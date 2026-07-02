//! The per-env episode lifecycle: the phase machine ([`EnvPhase`], [`EnvEpisode`]), the pure
//! reward/terminal decision ([`classify_step_end`], [`finalize_pending_step`]), the per-env
//! driver [`TrainingState::finalize_transitions`], and the reset/save systems ([`reset_crab`],
//! [`save_on_exit`]). The per-tick Sense→Think→Act step lives in [`super::step`].

use bevy::app::AppExit;
use bevy::prelude::*;

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{CrabAssets, CrabBodyPart, CrabEnvId, random_spawn_rotation};
use crate::bot::sensor::{CrabTargets, OBS_SIZE};
use crate::bot::{CrabSpawns, RESET_GRACE_TICKS, respawn_crab_rotated, settle_countdown};
use crate::training::algorithm::{NormalizedValue, StepEnd, Transition};
use crate::training::curriculum::{CURRICULUM_REACH_RADIUS, seed_target};
use crate::training::reward::{GRAB_REWARD, compute_reward, is_progress_glitch, planar_dist};

use super::state::TrainingState;
use super::step::{BodyState, StepInputs};

/// Episode length cap: a crab still alive after this many physics ticks is TRUNCATED (not
/// failed — GAE bootstraps its value; see `StepEnd::Truncated`). At 64 Hz this is ~23 s of
/// crab-time. The reward calibration is balanced over exactly this horizon (a full traverse's
/// progress must out-earn the integrated effort tax across these ticks — see
/// `reward::EFFORT_WEIGHT`), so it is named once here and shared, never duplicated as a magic
/// `1500` (`rl-train eval --ticks` derives its default from it too).
pub const MAX_EPISODE_TICKS: u32 = 1500;

/// Classify a recorded step's episode end from its three independent end conditions, in
/// PRIORITY order — the single place the terminal-vs-truncation contract is decided, kept pure
/// so it is unit-tested directly (not only through a full rollout):
///   * `grabbed` (a claw tip reached the target) or `fell` (a survival guard tripped) is a TRUE
///     [`StepEnd::Terminal`] — the episode genuinely ended, so GAE bootstraps ZERO past it. A
///     grab outranks the cap: a step that both grabs and crosses the cap is a success, not a
///     truncation.
///   * else `over_cap` (the step cap reached while still alive) is a [`StepEnd::Truncated`] — the
///     episode was cut short, so GAE must BOOTSTRAP the cut-short value (teaching the cap is a
///     dead end would be wrong).
///   * else the trajectory [`StepEnd::Continues`].
fn classify_step_end(grabbed: bool, fell: bool, over_cap: bool) -> StepEnd {
    if grabbed || fell {
        StepEnd::Terminal
    } else if over_cap {
        StepEnd::Truncated
    } else {
        StepEnd::Continues
    }
}

/// One env's post-physics `s_{t+1}` readings for a single finalize, grouped into NAMED fields so
/// the four same-typed scalars can't be transposed at the call site (two `f32`, two `Option<f32>`).
/// On the rescue path every field describes the FRESH spawn and is ignored.
struct PostStepPose {
    /// Carapace height — the survival-guard input (feeds no reward).
    height: f32,
    /// Fastest body-part speed this tick — the blow-up guard input.
    max_speed: f32,
    /// Carapace planar distance to the target this tick (`None` if the pose or target is absent)
    /// — the progress reward's `s_{t+1}` distance.
    d_now: Option<f32>,
    /// Closest claw-tip→target 3D distance this tick (`None` if absent) — the grab terminal's `d`.
    min_tip_dist: Option<f32>,
}

/// The pure outcome of finalizing ONE env's pending action against the pose it produced —
/// the reward/terminal decision, lifted out of the ECS driver loop so it is unit-testable.
struct StepFinalize {
    /// The completed transition to push into this env's rollout buffer.
    transition: Transition,
    /// Whether this step ENDED the episode (terminal/truncation/rescue) — the driver then
    /// resets the env and reseeds its target.
    ended: bool,
    /// Whether a present per-tick progress delta was ZEROED as non-physical (bddap/rl#175) — the
    /// driver tallies it for the horizon. Never set on the rescue path (its progress is a neutral
    /// `None`, not a glitch).
    progress_glitch: bool,
}

/// Finalize one env's [`Pending`] action against this tick's post-physics reading — a PURE
/// function of the pending plus this env's `s_{t+1}` scalars, touching no env/ECS state, so the
/// reward + terminal core is unit-testable apart from `finalize_transitions`' per-env driver
/// loop (bddap/rl#165). Two modes:
///   * `rescued` — the action drove the body non-finite and it was force-respawned this tick, so
///     `height`/`d_now`/`min_tip_dist` describe the FRESH spawn, not the action's result. The
///     action ends the episode as a terminal with NO progress or reach credit (crediting the
///     spawn teleport would be a huge spurious progress delta); the effort tax still applies — it
///     priced the DRIVE, not its result.
///   * otherwise the survival guards (height band + blow-up speed) and the sparse terminal grab
///     (a claw tip within [`CURRICULUM_REACH_RADIUS`], rl#95) decide the end, and alive past the
///     cap is a truncation ([`classify_step_end`]). Progress is the metres the carapace's distance
///     to the goal SHRANK from `pending.target_dist` (`s_t`) to `d_now` (`s_{t+1}`); a missing
///     `d_now`/`target_dist` earns no progress credit.
fn finalize_pending_step(
    pending: &Pending,
    pose: PostStepPose,
    over_cap: bool,
    rescued: bool,
) -> StepFinalize {
    let PostStepPose {
        height,
        max_speed,
        d_now,
        min_tip_dist,
    } = pose;
    let transition = |reward: f32, end: StepEnd| Transition {
        obs: pending.obs,
        action: pending.action,
        reward,
        value: pending.value,
        log_prob: pending.log_prob,
        end,
    };

    if rescued {
        return StepFinalize {
            transition: transition(compute_reward(None, pending.effort), StepEnd::Terminal),
            ended: true,
            progress_glitch: false,
        };
    }

    // Progress reward: the reduction in carapace→target distance this action produced.
    let distance_closed = pending.target_dist.zip(d_now).map(|(prev, now)| prev - now);
    let progress_glitch = is_progress_glitch(distance_closed);
    let mut reward = compute_reward(distance_closed, pending.effort);

    // Survival guard: an unphysical speed blow-up (before the solver NaNs and Rapier panics the
    // app) or a height outside the sim-sanity band ends the episode. The threshold is high —
    // direct torque is bounded, so vigorous limb-flinging is legal; only clearly unphysical speed
    // trips it. `height` feeds no reward, only this guard.
    let blowing_up = max_speed > 100.0 || !height.is_finite();
    let fell = !(0.02..=50.0).contains(&height) || blowing_up;
    // Sparse terminal grab (rl#95): a claw tip within the reach radius this tick adds the one-shot
    // bonus and ends the episode as a SUCCESS terminal (GAE bootstraps ZERO past it). `is_some_and`
    // makes a missing/NaN distance a non-grab (fail-safe, no spurious terminal).
    let grabbed = min_tip_dist.is_some_and(|d| d < CURRICULUM_REACH_RADIUS);
    if grabbed {
        reward += GRAB_REWARD;
    }
    let end = classify_step_end(grabbed, fell, over_cap);
    StepFinalize {
        transition: transition(reward, end),
        ended: end.ends_segment(),
        progress_glitch,
    }
}

/// Where an env sits in the record → reset → settle lifecycle. One field, not a
/// `needs_reset: bool` + `grace: u32` pair, so an illegal combination (a respawn pending
/// *while* already settling) is unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum EnvPhase {
    /// Live episode: transitions are recorded and termination is evaluated.
    #[default]
    Recording,
    /// Ended by a normal terminal/truncation; `reset_crab` will despawn+respawn
    /// this env's crab on its next run and move it to `Settling`. Set by
    /// `brain_step` and consumed by `reset_crab` in the same tick, so it is
    /// never observed across a tick boundary.
    AwaitingRespawn,
    /// Fresh crab dropping into the rest pose: `grace` ticks remain in which no
    /// transition is recorded and no termination is evaluated, while it lands
    /// and the motors take the load. Reached from `AwaitingRespawn` (a normal
    /// reset) or directly from a non-finite rescue, which respawns the crab itself
    /// and so skips `AwaitingRespawn`. `grace` is always ≥ 1 here — it returns to
    /// `Recording` the tick it would hit 0.
    Settling { grace: u32 },
}

/// A transition whose action has been chosen and obs/value/effort captured, but
/// whose reward and end need NEXT tick's post-physics pose to finalize.
///
/// The reward and end must score the pose `aₜ` *produced* — the carapace's distance closed
/// (progress) and the claw-tip distance (the grab terminal) at `s_{t+1}` — but the schedule is
/// Sense → Think (`brain_step`) → Act → physics, so when `brain_step` runs at tick `t` the pose
/// it can read is still `sₜ` (physics hasn't integrated `aₜ` yet). So everything known at tick
/// `t` is stashed here and the transition completed at tick `t+1`, in phase with the pose that
/// action caused.
#[derive(Clone)]
pub(crate) struct Pending {
    obs: [f32; OBS_SIZE],
    /// The policy's unbounded DRIVE `μ + σ·ε` this tick (see [`super::step::SampledAction`]).
    action: [f32; ACTION_SIZE],
    /// The value-head prediction — read by `take_rollouts` as the GAE tail bootstrap.
    pub(super) value: NormalizedValue,
    log_prob: f32,
    /// `Σ|dᵢ|^L` for this drive — the effort summand over the unbounded DRIVES (see
    /// [`action_effort`]), final at tick `t` and traveling with the drive it priced.
    /// [`compute_reward`] scales it by [`EFFORT_WEIGHT`] at finalization.
    effort: f32,
    /// Carapace planar distance to the target at `s_t` (the pose this action was chosen from).
    /// The progress reward is the REDUCTION in this distance to `s_{t+1}`, computed at
    /// finalization (`P·(target_dist − d_now)`). `None` if the carapace pose or target was
    /// absent at stash time, in which case the transition earns no progress credit.
    target_dist: Option<f32>,
}

/// Per-env episode accumulators. Each env's episode runs and resets
/// independently.
#[derive(Clone, Default)]
pub(crate) struct EnvEpisode {
    pub(crate) reward: f32,
    pub(crate) steps: u32,
    pub(crate) phase: EnvPhase,
    /// Closest claw-tip→target 3D euclidean distance seen at any tick this episode — the
    /// curriculum's competence signal (an episode "reached" if this drops below
    /// [`CURRICULUM_REACH_RADIUS`]). `None` until the first finite tip reading. The MIN
    /// over the whole episode (not the final-tick distance) is the honest "did it get
    /// there", since the crab need only touch the target once, and the target then
    /// stays fixed for the rest of the episode. A 3D radius (see [`dist_3d`]): a tip on the
    /// floor under a raised ball does not count.
    pub(crate) min_tip_dist: Option<f32>,
    /// Tick `t`'s chosen action awaiting tick `t+1`'s post-physics pose (see
    /// [`Pending`]). `None` outside a live recording stride — before the first action of
    /// an episode, and after its last is finalized or dropped at a reset/rescue boundary.
    pub(crate) pending: Option<Pending>,
}

impl TrainingState {
    /// Per env, finalize the PREVIOUS tick's pending transition with this tick's post-physics
    /// pose (see [`Pending`] for the one-tick phasing), then stash this tick's action as the
    /// next pending. On an episode end (terminal/truncation/rescue) push the transition, tally
    /// the reach signal, record the reward, and reset the env (seeding its next target).
    ///
    /// The heart of `brain_step`: the only writer of [`Transition`]s and the per-env episode
    /// lifecycle. Termination is survival guards only — jumping, flipping, and any other
    /// strategy the policy invents are legitimate (owner call: emergent behaviour is the
    /// point); the height band is sim sanity, not a behaviour bound.
    pub(super) fn finalize_transitions(
        &mut self,
        inputs: &StepInputs,
        targets: &mut CrabTargets,
        spawns: &CrabSpawns,
    ) {
        let body = inputs.body;
        let min_tip_dists = inputs.min_tip_dists;
        let rescued_envs = inputs.rescued_envs;
        // Index loop, not a zip: each iteration reads several parallel per-env arrays
        // (body/min_tip_dists/the StepInputs slices) AND mutates `self.envs[e]` /
        // `self.rollouts[e]`, so there is no single slice to iterate over.
        #[allow(clippy::needless_range_loop)]
        for e in 0..self.envs.len() {
            // A pending exists only across a live recording stride, so an env that is
            // settling (or whose crab is momentarily absent) has none to finalize and
            // nothing to stash — the policy is holding the rest pose, not acting.
            if matches!(self.envs[e].phase, EnvPhase::Settling { .. }) || body.poses[e].is_none() {
                continue;
            }

            // Finalize the action chosen last tick using this tick's pose, then push it and
            // update this env's episode accumulators. The reward/terminal decision is the pure
            // [`finalize_pending_step`]; this loop is only the driver (take pending → push →
            // reset on end). `min_tip_dists[e]`/pose are this tick's `s_{t+1}` — the result of
            // `pending`'s action — so the credit lands in phase with the pose it caused.
            let episode_ended = if let Some(pending) = self.envs[e].pending.take() {
                // The pose's second element (uprightness) is unused; `height` feeds only the
                // survival guard. On the rescue path this pose is the fresh spawn and is ignored.
                let (height, _upright) = body.poses[e].expect("poses[e].is_none() handled above");
                let d_now = carapace_target_dist(body, targets, e);
                // `steps` BEFORE this step's increment — alive PAST the cap ⇒ a truncation.
                let over_cap = self.envs[e].steps > MAX_EPISODE_TICKS;
                let rescued = rescued_envs.contains(&e);
                let fin = finalize_pending_step(
                    &pending,
                    PostStepPose {
                        height,
                        max_speed: body.max_speeds[e],
                        d_now,
                        min_tip_dist: min_tip_dists[e],
                    },
                    over_cap,
                    rescued,
                );
                // Surface a zeroed non-physical progress delta on the learner line (bddap/rl#175).
                if fin.progress_glitch {
                    self.progress_glitch_drops += 1;
                }
                let reward = fin.transition.reward;
                self.rollouts[e].push(fin.transition);
                let ep = &mut self.envs[e];
                ep.reward += reward;
                ep.steps += 1;
                fin.ended
            } else {
                // No pending yet: the first recording tick of an episode only chooses
                // an action (stashed below); its result, and thus its transition,
                // arrives next tick.
                false
            };

            // Stash this tick's action to finalize next tick — but only if the env is
            // still recording (a just-ended env is resetting below, and a rescued env
            // is being respawned). Settling/absent envs already `continue`d above.
            if !episode_ended && matches!(self.envs[e].phase, EnvPhase::Recording) {
                // Carapace→target distance at THIS pose (`s_t` for the action chosen now), for
                // next tick's finalize (see [`Pending::target_dist`]).
                let target_dist = carapace_target_dist(body, targets, e);
                self.envs[e].pending = Some(Pending {
                    obs: inputs.obs[e],
                    action: inputs.drives[e],
                    value: inputs.values[e],
                    log_prob: inputs.log_probs[e],
                    effort: inputs.efforts[e],
                    target_dist,
                });
            }

            if episode_ended {
                let ep = &self.envs[e];
                let ep_reward = ep.reward;
                // Did this episode reach the target (see [`EnvEpisode::min_tip_dist`]), read
                // before the reset clears it. Same `CURRICULUM_REACH_RADIUS` as the grab terminal
                // but over the episode MINIMUM, so a grab ⟹ reached: the grab fires the first
                // finalized tick the tip is inside the radius, which also drives the min inside it.
                let reached = ep.min_tip_dist.is_some_and(|d| d < CURRICULUM_REACH_RADIUS);
                // A rescued env was already despawned+respawned this tick by
                // rescue_nonfinite_crabs (runs .before(Sense)); a second respawn from reset_crab
                // would tear down that zero-tick-old fresh crab and rebuild an identical one. So
                // the rescue path owns the reset: straight to `Settling` here, taking the grace
                // itself, while a normal end goes to `AwaitingRespawn` for reset_crab to respawn.
                self.envs[e] = EnvEpisode {
                    phase: if rescued_envs.contains(&e) {
                        EnvPhase::Settling {
                            grace: RESET_GRACE_TICKS,
                        }
                    } else {
                        EnvPhase::AwaitingRespawn
                    },
                    ..EnvEpisode::default()
                };

                // New episode → fresh target around this env's spawn slot, so the next
                // episode poses a new target. Done here (the one place both
                // the normal and rescue ends converge) so target life tracks episode life.
                seed_target(targets, spawns, e, &mut self.rng);

                // Tally this finished episode's reach for the curriculum (drained per horizon
                // to the learner, like the rewards just below).
                self.reach_finished += 1;
                if reached {
                    self.reach_reached += 1;
                }

                self.recent_rewards.push(ep_reward);
                self.episode_count += 1;
            }
        }
    }

    /// Accumulate this tick's carapace drift-from-spawn over RECORDING envs (one sample each)
    /// into the horizon's walking diagnostic (see `drift_sum`). Recording-only, so a settle
    /// pose can't masquerade as a cold policy's ~0 reach.
    pub(super) fn accumulate_drift(&mut self, drifts: &[Option<f32>]) {
        for (e, drift) in drifts.iter().enumerate() {
            if matches!(self.envs[e].phase, EnvPhase::Recording)
                && let Some(d) = *drift
                && d.is_finite()
            {
                self.drift_sum += d as f64;
                self.drift_count += 1;
            }
        }
    }

    /// This tick's exploration `ε` per env (bddap/rl#161): a RECORDING env advances its
    /// temporally-correlated AR(1) noise; the only other phase live here is `Settling`
    /// (`AwaitingRespawn` is set and consumed later within this same `brain_step`, never seen at
    /// this point — see [`EnvPhase`]), which RE-SEEDS its noise to a fresh draw so its next episode
    /// begins uncorrelated. The returned `ε` for a Settling env is DISCARDED — `brain_step` forces
    /// its drive to the rest pose regardless — so the `[0.0; _]` is a placeholder, not a meaningful
    /// zero. Reset-on-not-Recording keeps the per-episode fresh start impossible to forget: the
    /// only env that ever consumes the noise is a Recording one, and it always advances from a draw
    /// re-seeded during its settle.
    pub(super) fn step_explore_noise(&mut self, n: usize) -> Vec<[f32; ACTION_SIZE]> {
        (0..n)
            .map(|e| {
                if matches!(self.envs[e].phase, EnvPhase::Recording) {
                    self.explore_noise.next(e, &mut self.rng)
                } else {
                    self.explore_noise.reset(e, &mut self.rng);
                    [0.0; ACTION_SIZE]
                }
            })
            .collect()
    }
}

/// Env `e`'s carapace planar distance to its target this tick — the quantity whose per-tick
/// REDUCTION is the progress reward (see [`crate::training::reward`]). `None` if the carapace
/// pose or the target is absent (mid-respawn / unseeded), so the transition earns no spurious
/// progress.
fn carapace_target_dist(body: &BodyState, targets: &CrabTargets, e: usize) -> Option<f32> {
    body.carapace_pos[e]
        .zip(targets.get(e))
        .map(|(pos, target)| planar_dist(pos, target))
}

/// System: rebuilds each env's crab when that env's episode ends by a normal
/// terminal/truncation — `brain_step` leaves such an env in
/// [`EnvPhase::AwaitingRespawn`], which this system consumes. An episode ended
/// by a non-finite *rescue* is deliberately NOT handled here — that crab was
/// already respawned this tick by [`rescue_nonfinite_crabs`], so the rescue path
/// goes straight to [`EnvPhase::Settling`] and never enters `AwaitingRespawn`;
/// a respawn here would just rebuild the fresh crab a second time.
///
/// A reset is a full despawn + respawn ([`respawn_crab_rotated`]): teleporting bodies
/// cannot repair a multibody whose joint state went non-finite — rapier 0.32
/// offers no way to rewrite multibody joint coordinates in place — and one
/// crab that tunnels through the floor would otherwise wedge its env forever.
/// The respawned crab starts in the overlap-free rest pose, so no unfold or
/// collision-group dance is needed; the grace just skips recording while it
/// takes load (see [`EnvPhase::Settling`]).
pub(crate) fn reset_crab(
    mut commands: Commands,
    mut training: NonSendMut<TrainingState>,
    mut actions: ResMut<CrabActions>,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>,
) {
    // Randomized-start curriculum: each respawning env gets a fresh random orientation
    // so the policy learns to stand (and right itself) from varied, even inverted,
    // starts instead of memorising the one bind pose. This is training-only — reset_crab
    // never runs in the demo (no `TrainingState`), which respawns upright. The rotation
    // is drawn from the run's seeded RNG so a resumed/replayed run reproduces it.
    for e in 0..training.envs.len() {
        if matches!(training.envs[e].phase, EnvPhase::AwaitingRespawn) {
            training.envs[e].phase = EnvPhase::Settling {
                grace: RESET_GRACE_TICKS,
            };
            if let Some(v) = actions.envs.get_mut(e) {
                *v = [0.0; ACTION_SIZE];
            }
            let origin = spawns.0.get(e).copied().unwrap_or(Vec3::ZERO);
            let init_rotation = random_spawn_rotation(&mut training.rng);
            respawn_crab_rotated(
                &mut commands,
                &assets,
                parts.iter().filter(|(_, id)| id.0 == e).map(|(ent, _)| ent),
                origin,
                e,
                init_rotation,
            );
        }
    }

    // Count the settle grace down on every settling env (including one just set
    // above, which lands at RESET_GRACE_TICKS-1 this tick); when the shared
    // countdown is spent it returns to Recording and the policy takes back over.
    for ep in training.envs.iter_mut() {
        if let EnvPhase::Settling { grace } = ep.phase {
            ep.phase = match settle_countdown(grace) {
                0 => EnvPhase::Recording,
                g => EnvPhase::Settling { grace: g },
            };
        }
    }
}

/// System: saves a final checkpoint when the app is about to exit.
pub(crate) fn save_on_exit(
    mut training: NonSendMut<TrainingState>,
    mut exit_events: bevy::prelude::MessageReader<AppExit>,
) {
    if training.saved_on_exit {
        return;
    }
    if exit_events.read().next().is_some() {
        info!("App exiting — saving final checkpoint...");
        training.save_checkpoint();
        training.saved_on_exit = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The terminal-vs-truncation contract the value targets depend on (rl#95): a GRAB or a
    /// fall is a TRUE terminal (GAE bootstrap 0); the step cap is a TRUNCATION (bootstrap the
    /// cut-short value); otherwise the trajectory continues. A grab OUTRANKS the cap — a step
    /// that both grabs and crosses the cap is a success, not a truncation — so the success
    /// return is never silently bootstrapped past.
    #[test]
    fn classify_step_end_terminal_vs_truncation() {
        // Grab ⇒ true terminal (does not bootstrap). This is the new sparse-grab path.
        assert_eq!(classify_step_end(true, false, false), StepEnd::Terminal);
        // A grab on the very tick the cap is hit is still a SUCCESS terminal, not a truncation.
        assert_eq!(classify_step_end(true, false, true), StepEnd::Terminal);
        // A fall (survival guard) is a true terminal too.
        assert_eq!(classify_step_end(false, true, false), StepEnd::Terminal);
        // Alive at the cap ⇒ truncation (bootstraps the value — must differ from a terminal).
        assert_eq!(classify_step_end(false, false, true), StepEnd::Truncated);
        // Otherwise the episode continues.
        assert_eq!(classify_step_end(false, false, false), StepEnd::Continues);
        // The bootstrap contract those map to: terminal/truncation end the segment, continue
        // does not — so GAE bootstraps 0 on the grab terminal, the value on truncation.
        assert!(classify_step_end(true, false, false).ends_segment());
        assert!(classify_step_end(false, false, true).ends_segment());
        assert!(!classify_step_end(false, false, false).ends_segment());
    }

    /// The pure per-env finalize (bddap/rl#165): the reward + terminal decision extracted from
    /// `finalize_transitions`' ECS driver loop, now unit-testable without a world. Pins each
    /// branch — rescue (terminal, no progress/reach credit, effort tax only), the sparse grab
    /// (terminal + one-shot bonus), a survival fall (terminal), a plain continue, a truncation at
    /// the cap, and the #175 glitch flag on a non-physical progress delta.
    #[test]
    fn finalize_pending_step_covers_each_terminal_branch() {
        let pend = |effort: f32, target_dist: Option<f32>| Pending {
            obs: [0.0; OBS_SIZE],
            action: [0.0; ACTION_SIZE],
            value: NormalizedValue(0.0),
            log_prob: 0.0,
            effort,
            target_dist,
        };
        const ALIVE_H: f32 = 1.0;
        const CALM: f32 = 1.0;
        let far_tip = Some(CURRICULUM_REACH_RADIUS * 4.0);
        // A live-pose reading closing `close` metres with the tip `tip` from the target.
        let pose = |close: Option<f32>, tip: Option<f32>| PostStepPose {
            height: ALIVE_H,
            max_speed: CALM,
            d_now: close,
            min_tip_dist: tip,
        };

        // Rescue: terminal, NO progress/reach credit — reward is the effort tax only — no glitch.
        // The fresh-spawn pose (NaN height, blow-up speed, a big distance delta) is IGNORED.
        let r = finalize_pending_step(
            &pend(0.7, Some(1.0)),
            PostStepPose {
                height: f32::NAN,
                max_speed: 1e9,
                d_now: Some(0.0),
                min_tip_dist: Some(0.0),
            },
            true,
            true,
        );
        assert_eq!(r.transition.end, StepEnd::Terminal);
        assert!(r.ended);
        assert!(
            !r.progress_glitch,
            "the rescue's None progress is not a glitch"
        );
        assert_eq!(
            r.transition.reward.to_bits(),
            compute_reward(None, 0.7).to_bits(),
            "a rescued step earns only the effort tax (no progress/grab credit)"
        );

        // Continue: alive, calm, closing 0.25 m (exactly representable, so the reward compares
        // bit-exact), tip far, under the cap → Continues, no bonus.
        let r = finalize_pending_step(
            &pend(0.0, Some(1.25)),
            pose(Some(1.0), far_tip),
            false,
            false,
        );
        assert_eq!(r.transition.end, StepEnd::Continues);
        assert!(!r.ended);
        assert_eq!(
            r.transition.reward.to_bits(),
            compute_reward(Some(0.25), 0.0).to_bits()
        );

        // Grab: a tip inside the reach radius ends the episode as a terminal carrying GRAB_REWARD.
        let r = finalize_pending_step(
            &pend(0.0, Some(1.0)),
            pose(Some(1.0), Some(0.0)),
            false,
            false,
        );
        assert_eq!(
            r.transition.end,
            StepEnd::Terminal,
            "a grab is a true terminal"
        );
        assert!(r.ended);
        assert_eq!(
            r.transition.reward.to_bits(),
            (compute_reward(Some(0.0), 0.0) + GRAB_REWARD).to_bits()
        );

        // Fall: a sub-floor height trips the survival guard → terminal, even with a far tip.
        let r = finalize_pending_step(
            &pend(0.0, None),
            PostStepPose {
                height: 0.0,
                max_speed: CALM,
                d_now: None,
                min_tip_dist: far_tip,
            },
            false,
            false,
        );
        assert_eq!(
            r.transition.end,
            StepEnd::Terminal,
            "a sub-floor height is a fall terminal"
        );
        assert!(r.ended);

        // Truncation: alive PAST the cap with no grab/fall → Truncated (bootstraps its value).
        let r = finalize_pending_step(&pend(0.0, Some(1.0)), pose(Some(1.0), far_tip), true, false);
        assert_eq!(r.transition.end, StepEnd::Truncated);
        assert!(r.ended);

        // Glitch: a present but non-physical progress delta (> 0.5 m/tick) is flagged and zeroed.
        let r = finalize_pending_step(
            &pend(0.0, Some(2.0)),
            pose(Some(0.0), far_tip),
            false,
            false,
        );
        assert!(
            r.progress_glitch,
            "a > 0.5 m/tick delta is a progress glitch"
        );
        assert_eq!(
            r.transition.reward.to_bits(),
            compute_reward(None, 0.0).to_bits(),
            "the glitched progress is dropped to zero (effort tax only)"
        );
    }
}
