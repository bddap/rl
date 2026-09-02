use bevy::prelude::*;

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{CrabAssets, CrabBodyPart, CrabEnvId, random_spawn_rotation};
use crate::bot::sensor::{CrabTargets, OBS_SIZE};
use crate::bot::{CrabSpawns, RESET_GRACE_TICKS, respawn_crab_rotated, settle_countdown};
use crate::training::algorithm::{NormalizedValue, StepEnd, Transition};
use crate::training::reward::{GRAB_REWARD, compute_reward, is_progress_glitch, planar_dist};
use crate::training::targets::{seed_target, tip_touch};

use super::state::WorkerState;
use super::step::{EnvStep, MaxPartSpeed};

pub const MAX_EPISODE_TICKS: u32 = 1500;

/// `next_value` is this tick's critic value — V of the state the pending action
/// produced — which a cap truncation must carry for its GAE bootstrap.
fn classify_step_end(grabbed: bool, over_cap: bool, next_value: NormalizedValue) -> StepEnd {
    if grabbed {
        StepEnd::Terminal
    } else if over_cap {
        StepEnd::Truncated { next_value }
    } else {
        StepEnd::Continues
    }
}

/// rl#343: a pose outside physical bounds is an ENGINE bug, not crab behavior — the
/// run hard-fails with the state instead of ending the episode and respawning. The
/// deleted `fell` terminal both hid the bug and made dying the dominant cold-start
/// strategy (rl#342 finding #1). No penalty, no recovery, by owner directive: fix the
/// engine bug the state points at.
fn assert_physics_integrity(
    env: usize,
    tick: u64,
    ep_step: u32,
    height: f32,
    ms: &MaxPartSpeed,
    drives: &[f32; ACTION_SIZE],
    trace: &super::trace::IntegrityTrace,
) {
    let blowing_up = ms.speed > 100.0 || !height.is_finite();
    if blowing_up || !(0.02..=50.0).contains(&height) {
        let tripped = if blowing_up {
            "blowing up (part speed > 100 m/s or non-finite pose)"
        } else {
            "carapace height outside [0.02, 50] m"
        };
        let part = ms
            .part
            .map_or("Carapace".to_string(), |id| format!("{id:?}"));
        let fast: Vec<String> = ms
            .fast_parts
            .iter()
            .map(|(id, lin, ang)| {
                let name = id.map_or("Carapace".to_string(), |id| format!("{id:?}"));
                format!("{name} lin {lin:.1} ang {ang:.1}")
            })
            .collect();
        panic!(
            "physics-integrity violation in training (rl#343): env {env} tick {tick} \
             episode-step {ep_step}: {tripped} — height {height} m, fastest part \
             {part} at lin {} m/s ang {} rad/s (bound tests lin.max(ang/3) = {} m/s), \
             all parts past 50: [{}], drive row {drives:?}. Training does not recover \
             from a broken physics state; fix the engine bug instead of respawning \
             over it.\nflight recorder (rl#349), oldest first:{}",
            ms.lin,
            ms.ang,
            ms.speed,
            fast.join("; "),
            trace.dump(env),
        );
    }
}

struct StepFinalize {
    transition: Transition,
    ended: bool,
    progress_glitch: bool,
}

fn finalize_pending_step(
    pending: &Pending,
    d_now: Option<f32>,
    min_tip_dist: Option<f32>,
    over_cap: bool,
    next_value: NormalizedValue,
    effort_weight: f32,
) -> StepFinalize {
    let distance_closed = pending.target_dist.zip(d_now).map(|(prev, now)| prev - now);
    let progress_glitch = is_progress_glitch(distance_closed);
    let mut reward = compute_reward(distance_closed, pending.effort, effort_weight);

    let grabbed = min_tip_dist.is_some_and(tip_touch);
    if grabbed {
        reward += GRAB_REWARD;
    }
    let end = classify_step_end(grabbed, over_cap, next_value);
    StepFinalize {
        transition: Transition {
            obs: pending.obs,
            action: pending.action,
            reward,
            value: pending.value,
            log_prob: pending.log_prob,
            end,
        },
        ended: end.ends_segment(),
        progress_glitch,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum EnvPhase {
    #[default]
    Recording,
    AwaitingRespawn,
    Settling {
        grace: u32,
    },
}

#[derive(Clone)]
pub(crate) struct Pending {
    obs: [f32; OBS_SIZE],
    action: [f32; ACTION_SIZE],
    pub(super) value: NormalizedValue,
    log_prob: f32,
    effort: f32,
    target_dist: Option<f32>,
}

#[derive(Clone, Default)]
pub(crate) struct EnvEpisode {
    pub(crate) reward: f32,
    pub(crate) steps: u32,
    pub(crate) phase: EnvPhase,
    pub(crate) min_tip_dist: Option<f32>,
    pub(crate) pending: Option<Pending>,
    /// This env's live random-shove burst, if any ([`super::shove`]).
    pub(crate) shove: super::shove::ShoveState,
}

impl WorkerState {
    pub(super) fn finalize_transitions(
        &mut self,
        steps: &[EnvStep],
        targets: &mut CrabTargets,
        spawns: &CrabSpawns,
        terrain: &crate::terrain::TerrainGrid,
    ) {
        #[allow(clippy::needless_range_loop)]
        for e in 0..self.mode.envs.len() {
            let step = &steps[e];
            if matches!(self.mode.envs[e].phase, EnvPhase::Settling { .. }) || step.height.is_none()
            {
                continue;
            }

            if pre_touched_target(&self.mode.envs[e], step.min_tip_dist) {
                debug_assert_eq!(
                    self.mode.envs[e].steps, 0,
                    "Recording with no pending ⇒ virgin"
                );
                self.mode.envs[e].min_tip_dist = None;
                let band_max_m = self.mode.band_max_m;
                seed_target(targets, spawns, e, band_max_m, &mut self.rng, terrain);
                continue;
            }

            let episode_ended = if let Some(pending) = self.mode.envs[e].pending.take() {
                let height = step.height.expect("height.is_none() handled above");
                assert_physics_integrity(
                    e,
                    self.mode.total_steps,
                    self.mode.envs[e].steps,
                    height,
                    &step.max_speed,
                    &step.drive,
                    &self.mode.trace,
                );
                let d_now = carapace_target_dist(step, targets, e);
                let over_cap = self.mode.envs[e].steps > MAX_EPISODE_TICKS;
                let fin = finalize_pending_step(
                    &pending,
                    d_now,
                    step.min_tip_dist,
                    over_cap,
                    step.value,
                    self.mode.effort_weight,
                );
                if fin.progress_glitch {
                    self.mode.telemetry.progress_glitch_drops += 1;
                }
                let reward = fin.transition.reward;
                self.mode.rollouts[e].push(fin.transition);
                let ep = &mut self.mode.envs[e];
                ep.reward += reward;
                ep.steps += 1;
                fin.ended
            } else {
                false
            };

            if !episode_ended && matches!(self.mode.envs[e].phase, EnvPhase::Recording) {
                let target_dist = carapace_target_dist(step, targets, e);
                self.mode.envs[e].pending = Some(Pending {
                    obs: step.obs,
                    action: step.drive,
                    value: step.value,
                    log_prob: step.log_prob,
                    effort: step.effort,
                    target_dist,
                });
            }

            if episode_ended {
                let ep = &self.mode.envs[e];
                let ep_reward = ep.reward;
                let reached = ep.min_tip_dist.is_some_and(tip_touch);
                self.mode.envs[e] = EnvEpisode {
                    phase: EnvPhase::AwaitingRespawn,
                    ..EnvEpisode::default()
                };

                // The episode's target bearing from the spawn origin — read BEFORE
                // seed_target replaces the target, binned onto the eval compass
                // (rl#276) so the per-bearing tally below and the chase eval speak
                // the same bearings.
                let bearing = targets.envs.get(e).copied().flatten().map(|t| {
                    let origin = spawns.origin(e);
                    crate::eval::bearing_bin((t.z - origin.z).atan2(t.x - origin.x))
                });

                let band_max_m = self.mode.band_max_m;
                seed_target(targets, spawns, e, band_max_m, &mut self.rng, terrain);

                let telemetry = &mut self.mode.telemetry;
                telemetry.reach_finished += 1;
                if reached {
                    telemetry.reach_reached += 1;
                }
                if let Some(bin) = bearing {
                    telemetry.reach_by_bearing[bin].1 += 1;
                    if reached {
                        telemetry.reach_by_bearing[bin].0 += 1;
                    }
                }

                self.record_episode_reward(ep_reward);
            }
        }
    }

    pub(super) fn accumulate_drift(&mut self, steps: &[EnvStep]) {
        for (ep, step) in self.mode.envs.iter().zip(steps) {
            if matches!(ep.phase, EnvPhase::Recording)
                && let Some(d) = step.drift
                && d.is_finite()
            {
                self.mode.telemetry.drift_sum += d as f64;
                self.mode.telemetry.drift_count += 1;
            }
        }
    }

    pub(super) fn step_explore_noise(&mut self, n: usize) -> Vec<[f32; ACTION_SIZE]> {
        let WorkerState { rng, mode, .. } = self;
        (0..n)
            .map(|e| {
                if matches!(mode.envs[e].phase, EnvPhase::Recording) {
                    mode.explore_noise.next(e, rng)
                } else {
                    mode.explore_noise.reset(e, rng);
                    [0.0; ACTION_SIZE]
                }
            })
            .collect()
    }
}

/// A target the settled rest pose is ALREADY touching (claw tip inside
/// REACH_RADIUS before the episode records a single step) would pay GRAB_REWARD
/// for doing nothing — a stream of do-nothing/+50 transitions poisoning the
/// update, and a fake `reached` in the logged stat. Only the close-disc
/// curriculum (rl#250) can realistically produce one — rest tips sit ~0.5 m from
/// origin, inside the ≥1.5 m chase band — so re-seed and try again instead of
/// scoring it. (Recording with no pending implies a virgin episode: every
/// non-ending tick re-arms pending and an ending tick resets the EnvEpisode.)
fn pre_touched_target(ep: &EnvEpisode, min_tip_dist: Option<f32>) -> bool {
    matches!(ep.phase, EnvPhase::Recording)
        && ep.pending.is_none()
        && min_tip_dist.is_some_and(tip_touch)
}

fn carapace_target_dist(step: &EnvStep, targets: &CrabTargets, e: usize) -> Option<f32> {
    step.carapace_pos
        .zip(targets.get(e))
        .map(|(pos, target)| planar_dist(pos, target))
}

#[allow(clippy::too_many_arguments)] // a bevy system's params are its dependency list
pub(crate) fn reset_crab(
    mut commands: Commands,
    mut training: NonSendMut<WorkerState>,
    mut actions: ResMut<CrabActions>,
    assets: Res<CrabAssets>,
    mut spawns: ResMut<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    terrain: Res<crate::terrain::Terrain>,
    parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>,
) {
    for e in 0..training.mode.envs.len() {
        if matches!(training.mode.envs[e].phase, EnvPhase::AwaitingRespawn) {
            training.mode.envs[e].phase = EnvPhase::Settling {
                grace: RESET_GRACE_TICKS,
            };
            let _ = actions.rest(e); // deliberate skip pre-spawn
            // Training samples a fresh locale per episode — the tile's whole
            // slope/relief distribution is the curriculum (rl#281 stage 4).
            let origin =
                crate::training::targets::random_episode_origin(&mut training.rng, &terrain);
            spawns.set_origin(e, origin);
            // The episode-end seeding (`finalize_transitions`, earlier this tick)
            // banded the target around the PREVIOUS locale — re-seed from the new
            // origin or every post-first episode would chase a point up to a tile
            // away (its obs/reward kilometers out of the trained band).
            let band_max_m = training.mode.band_max_m;
            seed_target(
                &mut targets,
                &spawns,
                e,
                band_max_m,
                &mut training.rng,
                &terrain,
            );
            let init_rotation = random_spawn_rotation(&mut training.rng);
            respawn_crab_rotated(
                &mut commands,
                &assets,
                &terrain,
                parts.iter().filter(|(_, id)| id.0 == e).map(|(ent, _)| ent),
                origin,
                e,
                init_rotation,
            );
        }
    }

    for ep in training.mode.envs.iter_mut() {
        if let EnvPhase::Settling { grace } = ep.phase {
            ep.phase = match settle_countdown(grace) {
                0 => EnvPhase::Recording,
                g => EnvPhase::Settling { grace: g },
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::reward::EFFORT_WEIGHT_DEFAULT;
    use crate::training::targets::REACH_RADIUS;

    /// A planar fixture grid, big enough for band sampling (≥ edge margin + band) —
    /// these tests pin lifecycle logic, and hand-checked geometry is exact on a plane.
    fn flat() -> crate::terrain::TerrainGrid {
        crate::terrain::TerrainGrid::flat(512.0)
    }

    /// A one-env worker plus the tick's [`EnvStep`] with a live body at the origin.
    fn worker_and_step(test: &str, min_tip_dist: Option<f32>) -> (WorkerState, EnvStep) {
        let config = crate::TrainConfig::scratch(
            &std::env::temp_dir().join(format!("rl_test_{test}")),
            1,
            7,
        );
        let ts = WorkerState::new_worker(&config, 0, crate::bot::arch::ArchId::DEFAULT);
        let step = EnvStep {
            height: Some(1.0),
            carapace_pos: Some(Vec3::ZERO),
            min_tip_dist,
            ..EnvStep::default()
        };
        (ts, step)
    }

    #[test]
    fn classify_step_end_terminal_vs_truncation() {
        let v = NormalizedValue(0.25);
        assert_eq!(classify_step_end(true, false, v), StepEnd::Terminal);
        assert_eq!(classify_step_end(true, true, v), StepEnd::Terminal);
        assert_eq!(
            classify_step_end(false, true, v),
            StepEnd::Truncated { next_value: v },
            "a truncation carries the successor value for its bootstrap"
        );
        assert_eq!(classify_step_end(false, false, v), StepEnd::Continues);
        assert!(classify_step_end(true, false, v).ends_segment());
        assert!(classify_step_end(false, true, v).ends_segment());
        assert!(!classify_step_end(false, false, v).ends_segment());
    }

    #[test]
    fn pre_touched_only_gates_a_virgin_recording_episode() {
        let ep = |phase, pending| EnvEpisode {
            phase,
            pending,
            ..Default::default()
        };
        let touching = Some(REACH_RADIUS * 0.5);
        assert!(
            pre_touched_target(&ep(EnvPhase::Recording, None), touching),
            "a rest-pose touch before any recorded step re-seeds"
        );
        assert!(!pre_touched_target(
            &ep(EnvPhase::Recording, None),
            Some(REACH_RADIUS * 4.0)
        ));
        let pending = Pending {
            obs: [0.0; OBS_SIZE],
            action: [0.0; ACTION_SIZE],
            value: NormalizedValue(0.0),
            log_prob: 0.0,
            effort: 0.0,
            target_dist: None,
        };
        assert!(
            !pre_touched_target(&ep(EnvPhase::Recording, Some(pending)), touching),
            "an in-flight step is an earned grab, finalized normally"
        );
    }

    /// The guard's actual promise: a pre-touched start records NOTHING — no
    /// transition, no reach tally, no leftover min_tip_dist — and the target is
    /// replaced, so the rest pose can never farm GRAB_REWARD.
    #[test]
    fn a_pre_touched_start_reseeds_without_scoring() {
        let (mut ts, step) = worker_and_step("pre_touched_reseed", Some(0.1));
        ts.mode.envs[0].min_tip_dist = Some(0.1);

        let mut targets = CrabTargets::default();
        targets.resize(1);
        let touched = Vec3::new(0.5, 0.2, 0.4);
        targets.envs[0] = Some(touched);
        let spawns = CrabSpawns::from_origins(vec![Vec3::ZERO]);
        ts.finalize_transitions(&[step], &mut targets, &spawns, &flat());

        assert_eq!(ts.mode.rollouts[0].len(), 0, "no transition recorded");
        assert_eq!(ts.mode.telemetry.reach_finished, 0, "no episode counted");
        assert!(
            ts.mode.envs[0].pending.is_none(),
            "the episode did not start"
        );
        assert_eq!(ts.mode.envs[0].min_tip_dist, None, "stale touch cleared");
        assert_ne!(targets.envs[0], Some(touched), "target replaced");
    }

    /// The rl#276 tally: a finished episode lands in the compass bin of ITS target
    /// (+Z of the spawn origin = 90° = bin 2), read before the reseed replaces it.
    #[test]
    fn a_finished_episode_bins_reach_by_target_bearing() {
        let (mut ts, step) = worker_and_step("bearing_bin_tally", Some(REACH_RADIUS * 4.0));
        ts.mode.envs[0].pending = Some(Pending {
            obs: [0.0; OBS_SIZE],
            action: [0.0; ACTION_SIZE],
            value: NormalizedValue(0.0),
            log_prob: 0.0,
            effort: 0.0,
            target_dist: None,
        });
        // An over-cap truncation ends the episode without needing a whole physics run.
        ts.mode.envs[0].steps = MAX_EPISODE_TICKS + 1;

        let mut targets = CrabTargets::default();
        targets.resize(1);
        targets.envs[0] = Some(Vec3::new(0.0, 0.2, 5.0));
        let spawns = CrabSpawns::from_origins(vec![Vec3::ZERO]);
        ts.finalize_transitions(&[step], &mut targets, &spawns, &flat());

        assert_eq!(
            ts.mode.telemetry.reach_finished, 1,
            "the episode finished and was counted"
        );
        let mut want = [(0, 0); crate::eval::EVAL_BEARINGS];
        want[2] = (0, 1);
        assert_eq!(
            ts.mode.telemetry.reach_by_bearing, want,
            "the episode tallies (unreached) in its target's compass bin"
        );
    }

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
        let far_tip = Some(REACH_RADIUS * 4.0);
        let succ_v = NormalizedValue(0.5);

        let r = finalize_pending_step(
            &pend(0.0, Some(1.25)),
            Some(1.0),
            far_tip,
            false,
            succ_v,
            EFFORT_WEIGHT_DEFAULT,
        );
        assert_eq!(r.transition.end, StepEnd::Continues);
        assert!(!r.ended);
        assert_eq!(
            r.transition.reward.to_bits(),
            compute_reward(Some(0.25), 0.0, EFFORT_WEIGHT_DEFAULT).to_bits()
        );

        let r = finalize_pending_step(
            &pend(0.0, Some(1.0)),
            Some(1.0),
            Some(0.0),
            false,
            succ_v,
            EFFORT_WEIGHT_DEFAULT,
        );
        assert_eq!(
            r.transition.end,
            StepEnd::Terminal,
            "a grab is a true terminal"
        );
        assert!(r.ended);
        assert_eq!(
            r.transition.reward.to_bits(),
            (compute_reward(Some(0.0), 0.0, EFFORT_WEIGHT_DEFAULT) + GRAB_REWARD).to_bits()
        );

        let r = finalize_pending_step(
            &pend(0.0, Some(1.0)),
            Some(1.0),
            far_tip,
            true,
            succ_v,
            EFFORT_WEIGHT_DEFAULT,
        );
        assert_eq!(
            r.transition.end,
            StepEnd::Truncated { next_value: succ_v },
            "the cap truncation carries this tick's value as its bootstrap"
        );
        assert!(r.ended);

        let r = finalize_pending_step(
            &pend(0.0, Some(2.0)),
            Some(0.0),
            far_tip,
            false,
            succ_v,
            EFFORT_WEIGHT_DEFAULT,
        );
        assert!(
            r.progress_glitch,
            "a > 0.5 m/tick delta is a progress glitch"
        );
        assert_eq!(
            r.transition.reward.to_bits(),
            compute_reward(None, 0.0, EFFORT_WEIGHT_DEFAULT).to_bits(),
            "the glitched progress is dropped to zero (effort tax only)"
        );
    }

    /// rl#343: a broken physics state in training panics with the diagnostics (env,
    /// tick, pose, what tripped) instead of scoring a terminal and respawning.
    #[test]
    #[should_panic(
        expected = "physics-integrity violation in training (rl#343): env 0 tick 0 \
                    episode-step 3: carapace height outside [0.02, 50] m"
    )]
    fn a_sub_floor_height_hard_fails_training() {
        assert_physics_integrity(
            0,
            0,
            3,
            0.0,
            &speed(1.0),
            &[0.0; ACTION_SIZE],
            &empty_trace(),
        );
    }

    #[test]
    #[should_panic(expected = "blowing up (part speed > 100 m/s or non-finite pose)")]
    fn a_blowup_hard_fails_training() {
        assert_physics_integrity(
            0,
            0,
            0,
            f32::NAN,
            &speed(1e9),
            &[0.0; ACTION_SIZE],
            &empty_trace(),
        );
    }

    #[test]
    fn a_live_crab_passes_the_integrity_check() {
        assert_physics_integrity(
            0,
            0,
            0,
            1.0,
            &speed(5.0),
            &[0.0; ACTION_SIZE],
            &empty_trace(),
        );
    }

    fn empty_trace() -> super::super::trace::IntegrityTrace {
        super::super::trace::IntegrityTrace::new(1)
    }

    fn speed(s: f32) -> MaxPartSpeed {
        MaxPartSpeed {
            speed: s,
            lin: s,
            ..MaxPartSpeed::default()
        }
    }
}
