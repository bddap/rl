use bevy::prelude::*;

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{CrabAssets, CrabBodyPart, CrabEnvId, random_spawn_rotation};
use crate::bot::sensor::{CrabTargets, OBS_SIZE};
use crate::bot::{CrabSpawns, RESET_GRACE_TICKS, respawn_crab_rotated, settle_countdown};
use crate::training::algorithm::{NormalizedValue, StepEnd, Transition};
use crate::training::reward::{GRAB_REWARD, compute_reward, is_progress_glitch, planar_dist};
use crate::training::targets::{seed_target, tip_touch};

use super::state::TrainingState;
use super::step::{BodyState, StepInputs};

pub const MAX_EPISODE_TICKS: u32 = 1500;

fn classify_step_end(grabbed: bool, fell: bool, over_cap: bool) -> StepEnd {
    if grabbed || fell {
        StepEnd::Terminal
    } else if over_cap {
        StepEnd::Truncated
    } else {
        StepEnd::Continues
    }
}

struct PostStepPose {
    height: f32,
    max_speed: f32,
    d_now: Option<f32>,
    min_tip_dist: Option<f32>,
}

struct StepFinalize {
    transition: Transition,
    ended: bool,
    progress_glitch: bool,
}

fn finalize_pending_step(
    pending: &Pending,
    pose: PostStepPose,
    over_cap: bool,
    rescued: bool,
    effort_weight: f32,
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
            transition: transition(
                compute_reward(None, pending.effort, effort_weight),
                StepEnd::Terminal,
            ),
            ended: true,
            progress_glitch: false,
        };
    }

    let distance_closed = pending.target_dist.zip(d_now).map(|(prev, now)| prev - now);
    let progress_glitch = is_progress_glitch(distance_closed);
    let mut reward = compute_reward(distance_closed, pending.effort, effort_weight);

    let blowing_up = max_speed > 100.0 || !height.is_finite();
    let fell = !(0.02..=50.0).contains(&height) || blowing_up;
    let grabbed = min_tip_dist.is_some_and(tip_touch);
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
}

impl TrainingState {
    pub(super) fn finalize_transitions(
        &mut self,
        inputs: &StepInputs,
        targets: &mut CrabTargets,
        spawns: &CrabSpawns,
        terrain: &crate::terrain::TerrainGrid,
    ) {
        let body = inputs.body;
        let min_tip_dists = inputs.min_tip_dists;
        let rescued_envs = inputs.rescued_envs;
        #[allow(clippy::needless_range_loop)]
        for e in 0..self.envs.len() {
            if matches!(self.envs[e].phase, EnvPhase::Settling { .. }) || body.poses[e].is_none() {
                continue;
            }

            if pre_touched_target(&self.envs[e], min_tip_dists[e]) {
                debug_assert_eq!(self.envs[e].steps, 0, "Recording with no pending ⇒ virgin");
                self.envs[e].min_tip_dist = None;
                let close_frac = self.close_frac;
                seed_target(targets, spawns, e, close_frac, &mut self.rng, terrain);
                continue;
            }

            let episode_ended = if let Some(pending) = self.envs[e].pending.take() {
                let (height, _upright) = body.poses[e].expect("poses[e].is_none() handled above");
                let d_now = carapace_target_dist(body, targets, e);
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
                    self.effort_weight,
                );
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
                false
            };

            if !episode_ended && matches!(self.envs[e].phase, EnvPhase::Recording) {
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
                let reached = ep.min_tip_dist.is_some_and(tip_touch);
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

                // The episode's target bearing from the spawn origin — read BEFORE
                // seed_target replaces the target, binned onto the eval compass
                // (rl#276) so the per-bearing tally below and the chase eval speak
                // the same bearings.
                let bearing = targets.envs.get(e).copied().flatten().map(|t| {
                    let origin = spawns.origin(e);
                    crate::eval::bearing_bin((t.z - origin.z).atan2(t.x - origin.x))
                });

                let close_frac = self.close_frac;
                seed_target(targets, spawns, e, close_frac, &mut self.rng, terrain);

                self.reach_finished += 1;
                if reached {
                    self.reach_reached += 1;
                }
                if let Some(bin) = bearing {
                    self.reach_by_bearing[bin].1 += 1;
                    if reached {
                        self.reach_by_bearing[bin].0 += 1;
                    }
                }

                self.recent_rewards.push(ep_reward);
                self.episode_count += 1;
            }
        }
    }

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

fn carapace_target_dist(body: &BodyState, targets: &CrabTargets, e: usize) -> Option<f32> {
    body.carapace_pos[e]
        .zip(targets.get(e))
        .map(|(pos, target)| planar_dist(pos, target))
}

pub(crate) fn reset_crab(
    mut commands: Commands,
    mut training: NonSendMut<TrainingState>,
    mut actions: ResMut<CrabActions>,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    terrain: Res<crate::terrain::Terrain>,
    parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>,
) {
    for e in 0..training.envs.len() {
        if matches!(training.envs[e].phase, EnvPhase::AwaitingRespawn) {
            training.envs[e].phase = EnvPhase::Settling {
                grace: RESET_GRACE_TICKS,
            };
            let _ = actions.rest(e); // deliberate skip pre-spawn
            let origin = spawns.origin(e);
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

    for ep in training.envs.iter_mut() {
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

    /// The training arena's flat grid — these tests pin FLAT-arena lifecycle behavior.
    fn flat() -> crate::terrain::TerrainGrid {
        crate::terrain::TerrainGrid::flat(crate::physics::world::ARENA_HALF_SIZE)
    }

    #[test]
    fn classify_step_end_terminal_vs_truncation() {
        assert_eq!(classify_step_end(true, false, false), StepEnd::Terminal);
        assert_eq!(classify_step_end(true, false, true), StepEnd::Terminal);
        assert_eq!(classify_step_end(false, true, false), StepEnd::Terminal);
        assert_eq!(classify_step_end(false, false, true), StepEnd::Truncated);
        assert_eq!(classify_step_end(false, false, false), StepEnd::Continues);
        assert!(classify_step_end(true, false, false).ends_segment());
        assert!(classify_step_end(false, false, true).ends_segment());
        assert!(!classify_step_end(false, false, false).ends_segment());
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
        let config = crate::TrainConfig::scratch(
            &std::env::temp_dir().join("rl_test_pre_touched_reseed"),
            1,
            7,
        );
        let mut ts = TrainingState::new(&config, None);
        ts.envs[0].min_tip_dist = Some(0.1);

        let mut targets = CrabTargets::default();
        targets.resize(1);
        let touched = Vec3::new(0.5, 0.2, 0.4);
        targets.envs[0] = Some(touched);
        let spawns = CrabSpawns::from_origins(vec![Vec3::ZERO]);
        let body = BodyState {
            poses: vec![Some((1.0, 1.0))],
            carapace_pos: vec![Some(Vec3::ZERO)],
            drifts: vec![None],
            max_speeds: vec![0.0],
        };
        let inputs = StepInputs {
            body: &body,
            min_tip_dists: &[Some(0.1)],
            obs: &[[0.0; OBS_SIZE]],
            drives: &[[0.0; ACTION_SIZE]],
            values: &[NormalizedValue(0.0)],
            log_probs: &[0.0],
            efforts: &[0.0],
            rescued_envs: &[],
        };
        ts.finalize_transitions(&inputs, &mut targets, &spawns, &flat());

        assert_eq!(ts.rollouts[0].len(), 0, "no transition recorded");
        assert_eq!(ts.reach_finished, 0, "no episode counted");
        assert!(ts.envs[0].pending.is_none(), "the episode did not start");
        assert_eq!(ts.envs[0].min_tip_dist, None, "stale touch cleared");
        assert_ne!(targets.envs[0], Some(touched), "target replaced");
    }

    /// The rl#276 tally: a finished episode lands in the compass bin of ITS target
    /// (+Z of the spawn origin = 90° = bin 2), read before the reseed replaces it.
    #[test]
    fn a_finished_episode_bins_reach_by_target_bearing() {
        let config = crate::TrainConfig::scratch(
            &std::env::temp_dir().join("rl_test_bearing_bin_tally"),
            1,
            7,
        );
        let mut ts = TrainingState::new(&config, None);
        ts.envs[0].pending = Some(Pending {
            obs: [0.0; OBS_SIZE],
            action: [0.0; ACTION_SIZE],
            value: NormalizedValue(0.0),
            log_prob: 0.0,
            effort: 0.0,
            target_dist: None,
        });

        let mut targets = CrabTargets::default();
        targets.resize(1);
        targets.envs[0] = Some(Vec3::new(0.0, 0.2, 5.0));
        let spawns = CrabSpawns::from_origins(vec![Vec3::ZERO]);
        let body = BodyState {
            poses: vec![Some((1.0, 1.0))],
            carapace_pos: vec![Some(Vec3::ZERO)],
            drifts: vec![None],
            max_speeds: vec![0.0],
        };
        let inputs = StepInputs {
            body: &body,
            min_tip_dists: &[Some(REACH_RADIUS * 4.0)],
            obs: &[[0.0; OBS_SIZE]],
            drives: &[[0.0; ACTION_SIZE]],
            values: &[NormalizedValue(0.0)],
            log_probs: &[0.0],
            efforts: &[0.0],
            // A rescue ends the episode without needing a whole physics run.
            rescued_envs: &[0],
        };
        ts.finalize_transitions(&inputs, &mut targets, &spawns, &flat());

        assert_eq!(ts.reach_finished, 1, "the episode finished and was counted");
        let mut want = [(0, 0); crate::eval::EVAL_BEARINGS];
        want[2] = (0, 1);
        assert_eq!(
            ts.reach_by_bearing, want,
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
        const ALIVE_H: f32 = 1.0;
        const CALM: f32 = 1.0;
        let far_tip = Some(REACH_RADIUS * 4.0);
        let pose = |close: Option<f32>, tip: Option<f32>| PostStepPose {
            height: ALIVE_H,
            max_speed: CALM,
            d_now: close,
            min_tip_dist: tip,
        };

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
            EFFORT_WEIGHT_DEFAULT,
        );
        assert_eq!(r.transition.end, StepEnd::Terminal);
        assert!(r.ended);
        assert!(
            !r.progress_glitch,
            "the rescue's None progress is not a glitch"
        );
        assert_eq!(
            r.transition.reward.to_bits(),
            compute_reward(None, 0.7, EFFORT_WEIGHT_DEFAULT).to_bits(),
            "a rescued step earns only the effort tax (no progress/grab credit)"
        );

        let r = finalize_pending_step(
            &pend(0.0, Some(1.25)),
            pose(Some(1.0), far_tip),
            false,
            false,
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
            pose(Some(1.0), Some(0.0)),
            false,
            false,
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
            &pend(0.0, None),
            PostStepPose {
                height: 0.0,
                max_speed: CALM,
                d_now: None,
                min_tip_dist: far_tip,
            },
            false,
            false,
            EFFORT_WEIGHT_DEFAULT,
        );
        assert_eq!(
            r.transition.end,
            StepEnd::Terminal,
            "a sub-floor height is a fall terminal"
        );
        assert!(r.ended);

        let r = finalize_pending_step(
            &pend(0.0, Some(1.0)),
            pose(Some(1.0), far_tip),
            true,
            false,
            EFFORT_WEIGHT_DEFAULT,
        );
        assert_eq!(r.transition.end, StepEnd::Truncated);
        assert!(r.ended);

        let r = finalize_pending_step(
            &pend(0.0, Some(2.0)),
            pose(Some(0.0), far_tip),
            false,
            false,
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
}
