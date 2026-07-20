use bevy::app::AppExit;
use bevy::prelude::*;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Tensor;
use tracing::{info, warn};

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::arch::GaussianHead;
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId};
use crate::bot::sensor::{CrabObservation, CrabTargets, OBS_SIZE};
use crate::bot::{CrabRescued, CrabSpawns};
use crate::training::algorithm::NormalizedValue;
use crate::training::reward::{action_effort, dist_3d, planar_dist};
use crate::training::targets::seed_target;

use super::lifecycle::{EnvEpisode, EnvPhase};
use super::state::TrainingState;

fn log_effort_probe(envs: &[EnvEpisode], efforts: &[f32], effort_weight: f32) {
    let mut count = 0usize;
    let mut effort_sum = 0.0f32;
    for (e, ep) in envs.iter().enumerate() {
        if matches!(ep.phase, EnvPhase::Recording) {
            count += 1;
            effort_sum += efforts[e];
        }
    }
    if count > 0 {
        let mean_effort = effort_sum / count as f32;
        info!(
            "EFFORTLOG n={count} mean_effort={mean_effort:.3} mean_tax={:.4}",
            effort_weight * mean_effort,
        );
    }
}

pub(super) struct SampledAction {
    drive: [f32; ACTION_SIZE],
    log_prob: f32,
}

pub(super) struct BodyState {
    pub(super) poses: Vec<Option<(f32, f32)>>,
    pub(super) carapace_pos: Vec<Option<Vec3>>,
    pub(super) drifts: Vec<Option<f32>>,
    pub(super) max_speeds: Vec<f32>,
}

pub(super) struct StepInputs<'a> {
    pub(super) body: &'a BodyState,
    pub(super) min_tip_dists: &'a [Option<f32>],
    pub(super) obs: &'a [[f32; OBS_SIZE]],
    pub(super) drives: &'a [[f32; ACTION_SIZE]],
    pub(super) values: &'a [NormalizedValue],
    pub(super) log_probs: &'a [f32],
    pub(super) efforts: &'a [f32],
    pub(super) rescued_envs: &'a [usize],
}

fn normalize_observations(
    training: &mut TrainingState,
    obs: &CrabObservation,
) -> Vec<[f32; OBS_SIZE]> {
    let n = training.envs.len();
    let mut obs_arrays: Vec<[f32; OBS_SIZE]> = Vec::with_capacity(n);
    for row in &obs.rows()[..n] {
        let normalized = training.obs_normalizer.normalize(row);
        let nonfinite = match training.normalizer_increment.as_mut() {
            Some(inc) => inc.observe(row),
            None => 0,
        };
        training.nonfinite_obs_elements += u64::from(nonfinite);
        obs_arrays.push(normalized);
    }
    obs_arrays
}

fn forward_pass(
    training: &TrainingState,
    obs_arrays: &[[f32; OBS_SIZE]],
) -> (GaussianHead<NdArray>, Vec<NormalizedValue>) {
    let n = obs_arrays.len();
    let device = training.device;
    let flat: Vec<f32> = obs_arrays.iter().flat_map(|a| a.iter().copied()).collect();
    let obs_batch = Tensor::<NdArray, 2>::from_data(
        burn::tensor::TensorData::new(flat, [n, OBS_SIZE]),
        &device,
    );
    let log_std_floor = training.log_std_floor;
    training.brain.with_inference(|inference_brain| {
        let head = GaussianHead::new(inference_brain.policy(obs_batch.clone()), log_std_floor);
        let values: Vec<NormalizedValue> = inference_brain
            .value(obs_batch)
            .flatten::<1>(0, 1)
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(NormalizedValue)
            .collect();
        (head, values)
    })
}

fn sample_actions(
    head: &GaussianHead<NdArray>,
    noise: &[[f32; ACTION_SIZE]],
    device: &NdArrayDevice,
) -> Vec<SampledAction> {
    let n = noise.len();
    let flat: Vec<f32> = noise.iter().flat_map(|r| r.iter().copied()).collect();
    let eps = Tensor::<NdArray, 2>::from_data(
        burn::tensor::TensorData::new(flat, [n, ACTION_SIZE]),
        device,
    );
    let drives = head.sample(eps);
    let log_probs: Vec<f32> = head
        .log_prob_rows(drives.clone())
        .to_data()
        .to_vec()
        .unwrap();
    let drive_data: Vec<f32> = drives.to_data().to_vec().unwrap();
    log_probs
        .into_iter()
        .zip(drive_data.chunks_exact(ACTION_SIZE))
        .map(|(lp, row)| {
            let log_prob = if lp.is_nan() || lp.is_infinite() {
                // Loud, like the drive-NaN guard below: a non-finite log-prob means the
                // policy head is emitting garbage, and a silent 0.0 would skew the PPO
                // importance ratio with no trace (#199 small-dupes: no silent zeroing).
                warn!("non-finite log_prob from the policy head, substituting 0.0");
                0.0
            } else {
                lp.clamp(-20.0, 20.0)
            };
            let mut drive = [0.0f32; ACTION_SIZE];
            let mut has_nan = false;
            for (d, &v) in drive.iter_mut().zip(row) {
                if v.is_nan() || v.is_infinite() {
                    has_nan = true;
                } else {
                    *d = v;
                }
            }
            if has_nan {
                warn!("NaN/Inf detected in NN drive, zeroing the offending joints");
            }
            SampledAction { drive, log_prob }
        })
        .collect()
}

fn gather_body_state(
    n: usize,
    spawns: &CrabSpawns,
    terrain: &crate::terrain::TerrainGrid,
    carapace_q: &Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: &Query<(&CrabEnvId, &bevy_rapier3d::prelude::Velocity), With<CrabBodyPart>>,
) -> BodyState {
    let mut poses: Vec<Option<(f32, f32)>> = vec![None; n];
    let mut carapace_pos: Vec<Option<Vec3>> = vec![None; n];
    let mut drifts: Vec<Option<f32>> = vec![None; n];
    for (env, transform) in carapace_q.iter() {
        if let Some(p) = poses.get_mut(env.0) {
            let up = transform.rotation * Vec3::Y;
            // Height ABOVE the local ground (rl#281) — the quantity the fall terminal
            // and height reward mean.
            let t = transform.translation;
            let height = t.y - terrain.height(t.x, t.z);
            *p = Some((height, up.dot(Vec3::Y)));
        }
        if let Some(c) = carapace_pos.get_mut(env.0) {
            *c = Some(transform.translation);
        }
        if let Some(d) = drifts.get_mut(env.0) {
            let origin = spawns.origin(env.0);
            *d = Some(planar_dist(transform.translation, origin));
        }
    }
    let mut max_speeds: Vec<f32> = vec![0.0; n];
    for (env, vel) in parts_q.iter() {
        if let Some(m) = max_speeds.get_mut(env.0) {
            let lin = vel.linear.length();
            let ang = vel.angular.length();
            let s = if lin.is_finite() && ang.is_finite() {
                lin.max(ang / 3.0)
            } else {
                f32::INFINITY
            };
            *m = m.max(s);
        }
    }
    BodyState {
        poses,
        carapace_pos,
        drifts,
        max_speeds,
    }
}

/// Batched, one query pass for N envs — the multi-env form of
/// `targets::closest_tip_dist` (eval/demo use that single-env one).
fn closest_tip_dists(
    n: usize,
    targets: &CrabTargets,
    claw_tips_q: &Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
) -> Vec<Option<f32>> {
    let mut min_tip_dists: Vec<Option<f32>> = vec![None; n];
    for (env, tip) in claw_tips_q.iter() {
        let Some(slot) = min_tip_dists.get_mut(env.0) else {
            continue;
        };
        let Some(target) = targets.get(env.0) else {
            continue;
        };
        if !tip.translation.is_finite() {
            continue;
        }
        let d = dist_3d(tip.translation, target);
        *slot = Some(slot.map_or(d, |cur| cur.min(d)));
    }
    min_tip_dists
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn brain_step(
    mut training: NonSendMut<TrainingState>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    mut targets: ResMut<CrabTargets>,
    spawns: Res<CrabSpawns>,
    terrain: Res<crate::terrain::Terrain>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: Query<(&CrabEnvId, &bevy_rapier3d::prelude::Velocity), With<CrabBodyPart>>,
    claw_tips_q: Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
    mut exit: MessageWriter<AppExit>,
    mut rescued: MessageReader<CrabRescued>,
) {
    let n = training.envs.len();
    let rescued_envs: Vec<usize> = rescued.read().map(|m| m.env).collect();
    if obs.rows().len() != n || actions.len() != n {
        return;
    }
    let device = training.device;

    let obs_arrays = normalize_observations(&mut training, &obs);
    let (head, values) = forward_pass(&training, &obs_arrays);
    let noise = training.step_explore_noise(n);
    let sampled = sample_actions(&head, &noise, &device);

    let drive_arrays: Vec<[f32; ACTION_SIZE]> = sampled.iter().map(|s| s.drive).collect();
    let log_probs: Vec<f32> = sampled.iter().map(|s| s.log_prob).collect();
    let efforts: Vec<f32> = sampled.iter().map(|s| action_effort(&s.drive)).collect();

    actions.set_rows(&drive_arrays);
    for (e, ep) in training.envs.iter().enumerate() {
        if matches!(ep.phase, EnvPhase::Settling { .. }) {
            let _ = actions.rest(e); // deliberate skip pre-spawn
        }
    }

    let body = gather_body_state(n, &spawns, &terrain, &carapace_q, &parts_q);

    for e in 0..n {
        if targets.get(e).is_none() {
            seed_target(&mut targets, &spawns, e, &mut training.rng, &terrain);
        }
    }

    let min_tip_dists = closest_tip_dists(n, &targets, &claw_tips_q);
    for (e, tip) in min_tip_dists.iter().enumerate() {
        if matches!(training.envs[e].phase, EnvPhase::Recording)
            && let Some(d) = *tip
        {
            let ep = &mut training.envs[e];
            ep.min_tip_dist = Some(ep.min_tip_dist.map_or(d, |cur| cur.min(d)));
        }
    }

    let inputs = StepInputs {
        body: &body,
        min_tip_dists: &min_tip_dists,
        obs: &obs_arrays,
        drives: &drive_arrays,
        values: &values,
        log_probs: &log_probs,
        efforts: &efforts,
        rescued_envs: &rescued_envs,
    };
    training.finalize_transitions(&inputs, &mut targets, &spawns, &terrain);

    if training.log_effort {
        log_effort_probe(&training.envs, &efforts, training.effort_weight);
    }
    training.accumulate_drift(&body.drifts);

    training.total_steps += 1;

    if training.tick_budget != 0 && training.total_steps == training.tick_budget {
        info!(
            "Tick budget reached ({} ticks) — stopping training.",
            training.tick_budget
        );
        exit.write(AppExit::Success);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrainConfig;
    use crate::bot::RESET_GRACE_TICKS;
    use crate::bot::arch::AnyBrain;
    use crate::training::TrainBackend;
    use crate::training::algorithm::{StepEnd, Transition};
    use crate::training::reward::GRAB_REWARD;
    use bevy::ecs::system::RunSystemOnce;

    use super::super::lifecycle::reset_crab;

    /// Env 0's target-local obs channels after a Sense pass over one hand-built
    /// carapace — read through the named-slot view, exactly what the policy steers by.
    fn observe_target_local(carapace: Transform, target: Option<Vec3>) -> Vec3 {
        use bevy_rapier3d::prelude::Velocity;

        let mut world = bevy::ecs::world::World::new();
        let mut obs = CrabObservation::default();
        obs.resize(1);
        let mut targets = CrabTargets::default();
        targets.resize(1);
        targets.envs[0] = target;
        world.insert_resource(obs);
        world.insert_resource(targets);
        world.insert_resource(CrabSpawns::from_origins(vec![Vec3::ZERO]));
        world.insert_resource(crate::terrain::Terrain::new(std::sync::Arc::new(
            crate::terrain::TerrainGrid::flat(64.0),
        )));
        world.spawn((CrabCarapace, CrabEnvId(0), carapace, Velocity::default()));
        world
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation");
        let obs = world.resource::<CrabObservation>();
        obs.env(0).expect("env 0 sized").target_local()
    }

    #[test]
    fn target_obs_points_toward_target() {
        let offset = Vec3::new(2.0, 0.5, -1.0);
        let local = observe_target_local(Transform::IDENTITY, Some(offset));
        assert!(
            (local - offset).length() < 1e-5,
            "identity pose: target-local {local:?} must equal the world offset {offset:?} \
             (points toward the target with the right sign)"
        );

        let yaw = Quat::from_rotation_y(std::f32::consts::PI);
        let local_rot = observe_target_local(Transform::from_rotation(yaw), Some(offset));
        let expected_rot = yaw.inverse() * offset;
        assert!(
            (local_rot - expected_rot).length() < 1e-5,
            "180° yaw: target-local {local_rot:?} must be the offset rotated into the body \
             frame {expected_rot:?}"
        );
        assert!(
            (local_rot.x + offset.x).abs() < 1e-5 && (local_rot.z + offset.z).abs() < 1e-5,
            "a 180° yaw must flip the body-local forward/right components: got {local_rot:?} \
             vs world offset {offset:?}"
        );
        assert!(
            (local_rot.y - offset.y).abs() < 1e-5,
            "yaw about Y must leave the body-local Y (height) component unchanged"
        );

        let pos = Vec3::new(3.0, 0.0, 4.0);
        let local_at = observe_target_local(Transform::from_translation(pos), Some(pos));
        assert!(
            local_at.length() < 1e-5,
            "carapace sitting on the target reads a zero target-local vector, got {local_at:?}"
        );
    }

    /// One-env training world built by the PRODUCTION env constructor
    /// ([`build_rollout_app`] — the headless server world plus the full training
    /// system set, shove included), so these tests — the same-seed determinism
    /// contract above all — certify the world rollouts actually run in.
    fn headless_training_app(checkpoint_dir: &std::path::Path, seed: u64) -> App {
        use crate::training::inproc::build_rollout_app;
        use clap::Parser;

        let config = TrainConfig::try_parse_from([
            "rl",
            "--checkpoint-dir",
            checkpoint_dir.to_str().expect("utf-8 checkpoint dir"),
            "--seed",
            &seed.to_string(),
        ])
        .expect("parse default TrainConfig");

        build_rollout_app(0, &config, crate::bot::arch::ArchId::DEFAULT)
    }

    fn body_part_entities(app: &mut App) -> std::collections::HashSet<Entity> {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<CrabBodyPart>>();
        q.iter(app.world()).collect()
    }

    #[test]
    fn same_seed_reproduces_the_rollout_trajectory() {
        const SEED: u64 = 0x00D3_7E2A;
        const TICKS: u32 = RESET_GRACE_TICKS + 80;
        const FORCE_RESET_AT: u32 = RESET_GRACE_TICKS + 20;

        fn run(seed: u64, initial_brain: &AnyBrain<TrainBackend>) -> Vec<Transition> {
            let dir = std::env::temp_dir()
                .join(format!("rl_test_determinism_{seed}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            // The production wiring includes the shove system (rl#298 stage 4), which
            // pins the shove draw stream — it consumes the training RNG every
            // recording tick — into the same-seed contract.
            let mut app = headless_training_app(&dir, seed);
            app.world_mut()
                .non_send_resource_mut::<TrainingState>()
                .brain
                .set(initial_brain.clone());
            for t in 0..TICKS {
                if t == FORCE_RESET_AT {
                    let mut q = app
                        .world_mut()
                        .query_filtered::<&mut Transform, With<CrabCarapace>>();
                    if let Ok(mut tr) = q.single_mut(app.world_mut()) {
                        tr.translation.y = -1.0;
                    }
                }
                app.update();
            }
            let traj = app.world().non_send_resource::<TrainingState>().rollouts[0]
                .transitions
                .clone();
            let _ = std::fs::remove_dir_all(&dir);
            traj
        }

        let seed_dir =
            std::env::temp_dir().join(format!("rl_test_determinism_seed_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&seed_dir);
        let brain = headless_training_app(&seed_dir, SEED)
            .world()
            .non_send_resource::<TrainingState>()
            .brain()
            .clone();
        let _ = std::fs::remove_dir_all(&seed_dir);

        let a = run(SEED, &brain);
        let b = run(SEED, &brain);
        assert!(!a.is_empty(), "the run must record transitions to compare");
        assert_eq!(
            a.len(),
            b.len(),
            "the same seed must record the same number of transitions"
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.obs, y.obs,
                "transition {i} obs diverged across identical seeds"
            );
            assert_eq!(
                x.action, y.action,
                "transition {i} action diverged across identical seeds"
            );
            assert_eq!(
                x.reward.to_bits(),
                y.reward.to_bits(),
                "transition {i} reward diverged across identical seeds"
            );
        }

        let c = run(SEED ^ 0xABCD, &brain);
        let differs =
            a.len() != c.len() || a.iter().zip(c.iter()).any(|(x, y)| x.action != y.action);
        assert!(differs, "a different seed must change the trajectory");
    }

    #[test]
    fn rescued_env_respawns_exactly_once() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_rescue_once_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x1234);

        for _ in 0..(RESET_GRACE_TICKS + 8) {
            app.update();
        }
        {
            let st = app.world().non_send_resource::<TrainingState>();
            assert!(
                matches!(st.envs[0].phase, EnvPhase::Recording),
                "settle grace elapsed and no reset pending — env is recording"
            );
            assert!(st.envs[0].steps > 0, "episode should have recorded steps");
        }
        let episodes_before = app
            .world()
            .non_send_resource::<TrainingState>()
            .episode_count;

        {
            let mut q = app
                .world_mut()
                .query_filtered::<&mut Transform, With<CrabCarapace>>();
            let mut t = q.single_mut(app.world_mut()).expect("carapace");
            t.translation = Vec3::splat(f32::NAN);
        }

        app.world_mut()
            .run_system_once(crate::bot::rescue_lost_crabs)
            .expect("rescue system");
        let rescued_set = body_part_entities(&mut app);
        assert!(
            rescued_set.iter().all(|&e| {
                app.world()
                    .get::<Transform>(e)
                    .is_some_and(|t| t.translation.is_finite())
            }),
            "rescue must leave a finite crab"
        );

        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step");

        {
            let st = app.world().non_send_resource::<TrainingState>();
            assert!(
                matches!(st.envs[0].phase, EnvPhase::Settling { grace } if grace == RESET_GRACE_TICKS),
                "rescue path takes the settle grace itself (Settling, not AwaitingRespawn) — \
                 being in Settling and not AwaitingRespawn is what stops reset_crab respawning again"
            );
            assert_eq!(
                st.episode_count,
                episodes_before + 1,
                "the rescue must still terminate the episode for training"
            );
        }

        app.world_mut()
            .run_system_once(reset_crab)
            .expect("reset_crab");

        let after_set = body_part_entities(&mut app);
        assert_eq!(
            after_set, rescued_set,
            "rescued env was respawned twice in one tick (issue #16): reset_crab \
             replaced the rescue's crab instead of leaving it alone"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    fn grab_within_radius_ends_episode_with_terminal_bonus() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_grab_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x6AB);

        for _ in 0..(RESET_GRACE_TICKS + 8) {
            app.update();
        }
        assert!(
            matches!(
                app.world().non_send_resource::<TrainingState>().envs[0].phase,
                EnvPhase::Recording
            ),
            "env 0 must be live-recording before the grab"
        );
        let episodes_before = app
            .world()
            .non_send_resource::<TrainingState>()
            .episode_count;

        let tip_pos = {
            let mut q = app
                .world_mut()
                .query_filtered::<(&CrabEnvId, &Transform), With<CrabClawTip>>();
            q.iter(app.world())
                .find(|(env, _)| env.0 == 0)
                .map(|(_, t)| t.translation)
                .expect("env 0 must have a claw tip")
        };
        app.world_mut().resource_mut::<CrabTargets>().envs[0] = Some(tip_pos);

        app.update();

        let st = app.world().non_send_resource::<TrainingState>();
        let last = st.rollouts[0]
            .transitions
            .last()
            .expect("env 0 recorded a transition");
        assert_eq!(
            last.end,
            StepEnd::Terminal,
            "a grab must end the episode as a TRUE terminal (GAE bootstrap 0), not a truncation"
        );
        assert!(
            last.reward >= GRAB_REWARD - 1.0,
            "the grabbing transition must carry the one-shot grab bonus (~{GRAB_REWARD}): got {}",
            last.reward
        );
        assert_eq!(
            st.episode_count,
            episodes_before + 1,
            "the grab must end the episode and count it"
        );
        assert!(
            !matches!(st.envs[0].phase, EnvPhase::Recording),
            "env 0 must have left Recording (reset for the next episode) after the grab"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    fn height_reward_pairs_with_the_action_that_produced_it() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_phase15_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x5678);

        for _ in 0..(RESET_GRACE_TICKS + 8) {
            app.update();
        }
        assert!(
            matches!(
                app.world().non_send_resource::<TrainingState>().envs[0].phase,
                EnvPhase::Recording
            ),
            "env must be recording before the hand-driven ticks"
        );

        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation A");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step A");
        let act_a = app.world().resource::<CrabActions>().rows()[0];

        {
            let mut q = app
                .world_mut()
                .query_filtered::<&mut Transform, With<CrabCarapace>>();
            let mut t = q.single_mut(app.world_mut()).expect("carapace");
            t.translation.y = -1.0;
        }

        let transitions_before = app.world().non_send_resource::<TrainingState>().rollouts[0].len();

        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation B");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step B");
        let act_b = app.world().resource::<CrabActions>().rows()[0];

        let st = app.world().non_send_resource::<TrainingState>();
        let last = st.rollouts[0]
            .transitions
            .last()
            .expect("a transition was pushed");

        assert_eq!(
            st.rollouts[0].len(),
            transitions_before + 1,
            "tick B finalizes exactly the one pending transition"
        );
        assert_eq!(
            last.end,
            StepEnd::Terminal,
            "the sub-floor height read at tick B must terminate the transition"
        );
        assert_ne!(
            act_a, act_b,
            "consecutive sampled actions differ, so the pairing below is decisive"
        );
        assert_eq!(
            last.action, act_a,
            "the terminal height (read at tick B) is paired with act_a — the tick-A \
             action whose physics result that height is — not tick B's action; this \
             is the one-tick phase the fix restores (issue #15)"
        );
        assert!(
            st.envs[0].pending.is_none(),
            "a terminated env carries no pending into its reset"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }
}
