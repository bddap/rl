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
use crate::training::reward::{action_effort, planar_dist};
use crate::training::targets::{closest_tip_dists, seed_target};

use super::lifecycle::{EnvEpisode, EnvPhase};
use super::state::WorkerState;

fn log_effort_probe(envs: &[EnvEpisode], steps: &[EnvStep], effort_weight: f32) {
    let mut count = 0usize;
    let mut effort_sum = 0.0f32;
    for (ep, step) in envs.iter().zip(steps) {
        if matches!(ep.phase, EnvPhase::Recording) {
            count += 1;
            effort_sum += step.effort;
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

struct SampledAction {
    drive: [f32; ACTION_SIZE],
    log_prob: f32,
}

/// The fastest part per env this tick, with enough identity to indict one joint
/// and separate a linear spike from an angular one when the rl#343 integrity
/// check trips (rl#346).
#[derive(Clone, Default)]
pub(super) struct MaxPartSpeed {
    /// The bound the integrity check tests: `lin.max(ang / 3)`.
    pub(super) speed: f32,
    pub(super) lin: f32,
    pub(super) ang: f32,
    /// `None` = the carapace.
    pub(super) part: Option<crate::bot::body::CrabJointId>,
    /// Every part past [`DETAIL_SPEED`] this tick — the whole-body picture the
    /// violation panic prints, distinguishing one whipping link from a
    /// tumbling assembly. Empty on a healthy tick, so the hot loop never
    /// allocates.
    pub(super) fast_parts: Vec<(Option<crate::bot::body::CrabJointId>, f32, f32)>,
}

/// Soft threshold (in bound units, `lin.max(ang / 3)`) past which a part joins
/// the diagnostic detail — half the 100 m/s violation bound.
const DETAIL_SPEED: f32 = 50.0;

/// Everything one tick knows about one env — body readings and NN outputs in ONE
/// struct, so index alignment across envs is by construction instead of eight
/// parallel slices held equal by convention (bddap/rl#337).
#[derive(Clone)]
pub(super) struct EnvStep {
    /// Carapace height ABOVE the local ground (rl#281) — the quantity the rl#343
    /// integrity bounds test. `None` while the env has no body.
    pub(super) height: Option<f32>,
    pub(super) carapace_pos: Option<Vec3>,
    /// Carapace planar drift from the spawn origin, for the drift telemetry.
    pub(super) drift: Option<f32>,
    pub(super) max_speed: MaxPartSpeed,
    /// Closest claw tip to this env's target this tick.
    pub(super) min_tip_dist: Option<f32>,
    pub(super) obs: [f32; OBS_SIZE],
    pub(super) drive: [f32; ACTION_SIZE],
    pub(super) value: NormalizedValue,
    pub(super) log_prob: f32,
    pub(super) effort: f32,
}

impl Default for EnvStep {
    fn default() -> Self {
        Self {
            height: None,
            carapace_pos: None,
            drift: None,
            max_speed: MaxPartSpeed::default(),
            min_tip_dist: None,
            obs: [0.0; OBS_SIZE],
            drive: [0.0; ACTION_SIZE],
            value: NormalizedValue(0.0),
            log_prob: 0.0,
            effort: 0.0,
        }
    }
}

fn normalize_observations(
    training: &mut WorkerState,
    obs: &CrabObservation,
) -> Vec<[f32; OBS_SIZE]> {
    let n = training.mode.envs.len();
    let mut obs_arrays: Vec<[f32; OBS_SIZE]> = Vec::with_capacity(n);
    for row in &obs.rows()[..n] {
        let normalized = training.obs_normalizer.normalize(row);
        let nonfinite = training.mode.increment.observe(row);
        training.mode.telemetry.nonfinite_obs_elements += u64::from(nonfinite);
        obs_arrays.push(normalized);
    }
    obs_arrays
}

fn forward_pass(
    training: &WorkerState,
    obs_arrays: &[[f32; OBS_SIZE]],
) -> (GaussianHead<NdArray>, Vec<NormalizedValue>) {
    let n = obs_arrays.len();
    let device = training.device;
    let flat: Vec<f32> = obs_arrays.iter().flat_map(|a| a.iter().copied()).collect();
    let obs_batch = Tensor::<NdArray, 2>::from_data(
        burn::tensor::TensorData::new(flat, [n, OBS_SIZE]),
        &device,
    );
    let log_std_floor = training.mode.log_std_floor;
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

/// Assemble this tick's [`EnvStep`]s: seed each env's slot from its NN outputs, then
/// scatter the body queries in by env id. One `Vec` built in one place — the length
/// checks here are the ONLY alignment seam, and they fail loud.
#[allow(clippy::too_many_arguments)]
fn gather_env_steps(
    n: usize,
    obs_arrays: Vec<[f32; OBS_SIZE]>,
    sampled: &[SampledAction],
    values: Vec<NormalizedValue>,
    min_tip_dists: Vec<Option<f32>>,
    spawns: &CrabSpawns,
    terrain: &crate::terrain::TerrainGrid,
    carapace_q: &Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: &Query<
        (
            &CrabEnvId,
            &bevy_rapier3d::prelude::Velocity,
            Option<&crate::bot::body::CrabJoint>,
        ),
        With<CrabBodyPart>,
    >,
) -> Vec<EnvStep> {
    assert!(
        obs_arrays.len() == n
            && sampled.len() == n
            && values.len() == n
            && min_tip_dists.len() == n,
        "per-env step data out of sync with the {n} training envs: obs {}, actions {}, \
         values {}, tip dists {} — the training world is mis-wired",
        obs_arrays.len(),
        sampled.len(),
        values.len(),
        min_tip_dists.len(),
    );
    let mut steps: Vec<EnvStep> = env_steps_from_nn(obs_arrays, sampled, values, min_tip_dists);

    for (env, transform) in carapace_q.iter() {
        if let Some(step) = steps.get_mut(env.0) {
            let t = transform.translation;
            step.height = Some(t.y - terrain.height(t.x, t.z));
            step.carapace_pos = Some(t);
            step.drift = Some(planar_dist(t, spawns.origin(env.0)));
        }
    }
    for (env, vel, joint) in parts_q.iter() {
        if let Some(m) = steps.get_mut(env.0).map(|s| &mut s.max_speed) {
            let lin = vel.linear.length();
            let ang = vel.angular.length();
            let s = if lin.is_finite() && ang.is_finite() {
                lin.max(ang / 3.0)
            } else {
                f32::INFINITY
            };
            if s > DETAIL_SPEED {
                m.fast_parts.push((joint.map(|j| j.id), lin, ang));
            }
            if s > m.speed {
                m.speed = s;
                m.lin = lin;
                m.ang = ang;
                m.part = joint.map(|j| j.id);
            }
        }
    }
    steps
}

fn env_steps_from_nn(
    obs_arrays: Vec<[f32; OBS_SIZE]>,
    sampled: &[SampledAction],
    values: Vec<NormalizedValue>,
    min_tip_dists: Vec<Option<f32>>,
) -> Vec<EnvStep> {
    obs_arrays
        .into_iter()
        .zip(sampled)
        .zip(values)
        .zip(min_tip_dists)
        .map(|(((obs, s), value), min_tip_dist)| EnvStep {
            obs,
            drive: s.drive,
            value,
            log_prob: s.log_prob,
            effort: action_effort(&s.drive),
            min_tip_dist,
            ..EnvStep::default()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn brain_step(
    mut training: NonSendMut<WorkerState>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    mut targets: ResMut<CrabTargets>,
    spawns: Res<CrabSpawns>,
    terrain: Res<crate::terrain::Terrain>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: Query<
        (
            &CrabEnvId,
            &bevy_rapier3d::prelude::Velocity,
            Option<&crate::bot::body::CrabJoint>,
        ),
        With<CrabBodyPart>,
    >,
    claw_tips_q: Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
    mut exit: MessageWriter<AppExit>,
    mut rescued: MessageReader<CrabRescued>,
) {
    let n = training.mode.envs.len();
    // rl#343: a rescue is the engine confessing it lost a crab (rl#137 non-finite /
    // rl#283 tunneled / rl#303 buried). Training hard-fails on it — the deleted
    // finalize-as-Terminal path silently traded an engine bug for a free respawn.
    if let Some(m) = rescued.read().next() {
        panic!(
            "physics-integrity violation in training (rl#343): rescue_lost_crabs \
             respawned env {} at tick {} (episode-step {}) — {:?} at `{}`. Training \
             does not recover from a broken physics state; fix the engine bug instead \
             of respawning over it.",
            m.env,
            training.mode.total_steps,
            training.mode.envs.get(m.env).map_or(0, |ep| ep.steps),
            m.reason,
            m.body,
        );
    }
    // Mis-sized obs/action resources mean the training world is mis-wired — fail
    // loud (bddap/rl#337; a silent return here dropped whole ticks with no trace).
    assert!(
        obs.rows().len() == n && actions.len() == n,
        "obs/action resources out of sync with the {n} training envs: obs {}, actions {} \
         — the training world is mis-wired",
        obs.rows().len(),
        actions.len(),
    );
    let device = training.device;

    let obs_arrays = normalize_observations(&mut training, &obs);
    let (head, values) = forward_pass(&training, &obs_arrays);
    let noise = training.step_explore_noise(n);
    let sampled = sample_actions(&head, &noise, &device);

    let drive_arrays: Vec<[f32; ACTION_SIZE]> = sampled.iter().map(|s| s.drive).collect();
    actions.set_rows(&drive_arrays);
    for (e, ep) in training.mode.envs.iter().enumerate() {
        if matches!(ep.phase, EnvPhase::Settling { .. }) {
            let _ = actions.rest(e); // deliberate skip pre-spawn
        }
    }

    for e in 0..n {
        if targets.get(e).is_none() {
            let band_max_m = training.mode.band_max_m;
            seed_target(
                &mut targets,
                &spawns,
                e,
                band_max_m,
                &mut training.rng,
                &terrain,
            );
        }
    }

    let steps = gather_env_steps(
        n,
        obs_arrays,
        &sampled,
        values,
        closest_tip_dists(n, &targets, &claw_tips_q),
        &spawns,
        &terrain,
        &carapace_q,
        &parts_q,
    );

    for (e, step) in steps.iter().enumerate() {
        if matches!(training.mode.envs[e].phase, EnvPhase::Recording)
            && let Some(d) = step.min_tip_dist
        {
            let ep = &mut training.mode.envs[e];
            ep.min_tip_dist = Some(ep.min_tip_dist.map_or(d, |cur| cur.min(d)));
        }
    }

    training.finalize_transitions(&steps, &mut targets, &spawns, &terrain);

    if training.mode.log_effort {
        log_effort_probe(&training.mode.envs, &steps, training.mode.effort_weight);
    }
    training.accumulate_drift(&steps);

    training.mode.total_steps += 1;

    if training.mode.tick_budget != 0 && training.mode.total_steps == training.mode.tick_budget {
        info!(
            "Tick budget reached ({} ticks) — stopping training.",
            training.mode.tick_budget
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
                .non_send_resource_mut::<WorkerState>()
                .brain
                .set(initial_brain.clone());
            for t in 0..TICKS {
                if t == FORCE_RESET_AT {
                    // Force an episode boundary mid-trajectory (over-cap truncation,
                    // rl#343 — an underground teleport would hard-fail the run) so the
                    // respawn/reset path is inside the same-seed contract.
                    app.world_mut()
                        .non_send_resource_mut::<WorkerState>()
                        .mode
                        .envs[0]
                        .steps = super::super::lifecycle::MAX_EPISODE_TICKS + 1;
                }
                app.update();
            }
            let traj = app.world().non_send_resource::<WorkerState>().mode.rollouts[0]
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
            .non_send_resource::<WorkerState>()
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

    /// rl#343: a rescue reaching the training loop is a physics-integrity violation —
    /// brain_step hard-fails with the state instead of finalizing the episode and
    /// letting the free respawn hide the engine bug.
    #[test]
    #[should_panic(
        expected = "physics-integrity violation in training (rl#343): rescue_lost_crabs \
                    respawned env 0"
    )]
    fn a_rescue_hard_fails_training() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_rescue_panics_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x1234);

        for _ in 0..(RESET_GRACE_TICKS + 8) {
            app.update();
        }
        {
            let st = app.world().non_send_resource::<WorkerState>();
            assert!(
                matches!(st.mode.envs[0].phase, EnvPhase::Recording),
                "settle grace elapsed and no reset pending — env is recording"
            );
        }

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
        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation");
        let _ = app.world_mut().run_system_once(brain_step);
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
                app.world().non_send_resource::<WorkerState>().mode.envs[0].phase,
                EnvPhase::Recording
            ),
            "env 0 must be live-recording before the grab"
        );
        let episodes_before = app.world().non_send_resource::<WorkerState>().episode_count;

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

        let st = app.world().non_send_resource::<WorkerState>();
        let last = st.mode.rollouts[0]
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
            !matches!(st.mode.envs[0].phase, EnvPhase::Recording),
            "env 0 must have left Recording (reset for the next episode) after the grab"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    fn reward_pairs_with_the_action_that_produced_it() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_phase15_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x5678);

        for _ in 0..(RESET_GRACE_TICKS + 8) {
            app.update();
        }
        assert!(
            matches!(
                app.world().non_send_resource::<WorkerState>().mode.envs[0].phase,
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

        // End the episode at tick B via over-cap truncation (rl#343 — the old
        // underground teleport would hard-fail the run, and the pairing pin below is
        // about pending mechanics, not the terminal kind).
        app.world_mut()
            .non_send_resource_mut::<WorkerState>()
            .mode
            .envs[0]
            .steps = super::super::lifecycle::MAX_EPISODE_TICKS + 1;

        let transitions_before =
            app.world().non_send_resource::<WorkerState>().mode.rollouts[0].len();

        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation B");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step B");
        let act_b = app.world().resource::<CrabActions>().rows()[0];

        let st = app.world().non_send_resource::<WorkerState>();
        let last = st.mode.rollouts[0]
            .transitions
            .last()
            .expect("a transition was pushed");

        assert_eq!(
            st.mode.rollouts[0].len(),
            transitions_before + 1,
            "tick B finalizes exactly the one pending transition"
        );
        assert_eq!(
            last.end,
            StepEnd::Truncated,
            "the over-cap read at tick B must end the transition"
        );
        assert_ne!(
            act_a, act_b,
            "consecutive sampled actions differ, so the pairing below is decisive"
        );
        assert_eq!(
            last.action, act_a,
            "the ending state (read at tick B) is paired with act_a — the tick-A \
             action whose physics result it is — not tick B's action; this is the \
             one-tick phase the fix restores (issue #15)"
        );
        assert!(
            st.mode.envs[0].pending.is_none(),
            "a terminated env carries no pending into its reset"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }
}
