//! The per-tick Sense→Think→Act step ([`brain_step`]): normalize observations, one batched
//! forward pass, sample a drive per env, gather this tick's body state, then hand off to the
//! episode lifecycle ([`super::lifecycle`]). [`TrainingState`] itself lives in [`super::state`].

use bevy::app::AppExit;
use bevy::prelude::*;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Tensor;
use tracing::{info, warn};

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId};
use crate::bot::sensor::{CrabObservation, CrabTargets, OBS_SIZE};
use crate::bot::{CrabRescued, CrabSpawns};
use crate::training::algorithm::{NormalizedValue, compute_log_prob, sample_action};
use crate::training::curriculum::seed_target;
use crate::training::reward::{EFFORT_WEIGHT, action_effort, dist_3d, planar_dist};

use super::lifecycle::{EnvEpisode, EnvPhase};
use super::state::TrainingState;

/// Effort/tax probe (RL_LOG_EFFORT only — inert otherwise): per tick, the mean drive effort
/// `Σ|d|²` and the resulting tax `EFFORT_WEIGHT·effort`, over the live RECORDING envs. Lets a
/// calibration run read how big a bite the tax takes out of the positive reward at the current
/// weight, without parsing rollouts.
fn log_effort_probe(envs: &[EnvEpisode], efforts: &[f32]) {
    if std::env::var_os("RL_LOG_EFFORT").is_none() {
        return;
    }
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
            EFFORT_WEIGHT * mean_effort,
        );
    }
}

/// One env's sample for this tick: the policy's unbounded neural DRIVE `μ + σ·ε` (see
/// [`sample_action`]) and its sampling log-prob. The drive is the RL action proper — the PPO
/// log-prob and the metabolic tax are both over it. The ±1 torque bound the sim runs is the
/// actuator's job, not stored here (`apply_actions` clamps every command), so the unbounded
/// drive is the single quantity and a saturating `|d|≫1` overshoot stays visible to the tax.
/// One row of [`sample_actions`].
pub(super) struct SampledAction {
    drive: [f32; ACTION_SIZE],
    /// Sampling log-prob of `drive` under the policy (NaN/Inf-guarded and clamped).
    log_prob: f32,
}

/// Per-env body state read off this tick's post-physics poses (the live `s_{t+1}`). Most
/// fields are off-reward (poses/speeds feed the survival guards, drift the walking diagnostic)
/// — the ONE that enters [`compute_reward`] is `carapace_pos`, from which the call site derives
/// the carapace→target distance whose per-tick REDUCTION is the progress reward. `None` for an
/// env whose crab is momentarily absent (mid-respawn).
pub(super) struct BodyState {
    /// `(carapace height, up·Y uprightness)`, the survival-guard input (only height is read).
    pub(super) poses: Vec<Option<(f32, f32)>>,
    /// Carapace world position — the progress-reward input: the call site computes its planar
    /// distance to the target and credits the per-tick reduction (the body's net ground
    /// covered). Measuring the ORIGIN's distance (not COM velocity) is what makes the progress
    /// term spin/limb-fling-proof (see [`crate::training::reward`] module header).
    pub(super) carapace_pos: Vec<Option<Vec3>>,
    /// Carapace planar (XZ) distance from spawn — the walking diagnostic.
    pub(super) drifts: Vec<Option<f32>>,
    /// Fastest body part (limbs blow up first), linear-scaled — the blow-up guard input.
    pub(super) max_speeds: Vec<f32>,
}

/// The per-env arrays captured this tick, bundled into one named borrow so
/// [`TrainingState::finalize_transitions`]'s call site can't transpose the several
/// same-typed slices (three `&[f32]`, the obs/action arrays). Every field is indexed by
/// env and has length `envs.len()`.
pub(super) struct StepInputs<'a> {
    /// Post-physics body readings (poses/speeds/drift) — the survival-guard +
    /// walking-diagnostic inputs.
    pub(super) body: &'a BodyState,
    /// Closest claw-tip→target 3D distance per env this tick — the grab terminal's `d` (and
    /// the curriculum "reached" signal's, via the per-episode minimum).
    pub(super) min_tip_dists: &'a [Option<f32>],
    /// Normalized observation fed to the policy this tick (stashed in the pending).
    pub(super) obs: &'a [[f32; OBS_SIZE]],
    /// The policy's unbounded DRIVE this tick, carried into the transition (see
    /// [`SampledAction`]).
    pub(super) drives: &'a [[f32; ACTION_SIZE]],
    /// Value-head output for this tick's observation.
    pub(super) values: &'a [NormalizedValue],
    /// Sampling log-prob of this tick's drive.
    pub(super) log_probs: &'a [f32],
    /// `Σ|dᵢ|^L` effort summand over the unbounded DRIVES (the metabolic tax input).
    pub(super) efforts: &'a [f32],
    /// Envs force-respawned this tick by the non-finite rescue (their pose is the fresh
    /// spawn, not the action's result — so the action ends the episode with no reach credit).
    pub(super) rescued_envs: &'a [usize],
}

/// Normalize every env's observation, feeding the shared running stats (and, in worker mode,
/// the per-horizon increment the thread ships back — see [`TrainingState::normalizer_snapshot`]).
/// Returns one normalized row per env, the forward pass's input.
fn normalize_observations(
    training: &mut TrainingState,
    obs: &CrabObservation,
) -> Vec<[f32; OBS_SIZE]> {
    let n = training.envs.len();
    let mut obs_arrays: Vec<[f32; OBS_SIZE]> = Vec::with_capacity(n);
    for e in 0..n {
        let normalized = training.obs_normalizer.normalize(&obs.envs[e]);
        // Count non-finite obs elements from the per-horizon increment (worker mode), the
        // same samples that ship to the learner — so the tally matches the shipped horizon
        // and the master's own skip isn't double-counted (bddap/rl#181).
        let nonfinite = match training.normalizer_increment.as_mut() {
            Some(inc) => inc.observe(&obs.envs[e]),
            None => 0,
        };
        training.nonfinite_obs_elements += u64::from(nonfinite);
        obs_arrays.push(normalized);
    }
    obs_arrays
}

/// ONE batched forward pass for all `n` envs: `[n, OBS_SIZE]` through the trunk once — this is
/// what makes N crabs cheaper than N apps. Returns each env's policy-mean row, the shared
/// `log_std`, and each value. Value-head outputs enter the type system as [`NormalizedValue`]
/// HERE (the single wrap point), so every stored value is in the head's normalized space.
fn forward_pass(
    training: &TrainingState,
    obs_arrays: &[[f32; OBS_SIZE]],
) -> (
    Vec<Tensor<NdArray, 1>>,
    Tensor<NdArray, 1>,
    Vec<NormalizedValue>,
) {
    let n = obs_arrays.len();
    let device = training.device;
    let flat: Vec<f32> = obs_arrays.iter().flat_map(|a| a.iter().copied()).collect();
    let obs_batch = Tensor::<NdArray, 2>::from_data(
        burn::tensor::TensorData::new(flat, [n, OBS_SIZE]),
        &device,
    );
    // Reuse the inference brain cached for these weights — rebuilt only when the rollout
    // thread reloads the learner's snapshot (once per horizon), not every tick.
    let log_std_floor = training.log_std_floor;
    let (means_batch, log_std, values) = training.brain.with_inference(|inference_brain| {
        let (means_batch, log_std) = inference_brain.policy(obs_batch.clone(), log_std_floor);
        let values: Vec<NormalizedValue> = inference_brain
            .value(obs_batch)
            .flatten::<1>(0, 1)
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(NormalizedValue)
            .collect();
        (means_batch, log_std, values)
    });
    let means_rows = (0..n)
        .map(|e| {
            means_batch
                .clone()
                .slice([e..e + 1, 0..ACTION_SIZE])
                .flatten(0, 1)
        })
        .collect();
    (means_rows, log_std, values)
}

/// Sample one DRIVE per env from its policy mean and the shared `log_std`, using `noise[e]` as
/// that env's standard-normal `ε` (the temporally-correlated draw from [`OuNoise`], aligned by
/// env), with the NaN/Inf guards the live solver needs: a non-finite log-prob becomes 0
/// (else clamped to ±20),
/// and any non-finite drive element zeroes that element (warning once for the row).
fn sample_actions(
    means_rows: &[Tensor<NdArray, 1>],
    log_std: &Tensor<NdArray, 1>,
    noise: &[[f32; ACTION_SIZE]],
    device: &NdArrayDevice,
) -> Vec<SampledAction> {
    means_rows
        .iter()
        .zip(noise)
        .map(|(means, &eps)| {
            let drive_tensor = sample_action(means, log_std, eps, device);
            // Log-prob of the ACTUAL sample (the unbounded drive): the quantity the PPO update
            // later recomputes its ratio over.
            let log_prob = compute_log_prob(means, log_std, &drive_tensor);
            let log_prob = if log_prob.is_nan() || log_prob.is_infinite() {
                0.0
            } else {
                log_prob.clamp(-20.0, 20.0)
            };

            let drive_data: Vec<f32> = drive_tensor.to_data().to_vec().unwrap();
            let mut drive = [0.0f32; ACTION_SIZE];
            let mut has_nan = false;
            for (i, &v) in drive_data.iter().enumerate().take(ACTION_SIZE) {
                if v.is_nan() || v.is_infinite() {
                    has_nan = true;
                    drive[i] = 0.0;
                } else {
                    drive[i] = v;
                }
            }
            if has_nan {
                warn!("NaN/Inf detected in NN drive, zeroing the offending joints");
            }
            SampledAction { drive, log_prob }
        })
        .collect()
}

/// Gather each env's [`BodyState`] from this tick's post-physics poses/velocities. Computed
/// from queries already in hand (no extra reads). Rapier writes each parentless link's world
/// pose straight into `Transform` every FixedUpdate tick, so these are the live `s_{t+1}`
/// readings, in phase with the deferred transition (`GlobalTransform` would be PostUpdate-stale).
fn gather_body_state(
    n: usize,
    spawns: &CrabSpawns,
    carapace_q: &Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: &Query<(&CrabEnvId, &bevy_rapier3d::prelude::Velocity), With<CrabBodyPart>>,
) -> BodyState {
    let mut poses: Vec<Option<(f32, f32)>> = vec![None; n];
    let mut carapace_pos: Vec<Option<Vec3>> = vec![None; n];
    let mut drifts: Vec<Option<f32>> = vec![None; n];
    for (env, transform) in carapace_q.iter() {
        if let Some(p) = poses.get_mut(env.0) {
            let up = transform.rotation * Vec3::Y;
            *p = Some((transform.translation.y, up.dot(Vec3::Y)));
        }
        // Carapace world position for the progress reward (see [`BodyState::carapace_pos`]).
        if let Some(c) = carapace_pos.get_mut(env.0) {
            *c = Some(transform.translation);
        }
        if let Some(d) = drifts.get_mut(env.0) {
            let origin = spawns.0.get(env.0).copied().unwrap_or(Vec3::ZERO);
            *d = Some(planar_dist(transform.translation, origin));
        }
    }
    // Fastest body part per env — limbs, not the carapace, blow up first (tiny eye-stalk
    // balls + acceleration motors), so the blowup guard must watch every body. NaN poisons
    // the max, so fold it in as +inf.
    let mut max_speeds: Vec<f32> = vec![0.0; n];
    for (env, vel) in parts_q.iter() {
        if let Some(m) = max_speeds.get_mut(env.0) {
            let lin = vel.linear.length();
            let ang = vel.angular.length();
            let s = if lin.is_finite() && ang.is_finite() {
                // Angular blowups (rad/s) run ~3x the linear scale before the solver NaNs;
                // fold both into one number on the linear scale.
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

/// Closest claw-tip→target 3D distance per env this tick (the grab terminal's `d`, see
/// [`dist_3d`]), folded over both claw tips. `None` when the env has no target or no claw tip
/// this tick (mid-respawn); a non-finite tip is skipped, not folded as a spurious hit.
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

/// System: runs the brain to produce actions each physics step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn brain_step(
    mut training: NonSendMut<TrainingState>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    mut targets: ResMut<CrabTargets>,
    spawns: Res<CrabSpawns>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: Query<(&CrabEnvId, &bevy_rapier3d::prelude::Velocity), With<CrabBodyPart>>,
    claw_tips_q: Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
    mut exit: MessageWriter<AppExit>,
    mut rescued: MessageReader<CrabRescued>,
) {
    let n = training.envs.len();
    // Envs whose crab went non-finite and was force-respawned this tick: the
    // pose this step reads is already the fresh crab back at spawn, so the
    // episode must end here or the teleport bleeds into the reward stream.
    let rescued_envs: Vec<usize> = rescued.read().map(|m| m.env).collect();
    if obs.envs.len() != n || actions.envs.len() != n {
        // Resources are sized by the spawn system; skip the tick(s) before that.
        return;
    }
    let device = training.device;
    // The horizon's target band (Copy), captured before the per-env borrows below so
    // both `seed_target` paths sample from the same band the learner set this horizon —
    // one band per horizon, identical for the lazy first-episode seed and every reset.
    let band = training.band;

    // Sense → Think: normalize, one batched forward pass, sample an action per env.
    let obs_arrays = normalize_observations(&mut training, &obs);
    let (means_rows, log_std, values) = forward_pass(&training, &obs_arrays);
    let noise = training.step_explore_noise(n);
    let sampled = sample_actions(&means_rows, &log_std, &noise, &device);

    let drive_arrays: Vec<[f32; ACTION_SIZE]> = sampled.iter().map(|s| s.drive).collect();
    let log_probs: Vec<f32> = sampled.iter().map(|s| s.log_prob).collect();
    let efforts: Vec<f32> = sampled.iter().map(|s| action_effort(&s.drive)).collect();

    // The unbounded drive is written as-is; the actuator's ±1 clamp (`apply_actions`) is the
    // single torque-bound source, so there is no second clamp here.
    actions.envs.copy_from_slice(&drive_arrays);
    // Settling envs hold the rest pose (action 0); the policy takes over at
    // step 0 of the new episode.
    for (e, ep) in training.envs.iter().enumerate() {
        if matches!(ep.phase, EnvPhase::Settling { .. })
            && let Some(v) = actions.envs.get_mut(e)
        {
            *v = [0.0; ACTION_SIZE];
        }
    }

    let body = gather_body_state(n, &spawns, &carapace_q, &parts_q);

    // Lazily seed the FIRST episode's target for any env still without one (envs
    // start Recording with no target; episode-end reset seeds every subsequent one).
    // Training-only by construction: only `brain_step` runs here, so the demo's
    // CrabTargets stays empty and its obs target vector stays zero.
    for e in 0..n {
        if targets.get(e).is_none() {
            seed_target(&mut targets, &spawns, e, band, &mut training.rng);
        }
    }

    let min_tip_dists = closest_tip_dists(n, &targets, &claw_tips_q);
    // Fold this tick's closest tip distance into each RECORDING env's episode minimum —
    // the curriculum's competence signal. Recording-only: a Settling env already holds
    // the NEXT episode's target (seeded at reset), so crediting its settle-pose distances
    // would contaminate that episode's reach with poses the policy never chose.
    for (e, tip) in min_tip_dists.iter().enumerate() {
        if matches!(training.envs[e].phase, EnvPhase::Recording)
            && let Some(d) = *tip
        {
            let ep = &mut training.envs[e];
            ep.min_tip_dist = Some(ep.min_tip_dist.map_or(d, |cur| cur.min(d)));
        }
    }

    // ONE far target per episode: seeded at reset (and lazily above for the first episode)
    // and then held FIXED — no mid-episode resample. A grab now ENDS the episode (the sparse
    // terminal in `finalize_transitions`), so a fixed goal makes touching it strictly optimal:
    // the dense progress field pulls the body in, and the one-shot grab bonus + done caps the
    // approach — there is no positive per-tick stream to farm by hovering (progress telescopes
    // to zero once arrived, and effort is a net cost), so the crab closes the last stretch and
    // grabs rather than loitering at the radius edge.

    // Act → record: finalize last tick's pending transition against this tick's pose, stash
    // this tick's, and roll over any episode that ended. The sole writer of `Transition`s.
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
    training.finalize_transitions(&inputs, &mut targets, &spawns, band);

    log_effort_probe(&training.envs, &efforts);
    training.accumulate_drift(&body.drifts);

    training.total_steps += 1;

    // Fixed-tick stop: exactly `tick_budget` physics ticks, then save+exit. Tick
    // count, never wall-clock, so the run is reproducible across machines/load.
    // `==` (steps increment by one) so the catch-up burst that crosses the budget
    // logs/requests exit once, not once per remaining tick in the burst.
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
    use crate::bot::brain::CrabBrain;
    use crate::training::TrainBackend;
    use crate::training::algorithm::{StepEnd, Transition};
    use crate::training::reward::GRAB_REWARD;
    use bevy::ecs::system::RunSystemOnce;

    use super::super::lifecycle::{reset_crab, save_on_exit};

    /// Drive `build_observation` over a single hand-placed carapace and return env 0's
    /// observation. No physics/rig — just the resources the system reads plus one
    /// carapace entity at the given world pose, so the body-state and target-local slots
    /// can be checked against an exact expected value (joint slots stay 0, no joints).
    fn observe_one_carapace(carapace: Transform, target: Option<Vec3>) -> [f32; OBS_SIZE] {
        use bevy_rapier3d::prelude::Velocity;

        let mut world = bevy::ecs::world::World::new();
        let mut obs = CrabObservation::default();
        obs.resize(1);
        let mut targets = CrabTargets::default();
        targets.resize(1);
        targets.envs[0] = target;
        world.insert_resource(obs);
        world.insert_resource(targets);
        world.insert_resource(CrabSpawns(vec![Vec3::ZERO]));
        world.spawn((CrabCarapace, CrabEnvId(0), carapace, Velocity::default()));
        world
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation");
        world.resource::<CrabObservation>().envs[0]
    }

    /// Index of the first target-local obs slot (the carapace-frame target vector lives in
    /// `[BASE, BASE+3)`). Taken from the observation layout's one home so this test can't
    /// drift from the slots the sensor actually writes.
    const TARGET_LOCAL_BASE: usize = crate::bot::sensor::TARGET_SLOT;

    /// The reach goal must enter the observation as a vector that points TOWARD the
    /// target in the carapace's OWN frame (correct sign), and that body-local vector must
    /// be orientation-invariant: yaw the body and the world offset is unchanged, but its
    /// body-frame coordinates counter-rotate. This is the property the policy relies on to
    /// "walk toward the target vector" from any heading — a sign flip here would train it
    /// to walk directly away.
    #[test]
    fn target_obs_points_toward_target() {
        let base = TARGET_LOCAL_BASE;

        // Identity pose at origin: the body frame equals the world frame, so the
        // target-local vector must equal the raw world offset to the target — same
        // direction, same sign (it points AT the target, not away).
        let offset = Vec3::new(2.0, 0.5, -1.0);
        let obs = observe_one_carapace(Transform::IDENTITY, Some(offset));
        let local = Vec3::new(obs[base], obs[base + 1], obs[base + 2]);
        assert!(
            (local - offset).length() < 1e-5,
            "identity pose: target-local {local:?} must equal the world offset {offset:?} \
             (points toward the target with the right sign)"
        );

        // Yaw the carapace 180° about Y, target fixed in WORLD. The world offset is
        // unchanged, but in the rotated body frame "forward" now points the other way, so
        // the body-local X and Z must FLIP sign (Y, the spin axis, is unchanged). This is
        // the orientation-invariance the obs frame buys: same goal, body-relative reading.
        let yaw = Quat::from_rotation_y(std::f32::consts::PI);
        let obs_rot = observe_one_carapace(Transform::from_rotation(yaw), Some(offset));
        let local_rot = Vec3::new(obs_rot[base], obs_rot[base + 1], obs_rot[base + 2]);
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

        // Off-origin too: target-local is the offset FROM the carapace, not the absolute
        // target — translating the body by the same vector as the target leaves it zero.
        let pos = Vec3::new(3.0, 0.0, 4.0);
        let obs_at = observe_one_carapace(Transform::from_translation(pos), Some(pos));
        let local_at = Vec3::new(obs_at[base], obs_at[base + 1], obs_at[base + 2]);
        assert!(
            local_at.length() < 1e-5,
            "carapace sitting on the target reads a zero target-local vector, got {local_at:?}"
        );
    }

    /// Headless training app (physics + bot + training), one fixed tick per
    /// `update()`, one env, with a fixed RNG `seed` so the run is deterministic. The
    /// windowless physics+bot stack is the shared [`crate::bot::headless::headless_stack`]
    /// (same builder the rollout workers use); this adds the training systems on top, so
    /// these tests exercise the exact stack the sole trainer runs. Unlike the rollout
    /// worker it keeps the single-world default pool (no K-thread scaling fix needed for
    /// one app).
    fn headless_training_app(checkpoint_dir: &std::path::Path, seed: u64) -> App {
        use crate::bot::headless::{HeadlessStack, WorldRole, headless_stack};
        use clap::Parser;

        // Point the checkpoint dir at an empty scratch path so no real checkpoint
        // loads; every other field keeps its default (tick budget 0 = unlimited,
        // so brain_step never writes AppExit during the test).
        let config = TrainConfig::try_parse_from([
            "rl",
            "--checkpoint-dir",
            checkpoint_dir.to_str().expect("utf-8 checkpoint dir"),
            "--seed",
            &seed.to_string(),
        ])
        .expect("parse default TrainConfig");

        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
        });

        // Wire the training world the same way the `rl learn` rollout worlds do
        // (see inproc::build_rollout_app): worker-mode TrainingState + the Sense→
        // Think→Act systems, so these tests exercise the brain_step / reset_crab /
        // rescue semantics the sole trainer runs. Worker index 0 → the seed is used
        // unmixed, so a fixed `seed` reproduces the trajectory exactly.
        let state = TrainingState::new_worker(&config, 0);
        app.insert_non_send_resource(state)
            .add_systems(
                FixedUpdate,
                (brain_step, reset_crab)
                    .chain()
                    .in_set(crate::bot::BotSet::Think),
            )
            .add_systems(Last, save_on_exit);
        app
    }

    fn body_part_entities(app: &mut App) -> std::collections::HashSet<Entity> {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<CrabBodyPart>>();
        q.iter(app.world()).collect()
    }

    /// Same seed ⇒ identical rollout trajectory. The two runs are handed the SAME initial
    /// brain (cloned in), so the seeded `StdRng` that drives action noise, target placement,
    /// and spawn rotation is the ONLY thing steering the trajectory — a regression that
    /// drops the seed on any of those sites would desync the runs. Copying the brain also
    /// makes the test robust to the process-global weight-init RNG a parallel test may touch
    /// (the per-state `StdRng` is not shared). Self-contained: a CPU single-world app, no
    /// learner thread and no GPU.
    #[test]
    fn same_seed_reproduces_the_rollout_trajectory() {
        const SEED: u64 = 0x00D3_7E2A;
        const TICKS: u32 = RESET_GRACE_TICKS + 80;
        // Force an episode end partway through so the reset path's spawn rotation
        // (`random_spawn_rotation`, drawn from the seeded rng) is exercised — otherwise a
        // random initial policy may never trip a terminal within the window, leaving that
        // RNG site unchecked. Applied identically in every run, so it can't desync them.
        const FORCE_RESET_AT: u32 = RESET_GRACE_TICKS + 20;

        fn run(seed: u64, initial_brain: &CrabBrain<TrainBackend>) -> Vec<Transition> {
            let dir = std::env::temp_dir()
                .join(format!("rl_test_determinism_{seed}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let mut app = headless_training_app(&dir, seed);
            // Start from the SAME weights so only the seed differs across runs.
            app.world_mut()
                .non_send_resource_mut::<TrainingState>()
                .brain
                .set(initial_brain.clone());
            for t in 0..TICKS {
                if t == FORCE_RESET_AT {
                    // Drop the carapace below the kill floor so the next tick terminates the
                    // episode → reset → seeded spawn rotation.
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

        // One fixed initial brain shared by every run (the determinism we test is the RNG
        // plumbing, not weight init).
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

        // A DIFFERENT seed must (almost surely) change the trajectory — otherwise the seed
        // isn't actually steering the run.
        let c = run(SEED ^ 0xABCD, &brain);
        let differs =
            a.len() != c.len() || a.iter().zip(c.iter()).any(|(x, y)| x.action != y.action);
        assert!(differs, "a different seed must change the trajectory");
    }

    /// A crab that goes non-finite is rescued (despawn+respawn) by `rescue_nonfinite_crabs`
    /// BEFORE Sense; the same tick, `brain_step` ends the episode. A rescued env must
    /// respawn EXACTLY ONCE (the rescue's) — `reset_crab` must leave it alone — yet the
    /// episode must still terminate for training.
    ///
    /// The post-tick episode state is identical whether or not reset_crab also respawns
    /// (both end at `Settling { grace: RESET_GRACE_TICKS - 1 }`), so the only discriminator
    /// is ENTITY IDENTITY: the crab after the tick must be the exact set the rescue spawned.
    /// So we drive the rescue tick by hand, capture the rescued entity set, then run
    /// brain_step + reset_crab and assert the set is untouched.
    #[test]
    fn rescued_env_respawns_exactly_once() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_rescue_once_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x1234);

        // Settle past grace (RESET_GRACE_TICKS) and record a few real steps, so
        // the rescued branch has a recorded step to mark terminal (steps > 0).
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

        // Poison the multibody the way a tunneling blowup does: a non-finite root
        // pose (the path rescue_nonfinite_crabs detects).
        {
            let mut q = app
                .world_mut()
                .query_filtered::<&mut Transform, With<CrabCarapace>>();
            let mut t = q.single_mut(app.world_mut()).expect("carapace");
            t.translation = Vec3::splat(f32::NAN);
        }

        // --- Drive the rescue tick by hand, all within one frame (no update() in
        // between, so the CrabRescued message survives for brain_step to read). ---

        // Phase A: rescue runs (.before(Sense) in the real schedule) — despawns the
        // NaN crab, spawns a fresh one, emits CrabRescued. Capture the fresh set.
        app.world_mut()
            .run_system_once(crate::bot::rescue_nonfinite_crabs)
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

        // Phase B: Sense → brain_step → reset_crab, the rest of the tick.
        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step");

        // After brain_step the rescued env must be in Settling (the rescue took the
        // grace itself), NOT AwaitingRespawn — that is what stops reset_crab from
        // respawning it again.
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

        // The crux: reset_crab must NOT have torn the rescue's crab down and built a
        // third one. The body-part entities after the full tick are EXACTLY the set
        // the rescue spawned.
        let after_set = body_part_entities(&mut app);
        assert_eq!(
            after_set, rescued_set,
            "rescued env was respawned twice in one tick (issue #16): reset_crab \
             replaced the rescue's crab instead of leaving it alone"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    /// The sparse-grab path end-to-end (rl#95): a claw tip within the reach radius of the target
    /// ENDS the episode as a TRUE terminal carrying the one-shot grab bonus, and the env resets.
    /// We force the grab by moving the target ONTO a live claw tip of env 0, so this tick's
    /// minimum tip distance is ~0 (well under `CURRICULUM_REACH_RADIUS`).
    #[test]
    fn grab_within_radius_ends_episode_with_terminal_bonus() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_grab_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x6AB);

        // Settle past grace and record a few real steps so env 0 has a pending action to
        // finalize against the grab pose this tick (steps > 0, Recording).
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

        // Place the target ON one of env 0's claw tips so the next finalize sees a tip distance
        // of ~0 → grab. The target then stays fixed until the grab-triggered reset reseeds it.
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

        // One tick: brain_step finalizes the pending action against a pose whose claw tip is on
        // the target → grab → Terminal + bonus, then reset_crab respawns the env.
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

    /// Each action's reward and the pose change it produced must occupy the SAME
    /// transition — the one-tick deferral [`super::super::lifecycle::Pending`] documents. This
    /// pins the phase at the unambiguous seam, a terminal: tick A chooses action `act_a` at a
    /// live height; then we drop the carapace below the kill floor so tick B reads `h < 0.02`
    /// and terminates. The terminal the kill-floor height produces must carry `act_a` (the
    /// action whose result that height IS), not tick B's action.
    #[test]
    fn height_reward_pairs_with_the_action_that_produced_it() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_phase15_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x5678);

        // Settle past grace and record a few real steps so the env is Recording
        // with a pending already primed (so tick A below is a steady-state tick).
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

        // Tick A (carapace at its live, above-floor height): Sense → brain_step.
        // This finalizes the pre-existing pending and stashes pending_A — whose
        // action is what brain_step just wrote to CrabActions.
        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation A");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step A");
        let act_a = app.world().resource::<CrabActions>().envs[0];

        // Drop the carapace below the kill floor (0.02 m) so tick B reads a
        // terminal height. With physics not stepped here, this is the exact pose
        // tick B's brain_step sees.
        {
            let mut q = app
                .world_mut()
                .query_filtered::<&mut Transform, With<CrabCarapace>>();
            let mut t = q.single_mut(app.world_mut()).expect("carapace");
            t.translation.y = -1.0;
        }

        let transitions_before = app.world().non_send_resource::<TrainingState>().rollouts[0].len();

        // Tick B: Sense → brain_step. h(s_B) = -1 < 0.02 finalizes pending_A as a
        // terminal. brain_step also writes tick B's own action — capture it to
        // prove the terminal carries act_a, not act_b.
        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation B");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step B");
        let act_b = app.world().resource::<CrabActions>().envs[0];

        let st = app.world().non_send_resource::<TrainingState>();
        let last = st.rollouts[0]
            .transitions
            .last()
            .expect("a transition was pushed");

        // Exactly one transition pushed at tick B (the finalized pending_A).
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
        // The discriminator only means something if the two actions differ; with
        // independent sampling on different observations they almost surely do.
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
        // The env resets after a terminal — no pending may straddle the reset.
        assert!(
            st.envs[0].pending.is_none(),
            "a terminated env carries no pending into its reset"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }
}
