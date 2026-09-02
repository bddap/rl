use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::body::{CrabBodyPart, CrabEnvId, CrabJoint, CrabJointId};

pub const ACTION_SIZE: usize = CrabJointId::COUNT;

/// One label per action channel, in [`CrabJointId::all`] order — the action half of
/// [`crate::bot::channel_layout_digest`] (bddap/rl#271).
pub(super) fn action_channel_labels() -> Vec<String> {
    CrabJointId::all()
        .iter()
        .map(|id| format!("drive:{id:?}"))
        .collect()
}

/// The per-env drive rows the actuator applies. The channel order is PRIVATE
/// (bddap/rl#271): whole rows move across the NN boundary ([`Self::rows`],
/// [`Self::set_row`], [`Self::set_rows`]); per-joint access is by [`CrabJointId`],
/// never by raw index.
///
/// The env-indexed writers return `false` (a no-op) on an unsized env:
/// `spawn_initial_crabs` sizes the rows on the first armed Update and FixedUpdate can
/// tick first, so callers in that window skip rather than panic. The `#[must_use]`
/// forces every caller to either act on the miss or mark the skip deliberate
/// (`let _ =`) — an unmarked drop is exactly a silent unlanded drive.
#[derive(Resource, Default)]
pub struct CrabActions {
    envs: Vec<[f32; ACTION_SIZE]>,
}

impl CrabActions {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![[0.0; ACTION_SIZE]; n];
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Whole drive rows, one per env — for aggregates and row-level comparisons.
    pub fn rows(&self) -> &[[f32; ACTION_SIZE]] {
        &self.envs
    }

    /// Land a whole policy-output row on env `e`.
    #[must_use = "false = env not sized; the drive did not land"]
    pub fn set_row(&mut self, e: usize, row: [f32; ACTION_SIZE]) -> bool {
        self.write(e, |r| *r = row)
    }

    /// Land the policy's whole batch.
    pub fn set_rows(&mut self, rows: &[[f32; ACTION_SIZE]]) {
        assert_eq!(
            rows.len(),
            self.envs.len(),
            "policy batch row count != sized envs"
        );
        self.envs.copy_from_slice(rows);
    }

    /// Rest pose: zero every drive of env `e`.
    #[must_use = "false = env not sized; the drive did not land"]
    pub fn rest(&mut self, e: usize) -> bool {
        self.fill(e, 0.0)
    }

    /// The same drive on every joint of env `e`.
    #[must_use = "false = env not sized; the drive did not land"]
    pub fn fill(&mut self, e: usize, v: f32) -> bool {
        self.write(e, |r| *r = [v; ACTION_SIZE])
    }

    /// Drive one named joint of env `e`.
    #[must_use = "false = env not sized; the drive did not land"]
    pub fn set_drive(&mut self, e: usize, id: CrabJointId, v: f32) -> bool {
        self.write(e, |r| r[id.index()] = v)
    }

    /// Env `e`'s drive on one named joint; `None` if that env isn't sized yet.
    pub fn drive(&self, e: usize, id: CrabJointId) -> Option<f32> {
        self.envs.get(e).map(|r| r[id.index()])
    }

    fn write(&mut self, e: usize, f: impl FnOnce(&mut [f32; ACTION_SIZE])) -> bool {
        match self.envs.get_mut(e) {
            Some(row) => {
                f(row);
                true
            }
            None => false,
        }
    }
}

pub fn bounded_drive(raw: f32) -> f32 {
    if raw.is_finite() {
        raw.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

pub fn applied_torque(id: CrabJointId, raw: f32) -> f32 {
    bounded_drive(raw) * id.drive_torque_ceiling()
}

/// Σ of every joint's drive ceiling — the most |torque| one tick can command across
/// the whole rig, hence the denominator that normalizes a tick's summed |torque| into
/// the [0, 1] saturation fraction the eval reports (rl#279).
pub fn total_drive_torque_ceiling() -> f32 {
    CrabJointId::all()
        .iter()
        .map(|id| id.drive_torque_ceiling())
        .sum()
}

/// Extra solver iterations a ZERO-DRIVE crab island runs (rl#392 rest convergence +
/// sleep). A resource, not a bare constant, so an ablation that zeroes drives can
/// pin it to 0 and change ONE thing (rl#332 F5: the soak's `--zero-drive-after`
/// used to swap the solver regime along with the drives).
#[derive(Resource, Clone, Copy)]
pub struct SettleExtraIterations(pub usize);

impl Default for SettleExtraIterations {
    fn default() -> Self {
        Self(super::body::CRAB_SETTLE_EXTRA_ITERATIONS)
    }
}

pub fn apply_actions(
    actions: Res<CrabActions>,
    settle: Res<SettleExtraIterations>,
    joints: Query<(Entity, &CrabJoint, &CrabEnvId, &MultibodyJoint)>,
    transforms: Query<&Transform>,
    mut forces: Query<(Entity, &mut ExternalForce), With<CrabBodyPart>>,
    mut solver_iters: Query<(&CrabEnvId, &mut AdditionalSolverIterations), With<CrabBodyPart>>,
    mut warned_nonfinite: Local<bool>,
) {
    let mut torque: HashMap<Entity, Vec3> = HashMap::new();

    for (child, joint, env, mj) in joints.iter() {
        let Some(values) = actions.envs.get(env.0) else {
            continue;
        };
        let id = joint.id;
        let raw = values[id.index()];
        if !raw.is_finite() && !*warned_nonfinite {
            error!(
                "crab actuator: non-finite drive ({raw}) on joint {id:?} — zeroed; a healthy \
                 brain never emits NaN/∞, so this flags a numerically-broken policy"
            );
            *warned_nonfinite = true;
        }
        let Ok(parent_tf) = transforms.get(mj.parent) else {
            continue;
        };
        let world_axis = parent_tf.rotation * joint.axis_local;
        let wrench = world_axis * applied_torque(id, raw);
        *torque.entry(child).or_default() += wrench;
        *torque.entry(mj.parent).or_default() -= wrench;
    }

    // A resting crab must be allowed to SLEEP (rl#392, the rl#377 pattern):
    // bevy_rapier force-wakes a body on any Changed ExternalForce, so an
    // unconditional write here resets every crab body's sleep timer each tick
    // and the awake contact solve rings the railed claw links forever. Skip
    // the write when the value is unchanged — a zero-drive policy then stops
    // touching the bodies after the first tick and rapier's pose-drift sleep
    // check retires the settled crab from the solver: rest is bit-exact rest.
    for (e, mut ef) in forces.iter_mut() {
        let next = ExternalForce {
            force: Vec3::ZERO,
            torque: torque.get(&e).copied().unwrap_or(Vec3::ZERO),
        };
        if *ef != next {
            *ef = next;
        }
    }

    // The two solver regimes, split by drive state through rapier's per-body
    // lever (rl#392): a DRIVEN crab runs the cheap global iteration counts (the
    // gait budget — this is most of the sim's per-tick cost), while a ZERO-DRIVE
    // crab gets extra solver iterations so its rest contacts actually converge,
    // settle below the sleep gates, and the whole multibody leaves the solver.
    // The elevated regime therefore lasts only the settle transient (~1 s);
    // asleep it costs nothing. Written only on drive-state edges — the value is
    // island-wide data, not a per-frame config fork.
    for (env, mut iters) in solver_iters.iter_mut() {
        let zero_drive = actions
            .envs
            .get(env.0)
            .is_none_or(|values| values.iter().all(|v| *v == 0.0));
        let want = if zero_drive { settle.0 } else { 0 };
        if iters.0 != want {
            iters.0 = want;
        }
    }
}
