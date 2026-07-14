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

pub fn apply_actions(
    actions: Res<CrabActions>,
    joints: Query<(Entity, &CrabJoint, &CrabEnvId, &MultibodyJoint)>,
    transforms: Query<&Transform>,
    mut forces: Query<(Entity, &mut ExternalForce), With<CrabBodyPart>>,
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

    for (e, mut ef) in forces.iter_mut() {
        ef.force = Vec3::ZERO;
        ef.torque = torque.get(&e).copied().unwrap_or(Vec3::ZERO);
    }
}
