use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::body::{CrabBodyPart, CrabEnvId, CrabJoint, CrabJointId};

pub const ACTION_SIZE: usize = CrabJointId::COUNT;

#[derive(Resource, Default)]
pub struct CrabActions {
    pub envs: Vec<[f32; ACTION_SIZE]>,
}

impl CrabActions {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![[0.0; ACTION_SIZE]; n];
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
