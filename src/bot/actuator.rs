//! Maps neural network outputs to joint torques.
//!
//! The NN emits one float per DOF; the actuator treats it as a signed **torque**
//! (a linear force for the prismatic pincer). The value is clamped to [-1, 1],
//! scaled by the joint's force ceiling, and applied as an internal couple —
//! +τ about the joint's world axis on the child link, −τ on the parent — so the
//! articulated solver feels exactly what a joint motor would deliver, except the
//! policy picks the torque directly. There is no position servo: the crab has
//! to learn to hold and move itself, servoing its own joints if that's what
//! works. Hard joint limits still cap travel.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::body::{CrabBodyPart, CrabEnvId, CrabJoint, CrabJointId};

/// Resource holding the current action vector for each environment's crab.
/// Written by the brain (NN or test controller), read by the actuator system.
#[derive(Resource, Default)]
pub struct CrabActions {
    /// `envs[e][CrabJointId::index()]` = env e's commanded torque in [-1, 1].
    pub envs: Vec<[f32; CrabJointId::COUNT]>,
}

impl CrabActions {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![[0.0; CrabJointId::COUNT]; n];
    }
}

/// Applies each env's action vector as joint torques.
///
/// Every crab part's `ExternalForce` is overwritten each step (set, not
/// accumulated) so a torque never lingers past the tick that commanded it. The
/// carapace is the parent of twelve joints, so it collects all their reaction
/// torques; it carries no prismatic child, so its force is left at zero here for
/// the demo poke (which runs after this system) to add to.
pub fn apply_actions(
    actions: Res<CrabActions>,
    joints: Query<(Entity, &CrabJoint, &CrabEnvId, &MultibodyJoint)>,
    transforms: Query<&Transform>,
    mut forces: Query<(Entity, &mut ExternalForce), With<CrabBodyPart>>,
) {
    let mut torque: HashMap<Entity, Vec3> = HashMap::new();
    let mut force: HashMap<Entity, Vec3> = HashMap::new();

    for (child, joint, env, mj) in joints.iter() {
        let Some(values) = actions.envs.get(env.0) else {
            continue;
        };
        let id = joint.id;
        let a = values[id.index()];
        let a = if a.is_finite() {
            a.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let Ok(parent_tf) = transforms.get(mj.parent) else {
            continue;
        };
        // The joint's free axis lives in the parent body frame; rotate it into
        // the world. The couple is equal and opposite, so it is a pure internal
        // joint torque — no net external wrench on the crab.
        let world_axis = parent_tf.rotation * id.joint_axis_local();
        let wrench = world_axis * (a * id.motor_max_force());
        // Revolute joints push a torque couple; the prismatic pincer pushes a
        // linear force couple.
        let map = match id {
            CrabJointId::ClawPincer(_) => &mut force,
            _ => &mut torque,
        };
        *map.entry(child).or_default() += wrench;
        *map.entry(mj.parent).or_default() -= wrench;
    }

    for (e, mut ef) in forces.iter_mut() {
        ef.force = force.get(&e).copied().unwrap_or(Vec3::ZERO);
        ef.torque = torque.get(&e).copied().unwrap_or(Vec3::ZERO);
    }
}
