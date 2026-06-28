//! Maps neural network outputs to joint torques.
//!
//! The NN emits one float per DOF; the actuator treats it as a signed **torque**.
//! The value is clamped to [-1, 1],
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

/// One commanded torque per actuated joint.
pub const ACTION_SIZE: usize = CrabJointId::COUNT;

/// Resource holding the current action vector for each environment's crab.
/// Written by the brain (NN or test controller), read by the actuator system.
#[derive(Resource, Default)]
pub struct CrabActions {
    /// `envs[e][CrabJointId::index()]` = env e's commanded torque in [-1, 1].
    pub envs: Vec<[f32; ACTION_SIZE]>,
}

impl CrabActions {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![[0.0; ACTION_SIZE]; n];
    }
}

/// Applies each env's action vector as joint torques.
///
/// Every crab part's `ExternalForce` is overwritten each step (set, not
/// accumulated) so a torque never lingers past the tick that commanded it. The
/// carapace is the parent of twelve joints, so it collects all their reaction
/// torques; no joint applies a linear force, so its force is left at zero here for
/// the demo poke (which runs after this system) to add to.
pub fn apply_actions(
    actions: Res<CrabActions>,
    joints: Query<(Entity, &CrabJoint, &CrabEnvId, &MultibodyJoint)>,
    transforms: Query<&Transform>,
    mut forces: Query<(Entity, &mut ExternalForce), With<CrabBodyPart>>,
) {
    let mut torque: HashMap<Entity, Vec3> = HashMap::new();

    for (child, joint, env, mj) in joints.iter() {
        let Some(values) = actions.envs.get(env.0) else {
            continue;
        };
        let id = joint.id;
        let a = values[id.index()];
        // The SOLE ±1 torque-bound for every `CrabActions` writer (the training policy's raw
        // drive, the demo, manual control): each writes an un-clamped value and this clamp is
        // where it becomes a bounded muscle command. Keeping the bound here (not at each
        // writer) lets the training tax see the policy's unbounded drive — a saturating
        // `|a|≫1` is penalized for the overshoot, then clamped to a physical torque here.
        let a = if a.is_finite() {
            a.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let Ok(parent_tf) = transforms.get(mj.parent) else {
            continue;
        };
        // Every joint is revolute (the pincer too): the free axis lives in the
        // parent body frame, so rotate it into world and push an equal-and-opposite
        // torque couple on child and parent — a free vector, internal by
        // construction (zero net torque/force on the crab).
        let world_axis = parent_tf.rotation * joint.axis_local;
        let wrench = world_axis * (a * id.drive_torque_ceiling());
        *torque.entry(child).or_default() += wrench;
        *torque.entry(mj.parent).or_default() -= wrench;
    }

    // Force stays zero here (no linear actuators); the carapace's demo poke runs
    // after this system and adds to it.
    for (e, mut ef) in forces.iter_mut() {
        ef.force = Vec3::ZERO;
        ef.torque = torque.get(&e).copied().unwrap_or(Vec3::ZERO);
    }
}
