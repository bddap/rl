//! Maps neural network outputs to joint motor commands.
//!
//! The NN outputs a vector of floats in [-1, 1], one per actuated DOF.
//! The actuator system scales these to joint-appropriate motor targets and
//! writes them into the Rapier `MultibodyJoint` components.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::body::{CrabEnvId, CrabJoint, CrabJointId};

/// Resource holding the current action vector for each environment's crab.
/// Written by the brain (NN or test controller), read by the actuator system.
#[derive(Resource, Default)]
pub struct CrabActions {
    /// `envs[e][CrabJointId::index()]` = env e's commanded value in [-1, 1].
    pub envs: Vec<[f32; CrabJointId::COUNT]>,
}

impl CrabActions {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![[0.0; CrabJointId::COUNT]; n];
    }
}

/// Maps action value in [-1, 1] to a motor target: -1 → the joint's lower
/// limit, 0 → the rest pose, +1 → the upper limit, linear on each side. The
/// two sides scale independently because crab joints have asymmetric ranges
/// (a knee bends far one way and not at all the other); action 0 = the
/// planted rest stance either way, so the policy's neutral output (and a
/// fresh episode) starts standing instead of sprawling at a range midpoint.
pub(crate) fn action_to_target(action: f32, id: &CrabJointId) -> f32 {
    // Guard against NaN/Inf — treat as zero (rest position)
    let a = if action.is_finite() {
        action.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let rest = id.default_position();
    let [lo, hi] = id.limits();
    if a >= 0.0 {
        rest + a * (hi - rest)
    } else {
        rest + a * (rest - lo)
    }
}

/// System that applies each env's `CrabActions` to that env's crab joints.
pub fn apply_actions(
    actions: Res<CrabActions>,
    mut joints: Query<(&CrabJoint, &CrabEnvId, &mut MultibodyJoint)>,
) {
    for (crab_joint, env, mut mj) in joints.iter_mut() {
        let Some(values) = actions.envs.get(env.0) else {
            continue;
        };
        let idx = crab_joint.id.index();
        let action = values[idx];
        let target = action_to_target(action, &crab_joint.id);
        let (stiffness, damping) = crab_joint.id.motor_stiffness_damping();
        let axis = crab_joint.id.joint_axis();

        let generic: &mut GenericJoint = mj.data.as_mut();
        generic.set_motor_position(axis, target, stiffness, damping);
        generic.set_motor_max_force(axis, crab_joint.id.motor_max_force());
    }
}
