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

/// Target position range per joint type, in radians (or meters for prismatic).
fn action_range(id: &CrabJointId) -> [f32; 2] {
    match id {
        CrabJointId::LegCoxa(_, _) => [-0.78, 0.78],  // ~±45°
        CrabJointId::LegFemur(_, _) => [-1.57, 0.78], // -90° to +45°
        CrabJointId::LegTibia(_, _) => [-0.1, 1.88],  // ~0° to ~108°
        CrabJointId::ClawUpper(_) => [-0.78, 1.57],
        CrabJointId::ClawFore(_) => [-1.57, 1.57],
        CrabJointId::ClawPincer(_) => [0.0, 0.06], // prismatic: 0 to 6cm
        CrabJointId::EyeStalk(_) => [-0.3, 0.78],
    }
}

/// Maps action value in [-1, 1] to a target position within the joint's range.
fn action_to_target(action: f32, range: &[f32; 2]) -> f32 {
    // Guard against NaN/Inf — treat as zero (default position)
    let a = if action.is_finite() {
        action.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let t = (a + 1.0) * 0.5; // [0, 1]
    range[0] + t * (range[1] - range[0])
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
        let range = action_range(&crab_joint.id);
        let target = action_to_target(action, &range);
        let (stiffness, damping) = crab_joint.id.motor_stiffness_damping();
        let axis = crab_joint.id.joint_axis();

        let generic: &mut GenericJoint = mj.data.as_mut();
        generic.set_motor_position(axis, target, stiffness, damping);
        generic.set_motor_max_force(axis, crab_joint.id.motor_max_force());
    }
}
