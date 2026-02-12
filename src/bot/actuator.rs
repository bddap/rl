//! Maps neural network outputs to joint motor commands.
//!
//! The NN outputs a vector of floats in [-1, 1], one per actuated DOF.
//! The actuator system scales these to joint-appropriate motor targets and
//! writes them into the Rapier `MultibodyJoint` components.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::body::{CrabJoint, CrabJointId};

/// Resource holding the current action vector for the crab.
/// Written by the brain (NN or test controller), read by the actuator system.
#[derive(Resource, Default)]
pub struct CrabActions {
    /// One float per DOF in [-1, 1]. Indexed by `CrabJointId::index()`.
    pub values: [f32; CrabJointId::COUNT],
}

/// Motor parameters per joint type, used to scale [-1,1] actions.
struct MotorParams {
    /// Target position range [min, max] in radians (or meters for prismatic).
    range: [f32; 2],
    stiffness: f32,
    damping: f32,
}

fn motor_params(id: &CrabJointId) -> MotorParams {
    match id {
        CrabJointId::LegCoxa(_, _) => MotorParams {
            range: [-0.78, 0.78], // ~±45°
            stiffness: 200.0,
            damping: 20.0,
        },
        CrabJointId::LegFemur(_, _) => MotorParams {
            range: [-1.57, 0.78], // -90° to +45°
            stiffness: 200.0,
            damping: 20.0,
        },
        CrabJointId::LegTibia(_, _) => MotorParams {
            range: [-0.1, 1.88], // ~0° to ~108°
            stiffness: 200.0,
            damping: 20.0,
        },
        CrabJointId::ClawUpper(_) => MotorParams {
            range: [-0.78, 1.57],
            stiffness: 250.0,
            damping: 25.0,
        },
        CrabJointId::ClawFore(_) => MotorParams {
            range: [-1.57, 1.57],
            stiffness: 250.0,
            damping: 25.0,
        },
        CrabJointId::ClawPincer(_) => MotorParams {
            range: [0.0, 0.06], // prismatic: 0 to 6cm
            stiffness: 250.0,
            damping: 25.0,
        },
        CrabJointId::EyeStalk(_) => MotorParams {
            range: [-0.3, 0.78],
            stiffness: 25.0,
            damping: 5.0,
        },
    }
}

/// Maps action value in [-1, 1] to a target position within the joint's range.
fn action_to_target(action: f32, params: &MotorParams) -> f32 {
    // Guard against NaN/Inf — treat as zero (default position)
    let a = if action.is_finite() { action.max(-1.0).min(1.0) } else { 0.0 };
    let t = (a + 1.0) * 0.5; // [0, 1]
    params.range[0] + t * (params.range[1] - params.range[0])
}

/// System that applies the current `CrabActions` to all crab joints.
pub fn apply_actions(
    actions: Res<CrabActions>,
    mut joints: Query<(&CrabJoint, &mut MultibodyJoint)>,
) {
    for (crab_joint, mut mj) in joints.iter_mut() {
        let idx = crab_joint.id.index();
        let action = actions.values[idx];
        let params = motor_params(&crab_joint.id);
        let target = action_to_target(action, &params);

        // Write motor target into the joint's generic data.
        let generic: &mut GenericJoint = mj.data.as_mut();

        // Determine the motor axis based on joint type.
        let axis = match &crab_joint.id {
            CrabJointId::ClawPincer(_) => JointAxis::LinX, // prismatic
            _ => JointAxis::AngX, // revolute (always the primary axis)
        };

        generic.set_motor_position(axis, target, params.stiffness, params.damping);
    }
}

/// Test system: generates sine-wave actions to verify motors work.
/// Add this system temporarily to see the crab wiggle.
#[allow(dead_code)]
pub fn test_wiggle(time: Res<Time>, mut actions: ResMut<CrabActions>) {
    let t = time.elapsed_secs();

    for i in 0..CrabJointId::COUNT {
        // Each joint gets a slightly different phase so it looks organic
        let phase = i as f32 * 0.5;
        let freq = 1.5; // Hz
        actions.values[i] = (t * freq * std::f32::consts::TAU + phase).sin() * 0.3;
    }
}
