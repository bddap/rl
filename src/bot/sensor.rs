//! Builds the observation vector from physics state.
//!
//! The observation vector contains:
//! - Per-joint: target position (last action), current angular velocity (3 floats per joint
//!   would be ideal, but we approximate with 1 relevant component)
//! - Body state: carapace position, orientation, linear/angular velocity
//!
//! For phase 1 (stand up), we don't need enemy state.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::actuator::CrabActions;
use super::body::{CrabCarapace, CrabJoint, CrabJointId};

/// Total observation size.
/// Per-joint: 2 floats (last_action, joint_velocity_magnitude)
/// Body: 3 (pos) + 4 (quat) + 3 (linvel) + 3 (angvel) = 13
pub const OBS_SIZE: usize = CrabJointId::COUNT * 2 + 13;

/// Resource holding the current observation vector.
#[derive(Resource)]
pub struct CrabObservation {
    pub values: [f32; OBS_SIZE],
}

impl Default for CrabObservation {
    fn default() -> Self {
        Self {
            values: [0.0; OBS_SIZE],
        }
    }
}

/// System that builds the observation vector each frame.
pub fn build_observation(
    actions: Res<CrabActions>,
    mut obs: ResMut<CrabObservation>,
    carapace_q: Query<(&Transform, &Velocity), With<CrabCarapace>>,
    joint_q: Query<(&CrabJoint, &Velocity)>,
) {
    let mut v = [0.0f32; OBS_SIZE];

    // -- Per-joint observations ------------------------------------------------
    for (crab_joint, vel) in joint_q.iter() {
        let idx = crab_joint.id.index();
        let base = idx * 2;

        // Last action (what the NN commanded)
        v[base] = actions.values[idx];

        // Joint velocity: use the DOF-appropriate velocity.
        // Prismatic joints (ClawPincer) → linear velocity magnitude.
        // Revolute joints (everything else) → angular velocity magnitude.
        v[base + 1] = match &crab_joint.id {
            CrabJointId::ClawPincer(_) => vel.linvel.length(),
            _ => vel.angvel.length(),
        };
    }

    // -- Body state (carapace) -------------------------------------------------
    let body_base = CrabJointId::COUNT * 2;

    if let Ok((transform, vel)) = carapace_q.single() {
        let pos = transform.translation;
        v[body_base] = pos.x;
        v[body_base + 1] = pos.y;
        v[body_base + 2] = pos.z;

        let rot = transform.rotation;
        v[body_base + 3] = rot.x;
        v[body_base + 4] = rot.y;
        v[body_base + 5] = rot.z;
        v[body_base + 6] = rot.w;

        v[body_base + 7] = vel.linvel.x;
        v[body_base + 8] = vel.linvel.y;
        v[body_base + 9] = vel.linvel.z;

        v[body_base + 10] = vel.angvel.x;
        v[body_base + 11] = vel.angvel.y;
        v[body_base + 12] = vel.angvel.z;
    }

    // Sanitize: replace any NaN/Inf with 0 to prevent NN corruption
    for val in v.iter_mut() {
        if val.is_nan() || val.is_infinite() {
            *val = 0.0;
        }
    }

    obs.values = v;
}
