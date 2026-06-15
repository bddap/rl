//! Builds the observation vector from physics state.
//!
//! The observation vector contains:
//! - Per-joint: current joint angle and signed joint DOF velocity
//! - Body state: carapace position, orientation, linear/angular velocity
//!
//! For phase 1 (stand up), we don't need enemy state.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::CrabSpawns;
use super::body::{CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, joint_angle};

/// Total observation size.
/// Per-joint: 2 floats (joint_angle, signed joint DOF velocity)
/// Body: 3 (pos) + 4 (quat) + 3 (linvel) + 3 (angvel) = 13
pub const OBS_SIZE: usize = CrabJointId::COUNT * 2 + 13;

/// Resource holding the current observation vector for each environment.
#[derive(Resource, Default)]
pub struct CrabObservation {
    /// `envs[e]` = env e's observation.
    pub envs: Vec<[f32; OBS_SIZE]>,
}

impl CrabObservation {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![[0.0; OBS_SIZE]; n];
    }
}

/// System that builds every env's observation vector each frame.
pub fn build_observation(
    spawns: Res<CrabSpawns>,
    mut obs: ResMut<CrabObservation>,
    carapace_q: Query<(Entity, &CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>,
    joint_q: Query<(
        Entity,
        &CrabJoint,
        &CrabEnvId,
        &MultibodyJoint,
        &Transform,
        &Velocity,
    )>,
) {
    for v in obs.envs.iter_mut() {
        *v = [0.0; OBS_SIZE];
    }

    // World rotation + velocity of every part, keyed by entity. A joint reads its
    // PARENT's motion from here: the velocity input is the signed DOF rate (the
    // joint coordinate's velocity), which is the part's motion RELATIVE to its
    // parent about the joint axis — that needs the parent's world rotation and
    // velocity. The carapace is included because it parents the coxae/claws/eyes.
    let mut motion: HashMap<Entity, (Quat, Vec3, Vec3)> = HashMap::new();
    for (e, _, _, _, t, vel) in joint_q.iter() {
        motion.insert(e, (t.rotation, vel.linear, vel.angular));
    }
    for (e, _, t, vel) in carapace_q.iter() {
        motion.insert(e, (t.rotation, vel.linear, vel.angular));
    }

    // -- Per-joint observations ------------------------------------------------
    for (_, crab_joint, env, mj, transform, vel) in joint_q.iter() {
        let Some(v) = obs.envs.get_mut(env.0) else {
            continue;
        };
        let idx = crab_joint.id.index();
        let base = idx * 2;

        let (parent_rot, parent_lin, parent_ang) =
            motion
                .get(&mj.parent)
                .copied()
                .unwrap_or((Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO));

        // Current joint angle (the coordinate the policy controls by torque).
        v[base] = joint_angle(crab_joint.id, parent_rot, transform.rotation);

        // SIGNED joint DOF rate = d(angle)/dt: the part's velocity relative to its
        // parent, projected onto the joint axis in world. Signed (not a magnitude)
        // so the policy can tell which way a joint is moving and damp it — an
        // unsigned magnitude hides the direction the controller needs to oppose.
        let axis_world = parent_rot * crab_joint.id.joint_axis_local();
        v[base + 1] = match &crab_joint.id {
            CrabJointId::ClawPincer(_) => (vel.linear - parent_lin).dot(axis_world),
            _ => (vel.angular - parent_ang).dot(axis_world),
        };
    }

    // -- Body state (carapace) -------------------------------------------------
    let body_base = CrabJointId::COUNT * 2;

    for (_, env, transform, vel) in carapace_q.iter() {
        let Some(v) = obs.envs.get_mut(env.0) else {
            continue;
        };
        // Position relative to this env's spawn origin: every crab observes
        // "how far have I drifted", not its absolute grid slot — otherwise x/z
        // would encode env identity and the policy could specialise per slot.
        let origin = spawns.0.get(env.0).copied().unwrap_or(Vec3::ZERO);
        let pos = transform.translation - origin;
        v[body_base] = pos.x;
        v[body_base + 1] = pos.y;
        v[body_base + 2] = pos.z;

        let rot = transform.rotation;
        v[body_base + 3] = rot.x;
        v[body_base + 4] = rot.y;
        v[body_base + 5] = rot.z;
        v[body_base + 6] = rot.w;

        v[body_base + 7] = vel.linear.x;
        v[body_base + 8] = vel.linear.y;
        v[body_base + 9] = vel.linear.z;

        v[body_base + 10] = vel.angular.x;
        v[body_base + 11] = vel.angular.y;
        v[body_base + 12] = vel.angular.z;
    }

    // Sanitize: replace any NaN/Inf with 0 to prevent NN corruption
    for v in obs.envs.iter_mut() {
        for val in v.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }
    }
}
