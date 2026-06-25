//! `RL_RIG_POSE` headless-screenshot diagnostic: drive the cheliped SHOULDERS to a
//! constant action — so the actuator pins them against their travel limit — with the
//! carapace held in place, so a rig limit/axis change can be eyeballed in the exact
//! offending pose (the arms lifted to their up-stop). The pose the owner reported as
//! "arms lift up through the carapace" is `RL_RIG_POSE=-1` (−action lifts the arm; see
//! the actuator). Render-only, gated entirely on the env var; never wired into training.

use bevy::prelude::*;
use bevy_rapier3d::prelude::{ExternalForce, Velocity};

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabCarapace, CrabJointId, Side};

use super::claw_demo::pin_correction;

/// The constant shoulder action both chelipeds are driven with (clamped to [-1, 1] by
/// the actuator, which then holds the joint at its corresponding limit). `slots` are
/// resolved through [`CrabJointId::index`] so a rig reorder can't drive the wrong DOF.
#[derive(Resource, Clone, Copy)]
pub(super) struct RigPose {
    action: f32,
    slots: [usize; 2],
}

/// Anchor for the body pin: the carapace pose captured the first tick, held every step
/// after so the lone arm torque can't tip the unsupported (policy-less) body out of frame.
#[derive(Resource, Default)]
pub(super) struct RigPosePin {
    target: Option<Transform>,
}

/// `RL_RIG_POSE=<float>` enables the harness, driving both shoulders with that constant
/// action — `-1` parks them at the up-stop (the offending pose). Absent or unparseable →
/// `None` (a normal screenshot; the harness is never added).
pub(super) fn rig_pose_from_env() -> Option<RigPose> {
    let raw = std::env::var("RL_RIG_POSE").ok()?;
    let action = raw.trim().parse::<f32>().ok().filter(|a| a.is_finite())?;
    Some(RigPose {
        action,
        slots: [
            CrabJointId::ClawShoulder(Side::Left).index(),
            CrabJointId::ClawShoulder(Side::Right).index(),
        ],
    })
}

/// System (BotSet::Think, after `policy_step`): overwrite both shoulder action slots with
/// the constant drive, leaving every other slot as the (unloaded → zero) policy set it.
pub(super) fn rig_pose_drive(pose: Res<RigPose>, mut actions: ResMut<CrabActions>) {
    let Some(a) = actions.envs.first_mut() else {
        return;
    };
    for slot in pose.slots {
        a[slot] = pose.action;
    }
}

/// System (FixedUpdate, after Act, before the physics step): capture the carapace pose
/// once, then PD-hold it there (reusing the claw demo's correction) so the body stays
/// framed while the arms drive to their stop.
pub(super) fn rig_pose_pin(
    mut pin: ResMut<RigPosePin>,
    mut carapace_q: Query<(&Transform, &Velocity, &mut ExternalForce), With<CrabCarapace>>,
) {
    let Ok((xform, vel, mut force)) = carapace_q.single_mut() else {
        return;
    };
    let target = *pin.target.get_or_insert(*xform);
    let (f, t) = pin_correction(&target, xform, vel);
    force.force += f;
    force.torque += t;
}
