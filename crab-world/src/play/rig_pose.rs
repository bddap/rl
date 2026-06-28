//! `RL_RIG_POSE` headless-screenshot diagnostic: drive a chosen joint FAMILY to a
//! constant action — so the actuator pins those joints against their travel limit —
//! with the carapace held in place, so a rig limit/axis change can be eyeballed in the
//! exact pose it produces. `RL_RIG_POSE_PART` selects the family: `shoulder` (default,
//! both chelipeds — the "arms lift up through the carapace" check, `RL_RIG_POSE=-1`) or
//! `legbasis` (all eight leg LIFT DOFs, to verify the 2-DOF coxa articulates —
//! bddap/rl#31). Render-only, gated entirely on the env var; never wired into training.

use bevy::prelude::*;
use bevy_rapier3d::prelude::{ExternalForce, Velocity};

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabCarapace, CrabJointId, Side};

/// The constant action a chosen joint family is driven with (clamped to [-1, 1] by the
/// actuator, which then holds each joint at its corresponding limit). `slots` are
/// resolved through [`CrabJointId::index`] so a rig reorder can't drive the wrong DOF.
#[derive(Resource, Clone)]
pub(super) struct RigPose {
    action: f32,
    slots: Vec<usize>,
}

/// Anchor for the body pin: the carapace pose captured the first tick, held every step
/// after so the lone arm torque can't tip the unsupported (policy-less) body out of frame.
#[derive(Resource, Default)]
pub(super) struct RigPosePin {
    target: Option<Transform>,
}

/// `RL_RIG_POSE=<float>` enables the harness, driving the family named by
/// `RL_RIG_POSE_PART` with that constant action. Absent or unparseable `RL_RIG_POSE` →
/// `None` (a normal screenshot; the harness is never added).
pub(super) fn rig_pose_from_env() -> Option<RigPose> {
    let raw = std::env::var("RL_RIG_POSE").ok()?;
    let action = raw.trim().parse::<f32>().ok().filter(|a| a.is_finite())?;
    let part = std::env::var("RL_RIG_POSE_PART").unwrap_or_default();
    let slots = match part.trim() {
        "legbasis" => (0..4)
            .flat_map(|leg| {
                [
                    CrabJointId::LegBasis(Side::Left, leg).index(),
                    CrabJointId::LegBasis(Side::Right, leg).index(),
                ]
            })
            .collect(),
        // Default (empty or "shoulder"): both chelipeds, the original behavior. A
        // non-empty but UNRECOGNIZED value is almost certainly a typo, and silently
        // diagnosing the wrong joint wastes a screenshot — so say so, loudly.
        other => {
            if !other.is_empty() && other != "shoulder" {
                eprintln!("RL_RIG_POSE_PART={other:?} unrecognized — driving the shoulders");
            }
            vec![
                CrabJointId::ClawShoulder(Side::Left).index(),
                CrabJointId::ClawShoulder(Side::Right).index(),
            ]
        }
    };
    Some(RigPose { action, slots })
}

/// System (BotSet::Think, after `policy_step`): overwrite both shoulder action slots with
/// the constant drive, leaving every other slot as the (unloaded → zero) policy set it.
pub(super) fn rig_pose_drive(pose: Res<RigPose>, mut actions: ResMut<CrabActions>) {
    let Some(a) = actions.envs.first_mut() else {
        return;
    };
    for &slot in &pose.slots {
        a[slot] = pose.action;
    }
}

/// System (FixedUpdate, after Act, before the physics step): capture the carapace pose
/// once, then PD-hold it there ([`pin_correction`]) so the body stays framed while the
/// arms drive to their stop.
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

/// PD gains for the carapace hold. The damping (KD) terms do the real work — arresting
/// the slow yaw/drift the lone arm torque induces — so they dominate; the restoring (KP)
/// terms only nudge the trunk back to where it settled. Both corrections are then clamped
/// (`PIN_MAX_*`) so no transient at the moment of capture can fling the light multibody
/// root out of frame.
const PIN_ROT_KP: f32 = 20.0;
const PIN_ROT_KD: f32 = 12.0;
const PIN_POS_KP: f32 = 60.0;
const PIN_POS_KD: f32 = 30.0;
const PIN_MAX_TORQUE: f32 = 12.0;
const PIN_MAX_FORCE: f32 = 120.0;

/// The clamped corrective `(force, torque)` that drives the carapace from its current
/// pose/velocity back toward `target`. The correction is an *external force/torque*, not
/// a velocity write or a body-type swap: on a Rapier multibody root only forces reach the
/// body (velocity writeback is a no-op, issue #14) and flipping the root to
/// `RigidBody::Fixed` mid-sim NaNs the solver. Per-body `Damping` is likewise ignored on a
/// multibody root, so a PD hold via `ExternalForce` is the one channel that actually pins
/// the trunk. Caller *adds* this onto `ExternalForce` after the actuator's baseline; see
/// [`rig_pose_pin`].
fn pin_correction(target: &Transform, xform: &Transform, vel: &Velocity) -> (Vec3, Vec3) {
    // Rotational PD: error as the axis-angle of the rotation that takes the current
    // orientation to the target, fed back against the current angular velocity.
    let err_rot = target.rotation * xform.rotation.inverse();
    let (axis, angle) = err_rot.to_axis_angle();
    let angle = if angle > std::f32::consts::PI {
        angle - std::f32::consts::TAU
    } else {
        angle
    };
    let torque =
        (axis * angle * PIN_ROT_KP - vel.angular * PIN_ROT_KD).clamp_length_max(PIN_MAX_TORQUE);

    // Positional PD: hold the trunk where it settled (catches any lateral skating).
    let err_pos = target.translation - xform.translation;
    let force = (err_pos * PIN_POS_KP - vel.linear * PIN_POS_KD).clamp_length_max(PIN_MAX_FORCE);
    (force, torque)
}
