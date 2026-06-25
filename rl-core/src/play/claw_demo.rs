//! The interactive right-claw inspection sweep (`RL_CLAW_DEMO=1`): a wrist+pincer sin/cos
//! drive plus a body-pin, on the demo's wall-clock, so a viewer can orbit/zoom a live,
//! continuously articulating claw. The pin holds the carapace still so only the claw moves.

use bevy::prelude::*;
use bevy_rapier3d::prelude::{ExternalForce, Velocity};

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabCarapace, CrabJointId, Side};

use super::demo::DemoSettle;
use super::manual_control::ManualControl;

/// Interactive right-claw inspection sweep (present only with `RL_CLAW_DEMO=1`): a
/// wrist+pincer sin/cos drive plus a body-pin, on the demo's wall-clock, so a viewer
/// can orbit/zoom a live, continuously articulating claw. `wrist`/`pincer` are resolved
/// through
/// [`CrabJointId::index`] (not hardcoded) so a rig reorder can't silently drive the
/// wrong slots. `f1`/`f2` (Hz, `RL_CLAW_DEMO_F1`/`F2`) are deliberately low and
/// unequal so the two DOFs trace a slow, non-repeating figure that reads clearly.
#[derive(Resource, Clone, Copy)]
pub(super) struct ClawDemo {
    f1: f32,
    f2: f32,
    wrist: usize,
    pincer: usize,
}

/// `RL_CLAW_DEMO=1` enables the sweep; `RL_CLAW_DEMO_F1`/`F2` override the wrist/pincer
/// frequencies (Hz; defaults 0.08 / 0.12). Any other/absent `RL_CLAW_DEMO` value → None
/// (policy drives normally).
pub(super) fn claw_demo_from_env() -> Option<ClawDemo> {
    if std::env::var("RL_CLAW_DEMO").ok()?.trim() != "1" {
        return None;
    }
    let freq = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or(d)
    };
    Some(ClawDemo {
        f1: freq("RL_CLAW_DEMO_F1", 0.08),
        f2: freq("RL_CLAW_DEMO_F2", 0.12),
        wrist: CrabJointId::ClawWrist(Side::Right).index(),
        pincer: CrabJointId::ClawPincer(Side::Right).index(),
    })
}

/// System (BotSet::Think, after `policy_step` and `demo_settle`): while the claw demo
/// is active and the fresh crab has finished settling, overwrite just the right wrist
/// and pincer action slots with `sin(2π·f1·t)` / `cos(2π·f2·t)` on Bevy's continuous
/// `Time` clock, leaving every other slot as the policy set it. `t` is the elapsed
/// wall-clock, so the sweep loops indefinitely at real-time speed regardless of frame
/// rate. Held off during the settle so the claw doesn't fight the body taking load,
/// and while hands-on manual control is active so the operator owns the joints.
pub(super) fn claw_demo_drive(
    demo: Res<ClawDemo>,
    settle: Res<DemoSettle>,
    manual: Option<Res<ManualControl>>,
    time: Res<Time>,
    mut actions: ResMut<CrabActions>,
) {
    if settle.0 > 0 || manual.is_some_and(|m| m.active) {
        return;
    }
    let Some(a) = actions.envs.first_mut() else {
        return;
    };
    let t = time.elapsed_secs();
    // The actuator clamps to the joint limits, so a unit-amplitude command sweeps each
    // DOF across its full range; no per-joint scaling needed here.
    a[demo.wrist] = (std::f32::consts::TAU * demo.f1 * t).sin();
    a[demo.pincer] = (std::f32::consts::TAU * demo.f2 * t).cos();
}

/// System (FixedUpdate, after `BotSet::Act`, before the physics step): the claw demo's
/// body pin. Captures the carapace pose once the demo settles and PD-corrects it back
/// every step ([`pin_correction`]), re-anchored on every reset — a respawn restarts the
/// settle, so the stale target is dropped and recaptured at the new rest pose. Without
/// this the one-claw torque slowly yaws the light crab and the articulation is lost;
/// see [`PinBody`].
pub(super) fn claw_demo_pin(
    settle: Res<DemoSettle>,
    mut pin: ResMut<PinBody>,
    mut carapace_q: Query<(&Transform, &Velocity, &mut ExternalForce), With<CrabCarapace>>,
) {
    if settle.0 > 0 {
        // Still settling (fresh spawn or post-reset): drop any stale anchor so the
        // target is recaptured at the pose the crab actually settles into this time.
        pin.target = None;
        return;
    }
    let Ok((xform, vel, mut force)) = carapace_q.single_mut() else {
        return;
    };
    let target = *pin.target.get_or_insert(*xform);
    let (f, t) = pin_correction(&target, xform, vel);
    force.force += f;
    force.torque += t;
}

/// Anchor target for the interactive claw demo's body pin ([`claw_demo_pin`]).
/// Driving one claw's torque reaction-torques the lightweight carapace, and on the
/// low-friction ground the whole crab slowly yaws/drifts — that body motion masks
/// the claw articulation. So we hold the carapace at the pose it settled into and
/// PD-correct it back each step. The correction is an *external force/torque*, not
/// a velocity write or a body-type swap: on a Rapier multibody root only forces
/// reach the body (velocity writeback is a no-op, issue #14) and flipping the root
/// to `RigidBody::Fixed` mid-sim NaNs the solver. Per-body `Damping` is likewise
/// ignored on a multibody root, so a PD hold via `ExternalForce` is the one channel
/// that actually pins the trunk. `target` is captured once, the render frame the
/// settle completes; `None` until then.
#[derive(Resource, Default)]
pub(super) struct PinBody {
    target: Option<Transform>,
}

/// PD gains for the carapace hold. The damping (KD) terms do the real work —
/// arresting the slow yaw/drift the claw induces — so they dominate; the restoring
/// (KP) terms only nudge the trunk back to where it settled. Both corrections are
/// then clamped (`PIN_MAX_*`) so no transient at the moment of capture can fling the
/// light multibody root out of frame. Calibrated against the demo poke
/// (force 70 / torque 4 visibly moves the body), so a hold lives in that ballpark.
const PIN_ROT_KP: f32 = 20.0;
const PIN_ROT_KD: f32 = 12.0;
const PIN_POS_KP: f32 = 60.0;
const PIN_POS_KD: f32 = 30.0;
const PIN_MAX_TORQUE: f32 = 12.0;
const PIN_MAX_FORCE: f32 = 120.0;

/// The clamped corrective `(force, torque)` that drives the carapace from its
/// current pose/velocity back toward `target` — the PD hold the interactive claw demo
/// uses to keep the body still while one claw articulates. Caller *adds* this onto
/// `ExternalForce` after the actuator has written the baseline; see [`claw_demo_pin`].
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
