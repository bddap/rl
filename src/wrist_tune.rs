//! Interactive wrist axis/amplitude tuner for the demo (`RL_WRIST_TUNE=1`).
//!
//! The right claw's wrist sweeps the hand back and forth while the owner dials in
//! two things live, then saves them: the sweep AMPLITUDE (the reported feedback is
//! that the wrist rotates too far) and the rotation AXIS (in case the bend axis
//! itself needs nudging). The chosen values are written to a file the daemon reads
//! to bake them into the rig.
//!
//! Why KINEMATIC, not the torque drive the claw demo uses: this is a tuning loop on
//! a human's clock, and rebuilding the Rapier revolute joint to change the axis or
//! re-clamping the limit to change the amplitude would stutter and lose the feel.
//! Instead we rotate the wrist link's `Transform` directly about the joint pivot and
//! let the existing skin pipeline ([`crate::bot::skin::drive_bones`], which sets each
//! hand bone to `link_transform * offset`) carry the skinned hand — so every axis or
//! amplitude change is visible the same frame, no joint rebuild. The pincer link is
//! the wrist's kinematic descendant, so it rides the same rotation: the whole hand
//! (palm + dactyl) swings as one rigid unit about the wrist pivot, matching what the
//! real wrist joint does.
//!
//! The override runs AFTER Rapier's writeback (which would otherwise put the rest
//! pose back into the link `Transform`) and before the skin reads it in `PostUpdate`.
//! The body is held at rest — all actions zeroed, carapace PD-pinned (reusing the
//! claw demo's pin) — so only the hand moves. Note the override is cosmetic-only: it
//! does not reach the multibody's reduced coordinates (a position write on a
//! multibody link is overwritten by its forward kinematics, the same reason a
//! velocity write is a no-op there, #14), so with `RL_DEBUG_COLLIDERS` the wrist
//! collider cage stays at rest while the skinned hand sweeps. That's fine: the owner
//! is judging the rendered hand, and the saved axis/amplitude is what gets baked.
//!
//! Axis frame: the axis is expressed in the SAME frame as `rig::bend_axis` (the
//! parent/shoulder link frame, which equals world at the rest pose the body is held
//! in), so the unit vector saved here drops straight onto `bend_axis` when baked.

use std::f32::consts::TAU;
use std::io::Write;
use std::path::PathBuf;

use bevy::prelude::*;
use bevy_rapier3d::plugin::PhysicsSet;
use bevy_rapier3d::prelude::*;

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{CrabCarapace, CrabJoint, CrabJointId, CrabRestPose, Side};
use crate::bot::{BotSet, body::CrabBodyPart};
use crate::play::{PinBody, pin_correction};

/// The wrist the tuner drives. The owner reported the RIGHT wrist, and matching the
/// claw demo keeps the two inspection modes on the same claw.
const TUNED_WRIST: CrabJointId = CrabJointId::ClawWrist(Side::Right);
const TUNED_PINCER: CrabJointId = CrabJointId::ClawPincer(Side::Right);

/// Sweep frequency (Hz): the claw demo's default wrist frequency, so the back-and-
/// forth has the same slow, readable cadence the owner is already used to.
const SWEEP_HZ: f32 = 0.08;

/// Live-adjust rates (per second, while the key is held) — fast enough to dial in a
/// value on stream, fine enough to settle on one. Degrees because that's how the
/// owner reads amplitude and az/el off the overlay.
const AMP_RATE_DEG: f32 = 25.0;
const ANGLE_RATE_DEG: f32 = 40.0;

/// Default save path; overridden by `RL_WRIST_AXIS_OUT`. The daemon reads this file
/// after the owner saves, so the location is a contract between the two.
const DEFAULT_OUT: &str = "/home/a/rl-demo/chosen-wrist-axis.txt";

/// Which canonical axis the current direction is snapped to, for the overlay. Cleared
/// to `None` the moment the owner nudges az/el off it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Snap {
    None,
    WorldX,
    WorldY,
    WorldZ,
    Bend,
}

impl Snap {
    fn label(self) -> &'static str {
        match self {
            Snap::None => "none",
            Snap::WorldX => "world X",
            Snap::WorldY => "world Y",
            Snap::WorldZ => "world Z",
            Snap::Bend => "bend_axis",
        }
    }
}

/// The tuner's live state. Present only under `RL_WRIST_TUNE=1`; its presence is also
/// what tells the camera/demo-controls systems to yield the arrow keys (see
/// [`crate::play`]). `az`/`el` parametrize the sweep axis (the same unit vector also
/// shown and saved); amplitude is the half-sweep angle in radians.
///
/// Init is deferred to the first frame the live wrist joint exists ([`captured`]):
/// the resource is built before the crab spawns, but the seed values — the current
/// bend axis and the joint's limit — come off the spawned joint, so reading them
/// from the live body avoids a second model load and can't drift from the rig.
#[derive(Resource)]
pub struct WristTune {
    amplitude: f32,
    az: f32,
    el: f32,
    snap: Snap,
    /// The wrist's rig bend axis, captured at init: the "snap to original" target and
    /// the seed direction. World-frame at rest (= parent frame, the bend-axis frame).
    bend_axis: Vec3,
    /// Joint limit magnitude (rad); the amplitude ceiling and the init value, since
    /// the claw demo's unit-amplitude torque drive effectively sweeps the full limit.
    max_amp: f32,
    captured: bool,
    out_path: PathBuf,
}

impl Default for WristTune {
    fn default() -> Self {
        Self {
            // Real values are seeded in `capture` from the live joint; these stand in
            // only for the pre-spawn frames before `captured` flips.
            amplitude: 0.0,
            az: 0.0,
            el: 0.0,
            snap: Snap::Bend,
            bend_axis: Vec3::Y,
            max_amp: 1.0,
            captured: false,
            out_path: std::env::var_os("RL_WRIST_AXIS_OUT")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_OUT)),
        }
    }
}

/// Unit vector from azimuth (around +Y) and elevation (up from the XZ plane). Same
/// spherical convention as the screenshot camera (`yaw = atan2(x, z)`), so the axis
/// the overlay shows reads consistently with the rest of the demo.
fn axis_from_az_el(az: f32, el: f32) -> Vec3 {
    let (sa, ca) = az.sin_cos();
    let (se, ce) = el.sin_cos();
    Vec3::new(ce * sa, se, ce * ca)
}

/// Inverse of [`axis_from_az_el`]: recover az/el from a (not necessarily unit) axis,
/// so a snap target or the seed bend axis can be expressed as the az/el the live
/// controls then edit.
fn az_el_from_axis(axis: Vec3) -> (f32, f32) {
    let a = axis.normalize_or_zero();
    (a.x.atan2(a.z), a.y.clamp(-1.0, 1.0).asin())
}

impl WristTune {
    fn axis(&self) -> Vec3 {
        axis_from_az_el(self.az, self.el)
    }

    /// Point the axis at `target` and record the snap for the overlay.
    fn set_snap(&mut self, target: Vec3, snap: Snap) {
        let (az, el) = az_el_from_axis(target);
        self.az = az;
        self.el = el;
        self.snap = snap;
    }
}

/// Build the tuner only behind `RL_WRIST_TUNE=1`. Off, nothing here is registered and
/// the demo/headless/screenshot paths are untouched. Exclusive with the claw demo:
/// both pin the body and drive the same wrist, so the caller picks one.
pub struct WristTunePlugin;

impl Plugin for WristTunePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WristTune>()
            .init_resource::<PinBody>()
            .add_systems(Startup, spawn_overlay)
            // Input + overlay on the render clock (per-frame dt for the hold-to-adjust
            // rates), same as `orbit_camera`.
            .add_systems(Update, (tune_input, update_overlay, toggle_physics))
            // Hold the body at rest: zero every action so the policy can't flail the
            // trunk, then PD-pin the carapace after the actuator writes its baseline.
            // After Think (so it wins over the policy) and before Act (so the actuator
            // applies the zeros), the same slot `demo_settle` uses.
            .add_systems(
                FixedUpdate,
                zero_actions.after(BotSet::Think).before(BotSet::Act),
            )
            .add_systems(
                FixedUpdate,
                pin_body.after(BotSet::Act).before(PhysicsSet::SyncBackend),
            )
            // The kinematic override runs AFTER writeback (which restores the rest
            // pose into the link transforms) so the skin, reading in PostUpdate, sees
            // the swept pose — not the rest pose Rapier just wrote.
            .add_systems(FixedUpdate, drive_wrist.after(PhysicsSet::Writeback));
    }
}

/// Hold the whole body at its rest pose by commanding zero torque everywhere; the
/// kinematic override is the only thing that moves the hand.
fn zero_actions(mut actions: ResMut<CrabActions>) {
    if let Some(a) = actions.envs.first_mut() {
        *a = [0.0; ACTION_SIZE];
    }
}

/// PD-pin the carapace at the pose it settles into, reusing the claw demo's hold so
/// the trunk doesn't drift. Re-captures the target until the body has come to rest
/// (low velocity) so it anchors a settled pose, not the spawn transient.
fn pin_body(
    mut pin: ResMut<PinBody>,
    mut carapace_q: Query<(&Transform, &Velocity, &mut ExternalForce), With<CrabCarapace>>,
) {
    let Ok((xform, vel, mut force)) = carapace_q.single_mut() else {
        return;
    };
    // Settled = barely moving. Until then keep re-anchoring so a fresh spawn or a
    // reset (R) re-pins at the new rest pose instead of yanking the body back to a
    // stale target.
    if vel.linear.length() > 0.05 || vel.angular.length() > 0.05 {
        pin.target = Some(*xform);
        return;
    }
    let target = *pin.target.get_or_insert(*xform);
    let (f, t) = pin_correction(&target, xform, vel);
    force.force += f;
    force.torque += t;
}

/// Rotate the right wrist (and its kinematic descendant, the pincer link) about the
/// wrist pivot by `amplitude·sin(2π·f·t)` on the demo's wall clock. Reads each link's
/// REST pose (its captured spawn `Transform`, whose translation IS the joint pivot,
/// since links spawn at identity anchored with `local_anchor2 = 0`) so the pivot can't
/// drift from the spawn. The pincer rides rigidly: rotating about the shared pivot
/// carries the dactyl with the palm, so the hand swings as one piece.
fn drive_wrist(
    time: Res<Time>,
    mut tune: ResMut<WristTune>,
    mut links: Query<(&CrabJoint, &CrabRestPose, &mut Transform), With<CrabBodyPart>>,
) {
    capture(&mut tune, &links);

    let angle = tune.amplitude * (TAU * SWEEP_HZ * time.elapsed_secs()).sin();
    let q = Quat::from_axis_angle(tune.axis(), angle);

    // Rest pivots of the two links (translation of their spawn transforms).
    let mut wrist_pivot = None;
    let mut pincer_pivot = None;
    for (joint, rest, _) in links.iter() {
        if joint.id == TUNED_WRIST {
            wrist_pivot = Some(rest.0.translation);
        } else if joint.id == TUNED_PINCER {
            pincer_pivot = Some(rest.0.translation);
        }
    }
    let Some(wrist_pivot) = wrist_pivot else {
        return;
    };

    for (joint, _, mut tf) in links.iter_mut() {
        // The wrist's pivot is its own link origin, so it just spins in place; the
        // pincer rides the same rotation about the wrist pivot (a point off its
        // origin), so its origin orbits too. Both take the rotation `q`.
        let new_translation = if joint.id == TUNED_WRIST {
            Some(wrist_pivot)
        } else if joint.id == TUNED_PINCER {
            pincer_pivot.map(|p| wrist_pivot + q * (p - wrist_pivot))
        } else {
            None
        };
        if let Some(translation) = new_translation {
            tf.translation = translation;
            tf.rotation = q;
        }
    }
}

/// Seed the live values from the spawned wrist joint, once. Deferred from plugin
/// build because the joint doesn't exist until the crab spawns.
fn capture(
    tune: &mut WristTune,
    links: &Query<(&CrabJoint, &CrabRestPose, &mut Transform), With<CrabBodyPart>>,
) {
    if tune.captured {
        return;
    }
    for (joint, ..) in links.iter() {
        if joint.id == TUNED_WRIST {
            tune.bend_axis = joint.axis_local.normalize_or_zero();
            // Effective demo range: the claw demo drives unit amplitude and the
            // actuator clamps to the limit, so the wrist currently sweeps its full
            // limit. Start there (the owner dials it DOWN).
            tune.max_amp = TUNED_WRIST.limits()[1].abs();
            tune.amplitude = tune.max_amp;
            let (az, el) = az_el_from_axis(tune.bend_axis);
            tune.az = az;
            tune.el = el;
            tune.snap = Snap::Bend;
            tune.captured = true;
            return;
        }
    }
}

/// Live controls (per render frame, hold-to-adjust). Arrow keys steer the axis; the
/// camera and demo controls yield them while the tuner is active (see [`crate::play`]).
fn tune_input(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut tune: ResMut<WristTune>) {
    let dt = time.delta_secs();
    let amp_step = AMP_RATE_DEG.to_radians() * dt;
    let ang_step = ANGLE_RATE_DEG.to_radians() * dt;

    // Amplitude on [ / ], clamped to [0, joint limit].
    if keys.pressed(KeyCode::BracketRight) {
        tune.amplitude = (tune.amplitude + amp_step).min(tune.max_amp);
    }
    if keys.pressed(KeyCode::BracketLeft) {
        tune.amplitude = (tune.amplitude - amp_step).max(0.0);
    }

    // Axis on the arrow keys: left/right = azimuth, up/down = elevation. Any nudge
    // drops the snap label — the direction is now free.
    let mut d_az = 0.0;
    let mut d_el = 0.0;
    if keys.pressed(KeyCode::ArrowLeft) {
        d_az -= ang_step;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        d_az += ang_step;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        d_el += ang_step;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        d_el -= ang_step;
    }
    if d_az != 0.0 || d_el != 0.0 {
        tune.az += d_az;
        // Clamp elevation shy of the poles so azimuth stays meaningful.
        tune.el = (tune.el + d_el).clamp(-1.55, 1.55);
        tune.snap = Snap::None;
    }

    // Snap to a canonical axis (tap).
    if keys.just_pressed(KeyCode::KeyX) {
        tune.set_snap(Vec3::X, Snap::WorldX);
    }
    if keys.just_pressed(KeyCode::KeyY) {
        tune.set_snap(Vec3::Y, Snap::WorldY);
    }
    if keys.just_pressed(KeyCode::KeyZ) {
        tune.set_snap(Vec3::Z, Snap::WorldZ);
    }
    if keys.just_pressed(KeyCode::KeyB) {
        let bend = tune.bend_axis;
        tune.set_snap(bend, Snap::Bend);
    }

    // Save (tap): append + echo.
    if keys.just_pressed(KeyCode::Enter) {
        save(&tune);
    }
}

/// Freeze/unfreeze the physics step on `F`. With physics live the carapace PD-pin still
/// sags slightly under gravity, and since the hand is driven kinematically about the
/// wrist's fixed spawn pivot, a drifting body leaves the swept hand behind. Pausing
/// Rapier holds the whole body still for inspection; `drive_wrist` runs after writeback
/// independent of the step, so the hand keeps sweeping while frozen.
fn toggle_physics(keys: Res<ButtonInput<KeyCode>>, mut cfg: Query<&mut RapierConfiguration>) {
    if !keys.just_pressed(KeyCode::KeyF) {
        return;
    }
    if let Ok(mut cfg) = cfg.single_mut() {
        cfg.physics_pipeline_active = !cfg.physics_pipeline_active;
    }
}

/// Append the chosen values to the out file and echo the same line to stderr. Flushes
/// (and fsyncs) immediately: the daemon may read the file the instant the owner saves,
/// so a buffered write would race it.
fn save(tune: &WristTune) {
    let axis = tune.axis();
    let line = format!(
        "wrist: amplitude_deg={:.1}  axis=Vec3::new({:.5}, {:.5}, {:.5})  // az={:.1} el={:.1}",
        tune.amplitude.to_degrees(),
        axis.x,
        axis.y,
        axis.z,
        tune.az.to_degrees(),
        tune.el.to_degrees(),
    );
    eprintln!("{line}");
    if let Some(dir) = tune.out_path.parent() {
        // Best-effort: the daemon owns the deploy dir, but creating it keeps the save
        // from silently failing if the demo runs somewhere it doesn't exist yet.
        let _ = std::fs::create_dir_all(dir);
    }
    let write = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&tune.out_path)
        .and_then(|mut f| {
            writeln!(f, "{line}")?;
            f.flush()?;
            f.sync_all()
        });
    if let Err(e) = write {
        eprintln!(
            "wrist tuner: could not write {}: {e}",
            tune.out_path.display()
        );
    }
}

/// Marks the tuner's on-screen readout.
#[derive(Component)]
struct TuneOverlay;

fn spawn_overlay(mut commands: Commands) {
    commands.spawn((
        TuneOverlay,
        // Seeded so the pre-first-update frame isn't blank; overwritten every frame.
        Text::new("wrist tuner"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.7, 1.0, 0.8)),
        // Dark panel behind the text: the readout sits over the lit scene and pale
        // carapace, where light glyphs alone washed out (the owner couldn't read it).
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.7)),
        Node {
            position_type: PositionType::Absolute,
            // Below the joint-graph label (top-left) and clear of the manual HUD
            // (top-right); the main demo HUD owns the bottom-left.
            top: Val::Px(96.0),
            left: Val::Px(12.0),
            padding: UiRect::all(Val::Px(6.0)),
            ..default()
        },
    ));
}

fn update_overlay(
    tune: Res<WristTune>,
    cfg: Query<&RapierConfiguration>,
    mut overlay: Query<&mut Text, With<TuneOverlay>>,
) {
    let Ok(mut text) = overlay.single_mut() else {
        return;
    };
    let frozen = cfg
        .single()
        .map(|c| !c.physics_pipeline_active)
        .unwrap_or(false);
    let axis = tune.axis();
    **text = format!(
        "WRIST TUNER (RL_WRIST_TUNE)\n\
         amplitude: {amp:.1} deg   (max {max:.1})\n\
         axis: ({x:.3}, {y:.3}, {z:.3})   az {az:.1}  el {el:.1}   snap: {snap}\n\
         physics: {phys}\n\
         [ / ] amplitude    arrows: az/el    X Y Z: snap world    B: snap bend_axis    F: freeze\n\
         Enter: save to {out}",
        amp = tune.amplitude.to_degrees(),
        max = tune.max_amp.to_degrees(),
        x = axis.x,
        y = axis.y,
        z = axis.z,
        az = tune.az.to_degrees(),
        el = tune.el.to_degrees(),
        snap = tune.snap.label(),
        phys = if frozen {
            "FROZEN (F)"
        } else {
            "live (F to freeze)"
        },
        out = tune.out_path.display(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// az/el and the unit axis must round-trip for the directions the snaps and the
    /// seed use, so what the overlay shows and what gets saved agree with the axis the
    /// kinematic drive actually rotates about.
    #[test]
    fn axis_az_el_round_trip() {
        for v in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(0.02072, 0.94040, 0.33945), // the right wrist's bend axis
            Vec3::new(-0.3, 0.5, 0.8),
        ] {
            let unit = v.normalize();
            let (az, el) = az_el_from_axis(unit);
            let back = axis_from_az_el(az, el);
            assert!(
                back.distance(unit) < 1e-5,
                "round-trip failed for {v:?}: got {back:?}"
            );
        }
    }

    /// A snap points the axis exactly at its target and labels it; nudging az/el then
    /// clears the label (handled in `tune_input`, mirrored here on the state).
    #[test]
    fn snap_sets_axis_and_label() {
        let mut t = WristTune::default();
        t.set_snap(Vec3::X, Snap::WorldX);
        assert!(t.axis().distance(Vec3::X) < 1e-5);
        assert_eq!(t.snap, Snap::WorldX);

        t.set_snap(Vec3::Z, Snap::WorldZ);
        assert!(t.axis().distance(Vec3::Z) < 1e-5);
        assert_eq!(t.snap, Snap::WorldZ);
    }
}
