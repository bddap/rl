//! The demo's cameras: the windowed orbit camera (mouse/keyboard/gamepad free-look that
//! tracks the tumbling crab) and the windowless offscreen camera the screenshot mode
//! renders through.

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

use crate::bot::body::CrabCarapace;
use crate::screenshot::{self, ShotTarget};

use super::ShotConfig;

/// Orbit camera state. `focus` tracks the crab so it stays centered.
#[derive(Component)]
pub(super) struct OrbitCamera {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        // Matches the screenshot framing: a 3/4 view from slightly above.
        Self {
            focus: Vec3::new(0.0, 0.4, 0.0),
            yaw: 0.64,
            pitch: 0.32,
            radius: 3.2,
        }
    }
}

pub(super) fn spawn_orbit_camera(mut commands: Commands) {
    let orbit = OrbitCamera::default();
    commands.spawn((
        camera_transform(&orbit),
        Camera3d::default(),
        Camera {
            // Dark fallback behind the night-sky skybox (see [`crate::sky`]); shown only
            // until the cubemap uploads.
            clear_color: ClearColorConfig::Custom(crate::sky::NIGHT_CLEAR),
            ..default()
        },
        orbit,
    ));
}

fn camera_transform(orbit: &OrbitCamera) -> Transform {
    let rot =
        Quat::from_axis_angle(Vec3::Y, orbit.yaw) * Quat::from_axis_angle(Vec3::X, -orbit.pitch);
    let eye = orbit.focus + rot * Vec3::new(0.0, 0.0, orbit.radius);
    Transform::from_translation(eye).looking_at(orbit.focus, Vec3::Y)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn orbit_camera(
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    carapace_q: Query<&Transform, (With<CrabCarapace>, Without<OrbitCamera>)>,
    mut cam_q: Query<(&mut OrbitCamera, &mut Transform), Without<CrabCarapace>>,
) {
    let Ok((mut orbit, mut transform)) = cam_q.single_mut() else {
        return;
    };
    let dt = time.delta_secs();
    let (mut d_yaw, mut d_pitch, mut d_zoom) = (0.0f32, 0.0f32, 0.0f32);

    // Mouse: right-drag to orbit, wheel to zoom.
    if mouse.pressed(MouseButton::Right) {
        for ev in motion.read() {
            d_yaw -= ev.delta.x * 0.006;
            d_pitch -= ev.delta.y * 0.006;
        }
    } else {
        motion.clear();
    }
    for ev in wheel.read() {
        d_zoom -= ev.y * 0.4;
    }

    // Keyboard orbit; -/= zoom. Right-yaw is the comma key, not the right arrow:
    // the right arrow toggles the collider wireframes (see `demo::demo_controls`), and
    // mouse right-drag already covers free-look orbiting in every direction.
    if keys.pressed(KeyCode::ArrowLeft) {
        d_yaw += dt;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        d_pitch += dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        d_pitch -= dt;
    }
    if keys.pressed(KeyCode::Comma) {
        d_yaw -= dt;
    }
    if keys.pressed(KeyCode::Minus) {
        d_zoom += dt * 3.0;
    }
    if keys.pressed(KeyCode::Equal) {
        d_zoom -= dt * 3.0;
    }

    // Gamepad: left stick orbits, triggers zoom. (The RIGHT stick is reserved for
    // --manual-control joint actuation; left-stick orbit is the convention anyway.)
    for gp in gamepads.iter() {
        let ls = gp.left_stick();
        if ls.length() > 0.15 {
            d_yaw -= ls.x * dt * 2.5;
            d_pitch += ls.y * dt * 2.5;
        }
        if gp.pressed(GamepadButton::RightTrigger2) {
            d_zoom += dt * 4.0;
        }
        if gp.pressed(GamepadButton::LeftTrigger2) {
            d_zoom -= dt * 4.0;
        }
    }

    orbit.yaw += d_yaw;
    orbit.pitch = (orbit.pitch + d_pitch).clamp(-1.3, 1.4);
    orbit.radius = (orbit.radius + d_zoom).clamp(1.0, 12.0);

    // Smoothly keep the (possibly tumbling) crab centered.
    if let Ok(crab) = carapace_q.single() {
        orbit.focus = orbit.focus.lerp(crab.translation, (dt * 4.0).min(1.0));
    }

    *transform = camera_transform(&orbit);
}

/// Screenshot-camera framing: the eye sits at this fixed offset from a focus
/// pinned to the crab's horizontal position at this fixed mid-body height.
const SHOT_CAM_OFFSET: Vec3 = Vec3::new(1.9, 0.95, 2.5);
const SHOT_CAM_FOCUS_Y: f32 = 0.5;

/// The single source for the offscreen camera's framing transform: focus pinned
/// to the crab's horizontal position (`crab_xz.x`/`.z`) at the fixed mid-body
/// height, eye at the fixed offset, looking at the focus. Both the initial spawn
/// and the per-frame tracking go through here so the two can't drift.
fn offscreen_camera_transform(crab_xz: Vec3) -> Transform {
    let focus = Vec3::new(crab_xz.x, SHOT_CAM_FOCUS_Y, crab_xz.z);
    Transform::from_translation(focus + SHOT_CAM_OFFSET).looking_at(focus, Vec3::Y)
}

/// Keep the offscreen camera aimed at the (possibly drifting) crab so it stays
/// centered in the screenshot. Tracks horizontal position only; the vertical
/// focus is fixed mid-body so framing doesn't bob.
pub(super) fn track_offscreen_camera(
    carapace_q: Query<&Transform, (With<CrabCarapace>, Without<Camera3d>)>,
    mut cam_q: Query<&mut Transform, With<Camera3d>>,
) {
    let (Ok(crab), Ok(mut cam)) = (carapace_q.single(), cam_q.single_mut()) else {
        return;
    };
    *cam = offscreen_camera_transform(crab.translation);
}

pub(super) fn spawn_offscreen_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cfg: Res<ShotConfig>,
) {
    let handle = images.add(screenshot::new_render_target(cfg.width, cfg.height));

    commands.spawn((
        screenshot::offscreen_camera_bundle(handle.clone()),
        // Initial framing on the crab's start position (origin). `track_offscreen_camera`
        // re-derives this every Update from the same source before any frame is captured.
        offscreen_camera_transform(Vec3::ZERO),
    ));
    commands.insert_resource(ShotTarget(handle));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical framing pins the focus to the crab's horizontal position at the
    /// fixed mid-body height and seats the eye at the fixed offset above it. Locks the
    /// single source so a future edit can't silently re-introduce a drifting literal.
    #[test]
    fn offscreen_framing_is_canonical() {
        let t = offscreen_camera_transform(Vec3::new(3.0, 99.0, -4.0));
        let focus = Vec3::new(3.0, SHOT_CAM_FOCUS_Y, -4.0);
        assert_eq!(t.translation, focus + SHOT_CAM_OFFSET);
        // Camera looks at the focus: forward points from eye to focus.
        let fwd = t.forward().as_vec3();
        assert!((fwd - (focus - t.translation).normalize()).length() < 1e-5);
    }
}
