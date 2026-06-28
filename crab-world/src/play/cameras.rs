//! The demo's cameras: the windowed orbit camera (mouse/keyboard/gamepad free-look that
//! tracks the tumbling crab) and the windowless offscreen camera the screenshot mode
//! renders through.

use bevy::camera::RenderTarget;
use bevy::core_pipeline::tonemapping::Tonemapping;
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
            clear_color: ClearColorConfig::Custom(Color::srgb(0.45, 0.7, 0.95)),
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

/// Default screenshot eye position relative to the tracked focus, and the fixed
/// height of that focus. The single source for the camera's framing.
const SHOT_CAM_OFFSET: Vec3 = Vec3::new(1.9, 0.95, 2.5);
const SHOT_CAM_FOCUS_Y: f32 = 0.5;

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
    let focus = Vec3::new(crab.translation.x, SHOT_CAM_FOCUS_Y, crab.translation.z);
    *cam = Transform::from_translation(focus + SHOT_CAM_OFFSET).looking_at(focus, Vec3::Y);
}

pub(super) fn spawn_offscreen_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cfg: Res<ShotConfig>,
) {
    let handle = images.add(screenshot::new_render_target(cfg.width, cfg.height));

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.25, 0.45, 0.75)),
            ..default()
        },
        RenderTarget::Image(handle.clone().into()),
        // Default tonemapping needs a LUT asset that may not be ready in a
        // windowless render; None keeps the offscreen pass simple.
        Tonemapping::None,
        Transform::from_xyz(1.9, 1.15, 2.5).looking_at(Vec3::new(0.0, 0.35, 0.0), Vec3::Y),
        // Make UI composite into THIS offscreen target. Bevy auto-targets UI at the
        // default-WINDOW camera, but the screenshot path has no window — without this
        // marker the controls overlay never draws into the captured texture (same fix
        // net/render.rs's screenshot camera carries). The windowed demo doesn't need it:
        // its window camera is the implicit UI target.
        bevy::ui::IsDefaultUiCamera,
    ));
    commands.insert_resource(ShotTarget(handle));
}
