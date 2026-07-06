
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

use crate::bot::body::CrabCarapace;
use crate::screenshot::{self, ShotTarget};

use super::ShotConfig;
use super::controls::{
    ORBIT_DRAG_BUTTON, ORBIT_PITCH_DOWN_KEY, ORBIT_PITCH_UP_KEY, ORBIT_YAW_LEFT_KEY,
    ORBIT_YAW_RIGHT_KEY, ZOOM_IN_KEY, ZOOM_IN_TRIGGER, ZOOM_OUT_KEY, ZOOM_OUT_TRIGGER, orbit_stick,
};

#[derive(Component)]
pub(super) struct OrbitCamera {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
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

    // Mouse: drag to orbit, wheel to zoom.
    if mouse.pressed(ORBIT_DRAG_BUTTON) {
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

    // Keyboard orbit + zoom (key choices and their why live beside DEMO_BINDINGS).
    if keys.pressed(ORBIT_YAW_LEFT_KEY) {
        d_yaw += dt;
    }
    if keys.pressed(ORBIT_PITCH_UP_KEY) {
        d_pitch += dt;
    }
    if keys.pressed(ORBIT_PITCH_DOWN_KEY) {
        d_pitch -= dt;
    }
    if keys.pressed(ORBIT_YAW_RIGHT_KEY) {
        d_yaw -= dt;
    }
    if keys.pressed(ZOOM_OUT_KEY) {
        d_zoom += dt * 3.0;
    }
    if keys.pressed(ZOOM_IN_KEY) {
        d_zoom -= dt * 3.0;
    }

    // Gamepad: stick orbits, triggers zoom.
    for gp in gamepads.iter() {
        let ls = orbit_stick(gp);
        if ls.length() > 0.15 {
            d_yaw -= ls.x * dt * 2.5;
            d_pitch += ls.y * dt * 2.5;
        }
        if gp.pressed(ZOOM_OUT_TRIGGER) {
            d_zoom += dt * 4.0;
        }
        if gp.pressed(ZOOM_IN_TRIGGER) {
            d_zoom -= dt * 4.0;
        }
    }

    orbit.yaw += d_yaw;
    orbit.pitch = (orbit.pitch + d_pitch).clamp(-1.3, 1.4);
    orbit.radius = (orbit.radius + d_zoom).clamp(1.0, 12.0);

    if let Ok(crab) = carapace_q.single() {
        orbit.focus = orbit.focus.lerp(crab.translation, (dt * 4.0).min(1.0));
    }

    *transform = camera_transform(&orbit);
}

const SHOT_CAM_OFFSET: Vec3 = Vec3::new(1.9, 0.95, 2.5);
const SHOT_CAM_FOCUS_Y: f32 = 0.5;

fn offscreen_camera_transform(crab_xz: Vec3) -> Transform {
    let focus = Vec3::new(crab_xz.x, SHOT_CAM_FOCUS_Y, crab_xz.z);
    Transform::from_translation(focus + SHOT_CAM_OFFSET).looking_at(focus, Vec3::Y)
}

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
        offscreen_camera_transform(Vec3::ZERO),
    ));
    commands.insert_resource(ShotTarget(handle));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offscreen_framing_is_canonical() {
        let t = offscreen_camera_transform(Vec3::new(3.0, 99.0, -4.0));
        let focus = Vec3::new(3.0, SHOT_CAM_FOCUS_Y, -4.0);
        assert_eq!(t.translation, focus + SHOT_CAM_OFFSET);
        let fwd = t.forward().as_vec3();
        assert!((fwd - (focus - t.translation).normalize()).length() < 1e-5);
    }
}
