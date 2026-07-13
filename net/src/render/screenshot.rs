use super::app::{install_armed_nn_crab, seed_round_crabs};
use super::driver::{
    FlightInput, PendingInput, ScriptedPackInput, coordinator, drive_client_sim, insert_core,
};
use super::hud::{spawn_hud, sync_controls_context, update_hud};
use super::input::gather_input;
use super::scene::{FpCamera, apply_transforms, follow_ground, reconcile_avatars, spawn_world};
use super::*;
use crate::net_loop::NetDriver;
use crab_world::screenshot::{self, ShotProgress, ShotTarget};

pub fn build_screenshot_app(
    mut client: ClientSim,
    cfg: ScreenshotConfig,
    external_crab: Option<crab_world::policy::Policy>,
    render_mode: super::RenderMode,
    pack: Input,
) -> App {
    let mut app = offscreen_app_scaffold();
    let armed_crab = external_crab.map(|policy| (policy, seed_round_crabs(&mut client, 1)));
    let coord = coordinator(None, client.peers(), client.me(), client.sim().clone());
    insert_core(&mut app, client, coord);
    app.insert_resource(ScriptedPackInput(pack));
    if let Some((policy, spawns)) = armed_crab {
        install_armed_nn_crab(&mut app, vec![policy], spawns);
    }
    finish_offscreen_app(&mut app, cfg, render_mode);
    app
}

pub fn build_net_screenshot_app(
    mut client: ClientSim,
    net: NetDriver,
    cfg: ScreenshotConfig,
    external_crab: crab_world::policy::Policy,
    render_mode: super::RenderMode,
) -> App {
    let mut app = offscreen_app_scaffold();
    let spawns = seed_round_crabs(&mut client, 1);
    let coord = coordinator(Some(net), client.peers(), client.me(), client.sim().clone());
    insert_core(&mut app, client, coord);
    install_armed_nn_crab(&mut app, vec![external_crab], spawns);
    finish_offscreen_app(&mut app, cfg, render_mode);
    app
}

fn offscreen_app_scaffold() -> App {
    let mut app = App::new();
    app.add_plugins(crab_world::app_boot::base_plugins(None));
    // The screenshot app has no menu: its round is installed before the first frame, so it is
    // in [`AppPhase::Playing`] for its whole life (one boot-time enter, never an exit). Saying
    // so keeps every Playing-gated system — the render-mode gizmos above all (rl#211) — on the
    // ONE stock `in_state` idiom, fail-closed: were the state merely absent, `in_state` would
    // silently return false and collider-view evidence shots would capture no cage.
    app.insert_state(super::AppPhase::Playing);
    app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
        Duration::from_secs_f64(1.0 / 60.0),
    ));
    app.add_plugins(crab_world::sky::NightSkyPlugin);
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(TICK_DT),
    ));
    // Everywhere else `ExternalCrabPlugin` provides this, but the crab-less screenshot path
    // (`fp-screenshot` without a checkpoint) skips the plugin, and `apply_transforms` /
    // `draw_vehicle_collider_wireframe` take it as a hard `Res` (rl#224).
    app.init_resource::<crate::external_crab::ArenaAnchor>();
    app
}

/// Wire the offscreen screenshot systems + render-mode onto a scaffolded app whose round is
/// already installed — shared by both builders so the capture path can't drift.
fn finish_offscreen_app(app: &mut App, cfg: ScreenshotConfig, render_mode: super::RenderMode) {
    // The overlay rides its plugin here too — one wiring of it, app-wide (the demo's
    // offscreen app does the same): the env overrides land after, replacing the plugin's
    // defaults, so an evidence shot can force reveal/device/context.
    let (force_reveal, active_device, active_context) =
        crab_world::controls::reveal_overrides_from_env::<GcrControls>();
    app.add_plugins(crab_world::controls::ControlsOverlayPlugin::<GcrControls>::default())
        .insert_resource(cfg)
        .init_resource::<ShotProgress>()
        .insert_resource(force_reveal)
        .insert_resource(active_device)
        .insert_resource(active_context)
        .add_systems(Startup, (spawn_world, spawn_offscreen_camera, spawn_hud))
        .add_systems(
            Update,
            (
                gather_input,
                drive_pilot_script,
                drive_client_sim,
                reconcile_avatars,
                apply_transforms,
                follow_ground,
                apply_shot_cam_offset,
                // Keep the controls HUD's context live like the windowed app — unless an
                // evidence shot forced one via RL_SHOW_CONTROLS_CONTEXT.
                sync_controls_context
                    .run_if(|| std::env::var_os("RL_SHOW_CONTROLS_CONTEXT").is_none())
                    .before(update_controls_ui::<GcrControls>),
                update_hud,
                capture_when_settled,
            )
                .chain(),
        );
    super::render_mode::register(app, render_mode);
}

/// A scripted local pilot for the offscreen apps: press the E-cycle at the given frames and
/// hold a constant forward drive while piloting. This drives the REAL input seams — the same
/// `PendingInput`/`FlightInput` the keyboard writes, feeding the real intent-on-the-wire path —
/// so a two-peer `net-screenshot` run is a live board/fly/cycle/exit verification (rl#191
/// increment 4), not a code-path fork.
#[derive(Resource, Clone)]
pub struct PilotScript {
    toggle_at: Vec<u64>,
    walk_at: Option<u64>,
    frame: u64,
}

impl PilotScript {
    pub fn new(toggle_at: Vec<u64>, walk_at: Option<u64>) -> Self {
        Self {
            toggle_at,
            walk_at,
            frame: 0,
        }
    }
}

fn drive_pilot_script(
    script: Option<ResMut<PilotScript>>,
    vehicle: Res<super::driver::LocalVehicle>,
    mut pending: ResMut<PendingInput>,
    mut flight: ResMut<FlightInput>,
) {
    let Some(mut script) = script else {
        return;
    };
    script.frame += 1;
    if script.toggle_at.contains(&script.frame) {
        pending.toggle_vehicle = true;
    }
    // While piloting, drive forward: plane throttle (wasd.y + rt), ship forward thrust + lift.
    // Runs after `gather_input` (which zeroes FlightInput off the absent devices), so the
    // script's hold wins the frame.
    if vehicle.kind().is_some() {
        flight.wasd = bevy::math::Vec2::new(0.0, 1.0);
        flight.rt = 0.5;
    } else if script.walk_at.is_some_and(|at| script.frame >= at) {
        // Walk a gentle arc on foot — a moving target, so the run exercises the hunt
        // against evasion, not just a standing grab (rl#236).
        pending.forward = 1.0;
        pending.yaw_delta = 0.02;
    }
}

#[derive(Resource, Clone)]
pub struct ScreenshotConfig {
    path: PathBuf,
    settle: u32,
    width: u32,
    height: u32,
    cam_yaw_deg: f32,
    cam_pitch_deg: f32,
    cam_fov_deg: Option<f32>,
}

impl ScreenshotConfig {
    pub fn new(path: PathBuf, settle: u32, width: u32, height: u32) -> Self {
        Self {
            path,
            settle,
            width,
            height,
            cam_yaw_deg: 0.0,
            cam_pitch_deg: 0.0,
            cam_fov_deg: None,
        }
    }

    pub fn with_fov(mut self, fov_deg: Option<f32>) -> Self {
        self.cam_fov_deg = fov_deg;
        self
    }

    pub fn with_cam_offset(mut self, yaw_deg: f32, pitch_deg: f32) -> Self {
        self.cam_yaw_deg = yaw_deg;
        self.cam_pitch_deg = pitch_deg;
        self
    }
}

fn spawn_offscreen_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cfg: Res<ScreenshotConfig>,
) {
    let handle = images.add(screenshot::new_render_target(cfg.width, cfg.height));
    let mut cam = commands.spawn((
        screenshot::offscreen_camera_bundle(handle.clone()),
        Transform::default(),
        FpCamera,
    ));
    // The windowed FP camera's perspective (shared render-frame-scaled near clip), plus an
    // optional wider FOV so the towering giant crab fits in one evidence frame.
    let mut projection = super::scene::fp_perspective();
    if let Some(fov_deg) = cfg.cam_fov_deg {
        projection.fov = fov_deg.to_radians();
    }
    cam.insert(Projection::Perspective(projection));
    commands.insert_resource(ShotTarget(handle));
}

fn apply_shot_cam_offset(
    cfg: Res<ScreenshotConfig>,
    mut cam_q: Query<&mut Transform, With<FpCamera>>,
) {
    if cfg.cam_yaw_deg == 0.0 && cfg.cam_pitch_deg == 0.0 {
        return;
    }
    let Ok(mut cam) = cam_q.single_mut() else {
        return;
    };
    let eye = cam.translation;
    let fwd = cam.forward().as_vec3();
    let right = cam.right().as_vec3();
    let rot = Quat::from_axis_angle(Vec3::Y, cfg.cam_yaw_deg.to_radians())
        * Quat::from_axis_angle(right, cfg.cam_pitch_deg.to_radians());
    let new_fwd = (rot * fwd).normalize();
    *cam = Transform::from_translation(eye).looking_at(eye + new_fwd, Vec3::Y);
}

fn capture_when_settled(
    mut commands: Commands,
    cfg: Res<ScreenshotConfig>,
    target: Res<ShotTarget>,
    mut progress: ResMut<ShotProgress>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(frame) = screenshot::advance_capture(&mut progress, cfg.settle, &mut exit) else {
        return;
    };
    screenshot::save_target_to(&mut commands, &target, cfg.path.clone());
    info!(
        "fp screenshot: captured at render frame {frame}, writing {}",
        cfg.path.display()
    );
    screenshot::finish_capture(&mut progress);
}
