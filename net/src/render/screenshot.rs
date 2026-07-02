//! Headless screenshot app: render one settled frame of the FP view to a PNG and exit —
//! the evidence path on a display-less box. Composes the same scene/HUD/input systems the
//! windowed client uses (see [`super::app`]) onto an offscreen camera.

use super::app::{install_armed_nn_crab, seed_external_crab_solo};
use super::driver::{ScriptedPackInput, coordinator, drive_lockstep, insert_core};
use super::hud::{spawn_hud, update_hud};
use super::input::gather_input;
use super::scene::{FpCamera, apply_transforms, reconcile_avatars, spawn_world};
use super::*;
use crate::net_loop::NetDriver;
use crab_world::screenshot::{self, ShotProgress, ShotTarget};

/// Build the HEADLESS screenshot app: GPU on, no window, render one settled frame of
/// the FP view to `path` and exit. The evidence path on a box with no display — it
/// proves the sim→render pipeline (crab, extraction marker, another player from the
/// local eyes) without 2-peer play. Solo only (no transport): one peer's input
/// completes every tick, which is all a single-frame render needs.
///
/// `external_crab` arms the real rapier-NN crab + skin via the SAME seed/stack/gate the windowed
/// solo client uses, so the shot composes the actual armed scene a player sees — the trained,
/// reposed-to-giant crab ("Sally"), not the static integer silhouette. `None` keeps the silhouette.
pub fn build_screenshot_app(
    mut ls: Lockstep,
    cfg: ScreenshotConfig,
    external_crab: Option<std::path::PathBuf>,
    render_mode: super::RenderMode,
) -> App {
    let mut app = offscreen_app_scaffold();
    // Arm the real NN crab BEFORE `ls` moves into core (seeds the crab's spawn pose); the stack +
    // gate go on after, mirroring the windowed `Boot::Round` solo path. One `Option<(dir, spawn)>`
    // so the checkpoint and its seeded spawn can't disagree (both present or both absent).
    let armed_crab: Option<(std::path::PathBuf, Pos)> =
        external_crab.map(|dir| (dir, seed_external_crab_solo(&mut ls)));
    // A solo host-authoritative server steps the round (the SAME stepper the windowed client runs —
    // no self-stepping lockstep). The pack (every non-local player) walks straight forward toward the
    // extraction (+Z) so the scene composes: the crab chases them up the +Z lane and out of the
    // stationary local camera's forward view, keeping the players in frame early (crab shot) and
    // clearing the lane to the extraction pillar later (objective shot). `ScriptedPackInput` supplies
    // that per-tick input; the driver records it into the server for each pack player exactly as a
    // wire peer's would be (deterministic, adds no nondeterminism). The local player (0) holds still
    // — it is the camera — so nothing needs to move it.
    let coord = coordinator(None, ls.peers(), ls.sim().clone());
    insert_core(&mut app, ls, coord);
    app.insert_resource(ScriptedPackInput(Input::new(0.0, 1.0, 0.0, 0)));
    // Known-armed at build (when a checkpoint was given): add the rapier-NN stack AND arm the gate
    // now — the SAME path the windowed `Boot::Round` solo client uses — so `spawn_world` hides the
    // static silhouette and the reposed-to-giant rig becomes the visible crab.
    if let Some((dir, spawn)) = armed_crab {
        // Solo screenshot round: add the stack and arm the gate (one path).
        install_armed_nn_crab(&mut app, dir, spawn);
    }
    finish_offscreen_app(&mut app, cfg, render_mode);
    app
}

/// Build the NETWORKED offscreen screenshot app: the same offscreen
/// render as [`build_screenshot_app`], but its round is a real [`Coordinator`](crate::net_loop) over
/// the wire — a HOST (the peer that formed the round with the lowest id) or a REMOTE CLIENT — instead
/// of a scripted solo. Run two of these and the client's captured frame is the evidence that the
/// remote client renders the host's authoritative state + articulated crab through the snapshot path
/// (no re-sim). `net` is the formed [`NetDriver`]; `ls` its agreed tick-0 lockstep. The crab is armed
/// on BOTH peers — the host steps + broadcasts it, the client spawns it frozen and poses it from the
/// host's articulation. This is a two-identical-peer evidence harness (both run the SAME resolved
/// checkpoint, so the round is weights/asset-synced by construction); the loud arm-refusal gate for
/// a genuine mismatch lives on the interactive `play` path (`arm_round`), not here.
pub fn build_net_screenshot_app(
    mut ls: Lockstep,
    net: NetDriver,
    cfg: ScreenshotConfig,
    external_crab: std::path::PathBuf,
    render_mode: super::RenderMode,
) -> App {
    let mut app = offscreen_app_scaffold();
    // Seed the crab spawn pose before `ls` moves into the coordinator, exactly as the windowed
    // `Boot::Round` path does — so host and client agree on where the crab starts.
    let spawn = seed_external_crab_solo(&mut ls);
    let coord = coordinator(Some(net), ls.peers(), ls.sim().clone());
    insert_core(&mut app, ls, coord);
    // Arm the NN crab on this peer: the host runs + broadcasts it, the client spawns it (frozen —
    // never pumped) as the render target its adopted articulation poses. One path (add + arm).
    install_armed_nn_crab(&mut app, external_crab, spawn);
    finish_offscreen_app(&mut app, cfg, render_mode);
    app
}

/// The shared offscreen render scaffold both screenshot builders use: no window, GPU on
/// (render-to-image), a fixed per-frame sim step, and the night sky. Keeping it in one place means
/// the solo and networked shots compose the identical scene — only their input SOURCE differs.
fn offscreen_app_scaffold() -> App {
    let mut app = App::new();
    // No window, GPU ON (render-to-image). A 60 Hz schedule runner with a real-time
    // step so the capture counter (render frames) also paces the sim and the GPU
    // pipeline warms over the same frames — mirrors play.rs's screenshot mode.
    app.add_plugins(crab_world::app_boot::base_plugins(None));
    app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
        Duration::from_secs_f64(1.0 / 60.0),
    ));
    // Night-sky skybox behind the captured FP frame (same sky as the windowed GCR client).
    app.add_plugins(crab_world::sky::NightSkyPlugin);
    // Advance the sim a FIXED amount per frame instead of by wall-clock, so the
    // composed scene is a function of the settle COUNT, not how fast software-Vulkan
    // renders each frame (otherwise a slower box advances the sim further before the
    // shot and the framing drifts). One tick's dt per frame → `settle` frames ≈
    // `settle` ticks, the deterministic exposure the evidence shot wants.
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(TICK_DT),
    ));
    app
}

/// Wire the offscreen screenshot systems + render-mode onto a scaffolded app whose round is
/// already installed — shared by both builders so the capture path can't drift.
fn finish_offscreen_app(app: &mut App, cfg: ScreenshotConfig, render_mode: super::RenderMode) {
    // Controls UI on the screenshot path too, so an evidence frame can prove the overlay +
    // hint draw — the shared env override forces it open headless, and picks the CONTEXT
    // (`RL_SHOW_CONTROLS_CONTEXT=foot|plane`) so one shot can record any context's legend
    // (see [`crab_world::controls::reveal_overrides_from_env`]).
    let (force_reveal, active_device, active_context) =
        crab_world::controls::reveal_overrides_from_env::<GcrControls>();
    app.insert_resource(cfg)
        .init_resource::<ShotProgress>()
        .insert_resource(force_reveal)
        .insert_resource(active_device)
        .insert_resource(active_context)
        .add_systems(
            Startup,
            (
                spawn_world,
                spawn_offscreen_camera,
                spawn_hud,
                spawn_controls_ui::<GcrControls>,
            ),
        )
        .add_systems(
            Update,
            (
                gather_input,
                drive_lockstep,
                reconcile_avatars,
                apply_transforms,
                apply_shot_cam_offset,
                update_hud,
                update_controls_ui::<GcrControls>,
                capture_when_settled,
            )
                .chain(),
        );
    // The crab render-mode cycle (mesh unless `render_mode` says otherwise), so an evidence frame
    // can capture any of the views. No determinism pin here either — see the note in
    // [`super::app::build_windowed_app`] (rl#199).
    super::render_mode::register(app, render_mode);
}

// ---------------------------------------------------------------------------
// Headless screenshot mode (evidence the sim→render path works)
// ---------------------------------------------------------------------------

/// Knobs for the headless screenshot app, and the resource its systems read.
#[derive(Resource, Clone)]
pub struct ScreenshotConfig {
    path: PathBuf,
    settle: u32,
    width: u32,
    height: u32,
    /// Screenshot-only camera pan/tilt for framing (see [`ScreenshotConfig::with_cam_offset`]).
    cam_yaw_deg: f32,
    cam_pitch_deg: f32,
    /// Screenshot-only vertical field of view (degrees). `None` keeps Bevy's default FOV;
    /// a wider value frames the whole towering crab in one first-person frame even though it
    /// stands only a body-length away (the giant fills the default FOV otherwise).
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

    /// Override the screenshot camera's vertical FOV (degrees) — widen it to fit the whole
    /// giant crab in one frame. `None`/unset keeps Bevy's default.
    pub fn with_fov(mut self, fov_deg: Option<f32>) -> Self {
        self.cam_fov_deg = fov_deg;
        self
    }

    /// Pan/tilt the screenshot camera by these degrees, applied at the local player's
    /// eye AFTER the first-person aim — so a single evidence frame can frame the giant
    /// crab, the extraction pillar, and the other players together when the towering
    /// crab would otherwise fill the dead-ahead view. Still a first-person shot (same
    /// eye, same sim yaw as the base); only the composition pans. Zero = straight
    /// first-person.
    pub fn with_cam_offset(mut self, yaw_deg: f32, pitch_deg: f32) -> Self {
        self.cam_yaw_deg = yaw_deg;
        self.cam_pitch_deg = pitch_deg;
        self
    }
}

/// The offscreen camera for the screenshot path. Its transform is driven by
/// [`apply_transforms`] (it carries the [`FpCamera`] marker), so the captured frame
/// is the genuine first-person view, not a separate angle.
fn spawn_offscreen_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cfg: Res<ScreenshotConfig>,
) {
    let handle = images.add(screenshot::new_render_target(cfg.width, cfg.height));
    let mut cam = commands.spawn((
        screenshot::offscreen_camera_bundle(handle.clone()),
        // This cam carries the first-person marker so `apply_transforms` drives its aim;
        // `Transform::default` is a placeholder until the first Update.
        Transform::default(),
        FpCamera,
    ));
    // Match the windowed FP camera's render-frame-scaled near plane (else the shrunk human world
    // clips at the default 0.1 m), plus an optional wider FOV so the towering giant crab fits in
    // one evidence frame.
    cam.insert(Projection::Perspective(PerspectiveProjection {
        near: super::scene::DEFAULT_CAMERA_NEAR * super::scene::world_render_scale(),
        fov: cfg
            .cam_fov_deg
            .map(f32::to_radians)
            .unwrap_or(PerspectiveProjection::default().fov),
        ..default()
    }));
    commands.insert_resource(ShotTarget(handle));
}

/// Screenshot-only: pan/tilt the FP camera by the configured offset for framing,
/// keeping its eye where [`apply_transforms`] placed it (so it's still the local
/// player's first-person view, just composed). Runs AFTER `apply_transforms`, which
/// owns the base FP aim. No-op when the offset is zero.
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
    // Yaw about world up, pitch about the camera's own right axis — pan then tilt.
    let rot = Quat::from_axis_angle(Vec3::Y, cfg.cam_yaw_deg.to_radians())
        * Quat::from_axis_angle(right, cfg.cam_pitch_deg.to_radians());
    let new_fwd = (rot * fwd).normalize();
    *cam = Transform::from_translation(eye).looking_at(eye + new_fwd, Vec3::Y);
}

/// After the sim has run a few ticks and the GPU pipeline has warmed, capture one PNG
/// of the FP view and exit. The settle/capture/exit bookkeeping is the shared
/// [`crab_world::screenshot`] primitive; this system just composes the FP scene's single
/// shot on top of it.
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
