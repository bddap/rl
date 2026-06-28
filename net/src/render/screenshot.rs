//! Headless screenshot app: render one settled frame of the FP view to a PNG and exit —
//! the evidence path on a display-less box. Composes the same scene/HUD/input systems the
//! windowed client uses (see [`super::app`]) onto an offscreen camera.

use super::*;
use super::app::{add_external_nn_crab, seed_external_crab_solo};
use super::driver::{InputSource, drive_lockstep, insert_core};
use super::hud::{spawn_hud, update_hud};
use super::input::gather_input;
use super::scene::{FpCamera, apply_transforms, spawn_world};
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
) -> App {
    let mut app = App::new();
    // No window, GPU ON (render-to-image). A 60 Hz schedule runner with a real-time
    // step so the capture counter (render frames) also paces the sim and the GPU
    // pipeline warms over the same frames — mirrors play.rs's screenshot mode.
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>(),
    );
    app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
        Duration::from_secs_f64(1.0 / 60.0),
    ));
    // Advance the sim a FIXED amount per frame instead of by wall-clock, so the
    // composed scene is a function of the settle COUNT, not how fast software-Vulkan
    // renders each frame (otherwise a slower box advances the sim further before the
    // shot and the framing drifts). One tick's dt per frame → `settle` frames ≈
    // `settle` ticks, the deterministic exposure the evidence shot wants.
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(TICK_DT),
    ));
    // Arm the real NN crab BEFORE `ls` moves into core (seeds the crab's spawn pose); the stack +
    // gate go on after, mirroring the windowed `Boot::Round` solo path. One `Option<(dir, spawn)>`
    // so the checkpoint and its seeded spawn can't disagree (both present or both absent).
    let armed_crab: Option<(std::path::PathBuf, Pos)> =
        external_crab.map(|dir| (dir, seed_external_crab_solo(&mut ls)));
    // Stand-in input for the absent peers so the sim advances and the scene composes:
    // walk them straight forward toward the extraction (+Z). The crab chases them up
    // the +Z lane and out of the stationary local camera's forward view, which keeps
    // the players in frame early (crab shot) and clears the lane to the extraction
    // pillar later (objective shot). Fed through the normal deterministic input path
    // (see [`InputSource::Scripted`]) — adds no nondeterminism.
    insert_core(
        &mut app,
        ls,
        InputSource::Scripted(Input::new(0.0, 1.0, 0.0, 0)),
    );
    // Known-armed at build (when a checkpoint was given): add the rapier-NN stack AND arm the gate
    // now — the SAME path the windowed `Boot::Round` solo client uses — so `spawn_world` hides the
    // static silhouette and the reposed-to-giant rig becomes the visible crab.
    let armed = armed_crab.is_some();
    if let Some((dir, spawn)) = armed_crab {
        add_external_nn_crab(&mut app, dir, spawn);
        app.insert_resource(crate::external_crab::ExternalCrabArmed);
    }
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
                apply_transforms,
                apply_shot_cam_offset,
                update_hud,
                update_controls_ui::<GcrControls>,
                capture_when_settled,
            )
                .chain(),
        );
    // When the NN crab is armed, the same single-threaded ECS pin the windowed solo client applies
    // (see `build_windowed_app`): rapier's multibody solver is nondeterministic — and can hit a NaN
    // that panics `step_simulation` — on the multi-threaded executor. Must run AFTER every system
    // is wired. Unnecessary for the silhouette shot (no physics), so gated on `armed`.
    if armed {
        crab_world::bot::headless::force_serial_schedules(&mut app);
    }
    app
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
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.5, 0.7, 0.92)),
            ..default()
        },
        RenderTarget::Image(handle.clone().into()),
        // Default tonemapping needs a LUT asset that may not be loaded in a windowless
        // render; None keeps the offscreen pass simple (mirrors play.rs).
        Tonemapping::None,
        Transform::default(),
        FpCamera,
        // Make UI render into THIS offscreen target. Bevy renders UI to the default-window
        // camera automatically, but the screenshot path has no window — without this marker
        // the HUD + controls overlay never composite into the captured texture, so an
        // evidence frame would miss them. The windowed client doesn't need it (its window
        // camera is the implicit UI target).
        bevy::ui::IsDefaultUiCamera,
    ));
    // Optional wider FOV so the towering giant crab fits in one evidence frame.
    if let Some(fov_deg) = cfg.cam_fov_deg {
        cam.insert(Projection::Perspective(PerspectiveProjection {
            fov: fov_deg.to_radians(),
            ..default()
        }));
    }
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
