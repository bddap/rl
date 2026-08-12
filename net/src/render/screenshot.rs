use super::app::{install_armed_nn_crab, seed_round_crabs};
use super::controls_sync::sync_controls_context;
use super::driver::{
    FlightInput, PendingInput, ScriptedPackInput, coordinator, drive_client_sim, insert_core,
};
use super::input::gather_input;
use super::scene::{
    FpCamera, apply_transforms, place_extraction_pillar, reconcile_avatars, spawn_world,
    sync_ground_anchor,
};
use super::*;
use crate::net_loop::NetDriver;
use crab_world::controls::ControlsOverrides;
use crab_world::screenshot::{self, ShotProgress, ShotTarget};

pub fn build_screenshot_app(
    mut client: ClientSim,
    cfg: ScreenshotConfig,
    nn_crab: Option<crab_world::policy::Policy>,
    view: crab_world::BootView,
    controls: ControlsOverrides<GcrControls>,
    pack: Input,
) -> App {
    let mut app = offscreen_app_scaffold(view);
    let armed_crab = nn_crab.map(|policy| (policy, seed_round_crabs(&mut client, 1)));
    let coord = coordinator(None, client.peers(), client.me(), client.sim().clone());
    insert_core(&mut app, client, coord);
    app.insert_resource(ScriptedPackInput(pack));
    if let Some((policy, spawns)) = armed_crab {
        install_armed_nn_crab(&mut app, vec![policy], spawns);
    }
    finish_offscreen_app(&mut app, cfg, view.render_mode, controls);
    app
}

pub fn build_net_screenshot_app(
    mut client: ClientSim,
    net: NetDriver,
    cfg: ScreenshotConfig,
    nn_crab: crab_world::policy::Policy,
    view: crab_world::BootView,
    controls: ControlsOverrides<GcrControls>,
) -> App {
    let mut app = offscreen_app_scaffold(view);
    let spawns = seed_round_crabs(&mut client, 1);
    let coord = coordinator(Some(net), client.peers(), client.me(), client.sim().clone());
    insert_core(&mut app, client, coord);
    install_armed_nn_crab(&mut app, vec![nn_crab], spawns);
    finish_offscreen_app(&mut app, cfg, view.render_mode, controls);
    app
}

fn offscreen_app_scaffold(view: crab_world::BootView) -> App {
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
    app.insert_resource(view.moon);
    app.add_plugins(crab_world::sky::NightSkyPlugin);
    app.add_plugins(crab_world::physics::ArenaWorldPlugin {
        ground_look: view.ground_look,
    });
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(TICK_DT),
    ));
    app
}

/// Wire the offscreen screenshot systems + render-mode onto a scaffolded app whose round is
/// already installed — shared by both builders so the capture path can't drift.
fn finish_offscreen_app(
    app: &mut App,
    cfg: ScreenshotConfig,
    render_mode: super::RenderMode,
    controls: ControlsOverrides<GcrControls>,
) {
    crab_world::controls::install_overlay(app, &controls);
    crab_world::chord::install_chords::<GcrControls>(app);
    // The rl#358 combo map, unpersisted by default — an evidence shot must never
    // write the real save; `--chord-map-file` overrides the resource after build.
    super::chord_map::install(app, None);
    app.add_systems(
        PreUpdate,
        drive_chord_script
            .after(bevy::input::InputSystems)
            .before(crab_world::chord::capture_chords::<GcrControls>),
    );
    app.insert_resource(cfg)
        .init_resource::<ShotProgress>()
        .add_systems(Startup, (spawn_world, spawn_offscreen_camera))
        .add_systems(
            Update,
            (
                gather_input,
                drive_pilot_script,
                drive_client_sim,
                // Before any pose/transform consumer, as in the windowed app: the
                // render origin must be current the frame it changes (rl#354).
                sync_ground_anchor.before(crab_world::ground::apply_ground_anchor),
                super::articulation::sample_crab_part_poses,
                reconcile_avatars,
                apply_transforms,
                place_extraction_pillar,
                apply_shot_cam_offset,
                // Keeps the controls context live like the windowed app. A shot that PINNED
                // a context is unaffected: ActiveContext::sync is a no-op while pinned.
                sync_controls_context.before(update_controls_ui::<GcrControls>),
                capture_when_settled,
            )
                .chain(),
        );
    super::render_mode::register(app, render_mode);
}

/// A scripted local pilot for the offscreen apps: issue a vehicle request at the given
/// frames and hold a constant forward drive while piloting. This drives the REAL input
/// seams — the same `PendingInput`/`FlightInput` the keyboard writes, feeding the real
/// intent-on-the-wire path — so a two-peer `net-screenshot` run is a live
/// board/fly/switch/exit verification (rl#191 increment 4), not a code-path fork.
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

/// Scripted chord entry for evidence shots (rl#330 stage 3): synthesizes the kb chord
/// modifier hold (right mouse) and WASD code taps into the REAL `ButtonInput`
/// resources, upstream of `capture_chords` — the shot exercises the live capture,
/// menu-filtering, and dispatch path, not a UI fork.
#[derive(Resource, Clone)]
pub struct ChordScript {
    /// Modifier hold windows `(from, to)`; `to = None` holds into the shot. Multiple
    /// windows let one clip enter SEVERAL codes (release executes each) — the rl#358
    /// map-growth evidence needs back-to-back discoveries.
    holds: Vec<(u64, Option<u64>)>,
    /// (frame, direction) code taps; each key is pressed for exactly one frame.
    taps: Vec<(u64, crab_world::chord::ChordDir)>,
    /// Frames at which to tap the chord-map layout-cycle key (M, rl#358) — the
    /// live-switch evidence input.
    cycles: Vec<u64>,
    frame: u64,
}

impl ChordScript {
    pub fn new(
        hold_from: u64,
        release_at: Option<u64>,
        taps: Vec<(u64, crab_world::chord::ChordDir)>,
    ) -> Self {
        Self {
            holds: vec![(hold_from, release_at)],
            taps,
            cycles: Vec::new(),
            frame: 0,
        }
    }

    /// Additional modifier hold windows beyond the first.
    pub fn with_holds(mut self, holds: Vec<(u64, Option<u64>)>) -> Self {
        self.holds.extend(holds);
        self
    }

    /// See [`ChordScript::cycles`].
    pub fn with_layout_cycles(mut self, cycles: Vec<u64>) -> Self {
        self.cycles = cycles;
        self
    }
}

fn drive_chord_script(
    script: Option<ResMut<ChordScript>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
) {
    let Some(mut script) = script else {
        return;
    };
    script.frame += 1;
    let frame = script.frame;
    let held = script
        .holds
        .iter()
        .any(|&(from, to)| frame >= from && to.is_none_or(|r| frame < r));
    if held {
        // Idempotent while already pressed — just_pressed fires only on the edge.
        mouse.press(crab_world::chord::CHORD_MODIFIER_MOUSE);
    } else if mouse.pressed(crab_world::chord::CHORD_MODIFIER_MOUSE) {
        mouse.release(crab_world::chord::CHORD_MODIFIER_MOUSE);
    }
    // Releases strictly before presses: for back-to-back same-key taps (frames N and
    // N+1 — Quit's ^^ opener) the N+1 press must land on a released key regardless
    // of spec order, or bevy's idempotent press() yields no just_pressed edge and the
    // capture silently holds a PREFIX of the typed code.
    for &(at, dir) in &script.taps {
        if script.frame == at + 1 {
            keys.release(crab_world::chord::dir_key(dir));
        }
    }
    for &(at, dir) in &script.taps {
        if script.frame == at {
            keys.press(crab_world::chord::dir_key(dir));
        }
    }
    // Layout-cycle taps ride the same release-before-press ordering as the code taps.
    if script.cycles.iter().any(|&at| script.frame == at + 1) {
        keys.release(super::chord_map::LAYOUT_CYCLE_KEY);
    }
    if script.cycles.iter().any(|&at| script.frame == at) {
        keys.press(super::chord_map::LAYOUT_CYCLE_KEY);
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
        // The game has no cycle verb anymore (one board code per vehicle, rl#330) —
        // the script keeps its historical foot→plane→ship→foot order locally so the
        // `--pilot-toggle-at` CLI contract is unchanged.
        use super::driver::VehicleRequest;
        use crab_world::vehicle::VehicleKind;
        pending.vehicle = Some(match vehicle.kind() {
            None => VehicleRequest::Board(VehicleKind::Plane),
            Some(VehicleKind::Plane) => VehicleRequest::Board(VehicleKind::Ship),
            Some(VehicleKind::Ship) => VehicleRequest::Exit,
        });
    }
    // While piloting, drive forward: plane throttle (wasd.y + rt), ship forward thrust + lift.
    // Runs after `gather_input` (which zeroes FlightInput off the absent devices), so the
    // script's hold wins the frame.
    if vehicle.kind().is_some() {
        flight.wasd = bevy::math::Vec2::new(0.0, 1.0);
        flight.rt = 0.5;
    } else if script.walk_at.is_some_and(|at| script.frame >= at) {
        // Walk a gentle arc on foot — a moving target, so the run exercises the hunt
        // against evasion, not just a standing kill (rl#236).
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
    cam_height_m: f32,
    cam_fov_deg: Option<f32>,
    /// `Some((count, every))`: capture `count` frames, one every `every` render
    /// frames after settle, as `<stem>.NNNN.png` — the animated-look evidence
    /// path (assemble with ffmpeg). `None`: the single shot at settle.
    anim: Option<(u32, u32)>,
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
            cam_height_m: 0.0,
            cam_fov_deg: None,
            anim: None,
        }
    }

    /// Capture a frame sequence instead of one shot (see [`Self::anim`]).
    pub fn with_anim(mut self, anim: Option<(u32, u32)>) -> Self {
        self.anim = anim.filter(|&(count, _)| count > 1);
        self
    }

    /// Raise the camera this far above the normal eye point — vista/altitude
    /// evidence shots (the ground art set, rl#304) without a piloted flight.
    pub fn with_cam_height(mut self, height_m: f32) -> Self {
        self.cam_height_m = height_m;
        self
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
    // The windowed FP camera's perspective (shared stature-scaled near clip), plus an
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
    if cfg.cam_yaw_deg == 0.0 && cfg.cam_pitch_deg == 0.0 && cfg.cam_height_m == 0.0 {
        return;
    }
    let Ok(mut cam) = cam_q.single_mut() else {
        return;
    };
    let eye = cam.translation + Vec3::Y * cfg.cam_height_m;
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
    let Some((count, every)) = cfg.anim else {
        screenshot::save_target_to(&mut commands, &target, cfg.path.clone());
        info!(
            "fp screenshot: captured at render frame {frame}, writing {}",
            cfg.path.display()
        );
        screenshot::finish_capture(&mut progress);
        return;
    };
    // Sequence capture: frame `settle + k*every` is shot number k. Time advances
    // ManualDuration(TICK_DT) per render frame, so `every` ticks = every/60 s of
    // shader time between shots — deterministic, replayable animation evidence.
    let since_settle = frame - cfg.settle;
    let every = every.max(1);
    if !since_settle.is_multiple_of(every) {
        return;
    }
    let shot = since_settle / every;
    let path = cfg.path.with_extension(format!("{shot:04}.png"));
    screenshot::save_target_to(&mut commands, &target, path.clone());
    info!(
        "fp screenshot: anim frame {shot}/{count} at render frame {frame} -> {}",
        path.display()
    );
    if shot + 1 >= count {
        screenshot::finish_capture(&mut progress);
    }
}
