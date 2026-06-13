//! Interactive play + screenshot modes.
//!
//! `DemoPlugin` loads a trained checkpoint and drives the crab with the policy
//! (deterministic, no learning) behind an orbit camera with poke/reset controls
//! — the "launch it and play" experience.
//!
//! `ScreenshotPlugin` renders one frame to a PNG and exits. It runs windowless
//! but with the GPU on (render-to-image), so the trained crab can be inspected
//! without a display or a human in the loop.

use std::path::{Path, PathBuf};

use bevy::app::AppExit;
use bevy::camera::RenderTarget;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy_rapier3d::plugin::PhysicsSet;
use bevy_rapier3d::prelude::*;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::{AutodiffModule, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;
use rand::Rng;

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabAssets, CrabBodyPart, CrabCarapace};
use crate::bot::brain::{ACTION_SIZE, CrabBrain};
use crate::bot::sensor::{CrabObservation, OBS_SIZE};
use crate::bot::{BotSet, CrabSpawns, respawn_crab};
use crate::training::session::{
    BRAIN_STEM, InferBackend, NORMALIZER_FILENAME, ObsNormalizer, TrainBackend,
};

/// A loaded policy that maps observations to actions for inference (no learning).
///
/// Non-send because the `ndarray` backend's tensors are not `Sync` (same reason
/// as `TrainingState`).
pub struct Policy {
    brain: CrabBrain<InferBackend>,
    normalizer: ObsNormalizer,
    device: NdArrayDevice,
    /// False when no checkpoint loaded — `act` then returns zero actions (a
    /// neutral, deterministic rest pose) instead of an untrained brain's noise,
    /// so a no-checkpoint render shows the body geometry cleanly.
    loaded: bool,
}

impl Policy {
    /// Load brain + normalizer from a checkpoint dir. Missing/corrupt files fall
    /// back to a zero-action policy so the app still launches (useful before the
    /// first checkpoint exists, and to inspect the body's neutral rest pose).
    pub fn load(checkpoint_dir: &Path) -> Self {
        let device = NdArrayDevice::Cpu;

        // Checkpoints are saved from the autodiff backend; load into it, then
        // `.valid()` down to the bare inference backend — the path training uses.
        let mut train_brain = CrabBrain::<TrainBackend>::new(&device);
        let mut loaded = false;
        let brain_path = checkpoint_dir.join(BRAIN_STEM);
        if brain_path.with_extension("bin").exists() {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            match recorder.load(brain_path.clone(), &device) {
                Ok(record) => {
                    train_brain = train_brain.load_record(record);
                    loaded = true;
                    info!("play: loaded brain from {}", brain_path.display());
                }
                Err(e) => warn!("play: failed to load brain ({e}) — using zero-action pose"),
            }
        } else {
            warn!(
                "play: no checkpoint at {} — using zero-action pose",
                brain_path.with_extension("bin").display()
            );
        }
        let brain = train_brain.valid();

        let mut normalizer = ObsNormalizer::new(5.0);
        let norm_path = checkpoint_dir.join(NORMALIZER_FILENAME);
        if norm_path.exists()
            && let Some(loaded) = ObsNormalizer::load(&norm_path)
        {
            info!("play: loaded normalizer from {}", norm_path.display());
            normalizer = loaded;
        }

        // Diagnostic: RL_RANDOM_POLICY drives the crab with the untrained
        // random-init brain even without a checkpoint, to see what a FRESH
        // policy does (vs the zero-action rest pose) — distinguishes a learned
        // behaviour from one the dynamics produce on their own.
        let loaded = loaded || std::env::var("RL_RANDOM_POLICY").is_ok_and(|v| v == "1");

        Self {
            brain,
            normalizer,
            device,
            loaded,
        }
    }

    /// Deterministic action: the policy mean (no exploration noise), so the crab
    /// holds a steady pose instead of jittering.
    fn act(&self, raw_obs: &[f32; OBS_SIZE]) -> [f32; ACTION_SIZE] {
        // No checkpoint → hold the neutral (zero-action) pose: a deterministic
        // view of the body geometry, not an untrained brain's noise.
        if !self.loaded {
            return [0.0; ACTION_SIZE];
        }
        let obs = self.normalizer.normalize_frozen(raw_obs);
        let input =
            Tensor::<InferBackend, 1>::from_floats(obs.as_slice(), &self.device).unsqueeze();
        let (means, _log_std) = self.brain.policy(input);
        let flat: Vec<f32> = means.flatten::<1>(0, 1).to_data().to_vec().unwrap();

        let mut out = [0.0f32; ACTION_SIZE];
        for (o, &v) in out.iter_mut().zip(flat.iter()) {
            *o = if v.is_finite() {
                v.clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        out
    }
}

/// System (BotSet::Think): run the policy and write the actions the actuator
/// will apply.
fn policy_step(
    policy: NonSend<Policy>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
) {
    if let (Some(o), Some(a)) = (obs.envs.first(), actions.envs.first_mut()) {
        *a = policy.act(o);
    }
}

fn add_inference(app: &mut App, checkpoint_dir: &Path) {
    app.insert_non_send_resource(Policy::load(checkpoint_dir));
    app.add_systems(FixedUpdate, policy_step.in_set(BotSet::Think));
}

// ---------------------------------------------------------------------------
// Demo: windowed, interactive
// ---------------------------------------------------------------------------

/// Windowed "play with the trained crab" mode.
pub struct DemoPlugin {
    pub checkpoint_dir: PathBuf,
}

impl Plugin for DemoPlugin {
    fn build(&self, app: &mut App) {
        add_inference(app, &self.checkpoint_dir);
        crate::player::graph::register(app);
        app.init_resource::<DemoSettle>()
            .init_resource::<PokeBurst>()
            .add_systems(Startup, (spawn_orbit_camera, spawn_hud))
            .add_systems(Update, (orbit_camera, demo_controls))
            .add_systems(
                FixedUpdate,
                (
                    demo_settle.after(BotSet::Think),
                    demo_fall_rescue,
                    demo_poke.after(BotSet::Act).before(PhysicsSet::SyncBackend),
                ),
            );
    }
}

/// Orbit camera state. `focus` tracks the crab so it stays centered.
#[derive(Component)]
struct OrbitCamera {
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

fn spawn_orbit_camera(mut commands: Commands) {
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
fn orbit_camera(
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

    // Keyboard arrows orbit; -/= zoom.
    if keys.pressed(KeyCode::ArrowLeft) {
        d_yaw += dt;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        d_yaw -= dt;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        d_pitch += dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        d_pitch -= dt;
    }
    if keys.pressed(KeyCode::Minus) {
        d_zoom += dt * 3.0;
    }
    if keys.pressed(KeyCode::Equal) {
        d_zoom -= dt * 3.0;
    }

    // Gamepad: right stick orbits, triggers zoom.
    for gp in gamepads.iter() {
        let rs = gp.right_stick();
        if rs.length() > 0.15 {
            d_yaw -= rs.x * dt * 2.5;
            d_pitch += rs.y * dt * 2.5;
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

/// Settle ticks remaining after a reset. The respawned crab starts in the
/// rest pose with the builder motors already holding it; the settle just
/// holds zero actions while it drops onto the ground and takes load.
#[derive(Resource, Default)]
struct DemoSettle(u32);

const DEMO_SETTLE_TICKS: u32 = 32;

/// A poke is a short force burst, not a velocity write: a multibody link's
/// velocity lives in the multibody's generalized coordinates, which the
/// `Velocity` component writeback never touches (issue #14 — the old poke
/// was a silent no-op). Per-link external forces, by contrast, are mapped
/// through the body Jacobians into generalized accelerations, so force is
/// the one channel that actually reaches a multibody root. Rapier never
/// auto-clears user forces, hence the countdown that zeroes them.
#[derive(Resource, Default)]
struct PokeBurst {
    ticks: u32,
    force: Vec3,
    torque: Vec3,
}

const POKE_TICKS: u32 = 8;
const POKE_FORCE: f32 = 70.0;
const POKE_TORQUE: f32 = 4.0;

/// System (FixedUpdate, after the actuator): adds the active poke burst on top
/// of the carapace's joint-reaction torques. The actuator overwrites every
/// link's `ExternalForce` each step, so the poke must run after it and *add*
/// rather than set — and it needs no cleanup, the actuator zeroes the baseline
/// next step.
fn demo_poke(
    mut burst: ResMut<PokeBurst>,
    mut carapace_q: Query<&mut ExternalForce, With<CrabCarapace>>,
) {
    if burst.ticks == 0 {
        return;
    }
    burst.ticks -= 1;
    let Ok(mut f) = carapace_q.single_mut() else {
        return;
    };
    f.force += burst.force;
    f.torque += burst.torque;
}

/// Demo reset: rebuild the crab fresh at spawn — the only reset that
/// survives a corrupted multibody (see [`respawn_crab`]) — and hold zero
/// actions while it takes load.
fn demo_respawn(
    commands: &mut Commands,
    assets: &CrabAssets,
    spawns: &CrabSpawns,
    parts: impl Iterator<Item = Entity>,
    settle: &mut DemoSettle,
    actions: &mut CrabActions,
) {
    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
    respawn_crab(commands, assets, parts, origin, 0);
    settle.0 = DEMO_SETTLE_TICKS;
    if let Some(a) = actions.envs.first_mut() {
        *a = [0.0; ACTION_SIZE];
    }
}

#[allow(clippy::too_many_arguments)]
fn demo_controls(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts_q: Query<Entity, With<CrabBodyPart>>,
    mut poke_burst: ResMut<PokeBurst>,
    mut actions: ResMut<CrabActions>,
    mut settle: ResMut<DemoSettle>,
) {
    let mut reset = keys.just_pressed(KeyCode::KeyR);
    let mut poke = keys.just_pressed(KeyCode::Space);
    let mut quit = keys.just_pressed(KeyCode::Escape);
    for gp in gamepads.iter() {
        reset |= gp.just_pressed(GamepadButton::South);
        poke |= gp.just_pressed(GamepadButton::West);
        quit |= gp.just_pressed(GamepadButton::Start);
    }

    if quit {
        exit.write(AppExit::Success);
    }
    if reset {
        demo_respawn(
            &mut commands,
            &assets,
            &spawns,
            parts_q.iter(),
            &mut settle,
            &mut actions,
        );
    }
    if poke {
        let mut rng = rand::thread_rng();
        let dir =
            Vec3::new(rng.gen_range(-1.0..1.0), 0.25, rng.gen_range(-1.0..1.0)).normalize_or_zero();
        *poke_burst = PokeBurst {
            ticks: POKE_TICKS,
            force: dir * POKE_FORCE,
            torque: Vec3::new(rng.gen_range(-1.0..1.0), 0.0, rng.gen_range(-1.0..1.0))
                * POKE_TORQUE,
        };
    }
}

/// System: the arena is finite and the policy can walk off it. A crab in
/// free fall under the world is lost to the viewer — rebuild it at spawn,
/// same as a training episode reset would.
fn demo_fall_rescue(
    mut commands: Commands,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts_q: Query<Entity, With<CrabBodyPart>>,
    carapace_q: Query<&Transform, With<CrabCarapace>>,
    mut actions: ResMut<CrabActions>,
    mut settle: ResMut<DemoSettle>,
) {
    let Ok(t) = carapace_q.single() else { return };
    if t.translation.y > -2.0 {
        return;
    }
    demo_respawn(
        &mut commands,
        &assets,
        &spawns,
        parts_q.iter(),
        &mut settle,
        &mut actions,
    );
}

/// System (FixedUpdate, after Think): while a demo settle is active, hold
/// zero actions so the motors keep the rest pose while the fresh crab takes
/// load.
fn demo_settle(mut settle: ResMut<DemoSettle>, mut actions: ResMut<CrabActions>) {
    if settle.0 == 0 {
        return;
    }
    if let Some(a) = actions.envs.first_mut() {
        *a = [0.0; ACTION_SIZE];
    }
    settle.0 -= 1;
}

fn spawn_hud(mut commands: Commands) {
    commands.spawn((
        Text::new(
            "Crab RL — trained policy\n\
             Right-drag / right-stick: orbit    wheel / triggers: zoom\n\
             R or (A): reset    Space or (X): poke    Esc or Start: quit",
        ),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.85)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

// ---------------------------------------------------------------------------
// Screenshot: windowless render-to-PNG
// ---------------------------------------------------------------------------

/// Renders one frame to a PNG after the crab settles, then exits.
pub struct ScreenshotPlugin {
    pub checkpoint_dir: PathBuf,
    pub path: PathBuf,
    pub settle: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Resource)]
struct ShotConfig {
    path: PathBuf,
    settle: u32,
    width: u32,
    height: u32,
}

/// The render-target image the offscreen camera draws into.
#[derive(Resource)]
struct ShotTarget(Handle<Image>);

#[derive(Resource, Default)]
struct ShotProgress {
    steps: u32,
    captured: bool,
    exit_countdown: i32,
}

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        add_inference(app, &self.checkpoint_dir);
        app.insert_resource(ShotConfig {
            path: self.path.clone(),
            settle: self.settle,
            width: self.width,
            height: self.height,
        })
        .init_resource::<ShotProgress>()
        .add_systems(Startup, spawn_offscreen_camera)
        .add_systems(
            Update,
            (track_offscreen_camera, capture_when_settled).chain(),
        );
    }
}

/// Keep the offscreen camera aimed at the (possibly drifting) crab so it stays
/// centered in the screenshot. Tracks horizontal position only; the vertical
/// focus is fixed mid-body so framing doesn't bob.
fn track_offscreen_camera(
    carapace_q: Query<&Transform, (With<CrabCarapace>, Without<Camera3d>)>,
    mut cam_q: Query<&mut Transform, With<Camera3d>>,
) {
    let (Ok(crab), Ok(mut cam)) = (carapace_q.single(), cam_q.single_mut()) else {
        return;
    };
    let focus = Vec3::new(crab.translation.x, 0.5, crab.translation.z);
    *cam =
        Transform::from_translation(focus + Vec3::new(1.9, 0.95, 2.5)).looking_at(focus, Vec3::Y);
}

fn spawn_offscreen_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cfg: Res<ShotConfig>,
) {
    let mut image =
        Image::new_target_texture(cfg.width, cfg.height, TextureFormat::bevy_default(), None);
    // COPY_SRC so the screenshot machinery can read the rendered texture back.
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let handle = images.add(image);

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
    ));
    commands.insert_resource(ShotTarget(handle));
}

#[allow(clippy::too_many_arguments)]
fn capture_when_settled(
    mut commands: Commands,
    cfg: Res<ShotConfig>,
    target: Res<ShotTarget>,
    mut progress: ResMut<ShotProgress>,
    mut exit: MessageWriter<AppExit>,
    carapace_q: Query<&Transform, With<CrabCarapace>>,
    meshes_q: Query<(), With<Mesh3d>>,
    lights_q: Query<(), With<DirectionalLight>>,
    cams_q: Query<&Camera>,
) {
    if progress.captured {
        progress.exit_countdown -= 1;
        if progress.exit_countdown <= 0 {
            exit.write(AppExit::Success);
        }
        return;
    }

    // settle is counted in RENDER frames, not physics steps: the GPU render
    // pipeline needs a few dozen frames to warm up (shader/pipeline compile,
    // asset upload) before the scene appears — earlier frames come out black.
    progress.steps += 1;
    if progress.steps < cfg.settle {
        return;
    }

    let carapace = carapace_q
        .single()
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);
    debug!(
        "screenshot scene: carapace={carapace:?} meshes={} lights={} cameras={}",
        meshes_q.iter().count(),
        lights_q.iter().count(),
        cams_q.iter().count(),
    );
    commands
        .spawn(Screenshot::image(target.0.clone()))
        .observe(save_to_disk(cfg.path.clone()));
    info!(
        "screenshot: captured at render frame {}, writing {}",
        progress.steps,
        cfg.path.display()
    );
    progress.captured = true;
    // Give the GPU readback + PNG encode a few frames to finish.
    progress.exit_countdown = 30;
}
