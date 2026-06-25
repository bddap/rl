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
use bevy_rapier3d::plugin::PhysicsSet;
use bevy_rapier3d::prelude::*;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::{AutodiffModule, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;
use rand::Rng;

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{
    self, CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabJoint, CrabJointId, Side,
};
use crate::bot::brain::CrabBrain;
use crate::bot::sensor::{CrabObservation, CrabTargets, OBS_SIZE};
use crate::bot::{BotSet, CrabSpawns, respawn_crab_rotated};
use crate::screenshot::{self, ShotProgress, ShotTarget};
use crate::training::session::{
    BRAIN_STEM, Curriculum, InferBackend, NORMALIZER_FILENAME, ObsNormalizer, RESET_GRACE_TICKS,
    TrainBackend, dist_3d, sample_target, settle_countdown,
};

// ---------------------------------------------------------------------------
// Demo control scheme — the demo's verbs for the reusable controls overlay
// (`crate::controls`). This is the single source of the on-screen LEGEND (it replaces the
// old hand-written HUD string, so the legend can't drift from the bindings). The demo's
// live input is analog/multi-key and read directly in its own systems (`orbit_camera`,
// `demo_controls`, `manual_control_step`, `toggle_graph`) — the map drives only the
// overlay, at the granularity that reads well (orbit's four arrow keys show as one glyph).
// ---------------------------------------------------------------------------

use crate::controls::{ControlEntry, ControlInput, ControlScheme, Glyph, KbBinding, PadBinding};

/// The demo's control scheme — a zero-size marker the overlay framework is instantiated
/// with. Disjoint from GCR's [`crate::net::controls::GcrControls`]: different verbs, own map.
pub struct DemoControls;

/// The demo's controllable verbs. The row key of [`DEMO_CONTROL_MAP`]; the per-app test
/// forces every variant to have exactly one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoAction {
    Orbit,
    Zoom,
    /// Rebuild the crab at a fresh random tilt.
    Rebuild,
    /// A random force/torque burst.
    Poke,
    Colliders,
    JointGraph,
    /// Hands-on manual gamepad control; the next two only apply while it's active.
    Manual,
    PickJoint,
    Torque,
    Quit,
    /// Hold to reveal the full control overlay.
    RevealControls,
}

/// The demo's keyboard glyph tokens. Some resolve to bundled icons (R/Space/Esc/Tab),
/// others to text keycaps for keys with no bundled icon (arrows/zoom/G) — see [`Glyph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoKey {
    R,
    Space,
    Escape,
    Tab,
    /// The orbit arrow keys (←/↑/↓ + comma), shown as one "Arrows" glyph.
    Arrows,
    /// The zoom keys (−/=), shown as one glyph.
    ZoomKeys,
    /// The collider-toggle key (→ / right arrow).
    ColliderKey,
    /// The joint-graph key (G).
    Graph,
}

/// The demo's mouse glyph tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoMouse {
    /// Right-drag to orbit (reuses the move icon).
    Drag,
    /// Wheel to zoom (text glyph; no bundled wheel icon).
    Wheel,
}

/// The demo's gamepad glyph tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoPad {
    LeftStick,
    RightStick,
    RightTrigger,
    LeftTrigger,
    South,
    West,
    East,
    North,
    Start,
    /// One glyph for two D-pad roles: collider toggle on Right, joint-pick on Up/Down.
    Dpad,
    View,
}

impl ControlScheme for DemoControls {
    type Action = DemoAction;
    type Key = DemoKey;
    type Pad = DemoPad;
    type Mouse = DemoMouse;

    fn map() -> &'static [ControlEntry<Self>] {
        &DEMO_CONTROL_MAP
    }

    fn reveal_action() -> DemoAction {
        DemoAction::RevealControls
    }

    fn key_glyph(key: DemoKey) -> Glyph {
        match key {
            DemoKey::R => Glyph::Icon("controls/keyboard_r.png"),
            DemoKey::Space => Glyph::Icon("controls/keyboard_space.png"),
            DemoKey::Escape => Glyph::Icon("controls/keyboard_escape.png"),
            DemoKey::Tab => Glyph::Icon("controls/keyboard_tab.png"),
            DemoKey::Arrows => Glyph::Label("Arrows"),
            DemoKey::ZoomKeys => Glyph::Label("- / ="),
            DemoKey::ColliderKey => Glyph::Label("Right"),
            DemoKey::Graph => Glyph::Label("G"),
        }
    }

    fn pad_glyph(pad: DemoPad) -> Glyph {
        match pad {
            DemoPad::LeftStick => Glyph::Icon("controls/xbox_stick_l.png"),
            DemoPad::RightStick => Glyph::Icon("controls/xbox_stick_r.png"),
            DemoPad::RightTrigger => Glyph::Icon("controls/xbox_rt.png"),
            DemoPad::South => Glyph::Icon("controls/xbox_button_a.png"),
            DemoPad::North => Glyph::Icon("controls/xbox_button_y.png"),
            DemoPad::Start => Glyph::Icon("controls/xbox_button_menu.png"),
            DemoPad::Dpad => Glyph::Icon("controls/xbox_dpad.png"),
            DemoPad::View => Glyph::Icon("controls/xbox_button_view.png"),
            // No bundled icon for these — text keycaps.
            DemoPad::LeftTrigger => Glyph::Label("LT"),
            DemoPad::West => Glyph::Label("X"),
            DemoPad::East => Glyph::Label("B"),
        }
    }

    fn mouse_glyph(mouse: DemoMouse) -> Glyph {
        match mouse {
            DemoMouse::Drag => Glyph::Icon("controls/mouse_move.png"),
            DemoMouse::Wheel => Glyph::Label("Wheel"),
        }
    }
}

impl ControlInput for DemoControls {
    fn key_code(key: DemoKey) -> Option<KeyCode> {
        // Only the reveal control (Tab) is polled by the overlay; the demo's other keys are
        // analog/multi-key and read directly in its own systems, so they map to no single
        // KeyCode here.
        match key {
            DemoKey::Tab => Some(KeyCode::Tab),
            _ => None,
        }
    }

    fn gamepad_button(pad: DemoPad) -> Option<GamepadButton> {
        // Likewise: only the reveal control (View) need resolve for the overlay's hold read.
        match pad {
            DemoPad::View => Some(GamepadButton::Select),
            _ => None,
        }
    }
}

/// THE demo control map — models the demo's CURRENT bindings (verified against
/// `orbit_camera`, `demo_controls`, `manual_control_step`, and `player::graph::toggle_graph`).
/// Reveal binding: hold Tab / hold pad View — both free in the demo's input set. Manual,
/// PickJoint, and Torque are gamepad-only (no keyboard binding), so the keyboard legend
/// omits them.
pub const DEMO_CONTROL_MAP: [ControlEntry<DemoControls>; 11] = [
    ControlEntry {
        action: DemoAction::Orbit,
        label: "Orbit camera",
        keyboard: KbBinding::new(&[DemoKey::Arrows], &[DemoMouse::Drag]),
        pad: PadBinding::new(&[DemoPad::LeftStick]),
    },
    ControlEntry {
        action: DemoAction::Zoom,
        label: "Zoom",
        keyboard: KbBinding::new(&[DemoKey::ZoomKeys], &[DemoMouse::Wheel]),
        pad: PadBinding::new(&[DemoPad::RightTrigger, DemoPad::LeftTrigger]),
    },
    ControlEntry {
        action: DemoAction::Rebuild,
        label: "Rebuild crab",
        keyboard: KbBinding::new(&[DemoKey::R], &[]),
        pad: PadBinding::new(&[DemoPad::South]),
    },
    ControlEntry {
        action: DemoAction::Poke,
        label: "Poke",
        keyboard: KbBinding::new(&[DemoKey::Space], &[]),
        pad: PadBinding::new(&[DemoPad::West]),
    },
    ControlEntry {
        action: DemoAction::Colliders,
        label: "Collider wireframes",
        keyboard: KbBinding::new(&[DemoKey::ColliderKey], &[]),
        pad: PadBinding::new(&[DemoPad::Dpad]),
    },
    ControlEntry {
        action: DemoAction::JointGraph,
        label: "Joint graph",
        keyboard: KbBinding::new(&[DemoKey::Graph], &[]),
        pad: PadBinding::new(&[DemoPad::North]),
    },
    ControlEntry {
        action: DemoAction::Manual,
        label: "Manual control",
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::East]),
    },
    ControlEntry {
        action: DemoAction::PickJoint,
        label: "Pick joint (manual)",
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::Dpad]),
    },
    ControlEntry {
        action: DemoAction::Torque,
        label: "Joint torque (manual)",
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::RightStick]),
    },
    ControlEntry {
        action: DemoAction::Quit,
        label: "Quit",
        keyboard: KbBinding::new(&[DemoKey::Escape], &[]),
        pad: PadBinding::new(&[DemoPad::Start]),
    },
    ControlEntry {
        action: DemoAction::RevealControls,
        label: "Controls",
        keyboard: KbBinding::hold(&[DemoKey::Tab], &[]),
        pad: PadBinding::hold(&[DemoPad::View]),
    },
];

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
    /// Live training checkpoint dir the demo hot-reloads from while running (None
    /// disables). `last_loaded` is the mtime of the brain file last swapped in, so
    /// we reload only when training has written a newer one. See [`Self::try_hot_reload`].
    live_dir: Option<PathBuf>,
    last_loaded: Option<std::time::SystemTime>,
}

/// Load a brain + normalizer from `dir`, or `None` if the brain file is absent or
/// fails to parse. Returning `None` (rather than a zero-action fallback) lets a
/// hot-reload keep the policy it has when it races a mid-save write, instead of
/// blanking the running demo to a rest pose on a torn read.
fn load_brain_normalizer(
    dir: &Path,
    device: &NdArrayDevice,
) -> Option<(CrabBrain<InferBackend>, ObsNormalizer)> {
    let brain_stem = dir.join(BRAIN_STEM);
    if !brain_stem.with_extension("bin").exists() {
        return None;
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder.load(brain_stem, device).ok()?;
    let brain = CrabBrain::<TrainBackend>::new(device)
        .load_record(record)
        .valid();
    // A checkpoint from a different rig (e.g. a stale 77-dim brain against the
    // current OBS_SIZE) loads fine here but its mismatched first-layer weight would
    // panic in the matmul at the first `policy()` call. Reject it as if it were
    // missing — None routes to the same zero-action / keep-current fallback a missing
    // brain takes — so a stale `checkpoints/` degrades to the rest pose instead of
    // crashing the demo/screenshot window on launch (rl#36).
    let (obs_dim, action_dim) = brain.io_dims();
    if obs_dim != OBS_SIZE || action_dim != ACTION_SIZE {
        warn!(
            "play: checkpoint dims ({obs_dim} obs, {action_dim} act) don't match the \
             current rig ({OBS_SIZE} obs, {ACTION_SIZE} act) — ignoring it",
        );
        return None;
    }
    let mut normalizer = ObsNormalizer::new(5.0);
    let norm_path = dir.join(NORMALIZER_FILENAME);
    if norm_path.exists()
        && let Some(loaded) = ObsNormalizer::load(&norm_path)
    {
        normalizer = loaded;
    }
    Some((brain, normalizer))
}

impl Policy {
    /// Load brain + normalizer from a checkpoint dir. Missing/corrupt files fall
    /// back to a zero-action policy so the app still launches (useful before the
    /// first checkpoint exists, and to inspect the body's neutral rest pose).
    pub fn load(checkpoint_dir: &Path) -> Self {
        let device = NdArrayDevice::Cpu;
        let (brain, normalizer, loaded) = match load_brain_normalizer(checkpoint_dir, &device) {
            Some((brain, normalizer)) => {
                info!("play: loaded checkpoint from {}", checkpoint_dir.display());
                (brain, normalizer, true)
            }
            None => {
                warn!(
                    "play: no usable checkpoint at {} — using zero-action pose",
                    checkpoint_dir.display()
                );
                // Random-init brain; `act` ignores it (returns the rest pose) while
                // `loaded` is false, unless RL_RANDOM_POLICY opts in below.
                let brain = CrabBrain::<TrainBackend>::new(&device).valid();
                (brain, ObsNormalizer::new(5.0), false)
            }
        };

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
            live_dir: None,
            last_loaded: None,
        }
    }

    /// If the live training dir holds a brain file newer than the one we're
    /// running, swap it in; returns whether it did. Safe against a mid-save race:
    /// a torn read makes [`load_brain_normalizer`] return `None` and we keep the
    /// current policy rather than blanking the demo to a rest pose.
    fn try_hot_reload(&mut self) -> bool {
        let Some(dir) = self.live_dir.clone() else {
            return false;
        };
        let brain_bin = dir.join(BRAIN_STEM).with_extension("bin");
        let Ok(mtime) = std::fs::metadata(&brain_bin).and_then(|m| m.modified()) else {
            return false; // no live brain file yet
        };
        if self.last_loaded == Some(mtime) {
            return false; // already running this checkpoint
        }
        let Some((brain, normalizer)) = load_brain_normalizer(&dir, &self.device) else {
            return false; // mid-save / unreadable — keep the current policy
        };
        self.brain = brain;
        self.normalizer = normalizer;
        self.loaded = true;
        self.last_loaded = Some(mtime);
        true
    }

    /// Deterministic action: the policy mean (no exploration noise), so the crab
    /// holds a steady pose instead of jittering. `pub` so the game's solo NN-crab drives
    /// the same inference the demo does — one policy implementation, two callers.
    pub fn act(&self, raw_obs: &[f32; OBS_SIZE]) -> [f32; ACTION_SIZE] {
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

/// System (BotSet::Think): run the policy and write the actions the actuator will
/// apply — unless manual control has taken over (then `manual_control_step` drives).
fn policy_step(
    policy: NonSend<Policy>,
    manual: Option<Res<ManualControl>>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
) {
    if manual.is_some_and(|m| m.active) {
        return;
    }
    if let (Some(o), Some(a)) = (obs.envs.first(), actions.envs.first_mut()) {
        *a = policy.act(o);
    }
}

/// Load the trained policy as a resource. The driver system that turns it into
/// actions (`policy_step`) is added by the caller, so `--manual-control` can swap
/// in [`manual_control_step`] instead.
fn add_inference(app: &mut App, checkpoint_dir: &Path, live_dir: Option<PathBuf>) {
    let mut policy = Policy::load(checkpoint_dir);
    policy.live_dir = live_dir;
    app.insert_non_send_resource(policy);
}

/// Hands-on gamepad control state. `active` is toggled live with Y/triangle (and
/// seeded by `--manual-control`); `selected` is the joint the right stick drives
/// ([`CrabJointId::index`]), None = all joints at zero torque until the D-pad picks one.
#[derive(Resource)]
struct ManualControl {
    active: bool,
    selected: Option<usize>,
}

/// Marker for the on-screen manual-control readout (the mode + the live joint).
#[derive(Component)]
struct ManualHud;

/// System (BotSet::Think): hands-on gamepad control as an alternative to the policy.
/// Y/triangle toggles it; while active the D-pad up/down cycles which joint is live
/// and the right stick's Y drives THAT joint's torque (effort, not a target angle),
/// every other joint held at zero — a human feeling the joint dynamics by hand. When
/// inactive it only refreshes the HUD; `policy_step` drives.
fn manual_control_step(
    gamepads: Query<&Gamepad>,
    joint_ids: Query<&CrabJoint>,
    mut manual: ResMut<ManualControl>,
    mut actions: ResMut<CrabActions>,
    mut hud: Query<&mut Text, With<ManualHud>>,
) {
    let Some(gp) = gamepads.iter().next() else {
        return;
    };
    // East (B / circle), NOT North: North already toggles the joint-telemetry
    // graph (player::graph), so sharing it fired both on one press.
    if gp.just_pressed(GamepadButton::East) {
        manual.active = !manual.active;
        manual.selected = None;
    }

    let n = CrabJointId::COUNT;
    let mut line = "POLICY  (press B / circle for hands-on manual control)".to_string();
    if manual.active {
        if gp.just_pressed(GamepadButton::DPadUp) {
            manual.selected = Some(manual.selected.map_or(0, |i| (i + 1) % n));
        }
        if gp.just_pressed(GamepadButton::DPadDown) {
            manual.selected = Some(manual.selected.map_or(0, |i| (i + n - 1) % n));
        }
        if let Some(a) = actions.envs.first_mut() {
            // Drive only the selected joint; hold everything else at zero torque.
            *a = [0.0; CrabJointId::COUNT];
            line = match manual.selected {
                Some(sel) => {
                    let v = gp.right_stick().y.clamp(-1.0, 1.0);
                    a[sel] = v;
                    let name = joint_ids
                        .iter()
                        .find(|j| j.id.index() == sel)
                        .map(|j| format!("{:?}", j.id))
                        .unwrap_or_else(|| format!("#{sel}"));
                    format!(
                        "MANUAL  (B: exit   D-pad: pick joint   R-stick Y: torque)\n\
                         joint {sel}/{n}  {name}   torque {v:+.2}"
                    )
                }
                None => {
                    "MANUAL  (B: exit   D-pad up/down: pick a joint, then R-stick Y to actuate)"
                        .to_string()
                }
            };
        }
    }
    if let Ok(mut text) = hud.single_mut() {
        **text = line;
    }
}

/// Top-right readout of the active driver (policy vs manual) and the live joint.
/// Top-right because the joint-telemetry graph owns the top-left corner.
fn spawn_manual_hud(mut commands: Commands) {
    commands.spawn((
        // Overwritten every frame by `manual_control_step`; seed it with the idle
        // (policy) line so the pre-first-update frame doesn't flash a stale label.
        Text::new("POLICY  (press B / circle for hands-on manual control)"),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.9, 0.4)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            right: Val::Px(12.0),
            ..default()
        },
        ManualHud,
    ));
}

/// Interval between demo hot-reload checks. Loading the brain is cheap but not
/// free, and training saves no faster than its PPO-update cadence, so a couple of
/// seconds keeps the demo live without re-reading the file every frame.
const HOT_RELOAD_INTERVAL_S: f32 = 2.0;

/// System (demo only): every [`HOT_RELOAD_INTERVAL_S`], swap in a newer checkpoint
/// from the live training dir so a left-open demo tracks training without a
/// relaunch. No-op unless `--live-checkpoint-dir` was given. See
/// [`Policy::try_hot_reload`].
fn hot_reload_policy(time: Res<Time>, mut policy: NonSendMut<Policy>, mut since: Local<f32>) {
    *since += time.delta_secs();
    if *since < HOT_RELOAD_INTERVAL_S {
        return;
    }
    *since = 0.0;
    if policy.try_hot_reload() {
        info!("play: hot-reloaded a newer checkpoint from live training");
    }
}

// ---------------------------------------------------------------------------
// Demo: windowed, interactive
// ---------------------------------------------------------------------------

/// Windowed "play with the trained crab" mode.
pub struct DemoPlugin {
    pub checkpoint_dir: PathBuf,
    pub live_checkpoint_dir: Option<PathBuf>,
    /// Start in hands-on gamepad control instead of the policy (toggle live with
    /// Y). See [`manual_control_step`] — a physics feel-test, not a learned driver.
    pub manual_control: bool,
}

impl Plugin for DemoPlugin {
    fn build(&self, app: &mut App) {
        add_inference(app, &self.checkpoint_dir, self.live_checkpoint_dir.clone());
        crate::player::graph::register(app);
        // The reusable controls overlay (corner hint + hold-to-reveal panel), driven by the
        // demo's own DEMO_CONTROL_MAP — replaces the old static bottom-left HUD text.
        app.add_plugins(crate::controls::ControlsOverlayPlugin::<DemoControls>::default());
        app.init_resource::<DemoSettle>()
            .init_resource::<DemoRetilt>()
            .init_resource::<PokeBurst>()
            .add_systems(Startup, (spawn_orbit_camera, spawn_target_ball))
            .add_systems(Update, (orbit_camera, demo_controls))
            .add_systems(
                FixedUpdate,
                (
                    // Settle zeroes the actions a fresh crab holds, so it must land
                    // before Act applies them (not merely after Think) — otherwise the
                    // crab takes a tick of stale policy torque mid-drop. The
                    // Sense→Think→Act chain alone leaves settle-vs-Act to Bevy's
                    // implicit conflict ordering; pin it.
                    demo_settle.after(BotSet::Think).before(BotSet::Act),
                    // Both rebuild the crab; run before Sense (as the training rescue
                    // does) so the fresh pose is observed and the zeroed actions reach
                    // Act this tick, not next. The periodic re-tilt keeps the passive
                    // stream showing the goofy righting loop; the fall-rescue is the
                    // safety net when the policy walks off the arena.
                    demo_fall_rescue.before(BotSet::Sense),
                    demo_retilt.before(BotSet::Sense),
                    demo_poke.after(BotSet::Act).before(PhysicsSet::SyncBackend),
                    // After Sense so it reads the same post-physics claw-tip and target
                    // state the observation just consumed — touch detection and the ball
                    // then agree with what the policy saw this tick.
                    target_ball.after(BotSet::Sense),
                ),
            );
        // Both drivers are always present; `ManualControl.active` (seeded by the
        // flag, toggled live with Y) decides which one writes the actions each tick.
        app.insert_resource(ManualControl {
            active: self.manual_control,
            selected: None,
        })
        .add_systems(Startup, spawn_manual_hud)
        .add_systems(
            FixedUpdate,
            (policy_step, manual_control_step).in_set(BotSet::Think),
        )
        .add_systems(Update, hot_reload_policy);

        // RL_CLAW_DEMO: the right-claw inspection sweep, on the demo's wall-clock. The
        // drive overwrites the wrist/pincer slots after the policy wrote them (so it
        // wins those two DOFs while the policy keeps the rest); the pin holds the
        // carapace still so only the claw articulates. Gated entirely on the env var.
        if let Some(claw) = claw_demo_from_env() {
            app.insert_resource(claw)
                .init_resource::<PinBody>()
                .add_systems(
                    FixedUpdate,
                    claw_demo_drive.in_set(BotSet::Think).after(policy_step),
                )
                .add_systems(
                    FixedUpdate,
                    claw_demo_pin
                        .after(BotSet::Act)
                        .before(PhysicsSet::SyncBackend),
                );
        }
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

    // Keyboard orbit; -/= zoom. Right-yaw is the comma key, not the right arrow:
    // the right arrow toggles the collider wireframes (see `demo_controls`), and
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

/// Settle ticks remaining after a reset. The respawned crab starts in the
/// rest pose with the builder motors already holding it; the settle just
/// holds zero actions while it drops onto the ground and takes load. Seeded from
/// training's [`RESET_GRACE_TICKS`] and decremented via the shared
/// [`settle_countdown`] so the demo's drop window stays identical to the one the
/// policy was trained under (0 = settled, policy back in control).
#[derive(Resource, Default)]
struct DemoSettle(u32);

/// Wall-clock since the last demo re-tilt. A passive stream needs the goofy
/// righting "journey" on a loop: a crab that lands on its feet (or never falls)
/// would otherwise just stand there, so [`demo_retilt`] re-tilts on this timer
/// regardless. See [`DEMO_RETILT_PERIOD_S`].
#[derive(Resource)]
struct DemoRetilt {
    since: f32,
}

impl Default for DemoRetilt {
    fn default() -> Self {
        // Seed near the period so the first re-tilt fires a few seconds in — the
        // initial spawn is upright (shared spawn path, see `spawn_initial_crabs`),
        // so this is what makes the stream go goofy shortly after launch without
        // touching that path.
        Self {
            since: DEMO_RETILT_PERIOD_S - 3.0,
        }
    }
}

/// How often the demo re-tilts the crab to a fresh random orientation. Long
/// enough for a full righting attempt (succeed or, with current weights, flail and
/// fail) to play out and read clearly; short enough that the passive stream never
/// sits on a static pose.
const DEMO_RETILT_PERIOD_S: f32 = 9.0;

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
/// survives a corrupted multibody (see [`respawn_crab_rotated`]) — and hold zero
/// actions while it takes load. `init_rotation` is the spawn tilt: the demo feeds
/// a fresh [`body::random_spawn_rotation`] every time so the crab lands at a random
/// goofy angle and visibly tries to right itself, the "journey" the stream shows.
fn demo_respawn(
    commands: &mut Commands,
    assets: &CrabAssets,
    spawns: &CrabSpawns,
    parts: impl Iterator<Item = Entity>,
    settle: &mut DemoSettle,
    actions: &mut CrabActions,
    init_rotation: Quat,
) {
    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
    respawn_crab_rotated(commands, assets, parts, origin, 0, init_rotation);
    settle.0 = RESET_GRACE_TICKS;
    if let Some(a) = actions.envs.first_mut() {
        *a = [0.0; ACTION_SIZE];
    }
}

/// A fresh random goofy spawn tilt for a demo respawn (see
/// [`body::random_spawn_rotation`]): mostly mild, sometimes fully inverted, random
/// yaw — so the demo crab keeps landing at new angles to right itself from.
fn random_demo_tilt() -> Quat {
    body::random_spawn_rotation(&mut rand::thread_rng())
}

/// Marker on the demo's red target ball — the visible stand-in for the target the
/// policy is reaching for. Demo-only: training renders nothing and reads the target
/// straight from [`CrabTargets`].
#[derive(Component)]
struct TargetBall;

/// 3D euclidean distance at which the DEMO counts a claw tip as having reached the
/// target and teleports it to a fresh far point (see [`target_ball`]). Set at the edge
/// of the claw's reach. Because the reach `d` is 3D, this 0.8 m is a SPHERE about the
/// target, not a ground-plane cylinder: a tip standing under a raised ball is no longer
/// "reached" until it is within 0.8 m in 3D. The training REWARD pays the smooth
/// `1 − tanh(d/S)` everywhere with no threshold, so this radius never enters a reward; it
/// defines a binary "reached it" event in two non-reward places that benefit from one
/// shared definition: the demo's ball-hop here, and the training curriculum's
/// per-episode competence signal. DERIVED from that curriculum constant
/// ([`crate::training::session::CURRICULUM_REACH_RADIUS`]) — one source, in the
/// always-compiled trainer — so the demo and curriculum can't drift apart on the
/// radius. (0.8 m, as the doc above describes.)
pub(crate) const DEMO_REACH_RADIUS: f32 = crate::training::session::CURRICULUM_REACH_RADIUS;

/// Radius (m) of the demo target ball. Bigger than [`DEMO_REACH_RADIUS`] so the
/// claw visibly reaches *into* the ball before it registers a reach and jumps —
/// a marker you can see from the orbit camera, not a pinprick.
const TARGET_BALL_RADIUS: f32 = 0.08;

/// Startup (demo only): spawn the red target ball. Its world position is driven
/// every tick by [`target_ball`] off [`CrabTargets`] — the same state the policy
/// observes and the reward scores — so it is a pure marker, never a second source of
/// truth. The target itself is seeded by `target_ball` (in FixedUpdate, after the
/// Startup that sizes [`CrabTargets`]), so the ball starts at the origin and snaps to
/// its target on the first tick.
fn spawn_target_ball(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(TARGET_BALL_RADIUS))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.05, 0.05),
            // Emissive so the ball reads as a bright, self-lit dot regardless of
            // where the scene lighting falls — unmistakable against the crab/ground.
            emissive: LinearRgba::new(1.6, 0.0, 0.0, 1.0),
            ..default()
        })),
        Transform::from_translation(Vec3::ZERO),
        TargetBall,
    ));
}

/// FixedUpdate (demo only): drive the red ball off env 0's target, and TELEPORT the
/// target to a fresh far point when a claw tip reaches it. [`CrabTargets`] is the single
/// source of truth — the same resource the observation reads; here we seed/relocate it
/// and snap the ball to it, so the ball can never disagree with the target the policy
/// perceives. Seeding and relocation both reuse [`sample_target`], the exact FAR-target
/// rule training samples, so a demo target is always the same kind of walk-to goal the
/// policy was trained on.
///
/// **Intentional train/demo divergence.** Training uses ONE fixed target per episode
/// (no resample on reach): the reward is a pure distance field with no reach-event
/// bonus, so resampling-on-reach would let the optimal policy hover just outside the
/// reach radius forever instead of touching — a degenerate optimum (see
/// `session::brain_step`). The DEMO deliberately teleports on reach anyway, purely for
/// watchability: it keeps the crab walking continuously to new goals instead of parking
/// on one. This is safe because the policy learned "walk toward the current target
/// vector" and so generalizes to a target that moves — the demo just exercises that on
/// a livelier schedule than training. Reached is the closest 3D euclidean claw-tip-to-target
/// distance within [`DEMO_REACH_RADIUS`]. (The demo runs no training target system, so
/// this is the only writer of env 0's target; the initial seed happens here rather than
/// at Startup because `BotPlugin`'s Startup resize of [`CrabTargets`] would otherwise
/// race and clear it.)
fn target_ball(
    spawns: Res<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    claw_tips_q: Query<(&body::CrabEnvId, &Transform), With<CrabClawTip>>,
    mut ball_q: Query<&mut Transform, (With<TargetBall>, Without<CrabClawTip>)>,
) {
    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);

    // Seed on first tick (target unset) so the demo always has a reach to show. An
    // explicit `RL_TARGET_BALL_AT` (screenshot evidence frames) pins the seed to a
    // chosen point; otherwise sample the reach box. Seeding here, not at Startup,
    // dodges a race with `BotPlugin`'s Startup resize of `CrabTargets`.
    // The demo runs no curriculum (it isn't training), so it samples targets from the
    // fixed rung-1 band — a sensible near-to-mid range that reads well on the orbit
    // camera. A trained policy generalizes to any in-arena target, so the demo band need
    // not track the rung the weights were trained at.
    let demo_band = Curriculum::start();
    let mut target = match targets.get(0) {
        Some(t) => t,
        None => target_ball_at_from_env()
            .unwrap_or_else(|| sample_target(origin, demo_band, &mut rand::thread_rng())),
    };

    // Closest 3D euclidean distance from either claw tip to the target (the reward's
    // `d`, env 0) — 3D so the ball relocates at the same reached-moment training does
    // (the reward and this test MUST share one `d`; see `session::dist_3d`).
    let mut min_dist = f32::INFINITY;
    for (env, tip) in claw_tips_q.iter() {
        if env.0 == 0 && tip.translation.is_finite() {
            min_dist = min_dist.min(dist_3d(tip.translation, target));
        }
    }

    // Reached → relocate to a fresh far target (demo watchability only; see the fn
    // doc on why training does NOT do this). The ball follows the new target below.
    if min_dist <= DEMO_REACH_RADIUS {
        target = sample_target(origin, demo_band, &mut rand::thread_rng());
    }

    // Write the one source of truth (seed or relocation), then snap the ball to it.
    if let Some(slot) = targets.envs.first_mut() {
        *slot = Some(target);
    }
    if let Ok(mut ball) = ball_q.single_mut() {
        ball.translation = target;
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
    // Always present in the demo: the Rapier debug-render plugin is added
    // unconditionally so this toggle works (RL_DEBUG_COLLIDERS only sets the
    // initial on/off — see main.rs).
    mut debug_render: ResMut<DebugRenderContext>,
) {
    let mut reset = keys.just_pressed(KeyCode::KeyR);
    let mut poke = keys.just_pressed(KeyCode::Space);
    let mut quit = keys.just_pressed(KeyCode::Escape);
    // Right arrow / D-pad Right toggles the collider wireframes live. The arrow
    // keys are otherwise the orbit camera, but its right-yaw moved to the comma
    // key so this single binding isn't double-bound (mouse right-drag still orbits).
    let mut toggle_colliders = keys.just_pressed(KeyCode::ArrowRight);
    for gp in gamepads.iter() {
        reset |= gp.just_pressed(GamepadButton::South);
        poke |= gp.just_pressed(GamepadButton::West);
        quit |= gp.just_pressed(GamepadButton::Start);
        toggle_colliders |= gp.just_pressed(GamepadButton::DPadRight);
    }

    if quit {
        exit.write(AppExit::Success);
    }
    if toggle_colliders {
        debug_render.enabled = !debug_render.enabled;
        info!("demo collider wireframes: {}", debug_render.enabled);
    }
    if reset {
        // A manual reset re-tilts too, so the owner can re-roll the righting attempt.
        demo_respawn(
            &mut commands,
            &assets,
            &spawns,
            parts_q.iter(),
            &mut settle,
            &mut actions,
            random_demo_tilt(),
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
    // A fresh respawn (this tick or a settling one) holds the crab near the origin,
    // not fallen — skipping while it settles also stops a same-tick double-respawn
    // racing `demo_retilt`, since despawns are deferred and the stale carapace would
    // still read as fallen in this query until the command flush.
    if settle.0 > 0 {
        return;
    }
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
        random_demo_tilt(),
    );
}

/// System: periodically re-tilt the demo crab to a fresh random orientation so the
/// passive stream always shows the goofy righting "journey" — even when the crab
/// lands on its feet or never falls (the fall-rescue alone wouldn't fire then). Held
/// off while a settle is in progress so a re-tilt can't interrupt the previous spawn
/// before it has landed and had its moment. See [`DemoRetilt`].
// A Bevy system: every parameter is a scheduler-injected `Res`/`ResMut`/`Query`, so the
// arity is the dependency list, not a refactor smell — bundling them into a SystemParam
// would only hide the wiring.
#[allow(clippy::too_many_arguments)]
fn demo_retilt(
    time: Res<Time>,
    mut retilt: ResMut<DemoRetilt>,
    mut commands: Commands,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts_q: Query<Entity, With<CrabBodyPart>>,
    mut actions: ResMut<CrabActions>,
    mut settle: ResMut<DemoSettle>,
) {
    if settle.0 > 0 {
        // Don't advance the clock mid-settle: time the period from when the crab is
        // actually up and acting, not from the spawn it hasn't landed from yet.
        return;
    }
    retilt.since += time.delta_secs();
    if retilt.since < DEMO_RETILT_PERIOD_S {
        return;
    }
    retilt.since = 0.0;
    demo_respawn(
        &mut commands,
        &assets,
        &spawns,
        parts_q.iter(),
        &mut settle,
        &mut actions,
        random_demo_tilt(),
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
    // Same countdown training's reset path runs (see `settle_countdown`); spent →
    // 0 (settled), which the demo treats as "policy back in control".
    settle.0 = settle_countdown(settle.0);
}

/// Interactive right-claw inspection sweep (present only with `RL_CLAW_DEMO=1`): a
/// wrist+pincer sin/cos drive plus a body-pin, on the demo's wall-clock, so a viewer
/// can orbit/zoom a live, continuously articulating claw. `wrist`/`pincer` are resolved
/// through
/// [`CrabJointId::index`] (not hardcoded) so a rig reorder can't silently drive the
/// wrong slots. `f1`/`f2` (Hz, `RL_CLAW_DEMO_F1`/`F2`) are deliberately low and
/// unequal so the two DOFs trace a slow, non-repeating figure that reads clearly.
#[derive(Resource, Clone, Copy)]
struct ClawDemo {
    f1: f32,
    f2: f32,
    wrist: usize,
    pincer: usize,
}

/// `RL_CLAW_DEMO=1` enables the sweep; `RL_CLAW_DEMO_F1`/`F2` override the wrist/pincer
/// frequencies (Hz; defaults 0.08 / 0.12). Any other/absent `RL_CLAW_DEMO` value → None
/// (policy drives normally).
fn claw_demo_from_env() -> Option<ClawDemo> {
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
fn claw_demo_drive(
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
fn claw_demo_pin(
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
pub(crate) struct PinBody {
    pub(crate) target: Option<Transform>,
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
pub(crate) fn pin_correction(
    target: &Transform,
    xform: &Transform,
    vel: &Velocity,
) -> (Vec3, Vec3) {
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

/// Read `RL_TARGET_BALL_AT="x,y,z"` into an explicit world target for the screenshot
/// ball, so an evidence frame can place the red ball at a chosen point in the arena
/// (and a second frame at a moved point) deterministically, instead of a random
/// sample. `None` (the default) lets [`target_ball`] sample a fresh far target.
fn target_ball_at_from_env() -> Option<Vec3> {
    let raw = std::env::var("RL_TARGET_BALL_AT").ok()?;
    let parts: Vec<f32> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    match parts.as_slice() {
        [x, y, z] => Some(Vec3::new(*x, *y, *z)),
        _ => None,
    }
}

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        add_inference(app, &self.checkpoint_dir, None);
        app.add_systems(FixedUpdate, policy_step.in_set(BotSet::Think));
        // RL_TARGET_BALL=1: render the demo's red target ball in the screenshot too,
        // off the same `CrabTargets` state, so the reach-target viz can be inspected
        // headless. `target_ball` seeds (honoring RL_TARGET_BALL_AT), drives, and
        // teleports the target exactly as in the demo. Inert by default — a plain
        // screenshot is unchanged.
        if std::env::var_os("RL_TARGET_BALL").is_some() {
            app.add_systems(Startup, spawn_target_ball)
                .add_systems(FixedUpdate, target_ball.after(BotSet::Sense));
        }
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
        // RL_SKIN_DIAG: print the settled-pose point-in-mesh audit one frame before
        // the capture, so the table describes the exact frame the screenshot records.
        if std::env::var_os("RL_SKIN_DIAG").is_some() {
            crate::bot::skin_diag::register(app, self.settle);
        }
        // RL_SKIN_ALPHA: render the skin translucent so markers/colliders read through
        // it — the visually-unambiguous inside/outside check.
        if std::env::var_os("RL_SKIN_ALPHA").is_some() {
            crate::bot::skin_diag::register_translucent(app);
        }
    }
}

/// Default screenshot eye position relative to the tracked focus, and the fixed
/// height of that focus. The single source for the camera's framing.
const SHOT_CAM_OFFSET: Vec3 = Vec3::new(1.9, 0.95, 2.5);
const SHOT_CAM_FOCUS_Y: f32 = 0.5;

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
    let focus = Vec3::new(crab.translation.x, SHOT_CAM_FOCUS_Y, crab.translation.z);
    *cam = Transform::from_translation(focus + SHOT_CAM_OFFSET).looking_at(focus, Vec3::Y);
}

fn spawn_offscreen_camera(
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
    // settle counted in RENDER frames (the GPU pipeline renders black for the first
    // few dozen frames while shaders/pipelines compile and assets upload); the shared
    // helper owns that gate + the post-capture exit countdown.
    let Some(frame) = screenshot::advance_capture(&mut progress, cfg.settle, &mut exit) else {
        return;
    };

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
    screenshot::save_target_to(&mut commands, &target, cfg.path.clone());
    info!(
        "screenshot: captured at render frame {frame}, writing {}",
        cfg.path.display()
    );
    screenshot::finish_capture(&mut progress);
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Module;

    /// Per-app control-map invariant: every [`DemoAction`] has exactly one
    /// [`DEMO_CONTROL_MAP`] row, is bound on at least one device, and the reveal control
    /// resolves to a hint glyph. The exhaustive `match` forces a new variant to be added to
    /// `ALL` (so it can't slip in unmapped); the framework does the runtime checks. Manual /
    /// PickJoint / Torque are gamepad-only — legitimately unbound on keyboard.
    #[test]
    fn demo_map_is_well_formed() {
        use crate::controls::assert_map_well_formed;
        const ALL: [DemoAction; 11] = [
            DemoAction::Orbit,
            DemoAction::Zoom,
            DemoAction::Rebuild,
            DemoAction::Poke,
            DemoAction::Colliders,
            DemoAction::JointGraph,
            DemoAction::Manual,
            DemoAction::PickJoint,
            DemoAction::Torque,
            DemoAction::Quit,
            DemoAction::RevealControls,
        ];
        fn classified(a: DemoAction) -> bool {
            match a {
                DemoAction::Orbit
                | DemoAction::Zoom
                | DemoAction::Rebuild
                | DemoAction::Poke
                | DemoAction::Colliders
                | DemoAction::JointGraph
                | DemoAction::Manual
                | DemoAction::PickJoint
                | DemoAction::Torque
                | DemoAction::Quit
                | DemoAction::RevealControls => true,
            }
        }
        assert!(ALL.iter().copied().all(classified));
        assert_map_well_formed::<DemoControls>(&ALL);
    }

    /// Every icon glyph the demo map surfaces points under `controls/`; the text-keycap
    /// glyphs (arrows/zoom/G/LT/X/B/Wheel) are intentional and need no asset.
    #[test]
    fn demo_icon_glyphs_are_under_controls_dir() {
        use crate::controls::{Device, Glyph};
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            for e in &DEMO_CONTROL_MAP {
                for glyph in e.glyphs(device) {
                    if let Glyph::Icon(path) = glyph {
                        assert!(
                            path.starts_with("controls/") && path.ends_with(".png"),
                            "glyph path {path:?} for {:?}/{device:?} is malformed",
                            e.action
                        );
                    }
                }
            }
        }
    }

    /// Save a freshly-initialised brain into `dir` the way training does, so a
    /// hot-reload has a real checkpoint file to pick up.
    fn save_brain(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let brain = CrabBrain::<TrainBackend>::new(&device);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .record(brain.into_record(), dir.join(BRAIN_STEM))
            .unwrap();
    }

    /// The demo's "always fresh" guarantee: when training writes a new checkpoint
    /// into the live dir, the running policy swaps it in (flipping to `loaded`),
    /// and it does NOT reload the same file twice. Also pins the safe no-ops: no
    /// `live_dir`, and a live dir with no brain yet, both leave the policy alone.
    #[test]
    fn hot_reload_swaps_in_a_new_checkpoint() {
        let tmp = std::env::temp_dir();
        let live = tmp.join(format!("rl-hotreload-live-{}", std::process::id()));
        let empty = tmp.join(format!("rl-hotreload-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
        std::fs::create_dir_all(&empty).unwrap();

        // No checkpoint anywhere → unloaded (holds the zero-action rest pose).
        let mut policy = Policy::load(&empty);
        assert!(
            !policy.loaded,
            "empty checkpoint dir should give an unloaded policy"
        );
        assert!(
            !policy.try_hot_reload(),
            "no live_dir set → nothing to reload"
        );

        // Point at a live dir that has no brain yet → still a no-op.
        policy.live_dir = Some(live.clone());
        assert!(
            !policy.try_hot_reload(),
            "live dir without a brain → no reload"
        );

        // Training writes a checkpoint → the policy picks it up exactly once.
        save_brain(&live);
        assert!(
            policy.try_hot_reload(),
            "a new brain in the live dir must reload"
        );
        assert!(
            policy.loaded,
            "a successful hot-reload marks the policy loaded"
        );
        assert!(
            !policy.try_hot_reload(),
            "the same checkpoint must not reload again"
        );

        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// Save a brain whose first trunk layer expects `obs_dim` inputs instead of the
    /// current `OBS_SIZE` — the on-disk shape a checkpoint from an older rig has. We
    /// can't get one from `CrabBrain::new` (it bakes in today's `OBS_SIZE`), so swap
    /// the `trunk_fc1` weight in the record for a `[obs_dim, HIDDEN]` tensor before
    /// recording. This is exactly the file that used to reach the matmul and panic.
    fn save_brain_with_obs_dim(dir: &Path, obs_dim: usize) {
        use burn::module::{Param, ParamId};
        std::fs::create_dir_all(dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let mut record = CrabBrain::<TrainBackend>::new(&device).into_record();
        let [_obs, hidden] = record.trunk_fc1.weight.shape().dims();
        let weight = Tensor::<TrainBackend, 2>::zeros([obs_dim, hidden], &device);
        record.trunk_fc1.weight = Param::initialized(ParamId::new(), weight);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder.record(record, dir.join(BRAIN_STEM)).unwrap();
    }

    /// rl#36: a checkpoint built for a different `OBS_SIZE` must degrade to the
    /// zero-action rest pose (as a missing checkpoint does), NOT panic in the matmul.
    /// Loading must leave the policy unloaded and `act` must return zeros without ever
    /// running the mismatched weights through a forward pass.
    #[test]
    fn dim_mismatched_checkpoint_falls_back_instead_of_panicking() {
        let tmp = std::env::temp_dir();
        let dir = tmp.join(format!("rl-dimmismatch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // A stale brain expecting OBS_SIZE+4 inputs (mirrors the seen 77-vs-73 drift).
        save_brain_with_obs_dim(&dir, OBS_SIZE + 4);

        let policy = Policy::load(&dir);
        assert!(
            !policy.loaded,
            "a dim-mismatched checkpoint must fall back to unloaded, not load"
        );
        // The real regression: this call hits the matmul for a loaded policy; with the
        // fallback it returns zeros and never touches the mismatched weights.
        assert_eq!(
            policy.act(&[0.0; OBS_SIZE]),
            [0.0; ACTION_SIZE],
            "an unloaded policy holds the zero-action pose"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
