//! First-person Bevy client for the deterministic gray-box (rl#38 render sub).
//!
//! This is the windowed `play` mode of the `game` binary: it makes the
//! giant-crab-rescue sim VISIBLE and PLAYABLE on top of the existing lockstep +
//! transport netcode. The split it honors is the one documented at the top of
//! [`crate::net::sim`]: **the sim is the authority, this client is a read-only
//! consumer that produces [`Input`]**. Rendering, the camera, mouse/gamepad input,
//! and tween interpolation are ALL client-side and add ZERO nondeterminism — the
//! only thing that ever crosses back into sim state is the per-tick [`Input`] each
//! peer broadcasts. Two peers running this client off the same input stream stay
//! bit-identical because none of the code here touches the sim except through
//! [`Lockstep::submit_local_input`].
//!
//! How the three layers wire together:
//! - **Lockstep** runs on a fixed-timestep accumulator ([`drive_lockstep`]) inside
//!   the Bevy app, NOT in Bevy's `FixedUpdate` — the sim's tick rate ([`TICK_HZ`])
//!   is its own clock, independent of the render/display rate. Each ready tick:
//!   drain the local [`PendingInput`] into `submit_local_input`, pump the transport
//!   (broadcast our [`TickMsg`], ingest peers'), then `try_advance`.
//! - **Render** ([`apply_transforms`]) reads `Lockstep::sim()` and tweens every
//!   entity between the previous tick's pose and the current one by the fractional
//!   accumulator, so motion is smooth at any frame rate even though the sim steps in
//!   discrete 30 Hz jumps.
//! - **Input** ([`gather_input`]) samples WASD + mouse + gamepad every render frame
//!   into [`PendingInput`]; the lockstep driver quantizes it to one [`Input`] per
//!   tick. Look pitch is integrated here and kept client-side (the sim models yaw
//!   only); the camera reads the authoritative yaw back from the sim.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use bevy::app::AppExit;
use bevy::camera::RenderTarget;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow, WindowMode};

use crate::net::lockstep::{Lockstep, TickMsg};
use crate::net::net_loop::{NetDriver, PeerMsg};
use crate::net::sim::{
    CRAB_SCALE, Crab, EXTRACT_RADIUS, Input, Outcome, Plane, Player, PlayerId, PlayerStatus, Pos,
    Pos3, Sim, UNIT, buttons, trig, trig_client,
};
use crate::net::telemetry::{TELEMETRY_TICK_EVERY, TelemetryEvent};

/// Sim tick rate (Hz). The deterministic sim advances at this fixed rate on every
/// peer; the client renders faster and interpolates between ticks. Matches the
/// headless driver's rate so a render peer and a headless peer stay in lockstep.
pub const TICK_HZ: u64 = 30;

/// Seconds per sim tick — the fixed dt the lockstep accumulator drains in.
const TICK_DT: f64 = 1.0 / TICK_HZ as f64;

/// Most sim ticks to apply in a single render frame, so a long stall (window drag,
/// GPU hitch) can't trigger an unbounded catch-up spiral that freezes the client.
/// Extra accumulated time past this is dropped — the sim falls a little behind real
/// time rather than locking up.
const MAX_TICKS_PER_FRAME: u32 = 8;

/// Eye height of the first-person camera above the player's ground position, in
/// meters (a ~1.8 m capsule on the ground at Y=0; eyes near the top).
const EYE_HEIGHT: f32 = 1.6;

/// Player capsule dimensions (meters): a person-sized avatar for the other peers.
const PLAYER_RADIUS: f32 = 0.4;
const PLAYER_HEIGHT: f32 = 1.8;

/// Plane gray-box dimensions (meters): a fuselage box + a wider, thinner wing box.
/// Just enough shape to read as an aircraft and show its facing — a placeholder, like
/// the crab box.
const PLANE_FUSELAGE_LEN: f32 = 6.0;
const PLANE_FUSELAGE_W: f32 = 1.2;
const PLANE_WINGSPAN: f32 = 9.0;
const PLANE_WING_CHORD: f32 = 1.6;

/// Mouse look sensitivity (radians per pixel of motion). Yaw feeds the sim as a
/// per-tick delta; pitch stays client-side.
const MOUSE_SENS: f32 = 0.0022;

/// Gamepad look speed (radians/second at full right-stick deflection), scaled by the
/// frame dt so it's frame-rate independent.
const PAD_LOOK_SPEED: f32 = 2.5;

/// Pitch clamp (radians) so the FP camera can't flip over the poles.
const PITCH_LIMIT: f32 = 1.5;

/// Convert a sim fixed-point coordinate to meters.
fn meters(coord: i64) -> f32 {
    coord as f32 / UNIT as f32
}

/// A sim ground position (XZ at Y=0) as a Bevy world point at height `y`. The sim's
/// right-handed XZ frame (+X right, +Z forward, +Y up) IS Bevy's frame, so this is a
/// direct unit conversion with no axis remap.
fn world(pos: Pos, y: f32) -> Vec3 {
    Vec3::new(meters(pos.x), y, meters(pos.z))
}

/// A sim 3D position ([`Pos3`], includes altitude) as a Bevy world point — the same
/// direct unit conversion as [`world`], but with the entity's own Y (a flying plane),
/// not an externally supplied ground height.
fn world3(pos: Pos3) -> Vec3 {
    Vec3::new(meters(pos.x), meters(pos.y), meters(pos.z))
}

/// Build the windowed first-person client app (no network = solo, `net = Some` =
/// real peers). Owns the `Lockstep` + optional `NetDriver` as resources; the caller
/// `run()`s it. Built here, not via a `Plugin` that holds the sim, because
/// `Plugin::build(&self)` can't move a non-`Clone` `Lockstep`/`NetDriver` out of
/// itself — inserting them as resources at construction is the clean path.
pub fn build_windowed_app(ls: Lockstep, net: Option<NetDriver>) -> App {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Giant Crab Rescue — first person".into(),
            mode: WindowMode::Windowed,
            ..default()
        }),
        ..default()
    }));
    let source = match net {
        Some(n) => InputSource::Networked(n),
        None => InputSource::Solo,
    };
    insert_core(&mut app, ls, source);
    app.add_systems(Startup, (spawn_world, spawn_fp_camera, spawn_hud))
        .add_systems(
            Update,
            (gather_input, drive_lockstep, apply_transforms, update_hud).chain(),
        )
        .add_systems(Update, (grab_cursor_once, exit_on_esc));
    app
}

/// Build the HEADLESS screenshot app: GPU on, no window, render one settled frame of
/// the FP view to `path` and exit. The evidence path on a box with no display — it
/// proves the sim→render pipeline (crab, extraction marker, another player from the
/// local eyes) without 2-peer play. Solo only (no transport): one peer's input
/// completes every tick, which is all a single-frame render needs.
pub fn build_screenshot_app(ls: Lockstep, cfg: ScreenshotConfig) -> App {
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
    app.insert_resource(cfg)
        .init_resource::<ShotProgress>()
        .add_systems(Startup, (spawn_world, spawn_offscreen_camera, spawn_hud))
        .add_systems(
            Update,
            (
                gather_input,
                drive_lockstep,
                apply_transforms,
                apply_shot_cam_offset,
                update_hud,
                capture_when_settled,
            )
                .chain(),
        );
    app
}

/// Shared setup for both apps: the sim + its input source, plus the input resources.
fn insert_core(app: &mut App, ls: Lockstep, input_source: InputSource) {
    let prev = SimSnapshot::capture(ls.sim());
    app.insert_non_send_resource(GameState {
        ls,
        input_source,
        accumulator: 0.0,
        prev,
    })
    .init_resource::<PendingInput>()
    .init_resource::<CameraPitch>()
    .init_resource::<CameraYaw>();
}

// ---------------------------------------------------------------------------
// Lockstep driver state (render-agnostic: owns the sim + transport)
// ---------------------------------------------------------------------------

/// Where the OTHER players' per-tick inputs come from this run. One field, three
/// mutually-exclusive cases — so "broadcast to real peers AND fabricate bot inputs"
/// (a meaningless combination) is unrepresentable rather than merely unreached.
enum InputSource {
    /// Real peers over the transport (windowed networked play): broadcast our tick
    /// message and ingest theirs.
    Networked(NetDriver),
    /// A single offline peer (windowed solo): our own input completes every tick, so
    /// there are no other inputs to supply.
    Solo,
    /// Stand-in input for the absent peers, fed for every non-local player each tick
    /// (headless screenshot only). It crosses the SAME deterministic `record_remote`
    /// path a wire peer would, so the sim can't distinguish it — a bot/replay input,
    /// not a back channel. Only ever a single-machine solo run, so no peer exists to
    /// disagree with it.
    Scripted(Input),
}

/// The networked sim, owned as a non-send Bevy resource and stepped on a
/// fixed-timestep accumulator. Non-send because [`NetDriver`] holds a tokio runtime
/// + the iroh session (not `Sync`); only the main thread drives it, so that's fine.
struct GameState {
    ls: Lockstep,
    /// Where the other players' inputs come from this run (real peers / none / a
    /// scripted stand-in). The sole writer of inputs other than the local controls.
    input_source: InputSource,
    /// Fractional-tick accumulator: render time elapsed since the last applied sim
    /// tick, in [0, TICK_DT). Drives both how many ticks to step and the render
    /// interpolation alpha.
    accumulator: f64,
    /// Renderable sim state as of the PREVIOUS applied tick. Render tweens from this
    /// toward the live sim by `alpha`. A snapshot (not the live sim) because we need
    /// "last tick" even after the sim has stepped to the current one.
    prev: SimSnapshot,
}

/// A minimal copy of the renderable sim state at one tick — the poses the client
/// tweens from. NOT a second source of truth: overwritten from the authoritative
/// [`Sim`] every tick, never fed back into it.
#[derive(Clone, Default)]
struct SimSnapshot {
    players: BTreeMap<PlayerId, Player>,
    planes: BTreeMap<PlayerId, Plane>,
    crab: Option<Crab>,
}

impl SimSnapshot {
    fn capture(sim: &Sim) -> Self {
        Self {
            players: sim.players().collect(),
            planes: sim.planes().collect(),
            crab: Some(sim.crab()),
        }
    }
}

/// The local player's input accumulated this render interval, applied to the sim at
/// the next tick boundary. Move axes are sampled each frame (last frame wins); the
/// yaw-look delta ACCUMULATES across the render frames between two ticks so no mouse
/// motion is dropped, and `action` latches if pressed any frame in the interval.
/// All drained when a tick consumes it.
#[derive(Resource, Default)]
struct PendingInput {
    strafe: f32,
    forward: f32,
    /// Accrued yaw-look this inter-tick interval, in radians (drained per tick).
    yaw_delta: f32,
    action: bool,
    /// Latches if RESTART (R) was pressed in this interval. Sent as
    /// [`buttons::RESTART`] so the restart rides the deterministic input stream and all
    /// peers restart on the same tick; the sim edge-triggers it. Drained per tick.
    restart: bool,
}

/// Client-side camera pitch (radians), integrated from look input. The sim models
/// only yaw (feet); pitch lives here and never reaches the sim, per the interface.
#[derive(Resource, Default)]
struct CameraPitch(f32);

/// Client-side camera YAW (radians), integrated from the same look input every frame.
/// While the local player is Alive the camera uses the AUTHORITATIVE sim yaw (so it
/// agrees with the avatar and peers) and this tracks it; once the player is downed or
/// extracted the sim freezes their yaw, so the camera falls back to this free value —
/// giving a spectator full free-look (yaw AND pitch), not pitch-only. Never reaches the
/// sim, so it adds no nondeterminism (a dead player's facing affects nothing).
#[derive(Resource, Default)]
struct CameraYaw(f32);

/// The sim's per-tick yaw turn cap, in radians. The sim clamps a tick's yaw delta to
/// `trig::TURN/24` turn-units (see [`crate::net::sim`]); we normalize our accrued
/// look radians by this same cap so full `look_yaw` deflection means exactly "the
/// most the sim turns in one tick" — commanding more would only make the camera lag
/// the avatar, since the sim would clamp it. Derived from the same integer `trig::TURN`
/// the sim uses, so the two can't drift.
const MAX_YAW_PER_TICK_RADIANS: f32 =
    (trig::TURN / 24) as f32 / trig::TURN as f32 * std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// Entity markers
// ---------------------------------------------------------------------------

/// A rendered avatar for sim player `id`. The local player's own avatar is hidden
/// (we see from its eyes) but still spawned so status handling stays uniform.
#[derive(Component)]
struct PlayerAvatar(PlayerId);

/// A rendered gray-box plane for the pilot with this id. The local pilot's own plane
/// is hidden (we view from its cockpit), like the local player's capsule.
#[derive(Component)]
struct PlaneAvatar(PlayerId);

/// The giant crab placeholder.
#[derive(Component)]
struct CrabAvatar;

/// The first-person camera, anchored to the local player each frame.
#[derive(Component)]
struct FpCamera;

/// The HUD status line (local Alive/Downed/Extracted + the round outcome).
#[derive(Component)]
struct StatusHud;

// ---------------------------------------------------------------------------
// Lockstep driver system
// ---------------------------------------------------------------------------

/// Advance the lockstep sim by real time on a fixed-timestep accumulator. This is
/// the ONLY writer of sim state, and it writes exactly one thing: the local
/// [`Input`] (drained from [`PendingInput`]) via `submit_local_input`. Everything
/// else is the existing deterministic machinery — pump the transport, then
/// `try_advance`. A desync fault is logged (lockstep can't recover); the client
/// keeps running so the operator sees it rather than a silent freeze.
fn drive_lockstep(
    mut state: NonSendMut<GameState>,
    mut pending: ResMut<PendingInput>,
    time: Res<Time>,
    mut reported_outcome: Local<bool>,
    mut next_tel_tick: Local<u64>,
    // Last sim tick this system saw, to detect a deterministic restart (RESTART rewinds
    // the sim to tick 0). When it does, the round-decided latch and telemetry cursor
    // below must reset, or the NEXT round never reports "decided" and tick-telemetry
    // stays suppressed until the counter climbs back past the stale watermark.
    mut last_tick: Local<u64>,
    // Cumulative lockstep fault count across the whole round (persists between system
    // runs), so telemetry reports the REAL running desync total — not a per-frame 0. This
    // is the live-debug alarm: a non-zero value on any deck means it has diverged.
    mut total_desyncs: Local<usize>,
) {
    state.accumulator += time.delta().as_secs_f64();

    // Clone out the telemetry handle (cheap: an mpsc sender + id) + the roster size so we
    // can READ the sim and push events without holding a borrow of `state.input_source`
    // (where the NetDriver that owns it lives). `None` unless this is networked play with
    // a collector. Telemetry never writes the sim — it only reports it.
    let (tel, roster_len) = match &state.input_source {
        InputSource::Networked(net) => (net.telemetry().cloned(), net.roster_len()),
        _ => (None, 0),
    };
    if *next_tel_tick == 0 {
        *next_tel_tick = TELEMETRY_TICK_EVERY;
    }

    let mut applied = 0u32;
    while state.accumulator >= TICK_DT && applied < MAX_TICKS_PER_FRAME {
        state.accumulator -= TICK_DT;
        applied += 1;

        // Snapshot the pre-step state for interpolation, then build THIS tick's local
        // input from the accumulated controls and hand it to the deterministic driver.
        state.prev = SimSnapshot::capture(state.ls.sim());

        let look_axis = (pending.yaw_delta / MAX_YAW_PER_TICK_RADIANS).clamp(-1.0, 1.0);
        let btns = (if pending.action { buttons::ACTION } else { 0 })
            | (if pending.restart { buttons::RESTART } else { 0 });
        let input = Input::new(pending.strafe, pending.forward, look_axis, btns);
        // Drain the accrued look + latched buttons; movement axes are re-sampled next frame.
        pending.yaw_delta = 0.0;
        pending.action = false;
        pending.restart = false;

        let me = state.ls.me();
        let issue_tick = state.ls.next_tick();
        let msg = state.ls.submit_local_input(input);

        // Supply the OTHER players' inputs for this tick from whichever source this run
        // uses, then record them into the deterministic driver. Collect them into a Vec
        // first — releasing the borrow of `state.input_source`/`state.ls` — before
        // recording into `state.ls`, since the source (`NetDriver`) and `ls` are fields
        // of the same non-send resource and an overlapping borrow wouldn't type-check.
        // Every path lands at `record_remote`, the same entry a wire peer takes, so the
        // sim can't tell the sources apart. Split the field borrows via `&mut *state`.
        let st = &mut *state;
        let peer_msgs: Vec<PeerMsg> = match &mut st.input_source {
            InputSource::Networked(net) => {
                net.broadcast(&msg);
                net.drain_inbox()
            }
            InputSource::Solo => Vec::new(),
            InputSource::Scripted(bot) => {
                // Stand in for the absent peers so the (otherwise-stalled) sim advances:
                // feed every non-local player this input at the SAME apply_tick the local
                // input got. Always a single-machine solo run, so no peer disagrees.
                let bot = *bot;
                st.ls
                    .sim()
                    .players()
                    .map(|(id, _)| id)
                    .filter(|&id| id != me)
                    .map(|pid| PeerMsg {
                        pid,
                        msg: TickMsg {
                            apply_tick: msg.apply_tick,
                            input: bot,
                            confirmed: None,
                        },
                    })
                    .collect()
            }
        };
        for from in peer_msgs {
            if from.pid != me
                && let Some(fault) = state.ls.record_remote(from.pid, from.msg)
            {
                *total_desyncs += 1;
                warn!("lockstep fault: {fault:?}");
                if let Some(t) = &tel {
                    t.send(TelemetryEvent::fault(&fault));
                }
            }
        }

        for fault in state.ls.try_advance() {
            *total_desyncs += 1;
            warn!("lockstep fault: {fault:?}");
            if let Some(t) = &tel {
                t.send(TelemetryEvent::fault(&fault));
            }
        }

        // Sampled telemetry: a Tick snapshot + the local input every TELEMETRY_TICK_EVERY
        // applied ticks. Read-only on the sim; best-effort (drops if the link can't keep
        // up). `roster` is the agreed player count (sync + accurate), `desyncs` is the real
        // running fault total, and the (tick, hash) is what the cross-peer desync check
        // needs.
        if let Some(t) = &tel
            && state.ls.sim().tick() >= *next_tel_tick
        {
            *next_tel_tick =
                (state.ls.sim().tick() / TELEMETRY_TICK_EVERY + 1) * TELEMETRY_TICK_EVERY;
            t.send(TelemetryEvent::tick(state.ls.sim(), *total_desyncs, roster_len));
            t.send(TelemetryEvent::input(issue_tick, input));
        }
    }

    if applied == MAX_TICKS_PER_FRAME {
        // Shed the backlog rather than spiral: drop accumulated time past one tick.
        state.accumulator = state.accumulator.min(TICK_DT);
    }

    // Restart detector: a RESTART press rewinds the sim to tick 0, so a tick lower than
    // last frame's means the round restarted. Clear the round-decided latch so the new
    // round can report its own outcome, and snap the telemetry cursor back to the new
    // (low) tick so sampled telemetry resumes immediately instead of waiting out a stale
    // watermark.
    let now_tick = state.ls.sim().tick();
    if now_tick < *last_tick {
        *reported_outcome = false;
        *next_tel_tick = (now_tick / TELEMETRY_TICK_EVERY + 1) * TELEMETRY_TICK_EVERY;
    }
    *last_tick = now_tick;

    if !*reported_outcome && state.ls.sim().outcome() != Outcome::Ongoing {
        *reported_outcome = true;
        info!("round decided: {:?}", state.ls.sim().outcome());
        if let Some(t) = &tel {
            t.send(TelemetryEvent::round_decided(state.ls.sim()));
        }
    }
}

// ---------------------------------------------------------------------------
// Input: WASD + mouse + gamepad → PendingInput
// ---------------------------------------------------------------------------

/// Sample local controls each render frame into [`PendingInput`] and integrate the
/// client-side camera pitch. Produces ONLY data destined for the next [`Input`] —
/// it never touches the sim. WASD/left-stick = move; mouse/right-stick = look
/// (yaw → sim, pitch → client); Space/RT or gamepad South = action.
#[allow(clippy::too_many_arguments)]
fn gather_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    cursor: Query<&CursorOptions, With<PrimaryWindow>>,
    mut pending: ResMut<PendingInput>,
    mut pitch: ResMut<CameraPitch>,
    mut yaw: ResMut<CameraYaw>,
) {
    let dt = time.delta_secs();

    // --- Move axes (last sample wins; re-sampled every frame) ---
    let mut strafe = 0.0f32;
    let mut forward = 0.0f32;
    if keys.pressed(KeyCode::KeyW) {
        forward += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        forward -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        strafe += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        strafe -= 1.0;
    }

    let mut action = keys.pressed(KeyCode::Space);
    // Restart the round (R). Latched here, sent as buttons::RESTART, edge-triggered in
    // the sim — so it restarts every peer in lockstep, not a local-only reset.
    if keys.just_pressed(KeyCode::KeyR) {
        pending.restart = true;
    }

    // --- Look (accumulated across frames) ---
    let mut d_yaw = 0.0f32;
    let mut d_pitch = 0.0f32;

    // Mouse look only when the cursor is grabbed (windowed play). In headless
    // screenshot mode there's no window/cursor, so this is simply skipped.
    let grabbed = cursor
        .iter()
        .next()
        .is_some_and(|c| c.grab_mode != CursorGrabMode::None);
    if grabbed {
        let d = mouse_motion.delta;
        d_yaw += d.x * MOUSE_SENS;
        d_pitch -= d.y * MOUSE_SENS;
    }

    // Gamepad: left stick moves, right stick looks, RT/South = action. Sticks have a
    // deadzone so a resting stick doesn't creep.
    for gp in gamepads.iter() {
        let ls = gp.left_stick();
        if ls.length() > 0.15 {
            strafe += ls.x;
            forward += ls.y;
        }
        let rs = gp.right_stick();
        if rs.length() > 0.15 {
            d_yaw += rs.x * PAD_LOOK_SPEED * dt;
            d_pitch += rs.y * PAD_LOOK_SPEED * dt;
        }
        action |= gp.pressed(GamepadButton::South) || gp.pressed(GamepadButton::RightTrigger2);
    }
    // Mouse-left also fires action, for mouse-only play.
    action |= mouse_buttons.pressed(MouseButton::Left);

    // Reconcile screen-right with the sim's X axis. The sim labels +X "strafe right"
    // and increasing yaw turns +Z toward +X — but a camera looking along +Z (yaw 0)
    // has its right axis at world −X, so world +X renders on the SCREEN-LEFT. Feeding
    // the player's "right" intents straight through would move the avatar and pan the
    // view the wrong way. Negating the two X-axis control intents (strafe and the
    // yaw-look delta) here — and only here — makes D / mouse-right / right-stick read
    // as screen-right, while the sim frame and the faithful world rendering stay
    // untouched (forward and pitch carry no X, so they're unaffected).
    pending.strafe = (-strafe).clamp(-1.0, 1.0);
    pending.forward = forward.clamp(-1.0, 1.0);
    pending.yaw_delta -= d_yaw;
    pending.action |= action;

    pitch.0 = (pitch.0 + d_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    // Integrate the client-side free-look yaw from the SAME (screen-corrected) delta the
    // sim yaw gets, so while alive it tracks the avatar and when dead it free-looks
    // seamlessly from the last facing. Wrap to keep it bounded over a long spectate.
    yaw.0 = (yaw.0 - d_yaw).rem_euclid(std::f32::consts::TAU);
}

/// Quit the game on Esc (windowed play only). Purely client-local — sends Bevy's
/// [`AppExit`] to end the run; no sim/lockstep involvement, so it can't desync a peer
/// (each client just closes its own window). The other peers play on.
fn exit_on_esc(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

/// Grab + hide the cursor once the window's [`CursorOptions`] exist, so mouse-look
/// works and the pointer stays captured. Runs every frame but no-ops after the first
/// successful grab. (Grabbing AFTER the window is live, rather than via the plugin's
/// initial options, avoids a too-early lock failing on some platforms.)
fn grab_cursor_once(
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    if let Ok(mut c) = cursor.single_mut() {
        c.grab_mode = CursorGrabMode::Locked;
        c.visible = false;
        *done = true;
    }
}

// ---------------------------------------------------------------------------
// Scene + interpolated transforms
// ---------------------------------------------------------------------------

/// Spawn the static gray-box world (ground + extraction marker + a light) and the
/// dynamic avatars (one capsule per sim player, the scaled crab). Poses are placed
/// every frame by [`apply_transforms`]; here we just create the meshes once.
fn spawn_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: NonSend<GameState>,
) {
    // Ground: a large gray plane at Y=0.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(400.0, 400.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.32, 0.34),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Sun-ish directional light so the gray-box reads with shape, plus a little
    // ambient so shadowed faces aren't pure black.
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(20.0, 40.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(bevy::light::GlobalAmbientLight {
        brightness: 220.0,
        ..default()
    });

    // Extraction point: a translucent green cylinder of the sim's EXTRACT_RADIUS,
    // capped with a bright pillar so it's unmistakable from across the map — the
    // objective marker.
    let ex = state.ls.sim().extraction().pos();
    let r = meters(EXTRACT_RADIUS);
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(r, 0.1))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.1, 0.9, 0.3, 0.55),
            emissive: LinearRgba::new(0.0, 1.2, 0.2, 1.0),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_translation(world(ex, 0.05)),
    ));
    // A tall bright glowing pillar at the point — the objective beacon. Made taller
    // than the giant crab (CRAB_SCALE players high) and thick enough to read at the
    // far end of the map, so the goal stays legible even when the towering crab is
    // between you and it.
    let pillar_h = PLAYER_HEIGHT * CRAB_SCALE as f32 * 1.2;
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.5, pillar_h))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.95, 0.3),
            emissive: LinearRgba::new(0.0, 2.2, 0.4, 1.0),
            ..default()
        })),
        Transform::from_translation(world(ex, pillar_h * 0.5)),
    ));

    // Player avatars: one capsule per sim player. The local player's is spawned too
    // (kept hidden in apply_transforms — we view from its eyes).
    let local = state.ls.me();
    for (id, _p) in state.ls.sim().players() {
        let is_local = id == local;
        let color = if is_local {
            Color::srgb(0.9, 0.8, 0.2)
        } else {
            Color::srgb(0.2, 0.5, 0.95)
        };
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(
                PLAYER_RADIUS,
                PLAYER_HEIGHT - 2.0 * PLAYER_RADIUS,
            ))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                ..default()
            })),
            Transform::from_translation(world(state.ls.sim().player(id).unwrap().pos(), 0.0)),
            PlayerAvatar(id),
        ));
    }

    // Pilot planes: one gray-box aircraft (fuselage + wing) per plane in the sim. The
    // root holds the pose (placed every frame by apply_transforms); the children give
    // it shape and a legible facing (+Z = nose, matching heading 0). The local pilot's
    // is spawned too but hidden in apply_transforms (cockpit view).
    let plane_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.62, 0.64, 0.67),
        perceptual_roughness: 0.7,
        ..default()
    });
    for (id, _plane) in state.ls.sim().planes() {
        let root = commands
            .spawn((
                Transform::from_translation(world3(state.ls.sim().plane(id).unwrap().pos())),
                Visibility::default(),
                PlaneAvatar(id),
            ))
            .id();
        // Fuselage: a long box down +Z (the nose direction).
        let fuselage = commands
            .spawn((
                Mesh3d(meshes.add(Cuboid::new(
                    PLANE_FUSELAGE_W,
                    PLANE_FUSELAGE_W,
                    PLANE_FUSELAGE_LEN,
                ))),
                MeshMaterial3d(plane_mat.clone()),
                Transform::default(),
            ))
            .id();
        // Wing: a wide, thin box across X, set a bit forward of center.
        let wing = commands
            .spawn((
                Mesh3d(meshes.add(Cuboid::new(
                    PLANE_WINGSPAN,
                    PLANE_FUSELAGE_W * 0.25,
                    PLANE_WING_CHORD,
                ))),
                MeshMaterial3d(plane_mat.clone()),
                Transform::from_xyz(0.0, 0.0, PLANE_FUSELAGE_LEN * 0.1),
            ))
            .id();
        commands.entity(root).add_children(&[fuselage, wing]);
    }

    // The giant crab: a big menacing box, CRAB_SCALE× a player, with a "head" wedge
    // so its facing is legible. Gray-box placeholder — the trained RL crab body is a
    // later concern (per the sim interface note).
    let crab_h = PLAYER_HEIGHT * CRAB_SCALE as f32;
    let crab_w = PLAYER_RADIUS * 2.0 * CRAB_SCALE as f32;
    let crab_root = commands
        .spawn((
            Transform::from_translation(world(state.ls.sim().crab().pos(), 0.0)),
            Visibility::default(),
            CrabAvatar,
        ))
        .id();
    // Body: a wide flat-ish box sitting on the ground.
    let body = commands
        .spawn((
            Mesh3d(meshes.add(Cuboid::new(crab_w * 1.6, crab_h * 0.5, crab_w))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.7, 0.18, 0.12),
                perceptual_roughness: 0.8,
                ..default()
            })),
            Transform::from_xyz(0.0, crab_h * 0.25, 0.0),
        ))
        .id();
    // A forward "claw" wedge at +Z (the crab's facing) so its orientation reads.
    let claw = commands
        .spawn((
            Mesh3d(meshes.add(Cuboid::new(crab_w * 0.3, crab_h * 0.25, crab_w * 0.9))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.85, 0.25, 0.15),
                ..default()
            })),
            Transform::from_xyz(0.0, crab_h * 0.3, crab_w * 1.0),
        ))
        .id();
    commands.entity(crab_root).add_children(&[body, claw]);
}

/// The three `&mut Transform` queries [`apply_transforms`] writes — player avatars,
/// the crab, the camera. Aliased (not inline) because Bevy needs the marker
/// `With`/`Without` filters to prove the three don't alias the same `Transform`, and
/// spelled inline that's the kind of type clippy's `type_complexity` flags. The
/// filters ARE the disjointness proof, so they can't be dropped — only named.
type AvatarXf<'w, 's> = Query<
    'w,
    's,
    (
        &'static PlayerAvatar,
        &'static mut Transform,
        &'static mut Visibility,
    ),
    Without<FpCamera>,
>;
type CrabXf<'w, 's> = Query<
    'w,
    's,
    &'static mut Transform,
    (With<CrabAvatar>, Without<PlayerAvatar>, Without<FpCamera>),
>;
type PlaneXf<'w, 's> = Query<
    'w,
    's,
    (
        &'static PlaneAvatar,
        &'static mut Transform,
        &'static mut Visibility,
    ),
    (
        Without<PlayerAvatar>,
        Without<CrabAvatar>,
        Without<FpCamera>,
    ),
>;
type CamXf<'w, 's> = Query<'w, 's, &'static mut Transform, With<FpCamera>>;

/// Place the FP camera and the dynamic avatars each frame, INTERPOLATED between the
/// previous tick's snapshot and the live sim by the fractional accumulator. This is
/// the smoothness layer: the sim jumps in 30 Hz steps, but every rendered frame
/// shows a pose `alpha` of the way from last tick to this one. Reads sim state
/// read-only; writes Bevy `Transform`s and (while the local player is alive) keeps the
/// client-side [`CameraYaw`] tracking the authoritative sim yaw — never the sim.
#[allow(clippy::too_many_arguments)]
fn apply_transforms(
    state: NonSend<GameState>,
    pitch: Res<CameraPitch>,
    mut yaw: ResMut<CameraYaw>,
    mut avatars: AvatarXf,
    mut crab_q: CrabXf,
    mut planes_q: PlaneXf,
    mut cam_q: CamXf,
) {
    let sim = state.ls.sim();
    let alpha = (state.accumulator / TICK_DT).clamp(0.0, 1.0) as f32;
    let local = state.ls.me();

    // Player avatars: lerp position and yaw from the previous snapshot to now.
    for (avatar, mut tf, mut vis) in avatars.iter_mut() {
        let Some(now) = sim.player(avatar.0) else {
            continue;
        };
        let prev = state.prev.players.get(&avatar.0).copied().unwrap_or(now);
        let pos = lerp_pos(prev.pos(), now.pos(), alpha);
        let yaw = lerp_yaw(prev.yaw(), now.yaw(), alpha);
        // Capsule center sits at half-height above the ground.
        *tf = Transform::from_translation(world(pos, PLAYER_HEIGHT * 0.5))
            .with_rotation(Quat::from_rotation_y(yaw));
        // Hide the local avatar (first-person), and any extracted player (gone safe).
        let hidden = avatar.0 == local || now.status() == PlayerStatus::Extracted;
        *vis = if hidden {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
        // A downed player falls onto its side so its status reads from the avatar.
        if now.status() == PlayerStatus::Downed {
            *tf = Transform::from_translation(world(pos, PLAYER_RADIUS)).with_rotation(
                Quat::from_rotation_y(yaw) * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            );
        }
    }

    // Crab: interpolate position + yaw.
    if let (Ok(mut tf), Some(crab_now), Some(crab_prev)) =
        (crab_q.single_mut(), Some(sim.crab()), state.prev.crab)
    {
        let pos = lerp_pos(crab_prev.pos(), crab_now.pos(), alpha);
        let yaw = lerp_yaw(crab_prev.yaw(), crab_now.yaw(), alpha);
        *tf =
            Transform::from_translation(world(pos, 0.0)).with_rotation(Quat::from_rotation_y(yaw));
    }

    // Planes: interpolate pose (3D position + heading + pitch) and orient the gray box
    // so +Z is the nose. Hide the local pilot's own plane (we fly from its cockpit).
    for (avatar, mut tf, mut vis) in planes_q.iter_mut() {
        let Some(now) = sim.plane(avatar.0) else {
            continue;
        };
        let prev = state.prev.planes.get(&avatar.0).copied().unwrap_or(now);
        *tf = plane_transform(prev, now, alpha);
        *vis = if avatar.0 == local {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }

    // FP camera. A PILOT flies from the cockpit: anchor the camera to the plane's
    // interpolated pose, looking along its heading+pitch (the authoritative sim
    // orientation), with the client pitch still added so the pilot can glance around.
    // An on-foot player keeps the ground eye view. Either way the orientation comes
    // from the sim, so what's seen matches where the sim says we point.
    if let Ok(mut cam) = cam_q.single_mut() {
        if let Some(plane_now) = sim.plane(local) {
            let plane_prev = state.prev.planes.get(&local).copied().unwrap_or(plane_now);
            let eye = lerp_pos3(plane_prev.pos(), plane_now.pos(), alpha);
            let heading = lerp_yaw(plane_prev.heading(), plane_now.heading(), alpha);
            // Pitch reuses lerp_yaw because it's a turn-unit angle too; since pitch is
            // bounded (never wraps), the shortest-arc handling is a harmless no-op here.
            let plane_pitch = lerp_yaw(plane_prev.pitch(), plane_now.pitch(), alpha);
            let look_dir = look_direction(heading, plane_pitch + pitch.0);
            *cam = Transform::from_translation(eye).looking_at(eye + look_dir, Vec3::Y);
        } else if let Some(now) = sim.player(local) {
            let prev = state.prev.players.get(&local).copied().unwrap_or(now);
            let pos = lerp_pos(prev.pos(), now.pos(), alpha);
            // Alive: aim by the AUTHORITATIVE sim yaw (so the view matches the avatar and
            // peers) and keep the free-look yaw tracking it. Downed/Extracted: the sim
            // freezes our yaw, so aim by the client-side CameraYaw instead — full
            // free-look (yaw+pitch) for a spectator, decoupled from the gated movement.
            let cam_yaw = if now.status() == PlayerStatus::Alive {
                let sim_yaw = lerp_yaw(prev.yaw(), now.yaw(), alpha);
                yaw.0 = sim_yaw;
                sim_yaw
            } else {
                yaw.0
            };
            let eye = world(pos, EYE_HEIGHT);
            let look_dir = look_direction(cam_yaw, pitch.0);
            *cam = Transform::from_translation(eye).looking_at(eye + look_dir, Vec3::Y);
        }
    }
}

/// The interpolated world transform for a plane: position lerped in 3D, orientation
/// from heading (about +Y) then pitch (nose up about the local right axis, +X). +Z is
/// the nose, matching the sim's heading-0 = +Z convention and the gray box's long axis.
/// Pitch is negated like the camera's: a positive sim pitch is nose-UP, but a positive
/// rotation about +X sends +Z toward −Y, so negate to make nose-up tilt the box up.
fn plane_transform(prev: Plane, now: Plane, alpha: f32) -> Transform {
    let pos = lerp_pos3(prev.pos(), now.pos(), alpha);
    let heading = lerp_yaw(prev.heading(), now.heading(), alpha);
    let pitch = lerp_yaw(prev.pitch(), now.pitch(), alpha);
    let rot = Quat::from_rotation_y(heading) * Quat::from_rotation_x(-pitch);
    Transform::from_translation(pos).with_rotation(rot)
}

/// Linear-interpolate two sim 3D positions (to meters) by `alpha` — the [`Pos3`]
/// analogue of [`lerp_pos`], including the altitude axis.
fn lerp_pos3(a: Pos3, b: Pos3, alpha: f32) -> Vec3 {
    Vec3::new(
        meters(a.x) + (meters(b.x) - meters(a.x)) * alpha,
        meters(a.y) + (meters(b.y) - meters(a.y)) * alpha,
        meters(a.z) + (meters(b.z) - meters(a.z)) * alpha,
    )
}

/// Linear-interpolate two sim positions (in meters) by `alpha`.
fn lerp_pos(a: Pos, b: Pos, alpha: f32) -> Pos {
    // Interpolate in fixed-point space, then `world()` converts to meters — keeps the
    // unit handling in one place. (a + (b-a)*alpha, rounded.)
    let lx = a.x as f32 + (b.x - a.x) as f32 * alpha;
    let lz = a.z as f32 + (b.z - a.z) as f32 * alpha;
    Pos {
        x: lx.round() as i64,
        z: lz.round() as i64,
    }
}

/// Interpolate two sim yaws (turn-unit integers) by `alpha`, taking the SHORTEST way
/// around the circle so a wrap from 359°→1° tweens through 0°, not backward through
/// the whole turn. Returns radians for the camera/avatar rotation.
fn lerp_yaw(a: i32, b: i32, alpha: f32) -> f32 {
    let ar = trig_client::turns_to_radians(a);
    let br = trig_client::turns_to_radians(b);
    let tau = std::f32::consts::TAU;
    let mut diff = br - ar;
    if diff > tau / 2.0 {
        diff -= tau;
    } else if diff < -tau / 2.0 {
        diff += tau;
    }
    ar + diff * alpha
}

/// The camera's look direction from a ground yaw and a client pitch. Compose yaw
/// (about +Y) with pitch (about the camera's local right axis, +X) and apply to the
/// base forward +Z: pitch tilts forward up/down in the YZ plane, then yaw swings it
/// horizontally. Pitch is negated because a positive rotation about +X sends +Z
/// toward −Y (down), and the control convention is positive-pitch = look UP.
fn look_direction(yaw_radians: f32, pitch_radians: f32) -> Vec3 {
    let rot = Quat::from_rotation_y(yaw_radians) * Quat::from_rotation_x(-pitch_radians);
    (rot * Vec3::Z).normalize()
}

/// Spawn the windowed first-person camera. Its transform is overwritten every frame
/// by [`apply_transforms`]; the sky-blue clear color frames the gray-box.
fn spawn_fp_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.5, 0.7, 0.92)),
            ..default()
        },
        Transform::default(),
        FpCamera,
    ));
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

fn spawn_hud(mut commands: Commands) {
    commands.spawn((
        Text::new("…"),
        TextFont {
            font_size: 22.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(14.0),
            left: Val::Px(14.0),
            ..default()
        },
        StatusHud,
    ));
}

/// Update the HUD: the local player's status, and the round outcome once decided.
fn update_hud(state: NonSend<GameState>, mut hud: Query<&mut Text, With<StatusHud>>) {
    let Ok(mut text) = hud.single_mut() else {
        return;
    };
    let sim = state.ls.sim();
    let status = sim
        .player(state.ls.me())
        .map(|p| match p.status() {
            PlayerStatus::Alive => "ALIVE",
            PlayerStatus::Downed => "DOWNED",
            PlayerStatus::Extracted => "EXTRACTED",
        })
        .unwrap_or("—");
    let outcome = match sim.outcome() {
        Outcome::Ongoing => String::new(),
        Outcome::Extracted => "\nROUND WON — extracted!".to_string(),
        Outcome::Wiped => "\nROUND LOST — wiped".to_string(),
    };
    **text = format!(
        "You: {status}   |   reach the green pillar, hold [Space]/(A) to extract — dodge the crab\n[R] restart   [Esc] quit{outcome}",
    );
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
        }
    }

    /// Pan/tilt the screenshot camera by these degrees, applied at the local player's
    /// eye AFTER the first-person aim — so a single evidence frame can frame the giant
    /// crab, the extraction pillar, and the other players together when the towering
    /// crab would otherwise fill the dead-ahead view. Still a first-person shot (same
    /// eye, same sim yaw as the base); only the composition pans, the play.rs
    /// `RL_CAM_*` convention. Zero = straight first-person.
    pub fn with_cam_offset(mut self, yaw_deg: f32, pitch_deg: f32) -> Self {
        self.cam_yaw_deg = yaw_deg;
        self.cam_pitch_deg = pitch_deg;
        self
    }
}

#[derive(Resource)]
struct ShotTarget(Handle<Image>);

#[derive(Resource, Default)]
struct ShotProgress {
    frames: u32,
    captured: bool,
    exit_countdown: i32,
}

/// The offscreen camera for the screenshot path. Its transform is driven by
/// [`apply_transforms`] (it carries the [`FpCamera`] marker), so the captured frame
/// is the genuine first-person view, not a separate angle.
fn spawn_offscreen_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cfg: Res<ScreenshotConfig>,
) {
    let mut image =
        Image::new_target_texture(cfg.width, cfg.height, TextureFormat::bevy_default(), None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let handle = images.add(image);
    commands.spawn((
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
    ));
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

/// After the sim has run a few ticks and the GPU pipeline has warmed (settle counted
/// in RENDER frames — early frames render black), capture one PNG of the FP view and
/// exit. Same shape as play.rs's capture: spawn a `Screenshot` observed by
/// `save_to_disk`, then a short countdown for the readback/encode to finish.
fn capture_when_settled(
    mut commands: Commands,
    cfg: Res<ScreenshotConfig>,
    target: Res<ShotTarget>,
    mut progress: ResMut<ShotProgress>,
    mut exit: MessageWriter<AppExit>,
) {
    if progress.captured {
        progress.exit_countdown -= 1;
        if progress.exit_countdown <= 0 {
            exit.write(AppExit::Success);
        }
        return;
    }
    progress.frames += 1;
    if progress.frames < cfg.settle {
        return;
    }
    commands
        .spawn(Screenshot::image(target.0.clone()))
        .observe(save_to_disk(cfg.path.clone()));
    info!(
        "fp screenshot: captured at render frame {}, writing {}",
        progress.frames,
        cfg.path.display()
    );
    progress.captured = true;
    progress.exit_countdown = 30;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame conversion must match the sim's documented right-handed XZ layout:
    /// +X right, +Z forward, Y up. A sim Pos maps straight through to Bevy XYZ with
    /// the given height — no axis swap or sign flip.
    #[test]
    fn world_maps_sim_frame_directly() {
        let p = Pos {
            x: 2 * UNIT,
            z: 5 * UNIT,
        };
        let v = world(p, 1.6);
        assert_eq!(v, Vec3::new(2.0, 1.6, 5.0));
    }

    /// The camera's flat (zero-pitch) facing must match the sim's yaw convention:
    /// yaw 0 looks +Z, a quarter turn looks +X — so what the player sees agrees with
    /// where the sim says it faces.
    #[test]
    fn camera_facing_matches_sim_yaw_convention() {
        let f0 = look_direction(0.0, 0.0);
        assert!(
            (f0 - Vec3::Z).length() < 1e-5,
            "yaw 0 should look +Z, got {f0:?}"
        );
        let fq = look_direction(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(
            (fq - Vec3::X).length() < 1e-5,
            "quarter turn should look +X, got {fq:?}"
        );
    }

    /// Look direction at zero pitch is the flat facing; pitching up tilts +Y without
    /// changing the horizontal heading sign.
    #[test]
    fn look_direction_pitches_without_flipping_heading() {
        let flat = look_direction(0.0, 0.0);
        assert!((flat - Vec3::Z).length() < 1e-5);
        let up = look_direction(0.0, 0.5);
        assert!(up.y > 0.0, "positive pitch looks up, got {up:?}");
        assert!(up.z > 0.0, "still facing +Z, got {up:?}");
    }

    /// Yaw interpolation takes the short way around the wrap: from just-below-a-full-
    /// turn to just-above-zero tweens FORWARD through 0, not backward through ~2π.
    #[test]
    fn yaw_lerp_takes_short_path_across_wrap() {
        // a ≈ 350°, b ≈ 10° (in turn units). Halfway should land near 0° (=360°),
        // i.e. the short 20° arc, not 180°.
        let a = trig::TURN - trig::TURN / 36; // ~350°
        let b = trig::TURN / 36; // ~10°
        let mid = lerp_yaw(a, b, 0.5);
        // Normalize to [-π, π] around 0.
        let mut n = mid % std::f32::consts::TAU;
        if n > std::f32::consts::PI {
            n -= std::f32::consts::TAU;
        }
        assert!(
            n.abs() < 0.2,
            "midpoint should be ~0 rad (short path), got {n}"
        );
    }

    /// Position interpolation is the plain linear midpoint in fixed-point space.
    #[test]
    fn pos_lerp_midpoint() {
        let a = Pos { x: 0, z: 0 };
        let b = Pos { x: 1000, z: -400 };
        let mid = lerp_pos(a, b, 0.5);
        assert_eq!(mid, Pos { x: 500, z: -200 });
    }

    /// A full-deflection look this tick must map to EXACTLY the sim's per-tick yaw cap
    /// — no more (the sim would clamp it and the camera would lag the avatar), no less
    /// (the player couldn't turn as fast as the sim allows). This pins the client's
    /// `look_yaw` normalization to the sim's `MAX_YAW_TURNS_PER_TICK`, the coupling
    /// that keeps the FP camera and the authoritative yaw in agreement.
    #[test]
    fn full_look_axis_turns_one_tick_cap() {
        // Drive a fresh sim one tick with look_yaw at full deflection; the yaw delta
        // must equal the sim's documented per-tick cap (TURN/24).
        let mut sim = Sim::new(0, &[PlayerId(0)]);
        let before = sim.player(PlayerId(0)).unwrap().yaw();
        // The client builds this exact input for a +MAX_YAW_PER_TICK_RADIANS look:
        // yaw_delta / MAX_YAW_PER_TICK_RADIANS, saturating the axis at full deflection.
        let look_axis = (MAX_YAW_PER_TICK_RADIANS / MAX_YAW_PER_TICK_RADIANS).clamp(-1.0, 1.0);
        assert_eq!(look_axis, 1.0, "a full-deflection look saturates the axis");
        let input = Input::new(0.0, 0.0, look_axis, 0);
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), input);
        sim.step(&inputs);
        let after = sim.player(PlayerId(0)).unwrap().yaw();
        let cap = trig::TURN / 24;
        assert_eq!(
            trig::wrap_turns(after - before),
            cap,
            "full look axis should turn exactly the sim's per-tick cap"
        );
    }

    /// WASD-shaped move + the action button map to the expected fixed-point [`Input`]:
    /// forward+right at full deflection quantize to +AXIS_SCALE, and pressing action
    /// sets the ACTION bit. (Mirrors how `gather_input`/`drive_lockstep` build the
    /// per-tick input from the accumulated controls.)
    #[test]
    fn move_and_action_map_to_input() {
        let i = Input::new(1.0, 1.0, 0.0, buttons::ACTION);
        assert_eq!(i.move_strafe, Input::AXIS_SCALE, "full right → +AXIS_SCALE");
        assert_eq!(
            i.move_forward,
            Input::AXIS_SCALE,
            "full forward → +AXIS_SCALE"
        );
        assert!(i.pressed(buttons::ACTION), "action bit set when pressed");
        let n = Input::new(0.0, 0.0, 0.0, 0);
        assert!(!n.pressed(buttons::ACTION), "no action bit when unpressed");
    }

    /// Pins the geometric fact that `gather_input`'s X-axis negation corrects: a camera
    /// facing +Z (yaw 0) has its RIGHT axis at world −X, so the sim's "+X = strafe
    /// right" renders on the SCREEN-LEFT. This is why the control layer negates strafe
    /// and yaw-look — keeping the proof in a test so a future camera change can't
    /// silently re-invert the controls.
    #[test]
    fn camera_right_is_negative_x_facing_plus_z() {
        let eye = Vec3::new(0.0, EYE_HEIGHT, 0.0);
        let cam =
            Transform::from_translation(eye).looking_at(eye + look_direction(0.0, 0.0), Vec3::Y);
        let right = cam.right().as_vec3();
        assert!(
            (right - Vec3::NEG_X).length() < 1e-5,
            "facing +Z, camera-right must be world −X (so sim +X is screen-left); got {right:?}"
        );
    }
}
