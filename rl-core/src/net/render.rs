//! First-person Bevy client for the deterministic gray-box (rl#38 render sub).
//!
//! This is the windowed `play` mode of the `game` binary: it makes the
//! giant-crab-rescue sim VISIBLE and PLAYABLE on top of the existing lockstep +
//! transport netcode. It boots to a client-side Host / Join menu (rl#58,
//! [`AppPhase`]/[`menu_ui`]) and builds the round only once the player chooses — the
//! menu is gated to its own pre-round phases and never touches the sim. The split it
//! honors is the one documented at the top of
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
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow, WindowMode};

use crate::screenshot::{self, ShotProgress, ShotTarget};

use crate::controls::{
    ActiveDevice, ForceRevealControls, PAD_STICK_DEADZONE, spawn_controls_ui, track_active_device,
    update_controls_ui,
};
use crate::net::cadence::PhysicsCadence;
use crate::net::controls::{self, Action, GcrControls};
use crate::net::lockstep::{Lockstep, TickMsg};
use crate::net::net_loop::{NetDriver, PeerMsg};
use crate::net::sim::{
    CRAB_SCALE, Crab, EXTRACT_RADIUS, Input, Outcome, Plane, Player, PlayerId, PlayerStatus, Pos,
    Pos3, Sim, UNIT, buttons, trig, trig_client,
};
use crate::net::telemetry::{TELEMETRY_TICK_EVERY, TelemetryEvent};

/// Sim tick rate (Hz). Re-exported from [`crate::net::sim::TICK_HZ`] (the one source)
/// so this windowed client and the headless driver advance at the same rate and stay
/// in lockstep; the client renders faster and interpolates between ticks.
pub use crate::net::sim::TICK_HZ;

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

/// How long Select/Back must be HELD to quit (seconds). A hold, not a tap, so a stray
/// press can't end the round for everyone on the couch — the kid-safe equivalent of
/// Esc. Client-local (sends AppExit, never touches the sim), so it can't desync a peer.
const PAD_QUIT_HOLD_SECS: f32 = 1.0;

/// One gamepad's contribution to this frame's control deltas: move axes from the left
/// stick, look deltas from the right. Raw f32 — like the keyboard/mouse contributions,
/// it crosses into the sim only after [`Input::new`] quantizes it (see [`gather_input`]).
struct PadAxes {
    strafe: f32,
    forward: f32,
    /// Yaw-look this frame (radians), already scaled by [`PAD_LOOK_SPEED`] and dt.
    d_yaw: f32,
    /// Pitch-look this frame (radians); client-only camera, never reaches the sim.
    d_pitch: f32,
}

/// Map a gamepad's two sticks to this frame's move + look deltas. The pure analog core
/// of the pad branch, factored out of [`gather_input`] so the determinism test can drive
/// the SAME arithmetic the client runs (no copy to drift) — the sub-deadzone-clears-to-0,
/// the look = stick·speed·dt scaling. Buttons aren't here: they're plain bool reads with
/// no quantization concern, so they stay inline at the call site. Frame-local and f32;
/// the result is quantized downstream by [`Input::new`], so it never enters the sim raw.
fn pad_stick_axes(left_stick: Vec2, right_stick: Vec2, dt: f32) -> PadAxes {
    // Deadzone on each stick's MAGNITUDE (not per-axis), so a resting stick's hardware
    // noise reads as exactly zero rather than creeping the avatar/view.
    let (mut strafe, mut forward) = (0.0, 0.0);
    if left_stick.length() > PAD_STICK_DEADZONE {
        strafe = left_stick.x;
        forward = left_stick.y;
    }
    let (mut d_yaw, mut d_pitch) = (0.0, 0.0);
    if right_stick.length() > PAD_STICK_DEADZONE {
        d_yaw = right_stick.x * PAD_LOOK_SPEED * dt;
        d_pitch = right_stick.y * PAD_LOOK_SPEED * dt;
    }
    PadAxes {
        strafe,
        forward,
        d_yaw,
        d_pitch,
    }
}

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

/// How the windowed client starts up: at the boot MENU (the interactive default — the
/// player picks Host/Join, rl#58), or straight into a prebuilt ROUND (the scripted
/// `--host`/`--join` flags, which form the match up front so tests/scripts never depend on
/// clicking the menu). One enum, two boots, so "has a menu AND a prebuilt round" is
/// unrepresentable rather than two bool flags.
pub enum Boot {
    /// Show the boot menu first; the sim is built only once the player chooses and the
    /// host-triggered lobby resolves. `seed` is the shared match seed and `telemetry` the
    /// optional collector id — both threaded to whichever formation the menu kicks off.
    Menu {
        seed: u64,
        telemetry: Option<crate::net::menu::EndpointId>,
    },
    /// Skip the menu and play this already-formed round immediately. The scripted entry
    /// (`--host`/`--join` = the formed lockstep + its driver; a host-alone `--host` that
    /// found no peer = a solo lockstep + `None`). Boxed because the lockstep + driver are
    /// large and `Menu` is tiny — without the box every `Menu` would carry that dead weight
    /// (the same reason [`crate::net::net_loop::MatchResult::Joined`] boxes).
    Round(Box<(Lockstep, Option<NetDriver>)>),
}

/// The windowed client's top-level phase (rl#56). The menu and lobby screens are PURE
/// client UI — no [`Lockstep`]/[`Sim`] exists until [`AppPhase::Playing`], which is entered
/// only after a choice (and, for networked roles, a host-commanded start). This is the
/// firewall that keeps the menu off the deterministic sim: the FP systems and the sim
/// resource are all gated to `Playing`, so menu state literally cannot reach the round
/// (it's built fresh on the transition from the unchanged formation machinery).
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppPhase {
    /// The boot menu: choose Host / Join (rl#58). egui only.
    #[default]
    Menu,
    /// A host-triggered lobby is forming on a background thread; show the live roster +
    /// (Host) join code + Start, and poll for the result. A Host-alone Start skips straight
    /// to its instant solo round without lingering here.
    Connecting,
    /// The round is live: the FP client runs exactly as before rl#56.
    Playing,
}

/// Build the windowed first-person client app. Starts at the boot menu or straight in a
/// round per [`Boot`]; owns the `Lockstep` + optional `NetDriver` as resources once
/// playing. Built here, not via a `Plugin` that holds the sim, because
/// `Plugin::build(&self)` can't move a non-`Clone` `Lockstep`/`NetDriver` out of
/// itself — inserting them as resources at the `Playing` transition is the clean path.
///
/// `external_crab` points at a trained checkpoint dir; when the round arms the crab the giant
/// crab is the REAL rapier-simulated NN body ([`crate::net::external_crab`]) instead of the
/// integer point-pursuer — on a SOLO round always, and on a NETWORKED round once peers agree on
/// a shared brain (the weights-digest handshake) and step it at the deterministic cadence (the
/// GCR fold, rl#82). A networked-UNSYNCED round keeps the integer pursuer.
/// Pin every process-global task pool to a single worker, so the armed float rapier-NN crab
/// ("real Sally") evolves bit-identically across peers in the windowed client (GCR#113).
///
/// MUST be called at process start — ahead of matchmaking and `solo_lockstep_for` (which build
/// sims) and `App::new` — because the pools latch their thread count on first use; see
/// [`crate::bot::test_util::pin_single_thread_pools`] for the env-var/`OnceLock` mechanism. Same
/// recipe the trainer and the #82 cross-peer probe run; exposed as one thin `pub` entry so `game`
/// (a separate crate) can call it without widening the crate-internal helper. The matching ECS
/// run-order pin is applied inside [`build_windowed_app`] via `force_serial_schedules`; that one
/// is `build`-internal because nothing outside needs it.
pub fn pin_process_pools() {
    crate::bot::test_util::pin_single_thread_pools();
}

pub fn build_windowed_app(boot: Boot, external_crab: Option<std::path::PathBuf>) -> App {
    // Loud canary for the one footgun the two-phase pin leaves: a caller that forgot
    // `pin_process_pools()` at process start would silently lose cross-peer determinism (fine
    // solo, desyncs only between peers). RAYON_NUM_THREADS is set to "1" only by the pin (and
    // an external override to anything else equally breaks determinism), so it's the canary.
    debug_assert_eq!(
        std::env::var("RAYON_NUM_THREADS").ok().as_deref(),
        Some("1"),
        "pin_process_pools() must run before build_windowed_app (and no external RAYON_NUM_THREADS override)"
    );
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Giant Crab Rescue".into(),
            mode: WindowMode::Windowed,
            ..default()
        }),
        ..default()
    }));
    app.init_state::<AppPhase>();

    // The FP round systems, gated to Playing. spawn_* moved off Startup to the Playing
    // transition (the sim doesn't exist until then); the per-frame systems run only while
    // playing so they never touch a not-yet-built GameState. The set is IDENTICAL to the
    // pre-rl#56 wiring — only the schedule gating is new, so the round itself is unchanged.
    //
    // `ensure_round_installed` is CHAINED ahead of the spawns: on the menu path it moves
    // the chosen round into GameState here (the sim must exist before spawn_world reads
    // it); on the scripted Boot::Round path GameState already exists, so it no-ops. The
    // chain is what guarantees the sim is live before the scene spawns — separate
    // OnEnter system sets have no ordering, which would race spawn_world ahead of the install.
    app.init_non_send_resource::<PendingRound>()
        .init_resource::<ActiveDevice>()
        // The windowed client never forces the overlay open — it's hold-to-reveal. Inserting
        // the resource here (false) keeps `update_controls_ui` reading a plain `Res`, not an
        // `Option<Res>`; only the screenshot path sets it true.
        .insert_resource(ForceRevealControls(false))
        .add_systems(
            OnEnter(AppPhase::Playing),
            (
                ensure_round_installed,
                spawn_world,
                spawn_fp_camera,
                spawn_hud,
                spawn_controls_ui::<GcrControls>,
                crate::build_info::spawn_build_info_overlay,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (gather_input, drive_lockstep, apply_transforms, update_hud)
                .chain()
                .run_if(in_state(AppPhase::Playing)),
        )
        .add_systems(
            Update,
            (
                grab_cursor_once,
                quit_game,
                // chained so the glyph swap reflects THIS frame's device, not last frame's.
                (track_active_device, update_controls_ui::<GcrControls>).chain(),
            )
                .run_if(in_state(AppPhase::Playing)),
        );

    // OUR policy-weights digest (rl#82, GCR), advertised in networked formation so peers can
    // agree on a shared brain (see [`crate::net::may_arm_external_crab`]). `0` for no checkpoint.
    // MUST equal the digest the per-tick bridge folds into the lockstep hash — both come from
    // [`crate::play::checkpoint_digest`] (here from the path; on the bridge via
    // `Policy::weights_digest()`), so the cadence-fold follow-up that arms the crab must source
    // its folded digest from the SAME checkpoint, or a hot-reload could split the two.
    let weights_digest = external_crab
        .as_deref()
        .map(crate::play::checkpoint_digest)
        .unwrap_or(0);
    // OUR crab-MODEL-asset digest (rl#100, GCR): the giant crab's rapier colliders are derived
    // from this asset ([`crate::bot::meshfit::crab_asset_digest`]), so peers must agree on it
    // before arming the float crab in lockstep — a different model builds different colliders
    // and desyncs even with identical brains. Computed unconditionally (it's a property of this
    // peer's installed crab model, independent of whether a checkpoint loaded); `0` for no model.
    let asset_digest = crate::bot::meshfit::crab_asset_digest();

    match boot {
        // Scripted boot: insert the round now and jump straight to Playing (the menu
        // states are never entered). NextState applied before the first frame, so
        // OnEnter(Playing) fires and the world spawns on frame one — no menu flash. The
        // scripted `--host`/`--join` path tests/scripts use; a host-alone `--host` that
        // found no peer is a solo round here, so it gets the real NN crab.
        Boot::Round(round) => {
            let (mut ls, net) = *round;
            let networked = net.is_some();
            // External NN crab: arm iff the shared GCR gate allows it — a SOLO round always,
            // a NETWORKED round only with synced weights ([`crate::net::may_arm_external_crab`],
            // the determinism guard; the `Some(dir)` arm already proves a checkpoint is present).
            // Capture the integer crab's spawn + hand the crab to external control BEFORE `ls`
            // moves into core.
            let nn = match external_crab {
                Some(dir)
                    if crate::net::may_arm_external_crab(
                        net.is_none(),
                        net.as_ref().is_some_and(NetDriver::weights_synced),
                        net.as_ref().is_some_and(NetDriver::assets_synced),
                    ) =>
                {
                    let crab = ls.sim().crab();
                    let spawn = crab.pos();
                    // Arm + seed the pose atomically with the crab's CURRENT spawn pose/yaw —
                    // writing back what's already there, so sim state is unchanged. Digest 0 to
                    // seed; the bridge's first post-step `hash_crab_physics` fills it before the
                    // first `sync_external_crab` push, so the seeded value is never cross-checked.
                    ls.initialize_external_crab(spawn, crab.yaw(), 0);
                    Some((dir, spawn))
                }
                _ => None,
            };
            let source = match net {
                Some(n) => InputSource::Networked(Box::new(n)),
                None => InputSource::Solo,
            };
            insert_core(&mut app, ls, source);
            if let Some((dir, spawn)) = nn {
                // Known-armed at build: add the stack AND arm the gate now, so the crab spawns
                // frame one.
                add_external_nn_crab(&mut app, dir, spawn);
                app.insert_resource(crate::net::external_crab::ExternalCrabArmed);
                // Networked: pin the feel knobs to defaults so a per-peer env override can't
                // walk the crab to a different (hashed) pose and desync — solo keeps its tuning.
                if networked {
                    app.world_mut()
                        .resource_mut::<crate::net::external_crab::ExternalCrabBridge>()
                        .pin_default_knobs();
                }
            }
            app.world_mut()
                .resource_mut::<NextState<AppPhase>>()
                .set(AppPhase::Playing);
        }
        // Interactive boot: add the menu plugin (egui menu + lobby poll). The sim is built
        // later, at the Playing transition, from the choice the menu records.
        Boot::Menu { seed, telemetry } => {
            app.add_plugins(menu_ui::MenuPlugin {
                seed,
                telemetry,
                weights_digest,
                asset_digest,
            });
            // NN crab on the round (rl#58 + GCR): the menu can't know at BUILD time whether the
            // round will be solo, networked-synced, or networked-unsynced, so add the whole NN
            // stack now with the gate OFF and no crab spawned (NumEnvs 0), and arm it only if
            // the resolved round may ([`ensure_round_installed`] → [`crate::net::may_arm_external_crab`]:
            // solo always, networked only with synced weights). The crab's arena spawn is a pure
            // function of the seed (a throwaway solo lockstep reads it), so it's known here
            // without the round existing yet. A missing/empty checkpoint keeps the integer crab
            // (the stack isn't added at all); a networked-UNSYNCED round leaves the gate off, so
            // it stays byte-identical to the integer-crab multiplayer round.
            if let Some(dir) = external_crab {
                let crab_spawn = crate::net::net_loop::solo_lockstep_for(seed)
                    .sim()
                    .crab()
                    .pos();
                add_external_nn_crab(&mut app, dir, crab_spawn);
                // Gate OFF: leave `ExternalCrabArmed` ABSENT (presence is the state). The
                // transition (`ensure_round_installed`) inserts it iff the round resolves solo.
                app.insert_resource(crate::bot::NumEnvs(0)); // no crab spawns behind the menu
                app.insert_resource(ExternalCrabStackInstalled(true)); // the transition may activate it
            }
        }
    }

    // Pin ECS run order: force every MAIN-world schedule onto the single-threaded executor so
    // systems never dispatch onto the global ComputeTaskPool, where thread scheduling would
    // reorder the float evolution that drives the armed NN crab and desync peers (GCR#113,
    // the same pin the trainer and #82 probe apply). Must run AFTER all systems are wired —
    // every plugin/`add_systems` above is in, the schedules now exist. This touches only the
    // main world's schedules (the sim); bevy's render sub-app keeps its own executor, so
    // rendering isn't forced serial here — only the global pool pin (`pin_windowed_pools`,
    // called at process start) caps render parallelism, the unavoidable cost of bevy sharing
    // three global pools across sim and render. Correctness > a few dropped frames.
    crate::bot::test_util::force_serial_schedules(&mut app);

    app
}

/// Whether the boot-menu app has the solo NN-crab stack installed at build (rl#58) — set when
/// a checkpoint was supplied on the menu path. The Playing transition reads it to decide
/// whether to ACTIVATE the NN crab (only when the round resolved solo). Absent/false ⇒ the
/// integer crab (no checkpoint, or the networked path).
#[derive(Resource, Default, Clone, Copy)]
struct ExternalCrabStackInstalled(bool);

/// Wire the real rapier-NN crab into the windowed solo app: the bot/physics/brain stack
/// (the SAME plugins `rl --demo` runs, so the crab steps the exact dynamics the policy
/// trained under) plus the [`external_crab::ExternalCrabPlugin`] bridge that walks it toward the
/// player and feeds its body position back into the sim. With no `sally.glb` the crab is
/// the rl#5 procedural fallback rig under a Rapier debug wireframe; with the model
/// present the cosmetic skin rides the same body.
fn add_external_nn_crab(app: &mut App, checkpoint_dir: std::path::PathBuf, crab_spawn: Pos) {
    use bevy_rapier3d::prelude::{DebugRenderMode, RapierDebugRenderPlugin};

    app.insert_resource(crate::Visuals(true))
        .insert_resource(crate::bot::NumEnvs(1))
        // Same fixed timestep + softened contact spring as training/demo, with Rapier in
        // FixedUpdate (lockstep with the Sense→Think→Act brain loop) — bundled in the one
        // order that applies the spring, so the solo crab's physics can't drift from what
        // the policy optimised under (see physics::CrabPhysicsPlugin).
        .add_plugins(crate::physics::CrabPhysicsPlugin)
        .add_plugins(crate::physics::PhysicsWorldPlugin)
        .add_plugins(crate::bot::BotPlugin)
        .add_plugins(crate::net::external_crab::ExternalCrabPlugin {
            checkpoint_dir,
            crab_spawn,
        });

    // Park the wall-clock FixedUpdate auto-pump; `drive_lockstep` pumps the body at the
    // deterministic [`PhysicsCadence`] instead (see [`park_fixed_auto_pump`]).
    park_fixed_auto_pump(app.world_mut());

    // The crab's true colliders as a wireframe — the in-engine view of the NN body when
    // no skin is loaded (and a useful overlay when one is). On by default for the solo
    // showcase so the body is always visible; the integer-crab placeholder box is hidden
    // in `apply_transforms` when the bridge is present.
    app.add_plugins(RapierDebugRenderPlugin {
        enabled: true,
        mode: DebugRenderMode::COLLIDER_SHAPES,
        ..default()
    });
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
    // Controls UI on the screenshot path too, so an evidence frame can prove the overlay +
    // hint draw — the shared env override forces it open headless (see
    // [`crate::controls::reveal_overrides_from_env`]).
    let (force_reveal, active_device) = crate::controls::reveal_overrides_from_env();
    app.insert_resource(cfg)
        .init_resource::<ShotProgress>()
        .insert_resource(force_reveal)
        .insert_resource(active_device)
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
    app
}

/// Shared setup for both apps: the sim + its input source, plus the input resources.
fn insert_core(app: &mut App, ls: Lockstep, input_source: InputSource) {
    install_round(app.world_mut(), ls, input_source);
}

/// Install the round resources into the world: the non-send [`GameState`] (sim + input
/// source) and the input resources. Factored out of [`insert_core`] so it can be called
/// BOTH at app build (the scripted/screenshot path) and from the menu's
/// `OnEnter(Playing)` transition system (rl#56), which only has a [`World`], not an
/// [`App`] — one definition so the round is set up identically however it was reached.
fn install_round(world: &mut World, ls: Lockstep, input_source: InputSource) {
    let prev = SimSnapshot::capture(ls.sim());
    world.insert_non_send_resource(GameState {
        ls,
        input_source,
        accumulator: 0.0,
        prev,
    });
    world.init_resource::<PendingInput>();
    world.init_resource::<CameraPitch>();
    world.init_resource::<CameraYaw>();
    world.init_resource::<LocalVehicle>();
}

/// The round the boot menu chose, parked here between the menu's Playing transition and
/// the `OnEnter(Playing)` install. Non-send because a [`menu::ReadyMatch`] holds a
/// `NetDriver` (tokio runtime). `None` on the scripted `Boot::Round` path, which installs
/// `GameState` at app build instead — so [`ensure_round_installed`] no-ops there.
#[derive(Default)]
struct PendingRound(Option<crate::net::menu::ReadyMatch>);

/// At the Playing transition, make sure a [`GameState`] exists before the scene spawns.
/// Chained ahead of `spawn_world` (which reads the sim). Two cases, one place:
/// - **Scripted (`Boot::Round`)**: `GameState` was inserted at app build — nothing to do.
/// - **Menu**: take the parked [`PendingRound`] (set by the menu on its choice) and
///   [`install_round`] it now, so the sim is live for the spawns.
///
/// On the menu path this is ALSO where the external NN crab is ARMED (rl#58 + GCR): if the NN
/// stack was installed at build ([`ExternalCrabStackInstalled`]) AND the resolved round may arm it
/// ([`crate::net::may_arm_external_crab`]: solo always, networked only with synced weights),
/// insert [`crate::net::external_crab::ExternalCrabArmed`] and hand the sim crab to external control
/// so the rapier-NN body drives it. A networked-UNSYNCED round (or no checkpoint) leaves the
/// gate off → the integer crab, byte-identical to today's multiplayer. The crab's arena spawn
/// was already seeded into the bridge at build (a pure function of the seed), so nothing about
/// the spawn depends on the round here.
///
/// Idempotent (guards on `GameState` already present), so it can't double-install if both
/// a scripted round and a stray pending one ever coexisted. Reaching Playing with neither a
/// pre-installed `GameState` (scripted) nor a parked round (menu) is an unreachable
/// logic-bug state — every menu path parks a round BEFORE requesting the transition — so we
/// panic HERE with a precise message rather than no-op and let the chained `spawn_world`
/// (which needs `GameState`) panic one system later with a cryptic missing-resource error.
fn ensure_round_installed(world: &mut World) {
    if world.get_non_send_resource::<GameState>().is_some() {
        return; // scripted path already installed the round at build time
    }
    let mut ready = world
        .get_non_send_resource_mut::<PendingRound>()
        .and_then(|mut p| p.0.take())
        .expect("entered Playing with no round to install — the menu must park a round before transitioning");
    // Arm the external NN crab iff the stack was installed at build (a checkpoint was present)
    // AND the resolved round may arm it (the shared GCR gate: solo always, networked only with
    // synced weights — a float crab on an unsynced networked round would desync peers).
    let has_nn_stack = world
        .get_resource::<ExternalCrabStackInstalled>()
        .is_some_and(|m| m.0);
    let networked = ready.net.is_some();
    let weights_synced = ready.net.as_ref().is_some_and(NetDriver::weights_synced);
    let assets_synced = ready.net.as_ref().is_some_and(NetDriver::assets_synced);
    if has_nn_stack
        && crate::net::may_arm_external_crab(ready.net.is_none(), weights_synced, assets_synced)
    {
        let crab = ready.lockstep.sim().crab();
        // Arm + seed atomically with the crab's current pose/yaw (writing back what's there →
        // no state change), removing the set-pose-before-arm footgun.
        ready
            .lockstep
            .initialize_external_crab(crab.pos(), crab.yaw(), 0);
        world.insert_resource(crate::net::external_crab::ExternalCrabArmed);
        // Networked: pin the feel knobs to defaults so a per-peer env override can't walk the
        // crab to a different (hashed) pose and desync — solo keeps its tuning.
        if networked {
            world
                .resource_mut::<crate::net::external_crab::ExternalCrabBridge>()
                .pin_default_knobs();
        }
    }
    let source = match ready.net {
        Some(n) => InputSource::Networked(Box::new(n)),
        None => InputSource::Solo,
    };
    install_round(world, ready.lockstep, source);
}

// ---------------------------------------------------------------------------
// Lockstep driver state (render-agnostic: owns the sim + transport)
// ---------------------------------------------------------------------------

/// Where the OTHER players' per-tick inputs come from this run. One field, three
/// mutually-exclusive cases — so "broadcast to real peers AND fabricate bot inputs"
/// (a meaningless combination) is unrepresentable rather than merely unreached.
enum InputSource {
    /// Real peers over the transport (windowed networked play): broadcast our tick
    /// message and ingest theirs. Boxed because [`NetDriver`] owns a tokio runtime +
    /// the iroh session (~200 bytes), dwarfing the other variants — without the box the
    /// whole enum (one per round) carries that weight even when solo.
    Networked(Box<NetDriver>),
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
    /// Latches if the enter/exit-vehicle key (E) was tapped this interval — a client-local
    /// toggle drained ONCE per frame in [`drive_lockstep`] (single-player only), never sent
    /// to the sim. Board a plane on foot / step out when piloting.
    toggle_vehicle: bool,
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

/// The local player's SINGLE-PLAYER vehicle, when piloting one. `None` = on foot.
///
/// Single-player vehicle flight lives ENTIRELY here in the play layer, NOT in the
/// deterministic sim ([`crate::net::sim`] stays integer-only and untouched): a solo round
/// needs no cross-peer lockstep, so the plane is a client-side body the client integrates
/// itself with the shared [`Plane::step`] flight formula. While piloting, the local foot
/// player feeds the sim a NEUTRAL input (it just stands at the boarding spot) and the camera
/// flies from this plane; stepping out drops it and returns the view to the foot player.
///
/// An enum (not two `Option`s) so the plane and its previous pose are present together or not
/// at all — the "flying but no prev pose" state is unrepresentable. `prev` is last applied
/// tick's pose, so [`apply_transforms`] tweens the cockpit camera the same way it interpolates
/// every sim body. Only ever `Piloting` on the windowed [`InputSource::Solo`] path.
#[derive(Resource, Default)]
enum LocalVehicle {
    #[default]
    OnFoot,
    Piloting {
        plane: Plane,
        prev: Plane,
    },
}

impl LocalVehicle {
    fn piloting(&self) -> bool {
        matches!(self, Self::Piloting { .. })
    }
}

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

/// Log + count each lockstep fault and forward it to telemetry — the shared reporting both
/// fault sites in [`drive_lockstep`] (`record_remote` and `advance_one`) use, so they can't
/// drift. A desync can't be recovered; we surface it (log + telemetry) and play on.
fn report_faults(
    faults: &[crate::net::lockstep::Fault],
    total: &mut usize,
    tel: &Option<crate::net::telemetry::TelemetrySender>,
) {
    for fault in faults {
        *total += 1;
        warn!("lockstep fault: {fault:?}");
        if let Some(t) = tel {
            t.send(TelemetryEvent::fault(fault));
        }
    }
}

/// Advance the crab's rapier physics + brain by exactly `steps` fixed steps, NOW, from the
/// lockstep driver — the deterministic replacement for Bevy's wall-clock `FixedUpdate`
/// auto-pump (disabled in [`add_external_nn_crab`] by parking `Time<Fixed>` at a huge timestep).
/// Each `run_schedule(FixedMain)` is one [`crate::physics::PHYSICS_DT`] step (rapier's own
/// fixed `dt`, independent of any clock), so N calls advance the body exactly `N · PHYSICS_DT`.
/// We mirror Bevy's own `run_fixed_main_schedule`: swap the generic `Time` to `Time<Fixed>`
/// for the step (so any system reading `Res<Time>` inside the fixed schedule sees the fixed
/// clock), then restore `Time<Virtual>` for the rest of `Update`/render. The step COUNT comes
/// from the integer [`PhysicsCadence`] (not wall clock), so every peer runs the identical
/// number of steps per lockstep tick and the per-tick `phys_digest` matches bit-for-bit.
pub(crate) fn pump_fixed_steps(world: &mut World, steps: u32) {
    use bevy::app::FixedMain;
    use bevy::time::{Fixed, Time, Virtual};

    if steps == 0 {
        return;
    }
    for _ in 0..steps {
        let fixed = world.resource::<Time<Fixed>>().as_generic();
        *world.resource_mut::<Time>() = fixed;
        world.run_schedule(FixedMain);
    }
    let virt = world.resource::<Time<Virtual>>().as_generic();
    *world.resource_mut::<Time>() = virt;
}

/// Park Bevy's wall-clock `FixedUpdate` auto-pump: stretch `Time<Fixed>`'s timestep to a day so
/// `run_fixed_main_schedule` never reaches its `expend()` threshold and auto-runs `FixedMain`
/// zero times. The deterministic crab body is then advanced ONLY by [`pump_fixed_steps`] at the
/// [`PhysicsCadence`] count — wall clock differs per machine/frame-rate, so letting it pump the
/// body would desync peers. The render clock (`Time<Virtual>`/`Time<Real>`) is untouched, and
/// rapier's `TimestepMode::Fixed { dt }` keeps its own `dt`, so each manual pump is still exactly
/// one `PHYSICS_DT` step. One source for the magic timestep so the windowed driver
/// ([`add_external_nn_crab`]) and the headless cross-peer probe
/// ([`crate::net::external_crab::run_cross_peer_probe`]) can't drift on it.
pub(crate) fn park_fixed_auto_pump(world: &mut World) {
    world
        .resource_mut::<bevy::time::Time<bevy::time::Fixed>>()
        .set_timestep(std::time::Duration::from_secs(86_400));
}

/// Advance the lockstep sim by real time on a fixed-timestep accumulator. This is the ONLY
/// writer of sim state, and (apart from the external crab pose) it writes exactly one thing:
/// the local [`Input`] (drained from [`PendingInput`]) via `submit_local_input`. Everything
/// else is the existing deterministic machinery — pump the transport, then advance.
///
/// GCR fold: when the external NN crab is armed ([`ExternalCrabArmed`] — solo OR a
/// networked round with synced weights, [`crate::net::may_arm_external_crab`]), the rapier
/// crab body is stepped INSIDE the lockstep tick: per APPLIED tick we run the deterministic
/// [`PhysicsCadence`] number of physics steps ([`pump_fixed_steps`]) and push the body's
/// resulting pose + weights-folded digest via [`external_crab::sync_external_crab`] BEFORE
/// applying that tick. We advance one tick at a time ([`Lockstep::advance_one`]) so each
/// applied tick gets its own physics batch + pose — a batched `try_advance` (which can apply
/// several ticks at once on catch-up) would smear one pose across them and desync peers. This
/// is an EXCLUSIVE system because pumping the fixed schedule needs `&mut World`.
///
/// A desync fault is logged (lockstep can't recover); the client keeps running so the
/// operator sees it rather than a silent freeze.
fn drive_lockstep(
    world: &mut World,
    mut reported_outcome: Local<bool>,
    mut next_tel_tick: Local<u64>,
    // Last sim tick this system saw, to detect a deterministic restart (RESTART rewinds
    // the sim to tick 0). When it does, the round-decided latch, telemetry cursor, AND the
    // physics cadence below must reset, or the NEXT round never reports "decided", tick
    // telemetry stays suppressed until the counter climbs past the stale watermark, and the
    // crab's per-tick step count starts mid-sequence (a needless cross-peer phase risk).
    mut last_tick: Local<u64>,
    // Cumulative lockstep fault count across the whole round (persists between system
    // runs), so telemetry reports the REAL running desync total — not a per-frame 0. This
    // is the live-debug alarm: a non-zero value on any deck means it has diverged.
    mut total_desyncs: Local<usize>,
    // The deterministic 64:30 physics/sim cadence, advanced once per APPLIED tick while armed.
    // A `Local` (per-round state) reset on the restart edge so two peers stay phase-aligned.
    mut cadence: Local<PhysicsCadence>,
) {
    // Whether the external NN crab drives the sim this round (solo, or networked + synced
    // weights). Read once — the gate is fixed for the round. When off, the integer pursuit
    // drives the crab and no physics is pumped, byte-identical to the pre-GCR path.
    let armed = world
        .get_resource::<crate::net::external_crab::ExternalCrabArmed>()
        .is_some();

    let delta = world.resource::<Time>().delta().as_secs_f64();

    // Clone out the telemetry handle (cheap: an mpsc sender + id) + the roster size so we can
    // READ the sim and push events without holding a borrow of the `NetDriver`. `None` unless
    // this is networked play with a collector. Telemetry never writes the sim.
    let (tel, roster_len) = {
        let state = world.non_send_resource::<GameState>();
        match &state.input_source {
            InputSource::Networked(net) => (net.telemetry().cloned(), net.roster_len()),
            _ => (None, 0),
        }
    };
    if *next_tel_tick == 0 {
        *next_tel_tick = TELEMETRY_TICK_EVERY;
    }

    world.non_send_resource_mut::<GameState>().accumulator += delta;

    // Single-player enter/exit a vehicle (client-local; the sim never sees it). Drain the
    // E-tap latch ONCE per frame: board a plane at the foot player's spot, or step back out.
    // Solo only — a networked round freezes its pilot set over the wire at formation (rl#43),
    // so this toggle is inert there and the deterministic lockstep is untouched.
    {
        let toggle = std::mem::take(&mut world.resource_mut::<PendingInput>().toggle_vehicle);
        let solo = toggle
            && matches!(
                world.non_send_resource::<GameState>().input_source,
                InputSource::Solo
            );
        if solo {
            if world.resource::<LocalVehicle>().piloting() {
                // Step out: drop the plane, the camera falls back to the foot player.
                *world.resource_mut::<LocalVehicle>() = LocalVehicle::OnFoot;
            } else {
                // Board: spawn a plane at the local player's current ground spot + facing, if
                // it is still alive to board. Reuses the sim's one plane-spawn definition.
                let state = world.non_send_resource::<GameState>();
                let me = state.ls.me();
                let boarding = state
                    .ls
                    .sim()
                    .player(me)
                    .filter(|p| p.status() == PlayerStatus::Alive)
                    .map(|p| Plane::spawn(p.pos(), p.yaw()));
                if let Some(plane) = boarding {
                    *world.resource_mut::<LocalVehicle>() =
                        LocalVehicle::Piloting { plane, prev: plane };
                }
            }
        }
    }

    let mut applied = 0u32;
    loop {
        // Pace by wall clock: one local input issued per sim tick, bounded per frame so a
        // stall can't trigger an unbounded catch-up spiral.
        {
            let state = world.non_send_resource::<GameState>();
            if state.accumulator < TICK_DT || applied >= MAX_TICKS_PER_FRAME {
                break;
            }
        }
        world.non_send_resource_mut::<GameState>().accumulator -= TICK_DT;
        applied += 1;

        // Build THIS tick's local input from the accumulated controls, draining the accrued
        // look + latched buttons (movement axes are re-sampled next frame).
        let input = {
            let mut pending = world.resource_mut::<PendingInput>();
            let look_axis = (pending.yaw_delta / MAX_YAW_PER_TICK_RADIANS).clamp(-1.0, 1.0);
            let btns = (if pending.action { buttons::ACTION } else { 0 })
                | (if pending.restart { buttons::RESTART } else { 0 });
            let input = Input::new(pending.strafe, pending.forward, look_axis, btns);
            pending.yaw_delta = 0.0;
            pending.action = false;
            pending.restart = false;
            input
        };

        // While piloting (single-player), the foot player feeds the sim a NEUTRAL input — it
        // just stands at the boarding spot — and the real input flies the client-side plane
        // below instead. On foot, the real input drives the sim as usual.
        let piloting = world.resource::<LocalVehicle>().piloting();
        let sim_input = if piloting { Input::default() } else { input };

        // Submit our input + gather the other players' inputs from whichever source this run
        // uses, then record them — every path lands at `record_remote`, the same entry a wire
        // peer takes, so the sim can't tell the sources apart.
        let (issue_tick, faults) = {
            let mut state = world.non_send_resource_mut::<GameState>();
            let me = state.ls.me();
            let issue_tick = state.ls.next_tick();
            let msg = state.ls.submit_local_input(sim_input);
            // Collect peer messages first (releasing the `input_source`/`ls` co-borrow via
            // `&mut *state`) before recording into `ls`.
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
            let mut faults = Vec::new();
            for from in peer_msgs {
                if from.pid != me
                    && let Some(fault) = state.ls.record_remote(from.pid, from.msg)
                {
                    faults.push(fault);
                }
            }
            (issue_tick, faults)
        };
        report_faults(&faults, &mut total_desyncs, &tel);

        // Fly the client-side plane one tick with the real input (single-player vehicle),
        // keeping last tick's pose for the camera interpolation. The sim never sees this — it
        // is the play layer's own body, so the deterministic core stays integer-only.
        if let LocalVehicle::Piloting { plane, prev } = &mut *world.resource_mut::<LocalVehicle>() {
            *prev = *plane;
            plane.step(input);
        }

        // Apply every now-ready tick, ONE at a time. Per applied tick: snapshot the pre-step
        // state for interpolation; if armed, step the crab body by the deterministic cadence
        // and push its resulting pose + digest BEFORE advancing, so this tick's
        // grab/extraction/outcome resolve against the real NN crab and every peer folds the
        // identical `phys_digest`. When not armed, the integer pursuit drives the crab and the
        // tick is byte-identical to the pre-GCR path.
        //
        // This inner drain is UNBOUNDED on purpose: it applies every tick whose inputs are
        // ready (a catch-up after a stall must apply them all, in order, to stay in lockstep —
        // and each applied tick advances the cadence, so peers stay phase-aligned regardless of
        // how the catch-up batches). `MAX_TICKS_PER_FRAME` bounds only input ISSUANCE (the outer
        // loop), which is what prevents a real-time spiral.
        loop {
            {
                let state = world.non_send_resource::<GameState>();
                if !state.ls.next_tick_ready() {
                    break;
                }
            }
            {
                let mut state = world.non_send_resource_mut::<GameState>();
                state.prev = SimSnapshot::capture(state.ls.sim());
            }
            if armed {
                let steps = cadence.steps_for_next_tick();
                pump_fixed_steps(world, steps);
                // Push the freshly-stepped body's pose + weights-folded digest + refresh the
                // hunted player — the shared handshake (one source with the headless probe).
                // `resource_scope` lifts the bridge out so we can hold it AND `GameState`'s
                // `ls` mutably at once (both live in the same `World`).
                world.resource_scope(
                    |world, mut bridge: Mut<crate::net::external_crab::ExternalCrabBridge>| {
                        let mut state = world.non_send_resource_mut::<GameState>();
                        crate::net::external_crab::sync_external_crab(&mut state.ls, &mut bridge);
                    },
                );
            }
            let (tick_faults, restarted) = {
                let mut state = world.non_send_resource_mut::<GameState>();
                let before = state.ls.sim().tick();
                let faults = state.ls.advance_one().expect("next_tick_ready was true");
                // A RESTART rewinds the sim to tick 0 INSIDE this step. Reset the cadence AT that
                // edge — not at end-of-frame — so the new round's first ticks step in phase on
                // every peer regardless of how each peer's frames batched the drain (an
                // end-of-frame reset would let one peer step a few post-restart ticks on the stale
                // phase before resetting, desyncing them).
                let restarted = state.ls.sim().tick() < before;
                if restarted {
                    *cadence = PhysicsCadence::default();
                }
                (faults, restarted)
            };
            // Same restart edge as the cadence reset above: re-seed the crab bridge to the
            // round's (rebuilt) spawn, so the next `sync_external_crab` pushes the spawn pose
            // instead of snapping the restarted crab onto the still-walking body's accumulated
            // position. The cross-peer-determinism argument lives on `restart_to_spawn`'s doc
            // (it fires off this same shared-input edge). Only meaningful while armed (else there
            // is no bridge driving the crab).
            if restarted && armed {
                let spawn = world.non_send_resource::<GameState>().ls.sim().crab().pos();
                world
                    .resource_mut::<crate::net::external_crab::ExternalCrabBridge>()
                    .restart_to_spawn(spawn);
            }
            report_faults(&tick_faults, &mut total_desyncs, &tel);
        }

        // Sampled telemetry: a Tick snapshot + the local input every TELEMETRY_TICK_EVERY
        // applied ticks. Read-only on the sim; best-effort (drops if the link can't keep up).
        if let Some(t) = &tel {
            let state = world.non_send_resource::<GameState>();
            if state.ls.sim().tick() >= *next_tel_tick {
                *next_tel_tick =
                    (state.ls.sim().tick() / TELEMETRY_TICK_EVERY + 1) * TELEMETRY_TICK_EVERY;
                t.send(TelemetryEvent::tick(state.ls.sim(), *total_desyncs, roster_len));
                // The input the SIM actually applied this tick (neutral while piloting).
                t.send(TelemetryEvent::input(issue_tick, sim_input));
            }
        }
    }

    if applied == MAX_TICKS_PER_FRAME {
        // Shed the backlog rather than spiral: drop accumulated time past one tick.
        let mut state = world.non_send_resource_mut::<GameState>();
        state.accumulator = state.accumulator.min(TICK_DT);
    }

    // Restart detector: a RESTART press rewinds the sim to tick 0, so a tick lower than last
    // frame's means the round restarted. Clear the round-decided latch and snap the telemetry
    // cursor back to the new (low) tick. (The cadence reset is NOT here — it must happen at the
    // exact rewind edge inside the drain above, or post-restart ticks applied in the same frame
    // would step on the stale phase; these two resets are reporting-only, so frame-relative is
    // fine.)
    let (now_tick, outcome) = {
        let state = world.non_send_resource::<GameState>();
        (state.ls.sim().tick(), state.ls.sim().outcome())
    };
    if now_tick < *last_tick {
        *reported_outcome = false;
        *next_tel_tick = (now_tick / TELEMETRY_TICK_EVERY + 1) * TELEMETRY_TICK_EVERY;
    }
    *last_tick = now_tick;

    if !*reported_outcome && outcome != Outcome::Ongoing {
        *reported_outcome = true;
        info!("round decided: {outcome:?}");
        if let Some(t) = &tel {
            let state = world.non_send_resource::<GameState>();
            t.send(TelemetryEvent::round_decided(state.ls.sim()));
        }
    }
}

// ---------------------------------------------------------------------------
// Input: WASD + mouse + gamepad → PendingInput
// ---------------------------------------------------------------------------

/// Sample local controls each render frame into [`PendingInput`] and integrate the
/// client-side camera pitch. Produces ONLY data destined for the next [`Input`] — it
/// never touches the sim. The game is fully playable on keyboard+mouse OR a gamepad
/// alone, the two additive:
/// - move: WASD / left stick / d-pad
/// - look: mouse / right stick (yaw → sim, pitch → client-only)
/// - action (extract): Space / mouse-left / pad South / pad RT
/// - restart: R / pad Start (edge-triggered → [`buttons::RESTART`], lockstep)
/// - quit: Esc / hold pad Select (handled in [`exit_on_esc`])
///
/// Analog stick magnitudes are raw f32 here, but the ONLY path from this function to
/// the sim is via [`Input::new`] in [`drive_lockstep`], which quantizes every axis to
/// the fixed-point grid — the identical boundary keyboard/mouse cross. So no f32 ever
/// reaches the deterministic sim; the i16 [`Input`] that each peer broadcasts is the
/// shared truth, and a pad input is bit-for-bit a keyboard input of the same magnitude.
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

    // Every DISCRETE key/button below is looked up in the one control map
    // (`controls::CONTROL_MAP`) via these helpers, never hardcoded — so the keys the
    // client polls are exactly the keys the on-screen legend shows (no drift).
    // `kc(action)` is the keyboard key; `pad_pressed`/`pad_just_pressed` fold the pad's
    // primary+alternate buttons. The ANALOG channels (stick→axis math, mouse motion,
    // D-pad digital move) aren't rebindable bindings, so they stay inline here.
    let kc = controls::key_code_for;
    let held = |a| kc(a).is_some_and(|k| keys.pressed(k));

    // --- Move axes (last sample wins; re-sampled every frame) ---
    let mut strafe = 0.0f32;
    let mut forward = 0.0f32;
    forward += held(Action::MoveForward) as i32 as f32;
    forward -= held(Action::MoveBack) as i32 as f32;
    strafe += held(Action::StrafeRight) as i32 as f32;
    strafe -= held(Action::StrafeLeft) as i32 as f32;

    let mut action = held(Action::Extract);
    // Restart the round (R). Latched here, sent as buttons::RESTART, edge-triggered in
    // the sim — so it restarts every peer in lockstep, not a local-only reset.
    if kc(Action::Restart).is_some_and(|k| keys.just_pressed(k)) {
        pending.restart = true;
    }
    // Enter/exit a vehicle (E). A tap-toggle, edge-triggered like restart, but client-local
    // (single-player) — `drive_lockstep` boards/steps-out; it never reaches the sim.
    if kc(Action::EnterExit).is_some_and(|k| keys.just_pressed(k)) {
        pending.toggle_vehicle = true;
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

    // Gamepad (full pad-only play): left stick moves, right stick looks, South/RT =
    // extract, Start (tap) = restart. Quit (hold North) and reveal-controls (hold Back)
    // live in `quit_game` / the overlay system. Sticks have a deadzone so a resting stick
    // doesn't creep. Stick magnitudes are raw f32 here but cross into the sim ONLY through
    // `Input::new`'s fixed-point quantization (below) — the SAME boundary keyboard/mouse
    // pass — so the analog values never reach the deterministic sim.
    for gp in gamepads.iter() {
        // The analog stick → axis arithmetic (deadzone + look scaling) lives in the pure
        // `pad_stick_axes` so the determinism test can exercise the SAME transform the
        // client runs, with no copy to drift out of sync.
        let pad = pad_stick_axes(gp.left_stick(), gp.right_stick(), dt);
        strafe += pad.strafe;
        forward += pad.forward;
        d_yaw += pad.d_yaw;
        d_pitch += pad.d_pitch;
        // D-pad mirrors WASD as a digital move (kids reach for it instinctively, and it's
        // the obvious second way to walk): full ±1.0 on each axis, summed with the stick
        // and clamped at the funnel below — the SAME path WASD takes, so it quantizes
        // identically. Up=forward, Down=back, Right/Left=strafe (pre-negation downstream).
        forward += (gp.pressed(GamepadButton::DPadUp) as i32
            - gp.pressed(GamepadButton::DPadDown) as i32) as f32;
        strafe += (gp.pressed(GamepadButton::DPadRight) as i32
            - gp.pressed(GamepadButton::DPadLeft) as i32) as f32;
        action |= controls::gamepad_buttons_for(Action::Extract).any(|b| gp.pressed(b));
        // Restart on Start (tap), edge-triggered exactly like keyboard R: latched here,
        // sent as buttons::RESTART, so every peer restarts on the same tick in lockstep (a
        // local-only reset would desync). Edge (just_pressed), not held. Quit is on its OWN
        // pad button (North, held), NOT Start — so beginning a quit can't fire this restart.
        if controls::gamepad_buttons_for(Action::Restart).any(|b| gp.just_pressed(b)) {
            pending.restart = true;
        }
        // Enter/exit vehicle on pad West (X), tap — same client-local toggle as keyboard E.
        if controls::gamepad_buttons_for(Action::EnterExit).any(|b| gp.just_pressed(b)) {
            pending.toggle_vehicle = true;
        }
    }
    // Mouse-left also fires action, for mouse-only play.
    if let Some(mb) = controls::MouseInput::Left.mouse_button() {
        action |= mouse_buttons.pressed(mb);
    }

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

/// Quit the game (windowed play only): the keyboard Quit key (Esc), or HOLD the gamepad
/// Quit button (North/Y) for [`PAD_QUIT_HOLD_SECS`]. Both bindings come from the one control
/// map ([`controls::CONTROL_MAP`]), so this matches the legend. Purely client-local — sends
/// Bevy's [`AppExit`]; no sim/lockstep involvement, so it can't desync a peer (each client
/// just closes its own window) and the others play on. The pad Quit is a HOLD on its OWN
/// button (not Start, which restarts), so a stray press can't end the round for the couch.
fn quit_game(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    mut quit_held: Local<f32>,
    mut exit: MessageWriter<AppExit>,
) {
    if controls::key_code_for(Action::Quit).is_some_and(|k| keys.just_pressed(k)) {
        exit.write(AppExit::Success);
        return;
    }
    // Accumulate held time while the pad Quit button is down on ANY pad; reset the instant
    // it's released, so only a sustained hold (not repeated taps) reaches the threshold.
    if pad_action_held(&gamepads, Action::Quit) {
        *quit_held += time.delta_secs();
        if *quit_held >= PAD_QUIT_HOLD_SECS {
            exit.write(AppExit::Success);
        }
    } else {
        *quit_held = 0.0;
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
    // Whether the solo NN crab is ACTIVE this round: when so, the placeholder crab box is
    // spawned hidden (the real rig is the crab). Keyed on the active gate, NOT the bridge's
    // presence — on the menu path the bridge exists even for a networked round, which must
    // keep the visible integer crab box. See the crab spawn below.
    external_crab_armed: Option<Res<crate::net::external_crab::ExternalCrabArmed>>,
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

    // The giant crab: Sally's real collider silhouette (see `spawn_crab_silhouette`),
    // CRAB_SCALE× a player, for the integer-pursuit crab. When the external NN crab is
    // armed (solo, or networked-synced) the real rapier rig (wireframe / skin) is the
    // crab, so this silhouette is spawned HIDDEN there — the armed gate is the tell — and
    // the rig shows instead. (We still spawn it so `apply_transforms`'s crab query is
    // satisfied either way; it just stays invisible.)
    let crab_hidden = external_crab_armed.is_some();
    let crab_h = PLAYER_HEIGHT * CRAB_SCALE as f32;
    let crab_root = commands
        .spawn((
            Transform::from_translation(world(state.ls.sim().crab().pos(), 0.0)),
            if crab_hidden {
                Visibility::Hidden
            } else {
                Visibility::default()
            },
            CrabAvatar,
        ))
        .id();
    // Body: Sally's ACTUAL physics colliders (carapace box + every leg/eye/claw capsule
    // from the render recipe), not a featureless placeholder box (#108). The shapes ride
    // `crab_root`, which `apply_transforms` re-poses to the sim crab every frame.
    spawn_crab_silhouette(&mut commands, &mut meshes, &mut materials, crab_root, crab_h);
}

/// Draw the giant integer-pursuit crab as its REAL physics colliders — the carapace
/// cuboid and every link capsule that [`crate::bot::body::render_recipe`] yields,
/// scaled to the giant height and oriented claws-forward (+Z, the crab's facing) — and
/// parent them to `crab_root`. Drawing the SAME shapes the sim body uses is the point
/// of #108: the cosmetic crab can't drift from the body it depicts, and it reads as
/// Sally instead of a box. `render_recipe` is the single model-vs-fallback selector
/// (shared with the trainer), so this never invents a second source of geometry. The
/// integer crab carries only a root pose (no per-limb articulation), so the silhouette
/// is the rest stance posed rigidly — the legs don't walk, but the shape is honest.
fn spawn_crab_silhouette(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    crab_root: Entity,
    crab_h: f32,
) {
    use crate::bot::rig::RestShape;

    let sil = crate::bot::rig::recipe_silhouette(&crate::bot::body::render_recipe());
    let shapes = || sil.limbs.iter().chain(std::iter::once(&sil.carapace));

    // Orient the rig claws-forward (+Z). The recipe's forward axis isn't necessarily +Z,
    // so DERIVE it from the geometry — carapace-center → limb centroid, flattened to the
    // ground plane — rather than hard-code a constant that could drift if the asset
    // changes. Both vectors are horizontal, so `from_rotation_arc` yields a pure yaw; a
    // degenerate recipe (no limbs) leaves the rig as-is (Vec3::Z → identity).
    let shape_mid = |s: &RestShape| match *s {
        RestShape::Capsule { a, b, .. } => (a + b) * 0.5,
        RestShape::Cuboid { center, .. } => center,
    };
    let fwd = if sil.limbs.is_empty() {
        Vec3::ZERO
    } else {
        let cc = shape_mid(&sil.carapace);
        let centroid = sil.limbs.iter().map(shape_mid).sum::<Vec3>() / sil.limbs.len() as f32;
        let mut d = centroid - cc;
        d.y = 0.0;
        d.normalize_or_zero()
    };
    let r = Quat::from_rotation_arc(
        if fwd.length_squared() < 1e-6 { Vec3::Z } else { fwd },
        Vec3::Z,
    );

    // AABB in the claws-forward frame, so we can scale the rig to the giant height and
    // stand its base (min-y) on the ground.
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    let mut grow = |p: Vec3| {
        lo = lo.min(p);
        hi = hi.max(p);
    };
    for s in shapes() {
        match *s {
            RestShape::Capsule { a, b, radius } => {
                let rad = Vec3::splat(radius);
                for p in [r * a, r * b] {
                    grow(p - rad);
                    grow(p + rad);
                }
            }
            RestShape::Cuboid { center, half } => {
                let cen = r * center;
                for sx in [-1.0_f32, 1.0] {
                    for sy in [-1.0_f32, 1.0] {
                        for sz in [-1.0_f32, 1.0] {
                            grow(cen + r * Vec3::new(sx * half.x, sy * half.y, sz * half.z));
                        }
                    }
                }
            }
        }
    }
    if !lo.is_finite() || !hi.is_finite() || (hi.y - lo.y) < 1e-4 {
        // Empty/degenerate recipe: never leave the crab invisible — keep a plain box.
        spawn_fallback_crab_box(commands, meshes, materials, crab_root, crab_h);
        return;
    }

    let scale = crab_h / (hi.y - lo.y);
    // Recenter horizontally on the root and stand the base on the ground (y=0).
    let origin = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    let map = |p: Vec3| (r * p - origin) * scale;

    let carapace_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.18, 0.12),
        perceptual_roughness: 0.8,
        ..default()
    });
    let limb_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.28, 0.18),
        perceptual_roughness: 0.7,
        ..default()
    });

    let mut children: Vec<Entity> = Vec::with_capacity(sil.limbs.len() + 1);
    for s in shapes() {
        let child = match *s {
            RestShape::Capsule { a, b, radius } => {
                let a = map(a);
                let b = map(b);
                let seg = b - a;
                let len = seg.length();
                let rot = if len > 1e-5 {
                    Quat::from_rotation_arc(Vec3::Y, seg / len)
                } else {
                    Quat::IDENTITY
                };
                commands
                    .spawn((
                        Mesh3d(meshes.add(Capsule3d::new(radius * scale, len))),
                        MeshMaterial3d(limb_mat.clone()),
                        Transform::from_translation((a + b) * 0.5).with_rotation(rot),
                    ))
                    .id()
            }
            RestShape::Cuboid { center, half } => commands
                .spawn((
                    Mesh3d(meshes.add(Cuboid::new(
                        half.x * 2.0 * scale,
                        half.y * 2.0 * scale,
                        half.z * 2.0 * scale,
                    ))),
                    MeshMaterial3d(carapace_mat.clone()),
                    Transform::from_translation(map(center)).with_rotation(r),
                ))
                .id(),
        };
        children.push(child);
    }
    commands.entity(crab_root).add_children(&children);
}

/// The pre-#108 placeholder: a plain red box crab with a forward claw wedge. Kept ONLY
/// as the safety net for a degenerate/empty collider recipe, so a broken asset shows a
/// box rather than an invisible crab.
fn spawn_fallback_crab_box(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    crab_root: Entity,
    crab_h: f32,
) {
    let crab_w = PLAYER_RADIUS * 2.0 * CRAB_SCALE as f32;
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
    vehicle: Res<LocalVehicle>,
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
    // interpolated pose, looking along its heading+pitch, with the client pitch still added
    // so the pilot can glance around. An on-foot player keeps the ground eye view.
    if let Ok(mut cam) = cam_q.single_mut() {
        if let LocalVehicle::Piloting { plane, prev } = &*vehicle {
            // Single-player: fly from the CLIENT-side plane (the play layer's own body — it
            // is not in the sim, so the deterministic core stays integer-only).
            *cam = plane_cockpit_camera(*prev, *plane, alpha, pitch.0);
        } else if let Some(plane_now) = sim.plane(local) {
            // A SIM-side pilot (networked vehicle, rl#43): same cockpit view from sim state.
            let plane_prev = state.prev.planes.get(&local).copied().unwrap_or(plane_now);
            *cam = plane_cockpit_camera(plane_prev, plane_now, alpha, pitch.0);
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

/// The first-person cockpit camera for a plane: eye at the interpolated 3D position,
/// looking along the interpolated heading + pitch with `extra_pitch` (the client free-look)
/// added on. The ONE cockpit-view formula, shared by the single-player client vehicle and
/// the sim-side networked pilot, so both fly from the identical view with no copy to drift.
fn plane_cockpit_camera(prev: Plane, now: Plane, alpha: f32, extra_pitch: f32) -> Transform {
    let eye = lerp_pos3(prev.pos(), now.pos(), alpha);
    let heading = lerp_yaw(prev.heading(), now.heading(), alpha);
    // Pitch reuses lerp_yaw because it's a turn-unit angle too; since pitch is bounded
    // (never wraps), the shortest-arc handling is a harmless no-op here.
    let plane_pitch = lerp_yaw(prev.pitch(), now.pitch(), alpha);
    let look_dir = look_direction(heading, plane_pitch + extra_pitch);
    Transform::from_translation(eye).looking_at(eye + look_dir, Vec3::Y)
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
    // Status + the one-line objective only. The control bindings are NOT duplicated here:
    // they live in the hold-to-reveal overlay + corner hint (the controls UI), which derive
    // from the one control map — so there's a single on-screen source for them.
    **text =
        format!("You: {status}   |   reach the green pillar, extract - dodge the crab{outcome}",);
}

/// Whether any connected gamepad currently HOLDS a button bound to `action`. The shared
/// read for the discrete pad buttons — folds the map's mapped buttons (via
/// [`controls::gamepad_buttons_for`]) across every pad. Factored out so `gather_input` and
/// `quit_game` don't each re-spell the nested any-any loop. (The overlay's own reveal-held
/// read lives in [`crate::controls`].)
fn pad_action_held(gamepads: &Query<&Gamepad>, action: Action) -> bool {
    gamepads
        .iter()
        .any(|gp| controls::gamepad_buttons_for(action).any(|b| gp.pressed(b)))
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
        // Make UI render into THIS offscreen target. Bevy renders UI to the default-window
        // camera automatically, but the screenshot path has no window — without this marker
        // the HUD + controls overlay never composite into the captured texture, so an
        // evidence frame would miss them. The windowed client doesn't need it (its window
        // camera is the implicit UI target).
        bevy::ui::IsDefaultUiCamera,
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

/// After the sim has run a few ticks and the GPU pipeline has warmed, capture one PNG
/// of the FP view and exit. The settle/capture/exit bookkeeping is the shared
/// [`crate::screenshot`] primitive; this system just composes the FP scene's single
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

// ---------------------------------------------------------------------------
// Boot menu (rl#58): client-side egui Host / Join + host-triggered lobby, gated to the
// Menu/Connecting phases. Builds the round ONLY at the Playing transition, so it can't
// touch the sim.
// ---------------------------------------------------------------------------

/// The boot-menu front-end: the egui Host / Join UI, the lobby (live roster + the host's
/// Start/Cancel), the background-formation poll, and the `OnEnter(Playing)` round installer.
/// This is the ONLY Bevy/egui code for the menu; the pure (testable, Bevy-free) connection
/// orchestration lives in [`crate::net::menu`]. The split keeps the determinism-relevant
/// claim simple: nothing here builds or reads a [`Lockstep`]/[`Sim`] except at the Playing
/// transition, from the unchanged formation machinery.
mod menu_ui {
    use bevy::prelude::*;
    use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

    use super::{AppPhase, PendingRound};
    use crate::net::menu::{
        self, ChooserItem, EndpointId, Formation, LobbyItem, MenuAction, MenuInput, MenuNav,
        StartChoice,
    };

    /// Wires the boot menu into the windowed app: the egui menu + connecting-poll pass.
    /// The round install at `OnEnter(Playing)` is `ensure_round_installed` in the parent
    /// module (always scheduled, chained ahead of the spawns) — the menu only *parks* its
    /// chosen round in [`PendingRound`]. Carries the shared match seed + optional telemetry
    /// collector so a networked Host/Join formation gets them.
    pub struct MenuPlugin {
        pub seed: u64,
        pub telemetry: Option<EndpointId>,
        /// Our NN-crab checkpoint digest (rl#82, GCR), `0` for none. Advertised in networked
        /// formation so peers can agree on a shared brain before arming the float crab.
        pub weights_digest: u64,
        /// Our crab-model-asset digest (rl#100, GCR), `0` for none. Advertised alongside
        /// `weights_digest` so peers can agree on a shared collider asset before arming.
        pub asset_digest: u64,
    }

    /// The camera the menu/connecting screens render into. bevy_egui 0.39 is
    /// camera-driven — it attaches its primary context to a [`Camera`] entity, so WITHOUT
    /// a camera the egui pass is skipped and the menu never draws. The round spawns its own
    /// `Camera3d` only at `OnEnter(Playing)`, so the menu needs this one of its own for the
    /// pre-round phases; it's despawned the instant we enter Playing so it never coexists
    /// with (or double-renders over) the FP camera.
    #[derive(Component)]
    struct MenuCamera;

    impl Plugin for MenuPlugin {
        fn build(&self, app: &mut App) {
            if !app.is_plugin_added::<EguiPlugin>() {
                app.add_plugins(EguiPlugin::default());
            }
            app.insert_non_send_resource(MenuState::new(
                self.seed,
                self.telemetry,
                self.weights_digest,
                self.asset_digest,
            ))
                // A 2D camera for the menu so bevy_egui has a context to render into.
                // Spawned on entering Menu (the default phase, so it fires at startup on the
                // menu boot; never on the scripted Boot::Round path, which supersedes Menu
                // with Playing before any transition). Re-entering Menu (Cancel/error from
                // Connecting) despawns any prior one first, so there's never a duplicate.
                .add_systems(OnEnter(AppPhase::Menu), (spawn_menu_camera, reset_menu_nav))
                // Tear it down as the round begins, before the FP Camera3d spawns, so the
                // two never coexist.
                .add_systems(OnEnter(AppPhase::Playing), despawn_menu_camera)
                // The menu + connecting poll draw in the egui pass (per render frame),
                // gated to the two pre-round phases so they vanish once Playing.
                .add_systems(
                    EguiPrimaryContextPass,
                    menu_screen.run_if(not(in_state(AppPhase::Playing))),
                );
        }
    }

    /// Spawn the menu's 2D camera (despawning any leftover first, so re-entering Menu from
    /// Connecting can't stack two). Without a camera bevy_egui renders nothing.
    fn spawn_menu_camera(mut commands: Commands, existing: Query<Entity, With<MenuCamera>>) {
        for e in existing.iter() {
            commands.entity(e).despawn();
        }
        commands.spawn((Camera2d, MenuCamera));
    }

    /// Despawn the menu camera as the round starts, so it doesn't linger into Playing and
    /// double-render over the FP `Camera3d`.
    fn despawn_menu_camera(mut commands: Commands, cams: Query<Entity, With<MenuCamera>>) {
        for e in cams.iter() {
            commands.entity(e).despawn();
        }
    }

    /// The menu's working state (non-send: a started [`Formation`] holds a tokio runtime
    /// via the `NetDriver`, which isn't `Send`). Holds the navigation FSM, the join-code text
    /// field, the in-flight formation, and any error to show. The finished round is parked in
    /// the parent's [`PendingRound`], not here. All pre-round UI bookkeeping — none of it is
    /// sim state.
    struct MenuState {
        seed: u64,
        telemetry: Option<EndpointId>,
        /// Our NN-crab checkpoint digest (rl#82, GCR), `0` for none — handed to
        /// [`crate::net::menu::begin`] so networked formation advertises it.
        weights_digest: u64,
        /// Our crab-model-asset digest (rl#100, GCR), `0` for none — handed to
        /// [`crate::net::menu::begin`] alongside `weights_digest`.
        asset_digest: u64,
        /// The pure navigation FSM ([`MenuNav`]) — focus + the chooser/lobby transition.
        /// Folded by controller/keyboard input AND egui clicks through one path, so every
        /// confirm (Start included) routes through the same tested dispatch.
        nav: MenuNav,
        /// Edge latch for left-stick menu nav: `true` while the stick is held past the nav
        /// threshold, cleared when it recenters, so a held stick steps one item, not many.
        stick_latched: bool,
        /// The join-code text the player is typing (an endpoint id), for Join-by-code.
        code_input: String,
        /// A networked Host/Join formation running on a background thread, while Connecting.
        forming: Option<Formation>,
        /// Last error to surface on the menu (bad code, formation failed), cleared when the
        /// player retries.
        error: Option<String>,
    }

    impl MenuState {
        fn new(
            seed: u64,
            telemetry: Option<EndpointId>,
            weights_digest: u64,
            asset_digest: u64,
        ) -> Self {
            Self {
                seed,
                telemetry,
                weights_digest,
                asset_digest,
                nav: MenuNav::new(),
                stick_latched: false,
                code_input: String::new(),
                forming: None,
                error: None,
            }
        }
    }

    /// Reset the navigation FSM to the chooser whenever we (re)enter the Menu phase, so a
    /// Cancel/error return from Connecting always lands on a clean Host-focused chooser.
    fn reset_menu_nav(mut state: NonSendMut<MenuState>) {
        state.nav = MenuNav::new();
        state.stick_latched = false;
    }

    /// The single egui system for the boot flow (rl#58, rl#82): poll the formation, gather
    /// controller/keyboard navigation, draw the current screen, and drive every transition
    /// through the pure [`MenuNav`] FSM + one exhaustive [`apply_action`] dispatch.
    ///
    /// Input unification (rl#82): controller (D-pad/stick + A/B), keyboard (arrows/WASD +
    /// Enter/Esc), and egui clicks ALL reduce to a [`MenuInput`] folded through `MenuNav`, so
    /// every confirm — Start included — takes the same wired path. A gamepad-only player (the
    /// Steam Deck case) can now operate the whole menu; before, the menu was mouse-click only,
    /// so on the Deck nothing in it worked, the dead lobby Start among them.
    ///
    /// Determinism: this only ever *selects/commands* a formation and reads its finished
    /// result. The round it parks (in [`PendingRound`]) is built by [`menu::ready_from`] /
    /// [`menu::solo_round`] from the unchanged barrier output — no sim state originates here.
    fn menu_screen(
        mut contexts: EguiContexts,
        mut state: NonSendMut<MenuState>,
        mut pending: NonSendMut<PendingRound>,
        phase: Res<State<AppPhase>>,
        mut next: ResMut<NextState<AppPhase>>,
        keys: Res<ButtonInput<KeyCode>>,
        gamepads: Query<&Gamepad>,
    ) -> Result {
        let ctx = contexts.ctx_mut()?;

        // Connecting: poll the background formation FIRST, so a finished match transitions
        // this frame before we draw or take any input on a screen that's about to vanish.
        if matches!(phase.get(), AppPhase::Connecting)
            && poll_formation(&mut state, &mut pending, &mut next)
        {
            return Ok(());
        }

        // The live roster size, the only context the FSM needs (host Start: solo vs networked).
        let lobby_len = state.forming.as_ref().map(|f| f.lobby_len()).unwrap_or(0);

        // Suppress KEYBOARD nav while egui has keyboard focus (the player is typing in the
        // join-code field) — otherwise Space/Enter/W/S double as nav and would, e.g., fire
        // Confirm mid-paste. The gamepad stays live (it never feeds the text field).
        let typing = ctx.wants_keyboard_input();

        // Controller / keyboard navigation: reduce raw input to FSM events, fold each, and
        // execute the action. A leaving action (Start/Cancel) makes the rest of the frame moot.
        let inputs = gather_menu_inputs(&keys, &gamepads, typing, &mut state.stick_latched);
        for input in inputs {
            let action = state.nav.step(input, lobby_len);
            if apply_action(action, &mut state, &mut pending, &mut next) {
                return Ok(());
            }
        }

        // Draw the current screen and route any click through the SAME FSM path (focus the
        // clicked item, then Confirm), so a click and a controller confirm can't diverge.
        match phase.get() {
            AppPhase::Menu => {
                if let Some(item) = draw_chooser(ctx, &mut state) {
                    state.nav.focus_chooser(item);
                    let action = state.nav.step(MenuInput::Confirm, lobby_len);
                    apply_action(action, &mut state, &mut pending, &mut next);
                }
            }
            AppPhase::Connecting => {
                let lobby = state
                    .forming
                    .as_ref()
                    .map(|f| f.roster())
                    .unwrap_or_default();
                if let Some(item) = draw_lobby(ctx, &state, &lobby) {
                    state.nav.focus_lobby(item);
                    let action = state.nav.step(MenuInput::Confirm, lobby_len);
                    apply_action(action, &mut state, &mut pending, &mut next);
                }
            }
            // Playing is gated out by the run condition; nothing to draw.
            AppPhase::Playing => {}
        }
        Ok(())
    }

    /// Reduce this frame's raw keyboard + gamepad state to [`MenuInput`] events (edge-
    /// triggered, so a held control steps one item). Controller: D-pad + left stick to move,
    /// South (A) to confirm, East (B) to back/cancel — folded across every connected pad.
    /// Keyboard: arrows/WASD to move, Enter/Space to confirm, Esc to back. The thin, untested
    /// input-gathering layer; all the navigation LOGIC lives in the unit-tested [`MenuNav`].
    fn gather_menu_inputs(
        keys: &ButtonInput<KeyCode>,
        gamepads: &Query<&Gamepad>,
        typing: bool,
        stick_latched: &mut bool,
    ) -> Vec<MenuInput> {
        let mut out = Vec::new();

        // Keyboard, edge-triggered — skipped entirely while a text field has focus (`typing`)
        // so keystrokes meant for the join-code field don't also drive navigation.
        if !typing {
            if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
                out.push(MenuInput::Up);
            }
            if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyS) {
                out.push(MenuInput::Down);
            }
            if keys.just_pressed(KeyCode::Enter)
                || keys.just_pressed(KeyCode::NumpadEnter)
                || keys.just_pressed(KeyCode::Space)
            {
                out.push(MenuInput::Confirm);
            }
            if keys.just_pressed(KeyCode::Escape) {
                out.push(MenuInput::Back);
            }
        }

        // Gamepad, folded across all pads. D-pad + face buttons are edge-triggered; the stick
        // is its own analog channel (handled below with a re-center latch).
        let mut up = false;
        let mut down = false;
        let mut confirm = false;
        let mut back = false;
        // Largest-magnitude left-stick Y across pads, so one player's stick drives the menu.
        let mut stick_y = 0.0f32;
        for gp in gamepads.iter() {
            up |= gp.just_pressed(GamepadButton::DPadUp);
            down |= gp.just_pressed(GamepadButton::DPadDown);
            confirm |= gp.just_pressed(GamepadButton::South);
            back |= gp.just_pressed(GamepadButton::East);
            let y = gp.left_stick().y;
            if y.abs() > stick_y.abs() {
                stick_y = y;
            }
        }
        // Stick → discrete nav with a re-center latch: emit ONE Up/Down on crossing the
        // threshold, then nothing until the stick falls back near center — so a held stick
        // moves one item, not a blur of them. Stick up (+Y) is Up.
        const NAV_THRESH: f32 = 0.6;
        if stick_y.abs() < NAV_THRESH * 0.5 {
            *stick_latched = false;
        } else if !*stick_latched && stick_y.abs() >= NAV_THRESH {
            *stick_latched = true;
            if stick_y > 0.0 {
                up = true;
            } else {
                down = true;
            }
        }

        if up {
            out.push(MenuInput::Up);
        }
        if down {
            out.push(MenuInput::Down);
        }
        if confirm {
            out.push(MenuInput::Confirm);
        }
        if back {
            out.push(MenuInput::Back);
        }
        out
    }

    /// Execute one [`MenuAction`]. The ONE place menu actions become side effects — the match
    /// is exhaustive, so a new action variant can't be added without wiring it here (a dead
    /// button is a compile error). Returns `true` if it changed the AppPhase (entered Playing,
    /// or moved to/from the lobby), so the caller stops drawing a screen that's leaving this
    /// frame — keeping the drawn screen consistent with the FSM's new state.
    fn apply_action(
        action: MenuAction,
        state: &mut MenuState,
        pending: &mut PendingRound,
        next: &mut NextState<AppPhase>,
    ) -> bool {
        match action {
            MenuAction::None => false,
            MenuAction::Host => {
                start_forming(state, &StartChoice::Host, next);
                true
            }
            MenuAction::Join => {
                // Parse the optional code: blank = discover on the LAN (no dial); a non-empty
                // field must parse to an endpoint id or it's a user error we surface. On a bad
                // code, revert the FSM to the chooser (it already advanced to the lobby on
                // Confirm) so the screen and AppPhase stay consistent.
                let trimmed = state.code_input.trim();
                let host = if trimmed.is_empty() {
                    None
                } else {
                    match trimmed.parse::<EndpointId>() {
                        Ok(id) => Some(id),
                        Err(_) => {
                            state.error = Some("That join code isn't a valid endpoint id.".into());
                            // The FSM advanced to the lobby on Confirm; revert it so the
                            // chooser (with the error) keeps drawing, consistent with the
                            // still-Menu AppPhase.
                            state.nav = MenuNav::Chooser {
                                focus: ChooserItem::Join,
                            };
                            return false;
                        }
                    }
                };
                start_forming(state, &StartChoice::Join(host), next);
                true
            }
            MenuAction::StartNetworked => {
                // Peers present: command the barrier's synchronized GO. The formed networked
                // round arrives on a later poll, which then enters Playing.
                if let Some(f) = &state.forming {
                    f.request_start();
                }
                false
            }
            MenuAction::StartSolo => {
                // Host-alone Start: abandon the wait (cancel the barrier so its session tears
                // down) and install the shared solo round INSTANTLY — the SAME deterministic
                // round the Alone fallback produces. No discovery dependency.
                if let Some(f) = &state.forming {
                    f.cancel();
                }
                state.forming = None;
                pending.0 = Some(menu::solo_round(state.seed));
                next.set(AppPhase::Playing);
                true
            }
            MenuAction::Cancel => {
                // Tell the barrier to bail and tear its session down (no ~12s LAN phantom),
                // drop the handle, and return to the menu.
                if let Some(f) = &state.forming {
                    f.cancel();
                }
                state.forming = None;
                next.set(AppPhase::Menu);
                true
            }
        }
    }

    /// Poll the background barrier; returns `true` if it had finished and we transitioned
    /// (parked the round → Playing, or returned to Menu on cancel/error) this frame.
    fn poll_formation(
        state: &mut MenuState,
        pending: &mut PendingRound,
        next: &mut NextState<AppPhase>,
    ) -> bool {
        let Some(result) = state.forming.as_ref().and_then(|f| f.poll()) else {
            return false;
        };
        // Done forming: drop the handle and act on the result.
        state.forming = None;
        match result {
            // A round formed (networked, or the Alone fallback): install it and play.
            // `ready_from` is `None` only for Cancelled, which the barrier reports after
            // tearing its session down — return to the menu, no phantom left behind.
            Ok(match_result) => match menu::ready_from(match_result, state.seed) {
                Some(ready) => {
                    pending.0 = Some(ready);
                    next.set(AppPhase::Playing);
                }
                None => next.set(AppPhase::Menu),
            },
            Err(e) => {
                state.error = Some(format!("Couldn't form a match: {e:#}"));
                next.set(AppPhase::Menu);
            }
        }
        true
    }

    /// Open the host-triggered lobby for a Host/Join choice and move to Connecting. Shared by
    /// the Host and Join actions so the "begin lobby + clear error + switch phase" sequence
    /// has one definition.
    fn start_forming(state: &mut MenuState, choice: &StartChoice, next: &mut NextState<AppPhase>) {
        state.error = None;
        state.forming = Some(menu::begin(
            choice,
            state.seed,
            state.telemetry,
            state.weights_digest,
            state.asset_digest,
        ));
        next.set(AppPhase::Connecting);
    }

    /// Draw the Host / Join chooser (rl#58 — no separate Solo button; Host-alone IS solo).
    /// The focused item (from [`MenuNav`]) is highlighted via `selectable_label`; returns the
    /// item the player clicked, if any, for the caller to route through the FSM.
    fn draw_chooser(ctx: &egui::Context, state: &mut MenuState) -> Option<ChooserItem> {
        let focus = match state.nav {
            MenuNav::Chooser { focus } => focus,
            // Off-screen (shouldn't happen in the Menu phase); default to Host highlight.
            _ => ChooserItem::Host,
        };
        let mut clicked = None;
        egui::Window::new("Giant Crab Rescue")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.heading("Giant Crab Rescue");
                ui.label("Rescue the giant crab. Reach the green pillar to extract.");
                ui.separator();

                // Host: open a lobby. Play alone (Start with nobody joined = an instant solo
                // round) or wait for others to Join by our code / the LAN, then Start.
                if ui
                    .selectable_label(
                        focus == ChooserItem::Host,
                        "Host (play alone or with others)",
                    )
                    .clicked()
                {
                    clicked = Some(ChooserItem::Host);
                }

                ui.separator();
                ui.label("…or join someone on your LAN:");
                ui.horizontal(|ui| {
                    ui.label("Join code:");
                    ui.add(
                        egui::TextEdit::singleline(&mut state.code_input)
                            .desired_width(260.0)
                            .hint_text("paste host code (optional)"),
                    );
                });
                if ui
                    .selectable_label(focus == ChooserItem::Join, "Join a match")
                    .clicked()
                {
                    clicked = Some(ChooserItem::Join);
                }

                ui.separator();
                ui.label("Controller: D-pad / stick to move · A to select · B to go back.");
                ui.label("Keyboard: arrows / WASD · Enter to select · Esc to back.");

                if let Some(err) = &state.error {
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(230, 120, 120), err);
                }
            });
        clicked
    }

    /// Draw the lobby / connecting screen: the role, the join code (Host) or dial status
    /// (Join), the live roster, and the focusable Start (host) + Cancel. Returns the clicked
    /// item, if any. Polling already happened in `menu_screen`, so this only renders + reports.
    fn draw_lobby(
        ctx: &egui::Context,
        state: &MenuState,
        lobby: &[EndpointId],
    ) -> Option<LobbyItem> {
        let (hosting, focus) = match state.nav {
            MenuNav::HostLobby { focus } => (true, focus),
            MenuNav::JoinLobby => (false, LobbyItem::Cancel),
            // Off-screen default; the lobby only draws in Connecting where nav is a lobby
            // variant. Fall back to the formation's role so the frame still renders sanely.
            MenuNav::Chooser { .. } => (
                state.forming.as_ref().is_some_and(|f| f.hosting),
                LobbyItem::Cancel,
            ),
        };
        let display_code = state.forming.as_ref().and_then(|f| f.display_code());
        let mut clicked = None;
        egui::Window::new("Lobby")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if hosting {
                    ui.heading("Hosting a match");
                    ui.label("Share this join code (or others can find you on the LAN):");
                    // The host's own code is its endpoint id, surfaced once the session binds.
                    match display_code {
                        Some(code) => {
                            // A selectable, read-only field so the player can copy the code.
                            let mut code_str = code.to_string();
                            ui.add(egui::TextEdit::singleline(&mut code_str).desired_width(360.0));
                        }
                        None => {
                            ui.label("(starting host — code will appear shortly)");
                        }
                    }
                } else {
                    ui.heading("Joining a match…");
                    match display_code {
                        Some(code) => {
                            ui.label(format!("Dialing host {}…", code.fmt_short()));
                        }
                        None => {
                            ui.label("Discovering a host on the LAN…");
                        }
                    }
                }

                // Live roster: the players currently in the lobby (rl#58). Host alone shows
                // just itself, which is the cue that Start = a solo round. When hosting, the
                // host's own id is its join code (`display_code`), so mark it "you"; a joiner
                // doesn't know which id is its own here, so nothing is marked for it.
                ui.separator();
                let me = if hosting { display_code } else { None };
                lobby_roster(ui, lobby, me);

                ui.separator();
                if hosting {
                    // The host commands the start (rl#58). Alone → an instant solo round;
                    // with peers → the synchronized networked start. The label reflects which,
                    // read from the live roster so it's honest about what Start does.
                    let solo = lobby.len() <= 1;
                    let label = if solo {
                        "Start (solo — nobody has joined)"
                    } else {
                        "Start the match"
                    };
                    if ui
                        .selectable_label(focus == LobbyItem::Start, label)
                        .clicked()
                    {
                        clicked = Some(LobbyItem::Start);
                    }
                } else {
                    // A joiner can't start; it waits for the host's GO.
                    ui.spinner();
                    ui.label("Waiting for the host to start…");
                }
                if ui
                    .selectable_label(focus == LobbyItem::Cancel, "Cancel")
                    .clicked()
                {
                    clicked = Some(LobbyItem::Cancel);
                }

                ui.separator();
                ui.label("Controller: A to select · B to cancel. Keyboard: Enter · Esc.");
            });
        clicked
    }

    /// Draw the lobby's live player list (rl#58): one line per player, `me` (if given)
    /// marked. `roster` is the barrier's current `live_set` (sorted by id bytes), empty until
    /// the session binds.
    fn lobby_roster(ui: &mut egui::Ui, roster: &[EndpointId], me: Option<EndpointId>) {
        if roster.is_empty() {
            ui.label("Players: (connecting…)");
            return;
        }
        ui.label(format!("Players in the lobby: {}", roster.len()));
        for id in roster {
            let tag = if Some(*id) == me { "  (you)" } else { "" };
            ui.label(format!("  • {}{}", id.fmt_short(), tag));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::menu::ReadyMatch;
    use crate::net::net_loop;

    /// The boot menu's handoff into the round (rl#56), exercised headlessly (no window):
    /// park a chosen [`ReadyMatch`] in [`PendingRound`], request the Playing transition,
    /// and prove `OnEnter(Playing)`'s [`ensure_round_installed`] builds a live
    /// [`GameState`] from it — the determinism-critical link the menu depends on (the menu
    /// only selects a round; this is where it actually becomes the sim). Uses
    /// `MinimalPlugins` + the state plumbing only, so it needs no display/GPU and can run
    /// on the headless box. (The egui UI + 2-peer formation still need on-device testing;
    /// this pins the part that decides which sim the round runs.)
    #[test]
    fn menu_handoff_installs_the_chosen_round() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(bevy::state::app::StatesPlugin)
            .init_state::<AppPhase>()
            .init_non_send_resource::<PendingRound>()
            .add_systems(OnEnter(AppPhase::Playing), ensure_round_installed);

        // Park a solo round (the same one the Solo button / Alone fallback produce) and ask
        // to enter Playing, exactly as the menu does on a choice.
        let seed = 0x1234_5678;
        app.world_mut()
            .insert_non_send_resource(PendingRound(Some(ReadyMatch {
                lockstep: net_loop::solo_lockstep_for(seed),
                net: None,
            })));
        app.world_mut()
            .resource_mut::<NextState<AppPhase>>()
            .set(AppPhase::Playing);

        // One update applies the transition and runs OnEnter(Playing).
        app.update();

        assert_eq!(
            *app.world().resource::<State<AppPhase>>().get(),
            AppPhase::Playing,
            "the transition must have entered Playing"
        );
        let gs = app
            .world()
            .get_non_send_resource::<GameState>()
            .expect("ensure_round_installed must build GameState from the parked round");
        // The installed sim is the chosen one: a single local player (solo), seeded as asked.
        assert_eq!(gs.ls.me(), crate::net::sim::PlayerId(0), "solo player id 0");
        assert!(
            matches!(gs.input_source, InputSource::Solo),
            "a solo handoff installs the Solo input source"
        );
        // And the parked round was consumed (taken), not left to double-install.
        assert!(
            app.world()
                .get_non_send_resource::<PendingRound>()
                .is_some_and(|p| p.0.is_none()),
            "the chosen round must be taken out of PendingRound"
        );
    }

    /// The GCR fold's manual fixed-step pump ([`pump_fixed_steps`]) must reproduce, bit-for-bit,
    /// the physics Bevy's wall-clock auto-pump produces — the stepping `bot::determinism_probe`
    /// proves deterministic. Build two identical headless crab worlds; step one with `app.update()`
    /// (the auto-pump path) and the other with `pump_fixed_steps` after parking `Time<Fixed>` (the
    /// windowed driver's path); drive both with the SAME scripted torque and assert their full
    /// articulated crab digests agree every tick. If they do, the windowed crab inherits ALL of
    /// the probe's determinism guarantees; if `pump_fixed_steps` ever double-stepped, skipped a
    /// fixed sub-schedule, or fed the wrong clock, this diverges. (Render-only — it needs the real
    /// rapier+bot stack — but headless: no window/GPU.)
    #[test]
    fn manual_pump_matches_auto_pump_step_for_step() {
        use crate::bot::actuator::{ACTION_SIZE, CrabActions};
        use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabJoint};
        use crate::bot::physics_digest::crab_state_digest;
        use crate::bot::test_util::{HeadlessStack, WorldRole, headless_stack};
        use bevy_rapier3d::prelude::Velocity;

        let build = || {
            headless_stack(HeadlessStack {
                num_envs: 1,
                role: WorldRole::Standalone,
            })
        };
        let mut auto = build();
        let mut manual = build();
        // One update each: run Startup (spawns the crab + sizes CrabActions) + one physics step,
        // so both worlds start from the identical post-spawn state.
        auto.update();
        manual.update();
        // Park the manual world's wall-clock auto-pump, exactly as `add_external_nn_crab` does, so
        // from here ONLY `pump_fixed_steps` advances its physics.
        park_fixed_auto_pump(manual.world_mut());

        let digest = |app: &mut App| -> u64 {
            let mut q = app.world_mut().query_filtered::<(
                &Transform,
                &Velocity,
                Option<&CrabJoint>,
                Option<&CrabCarapace>,
            ), With<CrabBodyPart>>();
            crab_state_digest(q.iter(app.world()))
        };
        let set_torque = |app: &mut App, a: [f32; ACTION_SIZE]| {
            app.world_mut().resource_mut::<CrabActions>().envs[0] = a;
        };

        let mut lcg: u64 = 0x1234_5678_9abc_def0;
        for t in 0..120u32 {
            let mut act = [0.0f32; ACTION_SIZE];
            for slot in act.iter_mut() {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *slot = ((lcg >> 40) as u32 as f32 / (1u32 << 24) as f32) * 1.6 - 0.8;
            }
            set_torque(&mut auto, act);
            set_torque(&mut manual, act);
            auto.update();
            pump_fixed_steps(manual.world_mut(), 1);
            assert_eq!(
                digest(&mut auto),
                digest(&mut manual),
                "manual pump diverged from auto-pump at tick {t}"
            );
        }
    }

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

    /// A stick resting inside the deadzone contributes exactly zero on every axis — the
    /// guard that hardware idle-noise can't creep the avatar or drift the view. Tests the
    /// REAL client transform (`pad_stick_axes`, which `gather_input` calls), so a future
    /// edit that drops/weakens the deadzone fails here.
    #[test]
    fn pad_sub_deadzone_sticks_contribute_nothing() {
        let inside = PAD_STICK_DEADZONE * 0.9;
        let a = pad_stick_axes(Vec2::new(inside, 0.0), Vec2::new(0.0, inside), 1.0 / 60.0);
        assert_eq!(
            (a.strafe, a.forward),
            (0.0, 0.0),
            "sub-deadzone move is zero"
        );
        assert_eq!(
            (a.d_yaw, a.d_pitch),
            (0.0, 0.0),
            "sub-deadzone look is zero"
        );
    }

    /// Past the deadzone, the left stick passes its raw magnitude straight to the move
    /// axes (analog, not bang-bang) and the right stick's look scales with both deflection
    /// and dt — pinning the frame-rate-independent look and the analog move feel.
    #[test]
    fn pad_above_deadzone_passes_move_and_scales_look_by_dt() {
        let dt = 1.0 / 60.0;
        let a = pad_stick_axes(Vec2::new(0.8, -0.6), Vec2::new(1.0, 0.0), dt);
        assert_eq!(a.strafe, 0.8, "left stick X → strafe, analog");
        assert_eq!(a.forward, -0.6, "left stick Y → forward, analog");
        assert!(
            (a.d_yaw - PAD_LOOK_SPEED * dt).abs() < 1e-6,
            "full right-stick-X look = PAD_LOOK_SPEED·dt, got {}",
            a.d_yaw
        );
        // Double the dt → double the per-frame look, the frame-rate independence that
        // keeps turn speed consistent across machines (the i16 it quantizes to is each
        // peer's own broadcast input, so this stays lockstep-safe — see net::desync_test).
        let b = pad_stick_axes(Vec2::ZERO, Vec2::new(1.0, 0.0), dt * 2.0);
        assert!(
            (b.d_yaw - 2.0 * a.d_yaw).abs() < 1e-6,
            "look is linear in dt"
        );
    }

    /// `pad_stick_axes` does NOT pre-negate any axis: the screen-relative X-negation is
    /// applied once, downstream in `gather_input` (the `-strafe` / `yaw_delta -= d_yaw`
    /// at the funnel), to BOTH keyboard and pad together. A positive stick X yields a
    /// positive raw strafe/yaw here; if this fn negated too, the pad would invert. Pins
    /// that the single negation site stays single (no double-negate, no pad-only flip).
    #[test]
    fn pad_axes_are_not_pre_negated() {
        let a = pad_stick_axes(Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0), 1.0 / 60.0);
        assert!(
            a.strafe > 0.0,
            "+stick X → +raw strafe (negation is downstream)"
        );
        assert!(
            a.d_yaw > 0.0,
            "+stick X → +raw yaw (negation is downstream)"
        );
    }
}
