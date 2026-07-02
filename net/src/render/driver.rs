//! Lockstep driver: owns the sim + transport and is the sole writer of sim state.
//!
//! The render-agnostic core of the windowed client — [`GameState`] (the sim + its input
//! source), the per-frame [`drive_lockstep`] tick pump, the deterministic fixed-step crab
//! pump ([`pump_fixed_steps`]), and the menu->round handoff ([`ensure_round_installed`]).
//! Holds no rendering; the scene/HUD/input live in sibling submodules.

use super::app::{ArmedRound, ExternalCrabStackInstalled};
use super::input::{CameraPitch, CameraYaw};
use super::*;
use crab_world::vehicle::{Vehicle, VehicleControl, VehicleKind};

/// Build the round's [`Coordinator`]: `None` ⇒ a solo internal server (roster of one), a host
/// driver ⇒ a server over the roster, a client driver ⇒ a remote client. `peers` is the sim's
/// participant set (solo ⇒ just the local player); `initial_sim` is the tick-0 world the server
/// owns (a clone of the client's freshly-built sim — solo/host step it authoritatively; a remote
/// client ignores it). The single home of the round's role choice ([[sp-is-mp-special-case]]).
pub(super) fn coordinator(
    net: Option<NetDriver>,
    peers: &[PlayerId],
    me: PlayerId,
    initial_sim: crate::sim::Sim,
) -> Box<Coordinator> {
    Box::new(Coordinator::for_round(net, peers, me, initial_sim))
}

/// Shared setup for both apps: the sim + its coordinator, plus the input resources.
pub(super) fn insert_core(app: &mut App, ls: Lockstep, coord: Box<Coordinator>) {
    install_round(app.world_mut(), ls, coord);
}

/// Install the round resources into the world: the non-send [`GameState`] (sim + coordinator)
/// and the input resources. Factored out of [`insert_core`] so it can be called BOTH at app
/// build (the screenshot path) and from the menu's `OnEnter(Playing)` transition system,
/// which only has a [`World`], not an [`App`] — one definition so the round is set up identically
/// however it was reached.
fn install_round(world: &mut World, ls: Lockstep, coord: Box<Coordinator>) {
    let prev = SimSnapshot::capture(&ls);
    world.insert_non_send_resource(GameState {
        ls,
        coord,
        accumulator: 0.0,
        prev,
        reported_outcome: false,
        next_tel_tick: next_sample_tick(0),
        cadence: PhysicsCadence::default(),
    });
    // OVERWRITE with fresh defaults (not `init_resource`, which keeps an existing value):
    // this install is the ONE owner of per-round input/camera state, so a round entered
    // after a disconnect return (rl#203) can't inherit the previous round's accrued look,
    // latched buttons, or piloting state.
    world.insert_resource(PendingInput::default());
    world.insert_resource(FlightInput::default());
    world.insert_resource(CameraPitch::default());
    world.insert_resource(CameraYaw::default());
    world.insert_resource(LocalVehicle::default());
}

/// The round the boot menu chose, parked here between the menu's Playing transition and
/// the `OnEnter(Playing)` install. Holds the [`ArmedRound`] PROOF, not a bare round — only
/// the arm gate ([`super::app::arm_round`]) can construct one, so no path can park an
/// unarmable round for install (impossible-by-construction). Non-send because the
/// round holds a `NetDriver` (tokio runtime). `None` on the scripted `Boot::Round` path,
/// which installs `GameState` at app build instead — so [`ensure_round_installed`] no-ops
/// there.
#[derive(Default)]
pub(super) struct PendingRound(pub(super) Option<ArmedRound>);

/// At the Playing transition, make sure a [`GameState`] exists before the scene spawns.
/// Chained ahead of `spawn_world` (which reads the sim). Two cases, one place:
/// - **Scripted (`Boot::Round`)**: `GameState` was inserted at app build — nothing to do.
/// - **Menu**: take the parked [`PendingRound`] (set by the menu on its choice) and
///   [`install_round`] it now, so the sim is live for the spawns.
///
/// On the menu path this is ALSO where the one giant crab — the real NN body — is ARMED:
/// insert [`crate::external_crab::ExternalCrabArmed`] and seed the sim crab so the
/// rapier-NN body drives it. The arm DECISION ([`crate::may_arm_external_crab`]: solo always,
/// networked only with synced weights+assets) is made UPSTREAM in the menu's `poll_formation`,
/// which refuses an unarmable networked round and returns to the chooser with an actionable
/// peer-mismatch message BEFORE requesting Playing. The [`ArmedRound`] taken from
/// [`PendingRound`] is the PROOF of that decision — only the gate can mint one — so
/// arming here without re-checking is sound by type, not by trust. It must arm here rather
/// than gate, because the chained `spawn_world` needs the installed `GameState`; a graceful
/// in-menu refusal is only possible before the transition, which is why the decision lives in
/// the menu. The crab's arena spawn was already seeded into the bridge at build (a pure
/// function of the seed), so nothing about the spawn depends on the round here.
///
/// Idempotent (guards on `GameState` already present), so it can't double-install if both
/// a scripted round and a stray pending one ever coexisted. Reaching Playing with neither a
/// pre-installed `GameState` (scripted) nor a parked round (menu) is an unreachable
/// logic-bug state — every menu path parks a round BEFORE requesting the transition — so we
/// panic HERE with a precise message rather than no-op and let the chained `spawn_world`
/// (which needs `GameState`) panic one system later with a cryptic missing-resource error.
pub(super) fn ensure_round_installed(world: &mut World) {
    if world.get_non_send_resource::<GameState>().is_some() {
        return; // scripted path already installed the round at build time
    }
    let mut ready = world
        .get_non_send_resource_mut::<PendingRound>()
        .and_then(|mut p| p.0.take())
        .expect("entered Playing with no round to install — the menu must park a round before transitioning")
        .into_ready();
    // Arm the one giant crab — the real NN body. Taking the round out of an [`ArmedRound`]
    // IS the arm verdict: only `poll_formation`'s gate pass can have minted it, so
    // there is no re-check here — a path that could reach this transition with an unarmable
    // round cannot exist. The checkpoint is required, so the stack is always
    // installed; a missing stack is a build-wiring bug, so assert it loudly.
    assert!(
        world.get_resource::<ExternalCrabStackInstalled>().is_some(),
        "the NN-crab stack must be installed before Playing (rl#114: the checkpoint is required)"
    );
    let crab = ready.lockstep.sim().crab();
    // Seed the pose with the crab's current pose/yaw (writing back what's there → no state change).
    ready
        .lockstep
        .set_external_crab_pose(crab.pos(), crab.yaw(), 0);
    // Arm the gate (the crab now walks at the player's actual position — nothing per-peer to
    // reconcile). One arm path, [`crate::external_crab::arm`].
    crate::external_crab::arm(world);
    // A round AFTER a disconnect return (rl#203): the previous round's crab body persists
    // across the menu (the stack installs once at build), still WARM and wherever it walked to.
    // Rebuild it COLD at this round's spawn, exactly as the restart edge does; a first round
    // has no body yet and the cold respawn no-ops (the guarded initial spawn places it).
    // Gated on the bridge — the stack's real presence signal — because a headless test
    // harness exercises this install with the marker but no crab plugins.
    if world.contains_resource::<crate::external_crab::ExternalCrabBridge>() {
        restart_crab_to_spawn(world, crab.pos());
    }
    // Clone the freshly-seeded sim for the authoritative server (solo/host); the client keeps its
    // own identical sim inside `ready.lockstep` and renders the snapshots the server emits into it.
    let coord = coordinator(
        ready.net,
        ready.lockstep.peers(),
        ready.lockstep.me(),
        ready.lockstep.sim().clone(),
    );
    install_round(world, ready.lockstep, coord);
}

/// Tear a finished round back OUT of the world — the `OnExit(Playing)` mirror of
/// [`install_round`] + the OnEnter spawns, so re-entering Playing installs a FRESH round
/// (rl#203: the disconnect return to the menu). Round ENTITIES despawn via
/// `DespawnOnExit(AppPhase::Playing)` at their spawn sites, and the per-round input/camera
/// resources are overwritten fresh by the next [`install_round`]; this removes only what
/// neither covers:
/// - [`GameState`], which is load-bearing twice: dropping it drops the round's [`NetDriver`]
///   (session teardown, so the host departs us cleanly) and its absence is what lets
///   [`ensure_round_installed`] install the next round instead of resuming the dead one —
///   and, with it, every per-round driver latch/watermark it carries (rl#210);
/// - the crab ARM gate ([`crate::external_crab::ExternalCrabArmed`] — presence IS the state,
///   so removal is the whole disarm). The NN stack + crab body persist across rounds by
///   design (the menu boot installs them once at build); [`ensure_round_installed`]
///   cold-respawns the body at the next round's spawn;
/// - any live piloting command (the same zeroing as `drive_lockstep`'s OnFoot arm), so no
///   stale force rides into the next round's physics pump.
pub(super) fn teardown_round(world: &mut World) {
    world.remove_non_send_resource::<GameState>();
    world.remove_resource::<crate::external_crab::ExternalCrabArmed>();
    if let Some(mut ctrl) = world.get_resource_mut::<VehicleControl>() {
        ctrl.active = false;
        FlightControl::default().write_into(&mut ctrl);
    }
}

/// End the round because the server stopped serving this remote client (rl#203) — the ONE
/// consumer of [`Exchanged::server_down`]. Loud everywhere it can be: an error log, a telemetry
/// Fault, and then EITHER the menu return (a [`RoundOver`] for the "connection lost — rejoin?"
/// prompt, on a [`super::app::BootedWithMenu`] app) or a loud process exit (the scripted
/// `--join`/`net-join` boots, which have no menu to return to).
fn end_round_server_down(
    world: &mut World,
    down: crate::net_loop::ServerDown,
    tel: Option<&crate::telemetry::TelemetrySender>,
) {
    let message = down.to_string();
    error!("leaving the round — {message}");
    if let Some(t) = tel {
        t.send(TelemetryEvent::Fault {
            msg: format!("client server-down (rl#203): {message}"),
        });
    }
    if world.get_resource::<super::app::BootedWithMenu>().is_some() {
        let host = world
            .non_send_resource::<GameState>()
            .coord
            .server_endpoint()
            .expect(
                "a ServerDown only occurs on the client arm, which always has a server endpoint",
            );
        world.insert_resource(super::app::RoundOver { message, host });
        world
            .resource_mut::<NextState<AppPhase>>()
            .set(AppPhase::Menu);
    } else {
        world.write_message(AppExit::error());
    }
}

// ---------------------------------------------------------------------------
// Lockstep driver state (render-agnostic: owns the sim + transport)
// ---------------------------------------------------------------------------

/// A scripted stand-in input fed to every NON-local player each tick, present ONLY in the headless
/// screenshot harness (`fp-screenshot`) so the pack walks and the scene composes. The host-
/// authoritative server records it for each pack player exactly as it would a wire peer's input, so
/// the sim can't distinguish it — a bot input, not a back channel. Absent in real play (windowed /
/// networked), where every non-local input arrives over the transport. A single-machine solo run, so
/// no peer exists to disagree with it.
#[derive(Resource, Clone, Copy)]
pub(super) struct ScriptedPackInput(pub(super) Input);

/// This peer's fixed role for a round — the ONE axis every per-tick branch in [`drive_lockstep`]
/// dispatches on, so "which role" lives in one place and the impossible both-server-and-client
/// state a pair of bools would allow can't exist (make-illegal-states-unrepresentable). It's the
/// render-side mirror of the coordinator's own arms ([`Coordinator::Server`]/[`Coordinator::Client`]),
/// NOT an SP/MP split — solo and host are the same [`PeerRole::ServerAuth`] (the headless
/// `fp-screenshot` harness is a solo server too, so it takes the same arm).
#[derive(Clone, Copy, PartialEq, Eq)]
enum PeerRole {
    /// Solo or host: this peer runs the authoritative server, steps it, and RENDERS the snapshot it
    /// emits — plus, when hosting, broadcasts state DOWN to remote clients.
    ServerAuth,
    /// A remote client: ADOPTS the host's snapshot into its own sim
    /// (no re-sim) and poses its frozen crab from the articulation, pumping no physics of its own.
    RemoteAdopt,
}

impl PeerRole {
    fn of(state: &GameState) -> Self {
        if state.coord.is_remote_client() {
            PeerRole::RemoteAdopt
        } else {
            PeerRole::ServerAuth
        }
    }

    /// Whether THIS peer may board a vehicle — the server-authoritative arm ONLY (solo OR host).
    /// Piloting is an input MODE, not an SP fork ([[rl-vehicles-plane-mode-required]]), so a HOST
    /// pilots in a networked round too — the real MP win, since a host pumps its own
    /// crab-world physics. The craft is a local, off-wire rapier body spawned + force-driven by
    /// `VehiclePlugin` in `FixedUpdate`, which is advanced ONLY by [`pump_fixed_steps`] inside the
    /// tick-drain loop. A `RemoteAdopt` client renders the host's crab from articulation and pumps
    /// NO physics of its own (it `break`s out of that loop), so a craft would never spawn there — a
    /// toggle-does-nothing trap ([[silent-fallback-antipattern]]). Remote-client piloting therefore
    /// waits until the craft is pumped/synced on the adopt arm (a later increment); boarding is
    /// gated OFF it here rather than exposed as an inert mode. The scripted harness has no avatar.
    fn can_pilot(self) -> bool {
        matches!(self, PeerRole::ServerAuth)
    }
}

/// The networked sim, owned as a non-send Bevy resource and stepped on a
/// fixed-timestep accumulator. Non-send because [`NetDriver`] holds a tokio runtime
/// + the iroh session (not `Sync`); only the main thread drives it, so that's fine.
pub(super) struct GameState {
    pub(super) ls: Lockstep,
    /// This peer's per-tick input coordinator: solo internal server, host, or remote
    /// client — the single path through which every non-local input flows. The sole writer of
    /// inputs other than the local controls.
    pub(super) coord: Box<Coordinator>,
    /// Fractional-tick accumulator: render time elapsed since the last applied sim
    /// tick, in [0, TICK_DT). Drives both how many ticks to step and the render
    /// interpolation alpha.
    pub(super) accumulator: f64,
    /// Renderable sim state as of the PREVIOUS applied tick. Render tweens from this
    /// toward the live sim by `alpha`. A snapshot (not the live sim) because we need
    /// "last tick" even after the sim has stepped to the current one.
    pub(super) prev: SimSnapshot,
    /// Round-decided latch: set when this round's decided outcome has been reported, cleared
    /// per Ongoing snapshot inside [`drive_lockstep`]'s drain arms (a RESTART revives a decided
    /// round without rewinding the tick, rl#204). Lives here — not a system `Local` — so a new
    /// round starts unlatched by construction (rl#210).
    reported_outcome: bool,
    /// The next telemetry sampling boundary ([`next_sample_tick`]), in this ROUND's ticks.
    /// Per-round for the same reason: a fresh Lockstep restarts at tick 0, so a surviving
    /// watermark would suppress tick telemetry until the counter climbed past it (rl#210).
    next_tel_tick: u64,
    /// The deterministic 64:30 physics/sim cadence, advanced once per APPLIED tick while
    /// armed and reset on the restart edge (reported by `step_next`, rl#204) so two peers
    /// stay phase-aligned.
    cadence: PhysicsCadence,
}

impl GameState {
    /// The authoritative server this peer runs, if any (solo or host) — `None` for a remote client.
    /// When `Some`, the local client renders the snapshots it emits rather than stepping `ls`'s sim
    /// itself.
    fn server(&self) -> Option<&crate::server::Server> {
        self.coord.server()
    }

    /// Mutable counterpart of [`GameState::server`], for the per-tick authoritative step.
    fn server_mut(&mut self) -> Option<&mut crate::server::Server> {
        self.coord.server_mut()
    }
}

/// A minimal copy of the renderable sim state at one tick — the poses the client
/// tweens from. NOT a second source of truth: overwritten every tick from the
/// authoritative [`CoreSnapshot`](crate::snapshot::CoreSnapshot) (via
/// [`capture`](SimSnapshot::capture)), never fed back into it.
#[derive(Clone, Default)]
pub(super) struct SimSnapshot {
    pub(super) players: BTreeMap<PlayerId, Player>,
    pub(super) crab: Option<Crab>,
}

impl SimSnapshot {
    /// Capture the renderable game state through the host-authoritative snapshot seam:
    /// the local client reads its interpolation source from the
    /// SAME serialized [`CoreSnapshot`](crate::snapshot::CoreSnapshot) a wire client will,
    /// via [`Lockstep::core_snapshot`], so SP and MP share one state-read path
    /// ([[sp-is-mp-special-case]]). Byte-identical to reading the sim directly.
    fn capture(ls: &Lockstep) -> Self {
        let snap = ls.core_snapshot();
        Self {
            players: snap.players,
            crab: Some(snap.crab),
        }
    }
}

/// The local player's input accumulated this render interval, applied to the sim at
/// the next tick boundary. Move axes are sampled each frame (last frame wins); the
/// yaw-look delta ACCUMULATES across the render frames between two ticks so no mouse
/// motion is dropped, and `action` latches if pressed any frame in the interval.
/// All drained when a tick consumes it.
#[derive(Resource, Default)]
pub(super) struct PendingInput {
    pub(super) strafe: f32,
    pub(super) forward: f32,
    /// Accrued yaw-look this inter-tick interval, in radians (drained per tick), integrated
    /// into the on-foot avatar's yaw. The in-air controls read the sticks/mouse directly
    /// ([`FlightInput`]), not this foot accumulator, so no field does double duty.
    pub(super) yaw_delta: f32,
    pub(super) action: bool,
    /// Latches if RESTART (R) was pressed in this interval. Sent as
    /// [`buttons::RESTART`] so the restart rides the input stream to the authoritative
    /// server (applied at the tick that consumes it) and every client adopts the reset
    /// via the snapshot; the sim edge-triggers it. Drained per tick.
    pub(super) restart: bool,
    /// Latches if the enter/exit-vehicle key (E) was tapped this interval — a client-local
    /// toggle drained ONCE per frame in [`drive_lockstep`] on the server-authoritative arm
    /// (solo or host — see `can_pilot`), never sent to the sim. Board a plane on foot / step
    /// out when piloting.
    pub(super) toggle_vehicle: bool,
}

/// RAW per-frame flight inputs for the CLIENT-LOCAL vehicle, sampled straight off the sticks/
/// triggers/bumpers in [`super::input::gather_input`] — NOT through the sim's merged move/look axes.
/// The plane (Ace Combat 6) maps the LEFT stick to attitude while the keyboard maps W/S to throttle;
/// those are the same merged axis to the sim, so the vehicle bridge reads pad and keyboard SEPARATELY
/// here to keep each craft's scheme intact. [`flight_control`] turns this into the per-craft
/// [`VehicleControl`] intents; nothing here touches the deterministic sim (the vehicle is
/// host-authoritative crab-world state off the wire).
#[derive(Resource, Default)]
pub(super) struct FlightInput {
    /// Pad left stick, deadzoned: x = +right, y = +up/forward.
    pub(super) left: Vec2,
    /// Pad right stick, deadzoned: x = +right, y = +up.
    pub(super) right: Vec2,
    /// This frame's mouse-look intent (already sensitivity-scaled): x = +right, y = +down.
    pub(super) mouse: Vec2,
    /// Keyboard move keys as a digital axis: x = D − A, y = W − S.
    pub(super) wasd: Vec2,
    /// Analog triggers, 0..1.
    pub(super) rt: f32,
    pub(super) lt: f32,
    /// Bumpers held.
    pub(super) lb: bool,
    pub(super) rb: bool,
    /// Match-velocity held (ship): pad A / keyboard Space.
    pub(super) match_vel: bool,
}

/// How much a banked turn auto-coordinates with yaw: rolling the plane also noses it into the turn a
/// little, so L/R reads as a turn, not just a barrel-roll. A clean seam for a future Expert toggle
/// (which would set this to 0 — pure roll on the stick).
const PLANE_TURN_COORDINATION: f32 = 0.3;

/// The ONE attitude/aim sensitivity knob (owner: vehicles were "too sensitive" on the controller).
/// Scales the ANALOG ROTATION sticks — the plane's left-stick pitch/roll and the ship's right-stick
/// pitch/yaw — before they hit the force model, so full deflection commands a fraction of the
/// available control torque instead of slamming it. 1.0 = raw (old, twitchy); <1 = gentler. Only the
/// analog sticks scale: the digital bumpers/keys are bang-bang and the translational thrusters keep
/// full authority, so this is purely the "how fast does the craft rotate per stick" feel knob the
/// owner trims. A single constant (referenced in both craft) so the two can't drift apart.
pub(super) const VEHICLE_STICK_SENS: f32 = 0.5;

/// The per-craft [`VehicleControl`] intents a set of raw [`FlightInput`]s produces. Pure (no World)
/// so a test can pin the directions — the plane's AC6 pitch (pull back = nose up) vs the ship's
/// camera-style aim, both craft's screen-reconciled horizontal, the ship's 6-DOF thrust axes, and the
/// [`VEHICLE_STICK_SENS`] scaling — without spinning a Bevy app.
#[derive(Debug, Default, PartialEq)]
pub(super) struct FlightControl {
    pub throttle_trim: f32,
    pub thrust: Vec3,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub match_velocity: bool,
}

/// Map raw flight inputs to the shared force-model intents, per craft:
/// - **Plane (AC6)**: left stick (or mouse) flies — pitch (AC6 flight-sim: pull back = nose UP) + roll
///   (negated for the body-pose/screen reconciliation, like the ship), with a coordinating yaw; RT/LT
///   (or W/S) trim the throttle lever; bumpers (or A/D) are the rudder. No direct thrusters (the plane
///   thrusts through its lever). Right stick is the camera (unused here).
/// - **Ship (Outer Wilds)**: left stick (or WASD) fires the body-frame thrusters (strafe/forward),
///   RT/LT thrust up/down; right stick (or mouse) AIMS (pitch camera-style/not-inverted, yaw negated
///   with strafe so stick-right reads screen-right); bumpers roll; A/Space matches velocity. No
///   throttle lever.
pub(super) fn flight_control(kind: VehicleKind, fi: &FlightInput) -> FlightControl {
    let clamp = |x: f32| x.clamp(-1.0, 1.0);
    match kind {
        VehicleKind::Plane => {
            // Pitch: the AC6 / flight-sim convention the owner asked for ("same plane control mapping
            // as Ace Combat 6", "actual flight simulator like controls") — pull the stick DOWN/back
            // (left.y < 0), or the mouse back (down, mouse.y > 0), to raise the nose. This DELIBERATELY
            // runs pitch (stick AND mouse) opposite the ship's camera-style aim and the on-foot look —
            // a plane pulls back to climb — so don't "unify" it back; stick and mouse agree with each
            // OTHER here, which is the property that matters. Analog stick scaled by VEHICLE_STICK_SENS
            // (the "too sensitive" fix); the mouse keeps FLIGHT_MOUSE_SENS.
            let pitch = clamp(-fi.left.y * VEHICLE_STICK_SENS + fi.mouse.y);
            // Roll: stick/mouse RIGHT banks right. NEGATED for the SAME body-pose/screen reconciliation
            // the ship strafe/yaw and the foot controls apply (see `gather_input` + the Ship arm): the
            // cockpit camera flies from the body pose looking along body +Z, so body +X renders
            // SCREEN-LEFT — without the negation, stick-right banked left. The coordinating yaw below
            // rides this reconciled roll, so a right bank turns the nose screen-right too.
            let roll = clamp(-(fi.left.x * VEHICLE_STICK_SENS + fi.mouse.x));
            // Rudder (bumpers / A,D) plus the coordinating yaw that turns a bank into a turn.
            // NEGATED at the source (RB/D command rudder-RIGHT, yet +yaw noses the plane toward body
            // +X, which renders SCREEN-LEFT) — the yaw torque has no reconciling negation of its own
            // (crab-world/vehicle.rs), so RB-right needs −yaw to swing the nose screen-right. Negate the
            // rudder input, NOT the torque, so the coordinating bank term below keeps its (correct) sign.
            let rudder = (fi.lb as i32 - fi.rb as i32) as f32 - fi.wasd.x;
            let yaw = clamp(rudder + PLANE_TURN_COORDINATION * roll);
            // Throttle lever trim: RT up / LT down (analog), or W/S on the keyboard.
            let throttle_trim = clamp(fi.rt - fi.lt + fi.wasd.y);
            FlightControl {
                throttle_trim,
                thrust: Vec3::ZERO,
                pitch,
                roll,
                yaw,
                match_velocity: false,
            }
        }
        VehicleKind::Ship => {
            // Direct body-frame thrusters: left stick / WASD = strafe (x) + forward (z); RT/LT =
            // vertical (y, up/down). Coast on momentum between taps. Strafe is NEGATED for the same
            // reason the foot controls negate it (see `gather_input`): a cockpit looking along body
            // +Z sees body +X on the SCREEN-LEFT, so stick-right must command −X to strafe right.
            let thrust = Vec3::new(
                clamp(-(fi.left.x + fi.wasd.x)),
                clamp(fi.rt - fi.lt),
                clamp(fi.left.y + fi.wasd.y),
            );
            // Aim with the right stick / mouse — camera-style, NOT inverted in pitch (push up = nose
            // up). Yaw is NEGATED (like the foot yaw-look and the strafe above) so stick/mouse-right
            // turns the view RIGHT: +yaw noses toward +X, which renders screen-left at this facing.
            // The analog stick is scaled by VEHICLE_STICK_SENS (the same "too sensitive" knob as the
            // plane); the mouse keeps its own FLIGHT_MOUSE_SENS.
            let pitch = clamp(fi.right.y * VEHICLE_STICK_SENS - fi.mouse.y);
            let yaw = clamp(-(fi.right.x * VEHICLE_STICK_SENS + fi.mouse.x));
            // Roll on the bumpers: LB banks right, RB banks left (owner playtest — the ship's
            // bumper twist read reversed; the plane rudder above uses the same [LB+, RB−] sense).
            let roll = clamp((fi.lb as i32 - fi.rb as i32) as f32);
            FlightControl {
                throttle_trim: 0.0,
                thrust,
                pitch,
                roll,
                yaw,
                match_velocity: fi.match_vel,
            }
        }
    }
}

/// This tick's LOCAL control intent, TAGGED by mode so the on-foot avatar's control axes and a
/// craft's flight axes can never both be live: the mode picks exactly one variant,
/// so a walker holding a throttle (or a pilot holding foot move-axes) is unrepresentable by
/// construction.
///
/// Client-local: the deterministic sim only ever applies the FOOT [`Input`] (neutral while
/// piloting — see [`LocalControl::sim_input`]), and the craft is host-authoritative crab-world
/// state OFF the wire. Networking the pilots (a wire that carries flight input) is rl#43 part 2;
/// until then the tag lives here at the client, where the piloting state actually is. Produced
/// once per tick in [`drive_lockstep`] from [`LocalVehicle`].
enum LocalControl {
    /// Walking: this foot [`Input`] drives the sim; no craft is active.
    OnFoot(Input),
    /// Piloting `kind`: the sim gets a NEUTRAL foot input and this per-craft [`FlightControl`]
    /// commands the rapier vehicle body via [`VehicleControl`].
    Piloting {
        kind: VehicleKind,
        control: FlightControl,
    },
}

impl LocalControl {
    /// The foot input the deterministic sim applies this tick: the real walker input on foot, a
    /// neutral (default) input while piloting — the pilot's foot avatar just stands at the boarding
    /// spot while the craft flies off the wire.
    fn sim_input(&self) -> Input {
        match self {
            LocalControl::OnFoot(input) => *input,
            LocalControl::Piloting { .. } => Input::default(),
        }
    }
}

impl FlightControl {
    /// Write these per-craft intents into the shared [`VehicleControl`] the force model reads.
    /// `active`/`kind` are the caller's (they drive spawn/despawn); this copies only the force axes,
    /// so on foot the caller writes `FlightControl::default()` (all zero) to leave no stale command.
    /// The exhaustive destructure (no `..`) makes a new `FlightControl` axis a COMPILE error here
    /// rather than a silently-dropped command.
    fn write_into(&self, ctrl: &mut VehicleControl) {
        let FlightControl {
            throttle_trim,
            thrust,
            pitch,
            roll,
            yaw,
            match_velocity,
        } = *self;
        ctrl.throttle_trim = throttle_trim;
        ctrl.thrust = thrust;
        ctrl.pitch = pitch;
        ctrl.roll = roll;
        ctrl.yaw = yaw;
        ctrl.match_velocity = match_velocity;
    }
}

/// A flyer's first-person cockpit pose in the crab's ARENA frame: a 3D position + a full attitude
/// quaternion (body→world). Read off the rapier vehicle body each applied tick; the renderer maps
/// it into render space (shifted to the crab's render spot, [`super::scene`]'s `cockpit_camera`)
/// and flies the one cockpit camera from it — so the plane and the ship share one camera
/// formula with no copy to drift.
#[derive(Clone, Copy)]
pub(super) struct CockpitPose {
    pub pos: Vec3,
    pub orient: Quat,
}

/// The local player's SINGLE-PLAYER vehicle, when piloting one. `OnFoot` = not in a vehicle.
///
/// The vehicle itself is a rapier rigidbody living in the crab's ONE physics world
/// ([`crab_world::vehicle`]) — official, host-authoritative game state that collides with Sally.
/// This resource is the net client's MIRROR of that body: it tracks "am I piloting and which
/// craft" (driving spawn/despawn via [`VehicleControl`]) and holds the body's last two arena poses
/// for the cockpit camera's interpolation. While piloting, the local foot player feeds the sim a
/// NEUTRAL input (it just stands at the boarding spot) and the camera flies from the vehicle;
/// stepping out returns the view to the foot player. The E/X control CYCLES foot → plane → ship →
/// foot.
///
/// `pose` is `(prev, now)` — the last two applied ticks' arena poses, so [`apply_transforms`] tweens
/// the cockpit camera the same way it interpolates every sim body. It is `None` from boarding until
/// the rapier body first reports a pose (the body spawns a tick or two after `VehicleControl` goes
/// active), so a fabricated seed pose is unrepresentable — the camera holds the foot view for those
/// frames rather than snapping in from a fake origin. Only ever piloting on a windowed SOLO round.
#[derive(Resource, Default)]
pub(super) enum LocalVehicle {
    #[default]
    OnFoot,
    Flying {
        kind: VehicleKind,
        pose: Option<(CockpitPose, CockpitPose)>,
    },
}

impl LocalVehicle {
    /// The vehicle kind currently piloted, or `None` on foot — the net→crab-world command that
    /// spawns the matching rapier body via [`VehicleControl`].
    pub(super) fn kind(&self) -> Option<VehicleKind> {
        match self {
            Self::OnFoot => None,
            Self::Flying { kind, .. } => Some(*kind),
        }
    }

    /// The controls CONTEXT this vehicle state presents — the single mapping from "what am I
    /// driving" to "which control set + legend is live". The overlay reads it via
    /// [`ActiveContext`]; the HUD names + labels it automatically.
    pub(super) fn context(&self) -> GcrContext {
        match self {
            Self::OnFoot => GcrContext::OnFoot,
            Self::Flying {
                kind: VehicleKind::Plane,
                ..
            } => GcrContext::Plane,
            Self::Flying {
                kind: VehicleKind::Ship,
                ..
            } => GcrContext::Ship,
        }
    }

    /// This vehicle's `(prev, now)` cockpit poses for the FP camera, or `None` on foot OR before the
    /// body's first pose has been read (boarding) — the renderer falls back to the foot view then.
    pub(super) fn cockpit_poses(&self) -> Option<(CockpitPose, CockpitPose)> {
        match self {
            Self::OnFoot => None,
            Self::Flying { pose, .. } => *pose,
        }
    }

    /// Refresh the mirrored pose from the rapier body's freshly-stepped arena Transform: shift `now`
    /// into `prev` and record the new pose, so the cockpit camera interpolates this tick's motion.
    /// The FIRST pose seeds both `prev` and `now` to it (no interpolation from a fabricated origin —
    /// the boarding-frame zoom glitch). No-op on foot.
    fn update_pose(&mut self, p: CockpitPose) {
        if let Self::Flying { pose, .. } = self {
            *pose = Some(match *pose {
                Some((_, now)) => (now, p),
                None => (p, p),
            });
        }
    }

    /// The NEXT vehicle in the enter/exit cycle (foot → plane → ship → foot). One place the cycle
    /// order lives, so the input toggle and any future caller can't disagree on it. A boarded craft
    /// starts with no pose (`None`); [`update_pose`] fills it from the spawned body.
    fn cycled(&self) -> Self {
        match self {
            Self::OnFoot => Self::Flying {
                kind: VehicleKind::Plane,
                pose: None,
            },
            Self::Flying {
                kind: VehicleKind::Plane,
                ..
            } => Self::Flying {
                kind: VehicleKind::Ship,
                pose: None,
            },
            Self::Flying {
                kind: VehicleKind::Ship,
                ..
            } => Self::OnFoot,
        }
    }
}

/// Read the single vehicle rigidbody's arena-frame pose (position + attitude) from the crab world,
/// or `None` if none is spawned (on foot, or the frame a freshly-boarded body hasn't appeared yet).
/// One body at most — `manage_vehicle` enforces it; we take the first. The renderer shifts this
/// arena pose to the crab's render spot in `cockpit_camera`.
fn read_vehicle_pose(world: &mut World) -> Option<CockpitPose> {
    let mut q = world.query_filtered::<&Transform, With<Vehicle>>();
    q.iter(world).next().map(|t| CockpitPose {
        pos: t.translation,
        orient: t.rotation,
    })
}

// ---------------------------------------------------------------------------
// Lockstep driver system
// ---------------------------------------------------------------------------

/// Advance the crab's rapier physics + brain by exactly `steps` fixed steps, NOW, from the
/// lockstep driver — the deterministic replacement for Bevy's wall-clock `FixedUpdate`
/// auto-pump (disabled in [`add_external_nn_crab`] by parking `Time<Fixed>` at a huge timestep).
/// Each `run_schedule(FixedMain)` is one [`crab_world::physics::PHYSICS_DT`] step (rapier's own
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
/// ([`add_external_nn_crab`]) and the headless probe
/// ([`crate::external_crab::run_headless_probe`]) can't drift on it.
pub(crate) fn park_fixed_auto_pump(world: &mut World) {
    world
        .resource_mut::<bevy::time::Time<bevy::time::Fixed>>()
        .set_timestep(std::time::Duration::from_secs(86_400));
}

/// Re-seed the crab bridge to the round's rebuilt `spawn` and cold-respawn the rapier body — run at
/// the exact RESTART edge (as reported by `Server::step_next`) in [`drive_lockstep`]'s
/// server-authoritative drain, the one arm that pumps a crab body. Re-seeding the bridge so
/// the next pose push is the spawn pose (not the still-walking body's accumulated position) is half
/// of it; the other half is the cold respawn — re-seeding alone leaves the rapier solver WARM, which
/// would desync a mid-game joiner's cold body against an incumbent's warm one.
/// Dropping + rebuilding the body makes every peer's solver state identically fresh off
/// the same shared-input restart edge, covering the plain RESTART button too (one shared edge).
/// `spawn` is read from whichever sim is authoritative for the round. Only meaningful while armed
/// (else no bridge drives the crab).
fn restart_crab_to_spawn(world: &mut World, spawn: crate::sim::Pos) {
    world
        .resource_mut::<crate::external_crab::ExternalCrabBridge>()
        .restart_to_spawn(spawn);
    crate::external_crab::cold_respawn_armed_crab(world);
}

/// Advance the game by real time on a fixed-timestep accumulator. This is the ONLY writer of
/// sim state, and (apart from the external crab pose) it writes exactly one thing: the local
/// [`Input`] (drained from [`PendingInput`]) via `submit_local_input`, shipped UP through the
/// [`Coordinator`]. On the server-authoritative arm the internal server then steps its own sim
/// and this peer's client adopts + broadcasts the emitted snapshot; a remote-adopt client
/// instead adopts the host's snapshots and steps nothing.
///
/// When the external NN crab is armed ([`ExternalCrabArmed`] — solo OR a
/// networked round with synced weights, [`crate::may_arm_external_crab`]), the rapier
/// crab body is stepped INSIDE the tick drain on the server-authoritative arm: per applied
/// tick we run the deterministic [`PhysicsCadence`] number of physics steps
/// ([`pump_fixed_steps`]) and hand the body's resulting pose + weights-folded digest to
/// [`Server::step_next`] as that tick's [`crate::server::CrabPose`]. One tick at a time, so
/// each applied tick gets its own physics batch + pose. This is an EXCLUSIVE system because
/// pumping the fixed schedule needs `&mut World`.
///
/// Takes NO system `Local`s: the per-round driver state (round-decided latch, telemetry
/// watermark, physics cadence) lives in [`GameState`], whose install/teardown IS the round
/// lifetime — a `Local` outlives the round across a Playing→Menu→Playing cycle (rl#210).
pub(super) fn drive_lockstep(world: &mut World) {
    // Whether the external NN crab drives the sim this round (solo, or networked + synced
    // weights). Read once — the gate is fixed for the round. A real round is always armed
    // (a round that can't arm Sally is refused at build); when off (e.g. behind the boot menu)
    // no physics is pumped and the crab holds its spawn.
    let armed = world
        .get_resource::<crate::external_crab::ExternalCrabArmed>()
        .is_some();

    // THIS peer's role for the round — fixed once, one source of truth for every branch below, so
    // "which role" can't drift or reach the impossible both-server-and-adopt state a pair of bools
    // would allow (make-illegal-states-unrepresentable). It mirrors the coordinator's own arms.
    let role = PeerRole::of(world.non_send_resource::<GameState>());

    let delta = world.resource::<Time>().delta().as_secs_f64();

    // Clone out the telemetry handle (cheap: an mpsc sender + id) + the roster size so we can
    // READ the sim and push events without holding a borrow of the `NetDriver`. `None` unless
    // this is networked play with a collector. Telemetry never writes the sim.
    let (tel, roster_len) = {
        let state = world.non_send_resource::<GameState>();
        let tel = state.coord.telemetry().cloned();
        // Roster size from the SIM's player set — the roster of record rides every adopted
        // `CoreSnapshot`, so this stays correct on the host, an incumbent, AND a mid-game joiner
        // after a live join. (`ls.peers()` is the round-START set and would go stale on the
        // host/incumbents when a joiner enters.)
        (tel, state.ls.sim().players().count())
    };

    world.non_send_resource_mut::<GameState>().accumulator += delta;

    // Enter/exit a vehicle. Drain the E-tap latch ONCE per frame and CYCLE foot → plane → ship →
    // foot. The actual craft is a rapier body spawned/despawned in the crab world from the resulting
    // `LocalVehicle` (mirrored into `VehicleControl` below). Available on the server-authoritative
    // arm — solo AND host (`can_pilot`): piloting is an input mode, not an SP fork,
    // so a host pilots in a networked round too. While piloting, the
    // foot player files NEUTRAL sim input (below), so the authoritative sim just parks the avatar at
    // the boarding spot; the craft itself is local, off-wire crab-world state pumped by this peer's
    // own `pump_fixed_steps`. A remote-adopt client pumps no physics, so it cannot spawn a craft yet
    // (see `can_pilot`); the scripted harness has no live avatar.
    {
        let toggle = std::mem::take(&mut world.resource_mut::<PendingInput>().toggle_vehicle);
        if toggle && role.can_pilot() {
            // Board from foot ONLY while the foot avatar is alive; a vehicle→vehicle switch and
            // stepping out need no foot. (Downed/extracted: can't board.)
            let alive = {
                let state = world.non_send_resource::<GameState>();
                let me = state.ls.me();
                state
                    .ls
                    .sim()
                    .player(me)
                    .is_some_and(|p| p.status() == PlayerStatus::Alive)
            };
            let mut vehicle = world.resource_mut::<LocalVehicle>();
            let boarding_from_foot = matches!(*vehicle, LocalVehicle::OnFoot);
            if !boarding_from_foot || alive {
                *vehicle = vehicle.cycled();
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

        // Build THIS tick's FOOT input from the accumulated controls, draining the accrued look +
        // latched buttons (movement axes are re-sampled next frame). Drained EVERY tick, even while
        // piloting, so accrued mouse-look and a latched Extract can't pile up and fire on stepping
        // out of the craft.
        let foot_input = {
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

        // Tag the control by the active mode: on foot the foot input drives the sim and no craft is
        // live; piloting, this frame's per-craft flight control commands the vehicle and the sim gets
        // a neutral foot input. Exactly one arm — a walker holding a throttle (or a pilot holding foot
        // move-axes) is unrepresentable. The `FlightInput` snapshot is per-frame, so the
        // intent is stable across the ticks pumped this frame.
        let local = match world.resource::<LocalVehicle>().kind() {
            None => LocalControl::OnFoot(foot_input),
            Some(kind) => LocalControl::Piloting {
                kind,
                control: flight_control(kind, world.resource::<FlightInput>()),
            },
        };
        let sim_input = local.sim_input();

        // The headless `fp-screenshot` harness (only) feeds a scripted input to the absent pack
        // players so the scene composes; real play has none (every non-local input is on the wire).
        let scripted_pack: Option<Input> = world.get_resource::<ScriptedPackInput>().map(|r| r.0);
        // Submit our input UP to the coordinator and, on a remote client, drain the host's state DOWN.
        // The newest crab pose a remote client drained this iteration is applied after the exchange's
        // `GameState` borrow is released — as is a server-down verdict (it ends the round).
        let mut pending_art: Option<crate::articulation::CrabArticulation> = None;
        let mut server_down: Option<crate::net_loop::ServerDown> = None;
        let issue_tick = {
            let mut state = world.non_send_resource_mut::<GameState>();
            // Plain `&mut GameState` (not `Mut`) so the adopt closure below can borrow the
            // `reported_outcome` field disjointly from the `ls` it runs against.
            let state = &mut *state;
            let me = state.ls.me();
            let msg = state.ls.submit_local_input(sim_input);
            // The tick this input was ISSUED as — the telemetry stamp. On a remote client this
            // differs from `ls.next_tick()` (the snapshot-apply cursor) by the transit lag.
            let issue_tick = msg.issue_tick;
            // Scripted screenshot: file the pack's bot input into the solo server for every non-local
            // rostered player at this tick, so the scene composes with the pack moving in step
            // (real play drains these off the transport instead). A no-op with no `ScriptedPackInput`.
            if let Some(bot) = scripted_pack
                && let Some(server) = state.coord.server_mut()
            {
                let others: Vec<PlayerId> = server
                    .roster()
                    .iter()
                    .copied()
                    .filter(|&p| p != me)
                    .collect();
                for pid in others {
                    server.record_remote(
                        pid,
                        TickMsg {
                            issue_tick: msg.issue_tick,
                            input: bot,
                        },
                    );
                }
            }
            // Ship our input to the (internal or remote) server. Solo runs the same exchange against a
            // roster of one — the sim advances on our own filed input alone. An Err is the
            // remote client's round-terminal server-down verdict (rl#203), handled below once
            // this borrow is released; the empty default keeps this tick inert meanwhile.
            let exch: Exchanged = match state.coord.exchange(msg) {
                Ok(exch) => exch,
                Err(down) => {
                    server_down = Some(down);
                    Exchanged::default()
                }
            };
            // Two roles diverge here:
            // - SERVER-AUTHORITATIVE (solo/host/fp-screenshot): the server already stepped these
            //   inputs into its OWN sim (assembled host-paced by `exchange`) and the local client
            //   renders the snapshot it emits below — nothing to do here.
            // - REMOTE ADOPT: the client ADOPTS the host's snapshot
            //   into its own sim (no re-sim) and stashes the newest articulation for the render apply
            //   after this borrow is released.
            match role {
                // Handled by the server step + snapshot render below.
                PeerRole::ServerAuth => {}
                PeerRole::RemoteAdopt => {
                    // Adopt the host's drained snapshots via the ONE shared client adopt policy
                    // ([`Lockstep::adopt_snapshots`]: arrival order, no tick gate — see its doc for
                    // the restart-freeze rationale).
                    if !exch.snapshots.is_empty() {
                        // Refresh the interpolation source from the PRE-adopt state, exactly as the
                        // stepping arm does before advancing — else `apply_transforms` would tween
                        // avatars + the FP camera from the tick-0 spawn every frame (a per-frame snap
                        // toward spawn), since this arm skips the drain loop that owns `prev`.
                        state.prev = SimSnapshot::capture(&state.ls);
                        // Clear the round-decided latch on every Ongoing snapshot IN the batch,
                        // not off the frame-end outcome: a batch spanning a restart AND a fresh
                        // decision (a catch-up after a stall) never shows Ongoing at the frame
                        // boundary, which would swallow the new round's report.
                        let reported_outcome = &mut state.reported_outcome;
                        state.ls.adopt_snapshots(exch.snapshots, |c| {
                            if c.sim().outcome() == Outcome::Ongoing {
                                *reported_outcome = false;
                            }
                        });
                        // Local-player prediction: the snapshot re-seated our own
                        // avatar to its round-trip-old authoritative position; replay our still-in-
                        // flight inputs on it so WASD feels responsive at input latency, not RTT.
                        // Remote players + the crab stay authoritative (pose from the host, tweened
                        // via `prev`), never predicted. Overwritten by the next snapshot each frame.
                        state.ls.reconcile_local_prediction();
                    }
                    // Newest crab pose (last-arrived, same reliable-stream reasoning) — applied to
                    // the World once `state` is released.
                    pending_art = exch.articulations.into_iter().next_back();
                }
            }
            issue_tick
        };
        // Render the host's crab pose: overwrite the client's frozen
        // crab entities + placement, so `drive_bones` skins the mesh to exactly the host's pose. The
        // `state` borrow above is released, so this can take the `&mut World` its query needs.
        if let Some(art) = pending_art {
            crate::render::articulation::apply(world, &art);
        }
        // The server stopped serving us (rl#203): surface it and LEAVE the round — never the
        // silent last-frame freeze. Nothing else this frame matters; the Playing→Menu
        // transition (or the scripted exit) takes it from here.
        if let Some(down) = server_down {
            end_round_server_down(world, down, tel.as_ref());
            return;
        }

        // Drive the rapier vehicle (server-authoritative arm) from the SAME tagged `local` control the
        // sim input came from — so the vehicle is active exactly when the sim got a neutral input, with
        // no second read of the mode to drift out of sync. The crab world's force system reads
        // `VehicleControl` on the next physics pump (spawn/despawn the body, apply thrust/lift/drag/
        // torque). The sim never sees the vehicle (the foot player gets neutral input above); it is
        // host-authoritative crab-world state OFF the wire, reading the RAW per-craft flight inputs
        // (Ace Combat 6 for the plane, Outer Wilds for the ship) rather than the sim's foot axes.
        // `VehicleControl` only exists when `VehiclePlugin` is installed — the WINDOWED play app
        // (`app.rs`), where you can actually board a craft. The headless screenshot app omits the
        // plugin (no piloting there), so skip the bridge rather than demand the resource.
        if world.get_resource::<VehicleControl>().is_some() {
            let mut ctrl = world.resource_mut::<VehicleControl>();
            match &local {
                // No body exists while inactive (`manage_vehicle` despawned it), but write zeros so no
                // stale piloting force can ride the despawn edge for a frame.
                LocalControl::OnFoot(_) => {
                    ctrl.active = false;
                    FlightControl::default().write_into(&mut ctrl);
                }
                LocalControl::Piloting { kind, control } => {
                    ctrl.active = true;
                    ctrl.kind = *kind;
                    control.write_into(&mut ctrl);
                }
            }
        }

        // Apply every now-ready tick, ONE at a time. Per applied tick: snapshot the pre-step state
        // for interpolation; if armed, step the crab body by the deterministic cadence and inject
        // its resulting pose + digest into the tick BEFORE the sim advances, so this tick's
        // grab/extraction/outcome resolve against the real NN crab and the digest is folded
        // identically. A real round is always armed (a round that can't arm Sally is refused
        // at build, never reaches here unarmed).
        //
        // The server-authoritative arm: the server steps its OWN sim with this
        // tick's assembled inputs + crab pose, emits a serialized snapshot the local client APPLIES
        // (no re-sim), captures the crab's render pose, and broadcasts both DOWN to remote clients. A
        // REMOTE ADOPT client never enters this loop — it adopted the host's snapshot above and
        // renders its pose from the articulation, running NO crab physics of its own.
        //
        // This inner drain is UNBOUNDED on purpose: it applies every ready tick (a catch-up after a
        // stall must apply them all, in order — and each applied tick advances the cadence, so the
        // physics phase stays aligned regardless of how the catch-up batches). `MAX_TICKS_PER_FRAME`
        // bounds only input ISSUANCE (the outer loop), which prevents a real-time spiral.
        loop {
            // A remote adopt client ran no sim to drain — it took the host's snapshot above and
            // renders from the articulation, never stepping. Nothing to apply here.
            if role == PeerRole::RemoteAdopt {
                break;
            }
            // Ready? Host-paced: `exchange` assembled one tick per issued local input, so this
            // drains exactly what this frame issued (a remote can delay nothing — rl#195).
            {
                let state = world.non_send_resource::<GameState>();
                if !state
                    .server()
                    .expect("server_auth ⇒ a server")
                    .next_tick_ready()
                {
                    break;
                }
            }
            // Capture the client's pre-step state as the interpolation source (the server applies
            // INTO it — the local client renders the snapshot, never stepping a sim of its own).
            {
                let mut state = world.non_send_resource_mut::<GameState>();
                state.prev = SimSnapshot::capture(&state.ls);
            }

            {
                // Pump this tick's crab physics, read the resulting pose, and aim the crab at the
                // AUTHORITATIVE server sim's nearest living player (pre-step).
                let crab_pose = if armed {
                    let steps = world
                        .non_send_resource_mut::<GameState>()
                        .cadence
                        .steps_for_next_tick();
                    pump_fixed_steps(world, steps);
                    // `resource_scope` lifts the bridge out so we can read it AND `GameState` at once.
                    let pose = world.resource_scope(
                        |world, mut bridge: Mut<crate::external_crab::ExternalCrabBridge>| {
                            let pose = crate::server::CrabPose {
                                pos: bridge.world_pos(),
                                yaw: bridge.yaw_turns(),
                                digest: bridge.phys_digest(),
                            };
                            let state = world.non_send_resource::<GameState>();
                            let hunt = state
                                .server()
                                .and_then(|s| s.sim().nearest_living_player_pos());
                            bridge.set_hunt_target(hunt);
                            pose
                        },
                    );
                    // Mirror the vehicle body's freshly-stepped pose for the cockpit camera;
                    // independent of the sim.
                    if let Some(p) = read_vehicle_pose(world) {
                        world.resource_mut::<LocalVehicle>().update_pose(p);
                    }
                    Some(pose)
                } else {
                    None
                };
                // Step the authoritative sim; a RESTART edge (reported by the step itself — the
                // tick stays monotone across a restart, rl#204) resets the cadence at that exact
                // edge (a post-restart tick stepping on the stale phase would desync; the reset
                // must be inside the drain, not end-of-frame).
                let (bytes, restarted) = {
                    let mut state = world.non_send_resource_mut::<GameState>();
                    let stepped = state
                        .server_mut()
                        .expect("server_auth ⇒ a server")
                        .step_next(crab_pose);
                    if stepped.restarted {
                        state.cadence = PhysicsCadence::default();
                    }
                    (stepped.snapshot, stepped.restarted)
                };
                // The ALWAYS-serialized hand-off (decode the bytes the server built; no by-reference
                // shortcut even in SP). The host renders this snapshot locally AND ships it — plus
                // the crab's render pose — DOWN to every remote client so they render the identical
                // state. Capture the pose BEFORE applying to the local
                // client (which never touches the crab entities), so it is the host's live crab.
                let snap = crate::snapshot::CoreSnapshot::from_bytes(&bytes)
                    .expect("the authoritative server's snapshot must decode");
                let articulation =
                    armed.then(|| crate::render::articulation::capture(world, snap.tick));
                {
                    let state = world.non_send_resource::<GameState>();
                    // No-op for solo (no transport); fans out for a host.
                    state.coord.broadcast_step(&snap, articulation.as_ref());
                }
                {
                    let mut state = world.non_send_resource_mut::<GameState>();
                    // Same per-tick latch clear as the adopt arm: an Ongoing tick means the
                    // round is (or just became, via RESTART) live, so the next decision must
                    // report even if this frame's unbounded catch-up drain also re-decides it.
                    if snap.outcome == Outcome::Ongoing {
                        state.reported_outcome = false;
                    }
                    state.ls.apply_core_snapshot(snap);
                }
                if restarted && armed {
                    let spawn = world
                        .non_send_resource::<GameState>()
                        .server()
                        .expect("server_auth ⇒ a server")
                        .sim()
                        .crab()
                        .pos();
                    restart_crab_to_spawn(world, spawn);
                }
                // No peer cross-check on the authoritative path — the host IS the source of truth.
            }
        }

        // Sampled telemetry: a Tick snapshot + the local input every TELEMETRY_TICK_EVERY
        // applied ticks. Read-only on the sim; best-effort (drops if the link can't keep up).
        if let Some(t) = &tel {
            let due = {
                let mut state = world.non_send_resource_mut::<GameState>();
                let due = state.ls.sim().tick() >= state.next_tel_tick;
                if due {
                    state.next_tel_tick = next_sample_tick(state.ls.sim().tick());
                    t.send(TelemetryEvent::tick(state.ls.sim(), roster_len));
                    // The input the SIM actually applied this tick (neutral while piloting).
                    t.send(TelemetryEvent::input(issue_tick, sim_input));
                }
                due
            };
            // Aggregated rescue surface: drain the window's `rescue_nonfinite_crabs`
            // tally into ONE Fault event carrying the count + last offending body, so a
            // frame-by-frame non-finite blowup shows on the hub feed as a filtered per-window
            // count instead of a per-step flood. A stable solo Sally never enters this branch
            // (`since_report` stays 0) — a nonzero count IS the alarm that she's exploding.
            if due
                && let Some(mut stats) = world.get_resource_mut::<crab_world::bot::RescueStats>()
                && stats.since_report > 0
            {
                let n = stats.since_report;
                let body = stats.last_body;
                stats.since_report = 0;
                let msg = match body {
                    Some(b) => format!(
                        "crab rescue: {n} non-finite respawn(s) this telemetry window \
                         (last offender: {b}) — armed Sally is going non-finite (rl#137)"
                    ),
                    None => format!(
                        "crab rescue: {n} non-finite respawn(s) this telemetry window (rl#137)"
                    ),
                };
                t.send(TelemetryEvent::Fault { msg });
            }
        }
    }

    if applied == MAX_TICKS_PER_FRAME {
        // Shed the backlog rather than spiral: drop accumulated time past one tick.
        let mut state = world.non_send_resource_mut::<GameState>();
        state.accumulator = state.accumulator.min(TICK_DT);
    }

    // Round-decided reporting: report once per decided round. The latch clears per applied /
    // adopted Ongoing SNAPSHOT inside the drain arms above (a RESTART revives a decided round
    // to Ongoing without rewinding the tick, rl#204 — and sampling only the frame-end outcome
    // would miss a batch that restarts AND re-decides within one catch-up frame). The
    // telemetry cursor needs no restart handling: the tick is monotone, so it never climbs
    // past a stale watermark.
    let mut state = world.non_send_resource_mut::<GameState>();
    let outcome = state.ls.sim().outcome();
    if !state.reported_outcome && outcome != Outcome::Ongoing {
        state.reported_outcome = true;
        info!("round decided: {outcome:?}");
        if let Some(t) = &tel {
            t.send(TelemetryEvent::round_decided(state.ls.sim()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FlightControl, LocalControl, PeerRole};
    use crate::sim::{Input, buttons};
    use crab_world::vehicle::VehicleKind;

    /// The tagged-control invariant: the deterministic sim only ever applies a FOOT
    /// [`Input`] — the real walker input on foot, a neutral one while piloting. A piloting player
    /// CANNOT leak a flight axis (throttle/pitch/roll) into the sim, because `Piloting` carries a
    /// [`FlightControl`], not an `Input`, and its `sim_input()` is `Input::default()` by
    /// construction — "walker holding a throttle" is unrepresentable.
    #[test]
    fn piloting_feeds_the_sim_a_neutral_foot_input() {
        let walk = Input::new(0.5, -0.5, 0.25, buttons::ACTION);
        assert_eq!(
            LocalControl::OnFoot(walk).sim_input(),
            walk,
            "on foot the real walker input drives the sim unchanged"
        );
        let flying = LocalControl::Piloting {
            kind: VehicleKind::Plane,
            control: FlightControl {
                throttle_trim: 1.0,
                pitch: 1.0,
                roll: -1.0,
                yaw: 1.0,
                match_velocity: true,
                ..Default::default()
            },
        };
        assert_eq!(
            flying.sim_input(),
            Input::default(),
            "piloting: the sim gets a neutral foot input, never a flight axis"
        );
    }

    #[test]
    fn only_the_server_authoritative_arm_can_pilot() {
        // The vehicle gate rides the role axis, so a HOST — a real networked round — can
        // pilot too, since it pumps its own crab-world physics. A remote-adopt client pumps NO physics (it renders the
        // host's crab from articulation), so a craft could never spawn there; boarding is gated OFF
        // it rather than exposed as an inert toggle ([[silent-fallback-antipattern]]). The scripted
        // harness has no live avatar. (A live windowed host/remote toggle needs a `NetDriver`, which
        // won't stand up headlessly — that is on-device territory; here we pin the role predicate.)
        assert!(
            PeerRole::ServerAuth.can_pilot(),
            "solo/host pumps physics ⇒ can pilot"
        );
        assert!(
            !PeerRole::RemoteAdopt.can_pilot(),
            "a remote client pumps no physics ⇒ no craft can spawn ⇒ cannot pilot yet"
        );
    }
}
