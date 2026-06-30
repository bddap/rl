//! Lockstep driver: owns the sim + transport and is the sole writer of sim state.
//!
//! The render-agnostic core of the windowed client — [`GameState`] (the sim + its input
//! source), the per-frame [`drive_lockstep`] tick pump, the deterministic fixed-step crab
//! pump ([`pump_fixed_steps`]), and the menu->round handoff ([`ensure_round_installed`]).
//! Holds no rendering; the scene/HUD/input live in sibling submodules.

use super::*;
use super::app::{ExternalCrabStackInstalled, crab_arm_failure};
use super::input::{CameraPitch, CameraYaw};
use crab_world::vehicle::{Vehicle, VehicleControl, VehicleKind};


/// Shared setup for both apps: the sim + its input source, plus the input resources.
pub(super) fn insert_core(app: &mut App, ls: Lockstep, input_source: InputSource) {
    install_round(app.world_mut(), ls, input_source);
}

/// Install the round resources into the world: the non-send [`GameState`] (sim + input
/// source) and the input resources. Factored out of [`insert_core`] so it can be called
/// BOTH at app build (the scripted/screenshot path) and from the menu's
/// `OnEnter(Playing)` transition system (rl#56), which only has a [`World`], not an
/// [`App`] — one definition so the round is set up identically however it was reached.
fn install_round(world: &mut World, ls: Lockstep, input_source: InputSource) {
    let prev = SimSnapshot::capture(&ls);
    world.insert_non_send_resource(GameState {
        ls,
        input_source,
        accumulator: 0.0,
        prev,
    });
    world.init_resource::<PendingInput>();
    world.init_resource::<FlightInput>();
    world.init_resource::<CameraPitch>();
    world.init_resource::<CameraYaw>();
    world.init_resource::<LocalVehicle>();
}

/// The round the boot menu chose, parked here between the menu's Playing transition and
/// the `OnEnter(Playing)` install. Non-send because a [`menu::ReadyMatch`] holds a
/// `NetDriver` (tokio runtime). `None` on the scripted `Boot::Round` path, which installs
/// `GameState` at app build instead — so [`ensure_round_installed`] no-ops there.
#[derive(Default)]
pub(super) struct PendingRound(pub(super) Option<crate::menu::ReadyMatch>);

/// At the Playing transition, make sure a [`GameState`] exists before the scene spawns.
/// Chained ahead of `spawn_world` (which reads the sim). Two cases, one place:
/// - **Scripted (`Boot::Round`)**: `GameState` was inserted at app build — nothing to do.
/// - **Menu**: take the parked [`PendingRound`] (set by the menu on its choice) and
///   [`install_round`] it now, so the sim is live for the spawns.
///
/// On the menu path this is ALSO where the one giant crab — the real NN body — is ARMED
/// (rl#58 + GCR): insert [`crate::external_crab::ExternalCrabArmed`] and seed the sim crab so the
/// rapier-NN body drives it. The arm DECISION ([`crate::may_arm_external_crab`]: solo always,
/// networked only with synced weights+assets) is made UPSTREAM in the menu's `poll_formation`,
/// which refuses an unarmable networked round and returns to the chooser with an actionable
/// peer-mismatch message (rl#115) BEFORE requesting Playing — so every round reaching this
/// transition is armable by construction, and there is no integer fallback to silently substitute
/// (rl#114). It must arm here rather than gate, because the chained `spawn_world` needs the
/// installed `GameState`; a graceful in-menu refusal is only possible before the transition, which
/// is why the decision lives in the menu. The crab's arena spawn was already seeded into the bridge
/// at build (a pure function of the seed), so nothing about the spawn depends on the round here.
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
        .expect("entered Playing with no round to install — the menu must park a round before transitioning");
    // Arm the one giant crab — the real NN body. The arm DECISION (the shared GCR gate: solo
    // always, networked only with synced weights+assets) is made UPSTREAM, at formation time: the
    // menu's `poll_formation` refuses an unarmable round before requesting Playing (rl#115), so by
    // construction every round that reaches this OnEnter(Playing) transition is armable — there's no
    // graceful recovery to attempt here anyway (the chained `spawn_world` needs the installed
    // `GameState`), which is exactly why the decision moved to the menu where a UI can show it. The
    // checkpoint is required (rl#114), so the stack is always installed; a missing stack is a
    // build-wiring bug, so assert it loudly. The arm invariant is a `debug_assert` — a regression
    // (a future path parking an unvalidated round) trips it in dev/test rather than silently arming
    // an unsynced crab into a desync (the silent-fallback class rl#114 deletes).
    assert!(
        world.get_resource::<ExternalCrabStackInstalled>().is_some(),
        "the NN-crab stack must be installed before Playing (rl#114: the checkpoint is required)"
    );
    debug_assert!(
        crab_arm_failure(&ready.net).is_none(),
        "ensure_round_installed got an unarmable round — poll_formation must gate it out before \
         Playing (rl#115)"
    );
    let crab = ready.lockstep.sim().crab();
    // Seed the pose with the crab's current pose/yaw (writing back what's there → no state change).
    ready
        .lockstep
        .set_external_crab_pose(crab.pos(), crab.yaw(), 0);
    // Arm the gate (the crab now walks at the player's actual position — nothing per-peer to
    // reconcile). One arm path, [`crate::external_crab::arm`].
    crate::external_crab::arm(world);
    // Clone the freshly-seeded sim for the authoritative server (solo/host); the client keeps its
    // own identical sim inside `ready.lockstep` and renders the snapshots the server emits into it.
    let source = InputSource::coordinated(
        ready.net,
        ready.lockstep.peers(),
        ready.lockstep.sim().clone(),
    );
    install_round(world, ready.lockstep, source);
}

// ---------------------------------------------------------------------------
// Lockstep driver state (render-agnostic: owns the sim + transport)
// ---------------------------------------------------------------------------

/// Where the OTHER players' per-tick inputs come from this run. Two mutually-exclusive cases —
/// so "real server-coordinated peers AND fabricated bot inputs" (a meaningless combination) is
/// unrepresentable rather than merely unreached.
pub(super) enum InputSource {
    /// Server-coordinated play (rl#151): solo (internal server, roster of one), host (internal
    /// server + remote clients), or a remote client of a server peer — all the SAME path through
    /// the [`Coordinator`], which is the SP=MP-uniformity proof. Boxed because it can own a
    /// [`NetDriver`] (tokio runtime + iroh session, ~200 bytes), dwarfing the `Scripted` variant.
    Coordinated(Box<Coordinator>),
    /// Stand-in input for the absent peers, fed for every non-local player each tick
    /// (headless screenshot only). It crosses the SAME deterministic `record_remote`
    /// path a wire peer would, so the sim can't distinguish it — a bot/replay input,
    /// not a back channel. Only ever a single-machine solo run, so no peer exists to
    /// disagree with it.
    Scripted(Input),
}

impl InputSource {
    /// Build the server-coordinated source for a round: `None` ⇒ a solo internal server, a host
    /// driver ⇒ a server over the roster, a client driver ⇒ a remote client. `peers` is the sim's
    /// participant set (solo ⇒ just the local player); `initial_sim` is the tick-0 world the server
    /// owns (a clone of the client's freshly-built sim — solo/host step it authoritatively; a remote
    /// client ignores it). The single home of the round's role choice.
    pub(super) fn coordinated(
        net: Option<NetDriver>,
        peers: &[PlayerId],
        initial_sim: crate::sim::Sim,
    ) -> Self {
        InputSource::Coordinated(Box::new(Coordinator::for_round(net, peers, initial_sim)))
    }
}

/// The networked sim, owned as a non-send Bevy resource and stepped on a
/// fixed-timestep accumulator. Non-send because [`NetDriver`] holds a tokio runtime
/// + the iroh session (not `Sync`); only the main thread drives it, so that's fine.
pub(super) struct GameState {
    pub(super) ls: Lockstep,
    /// Where the other players' inputs come from this run (real peers / none / a
    /// scripted stand-in). The sole writer of inputs other than the local controls.
    pub(super) input_source: InputSource,
    /// Fractional-tick accumulator: render time elapsed since the last applied sim
    /// tick, in [0, TICK_DT). Drives both how many ticks to step and the render
    /// interpolation alpha.
    pub(super) accumulator: f64,
    /// Renderable sim state as of the PREVIOUS applied tick. Render tweens from this
    /// toward the live sim by `alpha`. A snapshot (not the live sim) because we need
    /// "last tick" even after the sim has stepped to the current one.
    pub(super) prev: SimSnapshot,
}

impl GameState {
    /// The authoritative server this peer runs, if any (solo or host) — `None` for a remote client
    /// or the scripted screenshot harness. When `Some`, the local client renders the snapshots it
    /// emits rather than stepping `ls`'s sim itself (rl#151 increment 1).
    fn server(&self) -> Option<&crate::server::Server> {
        match &self.input_source {
            InputSource::Coordinated(c) => c.server(),
            InputSource::Scripted(_) => None,
        }
    }

    /// Mutable counterpart of [`GameState::server`], for the per-tick authoritative step.
    fn server_mut(&mut self) -> Option<&mut crate::server::Server> {
        match &mut self.input_source {
            InputSource::Coordinated(c) => c.server_mut(),
            InputSource::Scripted(_) => None,
        }
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
    /// Capture the renderable game state through the host-authoritative snapshot seam
    /// (bddap/rl#151 increment 0): the local client reads its interpolation source from the
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
    /// Accrued yaw-look this inter-tick interval, in radians (drained per tick). Flying,
    /// this drives the plane's ROLL (the stick's horizontal axis); on foot, the avatar yaw.
    pub(super) yaw_delta: f32,
    /// Accrued pitch-look this inter-tick interval, in radians (drained per tick). Flying,
    /// this drives the plane's PITCH (elevator); on foot it is unused for the sim (foot
    /// camera pitch is the separate client-local [`CameraPitch`]). Positive = nose up.
    pub(super) pitch_delta: f32,
    pub(super) action: bool,
    /// Latches if RESTART (R) was pressed in this interval. Sent as
    /// [`buttons::RESTART`] so the restart rides the deterministic input stream and all
    /// peers restart on the same tick; the sim edge-triggers it. Drained per tick.
    pub(super) restart: bool,
    /// Latches if the enter/exit-vehicle key (E) was tapped this interval — a client-local
    /// toggle drained ONCE per frame in [`drive_lockstep`] (single-player only), never sent
    /// to the sim. Board a plane on foot / step out when piloting.
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
/// so a test can pin the directions — the intuitive (push-up = nose-up) pitch shared by both craft,
/// the ship's 6-DOF thrust axes, and the [`VEHICLE_STICK_SENS`] scaling — without spinning a Bevy app.
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
/// - **Plane (AC6)**: left stick (or mouse) flies — pitch (push up = nose up, intuitive) + roll, with
///   a coordinating yaw; RT/LT (or W/S) trim the throttle lever; bumpers (or A/D) are the rudder. No
///   direct thrusters (the plane thrusts through its lever). Right stick is the camera (unused here).
/// - **Ship (Outer Wilds)**: left stick (or WASD) fires the body-frame thrusters (strafe/forward),
///   RT/LT thrust up/down; right stick (or mouse) AIMS (pitch + yaw, camera-style — NOT inverted);
///   bumpers roll; A/Space matches velocity. No throttle lever.
pub(super) fn flight_control(kind: VehicleKind, fi: &FlightInput) -> FlightControl {
    let clamp = |x: f32| x.clamp(-1.0, 1.0);
    match kind {
        VehicleKind::Plane => {
            // Intuitive (camera-style) pitch — the owner found the old AC6 inversion backwards on the
            // controller: push the stick UP (left.y > 0) → nose UP, matching the ship's aim and the
            // on-foot look. Mouse up (mouse.y < 0) also noses up. The analog stick is scaled by
            // VEHICLE_STICK_SENS (the "too sensitive" fix); the mouse keeps its own FLIGHT_MOUSE_SENS.
            let pitch = clamp(fi.left.y * VEHICLE_STICK_SENS - fi.mouse.y);
            let roll = clamp(fi.left.x * VEHICLE_STICK_SENS + fi.mouse.x);
            // Rudder (bumpers / A,D) plus the coordinating yaw that turns a bank into a turn.
            let rudder = (fi.rb as i32 - fi.lb as i32) as f32 + fi.wasd.x;
            let yaw = clamp(rudder + PLANE_TURN_COORDINATION * roll);
            // Throttle lever trim: RT up / LT down (analog), or W/S on the keyboard.
            let throttle_trim = clamp(fi.rt - fi.lt + fi.wasd.y);
            FlightControl { throttle_trim, thrust: Vec3::ZERO, pitch, roll, yaw, match_velocity: false }
        }
        VehicleKind::Ship => {
            // Direct body-frame thrusters: left stick / WASD = strafe (x) + forward (z); RT/LT =
            // vertical (y, up/down). Coast on momentum between taps.
            let thrust = Vec3::new(
                clamp(fi.left.x + fi.wasd.x),
                clamp(fi.rt - fi.lt),
                clamp(fi.left.y + fi.wasd.y),
            );
            // Aim with the right stick / mouse — camera-style, NOT inverted (push up = nose up). The
            // analog stick is scaled by VEHICLE_STICK_SENS (the same "too sensitive" knob as the
            // plane); the mouse keeps its own FLIGHT_MOUSE_SENS.
            let pitch = clamp(fi.right.y * VEHICLE_STICK_SENS - fi.mouse.y);
            let yaw = clamp(fi.right.x * VEHICLE_STICK_SENS + fi.mouse.x);
            // Roll on the bumpers.
            let roll = clamp((fi.rb as i32 - fi.lb as i32) as f32);
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
    pub(super) fn piloting(&self) -> bool {
        !matches!(self, Self::OnFoot)
    }

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
            Self::Flying { kind: VehicleKind::Plane, .. } => GcrContext::Plane,
            Self::Flying { kind: VehicleKind::Ship, .. } => GcrContext::Ship,
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
            Self::OnFoot => Self::Flying { kind: VehicleKind::Plane, pose: None },
            Self::Flying { kind: VehicleKind::Plane, .. } => {
                Self::Flying { kind: VehicleKind::Ship, pose: None }
            }
            Self::Flying { kind: VehicleKind::Ship, .. } => Self::OnFoot,
        }
    }
}

/// Read the single vehicle rigidbody's arena-frame pose (position + attitude) from the crab world,
/// or `None` if none is spawned (on foot, or the frame a freshly-boarded body hasn't appeared yet).
/// One body at most — `manage_vehicle` enforces it; we take the first. The renderer shifts this
/// arena pose to the crab's render spot in `cockpit_camera`.
fn read_vehicle_pose(world: &mut World) -> Option<CockpitPose> {
    let mut q = world.query_filtered::<&Transform, With<Vehicle>>();
    q.iter(world)
        .next()
        .map(|t| CockpitPose { pos: t.translation, orient: t.rotation })
}

// ---------------------------------------------------------------------------
// Lockstep driver system
// ---------------------------------------------------------------------------

/// Log + count each lockstep fault and forward it to telemetry — the shared reporting both
/// fault sites in [`drive_lockstep`] (`record_remote` and `advance_one`) use, so they can't
/// drift. A desync can't be recovered; we surface it (log + telemetry) and play on.
fn report_faults(
    faults: &[crate::lockstep::Fault],
    total: &mut usize,
    tel: &Option<crate::telemetry::TelemetrySender>,
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
/// ([`add_external_nn_crab`]) and the headless cross-peer probe
/// ([`crate::external_crab::run_cross_peer_probe`]) can't drift on it.
pub(crate) fn park_fixed_auto_pump(world: &mut World) {
    world
        .resource_mut::<bevy::time::Time<bevy::time::Fixed>>()
        .set_timestep(std::time::Duration::from_secs(86_400));
}

/// Re-seed the crab bridge to the round's rebuilt `spawn` and cold-respawn the rapier body — run at
/// the exact RESTART rewind edge, by BOTH drain arms ([`drive_lockstep`]). Re-seeding the bridge so
/// the next pose push is the spawn pose (not the still-walking body's accumulated position) is half
/// of it; the other half is the cold respawn — re-seeding alone leaves the rapier solver WARM, which
/// would desync a mid-game joiner's cold body against an incumbent's warm one (job 412, relocated to
/// the join). Dropping + rebuilding the body makes every peer's solver state identically fresh off
/// the same shared-input restart edge, covering the plain RESTART button too (one shared edge).
/// `spawn` is read from whichever sim is now authoritative for the round (the server's, or the
/// client's lockstep on the legacy arm). Only meaningful while armed (else no bridge drives the crab).
fn restart_crab_to_spawn(world: &mut World, spawn: crate::sim::Pos) {
    world
        .resource_mut::<crate::external_crab::ExternalCrabBridge>()
        .restart_to_spawn(spawn);
    crate::external_crab::cold_respawn_armed_crab(world);
}

/// Advance the lockstep sim by real time on a fixed-timestep accumulator. This is the ONLY
/// writer of sim state, and (apart from the external crab pose) it writes exactly one thing:
/// the local [`Input`] (drained from [`PendingInput`]) via `submit_local_input`. Everything
/// else is the existing deterministic machinery — pump the transport, then advance.
///
/// GCR fold: when the external NN crab is armed ([`ExternalCrabArmed`] — solo OR a
/// networked round with synced weights, [`crate::may_arm_external_crab`]), the rapier
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
pub(super) fn drive_lockstep(
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
    // weights). Read once — the gate is fixed for the round. A real round is always armed (rl#114:
    // a round that can't arm Sally is refused at build); when off (e.g. behind the boot menu) no
    // physics is pumped and the crab holds its spawn.
    let armed = world
        .get_resource::<crate::external_crab::ExternalCrabArmed>()
        .is_some();

    // Whether THIS peer runs the authoritative server for the round (solo or host): its local client
    // RENDERS the per-tick snapshot the server emits instead of stepping a sim of its own (rl#151
    // increment 1). Fixed for the round. A remote client and the headless scripted screenshot harness
    // keep the legacy lockstep advance below (the remote path migrates onto the snapshot in
    // increment 2). This is the Minecraft-model server/client role, NOT an SP/MP split — solo and
    // host take the SAME authoritative arm ([[sp-is-mp-special-case]]).
    let server_auth = world.non_send_resource::<GameState>().server().is_some();

    let delta = world.resource::<Time>().delta().as_secs_f64();

    // Clone out the telemetry handle (cheap: an mpsc sender + id) + the roster size so we can
    // READ the sim and push events without holding a borrow of the `NetDriver`. `None` unless
    // this is networked play with a collector. Telemetry never writes the sim.
    let (tel, roster_len) = {
        let state = world.non_send_resource::<GameState>();
        let tel = match &state.input_source {
            InputSource::Coordinated(c) => c.telemetry().cloned(),
            InputSource::Scripted(_) => None,
        };
        // Roster size from the AUTHORITATIVE lockstep peer set — correct on the host, an incumbent,
        // AND a mid-game joiner (its `Lockstep` is rebuilt over the new roster). One source of
        // truth, not a second copy in the driver's id_map (which a joiner only half-fills).
        (tel, state.ls.peers().len())
    };
    if *next_tel_tick == 0 {
        *next_tel_tick = TELEMETRY_TICK_EVERY;
    }

    world.non_send_resource_mut::<GameState>().accumulator += delta;

    // Single-player enter/exit a vehicle. Drain the E-tap latch ONCE per frame and CYCLE foot →
    // plane → ship → foot. The actual craft is a rapier body spawned/despawned in the crab
    // world from the resulting `LocalVehicle` (mirrored into `VehicleControl` below). Solo only —
    // a networked round is foot-only, so this toggle is inert there and the lockstep is untouched.
    {
        let toggle = std::mem::take(&mut world.resource_mut::<PendingInput>().toggle_vehicle);
        let solo = toggle
            && matches!(
                &world.non_send_resource::<GameState>().input_source,
                InputSource::Coordinated(c) if c.is_solo()
            );
        if solo {
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

        // Build THIS tick's local input from the accumulated controls, draining the accrued
        // look + latched buttons (movement axes are re-sampled next frame).
        let input = {
            let mut pending = world.resource_mut::<PendingInput>();
            let look_axis = (pending.yaw_delta / MAX_YAW_PER_TICK_RADIANS).clamp(-1.0, 1.0);
            // The pitch axis reuses the SAME per-tick radian scale as the yaw axis, so the
            // mouse/stick feels symmetric vertically and horizontally; the rapier vehicle then
            // applies each axis's own control torque (roll vs pitch) in its force model.
            let pitch_axis = (pending.pitch_delta / MAX_YAW_PER_TICK_RADIANS).clamp(-1.0, 1.0);
            let btns = (if pending.action { buttons::ACTION } else { 0 })
                | (if pending.restart { buttons::RESTART } else { 0 });
            let input =
                Input::new(pending.strafe, pending.forward, look_axis, btns).with_look_pitch(pitch_axis);
            pending.yaw_delta = 0.0;
            pending.pitch_delta = 0.0;
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
            let exch: Exchanged = match &mut st.input_source {
                // Server-coordinated: ship our input to the (internal or remote) server and get
                // back the OTHER players' inputs (+ any mid-game roster change to schedule). Solo
                // runs the same exchange against a roster of one — empty, so the sim advances on our
                // own filed input alone.
                InputSource::Coordinated(c) => c.exchange(me, msg),
                InputSource::Scripted(bot) => {
                    // Stand in for the absent peers so the (otherwise-stalled) sim advances:
                    // feed every non-local player this input at the SAME apply_tick the local
                    // input got. Always a single-machine solo run, so no peer disagrees + no joins.
                    let bot = *bot;
                    let peer_msgs = st
                        .ls
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
                        .collect();
                    Exchanged { peer_msgs, roster_changes: Vec::new() }
                }
            };
            // In the SERVER-AUTHORITATIVE path the server has already stepped these inputs into its
            // OWN sim (inside `exchange`'s host_assemble), and the local client renders the snapshot
            // it emits below — so the client never records peer inputs or schedules joins into its
            // own (non-stepping) lockstep. The legacy path (a remote client, or the scripted
            // screenshot harness) still records them and advances its lockstep itself.
            let mut faults = Vec::new();
            if !server_auth {
                // Schedule any roster change BEFORE recording inputs: a mid-game join the host
                // admitted or this client learned over the wire. `effective_tick` is JOIN_LEAD
                // ahead, so applying it now (well before the boundary) lets `advance_one` rebuild the
                // round in lockstep on every peer.
                for adm in &exch.roster_changes {
                    state.ls.schedule_roster_change(adm.effective_tick, &adm.roster);
                }
                for from in exch.peer_msgs {
                    if from.pid != me
                        && let Some(fault) = state.ls.record_remote(from.pid, from.msg)
                    {
                        faults.push(fault);
                    }
                }
            }
            (issue_tick, faults)
        };
        report_faults(&faults, &mut total_desyncs, &tel);

        // Drive the rapier vehicle (single-player): mirror the piloting state + this frame's flight
        // controls into `VehicleControl`, which the crab world's force system reads on the next
        // physics pump — spawn/despawn the body and apply thrust/lift/drag/torque. The sim never
        // sees the vehicle (the foot player gets neutral input above); it is host-authoritative
        // crab-world state OFF the wire, so it reads the RAW per-craft flight inputs (Ace Combat 6
        // for the plane, Outer Wilds for the ship — `flight_control`) rather than the sim's merged
        // move/look axes. The `FlightInput` snapshot is per-frame, so the intent is stable across the
        // ticks pumped this frame.
        // `VehicleControl` only exists when `VehiclePlugin` is installed — the WINDOWED play app
        // (`app.rs`), where you can actually board a craft. The headless screenshot app omits the
        // plugin (no piloting there), so skip the bridge rather than demand the resource.
        if world.get_resource::<VehicleControl>().is_some() {
            let kind = world.resource::<LocalVehicle>().kind();
            let fc = kind.map(|k| flight_control(k, world.resource::<FlightInput>()));
            let mut ctrl = world.resource_mut::<VehicleControl>();
            ctrl.active = kind.is_some();
            if let Some(k) = kind {
                ctrl.kind = k;
            }
            let fc = fc.unwrap_or_default();
            ctrl.throttle_trim = fc.throttle_trim;
            ctrl.thrust = fc.thrust;
            ctrl.pitch = fc.pitch;
            ctrl.roll = fc.roll;
            ctrl.yaw = fc.yaw;
            ctrl.match_velocity = fc.match_velocity;
        }

        // Apply every now-ready tick, ONE at a time. Per applied tick: snapshot the pre-step state
        // for interpolation; if armed, step the crab body by the deterministic cadence and inject
        // its resulting pose + digest into the tick BEFORE the sim advances, so this tick's
        // grab/extraction/outcome resolve against the real NN crab and the digest is folded
        // identically. A real round is always armed (rl#114: a round that can't arm Sally is refused
        // at build, never reaches here unarmed).
        //
        // Two arms (rl#151 increment 1), chosen by the round's fixed role:
        // - SERVER-AUTHORITATIVE (solo/host): the server steps its OWN sim with this tick's
        //   assembled inputs + crab pose and emits a serialized snapshot; the local client APPLIES
        //   it (no re-sim) and renders from it.
        // - LEGACY (remote client / scripted screenshot): the client steps its own lockstep via
        //   `advance_one` (the remote path migrates onto the snapshot in increment 2).
        // Both inject the SAME bridge pose + digest in the SAME pre-step position, so the
        // authoritative sim is byte-identical to what the lockstep advance produced ([[sp-is-mp-special-case]]).
        //
        // This inner drain is UNBOUNDED on purpose: it applies every ready tick (a catch-up after a
        // stall must apply them all, in order — and each applied tick advances the cadence, so the
        // physics phase stays aligned regardless of how the catch-up batches). `MAX_TICKS_PER_FRAME`
        // bounds only input ISSUANCE (the outer loop), which prevents a real-time spiral.
        loop {
            // Ready? The authoritative server gates on its own ledger/warmup; a legacy client gates
            // on its lockstep.
            {
                let state = world.non_send_resource::<GameState>();
                let ready = match state.server() {
                    Some(server) => server.next_tick_ready(),
                    None => state.ls.next_tick_ready(),
                };
                if !ready {
                    break;
                }
            }
            // Capture the client's pre-step state as the interpolation source — both arms render
            // from the client sim (server-auth applies INTO it; legacy steps it).
            {
                let mut state = world.non_send_resource_mut::<GameState>();
                state.prev = SimSnapshot::capture(&state.ls);
            }

            if server_auth {
                // Pump this tick's crab physics, read the resulting pose, and aim the crab at the
                // AUTHORITATIVE server sim's nearest living player (pre-step, exactly where
                // `sync_external_crab` reads it on the legacy arm).
                let crab_pose = if armed {
                    let steps = cadence.steps_for_next_tick();
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
                    // Mirror the vehicle body's freshly-stepped pose for the cockpit camera (as the
                    // legacy arm does); independent of the sim.
                    if let Some(p) = read_vehicle_pose(world) {
                        world.resource_mut::<LocalVehicle>().update_pose(p);
                    }
                    Some(pose)
                } else {
                    None
                };
                // Step the authoritative sim and detect a RESTART rewind, resetting the cadence at
                // that exact edge (same reasoning as the legacy arm: a post-restart tick stepping on
                // the stale phase would desync; the reset must be inside the drain, not end-of-frame).
                let (bytes, restarted) = {
                    let mut state = world.non_send_resource_mut::<GameState>();
                    let server = state.server_mut().expect("server_auth ⇒ a server");
                    let before = server.sim().tick();
                    let bytes = server.step_next(crab_pose);
                    let restarted = server.sim().tick() < before;
                    if restarted {
                        *cadence = PhysicsCadence::default();
                    }
                    (bytes, restarted)
                };
                // Apply the server's snapshot as the client's rendered state — the ALWAYS-serialized
                // hand-off (decode the bytes the server built; no by-reference shortcut even in SP).
                {
                    let mut state = world.non_send_resource_mut::<GameState>();
                    let snap = crate::snapshot::CoreSnapshot::from_bytes(&bytes)
                        .expect("the authoritative server's snapshot must decode");
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
                // No peer cross-check on the authoritative path (the host IS the source of truth);
                // faults only exist on the legacy peer-symmetric arm below.
            } else {
                // LEGACY arm: the client steps its own lockstep (remote client / scripted screenshot).
                if armed {
                    let steps = cadence.steps_for_next_tick();
                    pump_fixed_steps(world, steps);
                    // Push the freshly-stepped body's pose + weights-folded digest + refresh the
                    // hunted player — the shared handshake (one source with the headless probe).
                    world.resource_scope(
                        |world, mut bridge: Mut<crate::external_crab::ExternalCrabBridge>| {
                            let mut state = world.non_send_resource_mut::<GameState>();
                            crate::external_crab::sync_external_crab(&mut state.ls, &mut bridge);
                        },
                    );
                    if let Some(pose) = read_vehicle_pose(world) {
                        world.resource_mut::<LocalVehicle>().update_pose(pose);
                    }
                }
                let (tick_faults, restarted) = {
                    let mut state = world.non_send_resource_mut::<GameState>();
                    let before = state.ls.sim().tick();
                    let faults = state.ls.advance_one().expect("next_tick_ready was true");
                    let restarted = state.ls.sim().tick() < before;
                    if restarted {
                        *cadence = PhysicsCadence::default();
                    }
                    (faults, restarted)
                };
                if restarted && armed {
                    let spawn = world.non_send_resource::<GameState>().ls.sim().crab().pos();
                    restart_crab_to_spawn(world, spawn);
                }
                report_faults(&tick_faults, &mut total_desyncs, &tel);
            }
        }

        // Sampled telemetry: a Tick snapshot + the local input every TELEMETRY_TICK_EVERY
        // applied ticks. Read-only on the sim; best-effort (drops if the link can't keep up).
        if let Some(t) = &tel {
            let due = world.non_send_resource::<GameState>().ls.sim().tick() >= *next_tel_tick;
            if due {
                {
                    let state = world.non_send_resource::<GameState>();
                    *next_tel_tick =
                        (state.ls.sim().tick() / TELEMETRY_TICK_EVERY + 1) * TELEMETRY_TICK_EVERY;
                    t.send(TelemetryEvent::tick(state.ls.sim(), *total_desyncs, roster_len));
                    // The input the SIM actually applied this tick (neutral while piloting).
                    t.send(TelemetryEvent::input(issue_tick, sim_input));
                }
                // Aggregated rescue surface (rl#137): drain the window's `rescue_nonfinite_crabs`
                // tally into ONE Fault event carrying the count + last offending body, so a
                // frame-by-frame non-finite blowup shows on the hub feed as a filtered per-window
                // count instead of a per-step flood. A stable solo Sally never enters this branch
                // (`since_report` stays 0) — a nonzero count IS the alarm that she's exploding.
                if let Some(mut stats) = world.get_resource_mut::<crab_world::bot::RescueStats>()
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
