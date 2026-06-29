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
    let networked = ready.net.is_some();
    let crab = ready.lockstep.sim().crab();
    // Seed the pose with the crab's current pose/yaw (writing back what's there → no state change).
    ready
        .lockstep
        .set_external_crab_pose(crab.pos(), crab.yaw(), 0);
    // Arm the gate (and, networked, pin the lead so a per-peer env override can't desync the
    // hashed pose — solo keeps its tuning). One arm path, [`crate::external_crab::arm`].
    crate::external_crab::arm(world, networked);
    let source = InputSource::coordinated(ready.net, ready.lockstep.peers());
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
    /// participant set (solo ⇒ just the local player). The single home of the round's role choice.
    pub(super) fn coordinated(net: Option<NetDriver>, peers: &[PlayerId]) -> Self {
        InputSource::Coordinated(Box::new(Coordinator::for_round(net, peers)))
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

/// A minimal copy of the renderable sim state at one tick — the poses the client
/// tweens from. NOT a second source of truth: overwritten from the authoritative
/// [`Sim`] every tick, never fed back into it.
#[derive(Clone, Default)]
pub(super) struct SimSnapshot {
    pub(super) players: BTreeMap<PlayerId, Player>,
    pub(super) crab: Option<Crab>,
}

impl SimSnapshot {
    fn capture(sim: &Sim) -> Self {
        Self {
            players: sim.players().collect(),
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

/// A flyer's first-person cockpit pose in the crab's ARENA frame: a 3D position + a full attitude
/// quaternion (body→world). Read off the rapier vehicle body each applied tick; the renderer maps
/// it into render space (shifted to the crab's render spot, [`super::scene`]'s `cockpit_camera`)
/// and flies the one cockpit camera from it — so the plane and the helicopter share one camera
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
/// stepping out returns the view to the foot player. The E/X control CYCLES foot → plane →
/// helicopter → foot.
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
            Self::Flying { kind: VehicleKind::Helicopter, .. } => GcrContext::Helicopter,
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

    /// The NEXT vehicle in the enter/exit cycle (foot → plane → helicopter → foot). One place the
    /// cycle order lives, so the input toggle and any future caller can't disagree on it. A boarded
    /// craft starts with no pose (`None`); [`update_pose`] fills it from the spawned body.
    fn cycled(&self) -> Self {
        match self {
            Self::OnFoot => Self::Flying { kind: VehicleKind::Plane, pose: None },
            Self::Flying { kind: VehicleKind::Plane, .. } => {
                Self::Flying { kind: VehicleKind::Helicopter, pose: None }
            }
            Self::Flying { kind: VehicleKind::Helicopter, .. } => Self::OnFoot,
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
    // plane → helicopter → foot. The actual craft is a rapier body spawned/despawned in the crab
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
            // Schedule any roster change BEFORE recording inputs: a mid-game join the host admitted
            // or this client learned over the wire. `effective_tick` is JOIN_LEAD ahead, so applying
            // it now (well before the boundary) lets `advance_one` rebuild the round in lockstep on
            // every peer. The host applies its own admissions through this same path.
            for adm in &exch.roster_changes {
                state.ls.schedule_roster_change(adm.effective_tick, &adm.roster);
            }
            let mut faults = Vec::new();
            for from in exch.peer_msgs {
                if from.pid != me
                    && let Some(fault) = state.ls.record_remote(from.pid, from.msg)
                {
                    faults.push(fault);
                }
            }
            (issue_tick, faults)
        };
        report_faults(&faults, &mut total_desyncs, &tel);

        // Drive the rapier vehicle (single-player): mirror the piloting state + this tick's
        // controls into `VehicleControl`, which the crab world's force system reads on the next
        // physics pump — spawn/despawn the body and apply thrust/lift/drag/torque. The sim never
        // sees the vehicle (the foot player gets neutral input above); it is host-authoritative
        // crab-world state. Axes are screen-reconciled EXACTLY as the old integer model did
        // (`-look_yaw` → bank right, `-move_strafe` → yaw right, `look_pitch` → nose up), so the
        // controls + their on-screen labels are unchanged for the pilot.
        {
            let kind = world.resource::<LocalVehicle>().kind();
            let scale = Input::AXIS_SCALE as f32;
            let mut ctrl = world.resource_mut::<VehicleControl>();
            ctrl.active = kind.is_some();
            if let Some(k) = kind {
                ctrl.kind = k;
            }
            ctrl.throttle_trim = input.move_forward as f32 / scale;
            ctrl.pitch = input.look_pitch as f32 / scale;
            ctrl.roll = -(input.look_yaw as f32) / scale;
            ctrl.yaw = -(input.move_strafe as f32) / scale;
        }

        // Apply every now-ready tick, ONE at a time. Per applied tick: snapshot the pre-step
        // state for interpolation; if armed, step the crab body by the deterministic cadence
        // and push its resulting pose + digest BEFORE advancing, so this tick's
        // grab/extraction/outcome resolve against the real NN crab and every peer folds the
        // identical `phys_digest`. A real round is always armed (rl#114: a round that can't arm
        // Sally is refused at build, never reaches here unarmed).
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
                    |world, mut bridge: Mut<crate::external_crab::ExternalCrabBridge>| {
                        let mut state = world.non_send_resource_mut::<GameState>();
                        crate::external_crab::sync_external_crab(&mut state.ls, &mut bridge);
                    },
                );
                // Mirror the vehicle body's freshly-stepped arena pose into `LocalVehicle` for the
                // cockpit camera's interpolation (the float analogue of the `SimSnapshot::capture`
                // above). One body at most; read its Transform and shift `now`→`prev`. None while on
                // foot or before a boarded body has spawned — `cockpit_poses` then returns `None` and
                // the camera holds the foot view (no fabricated pose).
                if let Some(pose) = read_vehicle_pose(world) {
                    world.resource_mut::<LocalVehicle>().update_pose(pose);
                }
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
                    .resource_mut::<crate::external_crab::ExternalCrabBridge>()
                    .restart_to_spawn(spawn);
                // Re-seeding the bridge alone leaves the rapier solver WARM, which desyncs a
                // mid-game joiner's cold body against an incumbent's warm one (job 412, relocated to
                // the join). Drop + rebuild the body so every peer's solver state is identically
                // fresh. Same shared-input restart edge ⇒ bit-identical cross-peer; covers the plain
                // RESTART button too (one shared edge).
                crate::external_crab::cold_respawn_armed_crab(world);
            }
            report_faults(&tick_faults, &mut total_desyncs, &tel);
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
