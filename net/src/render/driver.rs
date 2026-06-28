//! Lockstep driver: owns the sim + transport and is the sole writer of sim state.
//!
//! The render-agnostic core of the windowed client — [`GameState`] (the sim + its input
//! source), the per-frame [`drive_lockstep`] tick pump, the deterministic fixed-step crab
//! pump ([`pump_fixed_steps`]), and the menu->round handoff ([`ensure_round_installed`]).
//! Holds no rendering; the scene/HUD/input live in sibling submodules.

use super::*;
use super::app::{ExternalCrabStackInstalled, crab_arm_failure_message};
use super::input::{CameraPitch, CameraYaw};


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
/// (rl#58 + GCR): if the resolved round may arm it ([`crate::may_arm_external_crab`]: solo
/// always, networked only with synced weights+assets), insert
/// [`crate::external_crab::ExternalCrabArmed`] and seed the sim crab so the rapier-NN body
/// drives it. A networked-UNSYNCED round CANNOT arm and, with no integer fallback (rl#114), FAILS
/// LOUD here with an actionable peer-mismatch message rather than silently substituting a fake
/// crab. The crab's arena spawn was already seeded into the bridge at build (a pure function of
/// the seed), so nothing about the spawn depends on the round here.
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
    // Arm the one giant crab — the real NN body — iff the resolved round may (the shared GCR gate:
    // solo always, networked only with synced weights+assets — a float crab on an unsynced
    // networked round would desync peers). The checkpoint is required (rl#114), so the stack is
    // always installed; a round that CAN'T arm FAILS LOUD here rather than substituting a fake
    // crab. The stack marker must be present (it always is on the menu path) — a missing stack is a
    // build-wiring bug, so assert it loudly too.
    assert!(
        world.get_resource::<ExternalCrabStackInstalled>().is_some(),
        "the NN-crab stack must be installed before Playing (rl#114: the checkpoint is required)"
    );
    let networked = ready.net.is_some();
    let weights_synced = ready.net.as_ref().is_some_and(NetDriver::weights_synced);
    let assets_synced = ready.net.as_ref().is_some_and(NetDriver::assets_synced);
    if !crate::may_arm_external_crab(ready.net.is_none(), weights_synced, assets_synced) {
        panic!("{}", crab_arm_failure_message(&ready.net));
    }
    let crab = ready.lockstep.sim().crab();
    // Seed the pose with the crab's current pose/yaw (writing back what's there → no state change).
    ready
        .lockstep
        .set_external_crab_pose(crab.pos(), crab.yaw(), 0);
    world.insert_resource(crate::external_crab::ExternalCrabArmed);
    // Networked: pin the walk-target lead to its default so a per-peer env override can't walk
    // the crab to a different (hashed) pose and desync — solo keeps its tuning.
    if networked {
        world
            .resource_mut::<crate::external_crab::ExternalCrabBridge>()
            .pin_default_lead();
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
pub(super) enum InputSource {
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
    pub(super) planes: BTreeMap<PlayerId, Plane>,
    pub(super) crab: Option<Crab>,
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
pub(super) struct PendingInput {
    pub(super) strafe: f32,
    pub(super) forward: f32,
    /// Accrued yaw-look this inter-tick interval, in radians (drained per tick).
    pub(super) yaw_delta: f32,
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

/// The local player's SINGLE-PLAYER vehicle, when piloting one. `None` = on foot.
///
/// Single-player vehicle flight lives ENTIRELY here in the play layer, NOT in the
/// deterministic sim ([`crate::sim`] stays integer-only and untouched): a solo round
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
pub(super) enum LocalVehicle {
    #[default]
    OnFoot,
    Piloting {
        plane: Plane,
        prev: Plane,
    },
}

impl LocalVehicle {
    pub(super) fn piloting(&self) -> bool {
        matches!(self, Self::Piloting { .. })
    }

    /// The controls CONTEXT this vehicle state presents — the single mapping from "what am I
    /// driving" to "which control set + legend is live". The overlay reads it via
    /// [`ActiveContext`]; adding a vehicle type means adding an arm here and its rows in
    /// [`crab_world::controls`], and the HUD names + labels it automatically.
    pub(super) fn context(&self) -> GcrContext {
        match self {
            Self::OnFoot => GcrContext::OnFoot,
            Self::Piloting { .. } => GcrContext::Plane,
        }
    }
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
