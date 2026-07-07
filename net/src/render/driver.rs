
use super::app::{ArmedRound, ExternalCrabStackInstalled};
use super::input::{CameraPitch, CameraYaw};
use super::*;
use crate::lockstep::PilotIntent;
use crab_world::vehicle::{PilotCommand, PilotId, Vehicle, VehicleControls, VehicleKind};

pub(super) fn coordinator(
    net: Option<NetDriver>,
    peers: &[PlayerId],
    me: PlayerId,
    initial_sim: crate::sim::Sim,
) -> Box<Coordinator> {
    Box::new(Coordinator::for_round(net, peers, me, initial_sim))
}

pub(super) fn insert_core(app: &mut App, ls: Lockstep, coord: Box<Coordinator>) {
    install_round(app.world_mut(), ls, coord);
}

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
        snap_buf: std::collections::VecDeque::new(),
        art_buf: std::collections::VecDeque::new(),
        logged_statuses: BTreeMap::new(),
    });
    world.insert_resource(PendingInput::default());
    world.insert_resource(FlightInput::default());
    world.insert_resource(CameraPitch::default());
    world.insert_resource(CameraYaw::default());
    world.insert_resource(LocalVehicle::default());
    world.insert_resource(super::articulation::RemoteVehicle::default());
}

#[derive(Default)]
pub(super) struct PendingRound(pub(super) Option<ArmedRound>);

pub(super) fn ensure_round_installed(world: &mut World) {
    if world.get_non_send_resource::<GameState>().is_some() {
        return;
    }
    let mut ready = world
        .get_non_send_resource_mut::<PendingRound>()
        .and_then(|mut p| p.0.take())
        .expect("entered Playing with no round to install — the menu must park a round before transitioning")
        .into_ready();
    assert!(
        world.get_resource::<ExternalCrabStackInstalled>().is_some(),
        "the NN-crab stack must be installed before Playing (rl#114: the checkpoint is required)"
    );
    let spawns = match world.get_resource::<crate::external_crab::ExternalCrabBridge>() {
        Some(bridge) => {
            let n = bridge.crab_count();
            super::app::seed_round_crabs(&mut ready.lockstep, n)
        }
        None => Vec::new(),
    };
    crate::external_crab::arm(world);
    if !spawns.is_empty() {
        restart_crab_to_spawn(world, &spawns);
    }
    let coord = coordinator(
        ready.net,
        ready.lockstep.peers(),
        ready.lockstep.me(),
        ready.lockstep.sim().clone(),
    );
    install_round(world, ready.lockstep, coord);
}

pub(super) fn teardown_round(world: &mut World) {
    world.remove_non_send_resource::<GameState>();
    world.remove_resource::<crate::external_crab::ExternalCrabArmed>();
    // Un-label the crab bodies that persist across rounds: labels are round state (the host
    // republishes on the next arm; a client re-adopts from the next round's articulation),
    // and a survivor here would float brain labels over the menu (the rl#211 class).
    if let Some(mut labels) = world.get_resource_mut::<crab_world::crab_view::CrabBrainLabels>() {
        labels.0.clear();
    }
    if let Some(mut ctrl) = world.get_resource_mut::<VehicleControls>() {
        ctrl.0.clear();
    }
    // Round state like the labels above: a survivor would suppress (or mis-measure) the next
    // round's remote-craft appeared/moved edges.
    world.remove_resource::<super::articulation::RemoteCraftWatch>();
}

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


#[derive(Resource, Clone, Copy)]
pub(super) struct ScriptedPackInput(pub(super) Input);

#[derive(Clone, Copy, PartialEq, Eq)]
enum PeerRole {
    ServerAuth,
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
}

fn pilot_of(pid: PlayerId) -> PilotId {
    PilotId(pid.0)
}

fn local_pilot(state: &GameState) -> PilotId {
    pilot_of(state.ls.me())
}

pub(super) struct GameState {
    pub(super) ls: Lockstep,
    pub(super) coord: Box<Coordinator>,
    pub(super) accumulator: f64,
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
    /// armed and reset on the restart edge (reported by `step_next`, rl#204) so every round
    /// pumps the same physics-step schedule from tick 0.
    cadence: PhysicsCadence,
    snap_buf: std::collections::VecDeque<crate::snapshot::CoreSnapshot>,
    art_buf: std::collections::VecDeque<crate::articulation::CrabArticulation>,
    /// Last logged per-player status, so Alive→Downed/Extracted edges (a crab strike landing,
    /// an extraction) each leave exactly one log line on every peer.
    logged_statuses: BTreeMap<PlayerId, PlayerStatus>,
}

const JITTER_BUF_MAX: usize = 3;
const JITTER_BUF_TARGET: usize = 1;

fn jitter_take(buffered: usize) -> usize {
    if buffered > JITTER_BUF_MAX {
        buffered - JITTER_BUF_TARGET
    } else {
        usize::from(buffered > 0)
    }
}

impl GameState {
    fn server(&self) -> Option<&crate::server::Server> {
        self.coord.server()
    }

    fn server_mut(&mut self) -> Option<&mut crate::server::Server> {
        self.coord.server_mut()
    }
}

#[derive(Clone, Default)]
pub(super) struct SimSnapshot {
    pub(super) players: BTreeMap<PlayerId, Player>,
    pub(super) crabs: Vec<Crab>,
}

impl SimSnapshot {
    fn capture(ls: &Lockstep) -> Self {
        let snap = ls.core_snapshot();
        Self {
            players: snap.players,
            crabs: snap.crabs,
        }
    }
}

#[derive(Resource, Default)]
pub(super) struct PendingInput {
    pub(super) strafe: f32,
    pub(super) forward: f32,
    pub(super) yaw_delta: f32,
    pub(super) action: bool,
    pub(super) restart: bool,
    pub(super) toggle_vehicle: bool,
}

#[derive(Resource, Default)]
pub(super) struct FlightInput {
    pub(super) left: Vec2,
    pub(super) right: Vec2,
    pub(super) mouse: Vec2,
    pub(super) wasd: Vec2,
    pub(super) rt: f32,
    pub(super) lt: f32,
    pub(super) lb: bool,
    pub(super) rb: bool,
    pub(super) match_vel: bool,
}

const PLANE_TURN_COORDINATION: f32 = 0.3;

pub(super) const VEHICLE_STICK_SENS: f32 = 0.5;

#[derive(Debug, Default, PartialEq)]
pub(super) struct FlightControl {
    pub throttle_trim: f32,
    pub thrust: Vec3,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub match_velocity: bool,
}

pub(super) fn flight_control(kind: VehicleKind, fi: &FlightInput) -> FlightControl {
    let clamp = |x: f32| x.clamp(-1.0, 1.0);
    match kind {
        VehicleKind::Plane => {
            let pitch = clamp(-fi.left.y * VEHICLE_STICK_SENS + fi.mouse.y);
            let roll = clamp(-(fi.left.x * VEHICLE_STICK_SENS + fi.mouse.x));
            let rudder = (fi.lb as i32 - fi.rb as i32) as f32 - fi.wasd.x;
            let yaw = clamp(rudder + PLANE_TURN_COORDINATION * roll);
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
            let thrust = Vec3::new(
                clamp(-(fi.left.x + fi.wasd.x)),
                clamp(fi.rt - fi.lt),
                clamp(fi.left.y + fi.wasd.y),
            );
            let pitch = clamp(fi.right.y * VEHICLE_STICK_SENS - fi.mouse.y);
            let yaw = clamp(-(fi.right.x * VEHICLE_STICK_SENS + fi.mouse.x));
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

enum LocalControl {
    OnFoot(Input),
    Piloting {
        kind: VehicleKind,
        control: FlightControl,
    },
}

impl LocalControl {
    fn sim_input(&self) -> Input {
        match self {
            LocalControl::OnFoot(input) => *input,
            LocalControl::Piloting { .. } => Input::default(),
        }
    }

    fn pilot_intent(&self) -> Option<PilotIntent> {
        match self {
            LocalControl::OnFoot(_) => None,
            LocalControl::Piloting { kind, control } => {
                let FlightControl {
                    throttle_trim,
                    thrust,
                    pitch,
                    roll,
                    yaw,
                    match_velocity,
                } = *control;
                Some(PilotIntent {
                    kind: *kind,
                    throttle_trim,
                    thrust: thrust.to_array(),
                    pitch,
                    roll,
                    yaw,
                    match_velocity,
                })
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct CockpitPose {
    pub pos: Vec3,
    pub orient: Quat,
}

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
    pub(super) fn kind(&self) -> Option<VehicleKind> {
        match self {
            Self::OnFoot => None,
            Self::Flying { kind, .. } => Some(*kind),
        }
    }

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

    pub(super) fn cockpit_poses(&self) -> Option<(CockpitPose, CockpitPose)> {
        match self {
            Self::OnFoot => None,
            Self::Flying { pose, .. } => *pose,
        }
    }

    fn update_pose(&mut self, p: CockpitPose) {
        if let Self::Flying { pose, .. } = self {
            *pose = Some(match *pose {
                Some((_, now)) => (now, p),
                None => (p, p),
            });
        }
    }

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

/// Read OUR OWN vehicle rigidbody's arena-frame pose (position + attitude) from the crab world,
/// or `None` if none is spawned (on foot, or the frame a freshly-boarded body hasn't appeared
/// yet). At most one body per pilot — `manage_vehicles` enforces it. Keyed by pilot because the
/// host's world will also carry REMOTE pilots' crafts (rl#191): the cockpit camera must fly from
/// ours alone. The renderer shifts this arena pose to the crab's render spot in `cockpit_camera`.
fn read_vehicle_pose(world: &mut World, me: PilotId) -> Option<CockpitPose> {
    let mut q = world.query::<(&Transform, &Vehicle)>();
    q.iter(world)
        .find(|(_, v)| v.pilot == me)
        .map(|(t, _)| CockpitPose {
            pos: t.translation,
            orient: t.rotation,
        })
}

/// The remote-client twin of [`read_vehicle_pose`]: our own craft's arena-frame pose out of an
/// adopted articulation. The host simulates the craft; its pose comes back per-pilot on the
/// wire. `None` keeps the cockpit camera off while `Flying { pose: None }` — usually the
/// request→grant window, but also a silent HOST REFUSAL (`file_intent`'s alive gate saw us
/// non-Alive; the client can't tell the two apart) and the unarmed round (no articulation at
/// all). Either way the camera simply holds off.
fn own_wire_pose(art: &crate::articulation::CrabArticulation, me: PilotId) -> Option<CockpitPose> {
    art.vehicles
        .iter()
        .find(|v| v.pilot == me.0)
        .map(|v| CockpitPose {
            pos: Vec3::from_array(v.pos),
            orient: Quat::from_array(v.rot),
        })
}

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

pub(crate) fn park_fixed_auto_pump(world: &mut World) {
    world
        .resource_mut::<bevy::time::Time<bevy::time::Fixed>>()
        .set_timestep(std::time::Duration::from_secs(86_400));
}

fn restart_crab_to_spawn(world: &mut World, spawns: &[crate::sim::Pos]) {
    world
        .resource_mut::<crate::external_crab::ExternalCrabBridge>()
        .restart_to_spawns(spawns);
    crate::external_crab::cold_respawn_armed_crab(world);
}

pub(super) fn drive_lockstep(world: &mut World) {
    let armed = world
        .get_resource::<crate::external_crab::ExternalCrabArmed>()
        .is_some();

    let role = PeerRole::of(world.non_send_resource::<GameState>());
    let me = local_pilot(world.non_send_resource::<GameState>());

    let delta = world.resource::<Time>().delta().as_secs_f64();

    let (tel, roster_len) = {
        let state = world.non_send_resource::<GameState>();
        let tel = state.coord.telemetry().cloned();
        (tel, state.ls.sim().players().count())
    };

    world.non_send_resource_mut::<GameState>().accumulator += delta;

    {
        let toggle = std::mem::take(&mut world.resource_mut::<PendingInput>().toggle_vehicle);
        if toggle {
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
                let next = vehicle.cycled();
                info!("vehicle: {:?} -> {:?}", vehicle.context(), next.context());
                *vehicle = next;
            }
        }
    }

    let mut applied = 0u32;
    loop {
        {
            let state = world.non_send_resource::<GameState>();
            if state.accumulator < TICK_DT || applied >= MAX_TICKS_PER_FRAME {
                break;
            }
        }
        world.non_send_resource_mut::<GameState>().accumulator -= TICK_DT;
        applied += 1;

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

        let local = match world.resource::<LocalVehicle>().kind() {
            None => LocalControl::OnFoot(foot_input),
            Some(kind) => LocalControl::Piloting {
                kind,
                control: flight_control(kind, world.resource::<FlightInput>()),
            },
        };
        let sim_input = local.sim_input();
        let pilot_intent = local.pilot_intent();

        let scripted_pack: Option<Input> = world.get_resource::<ScriptedPackInput>().map(|r| r.0);
        let mut pending_art: Option<crate::articulation::CrabArticulation> = None;
        let mut server_down: Option<crate::net_loop::ServerDown> = None;
        let issue_tick = {
            let mut state = world.non_send_resource_mut::<GameState>();
            // Plain `&mut GameState` (not `Mut`) so the adopt closure below can borrow the
            // `reported_outcome` field disjointly from the `ls` it runs against.
            let state = &mut *state;
            let me = state.ls.me();
            let msg = state.ls.submit_local_input(sim_input, pilot_intent);
            // The tick this input was ISSUED as — the telemetry stamp. On a remote client this
            // differs from `ls.next_tick()` (the snapshot-apply cursor) by the transit lag.
            let issue_tick = msg.issue_tick;
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
                            pilot: None,
                        },
                    );
                }
            }
            let exch: Exchanged = match state.coord.exchange(msg) {
                Ok(exch) => exch,
                Err(down) => {
                    server_down = Some(down);
                    Exchanged::default()
                }
            };
            match role {
                PeerRole::ServerAuth => {}
                PeerRole::RemoteAdopt => {
                    state.snap_buf.extend(exch.snapshots);
                    state.art_buf.extend(exch.articulations);
                    let take = jitter_take(state.snap_buf.len());
                    if take > 0 {
                        state.prev = SimSnapshot::capture(&state.ls);
                        let reported_outcome = &mut state.reported_outcome;
                        let snaps: Vec<_> = state.snap_buf.drain(..take).collect();
                        state.ls.adopt_snapshots(snaps, |c| {
                            if c.sim().outcome() == Outcome::Ongoing {
                                *reported_outcome = false;
                            }
                        });
                        state.ls.reconcile_local_prediction();
                        let adopted_tick = state.ls.next_tick();
                        while state
                            .art_buf
                            .front()
                            .is_some_and(|a| a.tick <= adopted_tick)
                        {
                            pending_art = state.art_buf.pop_front();
                        }
                    }
                }
            }
            issue_tick
        };
        if let Some(art) = pending_art {
            crate::render::articulation::apply(world, &art);
            super::articulation::publish_remote_vehicles(world, &art.vehicles, me);
            if let Some(p) = own_wire_pose(&art, me) {
                let mut vehicle = world.resource_mut::<LocalVehicle>();
                if matches!(&*vehicle, LocalVehicle::Flying { pose: None, .. }) {
                    // The request→grant edge (rl#191): the host accepted our intent and our
                    // craft's first pose just crossed the wire — the cockpit camera engages.
                    info!("cockpit engaged: own craft pose arrived on the wire");
                }
                vehicle.update_pose(p);
            }
        }
        if let Some(down) = server_down {
            end_round_server_down(world, down, tel.as_ref());
            return;
        }

        if role == PeerRole::ServerAuth && world.get_resource::<VehicleControls>().is_some() {
            let entries: BTreeMap<PilotId, PilotCommand> = {
                let state = world.non_send_resource::<GameState>();
                let server = state.coord.server().expect("server_auth ⇒ a server");
                server
                    .pilot_intents()
                    .iter()
                    .map(|(&pid, intent)| (pilot_of(pid), intent.to_command()))
                    .collect()
            };
            world.resource_mut::<VehicleControls>().0 = entries;
        }

        loop {
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
            {
                let mut state = world.non_send_resource_mut::<GameState>();
                state.prev = SimSnapshot::capture(&state.ls);
            }

            {
                let crab_poses = if armed {
                    let steps = world
                        .non_send_resource_mut::<GameState>()
                        .cadence
                        .steps_for_next_tick();
                    pump_fixed_steps(world, steps);
                    let poses = world.resource_scope(
                        |world, mut bridge: Mut<crate::external_crab::ExternalCrabBridge>| {
                            let poses = bridge.crab_poses();
                            let state = world.non_send_resource::<GameState>();
                            for idx in 0..bridge.crab_count() {
                                let hunt = state
                                    .server()
                                    .and_then(|s| s.sim().nearest_living_player_pos(idx));
                                bridge.set_hunt_target(idx, hunt);
                            }
                            poses
                        },
                    );
                    if let Some(p) = read_vehicle_pose(world, me) {
                        world.resource_mut::<LocalVehicle>().update_pose(p);
                    }
                    poses
                } else {
                    Vec::new()
                };
                let (bytes, restarted) = {
                    let mut state = world.non_send_resource_mut::<GameState>();
                    let stepped = state
                        .server_mut()
                        .expect("server_auth ⇒ a server")
                        .step_next(&crab_poses);
                    if stepped.restarted {
                        state.cadence = PhysicsCadence::default();
                    }
                    (stepped.snapshot, stepped.restarted)
                };
                let snap = crate::snapshot::CoreSnapshot::from_bytes(&bytes)
                    .expect("the authoritative server's snapshot must decode");
                let articulation =
                    armed.then(|| crate::render::articulation::capture(world, snap.tick));
                {
                    let state = world.non_send_resource::<GameState>();
                    state.coord.broadcast_step(&snap, articulation.as_ref());
                }
                if let Some(art) = &articulation {
                    super::articulation::publish_remote_vehicles(world, &art.vehicles, me);
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
                    let spawns: Vec<crate::sim::Pos> = world
                        .non_send_resource::<GameState>()
                        .server()
                        .expect("server_auth ⇒ a server")
                        .sim()
                        .crabs()
                        .iter()
                        .map(|c| c.pos())
                        .collect();
                    restart_crab_to_spawn(world, &spawns);
                }
            }
        }

        if let Some(t) = &tel {
            let due = {
                let mut state = world.non_send_resource_mut::<GameState>();
                let due = state.ls.sim().tick() >= state.next_tel_tick;
                if due {
                    state.next_tel_tick = next_sample_tick(state.ls.sim().tick());
                    t.send(TelemetryEvent::tick(state.ls.sim(), roster_len));
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

    // Chronic input-starvation surface (rl#213): reports appear at most once per second per
    // player, so once per frame — after the tick drain — is plenty. A remote-adopt client has
    // no server and drains nothing.
    crate::telemetry::surface_starvation(
        world.non_send_resource_mut::<GameState>().server_mut(),
        tel.as_ref(),
    );

    if applied == MAX_TICKS_PER_FRAME {
        let mut state = world.non_send_resource_mut::<GameState>();
        state.accumulator = state.accumulator.min(TICK_DT);
    }

    let mut state = world.non_send_resource_mut::<GameState>();
    let state = &mut *state;
    for (pid, p) in state.ls.sim().players() {
        match state.logged_statuses.insert(pid, p.status()) {
            Some(prev) if prev != p.status() => {
                info!("player {:?}: {:?} -> {:?}", pid, prev, p.status());
            }
            _ => {}
        }
    }
    state.logged_statuses.retain(|pid, _| state.ls.sim().player(*pid).is_some());
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
    use super::{FlightControl, JITTER_BUF_MAX, JITTER_BUF_TARGET, LocalControl, jitter_take};
    use crate::sim::{Input, buttons};
    use crab_world::vehicle::VehicleKind;

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
    fn own_wire_pose_picks_exactly_our_pilots_craft() {
        use crate::articulation::{CrabArticulation, VehiclePoseWire};
        use crab_world::vehicle::PilotId;
        let art = CrabArticulation {
            tick: 7,
            crabs: Vec::new(),
            vehicles: vec![
                VehiclePoseWire {
                    pilot: 0,
                    pos: [1.0, 2.0, 3.0],
                    rot: [0.0, 0.0, 0.0, 1.0],
                },
                VehiclePoseWire {
                    pilot: 2,
                    pos: [9.0, 8.0, 7.0],
                    rot: [0.0, 1.0, 0.0, 0.0],
                },
            ],
        };
        let ours = super::own_wire_pose(&art, PilotId(2)).expect("our craft is on the wire");
        assert_eq!(ours.pos.to_array(), [9.0, 8.0, 7.0]);
        assert_eq!(ours.orient.to_array(), [0.0, 1.0, 0.0, 0.0]);
        assert!(
            super::own_wire_pose(&art, PilotId(1)).is_none(),
            "no craft on the wire = the request→grant window: the camera holds off"
        );
    }

    #[test]
    fn jitter_take_paces_one_and_catches_down() {
        assert_eq!(jitter_take(0), 0, "empty ⇒ hold last state");
        for buffered in 1..=JITTER_BUF_MAX {
            assert_eq!(jitter_take(buffered), 1, "in-margin ⇒ even pacing");
        }
        assert_eq!(
            jitter_take(JITTER_BUF_MAX + 1),
            JITTER_BUF_MAX + 1 - JITTER_BUF_TARGET,
            "past the margin ⇒ drain to the target in one tick"
        );
        assert_eq!(jitter_take(10), 10 - JITTER_BUF_TARGET);
    }
}
