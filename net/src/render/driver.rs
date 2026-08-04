//! The windowed round driver: installs and tears down a round's world state through
//! the one round-scope registry, paces the fixed-tick loop ([`drive_client_sim`]),
//! bridges local input to the sim and the crab world's vehicles, and writes the
//! frame's [`RenderClock`] — THE clock every pose sampler reads.

use super::app::{ArmedRound, NnCrabStackInstalled};
use super::input::{CameraPitch, CameraYaw};
use super::pose::{Pose, PoseWindow};
use super::*;
use crate::client::PilotIntent;
use crab_world::vehicle::{PilotCommand, PilotId, Vehicle, VehicleControls, VehicleKind};

pub(super) fn coordinator(
    net: Option<NetDriver>,
    peers: &[PlayerId],
    me: PlayerId,
    initial_sim: crate::sim::Sim,
) -> Box<Coordinator> {
    Box::new(Coordinator::for_round(net, peers, me, initial_sim))
}

pub(super) fn insert_core(app: &mut App, client: ClientSim, coord: Box<Coordinator>) {
    install_round(app.world_mut(), client, coord);
}

/// Everything round-scoped, torn down by [`teardown_round`]. The thunks are registered
/// AT INSTALL, each adjacent to the state it scopes — one list, so install and teardown
/// can no longer be hand-maintained parallel lists that must deliberately not match:
/// rl#211, rl#258, and rl#274/rl#303 were each a stale survivor someone forgot to add
/// to the teardown twin, point-patched there; the registry deletes the twin (rl#336).
#[derive(Resource, Default)]
struct RoundScope(Vec<fn(&mut World)>);

/// Insert a freshly-defaulted round resource and register its teardown in the same
/// breath: reset to default at round end, so nothing stale survives into the menu or
/// under an ungated system until the next install.
fn round_resource<R: Resource + Default>(world: &mut World, scope: &mut Vec<fn(&mut World)>) {
    world.insert_resource(R::default());
    scope.push(|w| w.insert_resource(R::default()));
}

fn install_round(world: &mut World, client: ClientSim, coord: Box<Coordinator>) {
    let mut scope: Vec<fn(&mut World)> = Vec::new();
    let prev = SimSnapshot::capture(&client);
    world.insert_non_send_resource(GameState {
        client,
        coord,
        accumulator: 0.0,
        prev,
        reported_outcome: false,
        next_tel_tick: next_sample_tick(0),
        snap_buf: std::collections::VecDeque::new(),
        art_buf: std::collections::VecDeque::new(),
        stalled: false,
        logged_statuses: BTreeMap::new(),
    });
    scope.push(|w| {
        w.remove_non_send_resource::<GameState>();
    });
    round_resource::<PendingInput>(world, &mut scope);
    round_resource::<FlightInput>(world, &mut scope);
    round_resource::<CameraPitch>(world, &mut scope);
    round_resource::<CameraYaw>(world, &mut scope);
    round_resource::<LocalVehicle>(world, &mut scope);
    round_resource::<RenderClock>(world, &mut scope);
    round_resource::<super::articulation::RemoteVehicle>(world, &mut scope);
    round_resource::<super::articulation::CrabPartWindows>(world, &mut scope);
    round_resource::<crab_world::bot::skin::CrabRenderPose>(world, &mut scope);

    // Round state MADE elsewhere during arm/play, not inserted here — its teardown still
    // lives in this one registry:
    scope.push(|w| {
        w.remove_resource::<crate::crab_slot::NnCrabsArmed>();
    });
    // Un-label the crab bodies that persist across rounds: labels are round state (the host
    // republishes on the next arm; a client re-adopts from the next round's articulation),
    // and a survivor here would float brain labels over the menu (the rl#211 class).
    scope.push(|w| {
        if let Some(mut labels) = w.get_resource_mut::<crab_world::crab_view::CrabBrainLabels>() {
            labels.0.clear();
        }
    });
    scope.push(|w| {
        if let Some(mut ctrl) = w.get_resource_mut::<VehicleControls>() {
            ctrl.0.clear();
        }
    });
    // Craft bodies are round state like the controls that spawned them: FixedUpdate is
    // parked outside Playing, so a survivor would sit at its stale pose until the next
    // round's first pump — and a board landing on that round's first tick would match it
    // there instead of transforming at the walker (rl#258).
    scope.push(|w| {
        let crafts: Vec<Entity> = w
            .query_filtered::<Entity, With<Vehicle>>()
            .iter(w)
            .collect();
        for e in crafts {
            w.despawn(e);
        }
    });
    // A survivor would suppress (or mis-measure) the next round's remote-craft
    // appeared/moved edges.
    scope.push(|w| {
        w.remove_resource::<super::articulation::RemoteCraftWatch>();
    });
    world.insert_resource(RoundScope(scope));
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
        world.get_resource::<NnCrabStackInstalled>().is_some(),
        "the NN-crab stack must be installed before Playing (rl#114: the checkpoint is required)"
    );
    let spawns = match world.get_non_send_resource::<crate::crab_slot::CrabPolicies>() {
        Some(p) => super::app::seed_round_crabs(&mut ready.client, p.0.len()),
        None => Vec::new(),
    };
    crate::crab_slot::arm(world);
    if !spawns.is_empty() {
        crate::crab_slot::restart_crabs_to_spawns(world, &spawns);
    }
    let coord = coordinator(
        ready.net,
        ready.client.peers(),
        ready.client.me(),
        ready.client.sim().clone(),
    );
    install_round(world, ready.client, coord);
}

pub(super) fn teardown_round(world: &mut World) {
    let Some(RoundScope(scope)) = world.remove_resource::<RoundScope>() else {
        return;
    };
    for teardown in scope {
        teardown(world);
    }
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

/// The sim↔world correspondence, both directions (rl#258; one frame since rl#298
/// stage 5 — the world's coordinates ARE the sim's meters): a sim point stands ON the
/// ground via [`crab_world::terrain::TerrainGrid::place`] — THE spawn-on-surface
/// primitive, not a reimplementation — so a boarding on a mountainside authors its
/// craft at the walker's real elevation (rl#281 stage 6; on the flat grids the height
/// is exactly 0). `world_to_sim` is planar (drops y), so the two stay inverses.
fn sim_to_world(pos: crate::sim::Pos, terrain: &crab_world::terrain::TerrainGrid) -> Vec3 {
    let (x, z) = pos.to_meters();
    terrain.place(Vec2::new(x, z), 0.0)
}

fn world_to_sim(world: Vec3) -> crate::sim::Pos {
    crate::sim::Pos::from_meters(world.x, world.z)
}

/// The boarding player's walker state in world frame (rl#258): where the
/// craft must materialise, its facing, and the velocity to conserve. `prev` is the
/// walker one tick earlier (its last step is the velocity); recomputed per tick, but read
/// only on the spawn edge — and while piloting the walker already rides the craft.
fn boarding_of(
    now: crate::sim::Player,
    prev: crate::sim::Player,
    terrain: &crab_world::terrain::TerrainGrid,
) -> crab_world::vehicle::Boarding {
    let here = sim_to_world(now.pos(), terrain);
    let velocity = (here - sim_to_world(prev.pos(), terrain)) / TICK_DT as f32;
    // A walker can't out-run its walk speed: a bigger per-tick delta is a TELEPORT (the
    // round-RESTART respawn, a join slot), not motion — a craft boarded right after one
    // must start at rest, not inherit a cross-map fling.
    let max_walk = 2.0 * (crate::sim::PLAYER_SPEED as f32 * crate::sim::TICK_HZ as f32)
        / crate::sim::UNIT as f32;
    let velocity = if velocity.length() <= max_walk {
        velocity
    } else {
        Vec3::ZERO
    };
    crab_world::vehicle::Boarding {
        pos: here,
        yaw: crate::sim::trig_client::turns_to_radians(now.yaw()),
        velocity,
    }
}

/// Every spawned craft's pose bridged back into sim space — the per-tick pilot-follow
/// feed for [`crate::server::Server::step_next`] (rl#258): a piloting player's walker
/// rides its craft, so the sim never keeps a husk at the boarding spot.
fn pilot_shadows(world: &mut World) -> BTreeMap<PlayerId, crate::sim::PilotPose> {
    let mut q = world.query::<(&Transform, &Vehicle)>();
    q.iter(world)
        .map(|(t, v)| {
            let nose = t.rotation * Vec3::Z;
            (
                PlayerId(v.pilot.0),
                crate::sim::PilotPose {
                    pos: world_to_sim(t.translation),
                    yaw: crate::sim::trig_client::radians_to_turns(nose.x.atan2(nose.z)),
                },
            )
        })
        .collect()
}

fn local_pilot(state: &GameState) -> PilotId {
    pilot_of(state.client.me())
}

pub(super) struct GameState {
    pub(super) client: ClientSim,
    pub(super) coord: Box<Coordinator>,
    pub(super) accumulator: f64,
    pub(super) prev: SimSnapshot,
    /// Round-decided latch: set when this round's decided outcome has been reported, cleared
    /// per Ongoing snapshot inside [`drive_client_sim`]'s drain arms (a RESTART revives a decided
    /// round without rewinding the tick, rl#204). Lives here — not a system `Local` — so a new
    /// round starts unlatched by construction (rl#210).
    reported_outcome: bool,
    /// The next telemetry sampling boundary ([`next_sample_tick`]), in this ROUND's ticks.
    /// Per-round for the same reason: a fresh ClientSim restarts at tick 0, so a surviving
    /// watermark would suppress tick telemetry until the counter climbed past it (rl#210).
    next_tel_tick: u64,
    snap_buf: std::collections::VecDeque<crate::snapshot::CoreSnapshot>,
    art_buf: std::collections::VecDeque<crate::articulation::CrabArticulation>,
    /// Snapshot-stall latch (rl#273): the last remote-adopt drain iteration consumed a tick
    /// of render time but adopted nothing. While set, [`render_frac`] pins the clock at the
    /// end of the last adopted interval instead of wrapping — a stall renders as a clean
    /// hold, not a 30 Hz replay. Always false on the host: its drain steps every tick.
    stalled: bool,
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

/// The [`RenderClock`] fraction for this frame. Stalled (rl#273), the drain keeps
/// consuming `accumulator -= TICK_DT` to pace input submission while the sim tick
/// freezes, so the raw fraction would sweep 0→1 and wrap ~30×/s — every interpolated
/// surface replaying its last tick interval. Pinning at 1.0 holds the end of the last
/// adopted interval. Entry advances render time by at most one frame's sweep — the
/// same quantization every normal tick-crossing frame has — and recovery captures
/// `prev` at exactly the held pose before adopting, so the resume edge is seamless.
fn render_frac(accumulator: f64, stalled: bool) -> f32 {
    if stalled {
        1.0
    } else {
        (accumulator / TICK_DT).clamp(0.0, 1.0) as f32
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
    fn capture(client: &ClientSim) -> Self {
        let snap = client.core_snapshot();
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
    pub(super) vehicle: Option<VehicleRequest>,
}

/// A chord-issued vehicle change (rl#330): each vehicle has its own board code and
/// exit is a code of its own — there is no cycle verb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VehicleRequest {
    Board(VehicleKind),
    Exit,
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
pub(super) struct PlaneControl {
    pub throttle_trim: f32,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
}

#[derive(Debug, Default, PartialEq)]
pub(super) struct ShipControl {
    pub thrust: Vec3,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub match_velocity: bool,
}

/// Per-kind control surfaces — an enum, not a field union with the kind held apart, so
/// a control no kind can fly (a plane matching velocity, a ship trimming throttle) is
/// unrepresentable (rl#336). [`PilotIntent`] stays the wire-side union; the two meet
/// only in [`LocalControl::pilot_intent`].
#[derive(Debug, PartialEq)]
pub(super) enum FlightControl {
    Plane(PlaneControl),
    Ship(ShipControl),
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
            FlightControl::Plane(PlaneControl {
                throttle_trim,
                pitch,
                roll,
                yaw,
            })
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
            FlightControl::Ship(ShipControl {
                thrust,
                pitch,
                roll,
                yaw,
                match_velocity: fi.match_vel,
            })
        }
    }
}

enum LocalControl {
    OnFoot(Input),
    Piloting {
        control: FlightControl,
        /// This tick's foot input — only its pilot-surviving parts
        /// ([`Input::pilot_masked`]) reach the sim while the craft flies on `control`.
        foot: Input,
    },
}

impl LocalControl {
    fn sim_input(&self) -> Input {
        match self {
            LocalControl::OnFoot(input) => *input,
            LocalControl::Piloting { foot, .. } => foot.pilot_masked(),
        }
    }

    fn pilot_intent(&self) -> Option<PilotIntent> {
        let LocalControl::Piloting { control, .. } = self else {
            return None;
        };
        Some(match control {
            FlightControl::Plane(PlaneControl {
                throttle_trim,
                pitch,
                roll,
                yaw,
            }) => PilotIntent {
                kind: VehicleKind::Plane,
                throttle_trim: *throttle_trim,
                thrust: [0.0; 3],
                pitch: *pitch,
                roll: *roll,
                yaw: *yaw,
                match_velocity: false,
            },
            FlightControl::Ship(ShipControl {
                thrust,
                pitch,
                roll,
                yaw,
                match_velocity,
            }) => PilotIntent {
                kind: VehicleKind::Ship,
                throttle_trim: 0.0,
                thrust: thrust.to_array(),
                pitch: *pitch,
                roll: *roll,
                yaw: *yaw,
                match_velocity: *match_velocity,
            },
        })
    }
}

/// This frame's render time, written once per frame at the end of
/// [`drive_client_sim`]: the last stepped/adopted sim tick plus the accumulator
/// fraction into the next. THE one clock every [`super::pose::PoseWindow`] sampler
/// reads — the cockpit, the remote craft models, the wireframe pass, and the crab
/// body parts (both arms, rl#274) — so their notions of "now" cannot drift apart
/// within a frame.
#[derive(Resource, Clone, Copy, Default)]
pub(super) struct RenderClock {
    pub tick: u64,
    pub frac: f32,
}

#[derive(Resource, Default)]
pub(super) enum LocalVehicle {
    #[default]
    OnFoot,
    Flying {
        kind: VehicleKind,
        poses: PoseWindow,
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

    pub(super) fn cockpit_sample(&self, now_tick: u64, tick_frac: f32) -> Option<Pose> {
        match self {
            Self::OnFoot => None,
            Self::Flying { poses, .. } => poses.sample(now_tick, tick_frac),
        }
    }

    fn update_pose(&mut self, tick: u64, p: Pose) {
        if let Self::Flying { poses, .. } = self {
            poses.push(tick, p);
        }
    }
}

/// Read OUR OWN vehicle rigidbody's world-frame pose (position + attitude) from the crab
/// world, or `None` if none is spawned (on foot, or the frame a freshly-boarded body
/// hasn't appeared yet). At most one body per pilot — `manage_vehicles` enforces it.
/// Keyed by pilot because the host's world will also carry REMOTE pilots' crafts
/// (rl#191): the cockpit camera must fly from ours alone.
fn read_vehicle_pose(world: &mut World, me: PilotId) -> Option<Pose> {
    let mut q = world.query::<(&Transform, &Vehicle)>();
    q.iter(world)
        .find(|(_, v)| v.pilot == me)
        .map(|(t, _)| Pose {
            pos: t.translation,
            orient: t.rotation,
        })
}

/// The remote-client twin of [`read_vehicle_pose`]: our own craft's world-frame pose out
/// of an adopted articulation. The host simulates the craft; its pose comes back
/// per-pilot on the wire. `None` keeps the cockpit camera off while the pose ring is
/// empty — usually the request→grant window, but also a silent HOST REFUSAL
/// (`file_intent` saw [`crate::sim::PlayerStatus::may_board`] false; the client can't
/// tell the two apart) and the unarmed round (no articulation at all). Either way the
/// camera simply holds off.
fn own_wire_pose(art: &crate::articulation::CrabArticulation, me: PilotId) -> Option<Pose> {
    art.vehicles.iter().find(|v| v.pilot == me.0).map(|v| Pose {
        pos: Vec3::from_array(v.pos),
        orient: Quat::from_array(v.rot),
    })
}

/// Apply a chord-issued [`VehicleRequest`], if one is pending: boarding from foot needs
/// the sim's leave, switching craft-to-craft doesn't, and re-requesting the current
/// craft is a no-op (no pose-window reset mid-flight).
fn apply_vehicle_request(world: &mut World) {
    let Some(request) = world.resource_mut::<PendingInput>().vehicle.take() else {
        return;
    };
    let may_board = {
        let state = world.non_send_resource::<GameState>();
        let me = state.client.me();
        state
            .client
            .sim()
            .player(me)
            .is_some_and(|p| p.status().may_board())
    };
    let mut vehicle = world.resource_mut::<LocalVehicle>();
    let next = match request {
        VehicleRequest::Board(kind)
            if vehicle.kind() != Some(kind) && (vehicle.kind().is_some() || may_board) =>
        {
            Some(LocalVehicle::Flying {
                kind,
                poses: PoseWindow::default(),
            })
        }
        VehicleRequest::Exit if vehicle.kind().is_some() => Some(LocalVehicle::OnFoot),
        _ => None,
    };
    if let Some(next) = next {
        info!("vehicle: {:?} -> {:?}", vehicle.context(), next.context());
        *vehicle = next;
    }
}

/// Assemble this tick's [`LocalControl`] from the pending foot input (drained: the yaw
/// delta, action, and restart taps are consumed here, once per tick) and, piloting, the
/// flight input mapped through [`flight_control`].
fn local_control(world: &mut World) -> LocalControl {
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
    match world.resource::<LocalVehicle>().kind() {
        None => LocalControl::OnFoot(foot_input),
        Some(kind) => LocalControl::Piloting {
            control: flight_control(kind, world.resource::<FlightInput>()),
            foot: foot_input,
        },
    }
}

/// What one [`Coordinator::exchange`] round-trip produced for the rest of the frame.
struct TickOutcome {
    /// The tick this input was ISSUED as — the telemetry stamp. On a remote client this
    /// differs from `client.next_tick()` (the snapshot-apply cursor) by the transit lag.
    issue_tick: u64,
    /// The adopt-arm articulation to publish this tick (remote client only).
    articulation: Option<crate::articulation::CrabArticulation>,
    server_down: Option<crate::net_loop::ServerDown>,
}

/// Submit the local input and run this tick's coordinator exchange. On the remote-adopt
/// arm, also pace the jitter buffer and adopt what it releases (capturing `prev` for
/// interpolation and reconciling the local prediction).
fn exchange_tick(world: &mut World, role: PeerRole, local: &LocalControl) -> TickOutcome {
    let scripted_pack: Option<Input> = world.get_resource::<ScriptedPackInput>().map(|r| r.0);
    let mut state = world.non_send_resource_mut::<GameState>();
    // Plain `&mut GameState` (not `Mut`) so the adopt closure below can borrow the
    // `reported_outcome` field disjointly from the `client` it runs against.
    let state = &mut *state;
    let me = state.client.me();
    let msg = state
        .client
        .submit_local_input(local.sim_input(), local.pilot_intent());
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
    let mut server_down = None;
    let exch: Exchanged = match state.coord.exchange(msg) {
        Ok(exch) => exch,
        Err(down) => {
            server_down = Some(down);
            Exchanged::default()
        }
    };
    let mut articulation = None;
    match role {
        PeerRole::ServerAuth => {}
        PeerRole::RemoteAdopt => {
            state.snap_buf.extend(exch.snapshots);
            state.art_buf.extend(exch.articulations);
            let take = jitter_take(state.snap_buf.len());
            state.stalled = take == 0;
            if take > 0 {
                state.prev = SimSnapshot::capture(&state.client);
                let reported_outcome = &mut state.reported_outcome;
                let snaps: Vec<_> = state.snap_buf.drain(..take).collect();
                state.client.adopt_snapshots(snaps, |c| {
                    if c.sim().outcome() == Outcome::Ongoing {
                        *reported_outcome = false;
                    }
                });
                state.client.reconcile_local_prediction();
                let adopted_tick = state.client.next_tick();
                while state
                    .art_buf
                    .front()
                    .is_some_and(|a| a.tick <= adopted_tick)
                {
                    articulation = state.art_buf.pop_front();
                }
            }
        }
    }
    TickOutcome {
        issue_tick,
        articulation,
        server_down,
    }
}

/// Publish an adopted articulation to the render surfaces: crab poses, remote craft
/// models, and — if OUR craft's pose just crossed the wire — the cockpit window.
fn adopt_wire_articulation(
    world: &mut World,
    art: &crate::articulation::CrabArticulation,
    me: PilotId,
) {
    crate::render::articulation::adopt(world, art);
    super::articulation::publish_remote_vehicles(world, art.tick, &art.vehicles, me);
    if let Some(p) = own_wire_pose(art, me) {
        let mut vehicle = world.resource_mut::<LocalVehicle>();
        if matches!(&*vehicle, LocalVehicle::Flying { poses, .. } if poses.is_empty()) {
            // The request→grant edge (rl#191): the host accepted our intent and our
            // craft's first pose just crossed the wire — the cockpit camera engages.
            info!("cockpit engaged: own craft pose arrived on the wire");
        }
        vehicle.update_pose(art.tick, p);
    }
}

/// (Host) Bridge every pilot's intent into a [`PilotCommand`] for the crab world's
/// vehicle systems, anchored at each walker's [`boarding_of`] state this tick.
fn publish_pilot_commands(world: &mut World) {
    if world.get_resource::<VehicleControls>().is_none() {
        return;
    }
    let terrain = world.resource::<crab_world::terrain::Terrain>().clone();
    let entries: BTreeMap<PilotId, PilotCommand> = {
        let state = world.non_send_resource::<GameState>();
        let server = state.coord.server().expect("server_auth ⇒ a server");
        server
            .pilot_intents()
            .iter()
            .filter_map(|(&pid, intent)| {
                // No sim player (a departure racing its roster shrink) ⇒ no walker
                // to transform — the intent simply files no command this tick.
                let now = server.sim().player(pid)?;
                let prev = state.prev.players.get(&pid).copied().unwrap_or(now);
                let boarding = boarding_of(now, prev, &terrain);
                Some((pilot_of(pid), intent.to_command(boarding)))
            })
            .collect()
    };
    world.resource_mut::<VehicleControls>().0 = entries;
}

/// (Host) Drain every server tick this frame's exchange made ready: pump the crab
/// world, step the authoritative sim, broadcast the snapshot + articulation, and mirror
/// the step into our own client. Host-paced: `exchange` assembled one tick per issued
/// local input, so this drains exactly what this frame issued (a remote can delay
/// nothing — rl#195).
fn pump_host_ticks(world: &mut World, me: PilotId, armed: bool) {
    loop {
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
            state.prev = SimSnapshot::capture(&state.client);
        }

        let inputs = {
            let state = world.non_send_resource::<GameState>();
            crate::crab_slot::slot_inputs(state.server().expect("server_auth ⇒ a server").sim())
        };
        let (crab_poses, shadows) = if armed {
            let poses = crate::crab_slot::pump_crab_slot(world, &inputs);
            if let Some(p) = read_vehicle_pose(world, me) {
                world
                    .resource_mut::<LocalVehicle>()
                    .update_pose(inputs.stepping_into, p);
            }
            (poses, pilot_shadows(world))
        } else {
            // The one unarmed host: the crab-less screenshot path
            // (fp-screenshot without a checkpoint) — no crab world to read,
            // so the sim's crabs hold their spawn poses, clawless. Poses stay
            // mandatory; this is a diagnostics surface, not a served round.
            (inputs.fallback.clone(), BTreeMap::new())
        };
        let (bytes, restarted) = {
            let mut state = world.non_send_resource_mut::<GameState>();
            let stepped = state
                .server_mut()
                .expect("server_auth ⇒ a server")
                .step_next(&crab_poses, shadows);
            (stepped.snapshot, stepped.restarted)
        };
        let snap = crate::snapshot::CoreSnapshot::from_bytes(&bytes)
            .expect("the authoritative server's snapshot must decode");
        let articulation = armed.then(|| crate::render::articulation::capture(world, snap.tick));
        {
            let state = world.non_send_resource::<GameState>();
            state.coord.broadcast_step(&snap, articulation.as_ref());
        }
        if let Some(art) = &articulation {
            // The host renders its OWN Sally through the same windows a client
            // adopts (rl#274) — one interpolation mechanism on both arms.
            super::articulation::feed_crab_part_windows(world, art);
            super::articulation::publish_remote_vehicles(world, art.tick, &art.vehicles, me);
        }
        {
            let mut state = world.non_send_resource_mut::<GameState>();
            // Same per-tick latch clear as the adopt arm: an Ongoing tick means the
            // round is (or just became, via RESTART) live, so the next decision must
            // report even if this frame's unbounded catch-up drain also re-decides it.
            if snap.outcome == Outcome::Ongoing {
                state.reported_outcome = false;
            }
            state.client.apply_core_snapshot(snap);
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
            crate::crab_slot::restart_crabs_to_spawns(world, &spawns);
        }
    }
}

/// Emit the per-tick telemetry sample when the tick counter crosses the boundary, and —
/// on a sampling tick — drain the crab-rescue counters into per-window Fault events.
fn sample_telemetry(
    world: &mut World,
    t: &crate::telemetry::TelemetrySender,
    roster_len: usize,
    issue_tick: u64,
    sim_input: Input,
) {
    let due = {
        let mut state = world.non_send_resource_mut::<GameState>();
        let due = state.client.sim().tick() >= state.next_tel_tick;
        if due {
            state.next_tel_tick = next_sample_tick(state.client.sim().tick());
            t.send(TelemetryEvent::tick(state.client.sim(), roster_len));
            t.send(TelemetryEvent::input(issue_tick, sim_input));
        }
        due
    };
    // Aggregated rescue surface: drain the window's `rescue_lost_crabs`
    // tally into per-window Fault events carrying the count + last offending body,
    // so a frame-by-frame blowup shows on the hub feed as a filtered per-window
    // count instead of a per-step flood. One event PER REASON — a legitimate
    // hard-hit tunneling rescue (rl#283) reported as "going non-finite" would be a
    // false rl#137 alarm. A stable solo Sally never enters this branch (both
    // counters stay 0) — a nonzero count IS the alarm.
    if due
        && let Some(mut stats) = world.get_resource_mut::<crab_world::bot::RescueStats>()
        && (stats.since_nonfinite > 0 || stats.since_below_terrain > 0 || stats.since_buried > 0)
    {
        let (nf, bt, bu) = (
            stats.since_nonfinite,
            stats.since_below_terrain,
            stats.since_buried,
        );
        // `last_body` is reason-blind, so name it only when one reason fired —
        // else the rl#137 event could name a below-terrain offender or vice versa.
        let last = match stats.last_body {
            Some(b) if [nf, bt, bu].iter().filter(|&&n| n > 0).count() == 1 => {
                format!(" (last offender: {b})")
            }
            _ => String::new(),
        };
        stats.since_nonfinite = 0;
        stats.since_below_terrain = 0;
        stats.since_buried = 0;
        if nf > 0 {
            t.send(TelemetryEvent::Fault {
                msg: format!(
                    "crab rescue: {nf} non-finite respawn(s) this telemetry \
                     window{last} — armed Sally is going non-finite (rl#137)"
                ),
            });
        }
        if bt > 0 {
            t.send(TelemetryEvent::Fault {
                msg: format!(
                    "crab rescue: {bt} below-terrain respawn(s) this telemetry \
                     window{last} — fell below the terrain surface, tunneled or \
                     off the tile edge (rl#283 y-floor)"
                ),
            });
        }
        if bu > 0 {
            t.send(TelemetryEvent::Fault {
                msg: format!(
                    "crab rescue: {bu} buried-carapace respawn(s) this telemetry \
                     window{last} — carapace pinned under the one-sided \
                     heightfield sheet (rl#303)"
                ),
            });
        }
    }
}

/// Log each player's status edge exactly once per transition, on every peer, and report
/// the round's decided outcome once per decision (the latch clears per Ongoing tick in
/// the drain arms — a RESTART revives a decided round, rl#204).
fn log_round_edges(world: &mut World, tel: Option<&crate::telemetry::TelemetrySender>) {
    let mut state = world.non_send_resource_mut::<GameState>();
    let state = &mut *state;
    for (pid, p) in state.client.sim().players() {
        match state.logged_statuses.insert(pid, p.status()) {
            Some(prev) if prev != p.status() => {
                // A down is always a claw touch (rl#236 — no under-body disc); the crab
                // distance locates where near her body the strike landed.
                let (px, pz) = p.pos().to_meters();
                let from_crab = state
                    .client
                    .sim()
                    .crabs()
                    .iter()
                    .map(|c| {
                        let (cx, cz) = c.pos().to_meters();
                        (px - cx).hypot(pz - cz)
                    })
                    .fold(f32::INFINITY, f32::min);
                info!(
                    "player {:?}: {:?} -> {:?} ({from_crab:.2} m from crab center)",
                    pid,
                    prev,
                    p.status()
                );
            }
            _ => {}
        }
    }
    state
        .logged_statuses
        .retain(|pid, _| state.client.sim().player(*pid).is_some());
    let outcome = state.client.sim().outcome();
    if !state.reported_outcome && outcome != Outcome::Ongoing {
        state.reported_outcome = true;
        info!("round decided: {outcome:?}");
        if let Some(t) = tel {
            t.send(TelemetryEvent::round_decided(state.client.sim()));
        }
    }
}

/// The per-frame fixed-tick driver: accumulate render time, then per crossed tick
/// assemble local control, exchange it through the [`Coordinator`], and run the
/// role's arm — the host pumps + steps + broadcasts ([`pump_host_ticks`]); a remote
/// client adopts what [`exchange_tick`]'s jitter buffer released. Ends by writing the
/// frame's [`RenderClock`] and perf stats.
pub(super) fn drive_client_sim(world: &mut World) {
    let sim_started = std::time::Instant::now();
    let armed = world
        .get_resource::<crate::crab_slot::NnCrabsArmed>()
        .is_some();
    let role = PeerRole::of(world.non_send_resource::<GameState>());
    let me = local_pilot(world.non_send_resource::<GameState>());
    let delta = world.resource::<Time>().delta().as_secs_f64();
    let (tel, roster_len) = {
        let state = world.non_send_resource::<GameState>();
        let tel = state.coord.telemetry().cloned();
        (tel, state.client.sim().players().count())
    };

    world.non_send_resource_mut::<GameState>().accumulator += delta;

    apply_vehicle_request(world);

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

        let local = local_control(world);
        let sim_input = local.sim_input();
        let tick = exchange_tick(world, role, &local);
        if let Some(art) = tick.articulation {
            adopt_wire_articulation(world, &art, me);
        }
        if let Some(down) = tick.server_down {
            end_round_server_down(world, down, tel.as_ref());
            return;
        }

        if role == PeerRole::ServerAuth {
            publish_pilot_commands(world);
            pump_host_ticks(world, me, armed);
        }

        if let Some(t) = &tel {
            sample_telemetry(world, t, roster_len, tick.issue_tick, sim_input);
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

    log_round_edges(world, tel.as_ref());

    let clock = {
        let state = world.non_send_resource::<GameState>();
        RenderClock {
            tick: state.client.sim().tick(),
            frac: render_frac(state.accumulator, state.stalled),
        }
    };
    world.insert_resource(clock);

    // Feed the rl#331 perf readout/black box: this frame's whole-sim cost and how many
    // fixed ticks ran (pinned at MAX_TICKS_PER_FRAME = the sim can't keep up with wall
    // time — the death-spiral signature, distinct from a render/present stall).
    *world.resource_mut::<crab_world::debug_overlay::SimFrameStats>() =
        crab_world::debug_overlay::SimFrameStats {
            ms: sim_started.elapsed().as_secs_f32() * 1000.0,
            ticks: applied,
        };
}

#[cfg(test)]
mod tests {
    use super::{
        FlightControl, JITTER_BUF_MAX, JITTER_BUF_TARGET, LocalControl, PlaneControl, jitter_take,
        render_frac,
    };
    use crate::sim::{Input, buttons};

    #[test]
    fn piloting_feeds_the_sim_a_neutral_foot_input_except_restart() {
        let walk = Input::new(0.5, -0.5, 0.25, buttons::ACTION);
        assert_eq!(
            LocalControl::OnFoot(walk).sim_input(),
            walk,
            "on foot the real walker input drives the sim unchanged"
        );
        let flying = |foot| LocalControl::Piloting {
            control: FlightControl::Plane(PlaneControl {
                throttle_trim: 1.0,
                pitch: 1.0,
                roll: -1.0,
                yaw: 1.0,
            }),
            foot,
        };
        assert_eq!(
            flying(walk).sim_input(),
            Input::default(),
            "piloting: walk axes and ACTION never reach the sim, nor any flight axis"
        );
        assert_eq!(
            flying(Input::new(
                1.0,
                0.0,
                0.0,
                buttons::RESTART | buttons::ACTION
            ))
            .sim_input(),
            Input::new(0.0, 0.0, 0.0, buttons::RESTART),
            "RESTART is available in every context (rl#261) — it alone rides along"
        );
    }

    /// The two directions of the rl#258 sim↔world conversion (boarding spawn vs pilot
    /// follow) must be exact inverses, or a board+exit round-trip would drift the
    /// player — including on terrain, where sim_to_world lifts to the surface
    /// (world_to_sim is planar, so the lift can't leak back).
    #[test]
    fn sim_world_conversion_roundtrips() {
        let p = crate::sim::Pos {
            x: 12_340,
            z: -5_670,
        };
        let flat = crab_world::terrain::TerrainGrid::flat(16.0);
        let gcr = crab_world::terrain::TerrainGrid::gcr();
        for terrain in [&flat, &*gcr] {
            let world = super::sim_to_world(p, terrain);
            let back = super::world_to_sim(world);
            assert!(
                (back.x - p.x).abs() <= 1 && (back.z - p.z).abs() <= 1,
                "sim→world→sim drifted beyond grid quantization: {p:?} -> {back:?}"
            );
        }
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
                    kind: crab_world::vehicle::VehicleKind::Plane,
                    pos: [1.0, 2.0, 3.0],
                    rot: [0.0, 0.0, 0.0, 1.0],
                    thrust: [0, 0, 0],
                },
                VehiclePoseWire {
                    pilot: 2,
                    kind: crab_world::vehicle::VehicleKind::Ship,
                    pos: [9.0, 8.0, 7.0],
                    rot: [0.0, 1.0, 0.0, 0.0],
                    thrust: [0, 0, 0],
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

    #[test]
    fn render_frac_holds_at_one_through_a_snapshot_stall() {
        use crate::sim::TICK_DT;
        assert_eq!(render_frac(0.0, false), 0.0);
        assert_eq!(render_frac(TICK_DT * 0.5, false), 0.5);
        // The rl#273 wrap: a stalled drain keeps consuming TICK_DT to pace input
        // submission, so the raw fraction would rewind to ~0 here and replay the
        // last tick interval at 30 Hz. Pinned, the stall renders as a hold.
        assert_eq!(render_frac(TICK_DT * 0.02, true), 1.0);
        assert_eq!(render_frac(TICK_DT, false), 1.0);
        // The latch wiring itself (`stalled = take == 0` in the RemoteAdopt drain
        // arm, cleared by any adopting iteration and surviving zero-drain frames) is
        // not unit-tested: a remote-adopt GameState needs a live NetDriver. This
        // pins the pure half; the arm is the one line beside jitter_take's call.
    }
}
