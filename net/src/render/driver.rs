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
    assert!(
        world.get_resource::<RoundScope>().is_none(),
        "install_round over a live round — teardown_round must run first, or the \
         previous scope's teardowns would be silently dropped"
    );
    let mut scope: Vec<fn(&mut World)> = Vec::new();
    // Process-lifetime, NOT round-scoped: one trace file spans rounds (each marked
    // by its own `O` origin line), so it must not re-open — armed on first install.
    if world.get_resource::<super::pos_trace::PosTrace>().is_none() {
        world.insert_resource(super::pos_trace::PosTrace::from_env());
    }
    let prev = SimSnapshot::capture(&client);
    world.insert_non_send(GameState {
        client,
        coord,
        accumulator: 0.0,
        dt_window: DtWindow::default(),
        prev,
        reported_outcome: false,
        snap_buf: std::collections::VecDeque::new(),
        art_buf: std::collections::VecDeque::new(),
        stalled: false,
        logged_statuses: BTreeMap::new(),
        pending_pump: None,
    });
    scope.push(|w| {
        w.remove_non_send::<GameState>();
    });
    round_resource::<PendingInput>(world, &mut scope);
    round_resource::<FlightInput>(world, &mut scope);
    round_resource::<CameraPitch>(world, &mut scope);
    round_resource::<CameraYaw>(world, &mut scope);
    // The teardown reset IS a vehicle exit when the round ends mid-flight; without
    // this line the transition log skips the exit and the next ride reads
    // `OnFoot -> X` with no `X -> OnFoot` before it (rl#400: telemetry could not
    // segment rides across a round boundary).
    scope.push(|w| {
        let vehicle = w.resource::<LocalVehicle>();
        if vehicle.kind().is_some() {
            info!("vehicle: {:?} -> OnFoot (round end)", vehicle.context());
        }
    });
    round_resource::<LocalVehicle>(world, &mut scope);
    round_resource::<RenderClock>(world, &mut scope);
    round_resource::<super::articulation::RemoteVehicle>(world, &mut scope);
    round_resource::<super::articulation::CrabPartWindows>(world, &mut scope);
    round_resource::<crab_world::bot::skin::CrabRenderPose>(world, &mut scope);
    round_resource::<super::RenderOrigin>(world, &mut scope);

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
    // The combo map's replicated session set is round state riding a PERSISTENT
    // resource (the save must survive rounds; the last session's map must not) —
    // clear just that field (rl#398).
    scope.push(|w| {
        if let Some(mut map) = w.get_resource_mut::<super::chord_map::DiscoveredCodes>() {
            map.set_session(Default::default());
        }
    });
    world.insert_resource(RoundScope(scope));
}

#[derive(Default)]
pub(super) struct PendingRound(pub(super) Option<ArmedRound>);

pub(super) fn ensure_round_installed(world: &mut World) {
    if world.get_non_send::<GameState>().is_some() {
        return;
    }
    let mut ready = world
        .get_non_send_mut::<PendingRound>()
        .and_then(|mut p| p.0.take())
        .expect("entered Playing with no round to install — the menu must park a round before transitioning")
        .into_ready();
    assert!(
        world.get_resource::<NnCrabStackInstalled>().is_some(),
        "the NN-crab stack must be installed before Playing (rl#114: the checkpoint is required)"
    );
    let spawns = match world.get_non_send::<crate::crab_slot::CrabPolicies>() {
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

fn end_round_server_down(world: &mut World, down: crate::net_loop::ServerDown) {
    let message = down.to_string();
    // WARN, not ERROR: from the client's side losing the host is an expected round
    // ending (host quit) or the client's own link dying (deck suspend) — not a fault
    // in this process.
    warn!("leaving the round — {message}");
    if world.get_resource::<super::app::BootedWithMenu>().is_some() {
        let host = world
            .non_send::<GameState>()
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
/// The physics-side ceiling on what the craft bridge will vouch for, m/s: far above
/// any honest craft, far below anything that overflows the sim's i64 integration. A
/// blown-up rapier body (non-finite or absurd velocity — the class the rl#339 rescue
/// exists for) must not hand the authoritative walker a poisoned momentum.
const CRAFT_HANDOFF_MAX_MPS: f32 = 1000.0;

fn pilot_shadows(world: &mut World) -> BTreeMap<PlayerId, crate::sim::PilotPose> {
    // The craft's altitude and rapier velocity ride along (rl#355): the sim mirrors
    // both into the walker every piloted tick, so stepping out mid-air hands the
    // ballistic walker exactly the craft's altitude + momentum. Altitude is measured
    // through the sim's own `ground_at`, so the handoff and the landing read one
    // surface.
    let mut q = world.query::<(&Transform, &bevy_rapier3d::dynamics::Velocity, &Vehicle)>();
    let per_tick = |mps: f32| {
        let sane = if mps.is_finite() { mps } else { 0.0 };
        crate::sim::mps_to_grid_per_tick(
            sane.clamp(-CRAFT_HANDOFF_MAX_MPS, CRAFT_HANDOFF_MAX_MPS) as f64
        )
    };
    q.iter(world)
        .map(|(t, vel, v)| {
            let nose = t.rotation * Vec3::Z;
            let pos = world_to_sim(t.translation);
            let y = if t.translation.y.is_finite() {
                t.translation.y as f64
            } else {
                0.0
            };
            (
                PlayerId(v.pilot.0),
                crate::sim::PilotPose {
                    pos,
                    yaw: crate::sim::trig_client::radians_to_turns(super::scene::heading(nose)),
                    alt: crate::sim::meters_to_grid_f64(y) - crate::sim::ground_at(pos),
                    vel: crate::sim::Vel {
                        x: per_tick(vel.linear.x),
                        y: per_tick(vel.linear.y),
                        z: per_tick(vel.linear.z),
                    },
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
    dt_window: DtWindow,
    pub(super) prev: SimSnapshot,
    /// Round-decided latch: set when this round's decided outcome has been reported, cleared
    /// per Ongoing snapshot (a RESTART revives a decided round without rewinding the tick,
    /// rl#204). Lives here — not a system `Local` — so a new round starts unlatched by
    /// construction (rl#210).
    reported_outcome: bool,
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
    /// The host tick mid-pump (rl#396): the owed physics steps not yet run, if a tick
    /// is started but unfinalized. Paying a tick's whole 2-3-step physics pump inline
    /// in the crossing frame made TV frametimes bimodal, so the crossing frame runs
    /// only the exchange + the FIRST owed step, the rest spread one per frame
    /// ([`step_pending_pump`]) and force-complete before the next exchange
    /// ([`complete_pending_pump`]) — finalize (pose collect + hunt feed,
    /// authoritative sim step, snapshot + articulation broadcast) happens at
    /// owed-complete, so the wire still sees whole 30 Hz ticks. While `Some`, the
    /// client sim's tick sits one behind the exchanged tick (its walkers predicted
    /// into it) — the [`RenderClock`] write adds the tick back so render time never
    /// rewinds. Just a step count: the tick's slot inputs are re-derived at finalize,
    /// identical by construction (nothing touches the server sim between start and
    /// complete). Always `None` on a remote-adopt client (its frames are cheap; the
    /// host is the one peer that pumps physics).
    pending_pump: Option<u32>,
}

/// Even: an alternating fast/slow pair cancels exactly.
const DT_WINDOW: usize = 8;

/// `Time::delta` spaces render-thread completions, not displayed frames (rl#396).
/// A mean, not a median: a median drops time, and the host's tick stream would run slow.
/// Zero-filled, so the clock never leads wall time.
#[derive(Default)]
struct DtWindow([f64; DT_WINDOW]);

impl DtWindow {
    fn smooth(&mut self, raw: f64) -> f64 {
        self.0.rotate_left(1);
        self.0[DT_WINDOW - 1] = raw;
        self.0.iter().sum::<f64>() / DT_WINDOW as f64
    }
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
    pub(super) sprint: bool,
    pub(super) jump: bool,
    pub(super) slide: bool,
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
/// [`drive_client_sim`]: the last exchanged/adopted sim tick plus the accumulator
/// fraction into the next (on the host, a tick mid-deferral counts as exchanged —
/// rl#396). THE one clock every [`super::pose::PoseWindow`] sampler
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
        poses: Box<PoseWindow>,
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

    fn update_pose(&mut self, step: u64, p: Pose) {
        if let Self::Flying { poses, .. } = self {
            poses.push(step, p);
        }
    }
}

/// Keyed by pilot: the host's world also carries REMOTE pilots' crafts (rl#191), and
/// the cockpit camera must fly from ours alone.
fn own_wire_pose(art: &crate::articulation::CrabArticulation, me: PilotId) -> Option<Pose> {
    art.vehicles.iter().find(|v| v.pilot == me.0).map(|v| Pose {
        pos: Vec3::from_array(v.pos),
        orient: Quat::from_array(v.rot),
    })
}

/// The one pose-window feed on both arms (rl#274), stamped by physics step.
fn feed_pose_windows(
    world: &mut World,
    step: u64,
    art: &crate::articulation::CrabArticulation,
    me: PilotId,
) {
    super::articulation::feed_crab_part_windows(world, step, &art.crabs);
    super::articulation::publish_remote_vehicles(world, step, &art.vehicles, me);
    let Some(p) = own_wire_pose(art, me) else {
        return;
    };
    let mut vehicle = world.resource_mut::<LocalVehicle>();
    if matches!(&*vehicle, LocalVehicle::Flying { poses, .. } if poses.is_empty()) {
        info!("cockpit engaged: own craft's first pose arrived");
    }
    vehicle.update_pose(step, p);
    world
        .resource_mut::<super::pos_trace::PosTrace>()
        .craft(step, p.pos, p.orient);
}

/// Apply a chord-issued [`VehicleRequest`], if one is pending: boarding from foot needs
/// the sim's leave, switching craft-to-craft doesn't, and re-requesting the current
/// craft is a no-op (no pose-window reset mid-flight).
fn apply_vehicle_request(world: &mut World) {
    let Some(request) = world.resource_mut::<PendingInput>().vehicle.take() else {
        return;
    };
    let may_board = {
        let state = world.non_send::<GameState>();
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
                poses: Box::default(),
            })
        }
        VehicleRequest::Exit if vehicle.kind().is_some() => Some(LocalVehicle::OnFoot),
        _ => None,
    };
    // Orientation hand-off (rl#399), THE one place a mode switch carries the view
    // across. Stepping out keeps looking where the craft pointed: yaw already
    // carries through the sim (the pilot shadow mirrors the nose every piloted
    // tick, rl#258), but the on-foot pitch is a client-side accumulator
    // ([`CameraPitch`]) the flight never syncs — it must be set here or the foot
    // camera resumes on a stale value. Yaw is set too for the not-Alive camera
    // arm, which reads [`CameraYaw`] instead of the sim. Boarding needs nothing:
    // the craft spawns with the walker's facing (rl#258), and craft→craft morphs
    // the body in place, pose untouched.
    let exit_pose = match (&*vehicle, &next) {
        (LocalVehicle::Flying { poses, .. }, Some(LocalVehicle::OnFoot)) => poses.latest(),
        _ => None,
    };
    if let Some(next) = next {
        info!("vehicle: {:?} -> {:?}", vehicle.context(), next.context());
        *vehicle = next;
    }
    if let Some(pose) = exit_pose {
        let (yaw, pitch) = exit_look_angles(pose.orient);
        world.resource_mut::<CameraPitch>().0 = pitch;
        world.resource_mut::<CameraYaw>().0 = yaw;
    }
}

/// The on-foot look angles that keep the camera pointing where the craft's nose
/// points — the exit half of the rl#399 hand-off, the inverse of
/// [`super::scene::look_direction`]. Pitch clamps to the on-foot [`PITCH_LIMIT`];
/// roll has no on-foot analogue and drops.
fn exit_look_angles(orient: Quat) -> (f32, f32) {
    let nose = orient * Vec3::Z;
    let yaw = super::scene::heading(nose).rem_euclid(std::f32::consts::TAU);
    let pitch = nose
        .y
        .clamp(-1.0, 1.0)
        .asin()
        .clamp(-PITCH_LIMIT, PITCH_LIMIT);
    (yaw, pitch)
}

/// Assemble this tick's [`LocalControl`] from the pending foot input (drained: the yaw
/// delta, action, and restart taps are consumed here, once per tick) and, piloting, the
/// flight input mapped through [`flight_control`].
fn take_local_control(world: &mut World) -> LocalControl {
    let foot_input = {
        let mut pending = world.resource_mut::<PendingInput>();
        let look_axis = (pending.yaw_delta / MAX_YAW_PER_TICK_RADIANS).clamp(-1.0, 1.0);
        let btns = (if pending.action { buttons::ACTION } else { 0 })
            | (if pending.restart { buttons::RESTART } else { 0 })
            | (if pending.sprint { buttons::SPRINT } else { 0 })
            | (if pending.jump { buttons::JUMP } else { 0 })
            | (if pending.slide { buttons::SLIDE } else { 0 });
        let input = Input::new(pending.strafe, pending.forward, look_axis, btns);
        pending.yaw_delta = 0.0;
        pending.action = false;
        pending.restart = false;
        pending.sprint = false;
        pending.jump = false;
        pending.slide = false;
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
    /// The adopt-arm articulation to publish this tick (remote client only).
    articulation: Option<crate::articulation::CrabArticulation>,
    server_down: Option<crate::net_loop::ServerDown>,
}

/// Submit the local input and run this tick's coordinator exchange. On the remote-adopt
/// arm, also pace the jitter buffer and adopt what it releases (capturing `prev` for
/// interpolation and reconciling the local prediction).
fn exchange_tick(world: &mut World, role: PeerRole, local: &LocalControl) -> TickOutcome {
    let scripted_pack: Option<Input> = world.get_resource::<ScriptedPackInput>().map(|r| r.0);
    let mut state = world.non_send_mut::<GameState>();
    // Plain `&mut GameState` (not `Mut`) so the adopt closure below can borrow the
    // `reported_outcome` field disjointly from the `client` it runs against.
    let state = &mut *state;
    let me = state.client.me();
    let msg = state
        .client
        .submit_local_input(local.sim_input(), local.pilot_intent());
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
        articulation,
        server_down,
    }
}

fn adopt_wire_articulation(
    world: &mut World,
    art: &crate::articulation::CrabArticulation,
    me: PilotId,
) {
    super::articulation::adopt_brain_labels(world, art);
    feed_pose_windows(world, crate::cadence::cumulative_steps(art.tick), art, me);
}

/// (Host) Bridge every pilot's intent into a [`PilotCommand`] for the crab world's
/// vehicle systems, anchored at each walker's [`boarding_of`] state this tick.
fn publish_pilot_commands(world: &mut World) {
    if world.get_resource::<VehicleControls>().is_none() {
        return;
    }
    let terrain = world.resource::<crab_world::terrain::Terrain>().clone();
    let entries: BTreeMap<PilotId, PilotCommand> = {
        let state = world.non_send::<GameState>();
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

/// (Host) Start the server tick this frame's exchange assembled: capture `prev` and
/// run only the FIRST owed physics step — the rest spread one per frame
/// ([`step_pending_pump`]) and the tick finalizes at owed-complete
/// ([`complete_pending_pump`]), force-run before the next exchange. Host-paced:
/// `exchange` assembled at most the one tick this frame's issued local input
/// completed (a remote can delay nothing — rl#195), and `advance` asserts the
/// previous tick was stepped, so a missed force-complete fails loud.
fn start_host_tick(world: &mut World, me: PilotId, armed: bool) {
    {
        let state = world.non_send::<GameState>();
        debug_assert!(
            state.pending_pump.is_none(),
            "tick started over a live pending pump — complete_pending_pump precedes exchange"
        );
        if !state
            .server()
            .expect("server_auth ⇒ a server")
            .next_tick_ready()
        {
            return;
        }
    }
    {
        // Here, not at finalize: `prev` stays the last stepped tick while the
        // RenderClock counts the pending one.
        let mut state = world.non_send_mut::<GameState>();
        state.prev = SimSnapshot::capture(&state.client);
    }
    let remaining = if armed {
        let stepping_into = {
            let state = world.non_send::<GameState>();
            state.client.sim().tick() + 1
        };
        let owed = crate::cadence::steps_for_tick(stepping_into) - 1;
        host_physics_step(world, me, owed);
        owed
    } else {
        0
    };
    world.non_send_mut::<GameState>().pending_pump = Some(remaining);
    if remaining == 0 {
        // Nothing to spread (the unarmed crab-less screenshot host pumps no physics)
        // — finalize inline, frame-identical to the pre-rl#396 inline pump.
        complete_pending_pump(world, me, armed);
    }
}

/// One deferred physics step toward the pending host tick, on a frame that crossed no
/// tick; the last owed step finalizes in the same frame (the finalize tail — sim step
/// + broadcast — is cheap next to a physics step).
fn step_pending_pump(world: &mut World, me: PilotId, armed: bool) {
    let Some(remaining) = world.non_send::<GameState>().pending_pump else {
        return;
    };
    if remaining > 1 {
        host_physics_step(world, me, remaining - 1);
        world.non_send_mut::<GameState>().pending_pump = Some(remaining - 1);
    } else {
        complete_pending_pump(world, me, armed);
    }
}

/// (Host) One physics step toward the pending tick, leaving `owed_after` still owed.
/// Fed per step, not per tick: the RenderClock counts a pending tick, so a window
/// holding only finalized ticks clamps until the spread pump finishes (rl#396).
fn host_physics_step(world: &mut World, me: PilotId, owed_after: u32) {
    crate::crab_slot::pump_fixed_steps(world, 1);
    let stepping_into = world.non_send::<GameState>().client.sim().tick() + 1;
    let art = super::articulation::capture(world, stepping_into);
    let step = crate::cadence::cumulative_steps(stepping_into) - u64::from(owed_after);
    feed_pose_windows(world, step, &art, me);
}

/// Force-complete the pending host tick: run every still-owed physics step now, then
/// finalize — collect the crab poses + feed the next hunt, step the authoritative sim,
/// broadcast the snapshot + articulation, and mirror the step into our own client.
/// Runs before each exchange (the sim must sit at tick N before tick N+1's input
/// applies) and as [`step_pending_pump`]'s owed-complete tail. No-op when nothing is
/// pending.
fn complete_pending_pump(world: &mut World, me: PilotId, armed: bool) {
    let Some(remaining) = world.non_send_mut::<GameState>().pending_pump.take() else {
        return;
    };
    // Identical to a capture at start: the server sim is untouched between start and
    // complete (physics steps mutate only the crab world; exchange only assembles).
    let inputs = {
        let state = world.non_send::<GameState>();
        crate::crab_slot::slot_inputs(state.server().expect("server_auth ⇒ a server").sim())
    };
    let (crab_poses, shadows) = if armed {
        for owed in (0..remaining).rev() {
            host_physics_step(world, me, owed);
        }
        let poses = crate::crab_slot::finish_slot_tick(world, &inputs);
        (poses, pilot_shadows(world))
    } else {
        // The one unarmed host: the crab-less screenshot path
        // (fp-screenshot without a checkpoint) — no crab world to read,
        // so the sim's crabs hold their spawn poses, clawless. Poses stay
        // mandatory; this is a diagnostics surface, not a served round.
        (inputs.fallback.clone(), BTreeMap::new())
    };
    let (bytes, restarted) = {
        let mut state = world.non_send_mut::<GameState>();
        let stepped = state
            .server_mut()
            .expect("server_auth ⇒ a server")
            .step_next(&crab_poses, shadows);
        (stepped.snapshot, stepped.restarted)
    };
    let mut snap = crate::snapshot::CoreSnapshot::from_bytes(&bytes)
        .expect("the authoritative server's snapshot must decode");
    // rl#398: stamp the host's OWN discovered chord codes onto the outgoing
    // snapshot (rider metadata like `input_next` — the sim emits it empty) so a
    // joined client's combo map opens onto the session's discovered space. The
    // local save only, never the union: a replicated set stashed from some
    // earlier session this peer JOINED must not relay into rounds it hosts.
    snap.discovered = world
        .resource::<super::chord_map::DiscoveredCodes>()
        .codes()
        .clone();
    let articulation = armed.then(|| crate::render::articulation::capture(world, snap.tick));
    {
        let state = world.non_send::<GameState>();
        state.coord.broadcast_step(&snap, articulation.as_ref());
    }
    {
        let mut state = world.non_send_mut::<GameState>();
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
            .non_send::<GameState>()
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

/// Log each player's status edge exactly once per transition, on every peer, and report
/// the round's decided outcome once per decision (the latch clears per Ongoing tick in
/// the drain arms — a RESTART revives a decided round, rl#204).
fn log_round_edges(world: &mut World) {
    let mut state = world.non_send_mut::<GameState>();
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
    }
}

/// rl#371 trace: the local player's sim position after this tick's advance. On the
/// remote-adopt arm the sim cursor can hold across iterations (jitter-buffer stall);
/// the tick number on each line lets the reader collapse the repeats.
fn record_tick_trace(world: &mut World) {
    if world.resource::<super::pos_trace::PosTrace>().0.is_none() {
        return;
    }
    let Some((tick, pos, alt)) = ({
        let state = world.non_send::<GameState>();
        let sim = state.client.sim();
        sim.player(state.client.me())
            .map(|p| (sim.tick(), p.pos(), p.alt()))
    }) else {
        return;
    };
    world
        .resource_mut::<super::pos_trace::PosTrace>()
        .tick(tick, pos, alt);
}

pub(super) fn drive_client_sim(world: &mut World) {
    let raw = world.resource::<Time>().delta_secs_f64();
    let dt = world.non_send_mut::<GameState>().dt_window.smooth(raw);
    advance_frame(world, dt);
}

fn advance_frame(world: &mut World, dt: f64) {
    let sim_started = bevy::platform::time::Instant::now();
    let armed = world
        .get_resource::<crate::crab_slot::NnCrabsArmed>()
        .is_some();
    let role = PeerRole::of(world.non_send::<GameState>());
    let me = local_pilot(world.non_send::<GameState>());
    world.non_send_mut::<GameState>().accumulator += dt;

    apply_vehicle_request(world);

    let mut applied = 0u32;
    loop {
        {
            let state = world.non_send::<GameState>();
            if state.accumulator < TICK_DT || applied >= MAX_TICKS_PER_FRAME {
                break;
            }
        }
        world.non_send_mut::<GameState>().accumulator -= TICK_DT;
        applied += 1;

        if role == PeerRole::ServerAuth {
            // The previous tick's deferred steps must land before this tick's input
            // applies: the sim sits at tick N when tick N+1 assembles (rl#396), and
            // `Server::advance` asserts it.
            complete_pending_pump(world, me, armed);
        }

        let local = take_local_control(world);
        let sim_input = local.sim_input();
        let tick = exchange_tick(world, role, &local);
        if let Some(art) = tick.articulation {
            adopt_wire_articulation(world, &art, me);
        }
        if let Some(down) = tick.server_down {
            end_round_server_down(world, down);
            return;
        }

        if role == PeerRole::ServerAuth {
            publish_pilot_commands(world);
            start_host_tick(world, me, armed);
        }

        record_tick_trace(world);

        super::net_track::sample(world, role == PeerRole::ServerAuth, sim_input);
    }

    // A frame that crossed no tick pays one deferred physics step instead (rl#396) —
    // this spread is the whole point: every frame carries ~one step, none a whole
    // tick. A crossing frame already ran its step (and any force-completed
    // remainder), so it never doubles up here.
    if applied == 0 {
        step_pending_pump(world, me, armed);
    }

    // rl#398: publish the adopted snapshots' discovered chord codes to the combo
    // map — one unconditional path on both arms (the solo/host arm just mirrors its
    // own stamp back through the same seam a wire client consumes). Compared through
    // the immutable deref first: writing every frame would mark the resource changed
    // every frame, arming any future `Changed<DiscoveredCodes>` reader to fire
    // constantly (the set changes at most once per adopted snapshot).
    {
        let session = world
            .non_send::<GameState>()
            .client
            .session_discovered()
            .clone();
        let mut map = world.resource_mut::<super::chord_map::DiscoveredCodes>();
        if *map.session() != session {
            map.set_session(session);
        }
    }

    // Chronic input-starvation surface (rl#213): reports appear at most once per second per
    // player, so once per frame — after the tick drain — is plenty. A remote-adopt client has
    // no server and drains nothing.
    crate::net_loop::surface_starvation(world.non_send_mut::<GameState>().server_mut());

    if applied == MAX_TICKS_PER_FRAME {
        let mut state = world.non_send_mut::<GameState>();
        state.accumulator = state.accumulator.min(TICK_DT);
    }

    log_round_edges(world);

    // Once per tick: every crossing completes the tick before it starts its own.
    if applied > 0 && world.non_send::<GameState>().pending_pump.is_some() {
        let mut state = world.non_send_mut::<GameState>();
        let state = &mut *state;
        state
            .client
            .predict_pending_tick(state.coord.server().expect("a pending pump ⇒ a server"));
    }

    let clock = {
        let state = world.non_send::<GameState>();
        RenderClock {
            // Mid-deferral (rl#396) the host's sim sits one tick behind the exchanged
            // tick while the accumulator already crossed — count the pending tick or
            // render time rewinds ~a full tick on every crossing frame.
            tick: state.client.sim().tick() + u64::from(state.pending_pump.is_some()),
            frac: render_frac(state.accumulator, state.stalled),
        }
    };
    world.insert_resource(clock);

    // Feed the rl#331 perf readout/black box: this frame's whole-sim cost and how many
    // fixed ticks ran (pinned at MAX_TICKS_PER_FRAME = the sim can't keep up with wall
    // time — the death-spiral signature, distinct from a render/present stall).
    let sim_ms = sim_started.elapsed().as_secs_f32() * 1000.0;
    *world.resource_mut::<crab_world::debug_overlay::SimFrameStats>() =
        crab_world::debug_overlay::SimFrameStats {
            ms: sim_ms,
            ticks: applied,
        };
    world
        .resource_mut::<super::pos_trace::PosTrace>()
        .sim_cost(clock.tick, sim_ms, applied);
}

#[cfg(test)]
mod tests {
    use super::{
        DT_WINDOW, DtWindow, FlightControl, GameState, JITTER_BUF_MAX, JITTER_BUF_TARGET,
        LocalControl, PlaneControl, RenderClock, advance_frame, drive_client_sim, install_round,
        jitter_take, render_frac,
    };
    use crate::sim::{Input, TICK_DT, buttons};
    use bevy::prelude::*;

    /// A host round over a real armed crab world (rest-pose policy, flat ground): the
    /// host is `PlayerId(0)`, joined by `remotes` more players.
    fn armed_host(remotes: u8) -> App {
        use crab_world::bot::headless::{
            HeadlessStack, WorldRole, force_serial_schedules, headless_stack,
            pin_single_thread_pools,
        };

        pin_single_thread_pools();
        let peers: Vec<crate::sim::PlayerId> = (0..=remotes).map(crate::sim::PlayerId).collect();
        let mut client = crate::client::ClientSim::new(0xC0FFEE, &peers, peers[0]);
        let spawns = super::super::app::seed_round_crabs(&mut client, 1);
        let mut app = headless_stack(HeadlessStack {
            num_envs: spawns.len(),
            role: WorldRole::Standalone,
            grid: std::sync::Arc::new(crab_world::terrain::TerrainGrid::flat(16_384.0)),
            visuals: crab_world::Visuals(false),
        });
        app.add_plugins(crate::crab_slot::NnCrabPlugin::new(
            spawns
                .iter()
                .map(|_| crab_world::policy::Policy::rest())
                .collect(),
            spawns.clone(),
        ));
        app.add_plugins(crab_world::vehicle::VehiclePlugin);
        crate::crab_slot::arm(app.world_mut());
        crate::crab_slot::park_fixed_auto_pump(&mut app);
        crate::crab_slot::restart_crabs_to_spawns(app.world_mut(), &spawns);
        force_serial_schedules(&mut app);
        // Spawn the crab world's Update-side entities before installing the round.
        for _ in 0..8 {
            app.update();
        }
        let coord = super::coordinator(None, client.peers(), client.me(), client.sim().clone());
        install_round(app.world_mut(), client, coord);
        app.insert_resource(super::super::chord_map::DiscoveredCodes::load(None));
        app.init_resource::<crab_world::debug_overlay::SimFrameStats>();
        app
    }

    /// rl#409 repro: a ship boarded on a live host must answer the stick BOTH before
    /// and after an in-round RESTART. A double RESTART while piloting opened a
    /// dead-stick window: the walker track jumped to the rl#322 park ring and the
    /// craft then ignored ~6 min of stick input.
    #[test]
    fn ship_answers_the_stick_before_and_after_a_restart() {
        use crab_world::vehicle::{Vehicle, VehicleKind};

        let mut app = armed_host(0);
        let world = app.world_mut();
        // One full tick per call: a crossing frame (exchange + first owed step) then a
        // non-crossing frame that finalizes the spread pump (rl#396).
        fn tick(world: &mut World) {
            for dt in [TICK_DT * 1.01, 0.001] {
                advance_frame(world, dt);
            }
        }
        fn craft_pos(world: &mut World) -> Option<Vec3> {
            let mut q = world.query::<(&Transform, &Vehicle)>();
            q.iter(world).next().map(|(t, _)| t.translation)
        }
        // Hold full forward stick for `ticks`, then release.
        fn fly(world: &mut World, ticks: u32) {
            world.resource_mut::<super::FlightInput>().left = Vec2::new(0.0, 1.0);
            for _ in 0..ticks {
                tick(world);
            }
            world.resource_mut::<super::FlightInput>().left = Vec2::ZERO;
        }

        world.resource_mut::<super::PendingInput>().vehicle =
            Some(super::VehicleRequest::Board(VehicleKind::Ship));
        for _ in 0..3 {
            tick(world);
        }
        let boarded = craft_pos(world).unwrap_or_else(|| {
            let state = world.non_send::<GameState>();
            let intents = state.coord.server().map(|s| s.pilot_intents().len());
            let controls = world
                .get_resource::<crab_world::vehicle::VehicleControls>()
                .map(|c| c.0.len());
            let lv = world.resource::<super::LocalVehicle>().context();
            panic!(
                "the boarding request spawns the craft — intents {intents:?}, \
                 controls {controls:?}, local vehicle {lv:?}"
            )
        });

        fly(world, 90);
        let flown = craft_pos(world).expect("the craft persists while the intent is filed");
        assert!(
            (flown - boarded).length() > 1.0,
            "sanity: a fresh ship answers the stick ({boarded} -> {flown})"
        );

        // The session shape: RESTART double-tapped ~0.64 s (~19 ticks) apart, then the
        // craft sat parked long past rapier's sleep timer before the stick came back.
        world.resource_mut::<super::PendingInput>().restart = true;
        for _ in 0..19 {
            tick(world);
        }
        world.resource_mut::<super::PendingInput>().restart = true;
        for _ in 0..300 {
            tick(world);
        }
        let parked = craft_pos(world).expect("the craft survives a RESTART (rl#322 park)");

        fly(world, 90);
        let after = craft_pos(world).expect("the craft persists across the post-restart fly");
        assert!(
            (after - parked).length() > 1.0,
            "rl#409: the ship ignores the stick after an in-round RESTART \
             (parked {parked}, still at {after})"
        );
    }

    /// rl#396: the host spreads a tick's 2-3-step physics pump across frames. The
    /// crossing frame runs the exchange + only the FIRST owed step (the sim tick
    /// holds while [`RenderClock`] counts the pending tick, so render time never
    /// rewinds); a non-crossing frame runs the rest and finalizes; a back-to-back
    /// crossing force-completes first — `Server::advance` asserts on an unstepped
    /// tick, so a missed force-complete would panic right here.
    #[test]
    fn host_pump_spreads_steps_across_frames() {
        let mut app = armed_host(0);
        let world = app.world_mut();
        let mut render_time = 0.0_f64;
        let mut drive = |world: &mut World, dt: f64| {
            advance_frame(world, dt);
            let clock = *world.resource::<RenderClock>();
            let now = clock.tick as f64 + clock.frac as f64;
            assert!(
                now >= render_time,
                "render time rewound: {render_time} -> {now}"
            );
            render_time = now;
            (
                world.non_send::<GameState>().client.sim().tick(),
                world.non_send::<GameState>().pending_pump.is_some(),
                clock,
            )
        };

        let t0 = world.non_send::<GameState>().client.sim().tick();

        // Crossing frame: exchange + first owed step only — the sim tick holds, the
        // pump goes pending, and the clock counts the tick being stepped into.
        let (tick, pending, clock) = drive(world, TICK_DT * 1.01);
        assert_eq!(tick, t0, "the crossing frame must not finalize the tick");
        assert!(pending, "the crossing frame leaves the tick pending");
        assert_eq!(clock.tick, t0 + 1, "the clock counts the pending tick");
        assert_eq!(
            world
                .resource::<crab_world::debug_overlay::SimFrameStats>()
                .ticks,
            1
        );

        // Non-crossing frame: the remaining owed step(s) land and the tick
        // finalizes — sim step, snapshot, broadcast.
        let (tick, pending, clock) = drive(world, 0.001);
        assert_eq!(tick, t0 + 1, "owed-complete finalizes the tick");
        assert!(!pending);
        assert_eq!(clock.tick, t0 + 1);

        // Back-to-back crossings: each force-completes the previous pending tick
        // before its exchange (Server::advance asserts otherwise), then defers its
        // own — sustained tick rate is preserved one tick behind.
        let (tick, pending, _) = drive(world, TICK_DT * 1.01);
        assert_eq!(tick, t0 + 1);
        assert!(pending);
        let (tick, pending, clock) = drive(world, TICK_DT * 1.01);
        assert_eq!(
            tick,
            t0 + 2,
            "the crossing force-completed the pending tick"
        );
        assert!(pending, "and deferred its own");
        assert_eq!(clock.tick, t0 + 3);
    }

    /// rl#396: every host frame renders the walkers — the on-foot camera's position and
    /// heading, and a remote walker's avatar — exactly at the authoritative ticks'
    /// interpolation for its RenderClock, the frames clocked into a tick the spread pump
    /// has not stepped yet included. Inputs change every few frames, so a pending tick
    /// held, extrapolated, delayed or fed stale input shows.
    #[test]
    fn host_walkers_render_the_pending_tick_exactly() {
        use super::super::scene::{
            FpCamera, PlayerAvatar, apply_transforms, heading, lerp_pos, lerp_yaw,
            sync_ground_anchor,
        };
        use crate::sim::PlayerId;
        use bevy::ecs::system::RunSystemOnce;
        use std::f32::consts::{PI, TAU};

        let (me, remote) = (PlayerId(0), PlayerId(1));
        let mut app = armed_host(1);
        let world = app.world_mut();
        let cam = world.spawn((FpCamera, Transform::default())).id();
        let avatar = world
            .spawn((
                PlayerAvatar(remote),
                Transform::default(),
                Visibility::default(),
            ))
            .id();
        world.init_resource::<crab_world::ground::GroundAnchor>();
        world
            .run_system_once(sync_ground_anchor)
            .expect("sync_ground_anchor runs");
        let origin = world.resource::<super::super::RenderOrigin>().0;

        let mut stepped = std::collections::BTreeMap::new();
        let mut frames = Vec::new();
        for frame in 0..400u32 {
            let phase = frame / 5;
            {
                let mut pending = world.resource_mut::<super::PendingInput>();
                pending.forward = 1.0;
                pending.strafe = [0.5, -0.3, 0.0][phase as usize % 3];
                pending.yaw_delta = [0.03, -0.02, 0.0, 0.05][phase as usize % 4];
                pending.sprint = phase % 2 == 0;
            }
            world.insert_resource(super::ScriptedPackInput(Input::new(
                [0.4, -0.4][phase as usize % 2],
                1.0,
                [0.5, -0.2, 0.0][phase as usize % 3],
                if phase % 5 == 0 { buttons::SPRINT } else { 0 },
            )));
            // Frame deltas of 0.44–0.57 ticks: crossing frames land at every phase,
            // ticks span one to three frames, and the run meets every 64:30 bunching.
            advance_frame(
                world,
                TICK_DT * (0.44 + 0.13 * (f64::from(frame) * 0.618_034 % 1.0)),
            );
            world
                .run_system_once(apply_transforms)
                .expect("apply_transforms runs");
            let state = world.non_send::<GameState>();
            let sim = state.coord.server().expect("the host serves").sim();
            let walker = |pid| sim.player(pid).expect("both walkers stay in the round");
            stepped.insert(sim.tick(), (walker(me), walker(remote)));
            let at = |e: Entity| *world.get::<Transform>(e).expect("spawned above");
            frames.push((*world.resource::<RenderClock>(), at(cam), at(avatar)));
        }

        let xz = |t: &Transform| (t.translation.x, t.translation.z);
        let mut checked = 0;
        for (frame, (clock, cam, avatar)) in frames.iter().enumerate() {
            let (Some((me0, remote0)), Some((me1, remote1))) = (
                clock.tick.checked_sub(1).and_then(|t| stepped.get(&t)),
                stepped.get(&clock.tick),
            ) else {
                continue;
            };
            let alpha = clock.frac;
            let eye = lerp_pos(me0.pos(), me1.pos(), alpha).rel_meters(origin);
            assert_eq!(xz(cam), eye, "frame {frame}: camera off the stepped ticks");
            let yaw = lerp_yaw(me0.yaw(), me1.yaw(), alpha);
            let yaw_err = (heading(*cam.forward()) - yaw + PI).rem_euclid(TAU) - PI;
            assert!(
                yaw_err.abs() < 1e-4,
                "frame {frame}: camera heading {yaw_err} rad off the stepped ticks"
            );
            let body = lerp_pos(remote0.pos(), remote1.pos(), alpha).rel_meters(origin);
            assert_eq!(
                xz(avatar),
                body,
                "frame {frame}: remote walker off the stepped ticks"
            );
            checked += 1;
        }
        assert!(
            checked > frames.len() - 8,
            "only {checked} of {} frames had both stepped ticks",
            frames.len()
        );
    }

    /// rl#396: the host's [`super::pose::PoseWindow`] surfaces render its physics
    /// steps on time while the spread pump holds a tick pending. A ship flight and a
    /// settling crab at ~60 and ~144 fps frame deltas: every frame's cockpit and
    /// carapace samples must equal the physics steps, recorded as they ran,
    /// interpolated at that frame's [`RenderClock`] — capped at the steps run so far,
    /// which a crossing frame late in its tick phase trails by a fraction of a step.
    #[test]
    fn host_pose_windows_render_every_physics_step_on_time() {
        use super::super::articulation::sample_crab_part_poses;
        use bevy::ecs::system::RunSystemOnce;
        use crab_world::bot::body::{CrabBodyPart, CrabCarapace};
        use crab_world::bot::skin::CrabRenderPose;
        use crab_world::vehicle::{Vehicle, VehicleKind};

        /// Craft and carapace translations after each physics step, index = step − 1.
        #[derive(Resource, Default)]
        struct Stepped(Vec<(Option<Vec3>, Vec3)>);

        let r = crab_world::physics::PHYSICS_HZ as f64 / crate::sim::TICK_HZ as f64;
        for (rate, dt_lo, dt_span) in [("~60 fps", 0.44, 0.13), ("~144 fps", 0.19, 0.04)] {
            let mut app = armed_host(0);
            app.init_resource::<Stepped>();
            app.add_systems(
                FixedLast,
                |mut stepped: ResMut<Stepped>,
                 crafts: Query<&Transform, With<Vehicle>>,
                 carapaces: Query<&Transform, (With<CrabCarapace>, With<CrabBodyPart>)>| {
                    let craft = crafts.iter().next().map(|t| t.translation);
                    let carapace = carapaces.single().expect("one crab").translation;
                    stepped.0.push((craft, carapace));
                },
            );
            let world = app.world_mut();
            let carapace = world
                .query_filtered::<Entity, (With<CrabCarapace>, With<CrabBodyPart>)>()
                .single(world)
                .expect("one crab");
            world.resource_mut::<super::PendingInput>().vehicle =
                Some(super::VehicleRequest::Board(VehicleKind::Ship));
            world.resource_mut::<super::FlightInput>().left = Vec2::new(0.0, 1.0);
            let mut frames = Vec::new();
            for frame in 0..400u32 {
                advance_frame(
                    world,
                    TICK_DT * (dt_lo + dt_span * (f64::from(frame) * 0.618_034 % 1.0)),
                );
                world
                    .run_system_once(sample_crab_part_poses)
                    .expect("sample_crab_part_poses runs");
                let clock = *world.resource::<RenderClock>();
                let cockpit = world
                    .resource::<super::LocalVehicle>()
                    .cockpit_sample(clock.tick, clock.frac)
                    .map(|p| p.pos);
                let body = world.resource::<CrabRenderPose>().0[&carapace].translation;
                let ran = world.resource::<Stepped>().0.len();
                frames.push((clock, ran, cockpit, body));
            }
            let offset = world.resource::<super::super::RenderOrigin>().offset_m();
            let stepped = &world.resource::<Stepped>().0;

            let at = |t: f64, pick: fn(&(Option<Vec3>, Vec3)) -> Option<Vec3>| {
                let s = t.floor() as usize;
                let a = pick(&stepped[s - 1])?;
                let w = (t - s as f64) as f32;
                Some(if w == 0.0 {
                    a
                } else {
                    a.lerp(pick(&stepped[s])?, w)
                })
            };
            let (mut flown, mut settled) = (0, 0);
            for (frame, &(clock, ran, cockpit, body)) in frames.iter().enumerate() {
                let target = r * (clock.tick.saturating_sub(1) as f64 + clock.frac as f64) - 1.0;
                if clock.tick == 0 || target < 1.0 {
                    continue;
                }
                let shortfall = target - ran as f64;
                assert!(
                    shortfall < 0.25,
                    "{rate} frame {frame} (clock {} + {}): the clock is {shortfall} steps \
                     past the {ran} run",
                    clock.tick,
                    clock.frac,
                );
                let t = target.min(ran as f64);
                let off = |got: Vec3, want: Vec3| (got - want).length();
                let body_want = at(t, |s| Some(s.1)).expect("the crab is always stepped") - offset;
                assert!(
                    off(body, body_want) < 1e-4,
                    "{rate} frame {frame} (clock {} + {}): carapace {body} vs the \
                     physics steps' {body_want}",
                    clock.tick,
                    clock.frac,
                );
                settled += 1;
                if let (Some(want), Some(cockpit)) = (at(t, |s| s.0), cockpit) {
                    assert!(
                        off(cockpit, want) < 1e-4,
                        "{rate} frame {frame} (clock {} + {}): cockpit {cockpit} vs the \
                         physics steps' {want}",
                        clock.tick,
                        clock.frac,
                    );
                    flown += 1;
                }
            }
            let (first, last) = (stepped.first().unwrap().1, stepped.last().unwrap().1);
            let mut flight = stepped.iter().filter_map(|s| s.0);
            let (lift, cruise) = (flight.clone().next().unwrap(), flight.next_back().unwrap());
            assert!(
                (first - last).length() > 0.01 && (lift - cruise).length() > 1.0,
                "{rate}: sanity — the crab settled and the ship flew \
                 ({first} -> {last}, {lift} -> {cruise})"
            );
            assert!(
                flown > frames.len() / 2 && settled > frames.len() - 16,
                "{rate}: {flown} flown and {settled} settled frames of {}",
                frames.len()
            );
        }
    }

    /// A remote client stamps each adopted articulation at its tick's physics step —
    /// the clock [`super::pose::PoseWindow`] samples on — so a craft moving at constant
    /// velocity in physics time renders exactly on that clock through the 64:30
    /// staircase.
    #[test]
    fn client_cockpit_samples_adopted_ticks_on_the_step_clock() {
        use crate::articulation::{CrabArticulation, VehiclePoseWire};
        use crate::cadence::cumulative_steps;
        use crab_world::vehicle::{PilotId, VehicleKind};

        let r = crab_world::physics::PHYSICS_HZ as f64 / crate::sim::TICK_HZ as f64;
        let mut world = World::new();
        world.init_resource::<super::super::articulation::CrabPartWindows>();
        world.init_resource::<super::super::articulation::RemoteVehicle>();
        world.init_resource::<super::super::pos_trace::PosTrace>();
        world.insert_resource(super::LocalVehicle::Flying {
            kind: VehicleKind::Ship,
            poses: Box::default(),
        });
        let mut checked = 0;
        for tick in 1..=12u64 {
            let art = CrabArticulation {
                tick,
                crabs: Vec::new(),
                vehicles: vec![VehiclePoseWire {
                    pilot: 0,
                    kind: VehicleKind::Ship,
                    pos: [cumulative_steps(tick) as f32, 0.0, 0.0],
                    rot: [0.0, 0.0, 0.0, 1.0],
                    thrust: [0; 3],
                }],
            };
            super::adopt_wire_articulation(&mut world, &art, PilotId(0));
            if tick < 4 {
                continue;
            }
            for f in 0..4 {
                let frac = f as f32 / 4.0;
                let x = world
                    .resource::<super::LocalVehicle>()
                    .cockpit_sample(tick, frac)
                    .expect("flying with poses fed")
                    .pos
                    .x;
                let step = r * ((tick - 1) as f64 + frac as f64) - 1.0;
                assert!(
                    (f64::from(x) - step).abs() < 1e-4,
                    "tick {tick} frac {frac}: cockpit at step {x}, clock at {step}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 36);
    }

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

    #[test]
    fn dt_window_spreads_a_hitch_and_never_leads_wall_time() {
        let refresh = TICK_DT / 2.0;
        let mut window = DtWindow::default();
        let (mut wall, mut clock) = (0.0, 0.0);
        for raw in std::iter::once(0.25).chain(std::iter::repeat_n(refresh, 2 * DT_WINDOW)) {
            wall += raw;
            clock += window.smooth(raw);
            assert!(clock <= wall + 1e-12, "the clock ran ahead of wall time");
        }
        let trail = (DT_WINDOW - 1) as f64 / 2.0 * refresh;
        assert!(
            ((wall - clock) - trail).abs() < 1e-9,
            "the clock trails wall time by {} s once the hitch left the window, not the \
             steady {trail} s: the hitch's time was dropped",
            wall - clock
        );
    }

    #[test]
    fn bunched_time_deltas_advance_the_render_clock_evenly() {
        let me = crate::sim::PlayerId(0);
        let client = crate::client::ClientSim::new(0xC0FFEE, &[me], me);
        let coord = super::coordinator(None, client.peers(), client.me(), client.sim().clone());
        let mut world = World::new();
        world.init_resource::<Time>();
        world.insert_resource(super::super::chord_map::DiscoveredCodes::load(None));
        world.init_resource::<crab_world::debug_overlay::SimFrameStats>();
        install_round(&mut world, client, coord);

        let fast = 0.003;
        let mut prev = 0.0;
        for frame in 0..6 * DT_WINDOW {
            let raw = if frame % 2 == 0 { fast } else { TICK_DT - fast };
            world
                .resource_mut::<Time>()
                .advance_by(std::time::Duration::from_secs_f64(raw));
            drive_client_sim(&mut world);
            let clock = *world.resource::<RenderClock>();
            let now = clock.tick as f64 + f64::from(clock.frac);
            if frame >= DT_WINDOW {
                assert!(
                    (now - prev - 0.5).abs() < 1e-4,
                    "frame {frame}: the render clock advanced {} ticks, not half a tick",
                    now - prev
                );
            }
            prev = now;
        }
    }

    /// rl#399: stepping out of a craft must keep looking where the nose pointed —
    /// [`super::exit_look_angles`] is the inverse of the on-foot camera's `look_direction`,
    /// so feeding its angles back through must reproduce the nose direction. Roll is
    /// dropped by design. [`super::PITCH_LIMIT`] (1.5 rad) < π/2, so a nose steeper
    /// than 1.5 rad clamps and would not round-trip; the cases stay at ≤1.2 rad.
    #[test]
    fn exit_look_angles_match_the_craft_nose() {
        use super::super::scene::look_direction;
        use bevy::math::{EulerRot, Quat, Vec3};
        for (yaw, pitch, roll) in [
            (0.0, 0.0, 0.0),
            (2.1, 0.4, 0.0),
            (-1.3, -0.9, 1.0),
            (3.0, 1.2, -2.0),
        ] {
            let orient = Quat::from_euler(EulerRot::YXZ, yaw, -pitch, roll);
            let (cam_yaw, cam_pitch) = super::exit_look_angles(orient);
            let nose = (orient * Vec3::Z).normalize();
            let look = look_direction(cam_yaw, cam_pitch);
            assert!(
                nose.dot(look) > 0.9999,
                "exit view diverged from the nose: yaw={yaw} pitch={pitch} roll={roll} nose={nose:?} look={look:?}"
            );
        }
    }
}
