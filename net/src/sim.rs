//! Deterministic simulation core for lockstep multiplayer: peers run independent
//! copies and only inputs cross the wire (see [`crate::lockstep`]), so the sim
//! must be bit-identical across machines. The contract that buys that:
//!
//! - [`Sim::step`] is a pure function of `(prior state, inputs)`. No wall-clock, no
//!   thread-local/global RNG, no iteration over `HashMap`/`HashSet` (their order is
//!   randomized per process). Seed every random draw from [`Sim::rng`]; iterate
//!   players in `PlayerId` order.
//! - **No `f32`/`f64` arithmetic in the sim.** Floats round-trip differently across
//!   targets/compilers, so the whole world is integer fixed-point (positions, yaw,
//!   velocities) and angles go through an integer sin/cos table ([`trig`]). `f32`
//!   appears ONLY at the input boundary ([`Input::from_axes`]), which quantizes to
//!   the integer grid before anything reaches the sim.
//! - [`Sim::state_hash`] folds the FULL observable state into one `u64` — the desync
//!   detector. Every field added to the state MUST be hashed there, or a desync in it
//!   goes undetected.
//!
//! **Coordinate frame.** Right-handed: world is the XZ ground plane at Y=0, +X right,
//! +Z forward, +Y up. A yaw of 0 faces +Z and increases turning toward +X (see
//! [`trig::atan2_turns`]). The accessors below are read-only — rendering reads them but
//! never drives sim logic, which goes through [`Sim::step`] so it stays deterministic.

use std::collections::BTreeMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crab_world::fnv::Fnv;

use crate::snapshot::CoreSnapshot;

/// Identifies a player within the sim. Assigned from the connection set in a
/// deterministic order so every peer agrees which id is whom (see
/// [`crate::lockstep`]); the sim itself only relies on the ordering being
/// total and identical across peers, which `u8` gives for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlayerId(pub u8);

/// Button bits packed into [`Input::buttons`]. A bitfield (not a bool per action) so
/// growing the control set — jump, fire, enter-vehicle — costs no wire bytes and no
/// `WIRE_LEN` bump; unused bits stay 0. Only [`ACTION`](buttons::ACTION) is wired in
/// the gray-box (the extract/interact button).
pub mod buttons {
    /// Context action: at the extraction point, confirm the pickup. Bit 0.
    pub const ACTION: u8 = 1 << 0;
    /// Restart the round. Bit 1. Routed through the input stream (not a client-local
    /// reset) so the ONE authoritative server applies it and every client adopts the
    /// reset via the snapshot — a local-only reset would desync. Edge-triggered (see
    /// [`Sim::step`]). A restart is a STATE reset at the current tick, never a tick
    /// rewind: [`Sim::tick`] stays monotone so the server's tick space (the assemble
    /// cursor, the roster schedule) remains aligned with the sim across a restart
    /// (rl#204).
    pub const RESTART: u8 = 1 << 1;
}

/// One on-foot player's input for a single tick — the unit that crosses the wire (see
/// [`Input::to_bytes`]). This is a FOOT-ONLY control set: the only thing the deterministic
/// sim simulates is walkers, so the only input it carries is the walker's. A piloting player
/// feeds the sim a neutral (default) `Input` — its craft is client-local, host-authoritative
/// crab-world state off the wire (see [`crate::render::driver`]'s `LocalControl`/`FlightControl`),
/// so no flight axis lives here and a walker cannot carry one (the illegal on-foot+piloting
/// state is unrepresentable). Networking the pilots (a wire that carries flight input too) is
/// rl#43 part 2.
///
/// The move/look axes are facing-relative (named for the control intent), not world axes: at a
/// nonzero yaw they do not map to world X/Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Input {
    /// Strafe axis (right +, i.e. toward +X at yaw 0) in fixed-point units of
    /// 1/[`Input::AXIS_SCALE`], clamped to ±[`Input::AXIS_SCALE`].
    pub move_strafe: i16,
    /// Forward axis (forward +, i.e. toward +Z at yaw 0), same units/clamp as
    /// [`move_strafe`](Input::move_strafe).
    pub move_forward: i16,
    /// Yaw-look delta for THIS tick, in fixed-point units of 1/[`Input::AXIS_SCALE`]:
    /// ±[`Input::AXIS_SCALE`] is the per-tick yaw cap. The client integrates raw
    /// mouse/stick into this bounded delta and the sim adds it to the player's yaw —
    /// bounded so one tick can't spin a peer arbitrarily. Always a per-tick FOOT yaw
    /// delta; the in-air turn-rate is a different axis on a different type
    /// ([`crate::render::driver::FlightControl`]), so the two never share a field.
    pub look_yaw: i16,
    /// Action/button bitfield (see [`buttons`]). Held state, sampled each tick.
    pub buttons: u8,
}

impl Input {
    /// Fixed-point denominator: an axis value of `AXIS_SCALE` is full deflection
    /// (1.0). `move_strafe`, `move_forward`, and `look_yaw` all use this grid.
    pub const AXIS_SCALE: i16 = 1000;
    /// Wire size of one encoded [`Input`]: move_strafe(2) + move_forward(2) +
    /// look_yaw(2) + buttons(1). Drives [`crate::transport`]'s frame size — keep in
    /// sync with [`Input::to_bytes`].
    pub const WIRE_LEN: usize = 7;

    /// Full constructor: analog `strafe`, `forward`, and `look_yaw` axes in
    /// `[-1.0, 1.0]` plus the raw button bitfield. Quantizes the analog values to the
    /// fixed-point grid at the input boundary (not in the sim), so the sim stays
    /// integer-only and the value that crosses the wire is exactly the value applied.
    pub fn new(strafe: f32, forward: f32, look_yaw: f32, buttons: u8) -> Self {
        let q = |v: f32| (v.clamp(-1.0, 1.0) * Self::AXIS_SCALE as f32).round() as i16;
        Self {
            move_strafe: q(strafe),
            move_forward: q(forward),
            look_yaw: q(look_yaw),
            buttons,
        }
    }

    /// Move-only input (`strafe`, `forward`; neutral look, no buttons). The 2-axis
    /// constructor the netcode skeleton and its determinism tests were written against.
    pub fn from_axes(strafe: f32, forward: f32) -> Self {
        Self::new(strafe, forward, 0.0, 0)
    }

    /// Whether a button bit (see [`buttons`]) is held this tick.
    pub fn pressed(self, bit: u8) -> bool {
        self.buttons & bit != 0
    }

    /// The input the server substitutes for a tick where this player's stream is STARVED
    /// (transit lag): keep the held-state move axes, zero the rest — `look_yaw` is a per-tick
    /// DELTA (re-applying it would keep the avatar turning) and a re-fired button tap would
    /// double a grab/restart. See [`crate::server`]'s hold semantics.
    pub fn hold(self) -> Self {
        Self {
            move_strafe: self.move_strafe,
            move_forward: self.move_forward,
            look_yaw: 0,
            buttons: 0,
        }
    }

    /// Encode for the wire: little-endian, fixed width. Decoding the result yields
    /// exactly `self`.
    pub fn to_bytes(self) -> [u8; Self::WIRE_LEN] {
        let mut b = [0u8; Self::WIRE_LEN];
        b[0..2].copy_from_slice(&self.move_strafe.to_le_bytes());
        b[2..4].copy_from_slice(&self.move_forward.to_le_bytes());
        b[4..6].copy_from_slice(&self.look_yaw.to_le_bytes());
        b[6] = self.buttons;
        b
    }

    /// Inverse of [`Input::to_bytes`].
    pub fn from_bytes(b: [u8; Self::WIRE_LEN]) -> Self {
        Self {
            move_strafe: i16::from_le_bytes([b[0], b[1]]),
            move_forward: i16::from_le_bytes([b[2], b[3]]),
            look_yaw: i16::from_le_bytes([b[4], b[5]]),
            buttons: b[6],
        }
    }
}

/// Tick rate of the deterministic sim (Hz). The CANONICAL value: the headless driver
/// (`game`), the windowed client's render loop (`net::render`), and the per-tick speed
/// tuning here all read this one constant, so a render peer and a headless peer step
/// at the same rate and stay in lockstep. Lives in `sim` (the render-free determinism
/// core) so it is available even in the headless trainer build where `net::render` is
/// gated out. 30 Hz is plenty for the gray-box and keeps the lockstep stall window
/// forgiving on a LAN.
pub const TICK_HZ: u64 = 30;

/// Seconds per sim tick — the fixed dt one tick advances. Derived from [`TICK_HZ`] in
/// ONE place so every wall-clock pacer (the headless `net`/`solo` loops, the windowed
/// render accumulator) and the fixed per-frame screenshot step read the same value
/// instead of re-spelling `1.0 / TICK_HZ` and risking drift. Pure pacing/interpolation —
/// never part of the deterministic sim state.
pub const TICK_DT: f64 = 1.0 / TICK_HZ as f64;

/// Fixed-point world scale: a position/length value of [`UNIT`] equals one world
/// meter. All world coordinates, radii, and speeds are integers in these units.
pub const UNIT: i64 = 1000;

/// Player walk speed, in [`UNIT`]/tick at full stick. Tuned for the gray-box, not
/// realism: ~5 m/s at 30 Hz (`SPEED * 30 / UNIT`).
const PLAYER_SPEED: i64 = 166;

/// TEST-FIXTURE crab speed, in [`UNIT`]/tick. The production crab is driven entirely by
/// the rapier-NN body, so its real speed comes from physics, not a constant. This value only
/// powers the deterministic test driver ([`drive_crab_toward_prey`]) that walks a stand-in crab so
/// the grab/extraction/outcome MECHANICS stay exercised. Kept below [`PLAYER_SPEED`] (166) so the
/// dodge test's faster player can still outrun and beat it.
#[cfg(test)]
const CRAB_SPEED: i64 = 130;

/// Deterministic stand-in for the production crab driver, shared by every test in this
/// crate that needs a HUNTING crab (the sim mechanic tests + the `net` desync replay). Production
/// drives the crab from the rapier-NN body via [`Sim::set_external_crab_pose`]; tests have no
/// rapier stack, so this walks a crab toward the nearest living prey with the SAME integer math
/// the real bridge's hunt target uses, then pushes the pose. `#[cfg(test)]` only — it can NEVER
/// stand in for the NN crab in a release build, so it is not a production fallback. ONE
/// definition. Call it each tick BEFORE [`Sim::step`]: it moves
/// the crab once the round is `STARTUP_GRACE_TICKS` old, so the pose is in place for the first
/// armed step (the one that increments the tick PAST the grace and turns on grabs).
#[cfg(test)]
pub(crate) fn drive_crab_toward_prey(sim: &mut Sim) {
    if sim.tick() < sim.round_start + STARTUP_GRACE_TICKS {
        return; // hold spawn through the grace, matching the grab gate's head-start
    }
    let Some(target) = sim.nearest_living_player_pos() else {
        return;
    };
    let mut pos = sim.crab().pos();
    let dx = target.x - pos.x;
    let dz = target.z - pos.z;
    let yaw = trig::atan2_turns(dx, dz);
    let dist = isqrt_i128(dist2_i128(dx, dz));
    if dist <= CRAB_SPEED as i128 {
        pos = target;
    } else if dist > 0 {
        pos.x += (dx as i128 * CRAB_SPEED as i128 / dist) as i64;
        pos.z += (dz as i128 * CRAB_SPEED as i128 / dist) as i64;
    }
    sim.set_external_crab_pose(pos, yaw, 0);
}

/// Render hint only: how many times bigger than a player to draw the crab. It is a
/// scaled placeholder; the sim treats it as a point with a [`CRAB_GRAB_RADIUS`] reach.
pub const CRAB_SCALE: i64 = 12;

/// How close (in [`UNIT`]) the crab must get to a living player to "grab" (down) it.
/// Stands in for the giant crab's reach without modelling its limbs. Kept small so a
/// player who keeps a little gap survives. FEEL KNOB: exact reach is the owner's to
/// fine-tune; smaller = more beatable, larger = scarier.
pub const CRAB_GRAB_RADIUS: i64 = 3 * UNIT / 2;

/// Startup head-start, in ticks (~1 s at 30 Hz). For the first this-many ticks the
/// crab neither pursues NOR grabs (see [`Sim::step`]) — it holds its spawn while
/// players orient and break away, so no one is caught the instant the round starts.
/// Counted in ticks (sim state, identical on every peer), never wall-clock, so it stays
/// deterministic. FEEL KNOB.
const STARTUP_GRACE_TICKS: u64 = 30;

/// Minimum spawn separation (in [`UNIT`]) enforced between the crab and the NEAREST
/// player at round start (see [`Sim::spawn_crab`]). Belt-and-braces with
/// [`STARTUP_GRACE_TICKS`]: even if a future roster/layout would drop a player close to
/// the crab's nominal spawn, the crab is pushed out along the spawn line so no one
/// starts inside its reach. Comfortably larger than [`CRAB_GRAB_RADIUS`].
const MIN_CRAB_SPAWN_DISTANCE: i64 = 12 * UNIT;

/// How close (in [`UNIT`]) a living player must be to the extraction point, AND
/// holding [`buttons::ACTION`], to extract and win the round.
pub const EXTRACT_RADIUS: i64 = 2 * UNIT;

/// Per-tick cap on yaw turn from look input, in [`trig::TURN`] units (a full circle
/// is [`trig::TURN`]). At 30 Hz, [`trig::TURN`]/24 per tick ≈ half a turn per second
/// at full deflection — brisk but not instant, and bounded so a single tick can't
/// spin a peer wildly. `pub(crate)`: the render layer derives its radians-per-tick cap
/// from this one constant (`render::MAX_YAW_PER_TICK_RADIANS`), so the divisor is
/// stated exactly once.
pub(crate) const MAX_YAW_TURNS_PER_TICK: i32 = trig::TURN / 24;

/// What a player is doing in the round. Drives both sim logic (only `Alive` players
/// move, get hunted, and can extract) and rendering (downed = ragdoll/marker,
/// extracted = removed/safe).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerStatus {
    /// Up and playing: moves on input, is a crab target, can reach extraction.
    Alive,
    /// Grabbed by the crab — out of the round. In the gray-box this is terminal (no
    /// revive yet); it stops the player from moving or being targeted.
    Downed,
    /// Reached the extraction point and pressed action — safe, round-clearing.
    Extracted,
}

/// A point on the ground plane, in [`UNIT`] fixed-point world coordinates. The world
/// is the XZ-plane (`x` right, `z` forward) at Y=0 — so this is named `x`/`z`, not a
/// bare `(i64, i64)` whose `.1` a reader would mistake for the (unused) up axis. One
/// type for every entity's position; convert with [`Pos::to_meters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pos {
    pub x: i64,
    pub z: i64,
}

impl Pos {
    /// This position in meters, as `(x, z)` — THE fixed-point→meters rule (`coord / UNIT`),
    /// stated once so the render/bridge/probe conversions can't drift. The f32 view is for
    /// presentation and diagnostics only; sim logic never reads it back.
    pub fn to_meters(self) -> (f32, f32) {
        (self.x as f32 / UNIT as f32, self.z as f32 / UNIT as f32)
    }

    /// Inverse of [`Pos::to_meters`]: meters onto the fixed-point grid (truncating cast,
    /// exactly the cast the external-crab bridge has always used).
    pub fn from_meters(x_m: f32, z_m: f32) -> Self {
        Pos {
            x: (x_m * UNIT as f32) as i64,
            z: (z_m * UNIT as f32) as i64,
        }
    }
}

/// A first-person player: a capsule on the ground plane with a facing yaw. Position is
/// fixed-point world XZ at Y=0 (the player walks the ground); yaw is a [`trig`]
/// turn-unit angle the client reads to aim the FP camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player {
    pos: Pos,
    yaw: i32,
    status: PlayerStatus,
}

impl Player {
    /// Rebuild a [`Player`] from its parts — the inverse of the [`pos`](Player::pos) /
    /// [`yaw`](Player::yaw) / [`status`](Player::status) accessors, used by
    /// [`CoreSnapshot::from_bytes`](crate::snapshot::CoreSnapshot::from_bytes) to decode a
    /// player off the wire. `pub(crate)` (not `pub`): only the snapshot seam reconstructs a
    /// player; round logic still owns every in-round transition through [`Sim::step`].
    pub(crate) fn from_parts(pos: Pos, yaw: i32, status: PlayerStatus) -> Self {
        Self { pos, yaw, status }
    }

    /// World position on the ground plane (Y is 0).
    pub fn pos(self) -> Pos {
        self.pos
    }
    /// Facing angle in [`trig`] turn units (`0..`[`trig::TURN`]); convert to radians
    /// with [`trig_client::turns_to_radians`] for the camera.
    pub fn yaw(self) -> i32 {
        self.yaw
    }
    /// Alive / Downed / Extracted — read-only (round logic owns transitions; a render
    /// sub must not mutate it, which is why it's an accessor, not a public field).
    pub fn status(self) -> PlayerStatus {
        self.status
    }
}

/// The one giant crab: a ground-plane position with a facing yaw, driven from OUTSIDE the sim by
/// the real rapier-NN body ([`Sim::set_external_crab_pose`], the only way it moves). Rendered
/// [`CRAB_SCALE`]× a player; the sim models only its position and a [`CRAB_GRAB_RADIUS`] reach
/// (the limbs live in the NN body, not here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crab {
    pos: Pos,
    yaw: i32,
}

impl Crab {
    /// Rebuild a [`Crab`] from its integer pose — the inverse of the [`pos`](Crab::pos) /
    /// [`yaw`](Crab::yaw) accessors, used by
    /// [`CoreSnapshot::from_bytes`](crate::snapshot::CoreSnapshot::from_bytes) to decode the
    /// crab off the wire. `pub(crate)`: live, the crab pose only ever moves through
    /// [`Sim::set_external_crab_pose`] — this reconstructs it at the snapshot seam.
    pub(crate) fn from_parts(pos: Pos, yaw: i32) -> Self {
        Self { pos, yaw }
    }

    /// World position on the ground plane.
    pub fn pos(self) -> Pos {
        self.pos
    }
    /// Facing yaw in [`trig`] turn units; points toward its current target.
    pub fn yaw(self) -> i32 {
        self.yaw
    }
}

/// The fixed pickup point a player reaches to clear the round. A constant in the
/// gray-box, but carried in state (and hashed) so a later sub can move/randomize it
/// per round without reworking the desync check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractionPoint {
    pos: Pos,
}

impl ExtractionPoint {
    /// World position on the ground plane.
    pub fn pos(self) -> Pos {
        self.pos
    }
}

/// Round outcome — the win/lose state a client reads to end the round. Once it
/// leaves [`Outcome::Ongoing`] it never flips back (the round is over).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Round in progress: at least one player is still `Alive` and none has extracted.
    Ongoing,
    /// A player reached the extraction point — round won.
    Extracted,
    /// Every player has been downed — round lost.
    Wiped,
}

/// The full deterministic game state. Everything that affects future ticks lives
/// here and nowhere else (no globals, no wall-clock reads), so a peer can be
/// reconstructed from it alone and [`Sim::state_hash`] can cover it completely.
#[derive(Debug, Clone)]
pub struct Sim {
    tick: u64,
    /// First-person players. `BTreeMap` (per the determinism contract) so iteration
    /// is in `PlayerId` order on every peer.
    players: BTreeMap<PlayerId, Player>,
    crab: Crab,
    extraction: ExtractionPoint,
    /// The round result. Derived from the player statuses each tick by
    /// [`Sim::settle_outcome`] AND stored, because [`Sim::step`] needs it to freeze the
    /// world after the round is decided. Invariant: once it leaves `Ongoing` the
    /// statuses are frozen (step returns early), so the stored value can never disagree
    /// with them.
    outcome: Outcome,
    /// The one sanctioned randomness source (see [`Sim::rng`]). Its stream position is
    /// hashed and reproduced across peers, so it is genuine sim state — never reseed it
    /// mid-sim EXCEPT on a deterministic restart (see [`Sim::step`]'s RESTART handling,
    /// which every peer applies on the same tick). The crab pursuit draws nothing; the
    /// field stays so later loop variation (spawn jitter, crab feints) has a
    /// deterministic source ready.
    rng: ChaCha8Rng,
    /// Whether [`buttons::RESTART`] was held by some player on the PREVIOUS tick — the
    /// edge-trigger latch (see [`buttons::RESTART`]). It gates a future tick's behaviour
    /// and is identical on every peer, so it is folded into [`Sim::state_hash`].
    restart_held: bool,
    /// The tick the CURRENT round began at: 0 for a fresh sim, the restart tick after a
    /// [`buttons::RESTART`] (which resets state without rewinding the tick — rl#204). The
    /// startup grace ([`STARTUP_GRACE_TICKS`]) counts from here, so a restarted round gets
    /// its full crab head-start. Gates behaviour, so it is folded into [`Sim::state_hash`] —
    /// identical on every peer that STEPS. (An adopting client mirror never steps and keeps
    /// its construction value, so a stepping-vs-mirror hash comparison — the `game net`
    /// `--hash-log` proof — is sound only while no restart fires; same scoping as
    /// `restart_held` and `external_crab_digest`, likewise hashed but not snapshot-carried.)
    round_start: u64,
    /// The immutable round CONFIG — the match seed and the player roster — kept so a
    /// deterministic restart ([`buttons::RESTART`]) can rebuild the initial state in
    /// place. NOT folded into [`Sim::state_hash`]: it can't differ between in-sync peers,
    /// and a peer built with a different roster is already a different game the cross-check
    /// surfaces via the player state it does hash.
    config: RoundConfig,
    /// The peer-comparable digest of the REAL rapier crab's full physics state for this tick
    /// (every actuated body's pose + velocity bits — see [`crab_world::bot::physics_digest`]),
    /// supplied each tick by the deterministic driver alongside the pose via
    /// [`Sim::set_external_crab_pose`]. ALWAYS folded into [`Sim::state_hash`], so two peers whose
    /// float bodies — or whose policy weights, folded into this digest by the bridge — diverge
    /// desync on the tick it happens. Sound cross-peer only because the driver pumps a
    /// deterministic, wall-clock-free number of physics steps per lockstep tick
    /// ([`crate::cadence::PhysicsCadence`]), pushing ONE pose+digest per APPLIED tick. `0` until
    /// a pose is first pushed (a never-driven crab folds a constant `0`, still deterministic).
    external_crab_digest: u64,
}

/// The arguments that built a [`Sim`], retained so [`Sim::step`] can rebuild the
/// initial round on a deterministic restart. Not hashed — it never changes and is the
/// same on every peer (see [`Sim::config`]).
#[derive(Debug, Clone)]
struct RoundConfig {
    seed: u64,
    /// Foot players, in `PlayerId` order.
    players: Vec<PlayerId>,
}

impl Sim {
    /// Create the initial round: one player per id on a deterministic spawn ring, the
    /// giant crab off to one side, and the extraction point across the map. RNG seeded
    /// from `seed` (the shared match seed all peers agree on at session start). Layout
    /// is a pure function of the sorted id set, so the starting state is identical on
    /// every peer regardless of the order `players` arrives in.
    pub fn new(seed: u64, players: &[PlayerId]) -> Self {
        let mut sorted: Vec<PlayerId> = players.to_vec();
        sorted.sort();
        sorted.dedup();
        let config = RoundConfig {
            seed,
            players: sorted,
        };
        let (players, crab, extraction) = Self::spawn_state(&config);
        Self {
            tick: 0,
            players,
            crab,
            extraction,
            outcome: Outcome::Ongoing,
            rng: ChaCha8Rng::seed_from_u64(seed),
            restart_held: false,
            round_start: 0,
            config,
            external_crab_digest: 0,
        }
    }

    /// Drive the crab's ground position + facing yaw from outside the sim — the ONLY way the
    /// giant crab moves (rl#114: there is no built-in integer pursuit, so a round that can't arm
    /// the NN crab REFUSES loudly rather than substituting a fake crab). The real rapier-NN crab
    /// body ([`crate::external_crab`]) calls this each tick BEFORE advancing, so the
    /// grab/extraction checks resolve against the body's current position. `pos`/`yaw` are genuine
    /// hashed state; `phys_digest` is the float body's per-tick digest (see
    /// [`external_crab_digest`](Sim::external_crab_digest)). Seed it once at round setup with the
    /// crab's spawn pose, then push the body's pose each tick.
    pub fn set_external_crab_pose(&mut self, pos: Pos, yaw: i32, phys_digest: u64) {
        self.crab.pos = pos;
        self.crab.yaw = yaw;
        self.external_crab_digest = phys_digest;
    }

    /// Rebuild the round's WORLD to its spawn state from the stored [`config`](Sim::config)
    /// — the deterministic restart ([`buttons::RESTART`] in [`Sim::step`]). Rebuilds the
    /// SAME way construction does (both call [`Sim::spawn_state`]) and re-seeds the RNG from
    /// the (constant) match seed; since every peer restarts on the same tick from the same
    /// config, all peers land on the identical fresh state.
    ///
    /// Deliberately does NOT rewind [`tick`](Sim::tick): the restart is a state-reset AT the
    /// current tick, recorded in [`round_start`](Sim::round_start). The tick is the shared
    /// currency between the sim and the server's tick space (the assemble cursor, the pending
    /// tick, the roster schedule), all of which are monotone — a tick rewind desynced them
    /// and wedged the match permanently on `next_tick_ready` (rl#204), and would make the
    /// roster mirrors in [`crate::server::Server::step_next`] consult the ancient tick-0
    /// roster.
    ///
    /// Also leaves [`config`](Sim::config) and [`restart_held`](Sim::restart_held) alone:
    /// `config` is the rebuild source, and the restart-edge latch MUST survive the
    /// rebuild — clearing it would let a still-held R re-trigger every tick (the
    /// level-trigger bug `restart_is_edge_triggered_not_level` guards). [`Sim::step`] owns
    /// that latch; reset only touches the round/world fields.
    fn reset(&mut self) {
        let (players, crab, extraction) = Self::spawn_state(&self.config);
        self.round_start = self.tick;
        self.players = players;
        self.crab = crab;
        self.extraction = extraction;
        self.outcome = Outcome::Ongoing;
        self.rng = ChaCha8Rng::seed_from_u64(self.config.seed);
    }

    /// Spawn a mid-game joiner INTO the live round — the host-authoritative snapshot-transfer
    /// join ([[mp-minecraft-model]]). Inserts a fresh [`Player`] for `pid` and folds it into the
    /// [`config`](Sim::config) roster WITHOUT disturbing the ongoing state: the crab, extraction,
    /// incumbents, `tick`, `rng`, and `outcome` are all untouched, so the joiner drops into the
    /// match exactly where it stands rather than rebuilding the world to spawn state (which is
    /// what round RESTART does via [`reset`](Sim::reset)). The joiner never re-simulates the
    /// incumbents' warm rapier world — it renders the host's output pose carried in the snapshot —
    /// so warm-cache divergence is never on the wire. Only the authoritative host calls this (a
    /// client adopts the resulting snapshot), so the spawn position needs no cross-peer
    /// determinism — it reuses the standing ring convention (see [`spawn_state`](Sim::spawn_state))
    /// so the joiner appears with the pack, and incumbents keep their live positions (a cosmetic
    /// overlap they walk out of). Idempotent: a pid already present is left exactly as-is.
    pub fn spawn_joining_player(&mut self, pid: PlayerId) {
        if self.players.contains_key(&pid) {
            return;
        }
        // Keep config.players the sorted/deduped roster of record so a later RESTART rebuilds with
        // the joiner too (reset() respawns from config).
        self.config.players.push(pid);
        self.config.players.sort();
        self.config.players.dedup();
        // Ring slot from the joiner's index in the new roster — the same layout construction uses, so
        // the joiner spawns in formation. Position is cosmetic here (host-authoritative), not hashed
        // cross-peer state, so it need only be sane.
        let idx = self
            .config
            .players
            .iter()
            .position(|p| *p == pid)
            .unwrap_or(0) as i64;
        let n = self.config.players.len() as i64;
        let x = (idx - n / 2) * 2 * UNIT;
        self.players.insert(
            pid,
            Player {
                pos: Pos { x, z: 0 },
                yaw: 0,
                status: PlayerStatus::Alive,
            },
        );
    }

    /// Whether `pid` is a foot player in the round — the [`spawn_joining_player`](Sim::spawn_joining_player)
    /// idempotency guard the authoritative server checks before spawning a newly-rostered joiner.
    pub fn has_player(&self, pid: PlayerId) -> bool {
        self.players.contains_key(&pid)
    }

    /// Remove a DEPARTED player from the live round — the inverse of
    /// [`spawn_joining_player`](Sim::spawn_joining_player), driven the same way: the authoritative
    /// server derives it from the roster schedule ([`crate::server::Server::depart`] shrank the
    /// roster after the peer's link died), so the sim, the server, and the wire roster stay one
    /// source. Dropped from [`config`](Sim::config) too, so a later RESTART rebuilds without the
    /// departed player. Only the host calls this; clients adopt the resulting snapshot (which no
    /// longer carries the player). Idempotent: an absent pid is a no-op.
    pub fn despawn_departed_player(&mut self, pid: PlayerId) {
        self.players.remove(&pid);
        self.config.players.retain(|p| *p != pid);
    }

    /// The tick-0 entity layout for a [`RoundConfig`] — the SINGLE source of the spawn
    /// arrangement, shared by [`Sim::new`] and [`Sim::reset`] so the two can't drift. Pure
    /// function of the (integer, sorted) config, so every peer composes the identical
    /// starting world. Returns the foot players, the crab, and the extraction point.
    fn spawn_state(cfg: &RoundConfig) -> (BTreeMap<PlayerId, Player>, Crab, ExtractionPoint) {
        let mut map = BTreeMap::new();
        let n = cfg.players.len() as i64;
        for (i, &id) in cfg.players.iter().enumerate() {
            // Spawn ring near the origin; spacing in world units, all facing +Z
            // (yaw 0). Integer layout → identical everywhere.
            let x = (i as i64 - n / 2) * 2 * UNIT;
            map.insert(
                id,
                Player {
                    pos: Pos { x, z: 0 },
                    yaw: 0,
                    status: PlayerStatus::Alive,
                },
            );
        }
        // Extraction is across the map at +Z; players spawn at the origin.
        let extraction = ExtractionPoint {
            pos: Pos { x: 0, z: 40 * UNIT },
        };
        let crab = Self::spawn_crab(&map);
        (map, crab, extraction)
    }

    /// The crab's spawn pose for a given set of foot players. It sits BETWEEN spawn and
    /// the +Z extraction (around the midpoint, offset in X so it's an obstacle to dodge,
    /// not a head-on instant grab). Then, as a guard against any roster/layout that
    /// would drop a player right next to it, push it OUT along the spawn→nearest-player
    /// line until at least [`MIN_CRAB_SPAWN_DISTANCE`] away — so no one ever starts inside
    /// its reach. Pure integer math (the same deterministic isqrt the pursuit uses), so
    /// every peer computes the identical spawn.
    fn spawn_crab(players: &BTreeMap<PlayerId, Player>) -> Crab {
        let mut pos = Pos {
            x: 6 * UNIT,
            z: 20 * UNIT,
        };
        // Nearest foot player to the nominal spawn (PlayerId order via BTreeMap breaks
        // ties; planes aren't crab targets, so they don't gate the spawn).
        if let Some(nearest) = players
            .values()
            .min_by_key(|p| dist2_i128(pos.x - p.pos.x, pos.z - p.pos.z))
        {
            let dx = pos.x - nearest.pos.x;
            let dz = pos.z - nearest.pos.z;
            let d2 = dist2_i128(dx, dz);
            let min = MIN_CRAB_SPAWN_DISTANCE as i128;
            if d2 < min * min {
                // Too close: shove the crab to exactly MIN_CRAB_SPAWN_DISTANCE along the
                // vector from that player toward the crab (away from the player). If the
                // crab sits exactly on the player (d2==0) there's no direction to push
                // along — fall back to +X, an arbitrary but deterministic escape.
                let dist = isqrt_i128(d2);
                let (ux, uz, len) = if dist > 0 {
                    (dx as i128, dz as i128, dist)
                } else {
                    (1, 0, 1)
                };
                pos.x = nearest.pos.x + (ux * min / len) as i64;
                pos.z = nearest.pos.z + (uz * min / len) as i64;
            }
        }
        Crab { pos, yaw: 0 }
    }

    /// Every participant the sim simulates: each on-foot player. The lockstep boundary
    /// requires exactly one input per id in this set.
    fn participant_ids(&self) -> impl Iterator<Item = PlayerId> + '_ {
        self.players.keys().copied()
    }

    /// Fail-loud guard for the lockstep boundary (rl#105): every participant must have
    /// supplied this tick's input. Panics naming the missing id rather than letting
    /// [`Sim::step`] default it to neutral — one peer applying real input where another
    /// applied none diverges `state_hash` invisibly.
    fn require_complete_inputs(&self, inputs: &BTreeMap<PlayerId, Input>) {
        for id in self.participant_ids() {
            assert!(
                inputs.contains_key(&id),
                "lockstep input incomplete: no input for {id:?} (have {:?}); defaulting \
                 to neutral would desync peers — refusing",
                inputs.keys().collect::<Vec<_>>(),
            );
        }
    }

    /// Advance one tick: move each living player by its input (relative to facing),
    /// step the crab toward its nearest living prey, resolve grabs and extractions,
    /// then settle the round outcome. All in `PlayerId` order; pure integer math.
    ///
    /// `inputs` MUST hold an entry for every participant the sim tracks; a missing input
    /// panics ([`Sim::require_complete_inputs`]) rather than fabricating a neutral.
    ///
    /// Returns whether this tick was a RESTART edge — the world was rebuilt to spawn state
    /// at this (monotone) tick. An event, not state: the driver hangs its per-round resets
    /// (physics cadence, crab-body respawn) off this exact edge, which the tick counter can
    /// no longer signal since a restart doesn't rewind it (rl#204).
    pub fn step(&mut self, inputs: &BTreeMap<PlayerId, Input>) -> bool {
        self.require_complete_inputs(inputs);
        self.tick += 1;

        // Restart, edge-triggered: any player newly pressing RESTART rebuilds the round
        // to spawn state at this tick and stops. Checked BEFORE the freeze below so a
        // decided (won/lost) round can be restarted. Edge, not level: holding R restarts
        // once.
        let restart_now = inputs.values().any(|i| i.pressed(buttons::RESTART));
        let restart_edge = restart_now && !self.restart_held;
        self.restart_held = restart_now;
        if restart_edge {
            self.reset();
            return true;
        }

        // Once the round is decided, freeze the world: no more movement or grabs, so
        // every peer that reached the same outcome holds an identical final state.
        if self.outcome != Outcome::Ongoing {
            return false;
        }

        // 1) Players. Iterate the players map (BTreeMap → PlayerId order), not the
        //    inputs, so apply order is the state's own order, independent of how
        //    `inputs` was built.
        for (id, p) in self.players.iter_mut() {
            if p.status != PlayerStatus::Alive {
                continue;
            }
            // require_complete_inputs (step entry) guarantees an entry per participant.
            let inp = inputs[id];
            Self::advance_player(p, inp);
        }

        // The crab's grabs are disarmed during the startup grace (see [`STARTUP_GRACE_TICKS`]).
        // Counted from round_start, not tick 0, so a RESTART (state-reset at the current
        // tick) grants the fresh round its full head-start too.
        let armed = self.tick > self.round_start + STARTUP_GRACE_TICKS;

        // 2) Crab MOVE: the crab pose is whatever the external NN body last pushed
        //    ([`Sim::set_external_crab_pose`], the only mover); the grab/extraction below read
        //    `self.crab.pos`.

        // 3) Grabs: any living player within the crab's reach is downed (disarmed during
        //    the startup grace, so a player spawned near the crab isn't grabbed before
        //    they can move).
        let crab = self.crab.pos;
        if armed {
            for p in self.players.values_mut() {
                if p.status == PlayerStatus::Alive
                    && within(p.pos.x, p.pos.z, crab.x, crab.z, CRAB_GRAB_RADIUS)
                {
                    p.status = PlayerStatus::Downed;
                }
            }
        }

        // 4) Extraction: a living player at the point, holding ACTION, extracts.
        let ex = self.extraction.pos;
        for (id, p) in self.players.iter_mut() {
            if p.status == PlayerStatus::Alive
                && within(p.pos.x, p.pos.z, ex.x, ex.z, EXTRACT_RADIUS)
                && inputs[id].pressed(buttons::ACTION)
            {
                p.status = PlayerStatus::Extracted;
            }
        }

        // 5) Settle the round outcome (first decisive condition wins; extraction
        //    beats a wipe on the same tick — a rescue at the buzzer counts).
        self.outcome = self.settle_outcome();
        false
    }

    /// Move ONE player by one input tick: integrate the bounded yaw delta, then translate
    /// relative to the new facing — pure fixed-point, no dependence on other players or the
    /// crab. The single home for the foot-player mover, shared by [`step`](Sim::step) (which
    /// runs it per living player) and [`predict_player`](Sim::predict_player) (client-side
    /// prediction), so the two can't drift ([[collapse-global-mutable-state-not-coordinate]]).
    fn advance_player(p: &mut Player, inp: Input) {
        // Look: integrate the bounded yaw delta, wrapping into [0, TURN).
        let dyaw =
            (inp.look_yaw as i64 * MAX_YAW_TURNS_PER_TICK as i64 / Input::AXIS_SCALE as i64) as i32;
        p.yaw = trig::wrap_turns(p.yaw + dyaw);

        // Move relative to facing: forward is the yaw direction, strafe is +90°.
        // velocity = (forward * move_forward + right * move_strafe), each axis
        // scaled by PLAYER_SPEED, all in fixed-point with one final descale.
        let (sin, cos) = trig::sin_cos(p.yaw); // fixed-point, scale trig::ONE
        // Forward unit vector (sin, cos) in world XZ; right is (cos, -sin).
        let strafe = inp.move_strafe as i64; // units of AXIS_SCALE
        let forward = inp.move_forward as i64;
        let vx = sin * forward + cos * strafe;
        let vz = cos * forward - sin * strafe;
        // Two descales: by AXIS_SCALE (stick) and trig::ONE (the unit vector), then up by
        // PLAYER_SPEED. Integer division truncates identically on all targets.
        let denom = Input::AXIS_SCALE as i64 * trig::ONE as i64;
        p.pos.x += vx * PLAYER_SPEED / denom;
        p.pos.z += vz * PLAYER_SPEED / denom;
    }

    /// Client-side prediction of the LOCAL player only: re-apply one of its
    /// still-in-flight inputs on top of an adopted authoritative snapshot, so a remote client's
    /// own avatar responds at input latency instead of round-trip latency. Mirrors [`step`]'s
    /// per-player guards exactly — a player only moves while ALIVE and the round is ONGOING — so
    /// the replay lands where the host's own step will. Only the foot mover is predicted; the
    /// crab, grabs, extraction, and outcome are authoritative and never predicted (they arrive in
    /// the next snapshot). Purely cosmetic on the client: overwritten by the next snapshot, never
    /// fed back into the sim's hash or shipped anywhere ([[silent-fallback-antipattern]] — this is
    /// prediction, not a second authority).
    pub(crate) fn predict_player(&mut self, id: PlayerId, inp: Input) {
        if self.outcome != Outcome::Ongoing {
            return;
        }
        if let Some(p) = self.players.get_mut(&id)
            && p.status == PlayerStatus::Alive
        {
            Self::advance_player(p, inp);
        }
    }

    /// Ground position of the living player nearest the crab, or `None` if every player
    /// is downed/extracted. The NN-crab bridge reads this to aim the rapier crab at its
    /// prey, keeping the crab hunting the same player the round's grab logic resolves against.
    pub fn nearest_living_player_pos(&self) -> Option<Pos> {
        self.nearest_living_player().map(|p| p.pos)
    }

    /// The living player nearest the crab (ties broken by `PlayerId` order via the
    /// `<` comparison, so the choice is deterministic), or `None` if none are alive.
    fn nearest_living_player(&self) -> Option<Player> {
        let c = self.crab.pos;
        // Squared distance in i128 (positions are unbounded — see the crab step), so
        // the compare can't overflow on a long round.
        let mut best: Option<(i128, Player)> = None;
        for p in self.players.values() {
            if p.status != PlayerStatus::Alive {
                continue;
            }
            let d2 = dist2_i128(p.pos.x - c.x, p.pos.z - c.z);
            // `<` (strict) keeps the FIRST (lowest-id) of equal-distance players,
            // since we iterate in PlayerId order — a stable, peer-agnostic tie-break.
            if best.is_none_or(|(bd, _)| d2 < bd) {
                best = Some((d2, *p));
            }
        }
        best.map(|(_, p)| p)
    }

    /// Decide the round from current player statuses. Extraction (a win) takes
    /// priority over a wipe so a rescue on the same tick as the last grab still wins.
    fn settle_outcome(&self) -> Outcome {
        if self
            .players
            .values()
            .any(|p| p.status == PlayerStatus::Extracted)
        {
            return Outcome::Extracted;
        }
        if !self.players.is_empty()
            && self
                .players
                .values()
                .all(|p| p.status == PlayerStatus::Downed)
        {
            return Outcome::Wiped;
        }
        Outcome::Ongoing
    }

    /// Fold the entire observable state into one value. Equal hashes across peers ⇒
    /// in sync; any divergence flips it (see [`crate::lockstep`] desync check).
    ///
    /// Uses FNV-1a over a canonical byte serialization rather than `Hash`/`Hasher`:
    /// the algorithm is fixed in-code, so the value is stable across processes,
    /// builds, and machines — `DefaultHasher` is explicitly not (its seed/algorithm
    /// may change), which would make cross-peer comparison meaningless. EVERY field of
    /// every entity is folded; a field omitted here is a field whose desync is invisible.
    ///
    /// The exhaustive `let`-destructures below enforce that "EVERY field" promise (rl#70):
    /// no `..`, so a field added to `Sim` or any entity it hashes stops THIS function
    /// compiling until the field is folded in or bound to `_` as a deliberate exclusion —
    /// the forgot-to-hash-it desync becomes a compile error at the site that would be wrong.
    /// A runtime test perturbs each field and checks the hash moves, catching the dual slip
    /// (a field bound here but never written).
    pub fn state_hash(&self) -> u64 {
        // config is deliberately not hashed (see its field doc); bound to `_` so the destructure
        // stays exhaustive without folding it. external_crab_digest IS folded (below).
        let Sim {
            tick,
            players,
            crab,
            extraction,
            outcome,
            rng,
            restart_held,
            round_start,
            config: _,
            external_crab_digest,
        } = self;

        let mut h = Fnv::new();
        h.write(&tick.to_le_bytes());
        for (id, player) in players.iter() {
            let Player { pos, yaw, status } = player;
            h.write(&[id.0]);
            write_pos(&mut h, *pos);
            h.write(&yaw.to_le_bytes());
            h.write(&[status.tag()]);
        }
        let Crab { pos, yaw } = crab;
        write_pos(&mut h, *pos);
        h.write(&yaw.to_le_bytes());
        let ExtractionPoint { pos } = extraction;
        write_pos(&mut h, *pos);
        h.write(&[outcome.tag()]);
        // The restart edge-latch gates whether next tick's RESTART press fires, so a
        // divergence in it would desync the restart.
        h.write(&[u8::from(*restart_held)]);
        // round_start gates the post-restart startup grace, so a divergence in it would
        // arm the crab on different ticks across peers.
        h.write(&round_start.to_le_bytes());
        // Hash the RNG stream position so a desync in random draws is caught even before
        // it manifests in an entity. Cloning and drawing one block reflects the
        // generator's position without disturbing the real stream.
        h.write(&rand::Rng::r#gen::<u64>(&mut rng.clone()).to_le_bytes());
        // The float NN crab's per-tick physics+weights digest — always folded, so the
        // articulated body (and the policy weights, via the bridge) is part of the desync check,
        // not just the quantized 2D pose hashed above. A never-driven crab folds its seeded `0`.
        h.write(&external_crab_digest.to_le_bytes());
        h.finish()
    }

    /// The sanctioned randomness source for sim logic. Drawing from it advances shared
    /// state every peer tracks; never reach for `thread_rng`. (The gray-box crab is
    /// pure arithmetic and uses none; this is here for later loop variation.)
    ///
    /// Exposes the concrete `ChaCha8Rng` deliberately: the exact generator is part of
    /// the determinism contract (a different RNG would desync peers), so pinning it in
    /// the type is honest, not a leak.
    pub fn rng(&mut self) -> &mut ChaCha8Rng {
        &mut self.rng
    }

    /// Current tick count (number of [`Sim::step`] calls applied).
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Read-only view of all players in `PlayerId` order — for rendering/tests, never
    /// to drive sim logic (that goes through [`Sim::step`] so it stays deterministic).
    pub fn players(&self) -> impl Iterator<Item = (PlayerId, Player)> + '_ {
        self.players.iter().map(|(&id, &p)| (id, p))
    }

    /// Read one player's state (for rendering its FP view / a remote avatar).
    pub fn player(&self, id: PlayerId) -> Option<Player> {
        self.players.get(&id).copied()
    }

    /// The giant crab (for rendering the threat).
    pub fn crab(&self) -> Crab {
        self.crab
    }

    /// The extraction point (for rendering the objective marker).
    pub fn extraction(&self) -> ExtractionPoint {
        self.extraction
    }

    /// The round outcome (for ending the round / showing win-lose UI).
    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    /// Build the authoritative [`CoreSnapshot`] for this tick — the host-authoritative MP
    /// seam (see [`crate::snapshot`]). A client reads game state from this snapshot, not from
    /// a sim it stepped itself; single-player is the zero-remote case of the same path
    /// ([[sp-is-mp-special-case]]).
    ///
    /// Completeness is COMPILE-ENFORCED exactly like [`Sim::state_hash`]: the destructure
    /// below has NO `..`, so a newly-added authoritative `Sim` field stops this function
    /// compiling until it is carried into the snapshot or bound to `_` as a deliberate
    /// exclusion. The `_`-bound fields are out of the host->client surface by design:
    /// `extraction` is a fixed gray-box constant both sides derive from `config`; `rng`,
    /// `restart_held`, and `round_start` are peer-invariant round bookkeeping the server
    /// alone steps on; and `external_crab_digest` is not carried (the client renders the
    /// crab POSE carried in `crab`, never the solver).
    pub fn core_snapshot(&self) -> CoreSnapshot {
        let Sim {
            tick,
            players,
            crab,
            extraction: _,
            outcome,
            rng: _,
            restart_held: _,
            round_start: _,
            config,
            external_crab_digest: _,
        } = self;
        CoreSnapshot {
            tick: *tick,
            players: players.clone(),
            crab: *crab,
            outcome: *outcome,
            roster: config.players.clone(),
            // Input watermarks are SERVER coordination metadata, not sim state — the sim holds
            // none. [`crate::server::Server::step_next`] stamps them; the client's `Lockstep`
            // stashes + re-stamps them for its mirror re-emit.
            input_next: std::collections::BTreeMap::new(),
        }
    }

    /// Adopt a [`CoreSnapshot`] as this sim's authoritative state — the inverse of
    /// [`core_snapshot`](Sim::core_snapshot) over the carried fields. A client applies the
    /// host's latest snapshot here instead of stepping the sim itself.
    ///
    /// Overwrites exactly the carried fields and leaves the `_`-bound ones (see
    /// [`core_snapshot`](Sim::core_snapshot)) as they are. The destructure has NO `..`, so a
    /// new `CoreSnapshot` field must be handled here too.
    pub fn apply_core_snapshot(&mut self, snapshot: CoreSnapshot) {
        let CoreSnapshot {
            tick,
            players,
            crab,
            outcome,
            roster,
            // Coordination metadata, not sim state — the client's `Lockstep` stashes it
            // (prediction-window prune + mirror re-emit) before handing the snapshot here.
            input_next: _,
        } = snapshot;
        self.tick = tick;
        self.players = players;
        self.crab = crab;
        self.outcome = outcome;
        self.config.players = roster;
    }
}

impl PlayerStatus {
    /// Stable 1-byte tag — explicit (not `as u8` on the enum) so reordering or inserting
    /// variants can't silently shift the value out from under a peer on a different build.
    /// ONE encode source, shared by [`Sim::state_hash`] and the [`CoreSnapshot`] wire
    /// ([`crate::snapshot`]); [`from_tag`](PlayerStatus::from_tag) is its inverse, adjacent
    /// so the two can't drift.
    pub(crate) fn tag(self) -> u8 {
        match self {
            PlayerStatus::Alive => 0,
            PlayerStatus::Downed => 1,
            PlayerStatus::Extracted => 2,
        }
    }

    /// Decode a [`tag`](PlayerStatus::tag); `None` on an unknown value (a malformed wire
    /// snapshot, rejected loudly rather than defaulted).
    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(PlayerStatus::Alive),
            1 => Some(PlayerStatus::Downed),
            2 => Some(PlayerStatus::Extracted),
            _ => None,
        }
    }
}

impl Outcome {
    /// Stable 1-byte tag (see [`PlayerStatus::tag`]). ONE encode source for `state_hash` and
    /// the snapshot wire; [`from_tag`](Outcome::from_tag) is the adjacent inverse.
    pub(crate) fn tag(self) -> u8 {
        match self {
            Outcome::Ongoing => 0,
            Outcome::Extracted => 1,
            Outcome::Wiped => 2,
        }
    }

    /// Decode a [`tag`](Outcome::tag); `None` on an unknown value.
    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Outcome::Ongoing),
            1 => Some(Outcome::Extracted),
            2 => Some(Outcome::Wiped),
            _ => None,
        }
    }
}

/// Squared length of `(dx, dz)` in `i128`. Positions are unbounded (a player can flee
/// the slower crab indefinitely), so `i64·i64` could overflow on a marathon round —
/// which PANICS in a debug build but WRAPS in release, diverging two peers on
/// different build profiles. i128 makes every reachable coordinate's square fit.
fn dist2_i128(dx: i64, dz: i64) -> i128 {
    let dx = dx as i128;
    let dz = dz as i128;
    dx * dx + dz * dz
}

/// Whether `(ax, az)` and `(bx, bz)` are within `r` (all [`UNIT`] fixed-point).
/// Squared compare so there's no sqrt and no float; i128 squares so it can't overflow
/// on unbounded positions.
fn within(ax: i64, az: i64, bx: i64, bz: i64, r: i64) -> bool {
    dist2_i128(ax - bx, az - bz) <= (r as i128) * (r as i128)
}

/// Integer square root (floor) of a non-negative `i128`, via Newton's method on
/// integers. Deterministic on every target (no float `sqrt`, whose last bit can
/// differ across hardware); used to normalize the crab's pursuit vector from an i128
/// squared distance.
fn isqrt_i128(n: i128) -> i128 {
    debug_assert!(n >= 0);
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Deterministic integer trigonometry ([`trig`]) and its client-side float adapter
/// ([`trig_client`]) live in [`super::cordic`]; re-exported here so the sim and its
/// callers keep referring to them as `sim::trig` / `sim::trig_client`.
pub use super::cordic::{trig, trig_client};

/// Fold a [`Pos`] (both coordinates) into the state hash — one call per position so a hashed
/// entity can't accidentally fold X but forget Z. Destructured exhaustively so a new coordinate
/// forces a compile error here (the [`Sim::state_hash`] guard, extended to `Pos`). A free function rather than
/// an `Fnv` method because the hasher is the shared [`crab_world::fnv::Fnv`], which can't name sim's
/// `Pos`.
fn write_pos(h: &mut Fnv, p: Pos) {
    let Pos { x, z } = p;
    h.write(&x.to_le_bytes());
    h.write(&z.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn players(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    /// A complete neutral input map for every participant in `sim`. The lockstep
    /// boundary REQUIRES an entry per participant, so a test driving
    /// idle players spells out their neutral input instead of relying on a
    /// missing-key default.
    fn neutral_for(sim: &Sim) -> BTreeMap<PlayerId, Input> {
        sim.participant_ids()
            .map(|id| (id, Input::default()))
            .collect()
    }

    #[test]
    fn input_bytes_roundtrip() {
        for inp in [
            Input::default(),
            Input::from_axes(1.0, -1.0),
            Input::new(-0.5, 0.25, 1.0, buttons::ACTION),
            Input {
                move_strafe: i16::MIN,
                move_forward: i16::MAX,
                look_yaw: -123,
                buttons: 0xFF,
            },
        ] {
            assert_eq!(Input::from_bytes(inp.to_bytes()), inp);
        }
    }

    /// Mixed foot + piloting determinism. A piloting player is client-local: its
    /// craft flies OFF the wire, so the sim receives a NEUTRAL foot input for it (see
    /// [`crate::render::driver::LocalControl`]). This pins the sim-side contract that a round mixing
    /// a walking player (real foot input) with a "piloting" one (neutral) steps deterministically and
    /// the neutral player stays put — the sim has no separate pilot state to desync, so folding the
    /// pilot in honestly means folding a stationary walker.
    #[test]
    fn mixed_foot_and_neutral_pilot_steps_deterministically() {
        // Step a 2-player round 20 ticks: P0 walks forward (real foot input), P1 "pilots" → the sim
        // sees its NEUTRAL input. Returns the state hash plus each player's start/end position.
        let run = || {
            let mut sim = Sim::new(7, &players(2));
            let p0_start = sim.player(PlayerId(0)).unwrap().pos();
            let p1_start = sim.player(PlayerId(1)).unwrap().pos();
            for _ in 0..20 {
                let mut inputs = neutral_for(&sim);
                inputs.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
                sim.step(&inputs);
            }
            let p0_end = sim.player(PlayerId(0)).unwrap().pos();
            let p1_end = sim.player(PlayerId(1)).unwrap().pos();
            (sim.state_hash(), p0_start, p0_end, p1_start, p1_end)
        };
        let (h1, ..) = run();
        let (h2, p0_start, p0_end, p1_start, p1_end) = run();
        assert_eq!(
            h1, h2,
            "the same mixed foot+pilot inputs must reproduce the state hash"
        );
        assert_eq!(
            p1_start, p1_end,
            "a piloting (neutral-input) player's foot avatar stays put"
        );
        assert_ne!(
            p0_start, p0_end,
            "the walking player actually moved (not a no-op step)"
        );
    }

    #[test]
    fn wire_len_matches_encoding() {
        // The transport hardcodes a frame size from WIRE_LEN; the encoder must emit
        // exactly that many bytes or framing desyncs.
        assert_eq!(Input::default().to_bytes().len(), Input::WIRE_LEN);
    }

    #[test]
    fn from_axes_clamps_and_quantizes_and_is_neutral_look() {
        let i = Input::from_axes(2.0, -2.0);
        assert_eq!((i.move_strafe, i.move_forward), (1000, -1000));
        assert_eq!((i.look_yaw, i.buttons), (0, 0));
        assert_eq!(Input::from_axes(0.0, 0.0), Input::default());
    }

    #[test]
    fn spawn_is_deterministic_regardless_of_player_order() {
        let a = Sim::new(42, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        let b = Sim::new(42, &[PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn forward_input_moves_along_facing() {
        // At yaw 0 a player faces +Z; full forward stick should advance +Z by about
        // PLAYER_SPEED and not move in X.
        let mut sim = Sim::new(0, &players(1));
        let p0 = sim.player(PlayerId(0)).unwrap().pos();
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        sim.step(&inputs);
        let p1 = sim.player(PlayerId(0)).unwrap().pos();
        assert_eq!(p1.x, p0.x, "no X drift facing +Z");
        let dz = p1.z - p0.z;
        assert!(
            (dz - PLAYER_SPEED).abs() <= 1,
            "forward step ≈ PLAYER_SPEED, got {dz}"
        );
    }

    #[test]
    fn strafe_input_moves_sideways_along_x() {
        // At yaw 0 a player faces +Z, so its RIGHT is +X: a full positive strafe slides
        // +X by ≈PLAYER_SPEED with no +Z drift, a negative strafe mirrors to −X. The only
        // test pinning strafe's world direction (the others cover forward/look), so a
        // flipped sign — "strafing goes the wrong way" — is invisible without it.
        let mut sim = Sim::new(0, &players(1));
        let p0 = sim.player(PlayerId(0)).unwrap().pos();
        let mut right = BTreeMap::new();
        right.insert(PlayerId(0), Input::new(1.0, 0.0, 0.0, 0));
        sim.step(&right);
        let p1 = sim.player(PlayerId(0)).unwrap().pos();
        assert_eq!(p1.z, p0.z, "no Z drift strafing at yaw 0");
        let dx = p1.x - p0.x;
        assert!(
            (dx - PLAYER_SPEED).abs() <= 1,
            "strafe-right step ≈ +PLAYER_SPEED in X, got {dx}"
        );
        // And the opposite stick mirrors to −X (a fresh sim so the start is the same).
        let mut sim = Sim::new(0, &players(1));
        let mut left = BTreeMap::new();
        left.insert(PlayerId(0), Input::new(-1.0, 0.0, 0.0, 0));
        sim.step(&left);
        let dx_left = sim.player(PlayerId(0)).unwrap().pos().x - p0.x;
        assert_eq!(dx_left, -dx, "strafe-left mirrors strafe-right exactly");
    }

    #[test]
    fn predict_player_matches_step_for_the_local_avatar() {
        // Client-side prediction of ONE player must land exactly where `step` puts it — both run
        // the shared `advance_player` mover — so a remote client's local-avatar replay converges
        // byte-for-byte with the authoritative host. Move + turn, so the
        // facing-relative translate is exercised, not just an axis-aligned slide.
        let inp = Input::new(0.6, -0.3, 0.4, 0);
        let mut stepped = Sim::new(7, &players(1));
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), inp);
        stepped.step(&inputs);

        let mut predicted = Sim::new(7, &players(1));
        predicted.predict_player(PlayerId(0), inp);

        let sp = stepped.player(PlayerId(0)).unwrap();
        let pp = predicted.player(PlayerId(0)).unwrap();
        assert_eq!(
            (pp.pos(), pp.yaw()),
            (sp.pos(), sp.yaw()),
            "predicted local avatar must equal the stepped avatar"
        );

        // Predicting an id not in the round is a harmless no-op — no panic, no ghost player.
        let before = predicted.state_hash();
        predicted.predict_player(PlayerId(9), inp);
        assert_eq!(
            predicted.state_hash(),
            before,
            "predicting an absent player must change nothing"
        );
    }

    #[test]
    fn look_then_move_turns_the_heading() {
        // Apply a quarter-turn of look over enough ticks, then move forward: the
        // player should now travel along +X (yaw 90°), not +Z.
        let mut sim = Sim::new(0, &players(1));
        // Full positive look for the ticks needed to accrue a quarter turn.
        let ticks = ((trig::TURN / 4) / MAX_YAW_TURNS_PER_TICK) as usize;
        for _ in 0..ticks {
            let mut inp = BTreeMap::new();
            inp.insert(PlayerId(0), Input::new(0.0, 0.0, 1.0, 0));
            sim.step(&inp);
        }
        let before = sim.player(PlayerId(0)).unwrap().pos();
        let mut fwd = BTreeMap::new();
        fwd.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        sim.step(&fwd);
        let after = sim.player(PlayerId(0)).unwrap().pos();
        let dx = after.x - before.x;
        let dz = after.z - before.z;
        assert!(
            dx.abs() > dz.abs(),
            "after a ~quarter turn, forward should move mostly in X (dx={dx}, dz={dz})"
        );
    }

    #[test]
    fn crab_pursues_and_grabs_a_lone_player() {
        // One player standing still; a crab driven toward it (the external driver — production's
        // rapier-NN body, here the deterministic test stand-in) should close in and eventually
        // down it, ending the round as a wipe. The crab holds still through the startup grace, so
        // step PAST the grace before checking that distance closes.
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        let crab_armed = sim.crab().pos();
        let prey = sim.player(PlayerId(0)).unwrap().pos();
        let d_start = dist2(crab_armed, prey);
        drive_crab_toward_prey(&mut sim);
        sim.step(&neutral);
        let d_next = dist2(sim.crab().pos(), sim.player(PlayerId(0)).unwrap().pos());
        assert!(d_next < d_start, "crab must close distance once driven");
        // Run until the round resolves (bounded).
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        assert_eq!(
            sim.outcome(),
            Outcome::Wiped,
            "standing-still player gets caught"
        );
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed
        );
    }

    #[test]
    fn external_crab_pose_seeds_and_digest_is_hashed() {
        // `set_external_crab_pose` is the ONLY way the crab moves: it seeds/updates the
        // pose AND the per-tick physics digest, which is always folded into the hash.
        let pos = Pos {
            x: 7 * UNIT,
            z: -3 * UNIT,
        };
        let yaw = 123;

        let mut sim = Sim::new(0, &players(1));
        let digest = 0xFEED_FACE_DEAD_BEEF;
        sim.set_external_crab_pose(pos, yaw, digest);
        assert_eq!(sim.crab().pos(), pos, "must seed the pose");
        assert_eq!(sim.crab().yaw(), yaw, "must seed the yaw");
        // A different digest must move the hash — the desync teeth for the float body / weights
        // mismatch. (The digest is always folded now, no arm flag to gate it.)
        let h_seed = sim.state_hash();
        sim.set_external_crab_pose(pos, yaw, digest ^ 1);
        assert_ne!(
            h_seed,
            sim.state_hash(),
            "the external crab digest must be folded into the state hash"
        );
    }

    #[test]
    fn reaching_extraction_with_action_wins() {
        // End-to-end: a faster player who DODGES the crab and reaches the point holding
        // ACTION wins. The crab sits between spawn and extraction (at +X), so a straight
        // line gets grabbed; the player instead swings WIDE to -X (away from the crab),
        // dragging it off-axis, then runs up the far side to the point. The speed edge
        // (PLAYER_SPEED > CRAB_SPEED) makes the detour pay off.
        let mut sim = Sim::new(0, &players(1));
        let ex = sim.extraction().pos();
        // Waypoints: out to -X, up the far side, then the point itself.
        let route = [
            Pos {
                x: -30 * UNIT,
                z: 0,
            },
            Pos {
                x: -30 * UNIT,
                z: ex.z,
            },
            ex,
        ];
        let mut wp = 0usize;
        let mut won = false;
        for _ in 0..4000 {
            let p = sim.player(PlayerId(0)).unwrap();
            if p.status() != PlayerStatus::Alive {
                break;
            }
            let pp = p.pos();
            // Advance to the next waypoint once close to the current one.
            if wp < route.len() - 1 && within(pp.x, pp.z, route[wp].x, route[wp].z, UNIT) {
                wp += 1;
            }
            let target = route[wp];
            let want_yaw = trig::atan2_turns(target.x - pp.x, target.z - pp.z);
            // Nudge yaw toward want_yaw via the look axis (sign of the shortest delta).
            let delta = trig::wrap_turns(want_yaw - p.yaw());
            let look = if delta == 0 {
                0.0
            } else if delta < trig::TURN / 2 {
                1.0
            } else {
                -1.0
            };
            let mut inp = BTreeMap::new();
            inp.insert(PlayerId(0), Input::new(0.0, 1.0, look, buttons::ACTION));
            drive_crab_toward_prey(&mut sim);
            sim.step(&inp);
            if sim.outcome() == Outcome::Extracted {
                won = true;
                break;
            }
        }
        assert!(
            won,
            "a player who dodges the crab, reaches the point, and holds ACTION should extract"
        );
    }

    #[test]
    fn outcome_is_frozen_once_decided() {
        // After the round resolves, the WORLD freezes: no player or crab moves and the
        // outcome holds, so peers that reached the same outcome stay identical. (Tick
        // still advances — it counts steps — so we compare the game state, not the
        // tick-inclusive hash; the desync test already proves the hash tracks in step.)
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        assert_ne!(
            sim.outcome(),
            Outcome::Ongoing,
            "round should have resolved"
        );
        let snapshot = |s: &Sim| {
            (
                s.players().collect::<Vec<_>>(),
                s.crab(),
                s.extraction(),
                s.outcome(),
            )
        };
        let frozen = snapshot(&sim);
        // Keep calling the driver — once the round is decided every player is downed, so the driver
        // finds no living prey and pushes nothing, and `step` early-returns on the frozen outcome;
        // between the two the world stays frozen.
        for _ in 0..10 {
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        assert_eq!(
            snapshot(&sim),
            frozen,
            "a decided round must freeze the world"
        );
    }

    #[test]
    fn hash_changes_when_state_changes() {
        let mut sim = Sim::new(0, &players(2));
        let h0 = sim.state_hash();
        let mut inputs = neutral_for(&sim);
        inputs.insert(PlayerId(0), Input::from_axes(1.0, 1.0));
        sim.step(&inputs);
        assert_ne!(sim.state_hash(), h0);
    }

    #[test]
    #[should_panic(expected = "lockstep input incomplete")]
    fn missing_lockstep_input_panics_not_defaults_to_neutral() {
        // A tick stepped without EVERY participant's input must fail loud, not
        // silently default the absentee to neutral — which would desync a peer whose
        // input map WAS complete (the NN-crab pose rides on player state, in the hash).
        // Two players, only one input supplied → hard error at the boundary.
        let mut sim = Sim::new(0, &players(2));
        let mut partial = BTreeMap::new();
        partial.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        sim.step(&partial); // PlayerId(1)'s input is missing
    }

    // --- trig table sanity: pins the integer table to known trig values ---

    #[test]
    fn trig_table_hits_cardinal_points() {
        use trig::{ONE, TURN, cos, sin};
        // sin: 0 at 0/π, +ONE at quarter, −ONE at three-quarter.
        assert_eq!(sin(0), 0);
        assert_eq!(sin(TURN / 2), 0);
        assert!((sin(TURN / 4) - ONE).abs() <= 1);
        assert!((sin(3 * TURN / 4) + ONE).abs() <= 1);
        // cos: +ONE at 0, 0 at quarter, −ONE at half.
        assert!((cos(0) - ONE).abs() <= 1);
        assert!(cos(TURN / 4).abs() <= 1);
        assert!((cos(TURN / 2) + ONE).abs() <= 1);
    }

    #[test]
    fn trig_pythagorean_identity_holds() {
        use trig::{ONE, cos, sin};
        // sin²+cos² ≈ ONE² across the circle (integer table → small rounding slack).
        for k in 0..64 {
            let a = k * (trig::TURN / 64);
            let s = sin(a) as i64;
            let c = cos(a) as i64;
            let r2 = s * s + c * c;
            let one2 = (ONE as i64) * (ONE as i64);
            let err = (r2 - one2).abs();
            assert!(err <= one2 / 500, "sin²+cos² off at {a}: {r2} vs {one2}");
        }
    }

    #[test]
    fn atan2_recovers_cardinal_and_diagonal_directions() {
        use trig::{TURN, atan2_turns};
        // Convention: angle from +Z toward +X.
        let near = |a: i32, b: i32| trig::wrap_turns(a - b).min(trig::wrap_turns(b - a));
        assert!(near(atan2_turns(0, 1), 0) <= 2, "+Z is yaw 0");
        assert!(near(atan2_turns(1, 0), TURN / 4) <= 2, "+X is quarter turn");
        assert!(near(atan2_turns(0, -1), TURN / 2) <= 2, "−Z is half turn");
        assert!(
            near(atan2_turns(-1, 0), 3 * TURN / 4) <= 2,
            "−X is three-quarter turn"
        );
        assert!(
            near(atan2_turns(1, 1), TURN / 8) <= 2,
            "+X+Z diagonal is eighth turn"
        );
    }

    #[test]
    fn isqrt_matches_floor_sqrt() {
        for n in [
            0i128,
            1,
            2,
            3,
            4,
            8,
            15,
            16,
            17,
            100,
            1_000_000,
            1_000_003,
            i64::MAX as i128, // a value far past i64-squared range, to exercise the marathon case
            (i64::MAX as i128) * (i64::MAX as i128),
        ] {
            let r = isqrt_i128(n);
            assert!(
                r * r <= n && (r + 1) * (r + 1) > n,
                "isqrt({n})={r} not floor sqrt"
            );
        }
    }

    #[test]
    fn cordic_table_matches_f64_reference_exactly() {
        // The runtime sine table is built by integer CORDIC (no float) so two peers on
        // different libm agree. This test is the ONLY place float trig appears, and only
        // to PROVE the integer table equals the rounded f64 truth at every entry — if
        // CORDIC drifts, this fails. (The sim never runs this; it pins the table.)
        use trig::{ONE, TURN, cos, sin};
        for a in 0..TURN {
            let want = ((a as f64 / TURN as f64 * std::f64::consts::TAU).sin() * ONE as f64).round()
                as i32;
            assert_eq!(sin(a), want, "sin table off at {a}");
        }
        // cos derived as sin(a+quarter) — spot-check it tracks the reference too.
        for a in (0..TURN).step_by(257) {
            let want = ((a as f64 / TURN as f64 * std::f64::consts::TAU).cos() * ONE as f64).round()
                as i32;
            assert_eq!(cos(a), want, "cos off at {a}");
        }
    }

    #[test]
    fn state_hash_is_sensitive_to_every_hashed_field() {
        // Runtime half of the every-field-hashed guard. The COMPILE-TIME half lives in `state_hash`
        // itself: its exhaustive `let Sim { .. }` (and per-entity) destructures stop that
        // function compiling until a newly-added field is folded in or bound to `_`. This
        // test proves the other direction — that each field the destructure *names* is
        // actually written into the hash, not bound and silently dropped. Together: a new
        // field can be neither forgotten (compile error) nor faked (a binding that hashes
        // nothing fails here).
        //
        // `hash_after` clones the base, mutates one field, and returns the hash; a
        // hashed field must change it, the two excluded fields must not.
        let base = Sim::new(7, &players(2));
        let h0 = base.state_hash();
        let hash_after = |mutate: &dyn Fn(&mut Sim)| {
            let mut s = base.clone();
            mutate(&mut s);
            s.state_hash()
        };
        let foot = PlayerId(0);

        // Hashed fields: perturbing each must flip the hash.
        assert_ne!(hash_after(&|s| s.tick += 1), h0, "tick must be hashed");

        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().pos.x += 1),
            h0,
            "player pos.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().pos.z += 1),
            h0,
            "player pos.z must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().yaw += 1),
            h0,
            "player yaw must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().status = PlayerStatus::Downed),
            h0,
            "player status must be hashed"
        );

        // The crab POSE (driven externally by the NN body) is hashed so the quantized
        // 2D pose stays desync-safe.
        assert_ne!(
            hash_after(&|s| s.crab.pos.x += 1),
            h0,
            "crab pos.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.crab.pos.z += 1),
            h0,
            "crab pos.z must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.crab.yaw += 1),
            h0,
            "crab yaw must be hashed"
        );

        assert_ne!(
            hash_after(&|s| s.extraction.pos.x += 1),
            h0,
            "extraction pos.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.extraction.pos.z += 1),
            h0,
            "extraction pos.z must be hashed"
        );

        assert_ne!(
            hash_after(&|s| s.outcome = Outcome::Wiped),
            h0,
            "outcome must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.restart_held = !s.restart_held),
            h0,
            "restart_held must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.round_start += 1),
            h0,
            "round_start must be hashed (it gates the post-restart grace)"
        );
        // Advancing the generator (without touching anything else) must flip the hash, so a
        // desync in random draws is caught before it surfaces in an entity.
        assert_ne!(
            hash_after(&|s| {
                let _: u64 = rand::Rng::r#gen(s.rng());
            }),
            h0,
            "rng stream position must be hashed"
        );

        // `config` is excluded: perturbing must NOT flip the hash (field doc explains why;
        // nothing in production mutates it).
        assert_eq!(
            hash_after(&|s| s.config.seed ^= 0xdead_beef),
            h0,
            "config is deliberately not hashed (see Sim::config)"
        );
        // `external_crab_digest` is ALWAYS folded into the hash (the crab is always externally
        // driven — no integer-crab gate to fold it out), so perturbing it MUST move the hash:
        // the desync teeth for the float body / weights mismatch.
        assert_ne!(
            hash_after(&|s| s.external_crab_digest ^= 0xdead_beef),
            h0,
            "external_crab_digest must be hashed (always folded since rl#114)"
        );
    }

    #[test]
    fn core_snapshot_roundtrip_reproduces_authoritative_state() {
        // The snapshot-completeness proof: applying a serialized→
        // deserialized `CoreSnapshot` reproduces every authoritative field — the carried
        // ones `state_hash` observes, plus the (unhashed) roster. The COMPILE-TIME half is
        // the no-`..` destructure in `core_snapshot`/`apply_core_snapshot`; this is the
        // runtime half — that the carried fields are actually round-tripped, not bound and
        // dropped.
        //
        // Build a non-trivial source state: step with real movement so tick + player poses
        // carry real values, push a crab pose (digest 0 — the never-warmed value, so the
        // un-carried `external_crab_digest` stays equal across source and target without
        // diverging the hash), and down a player so a non-`Alive` status rides along.
        let mut original = Sim::new(7, &players(3));
        for _ in 0..5 {
            original.set_external_crab_pose(Pos { x: 4200, z: -1300 }, 77, 0);
            let mut inputs = neutral_for(&original);
            *inputs.get_mut(&PlayerId(0)).unwrap() = Input::from_axes(0.3, 1.0);
            original.step(&inputs);
        }
        original.players.get_mut(&PlayerId(1)).unwrap().status = PlayerStatus::Downed;

        let restored = CoreSnapshot::from_bytes(&original.core_snapshot().to_bytes())
            .expect("a freshly-built snapshot must round-trip through bytes");

        // Apply onto a sim that DIFFERS in every carried field but agrees on the un-carried
        // ones (a clone untouched on `extraction`/`rng`/`restart_held`/`round_start`/digest). If
        // apply restores every carried field the destructure names, the full `state_hash`
        // must match — a forgotten restore leaves one of these perturbations standing.
        let mut target = original.clone();
        target.tick = 999;
        target.players.get_mut(&PlayerId(0)).unwrap().pos.x += 12_345;
        target.players.get_mut(&PlayerId(2)).unwrap().yaw += 3;
        target.players.get_mut(&PlayerId(1)).unwrap().status = PlayerStatus::Extracted;
        target.crab.pos = Pos { x: -1, z: -2 };
        target.crab.yaw = 9;
        target.outcome = Outcome::Wiped;
        target.config.players = vec![PlayerId(0)]; // roster differs (unhashed, asserted below)
        assert_ne!(target.state_hash(), original.state_hash());

        target.apply_core_snapshot(restored);
        assert_eq!(
            target.state_hash(),
            original.state_hash(),
            "applying the round-tripped snapshot reproduces every hashed carried field"
        );
        // The roster is config-level (not folded into `state_hash`), so the hash check above
        // can't see it — assert directly that apply carried `config.players` back from the
        // snapshot's `roster` (target's was perturbed to a single id before apply).
        assert_eq!(
            target.config.players, original.config.players,
            "the snapshot must carry the roster too"
        );
    }

    // --- startup grace + deterministic restart ---

    #[test]
    fn crab_holds_and_cannot_grab_during_startup_grace() {
        // During the grace window the crab neither moves nor grabs, even with a player
        // standing on top of it — so no one is caught "the instant the game launched".
        // Spawn a lone player AT the crab's position to make the grab the only thing the
        // grace could be suppressing, then step through the grace and assert: crab
        // stationary, player still Alive, round still Ongoing.
        let mut sim = Sim::new(0, &players(1));
        let crab0 = sim.crab().pos();
        // Place the player exactly on the crab (the harshest case the grace must cover).
        sim.players.get_mut(&PlayerId(0)).unwrap().pos = crab0;
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral);
            assert_eq!(sim.crab().pos(), crab0, "crab holds its spawn during grace");
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "no grab during the startup grace"
            );
            assert_eq!(sim.outcome(), Outcome::Ongoing);
        }
        // The very next tick the crab is armed and grabs the co-located player.
        sim.step(&neutral);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "crab arms and grabs once the grace ends"
        );
    }

    #[test]
    fn crab_spawns_clear_of_every_player() {
        // No player starts within the crab's reach — the spawn guard keeps at least
        // MIN_CRAB_SPAWN_DISTANCE between the crab and the nearest player, for rosters
        // from 1 to 8. (Min distance is well outside CRAB_GRAB_RADIUS, so even with the
        // grace gone no one is in grab range at tick 0.)
        for n in 1..=8u8 {
            let sim = Sim::new(0, &players(n));
            let crab = sim.crab().pos();
            let nearest = sim
                .players()
                .map(|(_, p)| dist2_i128(crab.x - p.pos().x, crab.z - p.pos().z))
                .min()
                .unwrap();
            let min = MIN_CRAB_SPAWN_DISTANCE as i128;
            assert!(
                nearest >= min * min,
                "n={n}: nearest player {} closer than MIN_CRAB_SPAWN_DISTANCE",
                isqrt_i128(nearest)
            );
        }
    }

    #[test]
    fn restart_resets_the_round_to_spawn() {
        // Press RESTART and the WORLD rebuilds to its spawn state — players Alive at
        // spawn, crab at its spawn, outcome Ongoing, matching a fresh round — while the
        // TICK stays monotone: a restart is a state-reset at the current tick, never a
        // tick rewind, so the server's tick space stays aligned (rl#204). (The
        // hash also folds the restart edge-latch, which is legitimately set here because
        // R is held and clear in a never-restarted sim — so we compare the observable
        // round state, not the raw hash. The full-hash agreement BETWEEN peers who both
        // apply the restart is the lockstep test's job.) Run a few ticks and move first
        // so there's real state to discard.
        let mut sim = Sim::new(0xBEEF, &players(2));
        let fresh = Sim::new(0xBEEF, &players(2));
        let mut fwd = BTreeMap::new();
        fwd.insert(PlayerId(0), Input::new(0.3, 1.0, 0.5, 0));
        fwd.insert(PlayerId(1), Input::new(-0.2, 1.0, 0.0, 0));
        for _ in 0..50 {
            sim.step(&fwd);
        }
        // The world minus the tick: the tick deliberately does NOT rewind on restart.
        let world = |s: &Sim| {
            (
                s.players().collect::<Vec<_>>(),
                s.crab(),
                s.extraction(),
                s.outcome(),
            )
        };
        assert_ne!(
            world(&sim),
            world(&fresh),
            "the round should have diverged from spawn before restart"
        );
        // Press R (only player 0 holds it — one peer's press restarts everyone).
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        restart.insert(PlayerId(1), Input::default());
        let edge = sim.step(&restart);
        assert!(edge, "the press reports the restart edge");
        assert_eq!(sim.tick(), 51, "the tick stays monotone across a restart");
        assert_eq!(sim.outcome(), Outcome::Ongoing);
        assert_eq!(
            world(&sim),
            world(&fresh),
            "a restarted round's world matches a fresh one"
        );
    }

    #[test]
    fn restart_works_after_the_round_is_decided() {
        // The point of restart is to play again after a win/loss — so it must fire even
        // once the world is frozen (outcome != Ongoing). Wipe the player, confirm the
        // freeze, then RESTART and confirm a live round is back.
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        assert_eq!(sim.outcome(), Outcome::Wiped, "round should have ended");
        let tick_at_loss = sim.tick();
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        sim.step(&restart);
        assert_eq!(sim.outcome(), Outcome::Ongoing, "restart revives the round");
        assert_eq!(sim.tick(), tick_at_loss + 1, "the tick keeps counting");
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Alive,
            "the player is alive again after a post-loss restart"
        );
    }

    #[test]
    fn restart_is_edge_triggered_not_level() {
        // Holding RESTART across ticks must restart ONCE (on the press), then let the
        // round advance — otherwise a held key would re-spawn the world every tick.
        // The tick is monotone regardless (a restart never rewinds it), so the edge is
        // observed through `step`'s return value.
        let mut sim = Sim::new(0, &players(1));
        let mut held = BTreeMap::new();
        held.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(sim.step(&held), "first press restarts");
        // Keep holding: the latch is set, so subsequent ticks advance normally.
        assert!(!sim.step(&held), "a held key doesn't re-restart");
        assert!(!sim.step(&held), "still held, still no re-restart");
        assert_eq!(sim.tick(), 3, "every tick counted, restart included");
        // Release, then press again: that fresh edge restarts once more.
        assert!(!sim.step(&neutral_for(&sim)), "release: no restart");
        assert!(sim.step(&held), "a new press after release restarts again");
    }

    #[test]
    fn restart_keeps_two_peers_in_lockstep() {
        // The determinism contract under restart: two independent sims fed the identical
        // input stream — including a mid-round RESTART press — stay bit-identical
        // tick-for-tick. This is what makes restart safe over the wire (every peer
        // applies the same RESTART on the same tick).
        let mut a = Sim::new(0x5151, &players(2));
        let mut b = Sim::new(0x5151, &players(2));
        let mut restarts = 0u32;
        for t in 0..120u64 {
            let mut inputs = BTreeMap::new();
            // Both players move; player 1 presses R once at tick 40.
            let restart_bit = if t == 40 { buttons::RESTART } else { 0 };
            inputs.insert(PlayerId(0), Input::new(0.4, 1.0, 0.2, 0));
            inputs.insert(PlayerId(1), Input::new(-0.3, 1.0, -0.1, restart_bit));
            restarts += u32::from(a.step(&inputs));
            b.step(&inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "peers must stay bit-identical across a restart (tick {t})"
            );
        }
        // Non-vacuous: the restart actually fired, and the tick stayed monotone through it.
        assert_eq!(restarts, 1, "the mid-run restart fired exactly once");
        assert_eq!(a.tick(), 120, "a restart never rewinds the tick");
    }

    #[test]
    fn restart_grants_a_fresh_startup_grace() {
        // The grace window counts from the ROUND start, not tick 0 — a restart deep into
        // a match must give the fresh round its full crab head-start, or the crab would
        // be instantly armed and grab anyone respawned near it. Run well past the initial
        // grace, restart, park the player ON the crab, and assert the grace holds for
        // exactly STARTUP_GRACE_TICKS before the grab lands.
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..(3 * STARTUP_GRACE_TICKS) {
            sim.step(&neutral);
        }
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(sim.step(&restart), "the restart fires");
        // Harshest case: the player stands exactly on the restarted crab's spawn.
        let crab0 = sim.crab().pos();
        sim.players.get_mut(&PlayerId(0)).unwrap().pos = crab0;
        for i in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral);
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "no grab during the post-restart grace (tick {i} into the round)"
            );
        }
        sim.step(&neutral);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "the crab arms once the post-restart grace ends"
        );
    }

    fn dist2(a: Pos, b: Pos) -> i128 {
        dist2_i128(a.x - b.x, a.z - b.z)
    }
}
