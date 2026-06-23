//! Deterministic simulation core for lockstep multiplayer: peers run independent
//! copies and only inputs cross the wire (see [`crate::net::lockstep`]), so the sim
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

/// Identifies a player within the sim. Assigned from the connection set in a
/// deterministic order so every peer agrees which id is whom (see
/// [`crate::net::lockstep`]); the sim itself only relies on the ordering being
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
    /// reset) so every peer restarts on the SAME tick and stays in lockstep — a
    /// local-only reset would desync. Edge-triggered (see [`Sim::step`]).
    pub const RESTART: u8 = 1 << 1;
}

/// One player's input for a single tick — the unit that crosses the wire (see
/// [`Input::to_bytes`]). The move/look axes are facing-relative (named for the control
/// intent), not world axes: at a nonzero yaw they do not map to world X/Z.
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
    /// bounded so one tick can't spin a peer arbitrarily.
    pub look_yaw: i16,
    /// Action/button bitfield (see [`buttons`]). Held state, sampled each tick.
    pub buttons: u8,
}

impl Input {
    /// Fixed-point denominator: an axis value of `AXIS_SCALE` is full deflection
    /// (1.0). `move_strafe`, `move_forward`, and `look_yaw` all use this grid.
    pub const AXIS_SCALE: i16 = 1000;
    /// Wire size of one encoded [`Input`]: move_strafe(2) + move_forward(2) +
    /// look_yaw(2) + buttons(1). Drives [`crate::net::transport`]'s frame size — keep
    /// in sync with [`Input::to_bytes`].
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

/// Fixed-point world scale: a position/length value of [`UNIT`] equals one world
/// meter. All world coordinates, radii, and speeds are integers in these units.
pub const UNIT: i64 = 1000;

/// Player walk speed, in [`UNIT`]/tick at full stick. Tuned for the gray-box, not
/// realism: ~5 m/s at 30 Hz (`SPEED * 30 / UNIT`).
const PLAYER_SPEED: i64 = 166;

/// Crab pursuit speed, in [`UNIT`]/tick. Held meaningfully below [`PLAYER_SPEED`]
/// (166) so a dodging player can OUTRUN it and open distance — the tiny-vs-giant
/// asymmetry the loop is built on. Beatability comes mostly from this gap plus
/// [`CRAB_GRAB_RADIUS`] and the [`STARTUP_GRACE_TICKS`] head-start, not from one number.
/// FEEL KNOB: the exact value is for the owner to fine-tune by playing; keep the gap to
/// PLAYER_SPEED wide enough that running away actually works.
const CRAB_SPEED: i64 = 130;

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
/// spin a peer wildly.
const MAX_YAW_TURNS_PER_TICK: i32 = trig::TURN / 24;

// --- Plane flight tuning ---
// All integer, all named, few: crude arcade flight, not a sim. Values are in
// [`UNIT`]/tick for forces/speeds and [`trig`] turn units for the orientation rates.

/// Forward thrust at full throttle, in [`UNIT`]/tick² (added to velocity along the
/// facing each tick). Sized so full throttle climbs against [`PLANE_GRAVITY`] with
/// margin — the plane can gain altitude, not just arrest its fall.
const PLANE_THRUST: i64 = 40;

/// Downward pull per tick, in [`UNIT`]/tick² (subtracted from `vel.y`). Constant —
/// no lift model; throttle-along-facing is what holds altitude, so nose-up + throttle
/// is "climb" and idle is "sink".
const PLANE_GRAVITY: i64 = 16;

/// Velocity retained per tick, as a fraction of [`PLANE_DAMP_DEN`]. `< 1` so speed
/// bleeds off without thrust (drag) and the integrator can't run away to infinity —
/// terminal speed is `thrust * DEN / (DEN - NUM)`. Applied to all three axes.
const PLANE_DAMP_NUM: i64 = 98;
const PLANE_DAMP_DEN: i64 = 100;

/// Per-tick heading change at full yaw stick, in [`trig`] turn units. A touch brisker
/// than a walker's turn (this is a banking aircraft, not feet); still bounded so one
/// tick can't spin a peer.
const PLANE_YAW_RATE: i32 = trig::TURN / 90;

/// Per-tick pitch change at full pitch stick, in [`trig`] turn units.
const PLANE_PITCH_RATE: i32 = trig::TURN / 120;

/// Pitch clamp in [`trig`] turn units (≈±30°): the nose can climb or dive steeply but
/// not loop. With no roll there's no way to recover from inverted, so we never let it
/// get there — keeps the 2-angle orientation always upright.
const PLANE_MAX_PITCH: i32 = trig::TURN * 5 / 60;

/// Plane spawn altitude in [`UNIT`] (metres): start airborne so the pilot is flying
/// from tick 0 (no takeoff roll in this cut).
const PLANE_SPAWN_ALTITUDE: i64 = 30 * UNIT;

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
/// type for every entity's position; convert to meters with `x as f32 / UNIT as f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pos {
    pub x: i64,
    pub z: i64,
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

/// The one giant crab: a point pursuer on the ground plane with a facing yaw (it
/// turns to face its prey). Rendered [`CRAB_SCALE`]× a player; the sim models only
/// its position and a [`CRAB_GRAB_RADIUS`] reach (no limbs yet — gray-box).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crab {
    pos: Pos,
    yaw: i32,
}

impl Crab {
    /// World position on the ground plane.
    pub fn pos(self) -> Pos {
        self.pos
    }
    /// Facing yaw in [`trig`] turn units; points toward its current target.
    pub fn yaw(self) -> i32 {
        self.yaw
    }
}

/// A 3D point in [`UNIT`] fixed-point world coordinates — the flying entities' frame.
/// Feet live on the Y=0 plane and use the 2D [`Pos`]; a plane leaves the ground, so it
/// carries a full `(x, y, z)` (and so does its velocity). +Y is up. Integer only, like
/// everything else the sim evolves, so it folds into [`Sim::state_hash`] bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pos3 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

/// A pilotable plane: a flying body with 3D position + velocity and a 2D orientation
/// (heading yaw + pitch), all integer fixed-point so two peers evolve it identically.
///
/// Crude arcade flight, NOT a sim: throttle thrusts along the heading-and-pitch facing,
/// gravity pulls −Y every tick, and velocity damps so it can't integrate to infinity.
/// Roll is omitted on purpose — yaw + pitch is enough to fly a gray box and keeps the
/// orientation 2-angle (no integer quaternion needed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    pos: Pos3,
    vel: Pos3,
    /// Heading in [`trig`] turn units (the horizontal facing, like a player's yaw).
    heading: i32,
    /// Pitch in [`trig`] turn units: nose up/down off the horizon. Clamped to
    /// ±[`PLANE_MAX_PITCH`] so it can't loop (no roll to recover from inverted).
    pitch: i32,
}

impl Plane {
    /// World position (3D — includes altitude Y, unlike a player's ground [`Pos`]).
    pub fn pos(self) -> Pos3 {
        self.pos
    }
    /// World velocity in [`UNIT`]/tick (3D).
    pub fn vel(self) -> Pos3 {
        self.vel
    }
    /// Heading yaw in [`trig`] turn units; convert with [`trig_client::turns_to_radians`].
    pub fn heading(self) -> i32 {
        self.heading
    }
    /// Pitch in [`trig`] turn units (nose up = positive).
    pub fn pitch(self) -> i32 {
        self.pitch
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
    /// Pilotable planes, keyed by the [`PlayerId`] flying each one. A player is EITHER on
    /// foot (in `players`) OR piloting (here) — the two maps are disjoint, decided once at
    /// [`Sim::new_with_pilots`]. An empty map ⇒ the foot-only game, hashed to zero extra
    /// bytes, so the no-plane sim is byte-identical. `BTreeMap` for `PlayerId`-ordered
    /// iteration, per the contract.
    ///
    /// Planes don't participate in the objective: the crab (`nearest_living_player`),
    /// grabs, extraction, and `settle_outcome` all read only `players`, so a pilot can't
    /// be hunted or extract, and an all-pilot round never resolves (stays `Ongoing`).
    planes: BTreeMap<PlayerId, Plane>,
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
    /// The immutable round CONFIG — the match seed and the foot/pilot roster — kept so a
    /// deterministic restart ([`buttons::RESTART`]) can rebuild the initial state in
    /// place. NOT folded into [`Sim::state_hash`]: it can't differ between in-sync peers,
    /// and a peer built with a different roster is already a different game the cross-check
    /// surfaces via the player/plane state it does hash.
    config: RoundConfig,
    /// SOLO ONLY: the crab's ground position is driven from OUTSIDE the sim (by the real
    /// rapier-simulated NN crab — see [`Sim::set_external_crab_pose`]) instead of by the
    /// built-in integer point-pursuit. When set, [`Sim::step`] skips the pursuit move
    /// (block 2) and trusts whatever `set_external_crab_pose` last wrote; grabs, extraction,
    /// and outcome (blocks 3–5) then resolve against the real crab body's position.
    ///
    /// `false` for every networked round: a float rapier crab is NOT identical across
    /// peers, so the integer pursuit is the cross-peer-safe path and this is the solo
    /// showcase only. Not hashed (like [`config`](Sim::config)).
    crab_external: bool,
}

/// The arguments that built a [`Sim`], retained so [`Sim::step`] can rebuild the
/// initial round on a deterministic restart. Not hashed — it never changes and is the
/// same on every peer (see [`Sim::config`]).
#[derive(Debug, Clone)]
struct RoundConfig {
    seed: u64,
    /// Foot players, in `PlayerId` order. Disjoint from `pilots`.
    players: Vec<PlayerId>,
    /// Pilots (spawn flying a plane). Disjoint from `players`.
    pilots: Vec<PlayerId>,
}

impl Sim {
    /// Create the initial round: one player per id on a deterministic spawn ring, the
    /// giant crab off to one side, and the extraction point across the map. RNG seeded
    /// from `seed` (the shared match seed all peers agree on at session start). Layout
    /// is a pure function of the sorted id set, so the starting state is identical on
    /// every peer regardless of the order `players` arrives in.
    pub fn new(seed: u64, players: &[PlayerId]) -> Self {
        Self::new_with_pilots(seed, players, &[])
    }

    /// Like [`Sim::new`], but each id in `pilots` spawns PILOTING a plane instead of
    /// standing on foot. A pilot is removed from the foot `players` map and added to
    /// `planes`; the two stay disjoint. `pilots` not in the `players` set are ignored.
    /// With `pilots` empty this is byte-for-byte [`Sim::new`]. Callers MUST pass the same
    /// `pilots` on every peer or their sims diverge; the networked path passes none.
    pub fn new_with_pilots(seed: u64, players: &[PlayerId], pilots: &[PlayerId]) -> Self {
        let mut sorted: Vec<PlayerId> = players.to_vec();
        sorted.sort();
        sorted.dedup();
        // Pilots not in the player set are meaningless — keep only the ones that map to
        // a participant, in sorted order, so the retained config is canonical.
        let pilots: Vec<PlayerId> = sorted
            .iter()
            .copied()
            .filter(|id| pilots.contains(id))
            .collect();
        let config = RoundConfig {
            seed,
            players: sorted,
            pilots,
        };
        let (players, planes, crab, extraction) = Self::spawn_state(&config);
        Self {
            tick: 0,
            players,
            planes,
            crab,
            extraction,
            outcome: Outcome::Ongoing,
            rng: ChaCha8Rng::seed_from_u64(seed),
            restart_held: false,
            config,
            crab_external: false,
        }
    }

    /// Put the crab under EXTERNAL control (see [`crab_external`](Sim::crab_external)):
    /// [`Sim::step`] stops running the built-in integer pursuit, and the caller drives the
    /// crab's position each tick with [`Sim::set_external_crab_pose`]. Solo only — a
    /// float-driven crab is not cross-peer deterministic.
    pub fn enable_external_crab(&mut self, external: bool) {
        self.crab_external = external;
    }

    /// Set the crab's ground position + facing yaw directly (see
    /// [`enable_external_crab`](Sim::enable_external_crab)). Has effect only after
    /// `enable_external_crab(true)`; otherwise [`Sim::step`]'s pursuit overwrites it. Call
    /// it each tick BEFORE advancing, so the grab/extraction checks resolve against the
    /// body's current position. `pos`/`yaw` are genuine hashed state.
    pub fn set_external_crab_pose(&mut self, pos: Pos, yaw: i32) {
        self.crab.pos = pos;
        self.crab.yaw = yaw;
    }

    /// Rebuild the round to its tick-0 state from the stored [`config`](Sim::config) —
    /// the deterministic restart ([`buttons::RESTART`] in [`Sim::step`]). Rebuilds the
    /// SAME way construction does (both call [`Sim::spawn_state`]), so a restarted round
    /// is byte-identical to a freshly-constructed one. The RNG is re-seeded from the
    /// (constant) match seed; since every peer restarts on the same tick from the same
    /// config, all peers land on the identical fresh state.
    ///
    /// Deliberately leaves [`config`](Sim::config) and [`restart_held`](Sim::restart_held)
    /// alone: `config` is the rebuild source, and the restart-edge latch MUST survive the
    /// rebuild — clearing it would let a still-held R re-trigger every tick (the
    /// level-trigger bug `restart_is_edge_triggered_not_level` guards). [`Sim::step`] owns
    /// that latch; reset only touches the round/world fields.
    fn reset(&mut self) {
        let (players, planes, crab, extraction) = Self::spawn_state(&self.config);
        self.tick = 0;
        self.players = players;
        self.planes = planes;
        self.crab = crab;
        self.extraction = extraction;
        self.outcome = Outcome::Ongoing;
        self.rng = ChaCha8Rng::seed_from_u64(self.config.seed);
    }

    /// The tick-0 entity layout for a [`RoundConfig`] — the SINGLE source of the spawn
    /// arrangement, shared by [`Sim::new_with_pilots`] and [`Sim::reset`] so the two can't
    /// drift. Pure function of the (integer, sorted) config, so every peer composes the
    /// identical starting world. Returns the foot players, the pilots' planes, the crab,
    /// and the extraction point.
    fn spawn_state(
        cfg: &RoundConfig,
    ) -> (
        BTreeMap<PlayerId, Player>,
        BTreeMap<PlayerId, Plane>,
        Crab,
        ExtractionPoint,
    ) {
        let mut map = BTreeMap::new();
        let mut plane_map = BTreeMap::new();
        let n = cfg.players.len() as i64;
        for (i, &id) in cfg.players.iter().enumerate() {
            // Spawn ring near the origin; spacing in world units, all facing +Z
            // (yaw 0). Integer layout → identical everywhere. A pilot takes the same
            // ground XZ slot but starts airborne over it (a plane, not feet).
            let x = (i as i64 - n / 2) * 2 * UNIT;
            if cfg.pilots.contains(&id) {
                plane_map.insert(
                    id,
                    Plane {
                        pos: Pos3 {
                            x,
                            y: PLANE_SPAWN_ALTITUDE,
                            z: 0,
                        },
                        vel: Pos3::default(),
                        heading: 0,
                        pitch: 0,
                    },
                );
            } else {
                map.insert(
                    id,
                    Player {
                        pos: Pos { x, z: 0 },
                        yaw: 0,
                        status: PlayerStatus::Alive,
                    },
                );
            }
        }
        debug_assert!(
            map.keys().all(|id| !plane_map.contains_key(id)),
            "a participant is on foot XOR piloting — the two maps must stay disjoint"
        );
        // Extraction is across the map at +Z; players spawn at the origin.
        let extraction = ExtractionPoint {
            pos: Pos { x: 0, z: 40 * UNIT },
        };
        let crab = Self::spawn_crab(&map);
        (map, plane_map, crab, extraction)
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

    /// Advance one tick: move each living player by its input (relative to facing),
    /// step the crab toward its nearest living prey, resolve grabs and extractions,
    /// then settle the round outcome. All in `PlayerId` order; pure integer math.
    ///
    /// `inputs` must hold an entry for each player the sim knows about (the lockstep
    /// driver guarantees this by buffering a tick until all peers' inputs arrive). A
    /// missing input is treated as neutral — divergence here would be a logic bug, but
    /// defaulting keeps the step total rather than panicking on a dropped frame.
    pub fn step(&mut self, inputs: &BTreeMap<PlayerId, Input>) {
        self.tick += 1;

        // Restart, edge-triggered: any player newly pressing RESTART rebuilds the round
        // to tick 0 and stops. Checked BEFORE the freeze below so a decided (won/lost)
        // round can be restarted. Edge, not level: holding R restarts once.
        let restart_now = inputs.values().any(|i| i.pressed(buttons::RESTART));
        let restart_edge = restart_now && !self.restart_held;
        self.restart_held = restart_now;
        if restart_edge {
            self.reset();
            return;
        }

        // Once the round is decided, freeze the world: no more movement or grabs, so
        // every peer that reached the same outcome holds an identical final state.
        if self.outcome != Outcome::Ongoing {
            return;
        }

        // 1) Players. Iterate the players map (BTreeMap → PlayerId order), not the
        //    inputs, so apply order is the state's own order, independent of how
        //    `inputs` was built.
        for (id, p) in self.players.iter_mut() {
            if p.status != PlayerStatus::Alive {
                continue;
            }
            let inp = inputs.get(id).copied().unwrap_or_default();

            // Look: integrate the bounded yaw delta, wrapping into [0, TURN).
            let dyaw = (inp.look_yaw as i64 * MAX_YAW_TURNS_PER_TICK as i64
                / Input::AXIS_SCALE as i64) as i32;
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
            // Two descales: by AXIS_SCALE (stick) and trig::ONE (the unit vector),
            // then up by PLAYER_SPEED. Integer division truncates identically on all
            // targets.
            let denom = Input::AXIS_SCALE as i64 * trig::ONE as i64;
            p.pos.x += vx * PLAYER_SPEED / denom;
            p.pos.z += vz * PLAYER_SPEED / denom;
        }

        // 1b) Planes. Each pilot's plane integrates its own flight from its input,
        //     in PlayerId order (BTreeMap). Pure integer math — see `step_plane`.
        for (id, plane) in self.planes.iter_mut() {
            let inp = inputs.get(id).copied().unwrap_or_default();
            step_plane(plane, inp);
        }

        // The crab is disarmed during the startup grace (see [`STARTUP_GRACE_TICKS`]).
        let armed = self.tick > STARTUP_GRACE_TICKS;

        // 2) Crab: aim at and step toward the nearest living player. SKIPPED under
        //    external (solo NN) control — the real crab body owns the position then (see
        //    `crab_external`); the grab/extraction below still read `self.crab.pos`, so
        //    only the *move* is delegated, not the hunt.
        if !self.crab_external
            && armed
            && let Some(target) = self.nearest_living_player()
        {
            let t = target.pos;
            let dx = t.x - self.crab.pos.x;
            let dz = t.z - self.crab.pos.z;
            self.crab.yaw = trig::atan2_turns(dx, dz);
            // Step CRAB_SPEED along the (dx, dz) direction via an integer-normalized
            // move (deterministic isqrt), not float trig. Within one step's distance,
            // snap to the target so it doesn't overshoot/jitter around it. All of this
            // is i128: players can flee the slower crab forever, so positions are
            // unbounded, and an i64·i64 here (the squared distance OR the normalization
            // multiply) could overflow on a marathon round — which PANICS in debug but
            // WRAPS in release, i.e. desyncs two peers on different build profiles.
            let dist = isqrt_i128(dist2_i128(dx, dz));
            if dist <= CRAB_SPEED as i128 {
                self.crab.pos = t;
            } else if dist > 0 {
                self.crab.pos.x += (dx as i128 * CRAB_SPEED as i128 / dist) as i64;
                self.crab.pos.z += (dz as i128 * CRAB_SPEED as i128 / dist) as i64;
            }
        }

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
                && inputs
                    .get(id)
                    .copied()
                    .unwrap_or_default()
                    .pressed(buttons::ACTION)
            {
                p.status = PlayerStatus::Extracted;
            }
        }

        // 5) Settle the round outcome (first decisive condition wins; extraction
        //    beats a wipe on the same tick — a rescue at the buzzer counts).
        self.outcome = self.settle_outcome();
    }

    /// Ground position of the living player nearest the crab, or `None` if every player
    /// is downed/extracted. The solo NN-crab bridge reads this to aim the rapier crab at
    /// its prey (the same target the integer pursuit picks), keeping the showcase crab
    /// hunting the same player the round logic would.
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
    /// in sync; any divergence flips it (see [`crate::net::lockstep`] desync check).
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
        // config/crab_external are deliberately not hashed (see field docs); bound to `_`
        // so the destructure stays exhaustive without folding them.
        let Sim {
            tick,
            players,
            planes,
            crab,
            extraction,
            outcome,
            rng,
            restart_held,
            config: _,
            crab_external: _,
        } = self;

        let mut h = Fnv::new();
        h.write(&tick.to_le_bytes());
        for (id, player) in players.iter() {
            let Player { pos, yaw, status } = player;
            h.write(&[id.0]);
            h.write_pos(*pos);
            h.write(&yaw.to_le_bytes());
            h.write(&[status_tag(*status)]);
        }
        // Planes (PlayerId order): every evolving field. An empty plane map writes
        // nothing, so the foot-only sim's hash is unchanged.
        for (id, plane) in planes.iter() {
            let Plane {
                pos,
                vel,
                heading,
                pitch,
            } = plane;
            h.write(&[id.0]);
            h.write_pos3(*pos);
            h.write_pos3(*vel);
            h.write(&heading.to_le_bytes());
            h.write(&pitch.to_le_bytes());
        }
        let Crab { pos, yaw } = crab;
        h.write_pos(*pos);
        h.write(&yaw.to_le_bytes());
        let ExtractionPoint { pos } = extraction;
        h.write_pos(*pos);
        h.write(&[outcome_tag(*outcome)]);
        // The restart edge-latch gates whether next tick's RESTART press fires, so a
        // divergence in it would desync the restart.
        h.write(&[u8::from(*restart_held)]);
        // Hash the RNG stream position so a desync in random draws is caught even before
        // it manifests in an entity. Cloning and drawing one block reflects the
        // generator's position without disturbing the real stream.
        h.write(&rand::Rng::r#gen::<u64>(&mut rng.clone()).to_le_bytes());
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

    /// Read-only view of all planes in `PlayerId` order — for rendering each pilot's
    /// gray box. Empty in the foot-only game. Never drives sim logic (read-only, like
    /// [`Sim::players`]).
    pub fn planes(&self) -> impl Iterator<Item = (PlayerId, Plane)> + '_ {
        self.planes.iter().map(|(&id, &p)| (id, p))
    }

    /// The plane piloted by `id`, if that player is flying (for the FP camera / avatar).
    pub fn plane(&self, id: PlayerId) -> Option<Plane> {
        self.planes.get(&id).copied()
    }

    /// The giant crab (for rendering the threat).
    pub fn crab(&self) -> Crab {
        self.crab
    }

    /// Test-only read of the external-control flag (see [`crab_external`](Sim::crab_external)),
    /// so the MP byte-identical-invariant tests (rl#63) can assert the networked path leaves
    /// the crab under the deterministic integer pursuit. Not part of the production API: the
    /// flag is set-once internal state, never read back outside the determinism guard tests.
    #[cfg(test)]
    pub(crate) fn crab_is_external(&self) -> bool {
        self.crab_external
    }

    /// The extraction point (for rendering the objective marker).
    pub fn extraction(&self) -> ExtractionPoint {
        self.extraction
    }

    /// The round outcome (for ending the round / showing win-lose UI).
    pub fn outcome(&self) -> Outcome {
        self.outcome
    }
}

/// Stable 1-byte tag for a [`PlayerStatus`] in the state hash. Explicit (not
/// `as u8` on the enum) so reordering or inserting variants can't silently shift the
/// hashed value out from under a peer on a different build.
fn status_tag(s: PlayerStatus) -> u8 {
    match s {
        PlayerStatus::Alive => 0,
        PlayerStatus::Downed => 1,
        PlayerStatus::Extracted => 2,
    }
}

/// Stable 1-byte tag for an [`Outcome`] in the state hash (see [`status_tag`]).
fn outcome_tag(o: Outcome) -> u8 {
    match o {
        Outcome::Ongoing => 0,
        Outcome::Extracted => 1,
        Outcome::Wiped => 2,
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

/// Advance one plane one tick from its pilot input — crude arcade flight, pure integer
/// fixed-point so two peers evolve it bit-identically.
///
/// Control map (reuses the existing [`Input`] axes, no wire change):
/// - [`Input::move_forward`] → throttle: thrust along the heading-and-pitch facing.
/// - [`Input::look_yaw`] → yaw rate: turn the heading.
/// - [`Input::move_strafe`] → pitch rate: nose up (climb) / down (dive).
///
/// Order per tick: turn (yaw + pitch, clamped) → accelerate (thrust along the new
/// facing, minus gravity) → damp → integrate position. Forces are integers; the only
/// "trig" is the integer CORDIC [`trig`] table, never a float.
fn step_plane(plane: &mut Plane, inp: Input) {
    // 1) Orientation: integrate bounded yaw + pitch rates from the stick axes. Same
    //    descale pattern as the player's look (axis units of AXIS_SCALE → turn units).
    let dyaw = (inp.look_yaw as i64 * PLANE_YAW_RATE as i64 / Input::AXIS_SCALE as i64) as i32;
    plane.heading = trig::wrap_turns(plane.heading + dyaw);
    let dpitch =
        (inp.move_strafe as i64 * PLANE_PITCH_RATE as i64 / Input::AXIS_SCALE as i64) as i32;
    // Clamp pitch to ±MAX (no wrap — pitch is a bounded tilt, not a full circle; with no
    // roll, letting it pass vertical would leave the plane inverted with no recovery).
    plane.pitch = (plane.pitch + dpitch).clamp(-PLANE_MAX_PITCH, PLANE_MAX_PITCH);

    // 2) Facing unit vector (scale trig::ONE) from heading + pitch. Horizontal part is
    //    (sin,cos) of heading shrunk by cos(pitch); vertical is sin(pitch). At pitch 0
    //    this is the player's ground facing with vy=0; nose-up tilts thrust toward +Y.
    let (sh, ch) = trig::sin_cos(plane.heading); // heading sin/cos · ONE
    let (sp, cp) = trig::sin_cos(plane.pitch); // pitch sin/cos · ONE
    // fx,fz carry a trig::ONE² scale (two cosines/sines multiplied); fy carries ONE¹.
    // Bring fy up to ONE² too so all three axes share one scale before thrusting.
    let fx = sh * cp;
    let fz = ch * cp;
    let fy = sp * trig::ONE as i64;

    // 3) Thrust along the facing at the throttle, then gravity. Descale the facing's
    //    trig::ONE² and the throttle's AXIS_SCALE so the acceleration is in UNIT/tick².
    let throttle = inp.move_forward as i64; // units of AXIS_SCALE, sign = fwd/reverse
    let one2 = trig::ONE as i64 * trig::ONE as i64;
    let denom = one2 * Input::AXIS_SCALE as i64;
    plane.vel.x += fx * PLANE_THRUST * throttle / denom;
    plane.vel.y += fy * PLANE_THRUST * throttle / denom;
    plane.vel.z += fz * PLANE_THRUST * throttle / denom;
    plane.vel.y -= PLANE_GRAVITY;

    // 4) Damp (drag): bleed a fixed fraction of velocity so the integrator can't run
    //    away. Integer truncation toward zero is identical on every target.
    plane.vel.x = plane.vel.x * PLANE_DAMP_NUM / PLANE_DAMP_DEN;
    plane.vel.y = plane.vel.y * PLANE_DAMP_NUM / PLANE_DAMP_DEN;
    plane.vel.z = plane.vel.z * PLANE_DAMP_NUM / PLANE_DAMP_DEN;

    // 5) Integrate position. Don't sink through the ground: clamp Y≥0 and kill a
    //    downward velocity on contact, so a dive ends in a (crude) belly landing rather
    //    than falling forever below the world.
    plane.pos.x += plane.vel.x;
    plane.pos.y += plane.vel.y;
    plane.pos.z += plane.vel.z;
    if plane.pos.y < 0 {
        plane.pos.y = 0;
        if plane.vel.y < 0 {
            plane.vel.y = 0;
        }
    }
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

/// Integer fixed-point trigonometry for the deterministic sim.
///
/// Angles are **turn units**: a full circle is [`TURN`] (a power of two, so wrapping
/// is a cheap mask and there's no modulo bias). Sine/cosine come from a quarter-wave
/// lookup table in fixed-point (scale [`ONE`]).
///
/// **The table is built with INTEGER-ONLY math (CORDIC), never `f64::sin`.** A
/// hardware/compiler float `sin` is not guaranteed bit-identical across platforms
/// (libm/FMA/rounding differ), so building the table from one would reintroduce a
/// cross-machine desync — and a unit test, being same-binary, could never catch it.
/// Integer CORDIC is a fixed sequence of adds and shifts: identical on every target, so
/// two peers on different builds/OSes get the same table. No `f32`/`f64` appears
/// anywhere in this module (client float helpers live in [`super::trig_client`]).
pub mod trig {
    use std::sync::OnceLock;

    /// Angle units in a full turn. Power of two so [`wrap_turns`] is a mask.
    pub const TURN: i32 = 1 << 16;
    /// Fixed-point scale of sine/cosine outputs: `ONE` represents 1.0.
    pub const ONE: i32 = 1 << 14;

    /// Quarter-turn entry count for the sine table (`TURN/4`), the resolution of the
    /// angle→sine lookup.
    const QUARTER: usize = (TURN / 4) as usize;

    /// Quarter-wave sine table, `round(sin(k/TURN · 2π) · ONE)` for `k` in
    /// `0..=QUARTER`, built once by integer CORDIC. Quarter-wave + symmetry
    /// reconstructs the full circle, so the table is small and the four quadrants share
    /// one set of values (no quadrant can drift from another). Inclusive of `QUARTER`
    /// so `cos` (a quarter-turn phase shift) can index the endpoint.
    fn sine_table() -> &'static [i32; QUARTER + 1] {
        static T: OnceLock<[i32; QUARTER + 1]> = OnceLock::new();
        T.get_or_init(|| {
            let mut t = [0i32; QUARTER + 1];
            for (k, slot) in t.iter_mut().enumerate() {
                *slot = cordic_sin(k as i32);
            }
            t
        })
    }

    /// Internal CORDIC precision: fractional bits carried during the rotation so the
    /// rounded table entries are exact. [`PREC`] bits below the turn-unit/[`ONE`]
    /// scales.
    const PREC: u32 = 28;
    /// CORDIC iteration count — enough that the residual angle is far below one table
    /// step, giving `round(sin·ONE)` exact to the last bit (asserted in tests).
    const ITERS: usize = 28;

    /// `round(atan(2^-i) / (2π) · TURN · 2^PREC)` for `i` in `0..ITERS`: the fixed
    /// CORDIC micro-rotation angles, in `2^PREC`-scaled turn units. Integer constants
    /// so the rotation uses no float at runtime. They are NOT trusted on faith —
    /// [`tests::cordic_constants_match_reference`] re-derives them from `f64` and
    /// asserts equality, so a typo fails the build rather than silently skewing every
    /// angle.
    const ATAN_TURNS: [i64; ITERS] = [
        2199023255552,
        1298159229407,
        685911684590,
        348179481054,
        174765388006,
        87467890064,
        43744617923,
        21873643805,
        10936988781,
        5468515251,
        2734260233,
        1367130443,
        683565262,
        341782636,
        170891319,
        85445659,
        42722830,
        21361415,
        10680707,
        5340354,
        2670177,
        1335088,
        667544,
        333772,
        166886,
        83443,
        41722,
        20861,
    ];
    /// CORDIC start X = `round(ONE / K · 2^PREC)`, where `K = Π sqrt(1 + 2^-2i)` is the
    /// CORDIC gain. Pre-dividing by the gain means the rotation's final Y is directly
    /// `sin(a)·ONE·2^PREC` with no post-scale. Also re-derived and checked in tests.
    const X0: i64 = 2670726652173;

    /// Fixed-point CORDIC sine of a first-quadrant turn-unit angle `a` in
    /// `[0, QUARTER]`, returning `round(sin(a) · ONE)`. Pure integer: rotate the vector
    /// `(ONE/K, 0)` by `a` through the fixed [`ATAN_TURNS`] micro-rotations; the final
    /// Y is `sin(a)·ONE`. Only adds, shifts, and compile-time constants, so it is
    /// bit-identical on every target — the property the table needs and a float `sin`
    /// can't guarantee.
    fn cordic_sin(a: i32) -> i32 {
        let target = (a as i64) << PREC; // angle to rotate to, in 2^PREC turn units
        let mut x: i64 = X0;
        let mut y: i64 = 0;
        let mut ang: i64 = 0;
        for (i, &step) in ATAN_TURNS.iter().enumerate() {
            let i = i as u32;
            // Rotate toward `target`: each step turns by ±atan(2^-i), driving `ang`
            // toward the target while `(x, y)` tracks `(cos, sin)·ONE/K·2^PREC`.
            if ang < target {
                let (nx, ny) = (x - (y >> i), y + (x >> i));
                x = nx;
                y = ny;
                ang += step;
            } else {
                let (nx, ny) = (x + (y >> i), y - (x >> i));
                x = nx;
                y = ny;
                ang -= step;
            }
        }
        let half = 1i64 << (PREC - 1); // round-to-nearest when descaling
        ((y + half) >> PREC) as i32
    }

    /// Wrap a turn-unit angle into `[0, TURN)`. Mask, since `TURN` is a power of two —
    /// branch-free and identical for negative inputs (two's-complement `& (TURN-1)`).
    pub fn wrap_turns(a: i32) -> i32 {
        a & (TURN - 1)
    }

    /// Sine of a turn-unit angle, fixed-point (scale [`ONE`]). Reconstructs the full
    /// circle from the quarter-wave table by quadrant.
    pub fn sin(a: i32) -> i32 {
        let a = wrap_turns(a) as usize;
        let q = QUARTER;
        match a / q {
            0 => sine_table()[a],
            1 => sine_table()[2 * q - a],  // sin(π−x)=sin x
            2 => -sine_table()[a - 2 * q], // sin(π+x)=−sin x
            _ => -sine_table()[4 * q - a], // sin(2π−x)=−sin x
        }
    }

    /// Cosine of a turn-unit angle, fixed-point: `cos a = sin(a + quarter turn)`.
    pub fn cos(a: i32) -> i32 {
        sin(a + TURN / 4)
    }

    /// `(sin, cos)` together (one wrap, two table hits) for movement-by-facing.
    pub fn sin_cos(a: i32) -> (i64, i64) {
        (sin(a) as i64, cos(a) as i64)
    }

    /// Turn-unit angle of the vector `(x, z)`, measured from +Z toward +X — the
    /// world's convention (a player at yaw 0 faces +Z). Pure integer (octant
    /// decomposition + a binary search over the sine table), so it's deterministic
    /// across machines where a float `atan2` need not be. Used to point the crab at
    /// its prey; exact enough for a facing.
    pub fn atan2_turns(x: i64, z: i64) -> i32 {
        if x == 0 && z == 0 {
            return 0;
        }
        // Angle from +Z toward +X means |x| is the opposite leg and |z| the adjacent,
        // so the first-quadrant magnitude is atan(|x|/|z|). Reduce to the lower octant
        // by taking theta = atan(min/max) ∈ [0, TURN/8], then reflect to TURN/4−theta
        // when |x| is the LARGER leg (the angle is past the 45° diagonal).
        let ax = x.unsigned_abs() as u128;
        let az = z.unsigned_abs() as u128;
        let (hi, lo) = if ax >= az { (ax, az) } else { (az, ax) };
        // theta = atan(lo/hi) in [0, TURN/8]: binary-search the largest k whose
        // tan(k) ≤ lo/hi, compared as sin(k)*hi ≤ cos(k)*lo (cross-multiplied — no
        // division). sin/cos are monotone over the quarter, so the search is exact.
        let mut lo_k = 0i32;
        let mut hi_k = TURN / 8;
        while lo_k < hi_k {
            let mid = (lo_k + hi_k + 1) / 2;
            let s = sin(mid) as i128 * hi as i128;
            let c = cos(mid) as i128 * lo as i128;
            if s <= c {
                lo_k = mid;
            } else {
                hi_k = mid - 1;
            }
        }
        let theta = lo_k; // atan(min/max) in [0, TURN/8]
        // Magnitude of the angle off the +Z axis, in [0, TURN/4]. atan(|x|/|z|): equals
        // theta when |x| ≤ |z|, else its complement.
        let in_q = if ax <= az { theta } else { TURN / 4 - theta };
        // Fold into the correct quadrant by the signs of (x, z). Convention: angle
        // measured from +Z toward +X.
        let folded = match (x >= 0, z >= 0) {
            (true, true) => in_q,             // +x +z: [0, TURN/4)
            (true, false) => TURN / 2 - in_q, // +x −z: (TURN/4, TURN/2]
            (false, false) => TURN / 2 + in_q,
            (false, true) => TURN - in_q,
        };
        wrap_turns(folded)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cordic_constants_match_reference() {
            // The pinned CORDIC constants are large magic integers; re-derive them from
            // f64 and assert equality, so a typo fails the build instead of silently
            // skewing every table entry. Float appears ONLY in this test — the runtime
            // table build (`cordic_sin`) is integer-only and identical on every target.
            for (i, &pinned) in ATAN_TURNS.iter().enumerate() {
                let want = ((2.0f64.powi(-(i as i32))).atan() / std::f64::consts::TAU
                    * TURN as f64
                    * (1u64 << PREC) as f64)
                    .round() as i64;
                assert_eq!(pinned, want, "ATAN_TURNS[{i}] wrong");
            }
            let k: f64 = (0..ITERS as i32)
                .map(|i| (1.0 + 2.0f64.powi(-2 * i)).sqrt())
                .product();
            let want_x0 = (ONE as f64 / k * (1u64 << PREC) as f64).round() as i64;
            assert_eq!(X0, want_x0, "X0 wrong");
        }
    }
}

/// Client-side float helpers for turning the sim's integer angles into the radians a
/// renderer wants. **Deliberately OUTSIDE [`trig`]:** [`trig`] is the float-free
/// surface the sim itself calls, so keeping every `f32`/`f64` here means "no float in
/// the sim" is enforced by where things live, not by a comment a render edit can
/// ignore. Only client/render code (never [`Sim::step`]) calls this.
pub mod trig_client {
    use super::trig::{TURN, wrap_turns};

    /// Convert a turn-unit angle (e.g. [`super::Player::yaw`]) to radians for the FP
    /// camera. Returns `f32`, so by construction it can't be used in the sim.
    pub fn turns_to_radians(a: i32) -> f32 {
        (wrap_turns(a) as f32) / (TURN as f32) * std::f32::consts::TAU
    }
}

/// FNV-1a 64-bit. A tiny, fixed, allocation-free hash whose result is identical on
/// every machine — the property `state_hash` needs and `std::hash::DefaultHasher`
/// does not guarantee.
struct Fnv(u64);

impl Fnv {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
    /// Fold a [`Pos`] (both coordinates) — one call per position so a hashed entity
    /// can't accidentally fold X but forget Z. Destructured exhaustively so a new
    /// coordinate forces a compile error here (the rl#70 guard, extended to `Pos`).
    fn write_pos(&mut self, p: Pos) {
        let Pos { x, z } = p;
        self.write(&x.to_le_bytes());
        self.write(&z.to_le_bytes());
    }
    /// Fold a [`Pos3`] (all three coordinates) — one call per 3D position so a hashed
    /// flying entity can't fold X/Z but forget the altitude Y. Exhaustively destructured
    /// for the same reason as [`Fnv::write_pos`].
    fn write_pos3(&mut self, p: Pos3) {
        let Pos3 { x, y, z } = p;
        self.write(&x.to_le_bytes());
        self.write(&y.to_le_bytes());
        self.write(&z.to_le_bytes());
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn players(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
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
        // One player standing still; the crab should close in and eventually down it,
        // ending the round as a wipe. The crab holds still through the startup grace
        // (covered by its own test), so step PAST the grace before checking pursuit.
        let mut sim = Sim::new(0, &players(1));
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
        for _ in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral);
        }
        let crab_armed = sim.crab().pos();
        let prey = sim.player(PlayerId(0)).unwrap().pos();
        let d_start = dist2(crab_armed, prey);
        sim.step(&neutral);
        let d_next = dist2(sim.crab().pos(), sim.player(PlayerId(0)).unwrap().pos());
        assert!(d_next < d_start, "crab must close distance once armed");
        // Run until the round resolves (bounded).
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
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
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
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
        for _ in 0..10 {
            sim.step(&neutral);
        }
        assert_eq!(
            snapshot(&sim),
            frozen,
            "a decided round must freeze the world"
        );
    }

    #[test]
    fn no_pilots_is_byte_identical_to_plain_new() {
        // The flag-OFF invariant: `new_with_pilots(.., &[])` is the foot-only game,
        // hashed identically to `new` (empty plane map writes no bytes). If this ever
        // breaks, every existing replay/desync hash shifts and the no-plane game is no
        // longer the unchanged game.
        let players = players(3);
        let a = Sim::new(0xABCD, &players);
        let b = Sim::new_with_pilots(0xABCD, &players, &[]);
        assert_eq!(a.state_hash(), b.state_hash());
        assert_eq!(a.planes().count(), 0, "no pilots ⇒ no planes");
        assert_eq!(a.players().count(), 3);
    }

    #[test]
    fn pilot_replaces_foot_player_and_spawns_airborne() {
        // A pilot is in `planes`, NOT `players` (the two are disjoint), and starts above
        // the ground (positive altitude), flying from tick 0.
        let ids = players(2);
        let sim = Sim::new_with_pilots(7, &ids, &[PlayerId(1)]);
        assert!(sim.player(PlayerId(1)).is_none(), "a pilot is not on foot");
        assert!(sim.player(PlayerId(0)).is_some(), "non-pilot stays on foot");
        let plane = sim.plane(PlayerId(1)).expect("player 1 is a pilot");
        assert!(plane.pos().y > 0, "plane spawns airborne");
        assert_eq!(plane.heading(), 0, "spawns facing +Z like a player");
    }

    #[test]
    fn throttle_with_nose_up_gains_altitude() {
        // Full throttle while pitched up must climb: nose-up tilts thrust toward +Y
        // enough to beat gravity, so altitude rises over a few seconds. This is the
        // controllability check — the plane responds to its pitch+throttle inputs.
        let mut sim = Sim::new_with_pilots(0, &players(1), &players(1));
        let y0 = sim.plane(PlayerId(0)).unwrap().pos().y;
        let mut inp = BTreeMap::new();
        // Pitch up (strafe +) at full throttle (forward +).
        inp.insert(PlayerId(0), Input::new(1.0, 1.0, 0.0, 0));
        for _ in 0..120 {
            sim.step(&inp);
        }
        let y1 = sim.plane(PlayerId(0)).unwrap().pos().y;
        assert!(y1 > y0, "nose-up full throttle should climb, {y0} -> {y1}");
    }

    #[test]
    fn idle_plane_sinks_and_lands_without_falling_through() {
        // No throttle: gravity wins, the plane descends, and the ground clamp catches it
        // at Y=0 (never negative) — a crude belly landing, not an infinite fall.
        let mut sim = Sim::new_with_pilots(0, &players(1), &players(1));
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
        for _ in 0..400 {
            sim.step(&neutral);
        }
        let p = sim.plane(PlayerId(0)).unwrap();
        assert_eq!(
            p.pos().y,
            0,
            "an idle plane sinks to the ground and stops there"
        );
    }

    #[test]
    fn hash_covers_plane_state() {
        // Two pilot sims identical but for one tick of plane input must hash
        // differently — proves the plane's state is folded into the hash (not just
        // players/crab). One throttles, the other idles: their planes' velocity/pos
        // diverge, so the hashes must.
        let ids = players(1);
        let mut a = Sim::new_with_pilots(0, &ids, &ids);
        let mut b = Sim::new_with_pilots(0, &ids, &ids);
        assert_eq!(a.state_hash(), b.state_hash(), "same start ⇒ same hash");
        let mut throttle = BTreeMap::new();
        throttle.insert(PlayerId(0), Input::new(0.0, 1.0, 0.0, 0));
        a.step(&throttle);
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
        b.step(&neutral);
        assert_ne!(
            a.state_hash(),
            b.state_hash(),
            "plane motion must change the hash"
        );
    }

    #[test]
    fn mixed_foot_and_pilot_sims_stay_in_lockstep() {
        // A round with BOTH a foot player and a pilot must evolve identically on two
        // independent sims fed the same inputs — the determinism contract has to hold
        // across the two entity kinds at once, not just foot-only or pilot-only. Step
        // each tick with a movement input for the foot player AND a flight input for the
        // pilot in the same map, asserting the hashes agree every tick, then prove BOTH
        // actually moved (so a no-op match can't pass vacuously).
        let mut a = Sim::new_with_pilots(0xF007, &players(2), &[PlayerId(1)]);
        let mut b = Sim::new_with_pilots(0xF007, &players(2), &[PlayerId(1)]);
        assert!(a.player(PlayerId(0)).is_some(), "player 0 is on foot");
        assert!(a.plane(PlayerId(1)).is_some(), "player 1 is a pilot");
        let foot_spawn = a.player(PlayerId(0)).unwrap().pos();
        let plane_spawn = a.plane(PlayerId(1)).unwrap().pos();
        for _ in 0..300 {
            let mut inputs = BTreeMap::new();
            // Foot player walks forward; pilot throttles up with some yaw + nose-up pitch.
            inputs.insert(PlayerId(0), Input::new(0.0, 1.0, 0.0, 0));
            inputs.insert(PlayerId(1), Input::new(0.5, 1.0, 0.3, 0));
            a.step(&inputs);
            b.step(&inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "mixed foot+pilot sims must stay bit-identical tick-for-tick"
            );
        }
        assert_ne!(
            a.player(PlayerId(0)).unwrap().pos(),
            foot_spawn,
            "the foot player must have moved from spawn"
        );
        assert_ne!(
            a.plane(PlayerId(1)).unwrap().pos(),
            plane_spawn,
            "the pilot's plane must have moved from spawn"
        );
    }

    #[test]
    fn hash_changes_when_state_changes() {
        let mut sim = Sim::new(0, &players(2));
        let h0 = sim.state_hash();
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), Input::from_axes(1.0, 1.0));
        sim.step(&inputs);
        assert_ne!(sim.state_hash(), h0);
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
        // Runtime half of the rl#70 guard. The COMPILE-TIME half lives in `state_hash`
        // itself: its exhaustive `let Sim { .. }` (and per-entity) destructures stop that
        // function compiling until a newly-added field is folded in or bound to `_`. This
        // test proves the other direction — that each field the destructure *names* is
        // actually written into the hash, not bound and silently dropped. Together: a new
        // field can be neither forgotten (compile error) nor faked (a binding that hashes
        // nothing fails here).
        //
        // A sim with BOTH a foot player and a pilot, so the plane fields are real (an empty
        // plane map hashes to nothing — see `state_hash`) and every field has a value to
        // perturb. `hash_after` clones the base, mutates one field, and returns the hash; a
        // hashed field must change it, the two excluded fields must not.
        let base = Sim::new_with_pilots(7, &players(2), &[PlayerId(1)]);
        let h0 = base.state_hash();
        let hash_after = |mutate: &dyn Fn(&mut Sim)| {
            let mut s = base.clone();
            mutate(&mut s);
            s.state_hash()
        };
        let foot = PlayerId(0);
        let pilot = PlayerId(1);

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

        assert_ne!(
            hash_after(&|s| s.planes.get_mut(&pilot).unwrap().pos.x += 1),
            h0,
            "plane pos.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.planes.get_mut(&pilot).unwrap().pos.y += 1),
            h0,
            "plane pos.y must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.planes.get_mut(&pilot).unwrap().pos.z += 1),
            h0,
            "plane pos.z must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.planes.get_mut(&pilot).unwrap().vel.x += 1),
            h0,
            "plane vel.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.planes.get_mut(&pilot).unwrap().vel.y += 1),
            h0,
            "plane vel.y must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.planes.get_mut(&pilot).unwrap().vel.z += 1),
            h0,
            "plane vel.z must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.planes.get_mut(&pilot).unwrap().heading += 1),
            h0,
            "plane heading must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.planes.get_mut(&pilot).unwrap().pitch += 1),
            h0,
            "plane pitch must be hashed"
        );

        // The crab_external FLAG is excluded (below), but the crab POSE it drives is hashed
        // exactly like the internal-pursuit pose — that's what lets the solo NN crab's
        // quantized pose stay desync-safe.
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
        // Advancing the generator (without touching anything else) must flip the hash, so a
        // desync in random draws is caught before it surfaces in an entity.
        assert_ne!(
            hash_after(&|s| {
                let _: u64 = rand::Rng::r#gen(s.rng());
            }),
            h0,
            "rng stream position must be hashed"
        );

        // Excluded fields: perturbing must NOT flip the hash. Both have field docs
        // explaining why they are outside the cross-peer hash; mutating them here is purely
        // to witness the exclusion (nothing in production mutates `config`).
        assert_eq!(
            hash_after(&|s| s.config.seed ^= 0xdead_beef),
            h0,
            "config is deliberately not hashed (see Sim::config)"
        );
        assert_eq!(
            hash_after(&|s| s.crab_external = !s.crab_external),
            h0,
            "crab_external is deliberately not hashed (see Sim::crab_external)"
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
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
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
        // Press RESTART and the WORLD rebuilds to its tick-0 state: tick back to 0,
        // players Alive at spawn, crab at its spawn, outcome Ongoing — matching a fresh
        // round. (The hash also folds the restart edge-latch, which is legitimately set
        // here because R is held and clear in a never-restarted sim — so we compare the
        // observable round state, not the raw hash. The full-hash agreement BETWEEN
        // peers who both apply the restart is the lockstep test's job.) Run a few ticks
        // and move first so there's real state to discard.
        let mut sim = Sim::new(0xBEEF, &players(2));
        let fresh = Sim::new(0xBEEF, &players(2));
        let mut fwd = BTreeMap::new();
        fwd.insert(PlayerId(0), Input::new(0.3, 1.0, 0.5, 0));
        fwd.insert(PlayerId(1), Input::new(-0.2, 1.0, 0.0, 0));
        for _ in 0..50 {
            sim.step(&fwd);
        }
        let round = |s: &Sim| {
            (
                s.tick(),
                s.players().collect::<Vec<_>>(),
                s.crab(),
                s.extraction(),
                s.outcome(),
            )
        };
        assert_ne!(
            round(&sim),
            round(&fresh),
            "the round should have diverged from spawn before restart"
        );
        // Press R (only player 0 holds it — one peer's press restarts everyone).
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        restart.insert(PlayerId(1), Input::default());
        sim.step(&restart);
        assert_eq!(sim.tick(), 0, "restart resets the tick counter");
        assert_eq!(sim.outcome(), Outcome::Ongoing);
        assert_eq!(
            round(&sim),
            round(&fresh),
            "a restarted round's world matches a fresh one"
        );
    }

    #[test]
    fn restart_works_after_the_round_is_decided() {
        // The point of restart is to play again after a win/loss — so it must fire even
        // once the world is frozen (outcome != Ongoing). Wipe the player, confirm the
        // freeze, then RESTART and confirm a live round is back.
        let mut sim = Sim::new(0, &players(1));
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            sim.step(&neutral);
        }
        assert_eq!(sim.outcome(), Outcome::Wiped, "round should have ended");
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        sim.step(&restart);
        assert_eq!(sim.outcome(), Outcome::Ongoing, "restart revives the round");
        assert_eq!(sim.tick(), 0);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Alive,
            "the player is alive again after a post-loss restart"
        );
    }

    #[test]
    fn restart_is_edge_triggered_not_level() {
        // Holding RESTART across ticks must restart ONCE (on the press), then let the
        // round advance — otherwise a held key would pin the sim at tick 0 forever.
        let mut sim = Sim::new(0, &players(1));
        let mut held = BTreeMap::new();
        held.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        sim.step(&held); // tick 1 → restart fires, back to tick 0
        assert_eq!(sim.tick(), 0, "first press restarts");
        // Keep holding: the latch is set, so subsequent ticks advance normally.
        sim.step(&held);
        sim.step(&held);
        assert_eq!(sim.tick(), 2, "a held key doesn't re-restart every tick");
        // Release, then press again: that fresh edge restarts once more.
        sim.step(&BTreeMap::new()); // release (tick 3)
        sim.step(&held); // fresh press → restart
        assert_eq!(sim.tick(), 0, "a new press after release restarts again");
    }

    #[test]
    fn restart_keeps_two_peers_in_lockstep() {
        // The determinism contract under restart: two independent sims fed the identical
        // input stream — including a mid-round RESTART press — stay bit-identical
        // tick-for-tick. This is what makes restart safe over the wire (every peer
        // applies the same RESTART on the same tick).
        let mut a = Sim::new(0x5151, &players(2));
        let mut b = Sim::new(0x5151, &players(2));
        for t in 0..120u64 {
            let mut inputs = BTreeMap::new();
            // Both players move; player 1 presses R once at tick 40.
            let restart_bit = if t == 40 { buttons::RESTART } else { 0 };
            inputs.insert(PlayerId(0), Input::new(0.4, 1.0, 0.2, 0));
            inputs.insert(PlayerId(1), Input::new(-0.3, 1.0, -0.1, restart_bit));
            a.step(&inputs);
            b.step(&inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "peers must stay bit-identical across a restart (tick {t})"
            );
        }
        // And the restart actually happened: by tick 41 the sim is freshly at a low tick,
        // not 41 ticks deep.
        assert!(
            a.tick() < 120,
            "the mid-run restart rewound the tick counter"
        );
    }

    fn dist2(a: Pos, b: Pos) -> i128 {
        dist2_i128(a.x - b.x, a.z - b.z)
    }
}
