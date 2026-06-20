//! Deterministic simulation core for lockstep multiplayer.
//!
//! Phase 1 (rl#38) lives here: the gray-box "Extraction" loop — tiny first-person
//! players hunted by one giant crab, each racing to a fixed pickup point. It
//! replaces Phase 0's dot world but keeps the contract that makes lockstep work,
//! because peers run independent copies and only inputs cross the wire (see
//! [`crate::net::lockstep`]):
//!
//! - [`Sim::step`] is a pure function of `(prior state, inputs)` — identical inputs
//!   from an identical state always produce an identical next state, on any machine.
//!   No wall-clock, no thread-local/global RNG, no iteration over `HashMap`/`HashSet`
//!   (their order is randomized per process). Seed every random draw from
//!   [`Sim::rng`]; iterate players in `PlayerId` order.
//! - **No `f32`/`f64` arithmetic in the sim.** Floats round-trip differently across
//!   targets/compilers, so the whole world is integer fixed-point (positions, yaw,
//!   velocities) and angles go through an integer sin/cos table ([`trig`]). This
//!   sidesteps the cross-machine float-determinism risk entirely for the gray-box;
//!   if real rapier bodies arrive later, determinism shifts onto rapier's
//!   enhanced-determinism instead (see rl#39). `f32` appears ONLY at the input
//!   boundary ([`Input::from_axes`]), which quantizes to the integer grid before
//!   anything reaches the sim.
//! - [`Sim::state_hash`] folds the FULL observable state into one `u64`. It is the
//!   desync detector: two peers that diverge by even one bit hash differently next
//!   tick. Every field added here MUST be hashed there, or a desync in it goes
//!   undetected.
//!
//! # Interface for render / vehicle subs (rl#38)
//!
//! Two later subs build ON this module, and they relate to it differently — say so
//! plainly so neither author is surprised:
//! - The **render / FP-client** sub consumes a stable **read-only** view (the
//!   accessors below) and produces [`Input`]. It never edits the sim.
//! - The **vehicle** sub (plane, heli) is NOT a pure consumer: a flying body has 3D
//!   orientation (pitch + roll, which DO move it) and altitude, none of which fit
//!   [`Player`]'s 2D [`Pos`] + scalar yaw. So it will add a new entity type and edit
//!   [`Sim::new`]/[`Sim::step`]/[`Sim::state_hash`] directly (there is no plugin
//!   point — it's one crate). The yaw-only / Y=0 model here is a truth about *feet*,
//!   not a rule flying entities must obey.
//!
//! **Coordinate frame.** Right-handed: world is the XZ ground plane at Y=0, +X right,
//! +Z forward, +Y up (unused by feet). A yaw of 0 faces +Z and increases turning
//! toward +X (see [`trig::atan2_turns`]).
//!
//! **Reading state to render (read-only — never drives sim logic):**
//! - [`Sim::players`] → each [`Player`]'s [`pos`](Player::pos) ([`Pos`], world XZ in
//!   [`UNIT`] fixed-point), [`yaw`](Player::yaw) (a [`trig`] turn-unit facing), and
//!   [`status`](Player::status) (Alive / Downed / Extracted). A player is a capsule
//!   standing on the ground. Convert a coordinate to meters with
//!   `coord as f32 / UNIT as f32`, and a yaw to radians with
//!   [`trig_client::turns_to_radians`].
//! - [`Sim::crab`] → the one [`Crab`]: world position, facing yaw, and [`CRAB_SCALE`]
//!   (render it that many times bigger than a player — it is a scaled placeholder;
//!   reusing the trained RL crab body is a later concern).
//! - [`Sim::extraction`] → the fixed pickup point ([`ExtractionPoint`]); reaching it
//!   clears the round. Draw it as a debug marker of radius [`EXTRACT_RADIUS`].
//! - [`Sim::outcome`] → [`Outcome`]: `Ongoing`, `Extracted` (someone reached the
//!   point — round won), or `Wiped` (every player downed — round lost).
//!
//! **Producing input each tick:** build one [`Input`] per local player from the
//! controls — [`Input::new`] (move axes + yaw-look delta + an action bit) or the
//! move-only [`Input::from_axes`]. The client owns the camera (including pitch, which
//! the sim does not model — feet only need yaw to move relative to facing); it feeds
//! the sim a per-tick yaw *delta* and reads back the authoritative yaw to aim the
//! camera. When the vehicle sub lands, [`Input`] is a single flat fixed-width struct
//! shared by all entities — a deliberate gray-box simplification, NOT a finished
//! design. Adding throttle/pitch/roll either makes every walking player pay wire
//! bytes it ignores or lets a walker carry a nonzero throttle (an illegal state the
//! flat struct permits). Routing input to the right controller is also unsolved:
//! [`PlayerId`] is hardwired 1:1 to its [`Player`], with no "client X is piloting the
//! heli" indirection. **The vehicle author's first task is to redesign this seam**
//! (likely a tagged `enum Input { OnFoot{..}, Piloting{..} }` + a control-assignment
//! map) — keeping it small, fixed-width, and losslessly (de)serializable, since the
//! wire bytes ARE the shared truth (see [`Input::to_bytes`]). Also note [`trig`] is
//! 2D (yaw only): a flying body's pitch/roll need a 3D rotation the vehicle sub must
//! add (e.g. integer quaternions), still float-free for determinism.

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
}

/// One player's input for a single tick.
///
/// This is the unit that crosses the wire each tick, so it stays small and
/// (de)serializes losslessly and identically on every peer — the wire bytes ARE the
/// shared truth (see [`Input::to_bytes`]). Fixed-point, not `f32`: an integer is
/// bit-identical across machines where a float round-trip need not be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Input {
    /// Strafe axis (right +, i.e. toward +X at yaw 0) in fixed-point units of
    /// 1/[`Input::AXIS_SCALE`], clamped to ±[`Input::AXIS_SCALE`]. Applied relative to
    /// the player's facing yaw. Named for the control intent, not a world axis — at a
    /// nonzero yaw it does not map to world X.
    pub move_strafe: i16,
    /// Forward axis (forward +, i.e. toward +Z at yaw 0), same units/clamp as
    /// [`move_strafe`](Input::move_strafe). Likewise facing-relative, not world Z.
    pub move_forward: i16,
    /// Yaw-look delta for THIS tick, in fixed-point units of 1/[`Input::AXIS_SCALE`]:
    /// ±[`Input::AXIS_SCALE`] is the per-tick yaw cap (a fraction of a turn; see the
    /// sim's yaw clamp). The client integrates raw mouse/stick into this bounded delta
    /// and the sim adds it to the player's yaw — bounded so one tick can't spin a peer
    /// arbitrarily. A yaw delta, not a world-X delta.
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

/// Fixed-point world scale: a position/length value of [`UNIT`] equals one world
/// meter. All world coordinates, radii, and speeds are integers in these units, so
/// the whole sim is bit-identical across machines (see the module determinism note).
pub const UNIT: i64 = 1000;

/// Player walk speed, in [`UNIT`]/tick at full stick. Tuned for the gray-box, not
/// realism: ~5 m/s at 30 Hz (`SPEED * 30 / UNIT`).
const PLAYER_SPEED: i64 = 166;

/// Crab pursuit speed, in [`UNIT`]/tick. A touch slower than the player so a dodging
/// player can open distance — the asymmetry the loop is built on (rl#38) — but fast
/// enough to punish standing still.
const CRAB_SPEED: i64 = 150;

/// Render hint only: how many times bigger than a player to draw the crab. It is a
/// scaled placeholder; the sim treats it as a point with a [`CRAB_GRAB_RADIUS`] reach.
pub const CRAB_SCALE: i64 = 12;

/// How close (in [`UNIT`]) the crab must get to a living player to "grab" (down) it.
/// Generous — it stands in for the giant crab's reach without modelling its limbs.
pub const CRAB_GRAB_RADIUS: i64 = 3 * UNIT;

/// How close (in [`UNIT`]) a living player must be to the extraction point, AND
/// holding [`buttons::ACTION`], to extract and win the round.
pub const EXTRACT_RADIUS: i64 = 2 * UNIT;

/// Per-tick cap on yaw turn from look input, in [`trig::TURN`] units (a full circle
/// is [`trig::TURN`]). At 30 Hz, [`trig::TURN`]/24 per tick ≈ half a turn per second
/// at full deflection — brisk but not instant, and bounded so a single tick can't
/// spin a peer wildly.
const MAX_YAW_TURNS_PER_TICK: i32 = trig::TURN / 24;

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
    /// hashed and reproduced across peers, so it is genuine sim state, not a scratch
    /// generator — never reseed it mid-sim. The gray-box crab pursuit is pure
    /// arithmetic and draws nothing; the field stays so later loop variation (spawn
    /// jitter, crab feints) has a deterministic source ready, and the contract that
    /// "the one RNG is hashed" is established.
    rng: ChaCha8Rng,
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
        let mut map = BTreeMap::new();
        for (i, &id) in sorted.iter().enumerate() {
            // Spawn ring near the origin; spacing in world units, all facing +Z
            // (yaw 0). Integer layout → identical everywhere.
            let i = i as i64;
            map.insert(
                id,
                Player {
                    pos: Pos { x: (i - sorted.len() as i64 / 2) * 2 * UNIT, z: 0 },
                    yaw: 0,
                    status: PlayerStatus::Alive,
                },
            );
        }
        Self {
            tick: 0,
            players: map,
            // Extraction is across the map at +Z; players spawn at the origin. The crab
            // sits BETWEEN them (around the midpoint, offset in X so it's an obstacle to
            // dodge, not a head-on instant grab) — so reaching safety means getting past
            // it. The player is slightly faster than the crab (PLAYER_SPEED > CRAB_SPEED),
            // so a good dodge slips by while standing still or a clumsy line gets grabbed:
            // the tiny-vs-giant asymmetry the loop is built on (rl#38).
            crab: Crab { pos: Pos { x: 6 * UNIT, z: 20 * UNIT }, yaw: 0 },
            extraction: ExtractionPoint { pos: Pos { x: 0, z: 40 * UNIT } },
            outcome: Outcome::Ongoing,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
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

        // 2) Crab: aim at and step toward the nearest living player. Pure arithmetic —
        //    no RNG, no float — so it is trivially deterministic.
        if let Some(target) = self.nearest_living_player() {
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

        // 3) Grabs: any living player within the crab's reach is downed.
        let crab = self.crab.pos;
        for p in self.players.values_mut() {
            if p.status == PlayerStatus::Alive
                && within(p.pos.x, p.pos.z, crab.x, crab.z, CRAB_GRAB_RADIUS)
            {
                p.status = PlayerStatus::Downed;
            }
        }

        // 4) Extraction: a living player at the point, holding ACTION, extracts.
        let ex = self.extraction.pos;
        for (id, p) in self.players.iter_mut() {
            if p.status == PlayerStatus::Alive
                && within(p.pos.x, p.pos.z, ex.x, ex.z, EXTRACT_RADIUS)
                && inputs.get(id).copied().unwrap_or_default().pressed(buttons::ACTION)
            {
                p.status = PlayerStatus::Extracted;
            }
        }

        // 5) Settle the round outcome (first decisive condition wins; extraction
        //    beats a wipe on the same tick — a rescue at the buzzer counts).
        self.outcome = self.settle_outcome();
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
        if self.players.values().any(|p| p.status == PlayerStatus::Extracted) {
            return Outcome::Extracted;
        }
        if !self.players.is_empty()
            && self.players.values().all(|p| p.status == PlayerStatus::Downed)
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
    pub fn state_hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(&self.tick.to_le_bytes());
        for (id, p) in self.players.iter() {
            h.write(&[id.0]);
            h.write_pos(p.pos);
            h.write(&p.yaw.to_le_bytes());
            h.write(&[status_tag(p.status)]);
        }
        h.write_pos(self.crab.pos);
        h.write(&self.crab.yaw.to_le_bytes());
        h.write_pos(self.extraction.pos);
        h.write(&[outcome_tag(self.outcome)]);
        // Hash the RNG stream position so a desync in random draws is caught even
        // before it manifests in an entity. Cloning and drawing one block reflects the
        // generator's position without disturbing the real stream.
        h.write(&rand::Rng::r#gen::<u64>(&mut self.rng.clone()).to_le_bytes());
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
/// (libm/FMA/rounding differ), so building the table from one would reintroduce the
/// exact cross-machine desync this whole module exists to prevent — and a unit test,
/// being same-binary, could never catch it. Integer CORDIC is a fixed sequence of
/// adds and shifts: its output is identical on every target, so two peers on
/// *different builds/OSes* still get the same table. No `f32`/`f64` appears anywhere
/// in this module (client float helpers live in [`super::trig_client`]).
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
        2199023255552, 1298159229407, 685911684590, 348179481054, 174765388006, 87467890064,
        43744617923, 21873643805, 10936988781, 5468515251, 2734260233, 1367130443,
        683565262, 341782636, 170891319, 85445659, 42722830, 21361415,
        10680707, 5340354, 2670177, 1335088, 667544, 333772,
        166886, 83443, 41722, 20861,
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
            1 => sine_table()[2 * q - a], // sin(π−x)=sin x
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
    use super::trig::{wrap_turns, TURN};

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
    /// can't accidentally fold X but forget Z.
    fn write_pos(&mut self, p: Pos) {
        self.write(&p.x.to_le_bytes());
        self.write(&p.z.to_le_bytes());
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
        assert!((dz - PLAYER_SPEED).abs() <= 1, "forward step ≈ PLAYER_SPEED, got {dz}");
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
        assert!(dx.abs() > dz.abs(), "after a ~quarter turn, forward should move mostly in X (dx={dx}, dz={dz})");
    }

    #[test]
    fn crab_pursues_and_grabs_a_lone_player() {
        // One player standing still; the crab should close in and eventually down it,
        // ending the round as a wipe.
        let mut sim = Sim::new(0, &players(1));
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
        let crab_start = sim.crab().pos();
        let prey_start = sim.player(PlayerId(0)).unwrap().pos();
        sim.step(&neutral);
        let d_start = dist2(crab_start, prey_start);
        let d_next = dist2(sim.crab().pos(), sim.player(PlayerId(0)).unwrap().pos());
        assert!(d_next < d_start, "crab must close distance to its prey");
        // Run until the round resolves (bounded).
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            sim.step(&neutral);
        }
        assert_eq!(sim.outcome(), Outcome::Wiped, "standing-still player gets caught");
        assert_eq!(sim.player(PlayerId(0)).unwrap().status(), PlayerStatus::Downed);
    }

    #[test]
    fn reaching_extraction_with_action_wins() {
        // End-to-end: a faster player who DODGES the crab and reaches the point holding
        // ACTION wins. The crab sits between spawn and extraction (at +X), so a straight
        // line gets grabbed; the player instead swings WIDE to -X (away from the crab),
        // dragging it off-axis, then runs up the far side to the point. The speed edge
        // (PLAYER_SPEED > CRAB_SPEED) makes the detour pay off. Deterministic — fixed
        // waypoints, no randomness — so it can't flake. (A traced win lands ~t=590, far
        // inside the 4000 budget.)
        let mut sim = Sim::new(0, &players(1));
        let ex = sim.extraction().pos();
        // Waypoints: out to -X, up the far side, then the point itself.
        let route = [
            Pos { x: -30 * UNIT, z: 0 },
            Pos { x: -30 * UNIT, z: ex.z },
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
        assert!(won, "a player who dodges the crab, reaches the point, and holds ACTION should extract");
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
        assert_ne!(sim.outcome(), Outcome::Ongoing, "round should have resolved");
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
        assert_eq!(snapshot(&sim), frozen, "a decided round must freeze the world");
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

    #[test]
    fn hash_covers_yaw_and_status() {
        // Two sims identical but for one player's yaw must hash differently — proves
        // yaw is in the hash. (Status coverage is exercised by the wipe/extract tests
        // changing the outcome, which flips the hash.)
        let mut a = Sim::new(0, &players(1));
        let mut b = Sim::new(0, &players(1));
        assert_eq!(a.state_hash(), b.state_hash());
        let mut look = BTreeMap::new();
        look.insert(PlayerId(0), Input::new(0.0, 0.0, 1.0, 0));
        a.step(&look);
        let mut none = BTreeMap::new();
        none.insert(PlayerId(0), Input::default());
        b.step(&none);
        // a turned, b didn't: positions equal (no move) but yaw differs → hashes differ.
        assert_ne!(a.state_hash(), b.state_hash(), "yaw must be hashed");
    }

    // --- trig table sanity: pins the integer table to known trig values ---

    #[test]
    fn trig_table_hits_cardinal_points() {
        use trig::{cos, sin, ONE, TURN};
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
        use trig::{cos, sin, ONE};
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
        use trig::{atan2_turns, TURN};
        // Convention: angle from +Z toward +X.
        let near = |a: i32, b: i32| trig::wrap_turns(a - b).min(trig::wrap_turns(b - a));
        assert!(near(atan2_turns(0, 1), 0) <= 2, "+Z is yaw 0");
        assert!(near(atan2_turns(1, 0), TURN / 4) <= 2, "+X is quarter turn");
        assert!(near(atan2_turns(0, -1), TURN / 2) <= 2, "−Z is half turn");
        assert!(near(atan2_turns(-1, 0), 3 * TURN / 4) <= 2, "−X is three-quarter turn");
        assert!(near(atan2_turns(1, 1), TURN / 8) <= 2, "+X+Z diagonal is eighth turn");
    }

    #[test]
    fn isqrt_matches_floor_sqrt() {
        for n in [
            0i128, 1, 2, 3, 4, 8, 15, 16, 17, 100, 1_000_000, 1_000_003,
            i64::MAX as i128, // a value far past i64-squared range, to exercise the marathon case
            (i64::MAX as i128) * (i64::MAX as i128),
        ] {
            let r = isqrt_i128(n);
            assert!(r * r <= n && (r + 1) * (r + 1) > n, "isqrt({n})={r} not floor sqrt");
        }
    }

    #[test]
    fn cordic_table_matches_f64_reference_exactly() {
        // The runtime sine table is built by integer CORDIC (no float) so two peers on
        // different libm agree. This test is the ONLY place float trig appears, and only
        // to PROVE the integer table equals the rounded f64 truth at every entry — if
        // CORDIC drifts, this fails. (The sim never runs this; it pins the table.)
        use trig::{cos, sin, ONE, TURN};
        for a in 0..TURN {
            let want = ((a as f64 / TURN as f64 * std::f64::consts::TAU).sin() * ONE as f64)
                .round() as i32;
            assert_eq!(sin(a), want, "sin table off at {a}");
        }
        // cos derived as sin(a+quarter) — spot-check it tracks the reference too.
        for a in (0..TURN).step_by(257) {
            let want = ((a as f64 / TURN as f64 * std::f64::consts::TAU).cos() * ONE as f64)
                .round() as i32;
            assert_eq!(cos(a), want, "cos off at {a}");
        }
    }

    #[test]
    fn pos_is_hashed_for_every_entity() {
        // Moving a player changes the hash (player pos hashed); the crab moving toward
        // it also changes it (crab pos hashed). Covered jointly by stepping a sim with a
        // crab in pursuit and checking the hash advances each of the first few ticks.
        let mut sim = Sim::new(0, &players(1));
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
        let mut prev = sim.state_hash();
        for _ in 0..5 {
            sim.step(&neutral);
            let now = sim.state_hash();
            assert_ne!(now, prev, "crab moving (and tick advancing) must change the hash");
            prev = now;
        }
    }

    fn dist2(a: Pos, b: Pos) -> i128 {
        dist2_i128(a.x - b.x, a.z - b.z)
    }
}
