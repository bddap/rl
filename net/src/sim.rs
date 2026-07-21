use std::collections::BTreeMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crab_world::fnv::Fnv;

use crate::snapshot::CoreSnapshot;
use crate::wire::pos_bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlayerId(pub u8);

pub mod buttons {
    pub const ACTION: u8 = 1 << 0;
    pub const RESTART: u8 = 1 << 1;
    /// The foot buttons that stay live while piloting ([`super::Input::pilot_masked`]):
    /// RESTART works in every context (rl#261 — every cockpit legend in
    /// `net::controls` promises "Restart round"), while ACTION must NOT — the ship's
    /// brake shares Space/South with Extract, so a braking pilot over the pad would
    /// extract.
    pub const PILOT_MASK: u8 = RESTART;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Input {
    pub move_strafe: i16,
    pub move_forward: i16,
    pub look_yaw: i16,
    pub buttons: u8,
}

impl Input {
    pub const AXIS_SCALE: i16 = 1000;

    pub fn new(strafe: f32, forward: f32, look_yaw: f32, buttons: u8) -> Self {
        let q = |v: f32| (v.clamp(-1.0, 1.0) * Self::AXIS_SCALE as f32).round() as i16;
        Self {
            move_strafe: q(strafe),
            move_forward: q(forward),
            look_yaw: q(look_yaw),
            buttons,
        }
    }

    pub fn from_axes(strafe: f32, forward: f32) -> Self {
        Self::new(strafe, forward, 0.0, 0)
    }

    pub fn pressed(self, bit: u8) -> bool {
        self.buttons & bit != 0
    }

    /// The input the server substitutes for a tick where this player's stream is STARVED
    /// (transit lag): keep the held-state move axes, zero the rest — `look_yaw` is a per-tick
    /// DELTA (re-applying it would keep the avatar turning) and a re-fired button tap would
    /// double an extract/restart. See [`crate::server`]'s hold semantics.
    pub fn hold(self) -> Self {
        Self {
            move_strafe: self.move_strafe,
            move_forward: self.move_forward,
            look_yaw: 0,
            buttons: 0,
        }
    }

    /// The input a PILOTING player contributes to a tick — the craft flies on its
    /// `PilotIntent`, so the walk axes are zeroed and the buttons masked to
    /// [`buttons::PILOT_MASK`]. Applied by the client (`LocalControl::sim_input`) and
    /// RE-applied by the server at assembly, which doesn't trust the client to have
    /// (rl#191); one implementation so the two can't drift. A starved hold can't
    /// re-fire the surviving button ([`Input::hold`] zeroes buttons).
    pub fn pilot_masked(self) -> Self {
        Self {
            buttons: self.buttons & buttons::PILOT_MASK,
            ..Self::default()
        }
    }
}
// The wire codec (`to_bytes`/`from_bytes`/`WIRE_LEN`) is a transport concern and lives in
// `crate::wire`, with the rest of the byte-exact layouts.

pub const TICK_HZ: u64 = 30;

pub const TICK_DT: f64 = 1.0 / TICK_HZ as f64;

/// Fixed-point grid units per world meter. The world runs at the RIG's scale (rl#256):
/// one frame shared by the sim, the crab's physics arena, and the render — no
/// arena↔sim conversion, no render shrink. A player is ~0.051 m, so the grid is sized
/// for resolution at that stature: 10 µm, ~3× finer than the 1 mm cell the pre-rl#256
/// 35×-larger sim frame used.
pub const UNIT: i64 = 100_000;

/// On-foot pace, player heights per second — scale-free like
/// [`crab_world::eval::CRAB_CHARGE_SPEED_HEIGHTS_PER_S`]. The expression is the
/// pre-rl#256 tuning verbatim: 166 grid/tick at the old 1 mm grid, 30 Hz, 1.8 m
/// stature.
const PLAYER_SPEED_HEIGHTS_PER_S: f32 = 166.0 * 30.0 / (1000.0 * 1.8);

pub(crate) const PLAYER_SPEED: i64 =
    (PLAYER_SPEED_HEIGHTS_PER_S * PLAYER_HEIGHT * UNIT as f32 / TICK_HZ as f32 + 0.5) as i64;

/// Test-driver step per tick, folded from the ONE measured speed
/// ([`CRAB_CHARGE_SPEED_PER_S`], rl#257) so pursuit/grace tests exercise her honest
/// pace — a second bare speed here would drift from reality (and did: 130).
#[cfg(test)]
const CRAB_SPEED: i64 = CRAB_CHARGE_SPEED_PER_S / TICK_HZ as i64;

/// The claw tests' yardstick, 5/9 of a player height (≈0.028 m) — every claw-test
/// length is a multiple, so the geometry keeps one legible scale. Anchored on the
/// player, NOT on [`CLAW_DOWN_BUFFER`]: that's the tune-on-playtest feel knob, and
/// retuning it must not silently rescale this geometry (e.g. drop the "overhead"
/// claw under the player's height span).
#[cfg(test)]
const CLAW_M: i64 = PLAYER_HEIGHT_FP * 5 / 9;

/// A claw capsule at body height. `dx` slides it sideways off the player so the
/// near-miss cases stay one call.
#[cfg(test)]
fn claw_at(p: Pos, dx: i64, y: i64) -> ClawPose {
    ClawPose {
        a: Pos {
            x: p.x + dx - 2 * CLAW_M,
            z: p.z,
        },
        b: Pos {
            x: p.x + dx + 2 * CLAW_M,
            z: p.z,
        },
        a_y: y,
        b_y: y,
        radius: CLAW_M / 2,
    }
}

/// Advance every crab toward its nearest living player and return this tick's
/// [`Externals::crabs`] poses for the caller to feed `step()` — claws riding the
/// carapace points once past grace (downs are claw contact only, rl#236, so the
/// pursuit driver must bring claws, not just a pose, for a catch to land).
#[cfg(test)]
pub(crate) fn drive_crab_toward_prey(sim: &Sim) -> Vec<CrabPose> {
    let armed = sim.tick() >= sim.round_start + STARTUP_GRACE_TICKS;
    (0..sim.crabs().len())
        .map(|idx| {
            let mut pos = sim.crabs()[idx].pos();
            let mut yaw = sim.crabs()[idx].yaw();
            if armed && let Some(target) = sim.nearest_living_player_pos(idx) {
                let dx = target.x - pos.x;
                let dz = target.z - pos.z;
                yaw = trig::atan2_turns(dx, dz);
                let dist = isqrt_i128(dist2_i128(dx, dz));
                if dist <= CRAB_SPEED as i128 {
                    pos = target;
                } else if dist > 0 {
                    pos.x += (dx as i128 * CRAB_SPEED as i128 / dist) as i64;
                    pos.z += (dz as i128 * CRAB_SPEED as i128 / dist) as i64;
                }
            }
            CrabPose {
                pos,
                yaw,
                claws: if armed {
                    vec![claw_at(pos, 0, CLAW_M)]
                } else {
                    Vec::new()
                },
            }
        })
        .collect()
}

pub const CRAB_SCALE: i64 = 12;

/// Sally's nominal stature in world meters. The world's absolute scale IS the rig's
/// (rl#256), so this is a pinned copy of the rigs' measured natural height — fallback
/// 0.61146 m, mesh-fitted 0.61150 m; `spawn_clearance_matches_crab_body` holds it within 1%
/// of every rig she can wear. Human-scale constants derive from it via the sizing rule.
pub const CRAB_STATURE: f32 = 0.6115;

/// Player height in world meters — the crab sizing rule (she stands [`CRAB_SCALE`]
/// player-heights tall), inverted: the crab's stature is the world's ground truth and
/// players derive from HER (rl#256). Render capsules derive from this one constant.
pub const PLAYER_HEIGHT: f32 = CRAB_STATURE / CRAB_SCALE as f32;

/// `h` player heights on the fixed-point grid, rounded to the nearest unit — THE way
/// human-scale gameplay constants are written: the docs already reason in player
/// heights, and a ratio like `6.0 / 1.8` keeps a carried-over pre-rl#256 tuning's
/// provenance in the code instead of a rotting meter footnote.
const fn player_heights(h: f32) -> i64 {
    (h * PLAYER_HEIGHT * UNIT as f32 + 0.5) as i64
}

/// "Very close" slack around a claw collider (rl#249): the sim player is a point, so this
/// covers their body radius plus the near-miss feel margin — and absorbs one tick of claw
/// sweep (the capsule is sampled per tick, not swept). At her scale a graze this wide
/// (~1.1 player heights) reads as a hit. Pure feel parameter; tune on playtest.
pub const CLAW_DOWN_BUFFER: i64 = player_heights(2.0 / 1.8);

/// The player's height span for the claw check, on the fixed-point grid: a claw passing
/// clear overhead must not down anyone.
const PLAYER_HEIGHT_FP: i64 = player_heights(1.0);

const STARTUP_GRACE_TICKS: u64 = 30;

/// Sally's sustained full-charge ground speed, grid units per second — MEASURED, not
/// commanded: her speed is whatever the trained gait strides. Folded from the ONE
/// scale-free pinned pace ([`crab_world::eval::CRAB_CHARGE_SPEED_HEIGHTS_PER_S`]) ×
/// her stature ([`CRAB_STATURE`]) so the chase eval re-measures it every run and flags
/// drift after a retrain (rl#266) instead of the old bare 8_500 rotting silently.
const CRAB_CHARGE_SPEED_PER_S: i64 =
    (crab_world::eval::CRAB_CHARGE_SPEED_HEIGHTS_PER_S * CRAB_STATURE * UNIT as f32) as i64;

/// How long a fresh spawn is guaranteed before Sally at full charge can be on them —
/// the spawn-safety feel knob (rl#257). The old bare 19 m was tuned pre-rl#254 at
/// 1/35th her real speed and played as "seconds to live"; five seconds of charge is
/// time to orient and run. Tune on playtest.
const SPAWN_GRACE_SECS: i64 = 5;

/// Spawn clearance from the crab's sim pos, round-start and joiners alike (rl#247):
/// [`SPAWN_GRACE_SECS`] of her full charge, not a bare meter count (rl#257). Far
/// outside her carapace footprint (corner reach ~0.51 m), so no spawn lands inside
/// her claw shell; `spawn_clearance_matches_crab_body` cross-checks that floor
/// against every rig.
const MIN_CRAB_SPAWN_DISTANCE: i64 = CRAB_CHARGE_SPEED_PER_S * SPAWN_GRACE_SECS;

/// Spacing between player spawn slots along the z=0 spawn line.
const SPAWN_SLOT_PITCH: i64 = player_heights(2.0 / 1.8);

/// Reach-the-objective radius.
pub const EXTRACT_RADIUS: i64 = player_heights(2.0 / 1.8);

pub(crate) const MAX_YAW_TURNS_PER_TICK: i32 = trig::TURN / 24;

/// A fresh entropy seed for a real GCR launch (rl#305): the whole run layout
/// derives deterministically from the match seed, so the game entrypoints draw this
/// per launch (and log it — the seed alone reproduces the run's spawn), while the
/// probes/screenshot tools keep passing their pinned seed for byte-stable A/Bs.
pub fn random_match_seed() -> u64 {
    rand::random()
}

/// Rotate the layout-local vector `(lx, lz)` by `rot` [`trig`] turns — the
/// [`Sim::advance_player`] yaw convention: local +z maps to the facing of
/// `yaw == rot`, so a frame's `rot` doubles as its spawn yaw.
fn rotate(lx: i64, lz: i64, rot: i32) -> (i64, i64) {
    let (sin, cos) = trig::sin_cos(rot);
    (
        (lx * cos + lz * sin) / trig::ONE as i64,
        (lz * cos - lx * sin) / trig::ONE as i64,
    )
}

/// One line per layout draw, on whichever sim drew it — the HOST's line is the
/// authoritative record (a remote client's sim is a placeholder the snapshots
/// supersede, and it never restarts, so `restart` lines are host-only by
/// construction). `seed` + the restart count reproduce the run exactly (rl#305).
fn log_spawn(seed: u64, frame: &SpawnFrame, extraction: ExtractionPoint, why: &str) {
    let (ox, oz) = frame.origin.to_meters();
    let (ex, ez) = extraction.pos.to_meters();
    let deg = frame.rot as f32 / trig::TURN as f32 * 360.0;
    tracing::info!(
        "rl#305 {why} spawn: seed={seed:#x} origin=({ox:.1}, {oz:.1}) m \
         heading={deg:.0}° extraction=({ex:.1}, {ez:.1}) m"
    );
}

/// The run's spawn placement (rl#305): every layout point is authored in a local
/// frame — spawn line along local x through the origin, objective up local +z — and
/// mapped through one translate+rotate drawn per run from the match seed, so each
/// run opens at a fresh locale and heading on the tile instead of the fixed origin
/// Sally had memorized. Mid-round joiners place through the SAME frame
/// ([`Sim::nearest_clear_join_slot`]), so the join line is the run's spawn line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpawnFrame {
    origin: Pos,
    /// Layout heading in [`trig`] turns — also every spawned player's yaw.
    rot: i32,
}

/// Worst-case layout-local radius, before Sally's push-out: the objective's local z,
/// plus [`MIN_CRAB_SPAWN_DISTANCE`] again for a crab pushed a full clearance beyond
/// the widest spawn line a u8 roster can form. Everything the frame places stays
/// within this of its origin — except a many-crab base stagger (~0.12 m per extra
/// crab), which the sampling clamp's own 256 m edge margin absorbs for any plausible
/// crab count.
const LAYOUT_LOCAL_RADIUS: i64 = 2 * MIN_CRAB_SPAWN_DISTANCE
    + player_heights(10.0)
    + (u8::MAX as i64 / 2 + 1) * SPAWN_SLOT_PITCH;

impl SpawnFrame {
    /// Where a frame origin may land: the terrain sampling interior training draws
    /// episode locales from ([`crab_world::training::targets::sample_clamp_half`] —
    /// tile half-span less the band-fit edge margin), pulled in by
    /// [`LAYOUT_LOCAL_RADIUS`], so the whole layout lands on ground the brain has
    /// seen and every placed point clears the tile edge.
    fn origin_bound() -> i64 {
        let clamp_m = crab_world::training::targets::sample_clamp_half(
            &crab_world::terrain::TerrainGrid::gcr(),
        );
        let bound = meters_to_grid(clamp_m) - LAYOUT_LOCAL_RADIUS;
        assert!(
            bound > 0,
            "GCR tile interior ({clamp_m} m half-span) cannot fit the spawn layout \
             (local radius {LAYOUT_LOCAL_RADIUS} grid units)"
        );
        bound
    }

    fn draw(rng: &mut ChaCha8Rng) -> Self {
        use rand::Rng;
        let bound = Self::origin_bound();
        Self {
            origin: Pos {
                x: rng.gen_range(-bound..=bound),
                z: rng.gen_range(-bound..=bound),
            },
            rot: rng.gen_range(0..trig::TURN),
        }
    }

    /// Layout-local → world: rotate by `rot`, then translate to the origin.
    fn place(&self, lx: i64, lz: i64) -> Pos {
        let (x, z) = rotate(lx, lz, self.rot);
        Pos {
            x: self.origin.x + x,
            z: self.origin.z + z,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerStatus {
    Alive,
    Downed,
    Extracted,
}

impl PlayerStatus {
    /// The vehicle-boarding ability gate — the ONE formula; the host's intent filter and the
    /// client's toggle both consult it. Downed may board, provisionally (playtest 1, rl#262:
    /// "for now") — flip the balance here. Extracted is out of the round.
    pub fn may_board(self) -> bool {
        matches!(self, PlayerStatus::Alive | PlayerStatus::Downed)
    }
}

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
            x: meters_to_grid(x_m),
            z: meters_to_grid(z_m),
        }
    }
}

/// Scalar leg of [`Pos::from_meters`], for the heights that ride beside a `Pos`
/// (a claw capsule's y — [`ClawPose`]).
pub fn meters_to_grid(m: f32) -> i64 {
    (m * UNIT as f32) as i64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player {
    pos: Pos,
    yaw: i32,
    status: PlayerStatus,
}

impl Player {
    pub(crate) fn from_parts(pos: Pos, yaw: i32, status: PlayerStatus) -> Self {
        Self { pos, yaw, status }
    }

    pub fn pos(self) -> Pos {
        self.pos
    }
    pub fn yaw(self) -> i32 {
        self.yaw
    }
    pub fn status(self) -> PlayerStatus {
        self.status
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crab {
    pos: Pos,
    yaw: i32,
}

/// One of Sally's claw colliders as of this tick, bridged into sim space — THE down
/// mechanism, alone (rl#236 owner call): standing under her carapace is deliberately
/// safe-and-fun, so no center/footprint disc downs anyone; only a pincer touch does.
/// The capsule is the pincer's
/// real physics capsule (rl#249 — no separate hitbox to drift), as an XZ segment with
/// per-end heights ABOVE THE LOCAL GROUND SURFACE and the capsule radius, all on the
/// fixed-point grid. Surface-relative y is what makes [`Self::downs`]'s player span
/// (`0..=PLAYER_HEIGHT_FP`, a walker standing ON the ground) hold on the baked terrain
/// tile exactly as on the flat grids (rl#281 stage 6). External per-tick
/// INPUT, not round state: the host's crab slot captures it fresh from the one rapier
/// world into each step's [`Externals`], clients never see it (they receive the
/// resulting [`PlayerStatus`] via snapshot), and nothing stores it to hash or snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClawPose {
    pub a: Pos,
    pub b: Pos,
    pub a_y: i64,
    pub b_y: i64,
    pub radius: i64,
}

impl ClawPose {
    /// Whether this claw touches (within [`CLAW_DOWN_BUFFER`]) a player standing at `p`:
    /// the player's vertical span must meet the capsule's reach-fattened height band, and
    /// their XZ point must lie within reach of the capsule's XZ segment.
    fn downs(&self, p: Pos) -> bool {
        let reach = self.radius + CLAW_DOWN_BUFFER;
        let (lo, hi) = (
            self.a_y.min(self.b_y) - reach,
            self.a_y.max(self.b_y) + reach,
        );
        if hi < 0 || lo > PLAYER_HEIGHT_FP {
            return false;
        }
        let (cx, cz) = closest_on_segment(self.a, self.b, p);
        within(p.x, p.z, cx, cz, reach)
    }
}

impl Crab {
    pub(crate) fn from_parts(pos: Pos, yaw: i32) -> Self {
        Self { pos, yaw }
    }

    pub fn pos(self) -> Pos {
        self.pos
    }
    pub fn yaw(self) -> i32 {
        self.yaw
    }
}

/// A piloting player's craft pose in sim space — see [`Externals::pilots`].
#[derive(Debug, Clone, Copy)]
pub struct PilotPose {
    pub pos: Pos,
    pub yaw: i32,
}

/// One crab's world pose + claw colliders as of this tick — the mandatory per-crab
/// entry of [`Externals::crabs`] (rl#298 stage 5): Sally is world content, so her sim
/// pose is READ from the host's one rapier world every tick, never integrated
/// sim-side.
#[derive(Debug, Clone)]
pub struct CrabPose {
    pub pos: Pos,
    pub yaw: i32,
    /// This crab's claw colliders — see [`ClawPose`]. Empty only pre-spawn/grace.
    pub claws: Vec<ClawPose>,
}

/// This tick's external inputs to [`Sim::step`], captured fresh from the host's one
/// rapier world every tick. Parameters rather than sim state (rl#294): nothing stores
/// them, so a stale capture outliving the tick that measured it is unrepresentable, and
/// their exclusion from `state_hash`/`core_snapshot` needs no destructure discipline.
#[derive(Debug, Clone, Copy)]
pub struct Externals<'a> {
    /// Every crab's world pose + claws, one entry per sim crab — MANDATORY (rl#298
    /// stage 5): there is no inert-crab escape, so a host that runs no crab world
    /// cannot serve a round. [`Sim::step`] adopts the poses, then runs the claw-touch
    /// down check over the pooled claws (the check doesn't care which crab owns one).
    pub crabs: &'a [CrabPose],
    /// Every piloting player's craft pose, in sim space (rl#258): while a
    /// player flies, its ONE position is the craft's — the walker rides the craft instead
    /// of standing as a husk at the boarding spot, so the crab hunts the craft's shadow
    /// and stepping out resumes on foot right there. Membership doubles as the down
    /// exemption: a pilot is inside a hull, so claws act on the craft's REAL
    /// collider (rapier), never the walker. Clients see the resulting player pos.
    pub pilots: &'a BTreeMap<PlayerId, PilotPose>,
}

impl<'a> Externals<'a> {
    /// Crab poses with nobody piloting.
    pub(crate) fn crabs_only(crabs: &'a [CrabPose]) -> Self {
        const NO_PILOTS: &BTreeMap<PlayerId, PilotPose> = &BTreeMap::new();
        Self {
            crabs,
            pilots: NO_PILOTS,
        }
    }
}

/// Every crab held at its current sim pose, clawless — the test feed for steps where
/// the crabs are scenery (setup ticks, walker tests).
#[cfg(test)]
pub(crate) fn hold_poses(sim: &Sim) -> Vec<CrabPose> {
    sim.crabs()
        .iter()
        .map(|c| CrabPose {
            pos: c.pos(),
            yaw: c.yaw(),
            claws: Vec::new(),
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractionPoint {
    pos: Pos,
}

impl ExtractionPoint {
    pub fn pos(self) -> Pos {
        self.pos
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ongoing,
    Extracted,
    Wiped,
}

#[derive(Debug, Clone)]
pub struct Sim {
    tick: u64,
    players: BTreeMap<PlayerId, Player>,
    crabs: Vec<Crab>,
    extraction: ExtractionPoint,
    outcome: Outcome,
    rng: ChaCha8Rng,
    restart_held: bool,
    round_start: u64,
    spawn_frame: SpawnFrame,
    config: RoundConfig,
}

#[derive(Debug, Clone)]
struct RoundConfig {
    seed: u64,
    players: Vec<PlayerId>,
    crabs: usize,
}

impl Sim {
    pub fn new(seed: u64, players: &[PlayerId]) -> Self {
        let mut sorted: Vec<PlayerId> = players.to_vec();
        sorted.sort();
        sorted.dedup();
        let config = RoundConfig {
            seed,
            players: sorted,
            crabs: 1,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (spawn_frame, players, crabs, extraction) = Self::spawn_state(&config, &mut rng);
        log_spawn(seed, &spawn_frame, extraction, "round");
        Self {
            tick: 0,
            players,
            crabs,
            extraction,
            outcome: Outcome::Ongoing,
            rng,
            restart_held: false,
            round_start: 0,
            spawn_frame,
            config,
        }
    }

    pub fn configure_crabs(&mut self, crabs: usize) {
        assert!(crabs >= 1, "a round runs at least one giant crab (rl#114)");
        assert_eq!(
            self.tick, 0,
            "configure_crabs is round SETUP — the crab count is fixed once the round steps"
        );
        self.config.crabs = crabs;
        let (frame, players, crabs, extraction) = Self::spawn_state(&self.config, &mut self.rng);
        log_spawn(self.config.seed, &frame, extraction, "setup");
        self.spawn_frame = frame;
        self.players = players;
        self.crabs = crabs;
        self.extraction = extraction;
    }

    /// Adopt this tick's crab world poses — [`Sim::step`]'s first act, and callable
    /// alone only where a pose must land without stepping (a snapshot round-trip test).
    fn adopt_crab_poses(&mut self, crabs: &[CrabPose]) {
        assert_eq!(
            crabs.len(),
            self.crabs.len(),
            "one world pose per sim crab (rl#298 stage 5) — the crab world and the sim \
             disagree on the crab count"
        );
        for (c, pose) in self.crabs.iter_mut().zip(crabs) {
            c.pos = pose.pos;
            c.yaw = pose.yaw;
        }
    }

    fn reset(&mut self) {
        // The rng is deliberately NOT reseeded here: each restart draws the NEXT
        // frame off the seed's stream (rl#305) — reseeding would replay the same
        // layout every restart. Restarts stay deterministic in (seed, restart count),
        // which the log line below captures for repro.
        let (frame, players, crabs, extraction) = Self::spawn_state(&self.config, &mut self.rng);
        log_spawn(self.config.seed, &frame, extraction, "restart");
        self.round_start = self.tick;
        self.spawn_frame = frame;
        self.players = players;
        self.crabs = crabs;
        self.extraction = extraction;
        self.outcome = Outcome::Ongoing;
    }

    pub fn spawn_joining_player(&mut self, pid: PlayerId) {
        if self.players.contains_key(&pid) {
            return;
        }
        self.config.players.push(pid);
        self.config.players.sort();
        self.config.players.dedup();
        let idx = self
            .config
            .players
            .iter()
            .position(|p| *p == pid)
            .unwrap_or(0) as i64;
        let n = self.config.players.len() as i64;
        let x = (idx - n / 2) * SPAWN_SLOT_PITCH;
        self.players.insert(
            pid,
            Player {
                pos: self.nearest_clear_join_slot(x),
                yaw: self.spawn_frame.rot,
                status: PlayerStatus::Alive,
            },
        );
    }

    /// Nearest spawn-line slot to layout-local `x` clear of every crab by
    /// [`MIN_CRAB_SPAWN_DISTANCE`] — the same clearance round-start
    /// [`Self::spawn_crab`] keeps toward players, so a mid-round joiner gets the
    /// round-start guarantee and is never Downed before its first input (rl#247).
    /// Walks slots outward, alternating east/west, staying on the run's spawn line
    /// (the frame's local z=0 line, rl#305; no per-join grace: that would need
    /// wire-format state, and re-arming `round_start` would grace everyone
    /// mid-fight).
    fn nearest_clear_join_slot(&self, x: i64) -> Pos {
        // A crab blocks a closed 2·MIN chord of the line: at most this many slots. The
        // scan offers 2·blocked·crabs + 1 candidates, so one is always clear.
        let blocked_per_crab = 2 * MIN_CRAB_SPAWN_DISTANCE / SPAWN_SLOT_PITCH + 1;
        (0..=blocked_per_crab * self.crabs.len() as i64)
            .flat_map(|d| [d, -d])
            .map(|d| self.spawn_frame.place(x + d * SPAWN_SLOT_PITCH, 0))
            .find(|p| {
                self.crabs
                    .iter()
                    .all(|c| !within(p.x, p.z, c.pos.x, c.pos.z, MIN_CRAB_SPAWN_DISTANCE))
            })
            .expect("the candidate count outnumbers the slots the crabs can block")
    }

    pub fn has_player(&self, pid: PlayerId) -> bool {
        self.players.contains_key(&pid)
    }

    pub fn despawn_departed_player(&mut self, pid: PlayerId) {
        self.players.remove(&pid);
        self.config.players.retain(|p| *p != pid);
    }

    fn spawn_state(
        cfg: &RoundConfig,
        rng: &mut ChaCha8Rng,
    ) -> (
        SpawnFrame,
        BTreeMap<PlayerId, Player>,
        Vec<Crab>,
        ExtractionPoint,
    ) {
        let frame = SpawnFrame::draw(rng);
        let mut map = BTreeMap::new();
        let n = cfg.players.len() as i64;
        for (i, &id) in cfg.players.iter().enumerate() {
            let x = (i as i64 - n / 2) * SPAWN_SLOT_PITCH;
            map.insert(
                id,
                Player {
                    pos: frame.place(x, 0),
                    // Facing local +z — the party opens looking at the objective,
                    // whatever heading the frame drew.
                    yaw: frame.rot,
                    status: PlayerStatus::Alive,
                },
            );
        }
        // The objective sits BEYOND the crab's spawn ring by more than her ~0.48 m
        // claw-contact shell ([`ClawPose`] reach off her corner-most pincer) — with
        // a margin barely past her ring the rl#257 clearance bump parked her ON the
        // objective, its disc inside her claw shell at round start. Sally's bearing
        // jitter (rl#305) swings her at most a quarter turn off local +z, which only
        // GROWS her distance to the objective, so the clearance survives the jitter.
        let extraction = ExtractionPoint {
            pos: frame.place(0, MIN_CRAB_SPAWN_DISTANCE + player_heights(10.0)),
        };
        let crabs = (0..cfg.crabs)
            .map(|i| Self::spawn_crab(&map, &frame, i, rng))
            .collect();
        (frame, map, crabs, extraction)
    }

    fn spawn_crab(
        players: &BTreeMap<PlayerId, Player>,
        frame: &SpawnFrame,
        idx: usize,
        rng: &mut ChaCha8Rng,
    ) -> Crab {
        // The base pos staggers crabs and seeds the push-out BEARING; the clearance
        // clamp below (binding whenever the base sits inside MIN — every near-field
        // crab since rl#257 grew MIN past this base) sets the actual distance, so
        // round-start and joiner safety share the one constant. The per-crab bearing
        // jitter (rl#305, up to a quarter turn either side of local +z) keeps a run
        // from always opening with Sally square between the party and the objective;
        // at ±quarter-turn her ring stays clear of the extraction disc (the comment
        // on [`Self::spawn_state`]'s extraction margin).
        let jitter = rand::Rng::gen_range(rng, -trig::TURN / 4..=trig::TURN / 4);
        let (bx, bz) = rotate(
            player_heights(6.0 / 1.8) + idx as i64 * player_heights(8.0 / 1.8),
            player_heights(20.0 / 1.8),
            jitter,
        );
        let mut pos = frame.place(bx, bz);
        // Iterated clamp: one push-out from the base-nearest player sufficed for the
        // fixed pre-rl#305 bearing, but a jittered bearing can land the pushed pos
        // inside MIN of a DIFFERENT player (the party line's far slot) — so re-clamp
        // from the recomputed nearest until it is clear; clear-of-nearest ⇒ clear of
        // everyone. Each pass corrects by at most the party's line span (≪ MIN), so
        // a couple of passes settle it; the assert keeps a non-terminating geometry
        // loud instead of shipping a spawn inside her charge ring.
        let mut settled = false;
        for _ in 0..16 {
            let Some(nearest) = players
                .values()
                .min_by_key(|p| dist2_i128(pos.x - p.pos.x, pos.z - p.pos.z))
            else {
                settled = true;
                break;
            };
            let dx = pos.x - nearest.pos.x;
            let dz = pos.z - nearest.pos.z;
            let d2 = dist2_i128(dx, dz);
            let min = MIN_CRAB_SPAWN_DISTANCE as i128;
            if d2 >= min * min {
                settled = true;
                break;
            }
            let dist = isqrt_i128(d2);
            let (ux, uz, len) = if dist > 0 {
                (dx as i128, dz as i128, dist)
            } else {
                (1, 0, 1)
            };
            // Round each component AWAY from zero: truncation can land a hair
            // inside `min` (unseen while the clamp almost never bound; rl#257's
            // larger MIN binds every round). `len` is itself floored, which only
            // over-scales — so outward rounding guarantees distance ≥ min.
            let scale = |c: i128| {
                let num = c * min;
                (num / len + (num % len).signum()) as i64
            };
            pos.x = nearest.pos.x + scale(ux);
            pos.z = nearest.pos.z + scale(uz);
        }
        assert!(
            settled,
            "crab spawn clamp failed to clear every player within 16 passes (rl#305)"
        );
        Crab { pos, yaw: 0 }
    }

    fn participant_ids(&self) -> impl Iterator<Item = PlayerId> + '_ {
        self.players.keys().copied()
    }

    fn require_complete_inputs(&self, inputs: &BTreeMap<PlayerId, Input>) {
        for id in self.participant_ids() {
            assert!(
                inputs.contains_key(&id),
                "tick input incomplete: no input for {id:?} (have {:?}); defaulting \
                 to neutral would desync peers — refusing",
                inputs.keys().collect::<Vec<_>>(),
            );
        }
    }

    pub fn step(&mut self, inputs: &BTreeMap<PlayerId, Input>, externals: Externals<'_>) -> bool {
        self.require_complete_inputs(inputs);
        self.adopt_crab_poses(externals.crabs);
        self.tick += 1;

        let restart_now = inputs.values().any(|i| i.pressed(buttons::RESTART));
        let restart_edge = restart_now && !self.restart_held;
        self.restart_held = restart_now;
        if restart_edge {
            self.reset();
            return true;
        }

        if self.outcome != Outcome::Ongoing {
            return false;
        }

        for (id, p) in self.players.iter_mut() {
            if p.status != PlayerStatus::Alive {
                continue;
            }
            let inp = inputs[id];
            Self::advance_player(p, inp);
        }

        // A piloting player IS its craft (rl#258): its walker rides the craft's sim-space
        // shadow — one position whichever form the entity wears — so hunting, extraction
        // range and stepping out all resume from where the craft actually is.
        for (id, pp) in externals.pilots {
            if let Some(p) = self.players.get_mut(id) {
                p.pos = pp.pos;
                p.yaw = pp.yaw;
            }
        }

        let armed = self.tick > self.round_start + STARTUP_GRACE_TICKS;

        if armed {
            // Claw contact is the ONE down mechanism (rl#236) — see [`ClawPose`]. Pilots
            // are exempt: inside a hull there is no walker to claw — the crab strikes the
            // craft's REAL collider in the physics world instead (rl#258). Pooled across
            // crabs: the down check doesn't care which crab owns a claw.
            for claw in externals.crabs.iter().flat_map(|c| c.claws.iter()) {
                for (id, p) in self.players.iter_mut() {
                    if p.status == PlayerStatus::Alive
                        && !externals.pilots.contains_key(id)
                        && claw.downs(p.pos)
                    {
                        p.status = PlayerStatus::Downed;
                    }
                }
            }
        }

        let ex = self.extraction.pos;
        for (id, p) in self.players.iter_mut() {
            if p.status == PlayerStatus::Alive
                && within(p.pos.x, p.pos.z, ex.x, ex.z, EXTRACT_RADIUS)
                && inputs[id].pressed(buttons::ACTION)
            {
                p.status = PlayerStatus::Extracted;
            }
        }

        self.outcome = self.settle_outcome();
        false
    }

    fn advance_player(p: &mut Player, inp: Input) {
        let dyaw =
            (inp.look_yaw as i64 * MAX_YAW_TURNS_PER_TICK as i64 / Input::AXIS_SCALE as i64) as i32;
        p.yaw = trig::wrap_turns(p.yaw + dyaw);

        let (sin, cos) = trig::sin_cos(p.yaw);
        let strafe = inp.move_strafe as i64;
        let forward = inp.move_forward as i64;
        let vx = sin * forward + cos * strafe;
        let vz = cos * forward - sin * strafe;
        let denom = Input::AXIS_SCALE as i64 * trig::ONE as i64;
        p.pos.x += vx * PLAYER_SPEED / denom;
        p.pos.z += vz * PLAYER_SPEED / denom;
    }

    // Live only via `ClientSim::reconcile_local_prediction` (render-only) outside
    // tests — dead render-off (rl#248).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
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

    pub fn nearest_living_player_pos(&self, crab: usize) -> Option<Pos> {
        self.nearest_living_player(self.crabs[crab].pos)
            .map(|p| p.pos)
    }

    fn nearest_living_player(&self, c: Pos) -> Option<Player> {
        let mut best: Option<(i128, Player)> = None;
        for p in self.players.values() {
            if p.status != PlayerStatus::Alive {
                continue;
            }
            let d2 = dist2_i128(p.pos.x - c.x, p.pos.z - c.z);
            if best.is_none_or(|(bd, _)| d2 < bd) {
                best = Some((d2, *p));
            }
        }
        best.map(|(_, p)| p)
    }

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

    pub fn state_hash(&self) -> u64 {
        let Sim {
            tick,
            players,
            crabs,
            extraction,
            outcome,
            rng,
            restart_held,
            round_start,
            spawn_frame,
            config: _,
        } = self;

        // The hash covers exactly the state a snapshot ships and a peer adopts
        // ([`Self::core_snapshot`]/[`Self::apply_core_snapshot`]) — the SHARED world.
        // Host-private machinery (rng stream, restart latch, grace anchor, spawn
        // frame) is deliberately excluded: an adopt-only client never replicates it,
        // and since rl#305 each peer draws its own seed, so hashing it would desync
        // the cross-peer diff (`game net --hash-log`, #133) on every tick from 0.
        // Divergence there still surfaces here one draw later, through the crabs/
        // extraction/players it produces.
        let _ = (rng, restart_held, round_start, spawn_frame);
        let mut h = Fnv::new();
        h.write(&tick.to_le_bytes());
        for (id, player) in players.iter() {
            let Player { pos, yaw, status } = player;
            h.write(&[id.0]);
            h.write(&pos_bytes(*pos));
            h.write(&yaw.to_le_bytes());
            h.write(&[status.tag()]);
        }
        h.write(&(crabs.len() as u32).to_le_bytes());
        for crab in crabs {
            let Crab { pos, yaw } = crab;
            h.write(&pos_bytes(*pos));
            h.write(&yaw.to_le_bytes());
        }
        let ExtractionPoint { pos } = extraction;
        h.write(&pos_bytes(*pos));
        h.write(&[outcome.tag()]);
        h.finish()
    }

    pub fn rng(&mut self) -> &mut ChaCha8Rng {
        &mut self.rng
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn players(&self) -> impl Iterator<Item = (PlayerId, Player)> + '_ {
        self.players.iter().map(|(&id, &p)| (id, p))
    }

    pub fn player(&self, id: PlayerId) -> Option<Player> {
        self.players.get(&id).copied()
    }

    pub fn crabs(&self) -> &[Crab] {
        &self.crabs
    }

    pub fn extraction(&self) -> ExtractionPoint {
        self.extraction
    }

    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    pub fn core_snapshot(&self) -> CoreSnapshot {
        let Sim {
            tick,
            players,
            crabs,
            extraction,
            outcome,
            rng: _,
            restart_held: _,
            round_start: _,
            spawn_frame: _,
            config,
        } = self;
        CoreSnapshot {
            tick: *tick,
            players: players.clone(),
            crabs: crabs.clone(),
            // Per-run state since rl#305 (the host draws the layout from its own
            // seed), so it rides the snapshot like every other host-owned fact — a
            // client's locally-derived extraction is a placeholder until this lands.
            extraction: extraction.pos(),
            outcome: *outcome,
            roster: config.players.clone(),
            // Input watermarks are SERVER coordination metadata, not sim state — the sim holds
            // none. [`crate::server::Server::step_next`] stamps them; the client's `ClientSim`
            // stashes + re-stamps them for its mirror re-emit.
            input_next: std::collections::BTreeMap::new(),
        }
    }

    pub fn apply_core_snapshot(&mut self, snapshot: CoreSnapshot) {
        let CoreSnapshot {
            tick,
            players,
            crabs,
            extraction,
            outcome,
            roster,
            // Coordination metadata, not sim state — the client's `ClientSim` stashes it
            // (prediction-window prune + mirror re-emit) before handing the snapshot here.
            input_next: _,
        } = snapshot;
        self.tick = tick;
        self.players = players;
        self.config.crabs = crabs.len();
        self.crabs = crabs;
        self.extraction = ExtractionPoint { pos: extraction };
        self.outcome = outcome;
        self.config.players = roster;
    }
}

fn dist2_i128(dx: i64, dz: i64) -> i128 {
    let dx = dx as i128;
    let dz = dz as i128;
    dx * dx + dz * dz
}

fn within(ax: i64, az: i64, bx: i64, bz: i64, r: i64) -> bool {
    dist2_i128(ax - bx, az - bz) <= (r as i128) * (r as i128)
}

/// The point on segment `a`–`b` nearest to `p`, on the fixed-point grid. i128
/// intermediates: coordinates are bounded (|x| ≤ 100 km · UNIT, segments are claw-sized),
/// so the products stay far inside the type; the one truncating division costs at most a
/// grid unit (10 µm) — noise against [`CLAW_DOWN_BUFFER`].
fn closest_on_segment(a: Pos, b: Pos, p: Pos) -> (i64, i64) {
    let (dx, dz) = ((b.x - a.x) as i128, (b.z - a.z) as i128);
    let len2 = dx * dx + dz * dz;
    if len2 == 0 {
        return (a.x, a.z);
    }
    let t = ((p.x - a.x) as i128 * dx + (p.z - a.z) as i128 * dz).clamp(0, len2);
    (
        (a.x as i128 + dx * t / len2) as i64,
        (a.z as i128 + dz * t / len2) as i64,
    )
}

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

pub use super::cordic::{trig, trig_client};

#[cfg(test)]
mod tests {
    use super::*;

    fn players(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    fn neutral_for(sim: &Sim) -> BTreeMap<PlayerId, Input> {
        sim.participant_ids()
            .map(|id| (id, Input::default()))
            .collect()
    }

    /// Step with the crabs as scenery — held poses, no claws, nobody piloting.
    fn step_scenery(sim: &mut Sim, inputs: &BTreeMap<PlayerId, Input>) -> bool {
        let poses = hold_poses(sim);
        sim.step(inputs, Externals::crabs_only(&poses))
    }

    /// Every crab held at its current pose, with `claws` riding crab 0 — the common
    /// armed test feed.
    fn held_with_claws(sim: &Sim, claws: Vec<ClawPose>) -> Vec<CrabPose> {
        let mut poses = hold_poses(sim);
        poses[0].claws = claws;
        poses
    }

    #[test]
    fn mixed_foot_and_neutral_pilot_steps_deterministically() {
        let run = || {
            let mut sim = Sim::new(7, &players(2));
            let p0_start = sim.player(PlayerId(0)).unwrap().pos();
            let p1_start = sim.player(PlayerId(1)).unwrap().pos();
            for _ in 0..20 {
                let mut inputs = neutral_for(&sim);
                inputs.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
                step_scenery(&mut sim, &inputs);
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
    fn from_axes_clamps_and_quantizes_and_is_neutral_look() {
        let i = Input::from_axes(2.0, -2.0);
        assert_eq!((i.move_strafe, i.move_forward), (1000, -1000));
        assert_eq!((i.look_yaw, i.buttons), (0, 0));
        assert_eq!(Input::from_axes(0.0, 0.0), Input::default());
    }

    #[test]
    fn pilot_mask_keeps_restart_only_and_holds_cannot_refire_it() {
        let walking = Input::new(1.0, -1.0, 0.5, buttons::ACTION | buttons::RESTART);
        assert_eq!(
            walking.pilot_masked(),
            Input::new(0.0, 0.0, 0.0, buttons::RESTART),
            "piloting: walk axes and ACTION are stripped, RESTART survives (rl#261)"
        );
        assert_eq!(
            walking.pilot_masked().hold(),
            Input::default(),
            "a starved hold of a masked input can't re-fire RESTART"
        );
    }

    /// rl#305: the layout invariants that must survive EVERY frame draw — on-tile,
    /// crab clearance, and the objective outside the crab's reach — swept across
    /// seeds instead of proven at one origin.
    #[test]
    fn spawn_frame_randomizes_within_the_tile_interior() {
        let clamp = meters_to_grid(crab_world::training::targets::sample_clamp_half(
            &crab_world::terrain::TerrainGrid::gcr(),
        ));
        let on_tile = |p: Pos| p.x.abs() <= clamp && p.z.abs() <= clamp;
        let mut origins = std::collections::BTreeSet::new();
        for seed in 0..64u64 {
            let sim = Sim::new(seed, &players(3));
            origins.insert((sim.spawn_frame.origin.x, sim.spawn_frame.origin.z));
            for (_, p) in sim.players() {
                assert!(
                    on_tile(p.pos()),
                    "seed {seed}: player off-tile at {:?}",
                    p.pos()
                );
            }
            let ex = sim.extraction().pos();
            assert!(on_tile(ex), "seed {seed}: extraction off-tile at {ex:?}");
            for c in sim.crabs() {
                assert!(
                    on_tile(c.pos()),
                    "seed {seed}: crab off-tile at {:?}",
                    c.pos()
                );
                let min = MIN_CRAB_SPAWN_DISTANCE as i128;
                for (_, p) in sim.players() {
                    assert!(
                        dist2_i128(c.pos().x - p.pos().x, c.pos().z - p.pos().z) >= min * min,
                        "seed {seed}: spawn clearance broken"
                    );
                }
                assert!(
                    !within(ex.x, ex.z, c.pos().x, c.pos().z, 2 * EXTRACT_RADIUS),
                    "seed {seed}: the bearing jitter parked the crab on the objective"
                );
            }
        }
        assert!(
            origins.len() == 64,
            "64 seeds must draw 64 distinct locales, got {}",
            origins.len()
        );
    }

    #[test]
    fn spawn_is_deterministic_regardless_of_player_order() {
        let a = Sim::new(42, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        let b = Sim::new(42, &[PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert_eq!(a.state_hash(), b.state_hash());
    }

    /// Zero the player's yaw so an axis-aligned movement assertion holds: spawns face
    /// the run's random layout heading since rl#305, and these tests pin the MOVER's
    /// geometry, not the spawn draw.
    fn face_plus_z(sim: &mut Sim, id: PlayerId) {
        sim.players.get_mut(&id).expect("rostered").yaw = 0;
    }

    #[test]
    fn forward_input_moves_along_facing() {
        let mut sim = Sim::new(0, &players(1));
        face_plus_z(&mut sim, PlayerId(0));
        let p0 = sim.player(PlayerId(0)).unwrap().pos();
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        step_scenery(&mut sim, &inputs);
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
        let mut sim = Sim::new(0, &players(1));
        face_plus_z(&mut sim, PlayerId(0));
        let p0 = sim.player(PlayerId(0)).unwrap().pos();
        let mut right = BTreeMap::new();
        right.insert(PlayerId(0), Input::new(1.0, 0.0, 0.0, 0));
        step_scenery(&mut sim, &right);
        let p1 = sim.player(PlayerId(0)).unwrap().pos();
        assert_eq!(p1.z, p0.z, "no Z drift strafing at yaw 0");
        let dx = p1.x - p0.x;
        assert!(
            (dx - PLAYER_SPEED).abs() <= 1,
            "strafe-right step ≈ +PLAYER_SPEED in X, got {dx}"
        );
        let mut sim = Sim::new(0, &players(1));
        face_plus_z(&mut sim, PlayerId(0));
        let mut left = BTreeMap::new();
        left.insert(PlayerId(0), Input::new(-1.0, 0.0, 0.0, 0));
        step_scenery(&mut sim, &left);
        let dx_left = sim.player(PlayerId(0)).unwrap().pos().x - p0.x;
        assert_eq!(dx_left, -dx, "strafe-left mirrors strafe-right exactly");
    }

    #[test]
    fn predict_player_matches_step_for_the_local_avatar() {
        let inp = Input::new(0.6, -0.3, 0.4, 0);
        let mut stepped = Sim::new(7, &players(1));
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), inp);
        step_scenery(&mut stepped, &inputs);

        let mut predicted = Sim::new(7, &players(1));
        predicted.predict_player(PlayerId(0), inp);

        let sp = stepped.player(PlayerId(0)).unwrap();
        let pp = predicted.player(PlayerId(0)).unwrap();
        assert_eq!(
            (pp.pos(), pp.yaw()),
            (sp.pos(), sp.yaw()),
            "predicted local avatar must equal the stepped avatar"
        );

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
        let mut sim = Sim::new(0, &players(1));
        face_plus_z(&mut sim, PlayerId(0));
        let ticks = ((trig::TURN / 4) / MAX_YAW_TURNS_PER_TICK) as usize;
        for _ in 0..ticks {
            let mut inp = BTreeMap::new();
            inp.insert(PlayerId(0), Input::new(0.0, 0.0, 1.0, 0));
            step_scenery(&mut sim, &inp);
        }
        let before = sim.player(PlayerId(0)).unwrap().pos();
        let mut fwd = BTreeMap::new();
        fwd.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        step_scenery(&mut sim, &fwd);
        let after = sim.player(PlayerId(0)).unwrap().pos();
        let dx = after.x - before.x;
        let dz = after.z - before.z;
        assert!(
            dx.abs() > dz.abs(),
            "after a ~quarter turn, forward should move mostly in X (dx={dx}, dz={dz})"
        );
    }

    #[test]
    fn crab_pursues_and_claws_a_lone_player() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            let poses = drive_crab_toward_prey(&sim);
            sim.step(&neutral, Externals::crabs_only(&poses));
        }
        let crab_armed = sim.crabs()[0].pos();
        let prey = sim.player(PlayerId(0)).unwrap().pos();
        let d_start = dist2(crab_armed, prey);
        let poses = drive_crab_toward_prey(&sim);
        sim.step(&neutral, Externals::crabs_only(&poses));
        let d_next = dist2(sim.crabs()[0].pos(), sim.player(PlayerId(0)).unwrap().pos());
        assert!(d_next < d_start, "crab must close distance once driven");
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            let poses = drive_crab_toward_prey(&sim);
            sim.step(&neutral, Externals::crabs_only(&poses));
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
    fn adopted_crab_pose_seeds_and_is_hashed() {
        let pos = Pos {
            x: 7 * UNIT,
            z: -3 * UNIT,
        };
        let yaw = 123;

        let mut sim = Sim::new(0, &players(1));
        let pose = |pos| CrabPose {
            pos,
            yaw,
            claws: Vec::new(),
        };
        sim.adopt_crab_poses(&[pose(pos)]);
        assert_eq!(sim.crabs()[0].pos(), pos, "must seed the pose");
        assert_eq!(sim.crabs()[0].yaw(), yaw, "must seed the yaw");
        let h_seed = sim.state_hash();
        sim.adopt_crab_poses(&[pose(Pos {
            x: pos.x + 1,
            ..pos
        })]);
        assert_ne!(
            h_seed,
            sim.state_hash(),
            "the adopted crab pose must be folded into the state hash"
        );
    }

    #[test]
    fn reaching_extraction_with_action_wins() {
        let mut sim = Sim::new(0, &players(1));
        let ex = sim.extraction().pos();
        // The crab stays parked at her spawn: while CRAB_SPEED > PLAYER_SPEED (asserted
        // below — her honest pace since rl#257) the pure-pursuit test driver catches
        // EVERY on-foot route, so the old wide-dodge choreography can't win; catching a
        // standing player is pinned by `crab_pursues_and_claws_a_lone_player`, and THIS
        // test pins the extraction mechanic. No claws are fed here — a parked, clawless
        // crab cannot down anyone (rl#236) — so the run is a pure extraction exercise.
        const {
            assert!(
                CRAB_SPEED > PLAYER_SPEED,
                "the parked-crab premise inverted: re-measured charge speed no longer \
                 outruns players, a dodge route could win again"
            );
        }
        let route = [ex];
        let mut wp = 0usize;
        let mut won = false;
        for _ in 0..4000 {
            let p = sim.player(PlayerId(0)).unwrap();
            if p.status() != PlayerStatus::Alive {
                break;
            }
            let pp = p.pos();
            if wp < route.len() - 1 && within(pp.x, pp.z, route[wp].x, route[wp].z, UNIT) {
                wp += 1;
            }
            let target = route[wp];
            let want_yaw = trig::atan2_turns(target.x - pp.x, target.z - pp.z);
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
            step_scenery(&mut sim, &inp);
            if sim.outcome() == Outcome::Extracted {
                won = true;
                break;
            }
        }
        assert!(
            won,
            "a player who reaches the point clear of the crab and holds ACTION should extract"
        );
    }

    /// rl#247: a mid-round joiner must never materialize inside a crab's lethal reach —
    /// grace armed long ago, so an unlucky slot would Down them before their first input.
    #[test]
    fn joiner_never_spawns_inside_a_crab_spawn_clearance_disc() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..=STARTUP_GRACE_TICKS {
            step_scenery(&mut sim, &neutral);
        }
        // Park the crab dead on the joiner's roster slot (local x=0 for idx 1 of 2 —
        // the frame's origin), claw at her carapace point — downs are claw contact
        // only (rl#236).
        let parked = sim.spawn_frame.place(0, 0);
        let poses = vec![CrabPose {
            pos: parked,
            yaw: 0,
            claws: vec![claw_at(parked, 0, CLAW_M)],
        }];
        sim.adopt_crab_poses(&poses);
        sim.spawn_joining_player(PlayerId(1));
        let pos = sim.player(PlayerId(1)).unwrap().pos();
        // Slot selection stays on the run's spawn line: the chosen pos must be one of
        // the frame's line slots exactly (the scan only ever offers those).
        let blocked = 2 * MIN_CRAB_SPAWN_DISTANCE / SPAWN_SLOT_PITCH + 1;
        assert!(
            (-blocked..=blocked).any(|d| sim.spawn_frame.place(d * SPAWN_SLOT_PITCH, 0) == pos),
            "joiner slot {pos:?} is not a spawn-line slot of the run's frame"
        );
        assert!(
            !within(pos.x, pos.z, parked.x, parked.z, MIN_CRAB_SPAWN_DISTANCE),
            "joiner slot {pos:?} sits within the crab's spawn-clearance disc"
        );
        sim.step(&neutral_for(&sim), Externals::crabs_only(&poses));
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "the parked crab claws the host — proving the round is armed"
        );
        assert_eq!(
            sim.player(PlayerId(1)).unwrap().status(),
            PlayerStatus::Alive,
            "the joiner survives its first armed tick"
        );
    }

    #[test]
    fn joiner_keeps_its_roster_slot_when_clear() {
        let mut sim = Sim::new(0, &players(1));
        // Park the crab well clear: the round-start spawn sits ON the clearance ring
        // (within ~2 units of MIN since the rl#257 clamp binds), so relying on it
        // would hang this test's meaning off rounding crumbs. 2×MIN, not a bare meter
        // count — a re-measured charge speed grows the ring and a fixed park could
        // land back inside it (it did: 100 m vs the rl#266 ring).
        sim.adopt_crab_poses(&[CrabPose {
            pos: sim.spawn_frame.place(0, 2 * MIN_CRAB_SPAWN_DISTANCE),
            yaw: 0,
            claws: Vec::new(),
        }]);
        sim.spawn_joining_player(PlayerId(1));
        assert_eq!(
            sim.player(PlayerId(1)).unwrap().pos(),
            sim.spawn_frame.place(0, 0),
            "an unobstructed joiner takes its roster slot exactly"
        );
    }

    #[test]
    fn outcome_is_frozen_once_decided() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            let poses = drive_crab_toward_prey(&sim);
            sim.step(&neutral, Externals::crabs_only(&poses));
        }
        assert_ne!(
            sim.outcome(),
            Outcome::Ongoing,
            "round should have resolved"
        );
        let snapshot = |s: &Sim| {
            (
                s.players().collect::<Vec<_>>(),
                s.crabs().to_vec(),
                s.extraction(),
                s.outcome(),
            )
        };
        let frozen = snapshot(&sim);
        for _ in 0..10 {
            let poses = drive_crab_toward_prey(&sim);
            sim.step(&neutral, Externals::crabs_only(&poses));
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
        step_scenery(&mut sim, &inputs);
        assert_ne!(sim.state_hash(), h0);
    }

    #[test]
    #[should_panic(expected = "tick input incomplete")]
    fn missing_tick_input_panics_not_defaults_to_neutral() {
        let mut sim = Sim::new(0, &players(2));
        let mut partial = BTreeMap::new();
        partial.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        step_scenery(&mut sim, &partial);
    }

    #[test]
    fn trig_table_hits_cardinal_points() {
        use trig::{ONE, TURN, cos, sin};
        assert_eq!(sin(0), 0);
        assert_eq!(sin(TURN / 2), 0);
        assert!((sin(TURN / 4) - ONE).abs() <= 1);
        assert!((sin(3 * TURN / 4) + ONE).abs() <= 1);
        assert!((cos(0) - ONE).abs() <= 1);
        assert!(cos(TURN / 4).abs() <= 1);
        assert!((cos(TURN / 2) + ONE).abs() <= 1);
    }

    #[test]
    fn trig_pythagorean_identity_holds() {
        use trig::{ONE, cos, sin};
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
            i64::MAX as i128,
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
        use trig::{ONE, TURN, cos, sin};
        for a in 0..TURN {
            let want = ((a as f64 / TURN as f64 * std::f64::consts::TAU).sin() * ONE as f64).round()
                as i32;
            assert_eq!(sin(a), want, "sin table off at {a}");
        }
        for a in (0..TURN).step_by(257) {
            let want = ((a as f64 / TURN as f64 * std::f64::consts::TAU).cos() * ONE as f64).round()
                as i32;
            assert_eq!(cos(a), want, "cos off at {a}");
        }
    }

    #[test]
    fn state_hash_is_sensitive_to_every_hashed_field() {
        let base = Sim::new(7, &players(2));
        let h0 = base.state_hash();
        let hash_after = |mutate: &dyn Fn(&mut Sim)| {
            let mut s = base.clone();
            mutate(&mut s);
            s.state_hash()
        };
        let foot = PlayerId(0);

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
            hash_after(&|s| s.crabs[0].pos.x += 1),
            h0,
            "crab pos.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.crabs[0].pos.z += 1),
            h0,
            "crab pos.z must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.crabs[0].yaw += 1),
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
        // The other side of the contract: host-private machinery a peer never adopts
        // must NOT move the hash — it would desync the cross-peer diff (see the
        // comment in [`Sim::state_hash`]).
        assert_eq!(
            hash_after(&|s| s.restart_held = !s.restart_held),
            h0,
            "restart_held is host-private, deliberately unhashed"
        );
        assert_eq!(
            hash_after(&|s| s.round_start += 1),
            h0,
            "round_start is host-private, deliberately unhashed"
        );
        assert_eq!(
            hash_after(&|s| {
                let _: u64 = rand::Rng::r#gen(s.rng());
            }),
            h0,
            "the rng stream is host-private, deliberately unhashed"
        );
        assert_eq!(
            hash_after(&|s| s.spawn_frame.rot ^= 1),
            h0,
            "the spawn frame is host-private, deliberately unhashed"
        );
        assert_eq!(
            hash_after(&|s| s.config.seed ^= 0xdead_beef),
            h0,
            "config is deliberately not hashed (see Sim::config)"
        );
    }

    #[test]
    fn core_snapshot_roundtrip_reproduces_authoritative_state() {
        let mut original = Sim::new(7, &players(3));
        let posed = vec![CrabPose {
            pos: Pos { x: 4200, z: -1300 },
            yaw: 77,
            claws: Vec::new(),
        }];
        for _ in 0..5 {
            let mut inputs = neutral_for(&original);
            *inputs.get_mut(&PlayerId(0)).unwrap() = Input::from_axes(0.3, 1.0);
            original.step(&inputs, Externals::crabs_only(&posed));
        }
        original.players.get_mut(&PlayerId(1)).unwrap().status = PlayerStatus::Downed;

        let restored = CoreSnapshot::from_bytes(&original.core_snapshot().to_bytes())
            .expect("a freshly-built snapshot must round-trip through bytes");

        let mut target = original.clone();
        target.tick = 999;
        target.players.get_mut(&PlayerId(0)).unwrap().pos.x += 12_345;
        target.players.get_mut(&PlayerId(2)).unwrap().yaw += 3;
        target.players.get_mut(&PlayerId(1)).unwrap().status = PlayerStatus::Extracted;
        target.crabs[0].pos = Pos { x: -1, z: -2 };
        target.crabs[0].yaw = 9;
        target.outcome = Outcome::Wiped;
        target.config.players = vec![PlayerId(0)];
        assert_ne!(target.state_hash(), original.state_hash());

        target.apply_core_snapshot(restored);
        assert_eq!(
            target.state_hash(),
            original.state_hash(),
            "applying the round-tripped snapshot reproduces every hashed carried field"
        );
        assert_eq!(
            target.config.players, original.config.players,
            "the snapshot must carry the roster too"
        );
    }

    /// THE rl#236 owner call, pinned: standing under her carapace with no claw touching
    /// is SAFE — her body core downs nobody. A center-disc regression (the exact
    /// mechanism rl#236 deleted) fails here, not in a playtest.
    #[test]
    fn crab_body_overhead_without_a_claw_never_downs() {
        let mut sim = Sim::new(0, &players(1));
        let p = sim.player(PlayerId(0)).unwrap().pos();
        sim.adopt_crab_poses(&[CrabPose {
            pos: p,
            yaw: 0,
            claws: Vec::new(),
        }]);
        let neutral = neutral_for(&sim);
        for _ in 0..=STARTUP_GRACE_TICKS + 10 {
            step_scenery(&mut sim, &neutral);
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "her body core alone must never down (claw contact is the ONE mechanism)"
            );
        }
        // Same spot, now with a touching claw: downs — proving the round was armed and
        // the survival above was the mechanic, not a disarmed world.
        let poses = held_with_claws(&sim, vec![claw_at(p, 0, CLAW_M)]);
        sim.step(&neutral, Externals::crabs_only(&poses));
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
        );
    }

    #[test]
    fn claw_touch_downs_a_player_after_the_grace() {
        let mut sim = Sim::new(0, &players(1));
        let p = sim.player(PlayerId(0)).unwrap().pos();
        let poses = held_with_claws(&sim, vec![claw_at(p, 0, CLAW_M)]);
        let touching = Externals::crabs_only(&poses);
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral, touching);
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "no claw down during the startup grace"
            );
        }
        sim.step(&neutral, touching);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "a touching claw downs the player once armed"
        );
    }

    /// rl#258: a piloting player IS its craft — its walker rides the fed craft pose (one
    /// position, so the hunt targets the craft's shadow and stepping out resumes there),
    /// and a touching claw never downs it (the crab strikes the craft's real collider in
    /// the physics arena instead). A non-pilot at the same spot still goes down, so the
    /// exemption is the pilot set, not a weakened check.
    #[test]
    fn a_piloting_player_rides_its_craft_and_cannot_be_downed() {
        let mut sim = Sim::new(0, &players(2));
        let neutral = neutral_for(&sim);
        // Park both players' positions on the crab, touched by a claw. The pilot gets
        // there via its craft pose; the walker via the same feed minus pilot membership
        // (the server filters the pilots feed upstream, so passing it directly stands in
        // for "standing there on foot").
        let crab = sim.crabs()[0].pos();
        let poses = held_with_claws(&sim, vec![claw_at(crab, 0, CLAW_M)]);
        let both = |pilots: &[PlayerId]| {
            let mut m = BTreeMap::new();
            for &pid in pilots {
                m.insert(pid, PilotPose { pos: crab, yaw: 7 });
            }
            m
        };
        for _ in 0..=STARTUP_GRACE_TICKS + 1 {
            sim.step(
                &neutral,
                Externals {
                    crabs: &poses,
                    pilots: &both(&[PlayerId(0), PlayerId(1)]),
                },
            );
        }
        let p0 = sim.player(PlayerId(0)).unwrap();
        assert_eq!(p0.pos(), crab, "the walker rides the fed craft pose");
        assert_eq!(p0.yaw(), 7, "facing follows the craft too");
        assert_eq!(
            p0.status(),
            PlayerStatus::Alive,
            "inside a hull there is no walker to claw"
        );
        // Same spot, on foot: player 1 stops piloting (drops from the fed set) and downs.
        sim.step(
            &neutral,
            Externals {
                crabs: &poses,
                pilots: &both(&[PlayerId(0)]),
            },
        );
        assert_eq!(
            sim.player(PlayerId(1)).unwrap().status(),
            PlayerStatus::Downed,
            "the exemption ends the moment the player is on foot again"
        );
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Alive,
            "the still-piloting player stays exempt"
        );
    }

    #[test]
    fn claw_misses_do_not_down() {
        let mut sim = Sim::new(0, &players(1));
        let p = sim.player(PlayerId(0)).unwrap().pos();
        let neutral = neutral_for(&sim);
        let step_armed = |sim: &mut Sim, claw: ClawPose| {
            let poses = held_with_claws(sim, vec![claw]);
            for _ in 0..=STARTUP_GRACE_TICKS + 1 {
                sim.step(&neutral, Externals::crabs_only(&poses));
            }
            sim.player(PlayerId(0)).unwrap().status()
        };

        // Sweeping clear overhead: same XZ, but above the player's height span.
        assert_eq!(
            step_armed(&mut sim, claw_at(p, 0, 10 * CLAW_M)),
            PlayerStatus::Alive,
            "a claw passing overhead must not down anyone"
        );
        // Beside the player at body height, just past radius + buffer (measured from the
        // segment's near END — the segment check, not an endpoint/center approximation).
        let reach = CLAW_M / 2 + CLAW_DOWN_BUFFER;
        assert_eq!(
            step_armed(
                &mut sim,
                claw_at(p, 2 * CLAW_M + reach + CLAW_M / 10, CLAW_M)
            ),
            PlayerStatus::Alive,
            "a near miss beyond the buffer must not down"
        );
        // Same offset, but within reach of the near end: downs.
        assert_eq!(
            step_armed(
                &mut sim,
                claw_at(p, 2 * CLAW_M + reach - CLAW_M / 10, CLAW_M)
            ),
            PlayerStatus::Downed,
            "within the buffer of the capsule segment downs"
        );
    }

    #[test]
    fn crab_spawns_clear_of_every_player() {
        for n in 1..=8u8 {
            let sim = Sim::new(0, &players(n));
            let crab = sim.crabs()[0].pos();
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

    /// Pins [`MIN_CRAB_SPAWN_DISTANCE`] to the crab's actual body (rl#236). The world
    /// runs at rig scale (rl#256), so rig meters ARE world meters; [`CRAB_STATURE`] is
    /// pinned to every rig here. Against EVERY rig she can wear (the procedural fallback
    /// always; the mesh-fitted recipe when sally.glb resolves): spawns clear the carapace
    /// footprint's corner reach, so no spawn materializes under her body with her claws
    /// ([`ClawPose`] — the ONE down mechanism) already overhead.
    #[test]
    fn spawn_clearance_matches_crab_body() {
        use crab_world::bot::rig::{RestShape, fallback_recipe, recipe_silhouette};

        let mut recipes = vec![("fallback", fallback_recipe())];
        match crab_world::mesh_fallback::usable_model() {
            Ok(u) => recipes.push(("fitted", u.recipe.clone())),
            // No asset on this host is a legitimate degrade (sally.glb is not in git);
            // an asset that RESOLVES but can't be used is a breakage this test must not
            // paper over.
            Err(e) => assert!(
                crab_world::mesh_fallback::model_path().is_none(),
                "sally.glb resolves but is unusable: {e}"
            ),
        }

        for (name, recipe) in &recipes {
            let sil = recipe_silhouette(recipe);
            let RestShape::Cuboid { half, .. } = sil.carapace else {
                panic!("the carapace silhouette is a cuboid");
            };
            // One frame (rl#256): the rig IS world-scale. Pin the nominal stature to
            // every wearable rig so the derived human constants stay honest.
            assert!(
                (sil.natural_height() - CRAB_STATURE).abs() / CRAB_STATURE < 0.01,
                "{name}: natural height {} strays >1% from CRAB_STATURE {CRAB_STATURE}",
                sil.natural_height()
            );
            let corner_m = half.x.hypot(half.z);
            assert!(
                (MIN_CRAB_SPAWN_DISTANCE as f32 / UNIT as f32) > corner_m,
                "{name}: spawn clearance must exceed the carapace's corner reach {corner_m:.2} m"
            );
        }
    }

    #[test]
    fn restart_resets_the_round_to_spawn() {
        let mut sim = Sim::new(0xBEEF, &players(2));
        // The determinism mirror: same seed, same input history ⇒ the same restart
        // draw — the property that keeps every peer's picture of a restart identical.
        let mut twin = Sim::new(0xBEEF, &players(2));
        let world = |s: &Sim| {
            (
                s.players().collect::<Vec<_>>(),
                s.crabs().to_vec(),
                s.extraction(),
                s.outcome(),
            )
        };
        let opening = world(&sim);
        let mut fwd = BTreeMap::new();
        fwd.insert(PlayerId(0), Input::new(0.3, 1.0, 0.5, 0));
        fwd.insert(PlayerId(1), Input::new(-0.2, 1.0, 0.0, 0));
        for _ in 0..50 {
            step_scenery(&mut sim, &fwd);
            step_scenery(&mut twin, &fwd);
        }
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        restart.insert(PlayerId(1), Input::default());
        let edge = step_scenery(&mut sim, &restart);
        assert!(edge, "the press reports the restart edge");
        assert_eq!(sim.tick(), 51, "the tick stays monotone across a restart");
        assert_eq!(sim.outcome(), Outcome::Ongoing);
        assert!(
            sim.players()
                .all(|(_, p)| p.status() == PlayerStatus::Alive),
            "a restart revives everyone at spawn"
        );
        assert_ne!(
            world(&sim),
            opening,
            "the restart draws a FRESH spawn layout off the seed's stream (rl#305), \
             not a replay of the round-1 locale"
        );
        step_scenery(&mut twin, &restart);
        assert_eq!(
            world(&sim),
            world(&twin),
            "a restarted round is deterministic in (seed, input history)"
        );
    }

    #[test]
    fn restart_works_after_the_round_is_decided() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            let poses = drive_crab_toward_prey(&sim);
            sim.step(&neutral, Externals::crabs_only(&poses));
        }
        assert_eq!(sim.outcome(), Outcome::Wiped, "round should have ended");
        let tick_at_loss = sim.tick();
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        step_scenery(&mut sim, &restart);
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
        let mut sim = Sim::new(0, &players(1));
        let mut held = BTreeMap::new();
        held.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(step_scenery(&mut sim, &held), "first press restarts");
        assert!(
            !step_scenery(&mut sim, &held),
            "a held key doesn't re-restart"
        );
        assert!(
            !step_scenery(&mut sim, &held),
            "still held, still no re-restart"
        );
        assert_eq!(sim.tick(), 3, "every tick counted, restart included");
        let neutral = neutral_for(&sim);
        assert!(!step_scenery(&mut sim, &neutral), "release: no restart");
        assert!(
            step_scenery(&mut sim, &held),
            "a new press after release restarts again"
        );
    }

    #[test]
    fn restart_keeps_two_peers_in_lockstep() {
        let mut a = Sim::new(0x5151, &players(2));
        let mut b = Sim::new(0x5151, &players(2));
        let mut restarts = 0u32;
        for t in 0..120u64 {
            let mut inputs = BTreeMap::new();
            let restart_bit = if t == 40 { buttons::RESTART } else { 0 };
            inputs.insert(PlayerId(0), Input::new(0.4, 1.0, 0.2, 0));
            inputs.insert(PlayerId(1), Input::new(-0.3, 1.0, -0.1, restart_bit));
            restarts += u32::from(step_scenery(&mut a, &inputs));
            step_scenery(&mut b, &inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "peers must stay bit-identical across a restart (tick {t})"
            );
        }
        assert_eq!(restarts, 1, "the mid-run restart fired exactly once");
        assert_eq!(a.tick(), 120, "a restart never rewinds the tick");
    }

    #[test]
    fn restart_grants_a_fresh_startup_grace() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..(3 * STARTUP_GRACE_TICKS) {
            step_scenery(&mut sim, &neutral);
        }
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(step_scenery(&mut sim, &restart), "the restart fires");
        let crab0 = sim.crabs()[0].pos();
        sim.players.get_mut(&PlayerId(0)).unwrap().pos = crab0;
        // A claw touching the parked player on every post-restart tick, so only the
        // grace keeps them up.
        let poses = held_with_claws(&sim, vec![claw_at(crab0, 0, CLAW_M)]);
        let touching = Externals::crabs_only(&poses);
        for i in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral, touching);
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "no claw down during the post-restart grace (tick {i} into the round)"
            );
        }
        sim.step(&neutral, touching);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "the crab arms once the post-restart grace ends"
        );
    }

    #[test]
    fn multi_crab_round_spawns_hashes_and_snapshots_per_crab() {
        let mut sim = Sim::new(0, &players(2));
        sim.configure_crabs(2);
        assert_eq!(sim.crabs().len(), 2);
        assert_ne!(
            sim.crabs()[0].pos(),
            sim.crabs()[1].pos(),
            "crabs spawn staggered, not stacked"
        );
        for (i, crab) in sim.crabs().iter().enumerate() {
            let nearest = sim
                .players()
                .map(|(_, p)| dist2(crab.pos(), p.pos()))
                .min()
                .unwrap();
            let min = MIN_CRAB_SPAWN_DISTANCE as i128;
            assert!(
                nearest >= min * min,
                "crab {i} spawns clear of every player"
            );
        }

        assert_ne!(sim.state_hash(), Sim::new(0, &players(2)).state_hash());
        let h0 = sim.state_hash();
        let mut poses = hold_poses(&sim);
        poses[1].pos.x += 1;
        sim.adopt_crab_poses(&poses);
        assert_ne!(sim.state_hash(), h0, "crab 1's pose folds into the hash");

        // (Downing needs no per-crab leg: claws are a pooled, crab-agnostic feed —
        // the single-crab claw tests cover the mechanism, rl#236.)
        let snap = sim.core_snapshot();
        assert_eq!(snap.crabs.len(), 2);
        let restored = CoreSnapshot::from_bytes(&snap.to_bytes()).unwrap();
        let mut client = Sim::new(0, &players(2));
        client.apply_core_snapshot(restored);
        assert_eq!(
            client.crabs().len(),
            2,
            "an adopting client takes the host's crab count"
        );

        let mut restart = neutral_for(&sim);
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(step_scenery(&mut sim, &restart));
        assert_eq!(
            sim.crabs().len(),
            2,
            "restart rebuilds the configured count"
        );
    }

    #[test]
    #[should_panic(expected = "disagree on the crab count")]
    fn pose_count_mismatch_panics() {
        let mut sim = Sim::new(0, &players(1));
        sim.adopt_crab_poses(&[]);
    }

    fn dist2(a: Pos, b: Pos) -> i128 {
        dist2_i128(a.x - b.x, a.z - b.z)
    }
}
