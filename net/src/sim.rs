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
    /// double a grab/restart. See [`crate::server`]'s hold semantics.
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
    (PLAYER_SPEED_HEIGHTS_PER_S * PLAYER_HEIGHT * UNIT as f32 / TICK_HZ as f32) as i64;

/// Test-driver step per tick, folded from the ONE measured speed
/// ([`CRAB_CHARGE_SPEED_PER_S`], rl#257) so pursuit/grace tests exercise her honest
/// pace — a second bare speed here would drift from reality (and did: 130).
#[cfg(test)]
const CRAB_SPEED: i64 = CRAB_CHARGE_SPEED_PER_S / TICK_HZ as i64;

#[cfg(test)]
pub(crate) fn drive_crab_toward_prey(sim: &mut Sim) {
    if sim.tick() < sim.round_start + STARTUP_GRACE_TICKS {
        return;
    }
    for idx in 0..sim.crabs().len() {
        let Some(target) = sim.nearest_living_player_pos(idx) else {
            continue;
        };
        let mut pos = sim.crabs()[idx].pos();
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
        sim.set_external_crab_pose(idx, pos, yaw);
    }
}

pub const CRAB_SCALE: i64 = 12;

/// Sally's nominal stature in world meters. The world's absolute scale IS the rig's
/// (rl#256), so this is a pinned copy of the rigs' measured natural height — fallback
/// 0.61146 m, mesh-fitted 0.61150 m; `grab_reach_matches_crab_body` holds it within 1%
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

/// 2D reach of the grab around the crab's sim pos (the carapace ground point — the bridge
/// integrates carapace deltas): deep under her body core. rl#236 pinned this to the
/// full inscribed carapace footprint (~0.30 m); rl#249 shrank it, because that
/// disc downed prey her TRAINED near-field claw strike should engage — the circle shadowed
/// the claw mechanic rl#249 asks for (rl#236's own side observation). Now the core disc
/// covers only "she is standing on you"; the 0.17–0.48 m shell belongs to real claw
/// contact ([`ClawPose`]). Legs still deliberately do not grab.
/// `grab_reach_matches_crab_body` pins the disc within the carapace footprint, under
/// the claw shell.
pub const CRAB_GRAB_RADIUS: i64 = player_heights(6.0 / 1.8);

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
/// outside her carapace footprint (corner reach ~0.51 m) and [`CRAB_GRAB_RADIUS`];
/// `grab_reach_matches_crab_body` cross-checks that floor against every rig.
const MIN_CRAB_SPAWN_DISTANCE: i64 = CRAB_CHARGE_SPEED_PER_S * SPAWN_GRACE_SECS;

/// Spacing between player spawn slots along the z=0 spawn line.
const SPAWN_SLOT_PITCH: i64 = player_heights(2.0 / 1.8);

/// Reach-the-objective radius.
pub const EXTRACT_RADIUS: i64 = player_heights(2.0 / 1.8);

pub(crate) const MAX_YAW_TURNS_PER_TICK: i32 = trig::TURN / 24;

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

/// One of Sally's claw colliders as of this tick, bridged into sim space: the pincer's
/// real physics capsule (rl#249 — no separate hitbox to drift), as an XZ segment with
/// per-end heights and the capsule radius, all on the fixed-point grid. Like the crab
/// pose this is external per-tick INPUT, not round state: the server's bridge refreshes
/// it before every step, clients never see it (they receive the resulting
/// [`PlayerStatus`] via snapshot), so it is excluded from `state_hash` and
/// `core_snapshot`.
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

/// A piloting player's craft pose in sim space — see [`Sim::set_external_pilots`].
#[derive(Debug, Clone, Copy)]
pub struct PilotPose {
    pub pos: Pos,
    pub yaw: i32,
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
    /// All crabs' claw colliders as of this tick, pooled (the down check doesn't care
    /// which crab owns a claw). External per-tick input, not state — see [`ClawPose`].
    claws: Vec<ClawPose>,
    /// Every piloting player's craft pose, bridged back into sim space (rl#258): while a
    /// player flies, its ONE position is the craft's — the walker rides the craft instead
    /// of standing as a husk at the boarding spot, so the crab hunts the craft's shadow
    /// and stepping out resumes on foot right there. Membership doubles as the down
    /// exemption: a pilot is inside a hull, so grabs and claws act on the craft's REAL
    /// collider (rapier), never the walker. External per-tick input like [`ClawPose`]:
    /// server-only, overwritten before every step, excluded from `state_hash`/snapshots
    /// (clients see the resulting player pos).
    pilots: BTreeMap<PlayerId, PilotPose>,
    extraction: ExtractionPoint,
    outcome: Outcome,
    rng: ChaCha8Rng,
    restart_held: bool,
    round_start: u64,
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
        let (players, crabs, extraction) = Self::spawn_state(&config);
        Self {
            tick: 0,
            players,
            crabs,
            claws: Vec::new(),
            pilots: BTreeMap::new(),
            extraction,
            outcome: Outcome::Ongoing,
            rng: ChaCha8Rng::seed_from_u64(seed),
            restart_held: false,
            round_start: 0,
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
        let (players, crabs, extraction) = Self::spawn_state(&self.config);
        self.players = players;
        self.crabs = crabs;
        self.extraction = extraction;
    }

    pub fn set_external_crab_pose(&mut self, crab: usize, pos: Pos, yaw: i32) {
        assert!(
            crab < self.crabs.len(),
            "external crab pose for crab {crab}, but the round has {} — the bridge and the sim \
             disagree on the binding count",
            self.crabs.len()
        );
        self.crabs[crab].pos = pos;
        self.crabs[crab].yaw = yaw;
    }

    /// Refresh this tick's claw colliders (rl#249). Server-only, before each step; stale
    /// claws must not outlive the tick that measured them, so the bridge overwrites the
    /// whole set every time.
    pub fn set_external_claws(&mut self, claws: Vec<ClawPose>) {
        self.claws = claws;
    }

    /// Refresh this tick's piloting set + craft poses (rl#258). Server-only, before each
    /// step, whole-set overwrite like [`Self::set_external_claws`] — see [`Sim::pilots`].
    pub fn set_external_pilots(&mut self, pilots: BTreeMap<PlayerId, PilotPose>) {
        self.pilots = pilots;
    }

    fn reset(&mut self) {
        let (players, crabs, extraction) = Self::spawn_state(&self.config);
        self.round_start = self.tick;
        self.players = players;
        self.crabs = crabs;
        self.claws.clear();
        self.pilots.clear();
        self.extraction = extraction;
        self.outcome = Outcome::Ongoing;
        self.rng = ChaCha8Rng::seed_from_u64(self.config.seed);
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
                yaw: 0,
                status: PlayerStatus::Alive,
            },
        );
    }

    /// Nearest spawn-line slot to `x` clear of every crab by [`MIN_CRAB_SPAWN_DISTANCE`]
    /// — the same clearance round-start [`Self::spawn_crab`] keeps toward players, so a
    /// mid-round joiner gets the round-start guarantee and is never Downed before its
    /// first input (rl#247). Walks slots outward, alternating east/west, staying on the
    /// z=0 line (no per-join grace: that would need wire-format state, and re-arming
    /// `round_start` would grace everyone mid-fight).
    fn nearest_clear_join_slot(&self, x: i64) -> Pos {
        // A crab blocks a closed 2·MIN chord of the line: at most this many slots. The
        // scan offers 2·blocked·crabs + 1 candidates, so one is always clear.
        let blocked_per_crab = 2 * MIN_CRAB_SPAWN_DISTANCE / SPAWN_SLOT_PITCH + 1;
        (0..=blocked_per_crab * self.crabs.len() as i64)
            .flat_map(|d| [d, -d])
            .map(|d| Pos {
                x: x + d * SPAWN_SLOT_PITCH,
                z: 0,
            })
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

    fn spawn_state(cfg: &RoundConfig) -> (BTreeMap<PlayerId, Player>, Vec<Crab>, ExtractionPoint) {
        let mut map = BTreeMap::new();
        let n = cfg.players.len() as i64;
        for (i, &id) in cfg.players.iter().enumerate() {
            let x = (i as i64 - n / 2) * SPAWN_SLOT_PITCH;
            map.insert(
                id,
                Player {
                    pos: Pos { x, z: 0 },
                    yaw: 0,
                    status: PlayerStatus::Alive,
                },
            );
        }
        // The objective sits BEYOND the crab's spawn ring by more than her ~0.48 m
        // claw-contact shell (see [`CRAB_GRAB_RADIUS`]), preserving the layout
        // invariant spawn line < crab < extraction whatever MIN becomes — with a margin barely
        // past her ring the rl#257 clearance bump parked her ON the objective, its disc
        // inside her claw shell at round start.
        let extraction = ExtractionPoint {
            pos: Pos {
                x: 0,
                z: MIN_CRAB_SPAWN_DISTANCE + player_heights(10.0),
            },
        };
        let crabs = (0..cfg.crabs).map(|i| Self::spawn_crab(&map, i)).collect();
        (map, crabs, extraction)
    }

    fn spawn_crab(players: &BTreeMap<PlayerId, Player>, idx: usize) -> Crab {
        // The base pos staggers crabs and seeds the push-out BEARING; the clearance
        // clamp below (binding whenever the base sits inside MIN — every near-field
        // crab since rl#257 grew MIN past this base) sets the actual distance, so
        // round-start and joiner safety share the one constant.
        let mut pos = Pos {
            x: player_heights(6.0 / 1.8) + idx as i64 * player_heights(8.0 / 1.8),
            z: player_heights(20.0 / 1.8),
        };
        if let Some(nearest) = players
            .values()
            .min_by_key(|p| dist2_i128(pos.x - p.pos.x, pos.z - p.pos.z))
        {
            let dx = pos.x - nearest.pos.x;
            let dz = pos.z - nearest.pos.z;
            let d2 = dist2_i128(dx, dz);
            let min = MIN_CRAB_SPAWN_DISTANCE as i128;
            if d2 < min * min {
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
        }
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

    pub fn step(&mut self, inputs: &BTreeMap<PlayerId, Input>) -> bool {
        self.require_complete_inputs(inputs);
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
        for (id, pp) in &self.pilots {
            if let Some(p) = self.players.get_mut(id) {
                p.pos = pp.pos;
                p.yaw = pp.yaw;
            }
        }

        let armed = self.tick > self.round_start + STARTUP_GRACE_TICKS;

        if armed {
            // Pilots are exempt: inside a hull there is no walker to grab or claw — the
            // crab strikes the craft's REAL collider in the physics arena instead (rl#258).
            for crab in &self.crabs {
                for (id, p) in self.players.iter_mut() {
                    if p.status == PlayerStatus::Alive
                        && !self.pilots.contains_key(id)
                        && within(p.pos.x, p.pos.z, crab.pos.x, crab.pos.z, CRAB_GRAB_RADIUS)
                    {
                        p.status = PlayerStatus::Downed;
                    }
                }
            }
            for claw in &self.claws {
                for (id, p) in self.players.iter_mut() {
                    if p.status == PlayerStatus::Alive
                        && !self.pilots.contains_key(id)
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
            // External per-tick input, server-only — never replicated, so hashing it
            // would desync every host/client comparison (see [`ClawPose`]).
            claws: _,
            pilots: _,
            extraction,
            outcome,
            rng,
            restart_held,
            round_start,
            config: _,
        } = self;

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
        h.write(&[u8::from(*restart_held)]);
        h.write(&round_start.to_le_bytes());
        h.write(&rand::Rng::r#gen::<u64>(&mut rng.clone()).to_le_bytes());
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
            // Input, not state — the down decisions it produced are already in `players`.
            claws: _,
            pilots: _,
            extraction: _,
            outcome,
            rng: _,
            restart_held: _,
            round_start: _,
            config,
        } = self;
        CoreSnapshot {
            tick: *tick,
            players: players.clone(),
            crabs: crabs.clone(),
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

    #[test]
    fn mixed_foot_and_neutral_pilot_steps_deterministically() {
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

    #[test]
    fn spawn_is_deterministic_regardless_of_player_order() {
        let a = Sim::new(42, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        let b = Sim::new(42, &[PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn forward_input_moves_along_facing() {
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
        let mut sim = Sim::new(0, &players(1));
        let mut left = BTreeMap::new();
        left.insert(PlayerId(0), Input::new(-1.0, 0.0, 0.0, 0));
        sim.step(&left);
        let dx_left = sim.player(PlayerId(0)).unwrap().pos().x - p0.x;
        assert_eq!(dx_left, -dx, "strafe-left mirrors strafe-right exactly");
    }

    #[test]
    fn predict_player_matches_step_for_the_local_avatar() {
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
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        let crab_armed = sim.crabs()[0].pos();
        let prey = sim.player(PlayerId(0)).unwrap().pos();
        let d_start = dist2(crab_armed, prey);
        drive_crab_toward_prey(&mut sim);
        sim.step(&neutral);
        let d_next = dist2(sim.crabs()[0].pos(), sim.player(PlayerId(0)).unwrap().pos());
        assert!(d_next < d_start, "crab must close distance once driven");
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
    fn external_crab_pose_seeds_and_is_hashed() {
        let pos = Pos {
            x: 7 * UNIT,
            z: -3 * UNIT,
        };
        let yaw = 123;

        let mut sim = Sim::new(0, &players(1));
        sim.set_external_crab_pose(0, pos, yaw);
        assert_eq!(sim.crabs()[0].pos(), pos, "must seed the pose");
        assert_eq!(sim.crabs()[0].yaw(), yaw, "must seed the yaw");
        let h_seed = sim.state_hash();
        sim.set_external_crab_pose(
            0,
            Pos {
                x: pos.x + 1,
                ..pos
            },
            yaw,
        );
        assert_ne!(
            h_seed,
            sim.state_hash(),
            "the external crab pose must be folded into the state hash"
        );
    }

    #[test]
    fn reaching_extraction_with_action_wins() {
        let mut sim = Sim::new(0, &players(1));
        let ex = sim.extraction().pos();
        // The crab stays parked at her spawn: while CRAB_SPEED > PLAYER_SPEED (asserted
        // below — her honest pace since rl#257) the pure-pursuit test driver catches
        // EVERY on-foot route, so the old wide-dodge choreography can't win; catching a
        // standing player is pinned by `crab_pursues_and_grabs_a_lone_player`, and THIS
        // test pins the extraction mechanic. Her spawn ring still makes the run real:
        // the straight line to the point passes MIN_CRAB_SPAWN_DISTANCE-proportional
        // meters from her (the push-out bearing has a fixed x component), far clear of
        // [`CRAB_GRAB_RADIUS`] of an armed crab.
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
            sim.step(&inp);
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
    fn joiner_never_spawns_inside_a_crab_grab_disc() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..=STARTUP_GRACE_TICKS {
            sim.step(&neutral);
        }
        // Park the crab dead on the joiner's roster slot (x=0 for idx 1 of 2).
        let parked = Pos { x: 0, z: 0 };
        sim.set_external_crab_pose(0, parked, 0);
        sim.spawn_joining_player(PlayerId(1));
        let pos = sim.player(PlayerId(1)).unwrap().pos();
        assert_eq!(pos.z, 0, "slot selection stays on the spawn line");
        assert!(
            !within(pos.x, pos.z, parked.x, parked.z, MIN_CRAB_SPAWN_DISTANCE),
            "joiner slot {pos:?} sits within the crab's spawn-clearance disc"
        );
        sim.step(&neutral_for(&sim));
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "the parked crab grabs the host — proving the round is armed"
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
        sim.set_external_crab_pose(
            0,
            Pos {
                x: 0,
                z: 2 * MIN_CRAB_SPAWN_DISTANCE,
            },
            0,
        );
        sim.spawn_joining_player(PlayerId(1));
        assert_eq!(
            sim.player(PlayerId(1)).unwrap().pos(),
            Pos { x: 0, z: 0 },
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
                s.crabs().to_vec(),
                s.extraction(),
                s.outcome(),
            )
        };
        let frozen = snapshot(&sim);
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
    #[should_panic(expected = "tick input incomplete")]
    fn missing_tick_input_panics_not_defaults_to_neutral() {
        let mut sim = Sim::new(0, &players(2));
        let mut partial = BTreeMap::new();
        partial.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        sim.step(&partial);
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
        assert_ne!(
            hash_after(&|s| {
                let _: u64 = rand::Rng::r#gen(s.rng());
            }),
            h0,
            "rng stream position must be hashed"
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
        for _ in 0..5 {
            original.set_external_crab_pose(0, Pos { x: 4200, z: -1300 }, 77);
            let mut inputs = neutral_for(&original);
            *inputs.get_mut(&PlayerId(0)).unwrap() = Input::from_axes(0.3, 1.0);
            original.step(&inputs);
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

    #[test]
    fn crab_holds_and_cannot_grab_during_startup_grace() {
        let mut sim = Sim::new(0, &players(1));
        let crab0 = sim.crabs()[0].pos();
        sim.players.get_mut(&PlayerId(0)).unwrap().pos = crab0;
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral);
            assert_eq!(
                sim.crabs()[0].pos(),
                crab0,
                "crab holds its spawn during grace"
            );
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "no grab during the startup grace"
            );
            assert_eq!(sim.outcome(), Outcome::Ongoing);
        }
        sim.step(&neutral);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "crab arms and grabs once the grace ends"
        );
    }

    /// The claw tests' yardstick, 5/9 of a player height (≈0.028 m) — every claw-test
    /// length is a multiple, so the geometry keeps one legible scale. Anchored on the
    /// player, NOT on [`CLAW_DOWN_BUFFER`]: that's the tune-on-playtest feel knob, and
    /// retuning it must not silently rescale this geometry (e.g. drop the "overhead"
    /// claw under the player's height span).
    const CLAW_M: i64 = PLAYER_HEIGHT_FP * 5 / 9;

    /// A claw capsule at body height. `dx` slides it sideways off the player so the
    /// near-miss cases stay one call.
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

    #[test]
    fn claw_touch_downs_a_player_after_the_grace() {
        let mut sim = Sim::new(0, &players(1));
        let p = sim.player(PlayerId(0)).unwrap().pos();
        assert!(
            !within(
                p.x,
                p.z,
                sim.crabs()[0].pos().x,
                sim.crabs()[0].pos().z,
                CRAB_GRAB_RADIUS
            ),
            "the player must spawn outside the grab circle for this to isolate the claw path"
        );
        sim.set_external_claws(vec![claw_at(p, 0, CLAW_M)]);
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral);
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "no claw down during the startup grace"
            );
        }
        sim.step(&neutral);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "a touching claw downs the player once armed"
        );
    }

    /// rl#258: a piloting player IS its craft — its walker rides the fed craft pose (one
    /// position, so the hunt targets the craft's shadow and stepping out resumes there),
    /// and neither the grab circle nor a touching claw downs it (the crab strikes the
    /// craft's real collider in the physics arena instead). A non-pilot at the same spot
    /// still goes down, so the exemption is the pilot set, not a weakened check.
    #[test]
    fn a_piloting_player_rides_its_craft_and_cannot_be_downed() {
        let mut sim = Sim::new(0, &players(2));
        let neutral = neutral_for(&sim);
        // Park both players' positions on the crab: inside the grab circle AND touched by
        // a claw. The pilot gets there via its craft pose; the walker via the same feed
        // minus pilot membership (set_external_pilots is filtered upstream, so feeding it
        // directly stands in for "standing there on foot").
        let crab = sim.crabs()[0].pos();
        let claw = claw_at(crab, 0, CLAW_M);
        let both = |pilots: &[PlayerId]| {
            let mut m = BTreeMap::new();
            for &pid in pilots {
                m.insert(pid, PilotPose { pos: crab, yaw: 7 });
            }
            m
        };
        for _ in 0..=STARTUP_GRACE_TICKS + 1 {
            sim.set_external_claws(vec![claw]);
            sim.set_external_pilots(both(&[PlayerId(0), PlayerId(1)]));
            sim.step(&neutral);
        }
        let p0 = sim.player(PlayerId(0)).unwrap();
        assert_eq!(p0.pos(), crab, "the walker rides the fed craft pose");
        assert_eq!(p0.yaw(), 7, "facing follows the craft too");
        assert_eq!(
            p0.status(),
            PlayerStatus::Alive,
            "inside a hull there is no walker to grab or claw"
        );
        // Same spot, on foot: player 1 stops piloting (drops from the fed set) and downs.
        sim.set_external_claws(vec![claw]);
        sim.set_external_pilots(both(&[PlayerId(0)]));
        sim.step(&neutral);
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
            sim.set_external_claws(vec![claw]);
            for _ in 0..=STARTUP_GRACE_TICKS + 1 {
                sim.step(&neutral);
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

    /// Pins [`CRAB_GRAB_RADIUS`] and [`MIN_CRAB_SPAWN_DISTANCE`] to the crab's actual body
    /// (rl#236) and the grab to the claw-strike band (rl#249). The world runs at rig
    /// scale (rl#256), so rig meters ARE world meters; [`CRAB_STATURE`] is pinned to
    /// every rig here. Against EVERY rig she can wear (the procedural fallback always;
    /// the mesh-fitted recipe when sally.glb resolves): the grab never reaches beyond the
    /// carapace footprint's inscribed circle (the disc is anchored on the carapace, so
    /// inscribed = "certainly under her body" whatever her yaw), and spawns clear the
    /// footprint's corner reach. The rl#249 upper bound: the grab disc must stop short of
    /// the hunt-target band's far edge, else it downs prey before the trained near-field
    /// claw strike can engage and the claw trigger ([`ClawPose`]) is unreachable — exactly
    /// the shadowing rl#236 observed.
    #[test]
    fn grab_reach_matches_crab_body() {
        use crab_world::bot::rig::{RestShape, fallback_recipe, recipe_silhouette};

        let mut recipes = vec![("fallback", fallback_recipe())];
        match crab_world::mesh_fallback::usable_model() {
            Ok(u) => recipes.push(("fitted", u.recipe.clone())),
            // No asset on this host is a legitimate degrade (sally.glb is not in git);
            // an asset that RESOLVES but can't be used is a breakage this test must not
            // paper over.
            Err(e) => assert!(
                crab_world::bot::meshfit::model_path().is_none(),
                "sally.glb resolves but is unusable: {e}"
            ),
        }

        let grab_m = CRAB_GRAB_RADIUS as f32 / UNIT as f32;
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
            let inscribed_m = half.x.min(half.z);
            let corner_m = half.x.hypot(half.z);
            assert!(
                grab_m <= inscribed_m,
                "{name}: grab reach {grab_m:.2} m pokes beyond the carapace's inscribed \
                 footprint radius {inscribed_m:.2} m"
            );
            assert!(
                (MIN_CRAB_SPAWN_DISTANCE as f32 / UNIT as f32) > corner_m,
                "{name}: spawn clearance must exceed the carapace's corner reach {corner_m:.2} m"
            );
        }
        // (An rl#249 stanza compared grab_m against TARGET_ARENA_HALF — an ARENA-meter
        // band, ~318 sim m since the rl#254 frame fix, so the comparison pinned nothing.
        // The inscribed-footprint bound above carries the disc-vs-claw-shell intent.)
    }

    #[test]
    fn restart_resets_the_round_to_spawn() {
        let mut sim = Sim::new(0xBEEF, &players(2));
        let fresh = Sim::new(0xBEEF, &players(2));
        let mut fwd = BTreeMap::new();
        fwd.insert(PlayerId(0), Input::new(0.3, 1.0, 0.5, 0));
        fwd.insert(PlayerId(1), Input::new(-0.2, 1.0, 0.0, 0));
        for _ in 0..50 {
            sim.step(&fwd);
        }
        let world = |s: &Sim| {
            (
                s.players().collect::<Vec<_>>(),
                s.crabs().to_vec(),
                s.extraction(),
                s.outcome(),
            )
        };
        assert_ne!(
            world(&sim),
            world(&fresh),
            "the round should have diverged from spawn before restart"
        );
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
        let mut sim = Sim::new(0, &players(1));
        let mut held = BTreeMap::new();
        held.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(sim.step(&held), "first press restarts");
        assert!(!sim.step(&held), "a held key doesn't re-restart");
        assert!(!sim.step(&held), "still held, still no re-restart");
        assert_eq!(sim.tick(), 3, "every tick counted, restart included");
        assert!(!sim.step(&neutral_for(&sim)), "release: no restart");
        assert!(sim.step(&held), "a new press after release restarts again");
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
            restarts += u32::from(a.step(&inputs));
            b.step(&inputs);
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
            sim.step(&neutral);
        }
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(sim.step(&restart), "the restart fires");
        let crab0 = sim.crabs()[0].pos();
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

    #[test]
    fn multi_crab_round_spawns_hashes_and_grabs_per_crab() {
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
        let c1 = sim.crabs()[1];
        let nudged = Pos {
            x: c1.pos().x + 1,
            z: c1.pos().z,
        };
        sim.set_external_crab_pose(1, nudged, c1.yaw());
        assert_ne!(sim.state_hash(), h0, "crab 1's pose folds into the hash");

        let neutral = neutral_for(&sim);
        for _ in 0..=STARTUP_GRACE_TICKS {
            sim.step(&neutral);
        }
        let prey = sim.player(PlayerId(1)).unwrap().pos();
        sim.set_external_crab_pose(1, prey, 0);
        sim.step(&neutral);
        assert_eq!(
            sim.player(PlayerId(1)).unwrap().status(),
            PlayerStatus::Downed,
            "the second crab's reach downs a player"
        );

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
        assert!(sim.step(&restart));
        assert_eq!(
            sim.crabs().len(),
            2,
            "restart rebuilds the configured count"
        );
    }

    #[test]
    #[should_panic(expected = "disagree on the binding count")]
    fn pose_for_an_unconfigured_crab_panics() {
        let mut sim = Sim::new(0, &players(1));
        sim.set_external_crab_pose(1, Pos::default(), 0);
    }

    fn dist2(a: Pos, b: Pos) -> i128 {
        dist2_i128(a.x - b.x, a.z - b.z)
    }
}
