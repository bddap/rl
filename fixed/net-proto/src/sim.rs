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
    pub const SPRINT: u8 = 1 << 2;
    pub const JUMP: u8 = 1 << 3;
    pub const SLIDE: u8 = 1 << 4;
    /// The foot buttons that stay live while piloting ([`super::Input::pilot_masked`]):
    /// RESTART works in every context (rl#261 — every cockpit legend in
    /// `net::controls` promises "Restart round"), while the rest must NOT — the ship's
    /// brake shares South with Jump (rl#355) and Extract's triggers ride the flight
    /// controls, so an unmasked pilot would hop or extract mid-maneuver.
    pub const PILOT_MASK: u8 = RESTART;
    /// The held-state buttons a starved-tick hold preserves ([`super::Input::hold`]):
    /// sprint and slide are stances like the move axes, not taps — dropping them
    /// mid-lag would break a slide the player is still holding.
    pub const HOLD_MASK: u8 = SPRINT | SLIDE;
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
    /// double an extract/restart; the held-stance buttons ([`buttons::HOLD_MASK`]) persist
    /// like the axes. See [`crate::server`]'s hold semantics.
    pub fn hold(self) -> Self {
        Self {
            move_strafe: self.move_strafe,
            move_forward: self.move_forward,
            look_yaw: 0,
            buttons: self.buttons & buttons::HOLD_MASK,
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

pub const PLAYER_SPEED: i64 =
    (PLAYER_SPEED_HEIGHTS_PER_S * PLAYER_HEIGHT * UNIT as f32 / TICK_HZ as f32 + 0.5) as i64;

/// Sprint pace (rl#355) — 1.8× walk, the common run multiplier. Feel knob.
pub const SPRINT_SPEED: i64 = PLAYER_SPEED * 9 / 5;

/// On-foot gravity, grid units per tick² — folded from the ONE gravity the crafts fall
/// under, so a plane-exit handoff (rl#355) keeps falling at the rate the cockpit
/// taught the eye; a second 9.81 here would let a rapier retune silently change the
/// ballistic arc's slope at the switch.
const GRAVITY_PER_TICK2: i64 = (-crab_world::physics::PHYSICS_GRAVITY.y as f64 * UNIT as f64
    / (TICK_HZ * TICK_HZ) as f64) as i64;

/// The jump as a designer states it (rl#367): a full hold peaks this high, this many
/// ticks after liftoff; the fall is the ONE shared gravity, ~2× the derived rise — the
/// slow-up/fast-down asymmetry needs no third constant. Feel knobs, both.
const JUMP_HEIGHT: i64 = player_heights(5.0);
const JUMP_APEX_TICKS: i64 = 10;

const JUMP_RISE_GRAVITY: i64 = 2 * JUMP_HEIGHT / (JUMP_APEX_TICKS * JUMP_APEX_TICKS);

/// The +g/2 is the semi-implicit-Euler correction: the integrator sums velocities
/// AFTER each gravity step, so without it the discrete arc peaks at h·(1 − 1/t).
const JUMP_SPEED: i64 = 2 * JUMP_HEIGHT / JUMP_APEX_TICKS + JUMP_RISE_GRAVITY / 2;

/// Apex hang. Feel knob.
const JUMP_HANG_SPEED: i64 = JUMP_SPEED / 4;

const COYOTE_TICKS: u8 = 4;

const JUMP_BUFFER_TICKS: u8 = 4;

/// A slide sustains only above this pace — 1.5× walk, between the √2×-walk a diagonal
/// step reaches (the axes are direct-drive, not normalized) and sprint, so neither
/// plain nor diagonal walking can fake a slide but any sprint (or carried landing
/// momentum) can. Below it the skid ends and the axes take over again.
const SLIDE_MIN_SPEED: i64 = PLAYER_SPEED * 3 / 2;

/// Slide friction per tick, as a 63/64 decay: a boosted entry (2.25×, see
/// [`SLIDE_BOOST_NUM`]) reaches the [`SLIDE_MIN_SPEED`] cutoff (1.5×) in ~26 ticks
/// ≈ 0.85 s of skid — and a plane-speed landing rides it out far longer. Feel knob.
const SLIDE_KEEP_NUM: i64 = 63;
const SLIDE_KEEP_DEN: i64 = 64;

/// Slide entry boost, ×5/4: committing to a skid pays out a burst — sprint pace
/// (1.8×) jumps to 2.25× walk, then friction bleeds it back down. Without it a
/// slide was a pure slowdown next to just holding sprint (1.8→1.5 in 0.4 s):
/// strictly dominated and unobservable in play (rl#368). Edge-triggered on
/// [`Player::sliding`] and gated to entry from at-most-sprint pace, so a
/// plane-speed carried landing skids as before and a held slide through a jump
/// can't re-boost every touchdown into unbounded speed.
const SLIDE_BOOST_NUM: i64 = 5;
const SLIDE_BOOST_DEN: i64 = 4;

/// The tallest surface drop a grounded step absorbs, half a player height: a walk or
/// slide whose tick crosses a steeper drop-off goes AIRBORNE from the old height
/// (rl#355) instead of snapping down the face — one vertical regime with the jump, and
/// a plane-speed slide launches off a crest instead of hugging it.
const STEP_DOWN_MAX: i64 = PLAYER_HEIGHT_FP / 2;

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
/// her stature ([`CRAB_STATURE`]) so the pursuit/grace tests drive her at the pace
/// the instrument last measured. Feeds the test driver ONLY: spawn clearance is a
/// taste constant that deliberately does not track this pin (rl#397).
#[cfg(test)]
const CRAB_CHARGE_SPEED_PER_S: i64 =
    (crab_world::eval::CRAB_CHARGE_SPEED_HEIGHTS_PER_S * CRAB_STATURE * UNIT as f32) as i64;

/// Spawn clearance from the crab's sim pos, round-start and joiners alike (rl#247) —
/// the spawn-safety feel knob, a FIXED distance (rl#397). It deliberately does NOT
/// derive from the measured chase-speed pin: an instrument re-pin must never move
/// where players spawn or how safe a spawn feels (the rl#344 re-pin silently grew
/// this 1.69×). 7.98 m is the clearance the game was tuned at — ~5 s of charge at
/// the then-current gait (rl#257), time to orient and run. Tune on playtest, never
/// by derivation. Far outside her carapace footprint (corner reach ~0.51 m), so no
/// spawn lands inside her claw shell; `spawn_clearance_matches_crab_body`
/// cross-checks that floor against every rig.
const MIN_CRAB_SPAWN_DISTANCE: i64 = (7.98 * UNIT as f64) as i64;

/// [`MIN_CRAB_SPAWN_DISTANCE`] in world meters — the one conversion, so the rl#322
/// craft-park ring and the tests measure the same clearance the sim enforces.
pub const MIN_CRAB_SPAWN_DISTANCE_M: f32 = MIN_CRAB_SPAWN_DISTANCE as f32 / UNIT as f32;

/// Spacing between player spawn slots along the z=0 spawn line.
const SPAWN_SLOT_PITCH: i64 = player_heights(2.0 / 1.8);

/// Reach-the-objective radius.
pub const EXTRACT_RADIUS: i64 = player_heights(2.0 / 1.8);

pub const MAX_YAW_TURNS_PER_TICK: i32 = trig::TURN / 24;

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

    /// Inverse of [`Pos::to_meters`]: meters onto the fixed-point grid (truncating,
    /// not rounding).
    pub fn from_meters(x_m: f32, z_m: f32) -> Self {
        Pos {
            x: meters_to_grid(x_m),
            z: meters_to_grid(z_m),
        }
    }

    /// This position relative to `origin`, in meters — THE render-frame conversion
    /// (rl#354). Subtract on the i64 grid FIRST, then convert: the difference is
    /// small near the origin, so the f32 keeps the grid's full 10 µm resolution.
    /// Converting each side to f32 first quantizes at ~0.5–1.3 mm out at the
    /// rl#305 locales — a large fraction of the 0.051 m player's per-tick walking
    /// step, rendered as nearby-ground judder.
    pub fn rel_meters(self, origin: Pos) -> (f32, f32) {
        Pos {
            x: self.x - origin.x,
            z: self.z - origin.z,
        }
        .to_meters()
    }

    /// Absolute meters at f64 — for terrain sampling
    /// ([`crab_world::terrain::TerrainGrid::height_f64`]), where f32's ~1 mm
    /// quantization at the far locales is the rl#354 stair-step.
    pub fn to_meters_f64(self) -> (f64, f64) {
        (self.x as f64 / UNIT as f64, self.z as f64 / UNIT as f64)
    }
}

/// Scalar leg of [`Pos::from_meters`], for the heights that ride beside a `Pos`
/// (a claw capsule's y — [`ClawPose`]).
pub fn meters_to_grid(m: f32) -> i64 {
    (m * UNIT as f32) as i64
}

/// [`meters_to_grid`] at f64 — for heights that ride the f64 terrain sampler (rl#355).
pub fn meters_to_grid_f64(m: f64) -> i64 {
    (m * UNIT as f64) as i64
}

/// Meters-per-second onto the per-tick velocity grid — THE m/s → sim-velocity rule
/// (rl#355), stated once so the craft-velocity bridge cannot drift from the sim's own
/// speed constants.
pub fn mps_to_grid_per_tick(mps: f64) -> i64 {
    (mps * UNIT as f64 / TICK_HZ as f64) as i64
}

/// Terrain surface height under a sim point, grid units — the airborne walker's ground
/// reference (rl#355), and the ONE sampling route: the craft-altitude bridge
/// (`pilot_shadows`) reads it too, so the altitude handed off is measured against the
/// same surface the sim will land the walker on. Samples the committed GCR bake (the
/// one world every peer ships, already this sim's spawn-bounds source) at f64, the
/// same entry the render path uses.
pub fn ground_at(pos: Pos) -> i64 {
    let (x, z) = pos.to_meters_f64();
    meters_to_grid_f64(crab_world::terrain::TerrainGrid::gcr().height_f64(x, z))
}

/// A walker's carried velocity, grid units per tick (rl#355). `y` is vertical.
/// Airborne it IS the motion (ballistic — the plane-exit handoff sails on it); grounded
/// it is the record of the last step, so a jump or slide inherits walk/sprint pace, and
/// a slide decays it in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Vel {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player {
    pos: Pos,
    yaw: i32,
    status: PlayerStatus,
    /// Feet height above the terrain surface, grid units. 0 = grounded (glued to the
    /// surface, the pre-rl#355 invariant); > 0 = airborne, integrating under gravity.
    alt: i64,
    vel: Vel,
    /// Mid-skid (rl#368). State, not a derivation: the entry boost is
    /// edge-triggered on it (a held slide across a jump's touchdown must not
    /// re-boost — compounding boosts are an unbounded-speed exploit), and the
    /// render layer reads it for the first-person eye dip and avatar lean.
    sliding: bool,
    jump: JumpWindows,
}

/// The two jump-forgiveness windows (rl#367), ticks left in each: `coyote` opens on
/// every grounded tick so a JUMP shortly after the ground falls away still lifts
/// off; `buffer` opens on an airborne JUMP so a press shortly before touchdown fires
/// on the first grounded tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct JumpWindows {
    pub coyote: u8,
    pub buffer: u8,
}

impl Player {
    pub(crate) fn from_parts(
        pos: Pos,
        yaw: i32,
        status: PlayerStatus,
        alt: i64,
        vel: Vel,
        sliding: bool,
        jump: JumpWindows,
    ) -> Self {
        Self {
            pos,
            yaw,
            status,
            alt,
            vel,
            sliding,
            jump,
        }
    }

    /// A grounded, motionless walker — every spawn starts here.
    pub(crate) fn standing(pos: Pos, yaw: i32, status: PlayerStatus) -> Self {
        Self::from_parts(
            pos,
            yaw,
            status,
            0,
            Vel::default(),
            false,
            JumpWindows::default(),
        )
    }

    fn liftoff(&mut self) {
        self.vel.y = JUMP_SPEED;
        self.alt = self.alt.max(1);
        self.jump = JumpWindows::default();
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
    pub fn alt(self) -> i64 {
        self.alt
    }
    pub fn sliding(self) -> bool {
        self.sliding
    }
    pub fn vel(self) -> Vel {
        self.vel
    }
    pub fn jump(self) -> JumpWindows {
        self.jump
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crab {
    pos: Pos,
    yaw: i32,
}

/// One of Sally's claw colliders as of this tick, bridged into sim space — THE down
/// mechanism, alone (rl#236 held call): standing under her carapace is deliberately
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
    /// Whether this claw touches (within [`CLAW_DOWN_BUFFER`]) a player at `p` whose
    /// feet are `alt` over the surface: the player's vertical span must meet the
    /// capsule's reach-fattened height band, and their XZ point must lie within reach
    /// of the capsule's XZ segment. An airborne walker (rl#355) lifts its span with it,
    /// so sailing over a claw is safe passage — no separate exemption.
    fn downs(&self, p: Pos, alt: i64) -> bool {
        let reach = self.radius + CLAW_DOWN_BUFFER;
        let (lo, hi) = (
            self.a_y.min(self.b_y) - reach,
            self.a_y.max(self.b_y) + reach,
        );
        if hi < alt || lo > alt + PLAYER_HEIGHT_FP {
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
    /// Craft height above the terrain surface, grid units — may be briefly negative
    /// (a craft dipping through the sheet); the sim's adopt clamps. Mirrored into the
    /// walker every piloted tick so that stepping out mid-air resumes on foot AT the
    /// craft's altitude (rl#355).
    pub alt: i64,
    /// Craft velocity, grid units per tick — the momentum-handoff feed (rl#355): the
    /// walker's carried velocity mirrors it while piloting, so the tick the pilot
    /// steps out it sails ballistically with exactly the craft's last velocity.
    pub vel: Vel,
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
    pub fn crabs_only(crabs: &'a [CrabPose]) -> Self {
        const NO_PILOTS: &BTreeMap<PlayerId, PilotPose> = &BTreeMap::new();
        Self {
            crabs,
            pilots: NO_PILOTS,
        }
    }
}

/// Every crab held at its current sim pose, clawless — the test feed for steps where
/// the crabs are scenery (setup ticks, walker tests). Not `cfg(test)`: `net`'s tests
/// consume it too, and a downstream crate's test cfg can't see ours.
pub fn hold_poses(sim: &Sim) -> Vec<CrabPose> {
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
    /// alone only where a pose must land without stepping (tests placing crabs directly).
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
            Player::standing(
                self.nearest_clear_join_slot(x),
                self.spawn_frame.rot,
                PlayerStatus::Alive,
            ),
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
                // Facing local +z — the party opens looking at the objective,
                // whatever heading the frame drew.
                Player::standing(frame.place(x, 0), frame.rot, PlayerStatus::Alive),
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
                // The momentum handoff (rl#355): every piloted tick mirrors the
                // craft's altitude and velocity into the walker, so the tick the
                // pilot steps out (drops from this feed) the airborne integrator
                // above takes over from EXACTLY the craft's last state — no seam.
                p.alt = pp.alt.max(0);
                p.vel = pp.vel;
                p.jump = JumpWindows::default();
                if p.alt == 0 {
                    // Grounded ⇒ no vertical motion — the one invariant every other
                    // grounded site (touchdown, walk, slide) also maintains; a craft
                    // descending INTO the ground must not park a stale vy here.
                    p.vel.y = 0;
                }
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
                        && claw.downs(p.pos, p.alt)
                    {
                        p.status = PlayerStatus::Downed;
                    }
                }
            }
        }

        let ex = self.extraction.pos;
        for (id, p) in self.players.iter_mut() {
            // Grounded only (rl#355): extraction is stepping onto the pad, and a
            // walker sailing over it at plane speed must not scoop the win mid-air.
            if p.status == PlayerStatus::Alive
                && p.alt == 0
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

        if p.alt > 0 {
            // Releasing SLIDE mid-air re-arms the entry boost; holding it through a
            // jump keeps the skid armed so touchdown re-entry is boost-free.
            p.sliding &= inp.pressed(buttons::SLIDE);
            let jump = inp.pressed(buttons::JUMP);
            p.jump.buffer = if jump {
                JUMP_BUFFER_TICKS
            } else {
                p.jump.buffer.saturating_sub(1)
            };
            if jump && p.jump.coyote > 0 {
                p.liftoff();
            }
            p.jump.coyote = p.jump.coyote.saturating_sub(1);
            // Airborne (rl#355): ballistic — the carried velocity IS the motion, no
            // air control (the spec: sail with the craft's velocity and fall). The
            // altitude is terrain-relative, so integrate in absolute height and
            // re-express over the terrain under the new spot.
            let ground_before = ground_at(p.pos);
            p.pos.x += p.vel.x;
            p.pos.z += p.vel.z;
            // One carve-out from "no air control" (rl#367): holding JUMP shapes the
            // arc's height and hang — never steering. Released, and every airborne
            // state without JUMP (the whole plane-exit handoff, rl#355), falls under
            // the ONE shared gravity.
            p.vel.y -= if jump && p.vel.y > -JUMP_HANG_SPEED {
                JUMP_RISE_GRAVITY
            } else {
                GRAVITY_PER_TICK2
            };
            let abs_y = ground_before + p.alt + p.vel.y;
            p.alt = abs_y - ground_at(p.pos);
            if p.alt <= 0 {
                // Touchdown: glue back to the surface. The horizontal momentum
                // survives the landing tick so a held slide catches it next tick;
                // otherwise walking overwrites it with the axes.
                p.alt = 0;
                p.vel.y = 0;
            }
            return;
        }

        let ground_before = ground_at(p.pos);
        let speed2 = dist2_i128(p.vel.x, p.vel.z);
        let min = SLIDE_MIN_SPEED as i128;
        if inp.pressed(buttons::SLIDE) && speed2 > min * min {
            // Sliding (rl#355): a committed skid — the move axes are surrendered and
            // the carried momentum decays under friction until it drops to near
            // walking pace. Look stays live (handled above), steering does not.
            let sprint = SPRINT_SPEED as i128;
            if !p.sliding && speed2 <= sprint * sprint {
                // Entry burst (rl#368) — see [`SLIDE_BOOST_NUM`].
                p.vel.x = p.vel.x * SLIDE_BOOST_NUM / SLIDE_BOOST_DEN;
                p.vel.z = p.vel.z * SLIDE_BOOST_NUM / SLIDE_BOOST_DEN;
            }
            p.sliding = true;
            p.vel.x = p.vel.x * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN;
            p.vel.z = p.vel.z * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN;
            p.vel.y = 0;
            p.pos.x += p.vel.x;
            p.pos.z += p.vel.z;
        } else {
            p.sliding = false;
            // Walking: direct-drive axes, sprint scales the pace. The step is recorded
            // as the carried velocity so a jump (or slide entry) inherits walk/sprint
            // speed.
            let (sin, cos) = trig::sin_cos(p.yaw);
            let strafe = inp.move_strafe as i64;
            let forward = inp.move_forward as i64;
            let vx = sin * forward + cos * strafe;
            let vz = cos * forward - sin * strafe;
            let denom = Input::AXIS_SCALE as i64 * trig::ONE as i64;
            let speed = if inp.pressed(buttons::SPRINT) {
                SPRINT_SPEED
            } else {
                PLAYER_SPEED
            };
            p.vel.x = vx * speed / denom;
            p.vel.z = vz * speed / denom;
            p.vel.y = 0;
            p.pos.x += p.vel.x;
            p.pos.z += p.vel.z;
        }
        // A grounded step absorbs terrain drops up to [`STEP_DOWN_MAX`]; past that the
        // ground fell away — go airborne from the old height instead of snapping down
        // the face, the same regime the jump lands through.
        let drop = ground_before - ground_at(p.pos);
        if drop > STEP_DOWN_MAX {
            p.alt = drop;
        }
        p.jump.coyote = COYOTE_TICKS;
        if inp.pressed(buttons::JUMP) || p.jump.buffer > 0 {
            // Liftoff — from walk, sprint AND slide alike (rl#355; a slide-jump keeps
            // the skid's decayed momentum). Holding jump re-fires on every touchdown
            // (auto-hop) — a feel choice, not an edge-detect omission.
            p.liftoff();
        }
    }

    // Live only via `ClientSim::reconcile_local_prediction` (render-only) outside
    // tests (rl#248).
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
            let Player {
                pos,
                yaw,
                status,
                alt,
                vel,
                sliding,
                jump: JumpWindows { coyote, buffer },
            } = player;
            h.write(&[id.0]);
            h.write(&pos_bytes(*pos));
            h.write(&yaw.to_le_bytes());
            h.write(&[status.tag()]);
            h.write(&alt.to_le_bytes());
            let Vel { x, y, z } = vel;
            h.write(&x.to_le_bytes());
            h.write(&y.to_le_bytes());
            h.write(&z.to_le_bytes());
            h.write(&[*sliding as u8, *coyote, *buffer]);
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
            // Discovered chord codes are HOST-render progression metadata (rl#398), not
            // sim state — the host driver stamps them; `ClientSim` stashes + re-stamps.
            discovered: std::collections::BTreeSet::new(),
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
            // Likewise render-side metadata (rl#398): stashed by `ClientSim`, read by
            // the combo map, never sim state.
            discovered: _,
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

    /// Step with player 0 riding the fed craft state — the rl#355 handoff driver.
    fn step_piloted(sim: &mut Sim, alt: i64, vel: Vel) {
        let poses = hold_poses(sim);
        let pilot = sim.player(PlayerId(0)).unwrap();
        let pilots = BTreeMap::from([(
            PlayerId(0),
            PilotPose {
                pos: pilot.pos(),
                yaw: pilot.yaw(),
                alt,
                vel,
            },
        )]);
        sim.step(
            &neutral_for(sim),
            Externals {
                crabs: &poses,
                pilots: &pilots,
            },
        );
    }

    #[test]
    fn sprint_outpaces_walk_and_slide_rides_the_momentum_out() {
        let mut sim = Sim::new(11, &players(2));
        let start0 = sim.player(PlayerId(0)).unwrap().pos();
        let start1 = sim.player(PlayerId(1)).unwrap().pos();
        for _ in 0..10 {
            let mut inputs = neutral_for(&sim);
            inputs.insert(PlayerId(0), Input::new(0.0, 1.0, 0.0, buttons::SPRINT));
            inputs.insert(PlayerId(1), Input::from_axes(0.0, 1.0));
            step_scenery(&mut sim, &inputs);
        }
        let d = |a: Pos, b: Pos| dist2_i128(b.x - a.x, b.z - a.z);
        let sprint_d = d(start0, sim.player(PlayerId(0)).unwrap().pos());
        let walk_d = d(start1, sim.player(PlayerId(1)).unwrap().pos());
        assert!(
            sprint_d > walk_d * 3 && sprint_d < walk_d * 4,
            "sprint is 1.8× walk, so 10 ticks cover 3.24× the squared distance \
             (got {sprint_d} vs {walk_d})"
        );

        // Neutral axes + SLIDE: the sprint momentum carries the skid, decaying, and
        // the walker keeps moving with NO axis input.
        let mid = sim.player(PlayerId(0)).unwrap();
        let first_step = {
            let mut inputs = neutral_for(&sim);
            inputs.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::SLIDE));
            step_scenery(&mut sim, &inputs);
            d(mid.pos(), sim.player(PlayerId(0)).unwrap().pos())
        };
        assert!(first_step > 0, "the slide keeps moving on carried momentum");
        // Held long enough, friction bleeds it below the cutoff and the skid ends:
        // with the axes still neutral the walker stops.
        for _ in 0..120 {
            let mut inputs = neutral_for(&sim);
            inputs.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::SLIDE));
            step_scenery(&mut sim, &inputs);
        }
        let end = sim.player(PlayerId(0)).unwrap().pos();
        let mut inputs = neutral_for(&sim);
        inputs.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::SLIDE));
        step_scenery(&mut sim, &inputs);
        assert_eq!(
            end,
            sim.player(PlayerId(0)).unwrap().pos(),
            "below the slide cutoff the skid is over — neutral axes hold still"
        );
    }

    #[test]
    fn slide_jump_lifts_off_with_the_skid_momentum() {
        let mut sim = Sim::new(23, &players(1));
        for _ in 0..5 {
            let mut inputs = neutral_for(&sim);
            inputs.insert(PlayerId(0), Input::new(0.0, 1.0, 0.0, buttons::SPRINT));
            step_scenery(&mut sim, &inputs);
        }
        // One sliding tick, then jump out of the skid: the liftoff carries the
        // DECAYED slide velocity, not a fresh axis step.
        let mut slide = neutral_for(&sim);
        slide.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::SLIDE));
        step_scenery(&mut sim, &slide);
        let skid = sim.player(PlayerId(0)).unwrap().vel();
        let mut slide_jump = neutral_for(&sim);
        slide_jump.insert(
            PlayerId(0),
            Input::new(0.0, 0.0, 0.0, buttons::SLIDE | buttons::JUMP),
        );
        step_scenery(&mut sim, &slide_jump);
        let p = sim.player(PlayerId(0)).unwrap();
        assert!(p.alt() > 0, "slide-jump lifts off");
        assert_eq!(p.vel().y, JUMP_SPEED);
        assert_eq!(
            (p.vel().x, p.vel().z),
            (
                skid.x * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN,
                skid.z * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN
            ),
            "the jump inherits the skid's decayed momentum"
        );
    }

    /// The rl#368 fix: entering a slide from sprint pace bursts PAST sprint (×5/4,
    /// then friction) — without it the skid was a pure slowdown next to holding
    /// sprint, unobservable in play. The burst is entry-edge only.
    #[test]
    fn slide_entry_bursts_past_sprint_then_decays_without_reboost() {
        let mut sim = Sim::new(11, &players(1));
        for _ in 0..10 {
            let mut inputs = neutral_for(&sim);
            inputs.insert(PlayerId(0), Input::new(0.0, 1.0, 0.0, buttons::SPRINT));
            step_scenery(&mut sim, &inputs);
        }
        let carried = sim.player(PlayerId(0)).unwrap().vel();

        let mut slide = neutral_for(&sim);
        slide.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::SLIDE));
        step_scenery(&mut sim, &slide);
        let entry = sim.player(PlayerId(0)).unwrap();
        assert!(entry.sliding(), "the skid state is set on entry");
        let boosted =
            |v: i64| v * SLIDE_BOOST_NUM / SLIDE_BOOST_DEN * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN;
        assert_eq!(
            (entry.vel().x, entry.vel().z),
            (boosted(carried.x), boosted(carried.z)),
            "entry tick boosts ×5/4 before the friction step"
        );
        assert!(
            dist2_i128(entry.vel().x, entry.vel().z) > dist2_i128(carried.x, carried.z),
            "the entry burst outruns sprint pace"
        );

        // Second held tick: friction only — the boost must not re-fire mid-skid.
        let skid = entry.vel();
        let mut slide = neutral_for(&sim);
        slide.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::SLIDE));
        step_scenery(&mut sim, &slide);
        let p = sim.player(PlayerId(0)).unwrap();
        assert_eq!(
            (p.vel().x, p.vel().z),
            (
                skid.x * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN,
                skid.z * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN
            ),
            "mid-skid ticks decay only"
        );
    }

    /// A held slide across a jump keeps the skid armed: touchdown re-entry is
    /// boost-free, or auto-hop + slide would compound ×5/4 per hop into unbounded
    /// speed.
    #[test]
    fn held_slide_through_a_jump_does_not_reboost_on_touchdown() {
        let mut sim = Sim::new(23, &players(1));
        for _ in 0..5 {
            let mut inputs = neutral_for(&sim);
            inputs.insert(PlayerId(0), Input::new(0.0, 1.0, 0.0, buttons::SPRINT));
            step_scenery(&mut sim, &inputs);
        }
        let slide_held = |sim: &Sim, jump: bool| {
            let mut inputs = neutral_for(sim);
            let btns = buttons::SLIDE | if jump { buttons::JUMP } else { 0 };
            inputs.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, btns));
            inputs
        };
        let inputs = slide_held(&sim, false);
        step_scenery(&mut sim, &inputs);
        let inputs = slide_held(&sim, true);
        step_scenery(&mut sim, &inputs);
        assert!(sim.player(PlayerId(0)).unwrap().alt() > 0, "lifted off");
        for _ in 0..300 {
            if sim.player(PlayerId(0)).unwrap().alt() == 0 {
                break;
            }
            let inputs = slide_held(&sim, false);
            step_scenery(&mut sim, &inputs);
        }
        let landed = sim.player(PlayerId(0)).unwrap();
        assert_eq!(landed.alt(), 0, "came back down");
        assert!(
            landed.sliding(),
            "the held slide stayed armed through the air"
        );
        let carried = landed.vel();
        let inputs = slide_held(&sim, false);
        step_scenery(&mut sim, &inputs);
        let p = sim.player(PlayerId(0)).unwrap();
        assert_eq!(
            (p.vel().x, p.vel().z),
            (
                carried.x * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN,
                carried.z * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN
            ),
            "touchdown re-entry under a held slide decays only — no second boost"
        );
    }

    #[test]
    fn jump_works_from_sprint_rises_and_lands() {
        let mut sim = Sim::new(13, &players(1));
        let mut jump = neutral_for(&sim);
        jump.insert(
            PlayerId(0),
            Input::new(0.0, 1.0, 0.0, buttons::SPRINT | buttons::JUMP),
        );
        step_scenery(&mut sim, &jump);
        assert!(
            sim.player(PlayerId(0)).unwrap().alt() > 0,
            "jump lifts off from the sprint state"
        );
        let sprint_step = {
            let p = sim.player(PlayerId(0)).unwrap();
            debug_assert_eq!(p.vel().y, JUMP_SPEED);
            p.vel()
        };
        let before = sim.player(PlayerId(0)).unwrap().pos();
        let mut peak = 0;
        let mut ticks = 0;
        while sim.player(PlayerId(0)).unwrap().alt() > 0 {
            // Airborne: axes are surrendered — feed hard reverse to prove it.
            let mut inputs = neutral_for(&sim);
            inputs.insert(PlayerId(0), Input::from_axes(0.0, -1.0));
            step_scenery(&mut sim, &inputs);
            peak = peak.max(sim.player(PlayerId(0)).unwrap().alt());
            ticks += 1;
            assert!(ticks < 300, "a jump must come back down");
        }
        let after = sim.player(PlayerId(0)).unwrap().pos();
        // Flat-ground tapped apex is ~2.6 player heights; the half-height floor
        // leaves slack for the terrain slope the sprint carries the arc across (alt
        // is surface-relative).
        assert!(
            peak > PLAYER_HEIGHT_FP / 2,
            "the apex clears half a player height (got {peak})"
        );
        assert_eq!(
            (after.x - before.x, after.z - before.z),
            (sprint_step.x * ticks, sprint_step.z * ticks),
            "airborne motion is the carried sprint step, immune to the axes"
        );
        assert_eq!(sim.player(PlayerId(0)).unwrap().vel().y, 0, "landed clean");
    }

    /// A standing jump's altitude per tick, tick 0 = liftoff, ending on touchdown.
    /// `held(tick)` decides whether JUMP is down on that tick.
    fn arc(held: impl Fn(u64) -> bool) -> Vec<i64> {
        let mut sim = Sim::new(13, &players(1));
        let mut jump = neutral_for(&sim);
        jump.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::JUMP));
        step_scenery(&mut sim, &jump);
        let mut alts = vec![sim.player(PlayerId(0)).unwrap().alt()];
        while *alts.last().unwrap() > 0 {
            let tick = alts.len() as u64;
            let mut inputs = neutral_for(&sim);
            let b = if held(tick) { buttons::JUMP } else { 0 };
            inputs.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, b));
            step_scenery(&mut sim, &inputs);
            alts.push(sim.player(PlayerId(0)).unwrap().alt());
            assert!(alts.len() < 300, "a jump must come back down");
        }
        alts
    }

    #[test]
    fn held_jump_peaks_at_the_configured_height_on_the_configured_tick() {
        let alts = arc(|_| true);
        let apex = JUMP_APEX_TICKS as usize;
        assert!(
            (alts[apex] - JUMP_HEIGHT).abs() <= JUMP_RISE_GRAVITY,
            "the designed height on the designed tick, within one gravity step (got {})",
            alts[apex]
        );
        assert_eq!(
            alts.iter().max(),
            Some(&alts[apex]),
            "nothing later climbs above the designed apex"
        );
        let fall = alts.len() - 1 - apex;
        assert!(
            fall < apex,
            "the shared gravity brings her down faster than the rise ({fall} < {apex} ticks)"
        );
        let rise: Vec<i64> = alts.windows(2).map(|w| w[1] - w[0]).collect();
        let hang_ticks = rise[apex..]
            .windows(2)
            .take_while(|w| w[0] - w[1] == JUMP_RISE_GRAVITY)
            .count();
        assert_eq!(
            hang_ticks, 3,
            "the rise gravity carries past the apex until the fall passes JUMP_HANG_SPEED"
        );
        assert_eq!(
            rise[apex + hang_ticks] - rise[apex + hang_ticks + 1],
            GRAVITY_PER_TICK2,
            "once the hang ends every fall tick pulls at the ONE shared gravity"
        );
    }

    #[test]
    fn tapped_jump_peaks_lower_under_the_shared_gravity() {
        let alts = arc(|_| false);
        let peak = *alts.iter().max().unwrap();
        assert!(
            peak > PLAYER_HEIGHT_FP && peak < JUMP_HEIGHT * 3 / 5,
            "a tap clears a body but reaches nowhere near the held apex (got {peak})"
        );
        let steps: Vec<i64> = alts.windows(2).map(|w| w[1] - w[0]).collect();
        for w in steps.windows(2).take(steps.len() - 2) {
            assert_eq!(
                w[0] - w[1],
                GRAVITY_PER_TICK2,
                "released: the shared gravity, every tick"
            );
        }
    }

    #[test]
    fn early_release_shortens_the_jump() {
        let full = *arc(|_| true).iter().max().unwrap();
        let short = *arc(|t| t < 4).iter().max().unwrap();
        let tap = *arc(|_| false).iter().max().unwrap();
        assert!(
            tap < short && short < full,
            "height follows the hold ({tap} < {short} < {full})"
        );
    }

    #[test]
    fn buffered_jump_fires_on_landing() {
        // Measured against the landing each press produces: a press near the apex
        // re-engages the hang and stretches the arc.
        let run = |press_at: u64| -> (u64, bool) {
            let mut sim = Sim::new(13, &players(1));
            let mut jump = neutral_for(&sim);
            jump.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::JUMP));
            step_scenery(&mut sim, &jump);
            let mut tick = 0;
            loop {
                tick += 1;
                let b = if tick == press_at { buttons::JUMP } else { 0 };
                let mut inputs = neutral_for(&sim);
                inputs.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, b));
                step_scenery(&mut sim, &inputs);
                if sim.player(PlayerId(0)).unwrap().alt() == 0 {
                    break;
                }
                assert!(tick < 300, "a jump must come back down");
            }
            let n = neutral_for(&sim);
            step_scenery(&mut sim, &n);
            (tick, sim.player(PlayerId(0)).unwrap().vel().y == JUMP_SPEED)
        };
        let window = JUMP_BUFFER_TICKS as u64;
        let (landing, _) = run(u64::MAX);
        let mut in_window = 0;
        for press_at in 1..=landing {
            let (landed, fired) = run(press_at);
            let buffered = press_at + window > landed;
            assert_eq!(
                fired, buffered,
                "press on {press_at}, touchdown {landed}: fires iff within {window} ticks"
            );
            in_window += buffered as u64;
        }
        assert!(
            in_window > 0 && in_window < landing,
            "the scan straddles the window edge ({in_window} of {landing} presses buffered)"
        );
    }

    #[test]
    fn coyote_time_lifts_off_after_the_ground_falls_away() {
        // A plane-speed skid over a drop scanned off the shipped terrain.
        let carried = meters_to_grid(20.0) / TICK_HZ as i64;
        let skid = carried * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN;
        let step_off = || {
            let mut sim = Sim::new(13, &players(1));
            let mut snap = sim.core_snapshot();
            let p = snap.players[&PlayerId(0)];
            let ledge = (0..20_000)
                .map(|k| Pos {
                    x: p.pos().x + k * skid,
                    z: p.pos().z,
                })
                .find(|&at| {
                    let next = Pos {
                        x: at.x + skid,
                        z: at.z,
                    };
                    ground_at(at) - ground_at(next) > STEP_DOWN_MAX
                })
                .expect("the shipped terrain has a drop-off within reach of the spawn");
            snap.players.insert(
                PlayerId(0),
                Player::from_parts(
                    ledge,
                    p.yaw(),
                    p.status(),
                    0,
                    Vel {
                        x: carried,
                        y: 0,
                        z: 0,
                    },
                    true,
                    JumpWindows::default(),
                ),
            );
            sim.apply_core_snapshot(snap);
            let mut slide = neutral_for(&sim);
            slide.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::SLIDE));
            step_scenery(&mut sim, &slide);
            let p = sim.player(PlayerId(0)).unwrap();
            assert!(p.alt() > 0, "the skid carries her off the drop");
            assert_eq!(p.vel().y, 0, "stepped off, not jumped");
            sim
        };
        let jump_after = |neutral_ticks: u8| -> Sim {
            let mut sim = step_off();
            for _ in 0..neutral_ticks {
                let n = neutral_for(&sim);
                step_scenery(&mut sim, &n);
            }
            let mut jump = neutral_for(&sim);
            jump.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::JUMP));
            step_scenery(&mut sim, &jump);
            sim
        };
        let late = jump_after(COYOTE_TICKS - 1);
        assert_eq!(
            late.player(PlayerId(0)).unwrap().vel().y,
            JUMP_SPEED - JUMP_RISE_GRAVITY,
            "the last coyote tick still lifts off (then integrates)"
        );
        let too_late = jump_after(COYOTE_TICKS);
        assert!(
            too_late.player(PlayerId(0)).unwrap().vel().y < 0,
            "one tick past the window there is no jump"
        );
    }

    #[test]
    fn jumping_off_a_ledge_keeps_the_drop_height() {
        let carried = meters_to_grid(20.0) / TICK_HZ as i64;
        let skid = carried * SLIDE_KEEP_NUM / SLIDE_KEEP_DEN;
        let mut sim = Sim::new(13, &players(1));
        let mut snap = sim.core_snapshot();
        let p = snap.players[&PlayerId(0)];
        let ledge = (0..20_000)
            .map(|k| Pos {
                x: p.pos().x + k * skid,
                z: p.pos().z,
            })
            .find(|&at| {
                let next = Pos {
                    x: at.x + skid,
                    z: at.z,
                };
                ground_at(at) - ground_at(next) > STEP_DOWN_MAX
            })
            .expect("the shipped terrain has a drop-off within reach of the spawn");
        snap.players.insert(
            PlayerId(0),
            Player::from_parts(
                ledge,
                p.yaw(),
                p.status(),
                0,
                Vel {
                    x: carried,
                    y: 0,
                    z: 0,
                },
                true,
                JumpWindows::default(),
            ),
        );
        sim.apply_core_snapshot(snap);
        let mut inputs = neutral_for(&sim);
        inputs.insert(
            PlayerId(0),
            Input::new(0.0, 0.0, 0.0, buttons::SLIDE | buttons::JUMP),
        );
        step_scenery(&mut sim, &inputs);
        let p = sim.player(PlayerId(0)).unwrap();
        assert_eq!(p.vel().y, JUMP_SPEED, "lifted off");
        assert!(
            p.alt() > STEP_DOWN_MAX,
            "…from the ledge, not from the bottom of the drop (alt {})",
            p.alt()
        );
    }

    #[test]
    fn stepping_out_of_a_craft_never_jumps() {
        // A craft parked on the ground mirrors alt 0 into its pilot for a few ticks
        // (a grounded walker re-arms coyote every tick), then lifts and the pilot
        // steps out holding JUMP: the exit is the craft's momentum, never a liftoff.
        let mut sim = Sim::new(17, &players(1));
        for _ in 0..3 {
            step_piloted(&mut sim, 0, Vel::default());
        }
        let carried = Vel {
            x: 2_000,
            y: 0,
            z: 1_000,
        };
        step_piloted(&mut sim, meters_to_grid(1.0), carried);
        let mut inputs = neutral_for(&sim);
        inputs.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::JUMP));
        let poses = hold_poses(&sim);
        sim.step(&inputs, Externals::crabs_only(&poses));
        let p = sim.player(PlayerId(0)).unwrap();
        assert!(p.vel().y < 0, "gravity, not a jump (vy {})", p.vel().y);
        assert_eq!((p.vel().x, p.vel().z), (carried.x, carried.z));
    }
    #[test]
    fn plane_exit_hands_off_the_craft_momentum() {
        let mut sim = Sim::new(17, &players(1));
        let carried = Vel {
            x: 2_000,
            y: 0,
            z: 1_000,
        };
        let alt = meters_to_grid(1.0);
        for _ in 0..3 {
            step_piloted(&mut sim, alt, carried);
        }
        let exit = sim.player(PlayerId(0)).unwrap();
        assert_eq!(
            exit.alt(),
            alt,
            "the piloted walker mirrors the craft altitude"
        );
        assert_eq!(exit.vel(), carried, "…and the craft velocity");

        // The craft is gone (no pilots fed): the walker sails ballistically with the
        // carried velocity — the axes are dead weight — and falls under gravity.
        let mut inputs = neutral_for(&sim);
        inputs.insert(PlayerId(0), Input::from_axes(1.0, 1.0));
        let poses = hold_poses(&sim);
        sim.step(&inputs, Externals::crabs_only(&poses));
        let after = sim.player(PlayerId(0)).unwrap();
        assert_eq!(
            (after.pos().x - exit.pos().x, after.pos().z - exit.pos().z),
            (carried.x, carried.z),
            "first free tick advances by exactly the craft's velocity"
        );
        assert_eq!(
            after.vel().y,
            -GRAVITY_PER_TICK2,
            "gravity starts pulling the moment the craft is gone"
        );
        let mut ticks = 0;
        while sim.player(PlayerId(0)).unwrap().alt() > 0 {
            let n = neutral_for(&sim);
            step_scenery(&mut sim, &n);
            ticks += 1;
            assert!(ticks < 600, "the handoff must eventually land");
        }
        assert!(ticks > 3, "a 1 m drop is a real sail, not an instant snap");
    }

    #[test]
    fn airborne_walker_sails_over_the_claw_and_cannot_extract_midair() {
        let mut sim = Sim::new(19, &players(1));
        // Arm the round, then hoist the walker via a piloted tick and cut the feed.
        for _ in 0..(STARTUP_GRACE_TICKS + 1) {
            let n = neutral_for(&sim);
            step_scenery(&mut sim, &n);
        }
        let ex = sim.extraction().pos();
        let pilots = BTreeMap::from([(
            PlayerId(0),
            PilotPose {
                pos: ex,
                yaw: 0,
                alt: 20 * PLAYER_HEIGHT_FP,
                vel: Vel::default(),
            },
        )]);
        let poses = hold_poses(&sim);
        sim.step(
            &neutral_for(&sim),
            Externals {
                crabs: &poses,
                pilots: &pilots,
            },
        );
        // Airborne over the pad, claw sweeping the ground at the same spot, ACTION
        // held: neither the claw nor the extraction may connect.
        let mut action = neutral_for(&sim);
        action.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::ACTION));
        let p = sim.player(PlayerId(0)).unwrap();
        assert!(p.alt() > 2 * PLAYER_HEIGHT_FP, "the walker is airborne");
        let claws = held_with_claws(&sim, vec![claw_at(p.pos(), 0, CLAW_M)]);
        sim.step(&action, Externals::crabs_only(&claws));
        let p = sim.player(PlayerId(0)).unwrap();
        assert_eq!(
            p.status(),
            PlayerStatus::Alive,
            "a ground claw cannot reach a walker sailing far overhead"
        );
        // Let it land (no claws), then the same claw geometry downs a grounded walker.
        let mut ticks = 0;
        while sim.player(PlayerId(0)).unwrap().alt() > 0 {
            let n = neutral_for(&sim);
            step_scenery(&mut sim, &n);
            ticks += 1;
            assert!(ticks < 600, "the drop must land");
        }
        let p = sim.player(PlayerId(0)).unwrap();
        let claws = held_with_claws(&sim, vec![claw_at(p.pos(), 0, CLAW_M)]);
        sim.step(&neutral_for(&sim), Externals::crabs_only(&claws));
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "grounded, the identical claw connects — the exemption was the altitude"
        );
    }

    #[test]
    fn mixed_walking_and_idle_players_step_deterministically() {
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
            "the same mixed walking+idle inputs must reproduce the state hash"
        );
        assert_eq!(p1_start, p1_end, "a neutral-input player stays put");
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
            hash_after(&|s| s.players.get_mut(&foot).unwrap().jump.coyote += 1),
            h0,
            "player coyote window must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().jump.buffer += 1),
            h0,
            "player jump buffer must be hashed"
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

 /// THE rl#236 held call, pinned: standing under her carapace with no claw touching
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
                m.insert(
                    pid,
                    PilotPose {
                        pos: crab,
                        yaw: 7,
                        alt: 0,
                        vel: Vel::default(),
                    },
                );
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
    /// pinned to the one baked rig every process builds (rl#340 stage 10): spawns clear
    /// the carapace footprint's corner reach, so no spawn materializes under her body
    /// with her claws ([`ClawPose`] — the ONE down mechanism) already overhead.
    #[test]
    fn spawn_clearance_matches_crab_body() {
        use crab_world::bot::rig::{RestShape, baked_recipe, recipe_silhouette};

        let sil = recipe_silhouette(&baked_recipe());
        let RestShape::Cuboid { half, .. } = sil.carapace else {
            panic!("the carapace silhouette is a cuboid");
        };
        // One frame (rl#256): the rig IS world-scale. Pin the nominal stature to
        // the rig so the derived human constants stay honest.
        assert!(
            (sil.natural_height() - CRAB_STATURE).abs() / CRAB_STATURE < 0.01,
            "natural height {} strays >1% from CRAB_STATURE {CRAB_STATURE}",
            sil.natural_height()
        );
        let corner_m = half.x.hypot(half.z);
        assert!(
            MIN_CRAB_SPAWN_DISTANCE_M > corner_m,
            "spawn clearance must exceed the carapace's corner reach {corner_m:.2} m"
        );
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
