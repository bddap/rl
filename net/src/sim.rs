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
    /// bounded so one tick can't spin a peer arbitrarily. Flying, this axis is the
    /// stick's HORIZONTAL deflection → roll (ailerons); see [`step_plane`].
    pub look_yaw: i16,
    /// Pitch-look deflection for THIS tick, same fixed-point grid as [`look_yaw`].
    /// On foot the camera pitch is purely client-local (the sim has no foot pitch), so
    /// this is zero for walkers; it exists for the PLANE, where it is the stick's
    /// VERTICAL deflection → pitch (elevator). Carried on the wire alongside `look_yaw`
    /// so the one [`Input`] type drives both the (client-local) plane and a networked
    /// pilot through the single [`step_plane`] formula. Positive = nose up.
    pub look_pitch: i16,
    /// Action/button bitfield (see [`buttons`]). Held state, sampled each tick.
    pub buttons: u8,
}

impl Input {
    /// Fixed-point denominator: an axis value of `AXIS_SCALE` is full deflection
    /// (1.0). `move_strafe`, `move_forward`, and `look_yaw` all use this grid.
    pub const AXIS_SCALE: i16 = 1000;
    /// Wire size of one encoded [`Input`]: move_strafe(2) + move_forward(2) +
    /// look_yaw(2) + look_pitch(2) + buttons(1). Drives [`crate::transport`]'s frame
    /// size — keep in sync with [`Input::to_bytes`].
    pub const WIRE_LEN: usize = 9;

    /// Full constructor: analog `strafe`, `forward`, and `look_yaw` axes in
    /// `[-1.0, 1.0]` plus the raw button bitfield. `look_pitch` defaults to zero (the
    /// walker case — foot pitch is client-only); the plane sets it with
    /// [`with_look_pitch`](Input::with_look_pitch). Quantizes the analog values to the
    /// fixed-point grid at the input boundary (not in the sim), so the sim stays
    /// integer-only and the value that crosses the wire is exactly the value applied.
    pub fn new(strafe: f32, forward: f32, look_yaw: f32, buttons: u8) -> Self {
        let q = |v: f32| (v.clamp(-1.0, 1.0) * Self::AXIS_SCALE as f32).round() as i16;
        Self {
            move_strafe: q(strafe),
            move_forward: q(forward),
            look_yaw: q(look_yaw),
            look_pitch: 0,
            buttons,
        }
    }

    /// Set the analog `look_pitch` axis (`[-1.0, 1.0]`, positive = nose up) on an input,
    /// quantized like the other axes. Only the plane uses it (the elevator); foot input
    /// leaves it zero. Separate from [`new`](Input::new) so the many walker call sites
    /// stay 4-arg and only the flight path opts in.
    pub fn with_look_pitch(mut self, look_pitch: f32) -> Self {
        self.look_pitch = (look_pitch.clamp(-1.0, 1.0) * Self::AXIS_SCALE as f32).round() as i16;
        self
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
        b[6..8].copy_from_slice(&self.look_pitch.to_le_bytes());
        b[8] = self.buttons;
        b
    }

    /// Inverse of [`Input::to_bytes`].
    pub fn from_bytes(b: [u8; Self::WIRE_LEN]) -> Self {
        Self {
            move_strafe: i16::from_le_bytes([b[0], b[1]]),
            move_forward: i16::from_le_bytes([b[2], b[3]]),
            look_yaw: i16::from_le_bytes([b[4], b[5]]),
            look_pitch: i16::from_le_bytes([b[6], b[7]]),
            buttons: b[8],
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

/// TEST-FIXTURE crab speed, in [`UNIT`]/tick (rl#114). The production crab is driven entirely by
/// the rapier-NN body, so its real speed comes from physics, not a constant. This value only
/// powers the deterministic test driver ([`drive_crab_toward_prey`]) that walks a stand-in crab so
/// the grab/extraction/outcome MECHANICS stay exercised. Kept below [`PLAYER_SPEED`] (166) so the
/// dodge test's faster player can still outrun and beat it.
#[cfg(test)]
const CRAB_SPEED: i64 = 130;

/// Deterministic stand-in for the production crab driver (rl#114), shared by every test in this
/// crate that needs a HUNTING crab (the sim mechanic tests + the `net` desync replay). Production
/// drives the crab from the rapier-NN body via [`Sim::set_external_crab_pose`]; tests have no
/// rapier stack, so this walks a crab toward the nearest living prey with the SAME integer math
/// the real bridge's hunt target uses, then pushes the pose. `#[cfg(test)]` only — it can NEVER
/// stand in for the NN crab in a release build, so it is not a production fallback. ONE definition
/// (the manual's "one implementation per thing"). Call it each tick BEFORE [`Sim::step`]: it moves
/// the crab once `tick() >= STARTUP_GRACE_TICKS`, so the pose is in place for the first armed step
/// (the one that increments the tick PAST the grace and turns on grabs).
#[cfg(test)]
pub(crate) fn drive_crab_toward_prey(sim: &mut Sim) {
    if sim.tick() < STARTUP_GRACE_TICKS {
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
/// spin a peer wildly.
const MAX_YAW_TURNS_PER_TICK: i32 = trig::TURN / 24;

// --- Plane flight tuning ---
// Crude but PLANE-LIKE arcade flight (not an aerodynamics sim): forward AIRSPEED makes
// LIFT, lift holds you up, you BANK to turn, and you STALL (lift collapses, you sink)
// below a minimum speed. All integer fixed-point so two peers evolve it bit-identically.
// Forces/speeds are [`UNIT`]/tick(²); orientation rates are [`trig`] turn units/tick.
// Tuned at [`TICK_HZ`] = 30 and [`UNIT`] = 1000 (1 m), so a speed of 1000/tick = 30 m/s.

/// Forward thrust per tick at FULL throttle, in [`UNIT`]/tick² (along the facing). With
/// [`PLANE_DAMP`](PLANE_DAMP_NUM) the terminal forward speed is
/// `THRUST · DEN/(DEN−NUM)` ≈ 60 m/s — a brisk light aircraft.
const PLANE_THRUST: i64 = 40;

/// The throttle is a PERSISTENT lever in `0..`[`PLANE_THROTTLE_MAX`], like a real
/// throttle quadrant — not a momentary push. The pilot trims it up/down and it HOLDS
/// where set; thrust is `PLANE_THRUST · throttle / MAX`.
const PLANE_THROTTLE_MAX: i32 = 1000;

/// Throttle change per tick while the throttle-up / throttle-down control is held
/// (~0.6 s for a full idle↔max sweep at [`TICK_HZ`]).
const PLANE_THROTTLE_STEP: i32 = 56;

/// Downward pull per tick, in [`UNIT`]/tick² (subtracted from `vel.y`). Constant; LIFT
/// (which scales with airspeed) is what opposes it, so altitude follows from SPEED.
const PLANE_GRAVITY: i64 = 16;

/// Lift per unit of airspeed, as [`PLANE_LIFT_NUM`]/[`PLANE_LIFT_DEN`] of the speed
/// ([`UNIT`]/tick). The vertical lift is capped at [`PLANE_GRAVITY`] (a wing trims to
/// hold level, it doesn't balloon you upward on raw speed — you CLIMB by pitching up,
/// which vectors thrust skyward), so this ratio only sets WHERE lift first fully
/// supports the plane: `DEN · GRAVITY / NUM` = 800/tick = 24 m/s, the stall speed.
const PLANE_LIFT_NUM: i64 = 1;
const PLANE_LIFT_DEN: i64 = 50;

/// Stall speed in [`UNIT`]/tick (≈24 m/s): DERIVED as the airspeed at which the
/// airspeed-proportional lift first reaches gravity (`speed · LIFT_NUM/LIFT_DEN =
/// GRAVITY`), so it can never drift from the lift constants. BELOW it the wing stalls —
/// lift falls off with the SQUARE of the speed ratio, so it drops fast: the plane sinks,
/// accelerates downward, and recovers once flying speed returns.
const PLANE_STALL_SPEED: i64 = PLANE_LIFT_DEN * PLANE_GRAVITY / PLANE_LIFT_NUM;

/// Velocity retained per tick, as a fraction of [`PLANE_DAMP_DEN`] (drag). `< 1` so
/// speed bleeds off without thrust and the integrator can't run away to infinity —
/// terminal speed is `thrust · DEN/(DEN−NUM)`. Applied to all three axes.
const PLANE_DAMP_NUM: i64 = 98;
const PLANE_DAMP_DEN: i64 = 100;

/// Per-tick body-frame PITCH (elevator) rate at full stick, in [`trig`] turn units —
/// angular velocity about the craft's own right axis, integrated into its attitude
/// quaternion. Held input keeps pitching, so a sustained pull loops the plane; nothing
/// clamps it.
const PLANE_PITCH_RATE: i32 = trig::TURN / 120;

/// Per-tick body-frame ROLL (aileron) rate at full stick, in [`trig`] turn units. The brisk
/// one: banking is the PRIMARY way to turn (the banked wing's lift vector pulls the nose
/// round), so the ailerons respond quickly. Unbounded — a held aileron barrel-rolls.
const PLANE_ROLL_RATE: i32 = trig::TURN / 60;

/// Per-tick body-frame YAW (rudder) rate at full pedal, in [`trig`] turn units. Small — the
/// rudder is for fine yaw/coordination, not the main turn (that's the bank). About the
/// craft's OWN up axis, so it stays a rudder when the plane is banked or inverted.
const PLANE_RUDDER_RATE: i32 = trig::TURN / 240;

/// Plane spawn altitude in [`UNIT`] (metres): start airborne at a sensible cruise height.
const PLANE_SPAWN_ALTITUDE: i64 = 60 * UNIT;

/// Plane spawn forward speed in [`UNIT`]/tick (≈30 m/s): spawn at cruise so the wing is
/// already making lift and the pilot is FLYING from tick 0, not stalling off the spawn.
const PLANE_SPAWN_SPEED: i64 = 1000;

/// Throttle the plane spawns with (cruise setting; see [`PLANE_THROTTLE_MAX`]).
const PLANE_SPAWN_THROTTLE: i32 = 700;

// --- Helicopter flight tuning ---
// Arcade rotorcraft (not an aerodynamics sim): a persistent COLLECTIVE lever sets rotor
// lift straight out of the (tilted) rotor disc, so the craft HOVERS when lift balances
// gravity, CLIMBS/DESCENDS as the pilot trims collective, and TRANSLATES by tilting the
// disc with the cyclic — no airspeed is needed for lift, so it takes off and lands
// VERTICALLY (no runway). Yaw pedals (tail rotor) spin the heading at any speed, hover
// included. Integer fixed-point so two peers evolve it bit-identically; the same
// [`UNIT`]/tick(²) and [`trig`] turn-unit grids the plane uses.

/// Downward pull per tick, in [`UNIT`]/tick² — the same gravity the plane feels, so the
/// two vehicles fall alike.
const HELI_GRAVITY: i64 = PLANE_GRAVITY;

/// The collective is a PERSISTENT lever in `0..`[`HELI_COLLECTIVE_MAX`] (like the plane's
/// throttle quadrant): the pilot trims it and it HOLDS. Rotor lift is
/// `HELI_MAX_LIFT · collective / MAX`.
const HELI_COLLECTIVE_MAX: i32 = 1000;

/// Collective change per tick while the raise/lower control is held (~0.6 s for a full
/// idle↔max sweep at [`TICK_HZ`]) — matches the plane throttle's trim rate.
const HELI_COLLECTIVE_STEP: i32 = 56;

/// Rotor lift at FULL collective, in [`UNIT`]/tick² along the disc normal. Sized 2×
/// [`HELI_GRAVITY`] so HALF collective hovers (lift == gravity, disc level) and full
/// collective climbs hard — the hover trim is then DERIVED
/// ([`HELI_HOVER_COLLECTIVE`]), never a second literal to drift.
const HELI_MAX_LIFT: i64 = 2 * HELI_GRAVITY;

/// The collective at which a LEVEL rotor's lift exactly balances gravity — a steady hover.
/// DERIVED (`MAX_LIFT · c/MAX = GRAVITY`), so the hover trim can't drift from the lift
/// model. The craft SPAWNS here, so it hovers hands-off from tick 0.
const HELI_HOVER_COLLECTIVE: i32 =
    (HELI_GRAVITY * HELI_COLLECTIVE_MAX as i64 / HELI_MAX_LIFT) as i32;

/// Per-tick body-frame CYCLIC rate at full stick, in [`trig`] turn units — angular velocity
/// about the craft's own pitch/roll axes, integrated into its attitude quaternion. Tilting
/// the airframe leans the rotor's lift so the craft translates; held cyclic keeps rotating,
/// so a sustained input loops the helicopter — unbounded, no return-to-level.
const HELI_CYCLIC_RATE: i32 = trig::TURN / 90;

/// Per-tick body-frame YAW rate at full pedal (tail rotor), in [`trig`] turn units. Strong
/// authority — a helicopter pivots on the spot in a hover (≈150°/s at full pedal at
/// [`TICK_HZ`]). About the craft's OWN up axis, so the pedals stay a heading control even
/// when the airframe is pitched, banked, or inverted.
const HELI_YAW_RATE: i32 = trig::TURN / 72;

/// Velocity retained per tick (drag). Heavier than the plane's so the hover is stable and
/// the craft coasts to a stop when the cyclic centres — release it and it stops, it
/// doesn't glide. Terminal cruise ≈14 m/s at hover power, faster with more collective.
const HELI_DAMP_NUM: i64 = 98;
const HELI_DAMP_DEN: i64 = 100;

/// Helicopter spawn altitude, in [`UNIT`] (3 m): boots a few metres up, already hovering,
/// so the pilot sees it hold station from tick 0 (then climbs/descends/translates on input).
const HELI_SPAWN_ALTITUDE: i64 = 3 * UNIT;

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

/// The one giant crab: a ground-plane position with a facing yaw, driven from OUTSIDE the sim by
/// the real rapier-NN body (rl#114, via [`Sim::set_external_crab_pose`]) — there is no built-in
/// integer pursuit. Rendered [`CRAB_SCALE`]× a player; the sim models only its position and a
/// [`CRAB_GRAB_RADIUS`] reach (the limbs live in the NN body, not here).
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

/// A pilotable plane: a flying body with 3D position + velocity, an attitude quaternion,
/// and a persistent throttle — all integer fixed-point so two peers evolve it identically.
///
/// Plane-like arcade flight (see [`step_plane`]): forward airspeed makes lift along the
/// wing (the craft's up axis), lift holds altitude, BANKING turns (the banked wing's lift
/// vector pulls the nose round), and the wing stalls below [`PLANE_STALL_SPEED`]. Attitude
/// is a full quaternion integrated from body-frame angular rates, UNCLAMPED — the plane can
/// loop, roll, and fly inverted; there is no orientation bound and no auto-level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    pos: Pos3,
    vel: Pos3,
    /// Attitude (body→world rotation) as a [`trig::ONE`]-scaled unit quaternion. Integrated
    /// in the body frame each tick (see [`iquat::Quat::integrate`]); the nose is the body +Z
    /// axis, the wing-up the body +Y axis.
    orient: iquat::Quat,
    /// Throttle, a persistent lever in `0..`[`PLANE_THROTTLE_MAX`] (see there). Trimmed
    /// up/down by the pilot; thrust scales with it.
    throttle: i32,
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
    /// The attitude quaternion (body→world). The renderer reads it through [`CockpitPose`].
    pub fn orient(self) -> iquat::Quat {
        self.orient
    }
    /// Heading yaw in [`trig`] turn units — the nose projected to the ground plane. Derived
    /// from the attitude; used only to spawn the next craft facing the same way on a swap.
    pub fn heading(self) -> i32 {
        self.orient.yaw()
    }
    /// Throttle setting, `0..`[`PLANE_THROTTLE_MAX`].
    pub fn throttle(self) -> i32 {
        self.throttle
    }

    /// Spawn a plane airborne over a ground point, facing `heading` (turn units), already
    /// at cruise — flying forward at [`PLANE_SPAWN_SPEED`] with [`PLANE_SPAWN_THROTTLE`]
    /// set, wings level. The ONE spawn definition, shared by the deterministic pilot spawn
    /// ([`Sim::spawn_state`]) and the windowed client's single-player enter-vehicle toggle
    /// ([`crate::render`]) — so a plane boarded mid-round and a plane spawned at round start
    /// begin in the identical state, no second literal to drift. Spawning at speed (not at
    /// rest) means the wing makes lift immediately, so the pilot is flying, not stalling,
    /// from tick 0.
    pub fn spawn(ground: Pos, heading: i32) -> Self {
        // Initial velocity along the heading (level: no pitch component) at cruise speed.
        let (sh, ch) = trig::sin_cos(heading); // · ONE
        let vx = sh * PLANE_SPAWN_SPEED / trig::ONE as i64;
        let vz = ch * PLANE_SPAWN_SPEED / trig::ONE as i64;
        Self {
            pos: Pos3 {
                x: ground.x,
                y: PLANE_SPAWN_ALTITUDE,
                z: ground.z,
            },
            vel: Pos3 { x: vx, y: 0, z: vz },
            orient: iquat::Quat::from_yaw(heading),
            throttle: PLANE_SPAWN_THROTTLE,
        }
    }

    /// Advance this plane one tick under pilot `input` (crude arcade flight). The ONE flight
    /// integrator, shared by the deterministic sim step (pilots in [`Sim::step`]) and the
    /// windowed client's single-player vehicle ([`crate::render`]) so there is a single
    /// physics formula, no copy to drift. See [`step_plane`] for the control map and the
    /// integer-only determinism notes.
    pub fn step(&mut self, input: Input) {
        step_plane(self, input);
    }

    /// This plane's first-person cockpit pose (3D position + attitude). The renderer reads it
    /// through the shared [`CockpitPose`] so one camera formula serves every flyer (plane and
    /// helicopter) with no copy to drift.
    pub fn cockpit_pose(self) -> CockpitPose {
        CockpitPose {
            pos: self.pos,
            orient: self.orient,
        }
    }
}

/// A flyer's first-person cockpit pose: 3D position plus a full attitude quaternion (body→
/// world). The shared view both the [`Plane`] and the [`Helicopter`] hand the renderer, so
/// the one cockpit-camera formula (`crate::render`'s `cockpit_camera`) can fly either with no
/// copy to drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CockpitPose {
    pub pos: Pos3,
    pub orient: iquat::Quat,
}

/// A pilotable helicopter: a flying body with 3D position + velocity, an attitude quaternion,
/// and a persistent COLLECTIVE lever — integer fixed-point, like the [`Plane`], so two peers
/// evolve it bit-identically.
///
/// Arcade rotorcraft flight (see [`step_helicopter`]): rotor lift points out of the disc
/// (the craft's up axis), so at the hover collective a LEVEL craft just balances gravity and
/// HOVERS; trimming collective up/down climbs/descends; tilting the airframe with the cyclic
/// leans the lift vector so part of it pushes HORIZONTALLY — the craft translates and banks
/// into it. Yaw pedals spin the heading about the craft's own up axis (tail rotor) at any
/// speed, hover included. Attitude is a full quaternion integrated from body-frame rates,
/// UNCLAMPED and never auto-levelled — held cyclic loops the craft and it can fly inverted.
/// Unlike the plane, lift needs no airspeed, so it takes off and lands vertically — no runway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Helicopter {
    pos: Pos3,
    vel: Pos3,
    /// Attitude (body→world rotation) as a [`trig::ONE`]-scaled unit quaternion — the rotor
    /// disc normal is the body +Y axis. Integrated in the body frame each tick.
    orient: iquat::Quat,
    /// Collective, a persistent lever in `0..`[`HELI_COLLECTIVE_MAX`]: sets rotor lift.
    collective: i32,
}

impl Helicopter {
    /// World position (3D — includes altitude Y).
    pub fn pos(self) -> Pos3 {
        self.pos
    }
    /// World velocity in [`UNIT`]/tick (3D).
    pub fn vel(self) -> Pos3 {
        self.vel
    }
    /// The attitude quaternion (body→world). The renderer reads it through [`CockpitPose`].
    pub fn orient(self) -> iquat::Quat {
        self.orient
    }
    /// Heading in [`trig`] turn units — the nose projected to the ground plane (derived).
    pub fn heading(self) -> i32 {
        self.orient.yaw()
    }
    /// Collective setting, `0..`[`HELI_COLLECTIVE_MAX`].
    pub fn collective(self) -> i32 {
        self.collective
    }

    /// Spawn a helicopter HOVERING over a ground point, facing `heading` (turn units): a few
    /// metres up ([`HELI_SPAWN_ALTITUDE`]), at rest, disc level, collective at the derived
    /// [`HELI_HOVER_COLLECTIVE`] — so lift balances gravity and it holds station hands-off
    /// from tick 0. The ONE spawn definition, used by the windowed client's enter-vehicle
    /// toggle ([`crate::render`]); no second literal to drift.
    pub fn spawn(ground: Pos, heading: i32) -> Self {
        Self {
            pos: Pos3 {
                x: ground.x,
                y: HELI_SPAWN_ALTITUDE,
                z: ground.z,
            },
            vel: Pos3::default(),
            orient: iquat::Quat::from_yaw(heading),
            collective: HELI_HOVER_COLLECTIVE,
        }
    }

    /// Advance this helicopter one tick under pilot `input` (arcade rotorcraft flight). The
    /// ONE heli integrator; see [`step_helicopter`] for the control map and determinism notes.
    pub fn step(&mut self, input: Input) {
        step_helicopter(self, input);
    }

    /// This helicopter's first-person cockpit pose — the shared [`CockpitPose`] the renderer
    /// flies, identical formula to the plane's.
    pub fn cockpit_pose(self) -> CockpitPose {
        CockpitPose {
            pos: self.pos,
            orient: self.orient,
        }
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
    /// The immutable round CONFIG — the match seed and the foot/pilot roster — kept so a
    /// deterministic restart ([`buttons::RESTART`]) can rebuild the initial state in
    /// place. NOT folded into [`Sim::state_hash`]: it can't differ between in-sync peers,
    /// and a peer built with a different roster is already a different game the cross-check
    /// surfaces via the player/plane state it does hash.
    config: RoundConfig,
    /// The peer-comparable digest of the REAL rapier crab's full physics state for this tick
    /// (every actuated body's pose + velocity bits — see [`crab_world::bot::physics_digest`]),
    /// supplied each tick by the deterministic driver alongside the pose via
    /// [`Sim::set_external_crab_pose`]. ALWAYS folded into [`Sim::state_hash`], so two peers whose
    /// float bodies — or whose policy weights, folded into this digest by the bridge — diverge
    /// desync on the tick it happens. The giant crab's ground position is ALWAYS driven from
    /// OUTSIDE the sim by the real NN crab body (rl#114): there is no built-in integer
    /// point-pursuer to fall back to, so a round that can't arm the NN crab REFUSES LOUDLY rather
    /// than silently substituting a fake crab. `0` until a pose is first pushed (a never-driven
    /// crab — e.g. a headless determinism-machinery sim — folds a constant `0`, still
    /// deterministic).
    ///
    /// CADENCE (rl#82, the GCR fold): this digest is sound cross-peer ONLY because the rapier
    /// crab now steps a deterministic, wall-clock-free number of physics steps per lockstep tick
    /// — `net::render::drive_lockstep` pumps the fixed schedule itself via
    /// [`crate::cadence::PhysicsCadence`] (64:30), pushing ONE pose+digest per APPLIED tick,
    /// so every peer folds the identical digest into the identical tick regardless of frame rate.
    /// (Before the fold, physics ran on Bevy's wall-clock `FixedUpdate`, so the networked arm had
    /// to stay gated off — different-framerate peers would fold the same digest into a different
    /// number of ticks.)
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
            config,
            external_crab_digest: 0,
        }
    }

    /// Drive the crab's ground position + facing yaw from outside the sim — the ONLY way the
    /// giant crab moves (rl#114: there is no built-in integer pursuit). The real rapier-NN crab
    /// body ([`crate::external_crab`]) calls this each tick BEFORE advancing, so the
    /// grab/extraction checks resolve against the body's current position. `pos`/`yaw` are genuine
    /// hashed state; `phys_digest` is the float body's per-tick digest (see
    /// [`external_crab_digest`](Sim::external_crab_digest)) the desync check folds in — the bridge
    /// computes it from the rapier bodies and the loaded policy weights, so any cross-peer
    /// divergence is caught. Seed it once at round setup with the crab's spawn pose, then push the
    /// body's pose each tick.
    pub fn set_external_crab_pose(&mut self, pos: Pos, yaw: i32, phys_digest: u64) {
        self.crab.pos = pos;
        self.crab.yaw = yaw;
        self.external_crab_digest = phys_digest;
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
        let (players, crab, extraction) = Self::spawn_state(&self.config);
        self.tick = 0;
        self.players = players;
        self.crab = crab;
        self.extraction = extraction;
        self.outcome = Outcome::Ongoing;
        self.rng = ChaCha8Rng::seed_from_u64(self.config.seed);
    }

    /// The tick-0 entity layout for a [`RoundConfig`] — the SINGLE source of the spawn
    /// arrangement, shared by [`Sim::new`] and [`Sim::reset`] so the two can't drift. Pure
    /// function of the (integer, sorted) config, so every peer composes the identical
    /// starting world. Returns the foot players, the crab, and the extraction point.
    fn spawn_state(
        cfg: &RoundConfig,
    ) -> (BTreeMap<PlayerId, Player>, Crab, ExtractionPoint) {
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
    /// supplied this tick's input. Panics naming a missing id rather than letting
    /// [`Sim::step`] default it to neutral and silently desync peers.
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
    /// `inputs` MUST hold an entry for every participant the sim tracks — each on-foot
    /// player. A missing input is a fail-loud determinism fault, NOT a
    /// silent neutral: defaulting a dropped input to neutral would let one peer apply
    /// real input where another applied none, diverging `state_hash` invisibly (the
    /// NN-crab pose rides on player state, which feeds the hash). The lockstep driver
    /// guarantees completeness by stalling a tick until every peer's input arrives, so
    /// reaching `step` short is a bug — we panic rather than fabricate input (rl#105).
    pub fn step(&mut self, inputs: &BTreeMap<PlayerId, Input>) {
        self.require_complete_inputs(inputs);
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
            // require_complete_inputs (step entry) guarantees an entry per participant.
            let inp = inputs[id];

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

        // The crab's grabs are disarmed during the startup grace (see [`STARTUP_GRACE_TICKS`]).
        let armed = self.tick > STARTUP_GRACE_TICKS;

        // 2) Crab MOVE: the giant crab's position is driven entirely from outside the sim by the
        //    real rapier-NN crab body (rl#114, via [`Sim::set_external_crab_pose`]) — there is no
        //    built-in integer point-pursuer here. A round that can't arm the NN crab REFUSES at
        //    build time (see [`crate::render`]) rather than reaching `step` with a fake crab,
        //    so by the time we step, the crab pose is whatever the body last pushed. The
        //    grab/extraction below read `self.crab.pos`.

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
            h.write(&[status_tag(*status)]);
        }
        let Crab { pos, yaw } = crab;
        write_pos(&mut h, *pos);
        h.write(&yaw.to_le_bytes());
        let ExtractionPoint { pos } = extraction;
        write_pos(&mut h, *pos);
        h.write(&[outcome_tag(*outcome)]);
        // The restart edge-latch gates whether next tick's RESTART press fires, so a
        // divergence in it would desync the restart.
        h.write(&[u8::from(*restart_held)]);
        // Hash the RNG stream position so a desync in random draws is caught even before
        // it manifests in an entity. Cloning and drawing one block reflects the
        // generator's position without disturbing the real stream.
        h.write(&rand::Rng::r#gen::<u64>(&mut rng.clone()).to_le_bytes());
        // The float NN crab's per-tick physics+weights digest — ALWAYS folded (rl#114: the crab
        // is always externally driven). It makes the articulated body (and the policy weights,
        // via the bridge) part of the desync check, not just the quantized 2D `crab.pos`/`yaw`
        // hashed above. A never-driven crab folds the constant `0` it was seeded with — still
        // deterministic across peers.
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

/// Advance one plane one tick from its pilot input — plane-like arcade flight (see the
/// [`Plane`] docs), pure integer fixed-point so two peers evolve it bit-identically.
///
/// Flight-sim control map (the [`Input`] axes, re-read for flight):
/// - [`Input::move_forward`] → THROTTLE: trims the persistent throttle lever up (+) /
///   down (−). The flight-stick / WASD throttle.
/// - [`Input::look_yaw`] → ROLL (ailerons): the stick's horizontal deflection rolls the
///   craft about its nose; banking is what TURNS the plane (the banked wing's lift vector
///   pulls the velocity round — there is no separate heading coupling).
/// - [`Input::look_pitch`] → PITCH (elevator): the stick's vertical deflection rotates about
///   the craft's right axis; nose up (+) climbs, down (−) dives. Held, it loops the plane.
/// - [`Input::move_strafe`] → RUDDER (yaw): a small rotation about the craft's own up axis
///   for coordination; NOT the main turn.
///
/// Every angular input is a BODY-frame angular RATE integrated into the attitude quaternion,
/// UNCLAMPED and never auto-levelled (see [`iquat::Quat::integrate`]) — so the plane holds
/// whatever attitude the pilot flies it to, loops, rolls, and flies inverted. Order per tick:
/// trim throttle, integrate the body-frame rates into the attitude, apply thrust along the
/// nose and lift along the wing less gravity, damp (drag), then integrate position (with a
/// ground clamp). The only "trig" is the integer CORDIC [`trig`]/[`iquat`] math, never a float.
fn step_plane(plane: &mut Plane, inp: Input) {
    let axis = Input::AXIS_SCALE as i64;
    let one = trig::ONE as i64;

    // 1) Throttle: a persistent lever, trimmed up/down by the forward axis and held
    //    between ticks. Sign of move_forward = trim direction; magnitude = trim rate.
    let trim = (inp.move_forward as i64 * PLANE_THROTTLE_STEP as i64 / axis) as i32;
    plane.throttle = (plane.throttle + trim).clamp(0, PLANE_THROTTLE_MAX);

    // 2) Body-frame angular rates this tick → integrate the attitude (unclamped, no
    //    auto-level). `look_yaw`/`move_strafe` are screen-reconciled (`gather_input` negates
    //    them, screen-right↔sim-X), so a screen-right roll → right-wing-down and screen-left
    //    rudder → yaw left. Body-axis signs: +X noses DOWN, +Y turns toward +X (right), +Z
    //    raises the right wing — so nose-up, right-bank, and yaw-right negate to match.
    let pitch_up = (inp.look_pitch as i64 * PLANE_PITCH_RATE as i64 / axis) as i32;
    let bank_right = (-(inp.look_yaw as i64) * PLANE_ROLL_RATE as i64 / axis) as i32;
    let yaw_right = (-(inp.move_strafe as i64) * PLANE_RUDDER_RATE as i64 / axis) as i32;
    plane.orient = plane.orient.integrate(-pitch_up, yaw_right, -bank_right);

    // 3) The craft's nose (forward) and wing-up axes in the world.
    let (fx, fy, fz) = plane.orient.forward();
    let (ux, uy, uz) = plane.orient.up();

    // 4) Thrust along the nose (body +Z), scaled by the throttle lever — descale the
    //    forward vector's ONE and the throttle's MAX so the accel lands in UNIT/tick².
    let thr = plane.throttle as i64;
    let tdenom = one * PLANE_THROTTLE_MAX as i64;
    plane.vel.x += fx * PLANE_THRUST * thr / tdenom;
    plane.vel.y += fy * PLANE_THRUST * thr / tdenom;
    plane.vel.z += fz * PLANE_THRUST * thr / tdenom;

    // 5) Lift along the WING (body +Y). Lift grows with AIRSPEED, capped at gravity so level
    //    cruise just HOLDS altitude (you climb by pitching the nose up, step 4); below
    //    PLANE_STALL_SPEED the wing stalls (lift ∝ speed²) so the nose drops and you must
    //    regain speed. Applied as a vector along body-up, so banking tilts the lift: its
    //    horizontal share TURNS the craft and its vertical share shrinks (a hard turn loses
    //    altitude unless you pull up), and inverted the lift points down — you fall unless
    //    you push. No separate bank-to-turn coupling; the turn is the lift vector itself.
    let speed = isqrt_i128(
        plane.vel.x as i128 * plane.vel.x as i128
            + plane.vel.y as i128 * plane.vel.y as i128
            + plane.vel.z as i128 * plane.vel.z as i128,
    ) as i64;
    let mut lift = speed * PLANE_LIFT_NUM / PLANE_LIFT_DEN;
    if speed < PLANE_STALL_SPEED {
        lift = lift * speed / PLANE_STALL_SPEED; // stalled: drops off with speed²
    }
    lift = lift.min(PLANE_GRAVITY); // a wing trims to level, it doesn't balloon upward
    plane.vel.x += lift * ux / one;
    plane.vel.y += lift * uy / one;
    plane.vel.z += lift * uz / one;
    plane.vel.y -= PLANE_GRAVITY;

    // 6) Damp (drag): bleed a fixed fraction of velocity so the integrator can't run
    //    away. Integer truncation toward zero is identical on every target.
    plane.vel.x = plane.vel.x * PLANE_DAMP_NUM / PLANE_DAMP_DEN;
    plane.vel.y = plane.vel.y * PLANE_DAMP_NUM / PLANE_DAMP_DEN;
    plane.vel.z = plane.vel.z * PLANE_DAMP_NUM / PLANE_DAMP_DEN;

    // 7) Integrate position. Don't sink through the ground: clamp Y≥0 and kill a downward
    //    velocity on contact, so a dive ends in a (crude) belly landing rather than
    //    falling forever below the world.
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

/// Advance one helicopter one tick from its pilot input — arcade rotorcraft flight (see the
/// [`Helicopter`] docs), pure integer fixed-point so two peers evolve it bit-identically.
///
/// Control map (the [`Input`] axes, re-read for the heli):
/// - [`Input::move_forward`] → COLLECTIVE: trims the persistent collective lever up (+) /
///   down (−). More collective → more rotor lift → climb; less → descend. The W/S lever.
/// - [`Input::look_pitch`] → CYCLIC fore/aft: pushing the stick forward (mouse-up, positive
///   `look_pitch`) noses the craft DOWN and flies FORWARD; pulling back flies backward.
/// - [`Input::look_yaw`] → CYCLIC lateral: rolls the craft to translate sideways. Screen-
///   reconciled like the plane's ailerons (`gather_input` negates it), so screen-right banks
///   right and translates right.
/// - [`Input::move_strafe`] → YAW PEDALS (tail rotor): spin about the craft's OWN up axis,
///   authority at any speed (hover included). Negated so screen-left A yaws left, like the
///   plane rudder.
///
/// Same body-frame, RATE-based, UNCLAMPED attitude model as the plane (the ONE
/// [`iquat::Quat::integrate`]) — the cyclic and pedals are angular velocities about the
/// craft's own axes, so a held cyclic loops the helicopter and the pedals stay a heading
/// control even inverted; nothing auto-levels back to a hover. Order per tick: trim
/// collective → integrate the body-frame rates into the attitude → rotor lift along the
/// craft's up axis (the disc normal) − gravity → damp (drag) → integrate position (ground
/// clamp = a vertical landing). The only "trig" is the integer CORDIC [`trig`]/[`iquat`]
/// math, never a float.
fn step_helicopter(heli: &mut Helicopter, inp: Input) {
    let axis = Input::AXIS_SCALE as i64;
    let one = trig::ONE as i64;

    // 1) Collective: a persistent lever trimmed up/down by the forward axis (W/S), held
    //    between ticks. Sign of move_forward = trim direction; magnitude = trim rate.
    let trim = (inp.move_forward as i64 * HELI_COLLECTIVE_STEP as i64 / axis) as i32;
    heli.collective = (heli.collective + trim).clamp(0, HELI_COLLECTIVE_MAX);

    // 2) Body-frame angular rates this tick → integrate the attitude (unclamped, no
    //    auto-level). Cyclic forward (positive look_pitch) noses DOWN (+X rotation) so the
    //    tilted lift flies the craft forward; the screen-reconciled look_yaw/move_strafe
    //    negate as in the plane so screen-right banks right and screen-left A yaws left. Same
    //    body-axis signs as the plane: +X noses down, +Y turns right, +Z raises the right
    //    wing — so right-bank and yaw-right negate to match.
    let nose_down = (inp.look_pitch as i64 * HELI_CYCLIC_RATE as i64 / axis) as i32;
    let bank_right = (-(inp.look_yaw as i64) * HELI_CYCLIC_RATE as i64 / axis) as i32;
    let yaw_right = (-(inp.move_strafe as i64) * HELI_YAW_RATE as i64 / axis) as i32;
    heli.orient = heli.orient.integrate(nose_down, yaw_right, -bank_right);

    // 3) Rotor lift along the craft's up axis (the disc normal, body +Y), scaled by the
    //    collective. At the hover lever a LEVEL craft's lift exactly balances gravity, so it
    //    holds station; tilting the airframe leans the lift so part of it TRANSLATES the
    //    craft; inverted, the lift points down and drives the craft into the ground.
    let (ux, uy, uz) = heli.orient.up();
    let lift_mag = HELI_MAX_LIFT * heli.collective as i64 / HELI_COLLECTIVE_MAX as i64; // UNIT/tick²
    heli.vel.x += lift_mag * ux / one;
    heli.vel.y += lift_mag * uy / one - HELI_GRAVITY;
    heli.vel.z += lift_mag * uz / one;

    // 4) Damp (drag): bleed a fixed fraction of velocity so the hover is stable and the
    //    craft coasts to a stop when the cyclic centres. Integer truncation toward zero is
    //    identical on every target.
    heli.vel.x = heli.vel.x * HELI_DAMP_NUM / HELI_DAMP_DEN;
    heli.vel.y = heli.vel.y * HELI_DAMP_NUM / HELI_DAMP_DEN;
    heli.vel.z = heli.vel.z * HELI_DAMP_NUM / HELI_DAMP_DEN;

    // 6) Integrate position. Don't sink through the ground: clamp Y≥0 and kill a downward
    //    velocity on contact — a descent ends in a (crude) vertical landing.
    heli.pos.x += heli.vel.x;
    heli.pos.y += heli.vel.y;
    heli.pos.z += heli.vel.z;
    if heli.pos.y < 0 {
        heli.pos.y = 0;
        if heli.vel.y < 0 {
            heli.vel.y = 0;
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

/// Deterministic integer trigonometry ([`trig`]) and its client-side float adapter
/// ([`trig_client`]) live in [`super::cordic`]; re-exported here so the sim and its
/// callers keep referring to them as `sim::trig` / `sim::trig_client`.
pub use super::cordic::{iquat, iquat_client, trig, trig_client};

/// Fold a [`Pos`] (both coordinates) into the state hash — one call per position so a hashed
/// entity can't accidentally fold X but forget Z. Destructured exhaustively so a new coordinate
/// forces a compile error here (the rl#70 guard, extended to `Pos`). A free function rather than
/// an `Fnv` method because the hasher is the shared [`crab_world::fnv::Fnv`], which can't name sim's
/// `Pos`.
fn write_pos(h: &mut Fnv, p: Pos) {
    let Pos { x, z } = p;
    h.write(&x.to_le_bytes());
    h.write(&z.to_le_bytes());
}

/// Fold a [`Pos3`] (all three coordinates) into the state hash — one call per 3D position so a
/// hashed flying entity can't fold X/Z but forget the altitude Y. Exhaustively destructured for
/// the same reason as [`write_pos`].
fn write_pos3(h: &mut Fnv, p: Pos3) {
    let Pos3 { x, y, z } = p;
    h.write(&x.to_le_bytes());
    h.write(&y.to_le_bytes());
    h.write(&z.to_le_bytes());
}

/// Fold an attitude [`iquat::Quat`] (all four components) into the state hash — one call per
/// flyer orientation so a hashed craft can't fold some components and forget others.
/// Exhaustively destructured for the same reason as [`write_pos`].
fn write_quat(h: &mut Fnv, q: iquat::Quat) {
    let iquat::Quat { w, x, y, z } = q;
    h.write(&w.to_le_bytes());
    h.write(&x.to_le_bytes());
    h.write(&y.to_le_bytes());
    h.write(&z.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn players(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    /// A complete neutral input map for every participant in `sim`. The lockstep
    /// boundary now REQUIRES an entry per participant (rl#105), so a test driving
    /// idle players spells out their neutral input instead of relying on a
    /// missing-key default.
    fn neutral_for(sim: &Sim) -> BTreeMap<PlayerId, Input> {
        sim.participant_ids().map(|id| (id, Input::default())).collect()
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
                look_pitch: 4567,
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

    /// Pins the PLANE PITCH SIGN the cockpit legend depends on: positive `look_pitch`
    /// noses UP (climb), negative noses DOWN (dive). The HUD's "Pitch up (climb)" /
    /// "Pitch down (dive)" labels ride this sign; if it flips, they'd lie — so this fails
    /// loud rather than mislead the pilot. The nose's vertical component (the attitude's
    /// world-space forward Y) is the directly-observable "nose up" signal.
    #[test]
    fn step_plane_pitch_sign_matches_the_cockpit_legend() {
        let mut climb = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        climb.step(Input::default().with_look_pitch(1.0)); // positive look_pitch
        assert!(
            climb.orient().forward().1 > 0,
            "positive look_pitch must nose UP (forward.y > 0)"
        );
        let mut dive = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        dive.step(Input::default().with_look_pitch(-1.0)); // negative look_pitch
        assert!(
            dive.orient().forward().1 < 0,
            "negative look_pitch must nose DOWN (forward.y < 0)"
        );
    }

    /// Roll (ailerons) responds to the horizontal stick axis and banks to TURN, and the
    /// SIGN matches the screen: `gather_input` makes screen-right a NEGATIVE `look_yaw`
    /// (the screen-X↔sim reconcile), and `step_plane` negates it so screen-right banks
    /// RIGHT (right wing down). A banked wing's lift vector points partly sideways, so the
    /// craft accelerates that way — the plane TURNS by banking, with no separate heading
    /// coupling. And there is NO auto-level: the bank HOLDS when the stick centres (the
    /// owner's "auto-righting can go" — the pilot holds whatever attitude they set).
    #[test]
    fn step_plane_banks_right_and_turns_with_no_autolevel() {
        let mut p = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        // Screen-right (the negative look_yaw `gather_input` emits) banks the craft right —
        // its up-vector tilts toward +X (right wing down).
        for _ in 0..20 {
            p.step(Input::new(0.0, 0.0, -1.0, 0)); // screen-right = negative look_yaw
        }
        let (ux, _uy, _uz) = p.orient().up();
        assert!(ux > 0, "screen-right must bank RIGHT (up tilts +X), got {ux}");
        // The banked lift turns the craft right: it gains rightward (+X) velocity.
        assert!(
            p.vel().x > 0,
            "a right bank must turn the craft right (+X velocity), got {}",
            p.vel().x
        );
        // Release the stick: the bank HOLDS (no auto-level). The up-vector stays tilted.
        let banked = p.orient().up().0;
        for _ in 0..200 {
            p.step(Input::default());
        }
        assert!(
            p.orient().up().0 > banked / 2,
            "the bank must HOLD with no input (no auto-level), was {banked} now {}",
            p.orient().up().0
        );
    }

    /// The rudder (`move_strafe`) yaws the heading directly, and screen-left (A, which
    /// reaches the sim as POSITIVE `move_strafe` after `gather_input`'s negation) yaws
    /// LEFT (heading decreases). Small effect — pinned for sign, not magnitude.
    #[test]
    fn step_plane_rudder_yaws_left_on_positive_strafe() {
        let mut p = Plane::spawn(Pos { x: 0, z: 0 }, trig::TURN / 4);
        let h0 = p.heading();
        p.step(Input::new(1.0, 0.0, 0.0, 0)); // positive move_strafe = A = yaw left
        // Spawned at TURN/4, a small yaw can't underflow zero, so the raw difference is
        // already signed correctly (no wrap into the top of [0, TURN)).
        let dh = p.heading() - h0;
        assert!(dh < 0, "positive move_strafe (A) must yaw LEFT, got {dh}");
    }

    /// Throttle is a PERSISTENT lever: holding the up-trim (positive `move_forward`)
    /// raises it and it HOLDS when released; the down-trim lowers it. Bounded to
    /// `0..=MAX`.
    #[test]
    fn step_plane_throttle_is_a_persistent_lever() {
        let mut p = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        let t0 = p.throttle();
        p.step(Input::from_axes(0.0, 1.0)); // up-trim
        let t1 = p.throttle();
        assert!(t1 > t0, "up-trim must raise throttle");
        p.step(Input::default()); // released — holds
        assert_eq!(p.throttle(), t1, "throttle holds when the trim is released");
        // Drive it to the rails both ways.
        for _ in 0..100 {
            p.step(Input::from_axes(0.0, 1.0));
        }
        assert_eq!(p.throttle(), PLANE_THROTTLE_MAX, "up-trim saturates at MAX");
        for _ in 0..100 {
            p.step(Input::from_axes(0.0, -1.0));
        }
        assert_eq!(p.throttle(), 0, "down-trim bottoms at 0");
    }

    /// THE plane-feel test: at cruise (spawn) speed with wings level the plane HOLDS
    /// altitude (lift ≈ gravity), pitching up CLIMBS, and chopping the throttle bleeds
    /// speed until the wing STALLS and the plane sinks. Behavioral, over many ticks —
    /// the closest a headless test gets to "does it fly like a plane".
    #[test]
    fn step_plane_flight_envelope_cruise_climb_and_stall() {
        // Cruise: level, spawn throttle, neutral stick. Altitude should stay ~level over
        // a second of flight (no rocketing up, no falling out of the sky).
        let mut cruise = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        let y0 = cruise.pos().y;
        for _ in 0..30 {
            cruise.step(Input::default());
        }
        let drift = (cruise.pos().y - y0).abs();
        assert!(
            drift < 4 * UNIT,
            "level cruise must roughly hold altitude, drifted {drift} units"
        );

        // Climb: hold nose-up. Altitude must rise meaningfully above the cruise case.
        let mut climb = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        for _ in 0..30 {
            climb.step(Input::default().with_look_pitch(1.0));
        }
        assert!(
            climb.pos().y > cruise.pos().y + 3 * UNIT,
            "pitching up must climb (got {} vs cruise {})",
            climb.pos().y,
            cruise.pos().y
        );

        // Stall: chop the throttle and hold level. Speed bleeds below stall speed and the
        // plane sinks — ending up well below where it cruised.
        let mut stall = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        for _ in 0..120 {
            stall.step(Input::from_axes(0.0, -1.0)); // throttle to idle, holding
        }
        let speed = isqrt_i128(
            stall.vel().x as i128 * stall.vel().x as i128
                + stall.vel().y as i128 * stall.vel().y as i128
                + stall.vel().z as i128 * stall.vel().z as i128,
        ) as i64;
        assert!(
            speed < PLANE_STALL_SPEED,
            "idle throttle must bleed below stall speed, got {speed}"
        );
        assert!(
            stall.pos().y < cruise.pos().y,
            "a stalled plane must sink below cruise altitude"
        );
    }

    /// The defining helicopter property: at the spawn (hover) collective with the disc level
    /// and no input, lift EXACTLY balances gravity, so the craft holds station — same
    /// altitude, zero velocity, indefinitely. Integer-exact (the hover collective is derived
    /// to make `lift == gravity`), so it can't slowly sink or balloon.
    #[test]
    fn helicopter_hovers_hands_off() {
        let mut heli = Helicopter::spawn(Pos { x: 0, z: 0 }, 0);
        let spawn_y = heli.pos().y;
        for _ in 0..120 {
            heli.step(Input::default());
        }
        assert_eq!(heli.pos().y, spawn_y, "hover must hold altitude exactly");
        assert_eq!(
            (heli.vel().x, heli.vel().y, heli.vel().z),
            (0, 0, 0),
            "a hands-off hover stays at rest"
        );
        assert_eq!(
            heli.orient(),
            Helicopter::spawn(Pos { x: 0, z: 0 }, 0).orient(),
            "the craft stays level (attitude unchanged with no input)"
        );
    }

    /// Collective is vertical control: raising it (W) CLIMBS, dropping it (S) DESCENDS — and
    /// a full descent ENDS ON THE GROUND (vertical landing, no runway), not falling through it.
    #[test]
    fn helicopter_collective_climbs_descends_and_lands() {
        let mut heli = Helicopter::spawn(Pos { x: 0, z: 0 }, 0);
        let y0 = heli.pos().y;
        for _ in 0..30 {
            heli.step(Input::from_axes(0.0, 1.0)); // raise collective, hold
        }
        assert!(
            heli.pos().y > y0 + 3 * UNIT,
            "more collective must climb (got {} vs spawn {y0})",
            heli.pos().y
        );

        // From the climb, chop the collective and hold: it descends and settles ON the ground.
        for _ in 0..240 {
            heli.step(Input::from_axes(0.0, -1.0)); // collective to idle, holding
        }
        assert_eq!(heli.pos().y, 0, "idle collective lands it on the ground");
        assert_eq!(heli.vel().y, 0, "a landing kills the downward velocity");
    }

    /// Cyclic forward translates the craft FORWARD (push the stick = positive `look_pitch`)
    /// and noses the airframe DOWN to do it — tilt-to-move. Mostly +Z (its facing) with little
    /// sideways drift. And there is NO auto-level: the nose-down attitude HOLDS when the stick
    /// centres (rate-based, the owner's "auto-righting can go").
    #[test]
    fn helicopter_cyclic_pitches_down_to_fly_forward() {
        let mut heli = Helicopter::spawn(Pos { x: 0, z: 0 }, 0);
        // Pulse the cyclic forward to nose the craft down a modest amount (the rates are
        // unclamped now, so holding it would loop the craft — a pilot pulses to a tilt).
        for _ in 0..6 {
            heli.step(Input::default().with_look_pitch(1.0)); // push cyclic forward
        }
        assert!(
            heli.orient().forward().1 < 0,
            "flying forward noses the craft DOWN (forward.y < 0)"
        );
        // Release: the nose-down attitude HOLDS (no auto-level) and the tilted lift flies the
        // craft forward over the next couple of seconds.
        let nose_y = heli.orient().forward().1;
        for _ in 0..80 {
            heli.step(Input::default());
        }
        assert_eq!(
            heli.orient().forward().1,
            nose_y,
            "releasing the cyclic must NOT auto-level (attitude holds)"
        );
        assert!(
            heli.pos().z > 3 * UNIT,
            "cyclic forward must translate +Z (forward), got z={}",
            heli.pos().z
        );
        assert!(
            heli.pos().x.abs() < heli.pos().z / 4,
            "forward cyclic moves mostly forward, not sideways"
        );
    }

    /// Cyclic right banks the airframe right (right wing down → its up-vector tilts toward
    /// +X) and translates the craft to its RIGHT (+X). Screen-right reaches the sim as
    /// NEGATIVE `look_yaw` (the same reconcile the plane's ailerons use), so that's a right
    /// input here.
    #[test]
    fn helicopter_banks_right_to_translate_right() {
        let mut heli = Helicopter::spawn(Pos { x: 0, z: 0 }, 0);
        // Pulse a right bank to a modest tilt, then release (attitude holds) and coast.
        for _ in 0..6 {
            heli.step(Input::new(0.0, 0.0, -1.0, 0)); // screen-right = negative look_yaw
        }
        assert!(
            heli.orient().up().0 > 0,
            "banking right tilts the up-vector toward +X (right wing down)"
        );
        for _ in 0..80 {
            heli.step(Input::default());
        }
        assert!(
            heli.pos().x > 3 * UNIT,
            "a right bank must translate +X (right), got x={}",
            heli.pos().x
        );
    }

    /// Yaw pedals (the `move_strafe` axis, A/D) spin the heading even from a stationary
    /// hover — A reaches the sim as POSITIVE `move_strafe` (after `gather_input`'s negation)
    /// and must yaw LEFT, matching the plane's rudder sign.
    #[test]
    fn helicopter_pedals_yaw_in_a_hover() {
        // Spawn facing a quarter-turn so a small left yaw stays a plain decrease (no wrap
        // through 0 to confuse the sign check).
        let h0 = trig::TURN / 4;
        let mut heli = Helicopter::spawn(Pos { x: 0, z: 0 }, h0);
        for _ in 0..3 {
            heli.step(Input::new(1.0, 0.0, 0.0, 0)); // positive move_strafe = A = yaw left
        }
        assert!(
            heli.heading() < h0,
            "positive move_strafe (A) must yaw LEFT (heading decrease), got {} vs {h0}",
            heli.heading()
        );
    }

    /// The owner's #1 ask: BOTH craft can fly fully UPSIDE-DOWN — the world-space caps are
    /// gone, so a sustained pull loops the craft past vertical and inverts it, with the
    /// attitude staying a clean unit quaternion (no gimbal blowup / NaN — it's integer). A
    /// craft is inverted when its up-vector points DOWN (world up.y < 0).
    #[test]
    fn both_craft_can_fly_inverted_without_blowup() {
        // Plane: hold the nose up; within a loop it passes through inverted.
        let mut plane = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        let mut plane_inverted = false;
        for _ in 0..240 {
            plane.step(Input::default().with_look_pitch(1.0));
            if plane.orient().up().1 < 0 {
                plane_inverted = true;
            }
            assert_quat_unit(plane.orient());
        }
        assert!(plane_inverted, "the plane must be able to fly upside-down");

        // Helicopter: hold forward cyclic; it loops the airframe over the top, inverting it.
        let mut heli = Helicopter::spawn(Pos { x: 0, z: 0 }, 0);
        let mut heli_inverted = false;
        for _ in 0..240 {
            heli.step(Input::default().with_look_pitch(1.0));
            if heli.orient().up().1 < 0 {
                heli_inverted = true;
            }
            assert_quat_unit(heli.orient());
        }
        assert!(heli_inverted, "the helicopter must be able to fly upside-down");
    }

    /// The attitude quaternion stays UNIT (magnitude ≈ ONE) through thousands of ticks of
    /// hard, mixed control — the renormalize-every-tick keeps integer rounding from letting
    /// it drift or blow up. This is the "no NaN / no gimbal" guarantee in integer form.
    #[test]
    fn attitude_stays_unit_under_sustained_input() {
        let mut plane = Plane::spawn(Pos { x: 0, z: 0 }, 0);
        for i in 0..3000 {
            // Mixed pitch/roll/yaw so every body axis is exercised together.
            let inp = Input::new(1.0, 0.0, if i % 2 == 0 { 1.0 } else { -1.0 }, 0)
                .with_look_pitch(1.0);
            plane.step(inp);
            assert_quat_unit(plane.orient());
        }
    }

    /// A craft's attitude must be a unit quaternion to ±1 in the [`trig::ONE`] grid — the
    /// renormalize floors the magnitude near ONE, never NaN, never runaway.
    fn assert_quat_unit(q: iquat::Quat) {
        let one = trig::ONE as i64;
        let n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
        let mag = isqrt_i128(n2 as i128) as i64;
        assert!(
            (mag - one).abs() <= 2,
            "attitude must stay unit (mag {mag} vs ONE {one}) for {q:?}"
        );
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
        // `set_external_crab_pose` is the ONLY way the crab moves (rl#114): it seeds/updates the
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
        // rl#105: a tick stepped without EVERY participant's input must fail loud, not
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
        // Runtime half of the rl#70 guard. The COMPILE-TIME half lives in `state_hash`
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

        // The crab POSE (driven externally by the NN body, rl#114) is hashed so the quantized
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
        // `external_crab_digest` is ALWAYS folded into the hash now (rl#114: the crab is always
        // externally driven — no integer-crab gate to fold it out), so perturbing it MUST move the
        // hash. This is the desync teeth for the float body / weights mismatch (rl#82).
        assert_ne!(
            hash_after(&|s| s.external_crab_digest ^= 0xdead_beef),
            h0,
            "external_crab_digest must be hashed (always folded since rl#114)"
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
        let neutral = neutral_for(&sim);
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            drive_crab_toward_prey(&mut sim);
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
        sim.step(&neutral_for(&sim)); // release (tick 3)
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
