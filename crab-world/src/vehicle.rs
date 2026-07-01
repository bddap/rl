//! The player's single-player VEHICLE — a rapier rigidbody living in the crab's ONE
//! `bevy_rapier3d` world, so it is official host-authoritative game state that COLLIDES with
//! the trained crab "Sally" (owner 702/703/704). It replaces the old integer fixed-point flight
//! integrator (`net::sim`'s deleted `step_plane`/`step_helicopter`): ONE flight model, configured
//! as a plane or an Outer-Wilds-style ship by a [`VehicleParams`] set.
//!
//! ## Two craft, two feels, one force system (owner, botq#554)
//! - **Plane = Ace Combat 6.** A persistent THROTTLE LEVER drives thrust along the nose (+Z);
//!   wings make Bernoulli LIFT from forward airspeed; pitch/roll/yaw are control torques. The pilot
//!   turns by banking. (The AC6 input feel — intuitive pitch, left-stick fly, trigger throttle,
//!   bumper rudder — lives in the input→[`VehicleControl`] bridge, `net`'s `drive_lockstep`.)
//! - **Ship = Outer Wilds.** No wings (no lift) and no throttle lever — instead DIRECT body-frame
//!   thrusters on all three axes (strafe X / vertical Y / forward Z) that you tap and then COAST on
//!   (low drag = momentum). A held MATCH-VELOCITY brake bleeds relative velocity to rest (the cheap
//!   arena analog of OW's match-velocity-to-zero). Right-stick aims (pitch+yaw), bumpers roll.
//!
//! ## Force model (all UNCLAMPED — owner 702: no rotation caps, no auto-right by construction)
//! Each fixed step, before the rapier solve, [`apply_vehicle_forces`] reads the player's control
//! intents ([`VehicleControl`]) + the body's own pose/velocity and writes one [`ExternalForce`]
//! (overwritten, never accumulated):
//! - **thrust** = a LEVER term along the body thrust axis (plane nose; zero for the ship) PLUS a
//!   DIRECT body-frame term `rot · (thrust ⊙ direct_thrust)` (ship strafe/vertical/forward; zero
//!   for the plane). Both always summed — the unused term vanishes by its zeroed params, so there
//!   is no per-craft branch.
//! - **gravity** — rapier applies it to the dynamic body (no manual term).
//! - **drag** opposing velocity: linear `−k·v` plus quadratic `−k₂·|v|·v` (per-craft: the ship's is
//!   low, so it coasts), plus the match-velocity brake when held — speed is bounded without a cap.
//! - **lift** along body-up scaled by FORWARD airspeed (a wing): `up · (LIFT · max(0, v·forward))`.
//!   Zero for the ship.
//! - **torque** in the BODY frame from the pitch/roll/yaw axes (about +X / +Z / +Y), plus a mild
//!   per-craft angular drag so the craft is steerable. No leveling term and nothing caps the
//!   attitude, so rapier integrates the body-frame angular velocity freely — the craft loops,
//!   rolls, and flies inverted, now with real momentum and collisions.
//!
//! The plane and the ship are NOT two code paths: they are two [`VehicleParams`] sets driving the
//! one system. The difference is only the parameters (which thrust term is live, lift on/off, drag)
//! and how the bridge maps the sticks/triggers into the shared [`VehicleControl`] intents.
//!
//! ## Why it is policy-safe (owner: never destabilise the trained walking)
//! The vehicle carries the body's `VEHICLE_GROUP` collision bit (see [`vehicle_collision`]); at
//! TRAINING time no vehicle entity
//! exists, so the crab's collision filter naming that bit matches nothing and the trained physics
//! is bit-identical. A spawned vehicle is its own rapier island until it actually touches the crab,
//! so its mere presence changes nothing; only a deliberate strike transfers momentum (the headline
//! — bounce off Sally, shove her legs by mass). The crab's `physics_digest` folds only crab bodies,
//! so the vehicle never enters the lockstep desync hash either.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::bot::body::vehicle_collision;

// ---------------------------------------------------------------------------
// Tuning surface (ARENA scale: ±10 m box, ~0.5 m crab, 9.81 m/s² gravity). These are gray-box
// defaults — the owner feel-tests and trims them; the unit tests pin QUALITATIVE behaviour
// (accelerates, lift rises with airspeed, a torque inverts the body with no auto-right, drag
// bounds speed), never specific speeds, so a tuning change doesn't churn the tests.
// ---------------------------------------------------------------------------

/// Half-extents of the vehicle's box collider (m). Deliberately SMALLER than the ~0.5 m crab so it
/// reads as a little craft buzzing the giant Sally, and light enough to bounce off her without
/// bowling her over.
const VEHICLE_HALF: Vec3 = Vec3::new(0.22, 0.07, 0.34);

/// Collider density (kg/m³) → the vehicle masses ~2 kg at [`VEHICLE_HALF`]. Sets how hard a strike
/// shoves Sally's legs (rapier momentum transfer): heavy enough to push, light enough that her
/// trained gait recovers.
const VEHICLE_DENSITY: f32 = 50.0;

/// Throttle-lever trim per tick at full deflection (lever is 0..1) — a persistent quadrant, not a
/// momentary push (the plane's AC6 throttle: RT trims it up, LT down, it holds when neither).
const THROTTLE_TRIM_RATE: f32 = 0.03;

/// Extra linear drag (N per m/s) added while the ship holds MATCH VELOCITY — bleeds the relative
/// velocity to rest, the cheap arena analog of Outer Wilds' match-velocity-to-zero. Dwarfs the
/// ship's coasting drag so the brake is decisive, but it's still just drag (no teleport/clamp).
const MATCH_VEL_DAMP: f32 = 6.0;

/// Plane tuning (Ace Combat 6): a throttle LEVER drives forward (nose +Z) thrust, wings make lift
/// from forward airspeed, brisk roll for banked turns. No direct strafe/vertical thrusters.
const PLANE: VehicleParams = VehicleParams {
    thrust_axis: Vec3::Z, // nose — the lever pushes here
    lever_thrust: 60.0,
    direct_thrust: Vec3::ZERO,
    lift: 6.0,
    drag_lin: 1.5,
    drag_quad: 0.6,
    angular_drag: 0.9,
    pitch_torque: 6.0,
    roll_torque: 9.0,
    yaw_torque: 3.0,
};

/// Ship tuning (Outer Wilds): NO throttle lever and NO wings — DIRECT body-frame thrusters on all
/// three axes (X strafe, Y vertical, Z forward) you tap and coast on. The ship flies in ZERO-G (see
/// [`gravity_scale`]), so it floats neutrally and the thrusters are GENTLE — forward full-burn tops
/// out around 6 m/s, which in the ±10 m arena reads as the slow, weighty Outer-Wilds drift (a
/// stronger thruster would cross the box in under a second — a pinball, not a spaceship). Drag is LOW so the
/// coast carries; the match-velocity brake stops it on demand. Full aim authority (pitch/yaw on the
/// right stick); bumper roll is a GENTLE quarter of that so a barrel-roll is a slow deliberate twist.
///
/// Ship aim torque (pitch/yaw at full deflection, N·m); bumper roll derives from it so the two can't
/// drift. The one source for the ship's rotation feel.
const SHIP_AIM_TORQUE: f32 = 7.0;
const SHIP: VehicleParams = VehicleParams {
    thrust_axis: Vec3::Z, // unused (lever_thrust 0) — kept a valid unit axis
    lever_thrust: 0.0,
    direct_thrust: Vec3::new(3.0, 3.0, 4.0), // strafe / vertical / forward (N) — gentle, zero-g
    lift: 0.0,
    drag_lin: 0.35, // low → long coast (~6 s velocity time-constant at ~2 kg)
    drag_quad: 0.04,
    angular_drag: 1.1,
    pitch_torque: SHIP_AIM_TORQUE,
    // Bumper roll is one source: a quarter of the aim torque (owner playtest — a slow deliberate
    // twist, not a snap), so retuning the aim keeps the "quarter" relation instead of drifting.
    roll_torque: SHIP_AIM_TORQUE * 0.25,
    yaw_torque: SHIP_AIM_TORQUE,
};

/// Plane spawn: a few metres up over the arena centre, nosed forward (+Z) with a little cruise
/// speed so the wing is already making lift. Ship spawn: low, at rest, level — it hovers in place
/// off its thrusters from tick 0. Both are arena-frame (true scale), bounded by the ±10 m walls.
const PLANE_SPAWN_ALTITUDE: f32 = 4.0;
const PLANE_SPAWN_SPEED: f32 = 4.0;
const SHIP_SPAWN_ALTITUDE: f32 = 1.0;

/// Per-craft flight constants — the two parameter sets that configure the ONE force system as a
/// plane or a ship (see the module docs: not two code paths). Per-craft drag is what gives the
/// ship its long Outer-Wilds coast while the plane stays crisp.
#[derive(Clone, Copy)]
struct VehicleParams {
    /// Body-frame axis the LEVER thrust pushes along (plane = +Z nose). Unused when `lever_thrust`
    /// is 0 (the ship).
    thrust_axis: Vec3,
    /// Lever thrust force at full throttle (N), along [`thrust_axis`](Self::thrust_axis). 0 = no
    /// lever (the ship uses `direct_thrust` instead).
    lever_thrust: f32,
    /// DIRECT body-frame thruster force per axis (N): x = strafe (+right), y = vertical (+up),
    /// z = forward (+nose). Scaled component-wise by [`VehicleControl::thrust`]. `ZERO` = no direct
    /// thrusters (the plane, which thrusts only through its lever).
    direct_thrust: Vec3,
    /// Lift force per unit forward airspeed (N per m/s), along body-up. 0 for the wingless ship.
    lift: f32,
    /// Linear drag: `−(drag_lin·v + drag_quad·|v|·v)`. The ship's is LOW so it coasts.
    drag_lin: f32,
    drag_quad: f32,
    /// Angular drag (N·m per rad/s): bleeds spin so the craft is steerable. NOT a cap — it never
    /// bounds the ANGLE (owner 702), only the rate.
    angular_drag: f32,
    /// Control torque at full deflection (N·m): pitch about body +X, roll about +Z, yaw about +Y.
    pitch_torque: f32,
    roll_torque: f32,
    yaw_torque: f32,
}

/// Which craft the one vehicle model is configured as. The plane and ship share the force system;
/// this selects the [`VehicleParams`] + the spawn pose. (The input→[`VehicleControl`] mapping per
/// craft lives in `net`'s bridge — Ace Combat 6 for the plane, Outer Wilds for the ship.)
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum VehicleKind {
    #[default]
    Plane,
    Ship,
}

impl VehicleKind {
    fn params(self) -> VehicleParams {
        match self {
            VehicleKind::Plane => PLANE,
            VehicleKind::Ship => SHIP,
        }
    }

    /// Arena-frame spawn pose, facing +Z (the crab's forward), at the arena centre.
    fn spawn_transform(self) -> Transform {
        let y = match self {
            VehicleKind::Plane => PLANE_SPAWN_ALTITUDE,
            VehicleKind::Ship => SHIP_SPAWN_ALTITUDE,
        };
        Transform::from_xyz(0.0, y, 0.0)
    }

    /// Initial velocity: the plane spawns at cruise (so the wing flies from tick 0), the ship at
    /// rest (it holds station on its thrusters).
    fn spawn_velocity(self) -> Velocity {
        let linear = match self {
            VehicleKind::Plane => Vec3::new(0.0, 0.0, PLANE_SPAWN_SPEED),
            VehicleKind::Ship => Vec3::ZERO,
        };
        Velocity { linear, angular: Vec3::ZERO }
    }

    /// Rapier gravity multiplier. The plane is a real aircraft (gravity 1× — lift fights weight,
    /// stalls, dives). The Outer-Wilds ship flies in ZERO-G: it floats neutrally so its gentle
    /// thrusters drift it rather than fighting a constant fall, which is the whole thrust-and-coast
    /// feel (and is why its vertical thruster can be as gentle as the lateral ones).
    fn gravity_scale(self) -> f32 {
        match self {
            VehicleKind::Plane => 1.0,
            VehicleKind::Ship => 0.0,
        }
    }
}

/// Marks the one vehicle rigidbody and carries its persistent throttle lever (0..1, trimmed each
/// tick). On the component, not in [`VehicleControl`], so a fresh spawn starts at a known throttle
/// and the lever can't outlive the body.
#[derive(Component)]
pub struct Vehicle {
    pub kind: VehicleKind,
    /// Throttle lever, 0..1 — the plane's persistent quadrant, trimmed each tick by
    /// [`VehicleControl::throttle_trim`]. Unused by the ship (it has no lever). On the component,
    /// not in [`VehicleControl`], so a fresh spawn starts at a known throttle.
    throttle: f32,
}

/// The player's per-tick vehicle command, written by the net client each fixed step (the bridge),
/// read by [`apply_vehicle_forces`] + [`manage_vehicle`]. The bridge maps the raw sticks/triggers
/// into these per-craft (Ace Combat 6 for the plane, Outer Wilds for the ship), so this is the ONE
/// vocabulary the force model reads regardless of craft. `active`/`kind` drive spawn/despawn on the
/// E-cycle. Inert (no body, no forces) until `active`.
#[derive(Resource, Default)]
pub struct VehicleControl {
    /// Is the player currently piloting — spawns the body on the rising edge, despawns on the fall.
    pub active: bool,
    /// Which craft to be (cycled by the player). A change while active respawns the body.
    pub kind: VehicleKind,
    /// Throttle-lever trim this tick (plane: RT up / LT down), `-1..1`. The ship leaves it 0.
    pub throttle_trim: f32,
    /// DIRECT body-frame thrust intent (ship): x = strafe (+right), y = vertical (+up),
    /// z = forward (+nose), each `-1..1`. The plane leaves it `ZERO` (it thrusts via the lever).
    pub thrust: Vec3,
    /// Pitch (elevator): positive noses UP, `-1..1`.
    pub pitch: f32,
    /// Roll (ailerons): positive banks RIGHT, `-1..1`.
    pub roll: f32,
    /// Yaw (rudder): positive yaws RIGHT, `-1..1`.
    pub yaw: f32,
    /// Hold to brake toward rest (ship: Outer Wilds match-velocity-to-zero) — boosts linear drag
    /// this tick. The plane leaves it `false`.
    pub match_velocity: bool,
}

/// Plugin: the vehicle resource + the spawn/despawn manager + the force system. Added to the crab
/// world wherever a player can pilot (the net client and the headless policy-stability gate); the
/// trainer doesn't add it, so training never carries the vehicle systems. Both systems run in
/// `FixedUpdate` before `PhysicsSet::SyncBackend` (the rapier solve), chained so a body spawned this
/// tick is force-driven from the next — the same slot the crab actuator writes its torques in.
pub struct VehiclePlugin;

impl Plugin for VehiclePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VehicleControl>().add_systems(
            FixedUpdate,
            (manage_vehicle, apply_vehicle_forces)
                .chain()
                .before(PhysicsSet::SyncBackend),
        );
    }
}

/// Spawn the vehicle on the rising edge of [`VehicleControl::active`] (and respawn it on a kind
/// change); despawn it when the player steps out. One body at a time — the player flies a single
/// craft. The body is a dynamic box with the [`vehicle_collision`] groups (arena + every crab), so
/// it bounces off the walls and strikes Sally.
fn manage_vehicle(
    mut commands: Commands,
    control: Res<VehicleControl>,
    existing: Query<(Entity, &Vehicle)>,
) {
    let current = existing.iter().next();
    match (control.active, current) {
        // Piloting and the body already matches the chosen craft — nothing to do.
        (true, Some((_, v))) if v.kind == control.kind => {}
        // Stepped into a vehicle, or cycled to a different craft: (re)spawn it. Despawn any stale
        // body first so a kind switch can't leave two alive. A kind switch starts the new craft at
        // its spawn pose + a zero throttle lever (the fresh body's known state), not the old one's.
        (true, current) => {
            if let Some((e, _)) = current {
                commands.entity(e).despawn();
            }
            spawn_vehicle(&mut commands, control.kind);
        }
        // Stepped out: despawn the body if present.
        (false, Some((e, _))) => commands.entity(e).despawn(),
        (false, None) => {}
    }
}

/// The vehicle rigidbody's component bundle — the ONE definition shared by boarding
/// ([`spawn_vehicle`]) and the headless ram gate ([`spawn_ram_vehicle`]), so the collider, mass,
/// and collision groups can't drift between them. Mass comes from the collider density (so a
/// strike's momentum follows the size); `Velocity`/`ExternalForce` let the force system read the
/// body's motion and write its forces.
fn vehicle_bundle(kind: VehicleKind, transform: Transform, velocity: Velocity) -> impl Bundle {
    (
        Vehicle { kind, throttle: 0.0 },
        RigidBody::Dynamic,
        Collider::cuboid(VEHICLE_HALF.x, VEHICLE_HALF.y, VEHICLE_HALF.z),
        ColliderMassProperties::Density(VEHICLE_DENSITY),
        // Per-craft gravity: the plane falls (1×), the Outer-Wilds ship floats (zero-g).
        GravityScale(kind.gravity_scale()),
        vehicle_collision(),
        transform,
        velocity,
        ExternalForce::default(),
    )
}

/// Spawn one vehicle rigidbody of `kind` at its arena spawn pose — the boarding path
/// ([`manage_vehicle`]).
fn spawn_vehicle(commands: &mut Commands, kind: VehicleKind) {
    commands.spawn(vehicle_bundle(kind, kind.spawn_transform(), kind.spawn_velocity()));
}

/// Spawn a vehicle rigidbody directly into a `World` at an explicit pose + velocity, returning its
/// entity. For the headless crab-policy-stability gate: it places a ram vehicle on Sally (a pure
/// ballistic body, with no [`VehiclePlugin`] driving it) to verify her trained walking survives the
/// collision. Same bundle as boarding, so the gate exercises the real collider/mass/groups.
pub fn spawn_ram_vehicle(
    world: &mut World,
    kind: VehicleKind,
    transform: Transform,
    velocity: Velocity,
) -> Entity {
    world.spawn(vehicle_bundle(kind, transform, velocity)).id()
}

/// The ONE flight force model (see the module docs). Reads the control intents + the body's pose
/// and velocity, integrates the throttle lever, and overwrites the body's [`ExternalForce`] with
/// thrust + lift + drag (force) and the body-frame control torque + angular drag (torque). Gravity
/// is rapier's. Runs before the solve; the body's presence (not a flag) gates it, so it acts on
/// whatever vehicle [`manage_vehicle`] spawned.
fn apply_vehicle_forces(
    control: Res<VehicleControl>,
    mut q: Query<(&mut Vehicle, &Transform, &Velocity, &mut ExternalForce)>,
) {
    for (mut vehicle, transform, velocity, mut ef) in q.iter_mut() {
        let p = vehicle.kind.params();
        let rot = transform.rotation;
        let forward = rot * Vec3::Z;
        let up = rot * Vec3::Y;
        let v = velocity.linear;

        // Persistent throttle lever (plane), trimmed by `throttle_trim` and held between ticks
        // (clamped to 0..1). The ship leaves `throttle_trim` 0, so its lever idles unused.
        vehicle.throttle =
            (vehicle.throttle + control.throttle_trim * THROTTLE_TRIM_RATE).clamp(0.0, 1.0);

        // Thrust = LEVER term along the body thrust axis (plane nose; `lever_thrust` 0 ⇒ 0 for the
        // ship) + DIRECT body-frame thrusters `rot · (thrust ⊙ direct_thrust)` (ship strafe/
        // vertical/forward; `direct_thrust` ZERO ⇒ 0 for the plane). One expression, no per-craft
        // branch — each craft's unused term is zeroed by its params.
        let lever = (rot * p.thrust_axis) * (p.lever_thrust * vehicle.throttle);
        let direct = rot * (control.thrust * p.direct_thrust);
        let thrust = lever + direct;

        // Bernoulli lift along body-up, proportional to FORWARD airspeed (a wing). 0 for the
        // wingless ship. Backward motion makes no lift, so it can't suck a reversing craft down.
        let forward_airspeed = v.dot(forward).max(0.0);
        let lift = up * (p.lift * forward_airspeed);

        // Drag opposing velocity: linear + quadratic (per-craft: the ship's is low, so it coasts),
        // plus the match-velocity brake while held — bounds speed with no cap.
        let speed = v.length();
        let match_damp = if control.match_velocity { MATCH_VEL_DAMP } else { 0.0 };
        let drag = -v * (p.drag_lin + match_damp + p.drag_quad * speed);

        ef.force = thrust + lift + drag;

        // Body-frame control torque (UNCLAMPED — no leveling, no attitude bound): pitch about +X,
        // roll about +Z, yaw about +Y, rotated into the world frame rapier integrates. A mild
        // per-craft angular drag bleeds spin for control without bounding the angle. The pitch sign
        // is NEGATED: a positive +X torque rotates the nose (+Z) toward −Y (DOWN), so to make
        // positive `pitch` (the pilot's nose-UP intent) raise the nose we apply −pitch about +X.
        // The bridge gives each craft its feel (both craft pitch intuitively — push the stick up to
        // raise the nose — arriving here as positive `pitch`); roll/yaw carry their reconciling sign
        // from the bridge so "bank right"/"yaw right" hold.
        let body_torque = Vec3::new(
            -control.pitch * p.pitch_torque,
            control.yaw * p.yaw_torque,
            -control.roll * p.roll_torque,
        );
        ef.torque = rot * body_torque - velocity.angular * p.angular_drag;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::headless::headless_app;

    /// A windowless world with the vehicle systems, plus one vehicle spawned FAR from the arena
    /// (origin) so the force model is tested in free space — no ground/wall/crab contact to muddy
    /// the reading. Returns the body entity.
    fn app_with_vehicle(kind: VehicleKind, at: Vec3, vel: Vec3) -> (App, Entity) {
        let mut app = headless_app();
        app.add_plugins(VehiclePlugin);
        // Mark piloting so `manage_vehicle` keeps the body we spawn directly below (it despawns any
        // vehicle while `active` is false). Matching kind ⇒ no respawn, our entity persists.
        {
            let mut c = app.world_mut().resource_mut::<VehicleControl>();
            c.active = true;
            c.kind = kind;
        }
        // Spawn through the SHARED bundle (so the test exercises the real collider/mass/groups, not
        // a copy that could drift), then trim the throttle full so the thrust tests have authority.
        let transform = Transform::from_translation(at);
        let velocity = Velocity { linear: vel, angular: Vec3::ZERO };
        let e = app.world_mut().spawn(vehicle_bundle(kind, transform, velocity)).id();
        app.world_mut().entity_mut(e).get_mut::<Vehicle>().unwrap().throttle = 1.0;
        (app, e)
    }

    fn body(app: &App, e: Entity) -> (&Transform, &Velocity) {
        let ent = app.world().entity(e);
        (ent.get::<Transform>().unwrap(), ent.get::<Velocity>().unwrap())
    }

    const FAR: Vec3 = Vec3::new(500.0, 300.0, 500.0);

    /// Full throttle along the nose accelerates the plane forward (+Z). Held with a little starting
    /// speed (so lift roughly offsets gravity), the forward speed must RISE.
    #[test]
    fn plane_thrust_accelerates_forward() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, 3.0));
        app.world_mut().resource_mut::<VehicleControl>().throttle_trim = 1.0;
        let v0 = body(&app, e).1.linear.z;
        for _ in 0..30 {
            app.update();
        }
        let v1 = body(&app, e).1.linear.z;
        assert!(v1 > v0 + 1.0, "forward speed did not rise: {v0} -> {v1}");
    }

    /// Lift grows with FORWARD airspeed: a fast-flying plane gets more upward force than a slow one.
    /// Compare the vertical velocity gained over a few ticks at two airspeeds, throttle off so the
    /// only vertical players are lift vs gravity.
    #[test]
    fn lift_rises_with_airspeed() {
        let dy = |speed: f32| {
            let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, speed));
            app.world_mut().resource_mut::<VehicleControl>().throttle_trim = -1.0; // throttle to 0
            let y0 = body(&app, e).1.linear.y;
            for _ in 0..5 {
                app.update();
            }
            body(&app, e).1.linear.y - y0
        };
        let slow = dy(2.0);
        let fast = dy(12.0);
        assert!(
            fast > slow,
            "lift did not rise with airspeed: Δvy slow={slow} fast={fast}"
        );
    }

    /// DIRECTION pin (the sign the cockpit legend rides): a positive `pitch` (the pilot's nose-UP
    /// intent) raises the nose — the world-space forward vector's Y goes POSITIVE within a few
    /// ticks from level. Guards the +X-torque-noses-down trap that an inversion-only test misses.
    #[test]
    fn positive_pitch_raises_the_nose() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::ZERO);
        app.world_mut().resource_mut::<VehicleControl>().pitch = 1.0;
        for _ in 0..10 {
            app.update();
        }
        let nose_y = (body(&app, e).0.rotation * Vec3::Z).y;
        assert!(nose_y > 0.0, "positive pitch must raise the nose, got nose.y={nose_y}");
    }

    /// DIRECTION pin: a positive `yaw` control (a +Y torque) turns the nose toward +X (right). The
    /// driver feeds `yaw = -move_strafe`, so the player's A key (positive `move_strafe`) yaws LEFT —
    /// the sign the "Rudder left" label rides — but that reconciliation lives in `drive_lockstep`;
    /// here we pin only the force model's own convention.
    #[test]
    fn positive_yaw_turns_nose_right() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::ZERO);
        app.world_mut().resource_mut::<VehicleControl>().yaw = 1.0;
        for _ in 0..10 {
            app.update();
        }
        let nose_x = (body(&app, e).0.rotation * Vec3::Z).x;
        assert!(nose_x > 0.0, "positive yaw must turn the nose right (+X), got nose.x={nose_x}");
    }

    /// A held pitch input inverts the plane (body-up points DOWN) with NO return to level — the
    /// unclamped, no-auto-right invariant (owner 702). After enough ticks the world-space up vector
    /// flips below horizontal at least once and never snaps back on its own.
    #[test]
    fn pitch_input_loops_the_craft_without_autoright() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::ZERO);
        app.world_mut().resource_mut::<VehicleControl>().pitch = 1.0;
        let mut went_inverted = false;
        for _ in 0..240 {
            app.update();
            let up_y = (body(&app, e).0.rotation * Vec3::Y).y;
            if up_y < -0.3 {
                went_inverted = true;
            }
        }
        assert!(went_inverted, "held pitch never inverted the craft — a cap or auto-level crept in");
    }

    /// Drag bounds speed: a fast body with no thrust must SLOW down (drag opposes velocity), so a
    /// coasting craft can't run away to infinity.
    #[test]
    fn drag_bleeds_speed() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(20.0, 0.0, 0.0));
        app.world_mut().resource_mut::<VehicleControl>().throttle_trim = -1.0; // throttle to 0
        let s0 = body(&app, e).1.linear.length();
        for _ in 0..20 {
            app.update();
        }
        let s1 = body(&app, e).1.linear.length();
        assert!(s1 < s0, "drag did not bleed speed: {s0} -> {s1}");
    }

    /// Stepping into a vehicle spawns exactly one body; cycling kind respawns (still one); stepping
    /// out despawns it. The one-craft-at-a-time invariant `manage_vehicle` enforces.
    #[test]
    fn manage_spawns_and_despawns_one_vehicle() {
        let mut app = headless_app();
        app.add_plugins(VehiclePlugin);
        let count = |app: &mut App| {
            app.world_mut().query::<&Vehicle>().iter(app.world()).count()
        };

        app.update();
        assert_eq!(count(&mut app), 0, "no vehicle before piloting");

        {
            let mut c = app.world_mut().resource_mut::<VehicleControl>();
            c.active = true;
            c.kind = VehicleKind::Plane;
        }
        app.update();
        assert_eq!(count(&mut app), 1, "one vehicle after boarding");

        app.world_mut().resource_mut::<VehicleControl>().kind = VehicleKind::Ship;
        app.update();
        assert_eq!(count(&mut app), 1, "still one vehicle after a kind cycle");
        assert_eq!(
            app.world_mut().query::<&Vehicle>().single(app.world()).unwrap().kind,
            VehicleKind::Ship,
            "the body became the cycled kind",
        );

        app.world_mut().resource_mut::<VehicleControl>().active = false;
        app.update();
        assert_eq!(count(&mut app), 0, "vehicle despawned on step-out");
    }

    /// The ship's DIRECT body-frame thrusters (Outer Wilds): a forward thrust intent (+Z) with no
    /// throttle lever accelerates it along its nose. The plane can't do this (it has no direct
    /// thrusters and an idle lever) — this is the wingless 6-DOF thruster model. Thresholds are
    /// loose: the thrusters are deliberately GENTLE (OW drift in a small arena), so this pins the
    /// DIRECTION, not a magnitude.
    #[test]
    fn ship_direct_forward_thrust_accelerates() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::ZERO);
        app.world_mut().resource_mut::<VehicleControl>().thrust = Vec3::new(0.0, 0.0, 1.0);
        let v0 = body(&app, e).1.linear.z;
        for _ in 0..90 {
            app.update();
        }
        let v1 = body(&app, e).1.linear.z;
        assert!(v1 > v0 + 0.5, "ship forward thruster did not accelerate: {v0} -> {v1}");
    }

    /// The ship STRAFES from a lateral thrust intent (+X) — translational 6-DOF, no banking. A pure
    /// `thrust.x` at a level attitude gains +X velocity (the wing-less sideways thruster).
    #[test]
    fn ship_lateral_thrust_strafes() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::ZERO);
        app.world_mut().resource_mut::<VehicleControl>().thrust = Vec3::new(1.0, 0.0, 0.0);
        let v0 = body(&app, e).1.linear.x;
        for _ in 0..90 {
            app.update();
        }
        let v1 = body(&app, e).1.linear.x;
        assert!(v1 > v0 + 0.5, "ship lateral thruster did not strafe +X: {v0} -> {v1}");
    }

    /// The ship flies in ZERO-G (the Outer-Wilds float): with NO thrust it neither rises nor falls,
    /// so a gentle thruster drifts it rather than fighting a constant fall. The plane, by contrast,
    /// falls under gravity. Pins the per-craft `gravity_scale`.
    #[test]
    fn ship_is_zero_g_but_plane_falls() {
        let fall = |kind: VehicleKind| {
            let (mut app, e) = app_with_vehicle(kind, FAR, Vec3::ZERO);
            // Zero the lever (no thrust, hence no forward speed → no lift): the only vertical player
            // left is gravity, so this isolates the gravity scale.
            app.world_mut().entity_mut(e).get_mut::<Vehicle>().unwrap().throttle = 0.0;
            for _ in 0..30 {
                app.update();
            }
            body(&app, e).1.linear.y
        };
        assert!(fall(VehicleKind::Ship).abs() < 0.5, "ship must float (zero-g), not fall");
        assert!(fall(VehicleKind::Plane) < -1.0, "plane must fall under gravity");
    }

    /// Match-velocity brakes the coasting ship toward rest: a moving ship with no thrust but
    /// `match_velocity` held bleeds speed FASTER than coasting drag alone — the Outer Wilds
    /// match-velocity-to-zero feel, implemented as boosted drag (still no clamp/teleport).
    #[test]
    fn ship_match_velocity_brakes_harder_than_coast() {
        let drop = |brake: bool| {
            let (mut app, e) =
                app_with_vehicle(VehicleKind::Ship, FAR, Vec3::new(10.0, 0.0, 0.0));
            app.world_mut().resource_mut::<VehicleControl>().match_velocity = brake;
            let s0 = body(&app, e).1.linear.length();
            for _ in 0..20 {
                app.update();
            }
            s0 - body(&app, e).1.linear.length()
        };
        let coast = drop(false);
        let braked = drop(true);
        assert!(
            braked > coast,
            "match-velocity must brake harder than coasting drag: Δ coast={coast} braked={braked}"
        );
    }
}
