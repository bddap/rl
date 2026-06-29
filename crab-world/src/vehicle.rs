//! The player's single-player VEHICLE — a rapier rigidbody living in the crab's ONE
//! `bevy_rapier3d` world, so it is official host-authoritative game state that COLLIDES with
//! the trained crab "Sally" (owner 702/703/704). It replaces the old integer fixed-point flight
//! integrator (`net::sim`'s `step_plane`/`step_helicopter`, deleted in this migration): ONE flight
//! model, shared by the plane and the helicopter.
//!
//! ## Force model (all UNCLAMPED — owner 702: no rotation caps, no auto-right by construction)
//! Each fixed step, before the rapier solve, [`apply_vehicle_forces`] reads the player's quantized
//! control axes ([`VehicleControl`]) + the body's own pose/velocity and writes one [`ExternalForce`]
//! (overwritten, never accumulated):
//! - **thrust** along the body THRUST AXIS scaled by a persistent throttle lever — body-forward
//!   (+Z, the nose) for the plane, body-up (+Y, the rotor disc) for the helicopter.
//! - **gravity** — rapier applies it to the dynamic body (no manual term).
//! - **drag** opposing velocity: linear `−k·v` plus quadratic `−k₂·|v|·v`, so speed is bounded
//!   without a cap.
//! - **lift** along body-up scaled by FORWARD airspeed (Bernoulli — a wing for the plane, the
//!   rotor disc for the heli): `up · (LIFT · max(0, v·forward))`.
//! - **torque** in the BODY frame from the pitch/roll/yaw axes (about +X / +Z / +Y), plus a mild
//!   angular drag so the craft is steerable. No leveling term and nothing caps the attitude, so
//!   rapier integrates the body-frame angular velocity freely — the craft loops, rolls, and flies
//!   inverted, exactly as the old integer model did, now with real momentum and collisions.
//!
//! The plane and the helicopter are NOT two code paths: they are two [`VehicleParams`] sets driving
//! the one system, with the SAME axis→degree-of-freedom control mapping (the difference is only the
//! thrust axis + the tuning constants). One model, per owner 702.
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

/// Collider density (kg/m³) → the vehicle masses ~3 kg at [`VEHICLE_HALF`]. Sets how hard a strike
/// shoves Sally's legs (rapier momentum transfer): heavy enough to push, light enough that her
/// trained gait recovers.
const VEHICLE_DENSITY: f32 = 50.0;

/// Throttle-lever trim per tick at full deflection (lever is 0..1) — a persistent quadrant, not a
/// momentary push, like the old integer model's throttle/collective.
const THROTTLE_TRIM_RATE: f32 = 0.03;

/// Linear drag: `−(DRAG_LIN·v + DRAG_QUAD·|v|·v)` (N per m/s, N per (m/s)²). Bounds top speed and
/// bleeds coasting velocity so the craft doesn't run away.
const DRAG_LIN: f32 = 1.5;
const DRAG_QUAD: f32 = 0.6;

/// Angular drag (N·m per rad/s): bleeds spin so the craft is controllable. NOT a cap — it never
/// bounds the ANGLE, so full inversion stays free (owner 702); it only damps the rate, the angular
/// analogue of linear drag.
const ANGULAR_DRAG: f32 = 0.9;

/// Plane tuning: forward thrust, wing lift per unit forward airspeed, control torques.
const PLANE: VehicleParams = VehicleParams {
    thrust_axis: Vec3::Z, // nose
    thrust: 60.0,
    lift: 6.0,
    pitch_torque: 6.0,
    roll_torque: 9.0,
    yaw_torque: 3.0,
};

/// Helicopter tuning: thrust along the rotor disc (body-up); a touch of disc lift with airspeed;
/// brisker cyclic/pedal authority than the plane (it pivots in a hover).
const HELI: VehicleParams = VehicleParams {
    thrust_axis: Vec3::Y, // rotor disc normal
    thrust: 80.0,
    lift: 2.0,
    pitch_torque: 8.0,
    roll_torque: 8.0,
    yaw_torque: 6.0,
};

/// Plane spawn: a few metres up over the arena centre, nosed forward (+Z) with a little cruise
/// speed so the wing is already making lift. Helicopter spawn: low, at rest, level — it hovers off
/// the collective from tick 0. Both are arena-frame (true scale), bounded by the ±10 m walls.
const PLANE_SPAWN_ALTITUDE: f32 = 4.0;
const PLANE_SPAWN_SPEED: f32 = 4.0;
const HELI_SPAWN_ALTITUDE: f32 = 1.0;

/// Per-mode flight constants — the two parameter sets that configure the ONE force system as a
/// plane or a helicopter (see the module docs: not two code paths).
#[derive(Clone, Copy)]
struct VehicleParams {
    /// Body-frame axis thrust pushes along (plane = +Z nose, heli = +Y rotor disc).
    thrust_axis: Vec3,
    /// Thrust force at full throttle (N).
    thrust: f32,
    /// Lift force per unit forward airspeed (N per m/s), along body-up.
    lift: f32,
    /// Control torque at full deflection (N·m): pitch about body +X, roll about +Z, yaw about +Y.
    pitch_torque: f32,
    roll_torque: f32,
    yaw_torque: f32,
}

/// Which craft the one vehicle model is configured as. The plane and helicopter share the force
/// system and the control mapping; this selects the [`VehicleParams`] + the spawn pose.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum VehicleKind {
    #[default]
    Plane,
    Helicopter,
}

impl VehicleKind {
    fn params(self) -> VehicleParams {
        match self {
            VehicleKind::Plane => PLANE,
            VehicleKind::Helicopter => HELI,
        }
    }

    /// Arena-frame spawn pose, facing +Z (the crab's forward), at the arena centre.
    fn spawn_transform(self) -> Transform {
        let y = match self {
            VehicleKind::Plane => PLANE_SPAWN_ALTITUDE,
            VehicleKind::Helicopter => HELI_SPAWN_ALTITUDE,
        };
        Transform::from_xyz(0.0, y, 0.0)
    }

    /// Initial velocity: the plane spawns at cruise (so the wing flies from tick 0), the heli at
    /// rest (it lifts vertically off the collective).
    fn spawn_velocity(self) -> Velocity {
        let linear = match self {
            VehicleKind::Plane => Vec3::new(0.0, 0.0, PLANE_SPAWN_SPEED),
            VehicleKind::Helicopter => Vec3::ZERO,
        };
        Velocity { linear, angular: Vec3::ZERO }
    }
}

/// Marks the one vehicle rigidbody and carries its persistent throttle lever (0..1, trimmed each
/// tick). On the component, not in [`VehicleControl`], so a fresh spawn starts at a known throttle
/// and the lever can't outlive the body.
#[derive(Component)]
pub struct Vehicle {
    pub kind: VehicleKind,
    /// Throttle / collective lever, 0..1 — a persistent quadrant trimmed by the forward axis.
    throttle: f32,
}

/// The player's per-tick vehicle command, written by the net client each fixed step (the bridge),
/// read by [`apply_vehicle_forces`] + [`manage_vehicle`]. Axes are the quantized control inputs,
/// already screen-reconciled by the caller, in `-1..1`; `throttle_trim` trims the lever this tick.
/// `active`/`kind` drive spawn/despawn on the E-cycle. Inert (no body, no forces) until `active`.
#[derive(Resource, Default)]
pub struct VehicleControl {
    /// Is the player currently piloting — spawns the body on the rising edge, despawns on the fall.
    pub active: bool,
    /// Which craft to be (cycled by the player). A change while active respawns the body.
    pub kind: VehicleKind,
    /// Throttle / collective trim this tick (forward axis), `-1..1`.
    pub throttle_trim: f32,
    /// Pitch (elevator / fore-aft cyclic): positive noses UP, `-1..1`.
    pub pitch: f32,
    /// Roll (ailerons / lateral cyclic): positive banks RIGHT, `-1..1`.
    pub roll: f32,
    /// Yaw (rudder / tail rotor): positive yaws RIGHT, `-1..1`.
    pub yaw: f32,
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

/// The ONE flight force model (see the module docs). Reads the control axes + the body's pose and
/// velocity, integrates the throttle lever, and overwrites the body's [`ExternalForce`] with
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

        // Persistent throttle / collective lever, trimmed by the forward axis and held between
        // ticks (clamped to 0..1).
        vehicle.throttle =
            (vehicle.throttle + control.throttle_trim * THROTTLE_TRIM_RATE).clamp(0.0, 1.0);

        // Thrust along the body thrust axis (nose for the plane, rotor disc for the heli).
        let thrust = (rot * p.thrust_axis) * (p.thrust * vehicle.throttle);

        // Bernoulli lift along body-up, proportional to FORWARD airspeed (a wing / a rotor disc).
        // Backward motion makes no lift, so it can't suck a reversing craft down.
        let forward_airspeed = v.dot(forward).max(0.0);
        let lift = up * (p.lift * forward_airspeed);

        // Drag opposing velocity: linear + quadratic, so speed is bounded with no cap.
        let speed = v.length();
        let drag = -v * (DRAG_LIN + DRAG_QUAD * speed);

        ef.force = thrust + lift + drag;

        // Body-frame control torque (UNCLAMPED — no leveling, no attitude bound): pitch about +X,
        // roll about +Z, yaw about +Y, rotated into the world frame rapier integrates. A mild
        // angular drag bleeds spin for control without bounding the angle. The pitch sign is
        // NEGATED: a positive +X torque rotates the nose (+Z) toward −Y (DOWN), so to make positive
        // `pitch` (the pilot's nose-UP intent) raise the nose we apply −pitch about +X — the same
        // reconciliation the deleted integer model did. Roll/yaw already carry their reconciling
        // sign from `drive_lockstep` (−look_yaw, −move_strafe), so the labels (bank right, yaw
        // right) hold.
        let body_torque = Vec3::new(
            -control.pitch * p.pitch_torque,
            control.yaw * p.yaw_torque,
            -control.roll * p.roll_torque,
        );
        ef.torque = rot * body_torque - velocity.angular * ANGULAR_DRAG;
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

        app.world_mut().resource_mut::<VehicleControl>().kind = VehicleKind::Helicopter;
        app.update();
        assert_eq!(count(&mut app), 1, "still one vehicle after a kind cycle");
        assert_eq!(
            app.world_mut().query::<&Vehicle>().single(app.world()).unwrap().kind,
            VehicleKind::Helicopter,
            "the body became the cycled kind",
        );

        app.world_mut().resource_mut::<VehicleControl>().active = false;
        app.update();
        assert_eq!(count(&mut app), 0, "vehicle despawned on step-out");
    }
}
