use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::bot::body::VEHICLE_COLLISION;

const VEHICLE_HALF: Vec3 = Vec3::new(0.11, 0.035, 0.17);

const VEHICLE_DENSITY: f32 = 50.0;

const THROTTLE_TRIM_RATE: f32 = 0.03;

const MATCH_VEL_DAMP: f32 = 0.9;

const PLANE: VehicleParams = VehicleParams {
    lever_thrust: 4.0,
    direct_thrust: Vec3::ZERO,
    // 0.9 puts level flight at ~2.9 m/s — above PLANE_SPAWN_SPEED (no free ballooning at
    // spawn) and ~64% of the full-throttle terminal ~4.5 m/s, so climb costs throttle. At
    // 1.8 the plane out-lifted its ~2.6 N weight from 1.4 m/s (owner: "Bernoulli overdone",
    // rl#230); the level-flight band is pinned by `spawn_speed_sinks_high_speed_climbs`.
    lift: 0.9,
    drag_lin: 0.2,
    drag_quad: 0.15,
    angular_drag: 0.06,
    pitch_torque: 0.2,
    roll_torque: 0.3,
    yaw_torque: 0.1,
};

const SHIP_AIM_TORQUE: f32 = 0.2;
const SHIP: VehicleParams = VehicleParams {
    lever_thrust: 0.0,
    direct_thrust: Vec3::new(0.3, 0.3, 0.4),
    lift: 0.0,
    drag_lin: 0.05,
    drag_quad: 0.02,
    angular_drag: 0.07,
    pitch_torque: SHIP_AIM_TORQUE,
    roll_torque: SHIP_AIM_TORQUE * 0.25,
    yaw_torque: SHIP_AIM_TORQUE,
};

/// One spawn altitude for every craft, above the crab's standing-plus-flail reach: the ship
/// used to spawn at 0.5 m — inside Sally's body when she stood at her arena spawn, so her
/// wiggle physically batted the fresh craft meters away (rl#224).
const SPAWN_ALTITUDE: f32 = 2.0;
const PLANE_SPAWN_SPEED: f32 = 2.0;

#[derive(Clone, Copy)]
struct VehicleParams {
    lever_thrust: f32,
    direct_thrust: Vec3,
    lift: f32,
    drag_lin: f32,
    drag_quad: f32,
    angular_drag: f32,
    pitch_torque: f32,
    roll_torque: f32,
    yaw_torque: f32,
}

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

    fn spawn_transform(self) -> Transform {
        Transform::from_xyz(0.0, SPAWN_ALTITUDE, 0.0)
    }

    fn spawn_velocity(self) -> Velocity {
        let linear = match self {
            VehicleKind::Plane => Vec3::new(0.0, 0.0, PLANE_SPAWN_SPEED),
            VehicleKind::Ship => Vec3::ZERO,
        };
        Velocity {
            linear,
            angular: Vec3::ZERO,
        }
    }

    fn gravity_scale(self) -> f32 {
        match self {
            VehicleKind::Plane => 1.0,
            VehicleKind::Ship => 0.0,
        }
    }
}

/// Which player a craft belongs to — the crab-world key for per-player vehicles (rl#191: every
/// piloting player gets its OWN body in the host's one physics world). A plain `u8` newtype, not
/// `net`'s `PlayerId`, because this crate sits under `net`; the net bridge maps `PlayerId.0` in.
/// The server-authoritative local player is always pilot 0 (the host holds `PlayerId(0)` by
/// formation, and solo IS a host with a roster of one).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PilotId(pub u8);

/// Marks one pilot's vehicle rigidbody and carries its persistent throttle lever (0..1, trimmed
/// each tick). On the component, not in [`VehicleControls`], so a fresh spawn starts at a known
/// throttle and the lever can't outlive the body.
#[derive(Component)]
pub struct Vehicle {
    /// Whose craft this body is — at most one body per pilot ([`manage_vehicles`]).
    pub pilot: PilotId,
    pub kind: VehicleKind,
    throttle: f32,
}

#[derive(Clone, Copy)]
pub struct PilotCommand {
    /// Which craft to be (cycled by the player). A change while piloting respawns the body.
    pub kind: VehicleKind,
    pub throttle_trim: f32,
    pub thrust: Vec3,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub match_velocity: bool,
}

impl PilotCommand {
    /// Board `kind` with every axis neutral — the boarding-edge command (the per-tick axes are
    /// overwritten by the bridge each tick anyway). The one constructor, so choosing a craft is
    /// always explicit.
    pub fn new(kind: VehicleKind) -> Self {
        Self {
            kind,
            throttle_trim: 0.0,
            thrust: Vec3::ZERO,
            pitch: 0.0,
            roll: 0.0,
            yaw: 0.0,
            match_velocity: false,
        }
    }
}

/// Every currently-piloting player's command, keyed by pilot: an entry spawns + drives that
/// pilot's body; removing it despawns the body ([`manage_vehicles`]). The single seam the net
/// layer writes — solo/host write their own entry today; remote pilots' entries arrive off the
/// wire in a later rl#191 increment.
#[derive(Resource, Default)]
pub struct VehicleControls(pub std::collections::BTreeMap<PilotId, PilotCommand>);

pub struct VehiclePlugin;

impl Plugin for VehiclePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VehicleControls>().add_systems(
            FixedUpdate,
            (manage_vehicles, apply_vehicle_forces)
                .chain()
                .before(PhysicsSet::SyncBackend),
        );
    }
}

/// Keep the spawned bodies matched to [`VehicleControls`]: a new entry spawns that pilot's craft
/// (a kind change respawns it — the fresh body starts at its spawn pose + a zero throttle lever,
/// the known state, not the old craft's); a removed entry despawns it. At most one body per pilot,
/// provided bodies enter play only through this system (the headless gates spawn ram craft into
/// worlds WITHOUT the plugin): a body only spawns for a pilot with no matching body, and a
/// mismatched one is despawned in the same pass. Each body is a dynamic box with the
/// [`VEHICLE_COLLISION`] groups (arena + every crab), so it bounces off the walls and strikes
/// Sally.
fn manage_vehicles(
    mut commands: Commands,
    controls: Res<VehicleControls>,
    existing: Query<(Entity, &Vehicle)>,
) {
    let mut matched = std::collections::BTreeSet::new();
    for (e, v) in existing.iter() {
        match controls.0.get(&v.pilot) {
            // This pilot's body already is the chosen craft — keep it.
            Some(cmd) if cmd.kind == v.kind => {
                matched.insert(v.pilot);
            }
            // Kind changed or the pilot stepped out: despawn (a kind change spawns fresh below).
            _ => commands.entity(e).despawn(),
        }
    }
    for (&pilot, cmd) in &controls.0 {
        if !matched.contains(&pilot) {
            spawn_vehicle(&mut commands, pilot, cmd.kind);
        }
    }
}

fn vehicle_bundle(
    pilot: PilotId,
    kind: VehicleKind,
    transform: Transform,
    velocity: Velocity,
) -> impl Bundle {
    (
        Vehicle {
            pilot,
            kind,
            throttle: 0.0,
        },
        RigidBody::Dynamic,
        vehicle_collider(),
        ColliderMassProperties::Density(VEHICLE_DENSITY),
        GravityScale(kind.gravity_scale()),
        VEHICLE_COLLISION,
        transform,
        velocity,
        ExternalForce::default(),
    )
}

pub fn vehicle_collider() -> Collider {
    Collider::cuboid(VEHICLE_HALF.x, VEHICLE_HALF.y, VEHICLE_HALF.z)
}

/// Lateral spacing between different pilots' spawn spots (m). Crafts don't collide with each
/// other ([`VEHICLE_COLLISION`]'s filter omits VEHICLE_GROUP), so this is about legibility, not
/// physics: two players boarding the same tick must not materialise inside one another. Pilot 0
/// keeps the exact arena-centre pose solo always had. Unbounded by design: pilot ids are
/// couch-scale (the server allocates lowest-free), so the offset stays well inside the ±10 m
/// arena.
const VEHICLE_SPAWN_SPACING: f32 = 1.0;

/// Spawn `pilot`'s vehicle rigidbody of `kind` at its arena spawn pose — the boarding path
/// ([`manage_vehicles`]). Each pilot's spot is offset along +X by its id so simultaneous
/// boarders don't overlap.
fn spawn_vehicle(commands: &mut Commands, pilot: PilotId, kind: VehicleKind) {
    let mut transform = kind.spawn_transform();
    transform.translation.x += pilot.0 as f32 * VEHICLE_SPAWN_SPACING;
    commands.spawn(vehicle_bundle(
        pilot,
        kind,
        transform,
        kind.spawn_velocity(),
    ));
}

pub fn spawn_ram_vehicle(
    world: &mut World,
    kind: VehicleKind,
    transform: Transform,
    velocity: Velocity,
) -> Entity {
    // Pilot 0: the gate rams one craft with no pilot roster; 0 is the server-auth local player.
    world
        .spawn(vehicle_bundle(PilotId(0), kind, transform, velocity))
        .id()
}

/// The ONE flight force model (see the module docs). For each spawned body, reads ITS pilot's
/// command + the body's own pose and velocity, integrates the throttle lever, and overwrites the
/// body's [`ExternalForce`] with thrust + lift + drag (force) and the body-frame control torque +
/// angular drag (torque). Gravity is rapier's. Runs before the solve; the body's presence (not a
/// flag) gates it, so it acts on whatever [`manage_vehicles`] spawned. A body whose pilot has no
/// command is skipped, defensively: with the chained despawn (a sync point lands between the two
/// systems) a stepped-out pilot's body is gone before forces run, so the skip only guards a
/// future re-ordering — where applying the LAST piloted tick's command instead would thrust a
/// pilotless craft.
fn apply_vehicle_forces(
    controls: Res<VehicleControls>,
    mut q: Query<(&mut Vehicle, &Transform, &Velocity, &mut ExternalForce)>,
) {
    for (mut vehicle, transform, velocity, mut ef) in q.iter_mut() {
        let Some(control) = controls.0.get(&vehicle.pilot) else {
            continue;
        };
        let p = vehicle.kind.params();
        let rot = transform.rotation;
        let forward = rot * Vec3::Z;
        let up = rot * Vec3::Y;
        let v = velocity.linear;

        vehicle.throttle =
            (vehicle.throttle + control.throttle_trim * THROTTLE_TRIM_RATE).clamp(0.0, 1.0);

        // Thrust = LEVER term along the nose (`lever_thrust` 0 ⇒ 0 for the ship) + DIRECT
        // body-frame thrusters `rot · (thrust ⊙ direct_thrust)` (ship strafe/vertical/forward;
        // `direct_thrust` ZERO ⇒ 0 for the plane). One expression, no per-craft branch — each
        // craft's unused term is zeroed by its params.
        let lever = forward * (p.lever_thrust * vehicle.throttle);
        let direct = rot * (control.thrust * p.direct_thrust);
        let thrust = lever + direct;

        let forward_airspeed = v.dot(forward).max(0.0);
        let lift = up * (p.lift * forward_airspeed);

        let speed = v.length();
        let match_damp = if control.match_velocity {
            MATCH_VEL_DAMP
        } else {
            0.0
        };
        let drag = -v * (p.drag_lin + match_damp + p.drag_quad * speed);

        ef.force = thrust + lift + drag;

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

    /// The tests' one pilot (the server-authoritative local player's id).
    const P0: PilotId = PilotId(0);

    /// Board `pilot` into `kind` (neutral axes) — the tests' rising edge.
    fn board(app: &mut App, pilot: PilotId, kind: VehicleKind) {
        app.world_mut()
            .resource_mut::<VehicleControls>()
            .0
            .insert(pilot, PilotCommand::new(kind));
    }

    /// Mutate pilot 0's [`PilotCommand`] in place (it must have [`board`]ed).
    fn set_cmd(app: &mut App, f: impl FnOnce(&mut PilotCommand)) {
        let mut controls = app.world_mut().resource_mut::<VehicleControls>();
        f(controls.0.get_mut(&P0).expect("pilot 0 must have boarded"))
    }

    fn app_with_vehicle(kind: VehicleKind, at: Vec3, vel: Vec3) -> (App, Entity) {
        let mut app = headless_app();
        app.add_plugins(VehiclePlugin);
        // File pilot 0's command so `manage_vehicles` keeps the body we spawn directly below (it
        // despawns a body whose pilot has no entry). Matching kind ⇒ no respawn, our entity
        // persists.
        board(&mut app, P0, kind);
        let transform = Transform::from_translation(at);
        let velocity = Velocity {
            linear: vel,
            angular: Vec3::ZERO,
        };
        let e = app
            .world_mut()
            .spawn(vehicle_bundle(P0, kind, transform, velocity))
            .id();
        app.world_mut()
            .entity_mut(e)
            .get_mut::<Vehicle>()
            .unwrap()
            .throttle = 1.0;
        (app, e)
    }

    fn body(app: &App, e: Entity) -> (&Transform, &Velocity) {
        let ent = app.world().entity(e);
        (
            ent.get::<Transform>().unwrap(),
            ent.get::<Velocity>().unwrap(),
        )
    }

    const FAR: Vec3 = Vec3::new(500.0, 300.0, 500.0);

    /// rl#235 regression: nested (coxa) links were vehicle-transparent — neither
    /// NESTED_COLLISION's filter nor VEHICLE_COLLISION's carried the other side, so a
    /// low ram ghosted through a hip capsule under the shell edge. A vehicle overlapping
    /// a NESTED-grouped capsule must register a narrow-phase contact. The coxa is a
    /// proxy capsule carrying the exact groups `spawn_crab` gives a shell-nested link,
    /// because only the mesh-fitted Sally body nests links — the fallback body this test
    /// env spawns nests none.
    #[test]
    fn vehicle_contacts_nested_coxa() {
        use crate::bot::body::NESTED_COLLISION;
        use crate::bot::headless::tick;
        use bevy_rapier3d::plugin::context::{RapierContextColliders, RapierContextSimulation};

        let mut app = headless_app();
        tick(&mut app, 1);

        let coxa = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Collider::capsule_y(0.03, 0.02),
                NESTED_COLLISION,
                Transform::from_translation(FAR),
            ))
            .id();
        let vehicle = app
            .world_mut()
            .spawn(vehicle_bundle(
                P0,
                VehicleKind::Plane,
                Transform::from_translation(FAR),
                Velocity::default(),
            ))
            .id();
        tick(&mut app, 2);

        let mut q = app
            .world_mut()
            .query::<(&RapierContextColliders, &RapierContextSimulation)>();
        let (cols, sim) = q.single(app.world()).expect("one rapier context");
        let handle = |e: Entity| *cols.entity2collider().get(&e).expect("collider handle");
        assert!(
            sim.narrow_phase
                .contact_pair(handle(vehicle), handle(coxa))
                .is_some_and(|p| p.has_any_active_contact()),
            "vehicle overlapping a coxa produced no contact — NESTED_COLLISION and \
             VEHICLE_COLLISION must include each other's group (rl#235)"
        );
    }

    #[test]
    fn plane_thrust_accelerates_forward() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, 2.0));
        set_cmd(&mut app, |c| c.throttle_trim = 1.0);
        let v0 = body(&app, e).1.linear.z;
        for _ in 0..30 {
            app.update();
        }
        let v1 = body(&app, e).1.linear.z;
        assert!(v1 > v0 + 0.5, "forward speed did not rise: {v0} -> {v1}");
    }

    #[test]
    fn lift_rises_with_airspeed() {
        let dy = |speed: f32| {
            let (mut app, e) =
                app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, speed));
            set_cmd(&mut app, |c| c.throttle_trim = -1.0); // throttle to 0
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

    /// Pins the rl#230 feel-call: level flight lives strictly between spawn speed and the
    /// full-throttle terminal — at spawn speed (2 m/s) lift < weight so the plane settles
    /// instead of ballooning off the runway, while near terminal (4 m/s) lift > weight so
    /// altitude is still winnable with throttle. Both sides sample vertical velocity over a
    /// few zero-throttle ticks, level attitude, so lift vs gravity is the only vertical term.
    #[test]
    fn spawn_speed_sinks_high_speed_climbs() {
        let dvy = |speed: f32| {
            let (mut app, e) =
                app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, speed));
            set_cmd(&mut app, |c| c.throttle_trim = -1.0); // throttle to 0
            let y0 = body(&app, e).1.linear.y;
            for _ in 0..5 {
                app.update();
            }
            body(&app, e).1.linear.y - y0
        };
        let at_spawn = dvy(PLANE_SPAWN_SPEED);
        assert!(
            at_spawn < 0.0,
            "plane must sink at spawn speed (lift < weight), got Δvy={at_spawn}"
        );
        let near_terminal = dvy(4.0);
        assert!(
            near_terminal > 0.0,
            "plane must climb near full-throttle terminal speed, got Δvy={near_terminal}"
        );
    }

    #[test]
    fn positive_pitch_raises_the_nose() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::ZERO);
        set_cmd(&mut app, |c| c.pitch = 1.0);
        for _ in 0..10 {
            app.update();
        }
        let nose_y = (body(&app, e).0.rotation * Vec3::Z).y;
        assert!(
            nose_y > 0.0,
            "positive pitch must raise the nose, got nose.y={nose_y}"
        );
    }

    #[test]
    fn positive_yaw_turns_nose_right() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::ZERO);
        set_cmd(&mut app, |c| c.yaw = 1.0);
        for _ in 0..10 {
            app.update();
        }
        let nose_x = (body(&app, e).0.rotation * Vec3::Z).x;
        assert!(
            nose_x > 0.0,
            "positive yaw must turn the nose right (+X), got nose.x={nose_x}"
        );
    }

    #[test]
    fn pitch_input_loops_the_craft_without_autoright() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::ZERO);
        set_cmd(&mut app, |c| c.pitch = 1.0);
        let mut went_inverted = false;
        for _ in 0..240 {
            app.update();
            let up_y = (body(&app, e).0.rotation * Vec3::Y).y;
            if up_y < -0.3 {
                went_inverted = true;
            }
        }
        assert!(
            went_inverted,
            "held pitch never inverted the craft — a cap or auto-level crept in"
        );
    }

    #[test]
    fn drag_bleeds_speed() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(20.0, 0.0, 0.0));
        set_cmd(&mut app, |c| c.throttle_trim = -1.0); // throttle to 0
        let s0 = body(&app, e).1.linear.length();
        for _ in 0..20 {
            app.update();
        }
        let s1 = body(&app, e).1.linear.length();
        assert!(s1 < s0, "drag did not bleed speed: {s0} -> {s1}");
    }

    #[test]
    fn manage_spawns_and_despawns_one_vehicle() {
        let mut app = headless_app();
        app.add_plugins(VehiclePlugin);
        let count = |app: &mut App| {
            app.world_mut()
                .query::<&Vehicle>()
                .iter(app.world())
                .count()
        };

        app.update();
        assert_eq!(count(&mut app), 0, "no vehicle before piloting");

        board(&mut app, P0, VehicleKind::Plane);
        app.update();
        assert_eq!(count(&mut app), 1, "one vehicle after boarding");

        set_cmd(&mut app, |c| c.kind = VehicleKind::Ship);
        app.update();
        assert_eq!(count(&mut app), 1, "still one vehicle after a kind cycle");
        assert_eq!(
            app.world_mut()
                .query::<&Vehicle>()
                .single(app.world())
                .unwrap()
                .kind,
            VehicleKind::Ship,
            "the body became the cycled kind",
        );

        app.world_mut()
            .resource_mut::<VehicleControls>()
            .0
            .remove(&P0);
        app.update();
        assert_eq!(count(&mut app), 0, "vehicle despawned on step-out");
    }

    /// Per-pilot multiplicity (rl#191): two pilots board on the same tick ⇒ two bodies, each the
    /// kind ITS pilot chose, at distinct spawn spots (the per-pilot offset); one pilot stepping
    /// out despawns only ITS craft. Spawn/despawn bookkeeping only — a couple of ticks, so the
    /// crab standing at the arena origin never comes into play.
    #[test]
    fn each_pilot_gets_its_own_craft() {
        let p1 = PilotId(1);
        let mut app = headless_app();
        app.add_plugins(VehiclePlugin);
        // Warm the clock: the first update's zero delta runs no FixedUpdate (same dance as
        // `manage_spawns_and_despawns_one_vehicle`).
        app.update();
        board(&mut app, P0, VehicleKind::Ship);
        board(&mut app, p1, VehicleKind::Plane);
        app.update();

        let crafts: Vec<(PilotId, VehicleKind, f32)> = app
            .world_mut()
            .query::<(&Vehicle, &Transform)>()
            .iter(app.world())
            .map(|(v, t)| (v.pilot, v.kind, t.translation.x))
            .collect();
        assert_eq!(crafts.len(), 2, "one body per boarded pilot");
        let by = |p: PilotId| crafts.iter().find(|(o, ..)| *o == p).unwrap();
        assert_eq!(by(P0).1, VehicleKind::Ship, "pilot 0 got ITS chosen kind");
        assert_eq!(by(p1).1, VehicleKind::Plane, "pilot 1 got ITS chosen kind");
        assert_ne!(
            by(P0).2,
            by(p1).2,
            "pilots spawn at distinct spots (per-pilot offset)"
        );

        // Pilot 1 steps out: only ITS body despawns.
        app.world_mut()
            .resource_mut::<VehicleControls>()
            .0
            .remove(&p1);
        app.update();
        let left: Vec<PilotId> = app
            .world_mut()
            .query::<&Vehicle>()
            .iter(app.world())
            .map(|v| v.pilot)
            .collect();
        assert_eq!(
            left,
            vec![P0],
            "stepping out despawns only that pilot's craft"
        );
    }

    /// Each command drives only ITS pilot's body: two ships in free space (FAR — hermetic, like
    /// the sibling force tests), pilot 0 burning forward while pilot 1 idles. Only pilot 0's ship
    /// gains forward speed — the per-body command lookup, the property remote pilots ride on.
    #[test]
    fn a_command_drives_only_its_own_pilots_craft() {
        let p1 = PilotId(1);
        let mut app = headless_app();
        app.add_plugins(VehiclePlugin);
        board(&mut app, P0, VehicleKind::Ship);
        board(&mut app, p1, VehicleKind::Ship);
        set_cmd(&mut app, |c| c.thrust = Vec3::new(0.0, 0.0, 1.0));
        // Spawn both through the SHARED bundle at FAR (matching entries above, so
        // `manage_vehicles` keeps them and spawns no third).
        let spawn = |app: &mut App, pilot: PilotId, at: Vec3| {
            app.world_mut()
                .spawn(vehicle_bundle(
                    pilot,
                    VehicleKind::Ship,
                    Transform::from_translation(at),
                    Velocity::default(),
                ))
                .id()
        };
        let e0 = spawn(&mut app, P0, FAR);
        let e1 = spawn(&mut app, p1, FAR + Vec3::X * 5.0);
        for _ in 0..60 {
            app.update();
        }
        let vz = |app: &App, e| body(app, e).1.linear.z;
        assert!(
            vz(&app, e0) > 0.5,
            "pilot 0's thrust drives pilot 0's ship: vz={}",
            vz(&app, e0)
        );
        assert!(
            vz(&app, e1).abs() < 0.1,
            "pilot 1's idle ship must not pick up pilot 0's thrust: vz={}",
            vz(&app, e1)
        );
    }

    #[test]
    fn ship_direct_forward_thrust_accelerates() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::ZERO);
        set_cmd(&mut app, |c| c.thrust = Vec3::new(0.0, 0.0, 1.0));
        let v0 = body(&app, e).1.linear.z;
        for _ in 0..90 {
            app.update();
        }
        let v1 = body(&app, e).1.linear.z;
        assert!(
            v1 > v0 + 0.5,
            "ship forward thruster did not accelerate: {v0} -> {v1}"
        );
    }

    #[test]
    fn ship_lateral_thrust_strafes() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::ZERO);
        set_cmd(&mut app, |c| c.thrust = Vec3::new(1.0, 0.0, 0.0));
        let v0 = body(&app, e).1.linear.x;
        for _ in 0..90 {
            app.update();
        }
        let v1 = body(&app, e).1.linear.x;
        assert!(
            v1 > v0 + 0.5,
            "ship lateral thruster did not strafe +X: {v0} -> {v1}"
        );
    }

    #[test]
    fn ship_is_zero_g_but_plane_falls() {
        let fall = |kind: VehicleKind| {
            let (mut app, e) = app_with_vehicle(kind, FAR, Vec3::ZERO);
            app.world_mut()
                .entity_mut(e)
                .get_mut::<Vehicle>()
                .unwrap()
                .throttle = 0.0;
            for _ in 0..30 {
                app.update();
            }
            body(&app, e).1.linear.y
        };
        assert!(
            fall(VehicleKind::Ship).abs() < 0.5,
            "ship must float (zero-g), not fall"
        );
        assert!(
            fall(VehicleKind::Plane) < -1.0,
            "plane must fall under gravity"
        );
    }

    #[test]
    fn ship_match_velocity_brakes_harder_than_coast() {
        let drop = |brake: bool| {
            let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::new(10.0, 0.0, 0.0));
            set_cmd(&mut app, |c| c.match_velocity = brake);
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
