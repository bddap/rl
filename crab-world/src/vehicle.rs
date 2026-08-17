use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::bot::body::VEHICLE_COLLISION;

const VEHICLE_HALF: Vec3 = Vec3::new(0.11, 0.035, 0.17);

const VEHICLE_DENSITY: f32 = 50.0;

/// The craft body's mass (box volume × density — mirrors the cuboid
/// `ColliderMassProperties::Density`; revisit on a collider shape change) — the
/// drag-stability cap below.
const VEHICLE_MASS: f32 = VEHICLE_DENSITY * 8.0 * VEHICLE_HALF.x * VEHICLE_HALF.y * VEHICLE_HALF.z;

/// The craft's [`crate::physics::brake_coeff_max`] (one-step momentum cancel) on the
/// drag + match-velocity damp coefficient. The cap only engages past ~220 m/s (plane)
/// / ~277 m/s (ship) — 20×+ any flyable speed — so tuned feel is untouched.
const BRAKE_COEFF_MAX: f32 = crate::physics::brake_coeff_max(VEHICLE_MASS);

/// How far below the local surface a craft must sink before rescue: past the
/// heightfield's one-sided collider nothing pushes it back up (the crab's Buried
/// case), and at rl#339 escape speeds a single substep tunnels straight through.
/// Generous vs the crab's 2 m: a craft is never legitimately underground at all.
const CRAFT_BELOW_TERRAIN_RESCUE_M: f32 = 20.0;

/// Pulled-in margin for the escape park — see [`rescue_lost_crafts`].
const CRAFT_EDGE_PARK_MARGIN_M: f32 = 2.0;

/// Craft ground handling (thrust scales, ground roll, the rl#307 grounded-ship
/// feel) was tuned while the terrain had no Friction component — a 0.5↔0.5 Average
/// pair. The rl#318 grip fix raised the ground side
/// (`physics::world::GROUND_FRICTION` 2.5), which would have silently tripled
/// craft–ground friction (a plane's breakaway ~32%→~96% throttle); `Min` at the old
/// default keeps the craft–ground pair at its tuned 0.5 — `Average` could not, the
/// raised ground side dominates any craft coefficient. Side effect accepted: `Min`
/// also caps craft↔foot pairs at 0.5 (was 1.0 avg with `FOOT_FRICTION` 1.5; other
/// crab parts already paired at 0.5) — those contacts are ram/bump impulses, not
/// grip. Raise deliberately, not by drift.
pub(crate) const VEHICLE_FRICTION: Friction = Friction {
    coefficient: 0.5,
    combine_rule: CoefficientCombineRule::Min,
};

const THROTTLE_TRIM_RATE: f32 = 0.03;

const MATCH_VEL_DAMP: f32 = 0.9;

/// The plane's settled full-throttle top speed — the fastest thing in the game, THE
/// number every downstream band (wind/engine audio, ship comparisons) normalizes
/// against. Not free-floating: `full_throttle_top_speed_settles_in_band` measures the
/// realized settle and pins it within ±5% of this constant, so a params change that
/// moves the real top speed without updating this fails the ruler.
pub const PLANE_TOP_SPEED_MPS: f32 = 9.13;

// rl#379 (owner: "2x current"): every equilibrium speed doubled by scaling the
// speed-shaped terms — thrust ×2 + quadratic drag ÷2 doubles the thrust-vs-drag
// terminal (measured 4.64 → 9.13 m/s settled, `full_throttle_top_speed_settles_in_band`),
// lift ÷2 doubles the lift-vs-weight level-flight speed, and the full-authority
// speed doubles with cruise. Linear drag, grip, and every torque stay put: grip's
// force is already ∝ speed (same alignment τ), angular rates keep the tuned feel
// (turn RADIUS doubles with speed — that's the point of a faster plane).
const PLANE: VehicleParams = VehicleParams {
    lever_thrust: 8.0,
    direct_thrust: Vec3::ZERO,
    // 0.45 puts level flight at ~5.7 m/s — above the slow-flight band (a
    // freshly-boarded craft can't free-balloon) and ~62% of the full-throttle terminal
    // ~9.1 m/s, so climb costs throttle. At
    // 1.8 the plane out-lifted its ~2.6 N weight from 1.4 m/s (owner: "Bernoulli overdone",
    // rl#230); the level-flight band is pinned by `slow_flight_sinks_high_speed_climbs`.
    lift: 0.45,
    // Alignment time-constant ≈ mass/grip ≈ 0.3 s — velocity follows the nose well inside
    // one turn's sweep, so a coordinated turn stays near-aligned and pays only the
    // second-order grip loss (rl#255).
    grip: 0.8,
    drag_lin: 0.2,
    drag_quad: 0.075,
    angular_drag: 0.06,
    pitch_torque: 0.2,
    roll_torque: 0.3,
    yaw_torque: 0.1,
    // Full control authority just below the ~5.7 m/s level-flight speed, so cruise
    // and everything above it keeps the tuned feel (rl#382).
    control_speed_full: 5.0,
};

const SHIP_AIM_TORQUE: f32 = 0.015;
// rl#307 feel spec (owner): rotational inertia — a released spin persists and is bled
// by angular drag, never clamped; gravity "a bit"; drag "a bunch" (watercraft-heavy).
// The knob for the inertia FEEL is angular_drag: spin-down τ = I/angular_drag, and the
// craft box's yaw inertia is tiny (~0.0036 kg·m²), so the old 0.07 stopped a spin in
// ~50 ms — indistinguishable from a clamp. 0.005 puts yaw τ ≈ 0.7 s (roll, the bumper
// spin, ≈ 0.23 s on its slimmer axis); torques are scaled down with it so the
// terminal rates stay where they were (τ·rate ≈ the old feel, plus carry).
// Linear drag is 3× the old float (coast from top speed halves in ~1 s) with thrust
// raised to hold top speed ≈ 2.5 m/s, so "a bunch" reads as heavy stopping, not slow.
// The vertical thruster is the STRONGEST (rl#316): direct thrust is body-frame, so any
// axis stronger than the weight lets an attitude cheat hold altitude (nose-up converts
// forward thrust into lift) and gravity reads as idle-only. Weight 0.77 N > forward
// 0.75 N > strafe 0.55 N makes sustained lift-by-attitude impossible at any pitch/roll;
// 1.19 keeps the net climb force ≈ the rl#307 0.42 N.
const SHIP: VehicleParams = VehicleParams {
    lever_thrust: 0.0,
    direct_thrust: Vec3::new(0.55, 1.19, 0.75),
    lift: 0.0,
    grip: 0.0,
    drag_lin: 0.15,
    drag_quad: 0.06,
    angular_drag: 0.005,
    pitch_torque: SHIP_AIM_TORQUE,
    roll_torque: SHIP_AIM_TORQUE * 0.25,
    yaw_torque: SHIP_AIM_TORQUE,
    control_speed_full: 0.0,
};

/// How far above the boarding spot's ground point the craft's collider bottom
/// materialises: the craft's box is bigger than the walker, so an un-nudged in-place
/// spawn would intersect the ground and get a depenetration kick (rl#258).
const GROUND_CLEARANCE: f32 = 0.01;

#[derive(Clone, Copy)]
struct VehicleParams {
    lever_thrust: f32,
    direct_thrust: Vec3,
    lift: f32,
    grip: f32,
    drag_lin: f32,
    drag_quad: f32,
    angular_drag: f32,
    pitch_torque: f32,
    roll_torque: f32,
    yaw_torque: f32,
    /// Airspeed at which aerodynamic control surfaces reach full torque authority.
    /// Authority scales with dynamic pressure, (|v|/this)² capped at 1 — a
    /// near-stationary plane cannot yaw/pitch/roll in place (rl#382). 0 ⇒ the
    /// controls are thrusters, not surfaces (ship): full authority at any speed.
    control_speed_full: f32,
}

/// Discriminants are the wire bytes (0 reserved for "no craft" in the pilot-intent frame).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum VehicleKind {
    #[default]
    Plane = 1,
    Ship = 2,
}

impl VehicleKind {
    pub const ALL: [Self; 2] = [VehicleKind::Plane, VehicleKind::Ship];

    /// The craft's ONE wire byte, shared by every codec that ships a kind (the pilot-intent
    /// frame and the articulation vehicle poses) so two mappings can't drift.
    pub fn wire_byte(self) -> u8 {
        self as u8
    }

    pub fn from_wire_byte(b: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.wire_byte() == b)
    }

    fn params(self) -> VehicleParams {
        match self {
            VehicleKind::Plane => PLANE,
            VehicleKind::Ship => SHIP,
        }
    }

    fn gravity_scale(self) -> f32 {
        match self {
            VehicleKind::Plane => 1.0,
            // "A bit" of gravity that reads as gravity IN MOTION, not only at idle
            // (rl#316, owner: "ship shouldn't only sink when holding still"). The
            // weight must exceed every non-vertical thruster — 0.30 weighs ~0.77 N
            // against the 0.75 N forward / 0.55 N strafe pair — or a pitched/rolled
            // attitude converts body-frame thrust into lift and cruise never loses
            // altitude. The idle side stays a perceptible ramp (rl#313): release→
            // terminal ≈ 1 s building to ≈ 2.5 m/s, far gentler than the plane's
            // free fall. Hovering costs ~65% of the vertical thruster, which is
            // raised to keep the net climb authority at the rl#307 feel.
            VehicleKind::Ship => 0.30,
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

/// The boarding player's walker state in the arena frame — where a fresh craft
/// materialises (rl#258: the vehicle appears where the player is, one entity swaps form,
/// velocity conserved). The net bridge authors it from the authoritative sim;
/// [`spawn_vehicle`] only adds the ground-clearance nudge.
#[derive(Clone, Copy, Debug)]
pub struct Boarding {
    /// The walker's ground point (arena m).
    pub pos: Vec3,
    /// Facing, radians about +Y — the craft's nose starts along it.
    pub yaw: f32,
    /// The walker's velocity (arena m/s), conserved through the transform.
    pub velocity: Vec3,
}

#[derive(Clone, Copy)]
pub struct PilotCommand {
    /// Which craft to be (the player picks one per board code, rl#330). A change while
    /// piloting morphs the body in place ([`manage_vehicles`]).
    pub kind: VehicleKind,
    /// Where the pilot's body was when this command was authored — read only on the
    /// spawn edge (a pilot with no body yet).
    pub boarding: Boarding,
    pub throttle_trim: f32,
    pub thrust: Vec3,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub match_velocity: bool,
}

impl PilotCommand {
    /// Board `kind` at `boarding` with every axis neutral — the boarding-edge command
    /// (the per-tick axes are overwritten by the bridge each tick anyway). The one
    /// constructor, so choosing a craft and where it materialises is always explicit.
    pub fn new(kind: VehicleKind, boarding: Boarding) -> Self {
        Self {
            kind,
            boarding,
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
/// layer writes — the host authors an entry for EVERY filed pilot intent, its own and remote
/// pilots' alike (intents ride each client's input submission), so every craft has a real
/// body in the host's one physics world and the crab hunts boarded players wherever they fly
/// (rl#265).
#[derive(Resource, Default)]
pub struct VehicleControls(pub std::collections::BTreeMap<PilotId, PilotCommand>);

pub struct VehiclePlugin;

/// The boarding spawn edge ([`manage_vehicles`]) as an orderable seam.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct VehicleManageSet;

impl Plugin for VehiclePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VehicleControls>().add_systems(
            FixedUpdate,
            (
                manage_vehicles.in_set(VehicleManageSet),
                rescue_lost_crafts,
                apply_vehicle_forces,
            )
                .chain()
                .before(PhysicsSet::SyncBackend),
        );
    }
}

/// Keep the spawned bodies matched to [`VehicleControls`]: a new entry spawns that pilot's craft
/// at its command's [`Boarding`] pose; a kind change MORPHS the existing body in place — same
/// entity, pose and velocity carried through (rl#258), only the throttle lever resetting to the
/// known zero a fresh craft starts at; a removed entry despawns it. At most one body per pilot,
/// provided bodies enter play only through this system (the headless gates spawn ram craft into
/// worlds WITHOUT the plugin): a body only spawns for a pilot with no matching body. Each body is
/// a dynamic box with the [`VEHICLE_COLLISION`] groups (arena + every crab), so it bounces off
/// the walls and strikes Sally.
fn manage_vehicles(
    mut commands: Commands,
    controls: Res<VehicleControls>,
    terrain: Res<crate::terrain::Terrain>,
    mut existing: Query<(Entity, &mut Vehicle, &mut GravityScale)>,
) {
    let mut matched = std::collections::BTreeSet::new();
    for (e, mut v, mut gravity) in existing.iter_mut() {
        match controls.0.get(&v.pilot) {
            Some(cmd) => {
                if cmd.kind != v.kind {
                    // Everything kind-dependent in [`vehicle_bundle`] must be re-derived
                    // here, or a morphed craft drifts from a fresh one.
                    v.kind = cmd.kind;
                    v.throttle = 0.0;
                    *gravity = GravityScale(cmd.kind.gravity_scale());
                }
                matched.insert(v.pilot);
            }
            // The pilot stepped out: the craft despawns (the walker is the body again).
            None => commands.entity(e).despawn(),
        }
    }
    for (&pilot, cmd) in &controls.0 {
        if !matched.contains(&pilot) {
            spawn_vehicle(&mut commands, &terrain, pilot, cmd);
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
        VEHICLE_FRICTION,
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

/// One cuboid of a craft's rendered silhouette, in the craft's body frame (+Z is the nose).
pub struct VehiclePart {
    pub offset: Vec3,
    pub half: Vec3,
}

/// One rocket nozzle of a craft's rendered exhaust (rl#308), in the body frame (+Z is the
/// nose). Render-only data — geometry lives here beside [`VehicleKind::silhouette`] so the
/// hull and its nozzles can't drift apart, but physics never reads it.
#[derive(Clone, Copy)]
pub struct Nozzle {
    /// Where the plume roots, on the hull surface (m, body frame).
    pub offset: Vec3,
    /// Unit direction of the thrust this nozzle produces; the plume exits along `-axis`.
    pub axis: Vec3,
    /// Plume base radius (m).
    pub radius: f32,
    /// Plume length at full thrust (m).
    pub max_len: f32,
}

/// The plane's fuselage and the ship's starboard pontoon in fraction space (offset,
/// half) — the cuboids nozzles root on, named once and shared by
/// [`VehicleKind::silhouette`] and [`VehicleKind::nozzles`] so a hull reshape moves the
/// nozzles with it (the port pontoon is the x-mirror).
const PLANE_FUSELAGE: (Vec3, Vec3) = (Vec3::ZERO, Vec3::new(0.2, 0.8, 1.0));
const SHIP_PONTOON: (Vec3, Vec3) = (Vec3::new(0.74, -0.4, -0.1), Vec3::new(0.25, 0.5, 0.75));

impl VehicleKind {
    /// The craft's rendered shape: a few cuboids, every one inside the one collider box.
    /// Offsets and half-extents are FRACTIONS of [`VEHICLE_HALF`] (each axis's |offset| +
    /// half ≤ 1), so the mesh can never visually exceed the physics body and a collider
    /// resize rescales the model with it (rl#260); `silhouettes_stay_inside_the_collider`
    /// is the backstop.
    pub fn silhouette(self) -> Vec<VehiclePart> {
        let part = |offset: Vec3, half: Vec3| VehiclePart {
            offset: offset * VEHICLE_HALF,
            half: half * VEHICLE_HALF,
        };
        match self {
            // Slim full-length fuselage, full-span main wing, tailplane + fin at the stern.
            VehicleKind::Plane => vec![
                part(PLANE_FUSELAGE.0, PLANE_FUSELAGE.1),
                part(Vec3::new(0.0, 0.1, 0.18), Vec3::new(1.0, 0.15, 0.26)),
                part(Vec3::new(0.0, 0.2, -0.85), Vec3::new(0.5, 0.12, 0.12)),
                part(Vec3::new(0.0, 0.54, -0.85), Vec3::new(0.05, 0.45, 0.12)),
            ],
            // Catamaran: a broad hull slung low between two pontoons, bridge aft.
            VehicleKind::Ship => vec![
                part(Vec3::new(0.0, -0.25, 0.0), Vec3::new(0.6, 0.6, 1.0)),
                part(SHIP_PONTOON.0, SHIP_PONTOON.1),
                part(SHIP_PONTOON.0 * Vec3::new(-1.0, 1.0, 1.0), SHIP_PONTOON.1),
                part(Vec3::new(0.0, 0.58, -0.35), Vec3::new(0.35, 0.4, 0.3)),
            ],
        }
    }

    /// The craft's exhaust nozzles (rl#308): where a plume roots and which thrust axis
    /// fires it. Every nozzle sits on a face of its named hull cuboid ([`PLANE_FUSELAGE`] /
    /// [`SHIP_PONTOON`]), and sizes are hull fractions, so a reshape or collider resize
    /// moves and rescales the plumes with the craft (rl#260). The plane has one engine on
    /// the throttle lever; the ship's RCS pairs mirror its per-axis direct thrusters — a
    /// nozzle exists only on an axis [`VehicleParams`] can actually thrust
    /// (`nozzles_fire_only_thrustable_axes`).
    pub fn nozzles(self) -> Vec<Nozzle> {
        // Radius and length are fractions of the hull's smallest half-extent.
        let nozzle = |offset: Vec3, axis: Vec3, radius: f32, max_len: f32| Nozzle {
            offset: offset * VEHICLE_HALF,
            axis,
            radius: radius * VEHICLE_HALF.y,
            max_len: max_len * VEHICLE_HALF.y,
        };
        match self {
            // One main engine at the fuselage stern.
            VehicleKind::Plane => {
                let (c, h) = PLANE_FUSELAGE;
                vec![nozzle(c - h * Vec3::Z, Vec3::Z, 0.6, 8.0)]
            }
            // Per starboard pontoon face — main at the stern, retro at the bow, lift on
            // the keel, drop on the deck, strafe outboard — then the port x-mirror.
            VehicleKind::Ship => {
                let (c, h) = SHIP_PONTOON;
                let starboard = [
                    (c - h * Vec3::Z, Vec3::Z, 0.5, 5.7),
                    (c + h * Vec3::Z, Vec3::NEG_Z, 0.35, 2.6),
                    (c - h * Vec3::Y, Vec3::Y, 0.4, 3.1),
                    (c + h * Vec3::Y, Vec3::NEG_Y, 0.35, 2.6),
                    (c + h * Vec3::X, Vec3::NEG_X, 0.35, 2.6),
                ];
                let mirror = Vec3::new(-1.0, 1.0, 1.0);
                starboard
                    .into_iter()
                    .flat_map(|(offset, axis, radius, max_len)| {
                        [
                            nozzle(offset, axis, radius, max_len),
                            nozzle(offset * mirror, axis * mirror, radius, max_len),
                        ]
                    })
                    .collect()
            }
        }
    }
}

impl VehicleParams {
    /// 1.0 on each body axis the direct thrusters can fire — ONE encoding of the law
    /// (the force law needs no mask: a zero param zeroes its own term).
    fn direct_axes(&self) -> Vec3 {
        Vec3::select(self.direct_thrust.cmpgt(Vec3::ZERO), Vec3::ONE, Vec3::ZERO)
    }

    /// 1.0 when the throttle lever can thrust (along the nose, +Z).
    fn lever_live(&self) -> f32 {
        if self.lever_thrust > 0.0 { 1.0 } else { 0.0 }
    }

    /// Every axis the craft can thrust along — what `nozzles_fire_only_thrustable_axes`
    /// pins nozzle placement against.
    #[cfg(test)]
    fn live_axes(&self) -> Vec3 {
        self.direct_axes().max(Vec3::Z * self.lever_live())
    }
}

impl Vehicle {
    /// The craft's body-frame thrust COMMAND as per-axis fractions in [-1, 1] — the
    /// exhaust-intensity source (rl#308): the plane's throttle lever along +Z, the ship's
    /// per-axis direct thrusters. A nozzle's brightness is this dotted with its axis.
    /// Purely observational — [`apply_vehicle_forces`] reads the raw command, never this.
    pub fn thrust_fraction(&self, control: &PilotCommand) -> Vec3 {
        let p = self.kind.params();
        (control.thrust * p.direct_axes() + Vec3::Z * (p.lever_live() * self.throttle))
            .clamp(Vec3::NEG_ONE, Vec3::ONE)
    }
}

/// Spawn `pilot`'s vehicle rigidbody where its walker stands — the boarding path
/// ([`manage_vehicles`]): the craft materialises at the command's [`Boarding`] pose with the
/// walker's facing and velocity (rl#258), lifted just enough that the collider clears the
/// ground ([`clear_of_ground`] — the TERRAIN surface at the boarding spot, not y=0;
/// rl#283: a flat clamp buries the craft inside a hill). Since rl#281 stage 6 the
/// production [`Boarding`] author (net's sim bridge) poses y on the surface itself, so
/// the floor is a backstop, not the lift.
fn spawn_vehicle(
    commands: &mut Commands,
    terrain: &crate::terrain::TerrainGrid,
    pilot: PilotId,
    cmd: &PilotCommand,
) {
    let Boarding { pos, yaw, velocity } = cmd.boarding;
    let pos = clear_of_ground(pos, GROUND_CLEARANCE, terrain);
    commands.spawn(vehicle_bundle(
        pilot,
        cmd.kind,
        Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(yaw)),
        Velocity {
            linear: velocity,
            angular: Vec3::ZERO,
        },
    ));
}

/// Lift a craft pose just enough that its collider clears the terrain surface at its
/// xz (plus `clearance`); never pulls a higher pose down (a flying craft is
/// untouched). ONE formula for every path that poses a craft against ground it didn't
/// measure: the boarding spawn above and the round-RESTART park
/// ([`park_vehicles_near`]), both with [`GROUND_CLEARANCE`] — an uncleaned placement
/// can embed a craft in a hillside (rl#281 stage 6, rl#283).
pub fn clear_of_ground(pos: Vec3, clearance: f32, terrain: &crate::terrain::TerrainGrid) -> Vec3 {
    let floor = terrain.height(pos.x, pos.z) + VEHICLE_HALF.y + clearance;
    Vec3::new(pos.x, pos.y.max(floor), pos.z)
}

/// The round-RESTART park (rl#322): teleport every surviving craft onto a ring of
/// `radius` around `center` (Sally's fresh origin, planar). A restart re-pins the
/// whole round at the new sim spawns, and a craft left stranded across the map drags
/// its boarded pilot's shadow with it, defeating the reset. Crafts take the ring
/// slots FARTHEST from the `avoid` points (the other crabs' origins, in a multi-crab
/// round) — the ring itself keeps them clear of the center crab — dealt in
/// (pilot, entity) order over evenly-pitched candidates, so no two crafts share a
/// slot. Each lands on its LOCAL surface ([`clear_of_ground`]) at rest: velocities
/// AND the throttle lever zeroed — a lever held through the teleport would fly the
/// craft straight back off the ring — nose pointing outward, so throttle-up leaves
/// the crab rather than ramming her. Placement-only: no forces, colliders, or params
/// touched.
pub fn park_vehicles_near(
    world: &mut World,
    center: Vec2,
    radius: f32,
    avoid: &[Vec2],
    terrain: &crate::terrain::TerrainGrid,
) {
    let mut crafts: Vec<(PilotId, Entity)> = world
        .query::<(Entity, &Vehicle)>()
        .iter(world)
        .map(|(e, v)| (v.pilot, e))
        .collect();
    crafts.sort();
    // More candidates than crafts, so there are clear slots to prefer; the floor of
    // 16 bounds adjacent picks no closer than 2·r·sin(π/16) ≈ 0.39·r — over a craft
    // length at any near-Sally radius, even when every craft lands on one clear arc.
    let candidates = crafts.len().max(16);
    let mut slots: Vec<(f32, f32)> = (0..candidates)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / candidates as f32;
            let slot = center + radius * Vec2::new(angle.sin(), angle.cos());
            let clearance = avoid
                .iter()
                .map(|a| a.distance(slot))
                .fold(f32::INFINITY, f32::min);
            (angle, clearance)
        })
        .collect();
    slots.sort_by(|a, b| b.1.total_cmp(&a.1));
    slots.truncate(crafts.len());
    for ((_, e), (yaw, _)) in crafts.into_iter().zip(slots) {
        // Nose = body +Z rotated by `yaw` = (sin, cos) — the same direction as the
        // slot's radial, so the craft points outward by construction.
        let slot = center + radius * Vec2::new(yaw.sin(), yaw.cos());
        let pos = clear_of_ground(terrain.place(slot, 0.0), GROUND_CLEARANCE, terrain);
        let mut ent = world.entity_mut(e);
        *ent.get_mut::<Transform>().unwrap() =
            Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(yaw));
        *ent.get_mut::<Velocity>().unwrap() = Velocity::default();
        ent.get_mut::<Vehicle>().unwrap().throttle = 0.0;
    }
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

/// Park a craft physics can no longer recover — the vehicle half of
/// [`rescue_lost_crabs`](crate::bot::rescue_lost_crabs) (rl#339): a non-finite pose,
/// or a finite one off the terrain tile, where the ground collider ends
/// (`terrain.rs` — the rl#281 world-bounds gap), so nothing ever brings an escapee
/// back; meanwhile its shadow keeps streaming 1e8 m coordinates into the render
/// path, which is what froze the TV both times in rl#339. Parked at rest on the
/// local surface (velocities and lever zeroed, facing kept when finite) via the same
/// [`clear_of_ground`] every other placement uses; a non-finite craft has no
/// position left and parks at the arena origin.
fn rescue_lost_crafts(
    terrain: Res<crate::terrain::Terrain>,
    mut q: Query<(&mut Vehicle, &mut Transform, &mut Velocity)>,
) {
    // Park strictly inside the edge, not on it: a craft whose COM lands exactly on
    // the collider boundary can tip off the unsupported half and re-trigger forever.
    let bound = terrain.half_span_min() - CRAFT_EDGE_PARK_MARGIN_M;
    for (mut vehicle, mut transform, mut velocity) in q.iter_mut() {
        let pos = transform.translation;
        let finite = pos.is_finite() && transform.rotation.is_finite();
        let below = finite && terrain.height(pos.x, pos.z) - pos.y > CRAFT_BELOW_TERRAIN_RESCUE_M;
        if finite && !below && pos.x.abs() <= bound && pos.z.abs() <= bound {
            continue;
        }
        let anchor = if finite {
            Vec2::new(pos.x.clamp(-bound, bound), pos.z.clamp(-bound, bound))
        } else {
            Vec2::ZERO
        };
        // Yaw-only pose, like the restart park: the nose survives, any tumble doesn't.
        let nose = transform.rotation * Vec3::Z;
        let yaw = if finite { nose.x.atan2(nose.z) } else { 0.0 };
        let parked = clear_of_ground(terrain.place(anchor, 0.0), GROUND_CLEARANCE, &terrain);
        *transform = Transform::from_translation(parked).with_rotation(Quat::from_rotation_y(yaw));
        *velocity = Velocity::default();
        vehicle.throttle = 0.0;
        if finite {
            // Reachable by legitimate long flight (nothing else stops a craft at the
            // edge), so a warn — the fault tier is non-finite, as in the crab rescue.
            warn!(
                "rescue_lost_crafts: pilot {:?} craft left the world at translation={pos:?} \
                 — parked at rest at {parked:?} (rl#339)",
                vehicle.pilot
            );
        } else {
            error!(
                "rescue_lost_crafts: pilot {:?} craft went NON-FINITE — parked at rest at \
                 {parked:?} (rl#339; physics-correctness fault)",
                vehicle.pilot
            );
        }
    }
}

/// The ONE flight force model. For each spawned body, reads ITS pilot's
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
        let drag = -v * (p.drag_lin + match_damp + p.drag_quad * speed).min(BRAKE_COEFF_MAX);

        // Grip: the sideslip force of the wing+fuselage, as a spring from v to
        // `forward·|v|` — it mostly REDIRECTS momentum toward the nose; the along-track
        // component grip·|v|·(cosθ−1) ≤ 0 can never add speed and is second-order in the
        // slip angle θ, so a coordinated turn carries its speed instead of skidding it
        // off, and a sink converts into airspeed instead of a sideways fall (rl#255).
        let grip = (forward * speed - v) * p.grip;

        // Surface torque comes from airflow over the control surfaces — dynamic
        // pressure ~ ½ρv² — so scale by (|v|/full-authority speed)², capped at 1.
        // Total |v|, not the forward component: a diving or tail-sliding craft still
        // has airflow over its surfaces, so it stays recoverable (rl#382).
        let authority = if p.control_speed_full > 0.0 {
            (speed * speed / (p.control_speed_full * p.control_speed_full)).min(1.0)
        } else {
            1.0
        };
        let body_torque = authority
            * Vec3::new(
                -control.pitch * p.pitch_torque,
                control.yaw * p.yaw_torque,
                -control.roll * p.roll_torque,
            );
        // A parked craft must be allowed to SLEEP (rl#377): every write here marks
        // the component changed, and bevy_rapier force-wakes the body on any Changed
        // ExternalForce — resetting rapier's sleep timer each tick. At rest the
        // passive terms (drag, grip, lift, angular drag) feed on the contact solver's
        // own rest-noise velocities, re-deriving a not-quite-equal force every tick,
        // so the loop held a grounded craft awake forever — and at far-coordinate
        // locales (f32 position ULP ~1 mm at ~13 km) the awake contact solve rings at
        // up to ~1 rad/s of orientation chatter: the visible parked wiggle. Magnitude
        // can't separate noise from signal (settle-transient drag torques overlap
        // real control torques), so split by SOURCE: with the command neutral and
        // both velocities inside rapier's sleep thresholds, the passive terms are
        // rest noise by definition — hold the force at exactly zero and skip the
        // write, so sleep engages and rest is rest. Dissipative terms can never
        // sustain motion, so dropping them only inside the sleep band changes no
        // trajectory; any commanded input writes (and thus wakes) as before.
        let command_neutral = thrust == Vec3::ZERO && body_torque == Vec3::ZERO;
        let below_sleep_band = speed < SLEEP_LINEAR_THRESHOLD
            && velocity.angular.length_squared()
                < SLEEP_ANGULAR_THRESHOLD * SLEEP_ANGULAR_THRESHOLD;
        let next = if command_neutral && below_sleep_band {
            ExternalForce::default()
        } else {
            ExternalForce {
                force: thrust + lift + drag + grip,
                torque: rot * body_torque - velocity.angular * p.angular_drag,
            }
        };
        if *ef != next {
            *ef = next;
        }
    }
}

/// Rapier's default sleep thresholds (`RigidBodyActivation`; linear is normalized by
/// the 1 m length unit) — the band inside which the engine itself deems the body
/// rest-eligible, restated so [`apply_vehicle_forces`]'s no-write rest rule tracks
/// the engine's own sleep criterion.
const SLEEP_LINEAR_THRESHOLD: f32 = 0.4;
const SLEEP_ANGULAR_THRESHOLD: f32 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::headless::{flat_headless_app, headless_app};

    /// The tests' one pilot (the server-authoritative local player's id).
    const P0: PilotId = PilotId(0);

    /// A test boarding pose: standing at `pos`, facing +Z, at rest.
    fn standing_at(pos: Vec3) -> Boarding {
        Boarding {
            pos,
            yaw: 0.0,
            velocity: Vec3::ZERO,
        }
    }

    /// Board `pilot` into `kind` at `pos` (neutral axes) — the tests' rising edge.
    fn board_at(app: &mut App, pilot: PilotId, kind: VehicleKind, pos: Vec3) {
        app.world_mut()
            .resource_mut::<VehicleControls>()
            .0
            .insert(pilot, PilotCommand::new(kind, standing_at(pos)));
    }

    /// [`board_at`] at the arena origin.
    fn board(app: &mut App, pilot: PilotId, kind: VehicleKind) {
        board_at(app, pilot, kind, Vec3::ZERO);
    }

    /// Mutate pilot 0's [`PilotCommand`] in place (it must have [`board`]ed).
    fn set_cmd(app: &mut App, f: impl FnOnce(&mut PilotCommand)) {
        let mut controls = app.world_mut().resource_mut::<VehicleControls>();
        f(controls.0.get_mut(&P0).expect("pilot 0 must have boarded"))
    }

    fn app_with_vehicle(kind: VehicleKind, at: Vec3, vel: Vec3) -> (App, Entity) {
        let mut app = flat_headless_app();
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

    /// A hermetic free-space spot: hundreds of meters from the origin (and the crab),
    /// in the NEGATIVE quadrant so a test flying the craft outward (+X/+Z thrust)
    /// heads into the tile interior — from the positive corner it would cross the
    /// 512 m fixture edge mid-measurement and be parked by `rescue_lost_crafts`.
    const FAR: Vec3 = Vec3::new(-300.0, 300.0, -300.0);

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

    /// rl#298 stage 1 (one world, shared construction): the crab `spawn_crab` builds —
    /// the SAME construction code on the host and the trainer — exchanges momentum
    /// with world content BOTH ways, not just registers a contact pair: a ramming
    /// craft knocks her carapace back (the world pushes her), and the rammer leaves
    /// the impact slower than it arrived (she pushes back). This reaches the armed
    /// host by composition, not by rebuilding its whole assembly here: the physics
    /// core and collision groups are the ones the host installs (net's
    /// pump-equivalence test pins that this stack steps tick-for-tick what the client
    /// pumps), the rammer is the production `vehicle_bundle`, the omitted bridge
    /// never writes per-tick body poses (its recenter rebases spawn origins only),
    /// and net's carapace-delta test pins that a carapace push folds 1:1 into her
    /// game pose. Flat fixture grid — a zero-action ragdoll never settles on the GCR
    /// origin slope, and aiming a ballistic ram needs a stationary target. No
    /// `VehiclePlugin`, deliberately — a low-g Ship (gravity scale 0.30, rl#316)
    /// with no flight forces is ballistic over this sub-second ram, so the only
    /// thing that can bleed its speed is her body.
    #[test]
    fn ramming_craft_and_crab_exchange_momentum_both_ways() {
        use crate::bot::body::CrabCarapace;
        use crate::bot::headless::tick;

        let mut app = flat_headless_app();
        tick(&mut app, 192);

        let carapace = |app: &mut App| -> Vec3 {
            let mut q = app
                .world_mut()
                .query_filtered::<&Transform, With<CrabCarapace>>();
            q.single(app.world()).expect("carapace").translation
        };
        let carapace_speed = |app: &mut App| -> f32 {
            let mut q = app
                .world_mut()
                .query_filtered::<&Velocity, With<CrabCarapace>>();
            q.single(app.world()).expect("carapace").linear.length()
        };

        // The zero-action mesh ragdoll never sits STILL — solver rest noise
        // twitch-walks her decimeters over the settle (rl#340 stage 3 measured
        // spawn-to-ram drift spanning x ±0.3 m across bit-level physics
        // perturbations), so where and at what yaw the ram lands is draw-dependent
        // and a signed along-axis displacement assert measures her wander, not the
        // push. The push instrument is her carapace SPEED spike over the ram
        // window instead, asserted RELATIVE to this same-length pre-ram window —
        // the live noise floor, so wander creeping up to the signal fails loudly
        // instead of hollowing the instrument out. Five stage-3 draws measured
        // push peaks 0.44–0.58 m/s against wander peaks 0.11–0.14; the 0.15
        // clamp keeps the bar ≥0.3 even if a future plant goes quieter.
        const RAM_WINDOW: u32 = 60;
        let mut wander_peak = 0.0f32;
        for _ in 0..RAM_WINDOW {
            tick(&mut app, 1);
            wander_peak = wander_peak.max(carapace_speed(&mut app));
        }
        let p0 = carapace(&mut app);

        const RAM_SPEED: f32 = 8.0;
        let rammer = spawn_ram_vehicle(
            app.world_mut(),
            VehicleKind::Ship,
            Transform::from_translation(p0 + Vec3::new(1.2, 0.0, 0.0)),
            Velocity {
                linear: Vec3::new(-RAM_SPEED, 0.0, 0.0),
                angular: Vec3::ZERO,
            },
        );
        let mut pushed_peak = 0.0f32;
        for _ in 0..RAM_WINDOW {
            tick(&mut app, 1);
            pushed_peak = pushed_peak.max(carapace_speed(&mut app));
        }

        let p1 = carapace(&mut app);
        let v1 = app
            .world()
            .get::<Velocity>(rammer)
            .expect("the rammer body persists — nothing despawns it in this app")
            .linear;
        println!(
            "ram: carapace {p0:?} -> {p1:?}, peak speed {pushed_peak:.2} m/s \
             (pre-ram wander peak {wander_peak:.2}), rammer v1 {v1:?}"
        );
        assert!(
            pushed_peak > 2.0 * wander_peak.max(0.15),
            "the world must push her: an {RAM_SPEED} m/s ram peaked her carapace at \
             {pushed_peak:.2} m/s, not clearly above the pre-ram rest wander peak \
             of {wander_peak:.2}"
        );
        assert!(
            v1.x > -0.7 * RAM_SPEED,
            "she must push back: the rammer kept {v1:?} of its {RAM_SPEED} m/s"
        );
    }

    /// rl#260: the rendered silhouette must never poke outside the physics collider — for
    /// every part, |offset| + half must fit inside [`VEHICLE_HALF`] on every axis.
    #[test]
    fn silhouettes_stay_inside_the_collider() {
        for kind in VehicleKind::ALL {
            for (i, p) in kind.silhouette().iter().enumerate() {
                assert!(
                    p.half.cmpgt(Vec3::ZERO).all(),
                    "{kind:?} part {i} is degenerate: half {}",
                    p.half
                );
                let reach = p.offset.abs() + p.half;
                assert!(
                    reach.cmple(VEHICLE_HALF).all(),
                    "{kind:?} part {i} pokes outside the collider: reach {reach} > {VEHICLE_HALF}"
                );
            }
        }
    }

    /// rl#308: every nozzle roots on the hull (inside the collider box) and fires a unit
    /// axis the craft can actually thrust along — the plane's lever nose axis, or a ship
    /// direct-thrust axis. A nozzle on a dead axis would plume without force.
    #[test]
    fn nozzles_fire_only_thrustable_axes() {
        for kind in VehicleKind::ALL {
            let p = kind.params();
            for (i, n) in kind.nozzles().iter().enumerate() {
                assert!(
                    n.offset.abs().cmple(VEHICLE_HALF * 1.001).all(),
                    "{kind:?} nozzle {i} roots outside the collider: {}",
                    n.offset
                );
                assert!(
                    (n.axis.length() - 1.0).abs() < 1e-6,
                    "{kind:?} nozzle {i} axis is not unit: {}",
                    n.axis
                );
                assert!(n.radius > 0.0 && n.max_len > 0.0);
                assert!(
                    n.axis.abs().dot(p.live_axes()) > 0.99,
                    "{kind:?} nozzle {i} fires along a dead axis {}",
                    n.axis
                );
            }
        }
    }

    /// rl#308: the exhaust-intensity source. The plane's fraction is its throttle lever
    /// along +Z (direct axes dead); the ship's is the per-axis command (lever dead).
    #[test]
    fn thrust_fraction_mirrors_each_crafts_live_axes() {
        let mut cmd = PilotCommand::new(
            VehicleKind::Plane,
            Boarding {
                pos: Vec3::ZERO,
                yaw: 0.0,
                velocity: Vec3::ZERO,
            },
        );
        cmd.thrust = Vec3::new(1.0, -0.5, 0.25);

        let plane = Vehicle {
            pilot: PilotId(0),
            kind: VehicleKind::Plane,
            throttle: 0.6,
        };
        assert_eq!(
            plane.thrust_fraction(&cmd),
            Vec3::new(0.0, 0.0, 0.6),
            "the plane plumes on its throttle lever alone"
        );

        let ship = Vehicle {
            pilot: PilotId(0),
            kind: VehicleKind::Ship,
            throttle: 0.9,
        };
        assert_eq!(
            ship.thrust_fraction(&cmd),
            Vec3::new(1.0, -0.5, 0.25),
            "the ship plumes on its per-axis command; its dead lever adds nothing"
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

    /// Pins the rl#230 feel-call: level flight lives strictly between slow flight and the
    /// full-throttle terminal — at slow flight (4 m/s) lift < weight so the plane settles
    /// instead of ballooning off the runway, while near terminal (8 m/s) lift > weight so
    /// altitude is still winnable with throttle. Both sides sample vertical velocity over a
    /// few zero-throttle ticks, level attitude, so lift vs gravity is the only vertical term.
    #[test]
    fn slow_flight_sinks_high_speed_climbs() {
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
        let slow = dvy(4.0);
        assert!(
            slow < 0.0,
            "plane must sink in slow flight (lift < weight), got Δvy={slow}"
        );
        let near_terminal = dvy(8.0);
        assert!(
            near_terminal > 0.0,
            "plane must climb near full-throttle terminal speed, got Δvy={near_terminal}"
        );
    }

    // The pitch/yaw sign tests fly at 6 m/s (≥ full-authority speed): control
    // surfaces are aerodynamic, so a standstill craft has no authority (rl#382) —
    // the zero-speed behavior is pinned by `control_authority_scales_with_airspeed`.
    #[test]
    fn positive_pitch_raises_the_nose() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, 6.0));
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
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, 6.0));
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
        // Flying start — a standstill craft has no surface authority (rl#382).
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, 6.0));
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

    /// rl#255 regression: a sharp full-throttle turn must CARRY its speed, not skid it
    /// off. Hold hard yaw through a big nose sweep and require (a) the nose really swept,
    /// (b) the velocity followed the nose (the mechanism — the 0.93 floor is what grip
    /// buys: thrust alone re-aligns only to ~0.87 here, grip holds ~0.97), (c) most of
    /// the speed survived (the symptom).
    #[test]
    fn sharp_turn_carries_speed_and_velocity_follows_nose() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, 6.0));
        set_cmd(&mut app, |c| c.yaw = 1.0);
        let s0 = body(&app, e).1.linear.length();
        for _ in 0..120 {
            app.update();
        }
        let (t, vel) = body(&app, e);
        let nose = t.rotation * Vec3::Z;
        assert!(
            nose.z < 0.5,
            "hard yaw did not sweep the nose far enough to test the turn: nose.z={}",
            nose.z
        );
        let v = vel.linear;
        let alignment = v.normalize().dot(nose);
        assert!(
            alignment > 0.93,
            "velocity lagged the nose (skid) — grip must swing v to the new heading, \
             got v̂·nose={alignment}"
        );
        let s1 = v.length();
        assert!(
            s1 > 0.7 * s0,
            "sharp turn bled speed: {s0} -> {s1} (rl#255)"
        );
    }

    /// The grip term redirects, never adds: a velocity 90° off the nose must swing toward
    /// it without the speed growing. Gravity off and throttle zero, so grip's ≤ 0 power
    /// is the only term that could be caught adding energy.
    #[test]
    fn grip_swings_velocity_toward_nose_without_adding_speed() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(3.0, 0.0, 0.0));
        let mut ent = app.world_mut().entity_mut(e);
        ent.get_mut::<Vehicle>().unwrap().throttle = 0.0;
        ent.insert(GravityScale(0.0));
        let v0 = body(&app, e).1.linear;
        for _ in 0..60 {
            app.update();
        }
        let v1 = body(&app, e).1.linear;
        assert!(
            v1.z > v1.x.abs(),
            "velocity did not swing toward the nose (+Z): {v1}"
        );
        assert!(
            v1.length() <= v0.length(),
            "grip added speed with zero throttle: {} -> {}",
            v0.length(),
            v1.length()
        );
    }

    /// rl#382: aerodynamic control authority scales with dynamic pressure. Hold full
    /// yaw for a few ticks (zero throttle, zero gravity, so airspeed is the only
    /// variable) — a standstill plane must not turn at all, half-cruise gets partial
    /// authority, and at the full-authority speed the rate matches flying feel. The
    /// SHIP's controls are thrusters, not surfaces: full authority at standstill.
    #[test]
    fn control_authority_scales_with_airspeed() {
        let yaw_rate = |kind: VehicleKind, speed: f32| {
            let (mut app, e) = app_with_vehicle(kind, FAR, Vec3::new(0.0, 0.0, speed));
            let mut ent = app.world_mut().entity_mut(e);
            ent.get_mut::<Vehicle>().unwrap().throttle = 0.0;
            ent.insert(GravityScale(0.0));
            set_cmd(&mut app, |c| c.yaw = 1.0);
            for _ in 0..10 {
                app.update();
            }
            body(&app, e).1.angular.y
        };

        let standstill = yaw_rate(VehicleKind::Plane, 0.0);
        let half = yaw_rate(VehicleKind::Plane, 2.5);
        let cruise = yaw_rate(VehicleKind::Plane, 5.0);
        println!("yaw rate rad/s — standstill {standstill}, half {half}, cruise {cruise}");
        assert_eq!(
            standstill, 0.0,
            "a stationary plane has no airflow over its rudder — it must not yaw in place"
        );
        assert!(
            half > 0.05 * cruise && half < 0.6 * cruise,
            "half-speed authority should be partial (~(v/v_full)² = 25%), got \
             half={half} cruise={cruise}"
        );
        assert!(
            cruise > 1.0,
            "full-authority yaw rate must stay at the tuned flying feel, got {cruise}"
        );

        let ship_standstill = yaw_rate(VehicleKind::Ship, 0.0);
        assert!(
            ship_standstill > 0.0,
            "the ship aims with thrusters, not surfaces — standstill authority stays"
        );
    }

    /// rl#379 ruler: the settled full-throttle top speed, measured in free flight
    /// (level start at boarding speed, throttle pinned to 1, 20 s of sim). The
    /// median of the last 100 ticks is the "top speed" every retune quotes.
    #[test]
    fn full_throttle_top_speed_settles_in_band() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(0.0, 0.0, 2.0));
        set_cmd(&mut app, |c| c.throttle_trim = 1.0);
        let mut speeds = Vec::new();
        for _ in 0..600 {
            app.update();
            speeds.push(body(&app, e).1.linear.length());
        }
        for (t, s) in speeds.iter().enumerate().step_by(30) {
            println!("trace t={t} |v|={s:.3}");
        }
        let mut tail: Vec<f32> = speeds[500..].to_vec();
        tail.sort_by(f32::total_cmp);
        let settled = tail[tail.len() / 2];
        println!("settled top speed = {settled:.3} m/s");
        // rl#379: the owner-set target is 2× the pre-retune measurement (4.635 m/s,
        // same ruler); ±5% keeps the pin tight without chasing solver noise.
        let (lo, hi) = (PLANE_TOP_SPEED_MPS * 0.95, PLANE_TOP_SPEED_MPS * 1.05);
        assert!(
            (lo..hi).contains(&settled),
            "full-throttle top speed off PLANE_TOP_SPEED_MPS ({PLANE_TOP_SPEED_MPS}): \
             {settled} m/s"
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
        assert_eq!(count(&mut app), 1, "still one vehicle after a kind switch");
        assert_eq!(
            app.world_mut()
                .query::<&Vehicle>()
                .single(app.world())
                .unwrap()
                .kind,
            VehicleKind::Ship,
            "the body became the switched-to kind",
        );

        app.world_mut()
            .resource_mut::<VehicleControls>()
            .0
            .remove(&P0);
        app.update();
        assert_eq!(count(&mut app), 0, "vehicle despawned on step-out");
    }

    /// rl#258: boarding transforms in place — the craft materialises at the walker's spot
    /// with its facing and velocity, lifted just enough that the collider clears the ground.
    #[test]
    fn boarding_spawns_at_the_walker_with_velocity_conserved() {
        let mut app = flat_headless_app();
        app.add_plugins(VehiclePlugin);
        app.update();
        let vel = Vec3::new(0.4, 0.0, 0.1);
        app.world_mut().resource_mut::<VehicleControls>().0.insert(
            P0,
            PilotCommand::new(
                VehicleKind::Plane,
                Boarding {
                    pos: Vec3::new(3.0, 0.0, -2.0),
                    yaw: std::f32::consts::FRAC_PI_2,
                    velocity: vel,
                },
            ),
        );
        app.update();
        let mut q = app.world_mut().query::<(&Transform, &Velocity, &Vehicle)>();
        let (t, v, _) = q.single(app.world()).expect("one craft");
        // A physics tick already ran between the spawn and this read — small tolerances.
        assert!(
            (t.translation.x - 3.0).abs() < 0.05 && (t.translation.z - -2.0).abs() < 0.05,
            "the craft must materialise at the walker's spot, got {}",
            t.translation
        );
        assert!(
            t.translation.y >= VEHICLE_HALF.y - 0.01,
            "collider bottom must clear the ground, got centre y={}",
            t.translation.y
        );
        let nose = t.rotation * Vec3::Z;
        assert!(
            nose.x > 0.99,
            "yaw π/2 must point the nose along +X, got {nose}"
        );
        assert!(
            v.linear.distance(vel) < 0.2,
            "the walker's velocity is conserved (want {vel}, got {})",
            v.linear
        );
    }

    /// A deterministic hill on the APP'S OWN grid: the committed bake is fixed, so
    /// scan a coarse lattice around the origin for ground well above y=0 — the
    /// terrain tests' shared boarding spot.
    fn hill_on_the_tile(app: &App) -> (f32, f32) {
        let g = app.world().resource::<crate::terrain::Terrain>();
        (0..10_000)
            .map(|i| {
                (
                    ((i % 100) as f32 - 50.0) * 100.0,
                    ((i / 100) as f32 - 50.0) * 100.0,
                )
            })
            .find(|&(x, z)| g.height(x, z) > 10.0)
            .expect("a hill within ±5 km of the origin")
    }

    /// rl#322: the restart park lands the craft on its ring slot at rest with the
    /// throttle LEVER zeroed — a lever held through the teleport would immediately fly
    /// the craft back off the ring — nose outward, on the local surface.
    #[test]
    fn park_zeroes_the_lever_and_lands_on_the_surface() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, FAR, Vec3::new(30.0, -5.0, 10.0));
        app.world_mut()
            .entity_mut(e)
            .get_mut::<Velocity>()
            .unwrap()
            .angular = Vec3::new(1.0, 2.0, 3.0);

        let terrain = app.world().resource::<crate::terrain::Terrain>().clone();
        let center = Vec2::new(2.0, -3.0);
        park_vehicles_near(app.world_mut(), center, 8.0, &[], &terrain);

        let (t, v) = body(&app, e);
        let d = Vec2::new(t.translation.x, t.translation.z) - center;
        assert!(
            (d.length() - 8.0).abs() < 1e-3,
            "the craft parks on the ring: {} m out",
            d.length()
        );
        assert!(
            (t.rotation * Vec3::Z).dot(Vec3::new(d.x, 0.0, d.y).normalize()) > 0.99,
            "the nose points outward, away from the center"
        );
        assert_eq!((v.linear, v.angular), (Vec3::ZERO, Vec3::ZERO));
        let want_y = VEHICLE_HALF.y + GROUND_CLEARANCE;
        assert!(
            (t.translation.y - want_y).abs() < 1e-3,
            "flat grid: parked on the surface, y={} want {want_y}",
            t.translation.y
        );
        assert_eq!(
            app.world().entity(e).get::<Vehicle>().unwrap().throttle,
            0.0,
            "the throttle lever resets — same known zero a fresh craft starts at"
        );
    }

    /// rl#283: the boarding clamp is keyed to the TERRAIN surface at the boarding spot,
    /// not y=0 — a walker boarding on a mountainside gets a craft on the local ground,
    /// not one buried inside the hill.
    #[test]
    fn boarding_clamp_tracks_the_terrain_surface() {
        use crate::bot::headless::{HeadlessStack, WorldRole, headless_stack};

        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            grid: crate::terrain::TerrainGrid::gcr(),
            visuals: crate::Visuals(false),
        });
        app.add_plugins(VehiclePlugin);
        app.update();

        let (x, z) = hill_on_the_tile(&app);
        let want = {
            let g = app.world().resource::<crate::terrain::Terrain>();
            g.height(x, z) + VEHICLE_HALF.y + GROUND_CLEARANCE
        };

        board_at(&mut app, P0, VehicleKind::Plane, Vec3::new(x, 0.0, z));
        app.update();
        let mut q = app.world_mut().query::<(&Transform, &Vehicle)>();
        let (t, _) = q.single(app.world()).expect("one craft");
        assert!(
            (t.translation.y - want).abs() < 0.1,
            "craft must materialise on the local surface: y={}, want ≈{want}",
            t.translation.y
        );
    }

    /// rl#281 stage 4: a plane FLIES over the real tile — full throttle from a
    /// mountainside boarding, it makes real forward way and the heightfield holds it
    /// up the whole run (never sinks through the surface). With the boarding-clamp
    /// test above this is the vehicles-on-terrain verification: materialise on the
    /// local ground, then fly, with terrain contact live throughout.
    #[test]
    fn plane_flies_over_terrain_without_sinking_through() {
        use crate::bot::headless::{HeadlessStack, WorldRole, headless_stack};

        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            grid: crate::terrain::TerrainGrid::gcr(),
            visuals: crate::Visuals(false),
        });
        app.add_plugins(VehiclePlugin);
        app.update();

        let (x, z) = hill_on_the_tile(&app);
        board_at(&mut app, P0, VehicleKind::Plane, Vec3::new(x, 0.0, z));
        app.update();
        set_cmd(&mut app, |c| c.throttle_trim = 1.0);

        let start = {
            let mut q = app.world_mut().query::<(&Transform, &Vehicle)>();
            q.single(app.world()).expect("one craft").0.translation
        };
        let mut max_above = f32::MIN;
        for tick in 0..300 {
            app.update();
            let (t, g) = {
                let mut q = app.world_mut().query::<(&Transform, &Vehicle)>();
                let t = q.single(app.world()).expect("one craft").0.translation;
                let g = app.world().resource::<crate::terrain::Terrain>();
                (t, g.height(t.x, t.z))
            };
            assert!(t.is_finite(), "craft blew up at tick {tick}: {t}");
            // Soft contacts rest with ~cm penetration; anything past half a body
            // below the surface means the heightfield stopped holding.
            assert!(
                t.y > g - VEHICLE_HALF.y,
                "craft sank through the terrain at tick {tick}: y={} surface={g}",
                t.y
            );
            max_above = max_above.max(t.y - g);
        }
        let end = {
            let mut q = app.world_mut().query::<(&Transform, &Vehicle)>();
            q.single(app.world()).expect("one craft").0.translation
        };
        let way = Vec2::new(end.x - start.x, end.z - start.z).length();
        assert!(
            way > 2.0,
            "full throttle for 300 ticks must make real forward way over terrain, got {way} m"
        );
        // FLIGHT, not a skid: at some point the craft is clearly off the local ground
        // (several body heights — a grounded skid never leaves ~VEHICLE_HALF.y).
        assert!(
            max_above > 10.0 * VEHICLE_HALF.y,
            "craft never got airborne over the terrain, max clearance {max_above} m"
        );
    }

    /// rl#258: a kind switch morphs the SAME body in place — entity, pose and velocity all
    /// carry through; only the throttle lever and gravity scale become the new kind's.
    #[test]
    fn kind_switch_morphs_the_body_in_place() {
        let at = Vec3::new(1.0, 4.0, 2.0);
        let vel = Vec3::new(0.0, 0.0, 3.0);
        let (mut app, e) = app_with_vehicle(VehicleKind::Plane, at, vel);
        app.update();
        set_cmd(&mut app, |c| c.kind = VehicleKind::Ship);
        app.update();
        let ent = app.world().entity(e);
        let v = ent.get::<Vehicle>().expect("the SAME entity swapped form");
        assert_eq!(v.kind, VehicleKind::Ship);
        assert_eq!(v.throttle, 0.0, "the lever resets to the known fresh state");
        assert_eq!(
            ent.get::<GravityScale>().unwrap().0,
            VehicleKind::Ship.gravity_scale(),
            "gravity follows the new kind"
        );
        let pos = ent.get::<Transform>().unwrap().translation;
        assert!(
            pos.distance(at) < 1.0,
            "the body stays put through the morph (drifted to {pos})"
        );
        let linear = ent.get::<Velocity>().unwrap().linear;
        assert!(
            linear.distance(vel) < 1.0,
            "velocity carries through the morph (was {vel}, got {linear})"
        );
    }

    /// Per-pilot multiplicity (rl#191): two pilots board on the same tick ⇒ two bodies, each the
    /// kind ITS pilot chose, each where ITS walker stood (rl#258); one pilot stepping
    /// out despawns only ITS craft. Spawn/despawn bookkeeping only — a couple of ticks, so the
    /// crab standing at the arena origin never comes into play.
    #[test]
    fn each_pilot_gets_its_own_craft() {
        let p1 = PilotId(1);
        let mut app = flat_headless_app();
        app.add_plugins(VehiclePlugin);
        // Warm the clock: the first update's zero delta runs no FixedUpdate (same dance as
        // `manage_spawns_and_despawns_one_vehicle`).
        app.update();
        board_at(&mut app, P0, VehicleKind::Ship, Vec3::new(-2.0, 0.0, 0.0));
        board_at(&mut app, p1, VehicleKind::Plane, Vec3::new(3.0, 0.0, 1.0));
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
        assert_eq!(by(P0).2, -2.0, "pilot 0's craft is where ITS walker stood");
        assert_eq!(by(p1).2, 3.0, "pilot 1's craft is where ITS walker stood");

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

    /// rl#307/rl#316: the ship is "a bit" affected by gravity — an idle ship sinks
    /// perceptibly, but far more gently than the plane falls (heavy drag caps the
    /// sink rate too).
    #[test]
    fn ship_sinks_gently_but_plane_falls() {
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
        let ship = fall(VehicleKind::Ship);
        assert!(ship < -0.9, "ship must feel gravity, got Δvy={ship}");
        assert!(
            ship > -1.4,
            "but gently — drag caps the sink well below the plane's fall, got Δvy={ship}"
        );
        assert!(
            fall(VehicleKind::Plane) < -1.0,
            "plane must fall under gravity"
        );
    }

    /// rl#307: rotational inertia — releasing the yaw input must NOT stop the spin;
    /// the ship keeps rotating and angular drag brings it to rest, never an instant
    /// clamp (with the old 0.07 angular drag the spin died in ~3 ticks and the first
    /// assertion fails).
    #[test]
    fn ship_spin_persists_after_release_then_drag_stops_it() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::ZERO);
        set_cmd(&mut app, |c| c.yaw = 1.0);
        // 2 s ≈ 3 spin-up τ: near the terminal yaw rate.
        for _ in 0..128 {
            app.update();
        }
        let w0 = body(&app, e).1.angular.y;
        assert!(w0 > 1.0, "held yaw must build a real spin rate, got {w0}");
        set_cmd(&mut app, |c| c.yaw = 0.0);
        for _ in 0..16 {
            app.update();
        }
        let w1 = body(&app, e).1.angular.y;
        assert!(
            w1 > 0.5 * w0,
            "a quarter-second after release the ship must still be spinning \
             (rotational inertia, not a clamp): {w0} -> {w1}"
        );
        for _ in 0..256 {
            app.update();
        }
        let w2 = body(&app, e).1.angular.y;
        assert!(
            w2 < 0.1 * w0,
            "angular drag must bring the released spin to rest: {w0} -> {w2}"
        );
    }

    /// rl#307: drag "a bunch" — a coasting ship sheds half its speed within a second
    /// (watercraft-heavy), where the old 0.05 float coasted for many seconds.
    /// Gravity off (same isolation as `grip_swings_velocity_toward_nose_without_adding_speed`):
    /// this pins the DRAG decay, and since rl#316 the weight would feed a ~1.9 m/s sink
    /// into the speed magnitude over the same second.
    #[test]
    fn ship_coast_halves_speed_within_a_second() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::new(3.0, 0.0, 0.0));
        app.world_mut().entity_mut(e).insert(GravityScale(0.0));
        let s0 = body(&app, e).1.linear.length();
        for _ in 0..64 {
            app.update();
        }
        let s1 = body(&app, e).1.linear.length();
        assert!(
            s1 < 0.5 * s0,
            "heavy drag must halve a 3 m/s coast within 1 s: {s0} -> {s1}"
        );
    }

    /// rl#316 regression (owner: "ship shouldn't only sink when holding still — gravity
    /// doesn't work that way"). Two pins:
    ///
    /// `ship_cruise_sink_matches_rest_sink` — the SYMPTOM. At zero vertical thrust and
    /// level attitude, the settled sink at cruise must sit within ~25% of the settled
    /// sink at rest: the isotropic quadratic drag scales the VERTICAL drag coefficient
    /// with TOTAL speed, so a tuning whose rest sink is masked at cruise fails here.
    ///
    /// `ship_forward_thrust_cannot_lift_at_any_pitch` — the MECHANISM. Direct thrust is
    /// body-frame, so a thruster stronger than the weight holds altitude by attitude
    /// alone (nose-up converts forward thrust into lift). Weight 0.77 N > forward
    /// 0.75 N makes that impossible: 60° nose-up at full forward burn must still sink
    /// (the rl#313 tuning climbed at +1.4 m/s here).
    #[test]
    fn ship_cruise_sink_matches_rest_sink() {
        let settled_vy = |thrust: Vec3| {
            let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::ZERO);
            app.world_mut()
                .entity_mut(e)
                .get_mut::<Vehicle>()
                .unwrap()
                .throttle = 0.0;
            set_cmd(&mut app, |c| c.thrust = thrust);
            for _ in 0..512 {
                app.update();
            }
            body(&app, e).1.linear.y
        };
        let rest = -settled_vy(Vec3::ZERO);
        let cruise = -settled_vy(Vec3::new(0.0, 0.0, 1.0));
        let ratio = cruise / rest;
        assert!(
            (0.75..=1.25).contains(&ratio),
            "cruise sink must track rest sink (gravity acts in motion): \
             rest={rest:.2} m/s, cruise={cruise:.2} m/s (ratio {ratio:.2})"
        );
    }

    #[test]
    fn ship_forward_thrust_cannot_lift_at_any_pitch() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::ZERO);
        {
            let mut ent = app.world_mut().entity_mut(e);
            ent.get_mut::<Vehicle>().unwrap().throttle = 0.0;
            // 60° nose-up: forward burn's vertical component is 0.65 N < 0.77 N weight.
            ent.get_mut::<Transform>().unwrap().rotation =
                Quat::from_rotation_x(-std::f32::consts::FRAC_PI_3);
        }
        set_cmd(&mut app, |c| c.thrust = Vec3::new(0.0, 0.0, 1.0));
        for _ in 0..512 {
            app.update();
        }
        let vy = body(&app, e).1.linear.y;
        assert!(
            vy < -0.2,
            "full forward burn at 60° nose-up must still sink (weight > forward thrust), \
             got vy={vy:.2}"
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

    /// rl#313 (owner): the idle sink must FEEL like gravity — an integrated force
    /// with a perceptible buildup from release to terminal, never a constant-velocity
    /// step. Pins the kinematic trace's SHAPE: still slow at a quarter second, most
    /// of the ramp swept by 1 s, settled near terminal by 2.5 s. A velocity write
    /// (or a terminal masked by near-instant drag) fails the first bound; losing the
    /// ramp entirely (gravity tuned back to imperceptible) fails the terminal band.
    #[test]
    fn ship_idle_sink_ramps_like_gravity() {
        let (mut app, e) = app_with_vehicle(VehicleKind::Ship, FAR, Vec3::ZERO);
        app.world_mut()
            .entity_mut(e)
            .get_mut::<Vehicle>()
            .unwrap()
            .throttle = 0.0;
        let mut sink_at = std::collections::BTreeMap::new();
        for i in 1..=160u32 {
            app.update();
            if [16, 64, 160].contains(&i) {
                sink_at.insert(i, -body(&app, e).1.linear.y);
            }
        }
        let (quarter_s, one_s, terminal) = (sink_at[&16], sink_at[&64], sink_at[&160]);
        assert!(
            (2.3..2.8).contains(&terminal),
            "terminal sink must be fast enough to see, gentle enough to fly: {terminal} m/s"
        );
        assert!(
            quarter_s < 0.35 * terminal,
            "a quarter second in, the sink is still building — a step to terminal \
             is a velocity write, not gravity: {quarter_s} of {terminal} m/s"
        );
        assert!(
            (0.5..0.8).contains(&(one_s / terminal)),
            "most of the ramp sweeps through the first second — visibly \
             accelerating, not yet settled: {one_s} of {terminal} m/s"
        );
    }

    /// rl#339 regression: past dt·c/m = 1 the explicit quadratic drag flips from
    /// brake to amplifier — a craft kicked to a few hundred m/s by a crab strike
    /// escaped at a constant ~1e6 m/s. With the [`BRAKE_COEFF_MAX`] cap, ANY finite
    /// launch speed must decay, never grow.
    #[test]
    fn hypersonic_craft_brakes_instead_of_amplifying() {
        use crate::bot::headless::tick;
        // In-bounds launch (the rescue bound is the flat fixture's 512 m half-span)
        // so the out-of-world park can't mask a drag blowup; vertical UP so neither
        // the tile edge nor the below-terrain rescue is ever crossed while braking.
        let (mut app, e) =
            app_with_vehicle(VehicleKind::Plane, Vec3::new(0.0, 5.0, 0.0), Vec3::Y * 1e4);
        set_cmd(&mut app, |c| c.throttle_trim = -1.0);
        let mut prev = 1e4_f32;
        for _ in 0..16 {
            tick(&mut app, 1);
            let (t, v) = body(&app, e);
            let speed = v.linear.length();
            assert!(
                t.translation.is_finite() && speed.is_finite(),
                "craft went non-finite while braking"
            );
            // `<=`: an update before the first fixed step leaves the speed untouched.
            assert!(
                speed <= prev.max(10.0),
                "drag amplified instead of braking: {prev} -> {speed} m/s"
            );
            prev = speed;
        }
        assert!(
            prev < 100.0,
            "hypersonic launch must be dead within 16 ticks, still {prev} m/s"
        );
    }

    /// rl#339: a craft outside the tile interior (where no ground collider exists)
    /// is parked back on in-bounds terrain at rest by [`rescue_lost_crafts`].
    #[test]
    fn escaped_craft_parks_back_in_bounds() {
        use crate::bot::headless::tick;
        let mut app = flat_headless_app();
        app.add_plugins(VehiclePlugin);
        board(&mut app, P0, VehicleKind::Ship);
        tick(&mut app, 4); // spawn edge (the first update may run no fixed step)
        let bound = app
            .world()
            .resource::<crate::terrain::Terrain>()
            .half_span_min();
        let e = *app
            .world_mut()
            .query_filtered::<Entity, With<Vehicle>>()
            .iter(app.world())
            .collect::<Vec<_>>()
            .first()
            .expect("craft spawned");
        {
            let mut ent = app.world_mut().entity_mut(e);
            ent.get_mut::<Transform>().unwrap().translation = Vec3::new(bound + 5000.0, 40.0, 0.0);
            ent.get_mut::<Velocity>().unwrap().linear = Vec3::X * 1e6;
        }
        tick(&mut app, 4);
        let (t, v) = body(&app, e);
        assert!(
            t.translation.x.abs() <= bound && t.translation.z.abs() <= bound,
            "escaped craft not parked back in bounds: {:?}",
            t.translation
        );
        assert!(
            v.linear.length() < 1.0,
            "parked craft must be at rest, moving {:?}",
            v.linear
        );
    }

    /// rl#339: an escape impulse aimed DOWN tunnels through the one-sided
    /// heightfield in one substep and would free-fall forever with x/z in bounds —
    /// the below-terrain branch must lift it back onto the surface.
    #[test]
    fn buried_craft_parks_back_on_the_surface() {
        use crate::bot::headless::tick;
        let (mut app, e) =
            app_with_vehicle(VehicleKind::Plane, Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO);
        app.world_mut()
            .entity_mut(e)
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(0.0, -100.0, 0.0);
        tick(&mut app, 4);
        let (t, v) = body(&app, e);
        assert!(
            t.translation.y > -1.0,
            "buried craft must be lifted back onto the surface: {:?}",
            t.translation
        );
        assert!(v.linear.length() < 1.0, "rescued craft must be at rest");
    }

    /// rl#339: a non-finite craft pose (nothing else guards crafts — the pose
    /// sentinel and rescue cover crab body parts only) parks at the arena origin.
    #[test]
    fn nonfinite_craft_parks_at_origin() {
        use crate::bot::headless::tick;
        let (mut app, e) =
            app_with_vehicle(VehicleKind::Plane, Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO);
        app.world_mut()
            .entity_mut(e)
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::splat(f32::NAN);
        tick(&mut app, 4);
        let (t, v) = body(&app, e);
        assert!(
            t.translation.is_finite() && t.rotation.is_finite(),
            "craft still non-finite after rescue: {t:?}"
        );
        assert!(
            t.translation.x.abs() < 1.0 && t.translation.z.abs() < 1.0 && v.linear.length() < 1.0,
            "non-finite craft must park at rest at the origin: {:?} {:?}",
            t.translation,
            v.linear
        );
    }

    /// rl#377: a parked craft with a neutral command must fall ASLEEP — bit-exact
    /// stillness, not merely small motion. The bug: every apply_vehicle_forces write
    /// marks ExternalForce changed, bevy_rapier force-wakes the body on Changed, and
    /// the passive drag/grip/angular-drag terms re-derive a not-quite-equal force
    /// from the solver's own rest noise every tick — so the craft never slept, and
    /// the awake contact solve visibly chattered its orientation on the ground.
    #[test]
    fn parked_craft_falls_asleep() {
        use crate::bot::headless::tick;
        use bevy_rapier3d::plugin::context::RapierRigidBodySet;
        use bevy_rapier3d::prelude::RapierRigidBodyHandle;

        let mut app = flat_headless_app();
        app.add_plugins(VehiclePlugin);
        board_at(&mut app, P0, VehicleKind::Plane, Vec3::ZERO);
        // Spawn edge + ground settle + rapier's time_until_sleep (0.4 s ≈ 26 steps).
        tick(&mut app, 120);

        let (e, frozen) = {
            let mut q = app.world_mut().query::<(Entity, &Transform, &Vehicle)>();
            let (e, tf, _) = q.single(app.world()).expect("one craft");
            (e, *tf)
        };
        let handle = app
            .world()
            .entity(e)
            .get::<RapierRigidBodyHandle>()
            .expect("craft has a rapier body")
            .0;
        let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
        let set = set_q.single(app.world()).expect("one body set");
        assert!(
            set.bodies.get(handle).expect("live body").is_sleeping(),
            "a parked craft with a neutral command must be asleep by 120 ticks"
        );

        tick(&mut app, 60);
        let (t, _) = body(&app, e);
        assert!(
            t.translation == frozen.translation && t.rotation == frozen.rotation,
            "a sleeping parked craft must be bit-exact still: {:?} -> {:?}",
            frozen,
            t
        );
    }
}
