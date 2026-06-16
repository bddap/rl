//! Crab body definition: the `CrabJointId` action/observation joint set, the
//! per-instance joint + marker components, and `spawn_crab`, which instantiates
//! the rig-derived body recipe ([`super::rig`] owns the geometry).
//!
//! The body is a Rapier multibody tree rooted at the carapace. Only the locomotion
//! joints are policy-actuated and carry a [`CrabJointId`]: each leg's
//! coxa/merus/carpus and each claw's shoulder/wrist/pincer (all revolute). The rest
//! of the rig — proximal leg stubs, claw mid-segments, eye-stalks, palpi — spawns as
//! locked (fixed-joint) links with no `CrabJointId`, present in the body but
//! invisible to the policy. Promote a locked joint by adding a `CrabJointId` variant
//! and a rig actuation mapping ([`super::rig`]).

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::meshfit::{LoadedModel, PartId};
use super::rig::{self, RigLink, RigRecipe};

// ---------------------------------------------------------------------------
// Collision groups
// ---------------------------------------------------------------------------

/// Group 1: arena (ground + walls). Interacts with the crab body (group 2) and
/// the carapace-nested links (group 3) — the filter must name BOTH, or a nested
/// link sinks through the floor: collision is an AND of both directions, so the
/// arena listing group 3 is what lets a group-3 link contact the ground at all.
pub const ARENA_COLLISION: CollisionGroups =
    CollisionGroups::new(Group::GROUP_1, Group::GROUP_2.union(Group::GROUP_3));
/// Group 2: crab body parts. Interact with the arena AND with each other —
/// without self-collision the policy "tucks" legs through one another (free
/// interpenetration is an exploit, not a stance). Segments connected by a
/// joint are still contact-filtered per joint (`set_contacts_enabled(false)`),
/// so articulations don't fight their own contacts. N training crabs share
/// this group: their 4 m grid spacing exceeds any reach, so cross-crab
/// contact is geometrically impossible until we deliberately close the gap.
pub const CRAB_COLLISION: CollisionGroups =
    CollisionGroups::new(Group::GROUP_2, Group::GROUP_1.union(Group::GROUP_2));
/// Group 3: links whose collider center falls inside the carapace box — the
/// proximal stubs, and on this model the actuated coxa/claw-shoulder that ride
/// just under the shell. Membership is purely geometric (see `spawn_crab`), so it
/// catches actuated joints too, not only fixed stubs. They keep their mass but
/// collide only with the arena — never with the carapace or each other. A link
/// jammed inside the carapace collider just fights the solver every tick (the
/// near-massless pincers ring it as rest jitter, bddap/rl#20), and
/// `no_adjacent_contacts` can't filter it because its joint parent is another
/// nested link, not the carapace.
pub const NESTED_COLLISION: CollisionGroups = CollisionGroups::new(Group::GROUP_3, Group::GROUP_1);

/// Disable contacts between the two segments a joint connects. The joint
/// already constrains that pair, and their colliders overlap at the anchor by
/// construction — contacts there would only fight the articulation. All other
/// (non-adjacent) crab-part pairs DO collide; see [`CRAB_COLLISION`].
fn no_adjacent_contacts(joint: impl Into<TypedJoint>) -> TypedJoint {
    let mut joint = joint.into();
    let generic: &mut GenericJoint = joint.as_mut();
    generic.set_contacts_enabled(false);
    joint
}

/// Signed joint coordinate read off the two links' world orientations: the twist
/// of child-in-parent about the joint's free `axis_local` (the rig-derived axis
/// the actuator drives). Shared by the observation and the telemetry overlay so
/// both report the same angle the policy acts on.
pub fn joint_angle(axis_local: Vec3, parent_rot: Quat, child_rot: Quat) -> f32 {
    let q = (parent_rot.inverse() * child_rot).normalize();
    let v = Vec3::new(q.x, q.y, q.z);
    2.0 * v.dot(axis_local).atan2(q.w)
}

// ---------------------------------------------------------------------------
// Dimensions — tuned for a ~1m wide crab (game scale, not real life)
// ---------------------------------------------------------------------------

/// Carapace: wide, flat, slightly domed.
const CARAPACE_HALF_W: f32 = 0.5; // x (left-right)
const CARAPACE_HALF_H: f32 = 0.12; // y (up-down) — very flat
const CARAPACE_HALF_D: f32 = 0.35; // z (front-back)

/// Spawn height: how high above ground the carapace center starts. Model-scale:
/// in the bind pose the feet sit ~0.22 below the carapace centre, so ~0.3 drops
/// the crab onto its feet with a little clearance. (Was 0.58 for the larger
/// hand-coded body.)
pub const SPAWN_HEIGHT: f32 = 0.3;

/// Placeholder link-capsule dimensions (half-height, radius): `link_mesh` is one
/// uniformly-sized debug capsule reused for every rig link. The skinned model is the
/// real visual and the physics colliders are sized per-link from the rig recipe, so
/// this mesh is only a coarse stand-in in the debug render.
const COXA_LEN: f32 = 0.15;
const COXA_RAD: f32 = 0.045;

/// Leg-segment densities for the OFFLINE collider bake (`part_densities`); the live
/// rig body's masses come from `rig.rs`'s own densities. Tapered UP (denser toward
/// the tip): at game scale the distal links came out SUB-GRAM at density ~1 (the
/// tibia under a gram) — physically implausible and numerically twitchy (tiny inertia
/// → five-figure angular acceleration under load) — so denser distal segments land
/// each at ~10–14 g with a gentle inertia gradient up the leg.
const COXA_DENSITY: f32 = 6.0;
const FEMUR_DENSITY: f32 = 8.0;
const TIBIA_DENSITY: f32 = 14.0;

/// Carapace density. Heaviest part: the body's mass sits in the shell, over the
/// feet.
const CARAPACE_DENSITY: f32 = 5.0;
/// Every claw segment (upper/forearm/pincer). Kept LOW: the claw assembly hangs
/// off the front (+Z), and a dense one (was 3.0/2.5/4.0) put the CoM ahead of the
/// leg support so the crab pitched forward and couldn't stand.
const CLAW_DENSITY: f32 = 1.0;

// ---------------------------------------------------------------------------
// Torque ceilings — the magnitude an action of ±1 commands on each joint type.
// DIRECT-DRIVE torques (the policy's output IS the torque), and GRADED BY INERTIA:
// the lighter and more distal a link, the smaller its ceiling. A joint's
// snappiness is torque/inertia, so a flat ceiling makes a feather-light shin a
// hair-trigger while it's modest on the hip — the old flat 20 N·m on a sub-gram
// tibia was ~400,000 rad/s² of headroom and fed the mid-air helicopter. Each
// ceiling here is still 5-12x the ~0.5 N·m a joint needs to bear its share of the
// body, so standing keeps ample authority; only the absurd surplus is cut.
// ---------------------------------------------------------------------------

const COXA_TORQUE_CEILING: f32 = 6.0;
const FEMUR_TORQUE_CEILING: f32 = 4.0;
const TIBIA_TORQUE_CEILING: f32 = 2.5;
const CLAW_TORQUE_CEILING: f32 = 8.0;

/// Joint friction: a velocity-0 motor (stiffness 0 — no position servo; the policy
/// still commands all torque via `ExternalForce`), but FORCE-CAPPED by
/// [`CrabJointId::friction_cap`] so it saturates almost at once into a small
/// CONSTANT opposing torque — dry/Coulomb friction, not a viscous brake. This
/// coefficient is just the slope of the RAMP into that cap (hence the name): steep
/// enough to regularize the zero-velocity crossing, after which the cap binds and
/// the friction is the same small torque whether a joint creeps or flails — it is
/// NOT a damping gain. ForceBased so the cap is an honest N·m, not rescaled by the
/// link's effective mass. The friction must live on the joint: Rapier's per-body
/// `Damping` is a no-op on multibody links (only joint constraints and external
/// forces reach them, see #14). The free-flail speed it does NOT bound is caught by
/// the hard joint limits, which a limb reaches within a couple of ticks.
const FRICTION_RAMP: f32 = 4.0;

/// Breakaway level of the joint-friction motor (see [`FRICTION_RAMP`]) — the
/// constant torque an external load must exceed to back-drive the joint. Kept WELL
/// below the ~0.5 N·m needed to hold a leg (only ~1/5) so the legs flop loosely and
/// crumple under the lightest ground load instead of propping the body up, yet
/// nonzero because real joints have a little stiction and a touch of it regularizes
/// the zero-velocity crossing.
pub const LEG_FRICTION_CAP: f32 = 0.1;
const CLAW_FRICTION_CAP: f32 = 0.1;

/// Spring stiffness (natural frequency Hz, damping ratio) of every revolute
/// joint's constraint — the lock that holds the limb attached AND the hard end
/// stops. Rapier's joint default is `1e6` Hz: a near-rigid Baumgarte stop. At
/// that stiffness a limb driven hard into its limit overshoots it by up to
/// ~2.3 rad (130°!) in one tick, and the violent position-correction that snaps
/// it back is not momentum-conserving in the reduced-coordinate multibody
/// solver — with 30 joints slammed at once that residual *accumulates* into net
/// angular momentum, letting an airborne crab spin itself up from nothing
/// (issue #17: the ExternalForce couples are balanced to ~1e-5 N·m and contact
/// is ruled out, so the joint-LIMIT impulse was the leak, NOT the actuator).
/// 400 Hz keeps the stop firm — under standing/ground load the limb still holds
/// within ~0.1 rad of its limit, same as the rigid default — while capping the
/// airborne overshoot at ~0.2 rad, which cuts the spurious spin-up ~10× (a 68×
/// runaway under realistic drive becomes a bounded ~5× wobble). Damping 2.0
/// (> the default 1.0) softens the snap further without letting the limb sag
/// through its stop. It does NOT fully conserve L — the iterative solver has a
/// small irreducible drift floor — but it removes the runaway pumping.
pub const LIMIT_SOFTNESS: bevy_rapier3d::rapier::dynamics::SpringCoefficients<f32> =
    bevy_rapier3d::rapier::dynamics::SpringCoefficients {
        natural_frequency: 400.0,
        damping_ratio: 2.0,
    };

// ---------------------------------------------------------------------------
// Marker components for querying
// ---------------------------------------------------------------------------

/// Shared mesh/material handles for crab bodies, created once at startup.
/// Spawning goes through these because episode resets RESPAWN the whole crab
/// (a teleport keeps the dying pose's joint angles, which interpenetrate under
/// self-collision and explode) — per-spawn `Assets::add` would leak an asset
/// per body part per episode, unbounded over an overnight run.
#[derive(Resource)]
pub struct CrabAssets {
    body_mat: Handle<StandardMaterial>,
    leg_mat: Handle<StandardMaterial>,
    claw_mat: Handle<StandardMaterial>,
    eye_mat: Handle<StandardMaterial>,
    carapace_mesh: Handle<Mesh>,
    link_mesh: Handle<Mesh>,
    /// The rig-derived body recipe (the one model): every link's joint + capsule,
    /// read off the bind-pose skeleton once at startup. `spawn_crab` instantiates
    /// it. `None` only when the glTF model is unavailable.
    recipe: Option<RigRecipe>,
}

impl FromWorld for CrabAssets {
    fn from_world(world: &mut World) -> Self {
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        let body_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.45, 0.55), // blue-grey carapace
            perceptual_roughness: 0.7,
            ..default()
        });
        let leg_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.4, 0.15), // orange legs (Sally Lightfoot!)
            perceptual_roughness: 0.6,
            ..default()
        });
        let claw_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.7, 0.15, 0.15), // deep red claws
            perceptual_roughness: 0.5,
            ..default()
        });
        let eye_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.85, 0.7), // pale eye stalks
            perceptual_roughness: 0.3,
            ..default()
        });

        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        // The one model: derive the whole body's geometry from the glTF bind pose
        // once at startup. `spawn_crab` instantiates this recipe for every crab.
        let recipe = super::meshfit::model_path()
            .and_then(|p| LoadedModel::load(&p).ok())
            .and_then(|m| rig::build_recipe(&m));
        Self {
            body_mat,
            leg_mat,
            claw_mat,
            eye_mat,
            carapace_mesh: meshes.add(Cuboid::new(
                CARAPACE_HALF_W * 2.0,
                CARAPACE_HALF_H * 2.0,
                CARAPACE_HALF_D * 2.0,
            )),
            link_mesh: meshes.add(Capsule3d::new(COXA_RAD, COXA_LEN * 2.0)),
            recipe,
        }
    }
}

/// Marker for the crab's root carapace entity.
#[derive(Component)]
pub struct CrabCarapace;

/// Marker on the eye-tip link (the bone the eye rides). The reward reads its world
/// height (DeepMind-`stand`-style head height); the eye-stalks are locked, so they
/// carry no `CrabJoint` and this marker is how the reward locates them.
#[derive(Component)]
pub struct CrabEyeTip;

/// Marker applied to ALL crab body parts (carapace + limb segments).
#[derive(Component)]
pub struct CrabBodyPart;

/// Which training environment (crab instance) an entity belongs to. Every crab
/// entity carries one; systems group by it so N crabs sharing the world stay
/// independent samples. Demo/screenshot run a single env 0.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrabEnvId(pub usize);

/// A policy-driven joint on the crab: its observation/action slot key ([`id`](Self::id))
/// plus the per-instance data the sensor and actuator read. The free axis is
/// rig-derived — it varies per leg/side with the bind-pose bone geometry — so it
/// rides on the component, not a type-level constant. Locked rig joints (the
/// proximal leg stubs, claw mid-segments, eye-stalks, palpi) carry NO `CrabJoint`,
/// so they are invisible to the policy: present in the physics, out of the action space.
#[derive(Component, Clone, Copy, Debug)]
pub struct CrabJoint {
    pub id: CrabJointId,
    /// Free axis as a unit vector in the PARENT link's frame — the vector the
    /// actuator rotates into world to apply torque, and the sensor projects
    /// relative motion onto to read the DOF rate. Derived at spawn from the rig.
    pub axis_local: Vec3,
}

/// Every POLICY-ACTUATED joint — the locomotion-relevant subset of the crab's
/// articulation. The body has many more *physical* joints (the proximal leg
/// stubs, claw mid-segments, eye-stalks, palpi) that spawn from the rig as locked
/// links with no [`CrabJoint`]; promoting one to policy control means adding a
/// variant here (which grows the observation/action vector and the net).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CrabJointId {
    // Legs (8 = side L/R × leg 0..3 front→back). Coxa swings the leg off the
    // body; merus and carpus are the two load-bearing bends down the limb.
    LegCoxa(Side, u8),
    LegMerus(Side, u8),
    LegCarpus(Side, u8),
    // Claws (1 per side): shoulder lifts the arm, wrist bends the hand, pincer
    // opens/closes the movable finger.
    ClawShoulder(Side),
    ClawWrist(Side),
    ClawPincer(Side),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Side {
    Left,
    Right,
}

impl CrabJointId {
    /// Total policy-actuated DOFs — sets the observation/action vector width and
    /// the net's input/output size. Locked rig joints are excluded (no [`CrabJoint`]).
    pub const COUNT: usize = 8 * 3 + 2 * 3; // 24 leg + 6 claw = 30

    /// Flat observation/action slot for this joint (0..COUNT). A bijection — the
    /// `index_is_a_bijection` test pins that no two joints alias a slot.
    pub fn index(&self) -> usize {
        match self {
            CrabJointId::LegCoxa(side, leg) => side_offset(*side) * 12 + (*leg as usize) * 3,
            CrabJointId::LegMerus(side, leg) => side_offset(*side) * 12 + (*leg as usize) * 3 + 1,
            CrabJointId::LegCarpus(side, leg) => side_offset(*side) * 12 + (*leg as usize) * 3 + 2,
            CrabJointId::ClawShoulder(side) => 24 + side_offset(*side) * 3,
            CrabJointId::ClawWrist(side) => 24 + side_offset(*side) * 3 + 1,
            CrabJointId::ClawPincer(side) => 24 + side_offset(*side) * 3 + 2,
        }
    }

    /// Every joint, once. Order is irrelevant (callers needing a stable slot use
    /// [`index`](Self::index)); this lets a test enumerate the set without
    /// re-deriving it by hand. Yields exactly `COUNT` items.
    #[cfg(test)]
    fn all() -> impl Iterator<Item = CrabJointId> {
        [Side::Left, Side::Right].into_iter().flat_map(|side| {
            (0u8..4)
                .flat_map(move |leg| {
                    [
                        CrabJointId::LegCoxa(side, leg),
                        CrabJointId::LegMerus(side, leg),
                        CrabJointId::LegCarpus(side, leg),
                    ]
                })
                .chain([
                    CrabJointId::ClawShoulder(side),
                    CrabJointId::ClawWrist(side),
                    CrabJointId::ClawPincer(side),
                ])
        })
    }
}

impl CrabJointId {
    /// Peak DIRECT-DRIVE torque an action of ±1 commands on this joint — the
    /// magnitude the actuator applies via `ExternalForce`, graded by inertia
    /// (weakest on the lightest, most distal links). The policy's drive authority,
    /// distinct from the joint's friction motor ([`Self::friction_cap`]).
    pub fn drive_torque_ceiling(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(..) => COXA_TORQUE_CEILING,
            CrabJointId::LegMerus(..) => FEMUR_TORQUE_CEILING,
            CrabJointId::LegCarpus(..) => TIBIA_TORQUE_CEILING,
            CrabJointId::ClawShoulder(_)
            | CrabJointId::ClawWrist(_)
            | CrabJointId::ClawPincer(_) => CLAW_TORQUE_CEILING,
        }
    }

    /// Breakaway torque of this joint's friction motor (see [`FRICTION_RAMP`]):
    /// the constant an external load must beat to back-drive the joint, so legs
    /// crumple under ground contact and a modest command still actuates.
    pub fn friction_cap(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(..) | CrabJointId::LegMerus(..) | CrabJointId::LegCarpus(..) => {
                LEG_FRICTION_CAP
            }
            CrabJointId::ClawShoulder(_)
            | CrabJointId::ClawWrist(_)
            | CrabJointId::ClawPincer(_) => CLAW_FRICTION_CAP,
        }
    }

    /// Joint travel limits `[lo, hi]` in radians about the rig BIND POSE: the spawn
    /// bakes each link's bind orientation onto Rapier coordinate 0, so the range
    /// straddles 0 (0 = the bone's rest angle in the model). The policy commands
    /// torque, not position, so these are the hard stops the limb cannot pass, not
    /// a target. Per-family defaults, refined once the body stands.
    pub fn limits(&self) -> [f32; 2] {
        match self {
            CrabJointId::LegCoxa(..) => [-0.8, 0.8],
            CrabJointId::LegMerus(..) => [-1.0, 1.0],
            CrabJointId::LegCarpus(..) => [-1.1, 1.1],
            CrabJointId::ClawShoulder(_) => [-1.0, 1.0],
            CrabJointId::ClawWrist(_) => [-1.2, 1.2],
            CrabJointId::ClawPincer(_) => [-0.5, 0.2],
        }
    }
}

fn side_offset(side: Side) -> usize {
    match side {
        Side::Left => 0,
        Side::Right => 1,
    }
}

// ---------------------------------------------------------------------------
// Spawn the crab — instantiate the rig-derived recipe
// ---------------------------------------------------------------------------

/// Spawns a complete crab body at the given position, instantiating the
/// rig-derived [`RigRecipe`] (the one model). Returns the carapace entity.
///
/// Links are emitted parent-before-child so each joint can reference its already-
/// spawned parent entity. At rest every link is axis-aligned, so a link's world
/// position is just its parent's plus the joint anchor — tracked here to seed each
/// body's initial `Transform` near the pose the multibody solver will hold.
pub fn spawn_crab(
    commands: &mut Commands,
    assets: &CrabAssets,
    position: Vec3,
    env: usize,
) -> Entity {
    // Unreachable in normal runs: `main`'s preflight rejects a missing model or one
    // that builds no recipe (exit 1) before any spawn. The expect guards the path
    // that constructs `CrabAssets` without that preflight (e.g. a future caller).
    let recipe = assets
        .recipe
        .as_ref()
        .expect("CrabAssets built without a rig recipe — main's model preflight should have caught this");
    let origin = position + Vec3::new(0.0, SPAWN_HEIGHT, 0.0);

    // -- Carapace (root): the rigid trunk; shell/thorax/rostrum/abdomen ride it.
    let carapace = commands
        .spawn((
            CrabCarapace,
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            Collider::cuboid(
                recipe.carapace_half.x,
                recipe.carapace_half.y,
                recipe.carapace_half.z,
            ),
            CRAB_COLLISION,
            ColliderMassProperties::Density(recipe.carapace_density),
            Mesh3d(assets.carapace_mesh.clone()),
            MeshMaterial3d(assets.body_mat.clone()),
            Transform::from_translation(origin),
            Velocity::default(),
            ExternalForce::default(),
            // No `Damping`: Rapier applies per-body damping only to non-multibody
            // bodies, and the carapace is the multibody root, so it would be a no-op.
        ))
        .id();

    let mut ents: Vec<Entity> = Vec::with_capacity(recipe.links.len());
    let mut world_pos: Vec<Vec3> = Vec::with_capacity(recipe.links.len());
    // A world point inside the carapace box (centred at the spawn `origin`).
    let inside_carapace = |p: Vec3| (p - origin).abs().cmple(recipe.carapace_half).all();
    for link in &recipe.links {
        let (parent_ent, parent_pos) = if link.parent == rig::CARAPACE {
            (carapace, origin)
        } else {
            (ents[link.parent], world_pos[link.parent])
        };
        let here = parent_pos + link.anchor1; // identity rest rotation → plain add
        let collider = capsule_collider(link.center, link.col_rot, link.half_height, link.radius);
        // A link whose collider center sits inside the carapace box is a proximal
        // stub tucked under the shell; group it so it can't fight the carapace
        // collider (see [`NESTED_COLLISION`]). Distal limb segments reach outside the
        // box and keep full crab collision (ground + sibling limbs).
        let groups = if inside_carapace(here + link.center) {
            NESTED_COLLISION
        } else {
            CRAB_COLLISION
        };
        let joint = match link.actuated {
            Some(id) => rig_revolute(id, link.axis_local, link.anchor1),
            None => rig_fixed(link.anchor1),
        };
        let mut ec = commands.spawn((
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            collider,
            groups,
            ColliderMassProperties::Density(link.density),
            Mesh3d(assets.link_mesh.clone()),
            MeshMaterial3d(link_material(link, assets)),
            MultibodyJoint::new(parent_ent, joint),
            Transform::from_translation(here),
            Velocity::default(),
            ExternalForce::default(),
        ));
        if let Some(id) = link.actuated {
            ec.insert(CrabJoint {
                id,
                axis_local: link.axis_local,
            });
        }
        // Grippy feet: the distal leg bone (`004`) is what plants on the ground.
        if link.bone.starts_with("Def_leg") && link.bone.contains(".004.") {
            ec.insert(Friction::coefficient(1.5));
        }
        // The eye rides the stalk tip — mark it so the reward can read eye height.
        if link.bone.starts_with("Def_antennae_top") {
            ec.insert(CrabEyeTip);
        }
        ents.push(ec.id());
        world_pos.push(here);
    }

    carapace
}

/// Revolute joint for a policy-actuated link: free about `axis` (parent frame),
/// hard-limited, with a viscous friction motor and no contact against the adjacent
/// link. Coordinate 0 is the bind pose (links spawn axis-aligned), so the limits
/// straddle 0 directly — no rest bake (the hand-coded body needed one because its
/// flat-splay coordinate 0 sat outside the limits; the rig bind pose does not).
///
/// `RevoluteJointBuilder::new(axis)` writes `axis` into BOTH bodies' local frames,
/// which is only correct while children spawn at identity (parent frame == child
/// frame == world). When phase 2 bakes each bone's bind orientation into the child,
/// the child-side axis/anchor must be re-expressed in the child frame, or the solver
/// will constrain a different axis than the sensor/actuator read.
fn rig_revolute(id: CrabJointId, axis: Vec3, anchor1: Vec3) -> TypedJoint {
    let [lo, hi] = id.limits();
    let mut joint = no_adjacent_contacts(
        RevoluteJointBuilder::new(axis)
            .local_anchor1(anchor1)
            .local_anchor2(Vec3::ZERO)
            .limits([lo, hi])
            .motor_velocity(0.0, FRICTION_RAMP)
            .motor_max_force(id.friction_cap())
            .motor_model(MotorModel::ForceBased),
    );
    let generic: &mut GenericJoint = joint.as_mut();
    generic.raw.softness = LIMIT_SOFTNESS;
    joint
}

/// Fixed joint for a locked rig link: welds the child to its parent at the bind
/// pose (no DOF), so the joint is present in the body — promotable to actuated by
/// adding a [`CrabJointId`] for it — but invisible to the policy.
fn rig_fixed(anchor1: Vec3) -> TypedJoint {
    no_adjacent_contacts(
        FixedJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(Vec3::ZERO),
    )
}

/// A capsule collider offset within its link: the bone runs from the pivot (link
/// origin) along `rot·Y`, so the shape sits centred at `center` and oriented to
/// match — the same offset-baked-into-the-shape trick the fitted path uses.
fn capsule_collider(center: Vec3, rot: Quat, half_height: f32, radius: f32) -> Collider {
    let axis = rot * Vec3::Y * half_height;
    Collider::capsule(center - axis, center + axis, radius)
}

/// Debug material for a link, by limb family (legs orange, claws red, eye-stalks
/// and palpi pale) — read off the bone name.
fn link_material(link: &RigLink, assets: &CrabAssets) -> Handle<StandardMaterial> {
    if link.bone.starts_with("Def_leg") {
        assets.leg_mat.clone()
    } else if link.bone.starts_with("Def_pincer") {
        assets.claw_mat.clone()
    } else {
        assets.eye_mat.clone()
    }
}

/// Every physics part with the hand-coded density its mass is computed under, in
/// a deterministic order (carapace, legs L/R, claws L/R). The single source of the
/// part list + density used by the offline collider bake
/// ([`super::meshfit::bake_report`]): the fit keeps this density per part so a
/// fitted primitive's mass matches the body's at equal density. Lives here because
/// body.rs owns the joint set and these densities.
pub fn part_densities() -> Vec<(PartId, f32)> {
    let mut v = vec![(PartId::Carapace, CARAPACE_DENSITY)];
    for side in [Side::Left, Side::Right] {
        for leg in 0u8..4 {
            v.push((PartId::Joint(CrabJointId::LegCoxa(side, leg)), COXA_DENSITY));
            v.push((
                PartId::Joint(CrabJointId::LegMerus(side, leg)),
                FEMUR_DENSITY,
            ));
            v.push((
                PartId::Joint(CrabJointId::LegCarpus(side, leg)),
                TIBIA_DENSITY,
            ));
        }
    }
    for side in [Side::Left, Side::Right] {
        v.push((PartId::Joint(CrabJointId::ClawShoulder(side)), CLAW_DENSITY));
        v.push((PartId::Joint(CrabJointId::ClawWrist(side)), CLAW_DENSITY));
        v.push((PartId::Joint(CrabJointId::ClawPincer(side)), CLAW_DENSITY));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `index` must be a bijection onto `0..COUNT`. Every action, observation,
    /// and per-joint array is keyed by it, so a collision would silently alias
    /// two joints — one never actuated, the other actuated twice — and still
    /// "train". Pin that every joint maps to a distinct slot covering the range.
    #[test]
    fn index_is_a_bijection() {
        let mut seen = [false; CrabJointId::COUNT];
        let mut count = 0;
        for id in CrabJointId::all() {
            let i = id.index();
            assert!(i < CrabJointId::COUNT, "{id:?} index {i} out of range");
            assert!(!seen[i], "{id:?} aliases slot {i}");
            seen[i] = true;
            count += 1;
        }
        assert_eq!(
            count,
            CrabJointId::COUNT,
            "all() yielded the wrong joint count"
        );
        assert!(seen.iter().all(|&s| s), "index leaves a slot unfilled");
    }
}
