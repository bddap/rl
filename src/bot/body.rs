//! Crab body definition: a physically-simulated articulated body built as a
//! Rapier multibody tree.
//!
//! Body plan (inspired by Sally Lightfoot crab reference):
//!
//! ```text
//!                     [Carapace]  ← root, wide flat ellipsoid
//!                    /    |     \
//!             [Eye_L] [Eye_R]   (stalks, 1 DOF each)
//!            /                   \
//!     [Claw_L]                 [Claw_R]
//!       upper  ─ revolute        upper  ─ revolute
//!       fore   ─ revolute        fore   ─ revolute
//!       pincer ─ prismatic       pincer ─ prismatic
//!
//!     [Leg_L1..L4]             [Leg_R1..R4]
//!       coxa  ─ revolute (yaw)
//!       femur ─ revolute (pitch)
//!       tibia ─ revolute (pitch)
//! ```

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

// ---------------------------------------------------------------------------
// Collision groups
// ---------------------------------------------------------------------------

/// Group 1: arena (ground + walls). Interacts with crab.
pub const ARENA_COLLISION: CollisionGroups = CollisionGroups::new(Group::GROUP_1, Group::GROUP_2);
/// Group 2: crab body parts. Interact with the arena AND with each other —
/// without self-collision the policy "tucks" legs through one another (free
/// interpenetration is an exploit, not a stance). Segments connected by a
/// joint are still contact-filtered per joint (`set_contacts_enabled(false)`),
/// so articulations don't fight their own contacts. N training crabs share
/// this group: their 4 m grid spacing exceeds any reach, so cross-crab
/// contact is geometrically impossible until we deliberately close the gap.
pub const CRAB_COLLISION: CollisionGroups =
    CollisionGroups::new(Group::GROUP_2, Group::GROUP_1.union(Group::GROUP_2));

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

/// Build a revolute leg/claw/eye joint between the given anchors: viscous
/// friction, no contact with the adjacent segment, and — the point of this
/// helper — the rest pose baked into the joint frame so the crab spawns
/// *standing*. Everything but the two anchors derives from `id`, so the free
/// axis used for the constraint, the actuator, the sensor, and the bake is one
/// value (passing a mismatched axis is not expressible).
///
/// A Rapier multibody initialises every joint coordinate to 0. With identity
/// frames that puts each child at angle 0 relative to its parent — a flat splay
/// that, for the knees and coxae, is *outside* the joint's own limits, so the
/// solver snapped them on the first tick. Nothing drives a joint to
/// [`CrabJointId::default_position`] either: control is direct torque, not a
/// position servo. So we install the rest pose geometrically.
///
/// Rotating the parent frame by `Rot(axis, rest)` makes coordinate 0 *be* the
/// rest angle; the coordinate limits then shift by `−rest` to keep the physical
/// range `[lo, hi]` unchanged. The bake turns the frame about the joint's own
/// axis, so the principal (free) axis is fixed: the actuator torque and the
/// [`joint_angle`] sensor read true world orientation, not the Rapier
/// coordinate, and are untouched — only the spawn pose and the coordinate zero
/// move. The prismatic pincer is left alone (its coordinate-0 closed stop is
/// already legal), so this helper is revolute-only.
fn revolute_joint(id: CrabJointId, anchor1: Vec3, anchor2: Vec3) -> TypedJoint {
    let axis = id.joint_axis_local();
    let rest = id.default_position();
    let [lo, hi] = id.limits();
    let mut joint = no_adjacent_contacts(
        RevoluteJointBuilder::new(axis)
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .limits([lo - rest, hi - rest])
            .motor_velocity(0.0, FRICTION_RAMP)
            .motor_max_force(id.friction_cap())
            .motor_model(MotorModel::ForceBased),
    );
    let generic: &mut GenericJoint = joint.as_mut();
    generic.set_local_basis1(Quat::from_axis_angle(axis, rest) * generic.local_basis1());
    generic.raw.softness = LIMIT_SOFTNESS;
    joint
}

/// Signed joint coordinate read off the two links' world orientations: the
/// twist of child-in-parent about the joint's free axis. Shared by the
/// observation and the telemetry overlay so both report the same angle the
/// policy acts on. (The prismatic pincer barely twists, so it reads ~0 here;
/// its open/close shows up in the velocity term instead — it is the one joint
/// whose coordinate is a translation, and it is the least balance-relevant.)
pub fn joint_angle(id: CrabJointId, parent_rot: Quat, child_rot: Quat) -> f32 {
    let n = id.joint_axis_local();
    let q = (parent_rot.inverse() * child_rot).normalize();
    let v = Vec3::new(q.x, q.y, q.z);
    2.0 * v.dot(n).atan2(q.w)
}

// ---------------------------------------------------------------------------
// Dimensions — tuned for a ~1m wide crab (game scale, not real life)
// ---------------------------------------------------------------------------

/// Carapace: wide, flat, slightly domed.
const CARAPACE_HALF_W: f32 = 0.5; // x (left-right)
const CARAPACE_HALF_H: f32 = 0.12; // y (up-down) — very flat
const CARAPACE_HALF_D: f32 = 0.35; // z (front-back)

/// Spawn height: how high above ground the carapace center starts.
pub const SPAWN_HEIGHT: f32 = 0.58;

/// Leg segment dimensions (capsule half-height, radius).
const COXA_LEN: f32 = 0.15;
const COXA_RAD: f32 = 0.045;
const FEMUR_LEN: f32 = 0.18;
const FEMUR_RAD: f32 = 0.035;
const TIBIA_LEN: f32 = 0.2;
const TIBIA_RAD: f32 = 0.025;

/// Leg-segment densities, tapered UP (denser toward the tip). At game scale the
/// distal links came out SUB-GRAM at density ~1 (the tibia under a gram) —
/// physically implausible and numerically twitchy: at tiny inertia, any
/// load-bearing torque produces five-figure angular acceleration. Denser distal
/// segments compensate the shrinking cross-section, landing each at ~10–14 g with a
/// gentle inertia gradient up the leg. Geometry is unchanged, so the spawn pose holds.
const COXA_DENSITY: f32 = 6.0;
const FEMUR_DENSITY: f32 = 8.0;
const TIBIA_DENSITY: f32 = 14.0;

/// Claw segment dimensions.
const CLAW_UPPER_LEN: f32 = 0.2;
const CLAW_UPPER_RAD: f32 = 0.055;
const CLAW_FORE_LEN: f32 = 0.2;
const CLAW_FORE_RAD: f32 = 0.05;
const PINCER_HALF_W: f32 = 0.08;
const PINCER_HALF_H: f32 = 0.03;
const PINCER_HALF_D: f32 = 0.12;

/// Eye stalk dimensions.
const EYE_STALK_LEN: f32 = 0.08;
const _EYE_STALK_RAD: f32 = 0.02; // reserved for stalk mesh when we upgrade from sphere
const EYE_BALL_RAD: f32 = 0.03;

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
const EYE_TORQUE_CEILING: f32 = 0.5;

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
const EYE_FRICTION_CAP: f32 = 0.05;

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
    coxa_mesh: Handle<Mesh>,
    femur_mesh: Handle<Mesh>,
    tibia_mesh: Handle<Mesh>,
    claw_upper_mesh: Handle<Mesh>,
    claw_fore_mesh: Handle<Mesh>,
    pincer_mesh: Handle<Mesh>,
    eye_mesh: Handle<Mesh>,
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
            coxa_mesh: meshes.add(Capsule3d::new(COXA_RAD, COXA_LEN * 2.0)),
            femur_mesh: meshes.add(Capsule3d::new(FEMUR_RAD, FEMUR_LEN * 2.0)),
            tibia_mesh: meshes.add(Capsule3d::new(TIBIA_RAD, TIBIA_LEN * 2.0)),
            claw_upper_mesh: meshes.add(Capsule3d::new(CLAW_UPPER_RAD, CLAW_UPPER_LEN * 2.0)),
            claw_fore_mesh: meshes.add(Capsule3d::new(CLAW_FORE_RAD, CLAW_FORE_LEN * 2.0)),
            pincer_mesh: meshes.add(Cuboid::new(
                PINCER_HALF_W * 2.0,
                PINCER_HALF_H * 2.0,
                PINCER_HALF_D * 2.0,
            )),
            eye_mesh: meshes.add(Sphere::new(EYE_BALL_RAD)),
        }
    }
}

/// Marker for the crab's root carapace entity.
#[derive(Component)]
pub struct CrabCarapace;

/// Marker applied to ALL crab body parts (carapace + limb segments).
#[derive(Component)]
pub struct CrabBodyPart;

/// Which training environment (crab instance) an entity belongs to. Every crab
/// entity carries one; systems group by it so N crabs sharing the world stay
/// independent samples. Demo/screenshot run a single env 0.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrabEnvId(pub usize);

/// Identifies a specific joint on the crab for the sensor/actuator system.
#[derive(Component, Clone, Copy, Debug)]
pub struct CrabJoint {
    pub id: CrabJointId,
}

/// Every actuated joint on the crab.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CrabJointId {
    // Legs: side (L/R), leg index (0-3 front to back), segment
    LegCoxa(Side, u8),
    LegFemur(Side, u8),
    LegTibia(Side, u8),
    // Claws
    ClawUpper(Side),
    ClawFore(Side),
    ClawPincer(Side),
    // Eyes
    EyeStalk(Side),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Left,
    Right,
}

impl CrabJointId {
    /// Total number of actuated DOFs.
    pub const COUNT: usize = 8 * 3 + 2 * 3 + 2; // 24 + 6 + 2 = 32

    /// Returns a flat index for this joint (0..COUNT).
    pub fn index(&self) -> usize {
        match self {
            // Legs: 24 joints (8 legs × 3 segments)
            CrabJointId::LegCoxa(side, leg) => side_offset(*side) * 12 + (*leg as usize) * 3,
            CrabJointId::LegFemur(side, leg) => side_offset(*side) * 12 + (*leg as usize) * 3 + 1,
            CrabJointId::LegTibia(side, leg) => side_offset(*side) * 12 + (*leg as usize) * 3 + 2,
            // Claws: 6 joints (2 claws × 3)
            CrabJointId::ClawUpper(side) => 24 + side_offset(*side) * 3,
            CrabJointId::ClawFore(side) => 24 + side_offset(*side) * 3 + 1,
            CrabJointId::ClawPincer(side) => 24 + side_offset(*side) * 3 + 2,
            // Eyes: 2 joints
            CrabJointId::EyeStalk(side) => 30 + side_offset(*side),
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
                        CrabJointId::LegFemur(side, leg),
                        CrabJointId::LegTibia(side, leg),
                    ]
                })
                .chain([
                    CrabJointId::ClawUpper(side),
                    CrabJointId::ClawFore(side),
                    CrabJointId::ClawPincer(side),
                    CrabJointId::EyeStalk(side),
                ])
        })
    }
}

impl CrabJointId {
    /// The joint's free axis as a unit vector in the PARENT link's local frame
    /// — exactly the vector handed to `RevoluteJointBuilder::new` / the
    /// prismatic builder at spawn. Rotate it by the parent's world rotation to
    /// get the world axis a torque/force acts along, or to read the joint angle
    /// off the relative orientation. The left/right sign is baked in (the same
    /// action mirrors correctly across the body), so it is the single source
    /// for axis direction shared by the actuator, the sensor, and the overlay.
    pub fn joint_axis_local(&self) -> Vec3 {
        match self {
            CrabJointId::LegCoxa(side, _) => Vec3::new(0.0, side_sign(*side), 0.0),
            CrabJointId::LegFemur(side, _) | CrabJointId::LegTibia(side, _) => {
                Vec3::new(0.0, 0.0, side_sign(*side))
            }
            CrabJointId::ClawUpper(_) => Vec3::Z,
            CrabJointId::ClawFore(_) => Vec3::Y,
            CrabJointId::ClawPincer(_) => Vec3::Z,
            CrabJointId::EyeStalk(_) => Vec3::X,
        }
    }

    /// Peak DIRECT-DRIVE torque an action of ±1 commands on this joint — the
    /// magnitude the actuator applies via `ExternalForce`, graded by inertia (see
    /// the ceiling constants): weakest on the lightest, most distal links. Distinct
    /// from [`Self::friction_cap`] (the joint's friction motor) and from Rapier's
    /// own `motor_max_force` builder — this is the policy's drive authority, not a
    /// motor force.
    pub fn drive_torque_ceiling(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(..) => COXA_TORQUE_CEILING,
            CrabJointId::LegFemur(..) => FEMUR_TORQUE_CEILING,
            CrabJointId::LegTibia(..) => TIBIA_TORQUE_CEILING,
            CrabJointId::ClawUpper(_) | CrabJointId::ClawFore(_) | CrabJointId::ClawPincer(_) => {
                CLAW_TORQUE_CEILING
            }
            CrabJointId::EyeStalk(_) => EYE_TORQUE_CEILING,
        }
    }

    /// Breakaway torque of this joint's friction motor (see [`FRICTION_RAMP`]):
    /// the constant an external load must beat to back-drive the joint, so legs
    /// crumple under ground contact and a modest command still actuates.
    pub fn friction_cap(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(..) | CrabJointId::LegFemur(..) | CrabJointId::LegTibia(..) => {
                LEG_FRICTION_CAP
            }
            CrabJointId::ClawUpper(_) | CrabJointId::ClawFore(_) | CrabJointId::ClawPincer(_) => {
                CLAW_FRICTION_CAP
            }
            CrabJointId::EyeStalk(_) => EYE_FRICTION_CAP,
        }
    }

    /// The rest pose: this joint's angle in the planted stance the crab spawns
    /// in. SINGLE SOURCE — the spawn pose (`revolute_joint` bakes it into the
    /// joint frame so Rapier coordinate 0 *is* this angle) and the joint limits
    /// (centred on rest where the anatomy is symmetric) both derive from it. It
    /// is NOT a control target: the policy commands torque, not position, so an
    /// action of 0 is zero torque — it does not "hold rest". Tune standing
    /// geometry HERE (and SPAWN_HEIGHT), nowhere else.
    pub fn default_position(&self) -> f32 {
        match self {
            // Fan the legs front-to-back for a wide support polygon.
            CrabJointId::LegCoxa(_, leg_idx) => [-0.9_f32, -0.35, 0.35, 0.9][*leg_idx as usize],
            // Femur out near-horizontal + tibia folded back to vertical puts
            // the knee at the silhouette's high-out point with the foot
            // planted wide — the crab Λ. Tibia angle 0 = straight knee;
            // NEGATIVE folds crab-correct (down-under). Positive is the
            // backward "inward knee" bend (owner report, twice) — the femur
            // and tibia hinges share the +Z axis, so a positive tibia angle
            // rotates the shin up-and-out past the femur line.
            CrabJointId::LegFemur(_, _) => 1.2,
            CrabJointId::LegTibia(_, _) => -1.2,
            CrabJointId::ClawUpper(_) => 0.3,
            CrabJointId::ClawFore(_) => 0.0,
            CrabJointId::ClawPincer(_) => 0.03,
            CrabJointId::EyeStalk(_) => 0.2,
        }
    }

    /// Joint limits `[lo, hi]` in PHYSICAL angle: the range the joint
    /// constraint permits. `revolute_joint` feeds the builder these shifted by
    /// −rest, because the frame bake puts Rapier coordinate 0 at the rest angle;
    /// the physical range the limb can travel stays `[lo, hi]`. Leg ranges are
    /// ASYMMETRIC around rest where crab anatomy is one-directional: the femur
    /// sweeps straight-down to just past horizontal (never under the body), and
    /// the tibia is a one-way knee — angle 0 is a straight knee, the fold is
    /// strictly NEGATIVE (see default_position), and it always keeps some bend.
    pub fn limits(&self) -> [f32; 2] {
        let c = self.default_position();
        match self {
            CrabJointId::LegCoxa(_, _) => [c - 0.78, c + 0.78],
            CrabJointId::LegFemur(_, _) => [0.0, 1.8],
            CrabJointId::LegTibia(_, _) => [-1.89, -0.25],
            CrabJointId::ClawUpper(_) => [c - 1.17, c + 1.17],
            CrabJointId::ClawFore(_) => [c - 1.57, c + 1.57],
            CrabJointId::ClawPincer(_) => [0.0, 0.06],
            CrabJointId::EyeStalk(_) => [c - 0.54, c + 0.54],
        }
    }
}

fn side_offset(side: Side) -> usize {
    match side {
        Side::Left => 0,
        Side::Right => 1,
    }
}

fn side_sign(side: Side) -> f32 {
    match side {
        Side::Left => 1.0,
        Side::Right => -1.0,
    }
}

// ---------------------------------------------------------------------------
// Spawn the crab
// ---------------------------------------------------------------------------

/// Spawns a complete crab body at the given position.
/// Returns the carapace entity.
pub fn spawn_crab(
    commands: &mut Commands,
    assets: &CrabAssets,
    position: Vec3,
    env: usize,
) -> Entity {
    // -- Carapace (root) ---------------------------------------------------
    let carapace = commands
        .spawn((
            CrabCarapace,
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            Collider::cuboid(CARAPACE_HALF_W, CARAPACE_HALF_H, CARAPACE_HALF_D),
            CRAB_COLLISION,
            ColliderMassProperties::Density(5.0),
            Mesh3d(assets.carapace_mesh.clone()),
            MeshMaterial3d(assets.body_mat.clone()),
            Transform::from_translation(position + Vec3::new(0.0, SPAWN_HEIGHT, 0.0)),
            Velocity::default(),
            ExternalForce::default(),
            Damping {
                linear_damping: 0.5,
                angular_damping: 1.0,
            },
        ))
        .id();

    // -- Legs (4 per side) -------------------------------------------------
    for side in [Side::Left, Side::Right] {
        for leg_idx in 0u8..4 {
            spawn_leg(commands, assets, carapace, side, leg_idx, env);
        }
    }

    // -- Claws (1 per side) ------------------------------------------------
    for side in [Side::Left, Side::Right] {
        spawn_claw(commands, assets, carapace, side, env);
    }

    // -- Eyes (1 per side) -------------------------------------------------
    for side in [Side::Left, Side::Right] {
        spawn_eye(commands, assets, carapace, side, env);
    }

    carapace
}

// ---------------------------------------------------------------------------
// Leg: coxa → femur → tibia
// ---------------------------------------------------------------------------

fn spawn_leg(
    commands: &mut Commands,
    assets: &CrabAssets,
    carapace: Entity,
    side: Side,
    leg_idx: u8,
    env: usize,
) {
    let s = side_sign(side);

    // Leg attachment points spread along the side of the carapace.
    // Legs numbered 0 (front) to 3 (back).
    let z_positions = [0.22, 0.08, -0.08, -0.22];
    let z = z_positions[leg_idx as usize];

    let coxa_id = CrabJointId::LegCoxa(side, leg_idx);
    let femur_id = CrabJointId::LegFemur(side, leg_idx);
    let tibia_id = CrabJointId::LegTibia(side, leg_idx);

    // -- Coxa: rotates around Y (yaw — leg swings forward/back) -----------
    // The joint axis (from joint_axis_local) is side-mirrored — s·Y here, s·Z
    // for the femur/tibia. The policy emits one torque per joint *type* with no
    // side term and the rest bake applies one `rest` per type, so both mirror
    // into a symmetric stance only because the axis carries the side sign; a
    // fixed +Y would fan the two sides opposite ways.
    let coxa = commands
        .spawn((
            CrabBodyPart,
            CrabEnvId(env),
            CrabJoint {
                id: CrabJointId::LegCoxa(side, leg_idx),
            },
            RigidBody::Dynamic,
            Collider::capsule_y(COXA_LEN, COXA_RAD),
            CRAB_COLLISION,
            ColliderMassProperties::Density(COXA_DENSITY),
            Mesh3d(assets.coxa_mesh.clone()),
            MeshMaterial3d(assets.leg_mat.clone()),
            MultibodyJoint::new(
                carapace,
                revolute_joint(
                    coxa_id,
                    Vec3::new(s * CARAPACE_HALF_W, -0.02, z),
                    Vec3::new(-s * COXA_LEN, 0.0, 0.0),
                ),
            ),
            Velocity::default(),
            ExternalForce::default(),
        ))
        .id();

    // -- Femur: rotates around ±Z by side (pitch — leg lifts up/down) ------
    // Side-dependent axis (s·Z), like the tibia, so the per-joint-type torque
    // and rest bake mirror into a symmetric leg on both sides.
    let femur = commands
        .spawn((
            CrabBodyPart,
            CrabEnvId(env),
            CrabJoint {
                id: CrabJointId::LegFemur(side, leg_idx),
            },
            RigidBody::Dynamic,
            Collider::capsule_y(FEMUR_LEN, FEMUR_RAD),
            CRAB_COLLISION,
            ColliderMassProperties::Density(FEMUR_DENSITY),
            Mesh3d(assets.femur_mesh.clone()),
            MeshMaterial3d(assets.leg_mat.clone()),
            MultibodyJoint::new(
                coxa,
                revolute_joint(
                    femur_id,
                    Vec3::new(s * COXA_LEN, 0.0, 0.0),
                    Vec3::new(0.0, FEMUR_LEN, 0.0),
                ),
            ),
            Velocity::default(),
            ExternalForce::default(),
        ))
        .id();

    // -- Tibia: rotates around ±Z by side (pitch — lower leg bends) -------
    // Side-dependent axis (s·Z), matching the femur, so the knee bends the same
    // way on both sides.
    commands.spawn((
        CrabBodyPart,
        CrabEnvId(env),
        CrabJoint {
            id: CrabJointId::LegTibia(side, leg_idx),
        },
        RigidBody::Dynamic,
        Collider::capsule_y(TIBIA_LEN, TIBIA_RAD),
        CRAB_COLLISION,
        ColliderMassProperties::Density(TIBIA_DENSITY),
        Mesh3d(assets.tibia_mesh.clone()),
        MeshMaterial3d(assets.leg_mat.clone()),
        MultibodyJoint::new(
            femur,
            revolute_joint(
                tibia_id,
                Vec3::new(0.0, -FEMUR_LEN, 0.0),
                Vec3::new(0.0, TIBIA_LEN, 0.0),
            ),
        ),
        Friction::coefficient(1.5), // grippy feet
        Velocity::default(),
        ExternalForce::default(),
    ));
}

// ---------------------------------------------------------------------------
// Claw: upper arm → forearm → pincer
// ---------------------------------------------------------------------------

fn spawn_claw(
    commands: &mut Commands,
    assets: &CrabAssets,
    carapace: Entity,
    side: Side,
    env: usize,
) {
    let s = side_sign(side);
    let upper_id = CrabJointId::ClawUpper(side);
    let fore_id = CrabJointId::ClawFore(side);
    let pincer_id = CrabJointId::ClawPincer(side);

    // Claws attach at the front corners of the carapace
    let attach_point = Vec3::new(s * CARAPACE_HALF_W * 0.7, 0.05, CARAPACE_HALF_D * 0.9);

    // -- Upper arm: revolute around Z (pitch — raises/lowers claw) ---------
    let upper = commands
        .spawn((
            CrabBodyPart,
            CrabEnvId(env),
            CrabJoint {
                id: CrabJointId::ClawUpper(side),
            },
            RigidBody::Dynamic,
            Collider::capsule_y(CLAW_UPPER_LEN, CLAW_UPPER_RAD),
            CRAB_COLLISION,
            // Claws are light: the upper/forearm/pincer assemblies hang off the
            // front (+Z), and dense ones (was 3.0/2.5/4.0) put the centre of mass
            // ahead of the leg support so the crab pitched forward and couldn't
            // stand. Keep them low-mass so the CoM sits over the feet.
            ColliderMassProperties::Density(1.0),
            Mesh3d(assets.claw_upper_mesh.clone()),
            MeshMaterial3d(assets.claw_mat.clone()),
            MultibodyJoint::new(
                carapace,
                revolute_joint(
                    upper_id,
                    attach_point,
                    Vec3::new(0.0, -CLAW_UPPER_LEN * 0.5, 0.0),
                ),
            ),
            Velocity::default(),
            ExternalForce::default(),
        ))
        .id();

    // -- Forearm: revolute around Y (yaw — swings claw left/right) ---------
    let forearm = commands
        .spawn((
            CrabBodyPart,
            CrabEnvId(env),
            CrabJoint {
                id: CrabJointId::ClawFore(side),
            },
            RigidBody::Dynamic,
            Collider::capsule_y(CLAW_FORE_LEN, CLAW_FORE_RAD),
            CRAB_COLLISION,
            ColliderMassProperties::Density(1.0),
            Mesh3d(assets.claw_fore_mesh.clone()),
            MeshMaterial3d(assets.claw_mat.clone()),
            MultibodyJoint::new(
                upper,
                revolute_joint(
                    fore_id,
                    Vec3::new(0.0, CLAW_UPPER_LEN * 0.5, 0.0),
                    Vec3::new(0.0, 0.0, -CLAW_FORE_LEN * 0.5),
                ),
            ),
            Velocity::default(),
            ExternalForce::default(),
        ))
        .id();

    // -- Pincer: prismatic along Z (open/close) ----------------------------
    // Not rest-baked like the revolute joints: its coordinate-0 spawn is the
    // closed stop (limit lo = 0), which is already legal, so a frame bake would
    // only complicate the one translational joint for no gain.
    let pincer_joint = PrismaticJointBuilder::new(pincer_id.joint_axis_local())
        .local_anchor1(Vec3::new(0.0, 0.0, CLAW_FORE_LEN * 0.5))
        .local_anchor2(Vec3::new(0.0, 0.0, -PINCER_HALF_D))
        .limits(pincer_id.limits());

    commands.spawn((
        CrabBodyPart,
        CrabEnvId(env),
        CrabJoint {
            id: CrabJointId::ClawPincer(side),
        },
        RigidBody::Dynamic,
        Collider::cuboid(PINCER_HALF_W, PINCER_HALF_H, PINCER_HALF_D),
        CRAB_COLLISION,
        ColliderMassProperties::Density(1.0),
        Mesh3d(assets.pincer_mesh.clone()),
        MeshMaterial3d(assets.claw_mat.clone()),
        MultibodyJoint::new(forearm, no_adjacent_contacts(pincer_joint)),
        Velocity::default(),
        ExternalForce::default(),
    ));
}

// ---------------------------------------------------------------------------
// Eye stalk
// ---------------------------------------------------------------------------

fn spawn_eye(
    commands: &mut Commands,
    assets: &CrabAssets,
    carapace: Entity,
    side: Side,
    env: usize,
) {
    let s = side_sign(side);
    let eye_id = CrabJointId::EyeStalk(side);

    let attach = Vec3::new(s * 0.12, CARAPACE_HALF_H, CARAPACE_HALF_D * 0.7);

    // Eye stalk: revolute around X (pitch — looks up/down)
    commands.spawn((
        CrabBodyPart,
        CrabEnvId(env),
        CrabJoint {
            id: CrabJointId::EyeStalk(side),
        },
        RigidBody::Dynamic,
        Collider::ball(EYE_BALL_RAD),
        CRAB_COLLISION,
        ColliderMassProperties::Density(0.5),
        Mesh3d(assets.eye_mesh.clone()),
        MeshMaterial3d(assets.eye_mat.clone()),
        MultibodyJoint::new(
            carapace,
            revolute_joint(eye_id, attach, Vec3::new(0.0, -EYE_STALK_LEN, 0.0)),
        ),
        Velocity::default(),
        ExternalForce::default(),
    ));
}

/// Test-only re-export of the hand-coded collider dimensions and densities, so
/// the mesh-fit spike ([`super::meshfit`]) can compare its auto-derived colliders
/// against the live body's numbers without copying them (which would drift). The
/// `5.0`/`1.0`/`0.5` densities are inline at the spawn sites below; surface them
/// here as named consts to keep a single source.
#[cfg(test)]
pub mod reference {
    pub const CARAPACE_HALF_W: f32 = super::CARAPACE_HALF_W;
    pub const CARAPACE_HALF_H: f32 = super::CARAPACE_HALF_H;
    pub const CARAPACE_HALF_D: f32 = super::CARAPACE_HALF_D;
    pub const CARAPACE_DENSITY: f32 = 5.0; // inline at the carapace spawn

    pub const COXA_LEN: f32 = super::COXA_LEN;
    pub const COXA_RAD: f32 = super::COXA_RAD;
    pub const FEMUR_LEN: f32 = super::FEMUR_LEN;
    pub const FEMUR_RAD: f32 = super::FEMUR_RAD;
    pub const TIBIA_LEN: f32 = super::TIBIA_LEN;
    pub const TIBIA_RAD: f32 = super::TIBIA_RAD;
    pub const COXA_DENSITY: f32 = super::COXA_DENSITY;
    pub const FEMUR_DENSITY: f32 = super::FEMUR_DENSITY;
    pub const TIBIA_DENSITY: f32 = super::TIBIA_DENSITY;

    pub const CLAW_UPPER_LEN: f32 = super::CLAW_UPPER_LEN;
    pub const CLAW_UPPER_RAD: f32 = super::CLAW_UPPER_RAD;
    pub const CLAW_FORE_LEN: f32 = super::CLAW_FORE_LEN;
    pub const CLAW_FORE_RAD: f32 = super::CLAW_FORE_RAD;
    pub const PINCER_HALF_W: f32 = super::PINCER_HALF_W;
    pub const PINCER_HALF_H: f32 = super::PINCER_HALF_H;
    pub const PINCER_HALF_D: f32 = super::PINCER_HALF_D;
    pub const CLAW_DENSITY: f32 = 1.0; // inline at all three claw spawns

    pub const EYE_BALL_RAD: f32 = super::EYE_BALL_RAD;
    pub const EYE_DENSITY: f32 = 0.5; // inline at the eye spawn
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

    /// The rest pose must be legal: every joint's `default_position` lies inside
    /// its own limits. The spawn bakes coordinate 0 onto `default_position` (see
    /// `revolute_joint`), so a rest angle outside the limits would spawn the crab
    /// pre-violated — the very bug the frame bake fixed. This is the static,
    /// sim-free half of `crab_spawns_in_rest_pose_inside_limits`.
    #[test]
    fn rest_pose_is_inside_limits() {
        for id in CrabJointId::all() {
            let rest = id.default_position();
            let [lo, hi] = id.limits();
            assert!(
                (lo..=hi).contains(&rest),
                "{id:?} rest {rest:+.3} outside limits [{lo:+.3}, {hi:+.3}]"
            );
        }
    }
}
