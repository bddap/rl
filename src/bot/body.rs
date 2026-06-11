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

/// Crab groups during a post-reset settle: arena only, no crab-crab contacts.
/// A reset can't rewrite multibody joint coordinates (rapier 0.32), so the
/// limbs keep their dying pose and the motors must physically unfold them;
/// doing that under self-collision rams overlapping segments through each
/// other hard enough to NaN the solver. Settling crabs unfold contact-free,
/// then get [`CRAB_COLLISION`] back for the episode proper.
pub const CRAB_SETTLING_COLLISION: CollisionGroups =
    CollisionGroups::new(Group::GROUP_2, Group::GROUP_1);

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
// Motor parameters
// ---------------------------------------------------------------------------

const LEG_STIFFNESS: f32 = 200.0;
const LEG_DAMPING: f32 = 20.0;
const LEG_MAX_FORCE: f32 = 100.0;

const CLAW_STIFFNESS: f32 = 250.0;
const CLAW_DAMPING: f32 = 25.0;
const CLAW_MAX_FORCE: f32 = 150.0;

const EYE_STIFFNESS: f32 = 25.0;
const EYE_DAMPING: f32 = 5.0;
const EYE_MAX_FORCE: f32 = 10.0;

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
}

impl CrabJointId {
    /// Returns (stiffness, damping) for this joint's PD motor.
    pub fn motor_stiffness_damping(&self) -> (f32, f32) {
        match self {
            CrabJointId::EyeStalk(_) => (EYE_STIFFNESS, EYE_DAMPING),
            CrabJointId::ClawUpper(_) | CrabJointId::ClawFore(_) | CrabJointId::ClawPincer(_) => {
                (CLAW_STIFFNESS, CLAW_DAMPING)
            }
            _ => (LEG_STIFFNESS, LEG_DAMPING),
        }
    }

    /// Returns the Rapier joint axis for this DOF.
    /// Must match the axis used in the RevoluteJointBuilder / PrismaticJointBuilder.
    pub fn joint_axis(&self) -> JointAxis {
        match self {
            CrabJointId::LegCoxa(_, _) => JointAxis::AngY, // RevoluteJointBuilder::new(Vec3::Y)
            CrabJointId::LegFemur(_, _) => JointAxis::AngZ, // RevoluteJointBuilder::new(Vec3::Z)
            CrabJointId::LegTibia(_, _) => JointAxis::AngZ, // RevoluteJointBuilder::new(Vec3::Z)
            CrabJointId::ClawUpper(_) => JointAxis::AngZ,  // RevoluteJointBuilder::new(Vec3::Z)
            CrabJointId::ClawFore(_) => JointAxis::AngY,   // RevoluteJointBuilder::new(Vec3::Y)
            CrabJointId::ClawPincer(_) => JointAxis::LinZ, // PrismaticJointBuilder::new(Vec3::Z)
            CrabJointId::EyeStalk(_) => JointAxis::AngX,   // RevoluteJointBuilder::new(Vec3::X)
        }
    }

    /// Returns the motor_max_force for this joint type.
    pub fn motor_max_force(&self) -> f32 {
        match self {
            CrabJointId::EyeStalk(_) => EYE_MAX_FORCE,
            CrabJointId::ClawUpper(_) | CrabJointId::ClawFore(_) | CrabJointId::ClawPincer(_) => {
                CLAW_MAX_FORCE
            }
            _ => LEG_MAX_FORCE,
        }
    }

    /// The rest pose: this joint's angle in the planted stance. SINGLE SOURCE —
    /// the spawn motor target, the joint limits (rest ± half-width), and the
    /// actuator's action mapping (action 0 → rest) all derive from it, so the
    /// crab starts every episode planted and the policy's zero output holds
    /// that stance. Tune standing geometry HERE (+ SPAWN_HEIGHT), nowhere else.
    pub fn default_position(&self) -> f32 {
        match self {
            // Fan the legs front-to-back for a wide support polygon.
            CrabJointId::LegCoxa(_, leg_idx) => [-0.9_f32, -0.35, 0.35, 0.9][*leg_idx as usize],
            CrabJointId::LegFemur(_, _) => 0.5,
            CrabJointId::LegTibia(_, _) => 0.9,
            CrabJointId::ClawUpper(_) => 0.3,
            CrabJointId::ClawFore(_) => 0.0,
            CrabJointId::ClawPincer(_) => 0.03,
            CrabJointId::EyeStalk(_) => 0.2,
        }
    }

    /// Half-width of the commandable range around [`Self::default_position`].
    /// Joint limits and the actuator's action scaling both use this, so the
    /// reachable motion is never clipped by mismatched limits.
    pub fn action_half_width(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(_, _) => 0.78,
            CrabJointId::LegFemur(_, _) => 1.17,
            CrabJointId::LegTibia(_, _) => 0.99,
            CrabJointId::ClawUpper(_) => 1.17,
            CrabJointId::ClawFore(_) => 1.57,
            CrabJointId::ClawPincer(_) => 0.03, // prismatic: rest 0.03 ± 0.03 → 0..6 cm
            CrabJointId::EyeStalk(_) => 0.54,
        }
    }

    /// Joint limits: the commandable range, centered on the rest pose.
    pub fn limits(&self) -> [f32; 2] {
        let c = self.default_position();
        let hw = self.action_half_width();
        [c - hw, c + hw]
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
    // Coxa yaw axis is side-mirrored (s·Y), like the femur/tibia s·Z. The
    // actuator overwrites motor targets every step with a per-joint-type value
    // (no side term), so symmetry MUST come from the joint axis, not the spawn
    // motor_position: a fixed +Y made the same target fan the legs opposite ways
    // (the left looked un-splayed). With s·Y the same target mirrors correctly.
    let coxa_joint = RevoluteJointBuilder::new(Vec3::new(0.0, s, 0.0))
        .local_anchor1(Vec3::new(s * CARAPACE_HALF_W, -0.02, z))
        .local_anchor2(Vec3::new(-s * COXA_LEN, 0.0, 0.0))
        .limits(coxa_id.limits())
        .motor_position(coxa_id.default_position(), LEG_STIFFNESS, LEG_DAMPING)
        .motor_max_force(LEG_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

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
            ColliderMassProperties::Density(2.0),
            Mesh3d(assets.coxa_mesh.clone()),
            MeshMaterial3d(assets.leg_mat.clone()),
            MultibodyJoint::new(carapace, no_adjacent_contacts(coxa_joint)),
            Velocity::default(),
        ))
        .id();

    // -- Femur: rotates around ±Z by side (pitch — leg lifts up/down) ------
    // Side-dependent axis (s·Z), like the tibia, so the actuator's per-joint
    // target mirrors into a symmetric leg on both sides. Limits MUST match the
    // actuator's femur action_range [-1.57, 0.78] or the reachable lift is clipped.
    let femur_joint = RevoluteJointBuilder::new(Vec3::new(0.0, 0.0, s))
        .local_anchor1(Vec3::new(s * COXA_LEN, 0.0, 0.0))
        .local_anchor2(Vec3::new(0.0, FEMUR_LEN, 0.0))
        .limits(femur_id.limits())
        .motor_position(femur_id.default_position(), LEG_STIFFNESS, LEG_DAMPING)
        .motor_max_force(LEG_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

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
            ColliderMassProperties::Density(1.5),
            Mesh3d(assets.femur_mesh.clone()),
            MeshMaterial3d(assets.leg_mat.clone()),
            MultibodyJoint::new(coxa, no_adjacent_contacts(femur_joint)),
            Velocity::default(),
        ))
        .id();

    // -- Tibia: rotates around ±Z by side (pitch — lower leg bends) -------
    // Side-dependent axis (s·Z), matching the femur, so the knee bends the same
    // way on both sides.
    let tibia_joint = RevoluteJointBuilder::new(Vec3::new(0.0, 0.0, s))
        .local_anchor1(Vec3::new(0.0, -FEMUR_LEN, 0.0))
        .local_anchor2(Vec3::new(0.0, TIBIA_LEN, 0.0))
        .limits(tibia_id.limits())
        .motor_position(tibia_id.default_position(), LEG_STIFFNESS, LEG_DAMPING)
        .motor_max_force(LEG_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

    commands.spawn((
        CrabBodyPart,
        CrabEnvId(env),
        CrabJoint {
            id: CrabJointId::LegTibia(side, leg_idx),
        },
        RigidBody::Dynamic,
        Collider::capsule_y(TIBIA_LEN, TIBIA_RAD),
        CRAB_COLLISION,
        ColliderMassProperties::Density(1.0),
        Mesh3d(assets.tibia_mesh.clone()),
        MeshMaterial3d(assets.leg_mat.clone()),
        MultibodyJoint::new(femur, no_adjacent_contacts(tibia_joint)),
        Friction::coefficient(1.5), // grippy feet
        Velocity::default(),
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

    // Claws attach at the front corners of the carapace
    let attach_point = Vec3::new(s * CARAPACE_HALF_W * 0.7, 0.05, CARAPACE_HALF_D * 0.9);

    // -- Upper arm: revolute around Z (pitch — raises/lowers claw) ---------
    let upper_joint = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor1(attach_point)
        .local_anchor2(Vec3::new(0.0, -CLAW_UPPER_LEN * 0.5, 0.0))
        .limits(CrabJointId::ClawUpper(side).limits())
        .motor_position(
            CrabJointId::ClawUpper(side).default_position(),
            CLAW_STIFFNESS,
            CLAW_DAMPING,
        )
        .motor_max_force(CLAW_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

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
            MultibodyJoint::new(carapace, no_adjacent_contacts(upper_joint)),
            Velocity::default(),
        ))
        .id();

    // -- Forearm: revolute around Y (yaw — swings claw left/right) ---------
    let fore_joint = RevoluteJointBuilder::new(Vec3::Y)
        .local_anchor1(Vec3::new(0.0, CLAW_UPPER_LEN * 0.5, 0.0))
        .local_anchor2(Vec3::new(0.0, 0.0, -CLAW_FORE_LEN * 0.5))
        .limits(CrabJointId::ClawFore(side).limits())
        .motor_position(
            CrabJointId::ClawFore(side).default_position(),
            CLAW_STIFFNESS,
            CLAW_DAMPING,
        )
        .motor_max_force(CLAW_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

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
            MultibodyJoint::new(upper, no_adjacent_contacts(fore_joint)),
            Velocity::default(),
        ))
        .id();

    // -- Pincer: prismatic along Z (open/close) ----------------------------
    let pincer_joint = PrismaticJointBuilder::new(Vec3::Z)
        .local_anchor1(Vec3::new(0.0, 0.0, CLAW_FORE_LEN * 0.5))
        .local_anchor2(Vec3::new(0.0, 0.0, -PINCER_HALF_D))
        .limits(CrabJointId::ClawPincer(side).limits())
        .motor_position(
            CrabJointId::ClawPincer(side).default_position(),
            CLAW_STIFFNESS,
            CLAW_DAMPING,
        )
        .motor_max_force(CLAW_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

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

    let attach = Vec3::new(s * 0.12, CARAPACE_HALF_H, CARAPACE_HALF_D * 0.7);

    // Eye stalk: revolute around X (pitch — looks up/down)
    let eye_joint = RevoluteJointBuilder::new(Vec3::X)
        .local_anchor1(attach)
        .local_anchor2(Vec3::new(0.0, -EYE_STALK_LEN, 0.0))
        .limits(CrabJointId::EyeStalk(side).limits())
        .motor_position(
            CrabJointId::EyeStalk(side).default_position(),
            EYE_STIFFNESS,
            EYE_DAMPING,
        )
        .motor_max_force(EYE_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

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
        MultibodyJoint::new(carapace, no_adjacent_contacts(eye_joint)),
        Velocity::default(),
    ));
}
