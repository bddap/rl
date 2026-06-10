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
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

// ---------------------------------------------------------------------------
// Collision groups — prevent self-collision within the crab
// ---------------------------------------------------------------------------

/// Group 1: arena (ground + walls). Interacts with crab.
pub const ARENA_COLLISION: CollisionGroups = CollisionGroups::new(Group::GROUP_1, Group::GROUP_2);
/// Group 2: crab body parts. Interact with arena but NOT with each other.
pub const CRAB_COLLISION: CollisionGroups = CollisionGroups::new(Group::GROUP_2, Group::GROUP_1);

// ---------------------------------------------------------------------------
// Dimensions — tuned for a ~1m wide crab (game scale, not real life)
// ---------------------------------------------------------------------------

/// Carapace: wide, flat, slightly domed.
const CARAPACE_HALF_W: f32 = 0.5; // x (left-right)
const CARAPACE_HALF_H: f32 = 0.12; // y (up-down) — very flat
const CARAPACE_HALF_D: f32 = 0.35; // z (front-back)

/// Spawn height: how high above ground the carapace center starts.
const SPAWN_HEIGHT: f32 = 1.0;

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

/// Marker for the crab's root carapace entity.
#[derive(Component)]
pub struct CrabCarapace;

/// Marker applied to ALL crab body parts (carapace + limb segments).
#[derive(Component)]
pub struct CrabBodyPart;

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

    /// Returns the default (rest) motor position for this joint.
    pub fn default_position(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(_, leg_idx) => {
                let splay_angles = [0.3_f32, 0.1, -0.1, -0.3];
                splay_angles[*leg_idx as usize]
            }
            CrabJointId::LegFemur(_, _) => -0.4,
            CrabJointId::LegTibia(_, _) => 0.8,
            CrabJointId::ClawUpper(_) => 0.3,
            CrabJointId::ClawFore(_) => 0.0,
            CrabJointId::ClawPincer(_) => 0.0,
            CrabJointId::EyeStalk(_) => 0.2,
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
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    position: Vec3,
) -> Entity {
    let body_color = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.45, 0.55), // blue-grey carapace
        perceptual_roughness: 0.7,
        ..default()
    });
    let leg_color = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.4, 0.15), // orange legs (Sally Lightfoot!)
        perceptual_roughness: 0.6,
        ..default()
    });
    let claw_color = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.15, 0.15), // deep red claws
        perceptual_roughness: 0.5,
        ..default()
    });
    let eye_color = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.85, 0.7), // pale eye stalks
        perceptual_roughness: 0.3,
        ..default()
    });

    // -- Carapace (root) ---------------------------------------------------
    let carapace = commands
        .spawn((
            CrabCarapace,
            CrabBodyPart,
            RigidBody::Dynamic,
            Collider::cuboid(CARAPACE_HALF_W, CARAPACE_HALF_H, CARAPACE_HALF_D),
            CRAB_COLLISION,
            ColliderMassProperties::Density(5.0),
            Mesh3d(meshes.add(Cuboid::new(
                CARAPACE_HALF_W * 2.0,
                CARAPACE_HALF_H * 2.0,
                CARAPACE_HALF_D * 2.0,
            ))),
            MeshMaterial3d(body_color.clone()),
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
            spawn_leg(commands, meshes, &leg_color, carapace, side, leg_idx);
        }
    }

    // -- Claws (1 per side) ------------------------------------------------
    for side in [Side::Left, Side::Right] {
        spawn_claw(commands, meshes, &claw_color, carapace, side);
    }

    // -- Eyes (1 per side) -------------------------------------------------
    for side in [Side::Left, Side::Right] {
        spawn_eye(commands, meshes, &eye_color, carapace, side);
    }

    carapace
}

// ---------------------------------------------------------------------------
// Leg: coxa → femur → tibia
// ---------------------------------------------------------------------------

fn spawn_leg(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color: &Handle<StandardMaterial>,
    carapace: Entity,
    side: Side,
    leg_idx: u8,
) {
    let s = side_sign(side);

    // Leg attachment points spread along the side of the carapace.
    // Legs numbered 0 (front) to 3 (back).
    let z_positions = [0.22, 0.08, -0.08, -0.22];
    let z = z_positions[leg_idx as usize];

    // Angle the legs outward and slightly backward/forward. Wider fan = broader
    // front-back support, the axis the (front-heavy) crab tips over.
    let splay_angles = [0.5_f32, 0.2, -0.2, -0.5]; // radians from perpendicular
    let splay = splay_angles[leg_idx as usize];

    // -- Coxa: rotates around Y (yaw — leg swings forward/back) -----------
    // Coxa yaw axis is side-mirrored (s·Y), like the femur/tibia s·Z. The
    // actuator overwrites motor targets every step with a per-joint-type value
    // (no side term), so symmetry MUST come from the joint axis, not the spawn
    // motor_position: a fixed +Y made the same target fan the legs opposite ways
    // (the left looked un-splayed). With s·Y the same target mirrors correctly.
    let coxa_joint = RevoluteJointBuilder::new(Vec3::new(0.0, s, 0.0))
        .local_anchor1(Vec3::new(s * CARAPACE_HALF_W, -0.02, z))
        .local_anchor2(Vec3::new(-s * COXA_LEN, 0.0, 0.0))
        .limits([-FRAC_PI_4, FRAC_PI_4])
        .motor_position(splay, LEG_STIFFNESS, LEG_DAMPING)
        .motor_max_force(LEG_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

    let coxa = commands
        .spawn((
            CrabBodyPart,
            CrabJoint {
                id: CrabJointId::LegCoxa(side, leg_idx),
            },
            RigidBody::Dynamic,
            Collider::capsule_y(COXA_LEN, COXA_RAD),
            CRAB_COLLISION,
            ColliderMassProperties::Density(2.0),
            Mesh3d(meshes.add(Capsule3d::new(COXA_RAD, COXA_LEN * 2.0))),
            MeshMaterial3d(color.clone()),
            MultibodyJoint::new(carapace, coxa_joint.into()),
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
        .limits([-FRAC_PI_2, FRAC_PI_4])
        .motor_position(-0.4, LEG_STIFFNESS, LEG_DAMPING)
        .motor_max_force(LEG_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

    let femur = commands
        .spawn((
            CrabBodyPart,
            CrabJoint {
                id: CrabJointId::LegFemur(side, leg_idx),
            },
            RigidBody::Dynamic,
            Collider::capsule_y(FEMUR_LEN, FEMUR_RAD),
            CRAB_COLLISION,
            ColliderMassProperties::Density(1.5),
            Mesh3d(meshes.add(Capsule3d::new(FEMUR_RAD, FEMUR_LEN * 2.0))),
            MeshMaterial3d(color.clone()),
            MultibodyJoint::new(coxa, femur_joint.into()),
            Velocity::default(),
        ))
        .id();

    // -- Tibia: rotates around ±Z by side (pitch — lower leg bends) -------
    // Side-dependent axis (s·Z), matching the femur, so the knee bends the same
    // way on both sides.
    let tibia_joint = RevoluteJointBuilder::new(Vec3::new(0.0, 0.0, s))
        .local_anchor1(Vec3::new(0.0, -FEMUR_LEN, 0.0))
        .local_anchor2(Vec3::new(0.0, TIBIA_LEN, 0.0))
        .limits([-0.1, PI * 0.6])
        .motor_position(0.8, LEG_STIFFNESS, LEG_DAMPING)
        .motor_max_force(LEG_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

    commands.spawn((
        CrabBodyPart,
        CrabJoint {
            id: CrabJointId::LegTibia(side, leg_idx),
        },
        RigidBody::Dynamic,
        Collider::capsule_y(TIBIA_LEN, TIBIA_RAD),
        CRAB_COLLISION,
        ColliderMassProperties::Density(1.0),
        Mesh3d(meshes.add(Capsule3d::new(TIBIA_RAD, TIBIA_LEN * 2.0))),
        MeshMaterial3d(color.clone()),
        MultibodyJoint::new(femur, tibia_joint.into()),
        Friction::coefficient(1.5), // grippy feet
        Velocity::default(),
    ));
}

// ---------------------------------------------------------------------------
// Claw: upper arm → forearm → pincer
// ---------------------------------------------------------------------------

fn spawn_claw(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color: &Handle<StandardMaterial>,
    carapace: Entity,
    side: Side,
) {
    let s = side_sign(side);

    // Claws attach at the front corners of the carapace
    let attach_point = Vec3::new(s * CARAPACE_HALF_W * 0.7, 0.05, CARAPACE_HALF_D * 0.9);

    // -- Upper arm: revolute around Z (pitch — raises/lowers claw) ---------
    let upper_joint = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor1(attach_point)
        .local_anchor2(Vec3::new(0.0, -CLAW_UPPER_LEN * 0.5, 0.0))
        .limits([-FRAC_PI_4, FRAC_PI_2])
        .motor_position(0.3, CLAW_STIFFNESS, CLAW_DAMPING)
        .motor_max_force(CLAW_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

    let upper = commands
        .spawn((
            CrabBodyPart,
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
            Mesh3d(meshes.add(Capsule3d::new(CLAW_UPPER_RAD, CLAW_UPPER_LEN * 2.0))),
            MeshMaterial3d(color.clone()),
            MultibodyJoint::new(carapace, upper_joint.into()),
            Velocity::default(),
        ))
        .id();

    // -- Forearm: revolute around Y (yaw — swings claw left/right) ---------
    let fore_joint = RevoluteJointBuilder::new(Vec3::Y)
        .local_anchor1(Vec3::new(0.0, CLAW_UPPER_LEN * 0.5, 0.0))
        .local_anchor2(Vec3::new(0.0, 0.0, -CLAW_FORE_LEN * 0.5))
        .limits([-FRAC_PI_2, FRAC_PI_2])
        .motor_position(0.0, CLAW_STIFFNESS, CLAW_DAMPING)
        .motor_max_force(CLAW_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

    let forearm = commands
        .spawn((
            CrabBodyPart,
            CrabJoint {
                id: CrabJointId::ClawFore(side),
            },
            RigidBody::Dynamic,
            Collider::capsule_y(CLAW_FORE_LEN, CLAW_FORE_RAD),
            CRAB_COLLISION,
            ColliderMassProperties::Density(1.0),
            Mesh3d(meshes.add(Capsule3d::new(CLAW_FORE_RAD, CLAW_FORE_LEN * 2.0))),
            MeshMaterial3d(color.clone()),
            MultibodyJoint::new(upper, fore_joint.into()),
            Velocity::default(),
        ))
        .id();

    // -- Pincer: prismatic along Z (open/close) ----------------------------
    let pincer_joint = PrismaticJointBuilder::new(Vec3::Z)
        .local_anchor1(Vec3::new(0.0, 0.0, CLAW_FORE_LEN * 0.5))
        .local_anchor2(Vec3::new(0.0, 0.0, -PINCER_HALF_D))
        .limits([0.0, 0.06]) // slightly opens
        .motor_position(0.0, CLAW_STIFFNESS, CLAW_DAMPING)
        .motor_max_force(CLAW_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

    commands.spawn((
        CrabBodyPart,
        CrabJoint {
            id: CrabJointId::ClawPincer(side),
        },
        RigidBody::Dynamic,
        Collider::cuboid(PINCER_HALF_W, PINCER_HALF_H, PINCER_HALF_D),
        CRAB_COLLISION,
        ColliderMassProperties::Density(1.0),
        Mesh3d(meshes.add(Cuboid::new(
            PINCER_HALF_W * 2.0,
            PINCER_HALF_H * 2.0,
            PINCER_HALF_D * 2.0,
        ))),
        MeshMaterial3d(color.clone()),
        MultibodyJoint::new(forearm, pincer_joint.into()),
        Velocity::default(),
    ));
}

// ---------------------------------------------------------------------------
// Eye stalk
// ---------------------------------------------------------------------------

fn spawn_eye(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    color: &Handle<StandardMaterial>,
    carapace: Entity,
    side: Side,
) {
    let s = side_sign(side);

    let attach = Vec3::new(s * 0.12, CARAPACE_HALF_H, CARAPACE_HALF_D * 0.7);

    // Eye stalk: revolute around X (pitch — looks up/down)
    let eye_joint = RevoluteJointBuilder::new(Vec3::X)
        .local_anchor1(attach)
        .local_anchor2(Vec3::new(0.0, -EYE_STALK_LEN, 0.0))
        .limits([-0.3, FRAC_PI_4])
        .motor_position(0.2, EYE_STIFFNESS, EYE_DAMPING)
        .motor_max_force(EYE_MAX_FORCE)
        .motor_model(MotorModel::AccelerationBased);

    commands.spawn((
        CrabBodyPart,
        CrabJoint {
            id: CrabJointId::EyeStalk(side),
        },
        RigidBody::Dynamic,
        Collider::ball(EYE_BALL_RAD),
        CRAB_COLLISION,
        ColliderMassProperties::Density(0.5),
        Mesh3d(meshes.add(Sphere::new(EYE_BALL_RAD))),
        MeshMaterial3d(color.clone()),
        MultibodyJoint::new(carapace, eye_joint.into()),
        Velocity::default(),
    ));
}
