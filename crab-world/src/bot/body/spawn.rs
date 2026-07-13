use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::collision::{NESTED_COLLISION, crab_collision, no_adjacent_contacts};
use super::components::{
    CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint, CrabRestPose,
};
use super::joint_id::CrabJointId;
use crate::bot::rig;

pub const SPAWN_HEIGHT: f32 = 0.05;

const FRICTION_RAMP: f32 = 4.0;

pub const LIMIT_SOFTNESS: bevy_rapier3d::rapier::dynamics::SpringCoefficients<f32> =
    bevy_rapier3d::rapier::dynamics::SpringCoefficients {
        natural_frequency: 400.0,
        damping_ratio: 2.0,
    };

/// A random spawn orientation for respawns (every training reset, and demo resets):
/// ~80% a mild tilt (≤ ~25°) off upright, ~20% a heavy tilt up to fully inverted —
/// each about a random horizontal axis, with a random yaw on top. Forces the policy
/// to stand and right itself from a varied start rather than memorising the one bind
/// pose.
///
/// Callers are training resets (live iff `wgpu` — see `training`'s module allow) and
/// demo resets (`play`, render-gated), hence the two-feature allow.
#[cfg_attr(not(any(feature = "wgpu", feature = "render")), allow(dead_code))]
pub(crate) fn random_spawn_rotation(rng: &mut impl rand::Rng) -> Quat {
    use std::f32::consts::{PI, TAU};
    let yaw = rng.gen_range(0.0..TAU);
    let tilt = if rng.r#gen::<f32>() < 0.8 {
        rng.gen_range(0.0f32..0.44)
    } else {
        rng.gen_range(0.44..PI)
    };
    let az = rng.gen_range(0.0..TAU);
    let tilt_axis = Vec3::new(az.cos(), 0.0, az.sin());
    Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(tilt_axis, tilt)
}

pub fn spawn_crab(
    commands: &mut Commands,
    assets: &CrabAssets,
    position: Vec3,
    env: usize,
    init_rotation: Quat,
) -> Entity {
    let recipe = &assets.recipe;
    let origin = position + recipe.hub_bind_world + Vec3::new(0.0, SPAWN_HEIGHT, 0.0);

    let world_pos = rig::link_world_origins(&recipe.links, origin);

    let carapace_r = recipe.carapace_offset.length() + recipe.carapace_half.length();
    let mut low_unrot = origin.y - carapace_r;
    let mut low_rot = origin.y - carapace_r;
    for (link, &p) in recipe.links.iter().zip(&world_pos) {
        let r = link.bounding_radius();
        low_unrot = low_unrot.min(p.y - r);
        low_rot = low_rot.min((origin + init_rotation * (p - origin)).y - r);
    }
    let lift = (low_unrot - low_rot).max(0.0);
    let place = |p: Vec3| {
        Transform::from_translation(origin + init_rotation * (p - origin) + Vec3::Y * lift)
            .with_rotation(init_rotation)
    };

    let carapace = commands
        .spawn((
            CrabCarapace,
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            Collider::compound(vec![(
                recipe.carapace_offset,
                Quat::IDENTITY,
                Collider::cuboid(
                    recipe.carapace_half.x,
                    recipe.carapace_half.y,
                    recipe.carapace_half.z,
                ),
            )]),
            crab_collision(env),
            ColliderMassProperties::Density(recipe.carapace_density),
            place(origin),
            CrabRestPose(place(origin)),
            Velocity::default(),
            ExternalForce::default(),
        ))
        .id();

    let mut ents: Vec<Entity> = Vec::with_capacity(recipe.links.len());
    let inside_carapace = |p: Vec3| {
        (p - origin - recipe.carapace_offset)
            .abs()
            .cmple(recipe.carapace_half)
            .all()
    };
    for (i, link) in recipe.links.iter().enumerate() {
        if link.actuated.is_none() {
            ents.push(carapace);
            continue;
        }
        let parent_ent = match link.parent {
            None => carapace,
            Some(idx) => ents[idx],
        };
        let here = world_pos[i];
        let collider = match rig::link_rest_shape(link, Vec3::ZERO) {
            rig::RestShape::Capsule { a, b, radius } => Collider::capsule(a, b, radius),
            rig::RestShape::Cuboid { center, rot, half } => Collider::compound(vec![(
                center,
                rot,
                Collider::cuboid(half.x, half.y, half.z),
            )]),
        };
        let groups = if inside_carapace(here + link.center) {
            NESTED_COLLISION
        } else {
            crab_collision(env)
        };
        let id = link
            .actuated
            .expect("locked links are skipped before spawn");
        let joint = rig_revolute(id, link.axis_local, link.anchor1);
        let mut ec = commands.spawn((
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            collider,
            groups,
            ColliderMassProperties::Density(link.density),
            MultibodyJoint::new(parent_ent, joint),
            place(here),
            CrabRestPose(place(here)),
            Velocity::default(),
            ExternalForce::default(),
        ));
        ec.insert(CrabJoint {
            id,
            axis_local: link.axis_local,
        });
        if matches!(id, CrabJointId::ClawPincer(_)) {
            ec.insert(CrabClawTip);
        }
        if matches!(id, CrabJointId::LegCarpus(..)) {
            ec.insert(Friction::coefficient(1.5));
        }
        ents.push(ec.id());
    }

    carapace
}

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
