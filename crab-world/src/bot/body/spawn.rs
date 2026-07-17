use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::collision::{NESTED_COLLISION, crab_collision, no_adjacent_contacts};
use super::components::{
    CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint, CrabRestPose,
};
use super::joint_id::CrabJointId;
use crate::bot::rig;
use crate::terrain::TerrainGrid;

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
    terrain: &TerrainGrid,
    position: Vec3,
    env: usize,
    init_rotation: Quat,
) -> Entity {
    let recipe = &assets.recipe;
    let origin = position + recipe.hub_bind_world + Vec3::new(0.0, SPAWN_HEIGHT, 0.0);

    let world_pos = rig::link_world_origins(&recipe.links, origin);

    // Every part's bounding sphere in the unrotated bind pose (the carapace's wraps its
    // offset compound around `origin`, matching the collider bound below).
    let carapace_r = recipe.carapace_offset.length() + recipe.carapace_half.length();
    let spheres: Vec<(Vec3, f32)> = std::iter::once((origin, carapace_r))
        .chain(
            recipe
                .links
                .iter()
                .zip(&world_pos)
                .map(|(link, &p)| (p, link.center.length() + link.half_height + link.radius)),
        )
        .collect();
    let rotated = |p: Vec3| origin + init_rotation * (p - origin);
    let low_unrot = spheres
        .iter()
        .map(|(p, r)| p.y - r)
        .fold(f32::MAX, f32::min);
    let low_rot = spheres
        .iter()
        .map(|&(p, r)| rotated(p).y - r)
        .fold(f32::MAX, f32::min);
    // The bind pose clears a FLAT floor at `position.y` by construction. Two things can
    // still bury a part (rl#283): the init rotation swinging it below the unrotated low
    // point, and — on terrain — the ground itself rising above the origin sample under
    // an outlying part. Lift by both: restore the unrotated low point, then add the max
    // terrain rise under each part's bounding disc (exactly zero on flat grids, so
    // training placement is untouched). The 8-point rim + center sampling is approximate
    // — a gradient peaking between samples undershoots by ≲ (1−cos 22.5°)·r·slope, cm at
    // crab part radii — and the spheres are conservative bounds with SPAWN_HEIGHT on
    // top, so soft-contact depenetration absorbs the residual.
    const RIM: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let terrain_rise = spheres
        .iter()
        .flat_map(|&(p, r)| {
            let c = rotated(p);
            [
                (0.0, 0.0),
                (r, 0.0),
                (-r, 0.0),
                (0.0, r),
                (0.0, -r),
                (RIM * r, RIM * r),
                (RIM * r, -RIM * r),
                (-RIM * r, RIM * r),
                (-RIM * r, -RIM * r),
            ]
            .map(|(dx, dz)| terrain.height(c.x + dx, c.z + dz) - position.y)
        })
        .fold(0.0f32, f32::max);
    let lift = (low_unrot - low_rot).max(0.0) + terrain_rise;
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
        let cap = rig::link_capsule(link, Vec3::ZERO);
        let collider = Collider::capsule(cap.a, cap.b, cap.radius);
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

#[cfg(test)]
mod tests {
    use bevy::ecs::world::CommandQueue;

    use super::super::components::CrabBodyPart;
    use super::*;

    /// A 2×2 grid whose one cell ramps 5 m over 1 m toward +x — steep enough that a
    /// footprint-blind spawn at the origin sample buries the +x parts in the hill.
    /// Built through the real artifact codec deliberately (the parse path IS the seam
    /// spawns ride); note parse datum-shifts the declared heights to [-5, 0, -5, 0] —
    /// same ramp, shifted down.
    fn steep_ramp() -> TerrainGrid {
        let meta = br#"{"rows":2,"cols":2,"cell_size_m":1.0,"height_scale":1.0}"#;
        let mut bytes = b"RLTERR01".to_vec();
        bytes.extend((meta.len() as u32).to_le_bytes());
        bytes.extend(meta);
        for h in [0i16, 5, 0, 5] {
            bytes.extend(h.to_le_bytes());
        }
        TerrainGrid::parse(&bytes).expect("test artifact parses")
    }

    /// rl#283: the spawn lift must clear the TERRAIN under the whole footprint, not just
    /// a flat plane through the origin sample — on a steep slope the uphill parts would
    /// otherwise spawn inside the hill and take a depenetration kick.
    #[test]
    fn spawn_lift_clears_the_hill_under_the_footprint() {
        let grid = steep_ramp();
        let mut world = World::new();
        let assets = CrabAssets {
            recipe: crate::bot::rig::fallback_recipe(),
        };
        let position = grid.place(Vec2::ZERO, 0.0);
        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, &world);
        spawn_crab(&mut commands, &assets, &grid, position, 0, Quat::IDENTITY);
        queue.apply(&mut world);

        let mut parts = 0;
        let mut q = world.query_filtered::<&Transform, With<CrabBodyPart>>();
        for t in q.iter(&world) {
            let surface = grid.height(t.translation.x, t.translation.z);
            assert!(
                t.translation.y > surface,
                "part centre {:?} buried under the ramp (surface {surface})",
                t.translation
            );
            parts += 1;
        }
        assert!(parts > 10, "expected a whole crab, got {parts} parts");
    }
}
