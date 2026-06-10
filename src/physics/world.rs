use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::Visuals;
use crate::bot::body::ARENA_COLLISION;

/// Plugin that sets up the physics world: ground plane, lighting. Cameras are
/// spawned per app mode (fixed for training, orbit for demo, offscreen for
/// screenshots).
pub struct PhysicsWorldPlugin;

impl Plugin for PhysicsWorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_arena);
    }
}

/// Arena dimensions (half-extents).
const ARENA_HALF_SIZE: f32 = 10.0;
const GROUND_THICKNESS: f32 = 0.1;

fn setup_arena(
    mut commands: Commands,
    visuals: Res<Visuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if visuals.0 {
        // Directional light (sun)
        commands.spawn((
            DirectionalLight {
                shadows_enabled: true,
                illuminance: 10000.0,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.3, 0.0)),
        ));

        // Ambient light so shadows aren't pitch black
        commands.insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 300.0,
            ..default()
        });
    }

    // Ground plane — static rigid body with collider; add a mesh only when rendering.
    if visuals.0 {
        commands.spawn((
            RigidBody::Fixed,
            Collider::cuboid(ARENA_HALF_SIZE, GROUND_THICKNESS, ARENA_HALF_SIZE),
            ARENA_COLLISION,
            Mesh3d(meshes.add(Cuboid::new(
                ARENA_HALF_SIZE * 2.0,
                GROUND_THICKNESS * 2.0,
                ARENA_HALF_SIZE * 2.0,
            ))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.35, 0.55, 0.35),
                perceptual_roughness: 0.9,
                ..default()
            })),
            Transform::from_xyz(0.0, -GROUND_THICKNESS, 0.0),
        ));
    } else {
        commands.spawn((
            RigidBody::Fixed,
            Collider::cuboid(ARENA_HALF_SIZE, GROUND_THICKNESS, ARENA_HALF_SIZE),
            ARENA_COLLISION,
            Transform::from_xyz(0.0, -GROUND_THICKNESS, 0.0),
        ));
    }

    // Arena walls (just colliders, no visuals needed)
    let wall_height = 2.0;
    let wall_thickness = 0.5;
    let walls = [
        (
            Vec3::new(0.0, wall_height / 2.0, ARENA_HALF_SIZE + wall_thickness),
            Vec3::new(ARENA_HALF_SIZE, wall_height / 2.0, wall_thickness),
        ),
        (
            Vec3::new(0.0, wall_height / 2.0, -ARENA_HALF_SIZE - wall_thickness),
            Vec3::new(ARENA_HALF_SIZE, wall_height / 2.0, wall_thickness),
        ),
        (
            Vec3::new(ARENA_HALF_SIZE + wall_thickness, wall_height / 2.0, 0.0),
            Vec3::new(wall_thickness, wall_height / 2.0, ARENA_HALF_SIZE),
        ),
        (
            Vec3::new(-ARENA_HALF_SIZE - wall_thickness, wall_height / 2.0, 0.0),
            Vec3::new(wall_thickness, wall_height / 2.0, ARENA_HALF_SIZE),
        ),
    ];

    for (pos, half_extents) in walls {
        commands.spawn((
            RigidBody::Fixed,
            Collider::cuboid(half_extents.x, half_extents.y, half_extents.z),
            ARENA_COLLISION,
            Transform::from_translation(pos),
        ));
    }
}
