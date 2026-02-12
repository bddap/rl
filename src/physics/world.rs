use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

/// Plugin that sets up the physics world: ground plane, lighting, camera.
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera — overhead-ish view looking down at the arena
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 15.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

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

    // Ground plane — static rigid body with collider and visible mesh
    commands.spawn((
        RigidBody::Fixed,
        Collider::cuboid(ARENA_HALF_SIZE, GROUND_THICKNESS, ARENA_HALF_SIZE),
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

    // Arena walls (invisible, just colliders to keep things in bounds)
    let wall_height = 2.0;
    let wall_thickness = 0.5;
    let walls = [
        // (position, half-extents)
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
            Transform::from_translation(pos),
        ));
    }

    // Drop a test cube to prove physics is working
    commands.spawn((
        RigidBody::Dynamic,
        Collider::cuboid(0.5, 0.5, 0.5),
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.2, 0.2),
            ..default()
        })),
        Transform::from_xyz(0.0, 5.0, 0.0),
    ));
}
