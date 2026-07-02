use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::bot::body::ARENA_COLLISION;

/// Plugin that lays the arena PHYSICS — the ground + wall colliders. No meshes or lights:
/// the visible dressing is a SEPARATE [`ArenaVisualsPlugin`], so a host that draws its own
/// scene (the GCR client's gray-box world) adds the colliders alone and never spawns a second
/// coplanar ground quad to z-fight its own (rl#160). The standalone rl-demo arena, which draws
/// no other scene, adds both.
pub struct PhysicsWorldPlugin;

impl Plugin for PhysicsWorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_arena);
    }
}

/// Render-only arena dressing — the visible ground quad + lights — for a host that draws NO
/// other scene of its own (the standalone rl-demo arena). Kept OUT of [`PhysicsWorldPlugin`]
/// (which lays only the colliders) so the GCR client, which renders its own gray-box world,
/// gets the colliders WITHOUT a redundant second ground quad and lights coplanar with its own
/// (rl#160). The material/mesh-asset types this needs don't exist in the headless build, so the
/// whole plugin is render-only; it still honors `Visuals(false)` so a render-on caller that
/// wants no scene gets none.
#[cfg(feature = "render")]
pub struct ArenaVisualsPlugin;

#[cfg(feature = "render")]
impl Plugin for ArenaVisualsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_arena_visuals);
    }
}

/// Half-extent of the square arena (ground + walls) in metres. `pub(crate)` so the
/// reward's target-spawn clamp ([`crate::training::targets`]) can derive its inset
/// from the true wall position rather than hand-copying it — a wall move then can't
/// strand far targets inside a wall.
pub(crate) const ARENA_HALF_SIZE: f32 = 10.0;
const GROUND_THICKNESS: f32 = 0.1;
const WALL_HEIGHT: f32 = 2.0;
const WALL_THICKNESS: f32 = 0.5;

/// The four wall colliders' (center, half-extents). One source so the collider setup
/// and the render-only wall meshes describe the SAME walls — a wall move can't leave
/// the mesh behind the collider.
fn wall_boxes() -> [(Vec3, Vec3); 4] {
    [
        (
            Vec3::new(0.0, WALL_HEIGHT / 2.0, ARENA_HALF_SIZE + WALL_THICKNESS),
            Vec3::new(ARENA_HALF_SIZE, WALL_HEIGHT / 2.0, WALL_THICKNESS),
        ),
        (
            Vec3::new(0.0, WALL_HEIGHT / 2.0, -ARENA_HALF_SIZE - WALL_THICKNESS),
            Vec3::new(ARENA_HALF_SIZE, WALL_HEIGHT / 2.0, WALL_THICKNESS),
        ),
        (
            Vec3::new(ARENA_HALF_SIZE + WALL_THICKNESS, WALL_HEIGHT / 2.0, 0.0),
            Vec3::new(WALL_THICKNESS, WALL_HEIGHT / 2.0, ARENA_HALF_SIZE),
        ),
        (
            Vec3::new(-ARENA_HALF_SIZE - WALL_THICKNESS, WALL_HEIGHT / 2.0, 0.0),
            Vec3::new(WALL_THICKNESS, WALL_HEIGHT / 2.0, ARENA_HALF_SIZE),
        ),
    ]
}

/// Lay the arena PHYSICS: the ground collider + the four wall colliders. No meshes or
/// lights (those are render-only — see [`setup_arena_visuals`]), so this is the one
/// arena setup the headless trainer runs, touching no graphics types.
fn setup_arena(mut commands: Commands) {
    // Ground plane collider. The visible quad (render builds) is spawned separately and
    // sits exactly on this — see `setup_arena_visuals`.
    commands.spawn((
        RigidBody::Fixed,
        Collider::cuboid(ARENA_HALF_SIZE, GROUND_THICKNESS, ARENA_HALF_SIZE),
        ARENA_COLLISION,
        Transform::from_xyz(0.0, -GROUND_THICKNESS, 0.0),
    ));

    // Arena walls (just colliders; the demo doesn't render them).
    for (pos, half_extents) in wall_boxes() {
        commands.spawn((
            RigidBody::Fixed,
            Collider::cuboid(half_extents.x, half_extents.y, half_extents.z),
            ARENA_COLLISION,
            Transform::from_translation(pos),
        ));
    }
}

/// Render-only: the visible ground quad + lights, added by [`ArenaVisualsPlugin`] for a
/// standalone arena (rl-demo). Sits exactly on the collider [`setup_arena`] laid. Gated out of
/// the headless trainer entirely (its bevy build has no `StandardMaterial`/`Mesh3d`/
/// `DirectionalLight`). Still honors `Visuals(false)` so a render-on caller that wants no scene
/// gets none.
#[cfg(feature = "render")]
fn setup_arena_visuals(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !visuals.0 {
        return;
    }

    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.3, 0.0)),
    ));

    // Ambient light so shadows aren't pitch black.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        ..default()
    });

    // The visible ground quad, sitting exactly on the collider laid in `setup_arena`.
    commands.spawn((
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
}
