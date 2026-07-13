use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::bot::body::ARENA_COLLISION;

/// Which arena PHYSICS a world gets. The training/demo box and the GCR inference field
/// legitimately differ — training samples targets inside a ±10 m walled box, while the
/// GCR crab hunts a real player who can be anywhere on the map (players spawn ≥12 m out,
/// past the box walls — rl#209) — so the choice is an explicit parameter of the ONE
/// [`PhysicsWorldPlugin`], not a second parallel arena implementation.
pub enum Arena {
    /// The ±10 m walled box the policy trains in ([`ARENA_HALF_SIZE`]): a finite ground
    /// slab plus four walls. The trainer, its tests/eval, and the standalone rl-demo.
    WalledBox,
    /// An unbounded flat ground (a halfspace at the walled box's same y=0 surface) with NO
    /// walls, for GCR inference: the crab's per-round travel is unbounded so it can chase a
    /// player clear across the map (rl#209), and the player's flight vehicle — which clears
    /// the 2 m walls anyway — can never overfly a ground edge and fall forever.
    OpenField,
}

/// Plugin that lays the arena PHYSICS — the ground (+ wall) colliders per [`Arena`]. No
/// meshes or lights: the visible dressing is a SEPARATE [`ArenaVisualsPlugin`], so a host
/// that draws its own scene (the GCR client's gray-box world) adds the colliders alone and
/// never spawns a second coplanar ground quad to z-fight its own (rl#160). The standalone
/// rl-demo arena, which draws no other scene, adds both.
pub struct PhysicsWorldPlugin {
    pub arena: Arena,
}

impl Plugin for PhysicsWorldPlugin {
    fn build(&self, app: &mut App) {
        match self.arena {
            Arena::WalledBox => app.add_systems(Startup, setup_walled_box),
            Arena::OpenField => app.add_systems(Startup, setup_open_field),
        };
    }
}

#[cfg(feature = "render")]
pub struct ArenaVisualsPlugin;

#[cfg(feature = "render")]
impl Plugin for ArenaVisualsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_arena_visuals);
    }
}

pub(crate) const ARENA_HALF_SIZE: f32 = 10.0;
const GROUND_THICKNESS: f32 = 0.1;
const WALL_HEIGHT: f32 = 2.0;
const WALL_THICKNESS: f32 = 0.5;

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

/// Lay the [`Arena::WalledBox`] PHYSICS: the ground collider + the four wall colliders.
/// No meshes or lights (those are render-only — see [`setup_arena_visuals`]), so this is
/// the one arena setup the headless trainer runs, touching no graphics types.
fn setup_walled_box(mut commands: Commands) {
    commands.spawn((
        RigidBody::Fixed,
        Collider::cuboid(ARENA_HALF_SIZE, GROUND_THICKNESS, ARENA_HALF_SIZE),
        ARENA_COLLISION,
        Transform::from_xyz(0.0, -GROUND_THICKNESS, 0.0),
    ));

    for (pos, half_extents) in wall_boxes() {
        commands.spawn((
            RigidBody::Fixed,
            Collider::cuboid(half_extents.x, half_extents.y, half_extents.z),
            ARENA_COLLISION,
            Transform::from_translation(pos),
        ));
    }
}

/// Lay the [`Arena::OpenField`] PHYSICS: one unbounded halfspace ground whose surface is
/// the SAME y=0 plane the walled box's slab tops out at (default material, same
/// [`ARENA_COLLISION`] groups), so the crab plants and walks on contact dynamics identical
/// to what the policy trained under — only the walls and the ground's edge are gone.
fn setup_open_field(mut commands: Commands) {
    commands.spawn((
        RigidBody::Fixed,
        Collider::halfspace(Vec3::Y).expect("+Y is a unit normal"),
        ARENA_COLLISION,
        Transform::IDENTITY,
    ));
}

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
            shadow_maps_enabled: true,
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.3, 0.0)),
    ));

    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        ..default()
    });

    // The visible ground quad, sitting exactly on the collider laid in `setup_walled_box`.
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
