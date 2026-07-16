use std::sync::Arc;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::bot::body::ARENA_COLLISION;
use crate::terrain::{Terrain, TerrainGrid};

/// Which arena PHYSICS a world gets. The training/demo box, the GCR inference field, and
/// the baked terrain tile legitimately differ — training samples targets inside a ±10 m
/// walled box, while the GCR crab hunts a real player who can be anywhere on the map
/// (players spawn ≥12 m out, past the box walls — rl#209) — so the choice is an explicit
/// parameter of the ONE [`PhysicsWorldPlugin`], not a second parallel arena
/// implementation. Every variant is a height grid through the ONE terrain path
/// ([`crate::terrain`], rl#281): the ground collider is always [`TerrainGrid::collider`],
/// only the grid (and the walls) differ.
pub enum Arena {
    /// The ±10 m walled box the policy trains in ([`ARENA_HALF_SIZE`]): a flat grid
    /// plus four walls. The trainer, its tests/eval, and the standalone rl-demo.
    WalledBox,
    /// An unwalled flat grid at the walled box's same y=0 surface, for GCR inference:
    /// the crab's per-round travel is unbounded so it can chase a player clear across
    /// the map (rl#209). Spans ±[`OPEN_FIELD_HALF_SIZE`] — far past the cameras, the
    /// round clock, and every speed cap; kept flat until GCR's planar server sim learns
    /// terrain (rl#281 stages 3-5).
    OpenField,
    /// The committed GCR terrain bake ([`TerrainGrid::gcr`], rl#281) — real mountains,
    /// no walls. Live in rl-demo (`--terrain`) for the stage-3 taste loop; GCR itself
    /// flips once its sim and render are terrain-aware.
    Terrain,
}

impl Arena {
    fn grid(&self) -> Arc<TerrainGrid> {
        match self {
            Arena::WalledBox => Arc::new(TerrainGrid::flat(ARENA_HALF_SIZE)),
            Arena::OpenField => Arc::new(TerrainGrid::flat(OPEN_FIELD_HALF_SIZE)),
            Arena::Terrain => TerrainGrid::gcr(),
        }
    }
}

/// Plugin that lays the arena PHYSICS — the ground (+ wall) colliders per [`Arena`] —
/// and publishes the [`Terrain`] resource spawn-on-surface consumers sample. No meshes
/// or lights: the visible dressing is a SEPARATE [`ArenaVisualsPlugin`], so a host that
/// draws its own scene (the GCR client's gray-box world) adds the colliders alone and
/// never spawns a second coplanar ground quad to z-fight its own (rl#160). The
/// standalone rl-demo arena, which draws no other scene, adds both.
pub struct PhysicsWorldPlugin {
    pub arena: Arena,
}

impl Plugin for PhysicsWorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Terrain::new(self.arena.grid()));
        app.add_systems(Startup, setup_ground);
        if matches!(self.arena, Arena::WalledBox) {
            app.add_systems(Startup, setup_walls);
        }
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
/// Half-span of the [`Arena::OpenField`] flat grid. The old halfspace was truly
/// unbounded; a grid has an edge, so this is sized to make the edge unreachable in
/// practice (≥ the GCR terrain tile's own ~±15 km span). A real world-bounds policy is a
/// later stage of rl#281.
const OPEN_FIELD_HALF_SIZE: f32 = 16_384.0;
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

/// Lay the ground collider — the one terrain heightfield, whatever the arena. No meshes
/// or lights (those are render-only — see [`setup_arena_visuals`]), so this is the one
/// arena setup the headless trainer runs, touching no graphics types.
fn setup_ground(mut commands: Commands, terrain: Res<Terrain>) {
    commands.spawn((
        RigidBody::Fixed,
        terrain.collider(),
        ARENA_COLLISION,
        Transform::IDENTITY,
    ));
}

/// The [`Arena::WalledBox`] walls (same contact identity as the ground:
/// [`ARENA_COLLISION`]).
fn setup_walls(mut commands: Commands) {
    for (pos, half_extents) in wall_boxes() {
        commands.spawn((
            RigidBody::Fixed,
            Collider::cuboid(half_extents.x, half_extents.y, half_extents.z),
            ARENA_COLLISION,
            Transform::from_translation(pos),
        ));
    }
}

#[cfg(feature = "render")]
fn setup_arena_visuals(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    terrain: Res<Terrain>,
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

    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        ..default()
    });

    // The visible ground: the SAME grid the collider was built from (rl#281 — render
    // matches physics by construction). Flat arenas get their old flat green floor;
    // the terrain arena gets the actual mountains. Looks beyond correct geometry
    // (material, LOD, biome tint) are rl#281 stage 3.
    commands.spawn((
        Mesh3d(meshes.add(terrain.mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.55, 0.35),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::IDENTITY,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The open field must span at least the GCR terrain tile — the prose rationale for
    /// [`OPEN_FIELD_HALF_SIZE`] ("edge unreachable in practice") silently rots if a
    /// bigger tile is baked.
    #[test]
    fn open_field_spans_the_gcr_tile() {
        let g = TerrainGrid::gcr();
        assert!(OPEN_FIELD_HALF_SIZE >= g.extent_x() / 2.0);
        assert!(OPEN_FIELD_HALF_SIZE >= g.extent_z() / 2.0);
    }
}
