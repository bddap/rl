use std::sync::Arc;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

#[cfg(feature = "render")]
use crate::ground::{GroundDetail, GroundLook, GroundMaterial, GroundMaterialPlugin};

use crate::terrain::{Terrain, TerrainGrid};

/// Plugin that lays the arena PHYSICS — the ground collider — and publishes the
/// [`Terrain`] resource spawn-on-surface consumers sample. No meshes or lights, so a
/// headless host adds the collider alone; rendered surfaces add
/// [`ArenaWorldPlugin`], which is this plus the visible dressing.
pub struct PhysicsWorldPlugin {
    pub grid: Arc<TerrainGrid>,
}

impl Plugin for PhysicsWorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Terrain::new(self.grid.clone()));
        app.add_systems(Startup, setup_ground);
    }
}

/// The one arena every rendered surface installs — terrain physics on the
/// committed gcr bake plus the visible dressing — so the windowed client and the
/// screenshot scaffold cannot drift apart.
#[cfg(feature = "render")]
pub struct ArenaWorldPlugin {
    pub ground_look: GroundLook,
}

#[cfg(feature = "render")]
impl Plugin for ArenaWorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsWorldPlugin {
            grid: TerrainGrid::gcr(),
        });
        app.add_plugins(GroundMaterialPlugin {
            initial: self.ground_look,
        });
        app.add_plugins(crate::scatter::ScatterPlugin);
        app.add_systems(Startup, setup_arena_visuals);
        // Update, not Startup: cameras spawn after Startup in some modes (same
        // pattern as NightSkyPlugin's skybox attach).
        app.add_systems(Update, attach_vista_fog);
    }
}

/// Lay the ground collider — the one terrain heightfield.
fn setup_ground(mut commands: Commands, terrain: Res<Terrain>) {
    commands.spawn((RigidBody::Fixed, terrain.collider(), Transform::IDENTITY));
}

#[cfg(feature = "render")]
fn setup_arena_visuals(
    mut commands: Commands,
    terrain: Res<Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GroundMaterial>>,
    mut images: ResMut<Assets<Image>>,
    look: Res<GroundLook>,
) {
    // The directional light is the moon's (`crate::moon`, via NightSkyPlugin);
    // only the ambient starlight floor lives here.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.75, 0.82, 1.0),
        brightness: 400.0,
        ..default()
    });

    // The visible ground: the SAME grid the collider was built from (rl#281 — render
    // matches physics by construction). Macro tint rides the mesh's vertex colors
    // (biome bands); everything finer is the procedural world-space detail layer in
    // [`GroundDetail`]'s fragment shader (rl#304, which also carries the optic-flow
    // cue) — inside the one ground path, never a second surface.
    // The hydrology bake (rl#323): wetness + standing water derived from the SAME
    // grid as the mesh and collider, so the watershed look's moisture agrees with
    // the silhouette by construction.
    let moisture = images.add(crate::moisture::MoistureMap::bake(&terrain).image());
    commands.spawn((
        crate::ground::GroundMesh,
        Mesh3d(meshes.add(terrain.mesh())),
        MeshMaterial3d(materials.add(GroundMaterial {
            base: StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.95,
                ..default()
            },
            extension: GroundDetail::new(
                *look,
                moisture,
                Vec2::new(terrain.extent_x(), terrain.extent_z()),
            ),
        })),
        Transform::IDENTITY,
    ));
}

/// Give every camera aerial-perspective fog toward the night sky's horizon color.
/// Runs in `Update` because cameras can spawn any time; same attach pattern as the
/// skybox.
#[cfg(feature = "render")]
fn attach_vista_fog(
    mut commands: Commands,
    cams: Query<Entity, (With<Camera3d>, Without<DistanceFog>)>,
) {
    for cam in &cams {
        commands.entity(cam).insert(DistanceFog {
            color: crate::sky::horizon_fog_color(),
            falloff: FogFalloff::Linear {
                start: 600.0,
                end: 11000.0,
            },
            ..default()
        });
    }
}
