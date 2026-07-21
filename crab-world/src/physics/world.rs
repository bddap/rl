use std::sync::Arc;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

#[cfg(feature = "render")]
use crate::ground::{GroundDetail, GroundMaterial, GroundMaterialPlugin};

use crate::bot::body::ARENA_COLLISION;
use crate::terrain::{Terrain, TerrainGrid};

/// The plant sidecar's arena spelling — the one string a loadable checkpoint may
/// record. Kept as the digest/sidecar preimage across the rl#293 flip so identical
/// worlds never refuse-to-arm between peers on either side of it.
pub(crate) const ARENA_TERRAIN_KEY: &str = "terrain";

/// Parse a plant sidecar's `arena` value: `Ok(true)` = terrain, `Ok(false)` =
/// `walled_box` (a real pre-flip record — it PARSES so adoption can refuse it with
/// provenance intact, or the rl-demo `--terrain` waiver can view it; a parse-time
/// refusal would rank honest flat records below recordless ones). Unknown spellings
/// refuse: an unreadable plant must never load.
pub(crate) fn parse_arena_key(raw: &str) -> Result<bool, String> {
    match raw.trim() {
        k if k == ARENA_TERRAIN_KEY => Ok(true),
        "walled_box" => Ok(false),
        other => Err(format!("unknown arena {other:?} (terrain | walled_box)")),
    }
}

/// Plugin that lays the arena PHYSICS — the ground collider — and publishes the
/// [`Terrain`] resource spawn-on-surface consumers sample. Since the rl#293 flip
/// there is ONE production ground: the committed GCR bake ([`TerrainGrid::gcr`]) —
/// the trainer, the keep-best/honest evals, GCR, and rl-demo all build exactly it,
/// and it is the only grid production code can construct ([`TerrainGrid::flat`] is
/// test-gated). Tests may pass a fixture grid: a degenerate parameter of the one
/// seam, not a second ground path. No meshes or lights: the visible dressing is a
/// SEPARATE [`ArenaVisualsPlugin`] (split rl#160), so a headless host adds the
/// collider alone. Every rendered surface — rl-demo and, since rl#281 stage 6,
/// GCR — adds both: there is ONE way an arena looks.
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
/// committed gcr bake plus the visible dressing, visuals on. The world exists
/// whether or not a crab is armed (rl#296), so this is deliberately separate from
/// any crab stack: GCR's windowed and screenshot scaffolds and rl-demo all add
/// exactly this plugin, one formula that cannot drift per-surface. Headless hosts
/// keep composing [`PhysicsWorldPlugin`] alone.
#[cfg(feature = "render")]
pub struct ArenaWorldPlugin;

#[cfg(feature = "render")]
impl Plugin for ArenaWorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(crate::Visuals(true))
            .add_plugins(PhysicsWorldPlugin {
                grid: TerrainGrid::gcr(),
            })
            .add_plugins(ArenaVisualsPlugin);
    }
}

#[cfg(feature = "render")]
pub struct ArenaVisualsPlugin;

#[cfg(feature = "render")]
impl Plugin for ArenaVisualsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(GroundMaterialPlugin);
        app.add_systems(Startup, setup_arena_visuals);
        // Update, not Startup: cameras spawn after Startup in some modes (same
        // pattern as NightSkyPlugin's skybox attach).
        app.add_systems(Update, attach_vista_fog);
    }
}

/// Lay the ground collider — the one terrain heightfield. No meshes or lights (those
/// are render-only — see [`setup_arena_visuals`]), so this is the one arena setup the
/// headless trainer runs, touching no graphics types.
fn setup_ground(mut commands: Commands, terrain: Res<Terrain>) {
    commands.spawn((
        RigidBody::Fixed,
        terrain.collider(),
        ARENA_COLLISION,
        Transform::IDENTITY,
    ));
}

#[cfg(feature = "render")]
fn setup_arena_visuals(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    terrain: Res<Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GroundMaterial>>,
) {
    if !visuals.0 {
        return;
    }

    // A vista world (rl#281 stage 3) gets a low, cool moon-sun for long relief
    // shadows, plus cascades stretched from the ~150 m default to mountain scale
    // (30 m grid pitch makes coarse far cascades invisible).
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 9500.0,
            color: Color::srgb(0.85, 0.90, 1.0),
            ..default()
        },
        bevy::light::CascadeShadowConfigBuilder {
            maximum_distance: 9000.0,
            first_cascade_far_bound: 20.0,
            ..default()
        }
        .build(),
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.7, 0.0)),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.75, 0.82, 1.0),
        brightness: 400.0,
        ..default()
    });

    // The visible ground: the SAME grid the collider was built from (rl#281 — render
    // matches physics by construction). Macro tint rides the mesh's vertex colors
    // (biome bands); everything finer is the procedural world-space detail layer in
    // [`GroundDetail`]'s fragment shader (rl#304 — it replaced the rl#197 checker,
    // including its optic-flow duty) — inside the one ground path, never a second
    // surface.
    commands.spawn((
        Mesh3d(meshes.add(terrain.mesh())),
        MeshMaterial3d(materials.add(GroundMaterial {
            base: StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.95,
                ..default()
            },
            extension: GroundDetail::default(),
        })),
        Transform::IDENTITY,
    ));
}

/// Give every camera aerial-perspective fog toward the night sky's horizon color.
/// Runs in `Update` because cameras can spawn any time (the demo's orbit cam, the
/// offscreen shot cam); same attach pattern as the skybox.
#[cfg(feature = "render")]
fn attach_vista_fog(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    cams: Query<Entity, (With<Camera3d>, Without<DistanceFog>)>,
) {
    if !visuals.0 {
        return;
    }
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
