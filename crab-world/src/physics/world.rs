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
    /// no walls. Live in rl-demo (`--terrain`) for the stage-3 taste loop and, via
    /// [`RL_ARENA`], as the training/eval plant (stage 4); GCR itself flips once its
    /// sim and render are terrain-aware.
    Terrain,
}

/// Which arena the TRAINING stack builds — the world half of the plant, beside
/// `RL_JOINT_FRICTION_CAP` (bddap/rl#268): it must reach every sim inside one process
/// (the rollout workers, the in-process keep-best chase-eval) and the external
/// honest-eval binary, so it is the same read-once env knob pattern, recorded in the
/// checkpoint's plant sidecar and adopted by `run_eval` — a terrain-trained policy
/// measured on the flat box (or vice versa) is a mismeasure. Unset keeps the legacy
/// walled box bit-identical. Values: `walled_box`, `terrain`.
pub const ARENA_ENV: &str = "RL_ARENA";

/// The [`RL_ARENA`](ARENA_ENV) choices — the arenas a policy can TRAIN in. A subset of
/// [`Arena`] on purpose: `OpenField` is GCR's inference field, not a training plant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainArena {
    WalledBox,
    Terrain,
}

impl TrainArena {
    pub fn arena(self) -> Arena {
        match self {
            TrainArena::WalledBox => Arena::WalledBox,
            TrainArena::Terrain => Arena::Terrain,
        }
    }

    /// The knob/sidecar spelling — one source for parse + record.
    pub(crate) fn key(self) -> &'static str {
        match self {
            TrainArena::WalledBox => "walled_box",
            TrainArena::Terrain => "terrain",
        }
    }

    pub(crate) fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "walled_box" => Ok(TrainArena::WalledBox),
            "terrain" => Ok(TrainArena::Terrain),
            other => Err(format!("unknown arena {other:?} (walled_box | terrain)")),
        }
    }
}

static ARENA_OVERRIDE: std::sync::OnceLock<Option<TrainArena>> = std::sync::OnceLock::new();

/// The resolved per-run arena override, read once per process. A SET-but-invalid value
/// aborts instead of defaulting — silently training days in the wrong world is the
/// failure mode the knob exists to prevent (same policy as the plant knobs).
pub fn train_arena_override() -> Option<TrainArena> {
    *ARENA_OVERRIDE.get_or_init(|| match std::env::var(ARENA_ENV) {
        Err(std::env::VarError::NotPresent) => None,
        Ok(raw) => Some(
            TrainArena::parse(&raw)
                .unwrap_or_else(|e| panic!("{ARENA_ENV}={raw}: {e} — refusing an ambiguous plant")),
        ),
        Err(e @ std::env::VarError::NotUnicode(_)) => {
            panic!("{ARENA_ENV}: {e} — refusing an ambiguous plant")
        }
    })
}

/// The arena the training stack (rollout workers, keep-best gate, honest eval) builds.
pub fn train_arena() -> TrainArena {
    train_arena_override().unwrap_or(TrainArena::WalledBox)
}

/// Sidecar adoption (the arena leg of `adopt_recorded_plant`): install the recorded
/// arena override — `None` = the checkpoint trained in the default arena, which pins
/// a stray env override out — or verify an already-resolved one agrees.
pub(crate) fn adopt_train_arena(recorded: Option<TrainArena>) -> Result<(), String> {
    if ARENA_OVERRIDE.set(recorded).is_ok() {
        return Ok(());
    }
    match train_arena_override() {
        resolved if resolved == recorded => Ok(()),
        resolved => Err(format!(
            "checkpoint trained in arena {}, but this process already resolved {} — \
             refusing to mismeasure",
            recorded.map_or("<default>".into(), |a| a.key().to_string()),
            resolved.map_or("<default>".into(), |a| a.key().to_string()),
        )),
    }
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

/// The visible arena ground spawned by [`setup_arena_visuals`]. The PHYSICS ground
/// (collider) always sits at the physics-frame origin; a host whose render frame is a
/// translation of the physics frame (net's arena↔world anchor, rl#224) uses this
/// handle to keep the drawn surface at that translation. Hosts whose two frames
/// coincide (rl-demo) leave it at IDENTITY.
#[cfg(feature = "render")]
#[derive(Component)]
pub struct ArenaSurface;

#[cfg(feature = "render")]
pub struct ArenaVisualsPlugin;

#[cfg(feature = "render")]
impl Plugin for ArenaVisualsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_arena_visuals);
        // Update, not Startup: cameras spawn after Startup in some modes (same
        // pattern as NightSkyPlugin's skybox attach).
        app.add_systems(Update, attach_vista_fog);
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

    // Flat arenas keep the pre-terrain training-box light exactly; a vista world
    // (rl#281 stage 3) gets a lower, cooler moon-sun for long relief shadows, plus
    // cascades stretched from the ~150 m default to mountain scale (30 m grid pitch
    // makes coarse far cascades invisible).
    if !terrain.is_flat() {
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
    } else {
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
    }

    // The visible ground: the SAME grid the collider was built from (rl#281 — render
    // matches physics by construction). Tint rides the mesh's vertex colors (biome
    // bands on terrain, the old flat green on flat arenas), so the material stays
    // white and ONE material serves every arena.
    commands.spawn((
        ArenaSurface,
        Mesh3d(meshes.add(terrain.mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::IDENTITY,
    ));
}

/// Give every camera in a vista (real-relief, rendered) world aerial-perspective fog toward the night
/// sky's horizon color. Runs in `Update` because cameras can spawn any time (the
/// demo's orbit cam, the offscreen shot cam); same attach pattern as the skybox.
#[cfg(feature = "render")]
fn attach_vista_fog(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    terrain: Res<crate::terrain::Terrain>,
    cams: Query<Entity, (With<Camera3d>, Without<DistanceFog>)>,
) {
    if !visuals.0 || terrain.is_flat() {
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
