//! Ground scatter (bddap/rl#304, round 2): instanced grass tufts and pebbles that
//! hug the collider heightfield near the camera — the issue's "instanced ground
//! scatter with distance cull" direction. Strictly decoration: nothing here has a
//! collider, every base sits ON [`TerrainGrid::height`] (the physics surface), and
//! pieces stay tuft-sized (≤ ~0.5 m) so passing through one reads as brushing
//! grass, never as clipping geometry that lied about the ground.
//!
//! Layout is a camera-following chunk grid: the world tile is ~31 km, so scatter
//! exists only inside [`SPAWN_RADIUS_M`] of the active camera, spawned/despawned
//! per chunk as the camera moves, deterministic per chunk (same chunk → same tufts,
//! every session, every peer). Instances share 2 meshes + 3 material handles, so
//! the renderer draws the whole layer as a handful of instanced batches; each
//! instance carries a [`VisibilityRange`] dither-fade plus [`NotShadowCaster`], and
//! a camera above [`ALTITUDE_CUTOFF_M`] (plane mode) spawns nothing at all.

use std::collections::{HashMap, HashSet};
use std::f32::consts::TAU;

use bevy::camera::visibility::VisibilityRange;
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::sky::{hash3, rand01};
use crate::terrain::{Terrain, TerrainGrid, biome};

/// Chunk edge in meters. Small enough that a boundary crossing spawns a thin,
/// hitch-free sliver; big enough that the bookkeeping set stays tiny.
const CHUNK_M: f32 = 24.0;

/// Chunks whose center is inside this radius of the camera exist. Fade ends at
/// [`FADE_M`].1 + half a chunk diagonal before this, so spawn-in is never visible.
const SPAWN_RADIUS_M: f32 = 96.0;

/// Dither-fade band (meters from camera): fully there below .0, gone past .1.
/// Deliberately short — past ~70 m a tuft is sub-pixel and reads as speckle noise
/// (shimmer in motion), so the fade removes it before it degenerates.
const FADE_M: (f32, f32) = (45.0, 75.0);

/// Camera height above the ground past which no scatter exists — at plane altitude
/// every instance would be culled by [`FADE_M`] anyway, so skip the churn entirely.
/// A hysteresis pair (enter below .0, clear above .1): one threshold would respawn
/// and despawn the whole ~50-chunk disc every frame for a camera hovering at it.
const ALTITUDE_CUTOFF_M: (f32, f32) = (100.0, 140.0);

/// Chunk spawns per frame, so a teleport (or the first camera frame) amortizes its
/// full-disc fill over a few frames instead of hitching one — the fade band hides
/// the staggered arrival. A walking camera crosses one boundary at a time (< this).
const MAX_CHUNK_SPAWNS_PER_FRAME: usize = 12;

// The geometry that keeps every transition invisible, checked at compile time
// because these are exactly the knobs the taste loop tweaks:
// - fade-out completes before the spawn edge (farthest chunk corner) can pop;
// - everything is faded out before the altitude gate can clear the layer.
const _: () = {
    let half_diagonal = CHUNK_M * std::f32::consts::SQRT_2 / 2.0;
    assert!(SPAWN_RADIUS_M - half_diagonal > FADE_M.1);
    assert!(ALTITUDE_CUTOFF_M.0 >= FADE_M.1);
    assert!(ALTITUDE_CUTOFF_M.0 < ALTITUDE_CUTOFF_M.1);
};

/// Placement candidates hashed per chunk; biome weights then keep or drop each.
const CANDIDATES_PER_CHUNK: u32 = 72;

/// Global density trims on the biome weights (candidates kept at weight 1.0).
const TUFT_DENSITY: f32 = 0.85;
const PEBBLE_DENSITY: f32 = 0.5;

/// Registered by `ArenaVisualsPlugin` — the one arena path, so rl-demo, GCR, and
/// every MP peer grow the same scatter (headless hosts never compile this module).
pub struct ScatterPlugin;

impl Plugin for ScatterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Chunks>()
            .add_systems(Startup, setup_scatter_assets)
            .add_systems(Update, update_scatter);
    }
}

/// The shared handles every instance clones — 2 meshes × 3 materials keeps the
/// whole layer a handful of instanced draw batches.
#[derive(Resource)]
struct ScatterAssets {
    tuft: Handle<Mesh>,
    pebble: Handle<Mesh>,
    green: Handle<StandardMaterial>,
    dry: Handle<StandardMaterial>,
    stone: Handle<StandardMaterial>,
}

/// Live chunk parents by chunk coord (each parent's children are its instances).
#[derive(Resource, Default)]
struct Chunks(HashMap<IVec2, Entity>);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    GreenTuft,
    DryTuft,
    Pebble,
}

fn setup_scatter_assets(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !visuals.0 {
        return;
    }
    let tuft_material = |base: Color| StandardMaterial {
        base_color: base,
        perceptual_roughness: 1.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    };
    commands.insert_resource(ScatterAssets {
        tuft: meshes.add(tuft_mesh()),
        pebble: meshes.add(pebble_mesh()),
        green: materials.add(tuft_material(Color::srgb(0.46, 0.62, 0.30))),
        dry: materials.add(tuft_material(Color::srgb(0.76, 0.66, 0.40))),
        stone: materials.add(StandardMaterial {
            base_color: Color::srgb(0.44, 0.42, 0.39),
            perceptual_roughness: 1.0,
            ..default()
        }),
    });
}

/// One tuft: a ring of leaning single-triangle blades, all normals up (the classic
/// grass trick — blades shade like the ground they stand on instead of flashing
/// dark backfaces), vertex colors darkening toward the base. Instances vary by
/// yaw + uniform scale only, which is enough to hide that they share a mesh.
fn tuft_mesh() -> Mesh {
    const BLADES: usize = 7;
    let mut positions = Vec::with_capacity(BLADES * 3);
    let mut colors = Vec::with_capacity(BLADES * 3);
    for i in 0..BLADES {
        let r = |k: i32| rand01(hash3(i as i32, 304, k));
        let yaw = (i as f32 / BLADES as f32) * TAU + 0.6 * (r(0) - 0.5);
        let dir = Vec2::from_angle(yaw);
        let height = 0.20 + 0.30 * r(1);
        let lean = (0.12 + 0.38 * r(2)) * height;
        let side = Vec2::new(-dir.y, dir.x) * 0.032;
        let base = dir * 0.02;
        let tip = base + dir * lean;
        positions.push([base.x + side.x, 0.0, base.y + side.y]);
        positions.push([base.x - side.x, 0.0, base.y - side.y]);
        positions.push([tip.x, height, tip.y]);
        colors.extend([
            [0.8, 0.8, 0.75, 1.0],
            [0.8, 0.8, 0.75, 1.0],
            [1.0, 1.0, 0.9, 1.0],
        ]);
    }
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    let indices = (0..positions.len() as u32).collect::<Vec<_>>();
    Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

/// One pebble: a coarse icosphere; instances squash it per-axis so no two read
/// identical. Half its height sits below grade — rocks lie IN the dirt.
fn pebble_mesh() -> Mesh {
    Sphere::new(0.055)
        .mesh()
        .ico(0)
        .expect("0 subdivisions is always valid")
}

/// The deterministic contents of one chunk: world-space transforms sitting exactly
/// on the terrain surface, kinds weighted by the biome at each point.
fn chunk_instances(terrain: &TerrainGrid, chunk: IVec2) -> Vec<(Kind, Transform)> {
    let mut out = Vec::new();
    for i in 0..CANDIDATES_PER_CHUNK {
        let r = |k: i32| rand01(hash3(chunk.x, chunk.y, (i as i32) * 16 + k));
        let x = (chunk.x as f32 + r(0)) * CHUNK_M;
        let z = (chunk.y as f32 + r(1)) * CHUNK_M;
        let h = terrain.height(x, z);
        // Finite-difference up-normal at meter scale — the scale a tuft cares about.
        let dhdx = (terrain.height(x + 1.0, z) - terrain.height(x - 1.0, z)) / 2.0;
        let dhdz = (terrain.height(x, z + 1.0) - terrain.height(x, z - 1.0)) / 2.0;
        let normal_y = 1.0 / (1.0 + dhdx * dhdx + dhdz * dhdz).sqrt();

        let pick = r(2);
        let (kind, scale) = if pick < biome::tuft_weight(h, normal_y) * TUFT_DENSITY {
            let kind = if r(3) < biome::grass_dryness(h) {
                Kind::DryTuft
            } else {
                Kind::GreenTuft
            };
            (kind, Vec3::splat(0.9 + 0.9 * r(4)))
        } else if pick > 1.0 - biome::pebble_weight(h, normal_y) * PEBBLE_DENSITY {
            let s = 0.6 + 1.2 * r(4);
            (
                Kind::Pebble,
                Vec3::new(s, s * (0.45 + 0.3 * r(5)), 0.6 + 1.2 * r(6)),
            )
        } else {
            continue;
        };
        out.push((
            kind,
            Transform {
                // A hair below grade so a blade base never floats off a slope.
                translation: Vec3::new(x, h - 0.02, z),
                rotation: Quat::from_rotation_y(r(7) * TAU),
                scale,
            },
        ));
    }
    out
}

/// The chunk coords that should exist for a camera at `cam_xz`, `above_ground_m`
/// over the surface. Empty at plane altitude; `cutoff_m` is the caller's side of
/// the [`ALTITUDE_CUTOFF_M`] hysteresis pair.
fn wanted_chunks(cam_xz: Vec2, above_ground_m: f32, cutoff_m: f32) -> Vec<IVec2> {
    if above_ground_m > cutoff_m {
        return Vec::new();
    }
    let center = (cam_xz / CHUNK_M).floor().as_ivec2();
    let reach = (SPAWN_RADIUS_M / CHUNK_M).ceil() as i32;
    let mut out = Vec::new();
    for dz in -reach..=reach {
        for dx in -reach..=reach {
            let c = center + IVec2::new(dx, dz);
            let center_xz = (c.as_vec2() + 0.5) * CHUNK_M;
            if center_xz.distance(cam_xz) <= SPAWN_RADIUS_M {
                out.push(c);
            }
        }
    }
    out
}

/// Keep the chunk set matched to the active camera: despawn what fell out of
/// range, spawn what came into it. Steady-state (camera inside one chunk) this
/// touches nothing.
fn update_scatter(
    mut commands: Commands,
    terrain: Res<Terrain>,
    assets: Option<Res<ScatterAssets>>,
    mut chunks: ResMut<Chunks>,
    cams: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
) {
    // Asset presence IS the enabled flag: `setup_scatter_assets` inserts the
    // resource only on `Visuals(true)` surfaces.
    let Some(assets) = assets else { return };
    let Some(cam) = cams
        .iter()
        .find(|(c, _)| c.is_active)
        .map(|(_, t)| t.translation())
    else {
        return;
    };
    let cam_xz = Vec2::new(cam.x, cam.z);
    let above = cam.y - terrain.height(cam.x, cam.z);
    let cutoff = if chunks.0.is_empty() {
        ALTITUDE_CUTOFF_M.0
    } else {
        ALTITUDE_CUTOFF_M.1
    };
    let wanted: HashSet<IVec2> = wanted_chunks(cam_xz, above, cutoff).into_iter().collect();

    chunks.0.retain(|c, e| {
        wanted.contains(c) || {
            commands.entity(*e).despawn();
            false
        }
    });
    let mut spawned = 0;
    for c in wanted {
        if chunks.0.contains_key(&c) {
            continue;
        }
        if spawned == MAX_CHUNK_SPAWNS_PER_FRAME {
            break;
        }
        spawned += 1;
        let parent = commands
            .spawn((Transform::IDENTITY, Visibility::default()))
            .with_children(|p| {
                for (kind, tf) in chunk_instances(&terrain, c) {
                    let (mesh, mat) = match kind {
                        Kind::GreenTuft => (&assets.tuft, &assets.green),
                        Kind::DryTuft => (&assets.tuft, &assets.dry),
                        Kind::Pebble => (&assets.pebble, &assets.stone),
                    };
                    p.spawn((
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(mat.clone()),
                        tf,
                        NotShadowCaster,
                        VisibilityRange {
                            start_margin: 0.0..0.0,
                            end_margin: FADE_M.0..FADE_M.1,
                            use_aabb: false,
                        },
                    ));
                }
            })
            .id();
        chunks.0.insert(c, parent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same chunk → byte-identical contents, every call: the layer must look the
    /// same across sessions and MP peers with no state to sync.
    #[test]
    fn chunk_contents_are_deterministic() {
        let g = TerrainGrid::gcr();
        let a = chunk_instances(&g, IVec2::new(3, -7));
        let b = chunk_instances(&g, IVec2::new(3, -7));
        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len());
        for ((ka, ta), (kb, tb)) in a.iter().zip(&b) {
            assert_eq!(ka, kb);
            assert_eq!(ta.translation, tb.translation);
            assert_eq!(ta.rotation, tb.rotation);
            assert_eq!(ta.scale, tb.scale);
        }
    }

    /// Every instance stands ON the collider surface (render matches physics): the
    /// sampled height at its xz, minus the deliberate 2 cm sink.
    #[test]
    fn instances_sit_on_the_physics_surface() {
        let g = TerrainGrid::gcr();
        let mut checked = 0;
        for chunk in [IVec2::ZERO, IVec2::new(-5, 9), IVec2::new(40, -33)] {
            for (_, tf) in chunk_instances(&g, chunk) {
                let surface = g.height(tf.translation.x, tf.translation.z);
                assert!((tf.translation.y - (surface - 0.02)).abs() < 1e-4);
                checked += 1;
            }
        }
        assert!(checked > 20, "only {checked} instances across 3 chunks");
    }

    /// The origin plateau (dry-grass band) grows mostly straw tufts; deep valleys
    /// grow green ones — the scatter agrees with the tint it stands on.
    #[test]
    fn plateau_is_dry_valleys_are_green() {
        let g = TerrainGrid::gcr();
        let kinds: Vec<Kind> = chunk_instances(&g, IVec2::ZERO)
            .iter()
            .map(|(k, _)| *k)
            .collect();
        let dry = kinds.iter().filter(|k| **k == Kind::DryTuft).count();
        let green = kinds.iter().filter(|k| **k == Kind::GreenTuft).count();
        assert!(dry > green, "origin chunk: {dry} dry vs {green} green");
    }

    /// On-foot cameras get a disc of chunks around them; plane altitude gets none.
    #[test]
    fn chunk_disc_follows_the_camera_and_planes_get_none() {
        let near = wanted_chunks(Vec2::new(100.0, -50.0), 2.0, ALTITUDE_CUTOFF_M.0);
        assert!(near.contains(&IVec2::new(4, -3)), "camera's own chunk");
        // All within the radius (+ half a diagonal) of the camera.
        for c in &near {
            let center = (c.as_vec2() + 0.5) * CHUNK_M;
            assert!(center.distance(Vec2::new(100.0, -50.0)) <= SPAWN_RADIUS_M);
        }
        // Enough chunks to cover the fade disc, and none at plane altitude —
        // whichever side of the hysteresis pair a plane arrives on.
        assert!(near.len() > 30, "{} chunks", near.len());
        assert!(wanted_chunks(Vec2::ZERO, 600.0, ALTITUDE_CUTOFF_M.0).is_empty());
        assert!(wanted_chunks(Vec2::ZERO, 600.0, ALTITUDE_CUTOFF_M.1).is_empty());
    }
}
