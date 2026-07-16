//! The ONE terrain path (rl#281): a baked height grid drives BOTH the rapier
//! [`Collider`] and (render-gated) the visible mesh, so wherever this module's mesh is
//! what's drawn (rl-demo; GCR once its scene adopts it in stage 3) it cannot diverge
//! from what the crab stands on. The flat arenas are trivial constant grids through
//! this same seam — there is deliberately no flat-vs-terrain fork anywhere.
//!
//! Coordinates: the grid is centered on the world origin, +x = artifact column
//! (preview-PNG right), +z = artifact row (preview-PNG down), heights along +y.
//! [`TerrainGrid::height`] is triangle-exact against parry's default heightfield
//! subdivision — the sampler IS the collider surface, not a bilinear approximation
//! of it. Outside the tile the sampler clamps to the edge value, but the collider
//! ends at the tile edge; a world-bounds policy is a later stage of rl#281.

use std::sync::{Arc, OnceLock};

use bevy::prelude::*;
use bevy_rapier3d::prelude::Collider;

/// The stage-1 bake artifact for GCR (rl#281): seed 281, 1024², 30 m pitch.
static GCR_TERRAIN_BYTES: &[u8] = include_bytes!("../assets/terrain/gcr-seed281.terrain");

const MAGIC: &[u8; 8] = b"RLTERR01";

/// Cell-size cap for [`TerrainGrid::flat`], for f32 contact precision — parry solves
/// heightfield contacts against the containing cell's triangle, and one tile-spanning
/// cell on the ±16 km open field puts that triangle's vertices where f32 ulp is ~2 mm,
/// comparable to contact tolerances (and it measurably perturbed the rl#224 flail-walk).
/// 256 m keeps near-origin triangle vertices small at 129² points for the open field,
/// instead of megabytes of zeros per world on the probe-rebuild path.
const FLAT_CELL_MAX_M: f32 = 256.0;

/// The `.terrain` metadata fields the runtime consumes; the rest of the header
/// (seed, model rev, elevation range, …) is bake provenance and stays in the file.
#[derive(serde::Deserialize)]
struct Meta {
    rows: usize,
    cols: usize,
    cell_size_m: f32,
    height_scale: f32,
}

/// A heightmap over the world's XZ plane, centered on the origin. Heights are stored
/// datum-shifted (the tile's center sample sits at y=0, so spawns near the origin stay
/// near y=0 whatever the bake's absolute elevations) with the artifact's declarative
/// `height_scale` already applied.
pub struct TerrainGrid {
    rows: usize,
    cols: usize,
    /// World meters per cell edge (`cell_size_m` from the artifact).
    cell: f32,
    /// Row-major `[row * cols + col]`, like the artifact.
    heights: Vec<f32>,
}

impl TerrainGrid {
    /// Parse a `RLTERR01` artifact (see `scripts/terrain-bake/README.md`).
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, String> {
        let header = bytes.get(..12).ok_or("artifact shorter than its header")?;
        if &header[..8] != MAGIC {
            return Err(format!("bad magic {:?}", &header[..8]));
        }
        let json_len = u32::from_le_bytes(header[8..12].try_into().expect("4 bytes")) as usize;
        let meta_bytes = bytes
            .get(12..12 + json_len)
            .ok_or("truncated metadata blob")?;
        let meta: Meta =
            serde_json::from_slice(meta_bytes).map_err(|e| format!("bad metadata: {e}"))?;
        if meta.rows < 2 || meta.cols < 2 {
            return Err(format!("degenerate grid {}x{}", meta.rows, meta.cols));
        }
        if !(meta.cell_size_m > 0.0
            && meta.cell_size_m.is_finite()
            && meta.height_scale.is_finite())
        {
            return Err(format!(
                "bad scale knobs: cell_size_m={} height_scale={}",
                meta.cell_size_m, meta.height_scale
            ));
        }
        let grid = &bytes[12 + json_len..];
        let want = meta
            .rows
            .checked_mul(meta.cols)
            .and_then(|n| n.checked_mul(2))
            .ok_or("grid dimensions overflow")?;
        if grid.len() != want {
            return Err(format!(
                "grid is {} bytes, want {}x{}x2",
                grid.len(),
                meta.rows,
                meta.cols
            ));
        }
        let raw: Vec<i16> = grid
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let datum = raw[(meta.rows / 2) * meta.cols + meta.cols / 2];
        let heights = raw
            .iter()
            .map(|&h| (i32::from(h) - i32::from(datum)) as f32 * meta.height_scale)
            .collect();
        Ok(Self {
            rows: meta.rows,
            cols: meta.cols,
            cell: meta.cell_size_m,
            heights,
        })
    }

    /// A constant y=0 grid spanning ±`half_extent` — the flat arena, expressed through
    /// the one terrain path instead of a bespoke slab or halfspace.
    pub fn flat(half_extent: f32) -> Self {
        let cells = ((half_extent * 2.0) / FLAT_CELL_MAX_M).ceil().max(1.0) as usize;
        let n = cells + 1;
        Self {
            rows: n,
            cols: n,
            cell: half_extent * 2.0 / cells as f32,
            heights: vec![0.0; n * n],
        }
    }

    /// The committed GCR bake (parsed once; the artifact ships inside the binary so the
    /// headless trainer and every peer sample the identical world with no asset-path or
    /// download dependence).
    pub fn gcr() -> Arc<Self> {
        static GRID: OnceLock<Arc<TerrainGrid>> = OnceLock::new();
        GRID.get_or_init(|| {
            Arc::new(Self::parse(GCR_TERRAIN_BYTES).expect("committed artifact parses"))
        })
        .clone()
    }

    /// Full world-x span (grid columns).
    pub(crate) fn extent_x(&self) -> f32 {
        (self.cols - 1) as f32 * self.cell
    }

    /// Full world-z span (grid rows).
    pub(crate) fn extent_z(&self) -> f32 {
        (self.rows - 1) as f32 * self.cell
    }

    fn at(&self, row: usize, col: usize) -> f32 {
        self.heights[row * self.cols + col]
    }

    /// Surface height at world `(x, z)` — exact on the collider's triangles: parry's
    /// default (non-zigzag) subdivision splits each cell along the (row+1,col)—(row,col+1)
    /// diagonal, and this reproduces that split, so a point this fn puts ON the surface
    /// is on the physics surface. Clamps outside the tile (the edge continues flat).
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let u = ((x / self.extent_x() + 0.5) * (self.cols - 1) as f32)
            .clamp(0.0, (self.cols - 1) as f32);
        let v = ((z / self.extent_z() + 0.5) * (self.rows - 1) as f32)
            .clamp(0.0, (self.rows - 1) as f32);
        let col = (u as usize).min(self.cols - 2);
        let row = (v as usize).min(self.rows - 2);
        let (fu, fv) = (u - col as f32, v - row as f32);
        let h00 = self.at(row, col);
        let h10 = self.at(row + 1, col);
        let h01 = self.at(row, col + 1);
        let h11 = self.at(row + 1, col + 1);
        if fu + fv <= 1.0 {
            h00 + (h10 - h00) * fv + (h01 - h00) * fu
        } else {
            h11 + (h10 - h11) * (1.0 - fu) + (h01 - h11) * (1.0 - fv)
        }
    }

    /// THE spawn-on-surface primitive: the world point `height_above` meters over the
    /// surface at `xz`. Spawns and targets go through this so nothing seeds below a
    /// hill or floats over a valley. Takes the offset as a separate scalar — not a Vec3
    /// whose y is secretly an offset — so an already-lifted point can't be lifted twice.
    pub fn place(&self, xz: Vec2, height_above: f32) -> Vec3 {
        Vec3::new(xz.x, self.height(xz.x, xz.y) + height_above, xz.y)
    }

    /// The whole-tile static collider. Same grid as [`Self::height`] and [`Self::mesh`],
    /// transposed to parry's column-major layout (rows along +z, columns along +x).
    /// Built with `FIX_INTERNAL_EDGES`: the crab's soft contacts rest with ~cm
    /// penetration, and without the flag a foot straddling a cell seam takes a tilted
    /// ghost impulse from the neighbor triangle's edge — on the flat arenas the seam
    /// diagonal passes through the spawn origin, so the flag is what keeps the
    /// heightfield floor contact-equivalent to the slab it replaced.
    pub fn collider(&self) -> Collider {
        use bevy_rapier3d::rapier::geometry::SharedShape;
        use bevy_rapier3d::rapier::parry::shape::{HeightField, HeightFieldFlags};
        use bevy_rapier3d::rapier::parry::utils::Array2;

        let mut col_major = vec![0.0f32; self.rows * self.cols];
        for col in 0..self.cols {
            for row in 0..self.rows {
                col_major[col * self.rows + row] = self.at(row, col);
            }
        }
        let field = HeightField::with_flags(
            Array2::new(self.rows, self.cols, col_major),
            Vec3::new(self.extent_x(), 1.0, self.extent_z()),
            HeightFieldFlags::FIX_INTERNAL_EDGES,
        );
        SharedShape::new(field).into()
    }

    /// The visible terrain surface — the SAME vertices and cell diagonal as the
    /// collider's triangles, so render matches physics by construction. Smooth
    /// per-vertex normals via central differences (stage 3 owns looks beyond that).
    #[cfg(feature = "render")]
    pub fn mesh(&self) -> Mesh {
        use bevy::asset::RenderAssetUsages;
        use bevy::mesh::{Indices, PrimitiveTopology};

        let (rows, cols) = (self.rows, self.cols);
        let (ex, ez) = (self.extent_x(), self.extent_z());
        let mut positions = Vec::with_capacity(rows * cols);
        let mut normals = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let x = (col as f32 / (cols - 1) as f32 - 0.5) * ex;
                let z = (row as f32 / (rows - 1) as f32 - 0.5) * ez;
                positions.push([x, self.at(row, col), z]);
                let (c0, c1) = (col.saturating_sub(1), (col + 1).min(cols - 1));
                let (r0, r1) = (row.saturating_sub(1), (row + 1).min(rows - 1));
                let dhdx = (self.at(row, c1) - self.at(row, c0)) / (self.cell * (c1 - c0) as f32);
                let dhdz = (self.at(r1, col) - self.at(r0, col)) / (self.cell * (r1 - r0) as f32);
                normals.push(Vec3::new(-dhdx, 1.0, -dhdz).normalize().to_array());
            }
        }
        let mut indices = Vec::with_capacity((rows - 1) * (cols - 1) * 6);
        for row in 0..rows - 1 {
            for col in 0..cols - 1 {
                let v00 = (row * cols + col) as u32;
                let v10 = ((row + 1) * cols + col) as u32;
                let v01 = (row * cols + col + 1) as u32;
                let v11 = ((row + 1) * cols + col + 1) as u32;
                indices.extend([v00, v10, v01, v10, v11, v01]);
            }
        }
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_indices(Indices::U32(indices))
    }
}

/// The world's terrain, inserted by `PhysicsWorldPlugin` from its [`Arena`] choice.
/// `Arc` because eval worlds are rebuilt per bearing and training worlds per env batch —
/// the (possibly 4 MB) grid is shared, not re-parsed.
///
/// [`Arena`]: crate::physics::Arena
#[derive(Resource, Clone)]
pub struct Terrain(Arc<TerrainGrid>);

impl Terrain {
    /// Only `PhysicsWorldPlugin` constructs this, from its `Arena` — the grid the
    /// resource carries is BY CONSTRUCTION the one the ground collider was built from.
    /// Deliberately no `Default`: a defaulted flat grid where the arena's real grid was
    /// meant is exactly the sampler/collider divergence this module exists to prevent
    /// (a missing resource panics loudly instead).
    pub(crate) fn new(grid: Arc<TerrainGrid>) -> Self {
        Terrain(grid)
    }
}

impl std::ops::Deref for Terrain {
    type Target = TerrainGrid;

    fn deref(&self) -> &TerrainGrid {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_gcr_artifact_parses_and_is_datum_shifted() {
        let g = TerrainGrid::gcr();
        assert_eq!((g.rows, g.cols), (1024, 1024));
        assert_eq!(g.cell, 30.0);
        // Datum shift: the (rows/2, cols/2) sample sits at y=0, and sampling its world
        // position agrees (grid coord 512 of 0..=1023 → +0.5 cell off world center).
        assert_eq!(g.at(512, 512), 0.0);
        let x = (512.0 / 1023.0 - 0.5) * g.extent_x();
        let z = (512.0 / 1023.0 - 0.5) * g.extent_z();
        assert!(g.height(x, z).abs() < 1e-2);
        // Metadata pins elevation 295..=4508 m; shifted by the center sample the span
        // must be exactly preserved.
        let (min, max) = g
            .heights
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
        assert_eq!(max - min, (4508 - 295) as f32);
    }

    #[test]
    fn flat_grid_is_zero_everywhere_and_spans_the_arena() {
        let g = TerrainGrid::flat(10.0);
        assert_eq!(g.extent_x(), 20.0);
        assert_eq!(g.extent_z(), 20.0);
        for (x, z) in [(0.0, 0.0), (-10.0, 10.0), (7.3, -2.1), (500.0, -500.0)] {
            assert_eq!(g.height(x, z), 0.0, "flat at ({x},{z})");
        }
        assert_eq!(
            g.place(Vec2::new(1.0, -2.0), 0.3),
            Vec3::new(1.0, 0.3, -2.0)
        );
    }

    /// The sampler must agree with the collider's actual triangles — render/spawn
    /// matching physics is the whole point of the one-path seam. Samples the parry
    /// heightfield's triangle pair per probed cell and checks the plane height.
    #[test]
    fn sampler_is_exact_on_the_collider_triangles() {
        let g = TerrainGrid::gcr();
        let collider = g.collider();
        let hf = collider
            .as_heightfield()
            .expect("terrain collider is a heightfield");
        let raw = hf.raw;

        let mut checked = 0;
        for (row, col) in [(0, 0), (511, 512), (512, 511), (700, 200), (1022, 1022)] {
            let (t1, t2) = raw.triangles_at(row, col);
            for tri in [t1.expect("left tri"), t2.expect("right tri")] {
                // The triangle centroid is strictly inside, so no diagonal/edge ties.
                let c = (tri.a + tri.b + tri.c) / 3.0;
                let sampled = g.height(c.x, c.z);
                assert!(
                    (sampled - c.y).abs() < 1e-2,
                    "cell ({row},{col}): sampler {sampled} vs collider {}",
                    c.y
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 10);
    }

    /// Pin the artifact→world orientation: artifact row 0 / col 0 (the preview PNG's
    /// top-left) is the (-x, -z) corner of the world tile.
    #[test]
    fn artifact_row0_col0_is_the_minus_x_minus_z_corner() {
        let g = TerrainGrid::gcr();
        let (hx, hz) = (g.extent_x() / 2.0, g.extent_z() / 2.0);
        assert_eq!(g.height(-hx, -hz), g.at(0, 0));
        assert_eq!(g.height(hx, -hz), g.at(0, 1023));
        assert_eq!(g.height(-hx, hz), g.at(1023, 0));
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(TerrainGrid::parse(b"").is_err());
        assert!(TerrainGrid::parse(b"NOTMAGIC\0\0\0\0").is_err());
        // Right magic, truncated grid.
        let mut bytes = MAGIC.to_vec();
        let meta = br#"{"rows":2,"cols":2,"cell_size_m":1.0,"height_scale":1.0}"#;
        bytes.extend((meta.len() as u32).to_le_bytes());
        bytes.extend(meta);
        bytes.extend([0u8; 6]); // want 8
        assert!(TerrainGrid::parse(&bytes).is_err());
    }
}
