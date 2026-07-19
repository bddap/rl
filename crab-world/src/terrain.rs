//! The ONE terrain path (rl#281): a baked height grid drives BOTH the rapier
//! [`Collider`] and (render-gated) the visible mesh, so wherever this module's mesh is
//! what's drawn (rl-demo, GCR) it cannot diverge from what the crab stands on. Since
//! the rl#293 flip the committed GCR bake is the only production grid; tests may build
//! constant grids through this same seam ([`TerrainGrid::flat`], test-gated) — there
//! is deliberately no flat-vs-terrain fork anywhere.
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

#[cfg(feature = "render")]
use crate::sky::{hash3, rand01, smoothstep};

/// The stage-1 bake artifact for GCR (rl#281): seed 281, 1024², 30 m pitch.
static GCR_TERRAIN_BYTES: &[u8] = include_bytes!("../assets/terrain/gcr-seed281.terrain");

/// Digest of the committed bake artifact, for the MP plant handshake (rl#286): two
/// peers whose binaries embed different bakes stand on different mountains even when
/// both say "terrain", so the bytes themselves are what the digest must witness.
pub fn gcr_bake_digest() -> u64 {
    static DIGEST: OnceLock<u64> = OnceLock::new();
    *DIGEST.get_or_init(|| crate::fnv::fnv1a(GCR_TERRAIN_BYTES))
}

const MAGIC: &[u8; 8] = b"RLTERR01";

/// Cell-size cap for [`TerrainGrid::flat`], for f32 contact precision — parry solves
/// heightfield contacts against the containing cell's triangle, and one grid-spanning
/// cell on a ±16 km fixture puts that triangle's vertices where f32 ulp is ~2 mm,
/// comparable to contact tolerances (and it measurably perturbed the rl#224 flail-walk).
/// 256 m keeps near-origin triangle vertices small at 129² points for the largest
/// fixtures, instead of megabytes of zeros per world.
#[cfg(any(test, feature = "test-grid"))]
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
    /// Height span over the whole grid, computed once at construction.
    relief: f32,
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
        let heights: Vec<f32> = raw
            .iter()
            .map(|&h| (i32::from(h) - i32::from(datum)) as f32 * meta.height_scale)
            .collect();
        let (min, max) = heights
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
        Ok(Self {
            rows: meta.rows,
            cols: meta.cols,
            cell: meta.cell_size_m,
            heights,
            relief: max - min,
        })
    }

    /// A constant y=0 TEST grid spanning ±`half_extent`, expressed through the one
    /// terrain path instead of a bespoke slab or halfspace — for tests whose expected
    /// geometry is hand-computed on a plane. Test-gated since the rl#293 flat-arena
    /// deletion (production ground is [`Self::gcr`], impossible to fork by
    /// construction); net's tests reach it via the `test-grid` dev-dependency
    /// feature. Grids that run band logic ([`crate::training::targets`]) must span
    /// ≥ the edge margin + band — smaller grids fail the sampling clamp's assert.
    #[cfg(any(test, feature = "test-grid"))]
    pub fn flat(half_extent: f32) -> Self {
        let cells = ((half_extent * 2.0) / FLAT_CELL_MAX_M).ceil().max(1.0) as usize;
        let n = cells + 1;
        Self {
            rows: n,
            cols: n,
            cell: half_extent * 2.0 / cells as f32,
            heights: vec![0.0; n * n],
            relief: 0.0,
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
    pub fn extent_x(&self) -> f32 {
        (self.cols - 1) as f32 * self.cell
    }

    /// Full world-z span (grid rows).
    pub fn extent_z(&self) -> f32 {
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
    /// ghost impulse from the neighbor triangle's edge — on a flat grid the seam
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

    /// Height span over the whole grid — 0 for the flat test grids.
    pub fn relief(&self) -> f32 {
        self.relief
    }

    /// The visible terrain surface — the SAME vertices and cell diagonal as the
    /// collider's triangles, so render matches physics by construction. Smooth
    /// per-vertex normals via central differences; per-vertex biome tint from
    /// elevation + slope (rl#281 stage 3). No LOD: the 1024² tile is ~2M triangles in
    /// one static buffer, which the demo already renders live at full rate — decimation
    /// would only buy back memory we don't miss and open a render/physics seam gap.
    #[cfg(feature = "render")]
    pub fn mesh(&self) -> Mesh {
        use bevy::asset::RenderAssetUsages;
        use bevy::mesh::{Indices, PrimitiveTopology};

        let (rows, cols) = (self.rows, self.cols);
        let (ex, ez) = (self.extent_x(), self.extent_z());
        let mut positions = Vec::with_capacity(rows * cols);
        let mut normals = Vec::with_capacity(rows * cols);
        let mut colors = Vec::with_capacity(rows * cols);
        // UVs in mesh-local METERS (uv = xz) so the ground material can tile the
        // rl#197 pilot checker at a scale IT owns (bddap/rl#287): the 30 m-pitch
        // surface carries no high-frequency detail of its own, so the checker is the
        // landing-height/on-foot optic-flow cue (rl#293).
        let mut uvs = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let x = (col as f32 / (cols - 1) as f32 - 0.5) * ex;
                let z = (row as f32 / (rows - 1) as f32 - 0.5) * ez;
                let h = self.at(row, col);
                positions.push([x, h, z]);
                uvs.push([x, z]);
                let (c0, c1) = (col.saturating_sub(1), (col + 1).min(cols - 1));
                let (r0, r1) = (row.saturating_sub(1), (row + 1).min(rows - 1));
                let dhdx = (self.at(row, c1) - self.at(row, c0)) / (self.cell * (c1 - c0) as f32);
                let dhdz = (self.at(r1, col) - self.at(r0, col)) / (self.cell * (r1 - r0) as f32);
                let normal = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
                normals.push(normal.to_array());
                colors.push(biome_tint(h, normal.y, row as i32, col as i32));
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
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
    }
}

/// Per-vertex biome color (linear RGBA — multiplied into the terrain material):
/// a hypsometric ramp over datum-shifted elevation `h` in METERS — not the tile's
/// normalized range, because the datum shift puts y=0 where the crab lives (the tile's
/// center sample), so bands stay anchored to gameplay ground whatever a bake's absolute
/// span is (a normalized ramp painted the whole origin plateau as snow). Rock on steep
/// slopes, snow only on gentle high peaks, and a per-vertex hash jitter so
/// kilometer-scale bands don't read as flat paint.
#[cfg(feature = "render")]
fn biome_tint(h: f32, normal_y: f32, row: i32, col: i32) -> [f32; 4] {
    // (elevation m, srgb color) stops, low → high (bake.py's hillshade palette).
    const LAND: [(f32, [f32; 3]); 5] = [
        (-3200.0, [0.20, 0.36, 0.16]), // deep valley green
        (-1400.0, [0.27, 0.42, 0.18]), // lowland green
        (-500.0, [0.40, 0.50, 0.24]),  // dry grass
        (100.0, [0.60, 0.50, 0.32]),   // tan scree (crab country rim)
        (600.0, [0.59, 0.47, 0.39]),   // high brown
    ];
    const ROCK: [f32; 3] = [0.42, 0.39, 0.36];
    const SNOW: [f32; 3] = [0.88, 0.89, 0.93];
    const SNOWLINE_M: (f32, f32) = (350.0, 650.0);

    let lin = |c: [f32; 3]| LinearRgba::from(Color::srgb(c[0], c[1], c[2]));
    let mut c = lin(LAND[LAND.len() - 1].1);
    for w in LAND.windows(2) {
        let ((h0, c0), (h1, c1)) = (w[0], w[1]);
        if h < h1 {
            c = lin(c0).mix(&lin(c1), ((h - h0) / (h1 - h0)).clamp(0.0, 1.0));
            break;
        }
    }

    // Slope 0 → normal_y 1. Rock takes over from ~35° and owns ~55°+.
    let steep = 1.0 - normal_y;
    c = c.mix(&lin(ROCK), smoothstep(0.18, 0.42, steep));
    // Snowline: high AND not too steep (snow doesn't hold on cliffs).
    let snow = smoothstep(SNOWLINE_M.0, SNOWLINE_M.1, h) * (1.0 - smoothstep(0.18, 0.38, steep));
    c = c.mix(&lin(SNOW), snow);

    // ±6% value jitter, deterministic per vertex.
    let jitter = 1.0 + 0.06 * (2.0 * rand01(hash3(row, col, 0)) - 1.0);
    [c.red * jitter, c.green * jitter, c.blue * jitter, 1.0]
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
    fn relief_is_zero_flat_and_full_span_on_gcr() {
        assert_eq!(TerrainGrid::flat(10.0).relief(), 0.0);
        assert_eq!(TerrainGrid::gcr().relief(), (4508 - 295) as f32);
    }

    /// The rl#197 pilot-cue contract (bddap/rl#287 → rl#293): every ground mesh
    /// carries UVs in mesh-local METERS (uv = xz) so the ground material can tile the
    /// checker detail texture at a scale it owns.
    #[cfg(feature = "render")]
    #[test]
    fn mesh_uvs_are_mesh_local_meters() {
        use bevy::mesh::VertexAttributeValues;

        let mesh = TerrainGrid::gcr().mesh();
        let (pos, uv) = match (
            mesh.attribute(Mesh::ATTRIBUTE_POSITION),
            mesh.attribute(Mesh::ATTRIBUTE_UV_0),
        ) {
            (
                Some(VertexAttributeValues::Float32x3(p)),
                Some(VertexAttributeValues::Float32x2(u)),
            ) => (p, u),
            other => panic!("expected positions + Float32x2 UVs, got {other:?}"),
        };
        for (p, u) in pos.iter().zip(uv) {
            assert_eq!([p[0], p[2]], *u);
        }
    }

    /// The biome tint contract the taste loop leans on: banded colors with
    /// snow-bright peaks and darker valleys.
    #[cfg(feature = "render")]
    #[test]
    fn mesh_tint_terrain_banded() {
        use bevy::mesh::VertexAttributeValues;

        let colors = |m: &Mesh| match m.attribute(Mesh::ATTRIBUTE_COLOR) {
            Some(VertexAttributeValues::Float32x4(v)) => v.clone(),
            other => panic!("expected Float32x4 colors, got {other:?}"),
        };

        let g = TerrainGrid::gcr();
        let terr = colors(&g.mesh());
        let luma = |c: &[f32; 4]| c[0] + c[1] + c[2];
        // Somewhere up high there is snow (only the snow stop reaches this luma)...
        let max_luma = terr.iter().map(luma).fold(f32::MIN, f32::max);
        assert!(max_luma > 2.0, "no snow-bright vertex: max luma {max_luma}");
        // ...and a real population of dark green valley ground (not one lucky vertex).
        let green_dark = terr
            .iter()
            .filter(|c| luma(c) < 0.5 && c[1] > c[0] && c[1] > c[2])
            .count();
        assert!(
            green_dark > terr.len() / 100,
            "expected ≥1% dark green vertices, got {green_dark}/{}",
            terr.len()
        );
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
