//! The moisture map (bddap/rl#323): the ground look's wetness spine, derived from
//! the terrain height field itself rather than painted with noise. Moisture pools
//! at local minima and follows drainage, so the wet/dry map the player reads from
//! the plane is a real consequence of the silhouette — the same grid the collider
//! and the mesh are built from is the one this bakes over.
//!
//! Three classical hydrology passes, fixed physical thresholds (nothing is
//! normalized to the tile, so a flat fixture and a mountain tile are judged by the
//! same rules):
//!
//! 1. **Depression fill** (priority-flood, Barnes 2014): water drains off the tile
//!    edges; `fill − h` is positive exactly inside closed basins below their spill
//!    point — standing water, the literal "pools at local minima".
//! 2. **Flow accumulation** (D8 on the filled surface): upstream catchment area per
//!    cell, so valleys carry dampness along drainage lines even where nothing ponds.
//! 3. **Topographic wetness index** `ln(a / tan β)`: catchment against local slope —
//!    flat-and-fed reads wet, steep-and-starved reads dry.
//!
//! Baked once at arena setup into an `Rg8Unorm` texture (R wetness, G standing
//! water) sampled by the ground look shaders through [`crate::ground::GroundDetail`].

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::terrain::TerrainGrid;

/// Standing water saturates over this pond depth (meters): a knee-deep basin core
/// reads as a full puddle, a centimeter film only as damp ground.
const POND_FULL_M: f64 = 1.5;

/// TWI stops: below `DRY` a cell is bone dry, above `WET` fully soaked. On a 30 m
/// grid a ridge cell (catchment one cell, tan β ≈ 0.3) sits near 4.6 and a fed
/// valley flat (tan β ≈ 0.02, ~10³ cells upstream) near 14 — these stops spread
/// that physical range over the full wetness ramp.
const TWI_DRY: f32 = 5.5;
const TWI_WET: f32 = 12.5;

/// The baked hydrology fields, row-major like the grid. `data` is interleaved
/// `[wetness, pond]` bytes — exactly the `Rg8Unorm` texel layout.
pub struct MoistureMap {
    rows: usize,
    cols: usize,
    data: Vec<u8>,
}

/// 8-connected neighbors with their center distance in cells (1 or √2) — D8's
/// candidate set for both the fill frontier and steepest descent.
const NEIGHBORS: [(isize, isize, f64); 8] = [
    (-1, -1, std::f64::consts::SQRT_2),
    (-1, 0, 1.0),
    (-1, 1, std::f64::consts::SQRT_2),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, std::f64::consts::SQRT_2),
    (1, 0, 1.0),
    (1, 1, std::f64::consts::SQRT_2),
];

/// f64 keyed min-heap entry. Heights span kilometers, so the fill runs in f64: the
/// epsilon that forces filled flats to drain (1e-6 m) is below f32 resolution at
/// those magnitudes.
#[derive(PartialEq)]
struct Frontier(f64, u32);
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

impl MoistureMap {
    pub fn bake(grid: &TerrainGrid) -> Self {
        let (rows, cols) = grid.dims();
        let cell = grid.cell_m() as f64;
        let n = rows * cols;
        let h: Vec<f64> = grid.heights_row_major().iter().map(|&v| v as f64).collect();

        // Pass 1 — priority-flood depression fill, seeded from the whole boundary
        // (the tile edge is the outlet). Each cell locks in the lowest fill level
        // reachable from an outlet; the epsilon over the frontier makes filled
        // lake surfaces drain toward their spill instead of going flat.
        let mut fill = h.clone();
        let mut seen = vec![false; n];
        let mut heap: BinaryHeap<Reverse<Frontier>> = BinaryHeap::new();
        for i in 0..n {
            let (r, c) = (i / cols, i % cols);
            if r == 0 || c == 0 || r == rows - 1 || c == cols - 1 {
                seen[i] = true;
                heap.push(Reverse(Frontier(h[i], i as u32)));
            }
        }
        while let Some(Reverse(Frontier(level, i))) = heap.pop() {
            let (r, c) = (i as usize / cols, i as usize % cols);
            for (dr, dc, _) in NEIGHBORS {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                if nr < 0 || nc < 0 || nr as usize >= rows || nc as usize >= cols {
                    continue;
                }
                let j = nr as usize * cols + nc as usize;
                if !seen[j] {
                    seen[j] = true;
                    fill[j] = h[j].max(level + 1e-6);
                    heap.push(Reverse(Frontier(fill[j], j as u32)));
                }
            }
        }

        // Pass 2 — D8 flow accumulation over the filled surface, high to low: every
        // cell hands its catchment (itself included) to its steepest-descent
        // neighbor. The fill guarantees each interior cell HAS a lower neighbor, so
        // nothing strands short of the boundary.
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(|&a, &b| fill[b as usize].total_cmp(&fill[a as usize]));
        let mut acc = vec![1.0f64; n];
        for &i in &order {
            let i = i as usize;
            let (r, c) = (i / cols, i % cols);
            let mut best: Option<(usize, f64)> = None;
            for (dr, dc, dist) in NEIGHBORS {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                if nr < 0 || nc < 0 || nr as usize >= rows || nc as usize >= cols {
                    continue;
                }
                let j = nr as usize * cols + nc as usize;
                let drop = (fill[i] - fill[j]) / dist;
                if drop > 0.0 && best.is_none_or(|(_, d)| drop > d) {
                    best = Some((j, drop));
                }
            }
            if let Some((j, _)) = best {
                acc[j] += acc[i];
            }
        }

        // Pass 3 — combine into the two texels. Slope from the ORIGINAL heights
        // (central differences): ponds are flat on top but their ground is not.
        let mut wetness = vec![0.0f32; n];
        let mut pond = vec![0.0f32; n];
        for i in 0..n {
            let (r, c) = (i / cols, i % cols);
            let (r0, r1) = (r.saturating_sub(1), (r + 1).min(rows - 1));
            let (c0, c1) = (c.saturating_sub(1), (c + 1).min(cols - 1));
            let dhdx = (h[r * cols + c1] - h[r * cols + c0]) / (cell * (c1 - c0).max(1) as f64);
            let dhdz = (h[r1 * cols + c] - h[r0 * cols + c]) / (cell * (r1 - r0).max(1) as f64);
            let tan_b = (dhdx * dhdx + dhdz * dhdz).sqrt();
            // Specific catchment area in meters (upstream cells × cell width).
            let twi = ((acc[i] * cell) / (tan_b + 0.01)).ln() as f32;
            let depth = fill[i] - h[i];
            pond[i] = smoothstep(0.0, POND_FULL_M as f32, depth as f32);
            // Damp ground: TWI, floored by the pond (standing water is wet by
            // definition, whatever its catchment says).
            wetness[i] = smoothstep(TWI_DRY, TWI_WET, twi).max(pond[i]);
        }

        // One 3×3 box pass on wetness: D8 drainage lines are one cell wide, and a
        // single blur widens them past the bilinear filter's reach so streams read
        // as damp corridors instead of jaggies. The pond floor is re-applied after
        // the blur — a one-cell pond must not have its wet ground averaged away.
        let blurred = box3(&wetness, rows, cols);

        let data = blurred
            .iter()
            .zip(&pond)
            .map(|(&w, &p)| (w.max(p), p))
            .flat_map(|(w, p)| {
                [
                    (w.clamp(0.0, 1.0) * 255.0) as u8,
                    (p.clamp(0.0, 1.0) * 255.0) as u8,
                ]
            })
            .collect();
        Self { rows, cols, data }
    }

    /// The bake as a world-mapped texture: `uv = world_xz / extent + 0.5`, exactly
    /// the transform `TerrainGrid::height` uses, so texel (row, col) lands on the
    /// same world point as grid sample (row, col). Bilinear + clamp: the tile edge
    /// continues flat, so its moisture continues too.
    pub fn image(&self) -> Image {
        let mut image = Image::new(
            Extent3d {
                width: self.cols as u32,
                height: self.rows as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            self.data.clone(),
            TextureFormat::Rg8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::ClampToEdge,
            address_mode_v: ImageAddressMode::ClampToEdge,
            ..ImageSamplerDescriptor::linear()
        });
        image
    }

    /// Wetness at grid (row, col), 0..=1 — the R texel, for tests and tools.
    pub fn wetness(&self, row: usize, col: usize) -> f32 {
        self.data[(row * self.cols + col) * 2] as f32 / 255.0
    }

    /// Standing-water saturation at grid (row, col), 0..=1 — the G texel.
    pub fn pond(&self, row: usize, col: usize) -> f32 {
        self.data[(row * self.cols + col) * 2 + 1] as f32 / 255.0
    }
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn box3(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            let (r0, r1) = (r.saturating_sub(1), (r + 1).min(rows - 1));
            let (c0, c1) = (c.saturating_sub(1), (c + 1).min(cols - 1));
            let mut sum = 0.0;
            let mut count = 0.0;
            for rr in r0..=r1 {
                for cc in c0..=c1 {
                    sum += src[rr * cols + cc];
                    count += 1.0;
                }
            }
            out[r * cols + c] = sum / count;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode heights (meters, i16) through the real artifact codec. The datum
    /// shift subtracts the center sample, which moves every height equally —
    /// hydrology is shift-invariant, so tests state heights directly.
    fn grid(rows: usize, cols: usize, heights: &[i16]) -> TerrainGrid {
        TerrainGrid::test_grid(rows, cols, 30.0, 1.0, heights)
    }

    /// A closed bowl: rim at 40 m, floor at 0. Water cannot leave, so the floor
    /// ponds up to the spill and reads fully wet; the outside stays dry.
    #[test]
    fn bowl_ponds_at_its_local_minimum() {
        let n = 9;
        let mut h = vec![0i16; n * n];
        for r in 0..n {
            for c in 0..n {
                let ring = (r.abs_diff(4)).max(c.abs_diff(4));
                h[r * n + c] = match ring {
                    0 => 0,
                    1 => 10,
                    2 => 25,
                    3 => 40,
                    _ => 5,
                };
            }
        }
        let m = MoistureMap::bake(&grid(n, n, &h));
        assert_eq!(m.pond(4, 4), 1.0, "bowl floor is standing water");
        assert_eq!(m.wetness(4, 4), 1.0, "standing water is wet");
        assert_eq!(m.pond(4, 1), 0.0, "outside the rim nothing ponds");
    }

    /// A monotone slope drains everywhere: no cell sits below a spill point, so no
    /// standing water anywhere.
    #[test]
    fn monotone_slope_never_ponds() {
        let n = 12;
        let h: Vec<i16> = (0..n * n).map(|i| ((i / n) * 20) as i16).collect();
        let m = MoistureMap::bake(&grid(n, n, &h));
        for r in 0..n {
            for c in 0..n {
                assert_eq!(m.pond(r, c), 0.0, "pond on a draining slope at {r},{c}");
            }
        }
    }

    /// A V-valley between two ridges: the valley line collects the whole tile's
    /// drainage and must read wetter than the ridge crests above it.
    #[test]
    fn valley_line_is_wetter_than_the_ridges() {
        let n = 15;
        let mut h = vec![0i16; n * n];
        for r in 0..n {
            for c in 0..n {
                // Distance from the center column makes the walls; a gentle fall
                // along +r gives the valley itself an outlet.
                h[r * n + c] = (c.abs_diff(7) as i16) * 30 - (r as i16) * 2;
            }
        }
        let m = MoistureMap::bake(&grid(n, n, &h));
        let valley = m.wetness(12, 7);
        let ridge = m.wetness(7, 1).max(m.wetness(7, 13));
        assert!(
            valley > ridge + 0.2,
            "valley {valley} not clearly wetter than ridge {ridge}"
        );
    }

    /// The GCR tile itself: the bake must produce real contrast (both dry and wet
    /// ground exist) and at least some standing water — a mountain tile with no
    /// pond anywhere means the fill found nothing, i.e. the pass is broken.
    #[test]
    fn gcr_has_dry_ground_wet_ground_and_ponds() {
        let g = TerrainGrid::gcr();
        let m = MoistureMap::bake(&g);
        let (rows, cols) = g.dims();
        let mut dry = 0usize;
        let mut wet = 0usize;
        let mut ponded = 0usize;
        for r in 0..rows {
            for c in 0..cols {
                if m.wetness(r, c) < 0.15 {
                    dry += 1;
                }
                if m.wetness(r, c) > 0.85 {
                    wet += 1;
                }
                if m.pond(r, c) > 0.5 {
                    ponded += 1;
                }
            }
        }
        let n = rows * cols;
        assert!(dry > n / 10, "dry ground missing ({dry}/{n})");
        assert!(wet > n / 200, "wet ground missing ({wet}/{n})");
        assert!(ponded > 0, "no standing water on a 4 km-relief tile");
    }
}
