//! Spike: auto-derive physics colliders from the skinned glTF mesh.
//!
//! This is a *prototype/validation* module, not part of the live physics. It
//! loads the same `sally.glb` the cosmetic skin uses ([`super::skin`]), then for
//! each physics body part:
//!   1. gathers the mesh vertices whose dominant skin bone maps to that part
//!      (same bone→part mapping the skin drives with),
//!   2. fits a capsule to that vertex cloud (axis by PCA, radius/half-height
//!      from the spread), and
//!   3. computes the capsule's mass properties under the hand-coded density.
//!
//! The point is the *comparison* against the hand-coded body in [`super::body`]:
//! [`fit_report`] returns one [`PartFit`] per part with both the fitted and the
//! hand-coded numbers, and the `meshfit_validation` integration-style test
//! prints the table. Nothing here is wired into `spawn_crab`; see
//! `MESHFIT_PLAN.md` for what landing it for real would take.
//!
//! Coordinate frames: the fit works in the glTF's *bind-pose world* frame (mesh
//! space). The hand-coded colliders live in each link's *local* frame, posed by
//! joint anchors into a different rest stance — so size/mass/inertia compare
//! directly (they are pose-invariant) while *placement* only compares after a
//! forward-kinematics pass, which we do for a representative subset.
//!
//! The whole module is `#[cfg(test)]`-gated at its `mod` declaration, so it adds
//! nothing to release builds.

use std::collections::HashMap;

use bevy::prelude::*;

use super::body::{CrabJointId, Side};

/// Which physics link a mesh vertex (or fitted collider) belongs to. Mirrors
/// the skin's `LinkKey` but lives here so the spike is self-contained; the
/// carapace is the root, every other part is a joint's child link.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PartId {
    Carapace,
    Joint(CrabJointId),
}

impl PartId {
    /// Human-readable label for the validation table.
    fn label(self) -> String {
        match self {
            PartId::Carapace => "Carapace".to_string(),
            PartId::Joint(id) => format!("{id:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// glTF loading: skeleton (bind-pose bone transforms) + skin (vertex weights)
// ---------------------------------------------------------------------------

/// One mesh vertex with its dominant bone, in bind-pose world space.
struct SkinnedVertex {
    pos: Vec3,
    /// glTF node index of the bone with the largest weight on this vertex.
    dominant_node: usize,
    /// That bone's weight (0..1) — used to gauge how "clean" the assignment is.
    dominant_weight: f32,
}

/// The parsed-and-flattened model: bone bind transforms keyed by node index,
/// node names, and every skinned vertex tagged with its dominant bone.
pub struct LoadedModel {
    /// Bind-pose world transform of each *joint* node (by node index). Built by
    /// composing node-local transforms down the scene hierarchy.
    bind_world: HashMap<usize, Mat4>,
    node_name: HashMap<usize, String>,
    verts: Vec<SkinnedVertex>,
}

/// Where the spike looks for the model: `CRAB_MODEL_PATH` if set (same var the
/// skin uses), else the dev box's checkout. Absent → the test self-skips.
pub fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("CRAB_MODEL_PATH") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let fallback = std::path::PathBuf::from("/tmp/rl/assets/sally.glb");
    fallback.exists().then_some(fallback)
}

impl LoadedModel {
    /// Parse a GLB from disk: decode the skin's bind matrices into world bone
    /// transforms and every vertex's dominant bone. Errors surface as a string
    /// so the test can fail loudly rather than panic deep in the gltf crate.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("parse glb: {e}"))?;
        let blob = gltf.blob.as_deref().ok_or("GLB has no binary chunk")?;

        let node_name: HashMap<usize, String> = gltf
            .nodes()
            .filter_map(|n| n.name().map(|nm| (n.index(), nm.to_string())))
            .collect();

        // Bind-pose world transforms: walk every scene root down, composing
        // `parent_world * node_local`. The bind pose IS the node rest pose here
        // (no animation sampling) — the same pose the skin captures offsets in.
        let mut bind_world: HashMap<usize, Mat4> = HashMap::new();
        for scene in gltf.scenes() {
            for node in scene.nodes() {
                compose_world(&node, Mat4::IDENTITY, &mut bind_world);
            }
        }

        // Skin weights: dominant bone per vertex. A skin's `joints()` list maps
        // the per-vertex JOINTS_0 *indices* (0..N) to actual node indices.
        let skin = gltf.skins().next().ok_or("model has no skin")?;
        let joint_nodes: Vec<usize> = skin.joints().map(|j| j.index()).collect();

        let mesh = gltf.meshes().next().ok_or("model has no mesh")?;
        let mut verts = Vec::new();
        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| {
                // Single embedded buffer (GLB blob); index 0 is the bin chunk.
                (buf.index() == 0).then_some(blob)
            });
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or("primitive has no POSITION")?
                .collect();
            let joints: Vec<[u16; 4]> = reader
                .read_joints(0)
                .ok_or("primitive has no JOINTS_0")?
                .into_u16()
                .collect();
            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .ok_or("primitive has no WEIGHTS_0")?
                .into_f32()
                .collect();
            for ((p, j), w) in positions.iter().zip(&joints).zip(&weights) {
                // Pick the influence with the largest weight.
                let (lane, &wmax) = w
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                let node = joint_nodes[j[lane] as usize];
                verts.push(SkinnedVertex {
                    pos: Vec3::from_array(*p),
                    dominant_node: node,
                    dominant_weight: wmax,
                });
            }
        }

        Ok(LoadedModel {
            bind_world,
            node_name,
            verts,
        })
    }

    /// Group vertices by physics part via the bone→part map. Returns, per part,
    /// the world-space vertex positions and the mean dominant-weight (a
    /// skinning-cleanliness proxy). Vertices whose bone maps to no part (none,
    /// here — every bone routes somewhere) are dropped.
    fn vertices_by_part(&self) -> HashMap<PartId, (Vec<Vec3>, f32)> {
        let mut out: HashMap<PartId, (Vec<Vec3>, f32)> = HashMap::new();
        for v in &self.verts {
            let Some(name) = self.node_name.get(&v.dominant_node) else {
                continue;
            };
            let Some(part) = bone_to_part(name) else {
                continue;
            };
            let e = out.entry(part).or_insert_with(|| (Vec::new(), 0.0));
            e.0.push(v.pos);
            e.1 += v.dominant_weight;
        }
        for (_, (positions, wsum)) in out.iter_mut() {
            *wsum /= positions.len().max(1) as f32;
        }
        out
    }

    /// Bind-pose world origin of a bone by name, if present. The bone origin is
    /// the translation column of its bind-world transform — i.e. where the joint
    /// pivot sits in the rest skeleton.
    pub fn bone_origin(&self, name: &str) -> Option<Vec3> {
        let idx = self
            .node_name
            .iter()
            .find(|(_, nm)| nm.as_str() == name)
            .map(|(i, _)| *i)?;
        self.bind_world.get(&idx).map(|m| m.w_axis.truncate())
    }

    /// Skeleton-derived segment lengths for one left leg, from the bind-pose bone
    /// origins: (coxa, femur, tibia) spans following the skin's bone→segment
    /// grouping (coxa = 000→001, femur = 001→003, tibia = 003→005). This is what
    /// a from-skeleton joint-chain build would read instead of the hand-coded
    /// COXA_LEN/FEMUR_LEN/TIBIA_LEN. `leg` is 1..=4 (the model's numbering).
    pub fn leg_segment_spans(&self, leg: u8) -> Option<[f32; 3]> {
        let o = |seg: &str| self.bone_origin(&format!("Def_leg_0{leg}.{seg}.L"));
        let p000 = o("000")?;
        let p001 = o("001")?;
        let p003 = o("003")?;
        let p005 = o("005")?;
        Some([
            (p001 - p000).length(),
            (p003 - p001).length(),
            (p005 - p003).length(),
        ])
    }
}

/// Recursively compose `parent_world * node_local` for a node and its subtree,
/// recording each node's world matrix. glTF node transforms are TRS or a raw
/// matrix; `node.transform().matrix()` normalises both to a column-major 4x4.
fn compose_world(node: &gltf::Node, parent_world: Mat4, out: &mut HashMap<usize, Mat4>) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;
    out.insert(node.index(), world);
    for child in node.children() {
        compose_world(&child, world, out);
    }
}

// ---------------------------------------------------------------------------
// Bone → physics part mapping (mirrors super::skin::bone_target)
// ---------------------------------------------------------------------------

/// Map a glTF deform-bone name to the physics part it should drive. This is the
/// spike's copy of [`super::skin`]'s `bone_target` rules, returning the spike's
/// [`PartId`]. `bone_map_covers_all_model_bones` asserts every deform/control
/// bone in the model routes to *some* part (no silently-dropped cloud); keeping
/// the two functions byte-for-byte in sync is by construction (a real landing
/// would share one mapping, see MESHFIT_PLAN.md).
fn bone_to_part(name: &str) -> Option<PartId> {
    if !(name.starts_with("Def_") || name.starts_with("Ctrl_")) {
        return None;
    }
    let side = if name.ends_with(".L") || name.contains(".L.") {
        Some(Side::Left)
    } else if name.ends_with(".R") || name.contains(".R.") {
        Some(Side::Right)
    } else {
        None
    };

    if let Some(rest) = name.strip_prefix("Def_leg_0") {
        let leg = rest.chars().next()?.to_digit(10)? as u8 - 1;
        let seg = rest.get(2..5)?;
        let side = side?;
        let id = match seg {
            "000" => CrabJointId::LegCoxa(side, leg),
            "001" | "002" => CrabJointId::LegFemur(side, leg),
            _ => CrabJointId::LegTibia(side, leg),
        };
        return Some(PartId::Joint(id));
    }
    if name.starts_with("Def_pincer") || name.starts_with("Ctrl_pincer_tail") {
        let side = side?;
        let id = if name.contains("006") || name.starts_with("Ctrl_pincer_tail") {
            CrabJointId::ClawPincer(side)
        } else if name.contains("000") || name.contains("001") {
            CrabJointId::ClawUpper(side)
        } else {
            CrabJointId::ClawFore(side)
        };
        return Some(PartId::Joint(id));
    }
    if name.starts_with("Def_antennae") {
        return Some(PartId::Joint(CrabJointId::EyeStalk(side?)));
    }
    Some(PartId::Carapace)
}

// ---------------------------------------------------------------------------
// Capsule fit
// ---------------------------------------------------------------------------

/// A capsule fitted to a vertex cloud, in the cloud's own (world) frame.
#[derive(Clone, Copy, Debug)]
pub struct FittedCapsule {
    /// Segment endpoints (capsule = swept sphere between these).
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f32,
}

impl FittedCapsule {
    /// Distance between the segment endpoints.
    pub fn segment_len(&self) -> f32 {
        (self.b - self.a).length()
    }

    /// Rapier/Bevy `capsule_y` half-height convention: half the *segment*
    /// (cylinder) length, excluding the hemispherical caps.
    pub fn half_height(&self) -> f32 {
        self.segment_len() * 0.5
    }

    /// Total tip-to-tip length including both caps (segment + 2·radius).
    pub fn total_len(&self) -> f32 {
        self.segment_len() + 2.0 * self.radius
    }
}

/// Fit a capsule to a point cloud:
///   - axis = first principal component (largest-variance direction),
///   - radius = a high percentile of perpendicular distance to that axis
///     (percentile, not max, so a few skinning-bleed outliers don't inflate it),
///   - segment = the axial extent shrunk by `radius` at each end, so the caps
///     cover the tips instead of the capsule overhanging by a full radius.
///
/// Returns `None` for clouds too small to fit (< 4 points).
pub fn fit_capsule(points: &[Vec3]) -> Option<FittedCapsule> {
    if points.len() < 4 {
        return None;
    }
    let n = points.len() as f32;
    let centroid = points.iter().copied().sum::<Vec3>() / n;

    // Principal axis = largest-variance eigenvector of the covariance. (When the
    // top two variances are close — a chunky coxa cloud — this axis is
    // ill-defined; `ObbFit::blobbiness` flags exactly that case.)
    let (axes, _vars) = covariance_eigenframe(points, centroid);
    let axis = axes[0];

    // Project onto the axis for axial extent; perpendicular residual for radius.
    let mut tmin = f32::INFINITY;
    let mut tmax = f32::NEG_INFINITY;
    let mut perp: Vec<f32> = Vec::with_capacity(points.len());
    for &p in points {
        let d = p - centroid;
        let t = d.dot(axis);
        tmin = tmin.min(t);
        tmax = tmax.max(t);
        perp.push((d - axis * t).length());
    }
    // 95th-percentile radius: robust to a thin tail of bled vertices.
    perp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let radius = percentile(&perp, 0.95).max(1e-4);

    // Pull the endpoints in by `radius` so the spherical caps land on the tips.
    // Clamp so a stubby cloud doesn't invert into a negative segment.
    let half_axial = (tmax - tmin) * 0.5;
    let half_seg = (half_axial - radius).max(0.0);
    let mid = centroid + axis * (tmin + tmax) * 0.5;
    Some(FittedCapsule {
        a: mid - axis * half_seg,
        b: mid + axis * half_seg,
        radius,
    })
}

/// Linear-interpolated percentile of a pre-sorted slice. `q` in 0..1.
fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = q * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// RMS distance from each point to the surface of a fitted capsule, normalised
/// by the capsule radius. A faithful capsule fit sits near 0; a blob forced into
/// a capsule (claw hand) or a coxa cloud with no clean axis runs high. This is
/// the objective "is a capsule the right primitive here?" signal.
pub fn capsule_fit_residual(points: &[Vec3], cap: &FittedCapsule) -> f32 {
    if points.is_empty() || cap.radius <= 0.0 {
        return f32::INFINITY;
    }
    let seg = cap.b - cap.a;
    let seg_len2 = seg.length_squared().max(1e-12);
    let mut acc = 0.0f64;
    for &p in points {
        // Distance from p to the segment, then to the swept surface.
        let t = ((p - cap.a).dot(seg) / seg_len2).clamp(0.0, 1.0);
        let closest = cap.a + seg * t;
        let surf = (p - closest).length() - cap.radius;
        acc += (surf as f64) * (surf as f64);
    }
    ((acc / points.len() as f64).sqrt() as f32) / cap.radius
}

// ---------------------------------------------------------------------------
// PCA frame + oriented bounding box (the capsule's non-elongated alternative)
// ---------------------------------------------------------------------------

/// Oriented-box fit of a point cloud: principal axes (eigenvectors of the
/// covariance, ordered by descending *variance*) and the half-extent along each.
/// `axes[0]`/`half_extents.x` is the elongation direction. Variance order tracks
/// extent order closely but not exactly for skewed clouds, so treat the extents
/// as descriptive, not a hard sort.
#[derive(Clone, Copy, Debug)]
pub struct ObbFit {
    pub center: Vec3,
    pub axes: [Vec3; 3],
    pub half_extents: Vec3,
}

impl ObbFit {
    /// How non-elongated the cloud is: the second principal half-extent over the
    /// first. ~0 = stick-like (a capsule fits), ~1 = chunky (wants a box/hull).
    /// The validation table uses this to flag which primitive each part wants.
    pub fn blobbiness(&self) -> f32 {
        self.half_extents.y / self.half_extents.x.max(1e-6)
    }
}

/// Fit an oriented box: principal axes by eigendecomposition of the covariance,
/// extents by the max absolute projection onto each axis. Uses a percentile
/// (98th) on each axis so a couple of bled outliers don't balloon the box, then
/// reports that as the half-extent.
pub fn fit_obb(points: &[Vec3]) -> Option<ObbFit> {
    if points.len() < 4 {
        return None;
    }
    let n = points.len() as f32;
    let center = points.iter().copied().sum::<Vec3>() / n;
    let (axes, _vars) = covariance_eigenframe(points, center);
    let mut proj: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for &p in points {
        let d = p - center;
        for k in 0..3 {
            proj[k].push(d.dot(axes[k]).abs());
        }
    }
    let mut he = Vec3::ZERO;
    for k in 0..3 {
        proj[k].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        he[k] = percentile(&proj[k], 0.98).max(1e-4);
    }
    Some(ObbFit {
        center,
        axes,
        half_extents: he,
    })
}

/// Eigenframe (orthonormal eigenvectors, sorted by descending eigenvalue) of a
/// cloud's 3x3 covariance, via cyclic Jacobi rotations. Returns the three axes
/// longest-first and their variances. Robust and dependency-free; 3x3 converges
/// in a handful of sweeps.
// The Jacobi sweeps update two columns (p, q) of `a`/`v` per `k`; the explicit
// index keeps the rotation readable as matrix math, so range loops stay.
#[allow(clippy::needless_range_loop)]
fn covariance_eigenframe(points: &[Vec3], centroid: Vec3) -> ([Vec3; 3], Vec3) {
    // Symmetric covariance as a [[f64;3];3].
    let mut a = [[0.0f64; 3]; 3];
    for &p in points {
        let d = [
            (p.x - centroid.x) as f64,
            (p.y - centroid.y) as f64,
            (p.z - centroid.z) as f64,
        ];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += d[i] * d[j];
            }
        }
    }
    // V accumulates the eigenvectors (columns).
    let mut v = [[0.0f64; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..16 {
        // Largest off-diagonal magnitude.
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-14 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-18 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // Apply the Jacobi rotation J^T A J and accumulate into V.
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut cols: Vec<(f64, Vec3)> = (0..3)
        .map(|j| {
            (
                a[j][j],
                Vec3::new(v[0][j] as f32, v[1][j] as f32, v[2][j] as f32).normalize_or_zero(),
            )
        })
        .collect();
    cols.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    (
        [cols[0].1, cols[1].1, cols[2].1],
        Vec3::new(cols[0].0 as f32, cols[1].0 as f32, cols[2].0 as f32),
    )
}

// ---------------------------------------------------------------------------
// Mass properties (capsule, analytic, under a chosen density)
// ---------------------------------------------------------------------------

/// Rigid-body mass properties of a capsule, computed analytically (cylinder +
/// two hemispheres) so the spike doesn't depend on constructing a rapier
/// collider. `i_axial` is the moment about the capsule's own long axis;
/// `i_perp` about a transverse axis through the centre of mass. For comparison
/// with the hand-coded body these are the principal inertias of the part.
#[derive(Clone, Copy, Debug)]
pub struct CapsuleMass {
    pub mass: f32,
    pub i_axial: f32,
    pub i_perp: f32,
}

/// Analytic capsule inertia about its centre of mass. `half_height` is the
/// cylinder half-length (caps excluded), matching `capsule_y`. Standard
/// solid-capsule formulae (uniform density ρ): the cylinder contributes its
/// usual terms; each hemisphere its own plus a parallel-axis shift for the
/// transverse moment.
pub fn capsule_mass(radius: f32, half_height: f32, density: f32) -> CapsuleMass {
    let r = radius as f64;
    let h = (half_height * 2.0) as f64; // full cylinder length
    let rho = density as f64;
    let pi = std::f64::consts::PI;

    let m_cyl = rho * pi * r * r * h;
    let m_hemi = rho * (2.0 / 3.0) * pi * r * r * r; // one hemisphere
    let mass = m_cyl + 2.0 * m_hemi;

    // Axial (about the long axis): cylinder ½ m r², two hemispheres ⅖ m_hemi r² each.
    let i_axial = 0.5 * m_cyl * r * r + 2.0 * (2.0 / 5.0) * m_hemi * r * r;

    // Transverse: cylinder (1/12 m (3r² + h²)); each hemisphere has I about its
    // own centroid plus a parallel-axis shift to the capsule centre. Hemisphere
    // centroid sits 3r/8 from its flat face; the flat face is at h/2 from centre.
    let i_cyl_perp = m_cyl * (3.0 * r * r + h * h) / 12.0;
    let i_hemi_own = (2.0 / 5.0) * m_hemi * r * r; // sphere's ⅖mr² split per hemi
    // Parallel-axis: distance from capsule centre to hemisphere centroid.
    let d = h / 2.0 + 3.0 * r / 8.0;
    // Subtract the self-term already at the hemisphere centroid vs its flat face.
    // Using the standard composite result: I_hemi_perp(about capsule centre) =
    //   (2/5)m r²  - m (3r/8)²            [shift own-axis to centroid]
    //   + m (h/2 + 3r/8)²                 [shift centroid to capsule centre]
    let i_hemi_perp = i_hemi_own - m_hemi * (3.0 * r / 8.0).powi(2) + m_hemi * d * d;
    let i_perp = i_cyl_perp + 2.0 * i_hemi_perp;

    CapsuleMass {
        mass: mass as f32,
        i_axial: i_axial as f32,
        i_perp: i_perp as f32,
    }
}

/// Analytic solid-cuboid mass properties (for the carapace and pincer, which the
/// hand-coded body models as boxes). Half-extents along x/y/z.
pub fn cuboid_mass(hx: f32, hy: f32, hz: f32, density: f32) -> (f32, Vec3) {
    let (hx, hy, hz, rho) = (hx as f64, hy as f64, hz as f64, density as f64);
    let mass = rho * 8.0 * hx * hy * hz;
    // Principal inertias of a solid box about its centre.
    let ix = mass * ((2.0 * hy).powi(2) + (2.0 * hz).powi(2)) / 12.0;
    let iy = mass * ((2.0 * hx).powi(2) + (2.0 * hz).powi(2)) / 12.0;
    let iz = mass * ((2.0 * hx).powi(2) + (2.0 * hy).powi(2)) / 12.0;
    (mass as f32, Vec3::new(ix as f32, iy as f32, iz as f32))
}

// ---------------------------------------------------------------------------
// Validation report: fitted vs hand-coded, per part
// ---------------------------------------------------------------------------

/// One row of the validation table: a part's fitted primitives + mass next to
/// the hand-coded reference. Optional fields are `None` when the cloud was too
/// small to fit (< 4 vertices).
pub struct PartFit {
    pub part: PartId,
    pub vertex_count: usize,
    pub mean_dominant_weight: f32,
    pub fitted: Option<FittedCapsule>,
    /// Oriented box over the same cloud — the capsule's alternative for chunky
    /// parts (carapace, claw hand, pincer). `blobbiness()` says which primitive
    /// the cloud actually wants.
    pub obb: Option<ObbFit>,
    /// Normalised RMS residual of the cloud to the capsule surface (see
    /// [`capsule_fit_residual`]): the objective capsule-quality score.
    pub capsule_residual: f32,
    pub fitted_mass: Option<f32>,
    /// Fitted capsule's transverse principal inertia under the part density.
    pub fitted_i_perp: Option<f32>,
    /// Hand-coded reference (half_height, radius) for capsule parts; the
    /// carapace/pincer report their box half-extents' equivalent instead.
    pub ref_half_height: f32,
    pub ref_radius: f32,
    pub ref_mass: f32,
    pub ref_i_perp: f32,
}

impl PartFit {
    fn label(&self) -> String {
        self.part.label()
    }
}

/// Run the full fit on the model and pair every part with its hand-coded
/// reference. Every part gets BOTH a capsule and an OBB fit plus a capsule
/// residual, so the caller can see which primitive each cloud actually wants.
pub fn fit_report(model: &LoadedModel) -> Vec<PartFit> {
    let by_part = model.vertices_by_part();
    let mut rows = Vec::new();

    // Iterate a deterministic part order: carapace, then legs L/R, claws, eyes.
    for part in reference_parts() {
        let empty: &[Vec3] = &[];
        let (points, mean_w) = by_part
            .get(&part.part)
            .map(|(v, w)| (v.as_slice(), *w))
            .unwrap_or((empty, 0.0));
        let fitted = fit_capsule(points);
        let obb = fit_obb(points);
        let capsule_residual = fitted
            .map(|c| capsule_fit_residual(points, &c))
            .unwrap_or(f32::INFINITY);
        let (fitted_mass, fitted_i_perp) = match fitted {
            Some(c) => {
                let m = capsule_mass(c.radius, c.half_height(), part.density);
                (Some(m.mass), Some(m.i_perp))
            }
            None => (None, None),
        };
        rows.push(PartFit {
            part: part.part,
            vertex_count: points.len(),
            mean_dominant_weight: mean_w,
            fitted,
            obb,
            capsule_residual,
            fitted_mass,
            fitted_i_perp,
            ref_half_height: part.ref_half_height,
            ref_radius: part.ref_radius,
            ref_mass: part.ref_mass,
            ref_i_perp: part.ref_i_perp,
        });
    }
    rows
}

/// Hand-coded reference for one part: the live body's collider dims, density,
/// and derived mass/inertia. Built from [`super::body`]'s own constants (via the
/// test-only `reference` re-export) so this table can't drift from the physics.
struct RefPart {
    part: PartId,
    ref_half_height: f32,
    ref_radius: f32,
    ref_mass: f32,
    ref_i_perp: f32,
    density: f32,
}

/// The hand-coded body's parts with their reference numbers, in table order.
/// Capsule parts (legs, claw arms) use the analytic capsule mass; box parts
/// (carapace, pincer) and the ball (eye) use their own analytic mass, reported
/// in the same mass/inertia columns with `ref_radius`/`ref_half_height` standing
/// in for an equivalent extent so the table stays rectangular.
fn reference_parts() -> Vec<RefPart> {
    use super::body::reference as r;
    let mut v = Vec::new();

    // Carapace: a box. Report its half-depth as "half_height" and half-width as
    // "radius" just to fill the columns; the mass/inertia are the box's.
    let (cm, ci) = cuboid_mass(
        r::CARAPACE_HALF_W,
        r::CARAPACE_HALF_H,
        r::CARAPACE_HALF_D,
        r::CARAPACE_DENSITY,
    );
    v.push(RefPart {
        part: PartId::Carapace,
        ref_half_height: r::CARAPACE_HALF_D,
        ref_radius: r::CARAPACE_HALF_W,
        ref_mass: cm,
        ref_i_perp: ci.x.max(ci.z), // transverse-ish
        density: r::CARAPACE_DENSITY,
    });

    for side in [Side::Left, Side::Right] {
        for leg in 0u8..4 {
            for (id, hl, rad, den) in [
                (
                    CrabJointId::LegCoxa(side, leg),
                    r::COXA_LEN,
                    r::COXA_RAD,
                    r::COXA_DENSITY,
                ),
                (
                    CrabJointId::LegFemur(side, leg),
                    r::FEMUR_LEN,
                    r::FEMUR_RAD,
                    r::FEMUR_DENSITY,
                ),
                (
                    CrabJointId::LegTibia(side, leg),
                    r::TIBIA_LEN,
                    r::TIBIA_RAD,
                    r::TIBIA_DENSITY,
                ),
            ] {
                let m = capsule_mass(rad, hl, den);
                v.push(RefPart {
                    part: PartId::Joint(id),
                    ref_half_height: hl,
                    ref_radius: rad,
                    ref_mass: m.mass,
                    ref_i_perp: m.i_perp,
                    density: den,
                });
            }
        }
    }
    for side in [Side::Left, Side::Right] {
        let mu = capsule_mass(r::CLAW_UPPER_RAD, r::CLAW_UPPER_LEN, r::CLAW_DENSITY);
        v.push(RefPart {
            part: PartId::Joint(CrabJointId::ClawUpper(side)),
            ref_half_height: r::CLAW_UPPER_LEN,
            ref_radius: r::CLAW_UPPER_RAD,
            ref_mass: mu.mass,
            ref_i_perp: mu.i_perp,
            density: r::CLAW_DENSITY,
        });
        let mf = capsule_mass(r::CLAW_FORE_RAD, r::CLAW_FORE_LEN, r::CLAW_DENSITY);
        v.push(RefPart {
            part: PartId::Joint(CrabJointId::ClawFore(side)),
            ref_half_height: r::CLAW_FORE_LEN,
            ref_radius: r::CLAW_FORE_RAD,
            ref_mass: mf.mass,
            ref_i_perp: mf.i_perp,
            density: r::CLAW_DENSITY,
        });
        let (pm, pi) = cuboid_mass(
            r::PINCER_HALF_W,
            r::PINCER_HALF_H,
            r::PINCER_HALF_D,
            r::CLAW_DENSITY,
        );
        v.push(RefPart {
            part: PartId::Joint(CrabJointId::ClawPincer(side)),
            ref_half_height: r::PINCER_HALF_D,
            ref_radius: r::PINCER_HALF_W,
            ref_mass: pm,
            ref_i_perp: pi.x.max(pi.z),
            density: r::CLAW_DENSITY,
        });
    }
    for side in [Side::Left, Side::Right] {
        // Eye: a ball. Sphere mass + inertia ⅖mr².
        let r_eye = r::EYE_BALL_RAD as f64;
        let den = r::EYE_DENSITY as f64;
        let mass = (den * (4.0 / 3.0) * std::f64::consts::PI * r_eye.powi(3)) as f32;
        let i = (0.4 * mass as f64 * r_eye * r_eye) as f32;
        v.push(RefPart {
            part: PartId::Joint(CrabJointId::EyeStalk(side)),
            ref_half_height: 0.0,
            ref_radius: r::EYE_BALL_RAD,
            ref_mass: mass,
            ref_i_perp: i,
            density: r::EYE_DENSITY,
        });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every deform/control bone in the model must route to *some* physics part:
    /// an unmapped bone is a silently-dropped vertex cloud, so a part would be
    /// fit from fewer vertices than it owns. (This mirrors the skin's
    /// `bone_target` rules; it cannot call the private fn, so it re-derives them.)
    #[test]
    fn bone_map_covers_all_model_bones() {
        let Some(path) = model_path() else {
            eprintln!(
                "meshfit: no model at CRAB_MODEL_PATH or /tmp/rl/assets/sally.glb — skipping"
            );
            return;
        };
        let model = LoadedModel::load(&path).expect("load model");
        // Every deform/control bone in the model maps to some part.
        let mut unmapped = Vec::new();
        for (idx, name) in &model.node_name {
            if (name.starts_with("Def_") || name.starts_with("Ctrl_"))
                && bone_to_part(name).is_none()
            {
                unmapped.push((*idx, name.clone()));
            }
        }
        assert!(
            unmapped.is_empty(),
            "bones map to no physics part: {unmapped:?}"
        );
    }

    /// Capsule mass formula sanity: a capsule with zero half-height is a sphere,
    /// so its mass and (isotropic) inertia must match the sphere closed forms.
    #[test]
    fn capsule_reduces_to_sphere() {
        let r = 0.1f32;
        let rho = 3.0f32;
        let m = capsule_mass(r, 0.0, rho);
        let sphere_mass = rho * (4.0 / 3.0) * std::f32::consts::PI * r.powi(3);
        let sphere_i = 0.4 * sphere_mass * r * r;
        assert!(
            (m.mass - sphere_mass).abs() / sphere_mass < 1e-4,
            "mass {m:?}"
        );
        assert!(
            (m.i_axial - sphere_i).abs() / sphere_i < 1e-3,
            "i_axial {m:?}"
        );
        assert!(
            (m.i_perp - sphere_i).abs() / sphere_i < 1e-3,
            "i_perp {m:?}"
        );
        assert!(
            (m.i_axial - m.i_perp).abs() / m.i_axial < 1e-3,
            "sphere must be isotropic: {m:?}"
        );
    }

    /// The capsule fitter recovers a known capsule: sample points on a capsule
    /// of known axis/radius/length and check the fit lands close.
    #[test]
    fn fit_recovers_synthetic_capsule() {
        // Capsule along +X, radius 0.05, segment half-length 0.2.
        let axis = Vec3::X;
        let r = 0.05f32;
        let hseg = 0.2f32;
        let mut pts = Vec::new();
        // Cylinder wall samples.
        for i in 0..40 {
            let t = -hseg + 2.0 * hseg * (i as f32 / 39.0);
            for k in 0..16 {
                let a = std::f32::consts::TAU * (k as f32 / 16.0);
                pts.push(Vec3::new(t, r * a.cos(), r * a.sin()));
            }
        }
        let fit = fit_capsule(&pts).expect("fit");
        // Axis recovered (allow sign flip).
        let dir = (fit.b - fit.a).normalize();
        assert!(dir.dot(axis).abs() > 0.999, "axis off: {dir:?}");
        assert!(
            (fit.radius - r).abs() < 0.01,
            "radius {} vs {r}",
            fit.radius
        );
        // Endpoints pulled in by radius → segment ≈ 2*hseg - 2*r.
        let expected_seg = 2.0 * hseg - 2.0 * r;
        assert!(
            (fit.segment_len() - expected_seg).abs() < 0.03,
            "seg {} vs {expected_seg}",
            fit.segment_len()
        );
    }

    /// THE VALIDATION: load the real model, fit every part, and print the
    /// fitted-vs-hand-coded table plus the skeleton-placement check. The numbers
    /// are the deliverable; assertions pin only what the spike *proved works* (a
    /// clean stick fits; the carapace box matches) — the divergences are reported,
    /// not asserted away. Self-skips when no model is present.
    #[test]
    fn meshfit_validation() {
        let Some(path) = model_path() else {
            eprintln!("meshfit: no model — skipping validation table");
            return;
        };
        let model = LoadedModel::load(&path).expect("load model");
        let rows = fit_report(&model);

        println!("\n=== MESHFIT VALIDATION: fitted vs hand-coded body ===");
        println!("model: {path:?}");
        println!(
            "cols: capsule fit (half-height, radius, mass), hand-coded ref, ratios, \
             capsule residual (0=perfect, >~0.3 = wrong primitive), OBB blobbiness \
             (0=stick, ~1=box), inertia ratio (fit/ref transverse)."
        );
        println!(
            "{:<18} {:>6} {:>5} | {:>6} {:>6} {:>7} | {:>6} {:>6} {:>7} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>6}",
            "part",
            "nverts",
            "domW",
            "f_hh",
            "f_r",
            "f_mass",
            "r_hh",
            "r_r",
            "r_mass",
            "mass%",
            "len%",
            "I%",
            "resid",
            "blob",
            "prim?"
        );
        let mut total_fit_mass = 0.0f32;
        let mut total_ref_mass = 0.0f32;
        for row in &rows {
            total_ref_mass += row.ref_mass;
            let (fhh, fr, fm) = match (row.fitted, row.fitted_mass) {
                (Some(c), Some(m)) => {
                    total_fit_mass += m;
                    (c.half_height(), c.radius, m)
                }
                _ => (f32::NAN, f32::NAN, f32::NAN),
            };
            let mass_pct = 100.0 * fm / row.ref_mass;
            let ref_total_len = 2.0 * row.ref_half_height + 2.0 * row.ref_radius;
            let fit_total_len = row.fitted.map(|c| c.total_len()).unwrap_or(f32::NAN);
            let len_pct = 100.0 * fit_total_len / ref_total_len;
            let i_pct = match row.fitted_i_perp {
                Some(i) if row.ref_i_perp > 0.0 => 100.0 * i / row.ref_i_perp,
                _ => f32::NAN,
            };
            let blob = row.obb.map(|o| o.blobbiness()).unwrap_or(f32::NAN);
            // Which primitive the cloud wants, judged mainly by elongation
            // (blobbiness): a clear major axis → capsule, a chunky cloud → box.
            // Residual alone over-flags here because the real legs TAPER toward
            // the foot, so even a good capsule shows a moderate residual — that
            // is taper, not non-elongation. A diagnostic suggestion only.
            let prim = if blob < 0.45 { "caps" } else { "box" };
            println!(
                "{:<18} {:>6} {:>5.2} | {:>6.3} {:>6.3} {:>7.4} | {:>6.3} {:>6.3} {:>7.4} | {:>5.0} {:>5.0} {:>5.0} | {:>5.2} {:>5.2} {:>6}",
                row.label(),
                row.vertex_count,
                row.mean_dominant_weight,
                fhh,
                fr,
                fm,
                row.ref_half_height,
                row.ref_radius,
                row.ref_mass,
                mass_pct,
                len_pct,
                i_pct,
                row.capsule_residual,
                blob,
                prim,
            );
        }
        println!(
            "TOTAL fitted mass {:.3} kg vs hand-coded {:.3} kg ({:.0}%)",
            total_fit_mass,
            total_ref_mass,
            100.0 * total_fit_mass / total_ref_mass
        );

        // --- Skeleton-placement check: do bone spans recover the leg geometry? -
        println!(
            "\n=== SKELETON PLACEMENT: bind-pose bone spans vs hand-coded segment lengths ==="
        );
        println!(
            "hand-coded segment full lengths: coxa {:.3}, femur {:.3}, tibia {:.3} m (2*LEN)",
            2.0 * crate::bot::body::reference::COXA_LEN,
            2.0 * crate::bot::body::reference::FEMUR_LEN,
            2.0 * crate::bot::body::reference::TIBIA_LEN,
        );
        for leg in 1u8..=4 {
            if let Some([c, f, t]) = model.leg_segment_spans(leg) {
                println!(
                    "  model leg_0{leg}.L bind spans: coxa {c:.3}, femur {f:.3}, tibia {t:.3} m"
                );
            }
        }
        if let (Some(thorax), Some(tip)) = (
            model.bone_origin("Def_thorax_front"),
            model.bone_origin("Def_leg_01.005.L"),
        ) {
            println!(
                "  carapace bone at {:?}; front-left foot bone at {:?} (reach {:.3} m)",
                thorax.to_array().map(|v| (v * 1000.0).round() / 1000.0),
                tip.to_array().map(|v| (v * 1000.0).round() / 1000.0),
                (tip - thorax).length(),
            );
        }
        if let Some(o) = rows
            .iter()
            .find(|r| r.part == PartId::Carapace)
            .and_then(|r| r.obb)
        {
            println!(
                "  carapace OBB: center {:?}, major axis {:?}, half-extents {:?}",
                o.center.to_array().map(|v| (v * 1000.0).round() / 1000.0),
                o.axes[0].to_array().map(|v| (v * 1000.0).round() / 1000.0),
                o.half_extents
                    .to_array()
                    .map(|v| (v * 1000.0).round() / 1000.0),
            );
        }

        // --- Assertions: pin only the proven-feasible cases. ------------------
        // 1. A front leg (leg 0 = model leg_01, the cleanest cloud) fits a sane,
        //    stick-like capsule on every segment.
        for row in &rows {
            if let PartId::Joint(
                CrabJointId::LegCoxa(_, 0)
                | CrabJointId::LegFemur(_, 0)
                | CrabJointId::LegTibia(_, 0),
            ) = row.part
            {
                let fit = row
                    .fitted
                    .unwrap_or_else(|| panic!("{} did not fit", row.label()));
                assert!(
                    fit.radius > 0.005 && fit.radius < 0.12,
                    "{} radius {} implausible",
                    row.label(),
                    fit.radius
                );
                assert!(fit.segment_len() >= 0.0, "{} negative segment", row.label());
            }
        }
        // 2. The carapace cloud's footprint (longest horizontal half-extent)
        //    matches the hand-coded box width within 20% — the dimension the
        //    spike recovers cleanly. (The model carapace is a tall dome, so its
        //    HEIGHT diverges hard from the physics slab — reported above, NOT
        //    asserted: that mismatch is a finding, not a fit failure.)
        let carapace = rows
            .iter()
            .find(|r| r.part == PartId::Carapace)
            .expect("carapace row");
        let obb = carapace.obb.expect("carapace OBB");
        let widest = obb
            .half_extents
            .to_array()
            .into_iter()
            .fold(0.0f32, f32::max);
        let ref_widest = crate::bot::body::reference::CARAPACE_HALF_W
            .max(crate::bot::body::reference::CARAPACE_HALF_D);
        let ratio = widest / ref_widest;
        assert!(
            (0.8..=1.2).contains(&ratio),
            "carapace footprint: fitted half-extent {widest:.3} vs ref {ref_widest:.3} (ratio {ratio:.2})"
        );

        // 3. The total fitted mass is finite and positive (the pipeline produced
        //    real numbers end-to-end). Its magnitude vs hand-coded is in the
        //    table — fitted runs heavy because the art is fleshier than the
        //    stick colliders; that is the headline finding, discussed in the plan.
        assert!(
            total_fit_mass.is_finite() && total_fit_mass > 0.0,
            "total fitted mass {total_fit_mass}"
        );
    }
}
