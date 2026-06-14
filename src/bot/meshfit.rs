//! Spike → phase 1: auto-derive physics colliders from the skinned glTF mesh.
//!
//! This is a *prototype/validation* module, not part of the live physics. It
//! loads the same `sally.glb` the cosmetic skin uses ([`super::skin`]), then for
//! each physics body part:
//!   1. gathers the mesh vertices whose dominant skin bone maps to that part
//!      (same bone→part mapping the skin drives with),
//!   2. **chooses a collider primitive from the cloud's shape** — capsule, box,
//!      or ball — via [`choose_primitive`], instead of forcing a capsule on
//!      everything (the phase-0 spike's limitation), and
//!   3. computes that primitive's mass properties under the hand-coded density.
//!
//! Phase 1 adds, over the spike: data-driven primitive selection (elongation /
//! isotropy / flatness descriptors + surface residual), leg-taper handling (a
//! coverage-balanced capsule radius plus a tapered-cone residual that isolates
//! taper from non-stickness), and an explicit fix for the degenerate middle-coxa
//! blobs (routed to a box/ball, not a blown-up capsule).
//!
//! The point is the *comparison* against the hand-coded body in [`super::body`]:
//! [`fit_report`] returns one [`PartFit`] per part with the chosen primitive, its
//! residual, and both the fitted and hand-coded numbers; the `meshfit_validation`
//! integration-style test prints the table and asserts the fit quality. Nothing
//! here is wired into `spawn_crab`; see `MESHFIT_PLAN.md` for landing it for real
//! (phase 2: a typed `FittedBody` bake consuming exactly these choices).
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
// Taper: the legs thin toward the foot, so a constant-radius capsule cannot
// hug both ends. We measure the taper and report two things the spike's single
// p95 capsule could not: (1) the coverage-BALANCED constant radius that
// minimises the surface residual rather than enveloping every vertex, and
// (2) the residual a TAPERED cone (which rapier cannot represent) would leave —
// the floor that isolates "is this a stick?" from "does it taper?".
// ---------------------------------------------------------------------------

/// A cloud's taper along a capsule axis: the perpendicular radius at the `a` end
/// vs the `b` end, from a least-squares line through per-station radii.
#[derive(Clone, Copy, Debug)]
pub struct TaperFit {
    /// Fitted cone radius at the segment's `a` endpoint.
    pub r_a: f32,
    /// Fitted cone radius at the `b` endpoint.
    pub r_b: f32,
}

impl TaperFit {
    /// Thin-end / fat-end radius ratio in `0..=1`. 1.0 = no taper (a true
    /// cylinder); 0.5 = the thin end is half the fat end. The legs sit ~0.3–0.5.
    pub fn taper_ratio(&self) -> f32 {
        let (lo, hi) = (self.r_a.min(self.r_b), self.r_a.max(self.r_b));
        lo / hi.max(1e-6)
    }
}

/// Fit a linear radius profile r(t) = r_a + (r_b − r_a)·t along a capsule's axis
/// (t in 0..1 from `a` to `b`), so the cloud is modelled as a truncated cone
/// instead of a cylinder. Each axial bin contributes a high-percentile
/// perpendicular distance (its local cone radius); a least-squares line through
/// those bin radii gives the end radii. Caps are excluded by only binning points
/// whose projection lands on the segment interior.
fn fit_taper(points: &[Vec3], cap: &FittedCapsule) -> Option<TaperFit> {
    let seg = cap.b - cap.a;
    let seg_len2 = seg.length_squared();
    if seg_len2 < 1e-12 || points.len() < 16 {
        return None;
    }
    const BINS: usize = 8;
    let mut binned: [Vec<f32>; BINS] = Default::default();
    for &p in points {
        let t = (p - cap.a).dot(seg) / seg_len2;
        if !(0.0..=1.0).contains(&t) {
            continue; // cap region: its perpendicular spread isn't the cone radius
        }
        let closest = cap.a + seg * t;
        let perp = (p - closest).length();
        let bin = ((t * BINS as f32) as usize).min(BINS - 1);
        binned[bin].push(perp);
    }
    // Each populated bin → (t at its centre, p90 perpendicular = local radius).
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (i, b) in binned.iter_mut().enumerate() {
        if b.len() < 4 {
            continue;
        }
        b.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        xs.push((i as f32 + 0.5) / BINS as f32);
        ys.push(percentile(b, 0.90));
    }
    if xs.len() < 3 {
        return None;
    }
    // Ordinary least squares: radius = m·t + c.
    let n = xs.len() as f32;
    let sx: f32 = xs.iter().sum();
    let sy: f32 = ys.iter().sum();
    let sxx: f32 = xs.iter().map(|x| x * x).sum();
    let sxy: f32 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-9 {
        return None;
    }
    let m = (n * sxy - sx * sy) / denom;
    let c = (sy - m * sx) / n;
    Some(TaperFit {
        r_a: (c).max(1e-4),
        r_b: (m + c).max(1e-4),
    })
}

/// RMS distance from the cloud to a *tapered cone* surface (radius r_a→r_b along
/// the axis), normalised by the mean cone radius — the same units as
/// [`capsule_fit_residual`] so the two compare directly. This is the residual a
/// shape that could taper would leave; a constant-radius capsule cannot beat it,
/// so the gap between this and the capsule residual is exactly the cost of
/// taper. Cap regions are excluded (a cone has flat ends, not hemispheres).
fn cone_fit_residual(points: &[Vec3], cap: &FittedCapsule, taper: &TaperFit) -> f32 {
    let seg = cap.b - cap.a;
    let seg_len2 = seg.length_squared();
    let rmean = 0.5 * (taper.r_a + taper.r_b);
    if seg_len2 < 1e-12 || rmean <= 0.0 {
        return f32::INFINITY;
    }
    let mut acc = 0.0f64;
    let mut count = 0u32;
    for &p in points {
        let t = (p - cap.a).dot(seg) / seg_len2;
        if !(0.0..=1.0).contains(&t) {
            continue;
        }
        let closest = cap.a + seg * t;
        let r_here = taper.r_a + (taper.r_b - taper.r_a) * t;
        let surf = (p - closest).length() - r_here;
        acc += (surf as f64) * (surf as f64);
        count += 1;
    }
    if count == 0 {
        return f32::INFINITY;
    }
    ((acc / count as f64).sqrt() as f32) / rmean
}

/// Re-fit a capsule's *radius* to minimise the surface residual instead of
/// enveloping the cloud. The L2-optimal constant radius is the mean
/// perpendicular distance (the residual `Σ(dist−r)²` is minimised at `r =
/// mean(dist)`); a trimmed mean drops the skin-bleed tail. The axis and segment
/// centre are kept from `cap`, but the endpoints are re-pulled by the new
/// radius. This is the honest answer to "rapier can't taper": pick the single
/// radius that balances over- and under-coverage rather than the p95 that only
/// ever over-covers.
fn rebalance_capsule_radius(points: &[Vec3], cap: &FittedCapsule) -> FittedCapsule {
    let seg = cap.b - cap.a;
    let seg_len2 = seg.length_squared().max(1e-12);
    let axis = seg / seg.length().max(1e-9);
    let mut perp: Vec<f32> = points
        .iter()
        .map(|&p| {
            let t = ((p - cap.a).dot(seg) / seg_len2).clamp(0.0, 1.0);
            (p - (cap.a + seg * t)).length()
        })
        .collect();
    perp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Trim the top 5% (bleed) before averaging; mean of the rest = L2-optimal r.
    let keep = ((perp.len() as f32 * 0.95) as usize).max(1);
    let radius = (perp[..keep].iter().sum::<f32>() / keep as f32).max(1e-4);
    // Re-centre on the original axial midpoint and re-pull endpoints by the new
    // radius so the caps still land near the tips.
    let mid = 0.5 * (cap.a + cap.b);
    let half_axial = 0.5 * cap.segment_len() + cap.radius; // original axial half-extent
    let half_seg = (half_axial - radius).max(0.0);
    FittedCapsule {
        a: mid - axis * half_seg,
        b: mid + axis * half_seg,
        radius,
    }
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
    /// Half-extents sorted descending (longest first), regardless of which axis
    /// carried the most *variance*. The covariance order tracks extent order only
    /// loosely for skewed clouds (the carapace's top-variance axis comes out
    /// near-vertical), so every shape descriptor reads from this, not the raw
    /// `half_extents`/axis order — otherwise blobbiness mislabels exactly the
    /// degenerate clouds it exists to catch.
    pub fn sorted_half_extents(&self) -> [f32; 3] {
        let mut e = self.half_extents.to_array();
        e.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        e
    }

    /// How non-elongated the cloud is: the *second* half-extent over the longest
    /// (both extent-sorted). ~0 = stick-like (a capsule fits), ~1 = chunky (wants
    /// a box/hull). The validation table uses this to flag which primitive each
    /// part wants.
    pub fn blobbiness(&self) -> f32 {
        let e = self.sorted_half_extents();
        e[1] / e[0].max(1e-6)
    }

    /// Isotropy: the *shortest* half-extent over the longest (extent-sorted).
    /// ~1 = a near-cube/ball with no dominant axis (the degenerate middle coxae);
    /// ~0 = a thin slab or stick. PCA can't axis-align an isotropic blob, so this
    /// is the signal that a capsule fit will go degenerate (radius blow-up).
    pub fn isotropy(&self) -> f32 {
        let e = self.sorted_half_extents();
        e[2] / e[0].max(1e-6)
    }

    /// Flatness: the shortest half-extent over the *middle* one (extent-sorted).
    /// ~0 = a flat slab (two long axes, one thin — the carapace dome footprint,
    /// the claw palm); ~1 = chunky in all three. Distinguishes a box-worthy slab
    /// from an isotropic ball.
    pub fn flatness(&self) -> f32 {
        let e = self.sorted_half_extents();
        e[2] / e[1].max(1e-6)
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

// ---------------------------------------------------------------------------
// Ball fit (for isotropic blobs with no usable axis)
// ---------------------------------------------------------------------------

/// A ball fitted to a point cloud: centroid + a covering radius.
#[derive(Clone, Copy, Debug)]
pub struct FittedBall {
    pub center: Vec3,
    pub radius: f32,
}

/// Fit a ball: centroid centre, radius = a high percentile of the radial
/// distance (p90, robust to bleed). For a genuinely isotropic blob this hugs
/// the cloud about as well as anything; for an elongated cloud it over-covers
/// (which is why selection only reaches for a ball when isotropy says there is
/// no axis to exploit).
pub fn fit_ball(points: &[Vec3]) -> Option<FittedBall> {
    if points.len() < 4 {
        return None;
    }
    let n = points.len() as f32;
    let center = points.iter().copied().sum::<Vec3>() / n;
    let mut rad: Vec<f32> = points.iter().map(|&p| (p - center).length()).collect();
    rad.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(FittedBall {
        center,
        radius: percentile(&rad, 0.90).max(1e-4),
    })
}

/// Normalised RMS residual of a cloud to a ball surface (same units as the
/// capsule/cone residuals, so primitive residuals compare on one scale).
fn ball_fit_residual(points: &[Vec3], ball: &FittedBall) -> f32 {
    if points.is_empty() || ball.radius <= 0.0 {
        return f32::INFINITY;
    }
    let mut acc = 0.0f64;
    for &p in points {
        let surf = (p - ball.center).length() - ball.radius;
        acc += (surf as f64) * (surf as f64);
    }
    ((acc / points.len() as f64).sqrt() as f32) / ball.radius
}

/// Normalised RMS residual of a cloud to an oriented-box surface. Distance to a
/// box is computed in the box's own frame; normalised by the mean half-extent so
/// it shares the capsule/ball residual scale.
fn obb_fit_residual(points: &[Vec3], obb: &ObbFit) -> f32 {
    let e = Vec3::from_array(obb.sorted_half_extents());
    let scale = (e.x + e.y + e.z) / 3.0;
    if points.is_empty() || scale <= 0.0 {
        return f32::INFINITY;
    }
    let mut acc = 0.0f64;
    for &p in points {
        let d = p - obb.center;
        // Signed distance to an axis-aligned box in the OBB frame, summed over
        // the three principal axes (exact outside, conservative inside).
        let local = Vec3::new(
            d.dot(obb.axes[0]).abs() - obb.half_extents[0],
            d.dot(obb.axes[1]).abs() - obb.half_extents[1],
            d.dot(obb.axes[2]).abs() - obb.half_extents[2],
        );
        let outside = local.max(Vec3::ZERO).length();
        let inside = local.x.max(local.y.max(local.z)).min(0.0);
        let surf = outside + inside;
        acc += (surf as f64) * (surf as f64);
    }
    ((acc / points.len() as f64).sqrt() as f32) / scale
}

// ---------------------------------------------------------------------------
// Data-driven primitive selection
// ---------------------------------------------------------------------------

/// The collider primitive chosen for a part, with the fitted shape inside. This
/// is the phase-1 output: instead of a blanket capsule, every cloud is matched
/// to the primitive its own geometry supports, decided by the shape descriptors
/// below (elongation, isotropy, flatness) plus the surface residual.
#[derive(Clone, Copy, Debug)]
pub enum Primitive {
    /// Elongated, one dominant axis: legs, claw arms.
    Capsule(FittedCapsule),
    /// Chunky or flat, no single sweep axis: carapace, claw palm, pincer, the
    /// degenerate coxa blobs (a box hugs them; a capsule blows its radius up).
    Box(ObbFit),
    /// Round in all three axes — no axis to sweep *or* to lay a slab against. No
    /// `sally.glb` part is this round (even the eye is a short stalk → box), so
    /// it's the principled fallback for a future near-spherical cloud, not a
    /// current output. Kept so the chooser degrades sensibly rather than forcing
    /// a ball-shaped blob into a box.
    Ball(FittedBall),
}

impl Primitive {
    /// One-word tag for the validation table.
    pub fn tag(&self) -> &'static str {
        match self {
            Primitive::Capsule(_) => "capsule",
            Primitive::Box(_) => "box",
            Primitive::Ball(_) => "ball",
        }
    }
}

/// Why a primitive was chosen: the winning primitive, its surface residual, and
/// the extent-sorted shape descriptors that drove the decision — so the
/// validation table can show the *decision*, not just its outcome, and assert on
/// it.
#[derive(Clone, Debug)]
pub struct PrimitiveChoice {
    pub primitive: Primitive,
    /// Surface residual (normalised RMS) of the chosen primitive.
    pub chosen_residual: f32,
    /// Extent-sorted descriptors that drove the choice (see [`ObbFit`]).
    pub elongation: f32,
    pub isotropy: f32,
    pub flatness: f32,
    /// Human-readable one-liner: the rule that fired.
    pub reason: &'static str,
}

/// Selection thresholds. Picked from the measured descriptors on `sally.glb`
/// (clean leg sticks: elongation 0.29–0.40, isotropy 0.14–0.21; degenerate
/// middle coxae: isotropy 0.28–0.38 but elongation ~0.9; claw palm elongation
/// 0.57) with margin, and pinned by `meshfit_validation`'s asserts so a future
/// art change that crosses one is caught, not silently re-bucketed.
mod thresh {
    /// Below this elongation (mid extent / longest extent) the cloud has ONE
    /// clearly-dominant long axis — a stick a capsule can sweep. The clean leg
    /// segments sit 0.29–0.40; the chunky parts (coxa blobs, claw palm) run
    /// 0.5–0.9. This is the primary capsule gate.
    pub const ELONGATION_STICK: f32 = 0.45;
    /// At/above this isotropy (shortest extent / longest extent) even a cloud
    /// with a long axis is too round to trust a single sweep — a ball or box. No
    /// leg segment exceeds it; it's a backstop against a fat near-isotropic blob
    /// that happens to squeak under the elongation gate.
    pub const ISOTROPY_NO_AXIS: f32 = 0.45;
    /// Among non-stick clouds, at/above this isotropy AND not flat → a ball
    /// (round in all three); otherwise a box. Tuned so the genuine blobs round
    /// off while flat slabs (carapace, claw palm) and chunky-but-faced parts
    /// stay boxes.
    pub const ISOTROPY_BALL: f32 = 0.6;
    /// Flatness (shortest / middle extent) at/above which a non-stick cloud is
    /// "round enough" in its two minor axes to be a ball rather than a slab box.
    pub const FLATNESS_BALL: f32 = 0.7;
}

/// Choose the collider primitive for a vertex cloud from its shape, not a fixed
/// assumption. Capsule is tried FIRST and accepted only for a genuine stick (one
/// dominant long axis, not a round blob); everything else is a box, except a
/// cloud round in all three axes, which is a ball. Order matters: the stick test
/// must come before the box/ball split, or a flat slab's low flatness would be
/// mistaken for stickness.
///
/// Deliberately conservative: a part lands as a capsule only when its elongation
/// is low *and* it isn't isotropic. Returns `None` for clouds too small to fit
/// anything (< 4 points).
pub fn choose_primitive(points: &[Vec3]) -> Option<PrimitiveChoice> {
    let obb = fit_obb(points)?;
    let elongation = obb.blobbiness();
    let isotropy = obb.isotropy();
    let flatness = obb.flatness();

    // Capsule candidate + its residual, balanced (mean radius) so the leg taper
    // doesn't punish a genuine stick (see `rebalance_capsule_radius`).
    let capsule = fit_capsule(points).map(|c| rebalance_capsule_radius(points, &c));
    let capsule_residual = capsule
        .map(|c| capsule_fit_residual(points, &c))
        .unwrap_or(f32::INFINITY);

    let is_stick = elongation < thresh::ELONGATION_STICK && isotropy < thresh::ISOTROPY_NO_AXIS;
    let (primitive, chosen_residual, reason) = if let Some(c) = capsule.filter(|_| is_stick) {
        (
            Primitive::Capsule(c),
            capsule_residual,
            "elongated, one dominant axis → capsule",
        )
    } else if isotropy >= thresh::ISOTROPY_BALL && flatness >= thresh::FLATNESS_BALL {
        let ball = fit_ball(points)?;
        (
            Primitive::Ball(ball),
            ball_fit_residual(points, &ball),
            "round in all three axes → ball (no axis for a capsule)",
        )
    } else {
        (
            Primitive::Box(obb),
            obb_fit_residual(points, &obb),
            "chunky / flat slab, no clean sweep axis → box",
        )
    };

    Some(PrimitiveChoice {
        primitive,
        chosen_residual,
        elongation,
        isotropy,
        flatness,
        reason,
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

/// Analytic solid-ball mass properties: mass and the (isotropic) principal
/// inertia ⅖mr². Returned in the same `(mass, i_perp)` shape as the other
/// primitives so [`primitive_mass`] can stay uniform.
pub fn ball_mass(radius: f32, density: f32) -> (f32, f32) {
    let (r, rho) = (radius as f64, density as f64);
    let mass = rho * (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);
    let i = 0.4 * mass * r * r;
    (mass as f32, i as f32)
}

/// Mass + transverse principal inertia of whichever primitive was chosen, under
/// the part's hand-coded density — so fitted and reference mass/inertia compare
/// at equal density. For the box the transverse inertia is the larger of the two
/// non-vertical principal inertias (the table's `i_perp` convention); the ball is
/// isotropic so all three coincide.
pub fn primitive_mass(prim: &Primitive, density: f32) -> (f32, f32) {
    match prim {
        Primitive::Capsule(c) => {
            let m = capsule_mass(c.radius, c.half_height(), density);
            (m.mass, m.i_perp)
        }
        Primitive::Box(o) => {
            let e = o.half_extents;
            let (m, i) = cuboid_mass(e.x, e.y, e.z, density);
            (m, i.x.max(i.y).max(i.z))
        }
        Primitive::Ball(b) => ball_mass(b.radius, density),
    }
}

// ---------------------------------------------------------------------------
// Validation report: fitted vs hand-coded, per part
// ---------------------------------------------------------------------------

/// One row of the validation table: a part's chosen primitive + mass/inertia
/// next to the hand-coded reference, plus the evidence behind the choice.
/// Optional fields are `None` when the cloud was too small to fit (< 4 vertices).
pub struct PartFit {
    pub part: PartId,
    pub vertex_count: usize,
    /// Mean dominant-bone weight over the cloud — a skinning-cleanliness proxy. A
    /// low value (the middle coxae sit ~0.57) flags an attachment region where
    /// winner-take-all clustering smears the part, the root cause of degeneracy.
    pub mean_dominant_weight: f32,
    /// Oriented box over the cloud. Retained because the carapace footprint check
    /// reads its extents directly, and `blobbiness()`/`isotropy()` feed the choice.
    pub obb: Option<ObbFit>,
    /// Normalised RMS residual of the cloud to the *constant-radius* (spike p95)
    /// capsule surface: the baseline the balanced/cone residuals improve on.
    pub capsule_residual: f32,
    /// The phase-1 data-driven primitive choice (capsule/box/ball) + the residual
    /// it left and the rule that fired. `None` only for clouds too small to fit.
    /// This is what phase 2 will bake.
    pub choice: Option<PrimitiveChoice>,
    /// Linear taper of the cloud along the (capsule) axis, when one could be
    /// measured. `None` for non-elongated parts where taper is meaningless.
    pub taper: Option<TaperFit>,
    /// Residual a tapered cone leaves on this cloud (the floor a constant-radius
    /// capsule can't beat) — so the table shows how much of the capsule residual
    /// is taper vs genuine non-stick shape. `None` when no taper was fit.
    pub cone_residual: Option<f32>,
    /// Mass of the *chosen* primitive under the part density.
    pub fitted_mass: Option<f32>,
    /// Chosen primitive's transverse principal inertia under the part density.
    pub fitted_i_perp: Option<f32>,
    /// Hand-coded reference mass + transverse inertia (pose-invariant, so they
    /// compare directly regardless of primitive). Built from `body.rs`'s consts.
    pub ref_mass: f32,
    pub ref_i_perp: f32,
}

impl PartFit {
    fn label(&self) -> String {
        self.part.label()
    }
}

/// Run the full fit on the model and pair every part with its hand-coded
/// reference. Every part gets a data-driven primitive choice (capsule/box/ball
/// from [`choose_primitive`]), plus the raw capsule/OBB fits and a taper
/// analysis, so the caller sees both the decision and the evidence behind it.
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
        let choice = choose_primitive(points);
        // Taper + cone residual only make sense on the capsule axis; fit the raw
        // (covering) capsule's axis but report the cone floor against it.
        let (taper, cone_residual) =
            match fitted.and_then(|c| fit_taper(points, &c).map(|t| (c, t))) {
                Some((c, t)) => (Some(t), Some(cone_fit_residual(points, &c, &t))),
                None => (None, None),
            };
        let (fitted_mass, fitted_i_perp) = match &choice {
            Some(ch) => {
                let (m, i) = primitive_mass(&ch.primitive, part.density);
                (Some(m), Some(i))
            }
            None => (None, None),
        };
        rows.push(PartFit {
            part: part.part,
            vertex_count: points.len(),
            mean_dominant_weight: mean_w,
            obb,
            capsule_residual,
            choice,
            taper,
            cone_residual,
            fitted_mass,
            fitted_i_perp,
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
    ref_mass: f32,
    ref_i_perp: f32,
    density: f32,
}

/// The hand-coded body's parts with their reference mass + transverse inertia,
/// in table order. Capsule parts (legs, claw arms) use the analytic capsule
/// mass; box parts (carapace, pincer) the box; the eye the ball. Mass/inertia
/// are pose-invariant, so they compare to the fit regardless of which primitive
/// the fit chose — that comparison, not raw dimensions, is the table's point.
fn reference_parts() -> Vec<RefPart> {
    use super::body::reference as r;
    let mut v = Vec::new();

    let (cm, ci) = cuboid_mass(
        r::CARAPACE_HALF_W,
        r::CARAPACE_HALF_H,
        r::CARAPACE_HALF_D,
        r::CARAPACE_DENSITY,
    );
    v.push(RefPart {
        part: PartId::Carapace,
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
            ref_mass: mu.mass,
            ref_i_perp: mu.i_perp,
            density: r::CLAW_DENSITY,
        });
        let mf = capsule_mass(r::CLAW_FORE_RAD, r::CLAW_FORE_LEN, r::CLAW_DENSITY);
        v.push(RefPart {
            part: PartId::Joint(CrabJointId::ClawFore(side)),
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
            ref_mass: pm,
            ref_i_perp: pi.x.max(pi.z),
            density: r::CLAW_DENSITY,
        });
    }
    for side in [Side::Left, Side::Right] {
        // Eye: a ball. Sphere mass + inertia ⅖mr².
        let (mass, i) = ball_mass(r::EYE_BALL_RAD, r::EYE_DENSITY);
        v.push(RefPart {
            part: PartId::Joint(CrabJointId::EyeStalk(side)),
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
            "per part: the DATA-DRIVEN primitive (capsule/box/ball) + its surface residual, \
             then the fitted vs hand-coded mass/inertia, then the shape evidence."
        );
        println!(
            "cols: domW = mean dominant-bone weight (skinning cleanliness); prim = chosen \
             primitive; res = its surface residual (0=perfect); mass%/I% = fitted/ref; \
             cap_r/cone_r = constant-capsule (spike p95) vs tapered-cone residual (gap = taper \
             cost); taper = thin/fat radius ratio; elong/iso/flat = OBB extent-sorted shape \
             descriptors (stick: low elong+iso; slab: low flat; round: high iso+flat)."
        );
        println!(
            "{:<18} {:>6} {:>4} | {:>7} {:>5} | {:>7} {:>7} {:>5} {:>5} | {:>5} {:>5} | {:>5} {:>5} {:>5} {:>5}",
            "part",
            "nverts",
            "domW",
            "prim",
            "res",
            "f_mass",
            "r_mass",
            "mass%",
            "I%",
            "cap_r",
            "cone_r",
            "taper",
            "elong",
            "iso",
            "flat",
        );
        let mut total_fit_mass = 0.0f32;
        let mut total_ref_mass = 0.0f32;
        for row in &rows {
            total_ref_mass += row.ref_mass;
            let (tag, res) = match &row.choice {
                Some(ch) => (ch.primitive.tag(), ch.chosen_residual),
                None => ("none", f32::NAN),
            };
            let fm = match row.fitted_mass {
                Some(m) => {
                    total_fit_mass += m;
                    m
                }
                None => f32::NAN,
            };
            let mass_pct = 100.0 * fm / row.ref_mass;
            let i_pct = match row.fitted_i_perp {
                Some(i) if row.ref_i_perp > 0.0 => 100.0 * i / row.ref_i_perp,
                _ => f32::NAN,
            };
            let cap_r = row.capsule_residual;
            let cone_r = row.cone_residual.unwrap_or(f32::NAN);
            let taper = row.taper.map(|t| t.taper_ratio()).unwrap_or(f32::NAN);
            let elong = row
                .choice
                .as_ref()
                .map(|c| c.elongation)
                .unwrap_or(f32::NAN);
            let iso = row.choice.as_ref().map(|c| c.isotropy).unwrap_or(f32::NAN);
            let flat = row.choice.as_ref().map(|c| c.flatness).unwrap_or(f32::NAN);
            println!(
                "{:<18} {:>6} {:>4.2} | {:>7} {:>5.2} | {:>7.4} {:>7.4} {:>5.0} {:>5.0} | {:>5.2} {:>5.2} | {:>5.2} {:>5.2} {:>5.2} {:>5.2}",
                row.label(),
                row.vertex_count,
                row.mean_dominant_weight,
                tag,
                res,
                fm,
                row.ref_mass,
                mass_pct,
                i_pct,
                cap_r,
                cone_r,
                taper,
                elong,
                iso,
                flat,
            );
        }
        println!(
            "TOTAL fitted mass {:.3} kg vs hand-coded {:.3} kg ({:.0}%)",
            total_fit_mass,
            total_ref_mass,
            100.0 * total_fit_mass / total_ref_mass
        );

        // --- Phase-1 aggregate scorecard (the deliverable, in three numbers) ---
        let (mut caps, mut boxes, mut balls) = (0, 0, 0);
        for row in &rows {
            match row.choice.as_ref().map(|c| &c.primitive) {
                Some(Primitive::Capsule(_)) => caps += 1,
                Some(Primitive::Box(_)) => boxes += 1,
                Some(Primitive::Ball(_)) => balls += 1,
                None => {}
            }
        }
        println!(
            "\n--- phase-1 scorecard ---\nprimitives: {caps} capsule, {boxes} box, {balls} ball \
             (spike forced all {} into a capsule)",
            rows.len()
        );
        // Taper handling: mean residual of the capsule legs under the spike's
        // enveloping p95 radius vs the coverage-balanced radius. The gap is the
        // taper-handling win; it's largest on the hard-tapering tibiae.
        let leg_caps: Vec<&PartFit> = rows
            .iter()
            .filter(|r| {
                matches!(
                    r.part,
                    PartId::Joint(CrabJointId::LegFemur(..) | CrabJointId::LegTibia(..))
                ) && matches!(
                    r.choice.as_ref().map(|c| &c.primitive),
                    Some(Primitive::Capsule(_))
                )
            })
            .collect();
        if !leg_caps.is_empty() {
            let mean_p95 =
                leg_caps.iter().map(|r| r.capsule_residual).sum::<f32>() / leg_caps.len() as f32;
            let mean_bal = leg_caps
                .iter()
                .map(|r| r.choice.as_ref().unwrap().chosen_residual)
                .sum::<f32>()
                / leg_caps.len() as f32;
            let mean_cone = leg_caps.iter().filter_map(|r| r.cone_residual).sum::<f32>()
                / leg_caps.len() as f32;
            println!(
                "taper: leg-capsule residual {mean_p95:.3} (spike p95) → {mean_bal:.3} (balanced) \
                 → {mean_cone:.3} (cone floor); balanced cuts {:.0}% of the p95→cone gap",
                100.0 * (mean_p95 - mean_bal) / (mean_p95 - mean_cone).max(1e-6)
            );
        }
        // Degenerate-cluster fix: the middle coxae the spike blew up. Show the
        // residual the boxed fit leaves vs what the forced capsule would have.
        let mid_coxae: Vec<&PartFit> = rows
            .iter()
            .filter(|r| {
                matches!(
                    r.part,
                    PartId::Joint(CrabJointId::LegCoxa(_, 1) | CrabJointId::LegCoxa(_, 2))
                )
            })
            .collect();
        if !mid_coxae.is_empty() {
            let mean_box = mid_coxae
                .iter()
                .map(|r| r.choice.as_ref().unwrap().chosen_residual)
                .sum::<f32>()
                / mid_coxae.len() as f32;
            let mean_cap =
                mid_coxae.iter().map(|r| r.capsule_residual).sum::<f32>() / mid_coxae.len() as f32;
            println!(
                "degenerate coxae: {} middle-coxa blobs re-routed capsule→box; residual {mean_cap:.3} \
                 (forced capsule) → {mean_box:.3} (box), no radius blow-ups",
                mid_coxae.len()
            );
        }

        // Per-part primitive roll-up + the rule each part's choice fired.
        println!("\n--- primitive choice + reason (the phase-1 decision) ---");
        for row in &rows {
            if let Some(ch) = &row.choice {
                println!(
                    "  {:<18} {:>7}  — {}",
                    row.label(),
                    ch.primitive.tag(),
                    ch.reason
                );
            } else {
                println!(
                    "  {:<18} {:>7}  — cloud too small to fit",
                    row.label(),
                    "none"
                );
            }
        }

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

        // --- Assertions: pin the phase-1 fit-quality guarantees. --------------
        // The spike asserted only "a clean stick fits / the carapace footprint
        // matches"; phase 1 pins the whole decision: every part gets a primitive,
        // the right parts get capsules, the degenerate ones do NOT, no radius
        // blows up, and taper (not non-stickness) explains the leg residual.

        // (0) Every part is assigned a primitive — no cloud left unfit.
        for row in &rows {
            assert!(
                row.choice.is_some(),
                "{} got no primitive (cloud {} verts)",
                row.label(),
                row.vertex_count
            );
        }

        // (1) Femur and tibia on every leg are chosen as CAPSULES. The femur is
        // the clean near-cylindrical stick (taper ~0.85), so its balanced
        // residual is pinned tight (<0.42). The tibia tapers HARD (thin end ~half
        // the fat end) and the art adds a foot/joint bulge a single radius can't
        // hug, so its residual genuinely floors higher (~0.48 — the cone barely
        // beats it); we pin only that it stays under 0.55 and that the
        // coverage-balanced radius IMPROVES on the spike's enveloping p95 capsule
        // (assert (5b)). Honest divergence, bounded, not forced under a fake gate.
        for row in &rows {
            if let PartId::Joint(CrabJointId::LegFemur(..) | CrabJointId::LegTibia(..)) = row.part {
                let ch = row.choice.as_ref().unwrap();
                assert!(
                    matches!(ch.primitive, Primitive::Capsule(_)),
                    "{} should fit a capsule, got {} (elong {:.2}, iso {:.2})",
                    row.label(),
                    ch.primitive.tag(),
                    ch.elongation,
                    ch.isotropy
                );
                let bound = match row.part {
                    PartId::Joint(CrabJointId::LegFemur(..)) => 0.42,
                    _ => 0.55, // tibia: taper-limited floor, documented
                };
                assert!(
                    ch.chosen_residual < bound,
                    "{} balanced capsule residual {:.3} exceeds its bound {:.2}",
                    row.label(),
                    ch.chosen_residual,
                    bound
                );
            }
        }

        // (2) NO chosen capsule has a degenerate (blown-up) radius. The spike's
        // middle coxae blew the capsule radius to 0.20 (≈4.5× the hand-coded
        // 0.045); a real leg-segment radius is well under 0.12. Any capsule above
        // that means a blob got forced into a capsule — the exact failure phase 1
        // removes. (Box/ball parts are exempt; this guards only capsules.)
        for row in &rows {
            if let Some(PrimitiveChoice {
                primitive: Primitive::Capsule(c),
                ..
            }) = &row.choice
            {
                assert!(
                    c.radius > 0.005 && c.radius < 0.12,
                    "{} capsule radius {:.3} is degenerate (blob forced into a capsule)",
                    row.label(),
                    c.radius
                );
                assert!(
                    c.segment_len() >= 0.0,
                    "{} negative capsule segment",
                    row.label()
                );
            }
        }

        // (3) The degenerate middle coxae (legs 1 & 2 — isotropic hip blobs with
        // no PCA axis) are NOT capsules. The spike fit them as radius-0.20
        // capsules; phase 1 must route them to a box or ball instead. This is the
        // "fix the degenerate clusters" gate.
        for row in &rows {
            if let PartId::Joint(CrabJointId::LegCoxa(_, 1) | CrabJointId::LegCoxa(_, 2)) = row.part
            {
                let ch = row.choice.as_ref().unwrap();
                assert!(
                    !matches!(ch.primitive, Primitive::Capsule(_)),
                    "{} is a degenerate isotropic blob (iso {:.2}) and must NOT be a capsule, got {}",
                    row.label(),
                    ch.isotropy,
                    ch.primitive.tag()
                );
            }
        }

        // (4) The claw palm (ClawFore) is genuinely non-capsule (all axes
        // comparable) and must be a box/hull, not a capsule. The spike mis-fit it
        // as a radius-0.147 capsule (mass 8×); pin that it no longer does.
        for row in &rows {
            if let PartId::Joint(CrabJointId::ClawFore(_)) = row.part {
                let ch = row.choice.as_ref().unwrap();
                assert!(
                    matches!(ch.primitive, Primitive::Box(_)),
                    "{} (claw palm) should be a box, got {}",
                    row.label(),
                    ch.primitive.tag()
                );
            }
        }

        // (5) Taper handling, quantified two ways on every femur/tibia:
        //   (5a) The coverage-BALANCED capsule radius never does worse than the
        //        spike's enveloping p95 radius, and on the strongly-tapered
        //        segments (taper_ratio < 0.7 — the tibiae) it strictly improves.
        //        That improvement is the taper-handling win over the spike.
        //   (5b) The tapered-CONE residual is at or below the constant-capsule
        //        residual: a shape that can taper fits at least as well, i.e. the
        //        residual really is taper, not non-stickness. (For the clean
        //        near-cylindrical femur the gap is tiny; for the tibia the cone
        //        ties the balanced capsule — its remaining residual is the art's
        //        foot bulge, not taper, which is why (1) bounds the tibia looser.)
        for row in &rows {
            if let PartId::Joint(CrabJointId::LegFemur(..) | CrabJointId::LegTibia(..)) = row.part {
                let ch = row.choice.as_ref().unwrap();
                let balanced = ch.chosen_residual;
                let p95 = row.capsule_residual;
                let cone = row.cone_residual.expect("cone residual");
                let taper = row.taper.expect("taper").taper_ratio();
                assert!(
                    balanced <= p95 + 1e-3,
                    "{} balanced residual {:.3} worse than p95 {:.3}",
                    row.label(),
                    balanced,
                    p95
                );
                if taper < 0.7 {
                    assert!(
                        balanced < p95 - 1e-3,
                        "{} (taper {:.2}) balanced residual {:.3} should beat p95 {:.3}",
                        row.label(),
                        taper,
                        balanced,
                        p95
                    );
                }
                assert!(
                    cone <= p95 + 1e-3,
                    "{} cone residual {:.3} should not exceed capsule residual {:.3}",
                    row.label(),
                    cone,
                    p95
                );
                assert!(
                    (0.1..0.99).contains(&taper),
                    "{} taper ratio {:.2} outside a real leg's range",
                    row.label(),
                    taper
                );
            }
        }

        // (6) The carapace is chosen as a BOX and its footprint (longest
        // horizontal half-extent) matches the hand-coded box width within 20% —
        // the dimension the spike recovers cleanly. (Its HEIGHT still diverges:
        // the model is a dome, the physics a flat slab — reported in the table,
        // NOT asserted; that mismatch is a physics decision, not a fit failure.)
        let carapace = rows
            .iter()
            .find(|r| r.part == PartId::Carapace)
            .expect("carapace row");
        assert!(
            matches!(
                carapace.choice.as_ref().map(|c| &c.primitive),
                Some(Primitive::Box(_))
            ),
            "carapace should be a box, got {}",
            carapace
                .choice
                .as_ref()
                .map(|c| c.primitive.tag())
                .unwrap_or("none")
        );
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

        // (7) The total fitted mass is finite and positive (the pipeline produced
        // real numbers end-to-end). Its magnitude vs hand-coded is in the table —
        // fitted still runs heavy because the art is fleshier than the stick
        // colliders; that is the headline finding, discussed in the plan.
        assert!(
            total_fit_mass.is_finite() && total_fit_mass > 0.0,
            "total fitted mass {total_fit_mass}"
        );
    }
}
