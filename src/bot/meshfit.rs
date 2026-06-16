//! glTF skeleton loader + an offline collider-fitting library.
//!
//! [`LoadedModel`] parses `sally.glb` (the same model the cosmetic skin and the
//! rig-derived body read) into bind-pose bone transforms + skinned vertices — the
//! part the live body uses ([`super::rig`] reads bone origins from it).
//!
//! The rest is an OFFLINE collider-fitting pass: for each physics part it gathers
//! the mesh vertices whose dominant skin bone maps to that part, chooses a collider
//! primitive from the cloud's shape (capsule/box/ball, via [`choose_primitive`]),
//! solves the part's placement from the bind-pose bone chain, and derives mass from
//! the chosen [`Primitive`]. [`bake_report`] runs the fit, [`FittedBody::from_reports`]
//! assembles a typed table serialized to RON by `--bake-colliders` and hand-tuned by
//! `--edit-colliders`.
//!
//! NOTE: the rig body currently spawns PLACEHOLDER capsule colliders (see
//! [`super::rig`]); nothing loads a baked [`FittedBody`] at runtime. This fitting
//! half is offline tooling kept for the phase-2 rig collider re-fit (bddap/rl#16) —
//! run deterministically offline, never at spawn or in `build.rs`.
//!
//! **Size vs placement are fit from different sources, on purpose.** The cloud fit
//! gives pose-invariant *size* (capsule half-height/radius, box half-extents) — it
//! does not know where the part sits. *Placement* comes from the skeleton: the
//! proximal→distal bind-pose bone origins give each segment's pivot and direction.
//! So a re-fit of the mesh changes size; a re-rig changes placement; neither
//! silently perturbs the other.
//!
//! Coordinate frames: the cloud fit and bone origins are in the glTF's *bind-pose
//! world* frame; a [`Placement`] re-expresses a part's collider in its link-local
//! frame (origin at the joint pivot).

use std::collections::HashMap;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::body::{CrabJointId, Side};

/// Which physics link a fitted collider belongs to: the carapace is the root,
/// every other part is a joint's child link. Mirrors the skin's `LinkKey` but
/// carries [`CrabJointId`] so a baked [`FittedPart`] names the exact joint whose
/// link it replaces.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum PartId {
    Carapace,
    Joint(CrabJointId),
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

/// Where to find the model for an offline bake: `CRAB_MODEL_PATH` if set (same
/// var the skin uses), else the dev box's checkout. Absent → the bake/validation
/// tests self-skip.
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
        // Inverse-bind matrices (skin-joint order). Combined with the joints'
        // bind-world transforms they map a raw mesh vertex to its bind-pose WORLD
        // position — the position Bevy actually skins the visible mesh to. Fitting
        // colliders to that (not the raw, pre-skin POSITION attribute) is what keeps
        // them on the rendered mesh rather than offset by the mesh/armature frame.
        let inv_binds: Vec<Mat4> = skin
            .reader(|buf| (buf.index() == 0).then_some(blob))
            .read_inverse_bind_matrices()
            .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_else(|| vec![Mat4::IDENTITY; joint_nodes.len()]);

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
                let raw = Vec3::from_array(*p);
                // Bind-pose WORLD position: blend the influencing joints' bind
                // transforms (bindWorld · invBind) by skin weight — the same skinning
                // math the renderer uses, so the cloud sits where the visible mesh is.
                let mut world = Vec3::ZERO;
                let mut wsum = 0.0f32;
                for lane in 0..4 {
                    let wt = w[lane];
                    if wt <= 0.0 {
                        continue;
                    }
                    let ji = j[lane] as usize;
                    let jm = bind_world
                        .get(&joint_nodes[ji])
                        .copied()
                        .unwrap_or(Mat4::IDENTITY)
                        * inv_binds[ji];
                    world += wt * jm.transform_point3(raw);
                    wsum += wt;
                }
                let pos = if wsum > 1e-6 { world / wsum } else { raw };
                // Dominant influence tags the vertex for per-part bucketing.
                let (lane, &wmax) = w
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                verts.push(SkinnedVertex {
                    pos,
                    dominant_node: joint_nodes[j[lane] as usize],
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

    /// Group vertices by physics part via [`super::rig::part_for_bone`] — the one
    /// canonical bone→part mapping. Returns, per part, the world-space vertex
    /// positions and the mean dominant-weight (a skinning-cleanliness proxy).
    /// Vertices on a non-rig node (no part) are dropped.
    pub(crate) fn vertices_by_part(&self) -> HashMap<PartId, (Vec<Vec3>, f32)> {
        let mut out: HashMap<PartId, (Vec<Vec3>, f32)> = HashMap::new();
        for v in &self.verts {
            let Some(name) = self.node_name.get(&v.dominant_node) else {
                continue;
            };
            let Some(part) = super::rig::part_for_bone(name) else {
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

    /// World-space positions of every vertex whose dominant bone is one of `names`.
    /// Used to size the carapace box from the trunk's actual shell flesh.
    pub(crate) fn vertices_for_bones(&self, names: &[&str]) -> Vec<Vec3> {
        self.verts
            .iter()
            .filter(|v| {
                self.node_name
                    .get(&v.dominant_node)
                    .is_some_and(|n| names.contains(&n.as_str()))
            })
            .map(|v| v.pos)
            .collect()
    }

    /// Bind-pose world origin of a bone by name, if present. The bone origin is
    /// the translation column of its bind-world transform — i.e. where the joint
    /// pivot sits in the rest skeleton.
    pub fn bone_origin(&self, name: &str) -> Option<Vec3> {
        self.bone_bind_pose(name).map(|(o, _)| o)
    }

    /// Index of a bone by name, or `None` if absent. Linear scan over the node
    /// table; the model has a few hundred nodes, so callers that resolve many
    /// bones stay cheap enough without an inverse map.
    fn node_index(&self, name: &str) -> Option<usize> {
        self.node_name
            .iter()
            .find(|(_, nm)| nm.as_str() == name)
            .map(|(i, _)| *i)
    }

    /// Bind-pose world (origin, basis) of a bone by name. The basis is the bone's
    /// bind-world rotation — the orientation the physics link inherits, since each
    /// link rotates *with* its bone. Solving a [`Placement`] needs both: the origin
    /// is the joint pivot, and the basis is the frame the placement is expressed
    /// relative to (so `body.rs`'s rest bake re-applies it consistently). The
    /// rig-driven body spawn ([`super::rig`]) reads it per bone to place each link.
    pub(crate) fn bone_bind_pose(&self, name: &str) -> Option<(Vec3, Quat)> {
        let idx = self.node_index(name)?;
        self.bind_world.get(&idx).map(|m| {
            let (_, rot, trans) = m.to_scale_rotation_translation();
            (trans, rot)
        })
    }
}

/// Recursively compose `parent_world * node_local` for a node and its subtree,
/// recording each node's world matrix. glTF node transforms are TRS or a raw
/// matrix; `node.transform().matrix()` normalises both to a column-major 4x4.
fn compose_world(node: &gltf::Node, parent_world: Mat4, world: &mut HashMap<usize, Mat4>) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let w = parent_world * local;
    world.insert(node.index(), w);
    for child in node.children() {
        compose_world(&child, w, world);
    }
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

/// How well a *live* collider surface hugs the vertex cloud it stands in for, in
/// model units, in the cloud's own (bind-pose world) frame. Unlike
/// [`capsule_fit_residual`] (RMS, sign-collapsed, against a *re-fit* capsule), this
/// keeps the sign and scores the geometry the body actually spawns with: positive
/// surface distance = vertex OUTSIDE the collider (mesh pokes out → physics
/// under-coverage), negative = inside (collider margin / bulge). The split lets
/// `--verify-colliders` tell "mesh escapes the collider" from "collider oversized",
/// and the axis/radius diagnostics say *why* a capsule misses its limb.
pub(crate) struct ColliderScore {
    pub n: usize,
    /// Fraction of vertices more than 1 mm outside the collider.
    pub frac_outside: f32,
    /// 95th-percentile and max depth a vertex pokes out past the surface.
    pub poke_out_p95: f32,
    pub poke_out_max: f32,
    /// 95th-percentile depth the surface sits past the mesh on the covered side.
    pub bulge_p95: f32,
    /// Angle (deg) between the capsule axis and the cloud's principal axis — large
    /// = capsule pointed off the limb. `NaN` for a box or a too-small cloud.
    pub axis_skew_deg: f32,
    /// Live radius ÷ the cloud's p95 perpendicular spread about the live axis;
    /// `>1` fat, `<1` starved. `NaN` for a box or a too-small cloud.
    pub radius_ratio: f32,
}

impl ColliderScore {
    /// Aggregate a set of signed surface distances (capsule diagnostics filled in
    /// by [`score_capsule`]; left `NaN` for a box).
    fn from_signed(sd: &[f32]) -> Self {
        let inv = 1.0 / sd.len().max(1) as f32;
        let mut outs: Vec<f32> = sd.iter().copied().filter(|&d| d > 0.0).collect();
        let mut ins: Vec<f32> = sd
            .iter()
            .copied()
            .filter(|&d| d < 0.0)
            .map(f32::abs)
            .collect();
        let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
        outs.sort_by(cmp);
        ins.sort_by(cmp);
        let pctl = |s: &[f32], q| if s.is_empty() { 0.0 } else { percentile(s, q) };
        ColliderScore {
            n: sd.len(),
            frac_outside: sd.iter().filter(|&&d| d > 0.001).count() as f32 * inv,
            poke_out_p95: pctl(&outs, 0.95),
            poke_out_max: outs.last().copied().unwrap_or(0.0),
            bulge_p95: pctl(&ins, 0.95),
            axis_skew_deg: f32::NAN,
            radius_ratio: f32::NAN,
        }
    }
}

/// Score a cloud against a capsule given by its segment endpoints + radius (world).
pub(crate) fn score_capsule(points: &[Vec3], a: Vec3, b: Vec3, radius: f32) -> ColliderScore {
    let seg = b - a;
    let seg_len2 = seg.length_squared().max(1e-12);
    let sd: Vec<f32> = points
        .iter()
        .map(|&p| {
            let t = ((p - a).dot(seg) / seg_len2).clamp(0.0, 1.0);
            (p - (a + seg * t)).length() - radius
        })
        .collect();
    let mut s = ColliderScore::from_signed(&sd);
    // Diagnostics: how skewed is the live axis vs the cloud's true long axis, and
    // is the radius fat/starved relative to the cloud's spread about that axis.
    let axis = seg.normalize_or_zero();
    if points.len() >= 4 && axis.length_squared() > 0.5 {
        let centroid = points.iter().copied().sum::<Vec3>() / points.len() as f32;
        let (axes, _) = covariance_eigenframe(points, centroid);
        s.axis_skew_deg = axis.dot(axes[0]).abs().clamp(0.0, 1.0).acos().to_degrees();
        let mut perp: Vec<f32> = points
            .iter()
            .map(|&p| {
                let d = p - a;
                (d - axis * d.dot(axis)).length()
            })
            .collect();
        perp.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        s.radius_ratio = radius / percentile(&perp, 0.95).max(1e-4);
    }
    s
}

/// Score a cloud against a world-axis-aligned box (centre + half-extents).
pub(crate) fn score_box(points: &[Vec3], center: Vec3, half: Vec3) -> ColliderScore {
    let sd: Vec<f32> = points
        .iter()
        .map(|&p| {
            let local = (p - center).abs() - half;
            let outside = local.max(Vec3::ZERO).length();
            let inside = local.max_element().min(0.0);
            outside + inside
        })
        .collect();
    ColliderScore::from_signed(&sd)
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

/// The collider shape a part wants, **dimensions only** — pose-invariant, with no
/// position or orientation (those live in a [`Placement`]). This is the canonical
/// baked output: mass derives from it ([`Primitive::mass_properties`]) and the
/// body spawns a collider of exactly these dims, so there is one source for a
/// part's size and nothing to drift out of sync. The world-space cloud fit it is
/// distilled from is [`FittedShape`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Primitive {
    /// Elongated, one dominant axis: legs, claw arms. `half_height` is the
    /// cylinder half-length (caps excluded), matching rapier's `capsule_y`.
    Capsule { half_height: f32, radius: f32 },
    /// Chunky or flat, no single sweep axis: carapace, claw palm, pincer, the
    /// degenerate coxa blobs (a box hugs them; a capsule blows its radius up).
    Cuboid { half_extents: Vec3 },
    /// Round in all three axes — no axis to sweep *or* to lay a slab against. No
    /// `sally.glb` part is this round (even the eye is a short stalk → box), so
    /// it's the principled fallback for a future near-spherical cloud, not a
    /// current output. Kept so the chooser degrades sensibly rather than forcing
    /// a ball-shaped blob into a box.
    Ball { radius: f32 },
}

impl Primitive {
    /// One-word tag for the validation table and the bake reason lines.
    pub fn tag(&self) -> &'static str {
        match self {
            Primitive::Capsule { .. } => "capsule",
            Primitive::Cuboid { .. } => "box",
            Primitive::Ball { .. } => "ball",
        }
    }

    /// Mass + transverse principal inertia under `density`. The transverse moment
    /// is the table's `i_perp` convention: for the box, the larger of the two
    /// non-vertical principal inertias; the ball is isotropic so all three
    /// coincide. The single source for a part's mass — there are no stored mass
    /// fields to disagree with the dims.
    pub fn mass_properties(&self, density: f32) -> (f32, f32) {
        match *self {
            Primitive::Capsule {
                half_height,
                radius,
            } => {
                let m = capsule_mass(radius, half_height, density);
                (m.mass, m.i_perp)
            }
            Primitive::Cuboid { half_extents: e } => {
                let (m, i) = cuboid_mass(e.x, e.y, e.z, density);
                (m, i.x.max(i.y).max(i.z))
            }
            Primitive::Ball { radius } => ball_mass(radius, density),
        }
    }

    /// Reject dimensions that are non-finite or non-positive — the ones a foreign/
    /// hand-edited table can carry that serde accepts but rapier cannot build into a
    /// sane collider. A radius or extent must be strictly positive; a capsule's
    /// half-height may be 0 (that capsule is a sphere) but not negative.
    fn validate(&self) -> Result<(), String> {
        let positive = |label: &str, v: f32| {
            (v.is_finite() && v > 0.0)
                .then_some(())
                .ok_or_else(|| format!("{label} must be finite and > 0, got {v}"))
        };
        let non_negative = |label: &str, v: f32| {
            (v.is_finite() && v >= 0.0)
                .then_some(())
                .ok_or_else(|| format!("{label} must be finite and >= 0, got {v}"))
        };
        match *self {
            Primitive::Capsule {
                half_height,
                radius,
            } => {
                non_negative("capsule half_height", half_height)?;
                positive("capsule radius", radius)
            }
            Primitive::Cuboid { half_extents: e } => {
                positive("cuboid half_extent x", e.x)?;
                positive("cuboid half_extent y", e.y)?;
                positive("cuboid half_extent z", e.z)
            }
            Primitive::Ball { radius } => positive("ball radius", radius),
        }
    }
}

/// The world-space shape fitted to a vertex cloud, before it is distilled into a
/// pose-invariant [`Primitive`]. Carries absolute positions/axes. Production no
/// longer poses parts from this (the bake sizes/orients in the bone frame via
/// [`fit_part`]); the payloads survive only to drive the `#[cfg(test)]` validation
/// table's reported shape + residual, so they read as dead in a release build —
/// pending classifier removal (bddap/rl#25). Not serialized.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum FittedShape {
    Capsule(FittedCapsule),
    Box(ObbFit),
    Ball(FittedBall),
}

/// Why a primitive was chosen: the dimensioned primitive, the world-space fit it
/// came from, its surface residual, and the extent-sorted shape descriptors that
/// drove the decision — so the validation table can show the *decision*, not just
/// its outcome, and assert on it.
#[derive(Clone, Debug)]
pub struct PrimitiveChoice {
    /// The world-space fit the choice was distilled from (carries the residual and
    /// the box's fitted axes). Drives the `#[cfg(test)]` validation table's reported
    /// shape; the production bake goes through [`fit_part`], not this — so it's
    /// written but unread in a release build (bddap/rl#25).
    #[allow(dead_code)]
    pub shape: FittedShape,
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
/// 0.57) with margin.
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
    let (shape, chosen_residual, reason) = if let Some(c) = capsule.filter(|_| is_stick) {
        (
            FittedShape::Capsule(c),
            capsule_residual,
            "elongated, one dominant axis → capsule",
        )
    } else if isotropy >= thresh::ISOTROPY_BALL && flatness >= thresh::FLATNESS_BALL {
        let ball = fit_ball(points)?;
        (
            FittedShape::Ball(ball),
            ball_fit_residual(points, &ball),
            "round in all three axes → ball (no axis for a capsule)",
        )
    } else {
        (
            FittedShape::Box(obb),
            obb_fit_residual(points, &obb),
            "chunky / flat slab, no clean sweep axis → box",
        )
    };

    Some(PrimitiveChoice {
        shape,
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
/// two hemispheres) so mass derives from the dimensions without constructing a
/// rapier collider. `i_axial` is the moment about the capsule's own long axis;
/// `i_perp` about a transverse axis through the centre of mass. For comparison
/// with the hand-coded body these are the principal inertias of the part.
#[derive(Clone, Copy, Debug)]
pub struct CapsuleMass {
    pub mass: f32,
    /// Moment about the capsule's own long axis. Production reads only `i_perp`
    /// (the transverse moment the table compares); `i_axial` rounds out the
    /// principal inertias and pins the sphere-reduction test's isotropy check.
    #[cfg_attr(not(test), allow(dead_code))]
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
/// primitives so [`Primitive::mass_properties`] can stay uniform.
pub fn ball_mass(radius: f32, density: f32) -> (f32, f32) {
    let (r, rho) = (radius as f64, density as f64);
    let mass = rho * (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);
    let i = 0.4 * mass * r * r;
    (mass as f32, i as f32)
}

// ---------------------------------------------------------------------------
// Placement: where each part's collider sits, from the bind-pose skeleton
// ---------------------------------------------------------------------------

/// A part's collider pose in its **link-local frame**: the frame whose origin is
/// the joint pivot (the proximal bind-pose bone origin) and whose axes are the
/// body's, so it drops onto `body.rs`'s hand-authored parent anchor for that
/// joint. `center` is the collider centre relative to the pivot; `rotation`
/// orients the collider's canonical axes into the link frame (capsule +Y along the
/// bone; box identity, its axes being the bone frame's). The pivot-relative
/// encoding is why placement survives the runtime joint pose: only *where on the
/// parent* the joint attaches is hand-authored; how the collider hangs off it is this.
///
/// Stored as a translation+rotation rather than a [`Transform`] because the scale
/// is always 1 (a collider is its own size) and a bake artifact should carry no
/// field that is only ever the identity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub center: Vec3,
    pub rotation: Quat,
}

/// The bind-pose bone span that defines a part's segment, in glTF world space:
/// the proximal (pivot) and distal origins plus the proximal bone's bind-world
/// basis. The basis is what makes [`fit_part`] frame-correct — the collider is
/// posed relative to *this bone's* rest orientation, the same frame `body.rs`
/// rebuilds when it bakes the link's rest pose; without it the placement would be
/// in raw glTF world and land rotated off the limb. For the carapace the "segment"
/// is just the pivot bone and a +Z hint; its box is measured in the basis frame
/// (which, the root being identity, is world).
pub(crate) struct BoneSpan {
    pub(crate) proximal: Vec3,
    /// Bind-world rotation of the proximal bone — the link-local frame placement
    /// is expressed in.
    pub(crate) proximal_basis: Quat,
    pub(crate) distal: Vec3,
}

impl LoadedModel {
    /// The proximal/distal bind-pose bone origins bounding a part's segment, by
    /// the same bone→segment grouping the skin and the fit cluster with (coxa
    /// 000→001, femur 001→003, tibia 003→005; claw/eye chains analogously). The
    /// proximal origin is the joint pivot; the distal sets the segment direction
    /// and length. `None` if either bone is missing (e.g. a part with no skeleton
    /// correspondence).
    pub(crate) fn bone_span(&self, part: PartId) -> Option<BoneSpan> {
        // Resolve a span from its proximal/distal bone names: the proximal bone
        // gives the pivot AND the link-local basis; the distal gives direction.
        let span = |prox: &str, dist: &str| {
            let (proximal, proximal_basis) = self.bone_bind_pose(prox)?;
            Some(BoneSpan {
                proximal,
                proximal_basis,
                distal: self.bone_origin(dist)?,
            })
        };
        let leg = |leg: u8, side: Side, prox: &str, dist: &str| {
            let s = side_tag(side);
            span(
                &format!("Def_leg_0{}.{prox}.{s}", leg + 1),
                &format!("Def_leg_0{}.{dist}.{s}", leg + 1),
            )
        };
        match part {
            // Root: the carapace bone is the pivot; the distal hint points front
            // (+Z) so a degenerate direction never leaves the frame undefined.
            // The basis is IDENTITY, not the thorax bone's: the root link has no
            // rest bake — `body.rs` spawns it axis-aligned in world — so its link
            // frame IS world. That keeps the carapace's [`Placement`] (and the OBB
            // axes its rotation carries) in world, which `fitted_root` reads to map
            // each principal extent onto the world axis it's dominant on.
            PartId::Carapace => {
                let proximal = self
                    .bone_origin("Def_thorax_back")
                    .or_else(|| self.bone_origin("Def_thorax_front"))?;
                Some(BoneSpan {
                    proximal,
                    proximal_basis: Quat::IDENTITY,
                    distal: proximal + Vec3::Z,
                })
            }
            PartId::Joint(id) => match id {
                CrabJointId::LegCoxa(side, l) => leg(l, side, "000", "001"),
                CrabJointId::LegMerus(side, l) => leg(l, side, "002", "003"),
                CrabJointId::LegCarpus(side, l) => leg(l, side, "003", "004"),
                CrabJointId::ClawShoulder(side) => {
                    let s = side_tag(side);
                    span(
                        &format!("Def_pincer.000a.{s}"),
                        &format!("Def_pincer.001.{s}"),
                    )
                }
                CrabJointId::ClawWrist(side) => {
                    let s = side_tag(side);
                    span(
                        &format!("Def_pincer.005.{s}"),
                        &format!("Def_pincer.006b.{s}"),
                    )
                }
                CrabJointId::ClawPincer(side) => {
                    let s = side_tag(side);
                    span(
                        &format!("Def_pincer.006b.{s}"),
                        &format!("Def_pincer.006.{s}"),
                    )
                }
            },
        }
    }
}

/// glTF side suffix (`.L`/`.R`) for a [`Side`], the way the deform bones are
/// named.
fn side_tag(side: Side) -> &'static str {
    match side {
        Side::Left => "L",
        Side::Right => "R",
    }
}

/// Fit a box **axis-aligned in the proximal bone's frame**: its orientation is
/// the rig's (identity rotation in the link frame, exactly as a capsule rides its
/// bone) and its half-extents are the cloud's robust (p98) spread measured along
/// the bone axes. Bounding the cloud in the bone frame — the frame the mesh is
/// skinned into — keeps the box on the mesh by construction, where the old
/// point-cloud OBB drifted: a near-isotropic or flat cloud has no stable
/// eigenframe, so its principal axes (and the box with them) span off the limb.
/// The carapace root's basis is identity, so this reduces to a world-axis-aligned
/// box.
fn fit_box_bone_aligned(points: &[Vec3], span: &BoneSpan) -> (Primitive, Placement) {
    let to_local = span.proximal_basis.inverse();
    // Cloud into the link-local (bone) frame, pivot-relative.
    let local: Vec<Vec3> = points
        .iter()
        .map(|&p| to_local * (p - span.proximal))
        .collect();
    let center = local.iter().copied().sum::<Vec3>() / local.len() as f32;
    // Robust half-extent per bone axis (p98 of the centred |offset|), mirroring
    // `fit_obb` so a few skinning-bleed verts don't balloon the box.
    let mut proj: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for q in &local {
        let d = (*q - center).abs();
        for k in 0..3 {
            proj[k].push(d[k]);
        }
    }
    let mut he = Vec3::ZERO;
    for k in 0..3 {
        proj[k].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        he[k] = percentile(&proj[k], 0.98).max(1e-4);
    }
    (
        Primitive::Cuboid { half_extents: he },
        Placement {
            center,
            rotation: Quat::IDENTITY,
        },
    )
}

/// Fit a capsule stretched along the bone segment (proximal→distal) — the
/// "red→green" capsule: it spans exactly the bone, fattened by `radius` to cover
/// the cloud's spread perpendicular to the bone line. Orientation is the rig's
/// (the bone direction), never the point cloud, so it can't splay off the limb the
/// way a point-cloud box does. This is the fit for every *jointed* part; only the
/// carapace — whose "bone" is a single point with no segment to stretch along —
/// stays a box.
fn fit_capsule_redgreen(points: &[Vec3], span: &BoneSpan) -> (Primitive, Placement) {
    let pivot = span.proximal;
    let seg = span.distal - pivot;
    let len = seg.length();
    let axis = if len > 1e-6 { seg / len } else { Vec3::Y };
    // Radius = robust (p90) distance from the bone line, so a few skinning-bleed
    // verts don't inflate it.
    let mut perp: Vec<f32> = points
        .iter()
        .map(|&p| {
            let d = p - pivot;
            (d - axis * d.dot(axis)).length()
        })
        .collect();
    perp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let radius = percentile(&perp, 0.90).max(1e-3);
    // The capsule spans the whole bone: its hemispherical caps (each `radius` long)
    // reach the endpoints, so the cylinder half-length is the bone half-length less
    // one cap. Clamp at 0 for a stub shorter than its own radius (a near-sphere).
    let half_height = (0.5 * len - radius).max(0.0);
    let to_local = span.proximal_basis.inverse();
    let world_rot = if axis.dot(Vec3::Y) < -0.9999 {
        Quat::from_axis_angle(Vec3::X, std::f32::consts::PI)
    } else {
        Quat::from_rotation_arc(Vec3::Y, axis)
    };
    (
        Primitive::Capsule {
            half_height,
            radius,
        },
        Placement {
            center: to_local * (0.5 * seg),
            rotation: to_local * world_rot,
        },
    )
}

/// The canonical per-part collider fit, dispatched by the body plan the `PartId`
/// already encodes: the carapace is a bone-frame box (its "bone" is a single
/// point, no segment to stretch along), every jointed part a bone-stretched
/// capsule. Shared by the offline bake and the editor's seed so the two can't
/// drift.
///
/// Both shapes take their orientation from the rig, in the proximal bone's
/// bind-world frame. That matters because `body.rs` consumes the [`Placement`] in
/// the link's *rest* frame, not the bind pose the fit ran in: every
/// world-relative-to-pivot quantity is mapped into the link frame by `basis⁻¹`
/// (the proximal bone's bind rotation), and at spawn `body.rs` re-applies the
/// link's rest world rotation `L`, recovering `L · basis⁻¹ · (world vector)` — the
/// geometry as the limb actually sits, never rotated off the limb.
pub(crate) fn fit_part(part: PartId, points: &[Vec3], span: &BoneSpan) -> (Primitive, Placement) {
    match part {
        PartId::Carapace => fit_box_bone_aligned(points, span),
        PartId::Joint(_) => fit_capsule_redgreen(points, span),
    }
}

// ---------------------------------------------------------------------------
// Typed collider table (the offline-bake artifact)
// ---------------------------------------------------------------------------

/// One part of the baked body: which physics link it is, the dimensioned
/// collider [`Primitive`] to build there, its [`Placement`] in the link frame,
/// and the density mass is derived under. Mass is intentionally *not* stored —
/// it is `primitive.mass_properties(density)`, so the artifact cannot carry a
/// mass that disagrees with its own dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedPart {
    pub part: PartId,
    pub primitive: Primitive,
    pub placement: Placement,
    pub density: f32,
}

impl FittedPart {
    /// Mass + transverse principal inertia, derived from the primitive + density.
    pub fn mass_properties(&self) -> (f32, f32) {
        self.primitive.mass_properties(self.density)
    }

    /// Reject a part whose dimensions, placement, or density are non-finite or
    /// non-positive (see [`FittedBody::from_ron`]). Names the part so a bad table
    /// points at the offending entry.
    fn validate(&self) -> Result<(), String> {
        let label = || format!("{:?}", self.part);
        self.primitive
            .validate()
            .map_err(|e| format!("{}: {e}", label()))?;
        if !self.placement.center.is_finite() || !self.placement.rotation.is_finite() {
            return Err(format!("{}: non-finite placement", label()));
        }
        if !(self.density.is_finite() && self.density > 0.0) {
            return Err(format!(
                "{}: density must be finite and > 0, got {}",
                label(),
                self.density
            ));
        }
        Ok(())
    }
}

/// The offline-baked collider table: one [`FittedPart`] per physics link. Built
/// by [`bake_report`] from `sally.glb`, serialized to RON by `--bake-colliders`, and
/// consumed by `body.rs` behind `--body fitted` (the hand-coded body stays the
/// default). A `version` guards against a stale artifact silently feeding a body
/// whose format moved on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedBody {
    /// Artifact schema version. Bump when the meaning of a field changes so an
    /// old `.ron` is rejected loudly instead of mis-spawned.
    pub version: u32,
    pub parts: Vec<FittedPart>,
}

impl FittedBody {
    /// Current artifact schema version.
    pub const VERSION: u32 = 1;

    /// Serialize to pretty RON (the on-disk bake format).
    pub fn to_ron(&self) -> Result<String, String> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| format!("serialize FittedBody: {e}"))
    }

    /// Assemble from per-part [`bake_report`] rows — drops the reasoning, keeping
    /// just the [`FittedPart`]s the artifact stores.
    pub fn from_reports(reports: &[PartReport]) -> Self {
        FittedBody {
            version: Self::VERSION,
            parts: reports.iter().map(|r| r.fitted).collect(),
        }
    }

    /// Parse from RON, rejecting a version mismatch *and* structurally-valid but
    /// physically-impossible values. `--body-table` is a trust boundary (the file
    /// may be hand-edited or foreign), and a serde-valid table with a zero/negative/
    /// NaN dimension parses fine yet spawns a degenerate collider — a 0-radius
    /// capsule, a mirrored box — that detonates rapier deep inside the solver. Fail
    /// here, at the edge, with the offending part named.
    pub fn from_ron(s: &str) -> Result<Self, String> {
        let body: FittedBody = ron::from_str(s).map_err(|e| format!("parse FittedBody: {e}"))?;
        if body.version != Self::VERSION {
            return Err(format!(
                "FittedBody version {} != expected {} — re-bake the collider table",
                body.version,
                Self::VERSION
            ));
        }
        for fp in &body.parts {
            fp.validate()?;
        }
        Ok(body)
    }
}

/// One part's fit with the reasoning behind it: the baked [`FittedPart`], the
/// chooser's decision (primitive + residual + shape descriptors + the rule that
/// fired), and the bind-pose bone span the placement was solved from. Returned by
/// [`bake_report`] so the dev bake can print a reviewable why-this-shape summary
/// that the lean [`FittedBody`] artifact doesn't carry.
#[derive(Clone, Debug)]
pub struct PartReport {
    pub fitted: FittedPart,
    pub choice: PrimitiveChoice,
    /// Proximal→distal bind-pose bone-span length (m) the placement derived from.
    pub span_len: f32,
}

/// Fit the whole body from a loaded model, returning each part's [`FittedPart`]
/// alongside the reasoning behind it. Per part: choose the primitive from its
/// vertex cloud, solve placement from the skeleton bone span, and pair it with
/// the hand-coded density (mass is derived from those, not stored). Parts whose
/// cloud is too small to fit *or* whose skeleton span is missing are skipped —
/// the body loader falls back to the hand-coded link for any part the table
/// omits, so an incomplete fit degrades gracefully instead of spawning a hole.
pub fn bake_report(model: &LoadedModel) -> Vec<PartReport> {
    let by_part = model.vertices_by_part();
    let mut out = Vec::new();
    for (part, density) in super::body::part_densities() {
        let Some((points, _)) = by_part.get(&part) else {
            continue;
        };
        let (Some(choice), Some(span)) = (choose_primitive(points), model.bone_span(part)) else {
            continue;
        };
        let (primitive, placement) = fit_part(part, points, &span);
        out.push(PartReport {
            fitted: FittedPart {
                part,
                primitive,
                placement,
                density,
            },
            choice,
            span_len: (span.distal - span.proximal).length(),
        });
    }
    out
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
                && super::super::rig::part_for_bone(name).is_none()
            {
                unmapped.push((*idx, name.clone()));
            }
        }
        assert!(
            unmapped.is_empty(),
            "bones map to no physics part: {unmapped:?}"
        );
    }

    /// `from_ron` is the `--body-table` trust boundary: a serde-valid table with a
    /// non-finite or non-positive dimension/density must be rejected, not spawned
    /// into a degenerate collider. A clean part round-trips; each poisoned field
    /// fails loudly and names the part.
    #[test]
    fn from_ron_rejects_nonphysical_dims() {
        let good = FittedPart {
            part: PartId::Joint(CrabJointId::LegCoxa(Side::Left, 0)),
            primitive: Primitive::Capsule {
                half_height: 0.1,
                radius: 0.05,
            },
            placement: Placement {
                center: Vec3::ZERO,
                rotation: Quat::IDENTITY,
            },
            density: 1.0,
        };
        let body = |fp: FittedPart| FittedBody {
            version: FittedBody::VERSION,
            parts: vec![fp],
        };
        // A clean table parses.
        FittedBody::from_ron(&body(good).to_ron().unwrap()).expect("clean table parses");

        // Each poisoned field is rejected.
        let poison = [
            Primitive::Capsule {
                half_height: 0.1,
                radius: 0.0,
            }, // zero radius
            Primitive::Capsule {
                half_height: 0.1,
                radius: -0.05,
            }, // negative radius
            Primitive::Cuboid {
                half_extents: Vec3::new(0.1, f32::NAN, 0.1),
            }, // NaN extent
            Primitive::Ball { radius: 0.0 }, // zero radius
        ];
        for prim in poison {
            let ron = body(FittedPart {
                primitive: prim,
                ..good
            })
            .to_ron()
            .unwrap();
            assert!(
                FittedBody::from_ron(&ron).is_err(),
                "from_ron accepted a non-physical primitive: {prim:?}"
            );
        }
        // …and a bad density too.
        let ron = body(FittedPart {
            density: 0.0,
            ..good
        })
        .to_ron()
        .unwrap();
        assert!(
            FittedBody::from_ron(&ron).is_err(),
            "from_ron accepted a zero density"
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
}
