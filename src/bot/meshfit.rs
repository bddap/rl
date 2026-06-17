//! glTF skeleton loader + the collider-fit primitives the live body is built from.
//!
//! [`LoadedModel`] parses `sally.glb` (the same model the cosmetic skin and the
//! rig-derived body read) into bind-pose bone transforms + skinned vertices, and
//! exposes the per-part vertex clouds and bone origins [`super::rig`] reads to spawn
//! the body. The fit helpers operate on those clouds, all in the glTF's *bind-pose
//! world* frame:
//!
//! - [`fit_capsule`] — principal-axis capsule for a limb segment ([`super::rig`]
//!   sizes each leg/claw link with it at spawn).
//! - [`containing_obb`] — tightest oriented box enclosing a cloud ([`super::rig`]
//!   fits the carapace shell with it).
//! - [`score_capsule`]/[`score_box`] — signed surface agreement of a live collider
//!   against the cloud it stands in for (the `--verify-colliders` regression gate).

use std::collections::HashMap;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::body::CrabJointId;

/// Which physics link a collider belongs to: the carapace is the root, every other
/// part is a joint's child link. Mirrors the skin's `LinkKey` but carries
/// [`CrabJointId`] so a part names the exact joint whose link it stands in for.
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
                let pos = skin_to_bind_world(raw, *j, *w, &bind_world, &joint_nodes, &inv_binds);
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
    /// link rotates *with* its bone. The rig-driven body spawn ([`super::rig`]) reads
    /// the origin as the joint pivot to place each link.
    pub(crate) fn bone_bind_pose(&self, name: &str) -> Option<(Vec3, Quat)> {
        let idx = self.node_index(name)?;
        self.bind_world.get(&idx).map(|m| {
            let (_, rot, trans) = m.to_scale_rotation_translation();
            (trans, rot)
        })
    }
}

/// The model's surface as a triangle soup in bind-pose WORLD space: every vertex
/// skinned to where the renderer puts it, plus the triangle index list. Unlike
/// [`LoadedModel`]'s per-part `verts` (bucketed, no connectivity), this keeps the
/// global vertex order and the indices, so the triangles can be reconstructed for a
/// point-in-mesh test. Multiple mesh primitives are concatenated with their indices
/// offset, yielding one global soup.
pub struct BindMesh {
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

/// Parse a GLB into its bind-pose-world triangle soup ([`BindMesh`]): same skin math
/// as [`LoadedModel::load`] (via [`skin_to_bind_world`]) so the surface lands in the
/// exact frame the per-part clouds and bone origins live in, but retaining triangle
/// connectivity. A primitive with no index buffer is treated as a flat triangle list
/// (indices 0,1,2,…); per-primitive indices are offset by the running vertex base so
/// the concatenated soup stays consistent.
pub fn load_bind_mesh(path: &std::path::Path) -> Result<BindMesh, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("parse glb: {e}"))?;
    let blob = gltf.blob.as_deref().ok_or("GLB has no binary chunk")?;

    let mut bind_world: HashMap<usize, Mat4> = HashMap::new();
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            compose_world(&node, Mat4::IDENTITY, &mut bind_world);
        }
    }

    let skin = gltf.skins().next().ok_or("model has no skin")?;
    let joint_nodes: Vec<usize> = skin.joints().map(|j| j.index()).collect();
    let inv_binds: Vec<Mat4> = skin
        .reader(|buf| (buf.index() == 0).then_some(blob))
        .read_inverse_bind_matrices()
        .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
        .unwrap_or_else(|| vec![Mat4::IDENTITY; joint_nodes.len()]);

    let mesh = gltf.meshes().next().ok_or("model has no mesh")?;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for prim in mesh.primitives() {
        let reader = prim.reader(|buf| (buf.index() == 0).then_some(blob));
        let raw_positions: Vec<[f32; 3]> = reader
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

        let base = positions.len() as u32;
        for ((p, j), w) in raw_positions.iter().zip(&joints).zip(&weights) {
            positions.push(skin_to_bind_world(
                Vec3::from_array(*p),
                *j,
                *w,
                &bind_world,
                &joint_nodes,
                &inv_binds,
            ));
        }

        // Offset this primitive's indices into the global vertex range. An indexless
        // primitive is an implicit 0,1,2,… triangle list over its own vertices.
        let local: Vec<u32> = match reader.read_indices() {
            Some(idx) => idx.into_u32().collect(),
            None => (0..raw_positions.len() as u32).collect(),
        };
        for tri in local.chunks_exact(3) {
            triangles.push([base + tri[0], base + tri[1], base + tri[2]]);
        }
    }

    Ok(BindMesh {
        positions,
        triangles,
    })
}

/// Skin one raw mesh vertex to its bind-pose WORLD position: the weighted blend of
/// each influencing joint's `bindWorld · invBind`, the same linear-blend skinning
/// the renderer runs. Factored out so the per-part cloud ([`LoadedModel::load`]) and
/// the triangle-soup loader ([`load_bind_mesh`]) skin identically — a point-in-mesh
/// test is only meaningful if its query points and its surface share one frame, so
/// the two paths must not drift.
fn skin_to_bind_world(
    raw: Vec3,
    joints: [u16; 4],
    weights: [f32; 4],
    bind_world: &HashMap<usize, Mat4>,
    joint_nodes: &[usize],
    inv_binds: &[Mat4],
) -> Vec3 {
    let mut world = Vec3::ZERO;
    let mut wsum = 0.0f32;
    for lane in 0..4 {
        let wt = weights[lane];
        if wt <= 0.0 {
            continue;
        }
        let ji = joints[lane] as usize;
        let jm = bind_world
            .get(&joint_nodes[ji])
            .copied()
            .unwrap_or(Mat4::IDENTITY)
            * inv_binds[ji];
        world += wt * jm.transform_point3(raw);
        wsum += wt;
    }
    if wsum > 1e-6 { world / wsum } else { raw }
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

#[cfg(test)]
impl FittedCapsule {
    /// Distance between the segment endpoints. Test-only: production reads the
    /// endpoints (`a`/`b`) directly; this is the convenience the capsule-fit test uses.
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
    // top two variances are close — a chunky near-isotropic cloud — this axis is
    // ill-defined, so a capsule is the wrong primitive there; the carapace takes a
    // box instead.)
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

/// How well a *live* collider surface hugs the vertex cloud it stands in for, in
/// model units, in the cloud's own (bind-pose world) frame. It keeps the sign and
/// scores the geometry the body actually spawns with: positive
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

/// Score a cloud against an oriented box (centre + half-extents + box→world
/// rotation). Each point is brought into the box's own frame (`rot⁻¹·(p−center)`)
/// before the half-extent test, so the signed surface distance is measured against
/// the *oriented* faces. Pass `Quat::IDENTITY` for an axis-aligned box.
pub(crate) fn score_box(points: &[Vec3], center: Vec3, half: Vec3, rot: Quat) -> ColliderScore {
    let to_local = rot.inverse();
    let sd: Vec<f32> = points
        .iter()
        .map(|&p| {
            let local = (to_local * (p - center)).abs() - half;
            let outside = local.max(Vec3::ZERO).length();
            let inside = local.max_element().min(0.0);
            outside + inside
        })
        .collect();
    ColliderScore::from_signed(&sd)
}

// ---------------------------------------------------------------------------
// PCA frame + oriented bounding box (the carapace's collider fit)
// ---------------------------------------------------------------------------

/// Tightest oriented box that *contains* every point, as `(half_extents, center,
/// rotation)` where `rotation` maps a canonical axis-aligned box's local axes onto
/// the cloud's principal axes (so `rapier`'s `Collider::cuboid` + this rotation
/// reproduces the fit). Uses the full min/max span per principal axis — no
/// percentile trimming — and recentres on the span midpoint rather than the
/// centroid: the carapace box must enclose the shell so the mesh never pokes out,
/// and a skewed dome's centroid isn't its span centre. The
/// principal frame is forced right-handed (eigenvectors can come out left-handed)
/// so the basis is a valid rotation.
pub(crate) fn containing_obb(points: &[Vec3]) -> Option<(Vec3, Vec3, Quat)> {
    if points.len() < 4 {
        return None;
    }
    let centroid = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let (axes_pca, _vars) = covariance_eigenframe(points, centroid);
    // A left-handed eigenframe (det < 0) would make `from_mat3` yield a reflection,
    // not a rotation; flip the least-significant axis to restore right-handedness.
    let axes = if axes_pca[0].cross(axes_pca[1]).dot(axes_pca[2]) < 0.0 {
        [axes_pca[0], axes_pca[1], -axes_pca[2]]
    } else {
        axes_pca
    };
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &p in points {
        let d = p - centroid;
        let proj = Vec3::new(d.dot(axes[0]), d.dot(axes[1]), d.dot(axes[2]));
        lo = lo.min(proj);
        hi = hi.max(proj);
    }
    let half = ((hi - lo) * 0.5).max(Vec3::splat(1e-4));
    let mid = (hi + lo) * 0.5; // span centre, in principal coords relative to centroid
    let rotation = Quat::from_mat3(&Mat3::from_cols(axes[0], axes[1], axes[2]));
    // Map the principal-frame span centre back to world: centroid + Σ axis_k·mid_k.
    let center = centroid + axes[0] * mid.x + axes[1] * mid.y + axes[2] * mid.z;
    Some((half, center, rotation))
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
/// `i_perp` about a transverse axis through the centre of mass.
//
// Test-only: the analytic mass formula has no production caller — the live body
// lets rapier compute inertia from the spawned collider + density — but it is the
// independent ground truth `capsule_reduces_to_sphere` checks the closed forms
// against, so it stays gated to the test build rather than shipping unused.
#[cfg(test)]
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
#[cfg(test)]
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
