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
//! - [`score_capsule`]/[`score_box`] — signed surface agreement of a live collider
//!   against the cloud it stands in for (the `--verify-colliders` regression gate).

use std::collections::HashMap;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::body::CrabJointId;

/// Which physics link a collider belongs to: the carapace is the root, every other
/// part is a joint's child link. Carries [`CrabJointId`] so a part names the
/// exact joint whose link it stands in for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum PartId {
    Carapace,
    Joint(CrabJointId),
}

impl PartId {
    /// The carapace is the one rigid part — a single fixed box that never deforms;
    /// every other part is an articulated limb link. The skin strip leans on this: a
    /// limb seam vertex may keep a lane on the rigid shell it abuts, but a shell vertex
    /// must never keep a limb lane, or the limb would tug the rigid carapace (the #262
    /// bleed). Naming it here keeps that asymmetry from hiding as a bare `== Carapace`
    /// at the use site, and stays correct if a second rigid part is ever added.
    pub fn is_rigid(self) -> bool {
        matches!(self, PartId::Carapace)
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

/// Where to find the model, resolved the SAME way Bevy's `AssetServer` resolves
/// it for the skin (`crate::bot::skin`): `CRAB_MODEL_PATH` (default `sally.glb`)
/// names an asset under `<asset root>/assets/`, asset root being `BEVY_ASSET_ROOT`
/// or, unset, the crate dir. Sharing this resolution is the whole point — skin and
/// collider-fit must agree on one model. The old code read the var raw against the
/// CWD with a hardcoded `/tmp/rl` fallback, so on any host whose CWD wasn't the
/// asset root the skin loaded but collider-fit missed and the demo exited fatally
/// (bddap/rl#30). Missing model → `None`, and callers self-skip.
pub fn model_path() -> Option<std::path::PathBuf> {
    resolve(
        std::env::var_os("CRAB_MODEL_PATH").as_deref(),
        std::env::var_os("BEVY_ASSET_ROOT").as_deref(),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
        |p| p.exists(),
    )
}

/// Pure resolver, factored out so the path logic is testable without touching
/// process env: an absolute model path is used as-is; otherwise the path (default
/// `sally.glb`) is looked up under `<asset_root or crate_dir>/assets/`.
fn resolve(
    crab_model_path: Option<&std::ffi::OsStr>,
    asset_root: Option<&std::ffi::OsStr>,
    crate_dir: &std::path::Path,
    exists: impl Fn(&std::path::Path) -> bool,
) -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let rel = crab_model_path.map_or_else(|| PathBuf::from("sally.glb"), PathBuf::from);
    if rel.is_absolute() {
        return exists(&rel).then_some(rel);
    }
    let root = asset_root.map_or_else(|| crate_dir.to_path_buf(), PathBuf::from);
    let asset = root.join("assets").join(rel);
    exists(&asset).then_some(asset)
}

#[cfg(test)]
mod model_path_tests {
    use super::resolve;
    use std::path::{Path, PathBuf};

    // A relative CRAB_MODEL_PATH must land under <asset root>/assets — the exact
    // path the AssetServer hands the skin (bddap/rl#30).
    #[test]
    fn relative_resolves_under_asset_root() {
        let got = resolve(
            Some("sally.glb".as_ref()),
            Some("/srv/app".as_ref()),
            Path::new("/crate"),
            |p| p == Path::new("/srv/app/assets/sally.glb"),
        );
        assert_eq!(got, Some(PathBuf::from("/srv/app/assets/sally.glb")));
    }

    #[test]
    fn defaults_to_sally_under_crate_dir_when_unset() {
        let got = resolve(None, None, Path::new("/crate"), |p| {
            p == Path::new("/crate/assets/sally.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/crate/assets/sally.glb")));
    }

    #[test]
    fn absolute_path_used_as_is() {
        let got = resolve(
            Some("/models/x.glb".as_ref()),
            Some("/srv".as_ref()),
            Path::new("/crate"),
            |p| p == Path::new("/models/x.glb"),
        );
        assert_eq!(got, Some(PathBuf::from("/models/x.glb")));
    }

    #[test]
    fn none_when_missing() {
        assert_eq!(
            resolve(
                Some("sally.glb".as_ref()),
                Some("/srv".as_ref()),
                Path::new("/crate"),
                |_| false
            ),
            None
        );
    }
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
        compose_world(gltf.scenes().flat_map(|s| s.nodes()), &mut bind_world);

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
                let pos = skin_to_bind_world(raw, *j, *w, &bind_world, &joint_nodes, &inv_binds)?;
                // Dominant influence tags the vertex for per-part bucketing.
                let (lane, &wmax) = w
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                let dom = j[lane] as usize;
                verts.push(SkinnedVertex {
                    pos,
                    dominant_node: *joint_nodes.get(dom).ok_or_else(|| {
                        format!(
                            "JOINTS_0 index {dom} out of range (skin has {} joints)",
                            joint_nodes.len()
                        )
                    })?,
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

    /// Linear scan over the node table; the model has a few hundred nodes, so callers
    /// that resolve many bones stay cheap enough without an inverse map.
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
    compose_world(gltf.scenes().flat_map(|s| s.nodes()), &mut bind_world);

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
            )?);
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
) -> Result<Vec3, String> {
    let mut world = Vec3::ZERO;
    let mut wsum = 0.0f32;
    for lane in 0..4 {
        let wt = weights[lane];
        if wt <= 0.0 {
            continue;
        }
        // `ji` is a raw JOINTS_0 value straight off the mesh stream; a malformed
        // asset can put it past the skin's joint list, so bound it rather than let
        // the slice index panic deep in the loader (both arrays are joint-indexed).
        let ji = joints[lane] as usize;
        let &node = joint_nodes.get(ji).ok_or_else(|| {
            format!(
                "JOINTS_0 index {ji} out of range (skin has {} joints)",
                joint_nodes.len()
            )
        })?;
        let inv_bind = inv_binds.get(ji).copied().unwrap_or(Mat4::IDENTITY);
        let jm = bind_world.get(&node).copied().unwrap_or(Mat4::IDENTITY) * inv_bind;
        world += wt * jm.transform_point3(raw);
        wsum += wt;
    }
    Ok(if wsum > 1e-6 { world / wsum } else { raw })
}

/// Compose `parent_world * node_local` over every scene-root subtree, recording each
/// node's bind-world matrix. glTF node transforms are TRS or a raw matrix;
/// `node.transform().matrix()` normalises both to a column-major 4x4.
///
/// Iterative with an explicit work stack rather than recursion: glTF's `children()`
/// is just an index list, so a malformed asset can encode a cycle (or a chain deep
/// enough to blow the native stack), which recursion would turn into an abort the
/// train's auto-resume loop can't escape. The `visited` set both breaks cycles and
/// keeps the first (root-most) world transform a node gets — a valid scene is a
/// forest, so this is identical to the recursive walk there.
fn compose_world<'a>(
    roots: impl Iterator<Item = gltf::Node<'a>>,
    world: &mut HashMap<usize, Mat4>,
) {
    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut stack: Vec<(gltf::Node<'a>, Mat4)> = roots.map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent_world)) = stack.pop() {
        if !visited.insert(node.index()) {
            continue;
        }
        let w = parent_world * Mat4::from_cols_array_2d(&node.transform().matrix());
        world.insert(node.index(), w);
        for child in node.children() {
            stack.push((child, w));
        }
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
    /// Test-only: production reads the endpoints (`a`/`b`) directly; this is the
    /// convenience the capsule-fit test uses.
    pub fn segment_len(&self) -> f32 {
        (self.b - self.a).length()
    }
}

/// Perpendicular-spread percentile that sets the capsule radius. A high percentile
/// (not the max) so a thin tail of skinning-bled vertices doesn't inflate the radius.
/// 0.92 rather than 0.95 because the leg-coxa clouds carry a fatter outlier tail than
/// 5%: from p90 to p95 the middle-leg (legs 1–2) coxa radius jumps ~15–20% on a bulk
/// (p50) thickness that's near-identical across all eight legs — a fit artifact, not
/// real flesh. That extra envelope made adjacent middle-leg coxae interpenetrate ~64mm
/// at the settled rest pose, a penetration the solver fights into a visible body sag
/// (`--check-rest-colliders`); trimming the tail deflates the over-fit capsules in
/// proportion to their own tail (well-fit legs barely move), cutting that overlap ~28%
/// and letting the body settle higher. 0.92 is the floor that stays self-consistent
/// with the `--verify-colliders` radius-vs-spread gate: that gate measures the live
/// radius against the cloud's p95 spread and flags it "starved" below 0.85×, so a fit
/// percentile under ~0.90 would make a capsule fail the very gate that sized it.
const RADIUS_PERCENTILE: f32 = 0.92;

/// Fit a capsule to a point cloud:
///   - axis = first principal component (largest-variance direction),
///   - radius = [`RADIUS_PERCENTILE`] of perpendicular distance to that axis
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
    // Radius from a high perpendicular-spread percentile: robust to a tail of bled
    // vertices (see RADIUS_PERCENTILE for why that exact percentile, not the max).
    perp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let radius = percentile(&perp, RADIUS_PERCENTILE).max(1e-4);

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

// ---------------------------------------------------------------------------
// Point-in-mesh queries shared by the collider/pivot verifiers
// (`--verify-colliders`, `--verify-pivots`) and the skin-diag audit. Pure
// functions of a triangle soup, with no collider-fitting state.
// ---------------------------------------------------------------------------

/// Generalized winding number of a triangle soup at `p`: `(1/4π)·Σ` of each
/// triangle's signed solid angle, via the Van Oosterom–Strackee `atan2` formula.
/// ≈1 (or ≈−1 for the opposite global winding) inside a closed surface, ≈0 outside,
/// and — unlike parity ray-casting — degrades gracefully on a non-watertight mesh
/// (fractional values reveal exactly how open it is). Sign depends on triangle
/// orientation; the caller normalises it via the soup's signed volume sign so an
/// interior point reads +1 regardless of CW/CCW winding.
pub(crate) fn winding_number(p: Vec3, positions: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    let mut acc = 0.0f64;
    for t in tris {
        let a = positions[t[0] as usize] - p;
        let b = positions[t[1] as usize] - p;
        let c = positions[t[2] as usize] - p;
        let (la, lb, lc) = (a.length() as f64, b.length() as f64, c.length() as f64);
        let num = a.dot(b.cross(c)) as f64;
        let den =
            la * lb * lc + (a.dot(b) as f64) * lc + (b.dot(c) as f64) * la + (c.dot(a) as f64) * lb;
        acc += 2.0 * num.atan2(den);
    }
    (acc / (4.0 * std::f64::consts::PI)) as f32
}

/// Signed volume of the triangle soup (∑ of per-triangle tetrahedron volumes
/// `v0·(v1×v2)/6`). Its SIGN fixes the global winding convention without needing an
/// interior reference point: CCW-outward triangles give a positive volume and make
/// interior winding numbers read +1; CW give negative and −1. Robust where a single
/// "is this point inside?" probe isn't — the crab's vertex centroid lands in a
/// cavity, so it can't be trusted to orient the sign.
pub(crate) fn mesh_signed_volume(positions: &[Vec3], tris: &[[u32; 3]]) -> f64 {
    let mut acc = 0.0f64;
    for t in tris {
        let v0 = positions[t[0] as usize];
        let v1 = positions[t[1] as usize];
        let v2 = positions[t[2] as usize];
        acc += v0.dot(v1.cross(v2)) as f64 / 6.0;
    }
    acc
}

/// Unsigned distance from `p` to a triangle (`v0`,`v1`,`v2`) — the standard
/// closest-point-on-triangle clamp. Used to report HOW FAR a query point sits from
/// the mesh surface (the winding number gives inside/outside; this gives the depth).
fn point_tri_distance(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> f32 {
    let ab = v1 - v0;
    let ac = v2 - v0;
    let ap = p - v0;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - v1;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let cp = p - v2;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (v0 + ab * v - p).length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (v0 + ac * w - p).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (v1 + (v2 - v1) * w - p).length();
    }
    // Interior of the face: project onto its plane via barycentric weights.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (v0 + ab * v + ac * w - p).length()
}

/// Nearest unsigned surface distance from `p` over the whole triangle soup.
pub(crate) fn nearest_surface_distance(p: Vec3, positions: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    let mut best = f32::INFINITY;
    for t in tris {
        let d = point_tri_distance(
            p,
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        if d < best {
            best = d;
        }
    }
    best
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
                "meshfit: no model (set CRAB_MODEL_PATH under BEVY_ASSET_ROOT/assets) — skipping"
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

    /// A JOINTS_0 value past the skin's joint list (the kind a malformed asset can
    /// carry) must surface as `Err`, not a panic indexing the joint array — so the
    /// `Result<_, String>` loaders can report it cleanly. Exercises the lookup
    /// directly; hand-building a malformed GLB to drive it through `load` would be
    /// far more code for the same guard.
    #[test]
    fn out_of_range_joint_index_errors() {
        let joint_nodes = [7usize]; // one joint
        let inv_binds = [Mat4::IDENTITY];
        let bind_world = HashMap::new();
        let got = skin_to_bind_world(
            Vec3::ZERO,
            [3, 0, 0, 0], // lane 0 points at joint 3 — out of range
            [1.0, 0.0, 0.0, 0.0],
            &bind_world,
            &joint_nodes,
            &inv_binds,
        );
        assert!(
            got.is_err(),
            "out-of-range joint index should Err, got {got:?}"
        );
    }
}
