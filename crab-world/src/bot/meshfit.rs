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

mod containment;
mod fit;
mod mass;

// Re-export every item previously reachable as `meshfit::Foo`, so the split is
// invisible to external callers. `pub(crate)`/`#[cfg(test)]` visibility carries
// through the re-export unchanged.
pub use containment::{Containment, MeshContainment, aabb};
pub use fit::{
    CapsuleDiagnostics, ColliderScore, FittedCapsule, fit_capsule, score_box, score_capsule,
};
#[cfg(test)]
pub use mass::{CapsuleMass, capsule_mass};

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
/// names an asset under [`crate::assets::asset_root`]`/assets/`. The asset root is
/// the ONE source of truth ([`crate::assets::asset_root`]) — the bevy glyph load,
/// the skin, and this collider-fit loader all resolve against it, so they cannot
/// disagree about where assets live (bddap/rl#146). Missing model → `None`, and
/// callers self-skip.
pub fn model_path() -> Option<std::path::PathBuf> {
    resolve(
        std::env::var_os("CRAB_MODEL_PATH").as_deref(),
        &crate::assets::asset_root(),
        |p| p.exists(),
    )
}

/// FNV-1a/64 digest of the crab MODEL file's raw bytes (rl#100, GCR), or `0` when no
/// model resolves — the per-peer "same collider asset?" check the membership handshake
/// advertises alongside the policy-weights digest.
///
/// WHY hash the raw file bytes (not the post-load mesh or the fitted capsule spec): the
/// giant crab's rapier colliders are a DETERMINISTIC pure function of this file's bytes
/// GIVEN a fixed binary — `sally.glb` → [`ParsedGltf::open`] → [`skin_to_bind_world`] →
/// [`fit_capsule`]/box → `Collider::capsule`/`cuboid`. The binary is already an unstated
/// GCR baseline (two peers on different binaries desync on everything, not just colliders),
/// so the ONLY collider-affecting input not yet guarded is the asset file itself. Hashing
/// the bytes captures exactly that input — conservatively (any byte change ⇒ a mismatch ⇒ the
/// round refuses to arm the NN crab, rl#114) and WITHOUT re-introducing the float-reproducibility
/// question that hashing the skinned vertex cloud or the fitted f32 capsule would. Uses the
/// shared [`crate::fnv::fnv1a`] — the same build-stable FNV-1a/64 the other GCR digests use,
/// so two same-binary peers with byte-identical assets agree.
pub fn crab_asset_digest() -> u64 {
    let Some(path) = model_path() else {
        return 0; // no model resolves → "no collider asset", never counts as synced
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return 0; // unreadable → treat as no asset (a `0` digest never counts as synced → refuse)
    };
    crate::fnv::fnv1a(&bytes)
}

/// Pure resolver, factored out so the path logic is testable without touching
/// process env: an absolute model path is used as-is; otherwise the path (default
/// `sally.glb`) is looked up under `<asset_root>/assets/`. The `asset_root` is the
/// already-resolved root ([`crate::assets::asset_root`]) — this function never
/// re-derives it, so there is one place the root is decided.
fn resolve(
    crab_model_path: Option<&std::ffi::OsStr>,
    asset_root: &std::path::Path,
    exists: impl Fn(&std::path::Path) -> bool,
) -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let rel = crab_model_path.map_or_else(|| PathBuf::from("sally.glb"), PathBuf::from);
    if rel.is_absolute() {
        return exists(&rel).then_some(rel);
    }
    let asset = asset_root.join("assets").join(rel);
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
        let got = resolve(Some("sally.glb".as_ref()), Path::new("/srv/app"), |p| {
            p == Path::new("/srv/app/assets/sally.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/srv/app/assets/sally.glb")));
    }

    #[test]
    fn defaults_to_sally_under_asset_root() {
        let got = resolve(None, Path::new("/crate"), |p| {
            p == Path::new("/crate/assets/sally.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/crate/assets/sally.glb")));
    }

    #[test]
    fn absolute_path_used_as_is() {
        let got = resolve(Some("/models/x.glb".as_ref()), Path::new("/srv"), |p| {
            p == Path::new("/models/x.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/models/x.glb")));
    }

    #[test]
    fn none_when_missing() {
        assert_eq!(
            resolve(Some("sally.glb".as_ref()), Path::new("/srv"), |_| false),
            None
        );
    }
}

/// A GLB opened and decoded down to the skinning inputs both loaders share: the
/// binary blob, the bind-pose world transforms, and the skin's joint-node list +
/// inverse-bind matrices. [`LoadedModel::load`] and [`load_bind_mesh`] read the same
/// model for different end-products (per-part vertex clouds vs a triangle soup) but
/// must skin identically — folding the open/compose/skin-decode here is what stops
/// the two from drifting (they previously duplicated it and had to change in lockstep).
struct ParsedGltf {
    gltf: gltf::Gltf,
    blob: Vec<u8>,
    /// Bind-pose world transform of each node, by node index.
    bind_world: HashMap<usize, Mat4>,
    /// Skin joint node indices, in skin-joint order; a per-vertex JOINTS_0 value
    /// indexes this list.
    joint_nodes: Vec<usize>,
    /// Inverse-bind matrices in the same skin-joint order as `joint_nodes`.
    inv_binds: Vec<Mat4>,
}

impl ParsedGltf {
    /// Open the GLB and decode the inputs common to both loaders. The bind pose IS
    /// the node rest pose (no animation sampling) — the same pose the skin captures
    /// offsets in.
    fn open(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("parse glb: {e}"))?;
        let blob = gltf.blob.clone().ok_or("GLB has no binary chunk")?;

        // Bind-pose world transforms: walk every scene root down, composing
        // `parent_world * node_local`.
        let mut bind_world: HashMap<usize, Mat4> = HashMap::new();
        compose_world(gltf.scenes().flat_map(|s| s.nodes()), &mut bind_world);

        // A skin's `joints()` list maps the per-vertex JOINTS_0 *indices* (0..N) to
        // actual node indices.
        let skin = gltf.skins().next().ok_or("model has no skin")?;
        let joint_nodes: Vec<usize> = skin.joints().map(|j| j.index()).collect();
        // Inverse-bind matrices (skin-joint order). Combined with the joints'
        // bind-world transforms they map a raw mesh vertex to its bind-pose WORLD
        // position — the position Bevy actually skins the visible mesh to. Fitting
        // colliders to that (not the raw, pre-skin POSITION attribute) is what keeps
        // them on the rendered mesh rather than offset by the mesh/armature frame.
        // glTF lets a skin omit the IBMs to mean "all identity" — that is the only
        // legitimate empty case, so synthesise a full identity set of the right
        // length; `skin_to_bind_world` then treats a SHORTER array as the error it is.
        let inv_binds: Vec<Mat4> = skin
            .reader(|buf| (buf.index() == 0).then_some(blob.as_slice()))
            .read_inverse_bind_matrices()
            .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_else(|| vec![Mat4::IDENTITY; joint_nodes.len()]);

        Ok(ParsedGltf {
            gltf,
            blob,
            bind_world,
            joint_nodes,
            inv_binds,
        })
    }

    /// The single mesh's primitives. Both loaders read POSITION/JOINTS_0/WEIGHTS_0
    /// off these and skin via [`skin_to_bind_world`]; only their per-vertex output
    /// differs.
    fn primitives(&self) -> Result<gltf::mesh::iter::Primitives<'_>, String> {
        Ok(self
            .gltf
            .meshes()
            .next()
            .ok_or("model has no mesh")?
            .primitives())
    }

    /// Read a primitive's POSITION/JOINTS_0/WEIGHTS_0 and skin each vertex to its
    /// bind-pose WORLD position. Returns the skinned positions paired with their raw
    /// joints+weights so a caller can derive its own per-vertex product (dominant
    /// bone, triangle indices, …) without re-skinning.
    #[allow(clippy::type_complexity)]
    fn skin_primitive(
        &self,
        prim: &gltf::Primitive<'_>,
    ) -> Result<Vec<(Vec3, [u16; 4], [f32; 4])>, String> {
        let reader = prim.reader(|buf| (buf.index() == 0).then_some(self.blob.as_slice()));
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
        positions
            .iter()
            .zip(&joints)
            .zip(&weights)
            .map(|((p, &j), &w)| {
                let pos = skin_to_bind_world(
                    Vec3::from_array(*p),
                    j,
                    w,
                    &self.bind_world,
                    &self.joint_nodes,
                    &self.inv_binds,
                )?;
                Ok((pos, j, w))
            })
            .collect()
    }
}

impl LoadedModel {
    /// Parse a GLB from disk: decode the skin's bind matrices into world bone
    /// transforms and every vertex's dominant bone. Errors surface as a string
    /// so the test can fail loudly rather than panic deep in the gltf crate.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let parsed = ParsedGltf::open(path)?;
        let node_name: HashMap<usize, String> = parsed
            .gltf
            .nodes()
            .filter_map(|n| n.name().map(|nm| (n.index(), nm.to_string())))
            .collect();

        let mut verts = Vec::new();
        for prim in parsed.primitives()? {
            for (pos, j, w) in parsed.skin_primitive(&prim)? {
                // Dominant influence tags the vertex for per-part bucketing.
                let (lane, _) = w
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                let dom = j[lane] as usize;
                verts.push(SkinnedVertex {
                    pos,
                    dominant_node: *parsed.joint_nodes.get(dom).ok_or_else(|| {
                        format!(
                            "JOINTS_0 index {dom} out of range (skin has {} joints)",
                            parsed.joint_nodes.len()
                        )
                    })?,
                });
            }
        }

        Ok(LoadedModel {
            bind_world: parsed.bind_world,
            node_name,
            verts,
        })
    }

    /// Group vertices by physics part via [`super::rig::part_for_bone`] — the one
    /// canonical bone→part mapping. Returns the world-space vertex positions per
    /// part. Vertices on a non-rig node (no part) are dropped.
    pub fn vertices_by_part(&self) -> HashMap<PartId, Vec<Vec3>> {
        let mut out: HashMap<PartId, Vec<Vec3>> = HashMap::new();
        for v in &self.verts {
            let Some(name) = self.node_name.get(&v.dominant_node) else {
                continue;
            };
            let Some(part) = super::rig::part_for_bone(name) else {
                continue;
            };
            out.entry(part).or_default().push(v.pos);
        }
        out
    }

    /// World-space positions of every vertex whose dominant bone is one of `names`.
    /// Used to size the carapace box from the trunk's actual shell flesh.
    pub fn vertices_for_bones(&self, names: &[&str]) -> Vec<Vec3> {
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
    let parsed = ParsedGltf::open(path)?;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for prim in parsed.primitives()? {
        let base = positions.len() as u32;
        let skinned = parsed.skin_primitive(&prim)?;
        let prim_verts = skinned.len() as u32;
        positions.extend(skinned.into_iter().map(|(pos, _, _)| pos));

        // Offset this primitive's indices into the global vertex range. An indexless
        // primitive is an implicit 0,1,2,… triangle list over its own vertices.
        let reader = prim.reader(|buf| (buf.index() == 0).then_some(parsed.blob.as_slice()));
        let local: Vec<u32> = match reader.read_indices() {
            Some(idx) => idx.into_u32().collect(),
            None => (0..prim_verts).collect(),
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
        // `ji` is a raw JOINTS_0 value straight off the mesh stream. Three things
        // must hold for it to skin correctly, and a parseable-but-malformed asset can
        // break any of them; all three fail loud (return `Err`) rather than indexing
        // out of bounds or substituting identity, which would skin this vertex to the
        // wrong place SILENTLY. The glTF spec guarantees all three for a valid skin —
        // the joint index is in range, there is one IBM per joint, and every joint is
        // reachable from a scene root — so each gap is the asset's error to surface.
        let ji = joints[lane] as usize;
        let &node = joint_nodes.get(ji).ok_or_else(|| {
            format!(
                "JOINTS_0 index {ji} out of range (skin has {} joints)",
                joint_nodes.len()
            )
        })?;
        let inv_bind = inv_binds.get(ji).copied().ok_or_else(|| {
            format!(
                "skin has no inverse-bind matrix for joint index {ji} ({} IBMs, {} joints)",
                inv_binds.len(),
                joint_nodes.len()
            )
        })?;
        let bind = bind_world.get(&node).copied().ok_or_else(|| {
            format!("joint index {ji} (node {node}) is not reachable from any scene root")
        })?;
        let jm = bind * inv_bind;
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

    /// An inverse-bind-matrix array shorter than the joint list (a malformed skin —
    /// the glTF spec requires one IBM per joint) must `Err`, not silently substitute
    /// identity and skin the vertex to the wrong place. The joint index is IN range
    /// here, so this isolates the IBM-length guard from the out-of-range one above.
    #[test]
    fn missing_inverse_bind_matrix_errors() {
        let joint_nodes = [7usize]; // one joint, in range
        let inv_binds: [Mat4; 0] = []; // but no IBM for it
        let mut bind_world = HashMap::new();
        bind_world.insert(7usize, Mat4::IDENTITY);
        let got = skin_to_bind_world(
            Vec3::ZERO,
            [0, 0, 0, 0],
            [1.0, 0.0, 0.0, 0.0],
            &bind_world,
            &joint_nodes,
            &inv_binds,
        );
        assert!(
            got.is_err(),
            "missing inverse-bind matrix should Err, got {got:?}"
        );
    }

    /// A joint not reachable from any scene root (absent from `bind_world` — a
    /// malformed skin) must `Err` rather than silently skin against identity. The
    /// joint index is in range and has an IBM, isolating the reachability guard.
    #[test]
    fn unreachable_joint_node_errors() {
        let joint_nodes = [7usize]; // joint node 7, in range, with an IBM…
        let inv_binds = [Mat4::IDENTITY];
        let bind_world = HashMap::new(); // …but node 7 never got a bind-world transform
        let got = skin_to_bind_world(
            Vec3::ZERO,
            [0, 0, 0, 0],
            [1.0, 0.0, 0.0, 0.0],
            &bind_world,
            &joint_nodes,
            &inv_binds,
        );
        assert!(
            got.is_err(),
            "unreachable joint node should Err, got {got:?}"
        );
    }
}
