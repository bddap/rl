//! Skinned-glTF loading for the offline fitter: bind-pose vertex clouds per physics
//! part, bone bind poses, and the watertight bind mesh. Moved out of `crab-world`
//! with the rest of the fitter (bddap/rl#20 Phase 2) — the runtime never parses the
//! glb for physics anymore; it digests the bytes and reads the baked table.

use std::collections::HashMap;

use bevy::prelude::*;
use crab_world::bot::rig::{self, PartId};

struct SkinnedVertex {
    pos: Vec3,
    dominant_node: usize,
}

pub struct LoadedModel {
    bind_world: HashMap<usize, Mat4>,
    node_name: HashMap<usize, String>,
    verts: Vec<SkinnedVertex>,
}

struct ParsedGltf {
    gltf: gltf::Gltf,
    blob: Vec<u8>,
    bind_world: HashMap<usize, Mat4>,
    joint_nodes: Vec<usize>,
    inv_binds: Vec<Mat4>,
}

impl ParsedGltf {
    fn open(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        Self::from_slice(&bytes)
    }

    fn from_slice(bytes: &[u8]) -> Result<Self, String> {
        let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| format!("parse glb: {e}"))?;
        let blob = gltf.blob.clone().ok_or("GLB has no binary chunk")?;

        let mut bind_world: HashMap<usize, Mat4> = HashMap::new();
        compose_world(gltf.scenes().flat_map(|s| s.nodes()), &mut bind_world);

        let skin = gltf.skins().next().ok_or("model has no skin")?;
        let joint_nodes: Vec<usize> = skin.joints().map(|j| j.index()).collect();
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

    fn primitives(&self) -> Result<gltf::mesh::iter::Primitives<'_>, String> {
        Ok(self
            .gltf
            .meshes()
            .next()
            .ok_or("model has no mesh")?
            .primitives())
    }

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
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        Self::from_slice(&bytes)
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self, String> {
        let parsed = ParsedGltf::from_slice(bytes)?;
        let node_name: HashMap<usize, String> = parsed
            .gltf
            .nodes()
            .filter_map(|n| n.name().map(|nm| (n.index(), nm.to_string())))
            .collect();

        let mut verts = Vec::new();
        for prim in parsed.primitives()? {
            for (pos, j, w) in parsed.skin_primitive(&prim)? {
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

    pub fn vertices_by_part(&self) -> HashMap<PartId, Vec<Vec3>> {
        let mut out: HashMap<PartId, Vec<Vec3>> = HashMap::new();
        for v in &self.verts {
            let Some(name) = self.node_name.get(&v.dominant_node) else {
                continue;
            };
            let Some(part) = rig::part_for_bone(name) else {
                continue;
            };
            out.entry(part).or_default().push(v.pos);
        }
        out
    }

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

    pub fn bone_origin(&self, name: &str) -> Option<Vec3> {
        self.bone_bind_pose(name).map(|(o, _)| o)
    }

    fn node_index(&self, name: &str) -> Option<usize> {
        self.node_name
            .iter()
            .find(|(_, nm)| nm.as_str() == name)
            .map(|(i, _)| *i)
    }

    fn bone_bind_pose(&self, name: &str) -> Option<(Vec3, Quat)> {
        let idx = self.node_index(name)?;
        self.bind_world.get(&idx).map(|m| {
            let (_, rot, trans) = m.to_scale_rotation_translation();
            (trans, rot)
        })
    }

    /// Bones the model names that map to no physics part — must be empty for a
    /// bake to be honest (an unmapped bone's flesh would fit into nothing).
    pub fn unmapped_bones(&self) -> Vec<String> {
        self.node_name
            .values()
            .filter(|name| {
                (name.starts_with("Def_") || name.starts_with("Ctrl_"))
                    && rig::part_for_bone(name).is_none()
            })
            .cloned()
            .collect()
    }
}

impl rig::BindSource for LoadedModel {
    fn bone_origin(&self, name: &str) -> Option<Vec3> {
        LoadedModel::bone_origin(self, name)
    }
    fn trunk_vertices(&self) -> Vec<Vec3> {
        self.vertices_for_bones(&rig::TRUNK_BONES)
    }
}

pub struct BindMesh {
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

pub fn load_bind_mesh(path: &std::path::Path) -> Result<BindMesh, String> {
    let parsed = ParsedGltf::open(path)?;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for prim in parsed.primitives()? {
        let base = positions.len() as u32;
        let skinned = parsed.skin_primitive(&prim)?;
        let prim_verts = skinned.len() as u32;
        positions.extend(skinned.into_iter().map(|(pos, _, _)| pos));

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

    #[test]
    fn out_of_range_joint_index_errors() {
        let joint_nodes = [7usize];
        let inv_binds = [Mat4::IDENTITY];
        let bind_world = HashMap::new();
        let got = skin_to_bind_world(
            Vec3::ZERO,
            [3, 0, 0, 0],
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

    #[test]
    fn missing_inverse_bind_matrix_errors() {
        let joint_nodes = [7usize];
        let inv_binds: [Mat4; 0] = [];
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

    #[test]
    fn unreachable_joint_node_errors() {
        let joint_nodes = [7usize];
        let inv_binds = [Mat4::IDENTITY];
        let bind_world = HashMap::new();
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
