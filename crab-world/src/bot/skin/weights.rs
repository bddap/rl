use bevy::mesh::VertexAttributeValues;
use bevy::mesh::skinning::SkinnedMesh;
use bevy::platform::collections::HashSet;
use bevy::prelude::*;

use crate::bot::rig::PartId;

pub(super) fn register(app: &mut App) {
    app.init_resource::<StrippedMeshes>();
    app.add_systems(Update, strip_cross_part_weights);
}

#[derive(Resource, Default)]
struct StrippedMeshes(HashSet<AssetId<Mesh>>);

fn strip_cross_part_weights(
    mut meshes: ResMut<Assets<Mesh>>,
    mut stripped: ResMut<StrippedMeshes>,
    skinned: Query<(&Mesh3d, &SkinnedMesh)>,
    names: Query<&Name>,
) {
    for (mesh3d, skinned_mesh) in skinned.iter() {
        let id = mesh3d.0.id();
        if stripped.0.contains(&id) {
            continue;
        }
        let mut lane_parts: Vec<PartId> = Vec::with_capacity(skinned_mesh.joints.len());
        let mut all_named = true;
        for &joint in &skinned_mesh.joints {
            let Ok(name) = names.get(joint) else {
                all_named = false;
                break;
            };
            lane_parts
                .push(crate::bot::rig::part_for_bone(name.as_str()).unwrap_or(PartId::Carapace));
        }
        if !all_named {
            continue;
        }

        let Some(mesh) = meshes.get_mut(&mesh3d.0) else {
            continue;
        };
        let positions: Option<Vec<Vec3>> = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .map(|p| p.iter().map(|v| Vec3::from_array(*v)).collect());
        let (Some(joints), Some(weights)) = (
            read_u16x4(mesh, Mesh::ATTRIBUTE_JOINT_INDEX),
            read_f32x4(mesh, Mesh::ATTRIBUTE_JOINT_WEIGHT),
        ) else {
            stripped.0.insert(id);
            continue;
        };

        let region = positions
            .as_deref()
            .and_then(|pos| carapace_region(pos, &joints, &weights, &lane_parts));

        let new_weights: Vec<[f32; 4]> = (0..joints.len())
            .map(|i| {
                let pos = positions.as_ref().map(|p| p[i]);
                confine_vertex(pos, region, joints[i], weights[i], &lane_parts)
            })
            .collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, new_weights);
        stripped.0.insert(id);
        info!(
            "crab skin: stripped cross-part weights on {} verts (mesh {:?})",
            joints.len(),
            id
        );
    }
}

fn read_u16x4(mesh: &Mesh, attr: bevy::mesh::MeshVertexAttribute) -> Option<Vec<[u16; 4]>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Uint16x4(v) => Some(v.clone()),
        _ => None,
    }
}

fn read_f32x4(mesh: &Mesh, attr: bevy::mesh::MeshVertexAttribute) -> Option<Vec<[f32; 4]>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Float32x4(v) => Some(v.clone()),
        _ => None,
    }
}

fn confine_vertex(
    pos: Option<Vec3>,
    region: Option<(Vec3, Vec3)>,
    joints: [u16; 4],
    weights: [f32; 4],
    lane_parts: &[PartId],
) -> [f32; 4] {
    let Some(owner) = owner_part(pos, region, joints, weights, lane_parts) else {
        return weights;
    };
    let keep = |p: PartId| -> bool {
        p == owner || (!owner.is_rigid() && crate::bot::rig::parts_adjacent(owner, p))
    };
    let mut kept = [0.0f32; 4];
    let mut kept_sum = 0.0f32;
    for lane in 0..4 {
        if weights[lane] > 0.0 && keep(lane_part(joints, lane, lane_parts)) {
            kept[lane] = weights[lane];
            kept_sum += weights[lane];
        }
    }
    debug_assert!(kept_sum > 0.0, "the owner's own lane must survive");
    for w in &mut kept {
        *w /= kept_sum;
    }
    kept
}

fn owner_part(
    pos: Option<Vec3>,
    region: Option<(Vec3, Vec3)>,
    joints: [u16; 4],
    weights: [f32; 4],
    lane_parts: &[PartId],
) -> Option<PartId> {
    let in_region = match (region, pos) {
        (Some((lo, hi)), Some(p)) => p.cmpge(lo).all() && p.cmple(hi).all(),
        _ => false,
    };
    if in_region && has_shell_lane(joints, weights, lane_parts) {
        return Some(PartId::Carapace);
    }
    dominant_part(joints, weights, lane_parts)
}

fn lane_part(joints: [u16; 4], lane: usize, lane_parts: &[PartId]) -> PartId {
    lane_parts
        .get(joints[lane] as usize)
        .copied()
        .unwrap_or(PartId::Carapace)
}

fn dominant_part(joints: [u16; 4], weights: [f32; 4], lane_parts: &[PartId]) -> Option<PartId> {
    let mut sums: Vec<(PartId, f32)> = Vec::new();
    for (lane, &w) in weights.iter().enumerate() {
        if w <= 0.0 {
            continue;
        }
        let part = lane_part(joints, lane, lane_parts);
        match sums.iter_mut().find(|(p, _)| *p == part) {
            Some((_, s)) => *s += w,
            None => sums.push((part, w)),
        }
    }
    sums.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(p, _)| p)
}

fn has_shell_lane(joints: [u16; 4], weights: [f32; 4], lane_parts: &[PartId]) -> bool {
    (0..4).any(|l| weights[l] > 0.0 && lane_part(joints, l, lane_parts) == PartId::Carapace)
}

fn carapace_region(
    positions: &[Vec3],
    joints: &[[u16; 4]],
    weights: &[[f32; 4]],
    lane_parts: &[PartId],
) -> Option<(Vec3, Vec3)> {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for i in 0..positions.len() {
        if dominant_part(joints[i], weights[i], lane_parts) == Some(PartId::Carapace) {
            lo = lo.min(positions[i]);
            hi = hi.max(positions[i]);
            any = true;
        }
    }
    any.then_some((lo, hi))
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use crate::bot::body::{CrabJointId, Side};

    use super::{PartId, carapace_region, confine_vertex, dominant_part, has_shell_lane};

    fn strip(joints: [u16; 4], weights: [f32; 4], lane_parts: &[PartId]) -> [f32; 4] {
        confine_vertex(None, None, joints, weights, lane_parts)
    }

    #[test]
    fn strip_confines_disjoint_bleed_to_dominant_part() {
        let arm = PartId::Joint(CrabJointId::LegMerus(Side::Left, 0));
        let lane_parts = [PartId::Carapace, PartId::Carapace, arm, arm];
        let part_of = |joints: [u16; 4], lane: usize| lane_parts[joints[lane] as usize];

        let shell_joints = [0u16, 1, 2, 3];
        let shell = strip(shell_joints, [0.7, 0.1, 0.15, 0.05], &lane_parts);
        assert!(
            (shell.iter().sum::<f32>() - 1.0).abs() < 1e-6,
            "weights must renormalize to 1, got {shell:?}"
        );
        for (lane, &w) in shell.iter().enumerate() {
            if w > 0.0 {
                assert_eq!(
                    part_of(shell_joints, lane),
                    PartId::Carapace,
                    "shell vertex must keep only carapace lanes, got {shell:?}"
                );
            }
        }
        assert_eq!(
            shell[2], 0.0,
            "stray arm weight must be zeroed on the shell"
        );
        assert_eq!(
            shell[3], 0.0,
            "stray arm weight must be zeroed on the shell"
        );
        assert!((shell[0] - 0.875).abs() < 1e-6 && (shell[1] - 0.125).abs() < 1e-6);

        let arm_joints = [0u16, 2, 3, 1];
        let arm_v = strip(arm_joints, [0.1, 0.6, 0.3, 0.0], &lane_parts);
        assert!((arm_v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for (lane, &w) in arm_v.iter().enumerate() {
            if w > 0.0 {
                assert_eq!(
                    part_of(arm_joints, lane),
                    arm,
                    "arm vertex must keep arm lanes"
                );
            }
        }
        assert_eq!(
            arm_v[0], 0.0,
            "stray carapace weight must be zeroed on the disjoint arm"
        );

        let solo = strip([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0], &lane_parts);
        assert_eq!(solo, [0.5, 0.5, 0.0, 0.0]);

        let empty = strip([0, 0, 0, 0], [0.0; 4], &lane_parts);
        assert_eq!(empty, [0.0; 4]);
    }

    #[test]
    fn adjacency_matches_the_joint_chains() {
        use crate::bot::rig::parts_adjacent;
        let car = PartId::Carapace;
        let coxa_r0 = PartId::Joint(CrabJointId::LegCoxa(Side::Right, 0));
        let basis_r0 = PartId::Joint(CrabJointId::LegBasis(Side::Right, 0));
        let merus_r0 = PartId::Joint(CrabJointId::LegMerus(Side::Right, 0));
        let carpus_r0 = PartId::Joint(CrabJointId::LegCarpus(Side::Right, 0));
        let shoulder_r = PartId::Joint(CrabJointId::ClawShoulder(Side::Right));
        let wrist_r = PartId::Joint(CrabJointId::ClawWrist(Side::Right));
        let pincer_r = PartId::Joint(CrabJointId::ClawPincer(Side::Right));
        let coxa_r1 = PartId::Joint(CrabJointId::LegCoxa(Side::Right, 1));

        assert!(parts_adjacent(car, coxa_r0));
        assert!(parts_adjacent(coxa_r0, car));
        assert!(parts_adjacent(car, shoulder_r));
        assert!(parts_adjacent(coxa_r0, basis_r0));
        assert!(parts_adjacent(basis_r0, merus_r0));
        assert!(parts_adjacent(merus_r0, carpus_r0));
        assert!(parts_adjacent(shoulder_r, wrist_r));
        assert!(parts_adjacent(wrist_r, pincer_r));

        assert!(!parts_adjacent(car, merus_r0));
        assert!(!parts_adjacent(car, wrist_r));
        assert!(!parts_adjacent(car, pincer_r));
        assert!(!parts_adjacent(coxa_r0, merus_r0));
        assert!(!parts_adjacent(coxa_r0, carpus_r0));
        assert!(!parts_adjacent(shoulder_r, pincer_r));
        assert!(!parts_adjacent(coxa_r0, coxa_r1));
        assert!(!parts_adjacent(shoulder_r, coxa_r0));
    }

    #[test]
    fn strip_keeps_adjacent_seam_lanes_but_confines_disjoint() {
        let wrist = PartId::Joint(CrabJointId::ClawWrist(Side::Right));
        let pincer = PartId::Joint(CrabJointId::ClawPincer(Side::Right));
        let shoulder = PartId::Joint(CrabJointId::ClawShoulder(Side::Right));
        let coxa = PartId::Joint(CrabJointId::LegCoxa(Side::Right, 3));
        let carapace = PartId::Carapace;
        let lane_parts = [wrist, pincer, carapace, coxa, shoulder];

        let knuckle = strip([1, 0, 0, 0], [0.6, 0.4, 0.0, 0.0], &lane_parts);
        assert!((knuckle.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(knuckle[0] > 0.0, "pincer (winner) lane kept: {knuckle:?}");
        assert!(
            knuckle[1] > 0.0,
            "hand lane must survive at the adjacent seam (drag fix): {knuckle:?}"
        );
        assert!((knuckle[0] - 0.6).abs() < 1e-6 && (knuckle[1] - 0.4).abs() < 1e-6);

        let leg_seam = strip([3, 2, 0, 0], [0.65, 0.35, 0.0, 0.0], &lane_parts);
        assert!((leg_seam.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(
            leg_seam[1] > 0.0,
            "carapace lane must survive on a coxa-dominant seam vert: {leg_seam:?}"
        );

        let disjoint = strip([1, 2, 0, 0], [0.7, 0.3, 0.0, 0.0], &lane_parts);
        assert_eq!(disjoint[1], 0.0, "disjoint carapace lane must be zeroed");
        assert!((disjoint[0] - 1.0).abs() < 1e-6, "winner renormalized to 1");

        let shell = strip([2, 4, 1, 0], [0.7, 0.2, 0.1, 0.0], &lane_parts);
        assert_eq!(
            shell[1], 0.0,
            "adjacent shoulder lane on the shell is zeroed"
        );
        assert_eq!(
            shell[2], 0.0,
            "non-adjacent pincer lane on the shell is zeroed"
        );
        assert!((shell[0] - 1.0).abs() < 1e-6, "shell confined to carapace");
    }

    #[test]
    fn seam_drag_audit_on_model() {
        use crate::bot::rig::{part_for_bone, parts_adjacent};
        use crate::mesh_fallback::model_path;

        let Some(path) = model_path() else {
            eprintln!("seam_drag_audit_on_model: no model — skipping");
            return;
        };
        let bytes = std::fs::read(&path).expect("read glb");
        let gltf = gltf::Gltf::from_slice(&bytes).expect("parse glb");
        let blob = gltf.blob.as_deref().expect("glb blob");

        let skin = gltf.skins().next().expect("skin");
        let names: std::collections::HashMap<usize, String> = gltf
            .nodes()
            .filter_map(|n| n.name().map(|nm| (n.index(), nm.to_string())))
            .collect();
        let lane_parts: Vec<PartId> = skin
            .joints()
            .map(|j| {
                names
                    .get(&j.index())
                    .and_then(|nm| part_for_bone(nm))
                    .unwrap_or(PartId::Carapace)
            })
            .collect();

        let part_w = |js: [u16; 4], ws: [f32; 4], target: PartId| -> f32 {
            (0..4)
                .filter(|&l| lane_parts[js[l] as usize] == target)
                .map(|l| ws[l])
                .sum()
        };
        #[derive(Default)]
        struct Seam {
            verts: usize,
            new_anchor_sum: f32,
        }
        let mut seams: std::collections::HashMap<(PartId, PartId), Seam> =
            std::collections::HashMap::new();
        let mut disjoint_seen = 0usize;
        let mut disjoint_regressed = 0usize;

        for prim in gltf.meshes().next().expect("mesh").primitives() {
            let reader = prim.reader(|b| (b.index() == 0).then_some(blob));
            let joints: Vec<[u16; 4]> = reader.read_joints(0).expect("joints").into_u16().collect();
            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .expect("weights")
                .into_f32()
                .collect();
            for (&j, &w) in joints.iter().zip(&weights) {
                if w.iter().all(|&x| x <= 0.0) {
                    continue;
                }
                let dom = dominant_part(j, w, &lane_parts).expect("non-zero vertex");
                let new = strip(j, w, &lane_parts);
                let mut others: Vec<PartId> = Vec::new();
                for l in 0..4 {
                    let p = lane_parts[j[l] as usize];
                    if w[l] > 0.0 && p != dom && !others.contains(&p) {
                        others.push(p);
                    }
                }
                for anchor in others {
                    if parts_adjacent(dom, anchor) {
                        let s = seams.entry((dom, anchor)).or_default();
                        s.verts += 1;
                        s.new_anchor_sum += part_w(j, new, anchor);
                    } else {
                        disjoint_seen += 1;
                        if part_w(j, new, anchor) > 1e-6 {
                            disjoint_regressed += 1;
                        }
                    }
                }
            }
        }

        let mut rows: Vec<(&(PartId, PartId), &Seam)> = seams.iter().collect();
        rows.sort_by_key(|(_, s)| std::cmp::Reverse(s.verts));
        eprintln!("\n=== seam drag audit ({}) ===", path.display());
        eprintln!(
            "{:<48} {:>6} {:>16} {:>16}",
            "seam (winner -> anchor)", "verts", "wta anchor wt", "kept anchor wt"
        );
        for ((winner, anchor), s) in &rows {
            eprintln!(
                "{:<48} {:>6} {:>16.4} {:>16.4}",
                format!("{winner:?} -> {anchor:?}"),
                s.verts,
                0.0,
                s.new_anchor_sum / s.verts.max(1) as f32,
            );
        }
        eprintln!(
            "\ndisjoint (non-adjacent) cross-weights seen: {disjoint_seen}; regressed (leaked): {disjoint_regressed}"
        );
        eprintln!("=== end seam drag audit ===\n");

        assert!(
            disjoint_seen > 0,
            "audit saw no disjoint cross-weights — the #262 gate is vacuous"
        );
        assert_eq!(
            disjoint_regressed, 0,
            "a disjoint (e.g. carapace-vs-distal-limb) bleed leaked through — #262 regressed"
        );
    }

    #[test]
    fn carapace_region_verts_never_deform() {
        use crate::bot::rig::part_for_bone;
        use crate::mesh_fallback::model_path;

        let Some(path) = model_path() else {
            eprintln!("carapace_region_verts_never_deform: no model — skipping");
            return;
        };
        let bytes = std::fs::read(&path).expect("read glb");
        let gltf = gltf::Gltf::from_slice(&bytes).expect("parse glb");
        let blob = gltf.blob.as_deref().expect("glb blob");
        let skin = gltf.skins().next().expect("skin");
        let names: std::collections::HashMap<usize, String> = gltf
            .nodes()
            .filter_map(|n| n.name().map(|nm| (n.index(), nm.to_string())))
            .collect();
        let lane_parts: Vec<PartId> = skin
            .joints()
            .map(|j| {
                names
                    .get(&j.index())
                    .and_then(|nm| part_for_bone(nm))
                    .unwrap_or(PartId::Carapace)
            })
            .collect();

        let mut positions: Vec<Vec3> = Vec::new();
        let mut joints: Vec<[u16; 4]> = Vec::new();
        let mut weights: Vec<[f32; 4]> = Vec::new();
        for prim in gltf.meshes().next().expect("mesh").primitives() {
            let reader = prim.reader(|b| (b.index() == 0).then_some(blob));
            let ps: Vec<[f32; 3]> = reader.read_positions().expect("positions").collect();
            let js: Vec<[u16; 4]> = reader.read_joints(0).expect("joints").into_u16().collect();
            let ws: Vec<[f32; 4]> = reader
                .read_weights(0)
                .expect("weights")
                .into_f32()
                .collect();
            for ((p, j), w) in ps.iter().zip(&js).zip(&ws) {
                positions.push(Vec3::from_array(*p));
                joints.push(*j);
                weights.push(*w);
            }
        }

        let (lo, hi) =
            carapace_region(&positions, &joints, &weights, &lane_parts).expect("shell verts");
        let inside = |p: Vec3| p.cmpge(lo).all() && p.cmple(hi).all();

        let articulated_weight = |js: [u16; 4], ws: [f32; 4]| -> f32 {
            (0..4)
                .filter(|&l| !lane_parts[js[l] as usize].is_rigid())
                .map(|l| ws[l])
                .sum()
        };

        let mut shell_in_region = 0usize;
        let mut deforming = 0usize;
        for i in 0..positions.len() {
            let (j, w) = (joints[i], weights[i]);
            if w.iter().all(|&x| x <= 0.0) || !inside(positions[i]) {
                continue;
            }
            if !has_shell_lane(j, w, &lane_parts) {
                continue;
            }
            shell_in_region += 1;
            let new = confine_vertex(Some(positions[i]), Some((lo, hi)), j, w, &lane_parts);
            if articulated_weight(j, new) > 1e-6 {
                deforming += 1;
            }
        }

        eprintln!(
            "\n=== #37 carapace-region guard ({}) ===\nregion lo={lo:?} hi={hi:?}\nshell-flesh verts in region: {shell_in_region}; still deforming: {deforming}\n=== end #37 ===\n",
            path.display()
        );

        assert!(
            shell_in_region > 0,
            "no shell-flesh verts in the carapace region — the #37 guard is vacuous"
        );
        assert_eq!(
            deforming, 0,
            "{deforming} carapace-region shell verts still carry articulated-joint weight \
             — the rigid shell would morph with the limbs (#37)"
        );
    }

    #[test]
    fn carapace_claim_confines_shell_verts() {
        let arm = PartId::Joint(CrabJointId::ClawShoulder(Side::Left));
        let lane_parts = [PartId::Carapace, PartId::Carapace, arm, arm];
        let region = Some((Vec3::splat(-1.0), Vec3::splat(1.0)));
        let in_box = Vec3::ZERO;
        let out_box = Vec3::splat(5.0);

        let js = [0u16, 2, 3, 1];
        let ws = [0.2, 0.5, 0.2, 0.1];
        let weight_only = strip(js, ws, &lane_parts);
        assert!(
            weight_only[1] + weight_only[2] > 1e-6,
            "precondition: weight ownership leaves this shell vert limb-weighted: {weight_only:?}"
        );
        let new = confine_vertex(Some(in_box), region, js, ws, &lane_parts);
        assert_eq!(new[1], 0.0, "arm lane zeroed on the confined shell vert");
        assert_eq!(new[2], 0.0, "arm lane zeroed on the confined shell vert");
        assert!(
            (new.iter().sum::<f32>() - 1.0).abs() < 1e-6,
            "renormalized: {new:?}"
        );
        assert!((new[0] - 0.2 / 0.3).abs() < 1e-6 && (new[3] - 0.1 / 0.3).abs() < 1e-6);

        let limb = [2u16, 3, 0, 0];
        let limb_w = [0.6, 0.4, 0.0, 0.0];
        let limb_new = confine_vertex(Some(in_box), region, limb, limb_w, &lane_parts);
        assert_eq!(
            limb_new,
            strip(limb, limb_w, &lane_parts),
            "limb-only vert in region stays weight-owned, not shell-claimed"
        );
        assert!(
            limb_new[0] + limb_new[1] > 0.99,
            "limb stub still articulates"
        );

        let outside = confine_vertex(Some(out_box), region, js, ws, &lane_parts);
        assert_eq!(outside, strip(js, ws, &lane_parts));
    }
}
