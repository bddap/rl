//! Cross-part skin-weight strip (bddap/rl#32).
//!
//! [`strip_cross_part_weights`] rewrites each shared mesh asset's authored skin
//! weights once, applying one per-vertex rule — [`confine_vertex`], whose doc is the
//! canonical rationale. Pure weight-only output: pixel-identical at rest.

use bevy::mesh::VertexAttributeValues;
use bevy::mesh::skinning::SkinnedMesh;
use bevy::platform::collections::HashSet;
use bevy::prelude::*;

use crate::bot::meshfit::PartId;

pub(super) fn register(app: &mut App) {
    app.init_resource::<StrippedMeshes>();
    app.add_systems(Update, strip_cross_part_weights);
}

/// Meshes whose weights have already been confined to each vertex's owner part.
/// The skinned mesh asset is shared across crab instances, so the rewrite must
/// happen exactly once per asset; this records the ones done.
#[derive(Resource, Default)]
struct StrippedMeshes(HashSet<AssetId<Mesh>>);

/// Rewrite each skinned mesh asset's `WEIGHTS_0` exactly once, applying
/// [`confine_vertex`] — the one per-vertex deform rule, and the canonical WHY — to
/// every vertex ([`crate::bot::rig::part_for_bone`] assigns each bone its part).
///
/// Runs in `Update` and waits until the mesh asset is loaded AND every joint
/// entity has a `Name` (the scene finished spawning), since the strip maps each
/// `joint_index` → joint entity → `Name` → part. Until then it no-ops and retries.
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
        // Resolve every joint lane to its part. A joint without a `Name` yet means
        // the scene is still spawning — bail and retry next frame rather than bake
        // a wrong (all-Carapace) map. `part_for_bone` already maps unknown/non-rig
        // bones to Carapace, so once names exist this never produces a missing lane.
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
            continue; // asset not loaded yet
        };
        // Positions (raw mesh-local, the frame skinning starts from — GPU LBS moves
        // them, so the CPU attribute is the bind-pose-local position) define the
        // geometric carapace region below. Cloned out so the borrow ends before the
        // `&mut mesh` write; `None` (no/oddly-typed POSITION) just disables the
        // geometric claim, leaving pure weight ownership.
        let positions: Option<Vec<Vec3>> = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .map(|p| p.iter().map(|v| Vec3::from_array(*v)).collect());
        let (Some(joints), Some(weights)) = (
            read_u16x4(mesh, Mesh::ATTRIBUTE_JOINT_INDEX),
            read_f32x4(mesh, Mesh::ATTRIBUTE_JOINT_WEIGHT),
        ) else {
            // Not a skinned crab mesh (no u16x4/f32x4 weight attrs) — nothing to
            // confine. Still record it: this system re-runs each frame until a
            // mesh is in `stripped`, so an unrecorded one would be re-read forever.
            stripped.0.insert(id);
            continue;
        };

        // The carapace region: the AABB of every shell-dominant vertex. `None` if
        // positions are unavailable (then ownership is pure weight). See
        // `confine_vertex` for the per-vertex rule.
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

/// Read a `Uint16x4` mesh attribute as `[u16; 4]` lanes, or `None` for any other
/// stored format.
fn read_u16x4(mesh: &Mesh, attr: bevy::mesh::MeshVertexAttribute) -> Option<Vec<[u16; 4]>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Uint16x4(v) => Some(v.clone()),
        _ => None,
    }
}

/// Read a `Float32x4` mesh attribute as `[f32; 4]` lanes, or `None` for any other
/// stored format.
fn read_f32x4(mesh: &Mesh, attr: bevy::mesh::MeshVertexAttribute) -> Option<Vec<[f32; 4]>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Float32x4(v) => Some(v.clone()),
        _ => None,
    }
}

/// THE per-vertex deform rule — the whole strip, expressed once: a vertex belongs to
/// exactly one part, its [`owner_part`]; a lane survives iff its part is the owner
/// or, when the owner articulates, a part [hinged](crate::bot::rig::parts_adjacent)
/// to it; survivors renormalize to 1.0. An all-zero vertex (no owner) is returned
/// unchanged. Weight-only output: a vertex's REST position is its bind pose whatever
/// bones could move it, so the strip is pixel-identical at rest and only changes how
/// the surface deforms.
///
/// WHY strip at all (bddap/rl#32): sally.glb's authored skin paints cross-part bleed
/// — e.g. shell vertices carrying stray `WEIGHTS_0` on arm bones. Our deform bones
/// are driven RIGIDLY by the physics links ([`crate::bot::skin::pairing::drive_bones`]),
/// so GPU linear-blend skinning would drag that flesh along whenever the other part
/// moves. Two parts that share no hinge share no flesh, so a disjoint cross-weight is
/// pure bleed and never survives.
///
/// WHY hinged neighbours survive: pure winner-take-all also amputates the legitimate
/// blend at an ADJACENT joint seam, welding a seam vertex rigidly to one link so it
/// drags off the other when that link moves — the #32 dactyl-knuckle and
/// rear-leg/shell drag. Keeping the lanes of parts hinged to the owner lets a seam
/// bend with its joint; a surviving blend reproduces the authored bind-pose ratio.
///
/// WHY a RIGID owner keeps no neighbours: the carapace is a single rigid box that
/// never deforms ([`PartId::is_rigid`]), so any limb lane on a shell-owned vertex —
/// even an adjacent chain root's — has the moving limb tug the shell into a bulge
/// (the #262 artifact). A vertex that legitimately blends at a leg socket is limb-,
/// not shell-, owned and flexes via the clause above.
fn confine_vertex(
    pos: Option<Vec3>,
    region: Option<(Vec3, Vec3)>,
    joints: [u16; 4],
    weights: [f32; 4],
    lane_parts: &[PartId],
) -> [f32; 4] {
    let Some(owner) = owner_part(pos, region, joints, weights, lane_parts) else {
        return weights; // all-zero vertex: leave it be
    };
    let keep = |p: PartId| -> bool {
        p == owner || (!owner.is_rigid() && crate::bot::rig::parts_adjacent(owner, p))
    };
    // Zero every lane `keep` rejects, renormalize the survivors to sum to 1.0. At
    // least one lane IS kept: the owner came from a positively-weighted lane of its
    // own part ([`owner_part`]), and `keep(owner) == true`.
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

/// The one part that owns a vertex: position first, then weight. A shell-flesh
/// vertex — one carrying any carapace weight ([`has_shell_lane`]) whose `pos` lies
/// inside the carapace `region` — is the carapace's, regardless of its authored
/// weights; every other vertex belongs to its heaviest-weighted part
/// ([`dominant_part`]; `None` for an all-zero vertex). Either way the owner has a
/// positively-weighted lane of its own, so [`confine_vertex`] always keeps one.
///
/// WHY position outranks weight (bddap/rl#37): dominance is authored weight, and
/// weight cannot separate "shell vertex the artist painted limb-heavy" (must never
/// deform) from "limb vertex near the shell" (must keep articulating); only position
/// can. The geometric claim demands a carapace lane because a vertex with none is a
/// limb passing through the box (a leg/claw socket stub) — shell ownership would zero
/// its only lanes and collapse it to the origin. `region`/`pos` of `None` (positions
/// unavailable) disables the geometric claim, leaving pure weight ownership.
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

/// The part of skin-joint lane `lane` on this vertex; an out-of-range index defaults
/// to the carapace (matching `part_for_bone`'s unknown-bone fallback).
fn lane_part(joints: [u16; 4], lane: usize, lane_parts: &[PartId]) -> PartId {
    lane_parts
        .get(joints[lane] as usize)
        .copied()
        .unwrap_or(PartId::Carapace)
}

/// The vertex's dominant part — the one with the largest summed weight across its
/// lanes — or `None` for an all-zero vertex. The single source for weight-based
/// ownership, shared by [`owner_part`] and the carapace-region builder so the two
/// can't disagree about dominance.
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

/// Whether this vertex carries any weight on a carapace bone — i.e. whether it is
/// bound to the shell at all. Gates [`owner_part`]'s geometric claim (see its doc for
/// why a shell-lane-free vertex must stay weight-owned); gating on the same part the
/// claim awards is what guarantees a shell-owned vertex keeps a lane.
fn has_shell_lane(joints: [u16; 4], weights: [f32; 4], lane_parts: &[PartId]) -> bool {
    (0..4).any(|l| weights[l] > 0.0 && lane_part(joints, l, lane_parts) == PartId::Carapace)
}

/// The carapace region: the AABB of every shell-dominant vertex, in the mesh's raw
/// position frame — where [`owner_part`]'s geometric claim applies (bddap/rl#37).
/// Returns `None` if no vertex is shell-dominant (a non-crab mesh), which disables
/// the claim (ownership stays pure weight).
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

    /// [`confine_vertex`] with no geometry — pure weight ownership, the rule every
    /// vertex outside the carapace region gets.
    fn strip(joints: [u16; 4], weights: [f32; 4], lane_parts: &[PartId]) -> [f32; 4] {
        confine_vertex(None, None, joints, weights, lane_parts)
    }

    /// The strip (bddap/rl#32, refined for seams): a shell vertex carrying stray arm
    /// weight no longer drives the arm (the rigid carapace can't bulge), a vertex on a
    /// part DISJOINT from the carapace drops its stray carapace lane, and the survivors
    /// still sum to 1.0. Adjacent-seam blending is covered separately
    /// ([`strip_keeps_adjacent_seam_lanes_but_confines_disjoint`]). Pure (no GPU): a
    /// hand-built lane→part map stands in for the resolved joints.
    #[test]
    fn strip_confines_disjoint_bleed_to_dominant_part() {
        // A limb part DISJOINT from the carapace, so the cross-weight below is pure
        // bleed (no shared hinge) and must be stripped from either side. (The chain
        // roots ClawShoulder/LegCoxa DO hinge on the carapace — a real seam — so they'd
        // blend; the merus is two links out and shares no joint with the shell.)
        let arm = PartId::Joint(CrabJointId::LegMerus(Side::Left, 0));
        // Lane layout: 0,1 → Carapace; 2,3 → the arm.
        let lane_parts = [PartId::Carapace, PartId::Carapace, arm, arm];
        // Lane→part for a vertex's joint indices, used to re-check the result.
        let part_of = |joints: [u16; 4], lane: usize| lane_parts[joints[lane] as usize];

        // Shell vertex: dominated by carapace (0.7+0.1) with stray arm weight (0.2).
        // The arm lanes must end up at zero so the limb no longer drags the rigid
        // shell, and the carapace lanes renormalize to sum to 1.
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
        // Original carapace ratio (0.7 : 0.1) is preserved after renormalizing.
        assert!((shell[0] - 0.875).abs() < 1e-6 && (shell[1] - 0.125).abs() < 1e-6);

        // Arm vertex: dominated by the (disjoint) arm part (0.6+0.3) with stray
        // carapace weight (0.1). With no shared hinge it must keep its arm lanes and
        // drop the carapace lane.
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

        // Single-part vertex is unchanged (already sums to 1, one part).
        let solo = strip([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0], &lane_parts);
        assert_eq!(solo, [0.5, 0.5, 0.0, 0.0]);

        // All-zero vertex is left alone (no part to dominate).
        let empty = strip([0, 0, 0, 0], [0.0; 4], &lane_parts);
        assert_eq!(empty, [0.0; 4]);
    }

    /// Two parts in the same limb chain (claw-chain neighbours, leg-chain neighbours)
    /// and each chain root with the carapace must be adjacent; two parts on different
    /// limbs, or the carapace with a non-root joint, must NOT be. Pins the adjacency
    /// the strip keys off so a future chain edit can't silently re-confine a real seam
    /// or fuse a disjoint pair.
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

        // Adjacent: chain roots to the carapace, and within-chain neighbours. The leg
        // chain is coxa→basis→merus→carpus (the coxo-basal split), so each consecutive
        // pair hinges, but coxa→merus does not (the basis sits between them).
        assert!(parts_adjacent(car, coxa_r0));
        assert!(parts_adjacent(coxa_r0, car)); // symmetric
        assert!(parts_adjacent(car, shoulder_r));
        assert!(parts_adjacent(coxa_r0, basis_r0));
        assert!(parts_adjacent(basis_r0, merus_r0));
        assert!(parts_adjacent(merus_r0, carpus_r0));
        assert!(parts_adjacent(shoulder_r, wrist_r));
        assert!(parts_adjacent(wrist_r, pincer_r)); // the #32 thumb seam

        // NOT adjacent: the carapace to a non-root joint (the #262 bleed must stay
        // confined), across-the-chain skips, and two unrelated limbs.
        assert!(!parts_adjacent(car, merus_r0));
        assert!(!parts_adjacent(car, wrist_r));
        assert!(!parts_adjacent(car, pincer_r));
        assert!(!parts_adjacent(coxa_r0, merus_r0)); // skips the basis
        assert!(!parts_adjacent(coxa_r0, carpus_r0)); // skips the basis+merus
        assert!(!parts_adjacent(shoulder_r, pincer_r)); // skips the wrist
        assert!(!parts_adjacent(coxa_r0, coxa_r1)); // different legs
        assert!(!parts_adjacent(shoulder_r, coxa_r0)); // claw vs leg
    }

    /// Seam-only blending: at an ADJACENT hinge a vertex keeps both parts' lanes
    /// (so it bends with the joint instead of rigidly dragging off one side), but a
    /// vertex split across a NON-adjacent (disjoint) pair is still confined to its
    /// dominant part — the #262 carapace-vs-arm bleed must not return.
    #[test]
    fn strip_keeps_adjacent_seam_lanes_but_confines_disjoint() {
        let wrist = PartId::Joint(CrabJointId::ClawWrist(Side::Right));
        let pincer = PartId::Joint(CrabJointId::ClawPincer(Side::Right));
        let shoulder = PartId::Joint(CrabJointId::ClawShoulder(Side::Right));
        let coxa = PartId::Joint(CrabJointId::LegCoxa(Side::Right, 3));
        let carapace = PartId::Carapace;
        // Lane layout: 0→wrist (hand), 1→pincer (dactyl), 2→carapace, 3→coxa, 4→shoulder.
        let lane_parts = [wrist, pincer, carapace, coxa, shoulder];

        // (a) Knuckle seam vertex: pincer dominant (0.6) but a real hand lane (0.4) —
        // the #32 drag case. Both adjacent claw lanes survive; ratio preserved.
        let knuckle = strip([1, 0, 0, 0], [0.6, 0.4, 0.0, 0.0], &lane_parts);
        assert!((knuckle.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(knuckle[0] > 0.0, "pincer (winner) lane kept: {knuckle:?}");
        assert!(
            knuckle[1] > 0.0,
            "hand lane must survive at the adjacent seam (drag fix): {knuckle:?}"
        );
        assert!((knuckle[0] - 0.6).abs() < 1e-6 && (knuckle[1] - 0.4).abs() < 1e-6);

        // (b) Rear-leg seam vertex: coxa dominant (0.65) with a carapace lane (0.35).
        // The coxa hinges on the carapace (chain root), so the carapace lane survives
        // and the vertex blends back toward the shell instead of dragging fully off it.
        let leg_seam = strip([3, 2, 0, 0], [0.65, 0.35, 0.0, 0.0], &lane_parts);
        assert!((leg_seam.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(
            leg_seam[1] > 0.0,
            "carapace lane must survive on a coxa-dominant seam vert: {leg_seam:?}"
        );

        // Pincer dominant with a DISJOINT carapace lane (skips wrist+shoulder, no shared
        // hinge): the carapace lane is still zeroed — only adjacent neighbours blend.
        let disjoint = strip([1, 2, 0, 0], [0.7, 0.3, 0.0, 0.0], &lane_parts);
        assert_eq!(disjoint[1], 0.0, "disjoint carapace lane must be zeroed");
        assert!((disjoint[0] - 1.0).abs() < 1e-6, "winner renormalized to 1");

        // The #262 regression guard, the hard case: a shell vertex (carapace dominant)
        // carrying a stray SHOULDER lane. The shoulder IS a chain root hinged to the
        // carapace, yet because the rigid shell is what dominates, the limb lane is
        // still stripped — the arm must never tug the trunk into a bulge. (Pincer too,
        // which isn't even adjacent.)
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

    /// Quantitative seam audit on the real model (run with `--nocapture`): re-parses
    /// `sally.glb` and, for every vertex that blends across a part boundary, contrasts
    /// pure winner-take-all (the pre-seam-fix rule) with the current seam-aware rule.
    /// Read off the file, so it is independent of the Bevy spawn path.
    ///
    /// For each ADJACENT (winner → anchor) seam it prints the vertex count and the mean
    /// weight the vertex keeps on its anchor (the part it sits on but does not win)
    /// after each rule — winner-take-all zeroes it (drags fully off the anchor), the
    /// seam-aware rule keeps it where the anchor is allowed to flex. Because drag distance under a
    /// fixed joint rotation is proportional to the winning part's weight, the anchor
    /// weight handed back IS the fractional drag removed. The carapace-as-winner rows
    /// stay confined by design (the rigid shell never deforms), so their anchor weight
    /// is zero under both rules — that is the #262 guarantee, not a miss. A disjoint
    /// (non-adjacent) cross-weight must stay confined under both; the test fails if any
    /// leaks through. Skips cleanly when the model isn't present.
    #[test]
    fn seam_drag_audit_on_model() {
        use crate::bot::meshfit::model_path;
        use crate::bot::rig::{part_for_bone, parts_adjacent};

        let Some(path) = model_path() else {
            eprintln!("seam_drag_audit_on_model: no model — skipping");
            return;
        };
        let bytes = std::fs::read(&path).expect("read glb");
        let gltf = gltf::Gltf::from_slice(&bytes).expect("parse glb");
        let blob = gltf.blob.as_deref().expect("glb blob");

        // skin-joint index -> part, the same resolution the live strip does.
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

        // Summed weight on `target` across a vertex's lanes (for any weight array over
        // the same joint indices — original or stripped).
        let part_w = |js: [u16; 4], ws: [f32; 4], target: PartId| -> f32 {
            (0..4)
                .filter(|&l| lane_parts[js[l] as usize] == target)
                .map(|l| ws[l])
                .sum()
        };
        #[derive(Default)]
        struct Seam {
            verts: usize,
            new_anchor_sum: f32, // anchor weight kept by the new rule (old rule = 0)
        }
        let mut seams: std::collections::HashMap<(PartId, PartId), Seam> =
            std::collections::HashMap::new();
        let mut disjoint_seen = 0usize; // non-adjacent cross-weights (the #262 case)
        let mut disjoint_regressed = 0usize; // …that the new rule failed to confine

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
                // Each OTHER part this vertex weighs on = a boundary it straddles.
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

        // Keep the gate from going vacuous: if a future asset carries no disjoint
        // bleed to confine, the `== 0` check below would pass while testing nothing.
        assert!(
            disjoint_seen > 0,
            "audit saw no disjoint cross-weights — the #262 gate is vacuous"
        );
        // The regression gate: not one disjoint cross-weight may survive the new rule.
        assert_eq!(
            disjoint_regressed, 0,
            "a disjoint (e.g. carapace-vs-distal-limb) bleed leaked through — #262 regressed"
        );
    }

    /// The decisive #37 guard on the real model (run with `--nocapture`): no rigid
    /// shell vertex may deform with an articulated joint. Re-parses `sally.glb`, builds
    /// the [`carapace_region`] from its shell-dominant verts, runs the SAME per-vertex
    /// rule the live strip ([`super::strip_cross_part_weights`]) does, and asserts that
    /// EVERY shell-flesh vertex (one carrying any carapace weight) inside that region
    /// resolves to carapace-only — zero weight on any articulated joint bone.
    ///
    /// This FAILS on pre-#37 weight-only ownership: a shell vertex the artist weighted
    /// limb-heavy is limb-dominant, so it keeps its limb lanes and the rigid shell
    /// bulges with the joint (~1.2k such verts in sally.glb). The geometric claim
    /// drives that count to zero. Limb-only verts inside the box (a
    /// leg/claw socket stub passing through) are NOT shell flesh and are excluded — they
    /// must keep articulating, and confining them would zero their only lanes. Read off
    /// the file, independent of the Bevy spawn path; skips cleanly with no model.
    #[test]
    fn carapace_region_verts_never_deform() {
        use crate::bot::meshfit::model_path;
        use crate::bot::rig::part_for_bone;

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
        // skin-joint lane -> physics part, resolved exactly as the live strip does.
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

        // Weight a vertex's lanes onto the articulated (non-rigid) joints — how much it
        // still deforms with a moving limb. Must be zero for shell flesh in the region.
        let articulated_weight = |js: [u16; 4], ws: [f32; 4]| -> f32 {
            (0..4)
                .filter(|&l| !lane_parts[js[l] as usize].is_rigid())
                .map(|l| ws[l])
                .sum()
        };

        let mut shell_in_region = 0usize; // shell-flesh verts the geometric claim governs
        let mut deforming = 0usize; // …that STILL deform (the bug; must be 0)
        for i in 0..positions.len() {
            let (j, w) = (joints[i], weights[i]);
            if w.iter().all(|&x| x <= 0.0) || !inside(positions[i]) {
                continue;
            }
            // Limb-only verts inside the box are limb flesh passing through, not shell —
            // the geometric claim skips them (so does this assertion), and they keep flexing.
            if !has_shell_lane(j, w, &lane_parts) {
                continue;
            }
            shell_in_region += 1;
            // The exact per-vertex rule `strip_cross_part_weights` runs in production.
            let new = confine_vertex(Some(positions[i]), Some((lo, hi)), j, w, &lane_parts);
            if articulated_weight(j, new) > 1e-6 {
                deforming += 1;
            }
        }

        eprintln!(
            "\n=== #37 carapace-region guard ({}) ===\nregion lo={lo:?} hi={hi:?}\nshell-flesh verts in region: {shell_in_region}; still deforming: {deforming}\n=== end #37 ===\n",
            path.display()
        );

        // Not vacuous: sally.glb has thousands of shell verts inside the region, so a
        // future asset (or a bug) that strands zero would otherwise pass testing nothing.
        assert!(
            shell_in_region > 0,
            "no shell-flesh verts in the carapace region — the #37 guard is vacuous"
        );
        // The #37 done-state: not one rigid shell vertex deforms with a joint.
        assert_eq!(
            deforming, 0,
            "{deforming} carapace-region shell verts still carry articulated-joint weight \
             — the rigid shell would morph with the limbs (#37)"
        );
    }

    /// The geometric carapace claim ([`confine_vertex`] via [`super::owner_part`],
    /// bddap/rl#37), pure (no GPU): a hand-built lane→part map and explicit positions
    /// stand in for the model. Pins the three branches the model test can only assert
    /// in aggregate — the claim fires on a vertex weight ownership would have LEFT
    /// articulating (limb-dominant), which is exactly the #37 bug.
    #[test]
    fn carapace_claim_confines_shell_verts() {
        let arm = PartId::Joint(CrabJointId::ClawShoulder(Side::Left));
        // Lane layout: 0,1 → Carapace; 2,3 → an arm bone.
        let lane_parts = [PartId::Carapace, PartId::Carapace, arm, arm];
        let region = Some((Vec3::splat(-1.0), Vec3::splat(1.0)));
        let in_box = Vec3::ZERO; // inside the region
        let out_box = Vec3::splat(5.0); // outside it

        // The #37 case: a shell vertex the artist weighted ARM-heavy (arm dominant,
        // 0.7 > 0.3). Weight ownership keeps the arm lanes (limb-dominant), so the
        // shell would bulge — but inside the region the geometric claim hands it to
        // the carapace, renormalizing its shell lane(s) to 1 and zeroing the arm.
        let js = [0u16, 2, 3, 1];
        let ws = [0.2, 0.5, 0.2, 0.1];
        // Confirm weight-only ownership leaks (so the claim is doing real work here).
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

        // A limb-ONLY vertex inside the region (no carapace lane) is a socket stub
        // passing through — the geometric claim skips it (shell ownership would zero
        // it), so it stays weight-owned and keeps articulating.
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

        // The SAME shell vertex OUTSIDE the region stays weight-owned (the claim is
        // geometric — it governs only the shell box).
        let outside = confine_vertex(Some(out_box), region, js, ws, &lane_parts);
        assert_eq!(outside, strip(js, ws, &lane_parts));
    }
}
