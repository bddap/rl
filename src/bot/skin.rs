//! Optional skinned crab model riding the physics body.
//!
//! When `CRAB_MODEL_PATH` names a glTF inside the app's `assets/` directory
//! (e.g. `sally.glb`, fetched from the private bddap-bot/rl-assets repo), each
//! crab gets a skinned-mesh skin whose deform bones follow the physics links.
//! The physics body stays the single source of truth; the model is cosmetic,
//! and the colliders themselves are only ever shown by Rapier's debug-render
//! (`RL_DEBUG_COLLIDERS`), never as stand-in meshes.
//!
//! How following works: when the scene instance is ready, every deform bone is
//! matched to a physics link by name ([`super::rig::part_for_bone`]), and the bone's world
//! pose relative to its link is captured once: `offset = link⁻¹ · bone`. Each
//! frame after that the bone's world transform is set to `link · offset`, so
//! bones reproduce link *motion* exactly while keeping the model's own
//! proportions. Captured at spawn rest, where the model's bind pose and the
//! physics rest stance agree closely; residual mismatch is a constant
//! per-bone offset, not drift. Driven bones are reparented flat under the
//! skin root (identity transform), so their `Transform` IS their world pose —
//! no per-frame parent-chain inversions. Non-bone scene nodes (the skinned
//! mesh entity itself) are left alone; skinning reads joint GlobalTransforms,
//! which keep working across the reparent.
//!
//! Re-pairing on reset: an episode reset ([`crate::bot::respawn_crab`]) despawns
//! an env's physics parts and spawns fresh ones under the SAME env id. The skin
//! (keyed by env id) survives — `attach_skins` sees one already exists and
//! `reap_orphan_skins` sees the env still populated — but every [`BoneDrive`]
//! still points at a now-despawned part entity, so [`drive_bones`] reads dead
//! transforms and the model freezes in place. [`repair_skins`] detects this
//! (a bone's link entity went dead) and re-points each bone at the fresh part
//! playing the same role, reusing the captured offset: the respawn reproduces
//! the identical rest pose at the same origin, so the offset is still exact and
//! the skin stays visible the whole time — no re-settle, no flicker, no leak.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::VertexAttributeValues;
use bevy::mesh::skinning::SkinnedMesh;
use bevy::platform::collections::HashSet;
use bevy::prelude::*;

use super::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabRestPose};
use super::meshfit::PartId;

/// Present only when a model path was configured; all systems key off this.
#[derive(Resource)]
pub struct CrabModel {
    scene: Handle<Scene>,
}

/// One skin instance, attached to one crab (by env id).
#[derive(Component)]
pub struct CrabSkin {
    env: usize,
    /// Frames since the scene instance got children. Pairing waits for
    /// [`SETTLE_FRAMES`]: offsets captured against the crab still up at spawn
    /// height bake the subsequent ~0.2 m settle into every carapace-bound bone
    /// and sink the shell into the ground. The skin stays hidden (primitives
    /// showing) until it pairs.
    scene_frames: Option<u32>,
    /// Pairing done; `BoneDrive`s exist below this root.
    paired: bool,
}

/// Render frames between scene readiness and offset capture — enough for the
/// spawned crab to settle onto its feet (~1.5 s at 60 fps).
const SETTLE_FRAMES: u32 = 90;

/// A driven bone: follow `link`'s world transform with a fixed local offset.
#[derive(Component)]
pub struct BoneDrive {
    link: Entity,
    offset: Mat4,
}

/// The physics-link query shared by pairing and re-pairing: every part entity
/// of every crab, tagged with its env and its role (joint id or carapace).
type LinkQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static CrabEnvId,
        Option<&'static CrabJoint>,
        Option<&'static CrabCarapace>,
    ),
>;

/// Map one env's physics parts by the role a deform bone keys off. Built fresh
/// each time because a reset replaces the entities; [`super::rig::part_for_bone`]
/// resolves a bone name to a [`PartId`], this resolves that part to the live
/// entity. Eye-stalk links carry neither component, so they're absent here — the
/// eye bones ride the carapace cosmetically (the eye link is fixed to it).
fn link_map(links: &LinkQuery, env: usize) -> std::collections::HashMap<PartId, Entity> {
    links
        .iter()
        .filter(|(_, e, ..)| e.0 == env)
        .filter_map(|(e, _, joint, carapace)| match (joint, carapace) {
            (Some(j), _) => Some((PartId::Joint(j.id), e)),
            (_, Some(_)) => Some((PartId::Carapace, e)),
            _ => None,
        })
        .collect()
}

pub fn register(app: &mut App) {
    let Ok(path) = std::env::var("CRAB_MODEL_PATH") else {
        return;
    };
    let scene = app
        .world()
        .resource::<AssetServer>()
        .load(GltfAssetLabel::Scene(0).from_asset(path));
    app.insert_resource(CrabModel { scene });
    app.init_resource::<StrippedMeshes>();
    app.add_systems(
        Update,
        (
            attach_skins,
            reap_orphan_skins,
            reveal_skin,
            strip_cross_part_weights,
        ),
    );
    app.add_systems(
        PostUpdate,
        (
            // Re-point bones at the fresh parts BEFORE driving them, so a reset
            // frame already follows the new crab instead of the dead one.
            repair_skins.before(drive_bones),
            drive_bones.before(TransformSystems::Propagate),
            pair_bones.after(TransformSystems::Propagate),
        ),
    );
}

/// Give every skinless crab a skin root at its spawn point. Reactive, so
/// respawned crabs (episode resets, NaN rescues) are re-skinned automatically.
fn attach_skins(
    mut commands: Commands,
    model: Res<CrabModel>,
    assets: Res<super::body::CrabAssets>,
    crabs: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    skins: Query<&CrabSkin>,
) {
    // The body's carapace root spawns at the leg hub's bind-world position; the skin
    // renders its bones at glTF bind-world from this root, so subtracting the hub puts
    // the skin's skeleton in the body's exact frame — bone-for-link aligned at rest.
    // (Skip if the recipe is absent; the skin needs the model anyway.)
    let Some(hub) = assets.hub_bind_world() else {
        return;
    };
    for (env, t) in crabs.iter() {
        if skins.iter().any(|s| s.env == env.0) {
            continue;
        }
        commands.spawn((
            SceneRoot(model.scene.clone()),
            Transform::from_translation(t.translation - hub),
            // Hidden until paired: until then it would be a bind-pose statue.
            Visibility::Hidden,
            CrabSkin {
                env: env.0,
                scene_frames: None,
                paired: false,
            },
        ));
    }
}

/// A skin whose crab is gone (despawn-respawn reset) despawns with all its
/// (reparented) bones; `attach_skins` re-creates it for the replacement crab.
fn reap_orphan_skins(
    mut commands: Commands,
    skins: Query<(Entity, &CrabSkin)>,
    crabs: Query<&CrabEnvId, With<CrabCarapace>>,
) {
    for (root, skin) in skins.iter() {
        if !crabs.iter().any(|env| env.0 == skin.env) {
            commands.entity(root).despawn();
        }
    }
}

/// Once the glTF instance exists (one frame after its children appear, so
/// GlobalTransforms hold the bind pose), capture per-bone offsets and flatten
/// driven bones under the root. Setting `paired` hands visibility to
/// [`reveal_skin`], which shows the now-driven skin.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn pair_bones(
    mut commands: Commands,
    mut skins: Query<(Entity, &mut CrabSkin)>,
    children: Query<&Children>,
    names: Query<&Name>,
    globals: Query<&GlobalTransform>,
    rest_poses: Query<&CrabRestPose>,
    links: LinkQuery,
    meshes: Query<(), With<Mesh3d>>,
) {
    for (root, mut skin) in skins.iter_mut() {
        if skin.paired {
            continue;
        }
        match skin.scene_frames {
            None => {
                if children.get(root).is_ok_and(|c| !c.is_empty()) {
                    skin.scene_frames = Some(0);
                }
                continue;
            }
            Some(n) if n < SETTLE_FRAMES => {
                skin.scene_frames = Some(n + 1);
                continue;
            }
            Some(_) => {}
        }

        let link_of = link_map(&links, skin.env);

        let mut stack: Vec<Entity> = children
            .get(root)
            .map(|c| c.iter().collect())
            .unwrap_or_default();
        let mut paired = 0usize;
        while let Some(e) = stack.pop() {
            if let Ok(c) = children.get(e) {
                stack.extend(c.iter());
            }
            // A skinned mesh keeps the AABB of its bind pose; once the crab
            // walks far enough that those stale bounds leave the frustum, the
            // whole mesh is culled and the model vanishes mid-scene. One crab
            // is cheap to always draw.
            if meshes.get(e).is_ok() {
                commands.entity(e).insert(NoFrustumCulling);
            }
            let Ok(name) = names.get(e) else { continue };
            let Some(key) = super::rig::part_for_bone(name.as_str()) else {
                continue;
            };
            let Some(&link) = link_of.get(&key) else {
                continue;
            };
            let (Ok(bone_g), Ok(link_rest)) = (globals.get(e), rest_poses.get(link)) else {
                continue;
            };
            // Pair against the link's REST (bind) pose, not its live transform: the
            // body has already begun settling by now, and using the live pose would
            // bake that sag into the offset, leaving the skin riding above the
            // colliders. `drive_bones` then composes the LIVE link pose with this
            // bind offset, so the mesh reproduces the bind pose at rest and tracks
            // the physics exactly as it moves.
            let offset = link_rest.0.to_matrix().inverse() * bone_g.to_matrix();
            commands.entity(e).insert((
                BoneDrive { link, offset },
                ChildOf(root),
                Transform::from_matrix(bone_g.to_matrix()),
            ));
            paired += 1;
        }

        info!(
            "crab skin paired: env {} ({} bones driven)",
            skin.env, paired
        );
        // Driven bones now carry absolute world poses (see module docs), so the
        // root must be identity. attach_skins gave it the spawn translation; left in
        // place that transform would be applied a SECOND time on top of the
        // already-world bone poses, rendering the skin offset from the physics body.
        commands.entity(root).insert(Transform::default());
        skin.paired = true;
    }
}

/// Re-point a reset crab's bones at its fresh physics parts. A paired skin
/// whose env was respawned ([`crate::bot::respawn_crab`]) still drives the
/// despawned parts; this catches that — any one bone's link entity gone dead
/// flags the whole skin stale — and re-resolves every bone from its name to
/// the live part of the same role, keeping the captured offset (the respawn
/// reproduces the same rest pose at the same origin, so it stays exact). No
/// re-settle and no visibility change, so the skin never flickers.
fn repair_skins(
    mut bones: Query<(&mut BoneDrive, &Name)>,
    skins: Query<(&CrabSkin, &Children)>,
    links: LinkQuery,
) {
    for (skin, kids) in skins.iter() {
        if !skin.paired {
            continue; // pair_bones owns first-time pairing.
        }
        // Stale iff a bone targets a despawned link. `get` on a dead entity
        // fails; one such bone means the whole env was respawned.
        let stale = kids
            .iter()
            .filter_map(|b| bones.get(b).ok())
            .any(|(drive, _)| links.get(drive.link).is_err());
        if !stale {
            continue;
        }

        let link_of = link_map(&links, skin.env);
        for bone in kids.iter() {
            let Ok((mut drive, name)) = bones.get_mut(bone) else {
                continue;
            };
            if let Some(key) = super::rig::part_for_bone(name.as_str())
                && let Some(&link) = link_of.get(&key)
            {
                drive.link = link;
            }
        }
        info!("crab skin re-paired after reset: env {}", skin.env);
    }
}

/// Reveal a skin once it has paired. A skin spawns `Hidden` because an unpaired
/// one is a bind-pose statue sitting off the physics body; the moment its bones
/// are driven it should show. The colliders themselves are never rendered as
/// meshes (Rapier's debug-render is the physics view), so the skin is the whole
/// visible crab and just stays on after pairing. Writes only on change.
fn reveal_skin(mut roots: Query<(&CrabSkin, &mut Visibility)>) {
    for (skin, mut vis) in roots.iter_mut() {
        if skin.paired && *vis != Visibility::Visible {
            *vis = Visibility::Visible;
        }
    }
}

/// Every frame: driven bones follow their physics link. Links are top-level
/// entities, so their `Transform` is already world-space (and fresher than
/// `GlobalTransform`, which lags until propagation). Bones are top-level
/// children of an identity root, so writing world poses into `Transform` is
/// exact.
fn drive_bones(
    mut bones: Query<(&BoneDrive, &mut Transform)>,
    links: Query<&Transform, (With<CrabBodyPart>, Without<BoneDrive>)>,
) {
    for (drive, mut t) in bones.iter_mut() {
        if let Ok(link) = links.get(drive.link) {
            *t = Transform::from_matrix(link.to_matrix() * drive.offset);
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-part skin-weight strip (bddap/rl#32)
// ---------------------------------------------------------------------------

/// Meshes whose weights have already been confined to each vertex's dominant
/// part. The skinned mesh asset is shared across crab instances, so the rewrite
/// must happen exactly once per asset; this records the ones done.
#[derive(Resource, Default)]
struct StrippedMeshes(HashSet<AssetId<Mesh>>);

/// Confine every vertex's skin weights to its dominant physics part, once per
/// mesh asset.
///
/// WHY: sally.glb's skin paints cross-part bleed — many CARAPACE/shell vertices
/// carry stray `WEIGHTS_0` on arm bones (chiefly `ClawShoulder`, whose `members`
/// span the whole arm). Our deform bones are driven RIGIDLY by the physics links
/// ([`drive_bones`]), so standard GPU linear-blend skinning drags those shell
/// verts along when the arm moves and rips the rigid carapace into a bulge
/// ([`super::rig::part_for_bone`] is what assigns each bone its part). The physics
/// carapace is a single rigid box and never deforms; this is purely a skinning
/// artifact, so the fix lives entirely in the cosmetic mesh.
///
/// FIX: per vertex, sum weight per part across its (≤4) lanes, find the dominant
/// part, zero every lane outside it, and renormalize. Symmetric: an arm vertex
/// keeps its arm weights (the limb still articulates) and loses stray carapace
/// weight; a shell vertex keeps Carapace and loses stray arm weight (the shell
/// stays rigid). This only touches weights, so the bind pose is pixel-identical
/// (a vertex at rest sits at the same place regardless of which bones could move
/// it). It also fixes the minor leg-coxa bleed, not just ClawShoulder.
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
            lane_parts.push(super::rig::part_for_bone(name.as_str()).unwrap_or(PartId::Carapace));
        }
        if !all_named {
            continue;
        }

        let Some(mesh) = meshes.get_mut(&mesh3d.0) else {
            continue; // asset not loaded yet
        };
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

        let new_weights: Vec<[f32; 4]> = joints
            .iter()
            .zip(&weights)
            .map(|(j, w)| strip_to_dominant_part(*j, *w, &lane_parts))
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
/// stored format. Shared with `skin_diag` (which also walks skin joints/weights).
pub(crate) fn read_u16x4(
    mesh: &Mesh,
    attr: bevy::mesh::MeshVertexAttribute,
) -> Option<Vec<[u16; 4]>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Uint16x4(v) => Some(v.clone()),
        _ => None,
    }
}

/// Read a `Float32x4` mesh attribute as `[f32; 4]` lanes, or `None` for any other
/// stored format.
pub(crate) fn read_f32x4(
    mesh: &Mesh,
    attr: bevy::mesh::MeshVertexAttribute,
) -> Option<Vec<[f32; 4]>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Float32x4(v) => Some(v.clone()),
        _ => None,
    }
}

/// Confine one vertex's skin weights to its dominant part. `lane_parts[i]` is the
/// part of skin-joint lane `i`; `joints`/`weights` are the vertex's (≤4)
/// `(joint_index, weight)` lanes. Returns the rewritten weights: every lane whose
/// joint's part ≠ the dominant part is zeroed, and the survivors are renormalized
/// to sum to 1.0.
///
/// The dominant part is the one with the largest summed weight across the lanes.
/// A vertex with no weight at all is returned unchanged.
fn strip_to_dominant_part(joints: [u16; 4], weights: [f32; 4], lane_parts: &[PartId]) -> [f32; 4] {
    let part_of = |lane: usize| -> PartId {
        lane_parts
            .get(joints[lane] as usize)
            .copied()
            .unwrap_or(PartId::Carapace)
    };

    // Sum weight per part, tracking the dominant (largest-sum) part.
    let mut sums: Vec<(PartId, f32)> = Vec::new();
    for (lane, &w) in weights.iter().enumerate() {
        if w <= 0.0 {
            continue;
        }
        let part = part_of(lane);
        match sums.iter_mut().find(|(p, _)| *p == part) {
            Some((_, s)) => *s += w,
            None => sums.push((part, w)),
        }
    }
    let Some(&(dominant, _)) = sums
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return weights; // all-zero vertex: leave it be
    };

    // Keep only lanes in the dominant part.
    let mut kept = [0.0f32; 4];
    let mut kept_sum = 0.0f32;
    for lane in 0..4 {
        if weights[lane] > 0.0 && part_of(lane) == dominant {
            kept[lane] = weights[lane];
            kept_sum += weights[lane];
        }
    }

    // The dominant part has the max per-part sum, so at least one kept lane is
    // > 0: survivors always exist to renormalize against.
    debug_assert!(kept_sum > 0.0, "dominant part must carry positive weight");
    for w in &mut kept {
        *w /= kept_sum;
    }
    kept
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::*;

    use super::super::body::{
        CrabAssets, CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, Side,
    };
    use super::super::test_util::{headless_app, tick};
    use super::super::{CrabSpawns, respawn_crab};
    use super::{
        BoneDrive, CrabSkin, PartId, SETTLE_FRAMES, repair_skins, reveal_skin,
        strip_to_dominant_part,
    };

    /// The strip (bddap/rl#32): after confining each vertex to its dominant part,
    /// every surviving weight must belong to one part, the lane weights still sum to
    /// ~1.0, and a shell vertex carrying stray arm weight no longer drives the arm.
    /// Pure (no GPU): a hand-built lane→part map stands in for the resolved joints.
    #[test]
    fn strip_confines_weights_to_dominant_part() {
        let arm = PartId::Joint(CrabJointId::ClawShoulder(Side::Left));
        // Lane layout: 0,1 → Carapace; 2,3 → the arm.
        let lane_parts = [PartId::Carapace, PartId::Carapace, arm, arm];
        // Lane→part for a vertex's joint indices, used to re-check the result.
        let part_of = |joints: [u16; 4], lane: usize| lane_parts[joints[lane] as usize];

        // Shell vertex: dominated by carapace (0.7+0.1) with stray arm weight (0.2).
        // The arm lane (joint index 2) must end up at zero so ClawShoulder no longer
        // drags it, and the carapace lanes renormalize to sum to 1.
        let shell_joints = [0u16, 1, 2, 3];
        let shell = strip_to_dominant_part(shell_joints, [0.7, 0.1, 0.15, 0.05], &lane_parts);
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

        // Arm vertex: dominated by the arm (0.6+0.3) with stray carapace weight
        // (0.1). It must keep its arm lanes (the limb still articulates) and drop
        // the carapace lane.
        let arm_joints = [0u16, 2, 3, 1];
        let arm_v = strip_to_dominant_part(arm_joints, [0.1, 0.6, 0.3, 0.0], &lane_parts);
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
            "stray carapace weight must be zeroed on the arm"
        );

        // Single-part vertex is unchanged (already sums to 1, one part).
        let solo = strip_to_dominant_part([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0], &lane_parts);
        assert_eq!(solo, [0.5, 0.5, 0.0, 0.0]);

        // All-zero vertex is left alone (no part to dominate).
        let empty = strip_to_dominant_part([0, 0, 0, 0], [0.0; 4], &lane_parts);
        assert_eq!(empty, [0.0; 4]);
    }

    /// The role-resolving half of pairing: a bone name must map to the physics
    /// link the re-pair then targets. Pins the cases the test relies on so a
    /// mapping change can't make the re-pair test pass against the wrong link —
    /// including a now-coxa bone (`002`) that the old divergent map sent to the
    /// merus, which is what forked the rendered legs.
    #[test]
    fn bone_names_resolve_to_expected_links() {
        use super::super::rig::part_for_bone;
        assert_eq!(part_for_bone("Def_shell"), Some(PartId::Carapace));
        assert_eq!(
            part_for_bone("Def_leg_01.000.L"),
            Some(PartId::Joint(CrabJointId::LegCoxa(Side::Left, 0)))
        );
        assert_eq!(
            part_for_bone("Def_leg_01.002.L"),
            Some(PartId::Joint(CrabJointId::LegCoxa(Side::Left, 0)))
        );
        assert_eq!(
            part_for_bone("Def_leg_01.003.L"),
            Some(PartId::Joint(CrabJointId::LegMerus(Side::Left, 0)))
        );
    }

    /// The re-pair invariant: after an episode reset ([`respawn_crab`]) replaces
    /// an env's physics parts, a paired skin's bones must point at the NEW, live
    /// parts — not the despawned ones [`drive_bones`] would otherwise read as
    /// frozen garbage. Builds a skin by hand (no glTF needed: the bone→link
    /// contract is name-based) bound to the live crab, resets, and checks every
    /// bone re-homed onto the matching fresh part.
    #[test]
    fn skin_repairs_onto_fresh_parts_after_respawn() {
        let mut app = headless_app();
        // `repair_skins` is only auto-registered with a model loaded; add it
        // alone so the test drives exactly the system under test.
        app.add_systems(PostUpdate, repair_skins);
        tick(&mut app, 192); // spawn + settle the env-0 crab

        // Bind a synthetic skin to two live parts of distinct roles: the
        // carapace and the left front coxa.
        let carapace = find_part(&mut app, Role::Carapace);
        let coxa = find_part(&mut app, Role::Coxa);
        let (shell_bone, leg_bone) = app
            .world_mut()
            .run_system_once(move |mut commands: Commands| -> (Entity, Entity) {
                let root = commands
                    .spawn((
                        CrabSkin {
                            env: 0,
                            scene_frames: Some(SETTLE_FRAMES),
                            paired: true,
                        },
                        Transform::default(),
                        Visibility::Visible,
                    ))
                    .id();
                let bone = |commands: &mut Commands, name: &str, link: Entity| {
                    commands
                        .spawn((
                            BoneDrive {
                                link,
                                offset: Mat4::IDENTITY,
                            },
                            Name::new(name.to_owned()),
                            ChildOf(root),
                            Transform::default(),
                        ))
                        .id()
                };
                let shell = bone(&mut commands, "Def_shell", carapace);
                let leg = bone(&mut commands, "Def_leg_01.000.L", coxa);
                (shell, leg)
            })
            .unwrap();

        // Precondition: bones currently target the live OLD parts (not stale),
        // so the test proves the reset transition, not an already-correct state.
        assert_eq!(bone_link(&mut app, shell_bone), carapace);
        assert_eq!(bone_link(&mut app, leg_bone), coxa);

        respawn_env0(&mut app);
        // One update flushes the despawn+respawn and runs `repair_skins`.
        tick(&mut app, 1);

        // The old part entities are gone; the bones must now point at live ones.
        let new_carapace = find_part(&mut app, Role::Carapace);
        let new_coxa = find_part(&mut app, Role::Coxa);
        assert_ne!(new_carapace, carapace, "respawn should make new entities");
        assert_eq!(
            bone_link(&mut app, shell_bone),
            new_carapace,
            "shell bone must re-home onto the fresh carapace"
        );
        assert_eq!(
            bone_link(&mut app, leg_bone),
            new_coxa,
            "leg bone must re-home onto the fresh coxa"
        );
        // And both targets are live entities (a despawned id would not resolve).
        assert!(is_live(&mut app, new_carapace) && is_live(&mut app, new_coxa));
    }

    /// `reveal_skin` shows a skin only once it pairs: an unpaired root is a
    /// bind-pose statue off the body, so it must stay hidden, and the moment
    /// `paired` flips the skin must become (and remain) visible.
    #[test]
    fn reveal_skin_shows_only_after_pairing() {
        let mut app = headless_app();
        app.add_systems(Update, reveal_skin);

        let root = app
            .world_mut()
            .run_system_once(|mut commands: Commands| -> Entity {
                commands
                    .spawn((
                        CrabSkin {
                            env: 0,
                            scene_frames: None,
                            paired: false,
                        },
                        Transform::default(),
                        Visibility::Hidden,
                    ))
                    .id()
            })
            .unwrap();
        let visible = |app: &App| {
            matches!(
                app.world().get::<Visibility>(root),
                Some(Visibility::Visible)
            )
        };

        // Unpaired: stays hidden.
        tick(&mut app, 1);
        assert!(!visible(&app), "unpaired skin must stay hidden");

        // Pair it; reveal_skin flips it visible and leaves it there.
        app.world_mut().get_mut::<CrabSkin>(root).unwrap().paired = true;
        tick(&mut app, 1);
        assert!(visible(&app), "paired skin must become visible");
        tick(&mut app, 1);
        assert!(visible(&app), "paired skin must stay visible");
    }

    enum Role {
        Carapace,
        Coxa,
    }

    fn find_part(app: &mut App, role: Role) -> Entity {
        match role {
            Role::Carapace => {
                let mut q = app
                    .world_mut()
                    .query_filtered::<(Entity, &CrabEnvId), With<CrabCarapace>>();
                q.iter(app.world())
                    .find(|(_, e)| e.0 == 0)
                    .map(|(e, _)| e)
                    .expect("carapace")
            }
            Role::Coxa => {
                let mut q = app.world_mut().query::<(Entity, &CrabEnvId, &CrabJoint)>();
                q.iter(app.world())
                    .find(|(_, e, j)| e.0 == 0 && j.id == CrabJointId::LegCoxa(Side::Left, 0))
                    .map(|(e, _, _)| e)
                    .expect("left front coxa")
            }
        }
    }

    fn bone_link(app: &mut App, bone: Entity) -> Entity {
        app.world().get::<BoneDrive>(bone).expect("bone").link
    }

    fn is_live(app: &mut App, e: Entity) -> bool {
        let mut q = app.world_mut().query_filtered::<(), With<CrabBodyPart>>();
        q.get(app.world(), e).is_ok()
    }

    fn respawn_env0(app: &mut App) {
        app.world_mut()
            .run_system_once(
                |mut commands: Commands,
                 assets: Res<CrabAssets>,
                 spawns: Res<CrabSpawns>,
                 parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>| {
                    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
                    respawn_crab(
                        &mut commands,
                        &assets,
                        parts.iter().filter(|(_, id)| id.0 == 0).map(|(e, _)| e),
                        origin,
                        0,
                    );
                },
            )
            .expect("respawn system");
    }
}
