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
    phase: Pairing,
}

/// The skin's pairing lifecycle. A skin spawns before its glTF scene has
/// instantiated, settles with the body, then pairs its bones to the physics
/// links — strictly in that order. One enum makes the settle counter live only
/// inside [`Settling`](Pairing::Settling), so being [`Paired`](Pairing::Paired)
/// with a settle still outstanding is unrepresentable rather than merely avoided.
enum Pairing {
    /// Scene instance has no children yet, so its bind-pose GlobalTransforms
    /// aren't readable. The skin is a bind-pose statue off the body, kept hidden.
    Spawning,
    /// Scene is up; counting render frames so the body can settle before offsets
    /// are captured. Offsets taken against the crab still up at spawn height bake
    /// the subsequent ~0.2 m settle into every carapace-bound bone and sink the
    /// shell into the ground, so capture waits for [`SETTLE_FRAMES`]. Still hidden.
    Settling { frames: u32 },
    /// Bones captured and driven; `BoneDrive`s exist below this root and the skin
    /// is shown.
    Paired,
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
    // Skin iff a model resolves — the SAME `model_path` the body uses, so skin and
    // physics can't disagree about asset presence. The AssetServer wants an
    // asset-root-RELATIVE path, so feed it the relative `CRAB_MODEL_PATH` (default
    // `sally.glb`), not the absolute `model_path`.
    if super::meshfit::model_path().is_none() {
        return;
    }
    let rel = std::env::var("CRAB_MODEL_PATH").unwrap_or_else(|_| "sally.glb".to_string());
    let scene = app
        .world()
        .resource::<AssetServer>()
        .load(GltfAssetLabel::Scene(0).from_asset(rel));
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
    // (`register` only runs the skin systems when a model resolves, so the hub here is
    // always the real model's — the fallback body shows the debug wireframe, no skin.)
    let hub = assets.hub_bind_world();
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
                phase: Pairing::Spawning,
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
/// driven bones under the root. Reaching [`Pairing::Paired`] hands visibility to
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
        match skin.phase {
            Pairing::Paired => continue,
            Pairing::Spawning => {
                if children.get(root).is_ok_and(|c| !c.is_empty()) {
                    skin.phase = Pairing::Settling { frames: 0 };
                }
                continue;
            }
            Pairing::Settling { frames } if frames < SETTLE_FRAMES => {
                skin.phase = Pairing::Settling {
                    frames: frames + 1,
                };
                continue;
            }
            Pairing::Settling { .. } => {}
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
        skin.phase = Pairing::Paired;
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
        if !matches!(skin.phase, Pairing::Paired) {
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
        if matches!(skin.phase, Pairing::Paired) && *vis != Visibility::Visible {
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

/// Confine every vertex's skin weights to its dominant physics part's hinge
/// cluster, once per mesh asset.
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
/// WHY NOT pure winner-take-all: zeroing *every* non-dominant lane also amputates
/// the legitimate blend at an ADJACENT joint seam, welding a seam vertex rigidly to
/// one link so it drags off the other when that link moves — the dactyl-knuckle and
/// rear-leg/shell drag. [`strip_to_dominant_cluster`] (the per-vertex rule) keeps a
/// lane on any part *hinged* to the dominant one, so a seam flexes, but still strips
/// a disjoint cross-weight (and any limb lane on a carapace-dominant shell vertex, so
/// the rigid trunk never bulges).
///
/// WHY the geometric carapace-region override (bddap/rl#37): the cluster strip keys
/// off *dominance*, so a shell vertex the artist weighted limb-heavy is "limb-dominant"
/// and keeps its limb lanes — and bulges. Weight cannot separate "shell vertex authored
/// limb-heavy" from "limb vertex near the shell"; only position can. So every shell-flesh
/// vertex (one carrying any carapace weight) sitting inside the [`carapace_region`] AABB
/// is forced rigidly onto the carapace via [`confine_to_rigid`], regardless of its
/// authored weights. A vertex with no carapace weight is a limb passing through the box
/// (a leg/claw socket stub), so it is left to the cluster strip — confining it would zero
/// its only lanes and collapse it to the origin. Still weight-only output: a vertex's
/// REST position is its bind pose (all bones at bind), unchanged by which bones could move
/// it, so the strip is pixel-identical at rest and only changes how the surface deforms.
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
        // Positions (raw mesh-local, the frame skinning starts from — GPU LBS moves
        // them, so the CPU attribute is the bind-pose-local position) define the
        // geometric carapace region below. Cloned out so the borrow ends before the
        // `&mut mesh` write; `None` (no/oddly-typed POSITION) just skips the region
        // override and falls back to the per-vertex cluster strip.
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
        // positions are unavailable (then the region override is disabled and every
        // vertex takes the cluster strip). See `confine_vertex` for the per-vertex rule.
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

/// The per-vertex rule the strip applies (bddap/rl#37). A shell-flesh vertex (one with
/// any carapace weight) whose `pos` lies inside the carapace `region` is forced rigidly
/// onto the shell via [`confine_to_rigid`] — geometry overrides authored weight, since
/// weight alone can't tell a shell vertex painted limb-heavy from a limb vertex near the
/// shell. Every other vertex — outside the region, or limb-only (a leg/claw socket stub
/// passing through the box, which confining would zero out) — takes the seam-aware
/// [`strip_to_dominant_cluster`]. `region`/`pos` `None` (positions unavailable) disables
/// the override, leaving the pure weight-based strip.
fn confine_vertex(
    pos: Option<Vec3>,
    region: Option<(Vec3, Vec3)>,
    joints: [u16; 4],
    weights: [f32; 4],
    lane_parts: &[PartId],
) -> [f32; 4] {
    let in_region = match (region, pos) {
        (Some((lo, hi)), Some(p)) => p.cmpge(lo).all() && p.cmple(hi).all(),
        _ => false,
    };
    if in_region && has_rigid_lane(joints, weights, lane_parts) {
        confine_to_rigid(joints, weights, lane_parts)
    } else {
        strip_to_dominant_cluster(joints, weights, lane_parts)
    }
}

/// Confine one vertex's skin weights to its dominant part *and the limb parts joined
/// to it at a rig hinge*. `lane_parts[i]` is the part of skin-joint lane `i`;
/// `joints`/`weights` are the vertex's (≤4) `(joint_index, weight)` lanes. Returns
/// the rewritten weights, renormalized to sum to 1.0.
///
/// The dominant part is the one with the largest summed weight across the lanes. A
/// vertex with no weight at all is returned unchanged.
///
/// A lane survives iff its part is the dominant part or is
/// [`adjacent`](super::rig::parts_adjacent) to it — with one asymmetry for the
/// carapace, which alone among the parts is a single rigid box that never deforms:
///
/// - When a **limb joint** is dominant, keep every adjacent lane, the carapace
///   included. Two parts that meet at a joint (claw-chain neighbours, a chain root
///   and the carapace) share flesh the authored skin blends across the hinge;
///   zeroing the loser there welds the seam rigidly to one link, so it drags off the
///   other when that link moves — the #32 dactyl knuckle and the rear-leg/shell seam.
///   Blending keeps the seam vertex on both, so it bends with the joint.
/// - When the **carapace** is dominant (a shell vertex), keep ONLY the carapace and
///   strip every limb lane, even an adjacent chain root's. The shell is rigid; a
///   stray arm/leg lane there has the moving limb tug the shell into a bulge — the
///   #262 artifact — and a shell vertex that legitimately blended would be limb-, not
///   carapace-, dominant and is handled by the case above. So the rigid trunk is
///   never deformed by a limb, while limb seams still flex.
///
/// Disjoint parts (the carapace and a distal arm bone; two different limbs) share no
/// joint, so a cross-weight there is pure bleed and is stripped in either case. A
/// surviving blend reproduces the renderer's bind-pose weights, so at rest the
/// skinned surface is pixel-identical.
fn strip_to_dominant_cluster(
    joints: [u16; 4],
    weights: [f32; 4],
    lane_parts: &[PartId],
) -> [f32; 4] {
    let Some(dominant) = dominant_part(joints, weights, lane_parts) else {
        return weights; // all-zero vertex: leave it be
    };

    // A lane is kept if it is the dominant part, or — unless a rigid part (the
    // carapace) is what dominates — a part hinged to it. A seam vertex on a limb bends
    // with both links; a shell vertex stays welded to the shell so no limb can deform
    // it (see `PartId::is_rigid`).
    let keep = |p: PartId| -> bool {
        p == dominant || (!dominant.is_rigid() && super::rig::parts_adjacent(dominant, p))
    };
    renormalize_kept(joints, weights, lane_parts, keep)
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
/// lanes — or `None` for an all-zero vertex. The single source for "which part owns
/// this vertex", shared by the cluster strip and the carapace-region builder so the
/// two can't disagree about dominance.
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

/// Zero every lane whose part `keep` rejects, then renormalize the survivors to sum
/// to 1.0. The caller guarantees at least one lane is kept (so the sum is positive).
fn renormalize_kept(
    joints: [u16; 4],
    weights: [f32; 4],
    lane_parts: &[PartId],
    keep: impl Fn(PartId) -> bool,
) -> [f32; 4] {
    let mut kept = [0.0f32; 4];
    let mut kept_sum = 0.0f32;
    for lane in 0..4 {
        if weights[lane] > 0.0 && keep(lane_part(joints, lane, lane_parts)) {
            kept[lane] = weights[lane];
            kept_sum += weights[lane];
        }
    }
    debug_assert!(
        kept_sum > 0.0,
        "at least one lane must be kept to renormalize"
    );
    for w in &mut kept {
        *w /= kept_sum;
    }
    kept
}

/// Whether this vertex carries any weight on a rigid (carapace) bone — i.e. whether
/// it is bound to the shell at all. The carapace-region override only confines such
/// vertices; one with no rigid lane is a limb passing through the region (a leg/claw
/// socket stub), and zeroing its only (limb) lanes would collapse it to the origin.
fn has_rigid_lane(joints: [u16; 4], weights: [f32; 4], lane_parts: &[PartId]) -> bool {
    (0..4).any(|l| weights[l] > 0.0 && lane_part(joints, l, lane_parts).is_rigid())
}

/// Confine a shell-flesh vertex to the rigid shell: drop every articulated-joint lane
/// and renormalize onto the carapace lane(s). The geometric override for vertices
/// inside the carapace region (bddap/rl#37) — unlike the weight-based cluster strip,
/// it strips even an adjacent chain-root lane, because position (not weight) has
/// already established the vertex is shell, so no limb may tug it. `has_rigid_lane`
/// must hold (a carapace lane exists to renormalize onto).
fn confine_to_rigid(joints: [u16; 4], weights: [f32; 4], lane_parts: &[PartId]) -> [f32; 4] {
    renormalize_kept(joints, weights, lane_parts, PartId::is_rigid)
}

/// The carapace region: the AABB of every shell-dominant vertex, in the mesh's raw
/// position frame. Geometry, not weight, is what separates a shell vertex authored
/// limb-heavy from a limb vertex near the shell (bddap/rl#37); a shell-flesh vertex
/// inside this box is forced rigidly onto the carapace. Returns `None` if no vertex is
/// shell-dominant (a non-crab mesh), which disables the override (the cluster strip
/// still runs).
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
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::*;

    use super::super::body::{
        CrabAssets, CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, Side,
    };
    use super::super::test_util::{headless_app, tick};
    use super::super::{CrabSpawns, respawn_crab};
    use super::{
        BoneDrive, CrabSkin, Pairing, PartId, carapace_region, confine_vertex, has_rigid_lane,
        repair_skins, reveal_skin, strip_to_dominant_cluster,
    };

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
        let shell = strip_to_dominant_cluster(shell_joints, [0.7, 0.1, 0.15, 0.05], &lane_parts);
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
        let arm_v = strip_to_dominant_cluster(arm_joints, [0.1, 0.6, 0.3, 0.0], &lane_parts);
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
        let solo = strip_to_dominant_cluster([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0], &lane_parts);
        assert_eq!(solo, [0.5, 0.5, 0.0, 0.0]);

        // All-zero vertex is left alone (no part to dominate).
        let empty = strip_to_dominant_cluster([0, 0, 0, 0], [0.0; 4], &lane_parts);
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
                            phase: Pairing::Paired,
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
                            phase: Pairing::Spawning,
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
        app.world_mut().get_mut::<CrabSkin>(root).unwrap().phase = Pairing::Paired;
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

    /// Two parts in the same limb chain (claw-chain neighbours, leg-chain neighbours)
    /// and each chain root with the carapace must be adjacent; two parts on different
    /// limbs, or the carapace with a non-root joint, must NOT be. Pins the adjacency
    /// the strip keys off so a future chain edit can't silently re-confine a real seam
    /// or fuse a disjoint pair.
    #[test]
    fn adjacency_matches_the_joint_chains() {
        use super::super::rig::parts_adjacent;
        let car = PartId::Carapace;
        let coxa_r0 = PartId::Joint(CrabJointId::LegCoxa(Side::Right, 0));
        let merus_r0 = PartId::Joint(CrabJointId::LegMerus(Side::Right, 0));
        let carpus_r0 = PartId::Joint(CrabJointId::LegCarpus(Side::Right, 0));
        let shoulder_r = PartId::Joint(CrabJointId::ClawShoulder(Side::Right));
        let wrist_r = PartId::Joint(CrabJointId::ClawWrist(Side::Right));
        let pincer_r = PartId::Joint(CrabJointId::ClawPincer(Side::Right));
        let coxa_r1 = PartId::Joint(CrabJointId::LegCoxa(Side::Right, 1));

        // Adjacent: chain roots to the carapace, and within-chain neighbours.
        assert!(parts_adjacent(car, coxa_r0));
        assert!(parts_adjacent(coxa_r0, car)); // symmetric
        assert!(parts_adjacent(car, shoulder_r));
        assert!(parts_adjacent(coxa_r0, merus_r0));
        assert!(parts_adjacent(merus_r0, carpus_r0));
        assert!(parts_adjacent(shoulder_r, wrist_r));
        assert!(parts_adjacent(wrist_r, pincer_r)); // the #32 thumb seam

        // NOT adjacent: the carapace to a non-root joint (the #262 bleed must stay
        // confined), across-the-chain skips, and two unrelated limbs.
        assert!(!parts_adjacent(car, merus_r0));
        assert!(!parts_adjacent(car, wrist_r));
        assert!(!parts_adjacent(car, pincer_r));
        assert!(!parts_adjacent(coxa_r0, carpus_r0)); // skips the merus
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
        let knuckle = strip_to_dominant_cluster([1, 0, 0, 0], [0.6, 0.4, 0.0, 0.0], &lane_parts);
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
        let leg_seam = strip_to_dominant_cluster([3, 2, 0, 0], [0.65, 0.35, 0.0, 0.0], &lane_parts);
        assert!((leg_seam.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(
            leg_seam[1] > 0.0,
            "carapace lane must survive on a coxa-dominant seam vert: {leg_seam:?}"
        );

        // Pincer dominant with a DISJOINT carapace lane (skips wrist+shoulder, no shared
        // hinge): the carapace lane is still zeroed — only adjacent neighbours blend.
        let disjoint = strip_to_dominant_cluster([1, 2, 0, 0], [0.7, 0.3, 0.0, 0.0], &lane_parts);
        assert_eq!(disjoint[1], 0.0, "disjoint carapace lane must be zeroed");
        assert!((disjoint[0] - 1.0).abs() < 1e-6, "winner renormalized to 1");

        // The #262 regression guard, the hard case: a shell vertex (carapace dominant)
        // carrying a stray SHOULDER lane. The shoulder IS a chain root hinged to the
        // carapace, yet because the rigid shell is what dominates, the limb lane is
        // still stripped — the arm must never tug the trunk into a bulge. (Pincer too,
        // which isn't even adjacent.)
        let shell = strip_to_dominant_cluster([2, 4, 1, 0], [0.7, 0.2, 0.1, 0.0], &lane_parts);
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
    /// the old winner-take-all strip with the new cluster strip. Read off the file, so
    /// it is independent of the Bevy spawn path.
    ///
    /// For each ADJACENT (winner → anchor) seam it prints the vertex count and the mean
    /// weight the vertex keeps on its anchor (the part it sits on but does not win)
    /// after each rule — the old rule zeroes it (drags fully off the anchor), the new
    /// rule keeps it where the anchor is allowed to flex. Because drag distance under a
    /// fixed joint rotation is proportional to the winning part's weight, the anchor
    /// weight handed back IS the fractional drag removed. The carapace-as-winner rows
    /// stay confined by design (the rigid shell never deforms), so their anchor weight
    /// is zero under both rules — that is the #262 guarantee, not a miss. A disjoint
    /// (non-adjacent) cross-weight must stay confined under both; the test fails if any
    /// leaks through. Skips cleanly when the model isn't present.
    #[test]
    fn seam_drag_audit_on_model() {
        use super::super::meshfit::model_path;
        use super::super::rig::{part_for_bone, parts_adjacent};

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
        let dominant_of = |js: [u16; 4], ws: [f32; 4]| -> PartId {
            let mut sums: Vec<(PartId, f32)> = Vec::new();
            for l in 0..4 {
                if ws[l] <= 0.0 {
                    continue;
                }
                let p = lane_parts[js[l] as usize];
                match sums.iter_mut().find(|(q, _)| *q == p) {
                    Some((_, s)) => *s += ws[l],
                    None => sums.push((p, ws[l])),
                }
            }
            sums.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(p, _)| *p)
                .unwrap()
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
                let dom = dominant_of(j, w);
                let new = strip_to_dominant_cluster(j, w, &lane_parts);
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
            "seam (winner -> anchor)", "verts", "old anchor wt", "new anchor wt"
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
    /// dispatch the live strip ([`strip_cross_part_weights`]) does, and asserts that
    /// EVERY shell-flesh vertex (one carrying any carapace weight) inside that region
    /// resolves to carapace-only — zero weight on any articulated joint bone.
    ///
    /// This FAILS on the pre-#37 cluster-strip-only code: a shell vertex the artist
    /// weighted limb-heavy is limb-dominant, so the cluster strip keeps its limb lanes
    /// and the rigid shell bulges with the joint (~1.2k such verts in sally.glb). The
    /// geometric override drives that count to zero. Limb-only verts inside the box (a
    /// leg/claw socket stub passing through) are NOT shell flesh and are excluded — they
    /// must keep articulating, and confining them would zero their only lanes. Read off
    /// the file, independent of the Bevy spawn path; skips cleanly with no model.
    #[test]
    fn carapace_region_verts_never_deform() {
        use super::super::meshfit::model_path;
        use super::super::rig::part_for_bone;

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

        let mut shell_in_region = 0usize; // shell-flesh verts the override governs
        let mut deforming = 0usize; // …that STILL deform (the bug; must be 0)
        for i in 0..positions.len() {
            let (j, w) = (joints[i], weights[i]);
            if w.iter().all(|&x| x <= 0.0) || !inside(positions[i]) {
                continue;
            }
            // Limb-only verts inside the box are limb flesh passing through, not shell —
            // the override skips them (so does this assertion), and they keep flexing.
            if !has_rigid_lane(j, w, &lane_parts) {
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

    /// The geometric carapace-region rule ([`confine_vertex`], bddap/rl#37), pure (no
    /// GPU): a hand-built lane→part map and explicit positions stand in for the model.
    /// Pins the three branches the model test can only assert in aggregate — and unlike
    /// the weight-based strip, the override fires on a vertex the cluster strip would
    /// have LEFT articulating (limb-dominant), which is exactly the #37 bug.
    #[test]
    fn carapace_region_override_confines_shell_verts() {
        let arm = PartId::Joint(CrabJointId::ClawShoulder(Side::Left));
        // Lane layout: 0,1 → Carapace; 2,3 → an arm bone.
        let lane_parts = [PartId::Carapace, PartId::Carapace, arm, arm];
        let region = Some((Vec3::splat(-1.0), Vec3::splat(1.0)));
        let in_box = Vec3::ZERO; // inside the region
        let out_box = Vec3::splat(5.0); // outside it

        // The #37 case: a shell vertex the artist weighted ARM-heavy (arm dominant,
        // 0.7 > 0.3). The weight-based strip keeps the arm lanes (limb-dominant), so the
        // shell would bulge — but inside the region the geometric override forces it onto
        // the carapace, renormalizing its shell lane(s) to 1 and zeroing the arm.
        let js = [0u16, 2, 3, 1];
        let ws = [0.2, 0.5, 0.2, 0.1];
        // Confirm the OLD rule leaks (so the override is doing real work here).
        let old = strip_to_dominant_cluster(js, ws, &lane_parts);
        assert!(
            old[1] + old[2] > 1e-6,
            "precondition: cluster strip leaves this shell vert limb-weighted: {old:?}"
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
        // passing through — the override skips it (confining would zero it), so it takes
        // the cluster strip and keeps articulating.
        let limb = [2u16, 3, 0, 0];
        let limb_w = [0.6, 0.4, 0.0, 0.0];
        let limb_new = confine_vertex(Some(in_box), region, limb, limb_w, &lane_parts);
        assert_eq!(
            limb_new,
            strip_to_dominant_cluster(limb, limb_w, &lane_parts),
            "limb-only vert in region keeps the cluster strip, not the override"
        );
        assert!(
            limb_new[0] + limb_new[1] > 0.99,
            "limb stub still articulates"
        );

        // The SAME shell vertex OUTSIDE the region takes the cluster strip (the override
        // is geometric — it governs only the shell box).
        let outside = confine_vertex(Some(out_box), region, js, ws, &lane_parts);
        assert_eq!(outside, strip_to_dominant_cluster(js, ws, &lane_parts));
    }
}
