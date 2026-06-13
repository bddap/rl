//! Optional skinned crab model riding the physics body.
//!
//! When `CRAB_MODEL_PATH` names a glTF inside the app's `assets/` directory
//! (e.g. `sally.glb`, fetched from the private bddap-bot/rl-assets repo), each
//! crab gets a skinned-mesh skin whose deform bones follow the physics links.
//! Without the env var the primitive meshes render as before — the physics
//! body stays the single source of truth, the model is cosmetic.
//!
//! How following works: when the scene instance is ready, every deform bone is
//! matched to a physics link by name (`bone_target`), and the bone's world
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
//! `RL_SKIN_OVERLAY=1` keeps the primitive meshes visible under the skin —
//! the render-vs-physics debug view.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::prelude::*;

use super::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, Side};

/// Present only when a model path was configured; all systems key off this.
#[derive(Resource)]
pub struct CrabModel {
    scene: Handle<Scene>,
    /// Keep primitive meshes visible under the skin (debug overlay).
    overlay: bool,
    /// Uniform model scale; the bind pose is captured scaled, so bone offsets
    /// inherit it. The model's own proportions don't exactly match the physics
    /// body — this trades carapace fit against leg reach (`CRAB_MODEL_SCALE`).
    scale: f32,
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

/// Which physics body a deform bone follows.
#[derive(PartialEq, Eq, Hash)]
enum LinkKey {
    Carapace,
    Joint(CrabJointId),
}

pub fn register(app: &mut App) {
    let Ok(path) = std::env::var("CRAB_MODEL_PATH") else {
        return;
    };
    let overlay = std::env::var("RL_SKIN_OVERLAY").is_ok_and(|v| v == "1");
    // 1.2 seats the Sally model's shorter legs/shell on this physics body best
    // (judged by RL_SKIN_OVERLAY screenshots at 1.0/1.2).
    let scale = std::env::var("CRAB_MODEL_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.2);
    let scene = app
        .world()
        .resource::<AssetServer>()
        .load(GltfAssetLabel::Scene(0).from_asset(path));
    app.insert_resource(CrabModel {
        scene,
        overlay,
        scale,
    });
    app.add_systems(Update, (attach_skins, reap_orphan_skins));
    app.add_systems(
        PostUpdate,
        (
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
    crabs: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    skins: Query<&CrabSkin>,
) {
    for (env, t) in crabs.iter() {
        if skins.iter().any(|s| s.env == env.0) {
            continue;
        }
        // Model rest pose has feet at y=0, facing +z — same convention as the
        // physics crab, so the root sits on the ground under the carapace.
        commands.spawn((
            SceneRoot(model.scene.clone()),
            Transform::from_translation(Vec3::new(t.translation.x, 0.0, t.translation.z))
                .with_scale(Vec3::splat(model.scale)),
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
/// GlobalTransforms hold the bind pose), capture per-bone offsets, flatten
/// driven bones under the root, and hide the primitive meshes.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn pair_bones(
    mut commands: Commands,
    model: Res<CrabModel>,
    mut skins: Query<(Entity, &mut CrabSkin)>,
    children: Query<&Children>,
    names: Query<&Name>,
    globals: Query<&GlobalTransform>,
    links: Query<(
        Entity,
        &CrabEnvId,
        Option<&CrabJoint>,
        Option<&CrabCarapace>,
    )>,
    mut visibility: Query<&mut Visibility>,
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

        let link_of: std::collections::HashMap<LinkKey, Entity> = links
            .iter()
            .filter(|(_, env, ..)| env.0 == skin.env)
            .filter_map(|(e, _, joint, carapace)| match (joint, carapace) {
                (Some(j), _) => Some((LinkKey::Joint(j.id), e)),
                (_, Some(_)) => Some((LinkKey::Carapace, e)),
                _ => None,
            })
            .collect();

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
            let Some(key) = bone_target(name.as_str()) else {
                continue;
            };
            let Some(&link) = link_of.get(&key) else {
                continue;
            };
            let (Ok(bone_g), Ok(link_g)) = (globals.get(e), globals.get(link)) else {
                continue;
            };
            let offset = link_g.to_matrix().inverse() * bone_g.to_matrix();
            commands.entity(e).insert((
                BoneDrive { link, offset },
                ChildOf(root),
                Transform::from_matrix(bone_g.to_matrix()),
            ));
            paired += 1;
        }

        // Reveal the now-driven skin; it replaces the primitive look unless
        // overlaying for debug.
        if let Ok(mut vis) = visibility.get_mut(root) {
            *vis = Visibility::Visible;
        }
        if !model.overlay {
            for (e, env, ..) in links.iter() {
                if env.0 == skin.env
                    && let Ok(mut vis) = visibility.get_mut(e)
                {
                    *vis = Visibility::Hidden;
                }
            }
        }
        info!(
            "crab skin paired: env {} ({} bones driven)",
            skin.env, paired
        );
        skin.paired = true;
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

/// Map a glTF deform-bone name to the physics link it follows.
///
/// Model chains are finer than the physics body (6 bones per leg vs 3 links),
/// so consecutive bones share a link: 000/000b→coxa, 001/002→femur,
/// 003/004/005→tibia. The store's "antennae" sit where the eye stalks are —
/// mapped to them. Everything else deformable (shell, thorax, mouth, abdomen,
/// rostrum, palpi…) rides the carapace rigidly.
fn bone_target(name: &str) -> Option<LinkKey> {
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
        // "N.SEG[b].SIDE", N 1-4 front to back — same order as physics 0-3.
        let leg = rest.chars().next()?.to_digit(10)? as u8 - 1;
        let seg = rest.get(2..5)?;
        let side = side?;
        let id = match seg {
            "000" => CrabJointId::LegCoxa(side, leg),
            "001" | "002" => CrabJointId::LegFemur(side, leg),
            _ => CrabJointId::LegTibia(side, leg),
        };
        return Some(LinkKey::Joint(id));
    }
    if name.starts_with("Def_pincer") || name.starts_with("Ctrl_pincer_tail") {
        let side = side?;
        // 000a/000/001 = arm, 002-005 = palm + fixed jaw, 006* = movable jaw.
        let id = if name.contains("006") || name.starts_with("Ctrl_pincer_tail") {
            CrabJointId::ClawPincer(side)
        } else if name.contains("000") || name.contains("001") {
            CrabJointId::ClawUpper(side)
        } else {
            CrabJointId::ClawFore(side)
        };
        return Some(LinkKey::Joint(id));
    }
    if name.starts_with("Def_antennae") {
        return Some(LinkKey::Joint(CrabJointId::EyeStalk(side?)));
    }
    // Shell, thorax, abdomen, mouth, palpi, rostrum, neck… ride the carapace.
    Some(LinkKey::Carapace)
}
