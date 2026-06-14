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
//! the render-vs-physics debug view (and the demo's BOTH view mode).
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
use bevy::prelude::*;

use super::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, Side};

/// Present only when a model path was configured; all systems key off this.
#[derive(Resource)]
pub struct CrabModel {
    scene: Handle<Scene>,
    /// Uniform model scale; the bind pose is captured scaled, so bone offsets
    /// inherit it. The model's own proportions don't exactly match the physics
    /// body — this trades carapace fit against leg reach (`CRAB_MODEL_SCALE`).
    scale: f32,
}

/// Which render layers are shown. The crab is drawn twice — primitive part
/// meshes (one per collider) and the skinned glTF — and this is the single
/// source of truth for which is visible; [`apply_render_view`] enacts it on
/// the skin root and the primitives, and the demo cycles it with a keypress.
/// Toggling is just a [`Visibility`] flip (nothing despawns), so it's instant
/// and stateless.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderView {
    /// Skinned glTF only — the shipped look. Primitives hidden.
    Pretty,
    /// Primitive collider meshes only — the physics truth. Skin hidden.
    Physics,
    /// Both at once: primitives showing THROUGH/under the skin, the
    /// render-vs-physics debug overlay (was `RL_SKIN_OVERLAY=1`).
    Both,
}

impl RenderView {
    /// PRETTY → PHYSICS → BOTH → PRETTY. The demo's view key steps this.
    pub fn next(self) -> Self {
        match self {
            RenderView::Pretty => RenderView::Physics,
            RenderView::Physics => RenderView::Both,
            RenderView::Both => RenderView::Pretty,
        }
    }

    /// Should the primitive collider meshes be drawn in this view?
    fn show_primitives(self) -> bool {
        matches!(self, RenderView::Physics | RenderView::Both)
    }

    /// Should the skinned glTF be drawn in this view? (Only takes effect once a
    /// skin has paired; an unpaired skin stays hidden regardless — it would be
    /// a bind-pose statue.)
    fn show_skin(self) -> bool {
        matches!(self, RenderView::Pretty | RenderView::Both)
    }
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
#[derive(PartialEq, Eq, Hash, Debug)]
enum LinkKey {
    Carapace,
    Joint(CrabJointId),
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
/// each time because a reset replaces the entities; `bone_target` resolves a
/// bone name to a [`LinkKey`], this resolves that key to the live entity.
fn link_map(links: &LinkQuery, env: usize) -> std::collections::HashMap<LinkKey, Entity> {
    links
        .iter()
        .filter(|(_, e, ..)| e.0 == env)
        .filter_map(|(e, _, joint, carapace)| match (joint, carapace) {
            (Some(j), _) => Some((LinkKey::Joint(j.id), e)),
            (_, Some(_)) => Some((LinkKey::Carapace, e)),
            _ => None,
        })
        .collect()
}

pub fn register(app: &mut App) {
    let Ok(path) = std::env::var("CRAB_MODEL_PATH") else {
        return;
    };
    // RL_SKIN_OVERLAY=1 starts in the BOTH (debug overlay) view; otherwise the
    // shipped PRETTY look. The demo cycles it live from here either way.
    let view = if std::env::var("RL_SKIN_OVERLAY").is_ok_and(|v| v == "1") {
        RenderView::Both
    } else {
        RenderView::Pretty
    };
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
    app.insert_resource(CrabModel { scene, scale });
    app.insert_resource(view);
    app.add_systems(Update, (attach_skins, reap_orphan_skins, apply_render_view));
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
/// GlobalTransforms hold the bind pose), capture per-bone offsets and flatten
/// driven bones under the root. Setting `paired` hands visibility to
/// [`apply_render_view`], which reveals the now-driven skin.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn pair_bones(
    mut commands: Commands,
    mut skins: Query<(Entity, &mut CrabSkin)>,
    children: Query<&Children>,
    names: Query<&Name>,
    globals: Query<&GlobalTransform>,
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

        // Visibility (reveal the skin, hide/show primitives) is owned by
        // `apply_render_view`, which acts the moment `paired` flips true.
        info!(
            "crab skin paired: env {} ({} bones driven)",
            skin.env, paired
        );
        skin.paired = true;
    }
}

/// Re-point a reset crab's bones at its fresh physics parts. A paired skin
/// whose env was respawned ([`crate::bot::respawn_crab`]) still drives the
/// despawned parts; this catches that — any one bone's link entity gone dead
/// flags the whole skin stale — and re-resolves every bone from its name to
/// the live part of the same role, keeping the captured offset (the respawn
/// reproduces the same rest pose at the same origin, so it stays exact). No
/// re-settle and no visibility change, so the skin never flickers; re-hiding
/// the fresh primitives to match the view is left to [`apply_render_view`].
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
            if let Some(key) = bone_target(name.as_str())
                && let Some(&link) = link_of.get(&key)
            {
                drive.link = link;
            }
        }
        info!("crab skin re-paired after reset: env {}", skin.env);
    }
}

/// Enact [`RenderView`] on the skin root and the primitive part meshes — the
/// single owner of crab visibility. An unpaired skin stays hidden (it would be
/// a bind-pose statue) and its primitives stay shown, regardless of the view,
/// so the pre-pair settle still looks right; once paired the view decides.
/// Runs every frame so a reset's fresh (visible) primitives are re-corrected
/// without any per-reset bookkeeping. Writes only on change.
#[allow(clippy::type_complexity)]
fn apply_render_view(
    view: Res<RenderView>,
    skins: Query<&CrabSkin>,
    mut parts: Query<(&CrabEnvId, &mut Visibility), (With<CrabBodyPart>, With<Mesh3d>)>,
    mut roots: Query<(&CrabSkin, &mut Visibility), Without<CrabBodyPart>>,
) {
    // Which envs have a paired skin — only those let the view hide primitives /
    // show the skin; an env still waiting to pair shows primitives as a fallback.
    let paired = |env: usize| skins.iter().any(|s| s.env == env && s.paired);

    for (env, mut vis) in parts.iter_mut() {
        let want = if paired(env.0) && !view.show_primitives() {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
        if *vis != want {
            *vis = want;
        }
    }
    for (skin, mut vis) in roots.iter_mut() {
        let want = if skin.paired && view.show_skin() {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
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

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::*;

    use super::super::body::{
        CrabAssets, CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, Side,
    };
    use super::super::test_util::{headless_app, tick};
    use super::super::{CrabSpawns, respawn_crab};
    use super::{BoneDrive, CrabSkin, RenderView, SETTLE_FRAMES, apply_render_view, repair_skins};

    /// The role-resolving half of pairing: a bone name must map to the physics
    /// link the re-pair then targets. Pins the two cases the test relies on so a
    /// `bone_target` change can't make the re-pair test pass against the wrong link.
    #[test]
    fn bone_names_resolve_to_expected_links() {
        assert_eq!(
            super::bone_target("Def_shell"),
            Some(super::LinkKey::Carapace)
        );
        assert_eq!(
            super::bone_target("Def_leg_01.000.L"),
            Some(super::LinkKey::Joint(CrabJointId::LegCoxa(Side::Left, 0)))
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

    /// `apply_render_view` is the sole owner of crab visibility; pin that each
    /// view sets the skin root and the primitive parts as documented — and that
    /// the system's parts/roots queries don't alias (Bevy panics on a conflict,
    /// so merely running it is the disjointness check that the build can't give).
    #[test]
    fn render_view_drives_primitive_and_skin_visibility() {
        let mut app = headless_app();
        app.insert_resource(RenderView::Pretty);
        app.add_systems(Update, apply_render_view);
        tick(&mut app, 192);

        // A paired skin root for env 0 (no glTF: visibility keys off `paired`).
        let root = app
            .world_mut()
            .run_system_once(|mut commands: Commands| -> Entity {
                commands
                    .spawn((
                        CrabSkin {
                            env: 0,
                            scene_frames: Some(SETTLE_FRAMES),
                            paired: true,
                        },
                        Transform::default(),
                        Visibility::Hidden,
                    ))
                    .id()
            })
            .unwrap();

        let root_visible = |app: &App| {
            matches!(
                app.world().get::<Visibility>(root),
                Some(Visibility::Visible)
            )
        };
        let any_part_visible = |app: &mut App| {
            let mut q = app
                .world_mut()
                .query_filtered::<&Visibility, With<CrabBodyPart>>();
            q.iter(app.world())
                .any(|v| matches!(v, Visibility::Visible))
        };
        let all_parts_hidden = |app: &mut App| {
            let mut q = app
                .world_mut()
                .query_filtered::<&Visibility, With<CrabBodyPart>>();
            q.iter(app.world()).all(|v| matches!(v, Visibility::Hidden))
        };

        // PRETTY: skin shown, primitives hidden.
        tick(&mut app, 1);
        assert!(root_visible(&app), "pretty: skin visible");
        assert!(all_parts_hidden(&mut app), "pretty: primitives hidden");

        // PHYSICS: skin hidden, primitives shown.
        *app.world_mut().resource_mut::<RenderView>() = RenderView::Physics;
        tick(&mut app, 1);
        assert!(!root_visible(&app), "physics: skin hidden");
        assert!(any_part_visible(&mut app), "physics: primitives shown");

        // BOTH: skin shown AND primitives shown.
        *app.world_mut().resource_mut::<RenderView>() = RenderView::Both;
        tick(&mut app, 1);
        assert!(root_visible(&app), "both: skin visible");
        assert!(any_part_visible(&mut app), "both: primitives shown");
    }

    enum Role {
        Carapace,
        Coxa,
    }

    /// The single env-0 entity playing `role`.
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
