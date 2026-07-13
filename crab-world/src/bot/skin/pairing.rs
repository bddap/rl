use bevy::camera::visibility::NoFrustumCulling;
use bevy::prelude::*;

use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabRestPose};
use crate::bot::meshfit::PartId;

#[derive(Component)]
struct CrabSkin {
    env: usize,
    phase: Pairing,
}

enum Pairing {
    Spawning,
    Settling { frames: u32 },
    Paired,
}

const SETTLE_FRAMES: u32 = 90;

#[derive(Component)]
struct BoneDrive {
    link: Entity,
    offset: Mat4,
}

#[derive(Resource, Default, Clone)]
pub struct CrabSkinRepose(pub std::collections::BTreeMap<usize, SkinRepose>);

/// The physics-step-clock SAMPLED pose each crab body part renders at this frame, keyed
/// by part entity (rl#274). GCR's articulation sampler rebuilds it per frame on BOTH
/// arms — the host from its own capture, a client from its adopt — so every render
/// consumer (the skin's `drive_bones`, the collider cage, the brain labels) shows motion
/// uniform in render time without anything writing a rapier-owned `Transform` (rl#116).
/// A missing entry means no tick-stamped stream feeds this part (the standalone viewer,
/// training visuals, the window-fill grace frames): consumers render the physics
/// `Transform` directly.
#[derive(Resource, Default)]
pub struct CrabRenderPose(pub std::collections::BTreeMap<Entity, Transform>);

#[derive(Clone, Copy)]
pub struct SkinRepose {
    pub shift: Vec3,
}

impl SkinRepose {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_translation(self.shift)
    }
}

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

pub(super) fn register(app: &mut App) {
    app.init_resource::<CrabSkinRepose>();
    app.init_resource::<CrabRenderPose>();
    app.add_systems(Update, (attach_skins, reap_orphan_skins, reveal_skin));
    app.add_systems(
        PostUpdate,
        (
            repair_skins.before(drive_bones),
            drive_bones.before(TransformSystems::Propagate),
            pair_bones.after(TransformSystems::Propagate),
        ),
    );
}

fn attach_skins(
    mut commands: Commands,
    model: Res<super::CrabModel>,
    assets: Res<crate::bot::body::CrabAssets>,
    crabs: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    skins: Query<&CrabSkin>,
) {
    let hub = assets.hub_bind_world();
    for (env, t) in crabs.iter() {
        if skins.iter().any(|s| s.env == env.0) {
            continue;
        }
        commands.spawn((
            SceneRoot(model.scene.clone()),
            Transform::from_translation(t.translation - hub),
            Visibility::Hidden,
            CrabSkin {
                env: env.0,
                phase: Pairing::Spawning,
            },
        ));
    }
}

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
                skin.phase = Pairing::Settling { frames: frames + 1 };
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
            if meshes.get(e).is_ok() {
                commands.entity(e).insert(NoFrustumCulling);
            }
            let Ok(name) = names.get(e) else { continue };
            let Some(key) = crate::bot::rig::part_for_bone(name.as_str()) else {
                continue;
            };
            let Some(&link) = link_of.get(&key) else {
                continue;
            };
            let (Ok(bone_g), Ok(link_rest)) = (globals.get(e), rest_poses.get(link)) else {
                continue;
            };
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
        commands.entity(root).insert(Transform::default());
        skin.phase = Pairing::Paired;
    }
}

fn repair_skins(
    mut commands: Commands,
    mut bones: Query<(&mut BoneDrive, &Name)>,
    skins: Query<(&CrabSkin, &Children)>,
    links: LinkQuery,
) {
    for (skin, kids) in skins.iter() {
        if !matches!(skin.phase, Pairing::Paired) {
            continue;
        }
        let stale = kids
            .iter()
            .filter_map(|b| bones.get(b).ok())
            .any(|(drive, _)| links.get(drive.link).is_err());
        if !stale {
            continue;
        }

        let link_of = link_map(&links, skin.env);
        let mut repaired = 0usize;
        for bone in kids.iter() {
            let Ok((mut drive, name)) = bones.get_mut(bone) else {
                continue;
            };
            match crate::bot::rig::part_for_bone(name.as_str())
                .and_then(|key| link_of.get(&key).copied())
            {
                Some(link) => {
                    drive.link = link;
                    repaired += 1;
                }
                None => {
                    error!(
                        "crab skin repair: env {} bone {:?} has no live link after respawn \
                         — dropping its dead BoneDrive (skin will be missing this part)",
                        skin.env,
                        name.as_str()
                    );
                    commands.entity(bone).remove::<BoneDrive>();
                }
            }
        }
        info!(
            "crab skin re-paired after reset: env {} ({repaired} bones re-pointed)",
            skin.env
        );
    }
}

fn reveal_skin(
    mode: Option<Res<crate::crab_view::RenderMode>>,
    mut roots: Query<(&CrabSkin, &mut Visibility)>,
) {
    let show_mesh = mode.map(|m| m.shows_mesh()).unwrap_or(true);
    for (skin, mut vis) in roots.iter_mut() {
        let want = if matches!(skin.phase, Pairing::Paired) && show_mesh {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }
}

#[allow(clippy::type_complexity)]
fn drive_bones(
    mut bones: Query<(&BoneDrive, &mut Transform)>,
    links: Query<(&Transform, &CrabEnvId), (With<CrabBodyPart>, Without<BoneDrive>)>,
    sampled: Res<CrabRenderPose>,
    repose: Res<CrabSkinRepose>,
) {
    for (drive, mut t) in bones.iter_mut() {
        if let Ok((link, env)) = links.get(drive.link) {
            let m = repose.0.get(&env.0).map_or(Mat4::IDENTITY, |r| r.matrix());
            let link = sampled.0.get(&drive.link).unwrap_or(link);
            *t = Transform::from_matrix(m * link.to_matrix() * drive.offset);
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::*;

    use crate::bot::body::{
        CrabAssets, CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, Side,
    };
    use crate::bot::headless::{headless_app, tick};
    use crate::bot::{CrabSpawns, respawn_crab};

    use super::{BoneDrive, CrabSkin, Pairing, PartId, repair_skins, reveal_skin};

    /// The rl#274 seam: a sampled render pose, when present, is what a bone follows —
    /// the raw (rapier-owned) link `Transform` drives only where no stream feeds it.
    #[test]
    fn bones_follow_the_sampled_render_pose_when_present() {
        use bevy::ecs::system::RunSystemOnce;

        use super::{CrabRenderPose, CrabSkinRepose, drive_bones};
        use crate::bot::body::CrabBodyPart;

        let mut world = World::new();
        world.init_resource::<CrabSkinRepose>();
        world.init_resource::<CrabRenderPose>();
        let link = world
            .spawn((
                CrabBodyPart,
                CrabCarapace,
                CrabEnvId(0),
                Transform::from_xyz(1.0, 2.0, 3.0),
            ))
            .id();
        let bone = world
            .spawn((
                BoneDrive {
                    link,
                    offset: Mat4::IDENTITY,
                },
                Transform::IDENTITY,
            ))
            .id();

        world
            .resource_mut::<CrabRenderPose>()
            .0
            .insert(link, Transform::from_xyz(5.0, 6.0, 7.0));
        world.run_system_once(drive_bones).expect("bare world");
        assert_eq!(
            world.entity(bone).get::<Transform>().unwrap().translation,
            Vec3::new(5.0, 6.0, 7.0),
            "with a sampled pose the bone must follow it, not the physics Transform"
        );

        world.resource_mut::<CrabRenderPose>().0.clear();
        world.run_system_once(drive_bones).expect("bare world");
        assert_eq!(
            world.entity(bone).get::<Transform>().unwrap().translation,
            Vec3::new(1.0, 2.0, 3.0),
            "no stream (standalone viewer / grace frames): the physics Transform drives"
        );
    }

    #[test]
    fn bone_names_resolve_to_expected_links() {
        use crate::bot::rig::part_for_bone;
        assert_eq!(part_for_bone("Def_shell"), Some(PartId::Carapace));
        assert_eq!(
            part_for_bone("Def_leg_01.000.L"),
            Some(PartId::Joint(CrabJointId::LegCoxa(Side::Left, 0)))
        );
        assert_eq!(
            part_for_bone("Def_leg_01.002.L"),
            Some(PartId::Joint(CrabJointId::LegBasis(Side::Left, 0)))
        );
        assert_eq!(
            part_for_bone("Def_leg_01.003.L"),
            Some(PartId::Joint(CrabJointId::LegMerus(Side::Left, 0)))
        );
    }

    #[test]
    fn skin_repairs_onto_fresh_parts_after_respawn() {
        let mut app = headless_app();
        app.add_systems(PostUpdate, repair_skins);
        tick(&mut app, 192);

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

        assert_eq!(bone_link(&mut app, shell_bone), carapace);
        assert_eq!(bone_link(&mut app, leg_bone), coxa);

        respawn_env0(&mut app);
        tick(&mut app, 1);

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
        assert!(is_live(&mut app, new_carapace) && is_live(&mut app, new_coxa));
    }

    #[test]
    fn repair_skins_drops_dead_drive_when_unresolvable() {
        let mut app = headless_app();
        app.add_systems(PostUpdate, repair_skins);
        tick(&mut app, 192);

        let carapace = find_part(&mut app, Role::Carapace);
        let (shell_bone, orphan_bone) = app
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
                let orphan = bone(&mut commands, "NotARigBone", carapace);
                (shell, orphan)
            })
            .unwrap();

        respawn_env0(&mut app);
        tick(&mut app, 1);

        let new_carapace = find_part(&mut app, Role::Carapace);
        assert_ne!(new_carapace, carapace, "respawn should make new entities");
        assert_eq!(
            bone_link(&mut app, shell_bone),
            new_carapace,
            "resolvable bone must re-home onto the fresh carapace"
        );
        assert!(
            app.world().get::<BoneDrive>(orphan_bone).is_none(),
            "unresolvable bone's dead BoneDrive must be dropped, not left dangling"
        );
    }

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

        tick(&mut app, 1);
        assert!(!visible(&app), "unpaired skin must stay hidden");

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
                    let origin = spawns.origin(0);
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
