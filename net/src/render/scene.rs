use super::driver::{GameState, LocalVehicle, RenderClock};
use super::input::{CameraPitch, CameraYaw};
use super::pose::Pose;
use super::*;

#[derive(Component)]
pub(super) struct PlayerAvatar(PlayerId);

#[derive(Component)]
pub(super) struct CrabAvatar(pub(super) usize);

#[derive(Component)]
pub(super) struct FpCamera;

/// The extraction beacon, placed on the ground each frame by
/// [`place_extraction_pillar`] — spawn-time placement would freeze a client's pillar at
/// anchor ZERO, before the round's real anchor arrives on the articulation wire.
#[derive(Component)]
pub(super) struct ExtractionPillar;

/// The rendered ground height under a sim point: the arena surface sampled at the
/// point's arena-frame xz, lifted back through the anchor — world = arena + anchor
/// (anchor.y = 0 by construction, [`crate::external_crab::ArenaAnchor`]). Every sim
/// entity stands ON this surface; on the flat grids it is exactly the old y = 0.
fn ground_y(pos: Pos, terrain: &crab_world::terrain::Terrain, anchor: Vec3) -> f32 {
    let (x, z) = pos.to_meters();
    terrain.height(x - anchor.x, z - anchor.z) + anchor.y
}

/// Keep the drawn arena surface ([`crab_world::physics::ArenaSurface`]) at the arena
/// anchor: the mesh's vertices are arena-frame (they ARE the physics grid), so
/// translating the entity by the anchor renders the ground exactly where the physics
/// stands — the crab's articulated parts and every craft render through the same
/// anchor. Follows the resource rather than spawning at it because a joining client
/// adopts the anchor from the wire after Startup (and a round RESTART re-pins it).
pub(super) fn sync_arena_surface(
    placement: Res<crate::external_crab::ArenaAnchor>,
    mut surfaces: Query<&mut Transform, With<crab_world::physics::ArenaSurface>>,
) {
    for mut t in &mut surfaces {
        if t.translation != placement.0 {
            t.translation = placement.0;
        }
    }
}

/// Stand the extraction pillar on the ground under its sim point — per frame, like the
/// avatars, because both the anchor and (through it) the local surface height can move
/// under a fixed sim point (see [`ExtractionPillar`]).
pub(super) fn place_extraction_pillar(
    state: NonSend<GameState>,
    placement: Res<crate::external_crab::ArenaAnchor>,
    terrain: Res<crab_world::terrain::Terrain>,
    mut pillars: Query<&mut Transform, With<ExtractionPillar>>,
) {
    let ex = state.client.sim().extraction().pos();
    let pillar_h = crate::sim::CRAB_STATURE * 1.2;
    for mut t in &mut pillars {
        t.translation = world(ex, ground_y(ex, &terrain, placement.0) + pillar_h * 0.5);
    }
}

pub(super) fn spawn_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: NonSend<GameState>,
    external_crab_armed: Option<Res<crate::external_crab::ExternalCrabArmed>>,
    windows: Query<(), With<Window>>,
) {
    // Ground and lighting are the arena's own dressing (ArenaVisualsPlugin, rl#281):
    // the same grid the crab's physics runs on, terrain or flat, kept at the arena
    // anchor by [`sync_arena_surface`]. Nothing arena-shaped is spawned here.

    let ex = state.client.sim().extraction().pos();
    let pillar_h = crate::sim::CRAB_STATURE * 1.2;
    commands.spawn((
        DespawnOnExit(AppPhase::Playing),
        ExtractionPillar,
        Mesh3d(meshes.add(Cylinder::new(0.5 / 1.8 * PLAYER_HEIGHT, pillar_h))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.95, 0.3),
            emissive: LinearRgba::new(0.0, 2.2, 0.4, 1.0),
            ..default()
        })),
        Transform::from_translation(world(ex, pillar_h * 0.5)),
    ));

    commands.insert_resource(AvatarAssets {
        mesh: meshes.add(Capsule3d::new(
            PLAYER_RADIUS,
            PLAYER_HEIGHT - 2.0 * PLAYER_RADIUS,
        )),
        local: materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.8, 0.2),
            ..default()
        }),
        remote: materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.5, 0.95),
            ..default()
        }),
    });

    let armed = external_crab_armed.is_some();
    let have_model = crab_world::mesh_fallback::usable_model_path().is_some();
    let crab_hidden = armed && have_model;
    if armed && !have_model && !windows.is_empty() {
        let reason = crab_world::mesh_fallback::usable_model()
            .as_ref()
            .err()
            .map_or(crab_world::mesh_fallback::MESH_ABSENT_REASON, |s| {
                s.as_str()
            });
        let banner = crab_world::mesh_fallback::spawn_banner(&mut commands, reason);
        commands
            .entity(banner)
            .insert(DespawnOnExit(AppPhase::Playing));
    }
    for (idx, crab) in state.client.sim().crabs().iter().enumerate() {
        let crab_root = commands
            .spawn((
                DespawnOnExit(AppPhase::Playing),
                Transform::from_translation(world(crab.pos(), 0.0)),
                if crab_hidden {
                    Visibility::Hidden
                } else {
                    Visibility::default()
                },
                CrabAvatar(idx),
            ))
            .id();
        spawn_crab_silhouette(
            &mut commands,
            &mut meshes,
            &mut materials,
            crab_root,
            have_model,
        );
    }
}

#[derive(Resource)]
pub(super) struct AvatarAssets {
    mesh: Handle<Mesh>,
    local: Handle<StandardMaterial>,
    remote: Handle<StandardMaterial>,
}

pub(super) fn reconcile_avatars(
    mut commands: Commands,
    assets: Res<AvatarAssets>,
    state: NonSend<GameState>,
    avatars: Query<(Entity, &PlayerAvatar)>,
) {
    let sim = state.client.sim();
    let have: std::collections::HashSet<PlayerId> = avatars
        .iter()
        .filter_map(|(entity, avatar)| {
            if sim.player(avatar.0).is_some() {
                Some(avatar.0)
            } else {
                commands.entity(entity).despawn();
                None
            }
        })
        .collect();
    let local = state.client.me();
    for (id, p) in sim.players() {
        if have.contains(&id) {
            continue;
        }
        let material = if id == local {
            &assets.local
        } else {
            &assets.remote
        };
        commands.spawn((
            DespawnOnExit(AppPhase::Playing),
            Mesh3d(assets.mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(world(p.pos(), 0.0)),
            PlayerAvatar(id),
        ));
    }
}

fn spawn_crab_silhouette(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    crab_root: Entity,
    have_model: bool,
) {
    use crab_world::bot::rig::RestShape;

    let sil =
        crab_world::bot::rig::recipe_silhouette(&crab_world::bot::body::render_recipe(have_model));
    let shapes = || sil.shapes();

    let shape_mid = |s: &RestShape| match *s {
        RestShape::Capsule { a, b, .. } => (a + b) * 0.5,
        RestShape::Cuboid { center, .. } => center,
    };
    let fwd = if sil.limbs.is_empty() {
        Vec3::ZERO
    } else {
        let cc = shape_mid(&sil.carapace);
        let centroid = sil.limbs.iter().map(shape_mid).sum::<Vec3>() / sil.limbs.len() as f32;
        let mut d = centroid - cc;
        d.y = 0.0;
        d.normalize_or_zero()
    };
    let r = Quat::from_rotation_arc(
        if fwd.length_squared() < 1e-6 {
            Vec3::Z
        } else {
            fwd
        },
        Vec3::Z,
    );

    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    let mut grow = |p: Vec3| {
        lo = lo.min(p);
        hi = hi.max(p);
    };
    for s in shapes() {
        match *s {
            RestShape::Capsule { a, b, radius } => {
                let rad = Vec3::splat(radius);
                for p in [r * a, r * b] {
                    grow(p - rad);
                    grow(p + rad);
                }
            }
            RestShape::Cuboid { center, rot, half } => {
                for c in crab_world::bot::rig::cuboid_corners(center, rot, half) {
                    grow(r * c);
                }
            }
        }
    }
    let Some(_h) = crab_world::mesh_fallback::natural_body_height() else {
        unreachable!(
            "crab silhouette: render_recipe yielded a degenerate (zero natural-height) crab \
             — the collider recipe is broken"
        );
    };
    let origin = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    let map = |p: Vec3| r * p - origin;

    let carapace_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.18, 0.12),
        perceptual_roughness: 0.8,
        ..default()
    });
    let limb_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.28, 0.18),
        perceptual_roughness: 0.7,
        ..default()
    });

    let mut children: Vec<Entity> = Vec::with_capacity(sil.limbs.len() + 1);
    let carapace_ptr = &sil.carapace as *const RestShape;
    for s in shapes() {
        // Limb cuboids exist since rl#20 Phase 1; only the carapace slab wears the
        // carapace colour.
        let mat = if std::ptr::eq(s, carapace_ptr) {
            carapace_mat.clone()
        } else {
            limb_mat.clone()
        };
        let child = match *s {
            RestShape::Capsule { a, b, radius } => {
                let a = map(a);
                let b = map(b);
                let seg = b - a;
                let len = seg.length();
                let rot = if len > 1e-5 {
                    Quat::from_rotation_arc(Vec3::Y, seg / len)
                } else {
                    Quat::IDENTITY
                };
                commands
                    .spawn((
                        Mesh3d(meshes.add(Capsule3d::new(radius, len))),
                        MeshMaterial3d(mat),
                        Transform::from_translation((a + b) * 0.5).with_rotation(rot),
                    ))
                    .id()
            }
            RestShape::Cuboid { center, rot, half } => commands
                .spawn((
                    Mesh3d(meshes.add(Cuboid::new(half.x * 2.0, half.y * 2.0, half.z * 2.0))),
                    MeshMaterial3d(mat),
                    Transform::from_translation(map(center)).with_rotation(r * rot),
                ))
                .id(),
        };
        children.push(child);
    }
    commands.entity(crab_root).add_children(&children);
}

type AvatarXf<'w, 's> = Query<
    'w,
    's,
    (
        &'static PlayerAvatar,
        &'static mut Transform,
        &'static mut Visibility,
    ),
    Without<FpCamera>,
>;
type CrabXf<'w, 's> = Query<
    'w,
    's,
    (&'static CrabAvatar, &'static mut Transform),
    (Without<PlayerAvatar>, Without<FpCamera>),
>;
type CamXf<'w, 's> = Query<'w, 's, &'static mut Transform, With<FpCamera>>;

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_transforms(
    state: NonSend<GameState>,
    clock: Res<RenderClock>,
    pitch: Res<CameraPitch>,
    mut yaw: ResMut<CameraYaw>,
    vehicle: Res<LocalVehicle>,
    placement: Res<crate::external_crab::ArenaAnchor>,
    terrain: Res<crab_world::terrain::Terrain>,
    remote_crafts: Res<super::articulation::RemoteVehicle>,
    mut avatars: AvatarXf,
    mut crab_q: CrabXf,
    mut cam_q: CamXf,
) {
    let sim = state.client.sim();
    let alpha = clock.frac;
    let local = state.client.me();
    // Sim entities stand ON the arena surface (rl#281): every lift below is height
    // above the local ground, sampled at the entity's own interpolated spot.
    let ground = |pos: Pos| ground_y(pos, &terrain, placement.0);

    for (avatar, mut tf, mut vis) in avatars.iter_mut() {
        let Some(now) = sim.player(avatar.0) else {
            unreachable!(
                "avatar for departed player {:?} survived reconcile_avatars",
                avatar.0
            );
        };
        let prev = state.prev.players.get(&avatar.0).copied().unwrap_or(now);
        let pos = lerp_pos(prev.pos(), now.pos(), alpha);
        let yaw = lerp_yaw(prev.yaw(), now.yaw(), alpha);
        *tf = Transform::from_translation(world(pos, ground(pos) + PLAYER_HEIGHT * 0.5))
            .with_rotation(Quat::from_rotation_y(yaw));
        // A piloting player's ONE visible body is its craft (rl#258): the walker avatar
        // hides while a craft flies under that pilot's id. `RemoteVehicle` already excludes
        // the local pilot, whose avatar the first branch hides regardless.
        let in_craft = remote_crafts.contains(crab_world::vehicle::PilotId(avatar.0.0));
        let hidden = avatar.0 == local || now.status() == PlayerStatus::Extracted || in_craft;
        *vis = if hidden {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
        if now.status() == PlayerStatus::Downed {
            *tf = Transform::from_translation(world(pos, ground(pos) + PLAYER_RADIUS))
                .with_rotation(
                    Quat::from_rotation_y(yaw) * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                );
        }
    }

    for (avatar, mut tf) in crab_q.iter_mut() {
        let Some(crab_now) = sim.crabs().get(avatar.0).copied() else {
            continue;
        };
        let crab_prev = state.prev.crabs.get(avatar.0).copied().unwrap_or(crab_now);
        let pos = lerp_pos(crab_prev.pos(), crab_now.pos(), alpha);
        let yaw = lerp_yaw(crab_prev.yaw(), crab_now.yaw(), alpha);
        *tf = Transform::from_translation(world(pos, ground(pos)))
            .with_rotation(Quat::from_rotation_y(yaw));
    }

    if let Ok(mut cam) = cam_q.single_mut() {
        if let Some(pose) = vehicle.cockpit_sample(clock.tick, alpha) {
            // The STATIC arena→render frame (rl#224): anchoring on the live carapace here
            // made the whole cockpit view judder with Sally's every wiggle.
            *cam = cockpit_camera(pose, placement.0);
        } else if let Some(now) = sim.player(local) {
            let prev = state.prev.players.get(&local).copied().unwrap_or(now);
            let pos = lerp_pos(prev.pos(), now.pos(), alpha);
            let cam_yaw = if now.status() == PlayerStatus::Alive {
                let sim_yaw = lerp_yaw(prev.yaw(), now.yaw(), alpha);
                yaw.0 = sim_yaw;
                sim_yaw
            } else {
                yaw.0
            };
            let eye = world(pos, ground(pos) + EYE_HEIGHT);
            let look_dir = look_direction(cam_yaw, pitch.0);
            *cam = Transform::from_translation(eye).looking_at(eye + look_dir, Vec3::Y);
        }
    }
}

fn cockpit_camera(pose: Pose, shift: Vec3) -> Transform {
    let eye = pose.pos + shift;
    let rot = pose.orient;
    Transform::from_translation(eye).looking_at(eye + rot * Vec3::Z, rot * Vec3::Y)
}

pub(super) fn lerp_pos(a: Pos, b: Pos, alpha: f32) -> Pos {
    let lx = a.x as f32 + (b.x - a.x) as f32 * alpha;
    let lz = a.z as f32 + (b.z - a.z) as f32 * alpha;
    Pos {
        x: lx.round() as i64,
        z: lz.round() as i64,
    }
}

pub(super) fn lerp_yaw(a: i32, b: i32, alpha: f32) -> f32 {
    let ar = trig_client::turns_to_radians(a);
    let br = trig_client::turns_to_radians(b);
    let tau = std::f32::consts::TAU;
    let mut diff = br - ar;
    if diff > tau / 2.0 {
        diff -= tau;
    } else if diff < -tau / 2.0 {
        diff += tau;
    }
    ar + diff * alpha
}

pub(super) fn look_direction(yaw_radians: f32, pitch_radians: f32) -> Vec3 {
    let rot = Quat::from_rotation_y(yaw_radians) * Quat::from_rotation_x(-pitch_radians);
    (rot * Vec3::Z).normalize()
}

/// The FP cameras' perspective: Bevy's stock 0.1 m near plane assumes a 1.8 m human;
/// at the world's ~0.051 m player (rl#256) it would sit a player-height-and-a-half out
/// and clip near geometry (the looming crab's nearest legs, a cockpit), so it scales
/// with stature: the stock plane's fraction of the stock human, ≈ 2.8 mm. What actually clips in Bevy 0.18 is the oblique
/// `near_clip_plane` (a portals/mirrors feature), which DEFAULTS to the stock 0.1 m plane
/// independent of `near` — leave it stale and the view still clips at 0.1 render-m, ~2
/// eye-heights out (looking down while standing saw through the floor, rl#196) — so the two
/// move together here. The ONE perspective source for the windowed and screenshot FP
/// cameras, so their clips can't drift.
pub(super) fn fp_perspective() -> PerspectiveProjection {
    let near = 0.1 / 1.8 * PLAYER_HEIGHT;
    PerspectiveProjection {
        near,
        near_clip_plane: Vec4::new(0.0, 0.0, -1.0, -near),
        ..default()
    }
}

pub(super) fn spawn_fp_camera(mut commands: Commands) {
    commands.spawn((
        DespawnOnExit(AppPhase::Playing),
        Camera3d::default(),
        Projection::Perspective(fp_perspective()),
        Camera {
            clear_color: ClearColorConfig::Custom(crab_world::sky::NIGHT_CLEAR),
            ..default()
        },
        Transform::default(),
        FpCamera,
    ));
}
