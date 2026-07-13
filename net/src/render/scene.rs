use super::driver::{GameState, LocalVehicle, RenderClock};
use super::input::{CameraPitch, CameraYaw};
use super::pose::Pose;
use super::*;
use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy::math::Affine2;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

#[derive(Component)]
pub(super) struct PlayerAvatar(PlayerId);

#[derive(Component)]
pub(super) struct CrabAvatar(pub(super) usize);

#[derive(Component)]
pub(super) struct FpCamera;

/// Visual ground quad edge length (render m). The quad is re-centered on the camera
/// every frame ([`follow_ground`]), so all that matters is that its edge sits past the
/// cameras' 1000 render-m far plane in every direction — 4000 gives 2× margin. Together
/// the two make the visual ground unbounded by construction, matching the unbounded
/// physics ground: there is no reachable edge.
const GROUND_SIZE: f32 = 4000.0;

/// Checker repeat period (render m): one texture repeat is 2×2 coarse cells, each a
/// 16×16 sub-checker. Sized for the PILOT, who the cue serves (rl#197): the 2 m coarse
/// cells read from cruise altitude, the 0.125 m fine sub-cells (~2.5 player heights)
/// give optic flow at landing height and on foot.
const CHECKER_PERIOD: f32 = 4.0;

/// The ground quad, re-centered on the camera each frame by [`follow_ground`].
#[derive(Component)]
pub(super) struct GroundPlane;

/// Re-center the ground quad on the camera, snapped to the checker repeat period so the
/// pattern stays world-stationary (a non-multiple shift would make it swim underfoot).
/// With the quad's edge past the far plane ([`GROUND_SIZE`]), flying off the ground is
/// impossible, not merely far away.
pub(super) fn follow_ground(
    cams: Query<&Transform, (With<FpCamera>, Without<GroundPlane>)>,
    mut grounds: Query<&mut Transform, (With<GroundPlane>, Without<FpCamera>)>,
) {
    let Ok(cam) = cams.single() else { return };
    for mut ground in &mut grounds {
        ground.translation.x = (cam.translation.x / CHECKER_PERIOD).round() * CHECKER_PERIOD;
        ground.translation.z = (cam.translation.z / CHECKER_PERIOD).round() * CHECKER_PERIOD;
    }
}

/// The ground's repeat-tiled checker texture: a coarse 2×2 checker with a fainter
/// sub-checker on top (two spatial frequencies — see [`CHECKER_PERIOD`]), tinting the
/// material's `base_color` so the ground keeps its gray-box tone. Built with a full mip
/// chain — box-filtering per level — so the pattern fades to flat gray in the distance
/// instead of shimmering (a generated `Image` gets no auto-mips).
fn ground_checker(images: &mut Assets<Image>) -> Handle<Image> {
    const SIZE: usize = 256; // level-0 px; power of two so mip dims match wgpu's size>>level
    let mut levels: Vec<Vec<u8>> = vec![
        (0..SIZE * SIZE)
            .flat_map(|i| {
                let (x, y) = (i % SIZE, i / SIZE);
                let coarse = (x / (SIZE / 2) + y / (SIZE / 2)).is_multiple_of(2);
                let fine = (x / (SIZE / 32) + y / (SIZE / 32)).is_multiple_of(2);
                let v = 225 + if coarse { 16u8 } else { 0 } + if fine { 14 } else { 0 };
                [v, v, v, 255]
            })
            .collect(),
    ];
    let mut w = SIZE;
    while w > 1 {
        let prev = levels.last().unwrap();
        let half = w / 2;
        levels.push(
            (0..half * half)
                .flat_map(|i| {
                    let (x, y) = ((i % half) * 2, (i / half) * 2);
                    let at = |x: usize, y: usize| prev[(y * w + x) * 4] as u16;
                    let v = ((at(x, y) + at(x + 1, y) + at(x, y + 1) + at(x + 1, y + 1)) / 4) as u8;
                    [v, v, v, 255]
                })
                .collect(),
        );
        w = half;
    }
    let mip_count = levels.len() as u32;
    // `Image::new` asserts data == level-0 size, so build uninit and attach the full
    // mip chain (levels concatenated largest-first, bevy's expected layout) by hand.
    let mut image = Image::new_uninit(
        Extent3d {
            width: SIZE as u32,
            height: SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(levels.concat());
    image.texture_descriptor.mip_level_count = mip_count;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        // Most of a cockpit view meets the ground at grazing angles, where isotropic
        // trilinear blurs the checker to flat gray — right where the optic-flow cue
        // matters most.
        anisotropy_clamp: 16,
        ..ImageSamplerDescriptor::linear()
    });
    images.add(image)
}

pub(super) fn spawn_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    state: NonSend<GameState>,
    external_crab_armed: Option<Res<crate::external_crab::ExternalCrabArmed>>,
    windows: Query<(), With<Window>>,
) {
    commands.spawn((
        DespawnOnExit(AppPhase::Playing),
        GroundPlane,
        Mesh3d(meshes.add(Plane3d::default().mesh().size(GROUND_SIZE, GROUND_SIZE))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.32, 0.34),
            base_color_texture: Some(ground_checker(&mut images)),
            // Plane3d UVs span 0..1 across the whole quad; scale them so one texture
            // repeat covers CHECKER_PERIOD meters.
            uv_transform: Affine2::from_scale(Vec2::splat(GROUND_SIZE / CHECKER_PERIOD)),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    commands.spawn((
        DespawnOnExit(AppPhase::Playing),
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(20.0, 40.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(bevy::light::GlobalAmbientLight {
        brightness: 220.0,
        ..default()
    });

    let ex = state.client.sim().extraction().pos();
    let pillar_h = PLAYER_HEIGHT * CRAB_SCALE as f32 * 1.2;
    commands.spawn((
        DespawnOnExit(AppPhase::Playing),
        Mesh3d(meshes.add(Cylinder::new(0.014, pillar_h))),
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
            RestShape::Cuboid { center, half } => {
                let cen = r * center;
                for sx in [-1.0_f32, 1.0] {
                    for sy in [-1.0_f32, 1.0] {
                        for sz in [-1.0_f32, 1.0] {
                            grow(cen + r * Vec3::new(sx * half.x, sy * half.y, sz * half.z));
                        }
                    }
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
    for s in shapes() {
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
                        MeshMaterial3d(limb_mat.clone()),
                        Transform::from_translation((a + b) * 0.5).with_rotation(rot),
                    ))
                    .id()
            }
            RestShape::Cuboid { center, half } => commands
                .spawn((
                    Mesh3d(meshes.add(Cuboid::new(half.x * 2.0, half.y * 2.0, half.z * 2.0))),
                    MeshMaterial3d(carapace_mat.clone()),
                    Transform::from_translation(map(center)).with_rotation(r),
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
    remote_crafts: Res<super::articulation::RemoteVehicle>,
    mut avatars: AvatarXf,
    mut crab_q: CrabXf,
    mut cam_q: CamXf,
) {
    let sim = state.client.sim();
    let alpha = clock.frac;
    let local = state.client.me();

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
        *tf = Transform::from_translation(world(pos, PLAYER_HEIGHT * 0.5))
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
            *tf = Transform::from_translation(world(pos, PLAYER_RADIUS)).with_rotation(
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
        *tf =
            Transform::from_translation(world(pos, 0.0)).with_rotation(Quat::from_rotation_y(yaw));
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
            let eye = world(pos, EYE_HEIGHT);
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
/// with stature: 0.056 × [`PLAYER_HEIGHT`] ≈ 2.9 mm. What actually clips in Bevy 0.18 is the oblique
/// `near_clip_plane` (a portals/mirrors feature), which DEFAULTS to the stock 0.1 m plane
/// independent of `near` — leave it stale and the view still clips at 0.1 render-m, ~2
/// eye-heights out (looking down while standing saw through the floor, rl#196) — so the two
/// move together here. The ONE perspective source for the windowed and screenshot FP
/// cameras, so their clips can't drift.
pub(super) fn fp_perspective() -> PerspectiveProjection {
    let near = 0.056 * PLAYER_HEIGHT;
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
