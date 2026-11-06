use super::driver::{GameState, LocalVehicle, RenderClock};
use super::input::{CameraPitch, CameraYaw};
use super::pose::Pose;
use super::*;

#[derive(Component)]
pub(super) struct PlayerAvatar(pub(super) PlayerId);

#[derive(Component)]
pub(super) struct FpCamera;

/// The extraction beacon, placed on the ground each frame by
/// [`place_extraction_pillar`].
#[derive(Component)]
pub(super) struct ExtractionPillar;

/// Anchor the render frame on the round's locale (rl#334/rl#354): world-frame f32
/// coordinates quantize at ~0.5–2 mm out at the rl#305 corners — boiling speckle
/// in the detail octaves, and per-frame judder of everything near a walking eye —
/// so every rendered Transform is world − this anchor ([`super::RenderOrigin`];
/// [`crab_world::ground::GroundAnchor`] is its f32 shadow, consumed by the
/// terrain/scatter side). The extraction point is the anchor rather than the
/// spawn-frame origin because it is the one locale mark every peer holds
/// host-authoritatively (it ships in the snapshot; `spawn_frame` is each sim's
/// private draw, so a wire-joiner's disagrees with the host's by up to the whole
/// tile) — and it sits within meters of the spawn line. Constant per round, so
/// nothing pops mid-round; a wire-joiner may render a frame or two on its
/// placeholder extraction before the first snapshot lands — one re-anchor,
/// self-correcting at snapshot cadence. The `!=` guards are change-tick hygiene.
pub(super) fn sync_ground_anchor(
    state: NonSend<GameState>,
    mut origin: ResMut<super::RenderOrigin>,
    mut anchor: ResMut<crab_world::ground::GroundAnchor>,
) {
    let extraction = state.client.sim().extraction().pos();
    if origin.0 != extraction {
        origin.0 = extraction;
    }
    let (x, z) = extraction.to_meters();
    let locale = Vec2::new(x, z);
    if anchor.0 != locale {
        anchor.0 = locale;
    }
}

/// The rendered spot `above` meters over the ground under a sim point (one frame since
/// rl#298 stage 5: the world IS the sim in meters, rebased into the render frame,
/// rl#354). Every sim entity stands ON this surface — xz subtracts the origin on the
/// i64 grid (full fixed-point resolution), y samples the terrain at f64 so the eye
/// height cannot stair-step with f32's ~1 mm far-locale quantization; on the flat
/// grids the ground is exactly the old y = 0.
fn place_over(
    pos: Pos,
    above: f32,
    terrain: &crab_world::terrain::Terrain,
    origin: super::RenderOrigin,
) -> Vec3 {
    let (x, z) = pos.rel_meters(origin.0);
    let (ax, az) = pos.to_meters_f64();
    Vec3::new(x, terrain.height_f64(ax, az) as f32 + above, z)
}

/// Extraction beacon height, meters.
const PILLAR_HEIGHT: f32 = 0.73;

/// Stand the extraction pillar on the ground under its sim point — per frame, like the
/// avatars, in case a future round ever moves the extraction point.
pub(super) fn place_extraction_pillar(
    state: NonSend<GameState>,
    terrain: Res<crab_world::terrain::Terrain>,
    origin: Res<super::RenderOrigin>,
    mut pillars: Query<&mut Transform, With<ExtractionPillar>>,
) {
    let ex = state.client.sim().extraction().pos();
    for mut t in &mut pillars {
        t.translation = place_over(ex, PILLAR_HEIGHT * 0.5, &terrain, *origin);
    }
}

pub(super) fn spawn_world(
    mut commands: Commands,
    origin: Res<super::RenderOrigin>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: NonSend<GameState>,
) {
    // Ground and lighting are the world's own dressing (ArenaWorldPlugin, rl#281):
    // the same grid the physics runs on, rendered where it stands (one
    // frame). Nothing world-shaped is spawned here.

    let ex = state.client.sim().extraction().pos();
    commands.spawn((
        DespawnOnExit(AppPhase::Playing),
        ExtractionPillar,
        Mesh3d(meshes.add(Cylinder::new(0.5 / 1.8 * PLAYER_HEIGHT, PILLAR_HEIGHT))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.95, 0.3),
            emissive: LinearRgba::new(0.0, 2.2, 0.4, 1.0),
            ..default()
        })),
        Transform::from_translation(world(ex, PILLAR_HEIGHT * 0.5, *origin)),
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
}

#[derive(Resource)]
pub(super) struct AvatarAssets {
    mesh: Handle<Mesh>,
    local: Handle<StandardMaterial>,
    remote: Handle<StandardMaterial>,
}

pub(super) fn reconcile_avatars(
    mut commands: Commands,
    origin: Res<super::RenderOrigin>,
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
            Transform::from_translation(world(p.pos(), 0.0, *origin)),
            PlayerAvatar(id),
        ));
    }
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
type CamXf<'w, 's> = Query<'w, 's, &'static mut Transform, With<FpCamera>>;

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_transforms(
    state: NonSend<GameState>,
    clock: Res<RenderClock>,
    pitch: Res<CameraPitch>,
    mut yaw: ResMut<CameraYaw>,
    vehicle: Res<LocalVehicle>,
    terrain: Res<crab_world::terrain::Terrain>,
    origin: Res<super::RenderOrigin>,
    remote_crafts: Res<super::remote_vehicles::RemoteVehicle>,
    mut avatars: AvatarXf,
    mut cam_q: CamXf,
    time: Res<Time>,
    // Smoothed eye height (0 = uninitialized): the stand↔slide crouch eases at
    // [`SLIDE_EYE_RATE`] instead of cutting, so the dip reads as a body motion.
    mut eye_h: Local<f32>,
    // `Option`: the hand-built test worlds run this system without a round install.
    trace: Option<ResMut<super::pos_trace::PosTrace>>,
) {
    let sim = state.client.sim();
    let alpha = clock.frac;
    let local = state.client.me();
    // Sim entities stand ON the world surface (rl#281): every lift below is height
    // above the local ground, sampled at the entity's own interpolated spot.
    let place = |pos: Pos, above: f32| place_over(pos, above, &terrain, *origin);

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
        let alt = lerp_alt(prev.alt(), now.alt(), alpha);
        *tf = Transform::from_translation(place(pos, PLAYER_HEIGHT * 0.5 + alt))
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
        if now.sliding() {
            // A skidding walker leans back feet-first (rl#368) — the third-person
            // read of the slide, matching the local player's eye dip below.
            *tf = Transform::from_translation(place(pos, PLAYER_HEIGHT * 0.35 + alt))
                .with_rotation(
                    Quat::from_rotation_y(yaw) * Quat::from_rotation_x(std::f32::consts::FRAC_PI_3),
                );
        }
    }

    if let Ok(mut cam) = cam_q.single_mut() {
        if let Some(pose) = vehicle.cockpit_sample(clock.tick, alpha) {
            *cam = cockpit_camera(pose, *origin);
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
            // The airborne walker's eye rides its altitude (rl#355) — the plane-exit
            // handoff sails, so the camera sails with it and settles on touchdown.
            let alt = lerp_alt(prev.alt(), now.alt(), alpha);
            // Mid-slide the eye crouches to [`SLIDE_EYE_HEIGHT`] (rl#368), eased so
            // the drop reads as a body motion, not a camera cut.
            let target = if now.sliding() {
                SLIDE_EYE_HEIGHT
            } else {
                EYE_HEIGHT
            };
            if *eye_h == 0.0 {
                *eye_h = target;
            }
            *eye_h += (target - *eye_h) * (1.0 - (-time.delta_secs() * SLIDE_EYE_RATE).exp());
            let eye = place(pos, *eye_h + alt);
            let look_dir = look_direction(cam_yaw, pitch.0);
            *cam = Transform::from_translation(eye).looking_at(eye + look_dir, Vec3::Y);
        }
        clamp_camera_above_terrain(&mut cam, &terrain, *origin);
        if let Some(mut trace) = trace {
            trace.frame(clock.tick, alpha, time.delta(), &cam, origin.0);
        }
    }
}

/// The eye may never come closer to the surface — sampled at its own xz, exact on
/// both the mesh's and the collider's triangles — than this. Sized from
/// [`fp_perspective`]'s near plane: the near frustum's corners reach ~1.3×near from
/// the eye, so 4×near keeps them out of the sheet even nose-down on a steep slope.
/// Well under the on-foot [`EYE_HEIGHT`], so walking never feels it.
pub(super) const CAM_TERRAIN_CLEARANCE: f32 = 4.0 * FP_NEAR;

/// Floor the camera above the terrain sheet (rl#306). The on-foot eye already rides
/// the surface, but the cockpit flies the craft's raw physics pose: a landed or
/// ground-skimming craft's body center sits within soft-contact penetration of the
/// sheet — cm-scale, comparable to the whole craft at this world's stature — putting
/// the eye and its near frustum under the ground. Translation only; the pilot still
/// looks where the craft points.
pub(super) fn clamp_camera_above_terrain(
    cam: &mut Transform,
    terrain: &crab_world::terrain::TerrainGrid,
    origin: super::RenderOrigin,
) {
    // The camera lives in the render frame; the terrain sampler speaks absolute
    // world meters, so add the origin back — at f64, or the sum re-quantizes at
    // the very ~1 mm the frame exists to avoid (rl#354).
    let (ox, oz) = origin.0.to_meters_f64();
    let floor = terrain.height_f64(cam.translation.x as f64 + ox, cam.translation.z as f64 + oz)
        as f32
        + CAM_TERRAIN_CLEARANCE;
    cam.translation.y = cam.translation.y.max(floor);
}

/// The cockpit eye: the craft's physics pose carried into the render frame. The
/// pose only exists in absolute f32 world meters (rapier), so this subtraction is
/// as precise as physics itself — exact near the origin (Sterbenz).
fn cockpit_camera(pose: Pose, origin: super::RenderOrigin) -> Transform {
    let eye = pose.pos - origin.offset_m();
    let rot = pose.orient;
    Transform::from_translation(eye).looking_at(eye + rot * Vec3::Z, rot * Vec3::Y)
}

/// Interpolated feet-over-surface height in meters (rl#355) — the walker's sim
/// altitude, ready to add to a `place` lift.
pub(super) fn lerp_alt(a: i64, b: i64, alpha: f32) -> f32 {
    (a as f32 + (b - a) as f32 * alpha) / crate::sim::UNIT as f32
}

/// Lerp the per-tick DELTA in f32 and add it back on the i64 grid. The absolute
/// coordinate must never pass through f32: at the far locales it is ~1e9 grid
/// units, where f32 quantizes at 64 units (0.64 mm) — a third of a 60 Hz walking
/// step, seen as alternating long/short frame steps (the rl#371 residual jitter).
/// The delta is a few hundred units, exact in f32.
pub(super) fn lerp_pos(a: Pos, b: Pos, alpha: f32) -> Pos {
    Pos {
        x: a.x + ((b.x - a.x) as f32 * alpha).round() as i64,
        z: a.z + ((b.z - a.z) as f32 * alpha).round() as i64,
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

/// The yaw [`look_direction`] takes, recovered from a direction.
pub(super) fn heading(dir: Vec3) -> f32 {
    dir.x.atan2(dir.z)
}

/// The FP cameras' perspective: Bevy's stock 0.1 m near plane assumes a 1.8 m human;
/// at the world's ~0.051 m player (rl#256) it would sit a player-height-and-a-half out
/// and clip near geometry (a cockpit), so it scales
/// with stature: the stock plane's fraction of the stock human, ≈ 2.8 mm. The ONE
/// perspective source for the windowed and screenshot FP cameras, so their clips
/// can't drift.
pub(super) fn fp_perspective() -> PerspectiveProjection {
    PerspectiveProjection {
        near: FP_NEAR,
        ..default()
    }
}

/// The FP cameras' near-plane distance — the stock 0.1 m plane's fraction of the
/// stock 1.8 m human, scaled to this world's stature (see [`fp_perspective`]).
const FP_NEAR: f32 = 0.1 / 1.8 * PLAYER_HEIGHT;

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
