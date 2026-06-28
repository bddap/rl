//! Scene spawn + interpolated transforms: the gray-box world, avatars, the giant crab
//! silhouette, and the per-frame pose interpolation that smooths the 30 Hz sim to the
//! render rate. Reads the sim read-only; writes only Bevy `Transform`s (and the
//! client-side [`super::input::CameraYaw`] while alive).

use super::*;
use super::driver::{GameState, LocalVehicle};
use super::input::{CameraPitch, CameraYaw};


// ---------------------------------------------------------------------------
// Entity markers
// ---------------------------------------------------------------------------

/// A rendered avatar for sim player `id`. The local player's own avatar is hidden
/// (we see from its eyes) but still spawned so status handling stays uniform.
#[derive(Component)]
pub(super) struct PlayerAvatar(PlayerId);

/// A rendered gray-box plane for the pilot with this id. The local pilot's own plane
/// is hidden (we view from its cockpit), like the local player's capsule.
#[derive(Component)]
pub(super) struct PlaneAvatar(PlayerId);

/// The giant crab placeholder.
#[derive(Component)]
pub(super) struct CrabAvatar;

/// The first-person camera, anchored to the local player each frame.
#[derive(Component)]
pub(super) struct FpCamera;

// ---------------------------------------------------------------------------
// Scene + interpolated transforms
// ---------------------------------------------------------------------------

/// Spawn the static gray-box world (ground + extraction marker + a light) and the
/// dynamic avatars (one capsule per sim player, the scaled crab). Poses are placed
/// every frame by [`apply_transforms`]; here we just create the meshes once.
pub(super) fn spawn_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: NonSend<GameState>,
    // Whether the NN crab is ACTIVE this round: when so, the placeholder crab box is spawned
    // hidden (the real rig is the crab). A real round always arms it (rl#114); the box stays
    // visible only on the headless screenshot path, which renders a plain sim with no NN stack.
    // Keyed on the active gate, NOT the bridge's presence.
    external_crab_armed: Option<Res<crate::external_crab::ExternalCrabArmed>>,
) {
    // Ground: a large gray plane at Y=0.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(400.0, 400.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.32, 0.34),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Sun-ish directional light so the gray-box reads with shape, plus a little
    // ambient so shadowed faces aren't pure black.
    commands.spawn((
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

    // Extraction point: a tall bright glowing pillar — the objective beacon. Made
    // taller than the giant crab (CRAB_SCALE players high) and thick enough to read at
    // the far end of the map, so the goal stays legible even when the towering crab is
    // between you and it. (No ground disc: a flat EXTRACT_RADIUS marker at y=0 z-fought
    // the ground plane, and the pillar alone already marks the point unmistakably.)
    let ex = state.ls.sim().extraction().pos();
    let pillar_h = PLAYER_HEIGHT * CRAB_SCALE as f32 * 1.2;
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.5, pillar_h))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.95, 0.3),
            emissive: LinearRgba::new(0.0, 2.2, 0.4, 1.0),
            ..default()
        })),
        Transform::from_translation(world(ex, pillar_h * 0.5)),
    ));

    // Player avatars: one capsule per sim player. The local player's is spawned too
    // (kept hidden in apply_transforms — we view from its eyes).
    let local = state.ls.me();
    for (id, _p) in state.ls.sim().players() {
        let is_local = id == local;
        let color = if is_local {
            Color::srgb(0.9, 0.8, 0.2)
        } else {
            Color::srgb(0.2, 0.5, 0.95)
        };
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(
                PLAYER_RADIUS,
                PLAYER_HEIGHT - 2.0 * PLAYER_RADIUS,
            ))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                ..default()
            })),
            Transform::from_translation(world(state.ls.sim().player(id).unwrap().pos(), 0.0)),
            PlayerAvatar(id),
        ));
    }

    // Pilot planes: one gray-box aircraft (fuselage + wing) per plane in the sim. The
    // root holds the pose (placed every frame by apply_transforms); the children give
    // it shape and a legible facing (+Z = nose, matching heading 0). The local pilot's
    // is spawned too but hidden in apply_transforms (cockpit view).
    let plane_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.62, 0.64, 0.67),
        perceptual_roughness: 0.7,
        ..default()
    });
    for (id, _plane) in state.ls.sim().planes() {
        let root = commands
            .spawn((
                Transform::from_translation(world3(state.ls.sim().plane(id).unwrap().pos())),
                Visibility::default(),
                PlaneAvatar(id),
            ))
            .id();
        // Fuselage: a long box down +Z (the nose direction).
        let fuselage = commands
            .spawn((
                Mesh3d(meshes.add(Cuboid::new(
                    PLANE_FUSELAGE_W,
                    PLANE_FUSELAGE_W,
                    PLANE_FUSELAGE_LEN,
                ))),
                MeshMaterial3d(plane_mat.clone()),
                Transform::default(),
            ))
            .id();
        // Wing: a wide, thin box across X, set a bit forward of center.
        let wing = commands
            .spawn((
                Mesh3d(meshes.add(Cuboid::new(
                    PLANE_WINGSPAN,
                    PLANE_FUSELAGE_W * 0.25,
                    PLANE_WING_CHORD,
                ))),
                MeshMaterial3d(plane_mat.clone()),
                Transform::from_xyz(0.0, 0.0, PLANE_FUSELAGE_LEN * 0.1),
            ))
            .id();
        commands.entity(root).add_children(&[fuselage, wing]);
    }

    // The giant crab: Sally's collider silhouette (see `spawn_crab_silhouette`), CRAB_SCALE× a
    // player. Hidden only when the armed NN rig (rl#114) has a skin model to be the visible crab;
    // with no model the rig is mesh-less, so the silhouette must stay shown or the crab vanishes.
    // Always spawned so `apply_transforms`'s crab query is satisfied. Both renders use the one
    // `crab_render_scale`, so they can't mis-size.
    let crab_hidden = external_crab_armed.is_some() && crab_world::bot::meshfit::model_path().is_some();
    let crab_root = commands
        .spawn((
            Transform::from_translation(world(state.ls.sim().crab().pos(), 0.0)),
            if crab_hidden {
                Visibility::Hidden
            } else {
                Visibility::default()
            },
            CrabAvatar,
        ))
        .id();
    // Body: Sally's ACTUAL physics colliders (carapace box + every leg/eye/claw capsule
    // from the render recipe), not a featureless placeholder box (#108). The shapes ride
    // `crab_root`, which `apply_transforms` re-poses to the sim crab every frame.
    spawn_crab_silhouette(&mut commands, &mut meshes, &mut materials, crab_root);
}

/// The giant crab's target render height: a player's height blown up by [`CRAB_SCALE`]. The
/// world (spawn distance, camera framing, the extraction pillar) is dimensioned for a crab this
/// tall, so BOTH crab renders — the integer silhouette and the armed NN rig — fit to it.
const CRAB_RENDER_HEIGHT: f32 = PLAYER_HEIGHT * CRAB_SCALE as f32;

/// The uniform scale that fits the rest-pose crab rig to [`CRAB_RENDER_HEIGHT`]: the target
/// height over the rig's natural standing height
/// ([`crab_world::bot::rig::CrabSilhouette::natural_height`]). The ONE scale source for the giant —
/// the static integer silhouette ([`spawn_crab_silhouette`]) and the armed NN rig's skin
/// ([`crab_world::bot::skin::CrabSkinRepose`]) both use it, so they render the same-sized crab by
/// construction rather than two hand-tuned factors that could drift. The
/// body's natural height is ~0.6 m (NOT a player's `PLAYER_HEIGHT`), so a bare `CRAB_SCALE`
/// multiply would render the NN crab several× too small. `None` for a degenerate recipe (zero
/// natural height) — callers fall back to the plain box.
pub(crate) fn crab_render_scale() -> Option<f32> {
    // Memoized: `render_recipe()` re-reads + re-parses the 36 MB `sally.glb` and re-fits
    // the collider cloud on every call (~1 s of work), yet the result is a property of the
    // fixed binary+asset that never changes at runtime. `publish_skin_repose` calls this
    // EACH FRAME the giant crab is armed, so without the cache that 1 s parse was the whole
    // frame budget — the GCR ~0.7-fps slideshow (rl#129). Compute once, reuse forever.
    static SCALE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| {
        let h = crab_world::bot::rig::recipe_silhouette(&crab_world::bot::body::render_recipe())
            .natural_height();
        (h > 1e-4).then(|| CRAB_RENDER_HEIGHT / h)
    })
}

/// Draw the giant crab as its REAL physics colliders — the carapace cuboid and every link
/// capsule that [`crab_world::bot::body::render_recipe`] yields, scaled to the giant height and
/// oriented claws-forward (+Z, the crab's facing) — and parent them to `crab_root`. Drawing the
/// SAME shapes the sim body uses is the point of #108: the cosmetic crab can't drift from the body
/// it depicts, and it reads as Sally instead of a box. `render_recipe` is the single
/// model-vs-fallback selector (shared with the trainer), so this never invents a second source of
/// geometry. This static silhouette is the headless-screenshot placeholder (no articulation) — the
/// real armed round shows the walking NN rig instead; the silhouette is the rest stance posed
/// rigidly, so the legs don't walk, but the shape is honest.
fn spawn_crab_silhouette(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    crab_root: Entity,
) {
    use crab_world::bot::rig::RestShape;

    let sil = crab_world::bot::rig::recipe_silhouette(&crab_world::bot::body::render_recipe());
    let shapes = || sil.shapes();

    // Orient the rig claws-forward (+Z). The recipe's forward axis isn't necessarily +Z,
    // so DERIVE it from the geometry — carapace-center → limb centroid, flattened to the
    // ground plane — rather than hard-code a constant that could drift if the asset
    // changes. Both vectors are horizontal, so `from_rotation_arc` yields a pure yaw; a
    // degenerate recipe (no limbs) leaves the rig as-is (Vec3::Z → identity).
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
        if fwd.length_squared() < 1e-6 { Vec3::Z } else { fwd },
        Vec3::Z,
    );

    // AABB in the claws-forward frame, so we can scale the rig to the giant height and
    // stand its base (min-y) on the ground.
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
    // The SAME target-height fit the armed NN rig uses (one source, no drift). A degenerate
    // recipe (zero natural height) yields `None` — keep a plain box so the crab is never invisible.
    let Some(scale) = crab_render_scale() else {
        spawn_fallback_crab_box(commands, meshes, materials, crab_root, CRAB_RENDER_HEIGHT);
        return;
    };
    // Recenter horizontally on the root and stand the base on the ground (y=0).
    let origin = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    let map = |p: Vec3| (r * p - origin) * scale;

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
                        Mesh3d(meshes.add(Capsule3d::new(radius * scale, len))),
                        MeshMaterial3d(limb_mat.clone()),
                        Transform::from_translation((a + b) * 0.5).with_rotation(rot),
                    ))
                    .id()
            }
            RestShape::Cuboid { center, half } => commands
                .spawn((
                    Mesh3d(meshes.add(Cuboid::new(
                        half.x * 2.0 * scale,
                        half.y * 2.0 * scale,
                        half.z * 2.0 * scale,
                    ))),
                    MeshMaterial3d(carapace_mat.clone()),
                    Transform::from_translation(map(center)).with_rotation(r),
                ))
                .id(),
        };
        children.push(child);
    }
    commands.entity(crab_root).add_children(&children);
}

/// The pre-#108 placeholder: a plain red box crab with a forward claw wedge. Kept ONLY
/// as the safety net for a degenerate/empty collider recipe, so a broken asset shows a
/// box rather than an invisible crab.
fn spawn_fallback_crab_box(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    crab_root: Entity,
    crab_h: f32,
) {
    let crab_w = PLAYER_RADIUS * 2.0 * CRAB_SCALE as f32;
    let body = commands
        .spawn((
            Mesh3d(meshes.add(Cuboid::new(crab_w * 1.6, crab_h * 0.5, crab_w))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.7, 0.18, 0.12),
                perceptual_roughness: 0.8,
                ..default()
            })),
            Transform::from_xyz(0.0, crab_h * 0.25, 0.0),
        ))
        .id();
    // A forward "claw" wedge at +Z (the crab's facing) so its orientation reads.
    let claw = commands
        .spawn((
            Mesh3d(meshes.add(Cuboid::new(crab_w * 0.3, crab_h * 0.25, crab_w * 0.9))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.85, 0.25, 0.15),
                ..default()
            })),
            Transform::from_xyz(0.0, crab_h * 0.3, crab_w * 1.0),
        ))
        .id();
    commands.entity(crab_root).add_children(&[body, claw]);
}

/// The three `&mut Transform` queries [`apply_transforms`] writes — player avatars,
/// the crab, the camera. Aliased (not inline) because Bevy needs the marker
/// `With`/`Without` filters to prove the three don't alias the same `Transform`, and
/// spelled inline that's the kind of type clippy's `type_complexity` flags. The
/// filters ARE the disjointness proof, so they can't be dropped — only named.
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
    &'static mut Transform,
    (With<CrabAvatar>, Without<PlayerAvatar>, Without<FpCamera>),
>;
type PlaneXf<'w, 's> = Query<
    'w,
    's,
    (
        &'static PlaneAvatar,
        &'static mut Transform,
        &'static mut Visibility,
    ),
    (
        Without<PlayerAvatar>,
        Without<CrabAvatar>,
        Without<FpCamera>,
    ),
>;
type CamXf<'w, 's> = Query<'w, 's, &'static mut Transform, With<FpCamera>>;

/// Place the FP camera and the dynamic avatars each frame, INTERPOLATED between the
/// previous tick's snapshot and the live sim by the fractional accumulator. This is
/// the smoothness layer: the sim jumps in 30 Hz steps, but every rendered frame
/// shows a pose `alpha` of the way from last tick to this one. Reads sim state
/// read-only; writes Bevy `Transform`s and (while the local player is alive) keeps the
/// client-side [`CameraYaw`] tracking the authoritative sim yaw — never the sim.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_transforms(
    state: NonSend<GameState>,
    pitch: Res<CameraPitch>,
    mut yaw: ResMut<CameraYaw>,
    vehicle: Res<LocalVehicle>,
    mut avatars: AvatarXf,
    mut crab_q: CrabXf,
    mut planes_q: PlaneXf,
    mut cam_q: CamXf,
) {
    let sim = state.ls.sim();
    let alpha = (state.accumulator / TICK_DT).clamp(0.0, 1.0) as f32;
    let local = state.ls.me();

    // Player avatars: lerp position and yaw from the previous snapshot to now.
    for (avatar, mut tf, mut vis) in avatars.iter_mut() {
        let Some(now) = sim.player(avatar.0) else {
            continue;
        };
        let prev = state.prev.players.get(&avatar.0).copied().unwrap_or(now);
        let pos = lerp_pos(prev.pos(), now.pos(), alpha);
        let yaw = lerp_yaw(prev.yaw(), now.yaw(), alpha);
        // Capsule center sits at half-height above the ground.
        *tf = Transform::from_translation(world(pos, PLAYER_HEIGHT * 0.5))
            .with_rotation(Quat::from_rotation_y(yaw));
        // Hide the local avatar (first-person), and any extracted player (gone safe).
        let hidden = avatar.0 == local || now.status() == PlayerStatus::Extracted;
        *vis = if hidden {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
        // A downed player falls onto its side so its status reads from the avatar.
        if now.status() == PlayerStatus::Downed {
            *tf = Transform::from_translation(world(pos, PLAYER_RADIUS)).with_rotation(
                Quat::from_rotation_y(yaw) * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            );
        }
    }

    // Crab: interpolate position + yaw.
    if let (Ok(mut tf), Some(crab_now), Some(crab_prev)) =
        (crab_q.single_mut(), Some(sim.crab()), state.prev.crab)
    {
        let pos = lerp_pos(crab_prev.pos(), crab_now.pos(), alpha);
        let yaw = lerp_yaw(crab_prev.yaw(), crab_now.yaw(), alpha);
        *tf =
            Transform::from_translation(world(pos, 0.0)).with_rotation(Quat::from_rotation_y(yaw));
    }

    // Planes: interpolate pose (3D position + heading + pitch) and orient the gray box
    // so +Z is the nose. Hide the local pilot's own plane (we fly from its cockpit).
    for (avatar, mut tf, mut vis) in planes_q.iter_mut() {
        let Some(now) = sim.plane(avatar.0) else {
            continue;
        };
        let prev = state.prev.planes.get(&avatar.0).copied().unwrap_or(now);
        *tf = plane_transform(prev, now, alpha);
        *vis = if avatar.0 == local {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }

    // FP camera. A PILOT flies from the cockpit: anchor the camera to the plane's
    // interpolated pose, looking along its heading+pitch and BANKED by its roll. The mouse
    // now flies the plane (pitch/roll), so there is no separate free-look while piloting —
    // the view IS the plane's attitude. An on-foot player keeps the ground eye view.
    if let Ok(mut cam) = cam_q.single_mut() {
        if let LocalVehicle::Piloting { plane, prev } = &*vehicle {
            // Single-player: fly from the CLIENT-side plane (the play layer's own body — it
            // is not in the sim, so the deterministic core stays integer-only).
            *cam = plane_cockpit_camera(*prev, *plane, alpha);
        } else if let Some(plane_now) = sim.plane(local) {
            // A SIM-side pilot (networked vehicle, rl#43): same cockpit view from sim state.
            let plane_prev = state.prev.planes.get(&local).copied().unwrap_or(plane_now);
            *cam = plane_cockpit_camera(plane_prev, plane_now, alpha);
        } else if let Some(now) = sim.player(local) {
            let prev = state.prev.players.get(&local).copied().unwrap_or(now);
            let pos = lerp_pos(prev.pos(), now.pos(), alpha);
            // Alive: aim by the AUTHORITATIVE sim yaw (so the view matches the avatar and
            // peers) and keep the free-look yaw tracking it. Downed/Extracted: the sim
            // freezes our yaw, so aim by the client-side CameraYaw instead — full
            // free-look (yaw+pitch) for a spectator, decoupled from the gated movement.
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

/// The first-person cockpit camera for a plane: eye at the interpolated 3D position,
/// looking along the interpolated heading + pitch, with the horizon BANKED by the plane's
/// roll (the cockpit up-vector tilts with the wings, so a banked turn looks like one). The
/// ONE cockpit-view formula, shared by the single-player client vehicle and the sim-side
/// networked pilot, so both fly from the identical view with no copy to drift.
fn plane_cockpit_camera(prev: Plane, now: Plane, alpha: f32) -> Transform {
    let eye = lerp_pos3(prev.pos(), now.pos(), alpha);
    let heading = lerp_yaw(prev.heading(), now.heading(), alpha);
    // Pitch/roll reuse lerp_yaw because they're turn-unit angles too; since both are bounded
    // (never wrap), the shortest-arc handling is a harmless no-op here.
    let plane_pitch = lerp_yaw(prev.pitch(), now.pitch(), alpha);
    let roll = lerp_yaw(prev.roll(), now.roll(), alpha);
    let look_dir = look_direction(heading, plane_pitch);
    // Bank the camera's up-vector by rolling Y about the look direction. Positive sim roll
    // is right-wing-down, which tilts the horizon clockwise from the pilot's seat — a
    // negative rotation about the forward look axis.
    let up = Quat::from_axis_angle(look_dir, -roll) * Vec3::Y;
    Transform::from_translation(eye).looking_at(eye + look_dir, up)
}

/// The interpolated world transform for a plane: position lerped in 3D, orientation
/// from heading (about +Y), then pitch (nose up about the local right axis), then roll
/// (bank about the nose). +Z is the nose, matching the sim's heading-0 = +Z convention and
/// the gray box's long axis. Pitch is negated (a positive sim pitch is nose-UP, but a
/// positive rotation about +X sends +Z toward −Y); roll is negated likewise so a positive
/// sim roll (right wing down) drops the +X wing.
fn plane_transform(prev: Plane, now: Plane, alpha: f32) -> Transform {
    let pos = lerp_pos3(prev.pos(), now.pos(), alpha);
    let heading = lerp_yaw(prev.heading(), now.heading(), alpha);
    let pitch = lerp_yaw(prev.pitch(), now.pitch(), alpha);
    let roll = lerp_yaw(prev.roll(), now.roll(), alpha);
    let rot = Quat::from_rotation_y(heading)
        * Quat::from_rotation_x(-pitch)
        * Quat::from_rotation_z(-roll);
    Transform::from_translation(pos).with_rotation(rot)
}

/// Linear-interpolate two sim 3D positions (to meters) by `alpha` — the [`Pos3`]
/// analogue of [`lerp_pos`], including the altitude axis.
fn lerp_pos3(a: Pos3, b: Pos3, alpha: f32) -> Vec3 {
    Vec3::new(
        meters(a.x) + (meters(b.x) - meters(a.x)) * alpha,
        meters(a.y) + (meters(b.y) - meters(a.y)) * alpha,
        meters(a.z) + (meters(b.z) - meters(a.z)) * alpha,
    )
}

/// Linear-interpolate two sim positions (in meters) by `alpha`.
pub(super) fn lerp_pos(a: Pos, b: Pos, alpha: f32) -> Pos {
    // Interpolate in fixed-point space, then `world()` converts to meters — keeps the
    // unit handling in one place. (a + (b-a)*alpha, rounded.)
    let lx = a.x as f32 + (b.x - a.x) as f32 * alpha;
    let lz = a.z as f32 + (b.z - a.z) as f32 * alpha;
    Pos {
        x: lx.round() as i64,
        z: lz.round() as i64,
    }
}

/// Interpolate two sim yaws (turn-unit integers) by `alpha`, taking the SHORTEST way
/// around the circle so a wrap from 359°→1° tweens through 0°, not backward through
/// the whole turn. Returns radians for the camera/avatar rotation.
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

/// The camera's look direction from a ground yaw and a client pitch. Compose yaw
/// (about +Y) with pitch (about the camera's local right axis, +X) and apply to the
/// base forward +Z: pitch tilts forward up/down in the YZ plane, then yaw swings it
/// horizontally. Pitch is negated because a positive rotation about +X sends +Z
/// toward −Y (down), and the control convention is positive-pitch = look UP.
pub(super) fn look_direction(yaw_radians: f32, pitch_radians: f32) -> Vec3 {
    let rot = Quat::from_rotation_y(yaw_radians) * Quat::from_rotation_x(-pitch_radians);
    (rot * Vec3::Z).normalize()
}

/// Spawn the windowed first-person camera. Its transform is overwritten every frame
/// by [`apply_transforms`]; the sky-blue clear color frames the gray-box.
pub(super) fn spawn_fp_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.5, 0.7, 0.92)),
            ..default()
        },
        Transform::default(),
        FpCamera,
    ));
}
