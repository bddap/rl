//! Scene spawn + interpolated transforms: the gray-box world, avatars, the giant crab
//! silhouette, and the per-frame pose interpolation that smooths the 30 Hz sim to the
//! render rate. Reads the sim read-only; writes only Bevy `Transform`s (and the
//! client-side [`super::input::CameraYaw`] while alive).

use super::driver::{CockpitPose, GameState, LocalVehicle};
use super::input::{CameraPitch, CameraYaw};
use super::*;

// ---------------------------------------------------------------------------
// Entity markers
// ---------------------------------------------------------------------------

/// A rendered avatar for sim player `id`. The local player's own avatar is hidden
/// (we see from its eyes) but still spawned so status handling stays uniform.
#[derive(Component)]
pub(super) struct PlayerAvatar(PlayerId);

/// Marks the giant crab's render root — the entity [`apply_transforms`] re-poses to the sim
/// crab each frame.
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
    // Whether the NN crab is ACTIVE this round: when so AND a skin model resolves, the
    // physics-bones silhouette is spawned hidden (the skinned NN rig is the visible crab). A
    // real round always arms it (rl#114); the silhouette stays visible on the headless
    // screenshot path (a plain sim with no NN stack) and whenever no model resolves (the
    // silhouette is the allowed physics-bones view). Keyed on the active gate, NOT the
    // bridge's presence.
    external_crab_armed: Option<Res<crate::external_crab::ExternalCrabArmed>>,
    // A primary window exists on the WINDOWED play/solo surface (one `BorderlessFullscreen`
    // window) but NOT on the windowless fp-screenshot path (`primary_window: None`). The
    // missing-mesh banner is spawned only when a window exists, so it never pollutes a captured
    // screenshot frame.
    windows: Query<(), With<Window>>,
) {
    // The render-frame shrink: the human world (ground, players, the pillar, the camera)
    // renders this much smaller so the true-physics-size crab towers over it (render==physics; the
    // crab is NOT inflated — see [`world_render_scale`]). `world` already applies it to ground
    // POSITIONS; here it sizes the human-world MESHES. The crab silhouette is the lone exception —
    // it renders at native physics size.
    let rs = world_render_scale();

    // Ground: a large gray plane at Y=0.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(400.0 * rs, 400.0 * rs))),
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
        Mesh3d(meshes.add(Cylinder::new(0.5 * rs, pillar_h * rs))),
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
                PLAYER_RADIUS * rs,
                (PLAYER_HEIGHT - 2.0 * PLAYER_RADIUS) * rs,
            ))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                ..default()
            })),
            Transform::from_translation(world(state.ls.sim().player(id).unwrap().pos(), 0.0)),
            PlayerAvatar(id),
        ));
    }

    // The giant crab: Sally's collider silhouette (see `spawn_crab_silhouette`), at TRUE physics
    // size — it towers because the human world renders R× smaller around it ([`world_render_scale`]),
    // not because the crab is inflated. Hidden only when the armed NN rig (rl#114) has a skin model
    // to be the visible crab; with no model the rig is mesh-less, so the silhouette must stay shown
    // or the crab vanishes. Always spawned so `apply_transforms`'s crab query is satisfied. The
    // silhouette and the skin both render at native size, so they can't mis-size relative to each
    // other or to the colliders.
    let armed = external_crab_armed.is_some();
    // net's single asset source is the global preflight verdict: `CrabModelPath` is a `BotPlugin`
    // resource and the silhouette path runs WITHOUT the bot stack (the unarmed screenshot), so it
    // can't read that resource. net never overrides the resolver anyway, so `usable_model()` here and
    // the body's `CrabModelPath` (which defaults to the same verdict) agree — see
    // `crab_world::bot::body::CrabModelPath`. Preflighted (not existence-only `model_path()`), so a
    // present-but-broken glb resolves to `None` → the silhouette below draws the fallback recipe
    // instead of the real one (bddap/rl#154).
    let have_model = crab_world::mesh_fallback::usable_model_path().is_some();
    let crab_hidden = armed && have_model;
    // With the crab armed but NO model, the silhouette (the real colliders, NOT the real Sally
    // rig) stays shown AS the crab. Shipping that with no signal is the silent-fallback bug the
    // owner most hates (rl#706), so on the WINDOWED surface name it on screen with the banner
    // shared with rl-demo. The OTEL companion already fires in `game::resolve_render_mode`; only
    // the live window adds the banner here. The screenshot path has no window (`primary_window:
    // None`) so a UI band can't pollute the capture — there the OTEL error + visible silhouette
    // are the signal. NB the gates differ on purpose: the banner shows whenever the mesh is
    // absent (it answers "what am I looking at?"), while that OTEL error suppresses under an
    // explicit RL_RENDER_MODE override (it means "the missing mesh FORCED the fallback").
    if armed && !have_model && !windows.is_empty() {
        // `!have_model` ⇒ the verdict is `Err`, so name the ACTUAL cause (absent vs broken) on
        // screen (bddap/rl#154); `MESH_ABSENT_REASON` is only the fallback if the verdict somehow
        // lacks a reason.
        let reason = crab_world::mesh_fallback::usable_model()
            .as_ref()
            .err()
            .map_or(crab_world::mesh_fallback::MESH_ABSENT_REASON, |s| {
                s.as_str()
            });
        crab_world::mesh_fallback::spawn_banner(&mut commands, reason);
    }
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
    spawn_crab_silhouette(
        &mut commands,
        &mut meshes,
        &mut materials,
        crab_root,
        have_model,
    );
}

/// The giant crab's apparent-height target: a player's height times [`CRAB_SCALE`] — the crab
/// reads CRAB_SCALE players tall. We hit this RATIO by rendering the rest of the world that much
/// SMALLER (see [`world_render_scale`]), NOT by inflating the crab: the crab renders at its TRUE
/// physics size so a collider wireframe overlays it (render==physics). Kept only to derive the
/// world scale.
const CRAB_RENDER_HEIGHT: f32 = PLAYER_HEIGHT * CRAB_SCALE as f32;

/// The crab rig's natural standing height (m) — the span of its REAL physics colliders. Memoized:
/// the recipe + silhouette geometry it derives are a property of the fixed binary+asset that never
/// changes at runtime, yet this is read to set the world render scale on a per-frame path, so the
/// derivation is cached rather than re-run each read (rl#129). The underlying 36 MB glb parse + fit is
/// itself memoized once in [`crab_world::mesh_fallback::usable_model`] (rl#153), so
/// [`crab_world::bot::body::render_recipe`] here only clones that recipe. `None` for a degenerate
/// (zero-height) recipe — a broken collider asset with no honest geometry, which callers fail LOUD on
/// rather than drawing a stand-in.
fn natural_crab_height() -> Option<f32> {
    static H: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *H.get_or_init(|| {
        // Flips off the preflight verdict (`mesh_fallback::usable_model_path`), NOT the per-app
        // `CrabModelPath` resource: this answers "how big is the crab this run draws?" to set the
        // world render scale, a fixed binary+asset constant. Preflighted so a present-but-broken glb
        // draws the fallback recipe's height (what's actually rendered) instead of the real one
        // (bddap/rl#154). net has one asset source, so this and the silhouette below flip together.
        let h = crab_world::bot::rig::recipe_silhouette(&crab_world::bot::body::render_recipe(
            crab_world::mesh_fallback::usable_model_path().is_some(),
        ))
        .natural_height();
        (h > 1e-4).then_some(h)
    })
}

/// Display-only render-frame scale: rendered metres per sim metre for the HUMAN world — players,
/// planes, the arena, the camera. The giant crab renders at its TRUE physics size (1×, so a
/// collider wireframe overlays it natively — render==physics); the giant FEEL comes from rendering
/// everything ELSE this much smaller, so the crab still reads [`CRAB_SCALE`]× a player WITHOUT
/// inflating it (which would desync the wireframe from the colliders and force retraining Sally at
/// a bigger collider scale). The reciprocal of the old crab blow-up: the crab's natural height over
/// the giant target height. The ONE scale source — every sim→render ground position ([`world`])
/// and the human-world mesh sizes multiply by it, the crab does not (and the piloted vehicle, which
/// lives at true arena scale with the crab, doesn't either — its cockpit camera is shifted, not
/// shrunk).
/// Physics/training are untouched: this multiplies only Bevy `Transform`s. `1.0` on a degenerate
/// recipe (the silhouette path fails loud there anyway).
pub(crate) fn world_render_scale() -> f32 {
    natural_crab_height()
        .map(|h| h / CRAB_RENDER_HEIGHT)
        .unwrap_or(1.0)
}

/// Draw the giant crab as its REAL physics colliders — the carapace cuboid and every link
/// capsule that [`crab_world::bot::body::render_recipe`] yields, scaled to the giant height and
/// oriented claws-forward (+Z, the crab's facing) — and parent them to `crab_root`. Drawing the
/// SAME shapes the sim body uses is the point of #108: the cosmetic crab can't drift from the body
/// it depicts, and it reads as Sally instead of a box. `render_recipe` is the single
/// model-vs-fallback selector (shared with the trainer), so this never invents a second source of
/// geometry. This static silhouette is the honest physics-bones view (no articulation): the
/// headless-screenshot render, and the in-game view whenever no skin model resolves — the rest
/// stance posed rigidly, so the legs don't walk, but it is the REAL colliders, never a stand-in
/// box. The real armed round with a model shows the walking skinned NN rig instead. `have_model` is
/// the preflight verdict's flip (`mesh_fallback::usable_model_path().is_some()`) resolved ONCE in
/// `spawn_world` — net's single asset source, the SAME verdict the body's `CrabModelPath` defaults to,
/// so they agree — and `false` (absent or broken glb) draws the fallback recipe; `true` draws the
/// memoized real recipe the verdict already built (bddap/rl#153), never a re-parse or an `.expect()`.
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
        if fwd.length_squared() < 1e-6 {
            Vec3::Z
        } else {
            fwd
        },
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
    // The crab draws at its TRUE physics size (render==physics): no blow-up, so a collider
    // wireframe overlays it natively and Sally never has to be retrained at a bigger collider
    // scale. The giant FEEL instead comes from the R-shrunk human world ([`world_render_scale`]).
    // A degenerate recipe (zero natural height) is the ONE refusal: it can only mean the collider
    // recipe is broken (a real OR absent model both yield a positive-height procedural recipe), so
    // there is no honest crab geometry to draw. We refuse to paper over that with a placeholder box
    // (the silent-fallback bug — the only crabs allowed are the Sally mesh and this physics-bones
    // silhouette); fail LOUD instead.
    let Some(_h) = natural_crab_height() else {
        unreachable!(
            "crab silhouette: render_recipe yielded a degenerate (zero natural-height) crab \
             — the collider recipe is broken"
        );
    };
    // Recenter horizontally on the root and stand the base on the ground (y=0). No scale: native
    // physics size.
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
type CamXf<'w, 's> = Query<'w, 's, &'static mut Transform, With<FpCamera>>;
/// Read-only access to the crab carapace's ARENA-frame Transform — the anchor the cockpit camera
/// shifts the vehicle's arena pose against. Disjoint from the mutable Transform queries above by the
/// `Without` filters (a carapace is none of those entities), so Bevy lets them coexist.
type CarapaceXf<'w, 's> = Query<
    'w,
    's,
    &'static Transform,
    (
        With<crab_world::bot::body::CrabCarapace>,
        Without<CrabAvatar>,
        Without<PlayerAvatar>,
        Without<FpCamera>,
    ),
>;

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
    mut cam_q: CamXf,
    carapace_q: CarapaceXf,
) {
    let sim = state.ls.sim();
    let alpha = (state.accumulator / TICK_DT).clamp(0.0, 1.0) as f32;
    let local = state.ls.me();

    // Player avatars: lerp position and yaw from the previous snapshot to now.
    for (avatar, mut tf, mut vis) in avatars.iter_mut() {
        let Some(now) = sim.player(avatar.0) else {
            // No sim player behind this avatar: the player DEPARTED mid-match (rl#198 — the
            // adopted snapshot no longer carries them). Hide the capsule rather than leave it
            // frozen at their last pose; a rejoiner reusing the freed PlayerId un-hides it.
            *vis = Visibility::Hidden;
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

    // FP camera. A PILOT flies from the cockpit: anchor the camera to the vehicle's
    // interpolated pose, looking along its heading+pitch and BANKED by its roll. The mouse
    // now flies the craft (pitch/roll), so there is no separate free-look while piloting —
    // the view IS the craft's attitude. An on-foot player keeps the ground eye view.
    if let Ok(mut cam) = cam_q.single_mut() {
        if let Some((prev, now)) = vehicle.cockpit_poses() {
            // Single-player: fly from the rapier vehicle body (host-authoritative crab-world state,
            // not in the integer sim). Its pose is in the ARENA frame, so map it to render space
            // with the same shift the crab body uses — `world(crab) − arena_carapace` — so vehicle
            // and crab share one render frame and collide where they're drawn.
            let crab_now = sim.crab();
            let crab_prev = state.prev.crab.unwrap_or(crab_now);
            let crab_anchor = world(lerp_pos(crab_prev.pos(), crab_now.pos(), alpha), 0.0);
            let arena_carapace = carapace_q
                .iter()
                .next()
                .map(|t| t.translation)
                .unwrap_or(Vec3::ZERO);
            let shift = Vec3::new(
                crab_anchor.x - arena_carapace.x,
                0.0,
                crab_anchor.z - arena_carapace.z,
            );
            *cam = cockpit_camera(prev, now, alpha, shift);
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

/// The first-person cockpit camera for a flyer: eye at the interpolated 3D position, looking along
/// the craft's nose with the horizon banked/pitched/rolled by its full attitude — so a banked turn,
/// a loop, or inverted flight all look right. The ONE cockpit-view formula, flying EVERY craft (the
/// plane and the ship) from the shared [`CockpitPose`] with no copy to drift.
///
/// The pose is in the crab's ARENA frame (the rapier vehicle body lives in the ±10 m box with
/// Sally). `shift` is the pure XZ translate that carries an arena point to its render spot anchored
/// at the giant crab — the same `world(crab) − arena_carapace` the crab body's render uses — so the
/// vehicle and the crab share one render frame and collide where they're drawn. Altitude (Y) is kept
/// at TRUE arena scale (no shrink), like the crab's own bones, so a small craft reads correctly
/// against the giant Sally. The two ticks' attitudes are SLERPed (shortest-arc) for a smooth tween.
fn cockpit_camera(prev: CockpitPose, now: CockpitPose, alpha: f32, shift: Vec3) -> Transform {
    let eye = prev.pos.lerp(now.pos, alpha) + shift;
    let rot = prev.orient.slerp(now.orient, alpha);
    // Look along the craft's nose (+Z), with up its own up-vector (+Y) — so the pilot looks where
    // the craft is pointed, banked/pitched/rolled by its full attitude.
    Transform::from_translation(eye).looking_at(eye + rot * Vec3::Z, rot * Vec3::Y)
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
/// by [`apply_transforms`]; the night-sky skybox ([`crab_world::sky`]) paints the
/// background, with this dark clear color as the pre-upload fallback. The near plane is shrunk
/// by [`world_render_scale`] like the rest of the human frame: the whole world renders ~36×
/// smaller, so the default 0.1 m near plane would sit a player-height-and-a-half out and clip
/// near geometry (the looming crab's nearest legs, a cockpit) — scaling it keeps the same
/// relative near clip the unscaled world had.
pub(super) fn spawn_fp_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: DEFAULT_CAMERA_NEAR * world_render_scale(),
            ..default()
        }),
        Camera {
            clear_color: ClearColorConfig::Custom(crab_world::sky::NIGHT_CLEAR),
            ..default()
        },
        Transform::default(),
        FpCamera,
    ));
}

/// Bevy's default perspective near plane (m). We scale it by [`world_render_scale`] for the
/// shrunk GCR frame; named so the FP and screenshot cameras can't drift to different near clips.
pub(super) const DEFAULT_CAMERA_NEAR: f32 = 0.1;
