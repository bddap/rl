//! Scene spawn + interpolated transforms: the gray-box world, avatars, the giant crab
//! silhouette, and the per-frame pose interpolation that smooths the 30 Hz sim to the
//! render rate. Reads the sim read-only; writes only Bevy `Transform`s (and the
//! client-side [`super::input::CameraYaw`] while alive).

use super::driver::{CockpitPose, GameState, LocalVehicle};
use super::input::{CameraPitch, CameraYaw};
use super::*;
use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy::math::Affine2;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

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

/// Visual ground quad edge length (render m). The quad is re-centered on the camera
/// every frame ([`follow_ground`]), so all that matters is that its edge sits past the
/// cameras' 1000 render-m far plane in every direction — 4000 gives 2× margin. Together
/// the two make the visual ground unbounded by construction, matching the unbounded
/// physics ground: there is no reachable edge.
const GROUND_SIZE: f32 = 4000.0;

/// Checker repeat period (render m): one texture repeat is 2×2 coarse cells, each a
/// 16×16 sub-checker. Deliberately RENDER-frame meters, NOT human-world meters shrunk
/// by [`world_render_scale`]: the pilot flies at TRUE arena scale ([`cockpit_camera`])
/// and is who the cue serves (rl#197) — the 2 m coarse cells read from cruise altitude,
/// the 0.125 m fine sub-cells (≈4.5 human-m) give optic flow at landing height and on
/// foot.
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

/// Spawn the static gray-box world (ground + extraction marker + a light) and the giant
/// crab. Player avatars are spawned by [`reconcile_avatars`] as sim players appear. Poses
/// are placed every frame by [`apply_transforms`]; here we just create the meshes once.
pub(super) fn spawn_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
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
    // The render-frame shrink (see [`world_render_scale`]). `world` already applies it to
    // ground POSITIONS; here it sizes the human-world MESHES. The crab silhouette is the
    // lone exception — it renders at native physics size.
    let rs = world_render_scale();

    // Ground: a checkered gray plane at Y=0, camera-following ([`follow_ground`]) so it
    // has no reachable edge — the physics ground is unbounded, and a fixed quad puts
    // "the end of the world" seconds of flight out (rl#197). The checker's optic flow
    // is what gives the pilot altitude and speed; plain gray gave none. Every round
    // entity here is DespawnOnExit(Playing): leaving the round (rl#203's disconnect
    // return to the menu) tears the scene down, so re-entering Playing respawns it
    // fresh instead of stacking a second world.
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

    // Sun-ish directional light so the gray-box reads with shape, plus a little
    // ambient so shadowed faces aren't pure black.
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

    // Extraction point: a tall bright glowing pillar — the objective beacon. Made
    // taller than the giant crab (CRAB_SCALE players high) and thick enough to read at
    // the far end of the map, so the goal stays legible even when the towering crab is
    // between you and it. (No ground disc: a flat EXTRACT_RADIUS marker at y=0 z-fought
    // the ground plane, and the pillar alone already marks the point unmistakably.)
    let ex = state.ls.sim().extraction().pos();
    let pillar_h = PLAYER_HEIGHT * CRAB_SCALE as f32 * 1.2;
    commands.spawn((
        DespawnOnExit(AppPhase::Playing),
        Mesh3d(meshes.add(Cylinder::new(0.5 * rs, pillar_h * rs))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.95, 0.3),
            emissive: LinearRgba::new(0.0, 2.2, 0.4, 1.0),
            ..default()
        })),
        Transform::from_translation(world(ex, pillar_h * 0.5)),
    ));

    // Player avatars are NOT spawned here — [`reconcile_avatars`] owns the render roster
    // (rl#205). Here we just create their shared assets once: one capsule mesh and one
    // material per color, cloned per spawn (the same handle-sharing the crab silhouette's
    // materials use), so roster churn never accretes duplicate assets.
    commands.insert_resource(AvatarAssets {
        mesh: meshes.add(Capsule3d::new(
            PLAYER_RADIUS * rs,
            (PLAYER_HEIGHT - 2.0 * PLAYER_RADIUS) * rs,
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

    // The giant crab: Sally's collider silhouette (see `spawn_crab_silhouette`), at TRUE physics
    // size ([`world_render_scale`]). Hidden only when the armed NN rig has a skin model
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
    // instead of the real one.
    let have_model = crab_world::mesh_fallback::usable_model_path().is_some();
    let crab_hidden = armed && have_model;
    // With the crab armed but NO model, the silhouette (the real colliders, NOT the real Sally
    // rig) stays shown AS the crab. Shipping that with no signal is a silent fallback, so on
    // the WINDOWED surface name it on screen with the banner
    // shared with rl-demo. The OTEL companion already fires in
    // `crab_world::mesh_fallback::initial_render_mode` (via `game::resolve_render_mode`); only
    // the live window adds the banner here. The screenshot path has no window (`primary_window:
    // None`) so a UI band can't pollute the capture — there the OTEL error + visible silhouette
    // are the signal.
    if armed && !have_model && !windows.is_empty() {
        // `!have_model` ⇒ the verdict is `Err`, so name the ACTUAL cause (absent vs broken) on
        // screen; `MESH_ABSENT_REASON` is only the fallback if the verdict somehow
        // lacks a reason.
        let reason = crab_world::mesh_fallback::usable_model()
            .as_ref()
            .err()
            .map_or(crab_world::mesh_fallback::MESH_ABSENT_REASON, |s| {
                s.as_str()
            });
        let banner = crab_world::mesh_fallback::spawn_banner(&mut commands, reason);
        // Round-scoped like everything else spawned here — without the tag the band would
        // persist over the menu after a disconnect return and stack on re-entry.
        commands
            .entity(banner)
            .insert(DespawnOnExit(AppPhase::Playing));
    }
    let crab_root = commands
        .spawn((
            DespawnOnExit(AppPhase::Playing),
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

/// The avatar capsule's shared render assets — ONE mesh, one material per color — created
/// once in [`spawn_world`] and cloned per spawn.
#[derive(Resource)]
pub(super) struct AvatarAssets {
    mesh: Handle<Mesh>,
    local: Handle<StandardMaterial>,
    remote: Handle<StandardMaterial>,
}

/// The SINGLE owner of the render-side player roster: every frame, spawn a capsule avatar
/// for each sim player that lacks one and despawn each avatar whose player left the sim.
/// A one-shot setup spawn left the roster static — a mid-game joiner with a genuinely new
/// `PlayerId` was invisible to incumbents (rl#205) — and a departed player's capsule froze
/// at its last pose (#198); deriving the roster from sim state each frame makes both
/// unrepresentable, with no join/leave event channel to miss. A rejoiner reusing a freed
/// pid gets a fresh spawn — or, when the depart and rejoin ticks land in one frame's
/// batch, recycles the still-live capsule, which is indistinguishable: pose and
/// visibility re-derive from the sim each frame, and a recycled pid is always
/// remote→remote (the local pid can't be freed while this client runs), so the
/// spawn-time material stays right. Chained ahead of [`apply_transforms`], whose
/// auto-inserted sync point applies the spawns/despawns the same frame; it re-poses and
/// re-hides every avatar each frame, so the placeholder transform here never shows.
pub(super) fn reconcile_avatars(
    mut commands: Commands,
    assets: Res<AvatarAssets>,
    state: NonSend<GameState>,
    avatars: Query<(Entity, &PlayerAvatar)>,
) {
    let sim = state.ls.sim();
    let have: std::collections::HashSet<PlayerId> = avatars
        .iter()
        .filter_map(|(entity, avatar)| {
            if sim.player(avatar.0).is_some() {
                Some(avatar.0)
            } else {
                // The player DEPARTED mid-match (the adopted snapshot no longer carries
                // them): drop the capsule, freeing its pid for a fresh rejoin spawn.
                commands.entity(entity).despawn();
                None
            }
        })
        .collect();
    let local = state.ls.me();
    for (id, p) in sim.players() {
        if have.contains(&id) {
            continue;
        }
        // The local player's avatar is spawned too (kept hidden in apply_transforms — we
        // view from its eyes) so status handling stays uniform.
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

/// The giant crab's apparent-height target: a player's height times [`CRAB_SCALE`] — the crab
/// reads CRAB_SCALE players tall. Hit by shrinking the rest of the world, never the crab
/// (see [`world_render_scale`]). Kept only to derive the world scale.
const CRAB_RENDER_HEIGHT: f32 = PLAYER_HEIGHT * CRAB_SCALE as f32;

/// The crab rig's natural standing height (m) — the span of its REAL physics colliders. Memoized:
/// the recipe + silhouette geometry it derives are a property of the fixed binary+asset that never
/// changes at runtime, yet this is read to set the world render scale on a per-frame path, so the
/// derivation is cached rather than re-run each read. The underlying 36 MB glb parse + fit is
/// itself memoized once in [`crab_world::mesh_fallback::usable_model`], so
/// [`crab_world::bot::body::render_recipe`] here only clones that recipe. `None` for a degenerate
/// (zero-height) recipe — a broken collider asset with no honest geometry, which callers fail LOUD on
/// rather than drawing a stand-in.
fn natural_crab_height() -> Option<f32> {
    static H: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *H.get_or_init(|| {
        // Flips off the preflight verdict (`mesh_fallback::usable_model_path`), NOT the per-app
        // `CrabModelPath` resource: this answers "how big is the crab this run draws?" to set the
        // world render scale, a fixed binary+asset constant. Preflighted so a present-but-broken glb
        // draws the fallback recipe's height (what's actually rendered) instead of the real one.
        // net has one asset source, so this and the silhouette below flip together.
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
/// a bigger collider scale). The crab's natural height over the giant target height.
/// The ONE scale source — every sim→render ground position ([`world`])
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
/// SAME shapes the sim body uses means the cosmetic crab can't drift from the body
/// it depicts, and it reads as Sally instead of a box. `render_recipe` is the single
/// model-vs-fallback selector (shared with the trainer), so this never invents a second source of
/// geometry. This static silhouette is the honest physics-bones view (no articulation): the
/// headless-screenshot render, and the in-game view whenever no skin model resolves — the rest
/// stance posed rigidly, so the legs don't walk, but it is the REAL colliders, never a stand-in
/// box. The real armed round with a model shows the walking skinned NN rig instead. `have_model` is
/// the preflight verdict's flip (`mesh_fallback::usable_model_path().is_some()`) resolved ONCE in
/// `spawn_world` — net's single asset source, the SAME verdict the body's `CrabModelPath` defaults to,
/// so they agree — and `false` (absent or broken glb) draws the fallback recipe; `true` draws the
/// memoized real recipe the verdict already built, never a re-parse or an `.expect()`.
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
    // The crab draws at its TRUE physics size ([`world_render_scale`]).
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
            // [`reconcile_avatars`] (chained just ahead, and the sim only changes in
            // `drive_lockstep` before both) despawned every avatar without a sim player,
            // so one here can only mean the roster reconcile was unwired — fail loud
            // rather than pose a ghost.
            unreachable!(
                "avatar for departed player {:?} survived reconcile_avatars",
                avatar.0
            );
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
/// The pose is in the crab's ARENA frame (the rapier vehicle body lives on the open inference
/// field with Sally — rl#209). `shift` is the pure XZ translate that carries an arena point to its
/// render spot anchored
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

/// The FP cameras' perspective: Bevy's default 0.1 m near plane, shrunk by
/// [`world_render_scale`] like the rest of the human frame — the world renders ~36× smaller,
/// so unscaled it would sit a player-height-and-a-half out and clip near geometry (the
/// looming crab's nearest legs, a cockpit). What actually clips in Bevy 0.18 is the oblique
/// `near_clip_plane` (a portals/mirrors feature), which DEFAULTS to the stock 0.1 m plane
/// independent of `near` — leave it stale and the view still clips at 0.1 render-m, ~2
/// eye-heights out (looking down while standing saw through the floor, rl#196) — so the two
/// move together here. The ONE perspective source for the windowed and screenshot FP
/// cameras, so their clips can't drift.
pub(super) fn fp_perspective() -> PerspectiveProjection {
    let near = 0.1 * world_render_scale();
    PerspectiveProjection {
        near,
        near_clip_plane: Vec4::new(0.0, 0.0, -1.0, -near),
        ..default()
    }
}

/// Spawn the windowed first-person camera. Its transform is overwritten every frame
/// by [`apply_transforms`]; the night-sky skybox ([`crab_world::sky`]) paints the
/// background, with this dark clear color as the pre-upload fallback.
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
