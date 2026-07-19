use std::sync::Arc;

use bevy::prelude::*;
#[cfg(feature = "render")]
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    math::Affine2,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_rapier3d::prelude::*;

use crate::bot::body::ARENA_COLLISION;
use crate::terrain::{Terrain, TerrainGrid};

/// The plant sidecar's arena spelling — the one string a loadable checkpoint may
/// record. Kept as the digest/sidecar preimage across the rl#293 flip so identical
/// worlds never refuse-to-arm between peers on either side of it.
pub(crate) const ARENA_TERRAIN_KEY: &str = "terrain";

/// Parse a plant sidecar's `arena` value: `Ok(true)` = terrain, `Ok(false)` =
/// `walled_box` (a real pre-flip record — it PARSES so adoption can refuse it with
/// provenance intact, or the rl-demo `--terrain` waiver can view it; a parse-time
/// refusal would rank honest flat records below recordless ones). Unknown spellings
/// refuse: an unreadable plant must never load.
pub(crate) fn parse_arena_key(raw: &str) -> Result<bool, String> {
    match raw.trim() {
        k if k == ARENA_TERRAIN_KEY => Ok(true),
        "walled_box" => Ok(false),
        other => Err(format!("unknown arena {other:?} (terrain | walled_box)")),
    }
}

/// Plugin that lays the arena PHYSICS — the ground collider — and publishes the
/// [`Terrain`] resource spawn-on-surface consumers sample. Since the rl#293 flip
/// there is ONE production ground: the committed GCR bake ([`TerrainGrid::gcr`]) —
/// the trainer, the keep-best/honest evals, GCR, and rl-demo all build exactly it,
/// and it is the only grid production code can construct ([`TerrainGrid::flat`] is
/// test-gated). Tests may pass a fixture grid: a degenerate parameter of the one
/// seam, not a second ground path. No meshes or lights: the visible dressing is a
/// SEPARATE [`ArenaVisualsPlugin`] (split rl#160), so a headless host adds the
/// collider alone. Every rendered surface — rl-demo and, since rl#281 stage 6,
/// GCR — adds both: there is ONE way an arena looks.
pub struct PhysicsWorldPlugin {
    pub grid: Arc<TerrainGrid>,
}

impl Plugin for PhysicsWorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Terrain::new(self.grid.clone()));
        app.add_systems(Startup, setup_ground);
    }
}

/// The visible arena ground spawned by [`setup_arena_visuals`]. The PHYSICS ground
/// (collider) always sits at the physics-frame origin; a host whose render frame is a
/// translation of the physics frame (net's arena↔world anchor, rl#224) uses this
/// handle to keep the drawn surface at that translation. Hosts whose two frames
/// coincide (rl-demo) leave it at IDENTITY.
#[cfg(feature = "render")]
#[derive(Component)]
pub struct ArenaSurface;

#[cfg(feature = "render")]
pub struct ArenaVisualsPlugin;

#[cfg(feature = "render")]
impl Plugin for ArenaVisualsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_arena_visuals);
        // Update, not Startup: cameras spawn after Startup in some modes (same
        // pattern as NightSkyPlugin's skybox attach).
        app.add_systems(Update, attach_vista_fog);
    }
}

/// Lay the ground collider — the one terrain heightfield. No meshes or lights (those
/// are render-only — see [`setup_arena_visuals`]), so this is the one arena setup the
/// headless trainer runs, touching no graphics types.
fn setup_ground(mut commands: Commands, terrain: Res<Terrain>) {
    commands.spawn((
        RigidBody::Fixed,
        terrain.collider(),
        ARENA_COLLISION,
        Transform::IDENTITY,
    ));
}

#[cfg(feature = "render")]
fn setup_arena_visuals(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    terrain: Res<Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if !visuals.0 {
        return;
    }

    // A vista world (rl#281 stage 3) gets a low, cool moon-sun for long relief
    // shadows, plus cascades stretched from the ~150 m default to mountain scale
    // (30 m grid pitch makes coarse far cascades invisible).
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 9500.0,
            color: Color::srgb(0.85, 0.90, 1.0),
            ..default()
        },
        bevy::light::CascadeShadowConfigBuilder {
            maximum_distance: 9000.0,
            first_cascade_far_bound: 20.0,
            ..default()
        }
        .build(),
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.7, 0.0)),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.75, 0.82, 1.0),
        brightness: 400.0,
        ..default()
    });

    // The visible ground: the SAME grid the collider was built from (rl#281 — render
    // matches physics by construction). Tint rides the mesh's vertex colors (biome
    // bands), multiplied by the rl#197 pilot checker as a detail texture (rl#287 →
    // rl#293): the 30 m-pitch interpolated surface has no high-frequency detail of
    // its own, and the checker's fine cells are what give optic flow at landing
    // height and on foot — inside the one ground path, never a second surface.
    commands.spawn((
        ArenaSurface,
        Mesh3d(meshes.add(terrain.mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(ground_checker(&mut images)),
            // Mesh UVs are mesh-local METERS (terrain.rs); scale so one texture
            // repeat covers CHECKER_PERIOD meters — hosts move ArenaSurface by
            // translation only (net's anchor sync), so mesh meters stay render meters
            // and the pilot-scale cells hold.
            uv_transform: Affine2::from_scale(Vec2::splat(1.0 / CHECKER_PERIOD)),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::IDENTITY,
    ));
}

/// Checker repeat period (render m): one texture repeat is 2×2 coarse cells, each a
/// 16×16 sub-checker. Sized for the PILOT, who the cue serves (rl#197): the 2 m coarse
/// cells read from cruise altitude, the 0.125 m fine sub-cells (~2.5 player heights)
/// give optic flow at landing height and on foot.
#[cfg(feature = "render")]
const CHECKER_PERIOD: f32 = 4.0;

/// The ground's repeat-tiled checker detail texture: a coarse 2×2 checker with a
/// fainter sub-checker on top (two spatial frequencies — see [`CHECKER_PERIOD`]),
/// near-white so it modulates the mesh's biome vertex tint instead of replacing it.
/// Built with a full mip chain — box-filtering per level — so the pattern fades to
/// flat tint in the distance instead of shimmering (a generated `Image` gets no
/// auto-mips).
#[cfg(feature = "render")]
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
        // trilinear blurs the checker to flat tint — right where the optic-flow cue
        // matters most.
        anisotropy_clamp: 16,
        ..ImageSamplerDescriptor::linear()
    });
    images.add(image)
}

/// Give every camera aerial-perspective fog toward the night sky's horizon color.
/// Runs in `Update` because cameras can spawn any time (the demo's orbit cam, the
/// offscreen shot cam); same attach pattern as the skybox.
#[cfg(feature = "render")]
fn attach_vista_fog(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    cams: Query<Entity, (With<Camera3d>, Without<DistanceFog>)>,
) {
    if !visuals.0 {
        return;
    }
    for cam in &cams {
        commands.entity(cam).insert(DistanceFog {
            color: crate::sky::horizon_fog_color(),
            falloff: FogFalloff::Linear {
                start: 600.0,
                end: 11000.0,
            },
            ..default()
        });
    }
}
