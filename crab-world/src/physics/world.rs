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

/// Which arena PHYSICS a world gets. Since the rl#293 flip the baked terrain tile is
/// THE production ground — training, eval, GCR, demo — so `Terrain` is the only
/// variant a production world builds; the flat variants exist for tests that pin
/// flat-grid semantics until the stage-3 deletion removes them. Every variant is a
/// height grid through the ONE terrain path ([`crate::terrain`], rl#281): the ground
/// collider is always [`TerrainGrid::collider`], only the grid (and the walls) differ.
pub enum Arena {
    /// The legacy ±10 m walled training box ([`ARENA_HALF_SIZE`]): a flat grid plus
    /// four walls. Test-only since rl#293 (flat plants refuse to load); deleted in
    /// the epic's stage 3.
    WalledBox,
    /// An unwalled flat grid at y=0 spanning ±[`OPEN_FIELD_HALF_SIZE`] — GCR's
    /// pre-terrain inference field. Test-only since rl#293; deleted in stage 3.
    OpenField,
    /// The committed GCR terrain bake ([`TerrainGrid::gcr`], rl#281) — real
    /// mountains, no walls. THE canonical ground (rl#293): the trainer, the
    /// keep-best/honest evals, GCR, and rl-demo all build exactly this.
    Terrain,
}

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

impl Arena {
    fn grid(&self) -> Arc<TerrainGrid> {
        match self {
            Arena::WalledBox => Arc::new(TerrainGrid::flat(ARENA_HALF_SIZE)),
            Arena::OpenField => Arc::new(TerrainGrid::flat(OPEN_FIELD_HALF_SIZE)),
            Arena::Terrain => TerrainGrid::gcr(),
        }
    }
}

/// Plugin that lays the arena PHYSICS — the ground (+ wall) colliders per [`Arena`] —
/// and publishes the [`Terrain`] resource spawn-on-surface consumers sample. No meshes
/// or lights: the visible dressing is a SEPARATE [`ArenaVisualsPlugin`] (split rl#160),
/// so a headless host adds the colliders alone. Every rendered surface — rl-demo and,
/// since rl#281 stage 6, GCR — adds both: there is ONE way an arena looks.
pub struct PhysicsWorldPlugin {
    pub arena: Arena,
}

impl Plugin for PhysicsWorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Terrain::new(self.arena.grid()));
        app.add_systems(Startup, setup_ground);
        if matches!(self.arena, Arena::WalledBox) {
            app.add_systems(Startup, setup_walls);
        }
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

pub(crate) const ARENA_HALF_SIZE: f32 = 10.0;
/// Half-span of the [`Arena::OpenField`] flat grid. The old halfspace was truly
/// unbounded; a grid has an edge, so this is sized to make the edge unreachable in
/// practice (≥ the GCR terrain tile's own ~±15 km span). A real world-bounds policy is a
/// later stage of rl#281.
const OPEN_FIELD_HALF_SIZE: f32 = 16_384.0;
const WALL_HEIGHT: f32 = 2.0;
const WALL_THICKNESS: f32 = 0.5;

fn wall_boxes() -> [(Vec3, Vec3); 4] {
    [
        (
            Vec3::new(0.0, WALL_HEIGHT / 2.0, ARENA_HALF_SIZE + WALL_THICKNESS),
            Vec3::new(ARENA_HALF_SIZE, WALL_HEIGHT / 2.0, WALL_THICKNESS),
        ),
        (
            Vec3::new(0.0, WALL_HEIGHT / 2.0, -ARENA_HALF_SIZE - WALL_THICKNESS),
            Vec3::new(ARENA_HALF_SIZE, WALL_HEIGHT / 2.0, WALL_THICKNESS),
        ),
        (
            Vec3::new(ARENA_HALF_SIZE + WALL_THICKNESS, WALL_HEIGHT / 2.0, 0.0),
            Vec3::new(WALL_THICKNESS, WALL_HEIGHT / 2.0, ARENA_HALF_SIZE),
        ),
        (
            Vec3::new(-ARENA_HALF_SIZE - WALL_THICKNESS, WALL_HEIGHT / 2.0, 0.0),
            Vec3::new(WALL_THICKNESS, WALL_HEIGHT / 2.0, ARENA_HALF_SIZE),
        ),
    ]
}

/// Lay the ground collider — the one terrain heightfield, whatever the arena. No meshes
/// or lights (those are render-only — see [`setup_arena_visuals`]), so this is the one
/// arena setup the headless trainer runs, touching no graphics types.
fn setup_ground(mut commands: Commands, terrain: Res<Terrain>) {
    commands.spawn((
        RigidBody::Fixed,
        terrain.collider(),
        ARENA_COLLISION,
        Transform::IDENTITY,
    ));
}

/// The [`Arena::WalledBox`] walls (same contact identity as the ground:
/// [`ARENA_COLLISION`]).
fn setup_walls(mut commands: Commands) {
    for (pos, half_extents) in wall_boxes() {
        commands.spawn((
            RigidBody::Fixed,
            Collider::cuboid(half_extents.x, half_extents.y, half_extents.z),
            ARENA_COLLISION,
            Transform::from_translation(pos),
        ));
    }
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

    // Flat arenas keep the pre-terrain training-box light exactly; a vista world
    // (rl#281 stage 3) gets a lower, cooler moon-sun for long relief shadows, plus
    // cascades stretched from the ~150 m default to mountain scale (30 m grid pitch
    // makes coarse far cascades invisible).
    if !terrain.is_flat() {
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
    } else {
        commands.spawn((
            DirectionalLight {
                shadows_enabled: true,
                illuminance: 10000.0,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.3, 0.0)),
        ));
        commands.insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 300.0,
            ..default()
        });
    }

    // The visible ground: the SAME grid the collider was built from (rl#281 — render
    // matches physics by construction). Tint rides the mesh's vertex colors (biome
    // bands on terrain, the old flat green on flat arenas), so the material stays
    // white and ONE material serves every arena. A FLAT arena is otherwise a
    // featureless plane — no ground-speed or landing-height cue — so it alone layers
    // the rl#197 pilot checker over the green (bddap/rl#287); terrain self-cues with
    // relief, biome bands, and fog.
    commands.spawn((
        ArenaSurface,
        Mesh3d(meshes.add(terrain.mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: terrain.is_flat().then(|| ground_checker(&mut images)),
            // Flat-mesh UVs are mesh-local METERS (terrain.rs); scale so one texture
            // repeat covers CHECKER_PERIOD meters — hosts move ArenaSurface by
            // translation only (net's anchor sync), so mesh meters stay render meters
            // and the pilot-scale cells hold. Inert without a texture.
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

/// The flat ground's repeat-tiled checker texture: a coarse 2×2 checker with a fainter
/// sub-checker on top (two spatial frequencies — see [`CHECKER_PERIOD`]), tinting the
/// mesh's vertex-color green so the flat arenas keep their tone. Built with a full mip
/// chain — box-filtering per level — so the pattern fades to flat tint in the distance
/// instead of shimmering (a generated `Image` gets no auto-mips).
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

/// Give every camera in a vista (real-relief, rendered) world aerial-perspective fog toward the night
/// sky's horizon color. Runs in `Update` because cameras can spawn any time (the
/// demo's orbit cam, the offscreen shot cam); same attach pattern as the skybox.
#[cfg(feature = "render")]
fn attach_vista_fog(
    mut commands: Commands,
    visuals: Res<crate::Visuals>,
    terrain: Res<crate::terrain::Terrain>,
    cams: Query<Entity, (With<Camera3d>, Without<DistanceFog>)>,
) {
    if !visuals.0 || terrain.is_flat() {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The open field must span at least the GCR terrain tile — the prose rationale for
    /// [`OPEN_FIELD_HALF_SIZE`] ("edge unreachable in practice") silently rots if a
    /// bigger tile is baked.
    #[test]
    fn open_field_spans_the_gcr_tile() {
        let g = TerrainGrid::gcr();
        assert!(OPEN_FIELD_HALF_SIZE >= g.extent_x() / 2.0);
        assert!(OPEN_FIELD_HALF_SIZE >= g.extent_z() / 2.0);
    }
}
