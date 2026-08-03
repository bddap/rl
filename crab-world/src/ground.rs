//! The ground's look (bddap/rl#304): a procedural detail layer over the terrain
//! mesh's macro biome tint. All detail is computed in the fragment shader from
//! WORLD-SPACE position — no repeated texture anywhere in the ground path, so there
//! is no tile period to spot at any play scale, and the mesh stays the collider's
//! exact geometry (render matches physics; beauty is shading only).
//!
//! Every [`GroundLook`] is swappable at runtime and none of them can move physics or
//! eval: the swap reaches only the fragment shader the ground pipeline is specialized
//! with.

use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    MaterialPlugin,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

/// The one ground material: stock PBR (lighting, shadows, fog, vertex biome tint)
/// with the procedural detail fragment on top.
pub type GroundMaterial = ExtendedMaterial<StandardMaterial, GroundDetail>;

/// One interchangeable ground look — the shipped one, the surviving rl#304
/// competition entries, the rl#323 watershed unification, and the rl#329
/// night-bloom parameter sets — and the live selection: `GroundLook` is BOTH the
/// `--ground-look` value and the resource the in-game toggle cycles, the same
/// shape [`crate::crab_view::RenderMode`] uses, so there is one field to write and
/// no wrapper to keep in sync.
///
/// Each look is a whole fragment shader in `src/ground_looks/`, and the contract
/// every one of those files keeps is:
///
/// Inputs (all set up by [`GroundMaterial`]):
/// - `in: VertexOutput` (bevy_pbr forward io): `world_position` is true world space
///   in METERS, y up, terrain spans ~±15.3 km in xz; `world_normal` is the smooth
///   geometric normal. The mesh carries NO UVs — derive everything from world
///   position. The per-vertex biome tint (hypsometric bands, see terrain.rs
///   `biome`) arrives pre-multiplied into `pbr_input.material.base_color` via the
///   mesh COLOR attribute.
/// - `strengths: vec4<f32>` at `@group(#{MATERIAL_BIND_GROUP}) @binding(100)` — the
///   ONE extension uniform, the taste-iteration knob set (defaults + meaning in
///   [`GroundDetail`]). A look may reinterpret the lanes but must bind them at 100
///   (the Rust side is `AsBindGroup` on that binding).
/// - the hydrology bake at bindings 101 (texture, R wetness / G standing water),
///   102 (sampler), 103 (world-extent uniform) — optional to declare; watershed is
///   the consumer. World-mapped over the whole tile, so it cannot tile.
/// - the look's aesthetic parameter row at binding 104 (`array<vec4f, 8>`, lanes
///   documented on [`GroundLook::params`]) — optional to declare, REQUIRED for a
///   shader several looks share: a variant is a param row, not a shader fork
///   (the rl#333 seam). night_bloom is the consumer.
///
/// Rules (issue requirements, not style):
/// - entry point stays `fn fragment`. Only the FORWARD main pass draws these files
///   ([`GroundDetail::specialize`] leaves the prepass alone), so the
///   `PREPASS_PIPELINE` fork they carry is inert — it just keeps the file valid if
///   prepass wiring is ever added.
/// - world-space procedural only: no REPEATED textures (the world-spanning
///   hydrology bake above is the one sanctioned sample), and nothing that
///   lies about geometry — normal perturbation is fine, parallax/displacement is
///   NOT (the visual surface IS the collision heightfield).
/// - octaves faded by their on-screen footprint (`fwidth`): fine detail must exist
///   on foot and at landing height — the rl#197 optic-flow cue the deleted checker
///   used to carry — without shimmering from the plane.
/// - judge colors at rendered exposure (moon-sun + ambient are pre-exposed).
///
/// To add a look: a file in `src/ground_looks/`, a variant here, and a row in
/// [`ground_looks!`] — which is exhaustive over this enum, so a variant with no row
/// is a compile error rather than a look nobody can reach.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, clap::ValueEnum)]
pub enum GroundLook {
    #[default]
    Shipped,
    NightBloom,
    PatternedGround,
    WindCombed,
    CrackedLoam,
    WetNocturne,
    Watershed,
    // The night-bloom parameter sets (owner request 2026-08-01: "more variants of
    // what is now called night bloom. Parameterizing night bloom could go a long
    // way"): ONE shader, distinct rows in [`GroundLook::params`] — taste
    // candidates the owner judges in play. NightBloomAurora carries the rl#329
    // aurora palette (the glowy green/violet water colors) forward.
    NightBloomAurora,
    NightBloomEmber,
    NightBloomFrost,
    NightBloomRose,
    NightBloomFiligree,
}

/// The one place a look's shader UUID and its file sit together. Split across a
/// `match` and a separate registration call, they drift, and a look ends up drawing
/// its neighbour's shader with every test still green. A macro because
/// `load_internal_asset!` embeds via `include_str!` — the path must be a literal per
/// look, so no loop over the variants can build the registration list.
macro_rules! ground_looks {
    ($($($look:ident)|+ => $file:literal, $uuid:literal;)*) => {
        impl GroundLook {
            /// The look's shader. A const UUID handle rather than a loaded path
            /// because [`MaterialExtension::specialize`] — the only place that can
            /// repoint a pipeline's fragment stage — gets no world access to
            /// resolve one. Valid only once [`GroundMaterialPlugin`] has registered
            /// it, hence private. Several looks may share one shader (a row with
            /// `A | B | …`) — those are parameter sets over it, told apart by
            /// [`GroundLook::params`].
            fn shader(self) -> Handle<Shader> {
                match self {
                    $($(GroundLook::$look)|+ => uuid_handle!($uuid),)*
                }
            }
        }

        /// Compiled in, not loaded: `include_str!` makes a missing or misnamed look
        /// a BUILD error instead of a runtime "path not found" that un-renders the
        /// ground.
        fn register_look_shaders(app: &mut App) {
            $(load_internal_asset!(
                app,
                uuid_handle!($uuid),
                concat!("ground_looks/", $file, ".wgsl"),
                Shader::from_wgsl
            );)*
        }
    };
}

ground_looks! {
    Shipped => "shipped", "9f2a7c14-3b6d-4e58-9a01-5c7d2e8f4b60";
    NightBloom | NightBloomAurora | NightBloomEmber | NightBloomFrost
        | NightBloomRose | NightBloomFiligree => "night_bloom", "6b0d3f57-92ac-41e8-b74d-0a3e5c81f296";
    PatternedGround => "patterned_ground", "e83c1a06-5d4f-4b29-97e0-6c2b8d70a415";
    WindCombed => "wind_combed", "2d795be8-0c31-4a76-8b5e-91f4a03d6c27";
    CrackedLoam => "cracked_loam", "48a6f2d1-b90e-4c53-a812-7d05e9b34f6a";
    WetNocturne => "wet_nocturne", "7fe15b83-2a4c-49d0-91b6-c3820e5a7d14";
    Watershed => "watershed", "3a9d6c07-14be-4f82-a5c3-8e07b12d9f4b";
}

impl GroundLook {
    /// The look's aesthetic parameter row for the `@binding(104)` uniform (the
    /// rl#333 params seam): a variant of a shared shader is a row here, not a
    /// shader fork. Zero for every look whose shader doesn't read 104. Exhaustive
    /// over the enum, so a new variant without a row is a compile error.
    ///
    /// night_bloom.wgsl lanes:
    /// - `[0]` xyz night tint (base-color multiplier), w ground darkening under glow
    /// - `[1]` xyz vein color, w artery emissive intensity
    /// - `[2]` xyz knot-flush color, w knot mix strength
    /// - `[3]` xyz spore color, w spore emissive intensity
    /// - `[4]` x capillary intensity, y spore rarity threshold (higher = sparser),
    ///   z artery spacing (m), w capillary spacing (m)
    /// - `[5]` x artery width, y artery core width, z capillary width (noise-space)
    pub fn params(self) -> [Vec4; 8] {
        let zero = [Vec4::ZERO; 8];
        let row = |tint: [f32; 4], vein: [f32; 4], knot: [f32; 4], spore: [f32; 4], field: [f32; 4], widths: [f32; 3]| {
            let mut p = zero;
            p[0] = Vec4::from_array(tint);
            p[1] = Vec4::from_array(vein);
            p[2] = Vec4::from_array(knot);
            p[3] = Vec4::from_array(spore);
            p[4] = Vec4::from_array(field);
            p[5] = Vec4::new(widths[0], widths[1], widths[2], 0.0);
            p
        };
        match self {
            Self::Shipped
            | Self::PatternedGround
            | Self::WindCombed
            | Self::CrackedLoam
            | Self::WetNocturne
            | Self::Watershed => zero,
            // The original night bloom, constants unchanged.
            Self::NightBloom => row(
                [0.70, 0.88, 0.98, 0.55],
                [0.05, 0.85, 0.62, 2.2],
                [0.75, 0.10, 0.55, 0.55],
                [0.35, 0.85, 0.90, 1.4],
                [1.0, 0.86, 180.0, 14.0],
                [0.10, 0.035, 0.13],
            ),
            // The rl#329 aurora water palette (the liked glowy green/violet),
            // carried onto the vein web: green rivers, violet knots.
            Self::NightBloomAurora => row(
                [0.74, 0.84, 1.02, 0.60],
                [0.13, 1.00, 0.42, 2.4],
                [0.55, 0.18, 0.95, 0.70],
                [0.55, 0.95, 0.70, 1.2],
                [1.0, 0.88, 200.0, 16.0],
                [0.12, 0.04, 0.14],
            ),
            // Warm ember veins in dusk-brown ground, dense gold firefly spores.
            Self::NightBloomEmber => row(
                [0.88, 0.78, 0.66, 0.50],
                [1.00, 0.45, 0.10, 2.0],
                [0.85, 0.08, 0.05, 0.60],
                [1.00, 0.80, 0.35, 1.6],
                [1.0, 0.80, 150.0, 11.0],
                [0.08, 0.03, 0.11],
            ),
            // Icy filaments: thin sparse pale-blue webs, dense white glitter,
            // little wet-darkening — frost sitting ON the ground, not soaking it.
            Self::NightBloomFrost => row(
                [0.78, 0.90, 1.06, 0.25],
                [0.55, 0.80, 1.00, 1.6],
                [0.95, 0.98, 1.00, 0.40],
                [0.90, 0.95, 1.00, 2.0],
                [1.0, 0.75, 240.0, 18.0],
                [0.07, 0.025, 0.09],
            ),
            // The classic palette inverted: rose-magenta web, teal knots.
            Self::NightBloomRose => row(
                [0.80, 0.72, 0.92, 0.55],
                [0.95, 0.15, 0.55, 2.2],
                [0.10, 0.85, 0.75, 0.50],
                [1.00, 0.60, 0.80, 1.3],
                [1.0, 0.86, 180.0, 14.0],
                [0.10, 0.035, 0.13],
            ),
            // Structure as the differentiator: a 2×-denser, thinner, dimmer lace —
            // capillary-forward, spores nearly absent.
            Self::NightBloomFiligree => row(
                [0.70, 0.88, 0.98, 0.45],
                [0.10, 0.70, 0.85, 1.4],
                [0.75, 0.10, 0.55, 0.45],
                [0.35, 0.85, 0.90, 1.0],
                [1.2, 0.93, 90.0, 7.0],
                [0.06, 0.02, 0.08],
            ),
        }
    }
}

/// Strength knobs for the shader's layers, one uniform so a taste iteration is a
/// constant tweak, not a shader rewrite. Defaults are the shipped look; a look may
/// reinterpret the lanes (see [`GroundLook`]).
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
#[bind_group_data(GroundLook)]
pub struct GroundDetail {
    /// x: macro patchiness (hundreds of m), y: meso mottling (tens of m),
    /// z: fine on-foot detail, w: detail-normal strength.
    #[uniform(100)]
    pub strengths: Vec4,
    /// The hydrology bake over the terrain grid ([`crate::moisture::MoistureMap`]):
    /// R wetness, G standing water. World-mapped (`uv = world_xz / extent + 0.5`),
    /// so it has no repeat period — the one texture the no-tiling rule permits.
    #[texture(101)]
    #[sampler(102)]
    pub moisture: Handle<Image>,
    /// x, y: the moisture map's world extents in meters (terrain extent_x/extent_z)
    /// — the shader's world→uv scale. z, w unused.
    #[uniform(103)]
    pub moisture_extent: Vec4,
    /// The look's aesthetic parameter row ([`GroundLook::params`]) — optional for
    /// a shader to declare (`@binding(104)`); night_bloom is the consumer.
    #[uniform(104)]
    pub params: [Vec4; 8],
    /// Which look this material draws. Not a binding — it rides in the material to
    /// reach [`MaterialExtension::specialize`] as the pipeline key, so two ground
    /// materials share a pipeline iff they draw the same look and a swap is a
    /// re-specialization rather than a second material type.
    pub look: GroundLook,
}

impl GroundDetail {
    pub fn new(look: GroundLook, moisture: Handle<Image>, extent: Vec2) -> Self {
        Self {
            strengths: Vec4::new(0.55, 0.35, 0.45, 0.6),
            moisture,
            moisture_extent: Vec4::new(extent.x, extent.y, 0.0, 0.0),
            params: look.params(),
            look,
        }
    }
}

/// Required by `#[bind_group_data(GroundLook)]`: the derive calls `self.into()` to
/// produce the pipeline key.
impl From<&GroundDetail> for GroundLook {
    fn from(detail: &GroundDetail) -> Self {
        detail.look
    }
}

impl MaterialExtension for GroundDetail {
    /// A placeholder [`Self::specialize`] always overrides on the forward pass; it
    /// exists because the descriptor needs a fragment stage before specialization
    /// runs.
    fn fragment_shader() -> ShaderRef {
        GroundLook::Shipped.shader().into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let Some(fragment) = descriptor.fragment.as_mut() else {
            return Ok(());
        };
        // Bevy runs ONE specialize for both the forward and the prepass/shadow
        // pipelines. A look shader is forward-only (it writes a deferred/lit
        // output), so stamping it onto the prepass descriptor breaks the ground's
        // shadow pass wherever the prepass carries a fragment stage — adapters
        // without DEPTH_CLIP_CONTROL, which emulate unclipped depth in the shader.
        // Match the DEF NAME, not a whole `ShaderDefVal`: were bevy ever to push it
        // with a different payload, an equality check would silently stop matching
        // and break shadows only on the adapters nobody tests on.
        let is_prepass = fragment.shader_defs.iter().any(
            |def| matches!(def, bevy::shader::ShaderDefVal::Bool(name, _) if name == "PREPASS_PIPELINE"),
        );
        if is_prepass {
            return Ok(());
        }
        fragment.shader = key.bind_group_data.shader();
        Ok(())
    }
}

fn apply_ground_look(look: Res<GroundLook>, mut materials: ResMut<Assets<GroundMaterial>>) {
    if !look.is_changed() {
        return;
    }
    for (_, material) in materials.iter_mut() {
        material.extension.look = *look;
        material.extension.params = look.params();
    }
}

/// The transient "Ground: <name>" readout.
#[derive(Component)]
struct GroundLookReadout(Timer);

fn spawn_look_readout(mut commands: Commands) {
    commands.spawn((
        crate::crab_view::hud_corner_label(crate::crab_view::HudRow::GroundLook),
        GroundLookReadout(Timer::from_seconds(
            crate::crab_view::READOUT_SECS,
            TimerMode::Once,
        )),
    ));
}

/// Name the look on every change EXCEPT the boot one (`is_added`): booting into a
/// look is not a toggle, and a readout burned into the first seconds of every run
/// would land in every evidence screenshot.
fn update_look_readout(
    look: Res<GroundLook>,
    time: Res<Time>,
    mut readout: Query<(&mut Text, &mut GroundLookReadout)>,
) {
    let Ok((mut text, mut state)) = readout.single_mut() else {
        return;
    };
    if look.is_changed() && !look.is_added() {
        **text = format!("Ground: {}", crate::view_variant_name(&*look));
        state.0.reset();
    }
    if !state.0.is_finished() && state.0.tick(time.delta()).just_finished() {
        text.clear();
    }
}

/// Registers [`GroundMaterial`], every embedded look, and the swap system. Added by
/// `ArenaVisualsPlugin`, so every rendered surface gets it through the one arena
/// path. `initial` is the look the app boots in (`--ground-look`).
pub struct GroundMaterialPlugin {
    pub initial: GroundLook,
}

impl Plugin for GroundMaterialPlugin {
    fn build(&self, app: &mut App) {
        register_look_shaders(app);
        app.add_plugins(MaterialPlugin::<GroundMaterial>::default())
            .insert_resource(self.initial)
            .add_systems(Startup, spawn_look_readout)
            .add_systems(Update, (apply_ground_look, update_look_readout));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A copy-pasted UUID or param row would make two looks render identically —
    /// the toggle would still "work" and the player would just never see one of
    /// them. Identity is (shader, params): looks may share a shader on purpose
    /// (parameter sets), never a shader AND a row.
    #[test]
    fn every_look_renders_distinctly() {
        let all = <GroundLook as clap::ValueEnum>::value_variants();
        let mut ids: Vec<_> = all
            .iter()
            .map(|l| (l.shader().id(), l.params().map(|v| v.to_array().map(f32::to_bits))))
            .collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), all.len());
    }

    /// A parameterized look whose shader ignores binding 104 would collapse every
    /// variant onto one rendering. Any look sharing its shader with another must
    /// point at a file that declares the params binding.
    #[test]
    fn shared_shaders_read_the_params_binding() {
        let night_bloom = include_str!("ground_looks/night_bloom.wgsl");
        assert!(night_bloom.contains("@binding(104)"));
    }

    /// The runtime swap, end to end below the keybind: cycling the resource must
    /// repoint the LIVE ground material, which is what re-specializes its pipeline
    /// onto the next look's shader.
    #[test]
    fn cycling_repoints_the_live_ground_material() {
        let mut app = App::new();
        app.add_plugins(bevy::asset::AssetPlugin::default())
            .init_asset::<Shader>()
            .init_asset::<StandardMaterial>()
            .init_asset::<GroundMaterial>()
            .insert_resource(GroundLook::Shipped)
            .add_systems(Update, apply_ground_look);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<GroundMaterial>>()
            .add(GroundMaterial {
                base: StandardMaterial::default(),
                extension: GroundDetail::new(GroundLook::Shipped, Handle::default(), Vec2::ONE),
            });

        let next = crate::next_view_variant(GroundLook::Shipped);
        assert_ne!(next, GroundLook::Shipped);
        *app.world_mut().resource_mut::<GroundLook>() = next;
        app.update();

        let materials = app.world().resource::<Assets<GroundMaterial>>();
        assert_eq!(materials.get(&handle).unwrap().extension.look, next);
    }

    /// The toggle must reach every look and come home — a partial cycle would
    /// strand variants no player could ever see.
    #[test]
    fn cycling_visits_every_look_then_returns() {
        let all = <GroundLook as clap::ValueEnum>::value_variants();
        let mut look = GroundLook::default();
        let mut seen = vec![look];
        for _ in 1..all.len() {
            look = crate::next_view_variant(look);
            seen.push(look);
        }
        assert_eq!(
            crate::next_view_variant(look),
            GroundLook::default(),
            "cycle does not close"
        );
        assert_eq!(seen, all);
    }
}
