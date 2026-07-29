//! The ground's look (bddap/rl#304): a procedural detail layer over the terrain
//! mesh's macro biome tint. All detail is computed in the fragment shader from
//! WORLD-SPACE position — no repeated texture anywhere in the ground path, so there
//! is no tile period to spot at any play scale, and the mesh stays the collider's
//! exact geometry (render matches physics; beauty is shading only). The fine
//! octaves double as the rl#197 optic-flow cue the deleted checker used to carry:
//! high-frequency luminance detail at on-foot/landing scale, faded per-pixel by
//! screen footprint so altitude views get the macro layers without shimmer.
//!
//! Seven looks ship, swapped at runtime with no effect on physics or eval: the swap
//! reaches only the fragment shader the ground pipeline is specialized with.

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

/// One interchangeable ground look — the shipped one plus the six rl#304
/// competition entries — and the live selection: `GroundLook` is BOTH the
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
///
/// Rules (issue requirements, not style):
/// - entry point stays `fn fragment`. Only the FORWARD main pass draws these files
///   ([`GroundDetail::specialize`] leaves the prepass alone), so the
///   `PREPASS_PIPELINE` fork some of them carry is inert — it just keeps the file
///   valid if prepass wiring is ever added.
/// - world-space procedural only: no sampled textures (tiling), and nothing that
///   lies about geometry — normal perturbation is fine, parallax/displacement is
///   NOT (the visual surface IS the collision heightfield).
/// - judge colors at rendered exposure (moon-sun + ambient are pre-exposed).
///
/// To add a look: a file in `src/ground_looks/`, a variant here, and a row in
/// [`ground_looks!`] — which is exhaustive over this enum, so a variant with no row
/// is a compile error rather than a look nobody can reach.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, clap::ValueEnum)]
pub enum GroundLook {
    #[default]
    Shipped,
    CarvedStrata,
    NightBloom,
    PatternedGround,
    WindCombed,
    CrackedLoam,
    WetNocturne,
}

/// The one place a look's three facts live together — its shader UUID, its file,
/// and its on-screen name. Splitting them across a `match` and seven registration
/// calls is how a look ends up drawing its neighbour's shader with every test still
/// green.
macro_rules! ground_looks {
    ($($look:ident => $file:literal, $uuid:literal, $label:literal;)*) => {
        impl GroundLook {
            /// The look's shader. A const UUID handle rather than a loaded path
            /// because [`MaterialExtension::specialize`] — the only place that can
            /// repoint a pipeline's fragment stage — gets no world access to
            /// resolve one. Valid only once [`GroundMaterialPlugin`] has registered
            /// it, hence private.
            fn shader(self) -> Handle<Shader> {
                match self {
                    $(GroundLook::$look => uuid_handle!($uuid),)*
                }
            }

            /// What the on-screen toggle readout says.
            pub fn label(self) -> &'static str {
                match self {
                    $(GroundLook::$look => $label,)*
                }
            }
        }

        /// Compiled in, not loaded: `include_str!` makes a missing or misnamed look
        /// a BUILD error instead of a runtime "path not found" that un-renders the
        /// ground.
        fn register_look_shaders(app: &mut App) {
            $(load_internal_asset!(
                app,
                GroundLook::$look.shader(),
                concat!("ground_looks/", $file, ".wgsl"),
                Shader::from_wgsl
            );)*
        }
    };
}

ground_looks! {
    Shipped => "shipped", "9f2a7c14-3b6d-4e58-9a01-5c7d2e8f4b60", "Shipped";
    CarvedStrata => "carved_strata", "1c4e8a92-77b5-4d03-8fa6-2e91b0c5d7a3", "Carved strata";
    NightBloom => "night_bloom", "6b0d3f57-92ac-41e8-b74d-0a3e5c81f296", "Night-bloom";
    PatternedGround => "patterned_ground", "e83c1a06-5d4f-4b29-97e0-6c2b8d70a415", "Patterned ground";
    WindCombed => "wind_combed", "2d795be8-0c31-4a76-8b5e-91f4a03d6c27", "Wind-combed";
    CrackedLoam => "cracked_loam", "48a6f2d1-b90e-4c53-a812-7d05e9b34f6a", "Cracked loam & cobble";
    WetNocturne => "wet_nocturne", "7fe15b83-2a4c-49d0-91b6-c3820e5a7d14", "Wet nocturne";
}

impl GroundLook {
    /// Cycle order is DECLARATION order, straight off the enum — clap already
    /// maintains that list for `--ground-look`, so there is no second one to forget
    /// a variant in.
    fn all() -> &'static [Self] {
        <Self as clap::ValueEnum>::value_variants()
    }

    /// Advance one step and report where we landed.
    pub fn cycle(&mut self) -> Self {
        let all = Self::all();
        let i = all
            .iter()
            .position(|l| l == self)
            .expect("every variant is one of clap's value variants");
        *self = all[(i + 1) % all.len()];
        *self
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
    /// Which look this material draws. Not a binding — it rides in the material to
    /// reach [`MaterialExtension::specialize`] as the pipeline key, so two ground
    /// materials share a pipeline iff they draw the same look and a swap is a
    /// re-specialization rather than a second material type.
    pub look: GroundLook,
}

impl GroundDetail {
    pub fn new(look: GroundLook) -> Self {
        Self {
            strengths: Vec4::new(0.55, 0.35, 0.45, 0.6),
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
        if fragment.shader_defs.contains(&"PREPASS_PIPELINE".into()) {
            return Ok(());
        }
        fragment.shader = key.bind_group_data.shader();
        Ok(())
    }
}

/// Repoint every ground material at the selected look. Shading only — no physics,
/// mesh, or eval input is touched, by construction.
fn apply_ground_look(look: Res<GroundLook>, mut materials: ResMut<Assets<GroundMaterial>>) {
    if !look.is_changed() {
        return;
    }
    for (_, material) in materials.iter_mut() {
        material.extension.look = *look;
    }
}

/// How long the look's name stays up after a toggle. Long enough to read while
/// cycling, short enough to be gone from the frame you were judging.
const LOOK_READOUT_SECS: f32 = 2.5;

/// The transient "Ground: <name>" readout, sitting one row above the render-mode
/// label it shares a corner with.
#[derive(Component)]
struct GroundLookReadout(Timer);

fn spawn_look_readout(mut commands: Commands) {
    commands.spawn((
        crate::crab_view::hud_corner_label(1),
        GroundLookReadout(Timer::from_seconds(LOOK_READOUT_SECS, TimerMode::Once)),
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
        **text = format!("Ground: {}", look.label());
        state.0.reset();
    }
    if !state.0.is_finished() && state.0.tick(time.delta()).just_finished() {
        text.clear();
    }
}

/// Registers [`GroundMaterial`], the seven embedded looks, and the swap system.
/// Added by `ArenaVisualsPlugin`, so every rendered surface gets it through the one
/// arena path. `initial` is the look the app boots in (`--ground-look`).
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

    /// A copy-pasted UUID would make two looks the same shader — the toggle would
    /// still "work" and the player would just never see one of them.
    #[test]
    fn every_look_has_its_own_shader() {
        let mut ids: Vec<_> = GroundLook::all().iter().map(|l| l.shader().id()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), GroundLook::all().len());
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
                extension: GroundDetail::new(GroundLook::Shipped),
            });

        let next = app.world_mut().resource_mut::<GroundLook>().cycle();
        assert_ne!(next, GroundLook::Shipped);
        app.update();

        let materials = app.world().resource::<Assets<GroundMaterial>>();
        assert_eq!(materials.get(&handle).unwrap().extension.look, next);
    }

    /// The toggle must reach every look and come home — a partial cycle would
    /// strand variants no player could ever see.
    #[test]
    fn cycling_visits_every_look_then_returns() {
        let mut look = GroundLook::default();
        let mut seen = vec![look];
        for _ in 1..GroundLook::all().len() {
            seen.push(look.cycle());
        }
        assert_eq!(look.cycle(), GroundLook::default(), "cycle does not close");
        assert_eq!(seen, GroundLook::all());
    }
}
