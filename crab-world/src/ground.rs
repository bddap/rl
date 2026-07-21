//! The ground's look (bddap/rl#304): a procedural detail layer over the terrain
//! mesh's macro biome tint. All detail is computed in the fragment shader from
//! WORLD-SPACE position — no repeated texture anywhere in the ground path, so there
//! is no tile period to spot at any play scale, and the mesh stays the collider's
//! exact geometry (render matches physics; beauty is shading only). The fine
//! octaves double as the rl#197 optic-flow cue the deleted checker used to carry:
//! high-frequency luminance detail at on-foot/landing scale, faded per-pixel by
//! screen footprint so altitude views get the macro layers without shimmer.

use bevy::asset::embedded_asset;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

/// The one ground material: stock PBR (lighting, shadows, fog, vertex biome tint)
/// with the procedural detail fragment on top.
pub type GroundMaterial = ExtendedMaterial<StandardMaterial, GroundDetail>;

/// Strength knobs for the shader's layers, one uniform so a taste iteration is a
/// constant tweak, not a shader rewrite. Defaults are the shipped look.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct GroundDetail {
    /// x: macro patchiness (hundreds of m), y: meso mottling (tens of m),
    /// z: fine on-foot detail, w: detail-normal strength.
    #[uniform(100)]
    pub strengths: Vec4,
}

impl Default for GroundDetail {
    fn default() -> Self {
        Self {
            strengths: Vec4::new(0.55, 0.35, 0.45, 0.6),
        }
    }
}

impl MaterialExtension for GroundDetail {
    fn fragment_shader() -> ShaderRef {
        "embedded://crab_world/ground.wgsl".into()
    }
}

/// Registers [`GroundMaterial`] + its embedded shader. Added by `ArenaVisualsPlugin`,
/// so every rendered surface gets it through the one arena path.
pub struct GroundMaterialPlugin;

impl Plugin for GroundMaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "ground.wgsl");
        app.add_plugins(MaterialPlugin::<GroundMaterial>::default());
    }
}

#[cfg(test)]
mod tests {
    /// The URI in [`super::GroundDetail::fragment_shader`] must match what
    /// `embedded_asset!` actually registers — a drifted path is a runtime
    /// "Path not found" that silently un-renders the ground.
    #[test]
    fn embedded_shader_path_matches_the_material() {
        let p = bevy::asset::embedded_path!("ground.wgsl");
        assert_eq!(p, std::path::PathBuf::from("crab_world/ground.wgsl"));
    }
}
