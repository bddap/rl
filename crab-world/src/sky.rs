use bevy::asset::{RenderAssetUsages, load_internal_asset, uuid_handle};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::PrimitiveTopology;
use bevy::pbr::{Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

pub const NIGHT_CLEAR: Color = Color::srgb(0.02, 0.03, 0.09);

/// Procedural night sky (rl#325): one fullscreen triangle whose fragment shader
/// (`sky.wgsl`) evaluates the sky per pixel over the whole sphere — full wrap below
/// the horizon, resolution-independent stars, no cubemap bake or texture memory.
/// The look's math lives in the shader; the Rust side only spawns the triangle.
///
/// Carries the moon's boot knobs as a plugin field (same idiom as
/// `ArenaWorldPlugin { ground_look }`): a surface cannot get a sky without
/// stating its moon, so the disc and its light can never be forgotten apart.
pub struct NightSkyPlugin {
    pub moon: crate::moon::Moon,
}

const SKY_SHADER: Handle<Shader> = uuid_handle!("4f6b0e29-8d17-4c52-b3a9-7e04d1c68f5a");

impl Plugin for NightSkyPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SKY_SHADER, "sky.wgsl", Shader::from_wgsl);
        app.add_plugins(crate::moon::MoonPlugin { moon: self.moon })
            .add_plugins(MaterialPlugin::<StarSkyMaterial>::default())
            .add_systems(Startup, spawn_sky)
            .add_systems(Update, sync_moon_uniforms);
    }
}

/// Uniforms carry the [`crate::moon::Moon`] knobs to the shader. Colors are
/// authored srgb like every constant in `sky.wgsl` (the shader converts to
/// linear at the end).
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct StarSkyMaterial {
    /// xyz: world-space unit vector toward the moon; w: phase [0, 1).
    #[uniform(0)]
    moon_dir_phase: Vec4,
    /// rgb: moon tint (srgb); w: illuminated fraction (halo strength).
    #[uniform(1)]
    moon_color_frac: Vec4,
}

impl StarSkyMaterial {
    fn from_moon(moon: &crate::moon::Moon) -> Self {
        let c = moon.color().to_srgba();
        Self {
            moon_dir_phase: moon.direction().extend(moon.phase.rem_euclid(1.0)),
            moon_color_frac: Vec4::new(c.red, c.green, c.blue, moon.illuminated_fraction()),
        }
    }
}

/// Follow the moon knobs: any change to the resource re-uploads the sky's
/// uniforms next frame (same change-gating as the light sync in `moon.rs`).
fn sync_moon_uniforms(
    moon: Res<crate::moon::Moon>,
    mut materials: ResMut<Assets<StarSkyMaterial>>,
) {
    if !moon.is_changed() {
        return;
    }
    for (_, material) in materials.iter_mut() {
        *material = StarSkyMaterial::from_moon(&moon);
    }
}

impl Material for StarSkyMaterial {
    fn vertex_shader() -> ShaderRef {
        SKY_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SKY_SHADER.into()
    }
}

/// The mesh is three placeholder vertices: the vertex stage synthesizes the
/// fullscreen triangle from `vertex_index` alone, so the entity needs
/// [`NoFrustumCulling`] (its AABB is degenerate) and covers every 3D camera's view
/// wherever it "sits".
fn spawn_sky(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StarSkyMaterial>>,
    moon: Res<crate::moon::Moon>,
) {
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; 3]);
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StarSkyMaterial::from_moon(&moon))),
        NoFrustumCulling,
        NotShadowCaster,
    ));
}

/// Sky-gradient tone at the horizon (srgb). The shader's `HORIZON` must match: this
/// constant is what vista fog converges to, so terrain dissolves into the sky
/// ([`horizon_fog_color`]).
const HORIZON: Vec3 = Vec3::new(0.078, 0.122, 0.235);

/// The color the sky RENDERS at the horizon, for camera fog: distant geometry then
/// fades INTO the sky rather than into a mismatched haze (rl#281 stage 3 — it is
/// also what hides the baked tile's hard edge). Plain srgb, no HDR scaling: bevy
/// pre-exposes lights and the fog mix runs on those exposed values, so
/// `DistanceFog.color` is display-relative — and the sky shader writes its authored
/// srgb value straight to the pass (converted to linear, never pre-exposed), so the
/// horizon fragment lands on screen at ~this value.
pub(crate) fn horizon_fog_color() -> Color {
    Color::srgb(HORIZON.x, HORIZON.y, HORIZON.z)
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn hash3(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x1656_67b1);
    h ^= h >> 16;
    h
}

pub(crate) fn rand01(h: u32) -> f32 {
    (h & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader authors in srgb and fog must converge to the same horizon tone;
    /// a drifted constant re-opens the fog/sky seam (rl#281). Guards the Rust side
    /// of the pair; the WGSL side carries the matching constant.
    #[test]
    fn fog_color_is_the_horizon_constant() {
        let Color::Srgba(c) = horizon_fog_color() else {
            panic!("horizon fog should be srgb-constructed");
        };
        assert_eq!([c.red, c.green, c.blue], [HORIZON.x, HORIZON.y, HORIZON.z]);
    }

    /// sky.wgsl must keep the fog-matching horizon constant verbatim — the two
    /// sides of the seam live in two languages, so the test greps the shader
    /// source rather than trusting a comment to keep them aligned.
    #[test]
    fn shader_horizon_matches_rust() {
        let wgsl = include_str!("sky.wgsl");
        assert!(
            wgsl.contains("vec3(0.078, 0.122, 0.235)"),
            "sky.wgsl HORIZON no longer matches sky.rs HORIZON"
        );
        for entry in ["fn vertex(", "fn fragment("] {
            assert!(wgsl.contains(entry), "sky.wgsl lost entry point {entry}");
        }
    }
}
