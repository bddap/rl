//! Procedural night-sky skybox, shared by both rendered surfaces — the rl-demo
//! (orbit + screenshot cameras) and Giant Crab Rescue (first-person + screenshot
//! cameras).
//!
//! The cubemap is GENERATED in code, not loaded: a deep blue→near-black vertical
//! gradient, a faint tilted Milky-Way band, and hash-scattered stars of varied
//! brightness and tint. Because the art is ours it ships in the binary under rl's own
//! MIT/Apache license with no third-party-asset question (the owner's "generate
//! something" path — no redistribution doubt).
//!
//! Cheap: bevy's [`Skybox`] is one fullscreen pass behind everything (drawn at infinite
//! depth, so no far-plane / geometry cost) — it does not light the scene (that's
//! `EnvironmentMapLight`), only paints the background. [`NightSkyPlugin`] generates the
//! cubemap once and attaches the SAME handle to every `Camera3d` as it spawns, so all
//! four cameras across the two surfaces share one sky with no per-camera drift.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::Skybox;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};

/// Edge length (px) of one cubemap face. 1024 keeps stars crisp filling a full-screen
/// background; the one-time generate is 6·1024² texels of integer hashing (~tens of ms).
const FACE: usize = 1024;

/// `Skybox.brightness` (cd/m²). The skybox shader emits `sample.rgb * brightness *
/// exposure`, where `exposure` is the camera's (here the default Blender EV100, ≈ 1/998).
/// The cubemap already bakes final night colors in [0,1], so we set brightness to cancel
/// that exposure and reproduce them as-authored on the untonemapped screenshot cameras
/// (the owner's evidence path) — without the cancellation a brightness of 1.0 is crushed
/// to near-black. The windowed client's tonemapper then maps those same [0,1] values to a
/// dim night sky.
const SKY_BRIGHTNESS: f32 = 1000.0;

/// Clear color the cameras keep behind the skybox: visible only in the brief window
/// before the cubemap finishes uploading to the GPU (or if the skybox were ever absent).
/// A dark night tone near the gradient's dark end, so there's no bright-blue flash under
/// the night sky — the skybox, once uploaded, is the real background.
pub const NIGHT_CLEAR: Color = Color::srgb(0.02, 0.03, 0.09);

/// Bevy plugin: generate the night-sky cubemap once at startup and attach a [`Skybox`]
/// to every `Camera3d` that lacks one. Added by both surfaces' app builders; the
/// per-frame attach also covers cameras spawned after startup (GCR spawns its FP camera
/// on entering `Playing`).
pub struct NightSkyPlugin;

impl Plugin for NightSkyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, generate_sky)
            .add_systems(Update, attach_skybox);
    }
}

/// Handle to the generated cubemap, shared by every camera's [`Skybox`].
#[derive(Resource)]
struct NightSky(Handle<Image>);

fn generate_sky(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let handle = night_sky_cubemap(&mut images);
    commands.insert_resource(NightSky(handle));
}

fn attach_skybox(
    mut commands: Commands,
    sky: Option<Res<NightSky>>,
    cams: Query<Entity, (With<Camera3d>, Without<Skybox>)>,
) {
    let Some(sky) = sky else { return };
    for cam in &cams {
        commands.entity(cam).insert(Skybox {
            image: sky.0.clone(),
            brightness: SKY_BRIGHTNESS,
            rotation: Quat::IDENTITY,
        });
    }
}

/// Build the night-sky cubemap as a bevy cube [`Image`]. Every texel's color is a pure
/// function of its 3D ray direction ([`sky_color`]), so the six faces meet seamlessly
/// with no per-face orientation bookkeeping — only [`face_dir`]'s standard cube layout.
pub fn night_sky_cubemap(images: &mut Assets<Image>) -> Handle<Image> {
    let mut data = vec![0u8; FACE * FACE * 6 * 4];
    for f in 0..6 {
        for y in 0..FACE {
            for x in 0..FACE {
                let [r, g, b] = sky_color(face_dir(f, x, y));
                let i = ((f * FACE + y) * FACE + x) * 4;
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                data[i + 3] = 255;
            }
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: FACE as u32,
            height: (FACE * 6) as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        // Authored bytes are sRGB; the shader gets linear samples, which is what the
        // gradient/star math below is tuned against.
        TextureFormat::Rgba8UnormSrgb,
        // GPU-only: the sky never changes and nothing in the main world reads it back.
        RenderAssetUsages::RENDER_WORLD,
    );
    image
        .reinterpret_stacked_2d_as_array(6)
        .expect("6 equal-height stacked faces");
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    images.add(image)
}

/// The outward ray direction for texel `(x, y)` of cube face `f`, in the standard cube
/// layout (face order +X, −X, +Y, −Y, +Z, −Z). The exact handedness is irrelevant to
/// the look — every color is a function of this normalized direction, so faces are
/// seamless regardless — only that +Y maps to the zenith (top of the gradient).
fn face_dir(f: usize, x: usize, y: usize) -> Vec3 {
    let u = 2.0 * (x as f32 + 0.5) / FACE as f32 - 1.0;
    let v = 2.0 * (y as f32 + 0.5) / FACE as f32 - 1.0;
    match f {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        _ => Vec3::new(-u, -v, -1.0),
    }
    .normalize()
}

/// The final sRGB color of the sky in direction `dir`: base gradient + Milky-Way glow,
/// then any star on top.
fn sky_color(dir: Vec3) -> [u8; 3] {
    let mut c = base_gradient(dir) + milky_way(dir);
    c += star(dir);
    [to_u8(c.x), to_u8(c.y), to_u8(c.z)]
}

/// Deep-blue→dark-navy vertical gradient: lightest near the horizon, darkest at the
/// zenith — the usual night look (airglow lifts the horizon a touch). Values are sRGB
/// 0..1; the screenshot cameras are untonemapped so these bytes show as-is. Kept dark
/// enough to read as night but blue enough not to be a black void. Below the horizon
/// stays at the horizon tone; it's hidden by the ground anyway.
fn base_gradient(dir: Vec3) -> Vec3 {
    let horizon = Vec3::new(0.078, 0.122, 0.235); // ~(20,31,60)
    let zenith = Vec3::new(0.024, 0.039, 0.110); //  ~(6,10,28)
    let t = smoothstep(0.0, 0.9, dir.y.max(0.0));
    horizon.lerp(zenith, t)
}

/// A faint, patchy Milky-Way band on a tilted great circle, so it arcs across the sky
/// rather than ringing the horizon. Brightest on the band plane, fading out within
/// ~25°; low-frequency value noise breaks it into clouds so it doesn't read as a
/// painted stripe. Deliberately subtle — a glow, not a feature.
fn milky_way(dir: Vec3) -> Vec3 {
    // Pole of the band plane; the band itself is where dir ⟂ pole (dir·pole ≈ 0).
    let pole = Vec3::new(0.30, 0.86, -0.41).normalize();
    let off = dir.dot(pole).abs();
    let core = 1.0 - smoothstep(0.0, 0.45, off);
    if core <= 0.0 {
        return Vec3::ZERO;
    }
    let clouds = 0.4 + 0.6 * value_noise(dir * 7.0);
    let tint = Vec3::new(0.16, 0.17, 0.24); // dusty blue-white
    tint * (core * core * clouds)
}

/// A star at `dir`, or zero. Direction space is partitioned into a cube lattice
/// ([`STAR_GRID`] cells across the unit diameter); each cell deterministically holds at
/// most one star, placed safely inside the cell so its small footprint never crosses a
/// boundary (so checking only the home cell is exact). Brightness, size, and a slight
/// warm/cool tint are hashed per cell, with a few rare bright stars among many faint
/// ones. sRGB additive on top of the gradient.
fn star(dir: Vec3) -> Vec3 {
    // Coarse enough that a cell spans several texels: at 1024 px/face a cell is ~7 px,
    // so a star's footprint lands on multiple pixels instead of vanishing sub-pixel (the
    // bug a finer grid hid). ~12% of cells hold a star → a rich but not noisy field.
    const STAR_GRID: f32 = 90.0;
    const DENSITY: f32 = 0.12;

    let g = dir * STAR_GRID;
    let cell = g.floor();
    let h = hash3(cell.x as i32, cell.y as i32, cell.z as i32);
    if rand01(h) >= DENSITY {
        return Vec3::ZERO;
    }
    // Star center kept inside the cell (0.34..0.66), so its footprint stays in the home
    // cell and checking only that cell is exact.
    let center = Vec3::new(
        0.34 + 0.32 * rand01(h ^ 0x9e37_79b9),
        0.34 + 0.32 * rand01(h ^ 0x85eb_ca6b),
        0.34 + 0.32 * rand01(h ^ 0xc2b2_ae35),
    );
    let d = (g - cell) - center;
    // Most stars small/faint; a rare few large and bright (the ^3 skews low). `bright`
    // can exceed 1 so the brightest stars clamp to a pure-white core.
    let r = rand01(h ^ 0x27d4_eb2f);
    let bright = 0.5 + 0.8 * r * r * r;
    let size = 0.06 + 0.10 * r;
    let falloff = (-(d.length_squared()) / (size * size)).exp();
    if falloff < 0.015 {
        return Vec3::ZERO;
    }
    // Slight tint: warm (orange-ish) ↔ cool (blue-white) by another hash.
    let warm = rand01(h ^ 0x165e_67b1);
    let tint = Vec3::new(0.9 + 0.1 * warm, 0.92, 1.0 - 0.12 * warm);
    tint * (bright * falloff)
}

/// Smooth Hermite interpolation of `x` across `[edge0, edge1]`, clamped to `[0,1]`.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Cheap trilinearly-interpolated value noise in [0,1] over a 3D point — lattice corners
/// hashed to values, smoothstepped between. Used only to make the Milky Way patchy.
fn value_noise(p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let w = Vec3::new(
        smoothstep(0.0, 1.0, f.x),
        smoothstep(0.0, 1.0, f.y),
        smoothstep(0.0, 1.0, f.z),
    );
    let corner = |dx: i32, dy: i32, dz: i32| {
        rand01(hash3(i.x as i32 + dx, i.y as i32 + dy, i.z as i32 + dz))
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), w.x);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), w.x);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), w.x);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), w.x);
    let y0 = lerp(x00, x10, w.y);
    let y1 = lerp(x01, x11, w.y);
    lerp(y0, y1, w.z)
}

/// A stable integer hash of a 3D lattice cell — deterministic, so the sky is identical
/// every launch (no `rand`).
fn hash3(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32)
        .wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x1656_67b1);
    h ^= h >> 16;
    h
}

/// Map a hash to a float in [0,1).
fn rand01(h: u32) -> f32 {
    (h & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}

/// Clamp a linear-ish sRGB component in [0,1] to a byte.
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Face directions are unit length and the six faces tile all axes (the +Y face
    /// centre points to the zenith, the gradient's dark end).
    #[test]
    fn face_dirs_are_unit_and_oriented() {
        for f in 0..6 {
            let d = face_dir(f, FACE / 2, FACE / 2);
            assert!((d.length() - 1.0).abs() < 1e-4, "face {f} not unit: {d:?}");
        }
        // Centre of face 2 (+Y) is the zenith.
        assert!(face_dir(2, FACE / 2, FACE / 2).y > 0.99);
    }

    /// The base sky is dark everywhere (a night sky), and the zenith is darker than the
    /// horizon.
    #[test]
    fn gradient_is_dark_and_top_heavy() {
        let zen = base_gradient(Vec3::Y);
        let hor = base_gradient(Vec3::X);
        // Still firmly a night sky (the blue channel, the brightest, stays well below
        // mid), but bright enough to read as deep blue rather than a black void.
        assert!(zen.max_element() < 0.15, "zenith too bright: {zen:?}");
        assert!(hor.max_element() < 0.3, "horizon too bright: {hor:?}");
        assert!(zen.length() < hor.length(), "zenith should be darker than horizon");
    }

    /// Stars are sparse: only a small fraction of sampled directions light one up, so
    /// the sky reads as scattered points, not noise.
    #[test]
    fn stars_are_sparse() {
        let mut lit = 0;
        let n = 40;
        for f in 0..6 {
            for y in 0..n {
                for x in 0..n {
                    let dir = face_dir(f, x * FACE / n, y * FACE / n);
                    if star(dir).length() > 0.0 {
                        lit += 1;
                    }
                }
            }
        }
        let frac = lit as f32 / (6 * n * n) as f32;
        assert!(frac > 0.0 && frac < 0.25, "star coverage off: {frac}");
    }
}
