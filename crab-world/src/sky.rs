use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::Skybox;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};

const FACE: usize = 1024;

const SKY_BRIGHTNESS: f32 = 1000.0;

pub const NIGHT_CLEAR: Color = Color::srgb(0.02, 0.03, 0.09);

pub struct NightSkyPlugin;

impl Plugin for NightSkyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, generate_sky)
            .add_systems(Update, attach_skybox);
    }
}

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
            image: Some(sky.0.clone()),
            brightness: SKY_BRIGHTNESS,
            rotation: Quat::IDENTITY,
        });
    }
}

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
        TextureFormat::Rgba8UnormSrgb,
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

/// Below-horizon tone: near-black, darker than the zenith, so "down" is the darkest
/// direction in the sky and the horizon reads as a crisp bright→dark edge.
const BELOW_HORIZON: Vec3 = Vec3::new(0.008, 0.010, 0.022);

fn sky_color(dir: Vec3) -> [u8; 3] {
    let mut c = base_gradient(dir) + milky_way(dir);
    c += star(dir);
    c = c.lerp(BELOW_HORIZON, smoothstep(0.0, 0.05, -dir.y));
    [to_u8(c.x), to_u8(c.y), to_u8(c.z)]
}

fn base_gradient(dir: Vec3) -> Vec3 {
    let horizon = Vec3::new(0.078, 0.122, 0.235);
    let zenith = Vec3::new(0.024, 0.039, 0.110);
    let t = smoothstep(0.0, 0.9, dir.y.max(0.0));
    horizon.lerp(zenith, t)
}

fn milky_way(dir: Vec3) -> Vec3 {
    let pole = Vec3::new(0.30, 0.86, -0.41).normalize();
    let off = dir.dot(pole).abs();
    let core = 1.0 - smoothstep(0.0, 0.45, off);
    if core <= 0.0 {
        return Vec3::ZERO;
    }
    let clouds = 0.4 + 0.6 * value_noise(dir * 7.0);
    let tint = Vec3::new(0.16, 0.17, 0.24);
    tint * (core * core * clouds)
}

fn star(dir: Vec3) -> Vec3 {
    const STAR_GRID: f32 = 90.0;
    const DENSITY: f32 = 0.12;

    let g = dir * STAR_GRID;
    let cell = g.floor();
    let h = hash3(cell.x as i32, cell.y as i32, cell.z as i32);
    if rand01(h) >= DENSITY {
        return Vec3::ZERO;
    }
    let center = Vec3::new(
        0.34 + 0.32 * rand01(h ^ 0x9e37_79b9),
        0.34 + 0.32 * rand01(h ^ 0x85eb_ca6b),
        0.34 + 0.32 * rand01(h ^ 0xc2b2_ae35),
    );
    let d = (g - cell) - center;
    let r = rand01(h ^ 0x27d4_eb2f);
    let bright = 0.5 + 0.8 * r * r * r;
    let size = 0.06 + 0.10 * r;
    let falloff = (-(d.length_squared()) / (size * size)).exp();
    if falloff < 0.015 {
        return Vec3::ZERO;
    }
    let warm = rand01(h ^ 0x165e_67b1);
    let tint = Vec3::new(0.9 + 0.1 * warm, 0.92, 1.0 - 0.12 * warm);
    tint * (bright * falloff)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

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

fn hash3(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x1656_67b1);
    h ^= h >> 16;
    h
}

fn rand01(h: u32) -> f32 {
    (h & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_dirs_are_unit_and_oriented() {
        for f in 0..6 {
            let d = face_dir(f, FACE / 2, FACE / 2);
            assert!((d.length() - 1.0).abs() < 1e-4, "face {f} not unit: {d:?}");
        }
        assert!(face_dir(2, FACE / 2, FACE / 2).y > 0.99);
    }

    #[test]
    fn gradient_is_dark_and_top_heavy() {
        let zen = base_gradient(Vec3::Y);
        let hor = base_gradient(Vec3::X);
        assert!(zen.max_element() < 0.15, "zenith too bright: {zen:?}");
        assert!(hor.max_element() < 0.3, "horizon too bright: {hor:?}");
        assert!(
            zen.length() < hor.length(),
            "zenith should be darker than horizon"
        );
    }

    /// Below the horizon the FULL sky (gradient + band + stars) is darker than the
    /// zenith, so the horizon reads as a crisp edge and "starry" means up (rl#197).
    #[test]
    fn below_horizon_is_darkest() {
        let zenith = base_gradient(Vec3::Y).length();
        for dir in [Vec3::new(0.6, -0.3, 0.5).normalize(), Vec3::NEG_Y] {
            let [r, g, b] = sky_color(dir);
            let below = Vec3::new(r as f32, g as f32, b as f32) / 255.0;
            assert!(
                below.length() < zenith,
                "below-horizon sky should be darker than the zenith: {below:?}"
            );
        }
    }

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
