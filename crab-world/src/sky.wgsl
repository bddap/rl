// Procedural night sky (bddap/rl#325), evaluated per fragment over the whole
// sphere — below the horizon too, so peeking past the terrain edge shows sky, not
// void. Replaces the CPU-baked cubemap: resolution-independent (a star stays crisp
// at any FOV or screen size), no texture memory, and animatable later via globals
// if the nocturne direction wants twinkle.
//
// One fullscreen triangle pinned to the far plane (reverse-Z: clip z = 0), the same
// trick as bevy's own skybox pass, so coverage is total by construction and every
// opaque fragment in front wins the depth test.
//
// Colors are authored in srgb (the constants match the old bake's, shared history
// with `sky.rs`'s HORIZON) and converted to linear at the end — the exact value the
// srgb cubemap sampler used to hand the pass, so fog and tonemapping see the same
// sky as before. The old bake also multiplied by Skybox brightness 1000 to cancel
// bevy's light pre-exposure; a material fragment is never pre-exposed, so that
// counterweight vanishes rather than being ported.

#import bevy_pbr::mesh_view_bindings::view
#import rl::noise::{hash3, rand01}

struct SkyVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> SkyVertexOutput {
    // Indices 0,1,2 -> (-1,-1), (3,-1), (-1,3): one triangle covering clip space.
    let xy = vec2(f32(vertex_index & 1u), f32((vertex_index >> 1u) & 1u)) * 4.0 - vec2(1.0);
    return SkyVertexOutput(vec4(xy, 0.0, 1.0), xy);
}

@fragment
fn fragment(in: SkyVertexOutput) -> @location(0) vec4<f32> {
    // z flipped to keep the shipped star layout: the old pass sampled its cubemap
    // left-handed (bevy negates z), so the baked sky appeared as sky_color(x,y,-z).
    let dir = ray_direction(in.ndc) * vec3(1.0, 1.0, -1.0);
    let c = base_gradient(dir) + milky_way(dir) + star(dir);
    return vec4(srgb_to_linear(clamp(c, vec3(0.0), vec3(1.0))), 1.0);
}

// The fragment's view-space position IS the ray direction (camera at the origin);
// w=0 through world_from_view rotates it to world space without translation.
fn ray_direction(ndc: vec2<f32>) -> vec3<f32> {
    let view_pos = view.view_from_clip * vec4(ndc, 1.0, 1.0);
    let world_dir = view.world_from_view * vec4(view_pos.xyz / view_pos.w, 0.0);
    return normalize(world_dir.xyz);
}

// Sky-gradient tone at the horizon. MUST match sky.rs's HORIZON: vista fog
// converges to that constant, and any drift re-opens the fog/sky seam this shader
// exists to close.
const HORIZON: vec3<f32> = vec3(0.078, 0.122, 0.235);
const ZENITH: vec3<f32> = vec3(0.024, 0.039, 0.110);

// Symmetric in y: straight down (past the terrain) reads like straight up, so the
// sky wraps with no authored floor (the owner's ask; supersedes rl#197's
// darkest-below-horizon rule, which dated from when below-horizon was void).
fn base_gradient(dir: vec3<f32>) -> vec3<f32> {
    return mix(HORIZON, ZENITH, smoothstep(0.0, 0.9, abs(dir.y)));
}

fn milky_way(dir: vec3<f32>) -> vec3<f32> {
    let pole = normalize(vec3(0.30, 0.86, -0.41));
    let core = 1.0 - smoothstep(0.0, 0.45, abs(dot(dir, pole)));
    let clouds = 0.4 + 0.6 * value_noise(dir * 7.0);
    return vec3(0.16, 0.17, 0.24) * (core * core * clouds);
}

fn star(dir: vec3<f32>) -> vec3<f32> {
    let g = dir * 90.0; // star grid over the unit sphere
    let cell = floor(g);
    let h = hash3(vec3<i32>(cell));
    if rand01(h) >= 0.12 { // density
        return vec3(0.0);
    }
    let center = vec3(
        0.34 + 0.32 * rand01(h ^ 0x9e3779b9u),
        0.34 + 0.32 * rand01(h ^ 0x85ebca6bu),
        0.34 + 0.32 * rand01(h ^ 0xc2b2ae35u),
    );
    let d = (g - cell) - center;
    let r = rand01(h ^ 0x27d4eb2fu);
    let bright = 0.5 + 0.8 * r * r * r;
    let size = 0.06 + 0.10 * r;
    let falloff = exp(-dot(d, d) / (size * size));
    if falloff < 0.015 {
        return vec3(0.0);
    }
    let warm = rand01(h ^ 0x165e67b1u);
    return vec3(0.9 + 0.1 * warm, 0.92, 1.0 - 0.12 * warm) * (bright * falloff);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let w = smoothstep(vec3(0.0), vec3(1.0), fract(p));
    let x00 = mix(corner(i, 0, 0, 0), corner(i, 1, 0, 0), w.x);
    let x10 = mix(corner(i, 0, 1, 0), corner(i, 1, 1, 0), w.x);
    let x01 = mix(corner(i, 0, 0, 1), corner(i, 1, 0, 1), w.x);
    let x11 = mix(corner(i, 0, 1, 1), corner(i, 1, 1, 1), w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

fn corner(i: vec3<i32>, dx: i32, dy: i32, dz: i32) -> f32 {
    return rand01(hash3(i + vec3(dx, dy, dz)));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
    return select(hi, lo, c <= vec3(0.04045));
}
