// Procedural ground detail (bddap/rl#304) over the terrain mesh's vertex biome
// tint. Everything is derived from WORLD-SPACE position — no sampled texture, so
// no repeat period exists to spot from any altitude. Octaves are faded by their
// on-screen footprint (fwidth): the procedural analogue of mipmapping, so fine
// detail exists on foot and at landing height (the rl#197 optic-flow duty the old
// checker carried) but never shimmers from the plane.
//
// One of the interchangeable looks in this directory; the contract every
// file here keeps — inputs, binding 100, the `fragment` entry point — is
// documented once on `GroundLook` in ground.rs.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

// x: macro patchiness, y: meso mottling, z: fine on-foot detail, w: detail normal.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> strengths: vec4<f32>;

// Same integer-hash family as the Rust side's sky/terrain jitter (sky.rs hash3).
fn hash2(p: vec2<i32>, seed: u32) -> u32 {
    var h = bitcast<u32>(p.x) * 0x8da6b343u ^ bitcast<u32>(p.y) * 0xd8163841u ^ seed * 0xcb1ab31fu;
    h = h ^ (h >> 13u);
    h = h * 0x165667b1u;
    return h ^ (h >> 16u);
}

fn rand01(h: u32) -> f32 {
    return f32(h & 0xffffffu) / f32(0x1000000u);
}

// Value noise in [-1, 1]. Quintic (C2) interpolant: smoothstep's discontinuous
// second derivative creases along every lattice edge, and the rl#324 close-up
// octaves are large enough on screen to read those creases as a woven grid.
fn vnoise(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = rand01(hash2(i, seed));
    let b = rand01(hash2(i + vec2(1, 0), seed));
    let c = rand01(hash2(i + vec2(0, 1), seed));
    let d = rand01(hash2(i + vec2(1, 1), seed));
    return 2.0 * mix(mix(a, b, w.x), mix(c, d, w.x), w.y) - 1.0;
}

// 1 while the octave's wavelength spans many pixels, 0 once it is subpixel.
fn footprint_fade(wavelength: f32, fw: f32) -> f32 {
    return 1.0 - smoothstep(wavelength * 0.15, wavelength * 0.5, fw);
}

// The adaptive-descent fade (rl#324): wider margins than footprint_fade, so the
// finest octave the descent keeps still spans ~8+ pixels per wavelength. Letting
// octaves ride down toward Nyquist (footprint_fade's window) turns the fine tail
// into per-pixel salt noise — grain must stay coarse enough to read as surface.
fn grain_fade(wavelength: f32, fw: f32) -> f32 {
    return 1.0 - smoothstep(wavelength * 0.05, wavelength * 0.125, fw);
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let wp = in.world_position.xyz;
    let p = wp.xz;
    // Ground meters per pixel at this fragment — the octave-fade driver.
    let fw = max(max(fwidth(p.x), fwidth(p.y)), 1e-4);

    var rgb = pbr_input.material.base_color.rgb;

    // Vegetation mask from the biome tint's greenness: full patchiness on grass,
    // muted on scree/rock/snow (mineral ground varies less than growth does).
    let veg = clamp((rgb.g - max(rgb.r, rgb.b)) * 6.0, 0.0, 1.0);

    // Macro patchiness (hundreds of meters): kills the banded-paint read from the
    // plane. A warm/cool hue drift, not just value, so patches look like different
    // growth and soil rather than shadow.
    let macro_n = vnoise(p / 620.0, 11u) * 0.5
        + vnoise(p / 210.0, 12u) * 0.35
        + vnoise(p / 90.0, 13u) * 0.15;
    let warm = macro_n * strengths.x * mix(0.4, 1.0, veg);
    rgb *= vec3(1.0 + 0.25 * warm, 1.0 + 0.05 * warm, 1.0 - 0.18 * warm);
    rgb *= 1.0 + 0.50 * warm;

    // Meso mottling (tens of meters): the mid-range octave gap between biome bands
    // and on-foot detail.
    let meso_n = vnoise(p / 26.0, 21u) * footprint_fade(26.0, fw)
        + vnoise(p / 9.0, 22u) * 0.7 * footprint_fade(9.0, fw);
    rgb *= 1.0 + 0.35 * strengths.y * meso_n;

    // Fine detail (meters and below): the on-foot / landing-height optic-flow cue.
    // ADAPTIVE descent (rl#324): a fixed finest octave IS a resolution — its lattice
    // reads as an upscaled low-res texture the moment the camera gets closer than it
    // was tuned for. So descend by thirds from 2.6 m until the octave is subpixel
    // (footprint-faded to nothing), down to a 3.5 mm floor: finer than any standing
    // eye can resolve (~2 mm ground per pixel on foot at 720p) yet coarse enough
    // that the f32 ulp at the ±15 km map corners (~1 mm) never dominates a lattice
    // cell. Distant ground exits after one test — fw alone decides, so the loop is
    // quad-coherent and the near-fullscreen far ground pays nothing.
    // Each octave rotated 55.62° (90°/φ) from the last: value noise's lattice is
    // axis-aligned, and any octave landing near a multiple of 90° reads as a woven
    // plaid — 90°/φ keeps every small multiple ≥13° away from the axes (the golden
    // angle itself fails: 2×137.5° mod 90° = 5°, a visibly axis-aligned octave).
    let rot = mat2x2<f32>(0.5646, 0.8254, -0.8254, 0.5646);
    var fine_n = 0.0;
    var q = p;
    var wl = 2.6;
    var amp = 1.0;
    var seed = 31u;
    while wl > 0.0035 {
        let fade = grain_fade(wl, fw);
        if fade <= 0.001 {
            break;
        }
        fine_n += vnoise(q / wl, seed) * amp * fade;
        q = rot * q;
        wl /= 3.0;
        amp *= 0.72;
        seed += 1u;
    }
    rgb *= 1.0 + 0.30 * strengths.z * fine_n;

    // Grass clumps: darker tufted patches where the ground is vegetated.
    let tuft = smoothstep(0.15, 0.75, vnoise(p / 1.4, 34u)) * veg * footprint_fade(1.4, fw);
    rgb *= 1.0 - 0.30 * strengths.z * tuft;

    // Sedimentary strata on steep faces: elevation-banded value variation, so
    // cliffs read as layered rock instead of smeared vertex tint.
    let n_geo = normalize(in.world_normal);
    let steep = 1.0 - n_geo.y;
    let strata_mask = smoothstep(0.25, 0.55, steep);
    let strata = vnoise(vec2(wp.y / 7.0, (p.x + p.y) * 0.012), 41u);
    rgb *= 1.0 + 0.35 * strata_mask * strata * footprint_fade(7.0, max(fwidth(wp.y), 1e-4));

    pbr_input.material.base_color = vec4(rgb, pbr_input.material.base_color.a);

    // Detail normal from the fine-noise heightfield gradient: micro-relief up
    // close, faded with the same footprint rule so distant ground keeps the smooth
    // geometric normal. Same adaptive descent as the color detail (rl#324): a
    // relief lattice that stops at a fixed wavelength is the "low-res normal map"
    // the eye catches on foot, so descend by thirds from 45 cm until subpixel
    // (same 3.5 mm floor / f32-ulp rationale as above). The finite-difference step
    // scales with the wavelength so each octave's gradient stays honest; per-octave
    // gradient weight falls slower than the color amplitude (relief keeps more of
    // its punch as it refines — pebbles under moonlight, not blur).
    if strengths.w * footprint_fade(0.45, fw) > 0.001 {
        var grad = vec2(0.0);
        // Cumulative rotation for this octave's lattice; its transpose (== inverse)
        // carries the octave's gradient back to world space.
        var nrot = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
        var nwl = 0.45;
        var nweight = 0.06;
        var nseed = 51u;
        while nwl > 0.0035 {
            let fade = grain_fade(nwl, fw);
            if fade <= 0.001 {
                break;
            }
            let nq = nrot * p;
            let step = nwl * 0.27;
            let h0 = vnoise(nq / nwl, nseed);
            let hx = vnoise((nq + vec2(step, 0.0)) / nwl, nseed);
            let hz = vnoise((nq + vec2(0.0, step)) / nwl, nseed);
            grad += fade * (transpose(nrot) * vec2(hx - h0, hz - h0)) / step * nweight;
            nrot = rot * nrot;
            nwl /= 3.0;
            nweight *= 0.45;
            nseed += 1u;
        }
        // Soft-limit the stacked slope: unbounded octave sums tilt the normal past
        // grazing and light it as black speckle pits.
        grad *= 1.0 / (1.0 + 0.8 * length(grad));
        pbr_input.N = normalize(pbr_input.N + strengths.w * vec3(-grad.x, 0.0, -grad.y));
    }

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif
    return out;
}
