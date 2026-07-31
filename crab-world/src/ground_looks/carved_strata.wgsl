// Ground look: CARVED STRATA (bddap/rl#304 ground-shader competition, fable-1).
// The whole landscape reads as banded sedimentary rock — domain-warped strata
// keyed to ELEVATION, so from plane altitude the bands hug the topography like
// the contour art of a geological map, on slopes they read as exposed rock
// layers, and on foot they become mineral grain and gravel flecks. Vegetated
// valley floors keep their moonlit green; the mineral world shows through it
// where growth thins. Everything is derived from WORLD-SPACE position — no
// sampled texture, so no repeat period exists to spot from any altitude.
// Octaves are faded by their on-screen footprint (fwidth), so fine detail
// exists on foot but never shimmers from the plane.
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

// x: strata banding, y: macro patchiness, z: fine grain, w: detail normal.
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

// Value noise in [-1, 1], C1-smooth.
fn vnoise(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
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

// The strata palette, cyclic: rust → ochre → bone → slate-violet → rust. Linear
// RGB, chosen dim enough that moon-sun + ambient exposure lands them as deep
// mineral color, not paint.
fn strata_color(t: f32) -> vec3<f32> {
    let rust = vec3(0.44, 0.15, 0.07);
    let ochre = vec3(0.58, 0.35, 0.11);
    let bone = vec3(0.62, 0.54, 0.40);
    let slate = vec3(0.13, 0.12, 0.24);
    let u = fract(t) * 4.0;
    if u < 1.0 {
        return mix(rust, ochre, smoothstep(0.0, 1.0, u));
    } else if u < 2.0 {
        return mix(ochre, bone, smoothstep(0.0, 1.0, u - 1.0));
    } else if u < 3.0 {
        return mix(bone, slate, smoothstep(0.0, 1.0, u - 2.0));
    }
    return mix(slate, rust, smoothstep(0.0, 1.0, u - 3.0));
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

    let base = pbr_input.material.base_color.rgb;
    var rgb = base;

    // Vegetation mask from the biome tint's greenness: growth covers the rock in
    // the deep valleys; the mineral banding owns everything drier.
    let veg = clamp((base.g - max(base.r, base.b)) * 6.0, 0.0, 1.0);

    // ── Strata coordinate ──────────────────────────────────────────────────
    // Mostly ELEVATION (so bands follow the topography like contour lines),
    // warped by two macro noise octaves so layers buckle and fold like real
    // sediment instead of reading as a flat gradient, plus a slight horizontal
    // drift so even dead-flat ground still crosses bands.
    let warp = vnoise(p / 730.0, 61u) * 14.0 + vnoise(p / 173.0, 62u) * 5.0
        + vnoise(p / 47.0, 64u) * 1.8;
    let band_t = (wp.y + warp + (p.x + p.y) * 0.012) / 88.0;
    var strata_rgb = strata_color(band_t);

    // A finer band-within-band tier (11 m period): sub-layers inside each color
    // band, value-only, the "close enough to count the layers" read.
    let sub = vnoise(vec2(band_t * 32.0, (p.x - p.y) * 0.02), 63u);
    strata_rgb *= 1.0 + 0.28 * sub * footprint_fade(11.0, max(fwidth(wp.y), 1e-4));

    // Blend: mineral ground takes the full strata identity; vegetated valley
    // floors keep their green with only a strata undertone bleeding through.
    let strata_w = strengths.x * mix(0.92, 0.45, veg);
    // Preserve the scene's luminance scale: strata colors replace hue/chroma but
    // are lit by the same pre-exposed moonlight via the PBR pipe.
    rgb = mix(rgb, strata_rgb * (0.35 + 0.65 * length(base) / 0.55), clamp(strata_w, 0.0, 1.0));

    // ── Macro patchiness (hundreds of meters) ──────────────────────────────
    // Weathering stains across the bands so no band is one flat color from the
    // plane: value + a mild warm/cool tilt.
    let macro_n = vnoise(p / 540.0, 11u) * 0.6 + vnoise(p / 140.0, 12u) * 0.4;
    let stain = macro_n * strengths.y;
    rgb *= vec3(1.0 + 0.20 * stain, 1.0 + 0.06 * stain, 1.0 - 0.10 * stain);
    rgb *= 1.0 + 0.35 * stain;

    // ── Fine grain (meters and below): the on-foot optic-flow cue ──────────
    // Mineral grain aligned with nothing (isotropic), plus gravel flecks —
    // sparse bright chips of broken band material.
    var fine_n = vnoise(p / 2.2, 31u) * footprint_fade(2.2, fw)
        + vnoise(p / 0.8, 32u) * 0.8 * footprint_fade(0.8, fw)
        + vnoise(p / 0.29, 33u) * 0.6 * footprint_fade(0.29, fw);
    let grit_fade = vec2(footprint_fade(0.1, fw), footprint_fade(0.042, fw));
    if grit_fade.x + grit_fade.y > 0.001 {
        fine_n += vnoise(p / 0.1, 35u) * 0.45 * grit_fade.x
            + vnoise(p / 0.042, 36u) * 0.3 * grit_fade.y;
    }
    rgb *= 1.0 + 0.28 * strengths.z * fine_n;

    // Gravel flecks: rare cells whose hash clears a high threshold flash the
    // NEXT band's color — freshly broken rock scattered on the weathered top.
    let fleck_fade = footprint_fade(0.55, fw);
    if fleck_fade > 0.001 {
        let cell = vec2<i32>(floor(p / 0.55));
        let r = rand01(hash2(cell, 71u));
        if r > 0.93 {
            let fleck_rgb = strata_color(band_t + 0.25) * 1.4;
            let inner = vnoise(p / 0.18, 72u);
            let mask = smoothstep(0.1, 0.6, inner) * fleck_fade * strengths.z;
            rgb = mix(rgb, fleck_rgb * (0.35 + 0.65 * length(base) / 0.55), 0.8 * mask);
        }
    }

    // Vegetated ground keeps the round-2 grass clump read so valleys stay soft.
    let tuft = smoothstep(0.15, 0.75, vnoise(p / 1.4, 34u)) * veg * footprint_fade(1.4, fw);
    rgb *= 1.0 - 0.30 * strengths.z * tuft;

    pbr_input.material.base_color = vec4(rgb, pbr_input.material.base_color.a);

    // ── Detail normal ──────────────────────────────────────────────────────
    // Two duties: (a) close-up micro-relief as before; (b) a band-edge ledge —
    // each stratum boundary dips the normal slightly, so grazing moonlight
    // draws the layers as relief on slopes without touching geometry.
    let w_n = strengths.w * footprint_fade(0.45, fw);
    var grad = vec2(0.0);
    if w_n > 0.001 {
        let step = 0.12;
        let h0 = vnoise(p / 0.45, 51u);
        let hx = vnoise((p + vec2(step, 0.0)) / 0.45, 51u);
        let hz = vnoise((p + vec2(0.0, step)) / 0.45, 51u);
        grad = vec2(hx - h0, hz - h0) / step * 0.06 * w_n;
    }
    // Band-edge relief: derivative of a soft sawtooth in the strata coordinate,
    // projected onto the surface via the world-space gradient of band_t (≈ the
    // up direction flattened onto xz by the slope). Fades once bands go subpixel.
    let ledge_fade = footprint_fade(12.0, max(fwidth(wp.y), 1e-4)) * strengths.w;
    if ledge_fade > 0.001 {
        let saw = sin(band_t * 6.28318 * 4.0);
        let n_geo = normalize(in.world_normal);
        // Downslope direction in xz — where the band edge runs across the ground.
        let down = normalize(vec2(n_geo.x, n_geo.z) + vec2(1e-5));
        let slope = clamp(1.0 - n_geo.y, 0.0, 1.0);
        grad += down * saw * 0.05 * ledge_fade * slope;
    }
    if w_n + ledge_fade > 0.002 {
        pbr_input.N = normalize(pbr_input.N + vec3(-grad.x, 0.0, -grad.y));
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
