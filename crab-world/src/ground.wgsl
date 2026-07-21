// Procedural ground detail (bddap/rl#304) over the terrain mesh's vertex biome
// tint. Everything is derived from WORLD-SPACE position — no sampled texture, so
// no repeat period exists to spot from any altitude. Octaves are faded by their
// on-screen footprint (fwidth): the procedural analogue of mipmapping, so fine
// detail exists on foot and at landing height (the rl#197 optic-flow duty the old
// checker carried) but never shimmers from the plane.

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
    let fine_n = vnoise(p / 2.6, 31u) * footprint_fade(2.6, fw)
        + vnoise(p / 0.9, 32u) * 0.8 * footprint_fade(0.9, fw)
        + vnoise(p / 0.31, 33u) * 0.6 * footprint_fade(0.31, fw);
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

    // Detail normal from the fine-noise heightfield gradient: moonlit micro-relief
    // up close. Faded with the same footprint rule, so distant ground keeps the
    // smooth geometric normal.
    let w_n = strengths.w * footprint_fade(0.45, fw);
    if w_n > 0.001 {
        let step = 0.12;
        let h0 = vnoise(p / 0.45, 51u);
        let hx = vnoise((p + vec2(step, 0.0)) / 0.45, 51u);
        let hz = vnoise((p + vec2(0.0, step)) / 0.45, 51u);
        let grad = vec2(hx - h0, hz - h0) / step * 0.06;
        pbr_input.N = normalize(pbr_input.N + w_n * vec3(-grad.x, 0.0, -grad.y));
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
