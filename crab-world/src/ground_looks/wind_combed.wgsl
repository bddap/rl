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

// ─── THIS LOOK: WIND-COMBED (kimi competition variant 1) ────────────────────
// Art direction: the ground is COMBED, not mottled — every scale reads as
// aligned by wind and gravity, the way farmland and hillsides do from the air.
// A slowly varying wind field (bent onto slope contours as faces steepen —
// sediment combs AROUND a hill) orients anisotropic streak octaves, so the
// terrain has a directional GRAIN: long straw-and-green comb-lines in the
// meadows, sediment flow-lines on the steeps, short combed fiber underfoot.
// strengths lanes: x macro warm/cool drift, y meso comb streaks, z fine
// on-foot grain, w streak detail normal.

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

// x: macro drift, y: meso comb streaks, z: fine grain, w: detail normal.
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

// Anisotropic value noise: stretched to `along`×`across` meters in the comb
// frame `d`, so one sample is a streak, not a blot.
fn streak(p: vec2<f32>, d: vec2<f32>, along: f32, across: f32, seed: u32) -> f32 {
    let q = vec2(dot(p, d) / along, dot(p, vec2(-d.y, d.x)) / across);
    return vnoise(q, seed);
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Anchor-relative ground meters (rl#334): raw world xz quantizes at the fine
    // octaves' own scale out at the tile corners — see shipped.wgsl.
    let p = in.uv;
    // Ground meters per pixel at this fragment — the octave-fade driver.
    let fw = max(max(fwidth(p.x), fwidth(p.y)), 1e-4);

    var rgb = pbr_input.material.base_color.rgb;

    // Vegetation mask from the biome tint's greenness: combs read strongest on
    // growth, muted on scree/rock/snow.
    let veg = clamp((rgb.g - max(rgb.r, rgb.b)) * 6.0, 0.0, 1.0);

    let n_geo = normalize(in.world_normal);
    let steep = 1.0 - n_geo.y;

    // The comb direction: a wind angle from two low-frequency octaves, blended
    // toward the slope-contour direction as the face steepens. Contour is
    // sign-ambiguous — align it with the wind before blending so the two never
    // cancel. cross(up, N).xz = (N.z, -N.x).
    let wind_a = 3.14159265 * (vnoise(p / 700.0, 61u) * 0.7 + vnoise(p / 230.0, 62u) * 0.55);
    var d = vec2(cos(wind_a), sin(wind_a));
    let contour_w = smoothstep(0.04, 0.22, steep);
    if contour_w > 0.001 {
        var cd = vec2(n_geo.z, -n_geo.x);
        let cl = length(cd);
        if cl > 1e-4 {
            cd = cd / cl;
            if dot(cd, d) < 0.0 {
                cd = -cd;
            }
            d = normalize(mix(d, cd, contour_w));
        }
    }

    // Macro warm/cool drift (hundreds of meters): kills the banded-paint read
    // from the plane. A hue drift, not just value, so patches look like
    // different growth and soil rather than shadow.
    let macro_n = vnoise(p / 620.0, 11u) * 0.5
        + vnoise(p / 210.0, 12u) * 0.35
        + vnoise(p / 90.0, 13u) * 0.15;
    let warm = macro_n * strengths.x * mix(0.4, 1.0, veg);
    rgb *= vec3(1.0 + 0.22 * warm, 1.0 + 0.05 * warm, 1.0 - 0.16 * warm);
    rgb *= 1.0 + 0.50 * warm;

    // Meso comb streaks (tens of meters, direction-locked): the look's spine.
    // The ACROSS wavelength drives the footprint fade — that is the axis that
    // can shimmer; along-streak variation is too slow to.
    let comb = streak(p, d, 70.0, 9.0, 71u) * 0.6 * footprint_fade(9.0, fw)
        + streak(p, d, 24.0, 3.0, 72u) * 0.4 * footprint_fade(3.0, fw);
    let comb_v = comb * strengths.y;
    rgb *= 1.0 + 0.38 * comb_v;
    // Straw combs lighter, green combs darker: a hue tilt, not a gray wash.
    rgb *= vec3(1.0 + 0.12 * comb_v, 1.0 + 0.04 * comb_v, 1.0 - 0.08 * comb_v);

    // Fine on-foot grain: short combed fiber plus a little isotropic grit so
    // bare soil never reads as pure stripes. Branch-gated (screen-coherent,
    // distance-driven) so the far ground never pays for it.
    let grain_f = footprint_fade(0.5, fw);
    if grain_f > 0.001 {
        let g = streak(p, d, 3.4, 0.5, 81u) * grain_f
            + streak(p, d, 1.2, 0.18, 82u) * 0.7 * footprint_fade(0.18, fw)
            + vnoise(p / 0.24, 83u) * 0.45 * footprint_fade(0.24, fw);
        rgb *= 1.0 + 0.33 * strengths.z * g;
    }

    // Steep faces: comb lines become sediment flow — deepen their contrast and
    // pull toward mineral gray so cliffs read as combed scree, not striped
    // grass.
    let flow = smoothstep(0.25, 0.5, steep);
    if flow > 0.001 {
        let luma = dot(rgb, vec3(0.333333));
        rgb = mix(rgb, vec3(luma) * (1.0 + 0.35 * comb * strengths.y), flow * 0.5);
    }

    // Detail normal: streaks carry micro-relief across the comb direction only
    // (that is where they vary). Two taps, footprint-gated like the base.
    let w_n = strengths.w * footprint_fade(1.2, fw);
    if w_n > 0.001 {
        let perp = vec2(-d.y, d.x);
        let step = 0.3;
        let s0 = streak(p, d, 5.0, 1.2, 84u);
        let s1 = streak(p + perp * step, d, 5.0, 1.2, 84u);
        let g = (s1 - s0) / step * 0.045;
        pbr_input.N = normalize(pbr_input.N + w_n * vec3(-perp.x * g, 0.0, -perp.y * g));
    }

    pbr_input.material.base_color = vec4(rgb, pbr_input.material.base_color.a);

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
