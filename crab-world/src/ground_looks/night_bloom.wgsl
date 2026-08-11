// Ground look: NIGHT-BLOOM (bddap/rl#304 ground-shader competition, fable-2).
// The moonlit ground stays natural — cooled, darkened a touch — but the living
// valleys carry bioluminescence: dendritic glowing veins threading the low wet
// ground (from the plane they read as rivers of cold light), a drifting spore
// speckle that resolves on foot, and a soft teal bloom-halo around the veins.
// Rare vein knots flush magenta. Everything is derived from WORLD-SPACE
// position — no sampled texture, so no repeat period exists to spot from any
// altitude. Octaves are faded by their on-screen footprint (fwidth), so fine
// detail exists on foot but never shimmers from the plane.
//
// One of the interchangeable looks in this directory; the contract every
// file here keeps — inputs, binding 100, the `fragment` entry point — is
// documented once on `GroundLook` in ground.rs. Every night-bloom VARIANT is a
// parameter row over this one shader (binding 104, rl#329/rl#333) — palette,
// glow levels, vein spacing/width, spore density are all params, never forks.

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

#import rl::noise::{hash2, rand01, vnoise, footprint_fade}

// x: vein glow, y: macro patchiness, z: fine detail + spores, w: detail normal.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> strengths: vec4<f32>;
// The look's aesthetic parameter row — every night-bloom variant is a row over
// this ONE shader. Lane meanings live on `GroundLook::params` in ground.rs.
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var<uniform> params: array<vec4<f32>, 8>;

// A vein field: 1 on the zero-set of a warped noise, falling off over `width`
// (in noise-space units). The zero-set of smooth noise is a connected, branching
// web — dendritic without any simulation.
fn vein(n: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(0.0, width, abs(n));
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Anchor-relative ground meters (rl#334/rl#354: the mesh entity is translated
    // by −anchor, so world_position.xz is small and precise near play).
    let p = in.world_position.xz;
    // Ground meters per pixel at this fragment — the octave-fade driver.
    let fw = max(max(fwidth(p.x), fwidth(p.y)), 1e-4);

    var rgb = pbr_input.material.base_color.rgb;

    // Vegetation mask from the biome tint's greenness — the bloom lives where
    // things grow; mineral ground (scree/rock/snow) stays dark and quiet.
    let veg = clamp((rgb.g - max(rgb.r, rgb.b)) * 6.0, 0.0, 1.0);

    // Cool the base a step toward the variant's night tint so the warm moon
    // highlights and the glow both have somewhere to sit.
    rgb *= params[0].xyz;

    // Macro patchiness (hundreds of meters) — kept from the round-2 look but
    // biased cool/dark: dim mist-shadow patches instead of warm soil.
    let macro_n = vnoise(p / 620.0, 11u) * 0.5
        + vnoise(p / 210.0, 12u) * 0.35
        + vnoise(p / 90.0, 13u) * 0.15;
    rgb *= 1.0 + 0.40 * strengths.y * macro_n * mix(0.4, 1.0, veg);

    // Meso mottling + fine detail: the optic-flow duty, unchanged in role.
    let meso_n = vnoise(p / 26.0, 21u) * footprint_fade(26.0, fw)
        + vnoise(p / 9.0, 22u) * 0.7 * footprint_fade(9.0, fw);
    rgb *= 1.0 + 0.30 * strengths.y * meso_n;
    var fine_n = vnoise(p / 2.6, 31u) * footprint_fade(2.6, fw)
        + vnoise(p / 0.9, 32u) * 0.8 * footprint_fade(0.9, fw)
        + vnoise(p / 0.31, 33u) * 0.6 * footprint_fade(0.31, fw);
    let grit_fade = footprint_fade(0.11, fw);
    if grit_fade > 0.001 {
        fine_n += vnoise(p / 0.11, 35u) * 0.45 * grit_fade;
    }
    rgb *= 1.0 + 0.28 * strengths.z * fine_n;

    // ── The bloom ──────────────────────────────────────────────────────────
    // Two dendritic vein tiers on warped noise zero-sets: arteries (params[4].z
    // meters spacing, visible as glowing river-webs from the plane) and
    // capillaries (params[4].w m, the on-foot/mid tier). Warp keeps them organic.
    let artery_l = params[4].z;
    let capil_l = params[4].w;
    // Warp/sub-octave scales ride the spacings at the classic 61/180 and 4.7/14
    // ratios, so NightBloom's row keeps the original field (up to f32 rounding
    // of the ratios) and every variant warps in proportion.
    let wq = vnoise(p / (artery_l * (61.0 / 180.0)), 71u);
    let artery_n = vnoise(p / artery_l, 72u) + 0.35 * wq;
    let capil_n = vnoise(p / capil_l, 73u) + 0.4 * vnoise(p / (capil_l * (4.7 / 14.0)), 74u);
    // Arteries stay unfaded (macro feature); capillaries fade out by footprint.
    let artery = vein(artery_n, params[5].x) * (0.4 + 0.6 * vein(artery_n, params[5].y));
    let capil = vein(capil_n, params[5].z) * footprint_fade(capil_l, fw);
    // Spore speckle: bright cells rarer than the params[4].y threshold, on-foot only.
    var spore = 0.0;
    let spore_fade = footprint_fade(0.8, fw);
    if spore_fade > 0.001 {
        let cell = vec2<i32>(floor(p / 0.8));
        let r = rand01(hash2(cell, 75u));
        if r > params[4].y {
            let inner = vnoise(p / 0.26, 76u);
            spore = smoothstep(0.2, 0.75, inner) * spore_fade;
        }
    }

    // Glow lives in the growing lowlands; it dies on scree, rock, and snow.
    let habitat = veg;
    let glow_mask = strengths.x * habitat;
    let artery_g = artery * glow_mask;
    let capil_g = capil * glow_mask * 0.8;
    let spore_g = spore * glow_mask * strengths.z;

    // Colors: the variant's vein color for the web, its knot flush where artery
    // crests knot (second zero-set nearby), its spore color for the speckle.
    let knot = vein(vnoise(p / 43.0, 77u), 0.12);
    let vein_col = mix(params[1].xyz, params[2].xyz, params[2].w * knot);

    // The glow: emissive light the moon doesn't own. Levels are chosen against
    // the pre-exposed night — arteries ~2× lit-ground luminance at their core,
    // spores a quiet sparkle. Ground under the glow darkens (wet soil, by the
    // variant's params[0].w), so the light reads as coming FROM the ground, not
    // painted on it.
    let glow_total = artery_g + capil_g;
    rgb *= 1.0 - params[0].w * clamp(glow_total, 0.0, 1.0);
    pbr_input.material.base_color = vec4(rgb, pbr_input.material.base_color.a);
    let emissive = vein_col * (params[1].w * artery_g + params[4].x * capil_g)
        + params[3].xyz * params[3].w * spore_g;
    pbr_input.material.emissive = vec4(
        pbr_input.material.emissive.rgb + emissive,
        pbr_input.material.emissive.a,
    );

    // Detail normal: unchanged duty — moonlit micro-relief up close.
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
