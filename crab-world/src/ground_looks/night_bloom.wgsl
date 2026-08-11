// Ground look: NIGHT-BLOOM (bddap/rl#304 ground-shader competition, fable-2).
// The moonlit ground stays natural — cooled, darkened a touch — but the living
// valleys carry bioluminescence: dendritic glowing veins threading the low wet
// ground (from the plane they read as rivers of cold light), a drifting spore
// speckle that resolves on foot, and a soft teal bloom-halo around the veins.
// Rare vein knots flush magenta. Every night-bloom VARIANT is a parameter row
// over this one shader (rl#329/rl#333) — palette, glow levels, vein
// spacing/width, spore density are all params, never forks.
//
// One of the interchangeable `art()` modules in this directory; the contract —
// GroundCtx in, GroundArt out, the scaffold owns `fn fragment` and the detail
// layer — lives in rl::ground::art (ground_art.wgsl) and on `GroundLook`
// (ground.rs).

#define_import_path rl::ground::looks::night_bloom

#import rl::noise::{hash2, rand01, vnoise, footprint_fade}
#import rl::ground::art::{GroundCtx, GroundArt}

// A vein field: 1 on the zero-set of a warped noise, falling off over `width`
// (in noise-space units). The zero-set of smooth noise is a connected, branching
// web — dendritic without any simulation.
fn vein(n: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(0.0, width, abs(n));
}

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let p = ctx.p;
    let fw = ctx.fw;
    let strengths = ctx.strengths;

    var rgb = ctx.base;

    // The bloom lives where things grow; mineral ground stays dark and quiet.
    let veg = ctx.veg;

    // Cool the base a step toward the variant's night tint so the warm moon
    // highlights and the glow both have somewhere to sit.
    rgb *= params[0].xyz;

    // Macro patchiness (hundreds of meters) — kept from the round-2 look but
    // biased cool/dark: dim mist-shadow patches instead of warm soil.
    let macro_n = vnoise(p / 620.0, 11u) * 0.5
        + vnoise(p / 210.0, 12u) * 0.35
        + vnoise(p / 90.0, 13u) * 0.15;
    rgb *= 1.0 + 0.40 * strengths.y * macro_n * mix(0.4, 1.0, veg);

    // Meso mottling: the mid-range octave gap. Fine on-foot detail is the
    // scaffold's always-on layer (grain = 1 below).
    let meso_n = vnoise(p / 26.0, 21u) * footprint_fade(26.0, fw)
        + vnoise(p / 9.0, 22u) * 0.7 * footprint_fade(9.0, fw);
    rgb *= 1.0 + 0.30 * strengths.y * meso_n;

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
    let emissive = vein_col * (params[1].w * artery_g + params[4].x * capil_g)
        + params[3].xyz * params[3].w * spore_g;

    var out: GroundArt;
    out.rgb = rgb;
    out.roughness = ctx.rough;
    // Micro-relief is the scaffold's relief layer.
    out.n = ctx.n;
    out.emissive = emissive;
    out.glow = vec3(0.0);
    out.grain = 1.0;
    out.relief = 1.0;
    return out;
}
