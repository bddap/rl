// Ground look: WIND-COMBED (kimi competition variant 1).
// Art direction: the ground is COMBED, not mottled — every scale reads as
// aligned by wind and gravity, the way farmland and hillsides do from the air.
// A slowly varying wind field (bent onto slope contours as faces steepen —
// sediment combs AROUND a hill) orients anisotropic streak octaves, so the
// terrain has a directional GRAIN: long straw-and-green comb-lines in the
// meadows, sediment flow-lines on the steeps, short combed fiber underfoot.
//
// One of the interchangeable `art()` modules in this directory; the contract —
// GroundCtx in, GroundArt out, the scaffold owns `fn fragment` and the detail
// layer — lives in rl::ground::art (ground_art.wgsl) and on `GroundLook`
// (ground.rs).

#define_import_path rl::ground::looks::wind_combed

// strengths lanes here: x macro warm/cool drift, y meso comb streaks, z fine
// combed fiber (and the scaffold's grain gain), w streak detail normal (and
// the scaffold's relief gain).
#import rl::noise::{vnoise, footprint_fade}
#import rl::ground::art::{GroundCtx, GroundArt, default_art}

// Anisotropic value noise: stretched to `along`×`across` meters in the comb
// frame `d`, so one sample is a streak, not a blot.
fn streak(p: vec2<f32>, d: vec2<f32>, along: f32, across: f32, seed: u32) -> f32 {
    let q = vec2(dot(p, d) / along, dot(p, vec2(-d.y, d.x)) / across);
    return vnoise(q, seed);
}

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let p = ctx.p;
    let fw = ctx.fw;
    let strengths = ctx.strengths;

    var rgb = ctx.base;

    // Combs read strongest on growth, muted on scree/rock/snow.
    let veg = ctx.veg;

    let n_geo = ctx.n_geo;
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

    // Fine on-foot grain: short combed fiber — the directional identity the
    // shared layer cannot carry; isotropic grit is the scaffold's always-on
    // layer (grain = 1 below). Branch-gated (screen-coherent, distance-driven)
    // so the far ground never pays for it.
    let grain_f = footprint_fade(0.5, fw);
    if grain_f > 0.001 {
        let g = streak(p, d, 3.4, 0.5, 81u) * grain_f
            + streak(p, d, 1.2, 0.18, 82u) * 0.7 * footprint_fade(0.18, fw);
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
    var n = ctx.n;
    let w_n = strengths.w * footprint_fade(1.2, fw);
    if w_n > 0.001 {
        let perp = vec2(-d.y, d.x);
        let step = 0.3;
        let s0 = streak(p, d, 5.0, 1.2, 84u);
        let s1 = streak(p + perp * step, d, 5.0, 1.2, 84u);
        let g = (s1 - s0) / step * 0.045;
        n = normalize(n + w_n * vec3(-perp.x * g, 0.0, -perp.y * g));
    }

    var out = default_art(ctx);
    out.rgb = rgb;
    out.n = n;
    return out;
}
