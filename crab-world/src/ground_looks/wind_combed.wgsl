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

// strengths lanes here: x unused, y meso comb streaks, z fine combed fiber
// (and the scaffold's grain gain), w streak detail normal (and the scaffold's
// relief gain).
#import rl::noise::{footprint_fade, streak, wind_dir}
#import rl::ground::art::{GroundCtx, GroundArt, default_art}

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let p = ctx.p;
    let fw = ctx.fw;
    let strengths = ctx.strengths;

    var rgb = ctx.base;

    let n_geo = ctx.n_geo;
    let steep = 1.0 - n_geo.y;

    // The comb direction: the shared wind field (rl::noise wind_dir).
    let d = wind_dir(p, n_geo);

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
