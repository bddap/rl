// Ground look: SHIPPED (bddap/rl#304) — procedural detail over the terrain mesh's
// vertex biome tint. Everything is derived from the anchor-relative ground plane;
// octaves are faded by their on-screen footprint (fwidth), so detail exists on
// foot and at landing height but never shimmers from the plane.
//
// One of the interchangeable `art()` modules in this directory; the contract —
// GroundCtx in, GroundArt out, the scaffold owns `fn fragment` and the detail
// layer — lives in rl::ground::art (ground_art.wgsl) and on `GroundLook`
// (ground.rs).

#define_import_path rl::ground::looks::shipped

#import rl::noise::{vnoise, footprint_fade}
#import rl::ground::art::{GroundCtx, GroundArt, default_art}

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let p = ctx.p;
    let fw = ctx.fw;
    let strengths = ctx.strengths;

    var rgb = ctx.base;

    // Vegetation mask: tufts grow only where the biome tints growth.
    let veg = ctx.veg;

    // Meso mottling (tens of meters): the mid-range octave gap between biome bands
    // and on-foot detail.
    let meso_n = vnoise(p / 26.0, 21u) * footprint_fade(26.0, fw)
        + vnoise(p / 9.0, 22u) * 0.7 * footprint_fade(9.0, fw);
    rgb *= 1.0 + 0.35 * strengths.y * meso_n;

    // Fine on-foot detail is the scaffold's always-on layer (grain = 1 below).

    // Grass clumps: darker tufted patches where the ground is vegetated.
    let tuft = smoothstep(0.15, 0.75, vnoise(p / 1.4, 34u)) * veg * footprint_fade(1.4, fw);
    rgb *= 1.0 - 0.30 * strengths.z * tuft;

    // Sedimentary strata on steep faces: elevation-banded value variation, so
    // cliffs read as layered rock instead of smeared vertex tint.
    let steep = 1.0 - ctx.n_geo.y;
    let strata_mask = smoothstep(0.25, 0.55, steep);
    let strata = vnoise(vec2(ctx.wp.y / 7.0, (p.x + p.y) * 0.012), 41u);
    rgb *= 1.0 + 0.35 * strata_mask * strata * footprint_fade(7.0, ctx.fw_y);

    var out = default_art(ctx);
    out.rgb = rgb;
    return out;
}
