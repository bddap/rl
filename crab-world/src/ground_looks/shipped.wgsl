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
#import rl::ground::detail::{fine_color, relief_normal}
#import rl::ground::art::{GroundCtx, GroundArt}

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let p = ctx.p;
    let fw = ctx.fw;
    let strengths = ctx.strengths;

    var rgb = ctx.base;

    // Vegetation mask: full patchiness on grass, muted on scree/rock/snow
    // (mineral ground varies less than growth does).
    let veg = ctx.veg;

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

    // Fine on-foot detail: the rl#324 adaptive descent. Stage 4(a) keeps this
    // in-look call (grain = 0 below keeps the scaffold's layer inert) so the
    // flip's sweep matches current output; 4(b) deletes it and flips grain to 1.
    rgb *= 1.0 + fine_color(p, fw, 0.30 * strengths.z);

    // Grass clumps: darker tufted patches where the ground is vegetated.
    let tuft = smoothstep(0.15, 0.75, vnoise(p / 1.4, 34u)) * veg * footprint_fade(1.4, fw);
    rgb *= 1.0 - 0.30 * strengths.z * tuft;

    // Sedimentary strata on steep faces: elevation-banded value variation, so
    // cliffs read as layered rock instead of smeared vertex tint.
    let steep = 1.0 - ctx.n_geo.y;
    let strata_mask = smoothstep(0.25, 0.55, steep);
    let strata = vnoise(vec2(ctx.wp.y / 7.0, (p.x + p.y) * 0.012), 41u);
    rgb *= 1.0 + 0.35 * strata_mask * strata * footprint_fade(7.0, ctx.fw_y);

    var out: GroundArt;
    out.rgb = rgb;
    out.roughness = ctx.rough;
    // Micro-relief detail normal: kept in-look for stage 4(a) (relief = 0 below),
    // deleted in 4(b) where the scaffold's layer takes over.
    out.n = relief_normal(p, fw, ctx.n, strengths.w);
    out.emissive = vec3(0.0);
    out.glow = vec3(0.0);
    out.grain = 0.0;
    out.relief = 0.0;
    return out;
}
