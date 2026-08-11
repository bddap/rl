// Procedural ground detail (bddap/rl#304) over the terrain mesh's vertex biome
// tint. Everything is derived from the anchor-relative ground plane
// (world_position.xz, rl#334/rl#354) — no sampled texture, so no repeat period exists to spot from any
// altitude. Octaves are faded by their
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

#import rl::noise::{vnoise, footprint_fade}
#import rl::ground::detail::{fine_color, relief_normal}

// x: macro patchiness, y: meso mottling, z: fine on-foot detail, w: detail normal.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> strengths: vec4<f32>;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let wp = in.world_position.xyz;
    // Ground-plane meters, ANCHOR-relative (rl#334/rl#354): the terrain mesh's
    // entity is translated by −anchor (the round's locale origin), so this varying
    // is small — hence precise — near play. Raw world xz's ~1-2 mm quantization at
    // the tile's ±15 km corners is the fine octaves' own scale — the detail
    // dissolved into speckle that boiled whenever the eye moved. The anchor is
    // constant per round, so the pattern stays glued to the ground; it re-rolls
    // only across rounds, where the locale moves anyway.
    let p = in.world_position.xz;
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

    // Fine on-foot detail: the rl#324 adaptive descent, one copy in
    // rl::ground::detail (rl#333 seam 2).
    rgb *= 1.0 + fine_color(p, fw, 0.30 * strengths.z);

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

    // Micro-relief detail normal: same rl#324 adaptive descent, same one copy.
    pbr_input.N = relief_normal(p, fw, pbr_input.N, strengths.w);

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
