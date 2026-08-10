// Ground look: PATTERNED GROUND (bddap/rl#304 ground-shader competition, fable-3).
// The land is a living mosaic — periglacial patterned ground at two scales.
// From the plane: giant irregular plates (~120 m), each with its own hue tilt,
// read as a patchwork of fields seamed by darker ground. On foot: mudcrack
// polygons (~1.2 m) with grooved crack normals, dry earth between growth.
// Voronoi cellular structure everywhere — a macro identity value noise cannot
// fake. Everything is derived from WORLD-SPACE position — no sampled texture,
// so no repeat period exists to spot from any altitude. Cell tiers are faded by
// their on-screen footprint (fwidth), so nothing shimmers from the plane.
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

#import rl::noise::{hash2, rand01, vnoise, footprint_fade}

// x: plate mosaic, y: macro weathering, z: mudcracks + fine, w: detail normal.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> strengths: vec4<f32>;

// Voronoi over jittered lattice cells: returns (F1, F2 − F1, cell hash).
// F2 − F1 ≈ 0 on cell borders — the seam/crack driver; the hash gives each
// plate its own stable identity. 3×3 neighborhood, 9 hashes per call.
fn voronoi(q: vec2<f32>, seed: u32) -> vec3<f32> {
    let i = vec2<i32>(floor(q));
    let f = q - floor(q);
    var f1 = 8.0;
    var f2 = 8.0;
    var id = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let cell = i + vec2(dx, dy);
            let h = hash2(cell, seed);
            let site = vec2(f32(dx), f32(dy))
                + vec2(rand01(h), rand01(h * 0x9e3779b9u + 1u)) - f;
            let d = dot(site, site);
            if d < f1 {
                f2 = f1;
                f1 = d;
                id = rand01(h ^ 0x5bd1e995u);
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    return vec3(sqrt(f1), sqrt(f2) - sqrt(f1), id);
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

    // Vegetation mask from the biome tint's greenness: plates are strongest on
    // open/dry ground; deep growth softens the seams (roots hold the soil).
    let veg = clamp((rgb.g - max(rgb.r, rgb.b)) * 6.0, 0.0, 1.0);

    // ── Tier 1: giant plates (~120 m) — the patchwork-from-the-plane read ──
    // Domain-warped so plate edges meander instead of reading as a jittered grid.
    let warp = vec2(vnoise(p / 210.0, 81u), vnoise(p / 210.0, 82u)) * 0.55;
    let plate = voronoi(p / 120.0 + warp, 91u);
    // Per-plate identity: hue tilt (warm soil ↔ cool growth ↔ pale silt) and a
    // value step, so adjacent plates always separate.
    let hue = plate.z;
    let tilt = vec3(
        0.70 + 0.75 * hue,
        0.88 + 0.24 * abs(hue - 0.5),
        1.25 - 0.70 * hue,
    );
    let plate_w = strengths.x * mix(1.0, 0.65, veg);
    rgb *= mix(vec3(1.0), tilt * (0.82 + 0.5 * fract(hue * 7.31)), plate_w);
    // Seams: darker, damper ground between plates — wide soft shoulder plus a
    // tight dark core, both meaningful from altitude.
    let seam = (1.0 - smoothstep(0.0, 0.16, plate.y)) * 0.55
        + (1.0 - smoothstep(0.0, 0.05, plate.y)) * 0.45;
    rgb *= 1.0 - 0.42 * seam * plate_w;

    // ── Tier 2: meso plates (~14 m) — the claw-height stepping-stone read ──
    let meso = voronoi(p / 14.0 + warp * 3.0, 92u);
    let meso_fade = footprint_fade(14.0, fw);
    rgb *= 1.0 + (0.16 * (meso.z - 0.5) - 0.22 * (1.0 - smoothstep(0.0, 0.10, meso.y)))
        * strengths.x * meso_fade;

    // ── Macro weathering (hundreds of meters) ──────────────────────────────
    // Value-only stain across plates so the mosaic never reads as flat paint.
    let macro_n = vnoise(p / 560.0, 11u) * 0.6 + vnoise(p / 150.0, 12u) * 0.4;
    rgb *= 1.0 + 0.30 * strengths.y * macro_n;

    // ── Tier 3: mudcracks (~1.2 m) + fine grain — the on-foot read ─────────
    let crack_fade = footprint_fade(1.2, fw);
    var crack = 0.0;
    if crack_fade > 0.001 {
        let mud = voronoi(p / 1.2, 93u);
        crack = (1.0 - smoothstep(0.0, 0.09, mud.y)) * crack_fade;
        // Cracks open on dry open ground, close under vegetation.
        crack *= mix(1.0, 0.35, veg);
        rgb *= 1.0 - 0.38 * strengths.z * crack;
        // Slight per-polygon value so shards read individually.
        rgb *= 1.0 + 0.10 * (mud.z - 0.5) * strengths.z * crack_fade;
    }
    var fine_n = vnoise(p / 0.9, 32u) * 0.8 * footprint_fade(0.9, fw)
        + vnoise(p / 0.31, 33u) * 0.6 * footprint_fade(0.31, fw);
    let grit_fade = footprint_fade(0.11, fw);
    if grit_fade > 0.001 {
        fine_n += vnoise(p / 0.11, 35u) * 0.45 * grit_fade;
    }
    rgb *= 1.0 + 0.26 * strengths.z * fine_n;

    pbr_input.material.base_color = vec4(rgb, pbr_input.material.base_color.a);

    // ── Normals: crack grooves + micro-relief ──────────────────────────────
    // Mudcrack grooves dip toward the crack line (finite-difference of the
    // crack field), so grazing moonlight draws every polygon edge; plus the
    // round-2 micro-relief. Shading only — geometry untouched.
    let w_n = strengths.w * footprint_fade(0.45, fw);
    var grad = vec2(0.0);
    if w_n > 0.001 {
        let step = 0.12;
        let h0 = vnoise(p / 0.45, 51u);
        let hx = vnoise((p + vec2(step, 0.0)) / 0.45, 51u);
        let hz = vnoise((p + vec2(0.0, step)) / 0.45, 51u);
        grad = vec2(hx - h0, hz - h0) / step * 0.06 * w_n;
    }
    let groove_w = strengths.w * crack_fade;
    if groove_w > 0.001 {
        let gs = 0.08;
        let c0 = voronoi(p / 1.2, 93u).y;
        let cx = voronoi((p + vec2(gs, 0.0)) / 1.2, 93u).y;
        let cz = voronoi((p + vec2(0.0, gs)) / 1.2, 93u).y;
        // Depress the surface where F2−F1 shrinks: normals lean INTO cracks.
        grad += vec2(cx - c0, cz - c0) / gs * 0.10 * groove_w;
    }
    if length(grad) > 1e-4 {
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
