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

// ─── THIS LOOK: CRACKED LOAM & COBBLE (kimi competition variant 2) ──────────
// Art direction: the ground is a MOSAIC, not a wash. Domain-warped Voronoi
// cells at three scales carry everything: dried clay plates with dark crack
// seams in the dry bands, soft turf patches in the green valley floors,
// fist-sized cobble underfoot on the stony rims, and faint geology provinces
// from the plane. Structure over noise — the eye reads cells, cracks, and
// stones, never a blur. Elevation/slope styling mirrors the biome band edges
// in terrain.rs `biome` (one source: the vertex tint's stops).
// strengths lanes: x macro provinces, y meso crack network, z fine cobble,
// w cobble detail normal.

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

// x: macro provinces, y: meso cracks, z: fine cobble, w: detail normal.
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

struct Voro {
    f1: f32,   // distance to nearest feature point (cell units)
    edge: f32, // F2-F1: ~0 on a cell border
    id: vec2<i32>,
}

// Jittered-grid Voronoi, 3x3 search (F2 correctness needs the full ring).
fn voronoi(p: vec2<f32>, seed: u32) -> Voro {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    var f1 = 8.0;
    var f2 = 8.0;
    var id = vec2<i32>(0, 0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let cell = i + vec2<i32>(dx, dy);
            let h = hash2(cell, seed);
            let pt = vec2<f32>(f32(dx), f32(dy))
                + vec2<f32>(rand01(h), rand01(h ^ 0x9e3779b9u)) - f;
            let dd = dot(pt, pt);
            if dd < f1 {
                f2 = f1;
                f1 = dd;
                id = cell;
            } else if dd < f2 {
                f2 = dd;
            }
        }
    }
    let d1 = sqrt(f1);
    return Voro(d1, sqrt(f2) - d1, id);
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let wp = in.world_position.xyz;
    // Anchor-relative ground meters (rl#334): raw world xz quantizes at the fine
    // octaves' own scale out at the tile corners — see shipped.wgsl.
    let p = in.uv;
    // Ground meters per pixel at this fragment — the octave-fade driver.
    let fw = max(max(fwidth(p.x), fwidth(p.y)), 1e-4);

    var rgb = pbr_input.material.base_color.rgb;

    // Vegetation mask from the biome tint's greenness.
    let veg = clamp((rgb.g - max(rgb.r, rgb.b)) * 6.0, 0.0, 1.0);

    let n_geo = normalize(in.world_normal);
    let steep = 1.0 - n_geo.y;

    // Biome band edges mirrored from terrain.rs `biome` so cell styling follows
    // the same elevation logic the vertex tint and scatter placement use.
    let dry = smoothstep(-1400.0, -500.0, wp.y);   // LOWLAND_M → DRY_GRASS_M
    let stony = smoothstep(-500.0, 100.0, wp.y);   // DRY_GRASS_M → SCREE_M
    let rock = smoothstep(0.18, 0.42, steep);      // ROCK_STEEP
    let snow = smoothstep(350.0, 650.0, wp.y)      // SNOWLINE_M …
        * (1.0 - smoothstep(0.18, 0.38, steep));   // … on SNOW_HOLD_STEEP

    // ── macro provinces (420 m): faint polygonal geology from the plane ──
    // Per-cell tone, washed with plain noise so borders never read vector-hard.
    let pv = voronoi(p / 420.0, 100u);
    let prov = rand01(hash2(pv.id, 101u)) - 0.5;
    let prov_hue = rand01(hash2(pv.id, 102u)) - 0.5;
    let m_tone = mix(vnoise(p / 260.0, 103u), prov * 1.6, 0.6) * strengths.x;
    rgb *= 1.0 + 0.20 * m_tone;
    rgb *= vec3(1.0 + 0.09 * prov_hue * strengths.x, 1.0, 1.0 - 0.09 * prov_hue * strengths.x);

    // ── meso plates (8 m): crack seams + per-cell patchwork ──
    // Domain warp keeps the mosaic organic — no grid ever reads.
    let warp = vec2(vnoise(p / 34.0, 91u), vnoise((p + vec2(7.3, 3.1)) / 34.0, 92u)) * 2.6;
    let mv = voronoi((p + warp) / 8.0, 93u);
    // Seam width grows with dryness/rock: hairlines in turf, crevices in clay.
    let crack_w = mix(0.05, 0.16, max(dry, rock));
    let seam_f = footprint_fade(1.0, fw);
    let seam = (1.0 - smoothstep(0.0, crack_w, mv.edge))
        * strengths.y * mix(0.50, 0.95, max(dry * 0.8, rock)) * (1.0 - snow) * seam_f;
    rgb *= 1.0 - 0.62 * seam;
    // Seams on dry ground are shadowed red-brown soil, not gray paint.
    rgb = mix(rgb, rgb * vec3(1.15, 0.80, 0.70), clamp(seam, 0.0, 1.0) * 0.35);
    // Per-cell tint: a patchwork of soil and growth, never a uniform wash.
    let ch = rand01(hash2(mv.id, 94u)) - 0.5;
    let ch2 = rand01(hash2(mv.id, 95u)) - 0.5;
    let meso_f = footprint_fade(8.0, fw);
    rgb *= 1.0 + 0.32 * ch * strengths.y * meso_f;
    rgb *= vec3(1.0 + 0.10 * ch2 * strengths.y * meso_f, 1.0 + 0.02 * ch2 * strengths.y * meso_f, 1.0);
    // Plate centers lift a touch — dried clay curls, turf crowns.
    rgb *= 1.0 + 0.14 * strengths.y * (1.0 - smoothstep(0.0, 0.45, mv.f1)) * meso_f;

    // ── fine cobble (0.55 m): stones underfoot, branch-gated ──
    let cob_f = footprint_fade(0.55, fw);
    if cob_f > 0.001 {
        let cobble_amt = strengths.z * mix(0.55, 1.0, max(stony, rock)) * (1.0 - snow);
        let cwarp = vec2(vnoise(p / 1.7, 96u), vnoise((p + vec2(3.7, 9.2)) / 1.7, 97u)) * 0.18;
        let cp = (p + cwarp) / 0.55;
        let cv = voronoi(cp, 98u);
        let crev = 1.0 - smoothstep(0.0, 0.22, cv.edge);
        rgb *= 1.0 - 0.55 * cobble_amt * crev * cob_f;
        let dome = 1.0 - smoothstep(0.0, 0.5, cv.f1);
        rgb *= 1.0 + 0.22 * cobble_amt * dome * cob_f;
        // Per-stone tone so the cobble isn't one material repeated.
        let st = rand01(hash2(cv.id, 99u)) - 0.5;
        rgb *= 1.0 + 0.24 * st * cobble_amt * cob_f;
        // Detail normal: stones bulge — height falls with f1, so the perturbed
        // normal tilts down the f1 gradient, outward from each stone's center.
        let w_n = strengths.w * cob_f * cobble_amt;
        if w_n > 0.001 {
            let step = 0.14;
            let fx = voronoi(cp + vec2(step / 0.55, 0.0), 98u).f1;
            let fz = voronoi(cp + vec2(0.0, step / 0.55), 98u).f1;
            let grad = vec2(fx - cv.f1, fz - cv.f1) / step;
            pbr_input.N = normalize(pbr_input.N + w_n * 0.05 * vec3(grad.x, 0.0, grad.y));
        }
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
