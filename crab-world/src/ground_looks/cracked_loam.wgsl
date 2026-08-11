// Ground look: CRACKED LOAM & COBBLE (kimi competition variant 2).
// Art direction: the ground is a MOSAIC, not a wash. Domain-warped Voronoi
// cells at three scales carry everything: dried clay plates with dark crack
// seams in the dry bands, soft turf patches in the green valley floors,
// fist-sized cobble underfoot on the stony rims, and faint geology provinces
// from the plane. Structure over noise — the eye reads cells, cracks, and
// stones, never a blur. Elevation/slope styling mirrors the biome band edges
// in terrain.rs `biome` (one source: the vertex tint's stops).
//
// One of the interchangeable `art()` modules in this directory; the contract —
// GroundCtx in, GroundArt out, the scaffold owns `fn fragment` and the detail
// layer — lives in rl::ground::art (ground_art.wgsl) and on `GroundLook`
// (ground.rs).

#define_import_path rl::ground::looks::cracked_loam

#import rl::noise::{hash2, rand01, vnoise, footprint_fade}
#import rl::ground::art::{GroundCtx, GroundArt}

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

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let wp = ctx.wp;
    let p = ctx.p;
    let fw = ctx.fw;
    let strengths = ctx.strengths;

    var rgb = ctx.base;

    let veg = ctx.veg;

    let steep = 1.0 - ctx.n_geo.y;

    // Biome band edges mirrored from terrain.rs `biome` so cell styling follows
    // the same elevation logic the vertex tint and scatter placement use.
    let dry = smoothstep(-1400.0, -500.0, wp.y);   // LOWLAND_M → DRY_GRASS_M
    let stony = smoothstep(-500.0, 100.0, wp.y);   // DRY_GRASS_M → SCREE_M
    let rock = smoothstep(0.18, 0.42, steep);      // ROCK_STEEP
    let snow = ctx.snow;                           // SNOWLINE_M on SNOW_HOLD_STEEP

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
    var n = ctx.n;
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
            n = normalize(n + w_n * 0.05 * vec3(grad.x, 0.0, grad.y));
        }
    }

    var out: GroundArt;
    out.rgb = rgb;
    out.roughness = ctx.rough;
    out.n = n;
    out.emissive = vec3(0.0);
    out.glow = vec3(0.0);
    // The cobble above is structured near-field art, not the generic layer —
    // the scaffold's grain/relief ride on top at full strength.
    out.grain = 1.0;
    out.relief = 1.0;
    return out;
}
