// Ground look: PATTERNED GROUND (bddap/rl#304 ground-shader competition, fable-3).
// The land is a living mosaic — periglacial patterned ground at two scales.
// From the plane: giant irregular plates (~120 m), each with its own hue tilt,
// read as a patchwork of fields seamed by darker ground. On foot: mudcrack
// polygons (~1.2 m) with grooved crack normals, dry earth between growth.
// Voronoi cellular structure everywhere — a macro identity value noise cannot
// fake. Cell tiers are faded by their on-screen footprint (fwidth), so nothing
// shimmers from the plane.
//
// One of the interchangeable `art()` modules in this directory; the contract —
// GroundCtx in, GroundArt out, the scaffold owns `fn fragment` and the detail
// layer — lives in rl::ground::art (ground_art.wgsl) and on `GroundLook`
// (ground.rs).

#define_import_path rl::ground::looks::patterned_ground

#import rl::noise::{hash2, rand01, vnoise, footprint_fade}
#import rl::ground::art::{GroundCtx, GroundArt}

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

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let p = ctx.p;
    let fw = ctx.fw;
    let strengths = ctx.strengths;

    var rgb = ctx.base;

    // Plates are strongest on open/dry ground; deep growth softens the seams
    // (roots hold the soil).
    let veg = ctx.veg;

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

    // ── Tier 3: mudcracks (~1.2 m) — the on-foot read ──────────────────────
    // Fine isotropic grain is the scaffold's always-on layer (grain = 1 below).
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

    // ── Normals: crack grooves ─────────────────────────────────────────────
    // Mudcrack grooves dip toward the crack line (finite-difference of the
    // crack field), so grazing moonlight draws every polygon edge; isotropic
    // micro-relief is the scaffold's relief layer. Shading only — geometry
    // untouched.
    var grad = vec2(0.0);
    let groove_w = strengths.w * crack_fade;
    if groove_w > 0.001 {
        let gs = 0.08;
        let c0 = voronoi(p / 1.2, 93u).y;
        let cx = voronoi((p + vec2(gs, 0.0)) / 1.2, 93u).y;
        let cz = voronoi((p + vec2(0.0, gs)) / 1.2, 93u).y;
        // Depress the surface where F2−F1 shrinks: normals lean INTO cracks.
        grad += vec2(cx - c0, cz - c0) / gs * 0.10 * groove_w;
    }
    var n = ctx.n;
    if length(grad) > 1e-4 {
        n = normalize(n + vec3(-grad.x, 0.0, -grad.y));
    }

    var out: GroundArt;
    out.rgb = rgb;
    out.roughness = ctx.rough;
    out.n = n;
    out.emissive = vec3(0.0);
    out.glow = vec3(0.0);
    out.grain = 1.0;
    out.relief = 1.0;
    return out;
}
