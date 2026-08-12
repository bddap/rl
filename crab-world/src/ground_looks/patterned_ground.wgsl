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

// strengths lanes here: x plate mosaic (both tiers), y unused, z mudcracks
// (and the scaffold's grain gain), w crack grooves (and the scaffold's
// relief gain).
#import rl::noise::{hash2, rand01, vnoise, footprint_fade, voronoi, Voro}
#import rl::ground::art::{GroundCtx, GroundArt, default_art}

// Per-cell identity in [0, 1) from a Voro hit — the plate hue/value driver.
fn cell_id(v: Voro, seed: u32) -> f32 {
    return rand01(hash2(v.id, seed));
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
    let hue = cell_id(plate, 94u);
    let tilt = vec3(
        0.70 + 0.75 * hue,
        0.88 + 0.24 * abs(hue - 0.5),
        1.25 - 0.70 * hue,
    );
    let plate_w = strengths.x * mix(1.0, 0.65, veg);
    rgb *= mix(vec3(1.0), tilt * (0.82 + 0.5 * fract(hue * 7.31)), plate_w);
    // Seams: darker, damper ground between plates — wide soft shoulder plus a
    // tight dark core, both meaningful from altitude.
    let seam = (1.0 - smoothstep(0.0, 0.16, plate.edge)) * 0.55
        + (1.0 - smoothstep(0.0, 0.05, plate.edge)) * 0.45;
    rgb *= 1.0 - 0.42 * seam * plate_w;

    // ── Tier 2: meso plates (~14 m) — the claw-height stepping-stone read ──
    let meso = voronoi(p / 14.0 + warp * 3.0, 92u);
    let meso_fade = footprint_fade(14.0, fw);
    rgb *= 1.0 + (0.16 * (cell_id(meso, 95u) - 0.5) - 0.22 * (1.0 - smoothstep(0.0, 0.10, meso.edge)))
        * strengths.x * meso_fade;

    // ── Tier 3: mudcracks (~1.2 m) — the on-foot read ──────────────────────
    // Fine isotropic grain is the scaffold's always-on layer (grain = 1 below).
    let crack_fade = footprint_fade(1.2, fw);
    var crack = 0.0;
    if crack_fade > 0.001 {
        let mud = voronoi(p / 1.2, 93u);
        crack = (1.0 - smoothstep(0.0, 0.09, mud.edge)) * crack_fade;
        // Cracks open on dry open ground, close under vegetation.
        crack *= mix(1.0, 0.35, veg);
        rgb *= 1.0 - 0.38 * strengths.z * crack;
        // Slight per-polygon value so shards read individually.
        rgb *= 1.0 + 0.10 * (cell_id(mud, 96u) - 0.5) * strengths.z * crack_fade;
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
        let c0 = voronoi(p / 1.2, 93u).edge;
        let cx = voronoi((p + vec2(gs, 0.0)) / 1.2, 93u).edge;
        let cz = voronoi((p + vec2(0.0, gs)) / 1.2, 93u).edge;
        // Depress the surface where F2−F1 shrinks: normals lean INTO cracks.
        grad += vec2(cx - c0, cz - c0) / gs * 0.10 * groove_w;
    }
    var n = ctx.n;
    if length(grad) > 1e-4 {
        n = normalize(n + vec3(-grad.x, 0.0, -grad.y));
    }

    var out = default_art(ctx);
    out.rgb = rgb;
    out.n = n;
    return out;
}
