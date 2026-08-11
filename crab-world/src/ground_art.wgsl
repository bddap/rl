// rl::ground::art — the look contract (rl#333 seam 3): the context the ground
// scaffold (ground.wgsl) hands every look's `art()`, and the surface a look hands
// back. A look never owns `fn fragment` — it cannot skip the always-on detail
// layer, only modulate it through the typed `grain`/`relief` fields, and a full
// override is one explicit, reviewable line (`out.grain = 0.0`).

#define_import_path rl::ground::art

// The shipped-look strengths defaults (`GroundDetail::new`, ground.rs — the one
// Rust source). Looks that tune their constants at S = 1 normalize against this
// so a lane reads as a multiplier around the designed look; ONE copy here, since
// a per-look copy would silently rescale that look when the default moves.
const STRENGTH_DEFAULTS: vec4<f32> = vec4<f32>(0.55, 0.35, 0.45, 0.6);

struct GroundCtx {
    // Ground-plane meters, ANCHOR-relative (rl#334/rl#354): the terrain mesh's
    // entity is translated by −anchor, so this varying is small — hence precise —
    // near play. Raw world xz would quantize at ~1-2 mm out at the tile corners,
    // the fine octaves' own scale.
    p: vec2<f32>,
    // world_position.xyz; y is ABSOLUTE datum-shifted elevation (the anchor is
    // xz-only) — coarse elevation bands key on it.
    wp: vec3<f32>,
    // Ground meters per pixel at this fragment — the octave-fade driver.
    fw: f32,
    // The same footprint measure along elevation, for strata banded on wp.y.
    fw_y: f32,
    // The smooth geometric normal (normalize(world_normal)) — mask driver.
    n_geo: vec3<f32>,
    // The lighting normal (pbr_input.N, front-facing resolved) — what a look's
    // structural relief perturbs; returned via GroundArt.n.
    n: vec3<f32>,
    // The view vector (pbr_input.V) — glint/sheen driver.
    v: vec3<f32>,
    // The biome tint (vertex COLOR pre-multiplied into base_color) — the palette
    // every look grades over.
    base: vec3<f32>,
    // Vegetation mask from the biome tint's greenness — growth vs mineral ground.
    veg: f32,
    // Snow mask mirrored from terrain.rs `biome` (SNOWLINE_M / SNOW_HOLD_STEEP).
    snow: f32,
    // The material's base perceptual roughness.
    rough: f32,
    // The hydrology bake sample at this fragment (moisture.rs, rl#323):
    // R wetness, G standing water.
    hydro: vec4<f32>,
    // The strengths uniform (binding 100) — four per-look intensity lanes whose
    // SHIPPED-look meanings are x macro, y meso, z fine grain, w relief. z and w
    // are additionally the scaffold's detail-layer gains (composed with the
    // look's grain/relief fields); a look with structure of its own may also
    // read any lane as its intensity bucket — the divergent lane meanings are
    // documented in each look's header.
    strengths: vec4<f32>,
}

struct GroundArt {
    // Graded albedo.
    rgb: vec3<f32>,
    roughness: f32,
    // Look-level normal (strata ledges, cobble bulges…) — the scaffold's
    // micro-relief layer perturbs on top of this.
    n: vec3<f32>,
    // Pre-lighting emissive (night-bloom veins, watershed bloom).
    emissive: vec3<f32>,
    // Post-lighting additive radiance (dew/twinkle glints via sparkle(), aurora
    // wash): added after apply_pbr_lighting, before
    // main_pass_post_lighting_processing — where today's adds sit.
    glow: vec3<f32>,
    // Detail-layer color gain, multiplied into strengths.z; 1.0 = on.
    grain: f32,
    // Detail-layer normal gain, multiplied into strengths.w; 1.0 = on.
    relief: f32,
}

// The identity surface: the biome-tinted ground untouched, detail layer at full
// strength. Every look STARTS from this (`var out = default_art(ctx);`) so a
// field it does not set means "unchanged", never "zeroed" — a bare
// `var out: GroundArt` zero-inits, which would silently disable the detail
// layer (grain = 0) and feed a zero normal to lighting: exactly the two states
// this contract exists to make unwritable-by-accident.
fn default_art(ctx: GroundCtx) -> GroundArt {
    var out: GroundArt;
    out.rgb = ctx.base;
    out.roughness = ctx.rough;
    out.n = ctx.n;
    out.emissive = vec3(0.0);
    out.glow = vec3(0.0);
    out.grain = 1.0;
    out.relief = 1.0;
    return out;
}
