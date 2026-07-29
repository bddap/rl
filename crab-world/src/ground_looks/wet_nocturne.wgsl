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

// ─── THIS LOOK: WET NOCTURNE (kimi competition variant 3) ───────────────────
// Art direction: moonlit WET ground. A macro moisture field splits the world
// into dark saturated earth (roughness drops — broad moon sheen at grazing
// angles) and pale dry dust; the wettest flat cores hold mirror puddles that
// catch the moon; on foot, dew glints prick the near field. The structure is
// VALUE and ROUGHNESS contrast, not hue patchwork — a cool nocturne grade
// over the biome tint. Elevation/slope masks mirror the biome band edges in
// terrain.rs `biome`.
// strengths lanes: x moisture/sheen contrast, y puddles, z near-field detail
// (grain + dew sparkle), w micro-relief normal.

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

// x: moisture/sheen, y: puddles, z: near-field detail + sparkle, w: detail normal.
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

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let wp = in.world_position.xyz;
    let p = wp.xz;
    // Ground meters per pixel at this fragment — the octave-fade driver.
    let fw = max(max(fwidth(p.x), fwidth(p.y)), 1e-4);

    var rgb = pbr_input.material.base_color.rgb;

    // Lane strengths normalized to the shipped defaults (0.55/0.35/0.45/0.6) —
    // the constants below are tuned for S = 1 so a lane reads as a multiplier
    // around the designed look, not a fraction of it.
    let S = strengths / vec4(0.55, 0.35, 0.45, 0.6);

    let n_geo = normalize(in.world_normal);
    let steep = 1.0 - n_geo.y;

    // Snow mask mirrored from terrain.rs `biome` (SNOWLINE_M / SNOW_HOLD_STEEP)
    // — wet styling never touches snow.
    let snow = smoothstep(350.0, 650.0, wp.y)
        * (1.0 - smoothstep(0.18, 0.38, steep));

    // Moisture: domain-warped fbm, kilometers to tens of meters — soaked
    // basins against dry rims, with a HARD-ish threshold so the wet/dry split
    // reads as distinct territories, not a uniform damp.
    let mwarp = vec2(vnoise(p / 300.0, 111u), vnoise((p + vec2(5.1, 1.7)) / 300.0, 112u)) * 90.0;
    let pm = p + mwarp;
    let m0 = vnoise(pm / 520.0, 113u) * 0.35
        + vnoise(pm / 150.0, 114u) * 0.30
        + vnoise(pm / 48.0, 115u) * 0.35;
    let moist = smoothstep(0.08, 0.38, m0) * (1.0 - snow);

    // Nocturne grade: cool everything.
    rgb = mix(rgb, rgb * vec3(0.85, 1.00, 1.15), 0.55);

    // Sheen split: dry lifts toward pale dust, wet sinks toward saturated dark
    // earth. The VALUE contrast is the carrier at nadir views; the roughness
    // drop sells it at grazing angles (broad moon sheen on wet ground).
    rgb *= mix(vec3(1.0), vec3(1.18, 1.13, 1.04), (1.0 - moist) * 0.45 * S.x);
    rgb *= mix(vec3(1.0), vec3(0.55, 0.60, 0.68), moist * 0.90 * S.x);
    var rough = pbr_input.material.perceptual_roughness;
    rough = mix(mix(rough, 1.0, (1.0 - moist) * S.x), 0.35, moist * 0.75 * S.x);

    // Puddles: the wettest cores. The 48 m octave joins the mask so pools
    // exist at every frame scale (the macro field alone can miss a whole
    // mid-altitude frame); slopes past ~20° and high ground shed water but
    // are never fully barred. Fine noise frays the shoreline up close.
    let flat = 1.0 - smoothstep(0.10, 0.35, steep);
    let low_favor = mix(1.0, 0.45, smoothstep(-500.0, 200.0, wp.y));
    let pud_field = moist * 0.7
        + 0.35 * vnoise(pm / 48.0, 115u)
        + 0.14 * vnoise(p / 6.0, 116u)
        + 0.07 * vnoise(p / 1.3, 117u) * footprint_fade(1.3, fw);
    let pud_core = smoothstep(0.46, 0.60, pud_field);
    let pud = clamp(pud_core * flat * low_favor * S.y * (1.0 - snow), 0.0, 1.0);
    // Soaked rim band just outside each puddle — darker than either neighbor.
    let rim = clamp((smoothstep(0.32, 0.46, pud_field) - pud_core), 0.0, 1.0)
        * flat * low_favor * S.y * (1.0 - snow);
    rgb *= 1.0 - 0.45 * rim;
    // Puddle body: deep blue-black with a whisper of night sky.
    rgb = mix(rgb, vec3(0.05, 0.065, 0.095), pud * 0.92);
    rough = mix(rough, 0.05, pud);

    // Mud hollows: meter-scale wet pockets that gloss dark against the dry
    // dust — the on-foot carrier of the wet story, slope-independent (water
    // sits in hollows, not only in lakes).
    let mh_f = footprint_fade(2.5, fw);
    if mh_f > 0.001 {
        let mh = vnoise(p / 2.5, 123u) * 0.7 + vnoise(p / 0.8, 124u) * 0.3;
        let wet_micro = smoothstep(0.15, 0.55, mh + (moist - 0.5)) * (1.0 - pud) * (1.0 - snow) * mh_f;
        rgb *= 1.0 - 0.45 * wet_micro * S.z;
        rough = mix(rough, 0.18, wet_micro * 0.9 * S.z);
    }
    pbr_input.material.perceptual_roughness = rough;

    // Near-field grain: the rl#197 optic-flow duty — muted where wet (wet soil
    // reads smoother), gone inside puddles.
    let dryish = (1.0 - pud) * mix(1.0, 0.45, moist);
    let fine = vnoise(p / 2.2, 119u) * footprint_fade(2.2, fw)
        + vnoise(p / 0.7, 120u) * 0.7 * footprint_fade(0.7, fw);
    rgb *= 1.0 + 0.22 * S.z * fine * dryish;

    // Micro-relief normal: pocked dry soil, smoothed by moisture, flat on water.
    let w_n = S.w * footprint_fade(0.5, fw) * dryish;
    if w_n > 0.001 {
        let step = 0.12;
        let h0 = vnoise(p / 0.5, 121u);
        let hx = vnoise((p + vec2(step, 0.0)) / 0.5, 121u);
        let hz = vnoise((p + vec2(0.0, step)) / 0.5, 121u);
        let grad = vec2(hx - h0, hz - h0) / step * 0.05;
        pbr_input.N = normalize(pbr_input.N + w_n * vec3(-grad.x, 0.0, -grad.y));
    }

    pbr_input.material.base_color = vec4(rgb, pbr_input.material.base_color.a);

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);

    // Dew glints: sparse bright points on a 0.4 m jittered grid, near-field
    // only (footprint-faded), boosted on snow, drowned in puddles. Added
    // post-lighting — a glint is a glint, not a brighter patch of albedo.
    let spark_f = footprint_fade(0.4, fw) * S.z * (1.0 - pud);
    if spark_f > 0.001 {
        let cell = vec2<i32>(floor(p / 0.4));
        let h = rand01(hash2(cell, 118u));
        if h > 0.985 {
            let h2 = hash2(cell, 122u);
            let jp = vec2(rand01(h2), rand01(h2 ^ 0x9e3779b9u));
            let dpt = length(fract(p / 0.4) - jp);
            let star = 1.0 - smoothstep(0.0, 0.12, dpt);
            let ndv = max(dot(pbr_input.N, pbr_input.V), 0.0);
            let amt = spark_f * star * ndv * (h - 0.985) / 0.015 * (0.35 + 0.65 * snow);
            out.color = vec4(out.color.rgb + vec3(0.70, 0.78, 0.92) * amt, out.color.a);
        }
    }

    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif
    return out;
}
