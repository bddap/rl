// Ground look: WET NOCTURNE (kimi competition variant 3).
// Art direction: moonlit WET ground. A macro moisture field splits the world
// into dark saturated earth (roughness drops — broad moon sheen at grazing
// angles) and pale dry dust; the wettest flat cores hold mirror puddles that
// catch the moon; on foot, dew glints prick the near field. The structure is
// VALUE and ROUGHNESS contrast, not hue patchwork — a cool nocturne grade
// over the biome tint. Elevation/slope masks mirror the biome band edges in
// terrain.rs `biome`.
//
// One of the interchangeable `art()` modules in this directory; the contract —
// GroundCtx in, GroundArt out, the scaffold owns `fn fragment` and the detail
// layer — lives in rl::ground::art (ground_art.wgsl) and on `GroundLook`
// (ground.rs).

#define_import_path rl::ground::looks::wet_nocturne

#import rl::noise::{vnoise, footprint_fade, sparkle}
#import rl::ground::art::{GroundCtx, GroundArt}

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let wp = ctx.wp;
    let p = ctx.p;
    let fw = ctx.fw;

    var rgb = ctx.base;

    // Lane strengths normalized to the shipped defaults (0.55/0.35/0.45/0.6) —
    // the constants below are tuned for S = 1 so a lane reads as a multiplier
    // around the designed look, not a fraction of it.
    let S = ctx.strengths / vec4(0.55, 0.35, 0.45, 0.6);

    // Wet styling never touches snow.
    let snow = ctx.snow;

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
    var rough = ctx.rough;
    rough = mix(mix(rough, 1.0, (1.0 - moist) * S.x), 0.35, moist * 0.75 * S.x);

    // Puddles: the wettest cores. The 48 m octave joins the mask so pools
    // exist at every frame scale (the macro field alone can miss a whole
    // mid-altitude frame); slopes past ~20° and high ground shed water but
    // are never fully barred. Fine noise frays the shoreline up close.
    let steep = 1.0 - ctx.n_geo.y;
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

    // Near-field grain: the rl#197 optic-flow duty — muted where wet (wet soil
    // reads smoother), gone inside puddles. (Kept for stage 4(a)'s
    // identical-output sweep; 4(b) deletes it and sets grain = dryish so the
    // scaffold's adaptive layer inherits the same modulation.)
    let dryish = (1.0 - pud) * mix(1.0, 0.45, moist);
    let fine = vnoise(p / 2.2, 119u) * footprint_fade(2.2, fw)
        + vnoise(p / 0.7, 120u) * 0.7 * footprint_fade(0.7, fw);
    rgb *= 1.0 + 0.22 * S.z * fine * dryish;

    // Micro-relief normal: pocked dry soil, smoothed by moisture, flat on water.
    var n = ctx.n;
    let w_n = S.w * footprint_fade(0.5, fw) * dryish;
    if w_n > 0.001 {
        let step = 0.12;
        let h0 = vnoise(p / 0.5, 121u);
        let hx = vnoise((p + vec2(step, 0.0)) / 0.5, 121u);
        let hz = vnoise((p + vec2(0.0, step)) / 0.5, 121u);
        let grad = vec2(hx - h0, hz - h0) / step * 0.05;
        n = normalize(n + w_n * vec3(-grad.x, 0.0, -grad.y));
    }

    var out: GroundArt;
    out.rgb = rgb;
    out.roughness = rough;
    out.n = n;
    out.emissive = vec3(0.0);
    // Dew glints (rl::noise sparkle): near-field only (footprint-faded), boosted
    // on snow, drowned in puddles. Post-lighting additive radiance — the
    // scaffold adds it after apply_pbr_lighting.
    out.glow = sparkle(p, n, ctx.v, fw, S.z * (1.0 - pud), 0.35 + 0.65 * snow);
    out.grain = 0.0;
    out.relief = 0.0;
    return out;
}
