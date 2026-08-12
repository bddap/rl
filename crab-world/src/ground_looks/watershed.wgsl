// Ground look: WATERSHED — the other six as ground states of ONE world.
// The six competition entries were submitted as six worlds. They read as six
// worlds because each let a DIFFERENT FIELD drive the surface. Given one shared
// field stack they stop competing: each becomes the appearance of a different
// ground STATE in one place.
//
//   STATES — a soft partition; a fragment is in exactly one
//   slope                  →  bare streaked scree
//   moisture, dry end      →  cracked clay plates, cobble  cracked_loam
//   moisture, wet end      →  soaked basins, puddles
//   moisture, wettest+veg  →  bioluminescent vein webs     night_bloom
//
//   FIELDS — global modulations that ride ON the partition
//   wind direction         →  combed growth, scree flow    wind_combed
//   macro cell provinces   →  polygonal geology from air   cracked_loam ∪
//                                                          patterned_ground
//
// Moisture is the spine, and it is HYDROLOGICAL (rl#323): a bake over the same
// height grid the mesh and collider come from — priority-flood ponding, D8 flow
// accumulation, topographic wetness index (moisture.rs) — sampled by the
// scaffold as a world-mapped texture (ctx.hydro). Water literally pools at the
// local minima the player can see and follows the drainage lines between them,
// so the wet/dry map a player reads from the plane IS the terrain silhouette's
// consequence. That is what makes six languages read as one place instead of
// six wallpapers: nothing is decorative, every language is the visible
// consequence of where water is.
//
// The partition is also the optimization. A fragment pays only for the state it
// is in (region weights branch-gate each language), and the regions are hundreds
// of meters wide, so a warp is almost always uniform across one branch.
//
// patterned_ground is deliberately NOT a seventh language here: it is the same
// domain-warped Voronoi mosaic as cracked_loam at a different scale, and
// cracked_loam's carries the crack seams that couple to moisture. Its
// contribution survives as the macro province hue below.
//
// One of the interchangeable `art()` modules in this directory; the contract —
// GroundCtx in, GroundArt out, the scaffold owns `fn fragment` and the detail
// layer — lives in rl::ground::art (ground_art.wgsl) and on `GroundLook`
// (ground.rs). Designs B (naturalist) and C (nocturne) are param rows over this
// one module (`GroundLook::params`), never forks. Watershed's own row turns
// every design axis on: each use below multiplies or gates on a lane.

#define_import_path rl::ground::looks::watershed

// strengths lanes here: x macro provinces + moisture contrast, y meso structure
// (plates/comb/scree flow), z near-field (cobble/fiber/dew, and the scaffold's
// grain gain), w detail normal (cobble, and the scaffold's relief gain)
// — all normalized to S = strengths / STRENGTH_DEFAULTS.
#import rl::noise::{hash2, rand01, vnoise, footprint_fade, sparkle, streak, voronoi, cell_rand, vein, wind_dir}
#import rl::ground::art::{GroundCtx, GroundArt, STRENGTH_DEFAULTS, default_art}

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let wp = ctx.wp;
    let p = ctx.p;
    let fw = ctx.fw;

    let base = ctx.base;
    var rgb = base;
    var rough = ctx.rough;

    // Lane strengths normalized to the shipped defaults, so every constant below
    // is tuned at S = 1 and a lane reads as a multiplier around the designed look
    // rather than a fraction of it.
    let S = ctx.strengths / STRENGTH_DEFAULTS;

    // The design axes (header comment above): uniform, so every gate below is
    // uniform control flow and a disabled language costs nothing.
    let bloom_gain = params[0].x;
    let cellular = params[0].y;
    let hue_tilt = params[0].z;

    // ── The shared field stack ─────────────────────────────────────────────
    let n_geo = ctx.n_geo;
    let steep = 1.0 - n_geo.y;
    let veg = ctx.veg;
    let snow = ctx.snow;
    let stony = smoothstep(-500.0, 100.0, wp.y);       // DRY_GRASS_M → SCREE_M

    // MOISTURE — the spine, from the scaffold's hydrology-bake sample. Noise
    // only frays the wet/dry boundary — its gain peaks at the shoreline and dies
    // in the cores, so it can never move water uphill or invent a pond.
    let hydro = ctx.hydro;
    let standing = hydro.g;
    let mwarp = vec2(vnoise(p / 300.0, 111u), vnoise((p + vec2(5.1, 1.7)) / 300.0, 112u)) * 90.0;
    let pm = p + mwarp;
    let m_fbm = vnoise(pm / 150.0, 114u) * 0.5 + vnoise(pm / 48.0, 115u) * 0.5;
    let fray = hydro.r * (1.0 - hydro.r) * 4.0;
    let moist = clamp(hydro.r + (m_fbm - 0.5) * 0.55 * fray, 0.0, 1.0) * (1.0 - snow);

    // WIND — the shared direction field (rl::noise wind_dir): the same wind,
    // by construction, that combs wind_combed.
    let wind_d = wind_dir(p, n_geo);

    // ── The partition ──────────────────────────────────────────────────────
    // Three ground states, weights summing to 1. Soft everywhere, so the
    // transitions are the blend and no language ever has a visible border.
    let rock_w = smoothstep(0.20, 0.45, steep) * (1.0 - snow);
    let soil_w = 1.0 - rock_w;
    let wet_w = soil_w * moist;
    let dry_w = soil_w * (1.0 - moist);
    // Inside dry ground, turf combs and bare clay cracks — the two structural
    // languages are mutually exclusive rather than cross-hatched. The threshold
    // is low on purpose: anything the biome tints green at all is growth, so the
    // mosaic is confined to genuinely mineral ground (scree, rock rim) instead
    // of tiling the meadows.
    let turf = smoothstep(0.05, 0.30, veg);

    var grad = vec2(0.0);

    // ── Global grade + macro provinces ─────────────────────────────────────
    // One nocturne grade for the whole world: everything below sits in it, so
    // six palettes never argue about the time of day.
    rgb = mix(rgb, rgb * vec3(0.88, 1.00, 1.10), 0.45);

    // Polygonal geology provinces (420 m) — the faint patchwork-from-altitude
    // read. Per-cell tone and hue washed with plain noise so borders never look
    // vector-hard. This is the one place fable-3's per-plate identity lives —
    // `hue_tilt` is its own axis (Design B keeps the provinces, drops the hue),
    // and with `cellular` off the macro tone falls back to the plain noise wash.
    var m_tone = vnoise(p / 260.0, 103u);
    var prov_hue = 0.0;
    if cellular > 0.5 {
        let pv = voronoi(p / 420.0, 100u);
        let prov = cell_rand(pv, 101u) - 0.5;
        prov_hue = (cell_rand(pv, 102u) - 0.5) * hue_tilt;
        m_tone = mix(m_tone, prov * 1.6, 0.6);
    }
    m_tone *= S.x;
    rgb *= 1.0 + 0.18 * m_tone;
    rgb *= vec3(1.0 + 0.08 * prov_hue * S.x, 1.0, 1.0 - 0.08 * prov_hue * S.x);

    // ── FIELD: combed growth — the wind made visible on turf ───────────────
    // Not a state: wind combs grass wherever grass grows, wet or dry, so this
    // rides ON the partition rather than inside it (soaked turf just lies
    // flatter). Anisotropic octaves, faded on their ACROSS wavelength — the axis
    // that can shimmer; along-streak variation is too slow to.
    let comb_w = soil_w * turf * S.y * mix(1.0, 0.65, moist);
    if comb_w > 0.003 {
        let comb = streak(p, wind_d, 70.0, 9.0, 71u) * 0.6 * footprint_fade(9.0, fw)
            + streak(p, wind_d, 24.0, 3.0, 72u) * 0.4 * footprint_fade(3.0, fw);
        rgb *= 1.0 + 0.36 * comb * comb_w;
        // Straw combs lighter, green combs darker: a hue tilt, not a gray wash.
        rgb *= vec3(1.0 + 0.12 * comb * comb_w, 1.0 + 0.04 * comb * comb_w, 1.0 - 0.08 * comb * comb_w);
    }

    // ── STATE: rock — bare scree on the steeps ─────────────────────────────
    // The biome's own rock tint carries the color; the wind field on rock
    // becomes downslope scree streaking, so the steeps share the meadows'
    // directional language.
    if rock_w > 0.003 {
        let flow = streak(p, wind_d, 26.0, 3.2, 67u) * footprint_fade(3.2, fw);
        rgb *= 1.0 + 0.22 * flow * rock_w * S.y;
        rough = mix(rough, 0.95, rock_w * 0.5);
    }

    // ── STATE: dry soil — cracked plates and cobble ────────────────────────
    // The whole structural (Voronoi) tier sits behind `cellular`: Design C's dry
    // ground is dust, grain, and comb only — "zero Voronoi" is its cost story as
    // much as its look.
    if dry_w > 0.003 && cellular > 0.5 {
        // Plates (8 m), domain-warped so no grid ever reads. Seam width grows
        // with dryness: hairlines under turf, crevices in bare clay.
        let bare = dry_w * (1.0 - turf) * (1.0 - snow);
        let warp = vec2(vnoise(p / 34.0, 91u), vnoise((p + vec2(7.3, 3.1)) / 34.0, 92u)) * 2.6;
        let mv = voronoi((p + warp) / 8.0, 93u);
        let crack_w = mix(0.05, 0.16, 1.0 - moist);
        let seam = (1.0 - smoothstep(0.0, crack_w, mv.edge)) * S.y * bare * footprint_fade(1.0, fw);
        rgb *= 1.0 - 0.62 * seam;
        // Seams are shadowed red-brown soil, not gray paint.
        rgb = mix(rgb, rgb * vec3(1.15, 0.80, 0.70), clamp(seam, 0.0, 1.0) * 0.35);
        // Per-plate patchwork + a lifted center (dried clay curls, turf crowns).
        // Weighted toward bare ground so turf keeps only a whisper of the mosaic
        // — a cell tone under grass, never a honeycomb over it.
        let meso_f = footprint_fade(8.0, fw) * dry_w * S.y * mix(0.22, 1.0, 1.0 - turf);
        let ch = cell_rand(mv, 94u) - 0.5;
        rgb *= 1.0 + 0.30 * ch * meso_f;
        rgb *= 1.0 + 0.14 * (1.0 - smoothstep(0.0, 0.45, mv.dist)) * meso_f;

        // Cobble (0.55 m): fist-sized stones underfoot on stony/bare ground.
        let cob_f = footprint_fade(0.55, fw) * S.z * mix(0.45, 1.0, max(stony, 1.0 - turf)) * bare;
        if cob_f > 0.003 {
            let cwarp = vec2(vnoise(p / 1.7, 96u), vnoise((p + vec2(3.7, 9.2)) / 1.7, 97u)) * 0.18;
            let cp = (p + cwarp) / 0.55;
            let cv = voronoi(cp, 98u);
            rgb *= 1.0 - 0.55 * cob_f * (1.0 - smoothstep(0.0, 0.22, cv.edge));
            rgb *= 1.0 + 0.22 * cob_f * (1.0 - smoothstep(0.0, 0.5, cv.dist));
            rgb *= 1.0 + 0.24 * cob_f * (cell_rand(cv, 99u) - 0.5);
            // Stones bulge: height falls with f1, so the normal tilts outward
            // from each stone's center down the f1 gradient.
            let w_n = S.w * cob_f;
            if w_n > 0.001 {
                let step = 0.14;
                let fx = voronoi(cp + vec2(step / 0.55, 0.0), 98u).dist;
                let fz = voronoi(cp + vec2(0.0, step / 0.55), 98u).dist;
                grad -= (vec2(fx, fz) - cv.dist) / step * 0.05 * w_n;
            }
        }
    }
    // Dry ground drinks light: pale dust, high roughness. Outside the cellular
    // gate — a dry slope reads dry in every design, structured or not.
    if dry_w > 0.003 {
        rgb *= mix(vec3(1.0), vec3(1.16, 1.11, 1.03), dry_w * 0.42 * S.x);
        rough = mix(rough, 1.0, dry_w * 0.6 * S.x);
    }

    // ── STATE: wet basin — saturated earth, puddles, mud hollows ───────────
    var pud = 0.0;
    if wet_w > 0.003 {
        // Value contrast carries at nadir; the roughness drop sells it at
        // grazing angles as a broad moon sheen.
        rgb *= mix(vec3(1.0), vec3(0.58, 0.63, 0.70), wet_w * 0.78 * S.x);
        // Floor the roughness well above a mirror. At grazing incidence — half
        // the on-foot frame, where footprint fade has already removed every
        // detail octave — a low-roughness face becomes one broad specular sheet
        // with nothing on it, which is exactly how a wet MEADOW must not read.
        // Only the puddle cores are allowed to go glassy.
        rough = mix(rough, 0.62, wet_w * 0.70 * S.x);

        // Puddles are the bake's standing water — basin cores that sit below
        // their spill point — not noise maxima, so every pool the plane view
        // shows is a real depression the silhouette agrees with. Fine noise
        // frays the shoreline; damp ground can still bead into faint pools.
        let pud_field = standing * 1.1 + moist * 0.12
            + 0.10 * vnoise(p / 6.0, 116u)
            + 0.05 * vnoise(p / 1.3, 117u) * footprint_fade(1.3, fw);
        let pud_core = smoothstep(0.46, 0.60, pud_field);
        pud = clamp(pud_core * S.x * wet_w, 0.0, 1.0);
        // Soaked rim just outside each pool — darker than either neighbor.
        let rim = clamp(smoothstep(0.32, 0.46, pud_field) - pud_core, 0.0, 1.0) * wet_w;
        rgb *= 1.0 - 0.45 * rim;
        rgb = mix(rgb, vec3(0.05, 0.065, 0.095), pud * 0.92);
        rough = mix(rough, 0.05, pud);

        // Mud hollows: meter-scale wet pockets that gloss dark against the dust
        // — the on-foot carrier of the wet story (water sits in hollows, not
        // only in lakes).
        let mh_f = footprint_fade(2.5, fw);
        if mh_f > 0.001 {
            let mh = vnoise(p / 2.5, 123u) * 0.7 + vnoise(p / 0.8, 124u) * 0.3;
            let wet_micro = smoothstep(0.15, 0.55, mh + (moist - 0.5)) * (1.0 - pud) * wet_w * mh_f;
            rgb *= 1.0 - 0.50 * wet_micro * S.z;
            rough = mix(rough, 0.18, wet_micro * 0.9 * S.z);
            // Silt banding inside the hollows, combed by the same wind field —
            // the near-field texture that keeps a soaked slope from going blank.
            let silt = streak(p, wind_d, 6.0, 0.9, 125u) * footprint_fade(0.9, fw);
            rgb *= 1.0 + 0.20 * silt * wet_w * S.z;
        }
    }

    // ── STATE: bloom — bioluminescence in the wettest living basins ────────
    // The one emissive element in the set, and the one that would own any frame
    // it appeared in. Gating it to wet AND vegetated AND low turns it from a
    // change of fiction into the payoff of the hydrology: life lights up where
    // the water collects. The veins share the moisture field's own domain warp
    // (`pm`), so the webs meander along the basins instead of across them.
    var emissive = vec3(0.0);
    let bloom_w = clamp(wet_w * veg * smoothstep(0.45, 0.85, moist) * (1.0 - pud), 0.0, 1.0)
        * S.x * bloom_gain;
    if bloom_w > 0.003 {
        let artery_n = vnoise(pm / 180.0, 72u) + 0.35 * vnoise(pm / 61.0, 71u);
        let capil_n = vnoise(pm / 14.0, 73u) + 0.4 * vnoise(p / 4.7, 74u);
        let artery = vein(artery_n, 0.10) * (0.4 + 0.6 * vein(artery_n, 0.035)) * bloom_w;
        let capil = vein(capil_n, 0.13) * footprint_fade(14.0, fw) * bloom_w * 0.8;
        // Cold teal web, magenta where artery crests knot; pale cyan spores.
        let knot = vein(vnoise(pm / 43.0, 77u), 0.12);
        let vein_col = mix(vec3(0.05, 0.85, 0.62), vec3(0.75, 0.10, 0.55), 0.55 * knot);
        var spore = 0.0;
        let spore_f = footprint_fade(0.8, fw) * S.z;
        if spore_f > 0.001 {
            let cell = vec2<i32>(floor(p / 0.8));
            if rand01(hash2(cell, 75u)) > 0.86 {
                spore = smoothstep(0.2, 0.75, vnoise(p / 0.26, 76u)) * spore_f * bloom_w;
            }
        }
        // The ground under the glow darkens (soaked soil), so the light reads as
        // coming FROM the ground rather than painted on it.
        rgb *= 1.0 - 0.55 * clamp(artery + capil, 0.0, 1.0);
        emissive = vein_col * (2.2 * artery + 1.0 * capil)
            + vec3(0.35, 0.85, 0.90) * 1.4 * spore;
    }

    // ── Near field: the structured tier the shared layer cannot carry ──────
    // Combed fiber on turf and grass clumps ride ON the scaffold's isotropic
    // grain/relief (grain = relief = dryish below — muted where wet, gone
    // inside water, the same modulation the hand-rolled octaves had).
    // Branch-gated so the near-fullscreen far ground never pays for it.
    let dryish = (1.0 - pud) * mix(1.0, 0.72, moist);
    let grain_f = footprint_fade(2.4, fw) * S.z * dryish;
    if grain_f > 0.003 {
        let g = (streak(p, wind_d, 3.4, 0.5, 81u)
            + 0.7 * streak(p, wind_d, 1.2, 0.18, 82u) * footprint_fade(0.18, fw)) * turf * 0.7;
        rgb *= 1.0 + 0.28 * g * grain_f;
        // Grass clumps: darker tufted patches where the ground grows.
        rgb *= 1.0 - 0.28 * grain_f * turf
            * smoothstep(0.15, 0.75, vnoise(p / 1.4, 34u)) * footprint_fade(1.4, fw);
    }

    var n = ctx.n;
    if dot(grad, grad) > 1e-8 {
        n = normalize(n + vec3(-grad.x, 0.0, -grad.y));
    }

    var out = default_art(ctx);
    out.rgb = rgb;
    out.roughness = rough;
    out.n = n;
    out.emissive = emissive;
    // Dew glints (rl::noise sparkle): near-field only, boosted on snow AND
    // moisture, drowned in puddles. Post-lighting additive radiance — the
    // scaffold adds it after apply_pbr_lighting. Computed on the look-level
    // normal: the scaffold's relief layer runs after art(), a known small drift
    // from the deleted in-look micro-relief the old glints saw.
    out.glow = sparkle(p, n, ctx.v, fw, S.z * (1.0 - pud), (0.35 + 0.65 * snow) * (0.35 + 0.65 * moist));
    out.grain = dryish;
    out.relief = dryish;
    return out;
}
