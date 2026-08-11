// Ground look: WATERSHED — the other six as ground states of ONE world.
// The six competition entries were submitted as six worlds. They read as six
// worlds because each let a DIFFERENT FIELD drive the surface. Given one shared
// field stack they stop competing: each becomes the appearance of a different
// ground STATE in one place.
//
//   STATES — a soft partition; a fragment is in exactly one
//   slope                  →  exposed banded rock
//   moisture, dry end      →  cracked clay plates, cobble  cracked_loam
//   moisture, wet end      →  soaked basins, puddles       wet_nocturne
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
// one module (`GroundLook::params`), never forks. Watershed's own row is
// all-ones: every use below multiplies or gates on a lane, so Design A renders
// bit-identically to the pre-params shader.

#define_import_path rl::ground::looks::watershed

#import rl::noise::{hash2, rand01, vnoise, footprint_fade, sparkle}
#import rl::ground::art::{GroundCtx, GroundArt}

// Anisotropic value noise: stretched to `along`×`across` meters in frame `d`,
// so one sample is a streak, not a blot.
fn streak(p: vec2<f32>, d: vec2<f32>, along: f32, across: f32, seed: u32) -> f32 {
    let q = vec2(dot(p, d) / along, dot(p, vec2(-d.y, d.x)) / across);
    return vnoise(q, seed);
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

// 1 on the zero-set of a smooth field, falling off over `width`. The zero-set of
// smooth noise is a connected branching web — dendritic without any simulation.
fn vein(n: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(0.0, width, abs(n));
}

// The strata palette, cyclic: rust → ochre → bone → slate-violet → rust. Linear
// RGB, dim enough that moon-sun + ambient exposure lands them as deep mineral
// color, not paint.
fn strata_color(t: f32) -> vec3<f32> {
    let rust = vec3(0.44, 0.15, 0.07);
    let ochre = vec3(0.58, 0.35, 0.11);
    let bone = vec3(0.62, 0.54, 0.40);
    let slate = vec3(0.13, 0.12, 0.24);
    let u = fract(t) * 4.0;
    if u < 1.0 {
        return mix(rust, ochre, smoothstep(0.0, 1.0, u));
    } else if u < 2.0 {
        return mix(ochre, bone, smoothstep(0.0, 1.0, u - 1.0));
    } else if u < 3.0 {
        return mix(bone, slate, smoothstep(0.0, 1.0, u - 2.0));
    }
    return mix(slate, rust, smoothstep(0.0, 1.0, u - 3.0));
}

fn art(ctx: GroundCtx, params: array<vec4<f32>, 8>) -> GroundArt {
    let wp = ctx.wp;
    let p = ctx.p;
    let fw = ctx.fw;
    let fw_y = ctx.fw_y;

    let base = ctx.base;
    var rgb = base;
    var rough = ctx.rough;

    // Lane strengths normalized to the shipped defaults, so every constant below
    // is tuned at S = 1 and a lane reads as a multiplier around the designed look
    // rather than a fraction of it.
    let S = ctx.strengths / vec4(0.55, 0.35, 0.45, 0.6);

    // The design axes (header comment above): uniform, so every gate below is
    // uniform control flow and a disabled language costs nothing.
    let bloom_gain = params[0].x;
    let cellular = params[0].y;
    let strata_chroma = params[0].z;
    let hue_tilt = params[0].w;

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

    // WIND — the shared direction field: a slowly turning angle, bent onto the
    // slope contour as faces steepen (wind and gravity comb the same way there).
    // Contour is sign-ambiguous — align it with the wind before blending so the
    // two never cancel. cross(up, N).xz = (N.z, -N.x).
    let wind_a = 3.14159265 * (vnoise(p / 700.0, 61u) * 0.7 + vnoise(p / 230.0, 62u) * 0.55);
    var wind_d = vec2(cos(wind_a), sin(wind_a));
    let contour_w = smoothstep(0.04, 0.22, steep);
    if contour_w > 0.001 {
        var cd = vec2(n_geo.z, -n_geo.x);
        let cl = length(cd);
        if cl > 1e-4 {
            cd = cd / cl;
            if dot(cd, wind_d) < 0.0 {
                cd = -cd;
            }
            wind_d = normalize(mix(wind_d, cd, contour_w));
        }
    }

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
        let prov = rand01(hash2(pv.id, 101u)) - 0.5;
        prov_hue = (rand01(hash2(pv.id, 102u)) - 0.5) * hue_tilt;
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

    // ── STATE: rock — exposed banded sediment on the steeps ────────────────
    // Elevation-keyed strata, warped so layers buckle and fold. Confined to
    // slopes: this is the one palette in the set that is not green, and it lands
    // exactly where the biome tint is already rock, so it reads as the mountain
    // showing its bones rather than as paint spilled over the meadows.
    if rock_w > 0.003 {
        let warp = vnoise(p / 730.0, 63u) * 14.0 + vnoise(p / 173.0, 64u) * 5.0
            + vnoise(p / 47.0, 65u) * 1.8;
        let band_t = (wp.y + warp + (p.x + p.y) * 0.012) / 88.0;
        var strata_rgb = strata_color(band_t);
        // Design C: fable-1 reduced to value-only banding — the layers stay
        // legible, the one non-green palette stops arguing with the cool grade.
        strata_rgb = mix(
            vec3(dot(strata_rgb, vec3(0.2126, 0.7152, 0.0722))),
            strata_rgb,
            strata_chroma,
        );
        // Sub-layers inside each color band: the "close enough to count them" tier.
        let sub = vnoise(vec2(band_t * 32.0, (p.x - p.y) * 0.02), 66u);
        strata_rgb *= 1.0 + 0.28 * sub * footprint_fade(11.0, fw_y);
        // Luminance-matched so the strata replace hue/chroma but stay lit by the
        // same pre-exposed moonlight through the PBR pipe.
        let lum = 0.35 + 0.65 * length(base) / 0.55;
        rgb = mix(rgb, strata_rgb * lum, clamp(rock_w * S.y * 0.95 * mix(1.0, 0.5, veg), 0.0, 1.0));
        // Sediment flow-lines: the wind field on rock becomes downslope scree
        // streaking, so the steeps share the meadows' directional language.
        let flow = streak(p, wind_d, 26.0, 3.2, 67u) * footprint_fade(3.2, fw);
        rgb *= 1.0 + 0.22 * flow * rock_w * S.y;
        // Band-edge ledge relief: derivative of a soft sawtooth in the strata
        // coordinate, so grazing moonlight draws the layers without geometry.
        let ledge = footprint_fade(12.0, fw_y) * S.w * rock_w;
        if ledge > 0.001 {
            let down = normalize(vec2(n_geo.x, n_geo.z) + vec2(1e-5));
            grad += down * sin(band_t * 6.28318 * 4.0) * 0.05 * ledge * steep;
        }
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
        let ch = rand01(hash2(mv.id, 94u)) - 0.5;
        rgb *= 1.0 + 0.30 * ch * meso_f;
        rgb *= 1.0 + 0.14 * (1.0 - smoothstep(0.0, 0.45, mv.f1)) * meso_f;

        // Cobble (0.55 m): fist-sized stones underfoot on stony/bare ground.
        let cob_f = footprint_fade(0.55, fw) * S.z * mix(0.45, 1.0, max(stony, 1.0 - turf)) * bare;
        if cob_f > 0.003 {
            let cwarp = vec2(vnoise(p / 1.7, 96u), vnoise((p + vec2(3.7, 9.2)) / 1.7, 97u)) * 0.18;
            let cp = (p + cwarp) / 0.55;
            let cv = voronoi(cp, 98u);
            rgb *= 1.0 - 0.55 * cob_f * (1.0 - smoothstep(0.0, 0.22, cv.edge));
            rgb *= 1.0 + 0.22 * cob_f * (1.0 - smoothstep(0.0, 0.5, cv.f1));
            rgb *= 1.0 + 0.24 * cob_f * (rand01(hash2(cv.id, 99u)) - 0.5);
            // Stones bulge: height falls with f1, so the normal tilts outward
            // from each stone's center down the f1 gradient.
            let w_n = S.w * cob_f;
            if w_n > 0.001 {
                let step = 0.14;
                let fx = voronoi(cp + vec2(step / 0.55, 0.0), 98u).f1;
                let fz = voronoi(cp + vec2(0.0, step / 0.55), 98u).f1;
                grad -= (vec2(fx, fz) - cv.f1) / step * 0.05 * w_n;
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

    var out: GroundArt;
    out.rgb = rgb;
    out.roughness = rough;
    out.n = n;
    out.emissive = emissive;
    // Dew glints (rl::noise sparkle): near-field only, boosted on snow AND
    // moisture, drowned in puddles. Post-lighting additive radiance — the
    // scaffold adds it after apply_pbr_lighting.
    out.glow = sparkle(p, n, ctx.v, fw, S.z * (1.0 - pud), (0.35 + 0.65 * snow) * (0.35 + 0.65 * moist));
    out.grain = dryish;
    out.relief = dryish;
    return out;
}
