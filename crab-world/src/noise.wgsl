// rl::noise — the one copy of the procedural-noise helpers (rl#333 seam 1).
// Every ground look and sky.wgsl import from here.
// History's warning, and this module's reason to exist: as per-file copies these
// drifted twice (the rl#324 quintic fix reached 1 of 14 looks) — a fix here is
// fleet-wide by construction.

#define_import_path rl::noise

// Same integer-hash family as the Rust side's sky/terrain jitter (sky.rs hash3).
fn hash2(p: vec2<i32>, seed: u32) -> u32 {
    var h = bitcast<u32>(p.x) * 0x8da6b343u ^ bitcast<u32>(p.y) * 0xd8163841u ^ seed * 0xcb1ab31fu;
    h ^= h >> 13u;
    h *= 0x165667b1u;
    h ^= h >> 16u;
    return h;
}

// 3D sibling; constants match sky.rs's hash3 (bitcast matches Rust's `as u32` on
// negative cells).
fn hash3(v: vec3<i32>) -> u32 {
    var h = bitcast<u32>(v.x) * 0x8da6b343u
        ^ bitcast<u32>(v.y) * 0xd8163841u
        ^ bitcast<u32>(v.z) * 0xcb1ab31fu;
    h ^= h >> 13u;
    h *= 0x165667b1u;
    h ^= h >> 16u;
    return h;
}

fn rand01(h: u32) -> f32 {
    return f32(h & 0xffffffu) / f32(0x1000000u);
}

// Value noise in [-1, 1]. Quintic (C2) interpolant: smoothstep's discontinuous
// second derivative creases along every lattice edge, and the rl#324 close-up
// octaves are large enough on screen to read those creases as a woven grid.
fn vnoise(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
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

// The adaptive-descent fade (rl#324): wider margins than footprint_fade, so the
// finest octave the descent keeps still spans ~5+ pixels per wavelength. Letting
// octaves ride down toward Nyquist (footprint_fade's window) turns the fine tail
// into per-pixel salt noise — grain must stay coarse enough to read as surface.
// Window tuning (rl#353 stage 7): an 8 px floor caps the finest visible grain
// on foot at ~4 cm, which reads as low-frequency up close — this window (2.5×
// edge ratio) trades down to ~5.3 px, a third-finer near field, still above
// the salt line.
fn grain_fade(wavelength: f32, fw: f32) -> f32 {
    return 1.0 - smoothstep(wavelength * 0.075, wavelength * 0.1875, fw);
}

// Per-octave lattice rotation, 55.62° (90°/φ): value noise's lattice is
// axis-aligned, and any octave landing near a multiple of 90° reads as a woven
// plaid — 90°/φ keeps every small multiple ≥13° away from the axes (the golden
// angle itself fails: 2×137.5° mod 90° = 5°, a visibly axis-aligned octave).
const ROT: mat2x2<f32> = mat2x2<f32>(0.5646, 0.8254, -0.8254, 0.5646);

// Anisotropic value noise: stretched to `along`×`across` meters in frame `d`,
// so one sample is a streak, not a blot.
fn streak(p: vec2<f32>, d: vec2<f32>, along: f32, across: f32, seed: u32) -> f32 {
    let q = vec2(dot(p, d) / along, dot(p, vec2(-d.y, d.x)) / across);
    return vnoise(q, seed);
}

// The F1 member is `dist`, not `f1`: naga_oil 0.20 breaks a cross-module member
// named `f1` — importers fail naga validation with "invalid field accessor `f1`"
// (the in-module copies this fold replaced never crossed a module edge, so the
// name was safe there).
struct Voro {
    dist: f32, // F1: distance to nearest feature point (cell units)
    edge: f32, // F2-F1: ~0 on a cell border
    id: vec2<i32>,
}

// Jittered-grid Voronoi, 3x3 search (F2 correctness needs the full ring).
// Per-cell identity is `id`: hash it with a caller-owned seed for stable
// per-cell tones (e.g. rand01(hash2(v.id, seed))).
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

// The shared wind-direction field (seeds 61u/62u — one wind for every look that
// reads it): a slowly turning angle, bent onto the slope contour as faces
// steepen (wind and gravity comb the same way there). Contour is sign-ambiguous
// — align it with the wind before blending so the two never cancel.
// cross(up, N).xz = (N.z, -N.x).
fn wind_dir(p: vec2<f32>, n_geo: vec3<f32>) -> vec2<f32> {
    let wind_a = 3.14159265 * (vnoise(p / 700.0, 61u) * 0.7 + vnoise(p / 230.0, 62u) * 0.55);
    var d = vec2(cos(wind_a), sin(wind_a));
    let contour_w = smoothstep(0.04, 0.22, 1.0 - n_geo.y);
    if contour_w > 0.001 {
        var cd = vec2(n_geo.z, -n_geo.x);
        let cl = length(cd);
        if cl > 1e-4 {
            cd = cd / cl;
            if dot(cd, d) < 0.0 {
                cd = -cd;
            }
            d = normalize(mix(d, cd, contour_w));
        }
    }
    return d;
}

// Dew glints: sparse bright points on a 0.4 m jittered grid, cool moonlit white.
// Additive POST-lighting radiance — a glint is a glint, not a brighter patch of
// albedo. The grid size and its footprint fade live HERE, coupled — a caller
// repeating the 0.4 for its own fade is the drift this module exists to kill.
// The caller passes `mask` (strength × its drowning masks, e.g. puddles) and
// `boost` (its snow/moisture emphasis); both gates keep the miss case to one hash.
fn sparkle(p: vec2<f32>, n: vec3<f32>, v: vec3<f32>, fw: f32, mask: f32, boost: f32) -> vec3<f32> {
    let spark_f = footprint_fade(0.4, fw) * mask;
    if spark_f <= 0.001 {
        return vec3(0.0);
    }
    let cell = vec2<i32>(floor(p / 0.4));
    let h = rand01(hash2(cell, 118u));
    if h <= 0.985 {
        return vec3(0.0);
    }
    let h2 = hash2(cell, 122u);
    let jp = vec2(rand01(h2), rand01(h2 ^ 0x9e3779b9u));
    let star = 1.0 - smoothstep(0.0, 0.12, length(fract(p / 0.4) - jp));
    let ndv = max(dot(n, v), 0.0);
    let amt = spark_f * star * ndv * (h - 0.985) / 0.015 * boost;
    return vec3(0.70, 0.78, 0.92) * amt;
}
