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
// finest octave the descent keeps still spans ~8+ pixels per wavelength. Letting
// octaves ride down toward Nyquist (footprint_fade's window) turns the fine tail
// into per-pixel salt noise — grain must stay coarse enough to read as surface.
fn grain_fade(wavelength: f32, fw: f32) -> f32 {
    return 1.0 - smoothstep(wavelength * 0.05, wavelength * 0.125, fw);
}

// Per-octave lattice rotation, 55.62° (90°/φ): value noise's lattice is
// axis-aligned, and any octave landing near a multiple of 90° reads as a woven
// plaid — 90°/φ keeps every small multiple ≥13° away from the axes (the golden
// angle itself fails: 2×137.5° mod 90° = 5°, a visibly axis-aligned octave).
const ROT: mat2x2<f32> = mat2x2<f32>(0.5646, 0.8254, -0.8254, 0.5646);

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
