// rl::ground::detail — the always-on high-frequency ground layer (rl#333 seam 2):
// the rl#324 adaptive descents. One copy, called only by the ground scaffold
// (ground.wgsl), so every look composes with it by construction and a taste pass
// on on-foot detail is one edit, fleet-wide.

#define_import_path rl::ground::detail

#import rl::noise::{vnoise, footprint_fade, grain_fade, ROT}

// Fine color detail (meters and below): the on-foot / landing-height optic-flow cue.
// ADAPTIVE descent (rl#324): a fixed finest octave IS a resolution — its lattice
// reads as an upscaled low-res texture the moment the camera gets closer than it
// was tuned for. So descend by thirds from 2.6 m until the octave is subpixel
// (footprint-faded to nothing), down to a 3.5 mm floor: finer than any standing
// eye can resolve (~2 mm ground per pixel on foot at 720p). The anchor-relative
// p (rl#334) is what makes a floor this fine tenable — raw world xz quantizes
// at ~1-2 mm out at the corners, a large fraction of such a lattice cell.
// Distant ground exits after one test — fw alone decides, so the loop is
// quad-coherent and the near-fullscreen far ground pays nothing.
// Each octave rotated by ROT from the last (why that angle: rl::noise).
// `gain` is the scaffold's composed grain strength (`strengths.z * art.grain`)
// — inside the function, like relief_normal's, so both seams share one contract
// (pass the composed gain, get the finished contribution) and a zero-gain look
// skips the descent entirely instead of paying it and multiplying by nothing.
fn fine_color(p: vec2<f32>, fw: f32, gain: f32) -> f32 {
    // Exact-zero gate, not a relief_normal-style threshold: a threshold on
    // gain*fade drops sub-LSB contributions and breaks this module's
    // identical-output guarantee (measured: 188 px shift 1 LSB at 0.001). The
    // subpixel early-out is already the loop's first break; this only spares a
    // gain=0 look the descent.
    if gain == 0.0 {
        return 0.0;
    }
    var fine_n = 0.0;
    var q = p;
    var wl = 2.6;
    var amp = 1.0;
    var seed = 31u;
    while wl > 0.0035 {
        let fade = grain_fade(wl, fw);
        if fade <= 0.001 {
            break;
        }
        fine_n += vnoise(q / wl, seed) * amp * fade;
        q = ROT * q;
        wl /= 3.0;
        // 0.82/octave (rl#353 stage 7): a steeper decay peaks the spectrum at
        // the 2.6 m octave and on-foot ground reads as meter-scale blotch with
        // faint grain. The slower decay tilts weight toward the fine octaves —
        // and raises total near-field contrast with it, which is intended.
        amp *= 0.82;
        seed += 1u;
    }
    return gain * fine_n;
}

// Detail normal from the fine-noise heightfield gradient: micro-relief up
// close, faded with the same footprint rule so distant ground keeps the smooth
// geometric normal. Same adaptive descent as the color detail (rl#324): a
// relief lattice that stops at a fixed wavelength is the "low-res normal map"
// the eye catches on foot, so descend by thirds from 45 cm until subpixel
// (same 3.5 mm floor / f32-ulp rationale as fine_color). The finite-difference
// step scales with the wavelength so each octave's gradient stays honest;
// per-octave gradient weight falls slower than the color amplitude (relief keeps
// more of its punch as it refines — pebbles under moonlight, not blur).
// `gain` is the scaffold's composed relief strength (`strengths.w * art.relief`);
// at or below the fade threshold the incoming normal passes through untouched.
fn relief_normal(p: vec2<f32>, fw: f32, n: vec3<f32>, gain: f32) -> vec3<f32> {
    if gain * footprint_fade(0.45, fw) <= 0.001 {
        return n;
    }
    var grad = vec2(0.0);
    // Cumulative rotation for this octave's lattice; its transpose (== inverse)
    // carries the octave's gradient back to world space.
    var nrot = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    var nwl = 0.45;
    var nweight = 0.06;
    var nseed = 51u;
    while nwl > 0.0035 {
        let fade = grain_fade(nwl, fw);
        if fade <= 0.001 {
            break;
        }
        let nq = nrot * p;
        let step = nwl * 0.27;
        let h0 = vnoise(nq / nwl, nseed);
        let hx = vnoise((nq + vec2(step, 0.0)) / nwl, nseed);
        let hz = vnoise((nq + vec2(0.0, step)) / nwl, nseed);
        grad += fade * (transpose(nrot) * vec2(hx - h0, hz - h0)) / step * nweight;
        nrot = ROT * nrot;
        nwl /= 3.0;
        nweight *= 0.45;
        nseed += 1u;
    }
    // Soft-limit the stacked slope: unbounded octave sums tilt the normal past
    // grazing and light it as black speckle pits.
    grad *= 1.0 / (1.0 + 0.8 * length(grad));
    return normalize(n + gain * vec3(-grad.x, 0.0, -grad.y));
}
