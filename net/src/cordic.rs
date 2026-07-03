//! Deterministic integer trigonometry for the sim, extracted from [`super::sim`].
//!
//! [`trig`] is the float-free integer-CORDIC sin/cos/atan2 surface the lockstep sim
//! itself calls (it feeds [`super::sim::Sim::state_hash`], so it must be bit-identical
//! across machines); [`trig_client`] is the client-only float adapter, kept beside it
//! but deliberately out of [`trig`] so "no float in the sim" stays enforced by where
//! code lives. [`super::sim`] re-exports both, so existing `sim::trig` paths still resolve.

/// Integer fixed-point trigonometry for the deterministic sim.
///
/// Angles are **turn units**: a full circle is [`TURN`] (a power of two, so wrapping
/// is a cheap mask and there's no modulo bias). Sine/cosine come from a quarter-wave
/// lookup table in fixed-point (scale [`ONE`]).
///
/// **The table is built with INTEGER-ONLY math (CORDIC), never `f64::sin`.** A
/// hardware/compiler float `sin` is not guaranteed bit-identical across platforms
/// (libm/FMA/rounding differ), so building the table from one would reintroduce a
/// cross-machine desync — and a unit test, being same-binary, could never catch it.
/// Integer CORDIC is a fixed sequence of adds and shifts: identical on every target, so
/// two peers on different builds/OSes get the same table. No `f32`/`f64` appears
/// anywhere in this module (client float helpers live in [`super::trig_client`]).
pub mod trig {
    use std::sync::OnceLock;

    /// Angle units in a full turn. Power of two so [`wrap_turns`] is a mask.
    pub const TURN: i32 = 1 << 16;
    /// Fixed-point scale of sine/cosine outputs: `ONE` represents 1.0.
    pub const ONE: i32 = 1 << 14;

    /// Quarter-turn entry count for the sine table (`TURN/4`), the resolution of the
    /// angle→sine lookup.
    const QUARTER: usize = (TURN / 4) as usize;

    /// Quarter-wave sine table, `round(sin(k/TURN · 2π) · ONE)` for `k` in
    /// `0..=QUARTER`, built once by integer CORDIC. Quarter-wave + symmetry
    /// reconstructs the full circle, so the table is small and the four quadrants share
    /// one set of values (no quadrant can drift from another). Inclusive of `QUARTER`
    /// so `cos` (a quarter-turn phase shift) can index the endpoint.
    fn sine_table() -> &'static [i32; QUARTER + 1] {
        static T: OnceLock<[i32; QUARTER + 1]> = OnceLock::new();
        T.get_or_init(|| {
            let mut t = [0i32; QUARTER + 1];
            for (k, slot) in t.iter_mut().enumerate() {
                *slot = cordic_sin(k as i32);
            }
            t
        })
    }

    /// Internal CORDIC precision: fractional bits carried during the rotation so the
    /// rounded table entries are exact. [`PREC`] bits below the turn-unit/[`ONE`]
    /// scales.
    const PREC: u32 = 28;
    /// CORDIC iteration count — enough that the residual angle is far below one table
    /// step, giving `round(sin·ONE)` exact to the last bit (asserted in tests).
    const ITERS: usize = 28;

    /// `round(atan(2^-i) / (2π) · TURN · 2^PREC)` for `i` in `0..ITERS`: the fixed
    /// CORDIC micro-rotation angles, in `2^PREC`-scaled turn units. Integer constants
    /// so the rotation uses no float at runtime. They are NOT trusted on faith —
    /// [`tests::cordic_constants_match_reference`] re-derives them from `f64` and
    /// asserts equality, so a typo fails the build rather than silently skewing every
    /// angle.
    const ATAN_TURNS: [i64; ITERS] = [
        2199023255552,
        1298159229407,
        685911684590,
        348179481054,
        174765388006,
        87467890064,
        43744617923,
        21873643805,
        10936988781,
        5468515251,
        2734260233,
        1367130443,
        683565262,
        341782636,
        170891319,
        85445659,
        42722830,
        21361415,
        10680707,
        5340354,
        2670177,
        1335088,
        667544,
        333772,
        166886,
        83443,
        41722,
        20861,
    ];
    /// CORDIC start X = `round(ONE / K · 2^PREC)`, where `K = Π sqrt(1 + 2^-2i)` is the
    /// CORDIC gain. Pre-dividing by the gain means the rotation's final Y is directly
    /// `sin(a)·ONE·2^PREC` with no post-scale. Also re-derived and checked in tests.
    const X0: i64 = 2670726652173;

    /// Fixed-point CORDIC sine of a first-quadrant turn-unit angle `a` in
    /// `[0, QUARTER]`, returning `round(sin(a) · ONE)`. Pure integer: rotate the vector
    /// `(ONE/K, 0)` by `a` through the fixed [`ATAN_TURNS`] micro-rotations; the final
    /// Y is `sin(a)·ONE`. Only adds, shifts, and compile-time constants, so it is
    /// bit-identical on every target — the property the table needs and a float `sin`
    /// can't guarantee.
    fn cordic_sin(a: i32) -> i32 {
        let target = (a as i64) << PREC; // angle to rotate to, in 2^PREC turn units
        let mut x: i64 = X0;
        let mut y: i64 = 0;
        let mut ang: i64 = 0;
        for (i, &step) in ATAN_TURNS.iter().enumerate() {
            let i = i as u32;
            // Rotate toward `target`: each step turns by ±atan(2^-i), driving `ang`
            // toward the target while `(x, y)` tracks `(cos, sin)·ONE/K·2^PREC`.
            if ang < target {
                let (nx, ny) = (x - (y >> i), y + (x >> i));
                x = nx;
                y = ny;
                ang += step;
            } else {
                let (nx, ny) = (x + (y >> i), y - (x >> i));
                x = nx;
                y = ny;
                ang -= step;
            }
        }
        let half = 1i64 << (PREC - 1); // round-to-nearest when descaling
        ((y + half) >> PREC) as i32
    }

    /// Wrap a turn-unit angle into `[0, TURN)`. Mask, since `TURN` is a power of two —
    /// branch-free and identical for negative inputs (two's-complement `& (TURN-1)`).
    pub fn wrap_turns(a: i32) -> i32 {
        a & (TURN - 1)
    }

    /// Sine of a turn-unit angle, fixed-point (scale [`ONE`]). Reconstructs the full
    /// circle from the quarter-wave table by quadrant.
    pub fn sin(a: i32) -> i32 {
        let a = wrap_turns(a) as usize;
        let q = QUARTER;
        match a / q {
            0 => sine_table()[a],
            1 => sine_table()[2 * q - a],  // sin(π−x)=sin x
            2 => -sine_table()[a - 2 * q], // sin(π+x)=−sin x
            _ => -sine_table()[4 * q - a], // sin(2π−x)=−sin x
        }
    }

    /// Cosine of a turn-unit angle, fixed-point: `cos a = sin(a + quarter turn)`.
    pub fn cos(a: i32) -> i32 {
        sin(a + TURN / 4)
    }

    /// `(sin, cos)` together (one wrap, two table hits) for movement-by-facing.
    pub fn sin_cos(a: i32) -> (i64, i64) {
        (sin(a) as i64, cos(a) as i64)
    }

    /// Turn-unit angle of the vector `(x, z)`, measured from +Z toward +X — the
    /// world's convention (a player at yaw 0 faces +Z). Pure integer (octant
    /// decomposition + a binary search over the sine table), so it's deterministic
    /// across machines where a float `atan2` need not be. Used to point the crab at
    /// its prey; exact enough for a facing.
    pub fn atan2_turns(x: i64, z: i64) -> i32 {
        if x == 0 && z == 0 {
            return 0;
        }
        // Angle from +Z toward +X means |x| is the opposite leg and |z| the adjacent,
        // so the first-quadrant magnitude is atan(|x|/|z|). Reduce to the lower octant
        // by taking theta = atan(min/max) ∈ [0, TURN/8], then reflect to TURN/4−theta
        // when |x| is the LARGER leg (the angle is past the 45° diagonal).
        let ax = x.unsigned_abs() as u128;
        let az = z.unsigned_abs() as u128;
        let (hi, lo) = if ax >= az { (ax, az) } else { (az, ax) };
        // theta = atan(lo/hi) in [0, TURN/8]: binary-search the largest k whose
        // tan(k) ≤ lo/hi, compared as sin(k)*hi ≤ cos(k)*lo (cross-multiplied — no
        // division). sin/cos are monotone over the quarter, so the search is exact.
        let mut lo_k = 0i32;
        let mut hi_k = TURN / 8;
        while lo_k < hi_k {
            let mid = (lo_k + hi_k + 1) / 2;
            let s = sin(mid) as i128 * hi as i128;
            let c = cos(mid) as i128 * lo as i128;
            if s <= c {
                lo_k = mid;
            } else {
                hi_k = mid - 1;
            }
        }
        let theta = lo_k; // atan(min/max) in [0, TURN/8]
        // Magnitude of the angle off the +Z axis, in [0, TURN/4]. atan(|x|/|z|): equals
        // theta when |x| ≤ |z|, else its complement.
        let in_q = if ax <= az { theta } else { TURN / 4 - theta };
        // Fold into the correct quadrant by the signs of (x, z). Convention: angle
        // measured from +Z toward +X.
        let folded = match (x >= 0, z >= 0) {
            (true, true) => in_q,             // +x +z: [0, TURN/4)
            (true, false) => TURN / 2 - in_q, // +x −z: (TURN/4, TURN/2]
            (false, false) => TURN / 2 + in_q,
            (false, true) => TURN - in_q,
        };
        wrap_turns(folded)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cordic_constants_match_reference() {
            // The pinned CORDIC constants are large magic integers; re-derive them from
            // f64 and assert equality, so a typo fails the build instead of silently
            // skewing every table entry. Float appears ONLY in this test — the runtime
            // table build (`cordic_sin`) is integer-only and identical on every target.
            for (i, &pinned) in ATAN_TURNS.iter().enumerate() {
                let want = ((2.0f64.powi(-(i as i32))).atan() / std::f64::consts::TAU
                    * TURN as f64
                    * (1u64 << PREC) as f64)
                    .round() as i64;
                assert_eq!(pinned, want, "ATAN_TURNS[{i}] wrong");
            }
            let k: f64 = (0..ITERS as i32)
                .map(|i| (1.0 + 2.0f64.powi(-2 * i)).sqrt())
                .product();
            let want_x0 = (ONE as f64 / k * (1u64 << PREC) as f64).round() as i64;
            assert_eq!(X0, want_x0, "X0 wrong");
        }
    }
}

/// Client-side float helpers for turning the sim's integer angles into the radians a
/// renderer wants. **Deliberately OUTSIDE [`trig`]:** [`trig`] is the float-free
/// surface the sim itself calls, so keeping every `f32`/`f64` here means "no float in
/// the sim" is enforced by where things live, not by a comment a render edit can
/// ignore. Only client/render code (never [`crate::sim::Sim::step`]) calls this.
pub mod trig_client {
    use super::trig::{TURN, wrap_turns};

    /// Convert a turn-unit angle (e.g. [`crate::sim::Player::yaw`]) to radians for the FP
    /// camera. Returns `f32`, so by construction it can't be used in the sim.
    pub fn turns_to_radians(a: i32) -> f32 {
        (wrap_turns(a) as f32) / (TURN as f32) * std::f32::consts::TAU
    }

    /// Inverse of [`turns_to_radians`]: a float radians angle (e.g. an `atan2` heading) to
    /// wrapped turn units, nearest-turn rounded. Client/bridge-side like the forward — a
    /// place where a float measurement enters the integer sim's angle convention (the
    /// external-crab yaw bridge feeds `state_hash` through this), so the rounding must have
    /// exactly one spelling.
    pub fn radians_to_turns(r: f32) -> i32 {
        wrap_turns((r / std::f32::consts::TAU * TURN as f32).round() as i32)
    }
}
