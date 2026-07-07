pub mod trig {
    use std::sync::OnceLock;

    pub const TURN: i32 = 1 << 16;
    pub const ONE: i32 = 1 << 14;

    const QUARTER: usize = (TURN / 4) as usize;

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

    const PREC: u32 = 28;
    const ITERS: usize = 28;

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
    const X0: i64 = 2670726652173;

    fn cordic_sin(a: i32) -> i32 {
        let target = (a as i64) << PREC;
        let mut x: i64 = X0;
        let mut y: i64 = 0;
        let mut ang: i64 = 0;
        for (i, &step) in ATAN_TURNS.iter().enumerate() {
            let i = i as u32;
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
        let half = 1i64 << (PREC - 1);
        ((y + half) >> PREC) as i32
    }

    pub fn wrap_turns(a: i32) -> i32 {
        a & (TURN - 1)
    }

    pub fn sin(a: i32) -> i32 {
        let a = wrap_turns(a) as usize;
        let q = QUARTER;
        match a / q {
            0 => sine_table()[a],
            1 => sine_table()[2 * q - a],
            2 => -sine_table()[a - 2 * q],
            _ => -sine_table()[4 * q - a],
        }
    }

    pub fn cos(a: i32) -> i32 {
        sin(a + TURN / 4)
    }

    pub fn sin_cos(a: i32) -> (i64, i64) {
        (sin(a) as i64, cos(a) as i64)
    }

    pub fn atan2_turns(x: i64, z: i64) -> i32 {
        if x == 0 && z == 0 {
            return 0;
        }
        let ax = x.unsigned_abs() as u128;
        let az = z.unsigned_abs() as u128;
        let (hi, lo) = if ax >= az { (ax, az) } else { (az, ax) };
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
        let theta = lo_k;
        let in_q = if ax <= az { theta } else { TURN / 4 - theta };
        let folded = match (x >= 0, z >= 0) {
            (true, true) => in_q,
            (true, false) => TURN / 2 - in_q,
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

pub mod trig_client {
    use super::trig::{TURN, wrap_turns};

    pub fn turns_to_radians(a: i32) -> f32 {
        (wrap_turns(a) as f32) / (TURN as f32) * std::f32::consts::TAU
    }

    pub fn radians_to_turns(r: f32) -> i32 {
        wrap_turns((r / std::f32::consts::TAU * TURN as f32).round() as i32)
    }
}
