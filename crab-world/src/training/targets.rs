
use bevy::prelude::Vec3;

use crate::bot::CrabSpawns;
use crate::bot::sensor::CrabTargets;

pub(crate) const BAND_START_MIN: f32 = 1.5;
pub(crate) const TARGET_Y_MIN: f32 = 0.15;
pub(crate) const TARGET_Y_MAX: f32 = 0.7;
pub(crate) const TARGET_ARENA_HALF: f32 = crate::physics::world::ARENA_HALF_SIZE - 1.0;

pub(crate) const REACH_RADIUS: f32 = 0.8;
pub(crate) const SOLID_REACH_FRACTION: f32 = 0.6;

const NEAR_BIAS_EXP: f32 = 2.0;

pub(crate) fn sample_target(origin: Vec3, rng: &mut impl rand::Rng) -> Vec3 {
    let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
    let u: f32 = rng.gen_range(0.0..1.0);
    let dist = min + (max - min) * u.powf(NEAR_BIAS_EXP);
    let y = rng.gen_range(TARGET_Y_MIN..TARGET_Y_MAX);
    let at = |theta: f32| {
        Vec3::new(
            origin.x + dist * theta.cos(),
            y,
            origin.z + dist * theta.sin(),
        )
    };
    let in_arena = |p: &Vec3| p.x.abs() <= TARGET_ARENA_HALF && p.z.abs() <= TARGET_ARENA_HALF;

    for _ in 0..32 {
        let p = at(rng.gen_range(0.0..std::f32::consts::TAU));
        if in_arena(&p) {
            return p;
        }
    }
    let inward = Vec3::new(-origin.x, 0.0, -origin.z).normalize_or_zero();
    let theta = if inward == Vec3::ZERO {
        0.0
    } else {
        inward.z.atan2(inward.x)
    };
    at(theta)
}

pub(crate) fn seed_target(
    targets: &mut CrabTargets,
    spawns: &CrabSpawns,
    e: usize,
    rng: &mut rand::rngs::StdRng,
) {
    if let Some(slot) = targets.envs.get_mut(e) {
        let origin = spawns.0.get(e).copied().unwrap_or(Vec3::ZERO);
        *slot = Some(sample_target(origin, rng));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::reward::planar_dist;

    #[test]
    fn sampled_targets_lie_at_the_band_distance_and_inside_the_arena() {
        let mut rng = rand::thread_rng();
        let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
        for origin in [
            Vec3::ZERO,
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(8.0, 0.0, -8.0),
        ] {
            for _ in 0..2000 {
                let t = sample_target(origin, &mut rng);
                assert!(t.is_finite(), "a sampled target is always finite");
                assert!(
                    t.x.abs() <= TARGET_ARENA_HALF && t.z.abs() <= TARGET_ARENA_HALF,
                    "target {t:?} from {origin:?} must stay inside ±{TARGET_ARENA_HALF} m"
                );
                assert!(t.y >= TARGET_Y_MIN && t.y <= TARGET_Y_MAX);
                let d = planar_dist(t, origin);
                assert!(
                    d >= min - 1e-3 && d <= max + 1e-3,
                    "target from {origin:?} is at {d} m, outside the band [{min}, {max}]"
                );
            }
        }
    }

    // (No "band is the full arena" test: with TargetBand deleted the band IS the two
    // constants — a test would just restate them. The honest-distance test above pins
    // that samples span [BAND_START_MIN, TARGET_ARENA_HALF].)

    #[test]
    fn distance_draw_is_near_heavy_with_a_real_far_tail() {
        let mut rng = rand::thread_rng();
        let origin = Vec3::ZERO;
        let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
        let near_edge = 3.0;
        let far_edge = 6.0;
        let (mut near, mut far, n) = (0u32, 0u32, 20_000u32);
        for _ in 0..n {
            let d = planar_dist(sample_target(origin, &mut rng), origin);
            if d < near_edge {
                near += 1;
            }
            if d > far_edge {
                far += 1;
            }
        }
        let near_frac = near as f32 / n as f32;
        let far_frac = far as f32 / n as f32;
        assert!(
            near_frac > 0.40,
            "near ({min}-{near_edge} m) fraction {near_frac} should dominate (uniform would be ~0.2; EXP=2 ~0.45)"
        );
        assert!(
            far_frac > 0.15,
            "far (>{far_edge} m, up to {max} m) fraction {far_frac} must keep a FAT tail (EXP=2 ~0.22), not starve far"
        );
    }
}
