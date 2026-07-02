//! The target-distance band the policy trains on, and the target sampling defined over it.
//!
//! There is no growth curriculum: the band is FIXED at the full arena distance range. The reward
//! is a scale-free telescoping PROGRESS signal (`P·(d_prev − d_now)`, see [`super::reward`]) that
//! pays the same per-metre GRADIENT at any absolute distance — so the old near→far advancement
//! crutch (which existed only because the earlier reach reward went flat far out) is gone, and
//! the weights learn the far approach directly (the bitter lesson).
//!
//! What the scale-free reward does NOT remove is the credit-assignment HORIZON: a far target
//! demands a longer correct action sequence before the sparse grab terminal pays, so a far
//! episode leans harder on the dense progress crumbs. So the full-range distance is sampled
//! NEAR-HEAVY (see [`NEAR_BIAS_EXP`] / [`sample_target`]): most episodes get a near target — the
//! bootstrap the horizon needs — while a real tail still reaches the far edge so far pursuit is
//! learned, not forgotten. Be honest about what this is: a hand-chosen difficulty bias IS a mild
//! curriculum — the bitter-lesson ideal is a UNIFORM draw and the weights learning the far
//! approach unaided. What it is NOT is a GROWTH curriculum: it is stationary — nothing advances,
//! gates, persists, or drifts, the band never changes, only how a distance is drawn within it —
//! so the advancement-machinery drift that cost a live run cannot recur. It was adopted (job 748)
//! to re-bootstrap a warm-but-degraded policy that a uniform draw stalled on far targets; the
//! intended end state is to lean [`NEAR_BIAS_EXP`] back toward 1 (uniform) once reach is healthy.

use bevy::prelude::Vec3;

use crate::bot::CrabSpawns;
use crate::bot::sensor::CrabTargets;

/// Near edge of the target-distance band (m): the closest a target spawns from the env's
/// origin. Clears the ~1.3 m reach shell so even the nearest target demands a step, not a lean.
pub(crate) const BAND_START_MIN: f32 = 1.5;
/// Vertical band of the target (world Y). A modest claw-height span so a crab that
/// has walked up to the target still finishes with a real reach, not a foot-level
/// touch. Kept low and narrow — the reward is about getting THERE, so the target sits
/// just high enough to demand a genuine reach, no higher.
pub(crate) const TARGET_Y_MIN: f32 = 0.15;
pub(crate) const TARGET_Y_MAX: f32 = 0.7;
/// Far edge of the band, and the half-extent the target's planar position is clamped within:
/// a 1 m margin inside the arena walls, DERIVED from the wall position so a wall move can't
/// strand a far target in or beyond a wall where the crab can't stand on it. The margin leaves
/// room for the crab's own body at the goal.
pub(crate) const TARGET_ARENA_HALF: f32 = crate::physics::world::ARENA_HALF_SIZE - 1.0;

/// Per-episode reach radius (m): an episode "reached" if the crab's claw tip came within this
/// of the target at any tick. The CANONICAL reach distance — the demo's ball-hop
/// (`play::target_ball::DEMO_REACH_RADIUS`) and the sparse grab terminal
/// (`reward::GRAB_REWARD`) both derive from this one constant, so "reached" means the same
/// event a viewer sees the ball teleport on. Lives in the always-compiled trainer so the
/// headless build owns the source. A touch looser than zero so a near-miss the policy clearly
/// solved still counts.
pub(crate) const CURRICULUM_REACH_RADIUS: f32 = 0.8;
/// Reach-fraction at or above which the policy is judged to "reliably get there". Reused by
/// [`super::best`] as the solid-reach floor a checkpoint must clear to enter `ckpt/best/`, so a
/// collapse (reach below it) can never become the best. 0.6, not ~1.0: targets near the arena
/// edge clamp short and some spawns are awkward, so demanding unanimity would reject a policy
/// that has effectively mastered the task.
pub(crate) const SOLID_REACH_FRACTION: f32 = 0.6;

/// Exponent of the NEAR-HEAVY distance draw: a target distance is `min + (max−min)·u^EXP` for
/// `u ~ U[0,1)`, so `EXP > 1` biases the draw toward the near edge while still spanning the whole
/// band. At `EXP = 2` over the 1.5–9 m band the mass splits ~45 % into the near 1.5–3 m sub-band,
/// ~33 % into 3–6 m, ~22 % beyond 6 m — still near-dominant for the credit-assignment bootstrap,
/// but with a fatter far tail than `EXP = 3` (~16 %) so far pursuit re-trains instead of starving.
/// The single tunable knob: raise it to lean nearer, drop toward 1 for uniform. See the module doc
/// for why near-heavy over uniform. (Leaned 3→2 once near/mid were reliable and far was stalling:
/// deterministic eval showed EXP=3 over-specialized near, closing ~3.9 m short at 8 m; EXP=2
/// re-fed the far budget and cut the 8 m stall to ~1.1 m median.)
const NEAR_BIAS_EXP: f32 = 2.0;

/// Sample a fresh target world position for a crab whose env spawns at `origin`, at a planar
/// distance drawn NEAR-HEAVY (see [`NEAR_BIAS_EXP`]) from the fixed full-arena band
/// `[BAND_START_MIN, TARGET_ARENA_HALF)` and at EXACTLY
/// that distance — by construction the returned target's planar distance from `origin` is the
/// drawn distance, never less. A random distance in the band is fixed first; then a HEADING is
/// chosen so the full-distance target lands inside the arena (see [`TARGET_ARENA_HALF`]): random
/// headings are tried, falling back to aiming inward (toward the arena centre), which always fits
/// for an in-arena spawn.
///
/// WHY choose the heading rather than clamp the placed point: clamping the point into the arena
/// SHORTENS the distance for a spawn near a wall — an edge crab's "9 m" target clamped to the
/// wall is really ~2 m — so the policy would "master" a far distance by grabbing clamped-near
/// targets it never walked to (rl#159). Choosing the heading keeps the distance honest.
///
/// Honesty is of DISTANCE, not heading: the distance is drawn near-heavy but at its TRUE value
/// from every spawn (the bias is in the distribution, never in shortening a placed target); the
/// heading is uniform only where the arena permits it — a spawn near a wall can only aim inward,
/// so its targets cluster toward the arena centre. That directional bias is acceptable (the target
/// is observed in body axes and spawns are grid-symmetric); what must not bias is the distance.
///
/// Y is an independent claw-height draw. World-space (not carapace-relative) because the crab
/// spawns at varied orientations and walks: a point fixed in the world is an unambiguous goal
/// the observation re-expresses in body axes each tick. `pub(crate)` so the demo's red-ball
/// marker (`play::target_ball`) relocates its target through the very same rule training samples
/// — one sampling rule, so the demo can never pose a target training never saw.
pub(crate) fn sample_target(origin: Vec3, rng: &mut impl rand::Rng) -> Vec3 {
    let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
    // Near-heavy: u^EXP pushes the uniform draw toward `min` (near), still spanning to `max`.
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

    // Most headings fit for a central spawn; an edge spawn fits only the inward arc, so a
    // bounded random search lands a varied in-arena heading without computing the arc.
    for _ in 0..32 {
        let p = at(rng.gen_range(0.0..std::f32::consts::TAU));
        if in_arena(&p) {
            return p;
        }
    }
    // Fallback: aim from the spawn straight toward the arena centre. Moving `dist` toward
    // the centre from any in-arena spawn keeps both coordinates within the cap (the worst
    // case overshoots the centre to the opposite side at distance ≤ `dist` ≤ the cap), so
    // this always lands in-arena and never resorts to shortening the distance.
    let inward = Vec3::new(-origin.x, 0.0, -origin.z).normalize_or_zero();
    let theta = if inward == Vec3::ZERO {
        0.0
    } else {
        inward.z.atan2(inward.x)
    };
    at(theta)
}

/// Install a fresh target for env `e`, sampled around its spawn slot using the training
/// run's seeded RNG. The one home for "a new target is needed" —
/// called to seed the first episode (envs start target-less) and to refresh on every reset, so
/// both callers sample it identically. (Training holds the target fixed within an episode — no
/// resample on reach; see the reach-hover note in `brain_step`.)
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
        // The honest-distance invariant (rl#159): every sampled target is at EXACTLY the band
        // distance from its spawn AND inside the arena — from any origin, including hard against
        // a wall. The old position-clamp shortened the distance for an edge spawn, so the band
        // lied; this pins that it never does again. Checked from a central, an edge, and a
        // corner origin (where only the inward arc of headings fits).
        let mut rng = rand::thread_rng();
        let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
        for origin in [
            Vec3::ZERO,                // central — every heading fits
            Vec3::new(6.0, 0.0, 0.0),  // edge row — only the inward half fits
            Vec3::new(8.0, 0.0, -8.0), // corner — only a narrow inward arc fits
        ] {
            for _ in 0..2000 {
                let t = sample_target(origin, &mut rng);
                assert!(t.is_finite(), "a sampled target is always finite");
                assert!(
                    t.x.abs() <= TARGET_ARENA_HALF && t.z.abs() <= TARGET_ARENA_HALF,
                    "target {t:?} from {origin:?} must stay inside ±{TARGET_ARENA_HALF} m"
                );
                assert!(t.y >= TARGET_Y_MIN && t.y <= TARGET_Y_MAX);
                // The distance is the band distance, never shortened to fit the arena.
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
        // Pins the bootstrap sampling: most targets land near (so the credit-assignment horizon
        // stays short enough to learn) while a real tail still reaches the far arena (so far
        // pursuit is trained, not forgotten). Guards BOTH ways at EXP=2 (~45/33/22 near/mid/far):
        // near must still dominate a uniform draw (whose near fraction would be ~0.2), and the far
        // tail must stay fat — the whole point of the 3→2 lean was to un-starve far.
        let mut rng = rand::thread_rng();
        let origin = Vec3::ZERO;
        let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
        let near_edge = 3.0; // the 1.5-3 m near sub-band
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
