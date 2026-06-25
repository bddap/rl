//! The reward function (reach pull minus actuation cost) and the distance metrics it
//! is defined over. The reward signal is GLOBAL — a single claw-tip→target distance, no
//! gait term — so locomotion EMERGES rather than being hand-specified.

use bevy::prelude::Vec3;

use crate::bot::actuator::ACTION_SIZE;

/// Weight of the effort term `− EFFORT_WEIGHT·Σ|aᵢ|^L` (see [`compute_reward`]): the
/// per-command actuation cost. Binding constraint — effort is the convex cube of the
/// RAW (unbounded) outputs, so the weight fixes a break-even command size (tax = reach)
/// under which exploring-and-reaching nets positive. That break-even must exceed a real
/// stride or a COLD stand can't explore a gait and stays stuck in the stand basin just
/// paying the tax. At 0.005 break-even is ~|a|≈1.6/joint (a gait is explorable) while
/// deep saturation (|a|=3) still costs ~4 ≫ the W=0.6 reach, so flailing is still reined
/// in. Convex `EFFORT_EXP`=`L` keeps the gentlest sufficient command the cheapest.
pub(crate) const EFFORT_WEIGHT: f32 = 0.005;
const EFFORT_EXP: f32 = 3.0;

/// The effort summand `Σ|aᵢ|^L` that [`compute_reward`] weights by [`EFFORT_WEIGHT`],
/// taken over the RAW network outputs (the sampled PRE-clamp actions — see
/// `brain_step`), NOT the ±1-clamped actions the sim runs. The point is a gradient
/// PAST the clamp: `|a|^L` keeps rising beyond ±1, so an output that overshoots the
/// usable range is taxed in proportion to the overshoot and the policy is pulled back
/// into range. Taxing the clamped value instead would flatten the gradient at ±1 — a
/// saturating logit would pay a fixed toll but feel no pull off the rail.
pub(crate) fn action_effort(raw_actions: &[f32; ACTION_SIZE]) -> f32 {
    raw_actions.iter().map(|a| a.abs().powf(EFFORT_EXP)).sum()
}

/// Weight `W` and length scale `S` of the reach term `W·(1 − tanh(d/S))` (see
/// [`reach_bonus`]): `W` is the bonus a claw tip earns by reaching the target
/// dead-on; `S` sets how the pull decays with distance.
///
/// **Why `1 − tanh(d/S)` and not `exp(−d/S)`:** targets spawn metres away (the
/// curriculum band runs 1.5–9 m), and `exp(−d/0.4)` is ~0 (≈1e-3 at 3 m) out there —
/// no gradient, nothing pulling the crab across the arena. `tanh` has a long
/// polynomial-ish tail: with `S = 4 m`, `1 − tanh(d/S)` is ~0.36 at 3 m and ~0.10 at
/// 6 m, with a clearly non-zero slope the whole way (proved numerically in
/// [`tests::new_reach_term_has_gradient_at_spawn_distance`]). That non-vanishing slope
/// at spawn distance IS the walking signal — descend it by getting closer, i.e. walk.
///
/// `W` is set well above the effort cost of the gentle motion that closes the
/// distance, so a whole walk to a far target nets positive and the policy is paid to
/// set off (the tradeoff is pinned in [`tests::effort_cost_calibration`]).
const REACH_WEIGHT: f32 = 0.6;
const REACH_SCALE: f32 = 4.0;

/// Shaped proximity bonus `W·(1 − tanh(d/S))` (weight and scale on [`REACH_WEIGHT`]),
/// where `d` is the minimum 3D euclidean distance over (claw tip, target) pairs. The
/// reward's only positive term: strictly POSITIVE, maxing at `W` when a tip reaches the
/// target (`d`→0). `None` (no target, no claw tip) yields 0 — the reward then degrades to
/// just the effort tax (the demo path and any tip-less tick).
pub(crate) fn reach_bonus(min_tip_dist: Option<f32>) -> f32 {
    match min_tip_dist {
        Some(d) if d.is_finite() => REACH_WEIGHT * (1.0 - (d / REACH_SCALE).tanh()),
        _ => 0.0,
    }
}

/// The reward: `W·(1 − tanh(d/S)) − EFFORT_WEIGHT·Σ|aᵢ|^L`, the reach pull
/// ([`reach_bonus`]) minus the cost of the commands that earn it ([`action_effort`]),
/// where `d` is the closest 3D euclidean claw-tip-to-target distance.
///
/// The reach signal is GLOBAL — a single distance, no gait term, no "feet on the
/// ground" — so locomotion EMERGES instead of being hand-specified (owner's call:
/// mechanical terms don't scale to emergent behaviour). Height and uprightness are
/// observations, not reward: this function literally can't see them, so no pose can be
/// gamed for free reward — only closing `d` pays.
pub(crate) fn compute_reward(min_tip_dist: Option<f32>, effort: f32) -> f32 {
    reach_bonus(min_tip_dist) - EFFORT_WEIGHT * effort
}

/// Planar (XZ) distance between two world points. NOT the reach reward's `d` (that is
/// [`dist_3d`]); kept for the genuinely 2D diagnostics — the carapace's ground drift
/// from spawn and the curriculum band, both DEFINED on the floor plane.
pub(crate) fn planar_dist(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    (d.x * d.x + d.z * d.z).sqrt()
}

/// Full 3D euclidean distance between two world points — the reach reward's `d`. 3D (not
/// planar) so lowering a claw onto a low ball pays: a ground-only `d` would score a tip
/// hovering a metre above the target identically to one resting on it, leaving nothing to
/// pull the claw down the last stretch. `pub(crate)` so the demo's reached-test
/// (`play::target_ball`) measures the SAME `d` the reward does.
pub(crate) fn dist_3d(a: Vec3, b: Vec3) -> f32 {
    (a - b).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::curriculum::{BAND_START_MIN, TARGET_ARENA_HALF};

    #[test]
    fn height_does_not_change_the_reward() {
        // Guards against reintroducing a height arg while leaving the reach term inert:
        // `compute_reward` has no height argument, so tip distance must still move the
        // reward on its own.
        let effort = action_effort(&[0.3; ACTION_SIZE]);
        let near = compute_reward(Some(0.1), effort);
        let far = compute_reward(Some(2.0), effort);
        assert!(
            near > far,
            "reach still moves the reward (closer out-scores farther): {near} vs {far}"
        );
    }

    #[test]
    fn reach_bonus_rewards_reaching() {
        // The reach term is strictly positive, maxes at the target (d→0 ⇒ W), and
        // decreases monotonically with distance — a dense pull that stays alive across the
        // far band. Depends ONLY on the tip distance (uprightness is not in the reward).
        assert!(
            (reach_bonus(Some(0.0)) - REACH_WEIGHT).abs() < 1e-6,
            "a claw tip on the target earns the full reach weight"
        );
        assert!(
            reach_bonus(Some(0.1)) > reach_bonus(Some(0.5)),
            "closer to the target must out-reward farther"
        );
        // Still clearly positive at the FARTHEST the curriculum can push the target (the
        // arena cap) — the walking signal survives every rung.
        assert!(
            reach_bonus(Some(TARGET_ARENA_HALF)) > 0.0,
            "the reach bonus is strictly positive even at the arena-cap target distance"
        );
        assert!(
            reach_bonus(Some(BAND_START_MIN)) > reach_bonus(Some(TARGET_ARENA_HALF)),
            "the pull must slope downward across the whole curriculum span — the walking signal"
        );
        assert_eq!(
            reach_bonus(None),
            0.0,
            "no target (or no tip) contributes nothing — with no positive term left the \
             reward is just the effort tax"
        );
    }

    #[test]
    fn new_reach_term_has_gradient_at_spawn_distance() {
        // Numerically pins the why-tanh rationale (on [`REACH_WEIGHT`]): the `1 − tanh(d/S)`
        // term and its slope are clearly non-zero across the spawn band where an `exp` term
        // would be ~0, so a far target gives a real gradient to WALK down. Compares the two
        // at each distance via a finite-difference slope.
        const OLD_SCALE: f32 = 0.4; // an exp length scale, for the comparison
        let old_term = |d: f32| (-d / OLD_SCALE).exp();
        let new_term = |d: f32| 1.0 - (d / REACH_SCALE).tanh();
        let slope = |f: &dyn Fn(f32) -> f32, d: f32| {
            let h = 1e-3;
            (f(d + h) - f(d - h)) / (2.0 * h)
        };
        // Two distinct claims, because the curriculum band now spans a NEAR start edge
        // (where the cold policy bootstraps) out to the arena cap (the farthest rung) —
        // the old far-only band assumed the exp was ~0 everywhere, which is false at the
        // near edge (exp(−1.5/0.4)=exp(−3.75)≈0.024, small but not negligible).
        //
        // (a) NEAR edge — what makes bootstrapping possible is a strong ABSOLUTE gradient
        // at the spawn pose; the new term/slope just has to be clearly usable AND beat the
        // old (it does, ~3.7×), not dominate it 20×.
        assert!(
            new_term(BAND_START_MIN) > 0.3 && slope(&new_term, BAND_START_MIN).abs() > 0.1,
            "near edge must have a strong absolute reach gradient (the bootstrap signal): \
             term {} slope {}",
            new_term(BAND_START_MIN),
            slope(&new_term, BAND_START_MIN),
        );
        assert!(
            new_term(BAND_START_MIN) > old_term(BAND_START_MIN)
                && slope(&new_term, BAND_START_MIN).abs() > slope(&old_term, BAND_START_MIN).abs(),
            "even at the near edge the new term/slope must exceed the old",
        );
        // (b) FAR rungs (3 m out to the arena cap) — here an exp would have all but vanished
        // (exp(−7.5)…exp(−22.5)), so the tanh term/slope DOMINATE it. The term itself
        // shrinks toward the cap (1−tanh(9/4)≈0.022 at d=9), so the guarantees are: strictly
        // positive, a clearly non-zero SLOPE (the learning signal), and overwhelming
        // dominance of the exp — not an absolute term floor (which the curriculum's longer
        // reach can't assume).
        for &d in &[3.0, 4.5, TARGET_ARENA_HALF] {
            assert!(
                new_term(d) > 0.0,
                "new tanh term must stay strictly positive at far d={d}: {}",
                new_term(d)
            );
            assert!(
                new_term(d) > 20.0 * old_term(d),
                "new tanh term must dominate the old exp at far d={d}: new {} vs old {}",
                new_term(d),
                old_term(d)
            );
            assert!(
                slope(&new_term, d).abs() > 1e-3,
                "new tanh slope must be clearly non-zero at far d={d}: {}",
                slope(&new_term, d)
            );
            assert!(
                slope(&new_term, d).abs() > 20.0 * slope(&old_term, d).abs(),
                "new tanh slope must dominate the old exp slope at far d={d}: new {} vs old {}",
                slope(&new_term, d),
                slope(&old_term, d)
            );
        }
    }

    /// The reach reward must pull the tip toward the target IN 3D: a smaller 3D tip→target
    /// distance scores strictly higher, including when the only difference is HEIGHT — a
    /// claw lowered onto a low ball must beat one hovering above it at the same ground spot.
    #[test]
    fn closer_tip_in_3d_raises_reward() {
        let target = Vec3::new(1.0, 0.3, 0.0);
        // Two tips at the SAME ground position, differing only in height: one resting on
        // the ball, one a metre above it. A planar `d` ties these; the 3D `d` must rank
        // the on-target tip strictly closer.
        let on_ball = Vec3::new(1.0, 0.3, 0.0);
        let hovering = Vec3::new(1.0, 1.3, 0.0);
        let d_on = dist_3d(on_ball, target);
        let d_hover = dist_3d(hovering, target);
        assert!(
            d_on < d_hover,
            "3D distance must distinguish height: on-ball {d_on} should be < hovering {d_hover}"
        );
        // Same command effort both poses, so the reach term alone decides — closer ⇒ higher.
        let effort = action_effort(&[0.2; ACTION_SIZE]);
        assert!(
            compute_reward(Some(d_on), effort) > compute_reward(Some(d_hover), effort),
            "a tip resting on the ball must out-score one hovering a metre above it at the \
             same ground spot — the 3D reach pulls the claw DOWN, not just across"
        );
        // And monotone in general: any strictly smaller 3D distance scores strictly higher.
        for (near, far) in [(0.0_f32, 0.5_f32), (0.5, 2.0), (2.0, 6.0)] {
            assert!(
                reach_bonus(Some(near)) > reach_bonus(Some(far)),
                "reach reward must strictly increase as 3D distance shrinks: \
                 d={near} should beat d={far}"
            );
        }
    }

    #[test]
    fn reward_is_reach_minus_effort() {
        // Reward is EXACTLY `reach_bonus(d) − K·Σ|a|^L` — two terms, no height, no
        // uprightness, no hidden term. With no target and no command it is exactly zero
        // (nothing to reach, nothing to tax); a target adds the (ungated) reach term; a
        // command subtracts the tax.
        assert!(
            compute_reward(None, 0.0).abs() < 1e-6,
            "with no target and no effort, reward is exactly zero"
        );
        let expected = reach_bonus(Some(0.3)) - EFFORT_WEIGHT * action_effort(&[0.2; ACTION_SIZE]);
        assert!(
            (compute_reward(Some(0.3), action_effort(&[0.2; ACTION_SIZE])) - expected).abs() < 1e-6,
            "reward is exactly reach_bonus − K·effort"
        );
    }

    #[test]
    fn uprightness_does_not_change_the_reward() {
        // Uprightness lives in the observation, not the reward — `compute_reward` has no
        // uprightness argument, so a flat crab and a level one at the same tip distance and
        // command earn IDENTICAL reward. The consequence pinned here: a claw dangled onto
        // the target collects the FULL reach bonus ungated by pose, adding exactly
        // `reach_bonus(0) = W` over not reaching, at any pose.
        let effort = action_effort(&[0.3; ACTION_SIZE]);
        let no_reach = compute_reward(None, effort);
        let on_target = compute_reward(Some(0.0), effort);
        assert!(
            (on_target - no_reach - REACH_WEIGHT).abs() < 1e-6,
            "a claw on the target adds the full reach weight {REACH_WEIGHT} with no pose gate: \
             {on_target} − {no_reach}"
        );
    }

    #[test]
    fn higher_effort_lowers_the_reward() {
        // The tax is strictly increasing in command size, so a harder command always
        // scores below a gentler one — the lever that should make the crab economical
        // ("tired af"): it spends actuation only where reach pays for it.
        let still = compute_reward(None, action_effort(&[0.0; ACTION_SIZE]));
        let gentle = compute_reward(None, action_effort(&[0.3; ACTION_SIZE]));
        let hard = compute_reward(None, action_effort(&[0.9; ACTION_SIZE]));
        assert!(
            still > gentle && gentle > hard,
            "reward must fall as commanded effort rises: still {still} > gentle {gentle} > hard {hard}"
        );
        // With no target, a still policy pays NO tax and earns nothing — reward is zero.
        assert!(
            still.abs() < 1e-6,
            "a still policy with no target is untaxed and unrewarded: {still} should be zero"
        );
    }

    #[test]
    fn effort_cost_calibration() {
        // Pin the ordering that matters — reach is the ONLY positive term, so the tradeoff
        // is reach vs the cost of the motion that earns it (the tax is over the RAW
        // pre-clamp outputs, see `action_effort`):
        // 1. A still policy with no target pays no tax and earns nothing — reward is zero.
        let still = compute_reward(None, action_effort(&[0.0; ACTION_SIZE]));
        assert!(
            still.abs() < 1e-6,
            "a still policy with no target is zero: {still}"
        );
        // 2. Reaching the target with a MODERATE in-range command (|a| < 1) must still net
        //    POSITIVE — the reach payoff has to exceed the cost of the gentle motion that
        //    closes the distance, or the policy would rather lie still than walk. At weight
        //    0.005 a |a|=0.4 command across all 30 joints costs 0.005·30·0.4³ ≈ 0.0096, well under
        //    the W=0.6 reach payoff, so honest moderate motion that reaches stays worthwhile.
        let moderate_reach = compute_reward(Some(0.0), action_effort(&[0.4; ACTION_SIZE]));
        assert!(
            moderate_reach > 0.0,
            "reaching the target with a moderate command must net positive: {moderate_reach}"
        );
        // 3. A saturation-seeking command (raw outputs driven far past the ±1 the sim
        //    clamps to) is taxed BELOW that moderate reach even when it lands on the target —
        //    because the tax reads the raw outputs, |a|^L keeps climbing past the clamp, so
        //    the gradient pushes the policy OUT of saturation rather than letting it sit
        //    pinned at the rail for a flat toll. At |a|=3 the cost (0.005·30·27 ≈ 4.05)
        //    swamps any reach payoff, driving the reward deeply negative.
        let oversaturated = compute_reward(Some(0.0), action_effort(&[3.0; ACTION_SIZE]));
        assert!(
            oversaturated < moderate_reach,
            "saturation-seeking must be taxed below a moderate reach: {oversaturated} vs {moderate_reach}"
        );
    }
}
