//! The reward function and the distance metrics it is defined over. THREE terms: a
//! world-frame PROGRESS pull (the carapace's net distance CLOSED toward the goal this tick),
//! a near-field REACH grab bonus (closest claw-tip→target distance), minus an actuation-cost
//! tax. The reward stays GLOBAL — it pays high-level progress-through-the-world plus a
//! terminal grab, never a per-leg / foot-contact / gait-phase term — so the GAIT itself
//! EMERGES rather than being hand-specified (owner call: mechanical terms don't scale to
//! emergent behaviour).
//!
//! Why progress AND reach: the end task is "get to the player and grab." Progress is the
//! cross-arena pull (a lean cannot fake the BODY moving toward the goal — the gap the old
//! reach-only signal let a reacher game by leaning); reach is the last-metre grab bonus the
//! game's hit actually resolves on. Progress dominates while far, reach dominates while near.
//!
//! The progress term is a POTENTIAL-BASED shaping of `Φ = −distance`: each transition pays
//! `P·(d_prev − d_now)`, the reduction in the carapace's planar distance to the (fixed,
//! per-episode) target. It is the body's NET ground covered, measured on the carapace's own
//! transform origin — NOT a per-tick velocity. Two properties fall out, and both matter:
//!   * **Telescoping ⇒ oscillation-proof.** Summed over any path the per-tick deltas collapse
//!     to `P·(d_start − d_end)` — exactly the net distance closed. A closed wobble (return to
//!     where it started) sums to ZERO, of ANY shape or speed. A *clipped per-tick velocity*
//!     (the first cut of this design) does NOT have this: clipping the fast away-phase of an
//!     asymmetric wobble below the toward-phase pays net-positive for zero travel. The
//!     un-clamped potential closes that hole; the blow-up guard (`systems.rs`, > 100 m/s ends
//!     the episode) plus the effort tax are the fling ceiling instead of a per-tick clamp.
//!   * **Origin distance ⇒ spin/limb-fling-proof.** Measuring the carapace ORIGIN's distance
//!     (not COM velocity) means a pure spin about the body, or flinging a heavy limb to kick
//!     the COM, moves the origin little-to-none (and any recoil telescopes back) — only real
//!     body translation closes `d`.

use bevy::prelude::Vec3;

use crate::bot::actuator::ACTION_SIZE;

/// Weight of the effort term `− EFFORT_WEIGHT·Σ|aᵢ|^L` (see [`compute_reward`]): the
/// per-command actuation cost. Binding constraint — effort is the convex cube of the
/// RAW (unbounded) outputs, so the weight fixes a break-even command size under which a
/// real stride still nets positive. Now that PROGRESS (not reach) is what a stride earns, the
/// break-even is re-derived against the progress a stride collects per tick: a nominal ~0.5
/// m/s walk closes ~0.008 m/tick (at 60 Hz), worth `PROGRESS_WEIGHT·0.008 ≈ 0.2`, so
/// `tax(|a|) = EFFORT_WEIGHT·30·|a|³ = 0.2` at `|a| ≈ 1.1/joint`: a gait command (`|a| ≈
/// 0.4–0.8`) nets clearly positive on progress alone, while deep saturation (`|a| = 3`, cost
/// ≈ 4 ≫ 0.2) is still reined in. The recalibration CONFIRMS 0.005 rather than moving it.
/// Convex `EFFORT_EXP`=`L` keeps the gentlest sufficient command the cheapest.
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

/// Weight `P` of the progress term `P·(d_prev − d_now)` (see [`compute_reward`] and
/// [`progress_reward`]) — the reward per METRE of planar ground the carapace closes toward
/// the target. It is a function of DISPLACEMENT, not speed, so it is tick-rate independent and
/// (un-clamped) telescopes exactly to `P·(d_start − d_end)` over an episode.
///
/// 24 is chosen so:
/// * a full traversal of the curriculum band (targets 1.5→9 m out) pays ≈ 36→216 — comparable
///   to the old reach-only episode totals (~250), so the warm-started value head + return
///   normalizer barely rescale;
/// * one tick of honest walking (~0.5 m/s ⇒ ~0.008 m at 60 Hz) pays ≈ 0.2 — about 10× its
///   ~0.019 effort tax — a dense local gradient to set off and keep moving;
/// * it is the SAME per-tick signal strength the velocity-form first cut intended
///   (`P_vel·v = 0.4·0.5 = 0.2`), re-expressed as the exactly-telescoping potential rather
///   than a clipped per-tick velocity (which reopened an oscillation exploit — see the module
///   header). There is deliberately NO per-tick cap: the blow-up guard + effort tax bound a
///   fling, and the telescoping makes a fling-and-return net zero on its own.
const PROGRESS_WEIGHT: f32 = 24.0;

/// Glitch guard on the per-tick progress: a transition whose carapace planar distance changed
/// by MORE than this (metres in one ~1/60 s tick ⇒ > ~30 m/s of origin translation) is treated
/// as NON-PHYSICAL and earns ZERO progress, not `P·Δd`. No crab covers half a metre of ground
/// in a tick — the observed carapace speed peaks at ~3 m/s (≈0.05 m/tick), so this sits ~10×
/// above any real motion and NEVER fires on a physical trajectory. It exists only to stop a
/// rare solver hiccup (a finite-but-huge one-tick jump, still under the 100 m/s blow-up guard
/// that ends the episode) from injecting a `P·Δd` reward SPIKE into the value/return estimate.
/// Crucially it preserves the telescoping/oscillation-proofness for every realizable path
/// (all real steps are far below it, so none is dropped) and cannot be exploited — the policy
/// cannot deliberately produce a > 0.5 m/tick displacement to farm or to dodge the penalty.
const MAX_PROGRESS_STEP_M: f32 = 0.5;

/// Weight `W` and length scale `S` of the reach term `W·(1 − tanh(d/S))` (see
/// [`reach_bonus`]): `W` is the bonus a claw tip earns by reaching the target dead-on; `S`
/// sets how the pull decays with distance.
///
/// **Near-field only, by design.** The reach term is now the terminal GRAB bonus, not the
/// cross-arena pull (that is the progress term). So `S` is tightened to ~1 m: `1 − tanh(d/1)`
/// is ~0.27 of `W` at 1 m, ~0.10 at 1.5 m, and ~0.005 at 3 m — a smooth bonus confined to the
/// last metre or so, decaying to ≈0 well before the curriculum's far targets. `W` is halved
/// to 0.3 (from the old 0.6) so even dead-on reach cannot out-earn a steady stride from
/// across the arena. The long `tanh` tail at `S=4` was exactly what let a reacher score from
/// far away by leaning; tightening `S` and shrinking `W` removes that far-lean payoff without
/// a hard distance gate (smooth, no cliff at the boundary).
const REACH_WEIGHT: f32 = 0.3;
const REACH_SCALE: f32 = 1.0;

/// Shaped proximity bonus `W·(1 − tanh(d/S))` (weight and scale on [`REACH_WEIGHT`]),
/// where `d` is the minimum 3D euclidean distance over (claw tip, target) pairs. Strictly
/// POSITIVE, maxing at `W` when a tip reaches the target (`d`→0) and decaying to ≈0 by a few
/// metres (the near-field grab bonus). `None` (no target, no claw tip) yields 0.
pub(crate) fn reach_bonus(min_tip_dist: Option<f32>) -> f32 {
    match min_tip_dist {
        Some(d) if d.is_finite() => REACH_WEIGHT * (1.0 - (d / REACH_SCALE).tanh()),
        _ => 0.0,
    }
}

/// The weighted progress term `P·(d_prev − d_now)` — see [`PROGRESS_WEIGHT`]. `distance_closed`
/// is the metres the carapace's planar distance to the target SHRANK over the transition
/// (positive ⇒ closer, negative ⇒ farther). UN-clamped on purpose: that is what makes a closed
/// wobble telescope to exactly zero (the oscillation-proofness the design rests on). `None` —
/// a rescued / teleported body, or a missing pose/target — contributes 0: a teleport is not
/// EARNED travel (the same logic as the `None` reach credit on a rescue), and crediting the
/// spawn jump would be a huge spurious delta.
fn progress_reward(distance_closed: Option<f32>) -> f32 {
    match distance_closed {
        // The `abs() <= MAX_PROGRESS_STEP_M` arm drops only non-physical solver-glitch jumps
        // (see [`MAX_PROGRESS_STEP_M`]); every real step is far below it, so this neither
        // clamps legitimate motion nor breaks the telescoping.
        Some(delta) if delta.is_finite() && delta.abs() <= MAX_PROGRESS_STEP_M => {
            PROGRESS_WEIGHT * delta
        }
        _ => 0.0,
    }
}

/// The reward: `P·(d_prev − d_now) + W·(1 − tanh(d/S)) − EFFORT_WEIGHT·Σ|aᵢ|^L` — the
/// world-frame progress pull ([`progress_reward`]) plus the near-field reach grab bonus
/// ([`reach_bonus`]) minus the cost of the commands that earn them ([`action_effort`]).
///
/// Division of labour: PROGRESS dominates while far (the only cross-arena pull, and a lean
/// can't fake the body covering ground), REACH dominates in the last ~metre (it decays to ≈0
/// by a few metres). The signal stays GLOBAL — progress-through-the-world + a terminal grab,
/// no gait/foot/per-leg term — so locomotion EMERGES. Height and uprightness remain
/// OBSERVATIONS, not reward inputs: this function can't see them, so no pose can be gamed for
/// free reward — only closing ground toward the goal, or the last metre, pays.
pub(crate) fn compute_reward(
    distance_closed: Option<f32>,
    min_tip_dist: Option<f32>,
    effort: f32,
) -> f32 {
    progress_reward(distance_closed) + reach_bonus(min_tip_dist) - EFFORT_WEIGHT * effort
}

/// Planar (XZ) distance between two world points. The carapace→target distance the progress
/// term is the per-tick reduction OF (and the carapace→spawn drift diagnostic, and the
/// curriculum band) — all DEFINED on the floor plane. NOT the reach reward's `d` (that is the
/// 3D [`dist_3d`], so lowering a claw onto a low target pays).
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

    #[test]
    fn progress_closing_raises_receding_lowers() {
        // The core invariant: closing ground toward the target raises the reward, losing
        // ground lowers it, and the two are symmetric (the basis of the telescoping below).
        let effort = action_effort(&[0.3; ACTION_SIZE]);
        let still = compute_reward(Some(0.0), None, effort);
        let closing = compute_reward(Some(0.01), None, effort);
        let closing_more = compute_reward(Some(0.02), None, effort);
        let receding = compute_reward(Some(-0.01), None, effort);
        assert!(closing > still, "closing ground out-earns standing still");
        assert!(closing_more > closing, "closing more earns more (linear, un-capped)");
        assert!(receding < still, "losing ground lowers the reward below standing still");
        assert!(
            ((closing - still) - (still - receding)).abs() < 1e-6,
            "the progress term is symmetric: +δ closing gains what −δ receding loses"
        );
        // None (rescue/teleport, or missing pose/target) is neutral — no earned travel.
        assert!(
            (compute_reward(None, None, effort) - still).abs() < 1e-6,
            "a teleported/rescued body earns no progress (neutral, like standing still)"
        );
    }

    #[test]
    fn progress_is_linear_and_oscillation_proof() {
        // THE property the design rests on, and the one the clipped-velocity first cut LACKED:
        // because the progress term is linear over physical motion, any sequence of per-tick
        // deltas that returns the body to where it started (sums to 0) pays exactly 0 in total
        // — a wobble of ANY shape or speed can't be farmed. (A clipped per-tick velocity fails
        // this: the fast away-phase clips below the slow toward-phase ⇒ net-positive for zero
        // travel.) Pinned over an asymmetric closed loop of REALISTIC per-tick steps (all far
        // below MAX_PROGRESS_STEP_M, so none is dropped) summing to zero net displacement.
        let deltas = [0.02_f32, 0.05, -0.01, -0.08, 0.02]; // sums to exactly 0.0, all physical
        debug_assert!((deltas.iter().sum::<f32>()).abs() < 1e-6);
        let total: f32 = deltas.iter().map(|&d| progress_reward(Some(d))).sum();
        assert!(
            total.abs() < 1e-5,
            "a closed loop (Σδ = 0) must pay zero total progress, whatever the path: {total}"
        );
        // Linearity within the physical range (so the telescoping above is not a coincidence
        // of these numbers): progress(a) + progress(b) == progress(a+b).
        assert!(
            (progress_reward(Some(0.07)) + progress_reward(Some(-0.03))
                - progress_reward(Some(0.04)))
            .abs()
                < 1e-5,
            "progress is exactly linear in distance closed over physical steps (no clamp)"
        );
    }

    #[test]
    fn progress_glitch_guard_drops_nonphysical_jumps() {
        // The single-tick spike guard: a > MAX_PROGRESS_STEP_M jump (a solver hiccup, never a
        // real crab step) earns ZERO, not P·Δd — both directions, so it can't be farmed or
        // used to dodge a receding penalty. Real steps an order of magnitude smaller are paid
        // in full, so the guard never touches a physical trajectory.
        assert_eq!(
            progress_reward(Some(5.0)),
            0.0,
            "a non-physical forward jump (> 0.5 m/tick) earns no progress"
        );
        assert_eq!(
            progress_reward(Some(-5.0)),
            0.0,
            "a non-physical backward jump is likewise dropped (symmetric — no farm)"
        );
        // A brisk-but-real step (~3 m/s ⇒ ~0.05 m/tick) is well under the guard and paid fully.
        assert!(
            (progress_reward(Some(0.05)) - PROGRESS_WEIGHT * 0.05).abs() < 1e-6,
            "a physical step is paid in full — the guard never fires on real motion"
        );
    }

    #[test]
    fn progress_dominates_far_reach_dominates_near() {
        // The division of labour: while crossing the arena the progress a stride earns
        // dominates the (near-field) reach bonus at that distance, so leaning-from-afar stops
        // paying; in the final approach the reach grab bonus carries as progress tapers off.
        // FAR: a single honest stride (~0.5 m/s ⇒ ~0.008 m closed/tick) vs reach at 6 m.
        let stride = progress_reward(Some(0.5 / 60.0));
        assert!(
            stride > reach_bonus(Some(6.0)),
            "far from the goal, a stride's per-tick progress must exceed the reach bonus there: \
             {stride} vs {}",
            reach_bonus(Some(6.0)),
        );
        // NEAR/arrived: progress per tick → 0 as the body stops on the target; reach carries.
        let near_reach = reach_bonus(Some(0.1));
        let arrived_progress = progress_reward(Some(0.0005)); // creeping the last mm
        assert!(
            near_reach > arrived_progress,
            "near the goal the reach grab bonus must dominate the vanishing progress: \
             {near_reach} vs {arrived_progress}"
        );
    }

    #[test]
    fn reach_is_a_near_field_grab_bonus() {
        // Reach is the terminal grab bonus: strictly positive, maxes at W on the target,
        // monotone decreasing, and CONFINED to the near field (≈0 by a few metres) so it
        // cannot pull the body across the arena — that is now the progress term's job.
        assert!(
            (reach_bonus(Some(0.0)) - REACH_WEIGHT).abs() < 1e-6,
            "a claw tip on the target earns the full reach weight"
        );
        assert!(
            reach_bonus(Some(0.1)) > reach_bonus(Some(0.5)),
            "closer to the target must out-reward farther"
        );
        assert!(
            reach_bonus(Some(0.5)) > reach_bonus(Some(1.5)),
            "the reach bonus decreases monotonically with distance"
        );
        assert!(
            reach_bonus(Some(3.0)) < 0.01 * REACH_WEIGHT,
            "reach has all but vanished by a few metres — it is not a cross-arena pull: {}",
            reach_bonus(Some(3.0)),
        );
        assert_eq!(
            reach_bonus(None),
            0.0,
            "no target (or no tip) contributes nothing to the reach term"
        );
    }

    /// The reach reward must pull the tip toward the target IN 3D: a smaller 3D tip→target
    /// distance scores strictly higher, including when the only difference is HEIGHT — a
    /// claw lowered onto a low ball must beat one hovering above it at the same ground spot.
    #[test]
    fn closer_tip_in_3d_raises_reward() {
        let target = Vec3::new(1.0, 0.3, 0.0);
        let on_ball = Vec3::new(1.0, 0.3, 0.0);
        let hovering = Vec3::new(1.0, 1.3, 0.0);
        let d_on = dist_3d(on_ball, target);
        let d_hover = dist_3d(hovering, target);
        assert!(
            d_on < d_hover,
            "3D distance must distinguish height: on-ball {d_on} should be < hovering {d_hover}"
        );
        // Same command effort and no progress, so the reach term alone decides — closer ⇒ higher.
        let effort = action_effort(&[0.2; ACTION_SIZE]);
        assert!(
            compute_reward(None, Some(d_on), effort) > compute_reward(None, Some(d_hover), effort),
            "a tip resting on the ball must out-score one hovering a metre above it at the \
             same ground spot — the 3D reach pulls the claw DOWN, not just across"
        );
        for (near, far) in [(0.0_f32, 0.5_f32), (0.5, 1.5), (1.5, 3.0)] {
            assert!(
                reach_bonus(Some(near)) > reach_bonus(Some(far)),
                "reach reward must strictly increase as 3D distance shrinks: \
                 d={near} should beat d={far}"
            );
        }
    }

    #[test]
    fn reward_is_progress_plus_reach_minus_effort() {
        // Reward is EXACTLY `progress + reach − K·Σ|a|^L` — three terms, no height, no
        // uprightness, no hidden term. With no progress, no target and no command it is
        // exactly zero.
        assert!(
            compute_reward(None, None, 0.0).abs() < 1e-6,
            "with no progress, no target and no effort, reward is exactly zero"
        );
        let p = Some(0.01);
        let d = Some(0.3);
        let e = action_effort(&[0.2; ACTION_SIZE]);
        let expected = progress_reward(p) + reach_bonus(d) - EFFORT_WEIGHT * e;
        assert!(
            (compute_reward(p, d, e) - expected).abs() < 1e-6,
            "reward is exactly progress + reach − K·effort"
        );
    }

    #[test]
    fn higher_effort_lowers_the_reward() {
        // The tax is strictly increasing in command size, so a harder command always
        // scores below a gentler one — the lever that makes the crab economical: it spends
        // actuation only where progress (or reach) pays for it.
        let still = compute_reward(None, None, action_effort(&[0.0; ACTION_SIZE]));
        let gentle = compute_reward(None, None, action_effort(&[0.3; ACTION_SIZE]));
        let hard = compute_reward(None, None, action_effort(&[0.9; ACTION_SIZE]));
        assert!(
            still > gentle && gentle > hard,
            "reward must fall as commanded effort rises: still {still} > gentle {gentle} > hard {hard}"
        );
        assert!(
            still.abs() < 1e-6,
            "a still policy with no progress and no target is untaxed and unrewarded: {still} should be zero"
        );
    }

    #[test]
    fn effort_cost_calibration() {
        // The tradeoff that matters now is PROGRESS vs the cost of the stride that earns it
        // (the tax is over the RAW pre-clamp outputs — see `action_effort`):
        // 1. A still policy with no progress/target pays no tax and earns nothing — zero.
        let still = compute_reward(None, None, action_effort(&[0.0; ACTION_SIZE]));
        assert!(
            still.abs() < 1e-6,
            "a still policy with no progress/target is zero: {still}"
        );
        // 2. A real STRIDE — closing ~0.5 m/s of ground (≈0.008 m/tick at 60 Hz) with an
        //    in-range gait command (|a| < 1) — must net POSITIVE, valuing ONLY the progress
        //    (no reach credit). At weight 0.005 a |a|=0.6 command across 30 joints costs
        //    0.005·30·0.6³ ≈ 0.032, well under the stride's PROGRESS_WEIGHT·0.5/60 ≈ 0.2.
        let stride = compute_reward(Some(0.5 / 60.0), None, action_effort(&[0.6; ACTION_SIZE]));
        assert!(
            stride > 0.0,
            "a real stride must net positive after the tax, on progress alone: {stride}"
        );
        // 3. The break-even command size: the per-tick stride progress (≈0.2) equals the tax
        //    at |a| ≈ 1.1/joint, so a gait command is well inside the net-positive region.
        let stride_progress = progress_reward(Some(0.5 / 60.0));
        let breakeven = EFFORT_WEIGHT * action_effort(&[1.1; ACTION_SIZE]);
        assert!(
            (breakeven - stride_progress).abs() < 0.02,
            "effort break-even must sit at the per-tick stride progress: tax {breakeven} vs progress {stride_progress}"
        );
        // 4. A saturation-seeking command (raw outputs far past the ±1 clamp) is taxed BELOW
        //    a real stride even while closing ground — `|a|^L` keeps climbing past the clamp,
        //    so the gradient pushes the policy out of saturation. At |a|=3 the cost
        //    (0.005·30·27 ≈ 4.05) swamps the ≈0.2 stride progress, driving reward negative.
        let oversaturated =
            compute_reward(Some(0.5 / 60.0), None, action_effort(&[3.0; ACTION_SIZE]));
        assert!(
            oversaturated < stride,
            "saturation-seeking must be taxed below a real stride: {oversaturated} vs {stride}"
        );
    }
}
