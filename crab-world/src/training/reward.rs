//! The reward function and the distance metrics it is defined over. TWO continuous terms — a
//! world-frame PROGRESS pull (the carapace's net distance CLOSED toward the goal this tick)
//! minus an actuation-cost tax — plus a SPARSE TERMINAL grab event (a one-shot bonus + episode
//! `done`, applied at the episode boundary in `systems::finalize_transitions`, not here). The
//! reward stays GLOBAL — it pays high-level progress-through-the-world plus a terminal grab,
//! never a per-leg / foot-contact / gait-phase term — so the GAIT itself EMERGES rather than
//! being hand-specified (owner call: mechanical terms don't scale to emergent behaviour).
//!
//! Why progress AND a terminal grab: the end task is "get to the player and grab." Progress is
//! the cross-arena pull, dense the whole way (a lean cannot fake the BODY covering ground — the
//! gap the old reach-only signal let a reacher game by leaning). The grab is the SPARSE terminal
//! event the task actually resolves on: a claw tip inside the grab radius ends the episode with
//! a one-shot bonus. The approach EMERGES from progress alone — there is deliberately no per-tick
//! near-field proximity term. The old continuous reach integral did two harmful things at once:
//! it hand-specified the last-metre mechanic, and (un-telescoping, un-gated on contact) it paid a
//! crab to ARRIVE AND HOLD a claw in the near-field for the rest of the episode — farming far
//! more than an honest traverse earned (rl#95). A sparse terminal removes both: there is nothing
//! to hold-farm (the episode ends on contact), and the mechanic is no longer hand-shaped.
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

/// Weight of the metabolic-effort term `− EFFORT_WEIGHT·Σ|dᵢ|^L` (see [`compute_reward`]):
/// a GENTLE regularizer on the policy's neural DRIVE — the pre-clamp Gaussian sample `dᵢ =
/// μᵢ + σ·εᵢ` (UNBOUNDED), NOT the ±1-clamped command the sim runs (see [`action_effort`]).
/// The metabolic-cost analog: cost scales with the neural ACTIVATION the policy emits, not
/// with the bounded muscle output the joint produces. Two binding properties:
///   * **Live anti-saturation gradient.** Because the input is the UNBOUNDED drive, `|d|^L`
///     keeps rising past the ±1 torque clamp, so a policy that samples `|d| ≫ 1` to slam a
///     joint onto its rail PAYS for the overshoot — a gradient pulling `σ` (log_std) and `|μ|`
///     back off saturation. A tax on the clamped command could not: `|command| ≤ 1` is flat at
///     the rail, so a saturating joint would feel no pull off it.
///   * **`σ`-regularizer (via the advantage, not the loss).** The tax is a REWARD term, so
///     it reaches log_std through the policy gradient's normalized advantage — NOT as a direct
///     `∂/∂σ²` loss term. Because `E|d|² = μ² + σ²`, a wider policy samples larger drives on
///     average, so the quadratic tax lands as negative advantage that the gradient reduces by
///     lowering log_std. Crucially it DOMINATES the advantage in the early pure-noise regime
///     (progress ≈ 0, so the tax is nearly the whole signal) — counter-pressure on the entropy
///     bonus exactly when the policy would otherwise inflate into noise — then its share fades
///     once a gait forms and progress dominates the return.
///
/// **Calibration (physics 64 Hz = `physics::PHYSICS_DT`, integrated over a full
/// `systems::MAX_EPISODE_TICKS` episode).** `EFFORT_EXP = 2` (quadratic): the input is
/// unbounded, so a cube would explode over a noisy policy; the square is the standard
/// metabolic activation² and stays a regularizer, not the dominant term. A per-tick CAP is
/// deliberately avoided — it would re-flatten `|d|^L` past the cap and kill the anti-saturation
/// gradient — so the gentle quadratic is what bounds the cost instead. With
/// `EFFORT_WEIGHT = 0.0006`:
///   * a ~0.5 m/s stride closes `0.5·PHYSICS_DT ≈ 0.0078 m/tick`, worth `24·0.0078 ≈ 0.19`;
///   * a gait drive (`|d| ≈ 0.7`) taxes `0.0006·30·0.49 ≈ 0.009/tick` — ~5% of the stride's
///     progress, and ~13 integrated over a 3 m traverse vs its `24·3 = 72` progress: progress
///     dominates >5:1;
///   * the per-joint break-even sits at `|d| ≈ 3.2` (`0.0006·30·d² = 0.19`), far above any
///     gait. A NON-progressing saturating thrash (`|d| = 3`, ~0 progress) pays `0.16/tick`,
///     ≈ −240 over an episode — firmly net-negative. Convex, so the gentlest sufficient drive
///     is the cheapest.
pub(crate) const EFFORT_WEIGHT: f32 = 0.0006;
const EFFORT_EXP: f32 = 2.0;

/// The effort summand `Σ|dᵢ|^L` that [`compute_reward`] weights by [`EFFORT_WEIGHT`], taken
/// over the policy's neural DRIVES `dᵢ` — the PRE-clamp Gaussian samples `μᵢ + σ·εᵢ` the
/// policy actually drew (see `sample_actions`), NOT the ±1-clamped commands the sim runs. The
/// point is a gradient PAST the clamp: `|d|^L` keeps rising beyond ±1, so a drive that
/// overshoots the usable range is taxed in proportion to the overshoot and the policy is
/// pulled back into range. Taxing the clamped command instead flattens the gradient at ±1 — a
/// saturating drive would pay a fixed toll but feel no pull off the rail.
pub(crate) fn action_effort(drives: &[f32; ACTION_SIZE]) -> f32 {
    drives.iter().map(|d| d.abs().powf(EFFORT_EXP)).sum()
}

/// Weight `P` of the progress term `P·(d_prev − d_now)` (see [`compute_reward`] and
/// [`progress_reward`]) — the reward per METRE of planar ground the carapace closes toward
/// the target. It is a function of DISPLACEMENT, not speed, so it is tick-rate independent and
/// (un-clamped) telescopes exactly to `P·(d_start − d_end)` over an episode.
///
/// 24 is chosen so:
/// * a full traversal of the curriculum band (targets 1.5→9 m out) pays ≈ 36→216 in progress —
///   the ONLY cross-arena signal now that the per-tick reach integral is gone, and the dense
///   shaping that carries the body to within grab range at every band, near and far. The sparse
///   [`GRAB_REWARD`] is then a one-shot terminal bonus ON TOP, scaled (see there) to make a grab
///   the clearly-dominant outcome of a near-band episode without swamping the approach signal;
/// * one tick of honest walking (~0.5 m/s ⇒ ~0.0078 m at 64 Hz, `physics::PHYSICS_DT`) pays
///   ≈ 0.19 — ~20× its ~0.009 gait-drive effort tax — a dense local gradient to set off and
///   keep moving;
/// * it is the SAME per-tick signal strength the velocity-form first cut intended
///   (`P_vel·v = 0.4·0.5 = 0.2`), re-expressed as the exactly-telescoping potential rather
///   than a clipped per-tick velocity (which reopened an oscillation exploit — see the module
///   header). There is deliberately NO per-tick cap: the blow-up guard + effort tax bound a
///   fling, and the telescoping makes a fling-and-return net zero on its own.
const PROGRESS_WEIGHT: f32 = 24.0;

/// Glitch guard on the per-tick progress: a transition whose carapace planar distance changed
/// by MORE than this (metres in one 1/64 s tick ⇒ > ~32 m/s of origin translation) is treated
/// as NON-PHYSICAL and earns ZERO progress, not `P·Δd`. No crab covers half a metre of ground
/// in a tick — the observed carapace speed peaks at ~3 m/s (≈0.05 m/tick), so this sits ~10×
/// above any real motion and NEVER fires on a physical trajectory. It exists only to stop a
/// rare solver hiccup (a finite-but-huge one-tick jump, still under the 100 m/s blow-up guard
/// that ends the episode) from injecting a `P·Δd` reward SPIKE into the value/return estimate.
/// Crucially it preserves the telescoping/oscillation-proofness for every realizable path
/// (all real steps are far below it, so none is dropped) and cannot be exploited — the policy
/// cannot deliberately produce a > 0.5 m/tick displacement to farm or to dodge the penalty.
const MAX_PROGRESS_STEP_M: f32 = 0.5;

/// The one-shot TERMINAL bonus a grab earns (the sparse-terminal design — see the module
/// header). Applied at the episode boundary in `systems::finalize_transitions`, NOT inside
/// [`compute_reward`] (the per-tick continuous reward): a claw tip within the grab radius adds
/// this to the grabbing transition's reward and ends the episode as a TRUE terminal. The radius
/// is the SINGLE `curriculum::CURRICULUM_REACH_RADIUS`, shared with the curriculum "reached"
/// signal and the demo ball-hop, so no second radius can drift and a grab implies a reached
/// episode.
///
/// **Scale (relative to a band traverse's progress return, `PROGRESS_WEIGHT·distance`).** A
/// near-band traverse (1.5 m) earns ≈ 36 progress to arrive, a far-band one (9 m) ≈ 216. 50
/// makes a grab the clearly-DOMINANT outcome of a near-band episode (success ≈ 86, the grab
/// ~58 %) without reducing the approach progress to noise, while at the far band the journey
/// still dominates (≈ 266, the grab ~19 %). The approach itself is NOT sparse — dense progress
/// shaping carries the body to grab range at every band, so only the terminal event is sparse
/// and early learning is never signal-starved. A FLAT (not distance-shaped) bonus keeps the
/// last-metre mechanic un-hand-specified: the policy is told only that touching the target is
/// worth ~1.5 near-band traverses, and HOW the tip gets there emerges.
pub(crate) const GRAB_REWARD: f32 = 50.0;

/// The weighted progress term `P·(d_prev − d_now)` — see [`PROGRESS_WEIGHT`]. `distance_closed`
/// is the metres the carapace's planar distance to the target SHRANK over the transition
/// (positive ⇒ closer, negative ⇒ farther). UN-clamped on purpose: that is what makes a closed
/// wobble telescope to exactly zero (the oscillation-proofness the design rests on). `None` —
/// a rescued / teleported body, or a missing pose/target — contributes 0: a teleport is not
/// EARNED travel (the same logic as the `None` credit on a rescue), and crediting the
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

/// The per-tick continuous reward: `P·(d_prev − d_now) − EFFORT_WEIGHT·Σ|dᵢ|^L` — the
/// world-frame progress pull ([`progress_reward`]) minus the cost of the commands that earn it
/// ([`action_effort`]). The sparse terminal grab bonus ([`GRAB_REWARD`]) is NOT part of this
/// function — it is a one-shot event added at the episode boundary (`finalize_transitions`),
/// not a per-tick term.
///
/// The signal stays GLOBAL — progress-through-the-world (plus the terminal grab) with no
/// gait/foot/per-leg term — so locomotion EMERGES. Height and uprightness remain OBSERVATIONS,
/// not reward inputs: this function can't see them, so no pose can be gamed for free reward —
/// only closing ground toward the goal pays per tick, and touching the target pays once.
pub(crate) fn compute_reward(distance_closed: Option<f32>, effort: f32) -> f32 {
    progress_reward(distance_closed) - EFFORT_WEIGHT * effort
}

/// Planar (XZ) distance between two world points. The carapace→target distance the progress
/// term is the per-tick reduction OF (and the carapace→spawn drift diagnostic, and the
/// curriculum band) — all DEFINED on the floor plane. NOT the grab test's `d` (that is the
/// 3D [`dist_3d`], so lowering a claw onto a low target counts).
pub(crate) fn planar_dist(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    (d.x * d.x + d.z * d.z).sqrt()
}

/// Full 3D euclidean distance between two world points — the grab test's `d` (claw tip →
/// target). 3D (not planar) so lowering a claw onto a low ball counts: a ground-only `d` would
/// treat a tip hovering a metre above the target as a grab. `pub(crate)` so the demo's
/// reached-test (`play::target_ball`) measures the SAME `d` the grab/curriculum does.
pub(crate) fn dist_3d(a: Vec3, b: Vec3) -> f32 {
    (a - b).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::PHYSICS_DT;
    use crate::training::systems::MAX_EPISODE_TICKS;

    /// Ground a crab closes in one physics tick at speed `v` (m/s). The reward calibration is
    /// DERIVED from `physics::PHYSICS_DT` (64 Hz), so a tick-rate change can never silently
    /// desync the numbers from the sim the way the old hard-coded `/60.0` did.
    fn per_tick_closed(v: f32) -> f32 {
        v * PHYSICS_DT
    }

    #[test]
    fn progress_closing_raises_receding_lowers() {
        // The core invariant: closing ground toward the target raises the reward, losing
        // ground lowers it, and the two are symmetric (the basis of the telescoping below).
        let effort = action_effort(&[0.3; ACTION_SIZE]);
        let still = compute_reward(Some(0.0), effort);
        let closing = compute_reward(Some(0.01), effort);
        let closing_more = compute_reward(Some(0.02), effort);
        let receding = compute_reward(Some(-0.01), effort);
        assert!(closing > still, "closing ground out-earns standing still");
        assert!(closing_more > closing, "closing more earns more (linear, un-capped)");
        assert!(receding < still, "losing ground lowers the reward below standing still");
        assert!(
            ((closing - still) - (still - receding)).abs() < 1e-6,
            "the progress term is symmetric: +δ closing gains what −δ receding loses"
        );
        // None (rescue/teleport, or missing pose/target) is neutral — no earned travel.
        assert!(
            (compute_reward(None, effort) - still).abs() < 1e-6,
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
    fn grab_bonus_dominates_a_near_band_traverse() {
        // The sparse terminal grab must be the clearly-dominant outcome of a NEAR-band episode
        // (so closing the last stretch and touching the target beats anything the dense progress
        // shaping alone pays on the way), yet the far-band JOURNEY must still out-earn the bonus
        // (out there the traverse is the hard part). PROGRESS_WEIGHT·distance is the progress
        // return of a full traverse (telescoped, path-independent).
        let near_traverse = PROGRESS_WEIGHT * 1.5;
        let far_traverse = PROGRESS_WEIGHT * 9.0;
        assert!(
            GRAB_REWARD > near_traverse,
            "the grab bonus must dominate a near-band traverse's progress: {GRAB_REWARD} vs {near_traverse}"
        );
        assert!(
            far_traverse > GRAB_REWARD,
            "a far-band traverse's progress must still out-earn the grab bonus: {far_traverse} vs {GRAB_REWARD}"
        );
        // …but the bonus must not swamp the near-band APPROACH to noise — the approach progress
        // stays a meaningful fraction of the successful-episode return (here ~42 %).
        assert!(
            near_traverse > 0.3 * (near_traverse + GRAB_REWARD),
            "approach progress must remain a meaningful share of a successful near-band return, \
             not reduced to noise by the grab bonus"
        );
    }

    #[test]
    fn reward_is_progress_minus_effort_no_reach_term() {
        // Reward is EXACTLY `progress − K·Σ|d|^L` — two terms, no near-field reach integral, no
        // height, no uprightness, no hidden term (the grab is a sparse terminal event applied in
        // `finalize_transitions`, not a per-tick term here). With no progress and no command it
        // is exactly zero — in particular STANDING AT THE TARGET earns nothing per tick, so the
        // old hold-farming soft spot is gone: only closing ground pays, and touching pays once.
        assert!(
            compute_reward(None, 0.0).abs() < 1e-6,
            "with no progress and no effort, reward is exactly zero"
        );
        let p = Some(0.01);
        let e = action_effort(&[0.2; ACTION_SIZE]);
        let expected = progress_reward(p) - EFFORT_WEIGHT * e;
        assert!(
            (compute_reward(p, e) - expected).abs() < 1e-6,
            "reward is exactly progress − K·effort"
        );
    }

    #[test]
    fn holding_at_target_accrues_no_reward() {
        // The rl#95 fix, pinned: a crab parked on the target (no progress — it has arrived and is
        // not closing ground) accrues only the effort tax, i.e. ≤ 0 per tick — NEVER the old
        // ~0.21/tick near-field integral. Holding is now strictly worse than the one-shot grab
        // terminal, so there is nothing to farm by camping in the near field.
        let held = compute_reward(Some(0.0), action_effort(&[0.1; ACTION_SIZE]));
        assert!(
            held <= 0.0,
            "a crab holding on the target with no progress must accrue no positive reward: {held}"
        );
    }

    #[test]
    fn higher_drive_lowers_the_reward() {
        // The tax is strictly increasing in DRIVE magnitude, so a harder drive always
        // scores below a gentler one — the lever that makes the crab economical: it spends
        // neural activation only where progress pays for it.
        let still = compute_reward(None, action_effort(&[0.0; ACTION_SIZE]));
        let gentle = compute_reward(None, action_effort(&[0.3; ACTION_SIZE]));
        let hard = compute_reward(None, action_effort(&[0.9; ACTION_SIZE]));
        assert!(
            still > gentle && gentle > hard,
            "reward must fall as drive magnitude rises: still {still} > gentle {gentle} > hard {hard}"
        );
        assert!(
            still.abs() < 1e-6,
            "a still policy with no progress is untaxed and unrewarded: {still} should be zero"
        );
    }

    #[test]
    fn saturating_drive_costs_more_than_a_gentle_drive_at_the_same_command() {
        // THE property the OLD code failed and the whole effort design turns on: two drives that
        // produce the IDENTICAL ±1-clamped command the sim runs, but at different pre-clamp
        // magnitudes, must NOT cost the same. The tax is over the unbounded DRIVE, so a policy
        // that samples |d|≫1 to pin a joint on its rail pays strictly more than one that reaches
        // the same rail gently — a live gradient OFF saturation. (Old: tax over the clamped
        // action ⇒ both |a|=1 ⇒ identical cost ⇒ zero anti-saturation gradient.)
        let gentle_drive = [1.0_f32; ACTION_SIZE]; // sits exactly on the rail
        let saturating_drive = [5.0_f32; ACTION_SIZE]; // slams far past it
        // Both clamp to the SAME command — the sim cannot tell them apart.
        let gentle_cmd: Vec<f32> = gentle_drive.iter().map(|d| d.clamp(-1.0, 1.0)).collect();
        let sat_cmd: Vec<f32> = saturating_drive.iter().map(|d| d.clamp(-1.0, 1.0)).collect();
        assert_eq!(gentle_cmd, sat_cmd, "both drives produce the identical clamped command");
        // …yet the reward must charge the saturating drive strictly more.
        let r_gentle = compute_reward(Some(0.01), action_effort(&gentle_drive));
        let r_sat = compute_reward(Some(0.01), action_effort(&saturating_drive));
        assert!(
            r_sat < r_gentle,
            "a saturating drive must cost STRICTLY MORE than a gentle one at the same command: \
             sat {r_sat} vs gentle {r_gentle}"
        );
    }

    #[test]
    fn effort_cost_calibration() {
        // The tradeoff that matters is PROGRESS vs the cost of the DRIVE that earns it (the tax
        // is over the pre-clamp drives — see `action_effort`), all per-tick figures DERIVED
        // from `physics::PHYSICS_DT` (64 Hz):
        // 1. A still policy with no progress pays no tax and earns nothing — zero.
        let still = compute_reward(None, action_effort(&[0.0; ACTION_SIZE]));
        assert!(still.abs() < 1e-6, "a still policy with no progress is zero: {still}");
        // 2. A real STRIDE — closing ~0.5 m/s of ground (≈0.0078 m/tick at 64 Hz) with an
        //    in-range gait drive (|d| < 1) — must net POSITIVE on progress alone. A |d|=0.7
        //    drive over the actuated joints costs EFFORT_WEIGHT·N·0.49 (≈0.011 at N=38), well
        //    under the stride's 24·0.0078 ≈ 0.19.
        let stride = compute_reward(Some(per_tick_closed(0.5)), action_effort(&[0.7; ACTION_SIZE]));
        assert!(stride > 0.0, "a real stride must net positive after the tax, on progress alone: {stride}");
        // 3. The break-even DRIVE size — where the per-tick effort tax equals the per-tick stride
        //    progress — sits FAR above any in-range gait (|d| < 1): even a |d|=2 drive on EVERY
        //    joint is still net-positive, so a gait is deep in the net-positive region while
        //    saturation (below) is net-negative. Checked as a bracket, not pinned to a per-joint
        //    figure, so it holds as the actuated DOF count grows (bddap/rl#31 widened it to 38).
        let stride_progress = progress_reward(Some(per_tick_closed(0.5)));
        let big_gait_tax = EFFORT_WEIGHT * action_effort(&[2.0; ACTION_SIZE]);
        assert!(
            big_gait_tax < stride_progress,
            "break-even must sit above |d|=2/joint: tax {big_gait_tax} vs stride progress {stride_progress}"
        );
        // 4. A saturation-seeking drive (far past the ±1 clamp) is taxed BELOW a real stride
        //    even while closing ground — `|d|²` keeps climbing past the clamp, so the gradient
        //    pushes the policy out of saturation. At |d|=3 the cost (EFFORT_WEIGHT·N·9 ≈ 0.2 at
        //    N=38) swamps the ≈0.19 stride progress's margin, driving reward toward/below zero.
        let oversaturated =
            compute_reward(Some(per_tick_closed(0.5)), action_effort(&[3.0; ACTION_SIZE]));
        assert!(
            oversaturated < stride,
            "saturation-seeking must be taxed below a real stride: {oversaturated} vs {stride}"
        );
    }

    #[test]
    fn progress_episode_dominates_freezing() {
        // Defect (B) — the episode-scale mismatch — must be closed: a full band traverse must
        // CLEARLY out-earn standing still over a whole MAX_EPISODE_TICKS episode, and the
        // integrated effort tax must stay a LIGHT regularizer, never the dominant term.
        let ticks = MAX_EPISODE_TICKS as f32;
        // WALK: closes ~3 m over the episode (telescoped progress = P·Δd, path-independent),
        // paying a gait-drive tax (|d|≈0.7) every tick.
        let traverse_m = 3.0_f32;
        let walk_progress = PROGRESS_WEIGHT * traverse_m;
        let walk_tax = ticks * EFFORT_WEIGHT * action_effort(&[0.7; ACTION_SIZE]);
        let walk_total = walk_progress - walk_tax;
        // FREEZE: closes ~0 m, paying only the near-still drive tax (μ→0, σ≈0.2 ⇒ |d|≈0.1).
        let freeze_total = -(ticks * EFFORT_WEIGHT * action_effort(&[0.1; ACTION_SIZE]));
        assert!(walk_total > 0.0, "a full traverse must net positive over an episode: {walk_total}");
        assert!(
            walk_total > freeze_total + 30.0,
            "progress must EPISODE-DOMINATE: a traverse {walk_total} ≫ freezing {freeze_total}"
        );
        assert!(
            walk_progress > 4.0 * walk_tax,
            "progress {walk_progress} must dominate the integrated effort {walk_tax} (a light \
             regularizer, not the main term)"
        );
    }
}
