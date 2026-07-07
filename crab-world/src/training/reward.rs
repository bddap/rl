use bevy::prelude::Vec3;

use crate::bot::actuator::ACTION_SIZE;

pub(crate) const EFFORT_WEIGHT: f32 = 0.0006;
const EFFORT_EXP: f32 = 2.0;

pub(crate) fn action_effort(drives: &[f32; ACTION_SIZE]) -> f32 {
    drives.iter().map(|d| d.abs().powf(EFFORT_EXP)).sum()
}

const PROGRESS_WEIGHT: f32 = 24.0;

const MAX_PROGRESS_STEP_M: f32 = 0.5;

pub(crate) const GRAB_REWARD: f32 = 50.0;

fn progress_reward(distance_closed: Option<f32>) -> f32 {
    match distance_closed {
        Some(delta) if is_physical_step(delta) => PROGRESS_WEIGHT * delta,
        _ => 0.0,
    }
}

fn is_physical_step(delta: f32) -> bool {
    delta.is_finite() && delta.abs() <= MAX_PROGRESS_STEP_M
}

pub(crate) fn is_progress_glitch(distance_closed: Option<f32>) -> bool {
    matches!(distance_closed, Some(delta) if !is_physical_step(delta))
}

pub(crate) fn compute_reward(distance_closed: Option<f32>, effort: f32) -> f32 {
    progress_reward(distance_closed) - EFFORT_WEIGHT * effort
}

pub(crate) fn planar_dist(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    (d.x * d.x + d.z * d.z).sqrt()
}

pub(crate) fn dist_3d(a: Vec3, b: Vec3) -> f32 {
    (a - b).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::PHYSICS_DT;
    use crate::training::systems::MAX_EPISODE_TICKS;

    fn per_tick_closed(v: f32) -> f32 {
        v * PHYSICS_DT
    }

    #[test]
    fn progress_closing_raises_receding_lowers() {
        let effort = action_effort(&[0.3; ACTION_SIZE]);
        let still = compute_reward(Some(0.0), effort);
        let closing = compute_reward(Some(0.01), effort);
        let closing_more = compute_reward(Some(0.02), effort);
        let receding = compute_reward(Some(-0.01), effort);
        assert!(closing > still, "closing ground out-earns standing still");
        assert!(
            closing_more > closing,
            "closing more earns more (linear, un-capped)"
        );
        assert!(
            receding < still,
            "losing ground lowers the reward below standing still"
        );
        assert!(
            ((closing - still) - (still - receding)).abs() < 1e-6,
            "the progress term is symmetric: +δ closing gains what −δ receding loses"
        );
        assert!(
            (compute_reward(None, effort) - still).abs() < 1e-6,
            "a teleported/rescued body earns no progress (neutral, like standing still)"
        );
    }

    #[test]
    fn progress_is_linear_and_oscillation_proof() {
        let deltas = [0.02_f32, 0.05, -0.01, -0.08, 0.02];
        debug_assert!((deltas.iter().sum::<f32>()).abs() < 1e-6);
        let total: f32 = deltas.iter().map(|&d| progress_reward(Some(d))).sum();
        assert!(
            total.abs() < 1e-5,
            "a closed loop (Σδ = 0) must pay zero total progress, whatever the path: {total}"
        );
        assert!(
            (progress_reward(Some(0.07)) + progress_reward(Some(-0.03))
                - progress_reward(Some(0.04)))
            .abs()
                < 1e-5,
            "progress is exactly linear in distance closed over physical steps (no clamp)"
        );
    }

    #[test]
    fn progress_glitch_flag_matches_what_progress_zeroes() {
        let physical = [
            Some(0.0_f32),
            Some(0.05),
            Some(-0.05),
            Some(MAX_PROGRESS_STEP_M),
        ];
        for d in physical {
            assert!(
                !is_progress_glitch(d),
                "a physical/paid step is not a glitch: {d:?}"
            );
        }
        assert!(
            !is_progress_glitch(None),
            "None (rescue/teleport) is neutral, not a glitch"
        );
        let glitches = [
            Some(5.0_f32),
            Some(-5.0),
            Some(f32::NAN),
            Some(f32::INFINITY),
        ];
        for d in glitches {
            assert!(
                is_progress_glitch(d),
                "a non-physical present delta is a glitch: {d:?}"
            );
            assert_eq!(
                progress_reward(d),
                0.0,
                "a glitch delta must be zeroed: {d:?}"
            );
        }
    }

    #[test]
    fn progress_glitch_guard_drops_nonphysical_jumps() {
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
        assert!(
            (progress_reward(Some(0.05)) - PROGRESS_WEIGHT * 0.05).abs() < 1e-6,
            "a physical step is paid in full — the guard never fires on real motion"
        );
    }

    #[test]
    fn grab_bonus_dominates_a_near_band_traverse() {
        use crate::training::targets::{BAND_START_MIN, REACH_RADIUS, TARGET_ARENA_HALF};
        let near_approach = PROGRESS_WEIGHT * (BAND_START_MIN - REACH_RADIUS);
        let far_approach = PROGRESS_WEIGHT * (TARGET_ARENA_HALF - REACH_RADIUS);
        assert!(
            GRAB_REWARD > near_approach,
            "the grab bonus must dominate a near-band approach's progress: {GRAB_REWARD} vs {near_approach}"
        );
        assert!(
            far_approach > GRAB_REWARD,
            "a far-band approach's progress must still out-earn the grab bonus: {far_approach} vs {GRAB_REWARD}"
        );
        assert!(
            near_approach > 0.2 * (near_approach + GRAB_REWARD),
            "approach progress must remain a meaningful share of a successful near-band return, \
             not reduced to noise by the grab bonus"
        );
    }

    #[test]
    fn reward_is_progress_minus_effort_no_reach_term() {
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
        let held = compute_reward(Some(0.0), action_effort(&[0.1; ACTION_SIZE]));
        assert!(
            held <= 0.0,
            "a crab holding on the target with no progress must accrue no positive reward: {held}"
        );
    }

    #[test]
    fn higher_drive_lowers_the_reward() {
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
        let gentle_drive = [1.0_f32; ACTION_SIZE];
        let saturating_drive = [5.0_f32; ACTION_SIZE];
        let gentle_cmd: Vec<f32> = gentle_drive.iter().map(|d| d.clamp(-1.0, 1.0)).collect();
        let sat_cmd: Vec<f32> = saturating_drive
            .iter()
            .map(|d| d.clamp(-1.0, 1.0))
            .collect();
        assert_eq!(
            gentle_cmd, sat_cmd,
            "both drives produce the identical clamped command"
        );
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
        let still = compute_reward(None, action_effort(&[0.0; ACTION_SIZE]));
        assert!(
            still.abs() < 1e-6,
            "a still policy with no progress is zero: {still}"
        );
        let stride = compute_reward(
            Some(per_tick_closed(0.5)),
            action_effort(&[0.7; ACTION_SIZE]),
        );
        assert!(
            stride > 0.0,
            "a real stride must net positive after the tax, on progress alone: {stride}"
        );
        let stride_progress = progress_reward(Some(per_tick_closed(0.5)));
        let big_gait_tax = EFFORT_WEIGHT * action_effort(&[2.0; ACTION_SIZE]);
        assert!(
            big_gait_tax < stride_progress,
            "break-even must sit above |d|=2/joint: tax {big_gait_tax} vs stride progress {stride_progress}"
        );
        let oversaturated = compute_reward(
            Some(per_tick_closed(0.5)),
            action_effort(&[3.0; ACTION_SIZE]),
        );
        assert!(
            oversaturated < stride,
            "saturation-seeking must be taxed below a real stride: {oversaturated} vs {stride}"
        );
    }

    #[test]
    fn progress_episode_dominates_freezing() {
        // Episode-scale check: a full band traverse must CLEARLY out-earn standing still
        // over a whole MAX_EPISODE_TICKS episode, and the integrated effort tax must stay
        // a regularizer, never the dominant term (~4:1 progress-dominant at the current
        // EFFORT_WEIGHT).
        let ticks = MAX_EPISODE_TICKS as f32;
        let traverse_m = 3.0_f32;
        let walk_progress = PROGRESS_WEIGHT * traverse_m;
        let walk_tax = ticks * EFFORT_WEIGHT * action_effort(&[0.7; ACTION_SIZE]);
        let walk_total = walk_progress - walk_tax;
        let freeze_total = -(ticks * EFFORT_WEIGHT * action_effort(&[0.1; ACTION_SIZE]));
        assert!(
            walk_total > 0.0,
            "a full traverse must net positive over an episode: {walk_total}"
        );
        assert!(
            walk_total > freeze_total + 30.0,
            "progress must EPISODE-DOMINATE: a traverse {walk_total} ≫ freezing {freeze_total}"
        );
        assert!(
            walk_progress > 2.0 * walk_tax,
            "progress {walk_progress} must dominate the integrated effort {walk_tax} (a \
             regularizer, not the main term)"
        );
    }
}
