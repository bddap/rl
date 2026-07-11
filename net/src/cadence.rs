use crate::sim::TICK_HZ;
use crab_world::physics::PHYSICS_HZ;

/// Cumulative physics steps after `tick` whole sim ticks: the 64:30 staircase in
/// closed form, anchored at absolute tick 0. THE one source for the step bunching —
/// the host's per-tick pump and the renderer's physics-step-time cockpit
/// interpolation (rl#264) both derive from it, so they cannot disagree. Only the
/// server pumps physics (a remote-adopt client's `FixedUpdate` is parked), so no
/// cross-peer alignment state is needed — and none exists to reset on restart.
pub fn cumulative_steps(tick: u64) -> u64 {
    PHYSICS_HZ * tick / TICK_HZ
}

/// Physics steps the pump owes for the sim tick `tick` (1-based: the tick being
/// stepped INTO). 64:30 makes this 2 on most ticks, 3 on four ticks in every 30.
pub fn steps_for_tick(tick: u64) -> u32 {
    debug_assert!(tick >= 1, "tick is 1-based: the tick being stepped INTO");
    (cumulative_steps(tick) - cumulative_steps(tick - 1)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_second_emits_exactly_physics_hz_steps() {
        let total: u64 = (1..=TICK_HZ).map(|t| steps_for_tick(t) as u64).sum();
        assert_eq!(total, PHYSICS_HZ);
    }

    #[test]
    fn no_drift_over_many_seconds() {
        let secs = 1000u64;
        let total: u64 = (1..=secs * TICK_HZ).map(|t| steps_for_tick(t) as u64).sum();
        assert_eq!(total, secs * PHYSICS_HZ);
        assert_eq!(total, cumulative_steps(secs * TICK_HZ));
    }

    #[test]
    fn per_tick_count_is_floor_or_ceil() {
        let floor = (PHYSICS_HZ / TICK_HZ) as u32;
        let ceil = floor + u32::from(!PHYSICS_HZ.is_multiple_of(TICK_HZ));
        for t in 1..=(10 * TICK_HZ) {
            let n = steps_for_tick(t);
            assert!(
                n == floor || n == ceil,
                "step count {n} not in {{{floor},{ceil}}}"
            );
        }
    }

    #[test]
    fn staircase_stays_within_one_step_of_ideal() {
        // The renderer's uniform-clock sampling (rl#264) relies on |floor(r·t) − r·t| < 1.
        let r = PHYSICS_HZ as f64 / TICK_HZ as f64;
        for t in 0..(10 * TICK_HZ) {
            let ideal = r * t as f64;
            let actual = cumulative_steps(t) as f64;
            assert!(
                (ideal - actual).abs() < 1.0,
                "tick {t}: {actual} vs {ideal}"
            );
        }
    }
}
