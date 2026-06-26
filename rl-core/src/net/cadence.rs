//! Deterministic physics/sim cadence reconciliation for the networked NN crab (GCR).
//!
//! The crab body + its Sense→Think→Act brain step at [`crate::physics::PHYSICS_HZ`] (64 Hz);
//! the lockstep sim advances at [`crate::net::sim::TICK_HZ`] (30 Hz). On a NETWORKED round
//! every peer MUST advance the float body the SAME number of physics steps between two
//! lockstep ticks — otherwise the per-tick `phys_digest` folded into
//! [`crate::net::sim::Sim::state_hash`] differs and the peers desync by construction. Wall
//! clock can't decide that count (it differs per machine and per frame rate), so this is a
//! pure INTEGER accumulator keyed off the tick index alone: identical on every peer from the
//! shared [`Default`] start.
//!
//! 64 / 30 is not an integer, so the per-tick step count alternates (mostly 2, periodically
//! 3). The Bresenham-style accumulator emits a deterministic sequence that sums to exactly
//! [`PHYSICS_HZ`] steps over any [`TICK_HZ`] ticks, so the body advances exactly one second
//! of physics per one second of sim with no long-run drift.

use crate::net::sim::TICK_HZ;
use crate::physics::PHYSICS_HZ;

/// Doles out [`PHYSICS_HZ`] physics steps across [`TICK_HZ`] lockstep ticks. Each tick adds
/// `PHYSICS_HZ` of credit, emits the whole steps that buys (`credit / TICK_HZ`), and carries
/// the remainder — so over `TICK_HZ` ticks it emits exactly `PHYSICS_HZ` steps.
///
/// `Copy` and its whole state is one `u64`: two peers starting from [`Default`] and stepping
/// the same number of ticks always agree on the per-tick AND cumulative step counts, which is
/// the property the GCR fold relies on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhysicsCadence {
    /// Physics-step credit not yet spent as a whole step, in `PHYSICS_HZ·tick` units. Always
    /// `< TICK_HZ` once [`Self::steps_for_next_tick`] has returned; starts at 0.
    acc: u64,
}

impl PhysicsCadence {
    /// Physics steps to run for the next lockstep tick. Call EXACTLY once per ADVANCED tick
    /// (never for a stalled/attempted tick), so the accumulator stays in phase with the sim
    /// and so two peers that advanced the same ticks ran the same number of physics steps.
    pub fn steps_for_next_tick(&mut self) -> u32 {
        self.acc += PHYSICS_HZ;
        let n = self.acc / TICK_HZ;
        self.acc -= n * TICK_HZ;
        n as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Over exactly `TICK_HZ` ticks the cadence emits exactly `PHYSICS_HZ` steps — one second
    /// of physics per one second of sim, the no-drift invariant.
    #[test]
    fn one_second_emits_exactly_physics_hz_steps() {
        let mut c = PhysicsCadence::default();
        let total: u32 = (0..TICK_HZ).map(|_| c.steps_for_next_tick()).sum();
        assert_eq!(total as u64, PHYSICS_HZ);
        // And the accumulator has returned to its start, so it repeats cleanly forever.
        assert_eq!(c, PhysicsCadence::default());
    }

    /// No long-run drift: over many seconds the cumulative count stays exactly `k·PHYSICS_HZ`.
    #[test]
    fn no_drift_over_many_seconds() {
        let mut c = PhysicsCadence::default();
        let mut total: u64 = 0;
        let secs = 1000u64;
        for _ in 0..(secs * TICK_HZ) {
            total += c.steps_for_next_tick() as u64;
        }
        assert_eq!(total, secs * PHYSICS_HZ);
    }

    /// Every per-tick count is the floor or ceil of `PHYSICS_HZ / TICK_HZ` — the body never
    /// lurches by a big variable batch (which would read as a stutter and stress the solver).
    #[test]
    fn per_tick_count_is_floor_or_ceil() {
        let mut c = PhysicsCadence::default();
        let floor = (PHYSICS_HZ / TICK_HZ) as u32;
        let ceil = floor + u32::from(PHYSICS_HZ % TICK_HZ != 0);
        for _ in 0..(10 * TICK_HZ) {
            let n = c.steps_for_next_tick();
            assert!(n == floor || n == ceil, "step count {n} not in {{{floor},{ceil}}}");
        }
    }

    /// Determinism: two independently-constructed cadences (the two peers) produce the
    /// identical step sequence — the whole point of an integer, wall-clock-free accumulator.
    #[test]
    fn two_peers_agree_step_for_step() {
        let mut a = PhysicsCadence::default();
        let mut b = PhysicsCadence::default();
        for _ in 0..500 {
            assert_eq!(a.steps_for_next_tick(), b.steps_for_next_tick());
            assert_eq!(a, b);
        }
    }
}
