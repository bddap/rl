use crate::sim::TICK_HZ;
use crab_world::physics::PHYSICS_HZ;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhysicsCadence {
    acc: u64,
}

impl PhysicsCadence {
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

    #[test]
    fn one_second_emits_exactly_physics_hz_steps() {
        let mut c = PhysicsCadence::default();
        let total: u32 = (0..TICK_HZ).map(|_| c.steps_for_next_tick()).sum();
        assert_eq!(total as u64, PHYSICS_HZ);
        assert_eq!(c, PhysicsCadence::default());
    }

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

    #[test]
    fn per_tick_count_is_floor_or_ceil() {
        let mut c = PhysicsCadence::default();
        let floor = (PHYSICS_HZ / TICK_HZ) as u32;
        let ceil = floor + u32::from(!PHYSICS_HZ.is_multiple_of(TICK_HZ));
        for _ in 0..(10 * TICK_HZ) {
            let n = c.steps_for_next_tick();
            assert!(
                n == floor || n == ceil,
                "step count {n} not in {{{floor},{ceil}}}"
            );
        }
    }

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
