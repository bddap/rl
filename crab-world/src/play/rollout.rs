use bevy::prelude::*;
use rand::{SeedableRng, rngs::StdRng};

use crate::bot::actuator::ACTION_SIZE;
use crate::training::algorithm::OuNoise;

#[derive(Resource)]
pub struct RenderRollout {
    pub(super) band_max_m: f32,
    floor: Option<f32>,
    noise: OuNoise,
    rng: StdRng,
}

impl RenderRollout {
    pub fn new(band_max_m: f32, floor: Option<f32>, seed: u64) -> Self {
        let mut noise = OuNoise::new(1);
        let mut rng = StdRng::seed_from_u64(seed);
        noise.reset(0, &mut rng);
        Self {
            band_max_m,
            floor,
            noise,
            rng,
        }
    }

    pub(super) fn exploration(&mut self) -> Option<([f32; ACTION_SIZE], f32)> {
        self.floor
            .map(|floor| (self.noise.next(0, &mut self.rng), floor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::sensor::OBS_SIZE;
    use crate::policy::Policy;

    #[test]
    fn seeded_exploration_repeats_without_changing_mean_policy() {
        let policy = Policy::untrained();
        let obs = [0.0; OBS_SIZE];
        let mean = policy.act(&obs);
        assert_eq!(
            mean,
            policy.act_with_noise(&obs, Some(([0.0; ACTION_SIZE], -1.0)))
        );
        let mut a = RenderRollout::new(9.0, Some(-1.0), 351);
        let mut b = RenderRollout::new(9.0, Some(-1.0), 351);
        for _ in 0..8 {
            let aa = policy.act_with_noise(&obs, a.exploration());
            assert_eq!(aa, policy.act_with_noise(&obs, b.exploration()));
            assert_ne!(aa, mean);
            assert_eq!(policy.act(&obs), mean);
        }
        assert!(RenderRollout::new(9.0, None, 351).exploration().is_none());
    }
}
