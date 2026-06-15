pub mod algorithm;
pub mod inproc;
pub mod session;

use bevy::prelude::*;

use crate::TrainConfig;
use crate::bot::BotSet;

pub struct TrainingPlugin {
    config: TrainConfig,
    /// Periodic-checkpoint cadence for the single-process rollout boundary; see
    /// `Args::save_interval`. Only this (single-process) path honors it.
    save_interval: u32,
}

impl TrainingPlugin {
    pub fn new(config: TrainConfig, save_interval: u32) -> Self {
        Self {
            config,
            save_interval,
        }
    }
}

impl Plugin for TrainingPlugin {
    fn build(&self, app: &mut App) {
        let training_state =
            session::TrainingState::new_single_process(&self.config, self.save_interval);
        app.insert_non_send_resource(training_state)
            .add_systems(
                FixedUpdate,
                (session::brain_step, session::reset_crab)
                    .chain()
                    .in_set(BotSet::Think),
            )
            .add_systems(Last, session::save_on_exit);
    }
}
