pub mod algorithm;
pub mod replay;
pub mod session;

use bevy::prelude::*;

/// Plugin that manages RL training.
pub struct TrainingPlugin;

impl Plugin for TrainingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_non_send_resource(session::TrainingState::new())
            .add_systems(
                FixedUpdate,
                (
                    session::brain_step,
                    session::reset_crab,
                )
                    .chain(),
            );
    }
}
