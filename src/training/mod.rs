pub mod algorithm;
pub mod session;

use bevy::prelude::*;

use crate::Args;
use crate::bot::BotSet;

pub struct TrainingPlugin {
    args: Args,
}

impl TrainingPlugin {
    pub fn new(args: Args) -> Self {
        Self { args }
    }
}

impl Plugin for TrainingPlugin {
    fn build(&self, app: &mut App) {
        let training_state = session::TrainingState::new(&self.args);
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
