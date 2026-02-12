pub mod actuator;
pub mod body;
pub mod brain;
pub mod sensor;

use bevy::prelude::*;

/// Plugin that manages bot spawning and per-frame sensor/actuator updates.
pub struct BotPlugin;

impl Plugin for BotPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<actuator::CrabActions>()
            .init_resource::<sensor::CrabObservation>()
            .add_systems(Startup, spawn_initial_crab)
            .add_systems(
                FixedUpdate,
                (
                    // 1. Build observation from physics state
                    sensor::build_observation,
                    // 2. Brain step happens in TrainingPlugin (between sensor and actuator)
                    // 3. Apply actions to joint motors
                    actuator::apply_actions,
                )
                    .chain(),
            );
    }
}

fn spawn_initial_crab(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    body::spawn_crab(
        &mut commands,
        &mut meshes,
        &mut materials,
        Vec3::ZERO,
    );
}
