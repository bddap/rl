pub mod actuator;
pub mod body;
pub mod brain;
pub mod sensor;

use bevy::prelude::*;

/// Plugin that manages bot spawning and per-frame brain/actuator updates.
pub struct BotPlugin;

impl Plugin for BotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_initial_crab);
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
