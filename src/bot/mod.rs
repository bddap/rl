pub mod actuator;
pub mod body;
pub mod brain;
pub mod sensor;

use bevy::prelude::*;

use crate::HeadlessMode;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum BotSet {
    Sense,
    Think,
    Act,
}

pub struct BotPlugin;

impl Plugin for BotPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            FixedUpdate,
            (BotSet::Sense, BotSet::Think, BotSet::Act).chain(),
        );

        app.init_resource::<actuator::CrabActions>()
            .init_resource::<sensor::CrabObservation>()
            .add_systems(Startup, spawn_initial_crab)
            .add_systems(FixedUpdate, sensor::build_observation.in_set(BotSet::Sense))
            .add_systems(FixedUpdate, actuator::apply_actions.in_set(BotSet::Act));
    }
}

fn spawn_initial_crab(
    mut commands: Commands,
    headless: Res<HeadlessMode>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    if headless.0 {
        body::spawn_crab_headless(&mut commands, Vec3::ZERO);
    } else {
        let mut meshes = meshes.expect("Assets<Mesh> missing in graphical mode");
        let mut materials = materials.expect("Assets<StandardMaterial> missing in graphical mode");
        body::spawn_crab(&mut commands, &mut meshes, &mut materials, Vec3::ZERO);
    }
}
