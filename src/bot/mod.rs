pub mod actuator;
pub mod body;
pub mod brain;
pub mod sensor;

use bevy::prelude::*;
use bevy_rapier3d::plugin::PhysicsSet;

/// System sets that enforce Sense → Think → Act ordering across plugins.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum BotSet {
    /// Build observations from physics state.
    Sense,
    /// Neural network forward pass and RL bookkeeping.
    Think,
    /// Apply motor commands to joints.
    Act,
}

/// Plugin that manages bot spawning and per-frame sensor/actuator updates.
pub struct BotPlugin;

impl Plugin for BotPlugin {
    fn build(&self, app: &mut App) {
        // Enforce ordering: Sense → Think → Act, and run the whole loop BEFORE the
        // physics step (Rapier now lives in FixedUpdate too). So each tick observes
        // last step's state, picks an action, writes motor targets, THEN physics
        // integrates — one clean RL step per physics step.
        app.configure_sets(
            FixedUpdate,
            (BotSet::Sense, BotSet::Think, BotSet::Act)
                .chain()
                .before(PhysicsSet::SyncBackend),
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    body::spawn_crab(&mut commands, &mut meshes, &mut materials, Vec3::ZERO);
}
