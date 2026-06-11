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

/// How many crab environments to run in this world (training parallelism).
/// Demo/screenshot always use 1. Set from `--envs` before BotPlugin builds.
#[derive(Resource, Clone, Copy)]
pub struct NumEnvs(pub usize);

/// Ground-plane spawn origin of each env's crab, indexed by env id. Resets
/// teleport back here, and observations report carapace position relative to
/// it so the policy can't read its env identity (or absolute drift) from x/z.
#[derive(Resource, Default)]
pub struct CrabSpawns(pub Vec<Vec3>);

/// Crabs spaced on a grid: 4 m apart, centered on the origin. Legs reach
/// ~1.3 m and episodes end on a fall, so neighbours can't touch; the whole
/// grid fits the ±10 m arena up to N=16.
fn grid_offset(env: usize, n: usize) -> Vec3 {
    let cols = (n as f32).sqrt().ceil() as usize;
    let spacing = 4.0;
    let col = env % cols;
    let row = env / cols;
    let rows = n.div_ceil(cols);
    Vec3::new(
        (col as f32 - (cols as f32 - 1.0) / 2.0) * spacing,
        0.0,
        (row as f32 - (rows as f32 - 1.0) / 2.0) * spacing,
    )
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
            .init_resource::<CrabSpawns>()
            .init_resource::<body::CrabAssets>()
            .add_systems(Startup, spawn_initial_crabs)
            .add_systems(FixedUpdate, sensor::build_observation.in_set(BotSet::Sense))
            .add_systems(FixedUpdate, actuator::apply_actions.in_set(BotSet::Act));
    }
}

fn spawn_initial_crabs(
    mut commands: Commands,
    assets: Res<body::CrabAssets>,
    num_envs: Res<NumEnvs>,
    mut spawns: ResMut<CrabSpawns>,
    mut actions: ResMut<actuator::CrabActions>,
    mut obs: ResMut<sensor::CrabObservation>,
) {
    let n = num_envs.0;
    actions.resize(n);
    obs.resize(n);
    for env in 0..n {
        let origin = grid_offset(env, n);
        spawns.0.push(origin);
        body::spawn_crab(&mut commands, &assets, origin, env);
    }
}
