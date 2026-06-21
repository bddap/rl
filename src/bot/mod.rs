pub mod actuator;
pub mod body;
pub mod brain;
pub mod collider_check;
pub mod meshfit;
#[cfg(test)]
mod reset_test;
pub mod rig;
pub mod sensor;
#[cfg(test)]
mod sim_truth_test;
pub mod skin;
pub mod skin_diag;
// Not test-only: the `--check-rest-colliders` diagnostic ships in the binary and
// reuses `headless_app`/`tick` to build and settle the same windowless physics
// world the sim tests run, so one app builder serves both.
pub(crate) mod test_util;

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

/// Crabs spaced on a grid: 4 m apart, centered on the origin, so the whole grid
/// fits the ±10 m arena up to N=16. With far targets the crabs WALK and their paths
/// cross, but each env owns a private collision bit (see [`body::crab_collision`]),
/// so neighbours pass through one another — the grid is just the start layout, not a
/// no-touch guarantee.
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
            // Always present so `build_observation` (shared with the demo) can read
            // it; the training plugin populates a real target per env, the demo
            // leaves it empty (a zero target vector in the obs).
            .init_resource::<sensor::CrabTargets>()
            .init_resource::<CrabSpawns>()
            .init_resource::<body::CrabAssets>()
            .add_message::<CrabRescued>()
            .add_systems(Startup, spawn_initial_crabs)
            .add_systems(FixedUpdate, rescue_nonfinite_crabs.before(BotSet::Sense))
            .add_systems(FixedUpdate, sensor::build_observation.in_set(BotSet::Sense))
            .add_systems(FixedUpdate, actuator::apply_actions.in_set(BotSet::Act));

        // The optional skinned model only makes sense with rendering on; in
        // headless training there is no AssetServer to load it with.
        if app
            .world()
            .get_resource::<crate::Visuals>()
            .is_some_and(|v| v.0)
        {
            skin::register(app);
        }
    }
}

/// Sent when [`rescue_nonfinite_crabs`] had to rebuild an env's crab.
/// Training must treat it as an episode terminator: the replacement crab is
/// back at spawn, and letting the episode run on would smear that teleport
/// into the reward stream.
#[derive(Message)]
pub struct CrabRescued {
    pub env: usize,
}

/// System: rebuilds any crab whose pose has gone non-finite (solver blowup,
/// tunneling). Runs ahead of Sense — so observations never see NaN — and
/// ahead of the physics sync, so the poisoned multibody is removed before
/// rapier's solver touches it again: NaN joint coordinates make the motor
/// constraint clamp on NaN bounds and panic the app. This catches corruption
/// on the tick after it happens; a NaN that panics the solver within the very
/// step that created it is still fatal, and the outer auto-resume loop is the
/// backstop for that rarer case.
pub fn rescue_nonfinite_crabs(
    mut commands: Commands,
    assets: Res<body::CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts: Query<(Entity, &body::CrabEnvId, &Transform), With<body::CrabBodyPart>>,
    mut rescued: MessageWriter<CrabRescued>,
) {
    let mut sick: Vec<usize> = Vec::new();
    for (_, id, t) in parts.iter() {
        if (!t.translation.is_finite() || !t.rotation.is_finite()) && !sick.contains(&id.0) {
            sick.push(id.0);
        }
    }
    for env in sick {
        let origin = spawns.0.get(env).copied().unwrap_or(Vec3::ZERO);
        respawn_crab(
            &mut commands,
            &assets,
            parts
                .iter()
                .filter(|(_, id, _)| id.0 == env)
                .map(|(e, _, _)| e),
            origin,
            env,
        );
        rescued.write(CrabRescued { env });
    }
}

/// Tears down every entity of one env's crab and spawns a fresh one at
/// `origin`. This is the only reliable reset: rapier 0.32 cannot rewrite
/// multibody joint coordinates in place, so a crab whose multibody state has
/// gone non-finite (e.g. tunneled through the floor at speed) is
/// unrecoverable by teleport-and-zero — the poisoned joint state survives any
/// amount of Transform/Velocity writing and every "reset" reproduces the same
/// wedged pose. Dropping the whole tree and rebuilding always works.
pub fn respawn_crab(
    commands: &mut Commands,
    assets: &body::CrabAssets,
    parts: impl Iterator<Item = Entity>,
    origin: Vec3,
    env: usize,
) {
    respawn_crab_rotated(commands, assets, parts, origin, env, Quat::IDENTITY);
}

/// Like [`respawn_crab`] but spawns the fresh crab rigidly rotated by `init_rotation`
/// (training's randomized-start curriculum — see `reset_crab`). `Quat::IDENTITY`
/// reproduces the upright respawn exactly.
pub fn respawn_crab_rotated(
    commands: &mut Commands,
    assets: &body::CrabAssets,
    parts: impl Iterator<Item = Entity>,
    origin: Vec3,
    env: usize,
    init_rotation: Quat,
) {
    for e in parts {
        commands.entity(e).despawn();
    }
    body::spawn_crab(commands, assets, origin, env, init_rotation);
}

fn spawn_initial_crabs(
    mut commands: Commands,
    assets: Res<body::CrabAssets>,
    num_envs: Res<NumEnvs>,
    mut spawns: ResMut<CrabSpawns>,
    mut actions: ResMut<actuator::CrabActions>,
    mut obs: ResMut<sensor::CrabObservation>,
    mut targets: ResMut<sensor::CrabTargets>,
) {
    let n = num_envs.0;
    actions.resize(n);
    obs.resize(n);
    targets.resize(n);
    for env in 0..n {
        let origin = grid_offset(env, n);
        spawns.0.push(origin);
        // Initial spawn stays upright; the randomized-start curriculum kicks in on the
        // first reset (see `reset_crab`). A clean first frame also keeps any non-training
        // caller of this system (the demo) upright.
        body::spawn_crab(&mut commands, &assets, origin, env, Quat::IDENTITY);
    }
}
