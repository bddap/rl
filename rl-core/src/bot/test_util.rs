//! Windowless, GPU-less app that runs the real physics + bot plugins one fixed tick
//! per `app.update()`. Not test-only: [`headless_stack`] is the ONE builder for the
//! no-window physics+bot world, called by every headless entry — the sim tests, the
//! shipped `--check-rest-colliders` diagnostic ([`super::collider_check`]), and the
//! training rollout/test worlds (`training::inproc::build_rollout_app`,
//! `training::systems`'s test app) — so a physics knob added here reaches all of them
//! instead of one hand-synced copy silently missing it.

use std::time::Duration;

use bevy::app::{ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin};
use bevy::prelude::*;
// Only the `#[cfg(test)]` transform-vs-rapier assertion below reaches into Rapier's
// types now that the plugin stack is bundled in `physics::CrabPhysicsPlugin`.
#[cfg(test)]
use bevy_rapier3d::prelude::*;

use super::{BotPlugin, NumEnvs};
use crate::Visuals;
use crate::physics::PhysicsWorldPlugin;

/// The dimensions that legitimately differ between headless entry points: how many
/// envs the world drives, and whether it runs as one of K parallel rollout workers or
/// on its own. Everything else — the windowless render/winit setup, the production
/// fixed timestep + contact spring, the disabled log/gamepad plugins, the physics/bot
/// plugin stack, the one-physics-tick-per-`update()` clock — is identical for all
/// callers and lives once in [`headless_stack`].
pub struct HeadlessStack {
    /// `NumEnvs` for this world (1 for the demo/tests; `--envs` for a rollout worker).
    pub num_envs: usize,
    /// Whether this world is one of K parallel rollout workers or a standalone world —
    /// the only thing that changes the bevy task-pool / executor setup.
    pub role: WorldRole,
}

/// How a headless world relates to the rest of the process, which decides its bevy
/// task-pool and executor setup.
pub enum WorldRole {
    /// A standalone world: the demo, the sim tests, the single training-test app.
    /// Default bevy task pools.
    Standalone,
    /// One of K rollout-worker worlds run in parallel (see `build_rollout_app`). Pins
    /// the task pool to 1 thread and drives the world on `ScheduleRunnerPlugin`, so the
    /// K worlds don't all dispatch onto — and serialize on — the one global
    /// `ComputeTaskPool`. The caller MUST ALSO force every schedule onto the
    /// single-threaded executor AFTER adding its systems; that can't be done in
    /// [`headless_stack`] because the schedules don't exist until the caller wires them.
    RolloutWorker,
}

/// Build the shared windowless physics+bot `App` from [`HeadlessStack`]. The crab is
/// the rig-derived one model. Callers add their own systems (e.g. the training
/// Sense→Think→Act chain) on top of the returned app.
pub fn headless_stack(cfg: HeadlessStack) -> App {
    let mut app = App::new();

    let worker = matches!(cfg.role, WorldRole::RolloutWorker);

    // The windowless plugin set every headless entry shares — `DefaultPlugins` either
    // way, because the sim needs the non-render core plugins it brings (TransformPlugin
    // for GlobalTransform propagation, AssetPlugin, ScenePlugin, InputPlugin); a bare
    // MinimalPlugins omits those and rapier/bot systems then panic "Resource does not
    // exist". The difference between the two builds is ONLY which extra plugins
    // `DefaultPlugins` *contains*, and that's decided by bevy's feature set, not here:
    //
    // - render OFF (the trainer): bevy is built without its render feature, so
    //   `DefaultPlugins` automatically OMITS WindowPlugin/RenderPlugin/winit/gilrs/PBR
    //   (all `#[cfg(feature = "bevy_render"/"bevy_window"/…)]`) and instead includes
    //   `ScheduleRunnerPlugin`. No graphics crate is linked — the whole point of #51.
    // - render ON (the demo/game test builds, which also stand up this headless world):
    //   `DefaultPlugins` includes the window + render plugins, so we turn the window off
    //   and disable the GPU backend (`WgpuSettings::backends = None`) to stay windowless
    //   and GPU-less, matching the single-binary headless build before the split.
    //
    // Either way: drop LogPlugin (no rollout/test spam) and, for a rollout worker, pin
    // the task pool to one thread so K worlds don't serialize on the global pool.
    let mut plugins = DefaultPlugins.build().disable::<bevy::log::LogPlugin>();
    // render ON only: the window + GPU-backend plugins exist only when bevy's render
    // feature is on, so configure them under cfg (render-off they aren't in the group).
    #[cfg(feature = "render")]
    {
        plugins = plugins
            .set(bevy::window::WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .set(bevy::render::RenderPlugin {
                render_creation: bevy::render::settings::RenderCreation::Automatic(
                    bevy::render::settings::WgpuSettings {
                        backends: None,
                        ..default()
                    },
                ),
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<bevy::gilrs::GilrsPlugin>();
    }
    if worker {
        plugins = plugins.set(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions::with_num_threads(1),
        });
        // render OFF, `DefaultPlugins` already carries a ScheduleRunnerPlugin (no window
        // ⇒ the group includes it); a rollout worker drives its world on the runner loop
        // rather than once, so SET it to `run_loop(ZERO)` instead of adding a duplicate.
        #[cfg(not(feature = "render"))]
        {
            plugins = plugins.set(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
        }
    }
    app.add_plugins(plugins);

    // render ON, `DefaultPlugins` has a WindowPlugin (not a ScheduleRunnerPlugin), so a
    // worker needs the runner loop ADDED to drive `update()` with no winit event loop.
    #[cfg(feature = "render")]
    if worker {
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
    }

    // One fixed tick per app.update(): advance the virtual clock by exactly the physics
    // dt, so FixedUpdate runs exactly one step per update.
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f32(crate::physics::PHYSICS_DT),
    ));

    app.insert_resource(Visuals(false))
        .insert_resource(NumEnvs(cfg.num_envs))
        // The production fixed timestep + softened contact spring + Rapier plugin, in
        // the one order that applies the spring — bundled so headless callers can't run
        // physics the demo/training never uses (see physics::CrabPhysicsPlugin).
        .add_plugins(crate::physics::CrabPhysicsPlugin)
        .add_plugins(PhysicsWorldPlugin)
        .add_plugins(BotPlugin);
    app
}

/// A windowless, GPU-less app running the real physics + bot plugins, one fixed
/// physics tick ([`crate::physics::PHYSICS_DT`]) per `app.update()`. The crab is the
/// rig-derived one model. Thin wrapper over [`headless_stack`] for the sim tests and
/// `--check-rest-colliders`.
pub fn headless_app() -> App {
    headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
    })
}

pub fn tick(app: &mut App, n: u32) {
    for _ in 0..n {
        app.update();
    }
}

/// Render truthfulness at the transform level: every crab body part's bevy
/// `Transform` (what the mesh renders at) must equal rapier's rigid-body pose.
#[cfg(test)]
pub fn assert_transforms_match_rapier(app: &mut App) {
    use bevy_rapier3d::plugin::context::RapierRigidBodySet;

    use super::body::CrabBodyPart;

    let mut parts_q = app
        .world_mut()
        .query_filtered::<(Entity, &Transform, &RapierRigidBodyHandle), With<CrabBodyPart>>();
    let parts: Vec<(
        Entity,
        Transform,
        bevy_rapier3d::rapier::dynamics::RigidBodyHandle,
    )> = parts_q
        .iter(app.world())
        .map(|(e, t, h)| (e, *t, h.0))
        .collect();
    assert!(!parts.is_empty());

    let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
    let set = set_q.single(app.world()).expect("rapier context");

    for (e, t, h) in parts {
        let iso = set.bodies.get(h).expect("rapier body").position();
        let pt: Vec3 = iso.translation;
        let pq: Quat = iso.rotation;
        assert!(
            (t.translation - pt).length() < 1e-3,
            "{e:?}: bevy Transform {:?} != rapier body {:?}",
            t.translation,
            pt
        );
        assert!(
            t.rotation.dot(pq).abs() > 1.0 - 1e-4,
            "{e:?}: bevy rotation {:?} != rapier rotation {:?}",
            t.rotation,
            pq
        );
    }
}
