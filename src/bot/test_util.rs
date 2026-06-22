//! Windowless, GPU-less app that runs the real physics + bot plugins one fixed tick
//! per `app.update()`. Not test-only: [`headless_stack`] is the ONE builder for the
//! no-window physics+bot world, called by every headless entry — the sim tests, the
//! shipped `--check-rest-colliders` diagnostic ([`super::collider_check`]), and the
//! training rollout/test worlds (`training::inproc::build_rollout_app`,
//! `training::session`'s test app) — so a physics knob added here reaches all of them
//! instead of one hand-synced copy silently missing it.

use std::time::Duration;

use bevy::app::{ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin};
use bevy::prelude::*;
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

    // The windowless, GPU-less DefaultPlugins all three headless entries share. The
    // gamepad poller (GilrsPlugin) is dropped: a headless world has no input device,
    // so it would only spawn an idle gilrs thread. Built as a binding so the optional
    // 1-thread-pool `.set` below stays visible rather than buried in a chain.
    let mut plugins = DefaultPlugins
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
        .disable::<bevy::gilrs::GilrsPlugin>()
        // No log spam in the rollout threads / test output.
        .disable::<bevy::log::LogPlugin>();
    let worker = matches!(cfg.role, WorldRole::RolloutWorker);
    if worker {
        plugins = plugins.set(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions::with_num_threads(1),
        });
    }
    app.add_plugins(plugins);

    if worker {
        // No window/winit means no event loop to drive `update()`; the runner loop does.
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
    }

    // One fixed tick per app.update(): advance the virtual clock by exactly the physics
    // dt, so FixedUpdate runs exactly one step per update.
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f32(crate::physics::PHYSICS_DT),
    ));

    app.insert_resource(Visuals(false))
        .insert_resource(NumEnvs(cfg.num_envs))
        // Same fixed timestep as production (one source — see physics::fixed_timestep)
        // so headless callers can't pass under physics the demo/training run never uses.
        .insert_resource(crate::physics::fixed_timestep())
        // Same softened contact spring as production (physics::CONTACT_SOFTNESS).
        .insert_resource(crate::physics::rapier_context_init())
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
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
