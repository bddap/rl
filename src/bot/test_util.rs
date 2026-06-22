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

/// Knobs that legitimately differ between the headless entry points. Every field is
/// an EXPLICIT choice the caller makes so a divergence (a slower clock, logging on,
/// the K-thread pool) is visible at the call site, not an accident of a forked copy.
/// Everything NOT here — the windowless render/winit setup, the production fixed
/// timestep + contact spring, the physics/bot plugin stack — is identical for all
/// callers and lives once in [`headless_stack`].
pub struct HeadlessStack {
    /// `NumEnvs` for this world (1 for the demo/tests; `--envs` for a rollout worker).
    pub num_envs: usize,
    /// Virtual-clock advance per `app.update()` — drives exactly one FixedUpdate step
    /// when set to the physics dt, so a horizon is an exact tick count. Production uses
    /// [`crate::physics::PHYSICS_DT`]; pass that unless a test deliberately wants a
    /// different granularity.
    pub tick_dt: Duration,
    /// Keep bevy's `LogPlugin`? Headless callers disable it (no log spam in the
    /// rollout threads / test output); kept as a flag so turning it back on for one
    /// caller is a one-word, visible change.
    pub log: bool,
    /// Pin the bevy task pool to 1 thread and run on the `ScheduleRunnerPlugin` loop —
    /// the rollout-worker scaling fix (K worlds must not share the global compute pool;
    /// see `build_rollout_app`). The caller must ALSO force every schedule onto the
    /// single-threaded executor AFTER adding its systems — that can't be done here
    /// because the schedules don't exist until the caller wires them.
    pub single_thread_pool: bool,
}

/// Build the shared windowless physics+bot `App` from [`HeadlessStack`]. The crab is
/// the rig-derived one model. Callers add their own systems (e.g. the training
/// Sense→Think→Act chain) on top of the returned app.
pub fn headless_stack(cfg: HeadlessStack) -> App {
    let mut app = App::new();

    // The windowless, GPU-less DefaultPlugins all three headless entries share. The
    // gamepad poller (GilrsPlugin) is dropped: a headless world has no input device,
    // so it would only spawn an idle gilrs thread. Built as a binding so the optional
    // `.set`/`.disable` below stay visible rather than buried in a chain.
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
        .disable::<bevy::gilrs::GilrsPlugin>();
    if !cfg.log {
        plugins = plugins.disable::<bevy::log::LogPlugin>();
    }
    if cfg.single_thread_pool {
        plugins = plugins.set(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions::with_num_threads(1),
        });
    }
    app.add_plugins(plugins);

    if cfg.single_thread_pool {
        // No window/winit means no event loop to drive `update()`; the runner loop does.
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
    }

    // One fixed tick per app.update(): advance the virtual clock by exactly `tick_dt`,
    // so FixedUpdate runs one step per update.
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(cfg.tick_dt));

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
/// tick (1/64 s) per `app.update()`. The crab is the rig-derived one model. Thin
/// wrapper over [`headless_stack`] for the sim tests and `--check-rest-colliders`.
pub fn headless_app() -> App {
    headless_stack(HeadlessStack {
        num_envs: 1,
        tick_dt: Duration::from_secs_f32(crate::physics::PHYSICS_DT),
        log: false,
        single_thread_pool: false,
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
