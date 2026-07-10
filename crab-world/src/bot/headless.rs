use std::time::Duration;

use bevy::app::{ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin};
use bevy::prelude::*;
#[cfg(test)]
use bevy_rapier3d::prelude::*;

use super::{BotPlugin, NumEnvs};
use crate::Visuals;
use crate::physics::PhysicsWorldPlugin;

pub struct HeadlessStack {
    pub num_envs: usize,
    pub role: WorldRole,
    /// Which arena physics this world gets: the ±10 m walled training box, or the
    /// unbounded GCR inference field ([`crate::physics::Arena`]). A headless world that
    /// models the GCR client (the NN-crab probes, the pump-equivalence test) MUST pick
    /// `OpenField` so it steps the same world the client does.
    pub arena: crate::physics::Arena,
    /// `Visuals` for this world. `true` arms the render-gated systems (the skin, the
    /// GCR repose publisher) AND the rl#116 pose sentinel — the armed-render smoke
    /// test steps this exact configuration headless, the one the play-day crash
    /// showed no test covered. Headless training/eval/probe worlds stay `false`,
    /// which also keeps the write-`Transform`-to-teleport test idiom legal there.
    pub visuals: Visuals,
}

pub enum WorldRole {
    Standalone,
    RolloutWorker,
}

pub fn headless_stack(cfg: HeadlessStack) -> App {
    let mut app = App::new();

    let worker = matches!(cfg.role, WorldRole::RolloutWorker);

    let mut plugins = DefaultPlugins.build().disable::<bevy::log::LogPlugin>();
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
        #[cfg(not(feature = "render"))]
        {
            plugins = plugins.set(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
        }
    }
    app.add_plugins(plugins);

    #[cfg(feature = "render")]
    if worker {
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
    }

    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f32(crate::physics::PHYSICS_DT),
    ));

    app.insert_resource(cfg.visuals)
        .insert_resource(NumEnvs(cfg.num_envs))
        .add_plugins(crate::physics::CrabPhysicsPlugin)
        .add_plugins(PhysicsWorldPlugin { arena: cfg.arena })
        .add_plugins(BotPlugin);
    app
}

pub fn headless_app() -> App {
    headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        arena: crate::physics::Arena::WalledBox,
        visuals: Visuals(false),
    })
}

pub fn tick(app: &mut App, n: u32) {
    for _ in 0..n {
        app.update();
    }
}

pub fn pin_single_thread_pools() {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        unsafe { std::env::set_var("RAYON_NUM_THREADS", "1") };
    }
    // Only write when the value is wrong: this runs again for every in-process eval
    // (the keep-best gate, bddap/rl#233), and `set_var` in a threaded process races any
    // concurrent `getenv` — an already-correct pin must be a pure read.
    if std::env::var("MATMUL_NUM_THREADS").as_deref() != Ok("1") {
        if let Ok(prev) = std::env::var("MATMUL_NUM_THREADS") {
            eprintln!(
                "MATMUL_NUM_THREADS={prev} re-enables the shared matmul tree (deadlock risk \
                 under K>1 rollout threads, nondeterministic reduction order); forcing 1"
            );
        }
        unsafe { std::env::set_var("MATMUL_NUM_THREADS", "1") };
    }
    TaskPoolOptions::with_num_threads(1).create_default_pools();
}

pub fn force_serial_schedules(app: &mut App) {
    use bevy::ecs::schedule::{ExecutorKind, Schedules};
    let mut schedules = app.world_mut().resource_mut::<Schedules>();
    for (_label, schedule) in schedules.iter_mut() {
        schedule.set_executor_kind(ExecutorKind::SingleThreaded);
    }
}

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
            !super::pose_sentinel::pose_diverges(&t, pt, pq),
            "{e:?}: bevy Transform {:?}/{:?} != rapier body {:?}/{:?}",
            t.translation,
            t.rotation,
            pt,
            pq
        );
    }
}
