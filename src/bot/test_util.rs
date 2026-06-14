//! Shared harness for sim-level tests: a windowless, GPU-less app that runs
//! the real physics + bot plugins one fixed tick per `app.update()`.

use std::time::Duration;

use bevy::prelude::*;
use bevy_rapier3d::plugin::context::RapierRigidBodySet;
use bevy_rapier3d::prelude::*;

use super::body::CrabBodyPart;
use super::{BotPlugin, NumEnvs};
use crate::Visuals;
use crate::physics::PhysicsWorldPlugin;

pub fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
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
            .disable::<bevy::log::LogPlugin>(),
    );
    // One fixed tick (1/64 s) per app.update(), like headless training.
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(1.0 / 64.0),
    ));
    app.insert_resource(Visuals(false))
        .insert_resource(NumEnvs(1))
        // Same fixed timestep as production (one source — see physics::fixed_timestep)
        // so tests can't pass under physics the demo/training run never uses.
        .insert_resource(crate::physics::fixed_timestep())
        // Same softened contact spring as production (physics::CONTACT_SOFTNESS).
        .insert_resource(crate::physics::rapier_context_init())
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
        .add_plugins(PhysicsWorldPlugin)
        .add_plugins(BotPlugin);
    app
}

pub fn tick(app: &mut App, n: u32) {
    for _ in 0..n {
        app.update();
    }
}

/// Render truthfulness at the transform level: every crab body part's bevy
/// `Transform` (what the mesh renders at) must equal rapier's rigid-body pose.
pub fn assert_transforms_match_rapier(app: &mut App) {
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
