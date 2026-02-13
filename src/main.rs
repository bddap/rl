mod bot;
mod combat;
mod physics;
mod player;
mod training;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

/// Resource indicating whether we're running headless (no rendering).
#[derive(Resource, Clone, Copy)]
pub struct HeadlessMode(pub bool);

fn main() {
    let headless = std::env::args().any(|a| a == "--headless");

    let mut app = App::new();

    if headless {
        // Headless: use DefaultPlugins but skip window creation and rendering.
        // Set WGPU to skip GPU initialization entirely.
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
                .disable::<bevy::winit::WinitPlugin>(),
        );
        // Drive the main schedule as fast as possible (no sleep between ticks).
        app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::ZERO,
        ));
        // Speed up virtual time so FixedUpdate runs many iterations per frame.
        // At 100x speed, each real ms → 100ms virtual → ~6 fixed steps at 64Hz.
        // Raise max_delta to 10s so Bevy doesn't clip large virtual advances.
        let mut virtual_time = bevy::time::Time::<bevy::time::Virtual>::default();
        virtual_time.set_relative_speed(100.0);
        virtual_time.set_max_delta(std::time::Duration::from_secs(10));
        app.insert_resource(virtual_time);
    } else {
        // Windowed: full rendering
        app.add_plugins(DefaultPlugins);
        app.add_plugins(RapierDebugRenderPlugin::default());
    }

    app.insert_resource(HeadlessMode(headless))
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(physics::PhysicsWorldPlugin)
        .add_plugins(bot::BotPlugin)
        .add_plugins(training::TrainingPlugin)
        .add_systems(Startup, move || {
            if headless {
                info!("Headless training mode — no window, max speed physics");
            }
        });

    app.run();
}
