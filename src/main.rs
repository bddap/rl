mod bot;
mod combat;
mod physics;
mod player;
mod training;

use std::path::PathBuf;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use clap::Parser;

/// Crab Combat — RL-trained crab bots learn to stand, walk, and fight.
#[derive(Parser, Debug)]
#[command(version)]
pub struct Args {
    /// Run without a window at maximum simulation speed.
    #[arg(long)]
    headless: bool,

    /// Directory for checkpoint files. On startup, if the directory contains a
    /// previous checkpoint it will be loaded automatically. During training,
    /// checkpoints are saved here periodically and on exit.
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,

    /// Save a checkpoint every N PPO updates (0 to disable periodic saves).
    #[arg(long, default_value_t = 50)]
    save_interval: u32,
}

/// Resource indicating whether we're running headless (no rendering).
#[derive(Resource, Clone, Copy)]
pub struct HeadlessMode(pub bool);

fn main() {
    let args = Args::parse();
    let headless = args.headless;

    let mut app = App::new();

    if headless {
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
        app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::ZERO,
        ));
        let mut virtual_time = bevy::time::Time::<bevy::time::Virtual>::default();
        virtual_time.set_relative_speed(100.0);
        virtual_time.set_max_delta(std::time::Duration::from_secs(10));
        app.insert_resource(virtual_time);
    } else {
        app.add_plugins(DefaultPlugins);
        app.add_plugins(RapierDebugRenderPlugin::default());
    }

    app.insert_resource(HeadlessMode(headless))
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(physics::PhysicsWorldPlugin)
        .add_plugins(bot::BotPlugin)
        .add_plugins(training::TrainingPlugin::new(args))
        .add_systems(Startup, move || {
            if headless {
                info!("Headless training mode — no window, max speed physics");
            }
        });

    app.run();
}
