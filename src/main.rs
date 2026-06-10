mod bot;
mod combat;
mod physics;
mod play;
mod player;
mod training;

use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use clap::Parser;

/// Crab Combat — RL-trained crab bots learn to stand, walk, and fight.
#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Args {
    /// Train headless: no window, maximum simulation speed.
    #[arg(long)]
    headless: bool,

    /// Play with a trained crab: load the checkpoint, drive it with the policy
    /// (no learning), orbit camera + poke/reset controls.
    #[arg(long)]
    demo: bool,

    /// Render one frame to this PNG and exit (windowless, GPU on). For inspecting
    /// the trained crab without a display.
    #[arg(long, value_name = "PATH")]
    screenshot: Option<PathBuf>,

    /// Physics steps to simulate before taking the screenshot.
    #[arg(long, default_value_t = 200)]
    screenshot_settle: u32,

    /// Screenshot width in pixels.
    #[arg(long, default_value_t = 1280)]
    width: u32,

    /// Screenshot height in pixels.
    #[arg(long, default_value_t = 720)]
    height: u32,

    /// Directory for checkpoint files. On startup, if the directory contains a
    /// previous checkpoint it will be loaded automatically. During training,
    /// checkpoints are saved here periodically and on exit.
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,

    /// Save a checkpoint every N PPO updates (0 to disable periodic saves).
    #[arg(long, default_value_t = 50)]
    save_interval: u32,

    /// Stop headless training after exactly this many physics ticks (0 = run
    /// until killed). The budget is counted in ticks, never wall-clock, so a run
    /// simulates an identical amount regardless of machine speed or load — the
    /// "fixed ticks, not real time" guarantee an assumed time↔tick relation can't
    /// give.
    #[arg(long, default_value_t = 0)]
    ticks: u64,

    /// Benchmark only: skip NN inference in the train loop (hold zero actions),
    /// isolating physics + engine overhead from network cost. Training is
    /// meaningless under this flag — it exists to measure the per-step bottleneck.
    #[arg(long)]
    bench_skip_nn: bool,
}

/// What the process is doing this run. Train can be headless or windowed; demo
/// and screenshot always render.
#[derive(Clone)]
enum AppMode {
    Train,
    Demo,
    Screenshot { path: PathBuf, settle: u32 },
}

/// Whether to spawn visual assets (meshes, lights). False only for headless
/// training, where rendering is off entirely.
#[derive(Resource, Clone, Copy)]
pub struct Visuals(pub bool);

fn main() {
    let args = Args::parse();

    let mode = if let Some(path) = args.screenshot.clone() {
        AppMode::Screenshot {
            path,
            settle: args.screenshot_settle,
        }
    } else if args.demo {
        AppMode::Demo
    } else {
        AppMode::Train
    };

    // Headless training is the only mode without visuals.
    let visuals = !(matches!(mode, AppMode::Train) && args.headless);

    let mut app = App::new();

    match &mode {
        AppMode::Train if args.headless => {
            // No window, GPU off, run the schedule as fast as possible.
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
            app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(Duration::ZERO));
            app.insert_resource(fast_virtual_time());
            // Advance the clock by a FIXED amount per update instead of reading the
            // wall clock. With the 100x virtual speed + 10s max_delta above, each
            // update drains a fixed ~640 physics ticks regardless of how long the
            // frame actually took — so tick count is a function of updates, not
            // machine speed or load. Without this, a slow frame (e.g. a PPO update)
            // makes the next frame's wall-clock delta — and thus its tick count —
            // balloon, and a stall could spiral. Paired with --ticks, a run
            // simulates an exact, reproducible amount.
            app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                Duration::from_secs_f64(0.1),
            ));
        }
        AppMode::Screenshot { .. } => {
            // No window, but GPU ON so we can render to an image. Real-time 60 Hz
            // loop at default 1x: one physics step + one render per frame, so the
            // capture counter (render frames) also tracks simulated time and the
            // GPU pipeline warms over the same frames. (A fast/100x clock decouples
            // them — physics races while render frames crawl, and early frames
            // render black before the pipeline warms up.)
            app.add_plugins(
                DefaultPlugins
                    .set(bevy::window::WindowPlugin {
                        primary_window: None,
                        exit_condition: bevy::window::ExitCondition::DontExit,
                        ..default()
                    })
                    .disable::<bevy::winit::WinitPlugin>(),
            );
            app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
                Duration::from_secs_f64(1.0 / 60.0),
            ));
        }
        _ => {
            // Windowed: train-with-viz or demo.
            app.add_plugins(DefaultPlugins);
            if matches!(mode, AppMode::Train) {
                app.add_plugins(RapierDebugRenderPlugin::default());
            }
        }
    }

    app.insert_resource(Visuals(visuals))
        // Fixed physics dt: each Bevy tick advances the sim by exactly 1/64 s, so
        // physics is identical in headless training and the real-time demo and is
        // reproducible run-to-run. The default Variable timestep keys off
        // wall-clock delta, which in a headless loop with no render clock collapses
        // to ~0 dt — the crab never falls, training optimises a frozen spawn pose,
        // and the policy faceplants the moment real-time physics actually steps it.
        .insert_resource(TimestepMode::Fixed {
            dt: 1.0 / 64.0,
            substeps: 1,
        })
        // Run physics IN FixedUpdate, lockstep with the Sense→Think→Act brain loop
        // (which also lives in FixedUpdate). Rapier's default schedule is PostUpdate
        // — one physics step per rendered frame — while FixedUpdate runs as many
        // catch-up ticks as the (100x, headless) virtual clock accumulates. That
        // decouples them: the brain takes ~64 actions per single physics step, so
        // the crab sees a near-frozen world during training yet real dynamics in the
        // real-time demo. In FixedUpdate it's exactly one physics step per brain step
        // in both.
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
        .add_plugins(physics::PhysicsWorldPlugin)
        .add_plugins(bot::BotPlugin);

    match mode {
        AppMode::Train => {
            app.add_plugins(training::TrainingPlugin::new(args.clone()));
            if !args.headless {
                app.add_systems(Startup, spawn_fixed_camera);
            }
        }
        AppMode::Demo => {
            app.add_plugins(play::DemoPlugin {
                checkpoint_dir: args.checkpoint_dir.clone(),
            });
        }
        AppMode::Screenshot { path, settle } => {
            app.add_plugins(play::ScreenshotPlugin {
                checkpoint_dir: args.checkpoint_dir.clone(),
                path,
                settle,
                width: args.width,
                height: args.height,
            });
        }
    }

    app.run();
}

/// Virtual clock that runs 100× wall speed for headless/offscreen modes, so a
/// fixed number of physics steps elapses in a fraction of the real time.
fn fast_virtual_time() -> Time<Virtual> {
    let mut t = Time::<Virtual>::default();
    t.set_relative_speed(100.0);
    t.set_max_delta(Duration::from_secs(10));
    t
}

/// Fixed overhead camera for windowed training visualization.
fn spawn_fixed_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 15.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
