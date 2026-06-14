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

    /// Run the demo in a window instead of the default borderless fullscreen.
    #[arg(long)]
    windowed: bool,

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

    /// Directory the DEMO hot-reloads the policy from while running — the LIVE
    /// training output. The demo loads its initial policy from `--checkpoint-dir`
    /// (a stable copy) and then, every couple of seconds, swaps in a newer
    /// checkpoint that appears here, so a left-open demo tracks training without a
    /// relaunch. Unset, or a missing/half-written dir, = no swap. Demo mode only.
    #[arg(long, value_name = "PATH")]
    live_checkpoint_dir: Option<PathBuf>,

    /// Demo only: replace the trained policy with hands-on gamepad control — D-pad
    /// up/down picks a joint, the right stick drives its torque, all else held at
    /// zero. A physics feel-test, not a learned driver.
    #[arg(long)]
    manual_control: bool,

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

    /// Number of crab environments trained in parallel in one world (one batched
    /// NN pass per tick). Crabs sit on a 4 m grid; 16 is the most the ±10 m
    /// arena holds. Demo/screenshot modes always run 1.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..=16))]
    envs: u64,
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
            // Train-with-viz stays windowed; the demo defaults to borderless
            // fullscreen (the Steam launch target is a couch screen) unless
            // --windowed.
            let fullscreen = matches!(mode, AppMode::Demo) && !args.windowed;
            app.add_plugins(DefaultPlugins.set(bevy::window::WindowPlugin {
                primary_window: Some(bevy::window::Window {
                    title: "Crab RL".into(),
                    mode: if fullscreen {
                        bevy::window::WindowMode::BorderlessFullscreen(
                            bevy::window::MonitorSelection::Primary,
                        )
                    } else {
                        bevy::window::WindowMode::Windowed
                    },
                    ..default()
                }),
                ..default()
            }));
            if matches!(mode, AppMode::Train) {
                app.add_plugins(RapierDebugRenderPlugin::default());
            }
        }
    }

    // Parallel envs are a training concept; the interactive/render modes drive
    // exactly one crab.
    let num_envs = match &mode {
        AppMode::Train => args.envs as usize,
        AppMode::Demo | AppMode::Screenshot { .. } => 1,
    };

    app.insert_resource(Visuals(visuals))
        .insert_resource(bot::NumEnvs(num_envs))
        // Why a FIXED timestep at all: the default Variable timestep keys off
        // wall-clock delta, which in a headless loop with no render clock collapses
        // to ~0 dt — the crab never falls, training optimises a frozen spawn pose,
        // and the policy faceplants the moment real-time physics actually steps it.
        // The dt + sub-steps (shared with every headless test) live in one place,
        // physics::fixed_timestep, so production and test physics can't drift.
        .insert_resource(physics::fixed_timestep())
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
        .add_plugins(bot::BotPlugin)
        .add_systems(FixedUpdate, contact_audit);

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
                live_checkpoint_dir: args.live_checkpoint_dir.clone(),
                manual_control: args.manual_control,
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

/// Diagnostic (enable with RL_CONTACT_AUDIT=1): every 64 ticks, prints every
/// crab-part-vs-crab-part contact pair currently penetrating more than 5 mm,
/// deepest first. Ground contacts are excluded. Answers "are the legs
/// actually intersecting" with numbers instead of squinting at renders.
fn contact_audit(
    sim: Query<&bevy_rapier3d::plugin::context::RapierContextSimulation>,
    cols: Query<&bevy_rapier3d::plugin::context::RapierContextColliders>,
    parts: Query<
        (Option<&bot::body::CrabJoint>, Has<bot::body::CrabCarapace>),
        With<bot::body::CrabBodyPart>,
    >,
    mut tick: Local<u32>,
) {
    if std::env::var_os("RL_CONTACT_AUDIT").is_none() {
        return;
    }
    *tick += 1;
    if *tick % 64 != 2 {
        return;
    }
    let (Ok(sim), Ok(cols)) = (sim.single(), cols.single()) else {
        return;
    };
    let name = |p: (Option<&bot::body::CrabJoint>, bool)| {
        p.0.map(|j| format!("{:?}", j.id))
            .unwrap_or_else(|| "Carapace".to_string())
    };
    let mut worst: Vec<(f32, String, String)> = Vec::new();
    for pair in sim.narrow_phase.contact_pairs() {
        let (Some(e1), Some(e2)) = (
            cols.collider_entity(pair.collider1),
            cols.collider_entity(pair.collider2),
        ) else {
            continue;
        };
        let (Ok(p1), Ok(p2)) = (parts.get(e1), parts.get(e2)) else {
            continue;
        };
        let mut depth = 0.0f32;
        for m in &pair.manifolds {
            for pt in &m.points {
                depth = depth.max(-pt.dist);
            }
        }
        if depth > 0.005 {
            worst.push((depth, name(p1), name(p2)));
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "AUDIT tick {}: {} crab-crab pairs >5mm penetration",
        *tick,
        worst.len()
    );
    for (d, a, b) in worst.iter().take(6) {
        println!("  {:>4.0}mm {a} vs {b}", d * 1000.0);
    }
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
