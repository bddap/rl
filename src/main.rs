mod bot;
mod collider_editor;
mod combat;
mod debug_sliders;
mod physics;
mod play;
mod player;
mod training;

use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use clap::{Parser, Subcommand};

use training::session::STEPS_PER_ROLLOUT;

/// Crab Combat — RL-trained crab bots learn to stand, walk, and fight.
///
/// With no subcommand the binary runs the single-process path (train headless or
/// windowed, demo, or screenshot) from these flags — the established entrypoint.
/// The `learn` subcommand is the in-process multi-threaded trainer; its flags live
/// on the subcommand, so a stray `--workers` on a single-process run is a parse
/// error rather than a silent no-op.
#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Cli {
    #[command(flatten)]
    args: Args,

    #[command(subcommand)]
    command: Option<Command>,
}

/// The in-process multi-threaded training mode. One learner (the main thread) owns
/// the policy + optimizer + normalizer; K rollout THREADS each step their own
/// rapier world on their own core and feed buffers back over a channel. This buys
/// the wall-clock speedup the single-world `--envs` path can't (one world steps
/// single-threaded) without any multiprocess IPC. See `training::inproc`.
#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Run the in-process trainer: spawn K rollout threads, snapshot the policy to
    /// each, collect their rollouts, and run the PPO update. Resumes from
    /// `--checkpoint-dir` and stops at the `--ticks` budget, exactly as a
    /// single-process headless run does.
    Learn(LearnArgs),
}

/// Training/app config shared by the single-process path and the in-process
/// learner+rollout-thread (all of which build a `TrainingState`). The `learn` mode
/// flattens it so e.g. `--checkpoint-dir` / `--ticks` mean the same thing
/// everywhere.
#[derive(Parser, Debug, Clone)]
pub struct TrainConfig {
    /// Directory for checkpoint files. On startup, if the directory contains a
    /// previous checkpoint it will be loaded automatically. During training,
    /// checkpoints are saved here periodically and on exit.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,

    /// Stop training after this many physics ticks (0 = run until killed). The budget
    /// is counted in ticks, never wall-clock, so a run simulates an identical amount
    /// regardless of machine speed or load — the "fixed ticks, not real time"
    /// guarantee an assumed time↔tick relation can't give. The single-process path
    /// stops exactly at N; `learn` checks the budget once per PPO iteration, so it
    /// stops at the first iteration boundary at or after N (overshooting by up to one
    /// K·(--envs)·H iteration's worth of ticks).
    #[arg(long, default_value_t = 0)]
    pub ticks: u64,

    /// Benchmark only: skip NN inference in the train loop (hold zero actions),
    /// isolating physics + engine overhead from network cost. Training is
    /// meaningless under this flag — it exists to measure the per-step bottleneck.
    #[arg(long)]
    pub bench_skip_nn: bool,

    /// Number of crab environments trained in parallel in one world (one batched
    /// NN pass per tick). Crabs sit on a 4 m grid; 16 is the most the ±10 m
    /// arena holds. Demo/screenshot modes always run 1. (Under `learn` this is M,
    /// the env count PER rollout thread; total parallel envs = `--workers` × M.)
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..=16))]
    pub envs: u64,
}

/// Single-process / interactive config: the shared training config plus the
/// demo/screenshot/window knobs and the periodic-save cadence that only the
/// single-process path uses.
#[derive(Parser, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub train: TrainConfig,

    /// Save a checkpoint every N PPO updates (0 disables periodic saves). Lives only
    /// on the single-process path: it is read at the rollout boundary in `brain_step`,
    /// which `learn` never reaches — that learner checkpoints every iteration directly,
    /// so an interval would be dead there.
    #[arg(long, default_value_t = 50)]
    pub save_interval: u32,

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

    /// Demo only: show a TEMPORARY egui panel of physics sliders (contact spring,
    /// length_unit, joint limit spring + leg friction, restitution, substeps) that
    /// live-tune the running crab. A throwaway feel-tuning aid; off by default and
    /// never wired into training/headless. See `src/debug_sliders.rs`.
    #[arg(long)]
    debug_sliders: bool,

    /// DEV: fit colliders from the glTF model once, write the typed table to this RON
    /// path, then exit (no window, no physics) — the only place the fit runs. Model is
    /// `CRAB_MODEL_PATH`, else the dev `sally.glb`. Load it with `--body fitted`.
    #[arg(long, value_name = "OUT.ron")]
    bake_colliders: Option<PathBuf>,

    /// DEV: open the interactive collider-placement editor on this RON table, then
    /// exit. The table is seeded from the auto-fit (or resumed if it already exists);
    /// step bone-by-bone and hand-place each collider against the bind-pose mesh.
    /// Needs a window + GPU. See `collider_editor`.
    #[arg(long, value_name = "OUT.ron")]
    edit_colliders: Option<PathBuf>,

    /// Collider geometry + mass the body spawns with. `hand-coded` (default) is the
    /// tuned, trained-on body; `fitted` loads the baked `--body-table`, replacing only
    /// collider shape/mass/placement (joints/axes/limits/rest stance stay hand-coded).
    /// Unproven — never the default (see MESHFIT_PLAN.md phase 4).
    #[arg(long, value_enum, default_value_t = BodySel::HandCoded)]
    body: BodySel,

    /// Baked collider table that `--body fitted` loads (written by `--bake-colliders`).
    #[arg(long, value_name = "PATH", default_value = "colliders.ron")]
    body_table: PathBuf,
}

/// Learner orchestration: the shared training config plus how many rollout threads
/// to fan out.
#[derive(Parser, Debug, Clone)]
struct LearnArgs {
    #[command(flatten)]
    train: TrainConfig,

    /// Number of rollout threads K, each stepping its own world on its own core.
    /// Default is PHYSICAL cores minus 2 (floored at one) — physical, not logical,
    /// so it never oversubscribes a hyperthreaded pair onto one core — leaving the
    /// rest of the machine a couple of cores. Pass an explicit value to use more.
    /// Clamped to 1..=64.
    #[arg(long)]
    workers: Option<usize>,

    /// Rollout horizon H: physics ticks each thread rolls per iteration before
    /// handing its buffers back. Per-iteration sample count is K·(--envs)·H.
    #[arg(long, default_value_t = STEPS_PER_ROLLOUT as u64)]
    horizon: u64,

    /// Stop after this many PPO iterations (0 = unbounded). A benchmark / A-B knob;
    /// the production budget is `--ticks` (total physics ticks). Whichever limit is
    /// hit first stops the run.
    #[arg(long, default_value_t = 0)]
    iters: u64,

    /// Niceness applied to the whole process — the learner and its rollout threads
    /// share it (POSIX priority is per-process; higher = yields more CPU). Positive
    /// so a foreground game always preempts training even when the threads saturate
    /// their cores. Clamped to 0..=19 (0 disables; a negative nice would raise
    /// priority and needs privilege, so it is floored to 0 rather than attempted).
    #[arg(long, default_value_t = 10)]
    nice: i32,
}

/// `--body` selector: which collider source the crab spawns with.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
enum BodySel {
    HandCoded,
    Fitted,
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
    let cli = Cli::parse();

    // The `learn` entry point short-circuits the normal Bevy app: the learner steps
    // no world itself (it owns the policy and runs PPO), and spawns K rollout
    // threads that each drive their own headless app by hand. See `training::inproc`.
    if let Some(Command::Learn(l)) = cli.command {
        // run_learner owns nicing (it lowers process priority before building any
        // world) so a foreground game preempts training.
        training::inproc::run_learner(
            &l.train,
            training::inproc::default_workers(l.workers),
            l.horizon,
            l.iters,
            l.nice,
        );
        return;
    }

    let args = cli.args;

    // DEV bake: fit colliders once, write the table, exit — no bevy app needed.
    if let Some(out) = args.bake_colliders.clone() {
        bake_colliders(&out);
        return;
    }

    // DEV editor: open the interactive collider-placement window, then exit. Builds
    // its own minimal app (mesh + gizmos + egui), no physics or BotPlugin.
    if let Some(table) = args.edit_colliders.clone() {
        collider_editor::run(&table);
        return;
    }

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
        AppMode::Train => args.train.envs as usize,
        AppMode::Demo | AppMode::Screenshot { .. } => 1,
    };

    // Resolve the collider source up front so it is present before `BotPlugin`
    // builds `CrabAssets` from it (`--body fitted` reads the baked table now,
    // failing loudly if it is missing rather than silently spawning hand-coded).
    let body_source = resolve_body_source(args.body, &args.body_table);

    app.insert_resource(Visuals(visuals))
        .insert_resource(bot::NumEnvs(num_envs))
        // ORDER matters for `--body fitted`: this must precede `BotPlugin` (below),
        // whose `CrabAssets` eager-reads it at build to pre-bake the fitted meshes.
        // Inserted after it, `from_world` sees no source and uses the hand-coded
        // default — correct for hand-coded, wrong for a requested fitted body.
        .insert_resource(body_source)
        // Why a FIXED timestep at all: the default Variable timestep keys off
        // wall-clock delta, which in a headless loop with no render clock collapses
        // to ~0 dt — the crab never falls, training optimises a frozen spawn pose,
        // and the policy faceplants the moment real-time physics actually steps it.
        // The dt + sub-steps (shared with every headless test) live in one place,
        // physics::fixed_timestep, so production and test physics can't drift.
        .insert_resource(physics::fixed_timestep())
        // Seed Rapier's context with the softened contact spring (physics::
        // CONTACT_SOFTNESS) — one source, shared with training + tests — before the
        // plugin spawns the context from it.
        .insert_resource(physics::rapier_context_init())
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
            app.add_plugins(training::TrainingPlugin::new(
                args.train.clone(),
                args.save_interval,
            ));
            if !args.headless {
                app.add_systems(Startup, spawn_fixed_camera);
            }
        }
        AppMode::Demo => {
            app.add_plugins(play::DemoPlugin {
                checkpoint_dir: args.train.checkpoint_dir.clone(),
                live_checkpoint_dir: args.live_checkpoint_dir.clone(),
                manual_control: args.manual_control,
            });
            // TEMPORARY physics-tuning overlay, demo-only and off by default.
            if args.debug_sliders {
                app.add_plugins(debug_sliders::DebugSlidersPlugin);
            }
        }
        AppMode::Screenshot { path, settle } => {
            app.add_plugins(play::ScreenshotPlugin {
                checkpoint_dir: args.train.checkpoint_dir.clone(),
                path,
                settle,
                width: args.width,
                height: args.height,
            });
        }
    }

    app.run();
}

/// DEV `--bake-colliders`: load the glTF model, fit the typed collider table, and
/// write it to `out` as RON. Exits the process with a nonzero status on any
/// failure (missing model, fit error, write error) so a broken bake fails the
/// command instead of writing a half-table.
fn bake_colliders(out: &std::path::Path) {
    let Some(model_path) = bot::meshfit::model_path() else {
        eprintln!(
            "bake-colliders: no model — set CRAB_MODEL_PATH or place sally.glb at the dev path"
        );
        std::process::exit(1);
    };
    let model = match bot::meshfit::LoadedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bake-colliders: load {model_path:?}: {e}");
            std::process::exit(1);
        }
    };

    // Fit with the per-part reasoning, so the bake prints WHY each part got its
    // shape — the reviewable summary the lean artifact itself doesn't carry.
    let report = bot::meshfit::bake_report(&model);
    println!("fitting {} parts from {model_path:?}:", report.len());
    println!(
        "  {:<20} {:>7} {:>6} | {:>6} {:>5} {:>5} {:>5} | {:>7} {:>6}",
        "part", "prim", "res", "mass", "elong", "iso", "flat", "span_m", "reason"
    );
    for r in &report {
        let c = &r.choice;
        // Show the BAKED primitive + why, not the chooser's: every jointed part is
        // baked as a bone-stretched capsule regardless of the cloud descriptors, so
        // the chooser's tag/reason would be stale for those.
        let reason = if r.fitted.part == bot::meshfit::PartId::Carapace {
            c.reason
        } else {
            "jointed → capsule stretched along its bone (red→green)"
        };
        println!(
            "  {:<20} {:>7} {:>6.2} | {:>6.4} {:>5.2} {:>5.2} {:>5.2} | {:>7.3}  {}",
            format!("{:?}", r.fitted.part),
            r.fitted.primitive.tag(),
            c.chosen_residual,
            r.fitted.mass_properties().0,
            c.elongation,
            c.isotropy,
            c.flatness,
            r.span_len,
            reason,
        );
    }

    let body = bot::meshfit::FittedBody::from_reports(&report);
    let ron = match body.to_ron() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("bake-colliders: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = std::fs::write(out, ron) {
        eprintln!("bake-colliders: write {out:?}: {e}");
        std::process::exit(1);
    }
    // Total mass is derived from each part's primitive + density — a one-line
    // summary of what the table will weigh when spawned.
    let total_mass: f32 = body.parts.iter().map(|p| p.mass_properties().0).sum();
    println!(
        "baked {} parts → {out:?} (total fitted mass {total_mass:.3} kg)",
        body.parts.len()
    );
}

/// Resolve `--body` into the runtime [`bot::body::CrabBodySource`]. For `fitted`
/// this reads and validates the baked table now; a missing or stale table is a
/// hard error (exit) rather than a silent fall back to hand-coded, so an
/// explicit `--body fitted` can't quietly run the wrong physics.
fn resolve_body_source(sel: BodySel, table: &std::path::Path) -> bot::body::CrabBodySource {
    match sel {
        BodySel::HandCoded => bot::body::CrabBodySource::HandCoded,
        BodySel::Fitted => {
            let ron = std::fs::read_to_string(table).unwrap_or_else(|e| {
                eprintln!(
                    "--body fitted: read {table:?}: {e} (run `--bake-colliders {}` first)",
                    table.display()
                );
                std::process::exit(1);
            });
            match bot::meshfit::FittedBody::from_ron(&ron) {
                Ok(body) => bot::body::CrabBodySource::Fitted(body),
                Err(e) => {
                    eprintln!("--body fitted: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
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
pub(crate) fn fast_virtual_time() -> Time<Virtual> {
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
