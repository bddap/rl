//! `rl-demo` — the windowed demo + screenshot of a trained crab policy. Links
//! `crab-world` with render ON (bevy_render/bevy_pbr/wgpu), so it can put the crab on a
//! screen; it does no training (that's the headless `rl-train` binary).
//!
//! Modes (mutually exclusive; one is required):
//! - `--demo` — load a checkpoint and drive the crab with the policy (no learning)
//!   behind an orbit camera with poke/reset controls. Borderless fullscreen unless
//!   `--windowed`. The Steam launch target.
//! - `--screenshot PATH` — render one settled frame to a PNG and exit (windowless,
//!   GPU on) — inspect the trained crab without a display.
//!
//! This is the render half of the old single `rl` binary (rl#51); the training half
//! is `rl-train`, the multiplayer game is `game`.

use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use clap::Parser;
use crab_world::{TrainConfig, Visuals, bot, physics, play};

/// Crab Combat demo — render a trained crab policy.
///
/// One of `--demo` / `--screenshot` is required (a bare invocation has nothing to
/// show). The checkpoint to load comes from `--checkpoint-dir` (the shared
/// [`TrainConfig`]); training itself is the separate `rl-train` binary.
#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Args {
    #[command(flatten)]
    pub train: TrainConfig,

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
}

/// What this run is rendering. Both modes always render (the binary has no headless
/// path — that's `rl-train`).
#[derive(Clone)]
enum AppMode {
    Demo,
    Screenshot { path: PathBuf, settle: u32 },
}

fn main() {
    let args = Args::parse();

    // rl-demo is a PLAYER-FACING surface (the windowed couch demo and its screenshot),
    // so the canonical Sally mesh is REQUIRED here, not optional. The only crabs allowed
    // on a player-facing surface are the purchased Sally mesh and the physics bones we
    // built around it; a MISSING model must NOT silently fall back to the skinless
    // procedural body (that ships a non-Sally crab to the screen — the silent-fallback
    // bug). Fail loud and refuse the surface instead. This is the explicit player-facing
    // vs training split: the headless trainer (`rl-train`) is the OTHER binary and keeps
    // the no-skin procedural body by design (a fresh policy needs no cosmetic mesh) — see
    // `crab_world::bot::skin::register`, which only runs under render+visuals.
    let Some(p) = bot::meshfit::model_path() else {
        eprintln!(
            "rl-demo: no crab model resolved — refusing to render. The demo shows the \
             purchased Sally mesh (CRAB_MODEL_PATH, default `sally.glb`, under \
             BEVY_ASSET_ROOT/assets); it will NOT silently fall back to a skinless crab. \
             Fetch it with scripts/fetch-sally.sh, or point CRAB_MODEL_PATH at the model."
        );
        std::process::exit(1);
    };
    // Preflight the resolved model so a broken/incomplete asset fails fast with the real
    // reason, instead of panicking deep in Startup (or blaming CRAB_MODEL_PATH for a
    // parse error in a model that was present).
    match bot::meshfit::LoadedModel::load(&p) {
        Err(e) => {
            eprintln!("crab model {p:?}: {e}");
            std::process::exit(1);
        }
        // A model that loads but lacks the expected crab bones builds no recipe.
        // Reject it here, not as `spawn_crab`'s expect deep in Startup with a
        // message that wrongly blames a missing/corrupt file.
        Ok(model) => {
            if bot::rig::build_recipe(&model).is_none() {
                eprintln!(
                    "crab model {p:?}: loaded but has none of the expected crab bones (e.g. Def_leg_01.000.L)"
                );
                std::process::exit(1);
            }
        }
    }

    let mode = if let Some(path) = args.screenshot.clone() {
        AppMode::Screenshot {
            path,
            settle: args.screenshot_settle,
        }
    } else if args.demo {
        AppMode::Demo
    } else {
        // A bare `rl-demo` with no mode flag has nothing to show. (Training is the
        // `rl-train` binary.)
        eprintln!("no mode selected. rl-demo needs --demo or --screenshot.");
        std::process::exit(2);
    };

    let mut app = App::new();

    match &mode {
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
            // Rapier collider wireframes draw via gizmos, which DO render into the
            // offscreen screenshot camera (Bevy 0.18) — but only if the plugin is
            // present. The other arms add it; Screenshot has its own arm, so gate it
            // here on the same flag or the captured PNG never shows the colliders.
            if std::env::var_os("RL_DEBUG_COLLIDERS").is_some() {
                app.add_plugins(RapierDebugRenderPlugin {
                    // Collider shapes only — the default also draws per-body axes +
                    // joint markers, which on a 31-part body is an unreadable tangle.
                    mode: DebugRenderMode::COLLIDER_SHAPES,
                    ..default()
                });
                // Bright always-in-front markers at each joint pivot — the companion
                // to the collider cage, gated on the same flag (see body.rs).
                bot::body::register_pivot_markers(&mut app);
            }
        }
        AppMode::Demo => {
            // The demo defaults to borderless fullscreen (the Steam launch target is
            // a couch screen) unless --windowed.
            let fullscreen = !args.windowed;
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
            // With the stand-in primitive meshes removed, Rapier's debug-render is
            // the only in-engine view of the true colliders, so the skin can be
            // checked against the actual physics shapes. `enabled` only sets the
            // INITIAL state — the demo's right-arrow toggles `DebugRenderContext`
            // live — and the demo starts the cage on iff RL_DEBUG_COLLIDERS is set.
            app.add_plugins(RapierDebugRenderPlugin {
                enabled: std::env::var_os("RL_DEBUG_COLLIDERS").is_some(),
                // Collider shapes only — the default also draws per-body axes +
                // joint markers, which on a 31-part body is an unreadable tangle.
                mode: DebugRenderMode::COLLIDER_SHAPES,
                ..default()
            });
            // Pivot markers gate on RL_DEBUG_COLLIDERS: a deliberate diagnostic, not
            // default chrome.
            if std::env::var_os("RL_DEBUG_COLLIDERS").is_some() {
                bot::body::register_pivot_markers(&mut app);
            }
        }
    }

    // Demo and screenshot always render and drive exactly one crab (visuals on,
    // 1 env — parallel envs are a training concept, and training is `rl-train` only).
    app.insert_resource(Visuals(true))
        .insert_resource(bot::NumEnvs(1))
        // The crab's physics setup as one plugin: the shared dt + sub-steps + softened
        // contact spring, and Rapier running IN FixedUpdate (lockstep with the
        // Sense→Think→Act brain loop, one physics step per brain step). One source with
        // every headless test and the `rl-train learn` rollout worlds, and the init
        // ordering is enforced internally, so the demo can't drift from the physics
        // training optimizes (see physics::CrabPhysicsPlugin).
        .add_plugins(physics::CrabPhysicsPlugin)
        .add_plugins(physics::PhysicsWorldPlugin)
        .add_plugins(bot::BotPlugin)
        .add_systems(FixedUpdate, contact_audit);

    match mode {
        AppMode::Demo => {
            app.add_plugins(play::DemoPlugin {
                checkpoint_dir: args.train.checkpoint_dir.clone(),
                live_checkpoint_dir: args.live_checkpoint_dir.clone(),
                manual_control: args.manual_control,
            });
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
