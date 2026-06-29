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

    /// Render the ball-chase headless to this mp4 and exit (windowless, GPU on).
    /// Loads the policy from `--checkpoint-dir`, drives the crab after the moving
    /// target, captures every sim tick, and encodes with ffmpeg. Decoupled from
    /// wall-clock — one sim tick per rendered frame — so a slower-than-realtime render
    /// produces a true-speed clip. Also prints an objective `DRIVE_STATS` line
    /// (mean|drive|, mean effort) for comparing two policies' actuation.
    #[arg(long, value_name = "PATH")]
    render_video: Option<PathBuf>,

    /// `--render-video` clip length in SIMULATED seconds (ticks = seconds × the
    /// physics rate). The mp4 plays this many seconds of crab time; it is unrelated to
    /// how long the render itself takes (the render is as fast as compute allows).
    #[arg(long, default_value_t = 8.0)]
    seconds: f32,

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
    /// Offline ball-chase clip: step the sim one tick per frame, render that frame,
    /// encode to `path` at the end. `seconds` is the clip's SIMULATED length.
    RenderVideo { path: PathBuf, seconds: f32 },
}

fn main() {
    let args = Args::parse();

    // ALL env mutation happens HERE, before `otel::init` — `otel::init` spawns the OTLP
    // exporter's background threads (batch span/log processors, periodic metric reader) when
    // export is enabled, and `set_var` is unsound once another thread may `getenv`.
    //
    // SAFETY for every `set_var` below: program start, single-threaded, before both
    // `otel::init` and `App::new` spawn any thread — the same pattern (and justification) as
    // `crab_world::bot::headless`'s `set_var`.

    // Default the log filter to WARN. rl-demo disables bevy's LogPlugin (below) so the otel
    // subscriber is the only one, which means it ALSO governs OTLP export — and bevy/wgpu emit
    // a torrent of INFO every frame. At the default `info` that torrent floods the telemetry
    // sink when export is on (tens of MB per run), burying the signals that matter. WARN keeps
    // every error/warning — the canonical-mesh error and the checkpoint-mismatch refusal
    // included — and drops the per-frame noise. `RUST_LOG` still overrides for local debugging.
    if std::env::var_os("RUST_LOG").is_none() {
        unsafe { std::env::set_var("RUST_LOG", "warn") };
    }

    // rl-demo is a PLAYER-FACING surface (the windowed couch demo and its screenshot), so the
    // crab the player sees should be the purchased Sally mesh. But a MISSING/broken mesh is no
    // longer a hard refuse (owner 2026-06-28: too strict): instead fall back to the honest
    // physics-bones view — Rapier's debug-render of the REAL colliders, the same bones the body
    // uses — and emit a LOUD OTEL error so the absent asset is visible in telemetry. The one
    // thing still forbidden is a SILENT skinless body (the silent-fallback bug — ships a
    // non-Sally crab with no signal). This is the explicit player-facing vs training split: the
    // headless trainer (`rl-train`) keeps the no-skin procedural body by design.
    let mesh_status = crab_world::mesh_fallback::canonical_mesh_status();
    // The crab mesh THIS run renders, decided here ONCE from the full preflight: the real Sally
    // when usable, else `None` → the honest fallback. Inserted as `CrabModelPath` before `BotPlugin`
    // (below) so the body, skin, and silhouette all flip together off this one explicit value — see
    // `body::CrabModelPath` for why this replaces the old env-poison redirect. The physics bones are
    // then made VISIBLE by booting the render-mode cycle in `colliders` (see `initial_render_mode`).
    let mesh_state = crab_world::bot::body::CrabModelPath(if mesh_status.is_ok() {
        crab_world::bot::meshfit::model_path()
    } else {
        None
    });

    // OTEL/tracing: install the shared subscriber after `RUST_LOG` is final (above). The guard
    // must outlive the whole run, so it's bound here and dropped only when `main` returns.
    let _otel = otel::init("rl-demo");

    // `Some(reason)` ⇒ the canonical mesh is unusable, kept so the loudness lands in THREE
    // places, not just telemetry: the OTEL error below (stderr + OTLP), the forced colliders
    // render mode (`mesh_ok` decides the initial mode below), and — in the windowed demo — an
    // on-screen banner (see the Demo arm). The owner's bug (rl#706) was exactly this: a fallback
    // he could SEE but not identify, so the failure must name itself on the screen he's looking
    // at, not only in a log he isn't.
    let mesh_fallback_reason: Option<String> = mesh_status.err();
    if let Some(reason) = &mesh_fallback_reason {
        // LOUD via telemetry (stderr + OTLP), shared with the game surface so both name the
        // missing Sally identically (rl#706).
        crab_world::mesh_fallback::log_fallback(crab_world::mesh_fallback::Surface::RlDemo, reason);
    }
    let mesh_ok = mesh_fallback_reason.is_none();

    let mode = if let Some(path) = args.render_video.clone() {
        AppMode::RenderVideo {
            path,
            seconds: args.seconds,
        }
    } else if let Some(path) = args.screenshot.clone() {
        AppMode::Screenshot {
            path,
            settle: args.screenshot_settle,
        }
    } else if args.demo {
        AppMode::Demo
    } else {
        // A bare `rl-demo` with no mode flag has nothing to show. (Training is the
        // `rl-train` binary.)
        eprintln!("no mode selected. rl-demo needs --demo, --screenshot, or --render-video.");
        std::process::exit(2);
    };

    let mut app = App::new();

    match &mode {
        // Both windowless render-to-image modes share the same no-window-but-GPU-on plugin
        // set; only their schedule-runner cadence differs (set below). The screenshot's 60 Hz
        // loop ties render frames to simulated time so the pipeline warms over the same frames
        // it captures; render-video instead advances the sim itself (one tick per frame, see
        // `RenderVideoPlugin`) and runs the loop as fast as compute allows.
        AppMode::Screenshot { .. } | AppMode::RenderVideo { .. } => {
            // No window, but GPU ON so we can render to an image.
            app.add_plugins(
                DefaultPlugins
                    // Resolve the committed control glyphs from the bundled `assets/` dir
                    // regardless of cwd/which bin runs; `BEVY_ASSET_ROOT` overrides (deploy).
                    .set(AssetPlugin {
                        file_path: crab_world::assets::bevy_asset_path()
                            .to_string_lossy()
                            .into_owned(),
                        ..default()
                    })
                    .set(bevy::window::WindowPlugin {
                        primary_window: None,
                        exit_condition: bevy::window::ExitCondition::DontExit,
                        ..default()
                    })
                    .disable::<bevy::winit::WinitPlugin>()
                    // `otel::init` already installed the global tracing subscriber; bevy's
                    // LogPlugin would try to install a second and panic. Disable it — the
                    // otel subscriber carries the same stderr `fmt` output.
                    .disable::<bevy::log::LogPlugin>(),
            );
            // Screenshot paces at 60 Hz (render frames track sim time); render-video runs the
            // loop flat-out (Duration::ZERO) since `step_one_tick` — not wall-clock — advances
            // the sim, so a slower-than-realtime render still produces a true-speed clip.
            let interval = if matches!(mode, AppMode::RenderVideo { .. }) {
                Duration::ZERO
            } else {
                Duration::from_secs_f64(1.0 / 60.0)
            };
            app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(interval));
        }
        AppMode::Demo => {
            // The demo defaults to borderless fullscreen (the Steam launch target is
            // a couch screen) unless --windowed.
            let fullscreen = !args.windowed;
            app.add_plugins(
                DefaultPlugins
                    // Resolve the committed control glyphs from the bundled `assets/` dir
                    // regardless of cwd/which bin runs; `BEVY_ASSET_ROOT` overrides (deploy).
                    .set(AssetPlugin {
                        file_path: crab_world::assets::bevy_asset_path()
                            .to_string_lossy()
                            .into_owned(),
                        ..default()
                    })
                    .set(bevy::window::WindowPlugin {
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
                    })
                    // See the screenshot arm: otel owns the subscriber, so drop bevy's
                    // LogPlugin to avoid a second-subscriber panic.
                    .disable::<bevy::log::LogPlugin>(),
            );
        }
    }

    // The render-mode cycle: the SHARED collider wireframe + mode-naming HUD (the same
    // `crab_view` GCR uses — ONE wireframe impl, not Rapier's debug-render). Boots in `colliders`
    // when the canonical Sally mesh is missing (the honest physics view IS the crab the player
    // sees), else from the `RL_RENDER_MODE`/`RL_DEBUG_COLLIDERS` env (mesh otherwise). The demo's
    // Right-arrow / D-pad cycles it live (see `play::demo::demo_controls`).
    let initial_render_mode = if mesh_ok {
        crab_world::crab_view::RenderMode::from_env()
    } else {
        crab_world::crab_view::RenderMode::Colliders
    };
    crab_world::crab_view::register(&mut app, initial_render_mode);
    // Joint-pivot markers: the companion diagnostic, drawn with the cage (the draw self-gates on
    // the render mode now — see `body::draw_pivot_markers`).
    bot::body::register_pivot_markers(&mut app);

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
        // Hand the preflighted mesh choice to the bot plugin BEFORE it builds `CrabAssets`/skin,
        // so the fallback decision is this explicit resource, not a poisoned env (bddap/rl#147).
        .insert_resource(mesh_state)
        .add_plugins(bot::BotPlugin)
        .add_systems(FixedUpdate, contact_audit);

    match mode {
        AppMode::Demo => {
            // The windowed surface the owner is actually looking at: if the canonical mesh
            // failed, name the failure ON SCREEN so the physics-bones fallback can never be
            // mistaken for real Sally (rl#706). Only the windowed demo gets the banner —
            // the screenshot/render-video arms render to image and a banner would pollute
            // the capture; their loudness stays the OTEL error + the collider view.
            if let Some(reason) = mesh_fallback_reason {
                app.insert_resource(MeshFallbackBanner(reason))
                    .add_systems(Startup, spawn_mesh_fallback_banner);
            }
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
        AppMode::RenderVideo { path, seconds } => {
            app.add_plugins(play::RenderVideoPlugin {
                checkpoint_dir: args.train.checkpoint_dir.clone(),
                path,
                seconds,
                width: args.width,
                height: args.height,
            });
        }
    }

    app.run();
}

/// The human-readable cause of a canonical-mesh load failure, carried to the on-screen banner.
/// Present (inserted) only when the mesh is unusable, so the banner system runs iff there is a
/// failure to announce — a healthy run has no banner resource and no banner.
#[derive(Resource)]
struct MeshFallbackBanner(String);

/// Spawn the can't-miss top-center banner for the windowed demo when the Sally mesh failed to
/// load. The render below it is the collider view — the REAL colliders, but NOT the real Sally
/// rig — so the banner says exactly that, killing the owner's "is this even Sally?" ambiguity
/// (rl#706) the OTEL log alone left unanswered on the screen he's watching. The banner UI is
/// `crab_world::mesh_fallback::spawn_banner`, shared with the game surface so neither drifts.
fn spawn_mesh_fallback_banner(mut commands: Commands, banner: Res<MeshFallbackBanner>) {
    crab_world::mesh_fallback::spawn_banner(&mut commands, &banner.0);
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
