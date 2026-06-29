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
    let mesh_status = canonical_mesh_status();
    if mesh_status.is_err() {
        // Make every downstream consumer agree there is no usable model: the body recipe
        // (`crab_world::bot::body::render_recipe`) then takes the procedural-collider fallback
        // instead of panicking on a present-but-broken file, and the skin
        // (`bot::skin::register`) self-skips instead of half-loading a broken scene. Both read
        // `meshfit::model_path()`, so redirecting it at a guaranteed-absent path is the single
        // switch that flips them together. The physics bones are then made VISIBLE by forcing
        // the debug-render on (see `debug_colliders` below).
        unsafe {
            std::env::set_var(
                "CRAB_MODEL_PATH",
                "/nonexistent/rl-demo-canonical-mesh-unavailable.glb",
            );
        }
    }

    // OTEL/tracing: install the shared subscriber AFTER the env is final. The guard must
    // outlive the whole run, so it's bound here and dropped only when `main` returns.
    let _otel = otel::init("rl-demo");

    let mesh_ok = match mesh_status {
        Ok(()) => true,
        Err(reason) => {
            // LOUD via telemetry (stderr + OTLP), GRACEFUL in render. `target` namespaces the
            // record so the sink can filter for canonical-mesh failures.
            tracing::error!(
                target: "rl_demo::canonical_mesh",
                reason = %reason,
                "rl-demo: canonical Sally mesh could not be loaded — falling back to the \
                 physics-bones debug view (the real colliders). Fetch it with \
                 scripts/fetch-sally.sh or point CRAB_MODEL_PATH at the model."
            );
            false
        }
    };
    // Show the physics-bones view whenever the mesh is unusable, OR on explicit request. When
    // the mesh is fine the cage defaults off (the skinned Sally crab is the view); the demo's
    // right-arrow still toggles it live.
    let debug_colliders = !mesh_ok || std::env::var_os("RL_DEBUG_COLLIDERS").is_some();

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
            app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
                Duration::from_secs_f64(1.0 / 60.0),
            ));
            // Rapier collider wireframes draw via gizmos, which DO render into the
            // offscreen screenshot camera (Bevy 0.18) — but only if the plugin is
            // present. The other arms add it; Screenshot has its own arm, so gate it
            // here on the same flag or the captured PNG never shows the colliders.
            if debug_colliders {
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
            // With the stand-in primitive meshes removed, Rapier's debug-render is
            // the only in-engine view of the true colliders, so the skin can be
            // checked against the actual physics shapes. `enabled` only sets the
            // INITIAL state — the demo's right-arrow toggles `DebugRenderContext`
            // live — and the demo starts the cage on iff `debug_colliders` (RL_DEBUG_COLLIDERS
            // OR a missing canonical mesh, where the cage IS the crab the player sees).
            app.add_plugins(RapierDebugRenderPlugin {
                enabled: debug_colliders,
                // Collider shapes only — the default also draws per-body axes +
                // joint markers, which on a 31-part body is an unreadable tangle.
                mode: DebugRenderMode::COLLIDER_SHAPES,
                ..default()
            });
            // Pivot markers: a deliberate diagnostic, on with the cage.
            if debug_colliders {
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

/// Is the canonical Sally mesh present AND usable (loads + has the crab bones the rig needs)?
/// `Ok(())` means render the real skinned crab; `Err(reason)` carries a human-readable cause
/// for the OTEL error and the physics-bones fallback. This mirrors `crab_world::bot::body`'s
/// model-vs-fallback selection (the same `model_path` / `LoadedModel::load` / `build_recipe`
/// chain), so the demo's "is the mesh good?" verdict can't disagree with what the body would
/// actually spawn.
fn canonical_mesh_status() -> Result<(), String> {
    let Some(p) = bot::meshfit::model_path() else {
        return Err(
            "no crab model resolved (CRAB_MODEL_PATH / default `sally.glb` not found under \
             BEVY_ASSET_ROOT/assets)"
            .to_string(),
        );
    };
    let model = bot::meshfit::LoadedModel::load(&p).map_err(|e| format!("crab model {p:?}: {e}"))?;
    if bot::rig::build_recipe(&model).is_none() {
        return Err(format!(
            "crab model {p:?}: loaded but has none of the expected crab bones (e.g. Def_leg_01.000.L)"
        ));
    }
    Ok(())
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
