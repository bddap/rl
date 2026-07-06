
use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use clap::Parser;
use crab_world::{CheckpointArgs, Visuals, bot, physics, play};

#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Args {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[arg(long)]
    demo: bool,

    #[arg(long)]
    windowed: bool,

    #[arg(long, value_name = "PATH")]
    screenshot: Option<PathBuf>,

    #[arg(long, default_value_t = 200)]
    screenshot_settle: u32,

    #[arg(long, value_name = "PATH")]
    render_video: Option<PathBuf>,

    #[arg(long, default_value_t = 8.0)]
    seconds: f32,

    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_WIDTH)]
    width: u32,

    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_HEIGHT)]
    height: u32,

    #[arg(long, value_name = "PATH")]
    live_checkpoint_dir: Option<PathBuf>,

    #[arg(long)]
    manual_control: bool,
}

#[derive(Clone)]
enum AppMode {
    Demo,
    Screenshot {
        path: PathBuf,
        settle: u32,
    },
    RenderVideo {
        path: PathBuf,
        seconds: f32,
    },
}

fn main() {
    let args = Args::parse();


    if std::env::var_os("RUST_LOG").is_none() {
        unsafe { std::env::set_var("RUST_LOG", "warn,crab_world::play=info") };
    }

    let mesh_state =
        crab_world::bot::body::CrabModelPath(crab_world::mesh_fallback::usable_model_path());

    let _otel = otel::init("rl-demo");

    let mesh_fallback_reason: Option<String> = crab_world::mesh_fallback::usable_model()
        .as_ref()
        .err()
        .cloned();

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
        eprintln!("no mode selected. rl-demo needs --demo, --screenshot, or --render-video.");
        std::process::exit(2);
    };

    let mut app = App::new();

    match &mode {
        AppMode::Screenshot { .. } | AppMode::RenderVideo { .. } => {
            app.add_plugins(crab_world::app_boot::base_plugins(None));
            let interval = if matches!(mode, AppMode::RenderVideo { .. }) {
                Duration::ZERO
            } else {
                Duration::from_secs_f64(1.0 / 60.0)
            };
            app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(interval));
        }
        AppMode::Demo => {
            let fullscreen = !args.windowed;
            app.add_plugins(crab_world::app_boot::base_plugins(Some(
                bevy::window::Window {
                    title: "Crab RL".into(),
                    mode: if fullscreen {
                        bevy::window::WindowMode::BorderlessFullscreen(
                            bevy::window::MonitorSelection::Primary,
                        )
                    } else {
                        bevy::window::WindowMode::Windowed
                    },
                    ..default()
                },
            )));
        }
    }

    let initial_render_mode = crab_world::mesh_fallback::initial_render_mode(
        None,
        crab_world::mesh_fallback::Surface::RlDemo,
    );
    // Cage gate open always: the demo has no menu phase — it renders the round for its whole
    // life, so there is no screen the gizmos could leak behind (the gate exists for GCR, rl#211).
    crab_world::crab_view::register(&mut app, initial_render_mode, || true);
    bot::body::register_pivot_markers(&mut app);

    app.insert_resource(Visuals(true))
        .insert_resource(bot::NumEnvs(1))
        .add_plugins(physics::CrabPhysicsPlugin)
        // The demo mirrors the TRAINING world (targets sampled inside the box), so it keeps
        // the walled arena; only GCR inference runs the open field (rl#209).
        .add_plugins(physics::PhysicsWorldPlugin {
            arena: physics::Arena::WalledBox,
        })
        .add_plugins(physics::ArenaVisualsPlugin)
        .insert_resource(mesh_state)
        .add_plugins(bot::BotPlugin)
        .add_systems(FixedUpdate, contact_audit);

    match mode {
        AppMode::Demo => {
            if let Some(reason) = mesh_fallback_reason {
                app.insert_resource(MeshFallbackBanner(reason))
                    .add_systems(Startup, spawn_mesh_fallback_banner);
            }
            app.add_plugins(play::DemoPlugin {
                checkpoint_dir: args.checkpoint.checkpoint_dir.clone(),
                live_checkpoint_dir: args.live_checkpoint_dir.clone(),
                manual_control: args.manual_control,
            });
        }
        AppMode::Screenshot { path, settle } => {
            app.add_plugins(play::ScreenshotPlugin {
                checkpoint_dir: args.checkpoint.checkpoint_dir.clone(),
                path,
                settle,
                width: args.width,
                height: args.height,
            });
        }
        AppMode::RenderVideo { path, seconds } => {
            app.add_plugins(play::RenderVideoPlugin {
                checkpoint_dir: args.checkpoint.checkpoint_dir.clone(),
                path,
                seconds,
                width: args.width,
                height: args.height,
            });
        }
    }

    app.run();
}

#[derive(Resource)]
struct MeshFallbackBanner(String);

fn spawn_mesh_fallback_banner(mut commands: Commands, banner: Res<MeshFallbackBanner>) {
    crab_world::mesh_fallback::spawn_banner(&mut commands, &banner.0);
}

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
