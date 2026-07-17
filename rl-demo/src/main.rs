use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use clap::Parser;
use crab_world::controls::ControlsOverlayArgs;
use crab_world::{CheckpointArgs, RenderArgs, Visuals, bot, physics, play};

/// Watch the trained crab: a live demo window, a still, or a rendered video.
#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Args {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[command(flatten)]
    pub render: RenderArgs,

    /// Force-knobs for the controls legend — they act on the surfaces that HAVE one
    /// (`--demo`, `--screenshot`); `--render-video` shows no overlay.
    #[command(flatten)]
    pub controls: ControlsOverlayArgs,

    #[command(flatten)]
    pub otel: otel::OtelArgs,

    #[arg(long)]
    demo: bool,

    /// Run on the baked GCR terrain tile (rl#281) instead of the training walled box —
    /// the stage-3 taste-loop surface: real mountains, spawn-on-surface, no walls.
    #[arg(long)]
    terrain: bool,

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

    /// Seed for the demo RNG (ball placement, pokes); random when unset.
    #[arg(long, env = "RL_DEMO_SEED")]
    seed: Option<u64>,

    /// DIAGNOSTIC: when no usable checkpoint loads, drive an untrained random brain
    /// instead of the zero-action rest pose.
    #[arg(long, env = "RL_RANDOM_POLICY",
          value_parser = clap::builder::FalseyValueParser::new())]
    random_policy: bool,

    /// Spawn the chase target ball in --screenshot mode (the demo and --render-video
    /// always have one).
    #[arg(long, env = "RL_TARGET_BALL",
          value_parser = clap::builder::FalseyValueParser::new())]
    target_ball: bool,

    /// Pin the target ball here instead of sampling positions (a claw touch still
    /// re-rolls it).
    #[arg(long, env = "RL_TARGET_BALL_AT", value_name = "X,Y,Z", value_parser = parse_vec3)]
    target_ball_at: Option<Vec3>,

    /// Fix the --screenshot camera at this world position instead of the crab-tracking
    /// close-up (vista framing, rl#281). Pairs with --shot-focus.
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, requires = "shot_focus",
          allow_hyphen_values = true)]
    shot_cam: Option<Vec3>,

    /// Where the fixed --shot-cam camera looks.
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, requires = "shot_cam",
          allow_hyphen_values = true)]
    shot_focus: Option<Vec3>,

    /// Drive the selected joints at this action value in --screenshot mode (pose stills).
    #[arg(long, env = "RL_RIG_POSE", allow_negative_numbers = true,
          value_parser = parse_finite_f32)]
    rig_pose: Option<f32>,

    /// Which joints --rig-pose drives.
    #[arg(
        long,
        env = "RL_RIG_POSE_PART",
        value_enum,
        default_value_t,
        requires = "rig_pose"
    )]
    rig_pose_part: play::RigPosePart,

    /// Show the joint-trace graph overlay from launch (demo mode; also toggleable in-app).
    #[arg(long, env = "RL_GRAPH", value_parser = clap::builder::FalseyValueParser::new())]
    graph: bool,

    /// Capture the graph overlay to PATH once its traces fill (demo mode; implies --graph).
    #[arg(long, env = "RL_GRAPH_SHOT", value_name = "PATH")]
    graph_shot: Option<PathBuf>,

    /// DIAGNOSTIC: every 64th tick, log the deepest crab self-intersections (which body
    /// parts, by how much).
    #[arg(long, env = "RL_CONTACT_AUDIT", value_parser = clap::builder::FalseyValueParser::new())]
    contact_audit: bool,
}

fn parse_vec3(s: &str) -> Result<Vec3, String> {
    let parts: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>().map_err(|e| format!("{p:?}: {e}")))
        .collect::<Result<_, _>>()?;
    match parts.as_slice() {
        [x, y, z] => Ok(Vec3::new(*x, *y, *z)),
        _ => Err(format!("expected X,Y,Z (got {} components)", parts.len())),
    }
}

fn parse_finite_f32(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("{v} is not finite"))
    }
}

#[derive(Clone)]
enum AppMode {
    Demo,
    Screenshot { path: PathBuf, settle: u32 },
    RenderVideo { path: PathBuf, seconds: f32 },
}

fn main() {
    let args = Args::parse();

    // Resolved against the demo's scheme before any world is built: an unknown context id dies
    // HERE, at t=0, rather than silently capturing the default legend (rl#275).
    let controls = match args.controls.resolve::<play::DemoControls>() {
        Ok(controls) => controls,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    if std::env::var_os("RUST_LOG").is_none() {
        unsafe { std::env::set_var("RUST_LOG", "warn,crab_world::play=info") };
    }

    let mesh_state =
        crab_world::bot::body::CrabModelPath(crab_world::mesh_fallback::usable_model_path());

    let _otel = otel::init("rl-demo", args.otel);

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

    let initial_render_mode = args
        .render
        .resolve(crab_world::mesh_fallback::Surface::RlDemo);
    // Cage gate open always: the demo has no menu phase — it renders the round for its whole
    // life, so there is no screen the gizmos could leak behind (the gate exists for GCR, rl#211).
    crab_world::crab_view::register(&mut app, initial_render_mode, || true);
    bot::body::register_pivot_markers(&mut app);

    // The demo simulates the plant the checkpoint TRAINED on (rl#281 stage 5): the
    // sidecar drives the arena (and friction cap), so a terrain-trained brain shipped to
    // the TV/decks gets its mountains with no launcher flag to drift out of sync with
    // the weights. `--terrain` stays as a force for weight-less viewing (the stage-3
    // taste-loop recipe) and skips adoption — it is a VIEWING override, not a plant.
    // Adoption is boot-time: a live-checkpoint sync that changes the sidecar mid-run
    // takes effect on the next launch.
    let arena = if args.terrain {
        physics::Arena::Terrain
    } else {
        if let Err(err) = bot::body::adopt_recorded_plant(&args.checkpoint.checkpoint_dir) {
            eprintln!("{err}");
            std::process::exit(2);
        }
        physics::train_arena().arena()
    };

    app.insert_resource(Visuals(true))
        .insert_resource(bot::NumEnvs(1))
        .add_plugins(physics::CrabPhysicsPlugin)
        .add_plugins(physics::PhysicsWorldPlugin { arena })
        .add_plugins(physics::ArenaVisualsPlugin)
        .insert_resource(mesh_state)
        .add_plugins(bot::BotPlugin);

    if args.contact_audit {
        app.add_systems(FixedUpdate, contact_audit);
    }

    let overrides = play::PlayOverrides {
        seed: args.seed,
        random_policy: args.random_policy,
        target_ball_at: args.target_ball_at,
    };

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
                overrides,
                graph: args.graph,
                graph_shot: args.graph_shot.clone(),
                controls,
            });
        }
        AppMode::Screenshot { path, settle } => {
            app.add_plugins(play::ScreenshotPlugin {
                checkpoint_dir: args.checkpoint.checkpoint_dir.clone(),
                path,
                settle,
                width: args.width,
                height: args.height,
                overrides,
                target_ball: args.target_ball || args.target_ball_at.is_some(),
                rig_pose: args.rig_pose.map(|a| (a, args.rig_pose_part)),
                shot_view: args.shot_cam.zip(args.shot_focus),
                controls,
            });
        }
        AppMode::RenderVideo { path, seconds } => {
            app.add_plugins(play::RenderVideoPlugin {
                checkpoint_dir: args.checkpoint.checkpoint_dir.clone(),
                path,
                seconds,
                width: args.width,
                height: args.height,
                overrides,
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

/// Added only under `--contact-audit`: the gate is the system's PRESENCE, so an off run pays
/// nothing.
fn contact_audit(
    sim: Query<&bevy_rapier3d::plugin::context::RapierContextSimulation>,
    cols: Query<&bevy_rapier3d::plugin::context::RapierContextColliders>,
    parts: Query<
        (Option<&bot::body::CrabJoint>, Has<bot::body::CrabCarapace>),
        With<bot::body::CrabBodyPart>,
    >,
    mut tick: Local<u32>,
) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// See `game`'s twin: clap's own validity checks only run when the command is built.
    #[test]
    fn cli_is_well_formed() {
        use clap::CommandFactory;
        Args::command().debug_assert();
    }
}
