use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use clap::builder::FalseyValueParser;
use clap::{Parser, Subcommand};
use crab_world::controls::{ControlsOverlayArgs, ControlsOverrides};
use crab_world::{CheckpointArgs, RenderArgs, bot, physics, play};

/// Watch the trained crab: a live demo window, a still, or a rendered video.
#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Live borderless-fullscreen demo window — the TV/deck surface. Drives the
    /// checkpoint's brain and hot-reloads its weights as they change on disk.
    Demo(DemoArgs),
    /// Offscreen still of the settled scene, written to PATH.
    Screenshot(ScreenshotArgs),
    /// Offscreen fixed-timestep video render, written to PATH.
    RenderVideo(RenderVideoArgs),
}

/// Flags every mode shares.
#[derive(clap::Args, Debug)]
struct Common {
    #[command(flatten)]
    checkpoint: CheckpointArgs,

    #[command(flatten)]
    render: RenderArgs,

    #[command(flatten)]
    otel: otel::OtelArgs,

    /// Waive the checkpoint's `arena terrain` record requirement (rl#293): view a
    /// pre-terrain/recordless checkpoint on the canonical tile anyway. Terrain is the
    /// only ground — this flag no longer picks one, it only unlocks archive viewing.
    #[arg(long)]
    terrain: bool,

    /// Seed for the round's RNG (ball placement, pokes); random when unset.
    #[arg(long, env = "RL_DEMO_SEED")]
    seed: Option<u64>,

    /// DIAGNOSTIC: when no usable checkpoint loads, drive an untrained random brain
    /// instead of the zero-action rest pose.
    #[arg(long, env = "RL_RANDOM_POLICY", value_parser = FalseyValueParser::new())]
    random_policy: bool,

    /// Pin the chase target ball here instead of sampling positions (a claw touch
    /// still re-rolls it).
    #[arg(long, env = "RL_TARGET_BALL_AT", value_name = "X,Y,Z", value_parser = parse_vec3,
          allow_hyphen_values = true)]
    target_ball_at: Option<Vec3>,

    /// DIAGNOSTIC: every 64th tick, log the deepest crab self-intersections and
    /// crab-vs-terrain penetrations (which body parts, by how much).
    #[arg(long, env = "RL_CONTACT_AUDIT", value_parser = FalseyValueParser::new())]
    contact_audit: bool,
}

/// Output dimensions for the offscreen surfaces.
#[derive(clap::Args, Debug)]
struct SizeArgs {
    /// Output width in pixels.
    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_WIDTH)]
    width: u32,

    /// Output height in pixels.
    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_HEIGHT)]
    height: u32,
}

#[derive(clap::Args, Debug)]
struct DemoArgs {
    #[command(flatten)]
    common: Common,

    /// Force-knobs for the controls legend overlay.
    #[command(flatten)]
    controls: ControlsOverlayArgs,

    /// Run in a window instead of borderless fullscreen.
    #[arg(long)]
    windowed: bool,

    /// Checkpoint dir to hot-reload weights from as they change — the live
    /// training view.
    #[arg(long, value_name = "PATH")]
    live_checkpoint_dir: Option<PathBuf>,

    /// Drive the crab from the gamepad instead of the policy.
    #[arg(long)]
    manual_control: bool,

    /// Show the joint-trace graph overlay from launch (also toggleable in-app).
    #[arg(long, env = "RL_GRAPH", value_parser = FalseyValueParser::new())]
    graph: bool,

    /// Capture the graph overlay to PATH once its traces fill (implies --graph).
    #[arg(long, env = "RL_GRAPH_SHOT", value_name = "PATH")]
    graph_shot: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
struct ScreenshotArgs {
    #[command(flatten)]
    common: Common,

    /// Force-knobs for the controls legend overlay.
    #[command(flatten)]
    controls: ControlsOverlayArgs,

    #[command(flatten)]
    size: SizeArgs,

    /// Where the still is written.
    #[arg(value_name = "PATH")]
    path: PathBuf,

    /// Ticks to let the scene settle before capturing.
    #[arg(long, default_value_t = 200)]
    settle: u32,

    /// Spawn the chase target ball (demo and render-video always have one).
    #[arg(long, env = "RL_TARGET_BALL", value_parser = FalseyValueParser::new())]
    target_ball: bool,

    /// Fix the camera at this world position instead of the crab-tracking close-up
    /// (vista framing, rl#281). Pairs with --shot-focus.
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, requires = "shot_focus",
          allow_hyphen_values = true)]
    shot_cam: Option<Vec3>,

    /// Where the fixed --shot-cam camera looks.
    #[arg(long, value_name = "X,Y,Z", value_parser = parse_vec3, requires = "shot_cam",
          allow_hyphen_values = true)]
    shot_focus: Option<Vec3>,

    /// Drive the selected joints at this action value (pose stills).
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
}

#[derive(clap::Args, Debug)]
struct RenderVideoArgs {
    #[command(flatten)]
    common: Common,

    #[command(flatten)]
    size: SizeArgs,

    /// Where the video is written.
    #[arg(value_name = "PATH")]
    path: PathBuf,

    /// Length of the render in simulated seconds.
    #[arg(long, default_value_t = 8.0)]
    seconds: f32,
}

fn parse_vec3(s: &str) -> Result<Vec3, String> {
    let parts: Vec<f32> = s
        .split(',')
        .map(|p| parse_finite_f32(p.trim()).map_err(|e| format!("{p:?}: {e}")))
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

fn main() {
    match Cli::parse().command {
        Command::Demo(args) => run_demo(args),
        Command::Screenshot(args) => run_screenshot(args),
        Command::RenderVideo(args) => run_render_video(args),
    }
}

/// Resolved against the demo's scheme before any world is built: an unknown context id
/// dies HERE, at t=0, rather than silently capturing the default legend (rl#275).
fn resolve_controls(args: &ControlsOverlayArgs) -> ControlsOverrides<play::DemoControls> {
    match args.resolve::<play::DemoControls>() {
        Ok(controls) => controls,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    }
}

/// How the app is driven: a live window pumped by winit, or offscreen frames pumped
/// by a ScheduleRunner at the given interval. An enum so a mode can't get it
/// half-right (window plus no runner, or both).
enum Driver {
    Windowed(Box<bevy::window::Window>),
    Offscreen(Duration),
}

/// Mode-independent boot: logging, otel, plant adoption, and the shared plugin stack.
/// The returned [`otel::OtelGuard`] must outlive `app.run()`.
fn boot(common: &Common, driver: Driver) -> (App, otel::OtelGuard) {
    if std::env::var_os("RUST_LOG").is_none() {
        unsafe { std::env::set_var("RUST_LOG", "warn,crab_world::play=info") };
    }

    let otel_guard = otel::init("rl-demo", common.otel);

    let mut app = App::new();
    match driver {
        Driver::Windowed(window) => {
            app.add_plugins(crab_world::app_boot::base_plugins(Some(*window)));
        }
        Driver::Offscreen(interval) => {
            app.add_plugins(crab_world::app_boot::base_plugins(None));
            app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(interval));
        }
    }

    let view = common
        .render
        .resolve(crab_world::mesh_fallback::Surface::RlDemo);
    // Cage gate open always: the demo has no menu phase — it renders the round for its
    // whole life, so there is no screen the gizmos could leak behind (the gate exists
    // for GCR, rl#211).
    crab_world::crab_view::register(&mut app, view.render_mode, || true);
    bot::body::register_pivot_markers(&mut app);

    // The demo simulates the plant the checkpoint TRAINED on (rl#281 stage 5): the
    // sidecar's friction cap is adopted, and its `arena terrain` record is the load
    // condition (rl#293 — flat plants refuse). `--terrain` waives that record
    // requirement for weight-less or archive viewing (rl#285): the friction leg is
    // still adopted, so the weights keep their trained joint damping. Adoption is
    // boot-time: a live-checkpoint sync that changes the sidecar mid-run takes
    // effect on the next launch (or is refused by the hot-reload plant guard).
    let adopted = if common.terrain {
        bot::body::adopt_recorded_plant_forcing_terrain(&common.checkpoint.checkpoint_dir)
    } else {
        bot::body::adopt_recorded_plant(&common.checkpoint.checkpoint_dir)
    };
    if let Err(err) = adopted {
        eprintln!("{err}");
        std::process::exit(2);
    }

    app.insert_resource(bot::NumEnvs(1))
        .add_plugins(crab_world::sky::NightSkyPlugin { moon: view.moon })
        .add_plugins(physics::CrabPhysicsPlugin)
        .add_plugins(physics::ArenaWorldPlugin {
            ground_look: view.ground_look,
        })
        .insert_resource(bot::body::CrabModelPath(
            crab_world::mesh_fallback::usable_model_path(),
        ))
        .add_plugins(bot::BotPlugin);

    if common.contact_audit {
        app.add_systems(FixedUpdate, bot::collider_check::live_contact_audit);
    }

    (app, otel_guard)
}

impl Common {
    fn play_overrides(&self) -> play::PlayOverrides {
        play::PlayOverrides {
            seed: self.seed,
            random_policy: self.random_policy,
            target_ball_at: self.target_ball_at,
        }
    }
}

fn run_demo(args: DemoArgs) {
    let controls = resolve_controls(&args.controls);
    let window = bevy::window::Window {
        title: "Crab RL".into(),
        mode: if args.windowed {
            bevy::window::WindowMode::Windowed
        } else {
            bevy::window::WindowMode::BorderlessFullscreen(bevy::window::MonitorSelection::Primary)
        },
        ..default()
    };
    let overrides = args.common.play_overrides();
    let (mut app, _otel) = boot(&args.common, Driver::Windowed(Box::new(window)));
    if let Err(reason) = crab_world::mesh_fallback::usable_model() {
        app.insert_resource(MeshFallbackBanner(reason.clone()));
        app.add_systems(Startup, spawn_mesh_fallback_banner);
    }
    app.add_plugins(play::DemoPlugin {
        checkpoint_dir: args.common.checkpoint.checkpoint_dir,
        live_checkpoint_dir: args.live_checkpoint_dir,
        manual_control: args.manual_control,
        overrides,
        graph: args.graph,
        graph_shot: args.graph_shot,
        controls,
    });
    app.run();
}

fn run_screenshot(args: ScreenshotArgs) {
    let controls = resolve_controls(&args.controls);
    let (mut app, _otel) = boot(
        &args.common,
        Driver::Offscreen(Duration::from_secs_f64(1.0 / 60.0)),
    );
    let overrides = args.common.play_overrides();
    app.add_plugins(play::ScreenshotPlugin {
        checkpoint_dir: args.common.checkpoint.checkpoint_dir,
        path: args.path,
        settle: args.settle,
        width: args.size.width,
        height: args.size.height,
        overrides,
        target_ball: args.target_ball || args.common.target_ball_at.is_some(),
        rig_pose: args.rig_pose.map(|a| (a, args.rig_pose_part)),
        shot_view: args.shot_cam.zip(args.shot_focus),
        controls,
    });
    app.run();
}

fn run_render_video(args: RenderVideoArgs) {
    let overrides = args.common.play_overrides();
    let (mut app, _otel) = boot(&args.common, Driver::Offscreen(Duration::ZERO));
    app.add_plugins(play::RenderVideoPlugin {
        checkpoint_dir: args.common.checkpoint.checkpoint_dir,
        path: args.path,
        seconds: args.seconds,
        width: args.size.width,
        height: args.size.height,
        overrides,
    });
    app.run();
}

#[derive(Resource)]
struct MeshFallbackBanner(String);

fn spawn_mesh_fallback_banner(mut commands: Commands, banner: Res<MeshFallbackBanner>) {
    crab_world::mesh_fallback::spawn_banner(&mut commands, &banner.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// See `game`'s twin: clap's own validity checks only run when the command is built.
    #[test]
    fn cli_is_well_formed() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    /// The deployed launch shapes must keep parsing (deck + TV launchers, the
    /// render/screenshot recipes).
    #[test]
    fn deployed_invocations_parse() {
        for argv in [
            vec!["rl-demo", "demo", "--checkpoint-dir", "c"],
            vec![
                "rl-demo",
                "demo",
                "--checkpoint-dir",
                "c",
                "--live-checkpoint-dir",
                "c",
            ],
            vec!["rl-demo", "screenshot", "out.png", "--settle", "10"],
            vec!["rl-demo", "render-video", "out.mp4", "--seconds", "3"],
        ] {
            Cli::try_parse_from(&argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        }
    }

    /// A mode-mismatched flag is a hard parse error now, not silently ignored (rl#335).
    #[test]
    fn cross_mode_flags_rejected() {
        assert!(Cli::try_parse_from(["rl-demo", "render-video", "o.mp4", "--windowed"]).is_err());
        assert!(Cli::try_parse_from(["rl-demo", "demo", "--seconds", "3"]).is_err());
    }

    #[test]
    fn vec3_rejects_non_finite() {
        assert!(parse_vec3("1,2,3").is_ok());
        assert!(parse_vec3("nan,0,0").is_err());
        assert!(parse_vec3("1,inf,0").is_err());
    }
}
