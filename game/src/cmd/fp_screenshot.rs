use anyhow::Result;
use clap::Parser;
use net::render;
use net::sim::PlayerId;

use crab_world::RenderArgs;
use crab_world::controls::ControlsOverlayArgs;

use super::shared::{MATCH_SEED, gcr_controls, nn_crab_policy, render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, default_value = "fp.png")]
    out: std::path::PathBuf,
    #[arg(long, default_value_t = 90)]
    settle: u32,
    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_WIDTH)]
    width: u32,
    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_HEIGHT)]
    height: u32,
    #[arg(long, default_value_t = 2)]
    players: u8,
    #[arg(long, default_value_t = 0.0)]
    cam_yaw: f32,
    #[arg(long, default_value_t = 0.0)]
    cam_pitch: f32,
    #[arg(long)]
    cam_fov: Option<f32>,
    // No `env` here, unlike the other surfaces: this flag is the OPT-IN to arm a crab at all
    // (`.map(..)` below), so an exported RL_CRAB_CHECKPOINT_DIR would silently seed a crab into
    // an evidence shot that is meant to have none.
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<std::path::PathBuf>,

    #[command(flatten)]
    render: RenderArgs,

    #[command(flatten)]
    controls: ControlsOverlayArgs,

    /// Yaw axis held by the scripted pack (−1..1): non-zero makes the pack orbit instead
    /// of bee-lining, presenting the crab flank/bystander geometry (the claw-down regime,
    /// rl#249) rather than only frontal pursuit.
    #[arg(long, default_value_t = 0.0)]
    pack_look_yaw: f32,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let me = PlayerId(0);
    let players: Vec<PlayerId> = (0..args.players.max(1)).map(PlayerId).collect();
    let client = net::client::ClientSim::new(MATCH_SEED, &players, me);
    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(args.cam_yaw, args.cam_pitch)
        .with_fov(args.cam_fov);
    let nn_crab = args
        .nn_crab_checkpoint
        .map(|flag| nn_crab_policy(Some(flag)).map(|(_, policy)| policy))
        .transpose()?;
    let render_mode = render_mode(args.render);
    let controls = gcr_controls(&args.controls)?;
    let pack = net::sim::Input::new(0.0, 1.0, args.pack_look_yaw, 0);
    render::build_screenshot_app(client, cfg, nn_crab, render_mode, controls, pack).run();
    Ok(())
}
