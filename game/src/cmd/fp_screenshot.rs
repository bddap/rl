use anyhow::Result;
use clap::Parser;
use net::render;
use net::sim::PlayerId;

use super::shared::{MATCH_SEED, nn_crab_policy, resolve_render_mode};

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
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<std::path::PathBuf>,

    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let me = PlayerId(0);
    let players: Vec<PlayerId> = (0..args.players.max(1)).map(PlayerId).collect();
    let ls = net::lockstep::Lockstep::new(MATCH_SEED, &players, me);
    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(args.cam_yaw, args.cam_pitch)
        .with_fov(args.cam_fov);
    let external_crab = args
        .nn_crab_checkpoint
        .map(|flag| nn_crab_policy(Some(flag)).map(|(_, policy)| policy))
        .transpose()?;
    let render_mode = resolve_render_mode(args.render_mode.as_deref())?;
    render::build_screenshot_app(ls, cfg, external_crab, render_mode).run();
    Ok(())
}
