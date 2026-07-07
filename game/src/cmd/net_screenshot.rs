
use anyhow::Result;
use clap::Parser;
use net::{net_loop, render};

use super::shared::{MATCH_SEED, nn_crab_checkpoint_dir, parse_join_dial, resolve_render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, conflicts_with = "join")]
    host: bool,
    #[arg(long, value_name = "JOIN_CODE", num_args = 0..=1, default_missing_value = "", conflicts_with = "host")]
    join: Option<String>,
    #[arg(long, default_value_t = 5)]
    discover_secs: u64,
    #[arg(long, default_value_t = super::shared::DEFAULT_EXPECT)]
    expect: usize,
    #[arg(long, default_value = "net.png")]
    out: std::path::PathBuf,
    #[arg(long, default_value_t = 140)]
    settle: u32,
    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_WIDTH)]
    width: u32,
    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_HEIGHT)]
    height: u32,
    #[arg(long, default_value_t = 95.0)]
    cam_fov: f32,
    #[arg(long, default_value_t = 8.0)]
    cam_pitch: f32,
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<std::path::PathBuf>,
    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
    /// Press the vehicle E-cycle at this frame (repeatable) and hold a forward drive while
    /// piloting — a scripted pilot, so a two-peer run live-verifies board/fly/cycle/exit over
    /// the real wire (rl#191).
    #[arg(long, value_name = "FRAME")]
    pilot_toggle_at: Vec<u64>,
    /// From this frame on, the scripted pilot walks a gentle arc on foot (a moving target for
    /// the hunting crab).
    #[arg(long, value_name = "FRAME")]
    pilot_walk_at: Option<u64>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let external_crab = nn_crab_checkpoint_dir(args.nn_crab_checkpoint)?;
    let render_mode = resolve_render_mode(args.render_mode.as_deref())?;

    let dial = parse_join_dial(args.join.as_deref())?;
    let result = net_loop::connect_and_form_dialing(
        MATCH_SEED,
        args.discover_secs,
        args.expect,
        dial,
        None,
        None,
        crab_world::mesh_fallback::constructed_body_digest(),
        1,
    )?;

    let (ls, driver) = match result {
        net_loop::MatchResult::Joined(joined) => *joined,
        net_loop::MatchResult::Alone => {
            anyhow::bail!(
                "net-screenshot formed no peer within {}s — run a second peer (this mode needs a \
                 host + a client to show the remote-client render)",
                args.discover_secs
            )
        }
        net_loop::MatchResult::Cancelled => unreachable!("no interactive lobby on this path"),
    };
    let role = if driver.is_host() { "host" } else { "client" };
    println!(
        "net-screenshot: formed as {role}; rendering to {}",
        args.out.display()
    );

    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(0.0, args.cam_pitch)
        .with_fov(Some(args.cam_fov));
    let mut app = render::build_net_screenshot_app(ls, driver, cfg, external_crab, render_mode);
    if !args.pilot_toggle_at.is_empty() || args.pilot_walk_at.is_some() {
        app.insert_resource(render::PilotScript::new(
            args.pilot_toggle_at,
            args.pilot_walk_at,
        ));
    }
    app.run();
    Ok(())
}
