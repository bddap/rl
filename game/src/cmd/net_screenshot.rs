//! `net-screenshot`: form a real networked round over iroh and render one settled frame of a
//! peer's first-person view to a PNG — the OFFSCREEN, headless analogue of `play --host`/`--join`
//! (rl#151 increment 2 windowed). Run two of these: they discover over the LAN, the lowest-id peer
//! hosts (steps + broadcasts the authoritative snapshot + articulated crab), the other is a REMOTE
//! CLIENT (adopts the snapshot, renders the host's crab pose from the articulation, no re-sim). The
//! client's captured frame is the evidence that a windowed remote client renders the host's state.

use anyhow::Result;
use clap::Parser;
use iroh::EndpointId;
use net::{net_loop, render};

use super::shared::{MATCH_SEED, nn_crab_checkpoint_dir, resolve_render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    /// HOST the round directly (no menu): form over the LAN and start with whoever joins.
    #[arg(long, conflicts_with = "join")]
    host: bool,
    /// JOIN a host: dial its endpoint-id code, or (bare `--join`) discover over the LAN.
    #[arg(long, value_name = "JOIN_CODE", num_args = 0..=1, default_missing_value = "", conflicts_with = "host")]
    join: Option<String>,
    /// Wait this long for peers before starting.
    #[arg(long, default_value_t = 5)]
    discover_secs: u64,
    /// Expected peer count including us; proceeds with whoever showed up after `discover_secs`.
    #[arg(long, default_value_t = 2)]
    expect: usize,
    /// Output PNG for this peer's captured frame.
    #[arg(long, default_value = "net.png")]
    out: std::path::PathBuf,
    /// Frames (≈ ticks) to run before capturing — far enough into the round that the client has
    /// adopted the host's crab pose and the crab is in frame. Also GPU warmup.
    #[arg(long, default_value_t = 140)]
    settle: u32,
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    /// Vertical FOV (degrees) — widen to fit the towering giant crab in one frame.
    #[arg(long, default_value_t = 95.0)]
    cam_fov: f32,
    /// Tilt the camera up this many degrees from the dead-ahead FP aim.
    #[arg(long, default_value_t = 8.0)]
    cam_pitch: f32,
    /// The REQUIRED trained crab checkpoint (rl#114) — same resolution as `play`.
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<std::path::PathBuf>,
    /// Crab render view (default mesh).
    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
}

/// Form the networked round, then render one settled offscreen frame of THIS peer's view. The
/// peer's role (host / remote client) is decided by the formation (lowest endpoint id hosts); we
/// print it so the two captured PNGs can be told apart — the CLIENT's is the increment-2 evidence.
pub(crate) fn run(args: Args) -> Result<()> {
    // The one giant crab IS the trained NN body (rl#114) — resolve the checkpoint before forming so
    // we advertise our real weights digest; two peers arm Sally only when their brains agree.
    let external_crab = nn_crab_checkpoint_dir(args.nn_crab_checkpoint)?;
    let weights_digest = crab_world::play::checkpoint_digest(&external_crab);
    let render_mode = resolve_render_mode(args.render_mode.as_deref())?;

    let dial = match &args.join {
        Some(code) if !code.trim().is_empty() => Some(code.trim().parse::<EndpointId>()?),
        _ => None,
    };
    let result = net_loop::connect_and_form_dialing(
        MATCH_SEED,
        args.discover_secs,
        args.expect,
        dial,
        None,
        None,
        weights_digest,
        crab_world::bot::meshfit::crab_asset_digest(),
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
    println!("net-screenshot: formed as {role}; rendering to {}", args.out.display());

    // A networked round arms the deterministic NN crab, so pin the process task pools single-thread
    // BEFORE the app latches them (the same pin the windowed networked client uses). Harmless here
    // regardless of role; the client never steps the crab but shares the build path.
    render::pin_process_pools();
    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(0.0, args.cam_pitch)
        .with_fov(Some(args.cam_fov));
    let mut app = render::build_net_screenshot_app(ls, driver, cfg, external_crab, render_mode);
    app.run();
    Ok(())
}
