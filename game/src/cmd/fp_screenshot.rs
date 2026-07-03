//! `fp-screenshot`: render one settled first-person frame to a PNG and exit (no window).

use anyhow::Result;
use clap::Parser;
use net::render;
use net::sim::PlayerId;

use super::shared::{MATCH_SEED, nn_crab_checkpoint_dir, resolve_render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    /// Output PNG path for the captured first-person frame.
    #[arg(long, default_value = "fp.png")]
    out: std::path::PathBuf,
    /// Frames to render before capturing — both how far into the round and GPU warmup
    /// (early frames render black). The sim advances one tick per frame at a fixed dt, so
    /// the scene is deterministic, not machine-speed-dependent. ~40 keeps the players in
    /// frame; higher lets the round play out.
    #[arg(long, default_value_t = 90)]
    settle: u32,
    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_WIDTH)]
    width: u32,
    #[arg(long, default_value_t = crab_world::screenshot::DEFAULT_HEIGHT)]
    height: u32,
    /// Number of players to spawn in the solo scene, so the shot shows another
    /// player's avatar alongside the local one (player 0 is the local camera).
    #[arg(long, default_value_t = 2)]
    players: u8,
    /// Pan the screenshot camera this many degrees from the dead-ahead first-person aim
    /// (0 = straight ahead), to frame the towering crab + extraction pillar + other players
    /// together when the crab would otherwise fill the straight-ahead view.
    #[arg(long, default_value_t = 0.0)]
    cam_yaw: f32,
    /// Tilt the screenshot camera this many degrees (+ up) from the first-person aim.
    #[arg(long, default_value_t = 0.0)]
    cam_pitch: f32,
    /// Override the camera's vertical field of view (degrees). Widen it (e.g. 90) to fit the
    /// whole towering giant crab in one frame — it stands only a body-length from the player, so
    /// the default FOV otherwise shows just a slab of carapace. Unset keeps Bevy's default.
    #[arg(long)]
    cam_fov: Option<f32>,
    /// Arm the real trained rapier-NN crab ("Sally") + skin for the shot, instead of the static
    /// integer silhouette — the SAME `--nn-crab-checkpoint` resolution as `play`, so an evidence
    /// frame composes the actual armed crab the windowed solo client renders (reposed + scaled to
    /// the giant). A dir with no `brain.bin` errors out (as `play` does); omit for the silhouette shot.
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<std::path::PathBuf>,

    /// Capture the crab render view in this mode (default: mesh). `mesh+colliders` overlays the
    /// honest collider wireframe on the mesh; `colliders` shows the wireframe alone. The evidence
    /// path for the render-mode cycle + the missing-glb fallback.
    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
}

/// Headless first-person screenshot: build a solo lockstep with `--players`
/// participants (so a remote avatar is in frame beside the local one), render one
/// settled frame of the FP view to a PNG, and exit. The evidence path for the
/// sim→render pipeline on a display-less box.
pub(crate) fn run(args: Args) -> Result<()> {
    let me = PlayerId(0);
    let players: Vec<PlayerId> = (0..args.players.max(1)).map(PlayerId).collect();
    let ls = net::lockstep::Lockstep::new(MATCH_SEED, &players, me);
    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(args.cam_yaw, args.cam_pitch)
        .with_fov(args.cam_fov);
    // Same checkpoint resolution as `play --nn-crab-checkpoint`: a readable brain.bin arms the real
    // crab for the shot (errors out if pointed at a dir with none). `None` keeps the silhouette.
    let external_crab = args
        .nn_crab_checkpoint
        .map(|flag| nn_crab_checkpoint_dir(Some(flag)))
        .transpose()?;
    let render_mode = resolve_render_mode(args.render_mode.as_deref())?;
    render::build_screenshot_app(ls, cfg, external_crab, render_mode).run();
    Ok(())
}
