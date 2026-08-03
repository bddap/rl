use anyhow::Result;
use clap::Parser;
use net::render;
use net::sim::PlayerId;

use crab_world::RenderArgs;
use crab_world::controls::ControlsOverlayArgs;

use super::shared::{MATCH_SEED, boot_view, gcr_controls, nn_crab_policy};

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
    /// Raise the camera this many meters above the eye point (vista/altitude shots).
    #[arg(long, default_value_t = 0.0)]
    cam_height: f32,
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

    /// Boot with the developer debug overlay (rl#326) visible — offscreen has no F3.
    #[arg(long)]
    debug_overlay: bool,

    /// Scripted chord entry (rl#330 evidence): hold the kb chord modifier
    /// (right mouse) from this frame — the held-X context menu opens.
    #[arg(long)]
    chord_hold_at: Option<u64>,
    /// Release the scripted chord modifier at this frame, executing the typed code.
    /// Default: never — the menu stays open into the shot.
    #[arg(long)]
    chord_release_at: Option<u64>,
    /// Scripted code taps while held: `frame:dir`, comma-separated, dir ∈ U|D|L|R
    /// (e.g. `45:U,55:L`).
    #[arg(long, value_delimiter = ',')]
    chord_taps: Vec<String>,

    /// Capture this many frames as `<out-stem>.NNNN.png` instead of one shot —
    /// evidence clips for the animated ground looks (assemble with ffmpeg).
    #[arg(long)]
    anim_frames: Option<u32>,
    /// Render frames between captured anim frames (60 = 1 s of shader time).
    #[arg(long, default_value_t = 6)]
    anim_every: u32,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let me = PlayerId(0);
    let players: Vec<PlayerId> = (0..args.players.max(1)).map(PlayerId).collect();
    let client = net::client::ClientSim::new(MATCH_SEED, &players, me);
    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(args.cam_yaw, args.cam_pitch)
        .with_cam_height(args.cam_height)
        .with_fov(args.cam_fov)
        .with_anim(args.anim_frames.map(|count| (count, args.anim_every)));
    let nn_crab = args
        .nn_crab_checkpoint
        .map(|flag| nn_crab_policy(Some(flag)).map(|(_, policy)| policy))
        .transpose()?;
    let boot_view = boot_view(args.render);
    let controls = gcr_controls(&args.controls)?;
    let pack = net::sim::Input::new(0.0, 1.0, args.pack_look_yaw, 0);
    let mut app = render::build_screenshot_app(client, cfg, nn_crab, boot_view, controls, pack);
    if args.debug_overlay {
        app.insert_resource(crab_world::debug_overlay::DebugOverlay { visible: true });
    }
    if let Some(hold_at) = args.chord_hold_at {
        let taps = args
            .chord_taps
            .iter()
            .map(|spec| parse_chord_tap(spec))
            .collect::<Result<Vec<_>>>()?;
        app.insert_resource(render::ChordScript::new(
            hold_at,
            args.chord_release_at,
            taps,
        ));
    } else if !args.chord_taps.is_empty() || args.chord_release_at.is_some() {
        anyhow::bail!("--chord-taps/--chord-release-at need --chord-hold-at");
    }
    app.run();
    Ok(())
}

fn parse_chord_tap(spec: &str) -> Result<(u64, crab_world::chord::ChordDir)> {
    use crab_world::chord::ChordDir;
    let (frame, dir) = spec
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("chord tap {spec:?} is not frame:dir"))?;
    let dir = match dir {
        "U" => ChordDir::Up,
        "D" => ChordDir::Down,
        "L" => ChordDir::Left,
        "R" => ChordDir::Right,
        _ => anyhow::bail!("chord tap dir {dir:?} is not U|D|L|R"),
    };
    Ok((frame.parse()?, dir))
}
