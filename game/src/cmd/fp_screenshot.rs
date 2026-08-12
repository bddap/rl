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
    /// Extra modifier hold windows beyond --chord-hold-at/--chord-release-at:
    /// `from:to` pairs, `to` empty for never-release (e.g. `120:180,220:`) — lets one
    /// clip enter several codes (rl#358 map-growth evidence).
    #[arg(long, value_delimiter = ',')]
    chord_holds: Vec<String>,
    /// Tap the chord-map layout-cycle key (M) at these frames (rl#358 live-switch
    /// evidence).
    #[arg(long, value_delimiter = ',')]
    map_cycle_at: Vec<u64>,
    /// Pin the chord map's starting layout: hyperbolic | zoomquad | hyperatlas.
    #[arg(long)]
    chord_map_layout: Option<String>,
    /// Load/persist the shot's discovered-code set here (default: unpersisted,
    /// empty — an evidence shot must never touch the real save).
    #[arg(long)]
    chord_map_file: Option<std::path::PathBuf>,

    /// Capture this many frames as `<out-stem>.NNNN.png` instead of one shot —
    /// evidence clips for the animated ground looks (assemble with ffmpeg).
    #[arg(long)]
    anim_frames: Option<u32>,
    /// Render frames between captured anim frames (60 = 1 s of shader time).
    #[arg(long, default_value_t = 6)]
    anim_every: u32,

    /// Match seed — picks the run's spawn locale on the tile (rl#305), so evidence
    /// shots can target a specific coordinate regime (e.g. the far-corner f32
    /// precision band, rl#334).
    #[arg(long, default_value_t = MATCH_SEED)]
    seed: u64,
    /// From this frame on, the local player walks a gentle arc — motion evidence
    /// (rl#334: texture stability is only visible while the eye translates).
    #[arg(long, value_name = "FRAME", value_parser = clap::value_parser!(u64).range(1..))]
    walk_at: Option<u64>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let me = PlayerId(0);
    let players: Vec<PlayerId> = (0..args.players.max(1)).map(PlayerId).collect();
    let client = net::client::ClientSim::new(args.seed, &players, me);
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
    if let Some(walk_at) = args.walk_at {
        app.insert_resource(render::PilotScript::new(Vec::new(), Some(walk_at)));
    }
    if let Some(hold_at) = args.chord_hold_at {
        let taps = args
            .chord_taps
            .iter()
            .map(|spec| parse_chord_tap(spec))
            .collect::<Result<Vec<_>>>()?;
        // The script's frame counter starts at 1 and the modifier must be down before
        // a tap can buffer — a spec that can't fire must fail loud, not shoot a blank
        // evidence frame.
        if hold_at == 0 || taps.iter().any(|&(at, _)| at <= hold_at) {
            anyhow::bail!("chord frames must be ≥1 and taps after --chord-hold-at");
        }
        if args.chord_release_at.is_some_and(|r| r <= hold_at) {
            anyhow::bail!("--chord-release-at must be after --chord-hold-at");
        }
        let mut seen = std::collections::HashSet::new();
        if let Some(dup) = taps.iter().find(|&&t| !seen.insert(t)) {
            anyhow::bail!(
                "duplicate chord tap {}:{:?} (one edge, one tap)",
                dup.0,
                dup.1
            );
        }
        let holds = args
            .chord_holds
            .iter()
            .map(|spec| parse_hold(spec))
            .collect::<Result<Vec<_>>>()?;
        // The script holds the modifier on ANY covering window, so windows that
        // touch or overlap silently merge into one capture and the release edge —
        // the thing that executes the code — never fires between them. Demand a
        // ≥1-frame gap, in order, and a released primary window before extras.
        if !holds.is_empty() && args.chord_release_at.is_none() {
            anyhow::bail!("--chord-holds needs --chord-release-at to close the first window");
        }
        let mut windows = vec![(hold_at, args.chord_release_at)];
        windows.extend(holds.iter().copied());
        for pair in windows.windows(2) {
            let (prev, next) = (pair[0], pair[1]);
            match prev.1 {
                Some(end) if next.0 > end => {}
                _ => anyhow::bail!(
                    "chord hold windows must be in order with a ≥1-frame gap \
                     (got {:?} then {:?}) — touching windows merge into one capture",
                    prev,
                    next
                ),
            }
        }
        app.insert_resource(
            render::ChordScript::new(hold_at, args.chord_release_at, taps)
                .with_holds(holds)
                .with_layout_cycles(args.map_cycle_at.clone()),
        );
    } else if !args.chord_taps.is_empty()
        || args.chord_release_at.is_some()
        || !args.chord_holds.is_empty()
        || !args.map_cycle_at.is_empty()
    {
        anyhow::bail!(
            "--chord-taps/--chord-release-at/--chord-holds/--map-cycle-at need --chord-hold-at"
        );
    }
    if args.chord_map_layout.is_some() || args.chord_map_file.is_some() {
        let layout = args
            .chord_map_layout
            .as_deref()
            .map(render::chord_map::MapLayout::from_id)
            .transpose()
            .map_err(anyhow::Error::msg)?
            .unwrap_or_default();
        app.insert_resource(render::chord_map::ChordMapState::new(layout));
        if let Some(file) = args.chord_map_file {
            app.insert_resource(render::chord_map::DiscoveredCodes::load(Some(file)));
        }
    }
    app.run();
    Ok(())
}

fn parse_hold(spec: &str) -> Result<(u64, Option<u64>)> {
    let (from, to) = spec
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("chord hold {spec:?} is not from:to"))?;
    let from: u64 = from.parse()?;
    let to = if to.is_empty() {
        None
    } else {
        let to: u64 = to.parse()?;
        if to <= from {
            anyhow::bail!("chord hold {spec:?} releases before it holds");
        }
        Some(to)
    };
    Ok((from, to))
}

fn parse_chord_tap(spec: &str) -> Result<(u64, crab_world::chord::ChordDir)> {
    let (frame, dir) = spec
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("chord tap {spec:?} is not frame:dir"))?;
    // One UDLR alphabet — the chord map's codec, not a second letter match.
    let dir = match render::chord_map::code_from_str(dir).as_deref() {
        Some([d]) => *d,
        _ => anyhow::bail!("chord tap dir {dir:?} is not U|D|L|R"),
    };
    Ok((frame.parse()?, dir))
}
