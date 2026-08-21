use anyhow::Result;
use clap::Parser;
use net::render;
use net::sim::PlayerId;

use crab_world::RenderArgs;
use crab_world::controls::ControlsOverlayArgs;

use super::shared::{
    ChordScriptArgs, MATCH_SEED, boot_view, gcr_controls, nn_crab_policy, parse_hold,
};

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

    #[command(flatten)]
    chord: ChordScriptArgs,

    /// Capture this many frames as `<out-stem>.NNNN.png` instead of one shot —
    /// evidence clips for the animated ground looks (assemble with ffmpeg).
    #[arg(long)]
    anim_frames: Option<u32>,
    /// Render frames between captured anim frames (60 = 1 s of shader time).
    #[arg(long, default_value_t = 6)]
    anim_every: u32,

    /// Board a vehicle at this frame (repeatable, same foot→plane→ship→foot order as
    /// `net-screenshot`) and hold full forward drive while piloting — single-process
    /// flight evidence (rl#379: a clip at the plane's top speed needs no second peer).
    #[arg(long, value_name = "FRAME", value_parser = clap::value_parser!(u64).range(1..))]
    pilot_toggle_at: Vec<u64>,

    /// Match seed — picks the run's spawn locale on the tile (rl#305), so evidence
    /// shots can target a specific coordinate regime (e.g. the far-corner f32
    /// precision band, rl#334).
    #[arg(long, default_value_t = MATCH_SEED)]
    seed: u64,
    /// From this frame on, the local player walks a gentle arc — motion evidence
    /// (rl#334: texture stability is only visible while the eye translates).
    #[arg(long, value_name = "FRAME", value_parser = clap::value_parser!(u64).range(1..))]
    walk_at: Option<u64>,
    /// Hold JUMP over these on-foot frame windows, `from:to` pairs comma-separated
    /// (`to` empty = into the shot) — jump-feel evidence (rl#367): a short window
    /// taps, a long window rides the hold-to-float rise.
    #[arg(long, value_delimiter = ',')]
    jump_holds: Vec<String>,
    /// Hold SPRINT over these on-foot frame windows, same `from:to` shape — slide
    /// evidence (rl#368): the skid only enters above sprint pace.
    #[arg(long, value_delimiter = ',')]
    sprint_holds: Vec<String>,
    /// Hold SLIDE over these on-foot frame windows, same `from:to` shape (rl#368).
    #[arg(long, value_delimiter = ',')]
    slide_holds: Vec<String>,

    /// Walk dead straight instead of the default gentle arc — position-trace
    /// captures (rl#371) need a constant-velocity ground truth.
    #[arg(long)]
    walk_straight: bool,

    /// Hold every flight axis at zero while piloting instead of the default full
    /// forward drive — parked-craft captures (rl#377).
    #[arg(long)]
    pilot_park: bool,

    /// Model the frame clock as `1/HZ ± --frame-jitter-ms` per frame instead of the
    /// default one-sim-tick frames — reproduces a real display cadence (e.g. 60)
    /// beating against the 30 Hz sim (rl#371). Deterministic per --seed.
    #[arg(long, value_name = "HZ")]
    frame_hz: Option<f64>,
    /// Uniform ± bound, in ms, on each modeled frame delta (needs --frame-hz).
    #[arg(long, default_value_t = 0.0)]
    frame_jitter_ms: f64,
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
    if !args.pilot_toggle_at.is_empty()
        || args.walk_at.is_some()
        || !args.jump_holds.is_empty()
        || !args.sprint_holds.is_empty()
        || !args.slide_holds.is_empty()
    {
        let parse_holds = |specs: &[String]| {
            specs
                .iter()
                .map(|spec| parse_hold(spec))
                .collect::<Result<Vec<_>>>()
        };
        app.insert_resource(
            render::PilotScript::new(args.pilot_toggle_at, args.walk_at)
                .with_jump_holds(parse_holds(&args.jump_holds)?)
                .with_sprint_holds(parse_holds(&args.sprint_holds)?)
                .with_slide_holds(parse_holds(&args.slide_holds)?)
                .with_straight_walk(args.walk_straight)
                .with_park(args.pilot_park),
        );
    }
    if let Some(hz) = args.frame_hz {
        app.insert_resource(render::FrameDtModel::new(
            hz,
            args.frame_jitter_ms,
            args.seed,
        ));
    }
    args.chord.apply(&mut app)?;
    app.run();
    Ok(())
}
