//! `game` — the giant-crab rescue game (rl#38), built multiplayer-first on the
//! deterministic-lockstep + iroh netcode foundation (rl#39).
//!
//! This binary is the HEADLESS driver of the deterministic sim ([`rl::net::sim`] —
//! now the Phase 1 gray-box Extraction loop: first-person players, one giant crab,
//! an extraction point) over [`rl::net::lockstep`] and [`rl::net::transport`] (iroh
//! LAN discovery). It proves the netcode end to end — discovery, input exchange,
//! deterministic tick, desync detection — without a GPU (this box renders headlessly
//! at best). The windowed first-person client + the plane/heli vehicles are separate
//! later subs that plug into the same sim interface (documented on [`rl::net::sim`]);
//! they consume the state this driver advances, they don't replace it.
//!
//! Modes:
//! - `net` (default headless): bind an iroh endpoint, discover peers on the LAN,
//!   and run the lockstep loop for a fixed duration, printing per-second sync state.
//!   Run two copies on a LAN to see them find each other and stay in sync.
//! - `solo`: run the lockstep+sim loop with no network (one peer), for a quick
//!   smoke of the tick machinery (it stirs a placeholder input — real movement is
//!   the client's job, not this headless smoke's).
//! - `play`: the windowed first-person CLIENT ([`rl::net::render`]) — see the
//!   gray-box from the local player's eyes and play it, driving the SAME lockstep +
//!   transport as `net` (genuinely networked; `--solo` for a single offline peer).
//! - `fp-screenshot`: render one settled frame of the first-person view to a PNG and
//!   exit (GPU on, no window) — the headless evidence path for the sim→render
//!   pipeline on a box with no display.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, Subcommand};
use rl::net::lockstep::{INPUT_DELAY, Lockstep};
use rl::net::sim::{Input, PlayerId};
use rl::net::{net_loop, render, transport};

/// Tick rate of the deterministic sim. 30 Hz is plenty for Phase 0's dots and keeps
/// the lockstep stall window forgiving on a LAN; Phase 1 can raise it.
const TICK_HZ: u64 = 30;

#[derive(Parser)]
#[command(about = "Giant-crab rescue — Phase 0 netcode skeleton (rl#39)")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Networked headless run: discover LAN peers over iroh and run lockstep.
    Net(NetArgs),
    /// Single-peer headless smoke of the tick machinery (no network).
    Solo(SoloArgs),
    /// Windowed first-person client: see + play the gray-box on the lockstep sim.
    Play(PlayArgs),
    /// Render one frame of the first-person view to a PNG and exit (no window).
    FpScreenshot(FpScreenshotArgs),
}

#[derive(Parser)]
struct NetArgs {
    /// Wait this long for peers to be discovered before starting the tick loop.
    /// Discovery is mDNS, so a couple seconds covers a quiet LAN.
    #[arg(long, default_value_t = 4)]
    discover_secs: u64,
    /// Run the lockstep loop for this many seconds, then report and exit.
    #[arg(long, default_value_t = 10)]
    run_secs: u64,
    /// Expected peer count (including us). The loop waits up to `discover_secs` to
    /// reach it; if fewer are found it proceeds with whoever showed up (and a single
    /// peer simply runs solo over the network stack).
    #[arg(long, default_value_t = 2)]
    expect: usize,
}

#[derive(Parser)]
struct SoloArgs {
    #[arg(long, default_value_t = 5)]
    run_secs: u64,
}

#[derive(Parser)]
struct PlayArgs {
    /// Run a single offline peer (no network): see + play the sim solo. Without this
    /// the client discovers LAN peers over iroh and plays networked, exactly as `net`.
    #[arg(long)]
    solo: bool,
    /// Wait this long for peers before starting (networked play only).
    #[arg(long, default_value_t = 4)]
    discover_secs: u64,
    /// Expected peer count including us (networked play only); proceeds with whoever
    /// showed up after `discover_secs`.
    #[arg(long, default_value_t = 2)]
    expect: usize,
}

#[derive(Parser)]
struct FpScreenshotArgs {
    /// Output PNG path for the captured first-person frame.
    #[arg(long, default_value = "fp.png")]
    out: PathBuf,
    /// Frames to render before capturing. The sim advances one tick per frame (a
    /// fixed dt, so the composed scene is deterministic, not machine-speed-dependent),
    /// and the count also warms the GPU pipeline (early frames render black). So it's
    /// both "how far into the round" and "warmup"; ~40 keeps the players alive in
    /// frame, higher lets the round play out.
    #[arg(long, default_value_t = 90)]
    settle: u32,
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    /// Number of players to spawn in the solo scene, so the shot shows another
    /// player's avatar alongside the local one (player 0 is the local camera).
    #[arg(long, default_value_t = 2)]
    players: u8,
    /// Pan the screenshot camera this many degrees (about up) from the dead-ahead
    /// first-person aim, to frame the towering crab + the extraction pillar + the
    /// other players together (the giant crab fills the straight-ahead view). Still a
    /// first-person shot from the local eye; 0 = straight ahead.
    #[arg(long, default_value_t = 0.0)]
    cam_yaw: f32,
    /// Tilt the screenshot camera this many degrees (+ up) from the first-person aim.
    #[arg(long, default_value_t = 0.0)]
    cam_pitch: f32,
}

// Plain `main` (not `#[tokio::main]`): the windowed/screenshot client builds a Bevy
// app that owns the main thread and, for networked play, spins up its OWN tokio
// runtime inside `net_loop` — nesting that under an ambient `#[tokio::main]` runtime
// panics ("cannot start a runtime from within a runtime"). So each async mode (`net`)
// builds its runtime explicitly, and the sync modes (`solo`/`play`/`fp-screenshot`)
// never touch one they don't own.
fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    // No subcommand → the networked mode with its own defaults (parsed from an empty
    // arg list so the `default_value_t`s are the single source, not duplicated here).
    let command = Cli::parse()
        .command
        .unwrap_or_else(|| Command::Net(NetArgs::parse_from(["game"])));
    match command {
        Command::Net(args) => tokio::runtime::Runtime::new()?.block_on(run_net(args)),
        Command::Solo(args) => run_solo(args),
        Command::Play(args) => run_play(args),
        Command::FpScreenshot(args) => run_fp_screenshot(args),
    }
}

/// Windowed first-person client. Solo (one offline peer) or networked (discover LAN
/// peers and play in lockstep) per `--solo`. Builds the Bevy app from
/// [`render::build_windowed_app`] and runs it — the app owns the lockstep loop and
/// drives it on a fixed-timestep accumulator (see [`rl::net::render`]).
fn run_play(args: PlayArgs) -> Result<()> {
    let (ls, net) = if args.solo {
        let me = PlayerId(0);
        (Lockstep::new(MATCH_SEED, &[me], me), None)
    } else {
        let (ls, driver) =
            net_loop::connect_and_assign(MATCH_SEED, args.discover_secs, args.expect)?;
        (ls, Some(driver))
    };
    render::build_windowed_app(ls, net).run();
    Ok(())
}

/// Headless first-person screenshot: build a solo lockstep with `--players`
/// participants (so a remote avatar is in frame beside the local one), render one
/// settled frame of the FP view to a PNG, and exit. The evidence path for the
/// sim→render pipeline on a display-less box.
fn run_fp_screenshot(args: FpScreenshotArgs) -> Result<()> {
    let me = PlayerId(0);
    let players: Vec<PlayerId> = (0..args.players.max(1)).map(PlayerId).collect();
    let ls = Lockstep::new(MATCH_SEED, &players, me);
    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(args.cam_yaw, args.cam_pitch);
    render::build_screenshot_app(ls, cfg).run();
    Ok(())
}

/// Deterministic match seed. In Phase 0 it's a constant so independently-launched
/// peers agree without a handshake; Phase 1's session setup will negotiate it (the
/// lower-id peer proposes, say) — the sim already takes it as a parameter.
const MATCH_SEED: u64 = 0x6372_6162; // "crab"

/// Drive the lockstep sim from a constant local input, ticking at [`TICK_HZ`]. Pure
/// machinery check: no peers, so our own input completes every tick.
fn run_solo(args: SoloArgs) -> Result<()> {
    let me = PlayerId(0);
    let mut ls = Lockstep::new(MATCH_SEED, &[me], me);
    let tick_dt = Duration::from_secs_f64(1.0 / TICK_HZ as f64);
    let end = Instant::now() + Duration::from_secs(args.run_secs);
    let mut next = Instant::now();
    while Instant::now() < end {
        // A lazy circular stir so the dot visibly moves.
        let t = ls.next_tick() as f32 * 0.1;
        ls.submit_local_input(Input::from_axes(t.cos(), t.sin()));
        let desyncs = ls.try_advance();
        debug_assert!(desyncs.is_empty(), "solo can't desync");
        next += tick_dt;
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    let p = ls.sim().player(me).unwrap();
    let pos = p.pos();
    let crab = ls.sim().crab().pos();
    println!(
        "solo: {} ticks, player=({}, {}) yaw={} status={:?}, crab=({}, {}), outcome={:?}, hash={:#018x}",
        ls.sim().tick(),
        pos.x,
        pos.z,
        p.yaw(),
        p.status(),
        crab.x,
        crab.z,
        ls.sim().outcome(),
        ls.sim().state_hash()
    );
    Ok(())
}

/// Networked run: bind, discover, assign deterministic player ids from the sorted
/// endpoint-id set, then tick lockstep — broadcasting our input and ingesting peers'
/// each tick — and report whether we stayed in sync.
async fn run_net(args: NetArgs) -> Result<()> {
    let mut session = transport::start_session().await?;
    let my_eid = session.endpoint_id();
    println!("game endpoint id: {my_eid}");

    // Discover + freeze + assign ids via the shared cold-start (same code the windowed
    // client runs, so the two can't drift apart and desync). Replay any inputs that
    // arrived during discovery into the fresh sim.
    let frozen =
        net_loop::discover_and_freeze(&mut session, args.discover_secs, args.expect).await?;
    let me = frozen.me;
    let id_map = &frozen.id_map;
    let all_ids: Vec<PlayerId> = id_map.values().copied().collect();
    println!(
        "starting lockstep: {} player(s), I am {:?} ({})",
        all_ids.len(),
        me,
        my_eid.fmt_short()
    );

    let mut ls = Lockstep::new(MATCH_SEED, &all_ids, me);
    net_loop::replay_early(&mut ls, &frozen);

    let tick_dt = Duration::from_secs_f64(1.0 / TICK_HZ as f64);
    let mut ticker = tokio::time::interval(tick_dt);
    let end = Instant::now() + Duration::from_secs(args.run_secs);
    let mut total_desyncs = 0usize;
    // Report at fixed TICK boundaries (not wall-clock) so both peers print the SAME
    // ticks — the `(tick, hash)` lines are then directly comparable across peers, an
    // external check on the internal desync cross-check.
    let mut next_report_tick = TICK_HZ;

    while Instant::now() < end {
        ticker.tick().await;

        // Ingest everything the transport has for us this tick. A late-arriving hash
        // for an already-applied tick can surface a fault right here.
        while let Some(m) = session.try_recv() {
            if let Some(&pid) = id_map.get(&m.from)
                && let Some(f) = ls.record_remote(pid, m.msg)
            {
                report_fault(&mut total_desyncs, f);
            }
        }

        // Issue our input for this tick and tell every peer.
        let t = ls.next_tick() as f32 * 0.1;
        let msg = ls.submit_local_input(Input::from_axes(t.cos(), t.sin()));
        session.broadcast(&msg).await;

        // Advance every ready tick; surface faults found as we apply.
        for f in ls.try_advance() {
            report_fault(&mut total_desyncs, f);
        }

        // Report once the sim crosses each TICK_HZ boundary. The label is the actual
        // current tick and the hash is that same tick's state, so the pair is exact;
        // both peers cross the same boundaries, making the lines comparable.
        if ls.sim().tick() >= next_report_tick {
            next_report_tick = (ls.sim().tick() / TICK_HZ + 1) * TICK_HZ;
            println!(
                "tick={:>5} peers={} statehash={:#018x} desyncs={}",
                ls.sim().tick(),
                session.connected_peers().await.len(),
                ls.sim().state_hash(),
                total_desyncs,
            );
        }
    }

    println!(
        "done: {} ticks applied, {} desyncs, final hash {:#018x}",
        ls.sim().tick(),
        total_desyncs,
        ls.sim().state_hash()
    );
    if all_ids.len() > 1
        && ls.sim().tick() < (args.run_secs * TICK_HZ).saturating_sub(INPUT_DELAY + TICK_HZ)
    {
        // We applied far fewer ticks than wall time allowed → we spent the run
        // stalled waiting for a peer's input. Flag it; a healthy link keeps pace.
        eprintln!(
            "WARNING: only {} ticks in {}s — peer link stalled (missing inputs)",
            ls.sim().tick(),
            args.run_secs
        );
    }
    session.shutdown().await;
    Ok(())
}

/// Count and log a cross-check fault. A desync is unrecoverable in lockstep, but
/// Phase 0 keeps running so the test harness can observe how many ticks faulted
/// rather than aborting on the first.
fn report_fault(total: &mut usize, f: rl::net::lockstep::Fault) {
    use rl::net::lockstep::Fault;
    *total += 1;
    match f {
        Fault::Desync {
            tick,
            peer,
            local_hash,
            peer_hash,
        } => eprintln!(
            "DESYNC at tick {tick}: peer {peer:?} hash {peer_hash:#018x} != local {local_hash:#018x}"
        ),
        Fault::Unverifiable {
            tick,
            peer,
            peer_hash,
        } => eprintln!(
            "UNVERIFIABLE at tick {tick}: peer {peer:?} hash {peer_hash:#018x} fell out of our history window"
        ),
    }
}
