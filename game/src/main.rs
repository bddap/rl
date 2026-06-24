//! `game` — the giant-crab rescue game (rl#38), built multiplayer-first on the
//! deterministic-lockstep + iroh netcode foundation (rl#39).
//!
//! This binary is the HEADLESS driver of the deterministic sim ([`rl_core::net::sim`] —
//! now the Phase 1 gray-box Extraction loop: first-person players, one giant crab,
//! an extraction point) over [`rl_core::net::lockstep`] and [`rl_core::net::transport`] (iroh
//! LAN discovery). It proves the netcode end to end — discovery, input exchange,
//! deterministic tick, desync detection — without a GPU (this box renders headlessly
//! at best). The windowed first-person client + the plane/heli vehicles are separate
//! later subs that plug into the same sim interface (documented on [`rl_core::net::sim`]);
//! they consume the state this driver advances, they don't replace it.
//!
//! Modes:
//! - `net` (default headless): bind an iroh endpoint, discover peers on the LAN,
//!   and run the lockstep loop for a fixed duration, printing per-second sync state.
//!   Run two copies on a LAN to see them find each other and stay in sync.
//! - `solo`: run the lockstep+sim loop with no network (one peer), for a quick
//!   smoke of the tick machinery (it stirs a placeholder input — real movement is
//!   the client's job, not this headless smoke's).
//! - `play`: the windowed first-person CLIENT ([`rl_core::net::render`]) — see the
//!   gray-box from the local player's eyes and play it, on the SAME lockstep +
//!   transport as `net`. Boots to a Host / Join menu; `--host`/`--join <code>` skip it
//!   for scripting/tests.
//! - `fp-screenshot`: render one settled frame of the first-person view to a PNG and
//!   exit (GPU on, no window) — the headless evidence path for the sim→render
//!   pipeline on a box with no display.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, Subcommand};
use iroh::EndpointId;
use rl_core::net::lockstep::{INPUT_DELAY, Lockstep};
// `TICK_HZ` is the deterministic sim's tick rate. The ONE source lives in `net::sim`
// (so a render peer and this headless driver agree); import it rather than redeclare a
// second `30` that could silently drift from the sim's.
use rl_core::net::sim::{Input, PlayerId, TICK_HZ};
use rl_core::net::telemetry::{self, TELEMETRY_TICK_EVERY, TelemetryEvent, TelemetrySender};
use rl_core::net::{net_loop, render, transport};

#[derive(Parser)]
#[command(about = "Giant-crab rescue — Phase 1 gray-box extraction loop on deterministic lockstep + iroh")]
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
    /// Run the live-telemetry collector: bind a stable-id iroh endpoint, print its id,
    /// and stream every connected game's events as a merged human feed (for remote
    /// debugging — `Monitor` its stdout). Pass each game `--telemetry <printed id>`.
    TelemetryCollector(TelemetryCollectorArgs),
    /// Headless verification of the SOLO NN crab (no window / GPU / display): step the
    /// real rapier-NN crab for N ticks against a still player and log the crab's game
    /// position + its shrinking distance to the player — the evidence it WALKS toward the
    /// player under the trained policy. Runs the seed TWICE and compares the final state
    /// hash, the single-peer reproducibility check.
    NnCrabProbe(NnCrabProbeArgs),
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
    /// Stream live telemetry to the collector with this endpoint id (from
    /// `game telemetry-collector`). Opens a SEPARATE iroh connection on a distinct ALPN
    /// — the lockstep transport/determinism is untouched, and a telemetry failure never
    /// affects the match. Omit to run with no telemetry.
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,
}

#[derive(Parser)]
struct SoloArgs {
    #[arg(long, default_value_t = 5)]
    run_secs: u64,
}

#[derive(Parser)]
struct PlayArgs {
    /// Skip the menu and HOST a networked match directly (scripted/test entry): form over
    /// the LAN and start with whoever joins, or solo if nobody does. Equivalent to the
    /// menu's Host without the click. A Host that finds no peer IS the solo round — one
    /// codepath, no separate solo flag; bare `play` shows the menu instead.
    #[arg(long, conflicts_with = "join")]
    host: bool,
    /// Skip the menu and JOIN a host by its endpoint-id code (scripted/test entry): dial the
    /// code, then form. Equivalent to the menu's Join-by-code without the click. The bare
    /// flag `--join` with no value joins by LAN discovery (no explicit dial).
    #[arg(
        long,
        value_name = "JOIN_CODE",
        num_args = 0..=1,
        default_missing_value = "",
        conflicts_with = "host"
    )]
    join: Option<String>,
    /// Wait this long for peers before starting (the scripted `--host`/`--join` paths only).
    #[arg(long, default_value_t = 4)]
    discover_secs: u64,
    /// Expected peer count including us (the scripted `--host`/`--join` paths only); proceeds
    /// with whoever showed up after `discover_secs`.
    #[arg(long, default_value_t = 2)]
    expect: usize,
    /// Stream live telemetry to this collector endpoint id (networked play only; see
    /// `NetArgs::telemetry`). Separate ALPN/connection — never perturbs the lockstep.
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,

    /// Directory holding the trained crab policy (`brain.bin` + `normalizer.bin`). On a SOLO
    /// round (a Host-alone Start, or a scripted `--host` that found no peer) it drives the
    /// giant crab with the rapier-simulated NN body instead of the integer point-pursuer.
    /// Defaults to the `RL_CRAB_CHECKPOINT_DIR` env var, else `assets/weights` under the asset
    /// root; a missing/empty dir falls back to the integer crab (logged). Ignored on a
    /// NETWORKED round — a float rapier crab is not cross-peer deterministic, so multiplayer
    /// always keeps the integer crab.
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<PathBuf>,
}

#[derive(Parser)]
struct FpScreenshotArgs {
    /// Output PNG path for the captured first-person frame.
    #[arg(long, default_value = "fp.png")]
    out: PathBuf,
    /// Frames to render before capturing — both how far into the round and GPU warmup
    /// (early frames render black). The sim advances one tick per frame at a fixed dt, so
    /// the scene is deterministic, not machine-speed-dependent. ~40 keeps the players in
    /// frame; higher lets the round play out.
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
    /// Pan the screenshot camera this many degrees from the dead-ahead first-person aim
    /// (0 = straight ahead), to frame the towering crab + extraction pillar + other players
    /// together when the crab would otherwise fill the straight-ahead view.
    #[arg(long, default_value_t = 0.0)]
    cam_yaw: f32,
    /// Tilt the screenshot camera this many degrees (+ up) from the first-person aim.
    #[arg(long, default_value_t = 0.0)]
    cam_pitch: f32,
}

#[derive(Parser)]
struct TelemetryCollectorArgs {
    /// Path to the collector's persistent secret key (generated on first run). Pinning
    /// it keeps the collector's endpoint id STABLE across restarts, so the id baked into
    /// each game's `--telemetry` never goes stale.
    #[arg(long, default_value = rl_core::net::telemetry::DEFAULT_KEY_PATH)]
    key: PathBuf,
}

#[derive(Parser)]
struct NnCrabProbeArgs {
    /// Trained crab checkpoint dir (`brain.bin` + `normalizer.bin`). Same resolution as
    /// `play --nn-crab-checkpoint`: this flag, else `RL_CRAB_CHECKPOINT_DIR`, else
    /// `assets/weights` under the asset root.
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<PathBuf>,
    /// Sim ticks to step the crab for. Defaults high (1200 ≈ 40 s at 30 Hz) because the
    /// current `ckpt-best.locomotion` checkpoint locomotes SLOWLY (it leans/reaches toward
    /// the lead target rather than striding — its reward has no base-locomotion term), so a
    /// short run shows little net travel even though the crab is clearly NN-driven.
    #[arg(long, default_value_t = 1200)]
    ticks: u64,
    /// Log a sample every this-many ticks.
    #[arg(long, default_value_t = 100)]
    log_every: u64,
    /// Match seed (also the determinism check: the run is repeated with this seed and the
    /// final hashes compared).
    #[arg(long, default_value_t = 0x6372_6162)]
    seed: u64,
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
        Command::TelemetryCollector(args) => {
            tokio::runtime::Runtime::new()?.block_on(telemetry::run_collector(&args.key))
        }
        Command::NnCrabProbe(args) => run_nn_crab_probe(args),
    }
}

/// Headless NN-crab verification: step the real rapier crab for `--ticks` against a still
/// player, log its game position + shrinking distance to the player, and repeat the seed
/// to confirm the same trajectory hash twice (single-peer reproducibility). Prints a table
/// and a PASS/look-here verdict; exits nonzero if the crab never closed the gap (so it
/// doubles as a regression gate on "the policy actually drives the crab toward the player").
fn run_nn_crab_probe(args: NnCrabProbeArgs) -> Result<()> {
    use rl_core::net::solo_crab::run_headless_probe;

    let Some(dir) = nn_crab_checkpoint_dir(args.checkpoint) else {
        anyhow::bail!("nn-crab-probe: no brain.bin at the resolved checkpoint dir");
    };
    println!("nn-crab-probe: checkpoint={}", dir.display());
    println!("nn-crab-probe: seed={:#x} ticks={}", args.seed, args.ticks);

    let samples = run_headless_probe(&dir, args.seed, args.ticks, args.log_every);
    if samples.is_empty() {
        anyhow::bail!("nn-crab-probe: no samples — the crab never stepped");
    }

    println!("\n  tick   crab_x   crab_z   dist  | carapace x/y/z (walks?)  | claw→tgt");
    for s in &samples {
        println!(
            "  {:>5}  {:>7.2}  {:>7.2}  {:>6.2} | {:>7.2} {:>5.2} {:>7.2}  | {:>7.3}",
            s.tick,
            s.crab_x_m,
            s.crab_z_m,
            s.dist_to_prey_m,
            s.carapace_arena_x,
            s.carapace_y,
            s.carapace_arena_z,
            s.min_claw_to_target_m,
        );
    }

    let first = samples.first().unwrap().dist_to_prey_m;
    let last = samples.last().unwrap().dist_to_prey_m;
    let closed = first - last;
    println!(
        "\nnn-crab-probe: distance to player {first:.3} m → {last:.3} m  (closed {closed:.3} m)"
    );

    // Determinism (single peer): same seed twice ⇒ identical final hash + trajectory.
    let again = run_headless_probe(&dir, args.seed, args.ticks, args.log_every);
    let hash_a = samples.last().unwrap().state_hash;
    let hash_b = again.last().map(|s| s.state_hash).unwrap_or(0);
    let traj_match = samples.len() == again.len()
        && samples
            .iter()
            .zip(&again)
            .all(|(a, b)| a.state_hash == b.state_hash);
    println!(
        "nn-crab-probe: determinism — final hash A={hash_a:#018x} B={hash_b:#018x} ({}), \
         full trajectory {}",
        if hash_a == hash_b { "MATCH" } else { "DIFFER" },
        if traj_match { "MATCHES" } else { "DIFFERS" },
    );

    // Verdict: the crab must have closed the gap (the policy walked it toward the player)
    // AND the run must be reproducible. A crab that drifted away or sat still fails.
    if closed > 1.0 && traj_match {
        println!("nn-crab-probe: PASS — NN crab walked toward the player, reproducibly");
        Ok(())
    } else {
        anyhow::bail!(
            "nn-crab-probe: FAIL — closed {closed:.3} m (want > 1.0) / trajectory \
             reproducible = {traj_match}"
        )
    }
}

/// Windowed first-person client. The DEFAULT (`game play` with no flag) shows the boot menu
/// — the player picks Host / Join — and the round is built only after the choice, never
/// touching the deterministic sim before then (see [`rl_core::net::render::Boot`]). The
/// scripted flags bypass the menu for tests/scripts:
/// - `--host` → host directly: form over the LAN, start with whoever joins (solo if none).
/// - `--join [CODE]` → join directly: dial CODE (or LAN-discover if bare), then form.
///
/// Both scripted paths form the match UP FRONT and hand a ready round to
/// [`render::Boot::Round`], so they boot straight into play with no menu. They reuse the
/// SAME barrier as the menu and as `game net`, so the agreed roster + seed are identical
/// however play was reached; a Host-alone fallback yields a solo round when nobody shows.
fn run_play(args: PlayArgs) -> Result<()> {
    let boot = if args.host || args.join.is_some() {
        // Scripted host/join: dial the join code if joining (blank/absent = LAN discover),
        // form over the barrier, and hand the result to Boot::Round. Host never dials. This
        // is the default timer-closed barrier (no interactive lobby), so it can't be
        // cancelled — only Joined or the Alone solo fallback.
        let dial = match &args.join {
            Some(code) if !code.trim().is_empty() => Some(code.trim().parse::<EndpointId>()?),
            _ => None,
        };
        let result = net_loop::connect_and_form_dialing(
            MATCH_SEED,
            args.discover_secs,
            args.expect,
            dial,
            args.telemetry,
            None,
        )?;
        match result {
            net_loop::MatchResult::Joined(joined) => {
                let (ls, driver) = *joined;
                render::Boot::Round(Box::new((ls, Some(driver))))
            }
            // Nobody showed: play the shared solo round (the Host-alone outcome).
            net_loop::MatchResult::Alone => {
                render::Boot::Round(Box::new((net_loop::solo_lockstep_for(MATCH_SEED), None)))
            }
            // The scripted path runs no interactive lobby, so a Cancel is impossible.
            net_loop::MatchResult::Cancelled => {
                unreachable!("scripted --host/--join has no lobby to cancel")
            }
        }
    } else {
        // Interactive default: the boot menu builds the round after the player chooses.
        render::Boot::Menu {
            seed: MATCH_SEED,
            telemetry: args.telemetry,
        }
    };
    // The solo NN-crab checkpoint dir. Resolved unconditionally — it's just a path;
    // `build_windowed_app` applies it only on a solo round (the integer crab stands in on
    // every networked path, where a float rapier crab isn't cross-peer deterministic).
    // Skip a dir with no `brain.bin` so a bad path degrades to the integer crab rather
    // than every-frame load failures.
    let solo_crab = nn_crab_checkpoint_dir(args.nn_crab_checkpoint);
    render::build_windowed_app(boot, solo_crab).run();
    Ok(())
}

/// Resolve the solo NN-crab checkpoint dir: the `--nn-crab-checkpoint` flag (`flag`), else
/// the `RL_CRAB_CHECKPOINT_DIR` env var (deploy sets this), else `assets/weights` under the
/// asset root (`BEVY_ASSET_ROOT`, else the binary's cwd) — so a checkpoint can be chosen at
/// runtime, no recompile. `None` if the resolved dir has no `brain.bin`, and the caller then
/// keeps the integer crab.
fn nn_crab_checkpoint_dir(flag: Option<PathBuf>) -> Option<PathBuf> {
    let dir = flag
        .or_else(|| std::env::var_os("RL_CRAB_CHECKPOINT_DIR").map(PathBuf::from))
        .unwrap_or_else(|| {
            let root = std::env::var_os("BEVY_ASSET_ROOT").map_or_else(
                || std::env::current_dir().unwrap_or_default(),
                PathBuf::from,
            );
            root.join("assets").join("weights")
        });
    if dir.join("brain.bin").exists() {
        Some(dir)
    } else {
        eprintln!(
            "solo crab: no brain.bin under {} — using the integer point-pursuer crab \
             (set --nn-crab-checkpoint or RL_CRAB_CHECKPOINT_DIR to a trained checkpoint)",
            dir.display()
        );
        None
    }
}

/// Headless first-person screenshot: build a solo lockstep with `--players`
/// participants (so a remote avatar is in frame beside the local one), render one
/// settled frame of the FP view to a PNG, and exit. The evidence path for the
/// sim→render pipeline on a display-less box.
fn run_fp_screenshot(args: FpScreenshotArgs) -> Result<()> {
    let me = PlayerId(0);
    let players: Vec<PlayerId> = (0..args.players.max(1)).map(PlayerId).collect();
    // RL_VEHICLE=plane: make the NON-local players pilots and keep the local (player 0)
    // a ground observer, so the captured FP frame clearly shows a remote plane's gray box
    // flying — the evidence the plane renders. Needs `--players >= 2`: with one player
    // there's no remote plane to frame, so the shot stays on foot (a lone pilot would show
    // empty sky — worthless as evidence). Unset ⇒ the unchanged foot shot.
    let pilots: Vec<PlayerId> = match std::env::var("RL_VEHICLE").as_deref() {
        Ok("plane") if players.len() > 1 => players[1..].to_vec(),
        _ => Vec::new(),
    };
    let ls = Lockstep::new_with_pilots(MATCH_SEED, &players, me, &pilots);
    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(args.cam_yaw, args.cam_pitch);
    render::build_screenshot_app(ls, cfg).run();
    Ok(())
}

/// Deterministic match seed: a constant so independently-launched peers agree without a
/// handshake. The sim takes it as a parameter, so a future session setup can negotiate it
/// (the lower-id peer proposes, say) without touching the sim.
const MATCH_SEED: u64 = 0x6372_6162; // "crab"

/// Drive the lockstep sim from a constant local input, ticking at [`TICK_HZ`]. Pure
/// machinery check: no peers, so our own input completes every tick.
fn run_solo(args: SoloArgs) -> Result<()> {
    run_solo_round(args.run_secs)
}

/// One offline lockstep round for `run_secs`: a single peer whose own input completes
/// every tick (no network), ticking at [`TICK_HZ`] and printing a final summary. Shared by
/// the `solo` command and the headless `net` no-peer fallback, so the alone case runs the
/// SAME deterministic solo path — no second sim loop to drift.
fn run_solo_round(run_secs: u64) -> Result<()> {
    let me = PlayerId(0);
    let mut ls = Lockstep::new(MATCH_SEED, &[me], me);
    let tick_dt = Duration::from_secs_f64(1.0 / TICK_HZ as f64);
    let end = Instant::now() + Duration::from_secs(run_secs);
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

    // Open the telemetry side-channel (if configured) BEFORE forming the match, so the
    // collector sees the roster fill. Best-effort + isolated: separate iroh endpoint,
    // separate ALPN — see `rl_core::net::telemetry`. A failure yields `None` and the run is
    // byte-for-byte the no-telemetry run.
    let tel = net_loop::connect_telemetry(args.telemetry, my_eid).await;

    // Form one agreed match via the shared cold-start barrier (same code the windowed
    // client runs, so the two can't drift apart and desync). Replay any inputs that
    // arrived during formation into the fresh sim. If discovery finds no peer, tear down
    // the network side and run a solo round instead of awaiting an empty match.
    let frozen = match net_loop::form_match(
        &mut session,
        args.discover_secs,
        args.expect,
        tel.as_ref(),
        None, // headless: timer-closed barrier, no interactive lobby
    )
    .await?
    {
        net_loop::Formation::Agreed(frozen) => frozen,
        net_loop::Formation::Alone => {
            drop(tel);
            session.shutdown().await;
            return run_solo_round(args.run_secs);
        }
        // No cancel channel on the headless path, so a Cancel can never be signalled.
        net_loop::Formation::Cancelled => unreachable!("headless net has no lobby to cancel"),
    };
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
    // Telemetry-side sampling cursor (independent of the stdout report cadence) and a
    // one-shot latch so RoundDecided is reported exactly once.
    let mut next_tel_tick = TELEMETRY_TICK_EVERY;
    let mut reported_outcome = false;

    while Instant::now() < end {
        ticker.tick().await;

        // Ingest everything the transport has for us this tick. Only lockstep ticks
        // matter now; a stray barrier beat from a peer still winding down its formation
        // loop is ignored. A late-arriving hash for an already-applied tick can surface
        // a fault right here.
        while let Some(m) = session.try_recv() {
            if let transport::PeerWire::Tick(msg) = m.msg
                && let Some(&pid) = id_map.get(&m.from)
                && let Some(f) = ls.record_remote(pid, msg)
            {
                report_fault(&mut total_desyncs, f, tel.as_ref());
            }
        }

        // Issue our input for this tick and tell every peer.
        let t = ls.next_tick() as f32 * 0.1;
        let issue_tick = ls.next_tick();
        let input = Input::from_axes(t.cos(), t.sin());
        let msg = ls.submit_local_input(input);
        session.broadcast(&msg).await;

        // Advance every ready tick; surface faults found as we apply.
        for f in ls.try_advance() {
            report_fault(&mut total_desyncs, f, tel.as_ref());
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

        // Sampled telemetry: a Tick snapshot (+ the input we just issued) every
        // TELEMETRY_TICK_EVERY ticks, and a one-shot RoundDecided when the round ends.
        // All read-only on the sim; all best-effort (a send that can't keep up drops).
        if let Some(t) = tel.as_ref() {
            if ls.sim().tick() >= next_tel_tick {
                next_tel_tick = (ls.sim().tick() / TELEMETRY_TICK_EVERY + 1) * TELEMETRY_TICK_EVERY;
                // Agreed roster size (us + peers) — the same quantity render.rs and the
                // final snapshot report, so the feed's `roster` field means one thing
                // across every driver.
                t.send(TelemetryEvent::tick(ls.sim(), total_desyncs, all_ids.len()));
                t.send(TelemetryEvent::input(issue_tick, input));
            }
            if !reported_outcome && ls.sim().outcome() != rl_core::net::sim::Outcome::Ongoing {
                reported_outcome = true;
                t.send(TelemetryEvent::round_decided(ls.sim()));
            }
        }
    }

    println!(
        "done: {} ticks applied, {} desyncs, final hash {:#018x}",
        ls.sim().tick(),
        total_desyncs,
        ls.sim().state_hash()
    );
    // A final snapshot so the collector records where this deck ended even if the round
    // never "decided" within run_secs (the common case for a short headless run).
    if let Some(t) = tel.as_ref() {
        t.send(TelemetryEvent::tick(ls.sim(), total_desyncs, all_ids.len()));
    }
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
    // Give the best-effort telemetry queue a moment to flush its tail before the
    // process tears down the endpoint (the sender task drains on its own runtime). A
    // no-op when telemetry is off.
    if tel.is_some() {
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    drop(tel); // close the telemetry channel so its task finishes the stream cleanly
    session.shutdown().await;
    Ok(())
}

/// Count and log a cross-check fault. A desync is unrecoverable in lockstep, but we keep
/// running so the test harness can observe how many ticks faulted rather than aborting on
/// the first. Also mirrored to telemetry (best-effort) so a remote operator sees the
/// divergence the instant a deck does.
fn report_fault(
    total: &mut usize,
    f: rl_core::net::lockstep::Fault,
    telemetry: Option<&TelemetrySender>,
) {
    use rl_core::net::lockstep::Fault;
    *total += 1;
    if let Some(t) = telemetry {
        t.send(TelemetryEvent::fault(&f));
    }
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
