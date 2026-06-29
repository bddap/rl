//! `game` — Giant Crab Rescue (rl#38): the CLI over the deterministic-lockstep + iroh
//! sim ([`net::sim`]/[`net::lockstep`]/[`net::transport`]). First-person players reach
//! an extraction point while a trained-NN giant crab (Sally) hunts them; `solo` is just
//! the zero-remote-peer case of the one networked path, not a separate mode.
//!
//! Subcommands:
//! - `net` (default headless): discover peers over iroh and run the lockstep loop,
//!   printing per-second sync state — proves discovery/input-exchange/desync-detection.
//! - `solo`: the same loop with no peers, a quick smoke of the tick machinery.
//! - `play`: the windowed first-person client ([`net::render`]); Host/Join menu, or
//!   `--host`/`--join <code>` to skip it for scripting.
//! - `fp-screenshot`: render one settled frame to a PNG and exit (GPU, no window) — the
//!   headless evidence path for the sim→render pipeline.
//! - `nn-crab-probe` / `nn-crab-xpeer`: determinism gates for the armed NN crab —
//!   single-peer and cross-peer per-tick hash logs (rl#82/#114).
//! - `checkpoint-check`: verify a checkpoint's rig dims fit the crab before arming.
//! - `telemetry-collector`: sink for the OTLP-over-iroh telemetry stream.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use iroh::EndpointId;
use net::lockstep::{INPUT_DELAY, Lockstep};
// `TICK_HZ` is the deterministic sim's tick rate. The ONE source lives in `net::sim`
// (so a render peer and this headless driver agree); import it rather than redeclare a
// second `30` that could silently drift from the sim's.
use net::sim::{Input, PlayerId, TICK_HZ};
use net::telemetry::{self, TELEMETRY_TICK_EVERY, TelemetryEvent, TelemetrySender};
use net::{net_loop, render, transport};

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
    /// The decisive GCR #82 gate: run the real rapier-NN crab as the giant crab on TWO
    /// independent in-process peers exchanging lockstep inputs, and confirm their per-tick
    /// state hashes stay byte-identical — the float NN crab IS the deterministic multiplayer
    /// crab. Writes each peer's `<tick> <hash>` log (`--hash-log-a` / `--hash-log-b`) so they
    /// can be `diff`ed, and exits nonzero on any divergence (so it doubles as a CI gate).
    NnCrabXpeer(NnCrabXpeerArgs),
    /// Rig-compatibility gate for the release/deploy pipeline: load a checkpoint's
    /// `brain.bin`, read its `(obs, action)` dims, and compare them to THIS binary's
    /// compiled crab rig. Exits 0 only on an exact match; nonzero (with an actionable
    /// message) if the brain is missing/unreadable or its dims differ. A mismatched
    /// checkpoint loads "fine" at runtime but silently degrades the NN crab to its rest
    /// pose, so the release builder runs this against the checkpoint it's about to bundle
    /// and refuses to publish on a nonzero exit — the mismatch is loud, not silent.
    CheckpointCheck(CheckpointCheckArgs),
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
    /// Write a full per-tick `<tick> <state_hash>` log to this file (every applied tick,
    /// keyed by the true tick — unlike the coarse stdout report cadence). Two peers (or two
    /// machines) running the SAME match must produce logs that `diff` byte-identically over
    /// their overlapping tick range: the cross-peer (and cross-machine) determinism proof
    /// (rl#94). Omit for no log.
    #[arg(long, value_name = "FILE")]
    hash_log: Option<PathBuf>,
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

    /// Directory holding the trained crab policy (`brain.bin` + `normalizer.bin`) — REQUIRED: the
    /// giant crab IS the rapier-simulated NN body (rl#114, no integer fallback). It drives the crab
    /// on a SOLO round (a Host-alone Start, or a scripted `--host` that found no peer) AND — since
    /// the GCR fold (rl#82) — on a NETWORKED round once peers agree on the brain + collider asset
    /// (the weights/asset handshake, [`net::may_arm_external_crab`]): the float NN crab is
    /// then cross-peer deterministic (`enhanced-determinism`, proven by `game nn-crab-xpeer`). A
    /// missing/empty dir, or a networked round whose peers DON'T agree, FAILS LOUD with an
    /// actionable message rather than substituting a fake crab. Defaults to the
    /// `RL_CRAB_CHECKPOINT_DIR` env var, else `assets/weights` under the asset root.
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<PathBuf>,

    /// Start the debug-wireframe collider overlay in this mode (default: off, the
    /// player-facing showcase). `aligned` reposes the crab colliders to the giant render
    /// scale so the cage overlays the mesh; `raw` is rapier's debug-render at true physics
    /// scale (crab tiny + offset — the scale-mismatch diagnostic). F3 cycles it live. Unset
    /// falls back to the `RL_DEBUG_WIREFRAME` / `RL_DEBUG_COLLIDERS` env (off otherwise).
    #[arg(long, value_name = "off|aligned|raw")]
    debug_wireframe: Option<String>,
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
    /// Override the camera's vertical field of view (degrees). Widen it (e.g. 90) to fit the
    /// whole towering giant crab in one frame — it stands only a body-length from the player, so
    /// the default FOV otherwise shows just a slab of carapace. Unset keeps Bevy's default.
    #[arg(long)]
    cam_fov: Option<f32>,
    /// Make the NON-local players pilots (flying planes) so the captured frame shows a
    /// plane's gray box in the air — the evidence the plane renders. Needs `--players >= 2`
    /// (a lone pilot would show empty sky). Off ⇒ the unchanged on-foot shot.
    #[arg(long, default_value_t = false)]
    plane: bool,
    /// Arm the real trained rapier-NN crab ("Sally") + skin for the shot, instead of the static
    /// integer silhouette — the SAME `--nn-crab-checkpoint` resolution as `play`, so an evidence
    /// frame composes the actual armed crab the windowed solo client renders (reposed + scaled to
    /// the giant). A dir with no `brain.bin` errors out (as `play` does); omit for the silhouette shot.
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<PathBuf>,

    /// Capture the debug-wireframe collider overlay in this mode (default: off). `aligned`
    /// draws the crab colliders reposed to the giant render scale (overlays the mesh); `raw`
    /// is rapier's debug-render at true physics scale. The evidence path for the toggle.
    #[arg(long, value_name = "off|aligned|raw")]
    debug_wireframe: Option<String>,
}

#[derive(Parser)]
struct TelemetryCollectorArgs {
    /// Path to the collector's persistent secret key (generated on first run). Pinning
    /// it keeps the collector's endpoint id STABLE across restarts, so the id baked into
    /// each game's `--telemetry` never goes stale.
    #[arg(long, default_value = net::telemetry::DEFAULT_KEY_PATH)]
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
    /// Write a full per-tick `<tick> <state_hash>` log to this file (forces a sample every
    /// tick). Two machines running the SAME `(checkpoint, seed, ticks)` must produce
    /// byte-identical files — `diff` them for the on-hardware cross-machine determinism gate.
    #[arg(long, value_name = "FILE")]
    hash_log: Option<PathBuf>,
}

#[derive(Parser)]
struct NnCrabXpeerArgs {
    /// Trained crab checkpoint dir (`brain.bin` + `normalizer.bin`). Same resolution as
    /// `nn-crab-probe` / `play --nn-crab-checkpoint`.
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<PathBuf>,
    /// Sim ticks to run both peers for before comparing.
    #[arg(long, default_value_t = 600)]
    ticks: u64,
    /// Shared match seed (identical on both peers, as it would be on the wire).
    #[arg(long, default_value_t = 0x6372_6162)]
    seed: u64,
    /// Write peer A's per-tick `<tick> <state_hash>` log here. With B's log, `diff` them: a
    /// byte-identical pair is the cross-peer determinism proof.
    #[arg(long, value_name = "FILE", default_value = "xpeer_a.log")]
    hash_log_a: PathBuf,
    /// Write peer B's per-tick `<tick> <state_hash>` log here.
    #[arg(long, value_name = "FILE", default_value = "xpeer_b.log")]
    hash_log_b: PathBuf,
}

#[derive(Parser)]
struct CheckpointCheckArgs {
    /// Checkpoint dir holding the `brain.bin` to rig-check against this binary. Required —
    /// the gate names exactly the checkpoint it's about to ship, no implicit default.
    #[arg(long, value_name = "DIR")]
    checkpoint: PathBuf,
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
        Command::NnCrabXpeer(args) => run_nn_crab_xpeer(args),
        Command::CheckpointCheck(args) => run_checkpoint_check(args),
    }
}

/// Rig-compatibility gate (see [`Command::CheckpointCheck`]): does the checkpoint at
/// `--checkpoint` fit THIS binary's crab rig? crab-world owns the verdict
/// ([`crab_world::play::checkpoint_fits_rig`]) and the rig spec
/// ([`crab_world::play::rig_dims`]), so the binary answers for itself — no hand-kept number
/// to drift; here we only turn the verdict into a message + exit code. Any non-`Ok` verdict
/// is an error (→ nonzero exit), which the release builder treats as "do not publish this
/// checkpoint".
fn run_checkpoint_check(args: CheckpointCheckArgs) -> Result<()> {
    use crab_world::play::{RigDims, RigFit};
    let RigDims {
        obs: rig_obs,
        action: rig_act,
    } = crab_world::play::rig_dims();
    let dir = args.checkpoint.display();
    match crab_world::play::checkpoint_fits_rig(&args.checkpoint) {
        RigFit::Ok => {
            println!("checkpoint-check OK: {dir} matches the rig ({rig_obs} obs, {rig_act} act)");
            Ok(())
        }
        RigFit::Missing => bail!(
            "checkpoint-check: no readable brain.bin in {dir} (this binary's rig is {rig_obs} \
             obs, {rig_act} act) — nothing to ship",
        ),
        RigFit::Mismatch(RigDims { obs, action }) => bail!(
            "checkpoint-check MISMATCH: {dir} is {obs} obs, {action} act but this binary's rig \
             is {rig_obs} obs, {rig_act} act — the NN crab would silently hold its rest pose. \
             Retrain/redeploy on the current rig (or rebuild the binary to match the checkpoint).",
        ),
    }
}

/// The cross-peer NN-crab determinism gate (GCR #82): run the real rapier-NN crab on two
/// independent in-process peers exchanging lockstep inputs, write each peer's per-tick hash log,
/// and confirm they stayed byte-identical. Exits nonzero on any divergence so it doubles as a CI
/// gate on "the NN crab is the deterministic multiplayer crab".
/// Write per-tick `<tick> <hash>` lines (zero-padded 16-hex) to a file — the cross-machine
/// determinism log two peers `diff` to prove byte-identical sims. One writer for every gate
/// (probe + xpeer) so the line format can never drift between them.
fn write_tick_hash_log(
    path: &std::path::Path,
    entries: impl Iterator<Item = (u64, u64)>,
) -> Result<()> {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (tick, hash) in entries {
        writeln!(out, "{} {:#018x}", tick, hash).unwrap();
    }
    std::fs::write(path, out).with_context(|| format!("writing hash log to {}", path.display()))
}

fn run_nn_crab_xpeer(args: NnCrabXpeerArgs) -> Result<()> {
    use net::external_crab::run_cross_peer_probe;

    let dir = nn_crab_checkpoint_dir(args.checkpoint)?;
    println!("nn-crab-xpeer: checkpoint={}", dir.display());
    println!("nn-crab-xpeer: seed={:#x} ticks={}", args.seed, args.ticks);

    let result = run_cross_peer_probe(&dir, args.seed, args.ticks);
    if result.ticks.is_empty() {
        anyhow::bail!("nn-crab-xpeer: no ticks applied — the peers never advanced");
    }

    // Per-tick `<tick> <hash>` logs for each peer, so an operator can `diff` them directly.
    write_tick_hash_log(&args.hash_log_a, result.ticks.iter().map(|t| (t.tick, t.hash_a)))?;
    write_tick_hash_log(&args.hash_log_b, result.ticks.iter().map(|t| (t.tick, t.hash_b)))?;
    println!(
        "nn-crab-xpeer: wrote {} per-tick hashes per peer to {} / {}",
        result.ticks.len(),
        args.hash_log_a.display(),
        args.hash_log_b.display(),
    );

    println!(
        "nn-crab-xpeer: lockstep desync faults = {} (the peers' own cross-check)",
        result.faults
    );
    match result.first_divergence() {
        None => {
            let last = result.ticks.last().unwrap();
            println!(
                "nn-crab-xpeer: per-tick hashes IDENTICAL across both peers \
                 (final tick {} hash {:#018x})",
                last.tick, last.hash_a
            );
        }
        Some(d) => {
            println!(
                "nn-crab-xpeer: FIRST DIVERGENCE at tick {} — A={:#018x} B={:#018x}",
                d.tick, d.hash_a, d.hash_b
            );
        }
    }

    if result.is_deterministic() {
        println!(
            "nn-crab-xpeer: PASS — the trained NN crab is the deterministic multiplayer crab \
             (outcome 3: bit-identical across peers, 0 desyncs)"
        );
        Ok(())
    } else {
        anyhow::bail!(
            "nn-crab-xpeer: FAIL — the float NN crab DIVERGED across peers (outcome 4: \
             netcode-rethink trigger; the diverging hash logs are the evidence)"
        )
    }
}

/// Headless NN-crab verification: step the real rapier crab for `--ticks` against a still
/// player, log its game position + shrinking distance to the player, and repeat the seed
/// to confirm the same trajectory hash twice (single-peer reproducibility). Prints a table
/// and a PASS/look-here verdict; exits nonzero if the crab never closed the gap (so it
/// doubles as a regression gate on "the policy actually drives the crab toward the player").
fn run_nn_crab_probe(args: NnCrabProbeArgs) -> Result<()> {
    use net::external_crab::run_headless_probe;

    let dir = nn_crab_checkpoint_dir(args.checkpoint)?;
    println!("nn-crab-probe: checkpoint={}", dir.display());
    println!("nn-crab-probe: seed={:#x} ticks={}", args.seed, args.ticks);

    // For the cross-machine hash log we need EVERY tick, not the skimmable sample stride.
    let log_every = if args.hash_log.is_some() { 1 } else { args.log_every };
    let samples = run_headless_probe(&dir, args.seed, args.ticks, log_every);
    if samples.is_empty() {
        anyhow::bail!("nn-crab-probe: no samples — the crab never stepped");
    }

    // Cross-machine determinism gate: a plain `<tick> <hash>` line per tick. Two Decks running
    // the same (checkpoint, seed, ticks) must yield byte-identical files (see [`hash_log`]).
    if let Some(path) = &args.hash_log {
        write_tick_hash_log(path, samples.iter().map(|s| (s.tick, s.state_hash)))?;
        println!(
            "nn-crab-probe: wrote {} per-tick hashes to {}",
            samples.len(),
            path.display()
        );
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
    let again = run_headless_probe(&dir, args.seed, args.ticks, log_every);
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
/// touching the deterministic sim before then (see [`net::render::Boot`]). The
/// scripted flags bypass the menu for tests/scripts:
/// - `--host` → host directly: form over the LAN, start with whoever joins (solo if none).
/// - `--join [CODE]` → join directly: dial CODE (or LAN-discover if bare), then form.
///
/// Both scripted paths form the match UP FRONT and hand a ready round to
/// [`render::Boot::Round`], so they boot straight into play with no menu. They reuse the
/// SAME barrier as the menu and as `game net`, so the agreed roster + seed are identical
/// however play was reached; a Host-alone fallback yields a solo round when nobody shows.
fn run_play(args: PlayArgs) -> Result<()> {
    // The single-thread pin that keeps the NN crab cross-peer deterministic is decided LATER, in
    // `build_windowed_app`, from whether the formed round actually has a remote peer: matchmaking
    // below resolves the roster first, so a solo round (no peer) can skip the pin and run
    // multi-threaded (~60fps) while a networked round still pins (GCR#113). The pin lands before
    // `App::new()` latches the task pools — see `render::build_windowed_app`.
    // The REQUIRED NN-crab checkpoint dir — the one giant crab IS the trained NN body (rl#114): no
    // integer fallback, so a missing brain is a hard, actionable failure here. Resolved BEFORE the
    // handshake so the scripted host/join path can advertise our REAL weights digest (two peers arm
    // the NN crab only when their brains agree — see `net::may_arm_external_crab`).
    let external_crab = nn_crab_checkpoint_dir(args.nn_crab_checkpoint)?;
    let weights_digest = crab_world::play::checkpoint_digest(&external_crab);
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
            // Advertise our REAL weights + crab-asset digests so two scripted peers carrying the
            // same checkpoint arm the NN crab; a mismatch refuses the round (rl#114, no fallback).
            weights_digest,
            crab_world::bot::meshfit::crab_asset_digest(),
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
    // Arming AND the cross-peer single-thread pin are both decided in `build_windowed_app`: a SOLO
    // round always arms the NN crab and runs multi-threaded (no peer to stay in sync with); a
    // NETWORKED round arms it once peers agree on weights+assets (the digest handshake above) and
    // pins every task pool to one thread so the float crab evolves bit-identically across peers; a
    // round that can't agree FAILS LOUD rather than substituting a fake crab.
    // A scripted networked round whose peers disagree on the brain+colliders can't arm Sally and
    // refuses (rl#114) — surfaced here as a clean error exit with the actionable fix (rl#115), not a
    // panic/abort. The interactive menu handles its own unarmable case in-client.
    let wire = resolve_wire_mode(args.debug_wireframe.as_deref())?;
    render::build_windowed_app(boot, external_crab, wire)?.run();
    Ok(())
}

/// Resolve the REQUIRED NN-crab checkpoint dir: the `--nn-crab-checkpoint` flag (`flag`), else
/// the `RL_CRAB_CHECKPOINT_DIR` env var (deploy sets this), else `assets/weights` under the
/// asset root (`BEVY_ASSET_ROOT`, else the binary's cwd) — so a checkpoint can be chosen at
/// runtime, no recompile. A missing `brain.bin` is a HARD, ACTIONABLE failure (rl#114): the one
/// giant crab IS the trained NN body ("Sally"), and there is no integer point-pursuer to fall back
/// to, so rather than silently substituting a fake crab we refuse to launch with a message naming
/// the dir we searched and how to fix it.
fn nn_crab_checkpoint_dir(flag: Option<PathBuf>) -> Result<PathBuf> {
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
        Ok(dir)
    } else {
        anyhow::bail!(
            "rl#114: no trained crab brain (brain.bin) under {} — the giant crab IS the trained NN \
             body (\"Sally\"), and there is no integer stand-in. Point --nn-crab-checkpoint or the \
             RL_CRAB_CHECKPOINT_DIR env var at a trained checkpoint dir (deploy/rl-update must set \
             it, and EVERY device needs the IDENTICAL brain + crab model), then relaunch.",
            dir.display()
        );
    }
}

/// Headless first-person screenshot: build a solo lockstep with `--players`
/// participants (so a remote avatar is in frame beside the local one), render one
/// settled frame of the FP view to a PNG, and exit. The evidence path for the
/// sim→render pipeline on a display-less box.
fn run_fp_screenshot(args: FpScreenshotArgs) -> Result<()> {
    let me = PlayerId(0);
    let players: Vec<PlayerId> = (0..args.players.max(1)).map(PlayerId).collect();
    // `--plane`: make the NON-local players pilots and keep the local (player 0) a ground
    // observer, so the captured FP frame clearly shows a remote plane's gray box flying — the
    // evidence the plane renders. Needs `--players >= 2`: with one player there's no remote
    // plane to frame, so the shot stays on foot (a lone pilot would show empty sky —
    // worthless as evidence). Off ⇒ the unchanged foot shot.
    let pilots: Vec<PlayerId> = if args.plane && players.len() > 1 {
        players[1..].to_vec()
    } else {
        Vec::new()
    };
    let ls = Lockstep::new_with_pilots(MATCH_SEED, &players, me, &pilots);
    let cfg = render::ScreenshotConfig::new(args.out, args.settle, args.width, args.height)
        .with_cam_offset(args.cam_yaw, args.cam_pitch)
        .with_fov(args.cam_fov);
    // Same checkpoint resolution as `play --nn-crab-checkpoint`: a readable brain.bin arms the real
    // crab for the shot (errors out if pointed at a dir with none). `None` keeps the silhouette.
    let external_crab = args
        .nn_crab_checkpoint
        .map(|flag| nn_crab_checkpoint_dir(Some(flag)))
        .transpose()?;
    // The armed NN crab steps real rapier physics; pin the global task pools to one thread BEFORE
    // the app builds (idempotent OnceLock) so this single settled frame is byte-reproducible run to
    // run — a screenshot is evidence, so a stable solver/inference order is worth more than its
    // (one-frame) speed. The windowed client pins ONLY for a networked round (see
    // `render::build_windowed_app`); this evidence path always pins when a crab is armed. Harmless
    // for the silhouette shot (no physics). See `render::pin_process_pools`.
    if external_crab.is_some() {
        render::pin_process_pools();
    }
    let wire = resolve_wire_mode(args.debug_wireframe.as_deref())?;
    render::build_screenshot_app(ls, cfg, external_crab, wire).run();
    Ok(())
}

/// Resolve the `--debug-wireframe` flag into a [`render::WireMode`]: an explicit value is
/// parsed (rejecting an unknown token with an actionable error), and an absent flag falls
/// back to the `RL_DEBUG_WIREFRAME` / `RL_DEBUG_COLLIDERS` env (off by default).
fn resolve_wire_mode(flag: Option<&str>) -> Result<render::WireMode> {
    match flag {
        Some(s) => render::WireMode::parse(s)
            .ok_or_else(|| anyhow::anyhow!("--debug-wireframe must be one of off|aligned|raw")),
        None => Ok(render::WireMode::from_env()),
    }
}

/// Deterministic match seed: a constant so independently-launched peers agree without a
/// handshake. The sim takes it as a parameter, so a future session setup can negotiate it
/// (the lower-id peer proposes, say) without touching the sim.
const MATCH_SEED: u64 = 0x6372_6162; // "crab"

/// Drive the lockstep sim from a constant local input, ticking at [`TICK_HZ`]. Pure machinery
/// check of the sim/lockstep loop: no peers, so our own input completes every tick. Headless, so
/// there is no rapier-NN crab stack — the giant crab simply holds its spawn (rl#114: no integer
/// pursuit), which is fine here since this exercises player/lockstep determinism, not the crab.
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
    // separate ALPN — see `net::telemetry`. A failure yields `None` and the run is
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
        0,    // headless has no rapier-NN crab stack → 0 weights digest; the crab holds spawn (rl#114)
        crab_world::bot::meshfit::crab_asset_digest(), // honest crab-asset digest (rl#100)
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

    // Use the wire-negotiated pilot set so every peer spawns the identical foot/plane mix
    // (empty ⇒ the unchanged foot-only round).
    let mut ls = Lockstep::new_with_pilots(MATCH_SEED, &all_ids, me, &frozen.pilots);

    // Server-coordinated play (rl#151): the lowest-id peer (PlayerId 0) runs the match server; the
    // rest are remote clients of it. Solo (a single peer) is the same path with a roster of one. The
    // Server core (the input ledger + completeness gating) is the SAME type the windowed client runs
    // — only the async-vs-sync transport plumbing differs (headless awaits the session directly; the
    // Bevy client drives it through `NetDriver`/`Coordinator`). Inputs flow UP as `TickMsg`s, the
    // server broadcasts the complete `TickSet` DOWN; world state never crosses the wire.
    let am_host = me == PlayerId(0);
    let server_eid = *id_map
        .iter()
        .find(|(_, &pid)| pid == PlayerId(0))
        .map(|(eid, _)| eid)
        .expect("a frozen roster always contains PlayerId(0)");
    let mut server = am_host.then(|| {
        let mut s = net::server::Server::new(&all_ids);
        // Seed the ledger with any inputs a fast client sent during formation (idempotent if it
        // re-sends them once play begins).
        for (from, msg) in &frozen.early {
            if let Some(&pid) = id_map.get(from) {
                let _ = s.record(pid, *msg);
            }
        }
        s
    });

    let tick_dt = Duration::from_secs_f64(1.0 / TICK_HZ as f64);
    let mut ticker = tokio::time::interval(tick_dt);
    let end = Instant::now() + Duration::from_secs(args.run_secs);
    let mut total_desyncs = 0usize;
    // Coarse human progress: print roughly once per second of sim. This samples the FIRST
    // tick at/after each boundary, which a batched `try_advance` can overshoot by a tick or
    // two — so these lines are a liveness/hash eyeball, NOT a byte-exact cross-peer compare.
    // The authoritative cross-peer determinism proofs are the internal desync cross-check
    // (peer-advertised hashes) and the per-tick `--hash-log` (keyed by the true tick).
    let mut next_report_tick = TICK_HZ;
    // Telemetry-side sampling cursor (independent of the stdout report cadence) and a
    // one-shot latch so RoundDecided is reported exactly once.
    let mut next_tel_tick = TELEMETRY_TICK_EVERY;
    let mut reported_outcome = false;
    // Optional per-tick hash log (NetArgs::hash_log): every applied tick keyed by its true
    // tick, so two peers' logs diff byte-identically over their overlap — the cross-peer
    // (and cross-machine) determinism proof. Written one line per `advance_one` below.
    let mut hash_log = args
        .hash_log
        .as_ref()
        .map(|p| {
            std::fs::File::create(p)
                .map(std::io::BufWriter::new)
                .with_context(|| format!("creating hash log {}", p.display()))
        })
        .transpose()?;

    while Instant::now() < end {
        ticker.tick().await;

        // Ingest everything the transport has for us this tick. As the server: clients' input
        // `TickMsg`s, fed into the ledger. As a client: the server's assembled `TickSet`s. A stray
        // barrier beat from a peer still winding down formation is ignored either way.
        let mut sets: Vec<net::server::TickSet> = Vec::new();
        while let Some(m) = session.try_recv() {
            match m.msg {
                transport::PeerWire::Tick(msg) => {
                    if let (Some(srv), Some(&pid)) = (server.as_mut(), id_map.get(&m.from)) {
                        sets.extend(srv.record(pid, msg));
                    }
                }
                transport::PeerWire::TickSet(set) => {
                    if !am_host {
                        sets.push(set);
                    }
                }
                transport::PeerWire::Beat(_) => {}
            }
        }

        // Issue our input for this tick and route it through the server.
        let t = ls.next_tick() as f32 * 0.1;
        let issue_tick = ls.next_tick();
        let input = Input::from_axes(t.cos(), t.sin());
        let msg = ls.submit_local_input(input);
        if let Some(srv) = server.as_mut() {
            sets.extend(srv.record(me, msg));
            for s in &sets {
                session.broadcast_tickset(s).await;
            }
        } else {
            session.send_to(server_eid, &msg).await;
        }

        // Record the OTHER players' inputs from the assembled sets — the same `record_remote` entry
        // a mesh peer used to take, so the cross-check + advance below are unchanged. A late hash for
        // an already-applied tick can surface a fault here.
        for s in &sets {
            for pm in net::server::unpack_tickset(s, me) {
                if let Some(f) = ls.record_remote(pm.pid, pm.msg) {
                    report_fault(&mut total_desyncs, f, tel.as_ref());
                }
            }
        }

        // Advance every ready tick ONE AT A TIME so the hash log can record each tick's
        // closing hash at the instant it's applied — `try_advance` is exactly this loop, but
        // logging from its post-batch snapshot could miss a tick the batch already pruned.
        // Logging per `advance_one` writes every applied tick exactly once, regardless of how
        // many a single iteration catches up.
        while let Some(faults) = ls.advance_one() {
            for f in faults {
                report_fault(&mut total_desyncs, f, tel.as_ref());
            }
            if let Some((w, c)) = hash_log.as_mut().zip(ls.last_applied()) {
                use std::io::Write as _;
                writeln!(w, "{} {:#018x}", c.tick, c.hash).context("writing hash log")?;
            }
        }

        // Coarse progress print once the sim crosses each TICK_HZ boundary (see the
        // cadence note above — a batched advance can overshoot the boundary tick, so these
        // are not byte-comparable across peers; the `--hash-log` is).
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
            if !reported_outcome && ls.sim().outcome() != net::sim::Outcome::Ongoing {
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
    if let Some(mut w) = hash_log.take() {
        use std::io::Write as _;
        w.flush().context("flushing hash log")?; // surface a write error, don't swallow it on drop
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
    f: net::lockstep::Fault,
    telemetry: Option<&TelemetrySender>,
) {
    use net::lockstep::Fault;
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
