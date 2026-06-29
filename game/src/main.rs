//! `game` — Giant Crab Rescue (rl#38): the CLI over the deterministic-lockstep + iroh
//! sim ([`net::sim`]/[`net::lockstep`]/[`net::transport`]). First-person players reach
//! an extraction point while a trained-NN giant crab (Sally) hunts them; `solo` is just
//! the zero-remote-peer case of the one networked path, not a separate mode.
//!
//! Each subcommand lives in its own module under [`cmd`] (its `Args` struct + a `run` fn);
//! this file is only the entry point — parse [`Cli`], pick the [`cmd::Command`], dispatch:
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

mod cmd;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(
    about = "Giant-crab rescue — Phase 1 gray-box extraction loop on deterministic lockstep + iroh"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<cmd::Command>,
}

// Plain `main` (not `#[tokio::main]`): the windowed/screenshot client builds a Bevy
// app that owns the main thread and, for networked play, spins up its OWN tokio
// runtime inside `net_loop` — nesting that under an ambient `#[tokio::main]` runtime
// panics ("cannot start a runtime from within a runtime"). So each async mode (`net`,
// `telemetry-collector`) builds its runtime explicitly inside its `cmd::*::run`, and the
// sync modes (`solo`/`play`/`fp-screenshot`) never touch one they don't own.
fn main() -> Result<()> {
    // Installs the stderr fmt subscriber (so the game's `error!`/`warn!` surface locally) AND,
    // when a telemetry endpoint is configured, exports OTLP traces/logs/metrics — routing the
    // missing-mesh error (rl#706) and other faults onto the telemetry stream. Inert (stderr
    // only) when no endpoint is set, so it never perturbs lockstep. The guard flushes on drop,
    // so it must outlive the whole run — bound here, dropped only when `main` returns. `RUST_LOG`
    // overrides the default `info` level.
    let _otel = otel::init("game");
    // No subcommand → the networked mode with its own defaults (see `cmd::default_command`).
    let command = Cli::parse().command.unwrap_or_else(cmd::default_command);
    cmd::dispatch(command)
}
