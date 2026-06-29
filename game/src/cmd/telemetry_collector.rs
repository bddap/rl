//! `telemetry-collector`: the OTLP-over-iroh live-telemetry sink.

use anyhow::Result;
use clap::Parser;
use net::telemetry;

#[derive(Parser)]
pub(crate) struct Args {
    /// Path to the collector's persistent secret key (generated on first run). Pinning
    /// it keeps the collector's endpoint id STABLE across restarts, so the id baked into
    /// each game's `--telemetry` never goes stale.
    #[arg(long, default_value = net::telemetry::DEFAULT_KEY_PATH)]
    key: std::path::PathBuf,
}

/// Bind a stable-id iroh endpoint and stream every connected game's events as a merged human
/// feed. Builds its own tokio runtime (the collector is purely async, and `main` stays a plain
/// `fn` — see the runtime note there).
pub(crate) fn run(args: Args) -> Result<()> {
    tokio::runtime::Runtime::new()?.block_on(telemetry::run_collector(&args.key))
}
