mod cmd;

use anyhow::Result;
use clap::Parser;

/// Giant-crab rescue — Phase 1 gray-box extraction loop, host-authoritative over iroh.
#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    otel: otel::OtelArgs,

    #[command(subcommand)]
    command: Option<cmd::Command>,
}

fn main() -> Result<()> {
    // Parse BEFORE arming telemetry: the args decide whether to export at all, and a
    // `--help` or a bad flag should exit without spinning an exporter up first.
    let cli = Cli::parse();
    let _otel = otel::init("game", cli.otel);
    cmd::dispatch(cli.command.unwrap_or_else(cmd::default_command))
}
