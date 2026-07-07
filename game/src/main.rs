mod cmd;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(
    about = "Giant-crab rescue — Phase 1 gray-box extraction loop, host-authoritative over iroh"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<cmd::Command>,
}

fn main() -> Result<()> {
    let _otel = otel::init("game");
    let command = Cli::parse().command.unwrap_or_else(cmd::default_command);
    cmd::dispatch(command)
}
