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

#[cfg(test)]
mod tests {
    use super::*;

    /// clap validates a command (duplicate longs, a bad `global`, an action/value-parser
    /// mismatch) only when that command is BUILT — and subcommands build lazily, so a
    /// misconfiguration would otherwise surface as a panic for whoever first ran that one
    /// subcommand. This builds the whole tree, including every flattened Args struct.
    #[test]
    fn cli_is_well_formed() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
