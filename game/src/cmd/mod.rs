use anyhow::Result;
use clap::{Parser, Subcommand};

mod fp_screenshot;
mod net_screenshot;
mod play;
mod shared;

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Windowed first-person client: see + play the gray-box.
    Play(play::Args),
    /// Offscreen first-person screenshot of a local round — the evidence shot.
    FpScreenshot(fp_screenshot::Args),
    /// Offscreen screenshot from inside a live two-peer match (the remote-client render).
    NetScreenshot(net_screenshot::Args),
}

pub(crate) fn default_command() -> Command {
    Command::Play(play::Args::parse_from(["game"]))
}

pub(crate) fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Play(args) => play::run(args),
        Command::FpScreenshot(args) => fp_screenshot::run(args),
        Command::NetScreenshot(args) => net_screenshot::run(args),
    }
}
