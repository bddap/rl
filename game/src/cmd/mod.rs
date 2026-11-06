use anyhow::Result;
use clap::{Parser, Subcommand};

mod fp_screenshot;
mod net;
mod net_screenshot;
mod play;
mod shared;
mod solo;

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Networked headless run: discover LAN peers over iroh and run the host-authoritative loop.
    Net(net::Args),
    /// Headless single-peer round: run the sim for a few seconds and print its state hash.
    Solo(solo::Args),
    /// Windowed first-person client: see + play the gray-box.
    Play(play::Args),
    /// Offscreen first-person screenshot of a local round — the evidence shot.
    FpScreenshot(fp_screenshot::Args),
    /// Offscreen screenshot from inside a live two-peer match (the remote-client render).
    NetScreenshot(net_screenshot::Args),
}

pub(crate) fn default_command() -> Command {
    Command::Net(net::Args::parse_from(["game"]))
}

pub(crate) fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Net(args) => net::run(args),
        Command::Solo(args) => solo::run(args),
        Command::Play(args) => play::run(args),
        Command::FpScreenshot(args) => fp_screenshot::run(args),
        Command::NetScreenshot(args) => net_screenshot::run(args),
    }
}
