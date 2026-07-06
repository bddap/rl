
use anyhow::Result;
use clap::{Parser, Subcommand};

mod checkpoint_check;
mod fp_screenshot;
mod net;
mod net_join;
mod net_screenshot;
mod nn_crab_probe;
mod nn_crab_vehicle_stability;
mod play;
mod shared;
mod solo;
mod telemetry_collector;

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Networked headless run: discover LAN peers over iroh and run the host-authoritative loop.
    Net(net::Args),
    Solo(solo::Args),
    /// Windowed first-person client: see + play the gray-box.
    Play(play::Args),
    NetJoin(net_join::Args),
    FpScreenshot(fp_screenshot::Args),
    NetScreenshot(net_screenshot::Args),
    TelemetryCollector(telemetry_collector::Args),
    NnCrabProbe(nn_crab_probe::Args),
    CheckpointCheck(checkpoint_check::Args),
    NnCrabVehicleStability(nn_crab_vehicle_stability::Args),
}

pub(crate) fn default_command() -> Command {
    Command::Net(net::Args::parse_from(["game"]))
}

pub(crate) fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Net(args) => net::run(args),
        Command::Solo(args) => solo::run(args),
        Command::Play(args) => play::run(args),
        Command::NetJoin(args) => net_join::run(args),
        Command::FpScreenshot(args) => fp_screenshot::run(args),
        Command::NetScreenshot(args) => net_screenshot::run(args),
        Command::TelemetryCollector(args) => telemetry_collector::run(args),
        Command::NnCrabProbe(args) => nn_crab_probe::run(args),
        Command::CheckpointCheck(args) => checkpoint_check::run(args),
        Command::NnCrabVehicleStability(args) => nn_crab_vehicle_stability::run(args),
    }
}
