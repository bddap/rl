//! The subcommand dispatch table. Each subcommand owns a module here (its `Args` struct + a
//! `run` fn); cross-cutting helpers live in [`shared`]. `main` parses [`crate::Cli`] and hands
//! the chosen [`Command`] to [`dispatch`], so the binary's entry point stays a thin router and
//! every subcommand's plumbing is one focused file.

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
    /// Single-peer headless smoke of the tick machinery (no network).
    Solo(solo::Args),
    /// Windowed first-person client: see + play the gray-box.
    Play(play::Args),
    /// Windowed first-person client that JOINS a live match mid-game (GCR MP Stage 3): dial the
    /// host's endpoint id, get admitted at an agreed round-boundary tick over the new roster, and
    /// boot straight into the joiner's round as a normal networked Client. A digest mismatch is
    /// refused loudly (no silent wrong-crab fallback). The dialing analogue of `play --host`.
    NetJoin(net_join::Args),
    /// Render one frame of the first-person view to a PNG and exit (no window).
    FpScreenshot(fp_screenshot::Args),
    /// Form a real networked round over iroh and render one OFFSCREEN frame of this peer's view to
    /// a PNG (rl#151 increment 2 windowed): the headless analogue of `play --host`/`--join`. Run
    /// two — the remote CLIENT's captured frame is the evidence it renders the host's authoritative
    /// state + articulated crab through the snapshot path (no re-sim).
    NetScreenshot(net_screenshot::Args),
    /// Run the live-telemetry collector: bind a stable-id iroh endpoint, print its id,
    /// and stream every connected game's events as a merged human feed (for remote
    /// debugging — `Monitor` its stdout). Pass each game `--telemetry <printed id>`.
    TelemetryCollector(telemetry_collector::Args),
    /// Headless verification of the SOLO NN crab (no window / GPU / display): step the
    /// real rapier-NN crab for N ticks against a still player and log the crab's game
    /// position + its shrinking distance to the player — the evidence it WALKS toward the
    /// player under the trained policy. Runs the seed TWICE and compares the final state
    /// hash, the single-peer reproducibility check.
    NnCrabProbe(nn_crab_probe::Args),
    /// Rig-compatibility gate for the release/deploy pipeline: load a checkpoint's
    /// `brain.bin`, read its `(obs, action)` dims, and compare them to THIS binary's
    /// compiled crab rig. Exits 0 only on an exact match; nonzero (with an actionable
    /// message) if the brain is missing/unreadable or its dims differ. A mismatched
    /// checkpoint loads "fine" at runtime but silently degrades the NN crab to its rest
    /// pose, so the release builder runs this against the checkpoint it's about to bundle
    /// and refuses to publish on a nonzero exit — the mismatch is loud, not silent.
    CheckpointCheck(checkpoint_check::Args),
    /// Crab-policy-stability gate for the SP-vehicle→rapier migration (no window / GPU): run the
    /// trained NN crab, drop a real vehicle rigidbody onto it mid-walk (the same collider/mass/
    /// groups boarding spawns), and keep stepping. Verifies the headline (owner 703) — the
    /// vehicle↔crab collision is real and the trained walking RECOVERS (no NaN/explosion; the crab
    /// stands back up and keeps reaching). Exits nonzero if the crab explodes or fails to recover,
    /// so it gates the migration.
    NnCrabVehicleStability(nn_crab_vehicle_stability::Args),
}

/// The default when no subcommand is given: the networked mode with its own defaults (parsed
/// from an empty arg list so the `default_value_t`s stay the single source, not duplicated here).
pub(crate) fn default_command() -> Command {
    Command::Net(net::Args::parse_from(["game"]))
}

/// Route a parsed [`Command`] to its subcommand module. The async modes (`net`,
/// `telemetry-collector`) build their own tokio runtime inside their `run` — see the runtime
/// note in `main`.
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
