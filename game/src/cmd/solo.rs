//! `solo`: single-peer headless smoke of the tick machinery (no network).

use anyhow::Result;
use clap::Parser;

use super::shared::run_solo_round;

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, default_value_t = 5)]
    run_secs: u64,
}

/// Drive the lockstep sim from a constant local input, ticking at the sim's `TICK_HZ`. Pure
/// machinery check of the sim/lockstep loop: no peers, so our own input completes every tick.
/// Headless, so there is no rapier-NN crab stack — the giant crab simply holds its spawn
/// (rl#114: no integer pursuit), which is fine here since this exercises player/lockstep
/// determinism, not the crab. Runs the SAME [`run_solo_round`] as the headless `net` no-peer
/// fallback, so the alone case can't drift between the two.
pub(crate) fn run(args: Args) -> Result<()> {
    run_solo_round(args.run_secs)
}
