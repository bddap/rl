use anyhow::Result;
use clap::Parser;

use super::shared::run_solo_round;

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, default_value_t = 5)]
    run_secs: u64,
}

pub(crate) fn run(args: Args) -> Result<()> {
    run_solo_round(args.run_secs)
}
