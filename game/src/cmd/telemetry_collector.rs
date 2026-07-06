
use anyhow::Result;
use clap::Parser;
use net::telemetry;

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, default_value = net::telemetry::DEFAULT_KEY_PATH)]
    key: std::path::PathBuf,
}

pub(crate) fn run(args: Args) -> Result<()> {
    tokio::runtime::Runtime::new()?.block_on(telemetry::run_collector(&args.key))
}
