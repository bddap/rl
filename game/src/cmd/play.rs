use anyhow::Result;
use clap::Parser;
use iroh::EndpointId;
use net::render::{GameConfig, Launch, run_game};

use crab_world::RenderArgs;

use super::shared::parse_join_dial;

/// The native argv→[`GameConfig`] adapter — the game body itself is
/// [`net::render::run_game`], shared with the web adapter (rl#411).
#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, conflicts_with = "join")]
    host: bool,
    #[arg(
        long,
        value_name = "JOIN_CODE",
        num_args = 0..=1,
        default_missing_value = "",
        conflicts_with = "host"
    )]
    join: Option<String>,
    #[arg(long, default_value_t = super::shared::DEFAULT_DISCOVER_SECS)]
    discover_secs: u64,
    #[arg(long, default_value_t = super::shared::DEFAULT_EXPECT)]
    expect: usize,
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,

    #[arg(long, value_name = "DIR", env = net::render::CHECKPOINT_ENV, value_parser = crab_world::absolute_dir)]
    nn_crab_checkpoint: Vec<std::path::PathBuf>,

    #[command(flatten)]
    render: RenderArgs,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let launch = if args.host || args.join.is_some() {
        Launch::Lobby {
            dial: parse_join_dial(args.join.as_deref())?,
            discover_secs: args.discover_secs,
            expect: args.expect,
        }
    } else {
        Launch::Menu
    };
    run_game(GameConfig {
        launch,
        telemetry: args.telemetry,
        nn_crab_checkpoints: args.nn_crab_checkpoint,
        view: args.render,
        asset_root: crab_world::assets::native_asset_root(),
    })
}
