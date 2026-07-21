use anyhow::Result;
use clap::Parser;
use iroh::EndpointId;
use net::{formation, net_loop, render};

use crab_world::RenderArgs;

use super::shared::{nn_crab_policies, parse_join_dial, render_mode};

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

    #[arg(long, value_name = "DIR", env = super::shared::CHECKPOINT_ENV)]
    nn_crab_checkpoint: Vec<std::path::PathBuf>,

    #[command(flatten)]
    render: RenderArgs,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let nn_crabs = nn_crab_policies(args.nn_crab_checkpoint)?;
    // Per-launch entropy (rl#305): the run layout derives from this seed, so real play
    // opens somewhere fresh every launch (and every in-round RESTART re-draws); the
    // authoritative sim logs the seed for repro. Screenshot/probe tools keep the pinned
    // [`super::shared::MATCH_SEED`] instead.
    let seed = net::sim::random_match_seed();
    let boot = if args.host || args.join.is_some() {
        let dial = parse_join_dial(args.join.as_deref())?;
        let result = net_loop::connect_and_form_dialing(
            seed,
            args.discover_secs,
            args.expect,
            dial,
            args.telemetry,
            None,
            net::SyncStamp::local(nn_crabs.len() as u8),
        )?;
        match result {
            net_loop::MatchResult::Joined(joined) => {
                let (client, driver) = *joined;
                render::Boot::Round(Box::new((client, Some(driver))))
            }
            net_loop::MatchResult::Alone => {
                render::Boot::Round(Box::new((formation::solo_client_for(seed), None)))
            }
            net_loop::MatchResult::Cancelled => {
                unreachable!("scripted --host/--join has no lobby to cancel")
            }
        }
    } else {
        render::Boot::Menu {
            seed,
            telemetry: args.telemetry,
        }
    };
    let render_mode = render_mode(args.render);
    render::build_windowed_app(boot, nn_crabs, render_mode)?.run();
    Ok(())
}
