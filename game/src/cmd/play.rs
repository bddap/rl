use anyhow::Result;
use clap::Parser;
use iroh::EndpointId;
use net::{formation, net_loop, render};

use super::shared::{MATCH_SEED, nn_crab_policies, parse_join_dial, resolve_render_mode};

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

    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Vec<std::path::PathBuf>,

    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let external_crab = nn_crab_policies(args.nn_crab_checkpoint)?;
    let boot = if args.host || args.join.is_some() {
        let dial = parse_join_dial(args.join.as_deref())?;
        let result = net_loop::connect_and_form_dialing(
            MATCH_SEED,
            args.discover_secs,
            args.expect,
            dial,
            args.telemetry,
            None,
            crab_world::mesh_fallback::constructed_body_digest(),
            external_crab.len() as u8,
        )?;
        match result {
            net_loop::MatchResult::Joined(joined) => {
                let (ls, driver) = *joined;
                render::Boot::Round(Box::new((ls, Some(driver))))
            }
            net_loop::MatchResult::Alone => {
                render::Boot::Round(Box::new((formation::solo_lockstep_for(MATCH_SEED), None)))
            }
            net_loop::MatchResult::Cancelled => {
                unreachable!("scripted --host/--join has no lobby to cancel")
            }
        }
    } else {
        render::Boot::Menu {
            seed: MATCH_SEED,
            telemetry: args.telemetry,
        }
    };
    let render_mode = resolve_render_mode(args.render_mode.as_deref())?;
    render::build_windowed_app(boot, external_crab, render_mode)?.run();
    Ok(())
}
