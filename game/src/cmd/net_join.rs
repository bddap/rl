use anyhow::{Result, bail};
use clap::Parser;
use iroh::EndpointId;
use net::{net_loop, render};

use crab_world::RenderArgs;

use super::shared::{nn_crab_policies, render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(value_name = "HOST_ENDPOINT_ID")]
    host: EndpointId,
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,
    #[arg(long, value_name = "DIR", env = super::shared::CHECKPOINT_ENV)]
    nn_crab_checkpoint: Vec<std::path::PathBuf>,
    #[command(flatten)]
    render: RenderArgs,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let nn_crabs = nn_crab_policies(args.nn_crab_checkpoint)?;
    // The joiner's seed only shapes its pre-adopt placeholder sim — the host's
    // snapshot (extraction included, rl#305) supersedes it.
    let result = net_loop::connect_and_join(
        net::sim::random_match_seed(),
        args.host,
        args.telemetry,
        net::SyncStamp::local(nn_crabs.len() as u8),
    )?;

    let boot = match result {
        net_loop::JoinResult::Joined(joined) => {
            let (client, driver) = *joined;
            render::Boot::Round(Box::new((client, Some(driver))))
        }
        net_loop::JoinResult::Refused(reason) => {
            // rl-update only remedies build/asset mismatches — don't prescribe it for the
            // transient (Forming) or connection-loss (Departed) refusals.
            let advice = match reason {
                net::server::Refusal::Admission(_) => {
                    " Run rl-update so every device carries the same build and assets, then \
                     re-join."
                }
                net::server::Refusal::Departed | net::server::Refusal::Forming => "",
            };
            bail!("host refused our join: {reason}.{advice}")
        }
        net_loop::JoinResult::Unreachable => {
            bail!(
                "host {} was unreachable or not running a joinable match (no admission verdict \
                 within the timeout)",
                args.host.fmt_short(),
            )
        }
    };

    let render_mode = render_mode(args.render);
    render::build_windowed_app(boot, nn_crabs, render_mode)?.run();
    Ok(())
}
