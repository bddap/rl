use anyhow::{Result, bail};
use clap::Parser;
use iroh::EndpointId;
use net::{net_loop, render};

use super::shared::{MATCH_SEED, nn_crab_policies, resolve_render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(value_name = "HOST_ENDPOINT_ID")]
    host: EndpointId,
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Vec<std::path::PathBuf>,
    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let external_crab = nn_crab_policies(args.nn_crab_checkpoint)?;
    let asset_digest = crab_world::mesh_fallback::constructed_body_digest();

    let result = net_loop::connect_and_join(
        MATCH_SEED,
        args.host,
        args.telemetry,
        asset_digest,
        external_crab.len() as u8,
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

    let render_mode = resolve_render_mode(args.render_mode.as_deref())?;
    render::build_windowed_app(boot, external_crab, render_mode)?.run();
    Ok(())
}
