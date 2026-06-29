//! `net-join`: dial INTO a live match as a mid-game joiner (GCR MP Stage 3, rl#151).
//!
//! The dialing analogue of `play --host`: instead of forming a fresh match, it calls
//! [`net_loop::connect_and_join`] to enter an in-progress round over the round-boundary join,
//! then boots the windowed client straight into the joiner's [`Lockstep`] (built via
//! `join_at`) + its client [`NetDriver`] — the SAME `Boot::Round` path the scripted
//! `--host`/`--join` formation uses, so the joiner is just a normal Client with no SP/MP fork.

use anyhow::{Result, bail};
use clap::Parser;
use iroh::EndpointId;
use net::{net_loop, render};

use super::shared::{MATCH_SEED, nn_crab_checkpoint_dir, resolve_render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    /// The host's endpoint-id code (printed by the host's `game endpoint id: …` line) to dial
    /// into. We send our weight/collider digests as a join request; the host admits us at an
    /// agreed future tick over the new roster, or refuses loudly on a digest mismatch.
    #[arg(value_name = "HOST_ENDPOINT_ID")]
    host: EndpointId,
    /// Stream live telemetry to this collector endpoint id (separate ALPN/connection — never
    /// perturbs the lockstep; see `play --telemetry`).
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,
    /// Directory holding the trained crab policy (`brain.bin` + `normalizer.bin`) — REQUIRED, as
    /// for `play`: the joiner advertises its weights+asset digests so the host's admission gate
    /// can verify it runs the SAME Sally (a mismatch is refused, never a silent wrong crab).
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<std::path::PathBuf>,
    /// Start the crab render view in this mode (default: mesh). Same as `play --render-mode`.
    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
}

/// Dial the host, join the live round over the round-boundary mechanism, and boot the windowed
/// client into the joiner's session. The joiner resolves its REQUIRED checkpoint UP FRONT (so it
/// advertises its real digests on the join request), then on `Joined` hands the pre-built
/// `(Lockstep, NetDriver)` to [`render::Boot::Round`] — the joiner becomes a normal networked
/// Client. A `Refused` (digest mismatch the host turned away) or `Unreachable` host is a clean,
/// loud error exit — never a silent fallback to a fake/solo crab.
pub(crate) fn run(args: Args) -> Result<()> {
    let external_crab = nn_crab_checkpoint_dir(args.nn_crab_checkpoint)?;
    let weights_digest = crab_world::play::checkpoint_digest(&external_crab);
    let asset_digest = crab_world::bot::meshfit::crab_asset_digest();

    let result = net_loop::connect_and_join(
        MATCH_SEED,
        args.host,
        args.telemetry,
        weights_digest,
        asset_digest,
    )?;

    let boot = match result {
        net_loop::JoinResult::Joined(joined) => {
            let (ls, driver) = *joined;
            render::Boot::Round(Box::new((ls, Some(driver))))
        }
        net_loop::JoinResult::Refused(reason) => {
            bail!(
                "host refused our join: {reason}. We are running a different brain or Sally than \
                 the host — run rl-update on this device so every peer carries the same checkpoint."
            )
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
