//! `net-join`: dial INTO a live match as a host-authoritative mid-game joiner (GCR MP incr 4, rl#151).
//!
//! The dialing analogue of `play --host`: instead of forming a fresh match, it calls
//! [`net_loop::connect_and_join`] to enter an in-progress round, then boots the windowed client
//! straight into the joiner's placeholder [`Lockstep`] + its client [`NetDriver`] — the SAME
//! `Boot::Round` path the scripted `--host`/`--join` formation uses, so the joiner is just a normal
//! remote-adopt Client with no SP/MP fork. The host spawns the joiner into its LIVE authoritative
//! round, and the joiner boots from the host's next snapshot (drops into the ongoing match at its
//! live tick + crab pose), rather than every peer resetting to a fresh round.

use anyhow::{Result, bail};
use clap::Parser;
use iroh::EndpointId;
use net::{net_loop, render};

use super::shared::{MATCH_SEED, nn_crab_checkpoint_dir, resolve_render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    /// The host's endpoint-id code (printed by the host's `game endpoint id: …` line) to dial
    /// into. We send our collider digest as a join request; the host admits us at an
    /// agreed future tick over the new roster, or refuses loudly on a collider mismatch or an
    /// unarmed host.
    #[arg(value_name = "HOST_ENDPOINT_ID")]
    host: EndpointId,
    /// Stream live telemetry to this collector endpoint id (separate ALPN/connection — never
    /// perturbs the lockstep; see `play --telemetry`).
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,
    /// Directory holding the trained crab policy (`brain.bin` + `normalizer.bin`) — REQUIRED, as
    /// for `play`. Like a cold-start formation client (rl#199), the joiner never executes the
    /// brain — the host's self-gate is the one weights guard (rl#206) — so only the crab-ASSET
    /// digest is gated (a mismatch is refused, never a silent wrong crab).
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Option<std::path::PathBuf>,
    /// Start the crab render view in this mode (default: mesh). Same as `play --render-mode`.
    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
}

/// Dial the host, join the live round via host-authoritative snapshot adoption, and boot the
/// windowed client into the joiner's session. On `Joined` it hands the pre-built
/// `(Lockstep, NetDriver)` to [`render::Boot::Round`] — the joiner becomes a normal remote-adopt
/// Client that renders the host's snapshots; like a formation client, its own brain digest is
/// ungated (rl#206 — the host's self-gate guards the one brain that runs). A `Refused` (collider
/// mismatch OR a zero-digest host the gate turned away) or `Unreachable` host is a clean, loud
/// error exit — never a silent fallback to a fake/solo crab.
pub(crate) fn run(args: Args) -> Result<()> {
    // `nn_crab_checkpoint_dir` is the joiner's real pre-dial guard: it hard-fails on a
    // missing/refused/rig-mismatched checkpoint, and the join gate needs no weights digest from
    // us beyond that (rl#206) — only the crab-asset digest crosses the wire.
    let external_crab = nn_crab_checkpoint_dir(args.nn_crab_checkpoint)?;
    let asset_digest = crab_world::bot::meshfit::crab_asset_digest();

    let result = net_loop::connect_and_join(MATCH_SEED, args.host, args.telemetry, asset_digest)?;

    let boot = match result {
        net_loop::JoinResult::Joined(joined) => {
            let (ls, driver) = *joined;
            render::Boot::Round(Box::new((ls, Some(driver))))
        }
        net_loop::JoinResult::Refused(reason) => {
            bail!(
                "host refused our join: {reason}. Run rl-update so every device carries the same \
                 build and assets, then re-join."
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
