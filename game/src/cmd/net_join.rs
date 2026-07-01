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

/// Dial the host, join the live round via host-authoritative snapshot adoption, and boot the
/// windowed client into the joiner's session. The joiner resolves its REQUIRED checkpoint UP FRONT
/// and refuses to dial with a fake (zero-digest) crab, then on `Joined` hands the pre-built
/// `(Lockstep, NetDriver)` to [`render::Boot::Round`] — the joiner becomes a normal remote-adopt
/// Client that renders the host's snapshots. A `Refused` (digest mismatch OR a zero-digest host the
/// gate turned away) or `Unreachable` host is a clean, loud error exit — never a silent fallback to
/// a fake/solo crab.
pub(crate) fn run(args: Args) -> Result<()> {
    let external_crab = nn_crab_checkpoint_dir(args.nn_crab_checkpoint)?;
    let weights_digest = crab_world::play::checkpoint_digest(&external_crab);
    let asset_digest = crab_world::bot::meshfit::crab_asset_digest();

    // Joiner host-verify, our half (rl#151 incr 4): our own checkpoint must load to a REAL Sally
    // before we dial. With a zero digest we run the fake rest-pose crab, can't verify the host runs
    // the real one, and the host's self-gate would refuse us anyway — so fail fast and loud here
    // rather than after a QUIC round-trip. (An armed match only admits us when the host's digest
    // equals ours and is non-zero, so a successful join proves BOTH peers run the same real Sally.)
    if weights_digest == 0 {
        bail!(
            "our trained NN crab (\"Sally\") checkpoint at {} failed to load (weights digest 0) — \
             refusing to join a match with a fake crab. Run rl-update on this device so it carries \
             the real brain, then re-join.",
            external_crab.display()
        );
    }

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
