//! `play`: the windowed first-person client (menu, or scripted `--host`/`--join`).

use anyhow::Result;
use clap::Parser;
use iroh::EndpointId;
use net::{formation, net_loop, render};

use super::shared::{MATCH_SEED, nn_crab_checkpoint_dirs, parse_join_dial, resolve_render_mode};

#[derive(Parser)]
pub(crate) struct Args {
    /// Skip the menu and HOST a networked match directly (scripted/test entry): form over
    /// the LAN and start with whoever joins, or solo if nobody does. Equivalent to the
    /// menu's Host without the click. A Host that finds no peer IS the solo round — one
    /// codepath, no separate solo flag; bare `play` shows the menu instead.
    #[arg(long, conflicts_with = "join")]
    host: bool,
    /// Skip the menu and JOIN a host by its endpoint-id code (scripted/test entry): dial the
    /// code, then form. Equivalent to the menu's Join-by-code without the click. The bare
    /// flag `--join` with no value joins by LAN discovery (no explicit dial).
    #[arg(
        long,
        value_name = "JOIN_CODE",
        num_args = 0..=1,
        default_missing_value = "",
        conflicts_with = "host"
    )]
    join: Option<String>,
    /// Wait this long for peers before starting (the scripted `--host`/`--join` paths only).
    #[arg(long, default_value_t = super::shared::DEFAULT_DISCOVER_SECS)]
    discover_secs: u64,
    /// Expected peer count including us (the scripted `--host`/`--join` paths only); proceeds
    /// with whoever showed up after `discover_secs`.
    #[arg(long, default_value_t = super::shared::DEFAULT_EXPECT)]
    expect: usize,
    /// Stream live telemetry to this collector endpoint id (networked play only; see
    /// `NetArgs::telemetry`). Separate ALPN/connection — never perturbs the match.
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,

    /// Directory holding a trained crab policy (`brain.bin` + `normalizer.bin`) — REQUIRED: each
    /// giant crab IS a rapier-simulated NN body (rl#114, no integer fallback). REPEATABLE
    /// (rl#200): each occurrence binds one crab, in order, so a multi-architecture round runs
    /// one crab per checkpoint. The brains drive the crabs on a SOLO round (a Host-alone Start,
    /// or a scripted `--host` that found no peer) AND on a NETWORKED round once peers agree on
    /// the collider asset ([`net::may_arm_external_crab`]): the host then runs the
    /// authoritative crabs and clients render their broadcast poses. A missing/empty/unfitting
    /// dir, or a networked round whose peers DON'T agree, FAILS LOUD with an actionable message
    /// rather than substituting a fake crab. No flag defaults to ONE binding from the
    /// `RL_CRAB_CHECKPOINT_DIR` env var, else `assets/weights` under the asset root.
    #[arg(long, value_name = "DIR")]
    nn_crab_checkpoint: Vec<std::path::PathBuf>,

    /// Start the crab render view in this mode (default: mesh, the player-facing showcase).
    /// `mesh+colliders` overlays the honest collider wireframe on the mesh; `colliders` shows
    /// the wireframe alone. The `CycleRenderMode` control (V / pad B) cycles it live. Unset falls
    /// back to the `RL_RENDER_MODE` / `RL_DEBUG_COLLIDERS` env (mesh otherwise).
    #[arg(long, value_name = "mesh|mesh+colliders|colliders")]
    render_mode: Option<String>,
}

/// Windowed first-person client. The DEFAULT (`game play` with no flag) shows the boot menu
/// — the player picks Host / Join — and the round is built only after the choice, never
/// touching the deterministic sim before then (see [`net::render::Boot`]). The
/// scripted flags bypass the menu for tests/scripts:
/// - `--host` → host directly: form over the LAN, start with whoever joins (solo if none).
/// - `--join [CODE]` → join directly: dial CODE (or LAN-discover if bare), then form.
///
/// Both scripted paths form the match UP FRONT and hand a ready round to
/// [`render::Boot::Round`], so they boot straight into play with no menu. They reuse the
/// SAME barrier as the menu and as `game net`, so the agreed roster + seed are identical
/// however play was reached; a Host-alone fallback yields a solo round when nobody shows.
pub(crate) fn run(args: Args) -> Result<()> {
    // The REQUIRED brain bindings — each giant crab IS a trained NN body (rl#114): no integer
    // fallback, so a missing/unfitting brain is a hard, actionable failure here, per binding
    // (the rl#200 host fail-loud validation).
    let external_crab = nn_crab_checkpoint_dirs(args.nn_crab_checkpoint)?;
    let boot = if args.host || args.join.is_some() {
        // Scripted host/join: dial the join code if joining (blank/absent = LAN discover),
        // form over the barrier, and hand the result to Boot::Round. Host never dials. This
        // is the default timer-closed barrier (no interactive lobby), so it can't be
        // cancelled — only Joined or the Alone solo fallback.
        let dial = parse_join_dial(args.join.as_deref())?;
        let result = net_loop::connect_and_form_dialing(
            MATCH_SEED,
            args.discover_secs,
            args.expect,
            dial,
            args.telemetry,
            None,
            // Advertise our REAL crab-asset digest + binding count so peers agree on the
            // collider asset and the crab count; a mismatch refuses the round (rl#114/rl#200,
            // no fallback).
            crab_world::mesh_fallback::constructed_body_digest(),
            external_crab.len() as u8,
        )?;
        match result {
            net_loop::MatchResult::Joined(joined) => {
                let (ls, driver) = *joined;
                render::Boot::Round(Box::new((ls, Some(driver))))
            }
            // Nobody showed: play the shared solo round (the Host-alone outcome).
            net_loop::MatchResult::Alone => {
                render::Boot::Round(Box::new((formation::solo_lockstep_for(MATCH_SEED), None)))
            }
            // The scripted path runs no interactive lobby, so a Cancel is impossible.
            net_loop::MatchResult::Cancelled => {
                unreachable!("scripted --host/--join has no lobby to cancel")
            }
        }
    } else {
        // Interactive default: the boot menu builds the round after the player chooses.
        render::Boot::Menu {
            seed: MATCH_SEED,
            telemetry: args.telemetry,
        }
    };
    // Arming is decided in `build_windowed_app`: a SOLO round always arms the NN crabs; a NETWORKED
    // round arms them once peers agree on the crab-model asset (the digest handshake above); a
    // round that can't agree FAILS LOUD rather than substituting a fake crab.
    // A scripted networked round whose peers disagree on the colliders can't arm Sally and
    // refuses (rl#114) — surfaced here as a clean error exit with the actionable fix (rl#115), not a
    // panic/abort. The interactive menu handles its own unarmable case in-client.
    let render_mode = resolve_render_mode(args.render_mode.as_deref())?;
    render::build_windowed_app(boot, external_crab, render_mode)?.run();
    Ok(())
}
