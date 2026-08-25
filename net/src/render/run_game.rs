//! The ONE windowed-game entry (rl#411 stage 3): every platform runs the game by
//! assembling a [`GameConfig`] and calling [`run_game`]. Adapters own only the
//! platform inputs — the native CLI maps clap argv here, a web adapter maps its URL
//! query — so the body itself reads no argv and no env.

use std::path::PathBuf;

use anyhow::{Context, Result};
use iroh::EndpointId;

use super::app::{Boot, build_windowed_app};
use crate::{formation, net_loop};

/// The checkpoint-dir env fallback the deploy scripts export (deploy/rl-update sets
/// it). Named ONCE here and referenced by every `--nn-crab-checkpoint` / `--checkpoint`
/// flag that honors it, so the flags and [`nn_crab_policy`]'s error prose can't drift
/// apart.
pub const CHECKPOINT_ENV: &str = "RL_CRAB_CHECKPOINT_DIR";

/// Everything the windowed game needs to launch, platform inputs already resolved.
pub struct GameConfig {
    pub launch: Launch,
    /// LAN-discovery window when forming from a lobby.
    pub discover_secs: u64,
    /// Peers to wait for before the discovery window may close early.
    pub expect: usize,
    /// Telemetry collector to dial (fleet launchers pass one).
    pub telemetry: Option<EndpointId>,
    /// Checkpoint dirs, one armed crab each; empty = the default weights under the
    /// asset root.
    pub nn_crab_checkpoints: Vec<PathBuf>,
    pub view: crab_world::RenderArgs,
    /// Where the game's assets live — resolved by the adapter
    /// ([`crab_world::assets::native_asset_root`] on native), pinned process-wide
    /// before anything loads.
    pub asset_root: PathBuf,
}

pub enum Launch {
    /// Boot into the Host / Join menu.
    Menu,
    /// Scripted formation, no menu: form a match now — dialing an explicit host, or
    /// discovering — and boot straight into the round (solo when nobody shows).
    Lobby { dial: Option<EndpointId> },
}

pub fn run_game(config: GameConfig) -> Result<()> {
    crab_world::assets::set_asset_root(config.asset_root);
    let nn_crabs = nn_crab_policies(config.nn_crab_checkpoints)?;
    // Per-launch entropy (rl#305): the run layout derives from this seed, so real play
    // opens somewhere fresh every launch (and every in-round RESTART re-draws); the
    // authoritative sim logs the seed for repro. Screenshot/probe tools keep their
    // pinned seed instead.
    let seed = crate::sim::random_match_seed();
    let boot = match config.launch {
        Launch::Menu => Boot::Menu {
            seed,
            telemetry: config.telemetry,
        },
        Launch::Lobby { dial } => {
            let result = net_loop::connect_and_form_dialing(
                seed,
                config.discover_secs,
                config.expect,
                net_loop::DialTargets {
                    host: dial,
                    collector: config.telemetry,
                },
                crate::SyncStamp::local(nn_crabs.len() as u8),
            )?;
            match result {
                net_loop::MatchResult::Joined(joined) => {
                    let (client, driver) = *joined;
                    Boot::Round(Box::new((client, Some(driver))))
                }
                net_loop::MatchResult::Alone => {
                    Boot::Round(Box::new((formation::solo_client_for(seed), None)))
                }
                net_loop::MatchResult::Cancelled => {
                    unreachable!("a scripted lobby has no menu to cancel")
                }
            }
        }
    };
    let view = config
        .view
        .resolve(crab_world::mesh_fallback::Surface::Game);
    build_windowed_app(boot, nn_crabs, view)?.run();
    Ok(())
}

/// The launch gate: resolve the checkpoint dir and load it in ONE read — the returned
/// [`Policy`](crab_world::policy::Policy) is armed by construction, never re-read by
/// the plugin (rl#241: a classify-then-reload gate can straddle a checkpoint swap and
/// arm a rest-pose statue it never vetted). Returns the resolved dir alongside for
/// operator-facing labels.
pub fn nn_crab_policy(flag: Option<PathBuf>) -> Result<(PathBuf, crab_world::policy::Policy)> {
    use crab_world::policy::{CheckpointUnusable, RigDims};
    // The env fallback is clap's, declared on each subcommand's checkpoint flag
    // ([`CHECKPOINT_ENV`]). `fp-screenshot` deliberately opts OUT of it: there the flag
    // ARMS a crab at all, so the env would seed one into a shot meant to have none.
    let dir = flag.unwrap_or_else(|| {
        crab_world::assets::asset_root()
            .join("assets")
            .join("weights")
    });
    // Weights↔world (rl#281 stage 6, the rl-demo pattern): adopt the checkpoint's
    // recorded plant — arena + friction cap — before arming, so the brain plays in the
    // world it trained in and GCR serves a terrain brain its baked tile. Multi-binding
    // launches adopt each dir in turn; a disagreeing sidecar refuses here, at t=0.
    if let Err(err) = crab_world::bot::body::adopt_recorded_plant(&dir) {
        anyhow::bail!(
            "checkpoint under {} records a plant this launch can't adopt — {err}",
            dir.display()
        );
    }
    match crab_world::policy::load_armed(&dir) {
        Ok(policy) => Ok((dir, policy)),
        Err(CheckpointUnusable::Missing) => anyhow::bail!(
            "rl#114: no trained crab brain (brain.bin) under {} — the giant crab IS the trained NN \
             body (\"Sally\"), and there is no integer stand-in. Point this command's checkpoint \
             flag or {CHECKPOINT_ENV} at a trained checkpoint dir (deploy/rl-update must set it, \
             and EVERY device needs the IDENTICAL brain + crab model), then relaunch.",
            dir.display()
        ),
        Err(CheckpointUnusable::Refused(why)) => anyhow::bail!(
            "checkpoint under {} was REFUSED — {why}. Fix the checkpoint, then relaunch.",
            dir.display()
        ),
        Err(CheckpointUnusable::Mismatch(RigDims { obs, action })) => {
            let RigDims {
                obs: rig_obs,
                action: rig_act,
            } = crab_world::play::rig_dims();
            anyhow::bail!(
                "rl#199: checkpoint under {} was built for a DIFFERENT rig — its brain wants \
                 {obs} obs / {action} act but this binary's crab rig is {rig_obs} obs / \
                 {rig_act} act. Sally would launch as an inert rest-pose statue, so refusing to \
                 launch instead. Retrain/redeploy a checkpoint for this rig, or run a binary \
                 whose rig matches the checkpoint.",
                dir.display()
            )
        }
    }
}

/// [`nn_crab_policy`] over every `--nn-crab-checkpoint` binding (default binding when
/// none given) — the armed policies, one per crab.
pub fn nn_crab_policies(flags: Vec<PathBuf>) -> Result<Vec<crab_world::policy::Policy>> {
    if flags.is_empty() {
        return Ok(vec![nn_crab_policy(None)?.1]);
    }
    flags
        .into_iter()
        .enumerate()
        .map(|(idx, dir)| {
            nn_crab_policy(Some(dir))
                .map(|(_, policy)| policy)
                .with_context(|| format!("crab {idx}'s brain binding is unusable (rl#200)"))
        })
        .collect()
}
