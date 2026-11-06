//! The ONE windowed-game entry (rl#411 stage 3): every platform runs the game by
//! assembling a [`GameConfig`] and calling [`run_game`]. Adapters own only the
//! platform inputs — the native CLI maps clap argv here, a web adapter maps its URL
//! query — so the body itself reads no argv and no env.

use std::path::PathBuf;

use anyhow::Result;

use super::app::{Boot, build_windowed_app};
#[cfg(not(target_family = "wasm"))]
use crate::{formation, net_loop};

/// Everything the windowed game needs to launch, platform inputs already resolved.
pub struct GameConfig {
    pub launch: Launch,
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
    /// A CLI affordance — native-only (it blocks on the sync formation pacer).
    #[cfg(not(target_family = "wasm"))]
    Lobby {
        dial: Option<iroh::EndpointId>,
        /// LAN-discovery window.
        discover_secs: u64,
        /// Peers to wait for before the discovery window may close early.
        expect: usize,
    },
}

pub fn run_game(config: GameConfig) -> Result<()> {
    crab_world::assets::set_asset_root(config.asset_root);
    // Per-launch entropy (rl#305): the run layout derives from this seed, so real play
    // opens somewhere fresh every launch (and every in-round RESTART re-draws); the
    // authoritative sim logs the seed for repro. Screenshot tools keep their pinned
    // seed instead.
    let seed = net_proto::sim::random_match_seed();
    let boot = match config.launch {
        Launch::Menu => Boot::Menu { seed },
        #[cfg(not(target_family = "wasm"))]
        Launch::Lobby {
            dial,
            discover_secs,
            expect,
        } => {
            let result = net_loop::connect_and_form_dialing(
                seed,
                discover_secs,
                expect,
                dial,
                net_proto::SyncStamp::local(),
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
    build_windowed_app(boot, config.view.resolve())?.run();
    Ok(())
}
