//! The web platform half: the browser's JS event loop is the executor and the relay is
//! the only path ("in browsers, there will never be any direct addresses" — iroh).
//! Discovery is explicit dial from a URL/join code (the protocol's `DialTargets`
//! shape); there is no mDNS and no background publish. Relay posture: iroh's default
//! (n0-hosted) relays for now — self-hosting is an open owner question on rl#411.

use anyhow::{Context, Result};
use iroh::endpoint::presets;

use crate::{Session, spawn_router, transport_config};

/// The web platform state a [`Session`] carries: just the accept router — tasks ride
/// the JS event loop, so there is no runtime to own.
pub(crate) struct Guts {
    _router: iroh::protocol::Router,
}

impl Guts {
    /// The platform executor: the JS event loop (wasm-bindgen-futures under n0-future).
    pub(crate) fn spawn<F>(&self, fut: F)
    where
        F: std::future::Future<Output = ()> + 'static,
    {
        n0_future::task::spawn(fut);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Idempotent with an explicit `close`, same contract as native.
        self.close();
    }
}

impl Session {
    /// Graceful teardown of the endpoint, fire-and-forget onto the JS event loop —
    /// the browser has no thread to block. Same call shape as native.
    pub fn close(&self) {
        let ep = self.endpoint.clone();
        n0_future::task::spawn(async move {
            ep.close().await;
        });
    }
}

/// Bind a browser session. Async — the caller is platform entry code (a wasm-bindgen
/// future); everything above the [`Session`] surface stays sync.
pub async fn start_session() -> Result<Session> {
    let endpoint = iroh::Endpoint::builder(presets::Minimal)
        .relay_mode(iroh::RelayMode::Default)
        .transport_config(transport_config())
        .bind()
        .await
        .context("binding iroh endpoint (browser)")?;
    Ok(crate::assemble(endpoint, |endpoint, inbox, links| Guts {
        _router: spawn_router(endpoint, inbox, links),
    }))
}
