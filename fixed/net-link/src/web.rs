//! The web platform half: the browser's JS event loop is the executor and the relay is
//! the only path ("in browsers, there will never be any direct addresses" — iroh).
//! Discovery is explicit dial from a join code — a bare [`iroh::EndpointId`], resolved
//! to the peer's relay through the n0 defaults' pkarr-over-HTTPS lookup (rl#412
//! cross-play; relay hosting stays n0 by held call). There is no mDNS here.

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

/// Bind a browser session. Async — it runs on the JS event loop; the sync face the
/// game drives is [`crate::bind_session`].
async fn start_session() -> Result<Session> {
    let endpoint = iroh::Endpoint::builder(presets::N0)
        .transport_config(transport_config())
        .bind()
        .await
        .context("binding iroh endpoint (browser)")?;
    Ok(crate::assemble(endpoint, |endpoint, inbox, links| Guts {
        _router: spawn_router(endpoint, inbox, links),
    }))
}

/// The web face of [`crate::bind_session`]: the async bind rides the JS event loop;
/// its verdict comes back over a same-thread channel the poll drains.
pub(crate) struct Pending(std::sync::mpsc::Receiver<Result<Session>>);

pub(crate) fn begin_bind() -> Pending {
    let (tx, rx) = std::sync::mpsc::channel();
    n0_future::task::spawn(async move {
        // A send after the receiver is dropped (bind abandoned) just drops the
        // Session, whose own Drop closes the endpoint.
        let _ = tx.send(start_session().await);
    });
    Pending(rx)
}

impl Pending {
    pub(crate) fn poll(&mut self) -> Option<Result<Session>> {
        match self.0.try_recv() {
            Ok(v) => Some(v),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            // Only reachable if the bind task died without sending — a panic the
            // console already screamed about; surface it on the caller's error path.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::anyhow!("session bind task died")))
            }
        }
    }
}
