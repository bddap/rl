//! The native platform half: a tokio runtime OWNED by the session, direct LAN
//! addressing + mDNS discovery, relay off. Today's LAN game posture — a session binds
//! instantly, solo play needs zero network, peers find each other via mDNS or an
//! explicit join code.

use std::time::Duration;

use anyhow::{Context, Result};
use iroh::Endpoint;
use iroh::endpoint::presets;
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use n0_future::StreamExt;
use tokio::sync::mpsc;

use crate::{Links, Session, locked, spawn_router, transport_config, wire_connection};

const SERVICE_NAME: &str = "bddap-rl-game";

const ADDR_WAIT: Duration = Duration::from_secs(10);

const PUBLISH_SETTLE: Duration = Duration::from_millis(300);

/// The native platform state a [`Session`] carries: the owned runtime every link task
/// runs on, the accept router, and the mDNS auto-dial task.
pub(crate) struct Guts {
    _router: iroh::protocol::Router,
    discovery: tokio::task::JoinHandle<()>,
    rt: tokio::runtime::Runtime,
}

impl Guts {
    /// The platform executor: dial futures and their link tasks run here.
    pub(crate) fn spawn<F>(&self, fut: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        self.rt.spawn(fut);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.guts.discovery.abort();
        // Idempotent with an explicit `close` — the graceful paths close first; this
        // guarantees the endpoint never drops unclosed (iroh logs that at ERROR —
        // fleet-error 2026-08-01) even on early-exit paths.
        self.guts.rt.block_on(self.endpoint.close());
    }
}

impl Session {
    /// Graceful teardown of the endpoint — every pre-round exit path ends through here
    /// (`Drop` backstops the paths that never call it). Telemetry is its own component
    /// with its own teardown; the link knows nothing of it.
    pub fn close(&self) {
        self.guts.rt.block_on(self.endpoint.close());
    }
}

pub fn start_session() -> Result<Session> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let (endpoint, mdns) = rt.block_on(bind_endpoint())?;
    let guard = rt.enter();
    let my_id = endpoint.id();

    let (inbox_tx, inbox_rx) = mpsc::channel(256);
    let links: Links = Default::default();

    let router = spawn_router(&endpoint, inbox_tx.clone(), links.clone());

    let discovery = {
        let endpoint = endpoint.clone();
        let inbox = inbox_tx.clone();
        let links = links.clone();
        tokio::spawn(async move {
            let mut events = mdns.subscribe().await;
            while let Some(ev) = events.next().await {
                if let DiscoveryEvent::Discovered { endpoint_info, .. } = ev {
                    let peer = endpoint_info.endpoint_id;
                    if peer == my_id {
                        continue;
                    }
                    if my_id.as_bytes() >= peer.as_bytes() {
                        continue;
                    }
                    if locked(&links).contains_key(&peer) {
                        continue;
                    }
                    match endpoint.connect(peer, crate::ALPN).await {
                        Ok(conn) => {
                            if let Err(e) =
                                wire_connection(my_id, conn, inbox.clone(), links.clone(), true)
                                    .await
                            {
                                tracing::warn!(%peer, "dialing peer failed: {e:#}");
                            }
                        }
                        Err(e) => tracing::warn!(%peer, "connect to discovered peer failed: {e:#}"),
                    }
                }
            }
        })
    };
    drop(guard);

    Ok(Session {
        endpoint,
        inbox: inbox_rx,
        inbox_tx,
        links,
        epoch: n0_future::time::Instant::now(),
        guts: Guts {
            _router: router,
            discovery,
            rt,
        },
    })
}

async fn bind_endpoint() -> Result<(Endpoint, MdnsAddressLookup)> {
    let endpoint = Endpoint::builder(presets::Minimal)
        .relay_mode(iroh::RelayMode::Disabled)
        .transport_config(transport_config())
        .bind()
        .await
        .context("binding iroh endpoint")?;
    let mdns = MdnsAddressLookup::builder()
        .service_name(SERVICE_NAME)
        .build(endpoint.id())
        .context("starting mDNS discovery")?;
    endpoint
        .address_lookup()
        .context("endpoint has no address lookup registry")?
        .add(mdns.clone());

    // Publish our LAN address once a direct addr exists — in the background, so "is
    // networking up?" is a discovery property, never a boot gate: a session binds
    // instantly and solo play needs no network at all. Peers can't mDNS-find us until
    // this lands, which is exactly discovery's own timeline.
    let ep = endpoint.clone();
    tokio::spawn(async move {
        if let Err(e) = publish_lan_addr(&ep, SERVICE_NAME).await {
            tracing::warn!("LAN address publish failed (undiscoverable until retry): {e:#}");
        }
    });
    Ok((endpoint, mdns))
}

/// Wait for a direct addr, then publish the given service name for mDNS browsers.
/// Shared with the telemetry endpoints, which publish their own service names.
pub async fn publish_lan_addr(endpoint: &Endpoint, service_name: &str) -> Result<()> {
    wait_for_direct_addr(endpoint).await?;
    tokio::time::sleep(PUBLISH_SETTLE).await;
    let ud = iroh::endpoint_info::UserData::try_from(service_name.to_string())
        .context("building discovery user data")?;
    endpoint.set_user_data_for_address_lookup(Some(ud));
    Ok(())
}

async fn wait_for_direct_addr(endpoint: &Endpoint) -> Result<()> {
    use iroh::Watcher;
    let mut addrs = endpoint.watch_addr();
    let deadline = tokio::time::Instant::now() + ADDR_WAIT;
    loop {
        if addrs.get().ip_addrs().next().is_some() {
            return Ok(());
        }
        match tokio::time::timeout_at(deadline, addrs.updated()).await {
            Ok(Ok(_)) => continue,
            Ok(Err(_)) => anyhow::bail!("endpoint address watcher closed"),
            Err(_) => anyhow::bail!("no local IP address after {ADDR_WAIT:?} — is networking up?"),
        }
    }
}
