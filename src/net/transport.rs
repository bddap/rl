//! iroh transport: LAN peer discovery + per-tick message exchange over QUIC.
//!
//! This is the wire under [`crate::net::lockstep`]. It is deliberately thin: it
//! finds peers on the local network (mDNS, no server/relay — see [`bind_endpoint`]),
//! maintains one connection per peer, and ferries length-framed [`TickMsg`]s in and
//! out. It does NOT advance the sim or judge desync — that's the driver's job. The
//! split keeps the determinism-critical code (lockstep + sim) free of any async/IO.
//!
//! Connection direction is tie-broken by endpoint id (lower dials higher) so two
//! peers that discover each other open exactly one connection, not two racing ones.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use iroh::endpoint::{presets, Connection, RecvStream, SendStream};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{Endpoint, EndpointId};
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use n0_future::StreamExt;
use tokio::sync::mpsc;

use crate::net::lockstep::{Confirmed, TickMsg};

/// ALPN for the game's lockstep wire. `/0` is the framing in this module: a stream
/// of `[len:u32 LE][TickMsg bytes]`. Bumped on any incompatible wire change so
/// mismatched builds refuse to connect rather than desync.
pub const ALPN: &[u8] = b"bddap/rl-game/lockstep/0";

/// mDNS service name — scopes discovery to THIS game so we don't pick up unrelated
/// iroh endpoints on the LAN (the default `irohv1` service is shared by all iroh
/// apps). All peers must use the same name to find each other.
pub const SERVICE_NAME: &str = "bddap-rl-game";

/// Wire sentinel for [`TickMsg::confirmed`] == `None`. `u64::MAX` as the tick can
/// never occur in play (the sim would overflow long before), so it unambiguously
/// means "no tick confirmed yet" on the wire while the in-memory type stays an
/// honest `Option`.
const NO_CONFIRMED_TICK: u64 = u64::MAX;

/// Field offsets in the encoded [`TickMsg`], derived from the field widths so the
/// input growing (Phase 1 widened [`Input`]) shifts the trailing fields automatically
/// rather than silently corrupting them — the offsets can't drift from
/// [`crate::net::sim::Input::WIRE_LEN`] because they're computed from it.
const IN_LEN: usize = crate::net::sim::Input::WIRE_LEN;
const OFF_INPUT: usize = 8; // after apply_tick(8)
const OFF_CTICK: usize = OFF_INPUT + IN_LEN; // after input
const OFF_CHASH: usize = OFF_CTICK + 8; // after confirmed_tick(8)
/// Wire size of an encoded [`TickMsg`]: apply_tick(8) + input + confirmed_tick(8) +
/// confirmed_hash(8).
const TICKMSG_LEN: usize = OFF_CHASH + 8;

/// Encode a [`TickMsg`] to its fixed-width little-endian wire form.
fn encode_msg(m: &TickMsg) -> [u8; TICKMSG_LEN] {
    let (ctick, chash) = match m.confirmed {
        Some(c) => (c.tick, c.hash),
        None => (NO_CONFIRMED_TICK, 0),
    };
    let mut b = [0u8; TICKMSG_LEN];
    b[0..OFF_INPUT].copy_from_slice(&m.apply_tick.to_le_bytes());
    b[OFF_INPUT..OFF_CTICK].copy_from_slice(&m.input.to_bytes());
    b[OFF_CTICK..OFF_CHASH].copy_from_slice(&ctick.to_le_bytes());
    b[OFF_CHASH..].copy_from_slice(&chash.to_le_bytes());
    b
}

/// Inverse of [`encode_msg`].
fn decode_msg(b: &[u8; TICKMSG_LEN]) -> TickMsg {
    let ctick = u64::from_le_bytes(b[OFF_CTICK..OFF_CHASH].try_into().unwrap());
    let chash = u64::from_le_bytes(b[OFF_CHASH..].try_into().unwrap());
    TickMsg {
        apply_tick: u64::from_le_bytes(b[0..OFF_INPUT].try_into().unwrap()),
        input: crate::net::sim::Input::from_bytes(b[OFF_INPUT..OFF_CTICK].try_into().unwrap()),
        confirmed: (ctick != NO_CONFIRMED_TICK).then_some(Confirmed { tick: ctick, hash: chash }),
    }
}

/// A message received from a specific peer. `from` is the QUIC-authenticated peer
/// id — the trustworthy sender identity (never read the sender from the body), which
/// the lockstep driver needs so a peer can't inject input as someone else.
#[derive(Debug, Clone, Copy)]
pub struct FromPeer {
    pub from: EndpointId,
    pub msg: TickMsg,
}

/// How long to wait for the endpoint to enumerate at least one local IP address
/// before giving up. mDNS can only ANNOUNCE addresses the endpoint has discovered;
/// until it has one, swarm-discovery logs "no addresses, not announcing" and peers
/// never see us. On a normal LAN this resolves in well under a second.
const ADDR_WAIT: Duration = Duration::from_secs(10);

/// Build a LAN-only iroh endpoint: relay disabled and mDNS the only address lookup,
/// so discovery and connectivity stay on the local network with no internet
/// dependency (couch co-op is the target — the boys + dad on one LAN). Internet
/// relay/holepunching is a later layer (Steam), deliberately not wired here.
pub async fn bind_endpoint() -> Result<(Endpoint, MdnsAddressLookup)> {
    // `Minimal` gives a crypto provider and nothing networked; relay disabled so the
    // endpoint never reaches the internet, then mDNS is the sole address lookup. (The
    // `N0` preset would attach n0's DNS/pkarr publisher and a relay — internet
    // dependencies we don't want for couch co-op.)
    let endpoint = Endpoint::builder(presets::Minimal)
        .relay_mode(iroh::RelayMode::Disabled)
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

    // Wait until the endpoint has a direct address to advertise. NOT `online()` —
    // that blocks on a relay connection, which never comes with the relay disabled.
    wait_for_direct_addr(&endpoint).await?;

    // Force one more address publish AFTER the mDNS service loop is running. iroh's
    // first publish can land before that loop starts awaiting changes, so it misses
    // the addresses and swarm-discovery announces nothing ("no addresses") even
    // though `direct_addrs` is populated. Setting user data re-runs the publish path;
    // this publish the loop observes, so it announces our addresses. The value also
    // scopes discovery to this game (peers ignore endpoints whose user data isn't
    // ours). The short sleep ensures the freshly-spawned loop is already subscribed.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let ud = iroh::endpoint_info::UserData::try_from(SERVICE_NAME.to_string())
        .context("building discovery user data")?;
    endpoint.set_user_data_for_address_lookup(Some(ud));
    Ok((endpoint, mdns))
}

/// Block until the endpoint reports at least one direct IP address (or [`ADDR_WAIT`]
/// elapses, which on a LAN means something is wrong — surfaced as an error rather
/// than silently advertising nothing).
async fn wait_for_direct_addr(endpoint: &Endpoint) -> Result<()> {
    use iroh::Watcher;
    let mut addrs = endpoint.watch_addr();
    let deadline = tokio::time::Instant::now() + ADDR_WAIT;
    loop {
        if addrs.get().ip_addrs().next().is_some() {
            return Ok(());
        }
        match tokio::time::timeout_at(deadline, addrs.updated()).await {
            Ok(Ok(_)) => continue, // address set changed — re-check
            Ok(Err(_)) => anyhow::bail!("endpoint address watcher closed"),
            Err(_) => anyhow::bail!("no local IP address after {ADDR_WAIT:?} — is networking up?"),
        }
    }
}

/// Half a peer link: a [`TickMsg`] sender. The receive side feeds a shared mpsc
/// channel that [`Session`] owns, so the caller polls one queue for all peers.
#[derive(Clone, Debug)]
struct PeerLink {
    send: Arc<tokio::sync::Mutex<SendStream>>,
}

/// A running networked session: the bound endpoint, the discovery task, and the
/// per-peer links. Drive it by `submit`-ing the local tick message (fanned out to
/// every peer) and draining `recv` for peers' messages, then hand both to the
/// [`crate::net::lockstep::Lockstep`] driver.
pub struct Session {
    endpoint: Endpoint,
    _router: Router,
    /// Inbound messages from all peers, tagged with the authenticated sender.
    inbox: mpsc::Receiver<FromPeer>,
    /// Outbound links keyed by peer id. Grows as peers connect (discovery or accept).
    links: Arc<tokio::sync::Mutex<BTreeMap<EndpointId, PeerLink>>>,
    /// The mDNS discovery loop. Held only to abort it on drop (it owns the mDNS
    /// subscription); aborting on drop stops the dialing loop when the session ends.
    discovery: tokio::task::JoinHandle<()>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.discovery.abort();
    }
}

impl Session {
    /// Our own endpoint id — the stable identity peers dial and the value used to
    /// derive a deterministic [`crate::net::sim::PlayerId`] ordering across peers.
    pub fn endpoint_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// Endpoint ids of peers we currently hold a link to (excludes us).
    pub async fn connected_peers(&self) -> Vec<EndpointId> {
        self.links.lock().await.keys().copied().collect()
    }

    /// Send our tick message to every connected peer. A send failure on one link is
    /// logged and the link dropped — a dead peer must not block the others (lockstep
    /// will stall on the missing input, which is the correct, visible failure).
    pub async fn broadcast(&self, msg: &TickMsg) {
        let bytes = encode_msg(msg);
        // Snapshot the send halves under the map lock, then RELEASE it before any
        // network write. Holding the map lock across `write_framed().await` would let
        // one backpressured peer stall every other map user (link insert/remove,
        // connected_peers); the per-`send` mutex still serializes that peer's frames.
        let targets: Vec<(EndpointId, Arc<tokio::sync::Mutex<SendStream>>)> = {
            let links = self.links.lock().await;
            links.iter().map(|(id, link)| (*id, link.send.clone())).collect()
        };
        let mut dead = Vec::new();
        for (id, send) in targets {
            let mut s = send.lock().await;
            if let Err(e) = write_framed(&mut s, &bytes).await {
                tracing::warn!(%id, "peer send failed, dropping link: {e:#}");
                dead.push((id, send.clone()));
            }
        }
        if !dead.is_empty() {
            let mut links = self.links.lock().await;
            for (id, failed_send) in dead {
                // Only drop the link if it's still the SAME send half that failed —
                // a reconnect may have replaced it for this id between the write and
                // here, and we must not evict the fresh link.
                if links.get(&id).is_some_and(|l| Arc::ptr_eq(&l.send, &failed_send)) {
                    links.remove(&id);
                }
            }
        }
    }

    /// Pull the next peer message if one is ready, without blocking. The caller polls
    /// this each tick to feed [`crate::net::lockstep::Lockstep::record_remote`].
    pub fn try_recv(&mut self) -> Option<FromPeer> {
        self.inbox.try_recv().ok()
    }

    /// Wait for the next peer message (used by tests / a blocking step loop).
    pub async fn recv(&mut self) -> Option<FromPeer> {
        self.inbox.recv().await
    }

    /// Gracefully tear down: close the endpoint (drops connections, stops mDNS).
    pub async fn shutdown(self) {
        self.endpoint.close().await;
    }
}

/// Start a session: bind the LAN endpoint, accept incoming lockstep connections, and
/// spawn the discovery loop that dials newly-seen peers (lower id dials higher).
pub async fn start_session() -> Result<Session> {
    let (endpoint, mdns) = bind_endpoint().await?;
    let my_id = endpoint.id();

    let (inbox_tx, inbox_rx) = mpsc::channel(256);
    let links: Arc<tokio::sync::Mutex<BTreeMap<EndpointId, PeerLink>>> =
        Arc::new(tokio::sync::Mutex::new(BTreeMap::new()));

    // Accept side: register the lockstep protocol. Each accepted connection spawns a
    // reader feeding the shared inbox and records its send half as a link.
    let handler = LockstepProto {
        my_id,
        inbox: inbox_tx.clone(),
        links: links.clone(),
    };
    let router = Router::builder(endpoint.clone())
        .accept(ALPN, handler)
        .spawn();

    // Discovery side: subscribe to mDNS events; for each newly-seen peer that we
    // should dial (our id < theirs, the tie-break that prevents a double connect),
    // open a connection and wire it up exactly like an accepted one.
    let discovery = {
        let endpoint = endpoint.clone();
        let inbox = inbox_tx;
        let links = links.clone();
        tokio::spawn(async move {
            let mut events = mdns.subscribe().await;
            while let Some(ev) = events.next().await {
                if let DiscoveryEvent::Discovered { endpoint_info, .. } = ev {
                    let peer = endpoint_info.endpoint_id;
                    if peer == my_id {
                        continue;
                    }
                    // Tie-break: only the numerically-lower id dials, so the pair
                    // ends up with one connection. The other side accepts.
                    if my_id.as_bytes() >= peer.as_bytes() {
                        continue;
                    }
                    if links.lock().await.contains_key(&peer) {
                        continue; // already linked
                    }
                    match endpoint.connect(peer, ALPN).await {
                        Ok(conn) => {
                            if let Err(e) =
                                wire_connection(my_id, conn, inbox.clone(), links.clone()).await
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

    Ok(Session {
        endpoint,
        _router: router,
        inbox: inbox_rx,
        links,
        discovery,
    })
}

/// Protocol handler for inbound lockstep connections.
#[derive(Clone, Debug)]
struct LockstepProto {
    my_id: EndpointId,
    inbox: mpsc::Sender<FromPeer>,
    links: Arc<tokio::sync::Mutex<BTreeMap<EndpointId, PeerLink>>>,
}

impl ProtocolHandler for LockstepProto {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        if let Err(e) =
            wire_connection(self.my_id, connection, self.inbox.clone(), self.links.clone()).await
        {
            tracing::warn!("accepting lockstep connection failed: {e:#}");
        }
        Ok(())
    }
}

/// One-byte stream-open handshake the dialer sends first. A QUIC `open_bi()` only
/// allocates a stream id locally; the acceptor's `accept_bi()` doesn't return until
/// the opener actually SENDS, so without an immediate write the acceptor would block
/// (and not register its link) until the first per-tick broadcast — long after
/// discovery, risking an asymmetric participant set at session start. Sending this
/// byte right after `open_bi` resolves the acceptor's `accept_bi` immediately.
const HELLO: u8 = 0xA5;

/// Wire up one connection (dialed or accepted): open/accept the single bi-stream,
/// register the send half as this peer's link, and spawn a reader that decodes
/// framed [`TickMsg`]s into the shared inbox tagged with the peer's authenticated id.
///
/// Stream direction is pinned by the same id tie-break as discovery: the lower-id
/// peer dialed, so it OPENS the bi-stream and the higher-id peer ACCEPTS it. Both
/// ends compute the same answer from `my_id` vs the peer id, so exactly one opens —
/// no race, no double stream. The opener sends [`HELLO`] immediately (see its docs);
/// the acceptor consumes it before the message loop. After setup each side holds a
/// symmetric (send, recv) pair and the framing is direction-agnostic.
async fn wire_connection(
    my_id: EndpointId,
    conn: Connection,
    inbox: mpsc::Sender<FromPeer>,
    links: Arc<tokio::sync::Mutex<BTreeMap<EndpointId, PeerLink>>>,
) -> Result<()> {
    let peer = conn.remote_id();
    let dialer = my_id.as_bytes() < peer.as_bytes();
    let (mut send, mut recv) = if dialer {
        conn.open_bi().await.context("opening bi-stream")?
    } else {
        conn.accept_bi().await.context("accepting bi-stream")?
    };
    if dialer {
        send.write_all(&[HELLO]).await.context("sending hello")?;
    } else {
        let mut h = [0u8; 1];
        recv.read_exact(&mut h).await.context("reading hello")?;
        anyhow::ensure!(h[0] == HELLO, "bad stream-open byte {:#x}", h[0]);
    }

    let send = Arc::new(tokio::sync::Mutex::new(send));
    links.lock().await.insert(peer, PeerLink { send: send.clone() });

    let links_for_reader = links.clone();
    tokio::spawn(async move {
        if let Err(e) = read_loop(recv, peer, inbox).await {
            tracing::debug!(%peer, "peer read loop ended: {e:#}");
        }
        // Peer's stream closed → drop its link so broadcast stops targeting it, BUT
        // only if it's still THIS connection's link: a reconnect may have replaced it
        // for the same id, and a late EOF from the old reader must not evict the new.
        let mut links = links_for_reader.lock().await;
        if links.get(&peer).is_some_and(|l| Arc::ptr_eq(&l.send, &send)) {
            links.remove(&peer);
        }
    });
    Ok(())
}

/// Read framed [`TickMsg`]s from a peer's recv stream until it closes, forwarding
/// each into the shared inbox tagged with the authenticated peer id.
async fn read_loop(mut recv: RecvStream, peer: EndpointId, inbox: mpsc::Sender<FromPeer>) -> Result<()> {
    loop {
        let mut lenb = [0u8; 4];
        if recv.read_exact(&mut lenb).await.is_err() {
            return Ok(()); // clean EOF: peer closed the stream
        }
        let len = u32::from_le_bytes(lenb) as usize;
        anyhow::ensure!(len == TICKMSG_LEN, "unexpected frame length {len}");
        let mut body = [0u8; TICKMSG_LEN];
        recv.read_exact(&mut body).await.context("reading frame body")?;
        let msg = decode_msg(&body);
        if inbox.send(FromPeer { from: peer, msg }).await.is_err() {
            return Ok(()); // session dropped
        }
    }
}

/// Write one length-framed message to a send stream.
async fn write_framed(send: &mut SendStream, body: &[u8]) -> Result<()> {
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(body).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::sim::Input;

    #[test]
    fn tickmsg_wire_roundtrips() {
        // Both the confirmed-Some case and the None case (which uses the wire
        // sentinel) must round-trip exactly.
        for confirmed in [
            Some(Confirmed { tick: 1232, hash: 0xdead_beef_cafe_f00d }),
            None,
        ] {
            let m = TickMsg {
                apply_tick: 1234,
                input: Input::from_axes(0.5, -0.25),
                confirmed,
            };
            assert_eq!(decode_msg(&encode_msg(&m)), m);
        }
    }
}
