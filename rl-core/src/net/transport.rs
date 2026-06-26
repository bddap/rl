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
use iroh::endpoint::{Connection, RecvStream, SendStream, presets};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{Endpoint, EndpointId};
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use n0_future::StreamExt;
use tokio::sync::mpsc;

use crate::net::lockstep::{Confirmed, TickMsg};
use crate::net::membership::{self, Beat};

/// ALPN for the game's wire. The framing is `[len:u32 LE][kind:u8][body]`, where
/// `kind` ([`Frame`]) selects a per-tick [`TickMsg`] (the lockstep channel) or a
/// [`Beat`] (the match-formation barrier channel — rl#44). Bumped on any incompatible
/// wire change so mismatched builds refuse to connect rather than desync. `/1` added the
/// kind byte + the barrier channel over `/0`'s bare-TickMsg stream; `/2` added the host's
/// start-GO byte to every [`Beat`] (rl#58), shifting the barrier-frame layout; `/3` added the
/// per-peer policy-weights digest to every [`Beat`] (rl#82, GCR), shifting it again.
pub const ALPN: &[u8] = b"bddap/rl-game/lockstep/3";

/// mDNS service name — scopes discovery to THIS game so we don't pick up unrelated
/// iroh endpoints on the LAN (the default `irohv1` service is shared by all iroh
/// apps). All peers must use the same name to find each other.
pub const SERVICE_NAME: &str = "bddap-rl-game";

/// Frame kind, the byte right after the length prefix. Selects how the body decodes so
/// the lockstep channel ([`TickMsg`]) and the formation-barrier channel ([`Beat`]) can
/// share one QUIC stream without ambiguity. An unknown kind is a wire mismatch and the
/// frame is rejected (closing the link) rather than guessed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Frame {
    /// A per-tick lockstep [`TickMsg`] (the input/hash channel).
    Tick = 0,
    /// A [`Beat`]: heartbeat + roster advertisement for the formation barrier (rl#44).
    Beat = 1,
}

impl Frame {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Frame::Tick),
            1 => Some(Frame::Beat),
            _ => None,
        }
    }
}

/// A decoded frame from a peer: which channel it arrived on. The barrier driver reads
/// [`PeerWire::Beat`]s during formation; the lockstep driver reads
/// [`PeerWire::Tick`]s during play. Tagging at the transport (not the consumer) keeps
/// each driver from having to know the other's wire.
#[derive(Debug, Clone)]
pub enum PeerWire {
    Tick(TickMsg),
    Beat(Beat),
}

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

/// Encode a [`TickMsg`] body to its fixed-width little-endian wire form (without the
/// frame kind/length, which [`write_frame`] prepends).
fn encode_tick_body(m: &TickMsg) -> [u8; TICKMSG_LEN] {
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

/// Inverse of [`encode_tick_body`].
fn decode_tick_body(b: &[u8; TICKMSG_LEN]) -> TickMsg {
    let ctick = u64::from_le_bytes(b[OFF_CTICK..OFF_CHASH].try_into().unwrap());
    let chash = u64::from_le_bytes(b[OFF_CHASH..].try_into().unwrap());
    TickMsg {
        apply_tick: u64::from_le_bytes(b[0..OFF_INPUT].try_into().unwrap()),
        input: crate::net::sim::Input::from_bytes(b[OFF_INPUT..OFF_CTICK].try_into().unwrap()),
        confirmed: (ctick != NO_CONFIRMED_TICK).then_some(Confirmed {
            tick: ctick,
            hash: chash,
        }),
    }
}

/// A frame received from a specific peer. `from` is the QUIC-authenticated peer id —
/// the trustworthy sender identity (never read the sender from the body), which the
/// drivers need so a peer can't inject input (or a roster vote) as someone else.
#[derive(Debug, Clone)]
pub struct FromPeer {
    pub from: EndpointId,
    pub msg: PeerWire,
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
    /// Inbound frames from all peers, tagged with the authenticated sender.
    inbox: mpsc::Receiver<FromPeer>,
    /// A clone of the inbox SENDER, so a direct dial ([`Session::connect_direct`]) can
    /// wire its reader into the same shared inbox the discovery/accept paths feed.
    inbox_tx: mpsc::Sender<FromPeer>,
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

    /// This endpoint's dialable address (id + direct addresses) — what a peer passes to
    /// [`Session::connect_direct`] to reach us without mDNS. A hook for the
    /// multi-endpoint formation test and a future out-of-band "connect by code" path
    /// (Steam), where peers exchange addresses some way other than LAN multicast.
    pub fn local_addr(&self) -> iroh::EndpointAddr {
        self.endpoint.addr()
    }

    /// Dial a peer directly by its [`iroh::EndpointAddr`], bypassing mDNS, and wire it
    /// up exactly like a discovered/accepted peer (same `ALPN`, same reader into the
    /// shared inbox, same link bookkeeping). For tests and out-of-band connection
    /// setups; the normal LAN path is mDNS discovery in [`start_session`]. The id
    /// tie-break (lower id opens the stream) is enforced inside [`wire_connection`], so
    /// dialing the "wrong" direction still yields one correctly-oriented stream.
    pub async fn connect_direct(&self, addr: impl Into<iroh::EndpointAddr>) -> Result<()> {
        let conn = self
            .endpoint
            .connect(addr, ALPN)
            .await
            .context("direct-dialing peer")?;
        wire_connection(
            self.endpoint.id(),
            conn,
            self.inbox_tx.clone(),
            self.links.clone(),
        )
        .await
    }

    /// Send our per-tick [`TickMsg`] (the lockstep channel) to every connected peer.
    pub async fn broadcast(&self, msg: &TickMsg) {
        self.broadcast_frame(Frame::Tick, &encode_tick_body(msg))
            .await;
    }

    /// Send our [`Beat`] (the formation-barrier channel, rl#44) to every connected
    /// peer. Used only during cold-start; once the match starts the lockstep channel
    /// takes over. Separate from [`Session::broadcast`] so the barrier driver and the
    /// tick driver each speak only their own channel.
    pub async fn broadcast_beat(&self, beat: &Beat) {
        self.broadcast_frame(Frame::Beat, &membership::encode_beat(beat))
            .await;
    }

    /// Frame `body` with `kind` + length and send it to every connected peer. A send
    /// failure on one link is logged and the link dropped — a dead peer must not block
    /// the others (the lockstep stall on its missing input, or its expiry from the
    /// barrier, is the correct visible failure).
    async fn broadcast_frame(&self, kind: Frame, body: &[u8]) {
        // Snapshot the send halves under the map lock, then RELEASE it before any
        // network write. Holding the map lock across `write_frame().await` would let
        // one backpressured peer stall every other map user (link insert/remove,
        // connected_peers); the per-`send` mutex still serializes that peer's frames.
        let targets: Vec<(EndpointId, Arc<tokio::sync::Mutex<SendStream>>)> = {
            let links = self.links.lock().await;
            links
                .iter()
                .map(|(id, link)| (*id, link.send.clone()))
                .collect()
        };
        let mut dead = Vec::new();
        for (id, send) in targets {
            let mut s = send.lock().await;
            if let Err(e) = write_frame(&mut s, kind, body).await {
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
                if links
                    .get(&id)
                    .is_some_and(|l| Arc::ptr_eq(&l.send, &failed_send))
                {
                    links.remove(&id);
                }
            }
        }
    }

    /// Pull the next peer frame if one is ready, without blocking. The caller polls
    /// this each tick (lockstep) or each barrier round to feed the relevant driver.
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

    // Keep a sender clone for the session itself (direct-dial path) before the
    // discovery loop takes ownership of the original.
    let inbox_tx_for_session = inbox_tx.clone();

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
        inbox_tx: inbox_tx_for_session,
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
        if let Err(e) = wire_connection(
            self.my_id,
            connection,
            self.inbox.clone(),
            self.links.clone(),
        )
        .await
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
    links
        .lock()
        .await
        .insert(peer, PeerLink { send: send.clone() });

    let links_for_reader = links.clone();
    tokio::spawn(async move {
        if let Err(e) = read_loop(recv, peer, inbox).await {
            tracing::debug!(%peer, "peer read loop ended: {e:#}");
        }
        // Peer's stream closed → drop its link so broadcast stops targeting it, BUT
        // only if it's still THIS connection's link: a reconnect may have replaced it
        // for the same id, and a late EOF from the old reader must not evict the new.
        let mut links = links_for_reader.lock().await;
        if links
            .get(&peer)
            .is_some_and(|l| Arc::ptr_eq(&l.send, &send))
        {
            links.remove(&peer);
        }
    });
    Ok(())
}

/// Upper bound on a single frame body, to reject a hostile/garbled length before
/// allocating. The largest legitimate frame is a full-roster [`Beat`]
/// (kind + start + count + weights digest + 256×32-byte ids ≈ 8 KiB); 16 KiB is generous
/// slack and still bounds a bad length to a small allocation.
const MAX_FRAME_LEN: usize = 16 * 1024;

/// Read `[len:u32 LE][kind:u8][body]` frames from a peer's recv stream until it closes,
/// decoding each into a [`PeerWire`] (a lockstep tick or a barrier beat) and forwarding
/// it into the shared inbox tagged with the authenticated peer id. An unknown kind or a
/// body that fails to decode is a wire mismatch: surfaced as an error (which closes the
/// link) rather than silently skipped, so a mixed-version peer fails loudly.
async fn read_loop(
    mut recv: RecvStream,
    peer: EndpointId,
    inbox: mpsc::Sender<FromPeer>,
) -> Result<()> {
    loop {
        let mut lenb = [0u8; 4];
        if recv.read_exact(&mut lenb).await.is_err() {
            return Ok(()); // clean EOF: peer closed the stream
        }
        let len = u32::from_le_bytes(lenb) as usize;
        anyhow::ensure!(len >= 1, "frame length {len} has no room for a kind byte");
        anyhow::ensure!(len <= MAX_FRAME_LEN, "frame length {len} exceeds cap");
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf)
            .await
            .context("reading frame body")?;
        let kind = Frame::from_byte(buf[0])
            .with_context(|| format!("unknown frame kind {:#x}", buf[0]))?;
        let body = &buf[1..];
        let msg = match kind {
            Frame::Tick => {
                let arr: [u8; TICKMSG_LEN] = body.try_into().map_err(|_| {
                    anyhow::anyhow!("tick frame body is {} B, want {TICKMSG_LEN}", body.len())
                })?;
                PeerWire::Tick(decode_tick_body(&arr))
            }
            Frame::Beat => PeerWire::Beat(membership::decode_beat(body)?),
        };
        if inbox.send(FromPeer { from: peer, msg }).await.is_err() {
            return Ok(()); // session dropped
        }
    }
}

/// Write one `[len:u32 LE][kind:u8][body]` frame to a send stream. The length covers
/// the kind byte plus the body, so the reader allocates exactly the right buffer.
async fn write_frame(send: &mut SendStream, kind: Frame, body: &[u8]) -> Result<()> {
    let len = (1 + body.len()) as u32;
    send.write_all(&len.to_le_bytes()).await?;
    send.write_all(&[kind as u8]).await?;
    send.write_all(body).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::sim::Input;

    #[test]
    fn tick_body_wire_roundtrips() {
        // Both the confirmed-Some case and the None case (which uses the wire
        // sentinel) must round-trip exactly.
        for confirmed in [
            Some(Confirmed {
                tick: 1232,
                hash: 0xdead_beef_cafe_f00d,
            }),
            None,
        ] {
            let m = TickMsg {
                apply_tick: 1234,
                input: Input::from_axes(0.5, -0.25),
                confirmed,
            };
            assert_eq!(decode_tick_body(&encode_tick_body(&m)), m);
        }
    }

    #[test]
    fn frame_kind_byte_roundtrips() {
        // Each kind byte maps back to its variant; anything else is rejected (a wire
        // mismatch the read loop surfaces as an error).
        assert_eq!(Frame::from_byte(0), Some(Frame::Tick));
        assert_eq!(Frame::from_byte(1), Some(Frame::Beat));
        assert_eq!(Frame::from_byte(2), None);
        assert_eq!(Frame::from_byte(0xff), None);
    }
}
