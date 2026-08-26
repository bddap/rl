//! The link layer (rl#411): one poll-driven sync surface over iroh, one thin platform
//! impl per target. Everything protocol — frame kinds, codecs, fragmentation — is
//! `net_proto::codec`; this layer moves those bytes. Everything platform — runtime,
//! discovery, relay posture, clock source — lives in the platform module below the
//! [`Session`] surface: the per-frame surface is identical on both targets, so code
//! above this crate stays cfg-free. Construction is the one platform entry point
//! (native `start_session` is sync, web's is async on the JS event loop).
//!
//! - **native** ([`native`], non-wasm): tokio runtime OWNED by the session, direct
//!   LAN addressing + mDNS discovery, relay off.
//! - **web** ([`web`], wasm32): the JS event loop, relay-mode transport (browsers have
//!   no direct path — "in browsers, there will never be any direct addresses"),
//!   discovery = explicit dial from a URL/join code.
//!
//! Both speak the same ALPN and datagram semantics, so a browser peer and a native
//! peer interoperate in one match.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use iroh::endpoint::{Connection, IdleTimeout, QuicTransportConfig, RecvStream, SendStream};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{Endpoint, EndpointId};
use n0_future::time;
use tokio::sync::mpsc;

pub use net_proto::codec::PeerWire;
use net_proto::codec::{
    ALPN, Codec, Frame, MAX_FRAME_LEN, StateAssembler, StateCodec, decode_peer_wire,
    parse_state_datagram, state_datagrams,
};

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(not(target_family = "wasm"))]
use native as platform;
#[cfg(not(target_family = "wasm"))]
pub use native::{publish_lan_addr, start_session};

#[cfg(target_family = "wasm")]
mod web;
#[cfg(target_family = "wasm")]
use web as platform;
#[cfg(target_family = "wasm")]
pub use web::start_session;

#[derive(Debug, Clone)]
pub struct FromPeer {
    pub from: EndpointId,
    pub msg: PeerWire,
}

fn transport_config() -> QuicTransportConfig {
    QuicTransportConfig::builder()
        .keep_alive_interval(Duration::from_secs(1))
        .max_idle_timeout(Some(
            IdleTimeout::try_from(Duration::from_secs(5)).expect("constant timeout fits"),
        ))
        .build()
}

struct OutFrame {
    kind: Frame,
    body: Arc<[u8]>,
}

type Link = Arc<mpsc::Sender<OutFrame>>;

type LinkId = std::sync::Weak<mpsc::Sender<OutFrame>>;

const OUT_QUEUE_FRAMES: usize = 256;

/// Most concurrent peer links a session holds against INBOUND dials (rl#350). The roster
/// caps at [`net_proto::membership::MAX_MEMBERS`] players (≤ MAX_MEMBERS − 1 remote
/// links), so this leaves headroom for a pending joiner while denying a sybil dialer —
/// each accepted link costs three tasks plus an [`OUT_QUEUE_FRAMES`] out-queue —
/// unbounded growth. Our own dials are never refused: we choose them, so they're
/// already bounded.
const MAX_LINKS: usize = net_proto::membership::MAX_MEMBERS;

/// Longest an INBOUND connection may take to finish the bi-stream/HELLO handshake
/// (rl#350). Without a deadline the gate above is bypassable: `wire_connection` parks in
/// `accept_bi`/HELLO awaiting bytes only the dialer can send, our 1 s keep-alives hold
/// the idle connection open forever, and the parked accept tasks accumulate BEFORE the
/// [`MAX_LINKS`] check ever runs.
const ACCEPT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

const WRITE_STALL_TIMEOUT: Duration = Duration::from_secs(10);

// A std (not tokio) mutex: every critical section is a plain map touch with no await
// inside, and the sync lock is what lets the whole send/poll surface below be called
// straight from the frame loop — no runtime, no block_on.
type Links = Arc<std::sync::Mutex<BTreeMap<EndpointId, PeerLink>>>;

fn locked(links: &Links) -> std::sync::MutexGuard<'_, BTreeMap<EndpointId, PeerLink>> {
    // Poison-proof: the critical sections are map ops that don't panic, but if one ever
    // does, degrading to the map as-left (a dropped link at worst) beats poisoning every
    // later frame-loop send into a panic cascade.
    links
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone, Debug)]
struct PeerLink {
    send: Link,
    /// The QUIC connection itself — the datagram send path ([`Frame::via_datagram`]), and
    /// what link teardown closes so every per-link task exits promptly.
    conn: Connection,
    /// Who DIALED this connection — a property of the connection itself, so both ends agree
    /// on it. On a crossed dial (both sides connecting at once) each side keeps the link with
    /// the LOWER dialer id: a rule both compute identically, so the duplicate converges on
    /// the same survivor everywhere instead of a distributed coin-flip where each side closes
    /// (or stream-resets, by dropping the writer) the connection the other kept.
    dialer: EndpointId,
}

/// One link session: a poll-driven sync surface over iroh (rl#411 stage 2). The frame
/// loop calls send/broadcast/try_recv/connected_peers directly; dials run on the
/// platform executor and report through pollable channels. The platform half —
/// runtime/executor, discovery, relay posture — is [`platform::Guts`]; nothing here
/// is async-facing.
pub struct Session {
    endpoint: Endpoint,
    inbox: mpsc::Receiver<FromPeer>,
    inbox_tx: mpsc::Sender<FromPeer>,
    links: Links,
    /// The session clock's zero point. [`Session::now_ms`] is the one time axis the
    /// protocol drivers above see — `time::Instant` is tokio's on native and
    /// `performance.now`-backed web_time in browsers, monotonic on both.
    epoch: time::Instant,
    guts: platform::Guts,
}

impl Session {
    pub fn endpoint_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub fn connected_peers(&self) -> Vec<EndpointId> {
        locked(&self.links).keys().copied().collect()
    }

    pub fn local_addr(&self) -> iroh::EndpointAddr {
        self.endpoint.addr()
    }

    /// Monotonic milliseconds since the session bound — the injected `now_ms` axis every
    /// protocol-core driver (formation, join, membership timeouts) reads.
    pub fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// Drop a peer's link NOW — remove it and close the connection. The host uses this to
    /// reclaim a [`MAX_LINKS`] slot from a connected-but-never-rostered endpoint (rl#350);
    /// the link's tasks end on the closed connection, and their own `drop_if_same` cleanup
    /// is a no-op on the already-removed entry.
    pub fn disconnect(&self, peer: EndpointId) {
        let link = locked(&self.links).remove(&peer);
        if let Some(link) = link {
            link.conn.close(0u32.into(), b"disconnected by peer");
        }
    }

    /// Dial a peer by explicit address (a join code). Non-blocking: the dial runs on the
    /// platform executor; poll the returned channel for the verdict (callers use it for
    /// diagnostics — a failed dial is non-fatal while discovery may still find the peer).
    pub fn dial(
        &self,
        addr: impl Into<iroh::EndpointAddr>,
    ) -> std::sync::mpsc::Receiver<Result<()>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let endpoint = self.endpoint.clone();
        let inbox = self.inbox_tx.clone();
        let links = self.links.clone();
        let addr = addr.into();
        self.guts.spawn(async move {
            let dialed = async {
                let conn = endpoint
                    .connect(addr, ALPN)
                    .await
                    .context("direct-dialing peer")?;
                wire_connection(endpoint.id(), conn, inbox, links, true).await
            }
            .await;
            let _ = tx.send(dialed);
        });
        rx
    }

    pub fn send<M: Codec>(&self, peer: EndpointId, msg: &M) {
        const {
            assert!(
                !M::KIND.via_datagram(),
                "state frames go via broadcast_state"
            )
        }
        let bytes = msg.encode();
        self.send_frame(peer, M::KIND, bytes.as_ref().into());
    }

    pub fn broadcast<M: Codec>(&self, msg: &M) {
        const {
            assert!(
                !M::KIND.via_datagram(),
                "state frames go via broadcast_state"
            )
        }
        let bytes = msg.encode();
        self.broadcast_frame(M::KIND, bytes.as_ref().into());
    }

    /// Broadcast a state frame over unreliable unordered datagrams — fire-and-forget: no
    /// retransmit, no ordering, no writer queue to wedge. Under congestion the QUIC buffer
    /// drops the OLDEST buffered datagram first, which for full-state frames is exactly
    /// right (stale state is worthless).
    pub fn broadcast_state<M: StateCodec>(&self, msg: &M) {
        const {
            assert!(
                M::KIND.via_datagram(),
                "StateCodec kinds must route via datagram"
            )
        }
        let bytes = msg.encode();
        let body = bytes.as_ref();
        if body.len() > MAX_FRAME_LEN {
            // A real check, not debug-only: an over-cap frame's fragments would be rejected
            // by EVERY receiver, dropping every link at the broadcast rate — a deterministic
            // full-disconnect loop. Skipping the frame loses one tick of state (the next one
            // supersedes it); the latch keeps the design violation loud without the flood.
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::error!(
                    "outbound {:?} frame is {} B, over the {MAX_FRAME_LEN} B cap — skipped \
                     (and every same-size successor with it)",
                    M::KIND,
                    body.len()
                );
            }
            return;
        }
        let frags: Vec<bytes::Bytes> = state_datagrams(M::KIND, msg.tick(), body)
            .into_iter()
            .map(bytes::Bytes::from)
            .collect();
        let links = locked(&self.links);
        for link in links.values() {
            for frag in &frags {
                send_state_datagram(&link.conn, frag.clone());
            }
        }
    }

    fn send_frame(&self, peer: EndpointId, kind: Frame, body: Arc<[u8]>) {
        let wedged = {
            let links = locked(&self.links);
            let Some(link) = links.get(&peer) else { return };
            match link.send.try_send(OutFrame { kind, body }) {
                Err(mpsc::error::TrySendError::Full(_)) => Some(Arc::downgrade(&link.send)),
                _ => None,
            }
        };
        if let Some(link_id) = wedged {
            tracing::warn!(%peer, ?kind, "peer outbound queue full (not draining) — dropping link");
            drop_if_same(&self.links, peer, &link_id);
        }
    }

    fn broadcast_frame(&self, kind: Frame, body: Arc<[u8]>) {
        let mut wedged: Vec<(EndpointId, LinkId)> = Vec::new();
        {
            let links = locked(&self.links);
            for (id, link) in links.iter() {
                if let Err(mpsc::error::TrySendError::Full(_)) = link.send.try_send(OutFrame {
                    kind,
                    body: body.clone(),
                }) {
                    wedged.push((*id, Arc::downgrade(&link.send)));
                }
            }
        }
        for (id, link_id) in wedged {
            tracing::warn!(%id, "peer outbound queue full (not draining) — dropping link");
            drop_if_same(&self.links, id, &link_id);
        }
    }

    pub fn try_recv(&mut self) -> Option<FromPeer> {
        self.inbox.try_recv().ok()
    }
}

#[derive(Clone, Debug)]
struct GameProto {
    my_id: EndpointId,
    inbox: mpsc::Sender<FromPeer>,
    links: Links,
}

impl ProtocolHandler for GameProto {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let conn = connection.clone();
        match time::timeout(
            ACCEPT_HANDSHAKE_TIMEOUT,
            wire_connection(
                self.my_id,
                connection,
                self.inbox.clone(),
                self.links.clone(),
                false,
            ),
        )
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!("accepting game connection failed: {e:#}"),
            Err(_) => {
                tracing::warn!("inbound connection stalled in handshake — closing");
                conn.close(0u32.into(), b"handshake timeout");
            }
        }
        Ok(())
    }
}

/// The accept half both platforms share: route inbound ALPN connections through
/// [`wire_connection`], same as our own dials.
fn spawn_router(endpoint: &Endpoint, inbox: mpsc::Sender<FromPeer>, links: Links) -> Router {
    let handler = GameProto {
        my_id: endpoint.id(),
        inbox,
        links,
    };
    Router::builder(endpoint.clone())
        .accept(ALPN, handler)
        .spawn()
}

const INBOX_DEPTH: usize = 256;

/// The one session assembly both platform constructors share — inbox, links, epoch —
/// so a capacity change or a new field can't land in one platform and drift the other.
/// `make_guts` runs with the shared parts in hand (its router + any discovery tasks
/// consume them) inside whatever executor context the platform set up.
fn assemble(
    endpoint: Endpoint,
    make_guts: impl FnOnce(&Endpoint, mpsc::Sender<FromPeer>, Links) -> platform::Guts,
) -> Session {
    let (inbox_tx, inbox_rx) = mpsc::channel(INBOX_DEPTH);
    let links: Links = Default::default();
    let guts = make_guts(&endpoint, inbox_tx.clone(), links.clone());
    Session {
        endpoint,
        inbox: inbox_rx,
        inbox_tx,
        links,
        epoch: time::Instant::now(),
        guts,
    }
}

fn send_state_datagram(conn: &Connection, frag: bytes::Bytes) {
    use iroh::endpoint::SendDatagramError;
    if let Err(e) = conn.send_datagram(frag) {
        // ConnectionLost: the per-link stream tasks are already tearing this link down —
        // nothing to report. Every other variant is a design violation (both ends are our
        // build via the pinned ALPN, and fragments are sized under QUIC's guaranteed
        // datagram floor), and it means state frames silently stop flowing to that peer —
        // latch ONE loud error instead of a 60 Hz flood.
        if !matches!(e, SendDatagramError::ConnectionLost(_)) {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::error!(
                    "state datagram refused ({e}) — remote peers stop receiving game state"
                );
            }
        }
    }
}

const HELLO: u8 = 0xA5;

async fn wire_connection(
    my_id: EndpointId,
    conn: Connection,
    inbox: mpsc::Sender<FromPeer>,
    links: Links,
    dialed_by_me: bool,
) -> Result<()> {
    let peer = conn.remote_id();
    // Stream direction is by ID ORDER, not by who dialed: on a crossed dial both duplicate
    // connections then behave identically, so either can survive the dedup below.
    let opener = my_id.as_bytes() < peer.as_bytes();
    let (mut send, mut recv) = if opener {
        conn.open_bi().await.context("opening bi-stream")?
    } else {
        conn.accept_bi().await.context("accepting bi-stream")?
    };
    if opener {
        send.write_all(&[HELLO]).await.context("sending hello")?;
    } else {
        let mut h = [0u8; 1];
        recv.read_exact(&mut h).await.context("reading hello")?;
        anyhow::ensure!(h[0] == HELLO, "bad stream-open byte {:#x}", h[0]);
    }

    let dialer = if dialed_by_me { my_id } else { peer };
    let (tx, mut rx) = mpsc::channel::<OutFrame>(OUT_QUEUE_FRAMES);
    let tx = Arc::new(tx);
    let link_id: LinkId = Arc::downgrade(&tx);
    {
        let mut links = locked(&links);
        // Inbound capacity gate (rl#350), under the lock so it can't race past the cap. A
        // known peer's re-dial replaces its link rather than adding one, so it always passes.
        if !dialed_by_me && links.len() >= MAX_LINKS && !links.contains_key(&peer) {
            drop(links);
            conn.close(0u32.into(), b"connection capacity reached");
            return Ok(());
        }
        if let Some(existing) = links.get(&peer) {
            // Duplicate link. Keep the lower-dialer connection (see PeerLink::dialer); on a
            // SAME-dialer duplicate (a re-dial) the newer one wins — the old is stale.
            if existing.dialer.as_bytes() < dialer.as_bytes() {
                drop(links);
                conn.close(0u32.into(), b"crossed dial: lower-dialer link kept");
                return Ok(());
            }
        }
        if let Some(old) = links.insert(
            peer,
            PeerLink {
                send: tx,
                conn: conn.clone(),
                dialer,
            },
        ) {
            old.conn
                .close(0u32.into(), b"crossed dial: lower-dialer link kept");
        }
    }

    let links_for_writer = links.clone();
    let writer_id = link_id.clone();
    n0_future::task::spawn(async move {
        while let Some(f) = rx.recv().await {
            match time::timeout(WRITE_STALL_TIMEOUT, write_frame(&mut send, f.kind, &f.body)).await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::warn!(%peer, "peer send failed, dropping link: {e:#}");
                    break;
                }
                Err(_) => {
                    tracing::warn!(
                        %peer,
                        "peer write stalled >{WRITE_STALL_TIMEOUT:?} (alive but not reading) — dropping link"
                    );
                    break;
                }
            }
        }
        drop_if_same(&links_for_writer, peer, &writer_id);
    });

    let links_for_reader = links.clone();
    let reader_id = link_id.clone();
    let reader_inbox = inbox.clone();
    n0_future::task::spawn(async move {
        // WARN, not debug: read_loop returns Ok on every normal ending (clean EOF, session drop),
        // so an Err here is a real protocol violation — a mis-framed/unknown/truncated frame (e.g.
        // an ALPN-matched build with a drifted codec) — and must be visible, not a silent link
        // drop the joiner mis-reads as "host unreachable" ([[silent-fallback-antipattern]]).
        if let Err(e) = read_loop(recv, peer, reader_inbox).await {
            tracing::warn!(%peer, "peer read loop ended on a protocol violation: {e:#}");
        }
        drop_if_same(&links_for_reader, peer, &reader_id);
    });

    let links_for_dgram = links.clone();
    n0_future::task::spawn(async move {
        // Same loudness contract as the stream reader: Ok is every normal ending (the
        // connection closed), Err is a protocol violation. Dropping the link on either keeps
        // the failure mode LOUD — a peer whose state stopped flowing must read as departed,
        // never as a silently frozen world.
        if let Err(e) = datagram_loop(conn, peer, inbox).await {
            tracing::warn!(%peer, "peer datagram loop ended on a protocol violation: {e:#}");
        }
        drop_if_same(&links_for_dgram, peer, &link_id);
    });
    Ok(())
}

/// Receives state frames ([`Frame::via_datagram`]) — the unreliable unordered lane beside
/// [`read_loop`]'s reliable stream. Reassembles fragments per kind and delivers each complete
/// frame to the same inbox, strictly newest-first ([`StateAssembler`]).
async fn datagram_loop(
    conn: Connection,
    peer: EndpointId,
    inbox: mpsc::Sender<FromPeer>,
) -> Result<()> {
    let mut assemblers: BTreeMap<Frame, StateAssembler> = BTreeMap::new();
    loop {
        let Ok(d) = conn.read_datagram().await else {
            return Ok(());
        };
        let frag = parse_state_datagram(&d)?;
        let asm = assemblers.entry(frag.kind).or_default();
        if let Some(body) = asm.accept(&frag)? {
            let msg = decode_peer_wire(frag.kind, &body)?;
            if inbox.send(FromPeer { from: peer, msg }).await.is_err() {
                return Ok(());
            }
        }
    }
}

async fn read_loop(
    mut recv: RecvStream,
    peer: EndpointId,
    inbox: mpsc::Sender<FromPeer>,
) -> Result<()> {
    loop {
        let mut lenb = [0u8; 4];
        if recv.read_exact(&mut lenb).await.is_err() {
            return Ok(());
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
        anyhow::ensure!(
            !kind.via_datagram(),
            "state frame {kind:?} arrived on the reliable stream"
        );
        let msg = decode_peer_wire(kind, &buf[1..])?;
        if inbox.send(FromPeer { from: peer, msg }).await.is_err() {
            return Ok(());
        }
    }
}

fn drop_if_same(links: &Links, id: EndpointId, failed: &LinkId) {
    let mut links = locked(links);
    if links
        .get(&id)
        .is_some_and(|l| std::sync::Weak::ptr_eq(&Arc::downgrade(&l.send), failed))
        && let Some(l) = links.remove(&id)
    {
        // Explicit close, not handle-drop: the datagram loop holds a Connection clone, so
        // stream teardown alone would keep a dropped link's connection (and that loop)
        // alive until idle timeout.
        l.conn.close(0u32.into(), b"link dropped");
    }
}

async fn write_frame(send: &mut SendStream, kind: Frame, body: &[u8]) -> Result<()> {
    debug_assert!(
        body.len() < MAX_FRAME_LEN,
        "outbound {kind:?} frame is {} B, over the {MAX_FRAME_LEN} B cap every receiver enforces",
        1 + body.len()
    );
    let len = (1 + body.len()) as u32;
    send.write_all(&len.to_le_bytes()).await?;
    send.write_all(&[kind as u8]).await?;
    send.write_all(body).await?;
    Ok(())
}
