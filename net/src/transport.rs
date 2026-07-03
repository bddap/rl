//! iroh transport: LAN peer discovery + per-tick message exchange over QUIC.
//!
//! This is the wire under [`crate::lockstep`]. It is deliberately thin: it
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
use iroh::endpoint::{
    Connection, IdleTimeout, QuicTransportConfig, RecvStream, SendStream, presets,
};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{Endpoint, EndpointId};
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use n0_future::StreamExt;
use tokio::sync::mpsc;

use crate::articulation::CrabArticulation;
use crate::lockstep::TickMsg;
use crate::membership::{self, Beat};
use crate::server::{Admission, JoinRequest};
use crate::sim::PlayerId;
use crate::snapshot::CoreSnapshot;

/// ALPN for the game's wire. The framing is `[len:u32 LE][kind:u8][body]`, where
/// `kind` ([`Frame`]) selects how the body decodes. BUMP the trailing version on ANY
/// incompatible wire change — a new/changed frame kind or a shifted body layout — so
/// mismatched builds refuse at connect time rather than mis-frame or silently desync
/// mid-stream (the fleet updates atomically via rl-update, so a refused connect means
/// "update the other device", never a mixed-version match).
pub const ALPN: &[u8] = b"bddap/rl-game/lockstep/10";

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
pub(crate) enum Frame {
    /// A client→server [`TickMsg`]: one client's input for a tick.
    Tick = 0,
    /// A [`Beat`]: heartbeat + roster advertisement for the formation barrier.
    Beat = 1,
    // Byte 2 was the retired server→client `TickSet` frame (host-authority ships state,
    // not input sets); it is a rejected kind.
    /// A client→server [`JoinRequest`]: a would-be joiner's credentials when it dials a live match
    /// (mid-game join).
    JoinRequest = 3,
    // Byte 4 was the retired broadcast roster-change frame: incumbents learn a mid-game join
    // from the roster on every `Snapshot`, so nothing schedules roster changes client-side.
    /// A server→joiner refusal: the host isn't armed with a real brain, or the crab-asset digests
    /// disagreed — the joiner is turned away LOUDLY rather than admitted onto a wrong crab.
    Refuse = 5,
    /// A server→JOINER welcome: the [`Admission`] the host allocated for THIS joiner, sent UNICAST
    /// to it alone so a joiner can't mistake a concurrent joiner's allocation for its OWN (which
    /// would adopt the wrong [`crate::sim::PlayerId`]).
    Welcome = 6,
    /// A server→client [`CoreSnapshot`]: the host-authoritative full game state for one tick.
    /// A remote client never re-steps the sim from an input set; it ADOPTS this snapshot whole
    /// (state on the wire, not inputs), so warm-vs-cold physics divergence between peers never
    /// crosses the link. Adopted in ARRIVAL order, no tick gate — the ordered stream already delivers
    /// them in step order (rl#204: ticks stay monotone even across a host restart); see
    /// [`crate::lockstep::Lockstep::adopt_snapshots`].
    Snapshot = 7,
    /// A server→client [`CrabArticulation`]: the render-only per-part crab pose for one tick,
    /// broadcast beside a [`Frame::Snapshot`]. Not authoritative — float
    /// render garnish a windowed client writes onto its own frozen crab so it renders the host's
    /// exact pose without simulating physics. The `tick` inside is the version; a headless client
    /// (which renders nothing) decodes and ignores it.
    Articulation = 8,
}

impl Frame {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Frame::Tick),
            1 => Some(Frame::Beat),
            3 => Some(Frame::JoinRequest),
            5 => Some(Frame::Refuse),
            6 => Some(Frame::Welcome),
            7 => Some(Frame::Snapshot),
            8 => Some(Frame::Articulation),
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
    /// A client's input, received by the server.
    Tick(TickMsg),
    Beat(Beat),
    /// A would-be joiner's credentials, received by the host on a live-match dial.
    JoinRequest(JoinRequest),
    /// A refusal reason, received by a turned-away joiner — surfaced loudly, never a
    /// silent drop.
    Refuse(String),
    /// THIS joiner's own [`Admission`], unicast by the host — the joiner builds its session from it
    /// ([`crate::lockstep::Lockstep::join_at`]). Unicast so it can't confuse a concurrent joiner's
    /// allocation for its own.
    Welcome(Admission),
    /// The host-authoritative full game state for one tick, received by a remote client.
    /// The client adopts it via [`crate::lockstep::Lockstep::apply_core_snapshot`]
    /// instead of stepping its own sim.
    Snapshot(CoreSnapshot),
    /// The render-only crab pose for one tick, received by a windowed remote client.
    /// Written onto the client's frozen crab render entities via
    /// `net::render`; a headless client ignores it.
    Articulation(CrabArticulation),
}

/// Field offsets in the encoded [`TickMsg`], derived from the field widths so a change
/// to the input's width shifts the trailing fields automatically
/// rather than silently corrupting them — the offsets can't drift from
/// [`crate::sim::Input::WIRE_LEN`] because they're computed from it.
const IN_LEN: usize = crate::sim::Input::WIRE_LEN;
const OFF_INPUT: usize = 8; // after issue_tick(8)
/// Wire size of an encoded [`TickMsg`]: issue_tick(8) + input.
const TICKMSG_LEN: usize = OFF_INPUT + IN_LEN;

/// A wire message: owns its [`Frame`] kind byte and its body codec, so adding or changing a frame
/// touches ONE impl instead of rippling across [`Frame`], [`Frame::from_byte`], a bespoke send
/// wrapper, and the [`read_loop`] decode arm. `encode`/`decode` are byte-exact inverses (the
/// roundtrip tests pin every impl); [`Codec::Bytes`] keeps the per-tick [`TickMsg`] send zero-alloc
/// (a fixed array) while variable-length frames use `Vec<u8>`.
pub(crate) trait Codec: Sized {
    /// The kind byte this message frames as.
    const KIND: Frame;
    /// The encoded-body representation — a fixed array for fixed-width bodies (no heap), `Vec<u8>`
    /// for variable-length ones.
    type Bytes: AsRef<[u8]>;
    /// Encode the body WITHOUT the kind/length, which [`write_frame`] prepends.
    fn encode(&self) -> Self::Bytes;
    /// Decode a body, failing LOUDLY (which closes the link) on any short/overlong/malformed input
    /// rather than yielding a corrupt value.
    fn decode(body: &[u8]) -> Result<Self>;
}

impl Codec for TickMsg {
    const KIND: Frame = Frame::Tick;
    type Bytes = [u8; TICKMSG_LEN];

    /// Fixed-width little-endian: issue_tick(8) | input.
    fn encode(&self) -> [u8; TICKMSG_LEN] {
        let mut b = [0u8; TICKMSG_LEN];
        b[0..OFF_INPUT].copy_from_slice(&self.issue_tick.to_le_bytes());
        b[OFF_INPUT..].copy_from_slice(&self.input.to_bytes());
        b
    }

    fn decode(body: &[u8]) -> Result<Self> {
        let b: &[u8; TICKMSG_LEN] = body.try_into().map_err(|_| {
            anyhow::anyhow!("tick frame body is {} B, want {TICKMSG_LEN}", body.len())
        })?;
        Ok(TickMsg {
            issue_tick: u64::from_le_bytes(b[0..OFF_INPUT].try_into().unwrap()),
            input: crate::sim::Input::from_bytes(b[OFF_INPUT..].try_into().unwrap()),
        })
    }
}

/// Split `n` bytes off the front of `r`, advancing it; a typed error (naming `what`) on truncation
/// so a short/garbled frame fails LOUDLY at decode rather than corrupting silently. Shared by every
/// variable-length body codec.
fn take<'a>(r: &mut &'a [u8], n: usize, what: &str) -> Result<&'a [u8]> {
    anyhow::ensure!(r.len() >= n, "frame truncated reading {what}");
    let (head, tail) = r.split_at(n);
    *r = tail;
    Ok(head)
}

impl Codec for CoreSnapshot {
    const KIND: Frame = Frame::Snapshot;
    type Bytes = Vec<u8>;

    /// The body layout is owned by [`crate::snapshot`] (which owns the type and its
    /// deterministic little-endian form); transport owns only the kind byte.
    fn encode(&self) -> Vec<u8> {
        self.to_bytes()
    }

    fn decode(body: &[u8]) -> Result<Self> {
        // Surface a malformed host snapshot LOUDLY (closing the link) — a client must never render
        // a half-decoded authoritative state ([[silent-fallback-antipattern]]).
        CoreSnapshot::from_bytes(body).map_err(|e| anyhow::anyhow!("decoding snapshot frame: {e}"))
    }
}

impl Codec for CrabArticulation {
    const KIND: Frame = Frame::Articulation;
    type Bytes = Vec<u8>;

    /// The body layout is owned by [`crate::articulation`]; transport owns only the kind byte.
    fn encode(&self) -> Vec<u8> {
        self.to_bytes()
    }

    fn decode(body: &[u8]) -> Result<Self> {
        // Fail LOUDLY on a malformed pose — a client must never render a half-decoded articulation
        // (though a DROPPED one is merely a skipped render frame, superseded by the next tick).
        CrabArticulation::from_bytes(body)
            .map_err(|e| anyhow::anyhow!("decoding articulation frame: {e}"))
    }
}

impl Codec for Beat {
    const KIND: Frame = Frame::Beat;
    type Bytes = Vec<u8>;

    /// The [`Beat`] body layout is owned by [`crate::membership`] (which owns the type); transport
    /// owns only the kind byte.
    fn encode(&self) -> Vec<u8> {
        membership::encode_beat(self)
    }

    fn decode(body: &[u8]) -> Result<Self> {
        membership::decode_beat(body)
    }
}

impl Codec for JoinRequest {
    const KIND: Frame = Frame::JoinRequest;
    type Bytes = [u8; 8];

    /// `asset_digest(8)`, little-endian. The joiner's weights digest was dropped from the wire
    /// (rl#206) — a joiner never executes the brain, so the host self-gate is the one weights
    /// guard. The strict length checks make a pre-rl#206 16-byte frame a loud decode error, never
    /// a silently misread request (the fleet updates atomically via rl-update).
    fn encode(&self) -> [u8; 8] {
        self.asset_digest.to_le_bytes()
    }

    fn decode(body: &[u8]) -> Result<Self> {
        let mut r = body;
        let asset_digest = u64::from_le_bytes(take(&mut r, 8, "asset_digest")?.try_into().unwrap());
        anyhow::ensure!(
            r.is_empty(),
            "join-request frame has {} trailing bytes",
            r.len()
        );
        Ok(JoinRequest { asset_digest })
    }
}

impl Codec for Admission {
    // An Admission travels ONLY as the unicast joiner welcome — never broadcast it: a concurrent
    // joiner would read the frame as its OWN allocation and adopt the wrong PlayerId.
    const KIND: Frame = Frame::Welcome;
    type Bytes = Vec<u8>;

    /// `pid(1) | effective_tick(8) | u16 roster_len | roster pids…`. The seed is NOT carried — it is
    /// the shared build constant (`MATCH_SEED`) every peer already holds, so the joiner reuses it for
    /// `join_at`.
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(1 + 8 + 2 + self.roster.len());
        b.push(self.pid.0);
        b.extend_from_slice(&self.effective_tick.to_le_bytes());
        b.extend_from_slice(&(self.roster.len() as u16).to_le_bytes());
        for pid in &self.roster {
            b.push(pid.0);
        }
        b
    }

    fn decode(body: &[u8]) -> Result<Self> {
        let mut r = body;
        let pid = PlayerId(take(&mut r, 1, "pid")?[0]);
        let effective_tick =
            u64::from_le_bytes(take(&mut r, 8, "effective_tick")?.try_into().unwrap());
        let n = u16::from_le_bytes(take(&mut r, 2, "roster len")?.try_into().unwrap());
        let mut roster = Vec::with_capacity(n as usize);
        for _ in 0..n {
            roster.push(PlayerId(take(&mut r, 1, "roster pid")?[0]));
        }
        anyhow::ensure!(r.is_empty(), "welcome frame has {} trailing bytes", r.len());
        Ok(Admission {
            pid,
            effective_tick,
            roster,
        })
    }
}

/// A loud join REFUSAL body (a UTF-8 reason). Its own type so it routes as [`Frame::Refuse`] through
/// the one generic send path — a turned-away joiner surfaces the reason, never a silent drop.
pub(crate) struct Refuse(pub String);

impl Codec for Refuse {
    const KIND: Frame = Frame::Refuse;
    type Bytes = Vec<u8>;
    fn encode(&self) -> Vec<u8> {
        self.0.as_bytes().to_vec()
    }
    fn decode(body: &[u8]) -> Result<Self> {
        Ok(Refuse(
            String::from_utf8(body.to_vec()).context("refuse frame body is not UTF-8")?,
        ))
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

/// Pause between the direct address arriving and the re-publish, so the freshly-spawned
/// mDNS service loop is already subscribed and observes it (see [`publish_lan_addr`]).
const PUBLISH_SETTLE: Duration = Duration::from_millis(300);

/// Build a LAN-only iroh endpoint: relay disabled and mDNS the only address lookup,
/// so discovery and connectivity stay on the local network with no internet
/// dependency (couch co-op is the target — the boys + dad on one LAN). Internet
/// relay/holepunching is a later layer (Steam), deliberately not wired here.
pub async fn bind_endpoint() -> Result<(Endpoint, MdnsAddressLookup)> {
    // `Minimal` gives a crypto provider and nothing networked; relay disabled so the
    // endpoint never reaches the internet, then mDNS is the sole address lookup. (The
    // `N0` preset would attach n0's DNS/pkarr publisher and a relay — internet
    // dependencies we don't want for couch co-op.)
    //
    // Keep-alive + a short idle timeout so a SILENTLY-dead peer (crash, cable pull — no QUIC
    // close on the wire) is detected in seconds, not iroh's ~30s default: the timeout kills the
    // connection, the reader EOFs, the link is evicted, and the host DEPARTS the player
    // instead of stalling the match on its missing inputs. A graceful exit closes the
    // connection immediately and never needs this. Keep-alives are a packet a second per peer —
    // nothing at couch scale.
    let transport = QuicTransportConfig::builder()
        .keep_alive_interval(Duration::from_secs(1))
        .max_idle_timeout(Some(
            IdleTimeout::try_from(Duration::from_secs(5)).expect("constant timeout fits"),
        ))
        .build();
    let endpoint = Endpoint::builder(presets::Minimal)
        .relay_mode(iroh::RelayMode::Disabled)
        .transport_config(transport)
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

    // Wait for a direct address, then win the publish race (NOT `online()` — that blocks
    // on a relay connection, which never comes with the relay disabled).
    publish_lan_addr(&endpoint, SERVICE_NAME).await?;
    Ok((endpoint, mdns))
}

/// The mDNS publish-race dance, shared by the game transport and the telemetry endpoints
/// ([`crate::telemetry`]): wait for the endpoint to enumerate a direct address, then
/// force one more publish by setting the discovery user data to `service_name`.
///
/// iroh's first publish can land before the mDNS service loop starts awaiting changes, so
/// it misses the addresses and swarm-discovery announces nothing ("no addresses") even
/// though `direct_addrs` is populated. Setting user data re-runs the publish path; this
/// publish the loop observes, so it announces our addresses. The value also scopes
/// discovery to `service_name`, so endpoints in a different namespace ignore us.
pub(crate) async fn publish_lan_addr(endpoint: &Endpoint, service_name: &str) -> Result<()> {
    wait_for_direct_addr(endpoint).await?;
    tokio::time::sleep(PUBLISH_SETTLE).await;
    let ud = iroh::endpoint_info::UserData::try_from(service_name.to_string())
        .context("building discovery user data")?;
    endpoint.set_user_data_for_address_lookup(Some(ud));
    Ok(())
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

/// One queued outbound frame. The body is `Arc`'d so a broadcast encodes once and every peer's
/// queue shares the same bytes.
struct OutFrame {
    kind: Frame,
    body: Arc<[u8]>,
}

/// A peer's outbound frame queue: the caller ENQUEUES (never awaits a network write) and the
/// per-link writer task spawned in [`wire_connection`] owns the [`SendStream`] — so a dead or
/// backpressured peer can never block the caller, which on the windowed host is the bevy main
/// thread inside `block_on` (the rl#198 freeze). Bounded at [`OUT_QUEUE_FRAMES`]: a peer whose
/// queue FILLS is not draining (a wedged process whose runtime still answers keep-alives — the
/// idle timeout never fires for it), so the link is dropped LOUDLY and the departure machinery
/// takes over — never a silent frame drop to a peer presumed live ([[silent-fallback-antipattern]]).
/// `Arc`'d so cloned identity (`Weak::ptr_eq`) is how [`drop_if_same`] tells a stale link from a
/// reconnect's replacement — and so dropping the map entry closes the channel, which is what tells
/// the writer task to exit.
type Link = Arc<mpsc::Sender<OutFrame>>;

/// The identity handle the reader/writer tasks keep for eviction: weak, so a task holding it
/// doesn't keep its own channel (and hence itself) alive after the link is replaced or removed.
type LinkId = std::sync::Weak<mpsc::Sender<OutFrame>>;

/// Outbound queue depth per peer before the link is declared wedged and dropped. The windowed
/// host enqueues ~2 frames/tick at 30 Hz, so this is a few seconds of a peer not draining —
/// far past any healthy hiccup QUIC's own flow control absorbs.
const OUT_QUEUE_FRAMES: usize = 256;

/// Upper bound on ONE frame write before the writer declares the peer wedged. Only reachable for
/// a QUIC-alive peer whose stream flow-control window is full (it stopped reading); a genuinely
/// dead connection errors out at the idle timeout, well before this. Without it the parked write
/// would pin the stream (and the writer task) for as long as the peer stays alive-but-wedged.
const WRITE_STALL_TIMEOUT: Duration = Duration::from_secs(10);

/// The per-peer outbound link table, shared by the accept path, the discovery dialer, and every
/// reader task.
type Links = Arc<tokio::sync::Mutex<BTreeMap<EndpointId, PeerLink>>>;

/// Half a peer link: a frame queue into the peer's writer task. The receive side feeds a shared
/// mpsc channel that [`Session`] owns, so the caller polls one queue for all peers.
#[derive(Clone, Debug)]
struct PeerLink {
    send: Link,
}

/// A running networked session: the bound endpoint, the discovery task, and the
/// per-peer links. Drive it by `submit`-ing the local tick message (fanned out to
/// every peer) and draining `recv` for peers' messages, then hand both to the
/// [`crate::lockstep::Lockstep`] driver.
pub struct Session {
    endpoint: Endpoint,
    _router: Router,
    /// Inbound frames from all peers, tagged with the authenticated sender.
    inbox: mpsc::Receiver<FromPeer>,
    /// A clone of the inbox SENDER, so a direct dial ([`Session::connect_direct`]) can
    /// wire its reader into the same shared inbox the discovery/accept paths feed.
    inbox_tx: mpsc::Sender<FromPeer>,
    /// Outbound links keyed by peer id. Grows as peers connect (discovery or accept).
    links: Links,
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
    /// derive a deterministic [`crate::sim::PlayerId`] ordering across peers.
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

    /// Frame `msg` (its [`Codec::KIND`] + body) and send it to exactly one `peer` — the single
    /// unicast primitive. A client ships its input UP to the one server; a host UNICASTs a joiner its
    /// [`Welcome`] or [`Refuse`]. The message type picks the kind, so there is no per-kind wrapper. A
    /// send failure drops that link (same reconnect-safe policy as [`Session::broadcast`]); losing
    /// the server link stalls the client, the correct visible failure. No link to `peer` is a no-op.
    pub(crate) async fn send<M: Codec>(&self, peer: EndpointId, msg: &M) {
        let bytes = msg.encode();
        self.send_frame(peer, M::KIND, bytes.as_ref().into()).await;
    }

    /// Frame `msg` and send it to every connected peer — the single fan-out primitive (a
    /// [`CoreSnapshot`], a [`CrabArticulation`], a formation [`Beat`]). The body is
    /// identical for all, so one write reaches them; a dead peer is dropped inside
    /// [`Session::broadcast_frame`], never blocking the others.
    pub(crate) async fn broadcast<M: Codec>(&self, msg: &M) {
        let bytes = msg.encode();
        self.broadcast_frame(M::KIND, bytes.as_ref().into()).await;
    }

    /// Ship our per-tick input UP to the one server — the headless `net` harness's unicast
    /// entry (the windowed client goes through [`crate::net_loop::NetDriver`]). A concrete
    /// facade over the internal generic [`Session::send`] so the cross-crate caller needs no
    /// access to the crate-private [`Codec`].
    pub async fn send_to(&self, peer: EndpointId, msg: &TickMsg) {
        self.send(peer, msg).await;
    }

    /// Broadcast a host-authoritative [`CoreSnapshot`] DOWN to every client —
    /// the headless host's fan-out entry: the host ships game STATE, not an input set, and the
    /// client ADOPTS it rather than re-stepping. A concrete facade over the generic
    /// [`Session::broadcast`], keeping [`Codec`] crate-private (the windowed path fans out through
    /// [`crate::net_loop::NetDriver`] directly).
    pub async fn broadcast_snapshot(&self, snapshot: &CoreSnapshot) {
        self.broadcast(snapshot).await;
    }

    /// Broadcast a render-only [`CrabArticulation`] DOWN to every client
    /// — the windowed host ships it beside each [`CoreSnapshot`] so a joiner renders the
    /// host's exact crab pose. A concrete facade over the generic [`Session::broadcast`], keeping
    /// [`Codec`] crate-private (the windowed host fans out through [`crate::net_loop::NetDriver`]).
    pub async fn broadcast_articulation(&self, articulation: &CrabArticulation) {
        self.broadcast(articulation).await;
    }

    /// Frame `body` with `kind` and queue it to one `peer` (the unicast analogue of
    /// [`Self::broadcast_frame`]). ENQUEUE-only — the peer's writer task performs the network
    /// write, so this never blocks on the peer; a write failure surfaces there and drops the
    /// link. No link to `peer` (or a writer already dead) is a no-op (surfaced as the
    /// higher-level stall, not a panic); a FULL queue means the peer stopped draining — drop the
    /// link loudly (see [`OUT_QUEUE_FRAMES`]).
    async fn send_frame(&self, peer: EndpointId, kind: Frame, body: Arc<[u8]>) {
        let wedged = {
            let links = self.links.lock().await;
            let Some(link) = links.get(&peer) else { return };
            match link.send.try_send(OutFrame { kind, body }) {
                Err(mpsc::error::TrySendError::Full(_)) => Some(Arc::downgrade(&link.send)),
                _ => None, // sent, or Closed (the writer died; eviction is under way)
            }
        };
        if let Some(link_id) = wedged {
            tracing::warn!(%peer, ?kind, "peer outbound queue full (not draining) — dropping link");
            drop_if_same(&self.links, peer, &link_id).await;
        }
    }

    /// Frame `body` with `kind` and queue it to every connected peer. ENQUEUE-only, like
    /// [`Self::send_frame`]: a dead peer's write failure surfaces in its writer task (which drops
    /// the link) and can never block the others — nor the caller, which on the windowed host is
    /// the game's main thread. A peer whose queue is FULL is dropped loudly. Eviction
    /// happens after the map lock is released — [`drop_if_same`] re-takes it.
    async fn broadcast_frame(&self, kind: Frame, body: Arc<[u8]>) {
        let mut wedged: Vec<(EndpointId, LinkId)> = Vec::new();
        {
            let links = self.links.lock().await;
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
            drop_if_same(&self.links, id, &link_id).await;
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
    let links: Links = Arc::new(tokio::sync::Mutex::new(BTreeMap::new()));

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
    links: Links,
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
    links: Links,
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

    let (tx, mut rx) = mpsc::channel::<OutFrame>(OUT_QUEUE_FRAMES);
    let tx = Arc::new(tx);
    let link_id: LinkId = Arc::downgrade(&tx);
    // Replacing an existing entry drops the old link's only strong Arc, which closes the old
    // channel — the old writer task then drains, exits, and drops its (old-connection) stream.
    links.lock().await.insert(peer, PeerLink { send: tx });

    // Writer: the ONE owner of the send stream. Draining the queue here (instead of the caller
    // awaiting the write) is what keeps a dead/backpressured peer from ever blocking the game
    // thread; a write failure or stall evicts the link, and the reader EOF below (or a
    // roster-level departure) is how the rest of the system learns the peer is gone.
    let links_for_writer = links.clone();
    let writer_id = link_id.clone();
    tokio::spawn(async move {
        while let Some(f) = rx.recv().await {
            match tokio::time::timeout(WRITE_STALL_TIMEOUT, write_frame(&mut send, f.kind, &f.body))
                .await
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
        drop_if_same(&links_for_writer, peer, &writer_id).await;
    });

    let links_for_reader = links.clone();
    tokio::spawn(async move {
        // WARN, not debug: read_loop returns Ok on every normal ending (clean EOF, session drop),
        // so an Err here is a real protocol violation — a mis-framed/unknown/truncated frame (e.g.
        // an ALPN-matched build with a drifted codec) — and must be visible, not a silent link
        // drop the joiner mis-reads as "host unreachable" ([[silent-fallback-antipattern]]).
        if let Err(e) = read_loop(recv, peer, inbox).await {
            tracing::warn!(%peer, "peer read loop ended on a protocol violation: {e:#}");
        }
        // Peer's stream closed → drop its link so broadcast stops targeting it (reconnect-safe: a
        // late EOF from a link a reconnect already replaced must not evict the fresh one).
        drop_if_same(&links_for_reader, peer, &link_id).await;
    });
    Ok(())
}

/// Upper bound on a single frame body, to reject a hostile/garbled length before
/// allocating. The two largest legitimate frames both fit comfortably: a full-roster [`Beat`]
/// (kind + start + count + weights digest + 256×32-byte ids ≈ 8 KiB) and a full-roster
/// [`CoreSnapshot`] (~14 B/player, ≈ 4 KiB at 256); 16 KiB is generous
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
        // Each arm is one line: the kind names the [`Codec`] type, which decodes its own body.
        let msg = match kind {
            Frame::Tick => PeerWire::Tick(TickMsg::decode(body)?),
            Frame::Beat => PeerWire::Beat(Beat::decode(body)?),
            Frame::JoinRequest => PeerWire::JoinRequest(JoinRequest::decode(body)?),
            Frame::Refuse => PeerWire::Refuse(Refuse::decode(body)?.0),
            Frame::Welcome => PeerWire::Welcome(Admission::decode(body)?),
            Frame::Snapshot => PeerWire::Snapshot(CoreSnapshot::decode(body)?),
            Frame::Articulation => PeerWire::Articulation(CrabArticulation::decode(body)?),
        };
        if inbox.send(FromPeer { from: peer, msg }).await.is_err() {
            return Ok(()); // session dropped
        }
    }
}

/// Evict `id`'s link, but ONLY if it is still `failed` — a reconnect may have replaced the link for
/// this id since the failure, and the fresh link must survive a late failure/EOF from the stale
/// one. The single link-eviction path, shared by the writer (send failure) and reader (EOF) tasks.
async fn drop_if_same(links: &Links, id: EndpointId, failed: &LinkId) {
    let mut links = links.lock().await;
    if links
        .get(&id)
        .is_some_and(|l| std::sync::Weak::ptr_eq(&Arc::downgrade(&l.send), failed))
    {
        links.remove(&id);
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
    use crate::sim::Input;

    #[test]
    fn tick_body_wire_roundtrips() {
        let m = TickMsg {
            issue_tick: 1234,
            input: Input::from_axes(0.5, -0.25),
        };
        assert_eq!(TickMsg::decode(TickMsg::encode(&m).as_ref()).unwrap(), m);
    }

    #[test]
    fn frame_kind_byte_roundtrips() {
        // Each kind byte maps back to its variant; anything else is rejected (a wire
        // mismatch the read loop surfaces as an error).
        assert_eq!(Frame::from_byte(0), Some(Frame::Tick));
        assert_eq!(Frame::from_byte(1), Some(Frame::Beat));
        // byte 2 was the retired server→client `TickSet` frame; it is now a rejected kind.
        assert_eq!(Frame::from_byte(2), None);
        assert_eq!(Frame::from_byte(3), Some(Frame::JoinRequest));
        // byte 4 was the retired broadcast roster-change frame; it is now a rejected kind.
        assert_eq!(Frame::from_byte(4), None);
        assert_eq!(Frame::from_byte(5), Some(Frame::Refuse));
        assert_eq!(Frame::from_byte(6), Some(Frame::Welcome));
        assert_eq!(Frame::from_byte(7), Some(Frame::Snapshot));
        assert_eq!(Frame::from_byte(8), Some(Frame::Articulation));
        assert_eq!(Frame::from_byte(9), None);
        assert_eq!(Frame::from_byte(0xff), None);
    }

    #[test]
    fn articulation_wire_roundtrips() {
        use crate::articulation::{CrabArticulation, PartTransform, ReposeWire, VehiclePoseWire};
        // A render pose round-trips byte-for-byte through the frame codec, and a truncated body is a
        // loud error (never a half-decoded pose on the client).
        let art = CrabArticulation {
            tick: 909,
            parts: vec![
                PartTransform {
                    part: 0,
                    pos: [0.5, -1.0, 2.0],
                    rot: [0.0, 0.0, 0.0, 1.0],
                },
                PartTransform {
                    part: 12,
                    pos: [3.0, 4.0, 5.0],
                    rot: [0.5, 0.5, 0.5, 0.5],
                },
            ],
            repose: Some(ReposeWire {
                shift: [1.0, 0.0, -2.0],
                pivot: [0.0, 0.5, 0.0],
                scale: 9.0,
            }),
            vehicle: Some(VehiclePoseWire {
                pos: [2.0, 5.5, -1.0],
                rot: [0.0, std::f32::consts::FRAC_1_SQRT_2, 0.0, std::f32::consts::FRAC_1_SQRT_2],
            }),
        };
        let body = CrabArticulation::encode(&art);
        assert_eq!(CrabArticulation::decode(&body).unwrap(), art);
        assert!(CrabArticulation::decode(&body[..body.len() - 1]).is_err());
    }

    #[test]
    fn snapshot_wire_roundtrips() {
        use crate::sim::{PlayerId, Pos, Sim};
        // A stepped sim's authoritative snapshot round-trips byte-for-byte through the frame codec,
        // and a truncated body is a loud error (never a half-decoded state on the client).
        let mut sim = Sim::new(3, &[PlayerId(0), PlayerId(1)]);
        sim.set_external_crab_pose(Pos { x: 77, z: -88 }, 5, 0);
        let snap = sim.core_snapshot();
        let body = CoreSnapshot::encode(&snap);
        assert_eq!(CoreSnapshot::decode(&body).unwrap(), snap);
        assert!(CoreSnapshot::decode(&body[..body.len() - 1]).is_err());
    }

    #[test]
    fn join_request_wire_roundtrips() {
        let req = JoinRequest {
            asset_digest: 0xdead_beef_cafe_f00d,
        };
        assert_eq!(
            JoinRequest::decode(JoinRequest::encode(&req).as_ref()).unwrap(),
            req
        );
        // A short body is a loud error, not a guessed-at request.
        assert!(JoinRequest::decode(&[0u8; 4]).is_err());
        // Trailing bytes too — notably a pre-rl#206 16-byte (weights|assets) frame must be a loud
        // version-skew error, never silently truncated to just its first field.
        assert!(JoinRequest::decode(&[0u8; 16]).is_err());
    }

    #[test]
    fn welcome_wire_roundtrips() {
        use crate::sim::PlayerId;
        // The empty-roster edge and a multi-member roster both round-trip byte-for-byte.
        for roster in [vec![], vec![PlayerId(0), PlayerId(1), PlayerId(2)]] {
            let adm = Admission {
                pid: PlayerId(2),
                effective_tick: 1_234_567,
                roster,
            };
            assert_eq!(
                Admission::decode(Admission::encode(&adm).as_ref()).unwrap(),
                adm
            );
        }
        // Truncation (a roster_len claiming more pids than present) is a loud error.
        let mut body = Admission::encode(&Admission {
            pid: PlayerId(0),
            effective_tick: 7,
            roster: vec![PlayerId(0), PlayerId(1)],
        });
        body.pop(); // drop the last pid byte → count says 2 but only 1 present
        assert!(Admission::decode(&body).is_err());
    }
}
