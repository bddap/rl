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
use iroh::endpoint::{Connection, RecvStream, SendStream, presets};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{Endpoint, EndpointId};
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use n0_future::StreamExt;
use tokio::sync::mpsc;

use crate::lockstep::{Confirmed, TickMsg};
use crate::membership::{self, Beat};
use crate::server::{Admission, JoinRequest, TickSet};
use crate::sim::{Input, PlayerId};

/// ALPN for the game's wire. The framing is `[len:u32 LE][kind:u8][body]`, where
/// `kind` ([`Frame`]) selects a client→server [`TickMsg`] (an input), a server→client
/// [`TickSet`] (the assembled input set), or a [`Beat`] (the match-formation barrier channel
/// — rl#44). Bumped on any incompatible wire change so mismatched builds refuse to connect
/// rather than desync. `/1` added the kind byte + the barrier channel over `/0`'s bare-TickMsg
/// stream; `/2` added the host's start-GO byte to every [`Beat`] (rl#58), shifting the
/// barrier-frame layout; `/3` added the per-peer policy-weights digest to every [`Beat`]
/// (rl#82, GCR); `/4` replaced the P2P tick MESH with the server-coordinated model (rl#151):
/// inputs go UP to the server as [`TickMsg`]s, the server broadcasts the complete [`TickSet`]
/// DOWN — a new frame kind and a topology the old mesh peers can't speak.
pub const ALPN: &[u8] = b"bddap/rl-game/lockstep/5";

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
    /// A client→server [`TickMsg`]: one client's input for a tick (+ its confirmed hash).
    Tick = 0,
    /// A [`Beat`]: heartbeat + roster advertisement for the formation barrier (rl#44).
    Beat = 1,
    /// A server→client [`TickSet`]: the complete input set for one tick (rl#151).
    TickSet = 2,
    /// A client→server [`JoinRequest`]: a would-be joiner's credentials when it dials a live match
    /// (Stage 3 mid-game join, rl#151).
    JoinRequest = 3,
    /// A server→client [`Admission`]: the roster change for a mid-game join, broadcast DOWN to every
    /// client (incumbents schedule it; the joiner builds its session from it). Stage 3, rl#151.
    RosterChange = 4,
    /// A server→joiner refusal: the joiner's weight/collider digests disagreed, so it is turned away
    /// LOUDLY rather than admitted onto a wrong crab. Stage 3, rl#151.
    Refuse = 5,
    /// A server→JOINER welcome: the [`Admission`] the host allocated for THIS joiner, sent UNICAST
    /// to it alone. Distinct from the broadcast [`Frame::RosterChange`] (which incumbents schedule)
    /// precisely so a joiner can't mistake a concurrent joiner's broadcast change for its OWN
    /// allocation and adopt the wrong [`crate::sim::PlayerId`]. Same payload as `RosterChange`.
    /// Stage 3, rl#151.
    Welcome = 6,
}

impl Frame {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Frame::Tick),
            1 => Some(Frame::Beat),
            2 => Some(Frame::TickSet),
            3 => Some(Frame::JoinRequest),
            4 => Some(Frame::RosterChange),
            5 => Some(Frame::Refuse),
            6 => Some(Frame::Welcome),
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
    /// The server's assembled input set for a tick, received by a client.
    TickSet(TickSet),
    /// A would-be joiner's credentials, received by the host on a live-match dial (Stage 3).
    JoinRequest(JoinRequest),
    /// A roster change ([`Admission`]), received by a client: incumbents schedule it (Stage 3).
    RosterChange(Admission),
    /// A refusal reason, received by a turned-away joiner (Stage 3) — surfaced loudly, never a
    /// silent drop.
    Refuse(String),
    /// THIS joiner's own [`Admission`], unicast by the host — the joiner builds its session from it
    /// ([`crate::lockstep::Lockstep::join_at`]). Distinct from [`Self::RosterChange`] so it can't
    /// confuse another joiner's broadcast change for its own allocation (Stage 3).
    Welcome(Admission),
}

/// Wire sentinel for [`TickMsg::confirmed`] == `None`. `u64::MAX` as the tick can
/// never occur in play (the sim would overflow long before), so it unambiguously
/// means "no tick confirmed yet" on the wire while the in-memory type stays an
/// honest `Option`.
const NO_CONFIRMED_TICK: u64 = u64::MAX;

/// Field offsets in the encoded [`TickMsg`], derived from the field widths so the
/// input growing (Phase 1 widened [`Input`]) shifts the trailing fields automatically
/// rather than silently corrupting them — the offsets can't drift from
/// [`crate::sim::Input::WIRE_LEN`] because they're computed from it.
const IN_LEN: usize = crate::sim::Input::WIRE_LEN;
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
        input: crate::sim::Input::from_bytes(b[OFF_INPUT..OFF_CTICK].try_into().unwrap()),
        confirmed: (ctick != NO_CONFIRMED_TICK).then_some(Confirmed {
            tick: ctick,
            hash: chash,
        }),
    }
}

/// Encode a [`TickSet`] body to its wire form (without the frame kind/length): the apply tick,
/// then a `u16`-counted list of `(player, input)` pairs, then a `u16`-counted list of
/// `(player, ctick, chash)` confirmed-hash triples. `u16` counts because a roster can be up to 256
/// players (a `u8` count would overflow at the max), even though couch play is far smaller.
fn encode_tickset_body(set: &TickSet) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + 2 + set.inputs.len() * (1 + IN_LEN) + 2 + set.confirmed.len() * 17);
    b.extend_from_slice(&set.apply_tick.to_le_bytes());
    b.extend_from_slice(&(set.inputs.len() as u16).to_le_bytes());
    for (pid, input) in &set.inputs {
        b.push(pid.0);
        b.extend_from_slice(&input.to_bytes());
    }
    b.extend_from_slice(&(set.confirmed.len() as u16).to_le_bytes());
    for (pid, c) in &set.confirmed {
        b.push(pid.0);
        b.extend_from_slice(&c.tick.to_le_bytes());
        b.extend_from_slice(&c.hash.to_le_bytes());
    }
    b
}

/// Inverse of [`encode_tickset_body`]. A malformed body (truncated, or a length that overruns) is a
/// wire mismatch surfaced as an error (closing the link), never a silently-short set.
fn decode_tickset_body(b: &[u8]) -> Result<TickSet> {
    let mut r = b;
    fn take<'a>(r: &mut &'a [u8], n: usize, what: &str) -> Result<&'a [u8]> {
        anyhow::ensure!(r.len() >= n, "tick-set frame truncated reading {what}");
        let (head, tail) = r.split_at(n);
        *r = tail;
        Ok(head)
    }
    let apply_tick = u64::from_le_bytes(take(&mut r, 8, "apply_tick")?.try_into().unwrap());
    let n_inputs = u16::from_le_bytes(take(&mut r, 2, "input count")?.try_into().unwrap());
    let mut inputs = BTreeMap::new();
    for _ in 0..n_inputs {
        let pid = PlayerId(take(&mut r, 1, "input player")?[0]);
        let input = Input::from_bytes(take(&mut r, IN_LEN, "input")?.try_into().unwrap());
        inputs.insert(pid, input);
    }
    let n_confirmed = u16::from_le_bytes(take(&mut r, 2, "confirmed count")?.try_into().unwrap());
    let mut confirmed = BTreeMap::new();
    for _ in 0..n_confirmed {
        let pid = PlayerId(take(&mut r, 1, "confirmed player")?[0]);
        let tick = u64::from_le_bytes(take(&mut r, 8, "confirmed tick")?.try_into().unwrap());
        let hash = u64::from_le_bytes(take(&mut r, 8, "confirmed hash")?.try_into().unwrap());
        confirmed.insert(pid, Confirmed { tick, hash });
    }
    anyhow::ensure!(r.is_empty(), "tick-set frame has {} trailing bytes", r.len());
    Ok(TickSet {
        apply_tick,
        inputs,
        confirmed,
    })
}

/// Split `n` bytes off the front of `r`, advancing it; a typed error (naming `what`) on truncation
/// so a short/garbled frame fails LOUDLY at decode rather than corrupting silently. Shared by the
/// Stage-3 join codecs below.
fn take<'a>(r: &mut &'a [u8], n: usize, what: &str) -> Result<&'a [u8]> {
    anyhow::ensure!(r.len() >= n, "join frame truncated reading {what}");
    let (head, tail) = r.split_at(n);
    *r = tail;
    Ok(head)
}

/// Encode a [`JoinRequest`] body: `weights_digest(8) | asset_digest(8)`, little-endian.
fn encode_join_request_body(req: &JoinRequest) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&req.weights_digest.to_le_bytes());
    b[8..16].copy_from_slice(&req.asset_digest.to_le_bytes());
    b
}

/// Inverse of [`encode_join_request_body`].
fn decode_join_request_body(b: &[u8]) -> Result<JoinRequest> {
    let mut r = b;
    let weights_digest = u64::from_le_bytes(take(&mut r, 8, "weights_digest")?.try_into().unwrap());
    let asset_digest = u64::from_le_bytes(take(&mut r, 8, "asset_digest")?.try_into().unwrap());
    anyhow::ensure!(r.is_empty(), "join-request frame has {} trailing bytes", r.len());
    Ok(JoinRequest { weights_digest, asset_digest })
}

/// Encode an [`Admission`] (roster-change) body: `pid(1) | effective_tick(8) | roster_len(u16) |
/// roster pids…`. The seed is NOT carried — it is the shared build constant every peer already
/// holds (`MATCH_SEED`), so the joiner reuses it for `join_at`.
fn encode_roster_change_body(adm: &Admission) -> Vec<u8> {
    let mut b = Vec::with_capacity(1 + 8 + 2 + adm.roster.len());
    b.push(adm.pid.0);
    b.extend_from_slice(&adm.effective_tick.to_le_bytes());
    b.extend_from_slice(&(adm.roster.len() as u16).to_le_bytes());
    for pid in &adm.roster {
        b.push(pid.0);
    }
    b
}

/// Inverse of [`encode_roster_change_body`].
fn decode_roster_change_body(b: &[u8]) -> Result<Admission> {
    let mut r = b;
    let pid = PlayerId(take(&mut r, 1, "pid")?[0]);
    let effective_tick = u64::from_le_bytes(take(&mut r, 8, "effective_tick")?.try_into().unwrap());
    let n = u16::from_le_bytes(take(&mut r, 2, "roster len")?.try_into().unwrap());
    let mut roster = Vec::with_capacity(n as usize);
    for _ in 0..n {
        roster.push(PlayerId(take(&mut r, 1, "roster pid")?[0]));
    }
    anyhow::ensure!(r.is_empty(), "roster-change frame has {} trailing bytes", r.len());
    Ok(Admission { pid, effective_tick, roster })
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

/// Half a peer link: a [`TickMsg`] sender. The receive side feeds a shared mpsc
/// channel that [`Session`] owns, so the caller polls one queue for all peers.
#[derive(Clone, Debug)]
struct PeerLink {
    send: Arc<tokio::sync::Mutex<SendStream>>,
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

    /// Send our input [`TickMsg`] UP to the server `peer` (rl#151). A client routes its input to
    /// exactly one endpoint — the match server — not the whole mesh; the server assembles the
    /// complete set and broadcasts it back. A send failure drops that link (the same policy as
    /// [`Session::broadcast_frame`]); losing the link to the server stalls the client, which is the
    /// correct visible failure. No link to `peer` is a no-op (we are not connected to the server —
    /// surfaced as the lockstep stall, not a panic).
    pub async fn send_to(&self, peer: EndpointId, msg: &TickMsg) {
        let send = {
            let links = self.links.lock().await;
            links.get(&peer).map(|l| l.send.clone())
        };
        let Some(send) = send else { return };
        let mut s = send.lock().await;
        if let Err(e) = write_frame(&mut s, Frame::Tick, &encode_tick_body(msg)).await {
            tracing::warn!(%peer, "sending input to server failed: {e:#}");
            drop(s);
            let mut links = self.links.lock().await;
            if links.get(&peer).is_some_and(|l| Arc::ptr_eq(&l.send, &send)) {
                links.remove(&peer);
            }
        }
    }

    /// Broadcast a server-assembled [`TickSet`] DOWN to every connected client (rl#151). Called
    /// only by the peer running the server; the complete set is identical for all clients, so one
    /// fan-out reaches them all. A dead client is dropped inside [`Session::broadcast_frame`] (its
    /// missing future inputs are the server's visible failure, not a block on the others).
    pub async fn broadcast_tickset(&self, set: &TickSet) {
        self.broadcast_frame(Frame::TickSet, &encode_tickset_body(set))
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

    /// Send our [`JoinRequest`] UP to the host `peer` when dialing a live match (Stage 3). Routed to
    /// exactly the host (the only peer a joiner is connected to at dial time); the host gates it
    /// ([`crate::server::may_admit_joiner`]) and replies with a [`Frame::RosterChange`] (admitted)
    /// or a [`Frame::Refuse`] (turned away). A no-link is a no-op surfaced as the join stalling.
    pub async fn send_join_request(&self, peer: EndpointId, req: &JoinRequest) {
        self.send_frame(peer, Frame::JoinRequest, &encode_join_request_body(req))
            .await;
    }

    /// Broadcast a roster change ([`Admission`]) DOWN to every connected client (Stage 3 mid-game
    /// join). One fan-out reaches incumbents (who schedule the change) AND the joiner (who builds its
    /// session from it), exactly as [`Self::broadcast_tickset`] fans the per-tick set; the change is
    /// identical for all, so every peer schedules the same rebuild at the same tick.
    pub async fn broadcast_roster_change(&self, adm: &Admission) {
        self.broadcast_frame(Frame::RosterChange, &encode_roster_change_body(adm))
            .await;
    }

    /// Tell a would-be joiner `peer` it is REFUSED, with `reason` (Stage 3). A loud turn-away — the
    /// joiner surfaces the reason rather than being silently dropped onto a mismatched crab.
    pub async fn send_refuse(&self, peer: EndpointId, reason: &str) {
        self.send_frame(peer, Frame::Refuse, reason.as_bytes()).await;
    }

    /// UNICAST a just-admitted joiner `peer` its OWN [`Admission`] (Stage 3). Separate from the
    /// broadcast [`Self::broadcast_roster_change`] so the joiner reads only the allocation meant for
    /// it — never a concurrent joiner's broadcast change (which would hand it the wrong PlayerId).
    pub async fn welcome_joiner(&self, peer: EndpointId, adm: &Admission) {
        self.send_frame(peer, Frame::Welcome, &encode_roster_change_body(adm))
            .await;
    }

    /// Frame `body` with `kind` + length and send it to one `peer` (the unicast analogue of
    /// [`Self::broadcast_frame`]). A send failure drops that link — the same policy as the broadcast
    /// path. No link to `peer` is a no-op (surfaced as the higher-level stall, not a panic).
    async fn send_frame(&self, peer: EndpointId, kind: Frame, body: &[u8]) {
        let send = {
            let links = self.links.lock().await;
            links.get(&peer).map(|l| l.send.clone())
        };
        let Some(send) = send else { return };
        let mut s = send.lock().await;
        if let Err(e) = write_frame(&mut s, kind, body).await {
            tracing::warn!(%peer, ?kind, "sending frame to peer failed: {e:#}");
            drop(s);
            let mut links = self.links.lock().await;
            if links.get(&peer).is_some_and(|l| Arc::ptr_eq(&l.send, &send)) {
                links.remove(&peer);
            }
        }
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
/// allocating. The two largest legitimate frames both fit comfortably: a full-roster [`Beat`]
/// (kind + start + count + weights digest + 256×32-byte ids ≈ 8 KiB) and a full-roster
/// [`crate::server::TickSet`] (256 inputs + 256 confirmed triples ≈ 7 KiB); 16 KiB is generous
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
            Frame::TickSet => PeerWire::TickSet(decode_tickset_body(body)?),
            Frame::JoinRequest => PeerWire::JoinRequest(decode_join_request_body(body)?),
            Frame::RosterChange => PeerWire::RosterChange(decode_roster_change_body(body)?),
            Frame::Refuse => PeerWire::Refuse(
                String::from_utf8(body.to_vec()).context("refuse frame body is not UTF-8")?,
            ),
            // Same payload as RosterChange — only the frame kind (unicast welcome vs broadcast
            // notice) distinguishes the joiner's own allocation from an incumbent's notice.
            Frame::Welcome => PeerWire::Welcome(decode_roster_change_body(body)?),
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
    use crate::sim::Input;

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
        assert_eq!(Frame::from_byte(2), Some(Frame::TickSet));
        assert_eq!(Frame::from_byte(3), Some(Frame::JoinRequest));
        assert_eq!(Frame::from_byte(4), Some(Frame::RosterChange));
        assert_eq!(Frame::from_byte(5), Some(Frame::Refuse));
        assert_eq!(Frame::from_byte(6), Some(Frame::Welcome));
        assert_eq!(Frame::from_byte(7), None);
        assert_eq!(Frame::from_byte(0xff), None);
    }

    #[test]
    fn join_request_wire_roundtrips() {
        let req = JoinRequest { weights_digest: 0xdead_beef_cafe_f00d, asset_digest: 0x0102_0304 };
        assert_eq!(decode_join_request_body(&encode_join_request_body(&req)).unwrap(), req);
        // A short body is a loud error, not a guessed-at request.
        assert!(decode_join_request_body(&[0u8; 8]).is_err());
        // Trailing bytes too (a longer/mixed-version frame must not be silently truncated).
        assert!(decode_join_request_body(&[0u8; 17]).is_err());
    }

    #[test]
    fn roster_change_wire_roundtrips() {
        use crate::sim::PlayerId;
        // The empty-roster edge and a multi-member roster both round-trip byte-for-byte.
        for roster in [vec![], vec![PlayerId(0), PlayerId(1), PlayerId(2)]] {
            let adm = Admission { pid: PlayerId(2), effective_tick: 1_234_567, roster };
            assert_eq!(decode_roster_change_body(&encode_roster_change_body(&adm)).unwrap(), adm);
        }
        // Truncation (a roster_len claiming more pids than present) is a loud error.
        let mut body = encode_roster_change_body(&Admission {
            pid: PlayerId(0),
            effective_tick: 7,
            roster: vec![PlayerId(0), PlayerId(1)],
        });
        body.pop(); // drop the last pid byte → count says 2 but only 1 present
        assert!(decode_roster_change_body(&body).is_err());
    }

    #[test]
    fn tickset_wire_roundtrips() {
        use crate::sim::PlayerId;
        use std::collections::BTreeMap;
        // A multi-player set with a partial confirmed map (the common case: only some clients'
        // hashes advanced this set) round-trips byte-for-byte, including the empty-confirmed case.
        for confirmed in [
            BTreeMap::from([(PlayerId(1), Confirmed { tick: 12, hash: 0xfeed })]),
            BTreeMap::new(),
        ] {
            let set = TickSet {
                apply_tick: 99,
                inputs: BTreeMap::from([
                    (PlayerId(0), Input::from_axes(0.5, -0.5)),
                    (PlayerId(1), Input::from_axes(-1.0, 0.25)),
                ]),
                confirmed,
            };
            let body = encode_tickset_body(&set);
            assert_eq!(decode_tickset_body(&body).unwrap(), set);
        }
    }

    #[test]
    fn truncated_tickset_is_an_error_not_a_short_set() {
        use crate::sim::PlayerId;
        use std::collections::BTreeMap;
        let set = TickSet {
            apply_tick: 1,
            inputs: BTreeMap::from([(PlayerId(0), Input::from_axes(1.0, 0.0))]),
            confirmed: BTreeMap::new(),
        };
        let body = encode_tickset_body(&set);
        // Lop off the last byte: decode must fail loudly rather than yield a corrupt/short set.
        assert!(decode_tickset_body(&body[..body.len() - 1]).is_err());
    }
}
