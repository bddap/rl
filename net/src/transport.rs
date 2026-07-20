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
use crate::client::{PilotIntent, TickMsg};
use crate::membership::{self, Beat};
use crate::server::{Admission, AdmissionRefusal, JoinRequest, Refusal};
use crate::sim::PlayerId;
use crate::snapshot::CoreSnapshot;

// v15: state frames (snapshot/articulation) moved off the reliable bi-stream onto QUIC
// datagrams (rl#259), and the stale "lockstep" vocabulary retired with the same bump (rl#244).
// v16: articulation vehicle poses carry a kind byte (rl#260).
// v17: world re-based to rig scale — UNIT 1000→100_000 grid/m and every constant's
// physical meaning changed with it (rl#256); absolute wire positions are incompatible.
// v18: beats + join requests carry the effective-plant digest (arena/bake/friction,
// rl#286); the widened frames are incompatible.
pub const ALPN: &[u8] = b"bddap/rl-game/hostauth/18";

pub const SERVICE_NAME: &str = "bddap-rl-game";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum Frame {
    Tick = 0,
    /// A [`Beat`]: heartbeat + roster advertisement for the formation barrier.
    Beat = 1,
    JoinRequest = 3,
    Refuse = 5,
    Welcome = 6,
    Snapshot = 7,
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

    /// State frames ride unreliable unordered QUIC datagrams; everything else (inputs,
    /// formation, admission) stays on the reliable ordered bi-stream, where a retransmit
    /// stall is the price of ordering. ONE source for the routing: senders const-assert the
    /// path at monomorphization, and each receive path rejects the other's kinds, so a kind
    /// can never ride both (rl#259).
    const fn via_datagram(self) -> bool {
        matches!(self, Frame::Snapshot | Frame::Articulation)
    }
}

#[derive(Debug, Clone)]
pub enum PeerWire {
    Tick(TickMsg),
    Beat(Beat),
    /// A would-be joiner's credentials, received by the host on a live-match dial.
    JoinRequest(JoinRequest),
    Refuse(Refusal),
    Welcome(Admission),
    Snapshot(CoreSnapshot),
    Articulation(CrabArticulation),
}

const IN_LEN: usize = crate::sim::Input::WIRE_LEN;
const OFF_INPUT: usize = 8; // after issue_tick(8)
const OFF_PILOT: usize = OFF_INPUT + IN_LEN;
const PILOT_LEN: usize = 1 + 4 + 12 + 4 + 4 + 4 + 1;
const TICKMSG_LEN: usize = OFF_PILOT + PILOT_LEN;

pub(crate) trait Codec: Sized {
    const KIND: Frame;
    type Bytes: AsRef<[u8]>;
    fn encode(&self) -> Self::Bytes;
    fn decode(body: &[u8]) -> Result<Self>;
}

/// A tick-stamped full-state frame — safe to LOSE (the next tick's frame supersedes it) and
/// safe to REORDER (the receiver drops anything at or below the newest tick it delivered) —
/// which is what qualifies it for the datagram path ([`Frame::via_datagram`]). `tick` is the
/// frame's sim tick: it keys both fragment reassembly and the receiver's stale-drop, and the
/// sim tick is monotone even across an in-round RESTART, so "newest tick wins" is always
/// correct within a connection's lifetime.
pub(crate) trait StateCodec: Codec {
    fn tick(&self) -> u64;
}

impl Codec for TickMsg {
    const KIND: Frame = Frame::Tick;
    type Bytes = [u8; TICKMSG_LEN];

    fn encode(&self) -> [u8; TICKMSG_LEN] {
        let mut b = [0u8; TICKMSG_LEN];
        b[0..OFF_INPUT].copy_from_slice(&self.issue_tick.to_le_bytes());
        b[OFF_INPUT..OFF_PILOT].copy_from_slice(&self.input.to_bytes());
        if let Some(p) = &self.pilot {
            let w = &mut b[OFF_PILOT..];
            w[0] = p.kind.wire_byte();
            w[1..5].copy_from_slice(&p.throttle_trim.to_le_bytes());
            for (i, t) in p.thrust.iter().enumerate() {
                w[5 + 4 * i..9 + 4 * i].copy_from_slice(&t.to_le_bytes());
            }
            w[17..21].copy_from_slice(&p.pitch.to_le_bytes());
            w[21..25].copy_from_slice(&p.roll.to_le_bytes());
            w[25..29].copy_from_slice(&p.yaw.to_le_bytes());
            w[29] = p.match_velocity as u8;
        }
        b
    }

    fn decode(body: &[u8]) -> Result<Self> {
        let b: &[u8; TICKMSG_LEN] = body.try_into().map_err(|_| {
            anyhow::anyhow!("tick frame body is {} B, want {TICKMSG_LEN}", body.len())
        })?;
        let p = &b[OFF_PILOT..];
        let f = |off: usize| f32::from_le_bytes(p[off..off + 4].try_into().unwrap());
        let pilot = match p[0] {
            0 => {
                anyhow::ensure!(
                    p[1..].iter().all(|&x| x == 0),
                    "on-foot tick frame has nonzero pilot-intent bytes"
                );
                None
            }
            b => {
                let kind = crab_world::vehicle::VehicleKind::from_wire_byte(b)
                    .with_context(|| format!("unknown pilot-intent kind byte {b:#x}"))?;
                let match_velocity = match p[29] {
                    0 => false,
                    1 => true,
                    x => anyhow::bail!("pilot-intent match_velocity flag is {x}, want 0/1"),
                };
                Some(PilotIntent {
                    kind,
                    throttle_trim: f(1),
                    thrust: [f(5), f(9), f(13)],
                    pitch: f(17),
                    roll: f(21),
                    yaw: f(25),
                    match_velocity,
                })
            }
        };
        Ok(TickMsg {
            issue_tick: u64::from_le_bytes(b[0..OFF_INPUT].try_into().unwrap()),
            input: crate::sim::Input::from_bytes(b[OFF_INPUT..OFF_PILOT].try_into().unwrap()),
            pilot,
        })
    }
}

fn take<'a>(r: &mut &'a [u8], n: usize, what: &str) -> Result<&'a [u8]> {
    anyhow::ensure!(r.len() >= n, "frame truncated reading {what}");
    let (head, tail) = r.split_at(n);
    *r = tail;
    Ok(head)
}

impl Codec for CoreSnapshot {
    const KIND: Frame = Frame::Snapshot;
    type Bytes = Vec<u8>;

    fn encode(&self) -> Vec<u8> {
        self.to_bytes()
    }

    fn decode(body: &[u8]) -> Result<Self> {
        CoreSnapshot::from_bytes(body).map_err(|e| anyhow::anyhow!("decoding snapshot frame: {e}"))
    }
}

impl StateCodec for CoreSnapshot {
    fn tick(&self) -> u64 {
        self.tick
    }
}

impl Codec for CrabArticulation {
    const KIND: Frame = Frame::Articulation;
    type Bytes = Vec<u8>;

    fn encode(&self) -> Vec<u8> {
        self.to_bytes()
    }

    fn decode(body: &[u8]) -> Result<Self> {
        CrabArticulation::from_bytes(body)
            .map_err(|e| anyhow::anyhow!("decoding articulation frame: {e}"))
    }
}

impl StateCodec for CrabArticulation {
    fn tick(&self) -> u64 {
        self.tick
    }
}

impl Codec for Beat {
    const KIND: Frame = Frame::Beat;
    type Bytes = Vec<u8>;

    fn encode(&self) -> Vec<u8> {
        membership::encode_beat(self)
    }

    fn decode(body: &[u8]) -> Result<Self> {
        membership::decode_beat(body)
    }
}

impl Codec for JoinRequest {
    const KIND: Frame = Frame::JoinRequest;
    type Bytes = [u8; 17];

    fn encode(&self) -> [u8; 17] {
        let mut out = [0u8; 17];
        out[0..8].copy_from_slice(&self.stamp.body_digest.to_le_bytes());
        out[8..16].copy_from_slice(&self.stamp.plant_digest.to_le_bytes());
        out[16] = self.stamp.crab_count;
        out
    }

    fn decode(body: &[u8]) -> Result<Self> {
        let mut r = body;
        let body_digest = u64::from_le_bytes(take(&mut r, 8, "body_digest")?.try_into().unwrap());
        let plant_digest = u64::from_le_bytes(take(&mut r, 8, "plant_digest")?.try_into().unwrap());
        let crab_count = take(&mut r, 1, "crab_count")?[0];
        anyhow::ensure!(
            r.is_empty(),
            "join-request frame has {} trailing bytes",
            r.len()
        );
        Ok(JoinRequest {
            stamp: crate::SyncStamp {
                body_digest,
                plant_digest,
                crab_count,
            },
        })
    }
}

impl Codec for Admission {
    const KIND: Frame = Frame::Welcome;
    type Bytes = Vec<u8>;

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

impl Codec for Refusal {
    const KIND: Frame = Frame::Refuse;
    type Bytes = Vec<u8>;

    fn encode(&self) -> Vec<u8> {
        match self {
            Refusal::Admission(AdmissionRefusal::BodyMismatch { host, joiner }) => {
                let mut b = vec![0u8];
                b.extend_from_slice(&host.to_le_bytes());
                b.extend_from_slice(&joiner.to_le_bytes());
                b
            }
            Refusal::Admission(AdmissionRefusal::CrabCountMismatch { host, joiner }) => {
                vec![1, *host, *joiner]
            }
            Refusal::Departed => vec![2],
            Refusal::Forming => vec![3],
            Refusal::Admission(AdmissionRefusal::PlantMismatch { host, joiner }) => {
                let mut b = vec![4u8];
                b.extend_from_slice(&host.to_le_bytes());
                b.extend_from_slice(&joiner.to_le_bytes());
                b
            }
        }
    }

    fn decode(body: &[u8]) -> Result<Self> {
        let mut r = body;
        let verdict = match take(&mut r, 1, "refusal tag")?[0] {
            0 => Refusal::Admission(AdmissionRefusal::BodyMismatch {
                host: u64::from_le_bytes(take(&mut r, 8, "host digest")?.try_into().unwrap()),
                joiner: u64::from_le_bytes(take(&mut r, 8, "joiner digest")?.try_into().unwrap()),
            }),
            1 => Refusal::Admission(AdmissionRefusal::CrabCountMismatch {
                host: take(&mut r, 1, "host crab count")?[0],
                joiner: take(&mut r, 1, "joiner crab count")?[0],
            }),
            2 => Refusal::Departed,
            3 => Refusal::Forming,
            4 => Refusal::Admission(AdmissionRefusal::PlantMismatch {
                host: u64::from_le_bytes(take(&mut r, 8, "host plant digest")?.try_into().unwrap()),
                joiner: u64::from_le_bytes(
                    take(&mut r, 8, "joiner plant digest")?.try_into().unwrap(),
                ),
            }),
            x => anyhow::bail!("unknown refusal tag {x:#x}"),
        };
        anyhow::ensure!(r.is_empty(), "refuse frame has {} trailing bytes", r.len());
        Ok(verdict)
    }
}

#[derive(Debug, Clone)]
pub struct FromPeer {
    pub from: EndpointId,
    pub msg: PeerWire,
}

const ADDR_WAIT: Duration = Duration::from_secs(10);

const PUBLISH_SETTLE: Duration = Duration::from_millis(300);

pub async fn bind_endpoint() -> Result<(Endpoint, MdnsAddressLookup)> {
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

    publish_lan_addr(&endpoint, SERVICE_NAME).await?;
    Ok((endpoint, mdns))
}

pub(crate) async fn publish_lan_addr(endpoint: &Endpoint, service_name: &str) -> Result<()> {
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

struct OutFrame {
    kind: Frame,
    body: Arc<[u8]>,
}

type Link = Arc<mpsc::Sender<OutFrame>>;

type LinkId = std::sync::Weak<mpsc::Sender<OutFrame>>;

const OUT_QUEUE_FRAMES: usize = 256;

const WRITE_STALL_TIMEOUT: Duration = Duration::from_secs(10);

type Links = Arc<tokio::sync::Mutex<BTreeMap<EndpointId, PeerLink>>>;

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

// State frames ride datagrams so a lost/reordered packet costs ONE stale frame (superseded a
// tick later) instead of stalling every later message behind a retransmit — the rl#259
// "feels like tcp synchronization" jitter. A frame bigger than one datagram is split into
// tick-stamped fragments; a lost fragment drops that WHOLE frame (never blocks, never
// retransmits — the next tick's frame replaces it).
//
// Datagram layout: kind(1) ++ tick(8 LE) ++ frag_idx(1) ++ frag_count(1) ++ payload.
const DGRAM_HDR_LEN: usize = 11;
/// Sized so header + payload stays under QUIC's guaranteed datagram floor ("a little over a
/// kilobyte" before path-MTU discovery grows it) — a fragment can never hit `TooLarge`.
const DGRAM_FRAG_PAYLOAD: usize = 1024;
const DGRAM_MAX_FRAGS: usize = MAX_FRAME_LEN.div_ceil(DGRAM_FRAG_PAYLOAD);

fn state_datagrams(kind: Frame, tick: u64, body: &[u8]) -> Vec<bytes::Bytes> {
    let count = body.len().div_ceil(DGRAM_FRAG_PAYLOAD).max(1);
    debug_assert!(
        count <= DGRAM_MAX_FRAGS,
        "outbound {kind:?} frame is {} B, over the {MAX_FRAME_LEN} B cap every receiver enforces",
        body.len()
    );
    (0..count)
        .map(|i| {
            let chunk =
                &body[i * DGRAM_FRAG_PAYLOAD..((i + 1) * DGRAM_FRAG_PAYLOAD).min(body.len())];
            let mut d = Vec::with_capacity(DGRAM_HDR_LEN + chunk.len());
            d.push(kind as u8);
            d.extend_from_slice(&tick.to_le_bytes());
            d.push(i as u8);
            d.push(count as u8);
            d.extend_from_slice(chunk);
            bytes::Bytes::from(d)
        })
        .collect()
}

struct DgramFrag<'a> {
    kind: Frame,
    tick: u64,
    idx: u8,
    count: u8,
    payload: &'a [u8],
}

fn parse_state_datagram(d: &[u8]) -> Result<DgramFrag<'_>> {
    anyhow::ensure!(
        d.len() >= DGRAM_HDR_LEN,
        "datagram is {} B, too short for a fragment header",
        d.len()
    );
    let kind = Frame::from_byte(d[0])
        .with_context(|| format!("unknown datagram frame kind {:#x}", d[0]))?;
    anyhow::ensure!(
        kind.via_datagram(),
        "control frame {kind:?} arrived as a datagram"
    );
    let tick = u64::from_le_bytes(d[1..9].try_into().expect("length checked above"));
    let (idx, count) = (d[9], d[10]);
    anyhow::ensure!(
        idx < count && (count as usize) <= DGRAM_MAX_FRAGS,
        "datagram fragment {idx}/{count} is out of range"
    );
    let payload = &d[DGRAM_HDR_LEN..];
    let last = idx + 1 == count;
    anyhow::ensure!(
        if last {
            payload.len() <= DGRAM_FRAG_PAYLOAD
        } else {
            payload.len() == DGRAM_FRAG_PAYLOAD
        },
        "datagram fragment {idx}/{count} payload is {} B",
        payload.len()
    );
    Ok(DgramFrag {
        kind,
        tick,
        idx,
        count,
        payload,
    })
}

/// Reassembles ONE state-frame kind's fragments, delivering each complete body at most once
/// and strictly newer-than-last: the wire is unordered, so the receiver enforces monotonicity
/// (a stale frame is dropped, never delivered late), and a newer tick's first fragment
/// abandons any incomplete older assembly — loss costs one frame, never a stall.
#[derive(Default)]
struct StateAssembler {
    delivered: Option<u64>,
    tick: u64,
    frags: Vec<Option<Vec<u8>>>,
    have: usize,
}

impl StateAssembler {
    fn accept(&mut self, f: &DgramFrag) -> Result<Option<Vec<u8>>> {
        if self.delivered.is_some_and(|d| f.tick <= d) {
            return Ok(None);
        }
        if self.frags.is_empty() || self.tick != f.tick {
            if !self.frags.is_empty() && f.tick < self.tick {
                // Older than the frame mid-assembly — superseded before it completed.
                return Ok(None);
            }
            self.tick = f.tick;
            self.frags = vec![None; f.count as usize];
            self.have = 0;
        }
        anyhow::ensure!(
            self.frags.len() == f.count as usize,
            "tick {} re-announced with fragment count {} (was {})",
            f.tick,
            f.count,
            self.frags.len()
        );
        let slot = &mut self.frags[f.idx as usize];
        if slot.is_none() {
            self.have += 1;
        }
        *slot = Some(f.payload.to_vec());
        if self.have < self.frags.len() {
            return Ok(None);
        }
        self.delivered = Some(self.tick);
        let body = std::mem::take(&mut self.frags)
            .into_iter()
            .flat_map(|s| s.expect("all fragments present"))
            .collect();
        self.have = 0;
        Ok(Some(body))
    }
}

pub struct Session {
    endpoint: Endpoint,
    _router: Router,
    inbox: mpsc::Receiver<FromPeer>,
    inbox_tx: mpsc::Sender<FromPeer>,
    links: Links,
    discovery: tokio::task::JoinHandle<()>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.discovery.abort();
    }
}

impl Session {
    pub fn endpoint_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub async fn connected_peers(&self) -> Vec<EndpointId> {
        self.links.lock().await.keys().copied().collect()
    }

    pub fn local_addr(&self) -> iroh::EndpointAddr {
        self.endpoint.addr()
    }

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
            true,
        )
        .await
    }

    pub(crate) async fn send<M: Codec>(&self, peer: EndpointId, msg: &M) {
        const {
            assert!(
                !M::KIND.via_datagram(),
                "state frames go via broadcast_state"
            )
        }
        let bytes = msg.encode();
        self.send_frame(peer, M::KIND, bytes.as_ref().into()).await;
    }

    pub(crate) async fn broadcast<M: Codec>(&self, msg: &M) {
        const {
            assert!(
                !M::KIND.via_datagram(),
                "state frames go via broadcast_state"
            )
        }
        let bytes = msg.encode();
        self.broadcast_frame(M::KIND, bytes.as_ref().into()).await;
    }

    /// Broadcast a state frame over unreliable unordered datagrams — fire-and-forget: no
    /// retransmit, no ordering, no writer queue to wedge. Under congestion the QUIC buffer
    /// drops the OLDEST buffered datagram first, which for full-state frames is exactly
    /// right (stale state is worthless).
    pub(crate) async fn broadcast_state<M: StateCodec>(&self, msg: &M) {
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
        let frags = state_datagrams(M::KIND, msg.tick(), body);
        let links = self.links.lock().await;
        for link in links.values() {
            for frag in &frags {
                send_state_datagram(&link.conn, frag.clone());
            }
        }
    }

    pub async fn send_to(&self, peer: EndpointId, msg: &TickMsg) {
        self.send(peer, msg).await;
    }

    // The public faces of `broadcast_state` — `Codec`/`StateCodec` are crate-internal, so
    // out-of-crate callers (the game harnesses) get typed entry points instead.
    pub async fn broadcast_snapshot(&self, snapshot: &CoreSnapshot) {
        self.broadcast_state(snapshot).await;
    }

    pub async fn broadcast_articulation(&self, articulation: &CrabArticulation) {
        self.broadcast_state(articulation).await;
    }

    async fn send_frame(&self, peer: EndpointId, kind: Frame, body: Arc<[u8]>) {
        let wedged = {
            let links = self.links.lock().await;
            let Some(link) = links.get(&peer) else { return };
            match link.send.try_send(OutFrame { kind, body }) {
                Err(mpsc::error::TrySendError::Full(_)) => Some(Arc::downgrade(&link.send)),
                _ => None,
            }
        };
        if let Some(link_id) = wedged {
            tracing::warn!(%peer, ?kind, "peer outbound queue full (not draining) — dropping link");
            drop_if_same(&self.links, peer, &link_id).await;
        }
    }

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

    pub fn try_recv(&mut self) -> Option<FromPeer> {
        self.inbox.try_recv().ok()
    }

    pub async fn recv(&mut self) -> Option<FromPeer> {
        self.inbox.recv().await
    }

    pub async fn shutdown(self) {
        self.endpoint.close().await;
    }
}

pub async fn start_session() -> Result<Session> {
    let (endpoint, mdns) = bind_endpoint().await?;
    let my_id = endpoint.id();

    let (inbox_tx, inbox_rx) = mpsc::channel(256);
    let links: Links = Arc::new(tokio::sync::Mutex::new(BTreeMap::new()));

    let handler = GameProto {
        my_id,
        inbox: inbox_tx.clone(),
        links: links.clone(),
    };
    let router = Router::builder(endpoint.clone())
        .accept(ALPN, handler)
        .spawn();

    let inbox_tx_for_session = inbox_tx.clone();

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
                    if my_id.as_bytes() >= peer.as_bytes() {
                        continue;
                    }
                    if links.lock().await.contains_key(&peer) {
                        continue;
                    }
                    match endpoint.connect(peer, ALPN).await {
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

    Ok(Session {
        endpoint,
        _router: router,
        inbox: inbox_rx,
        inbox_tx: inbox_tx_for_session,
        links,
        discovery,
    })
}

#[derive(Clone, Debug)]
struct GameProto {
    my_id: EndpointId,
    inbox: mpsc::Sender<FromPeer>,
    links: Links,
}

impl ProtocolHandler for GameProto {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        if let Err(e) = wire_connection(
            self.my_id,
            connection,
            self.inbox.clone(),
            self.links.clone(),
            false,
        )
        .await
        {
            tracing::warn!("accepting game connection failed: {e:#}");
        }
        Ok(())
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
        let mut links = links.lock().await;
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
    let reader_id = link_id.clone();
    let reader_inbox = inbox.clone();
    tokio::spawn(async move {
        // WARN, not debug: read_loop returns Ok on every normal ending (clean EOF, session drop),
        // so an Err here is a real protocol violation — a mis-framed/unknown/truncated frame (e.g.
        // an ALPN-matched build with a drifted codec) — and must be visible, not a silent link
        // drop the joiner mis-reads as "host unreachable" ([[silent-fallback-antipattern]]).
        if let Err(e) = read_loop(recv, peer, reader_inbox).await {
            tracing::warn!(%peer, "peer read loop ended on a protocol violation: {e:#}");
        }
        drop_if_same(&links_for_reader, peer, &reader_id).await;
    });

    let links_for_dgram = links.clone();
    tokio::spawn(async move {
        // Same loudness contract as the stream reader: Ok is every normal ending (the
        // connection closed), Err is a protocol violation. Dropping the link on either keeps
        // the failure mode LOUD — a peer whose state stopped flowing must read as departed,
        // never as a silently frozen world.
        if let Err(e) = datagram_loop(conn, peer, inbox).await {
            tracing::warn!(%peer, "peer datagram loop ended on a protocol violation: {e:#}");
        }
        drop_if_same(&links_for_dgram, peer, &link_id).await;
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

const MAX_FRAME_LEN: usize = 16 * 1024;

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

fn decode_peer_wire(kind: Frame, body: &[u8]) -> Result<PeerWire> {
    Ok(match kind {
        Frame::Tick => PeerWire::Tick(TickMsg::decode(body)?),
        Frame::Beat => PeerWire::Beat(Beat::decode(body)?),
        Frame::JoinRequest => PeerWire::JoinRequest(JoinRequest::decode(body)?),
        Frame::Refuse => PeerWire::Refuse(Refusal::decode(body)?),
        Frame::Welcome => PeerWire::Welcome(Admission::decode(body)?),
        Frame::Snapshot => PeerWire::Snapshot(CoreSnapshot::decode(body)?),
        Frame::Articulation => PeerWire::Articulation(CrabArticulation::decode(body)?),
    })
}

async fn drop_if_same(links: &Links, id: EndpointId, failed: &LinkId) {
    let mut links = links.lock().await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Input;

    #[test]
    fn tick_body_wire_roundtrips() {
        let on_foot = TickMsg {
            issue_tick: 1234,
            input: Input::from_axes(0.5, -0.25),
            pilot: None,
        };
        assert_eq!(
            TickMsg::decode(TickMsg::encode(&on_foot).as_ref()).unwrap(),
            on_foot
        );
        let piloting = TickMsg {
            pilot: Some(PilotIntent {
                kind: crab_world::vehicle::VehicleKind::Ship,
                throttle_trim: -0.5,
                thrust: [0.25, -1.0, 1.0],
                pitch: 0.125,
                roll: -0.75,
                yaw: 1.0,
                match_velocity: true,
            }),
            ..on_foot
        };
        assert_eq!(
            TickMsg::decode(TickMsg::encode(&piloting).as_ref()).unwrap(),
            piloting
        );
    }

    #[test]
    fn tick_pilot_block_is_strict() {
        let on_foot = TickMsg {
            issue_tick: 7,
            input: Input::from_axes(0.0, 1.0),
            pilot: None,
        };
        let mut b = TickMsg::encode(&on_foot);
        b[TICKMSG_LEN - 1] = 1;
        assert!(
            TickMsg::decode(&b).is_err(),
            "nonzero on-foot tail rejected"
        );
        let mut b = TickMsg::encode(&on_foot);
        b[OFF_PILOT] = 3;
        assert!(TickMsg::decode(&b).is_err(), "unknown kind byte rejected");
        assert!(
            TickMsg::decode(&b[..OFF_PILOT]).is_err(),
            "short frame rejected"
        );
    }

    #[test]
    fn refusal_wire_roundtrips_typed() {
        for verdict in [
            Refusal::Admission(AdmissionRefusal::BodyMismatch {
                host: 0x1122_3344_5566_7788,
                joiner: 0x99aa_bbcc_ddee_ff00,
            }),
            Refusal::Admission(AdmissionRefusal::CrabCountMismatch { host: 3, joiner: 1 }),
            Refusal::Admission(AdmissionRefusal::PlantMismatch {
                host: 0x7e44_a100_0bad_5eed,
                joiner: 0x7e44_a100_0bad_5eee,
            }),
            Refusal::Departed,
            Refusal::Forming,
        ] {
            assert_eq!(
                Refusal::decode(Refusal::encode(&verdict).as_ref()).unwrap(),
                verdict
            );
        }
        assert!(Refusal::decode(&[9]).is_err());
        assert!(Refusal::decode(&[2, 0]).is_err());
        assert!(Refusal::decode(b"host gone").is_err());
    }

    /// Push one fragment through an assembler, unwrapping the protocol-violation layer.
    fn feed(asm: &mut StateAssembler, d: &bytes::Bytes) -> Option<Vec<u8>> {
        asm.accept(&parse_state_datagram(d).expect("well-formed fragment"))
            .expect("well-formed fragment sequence")
    }

    #[test]
    fn state_datagrams_fragment_and_reassemble() {
        let mut asm = StateAssembler::default();

        // A small body: one fragment, delivered immediately.
        let small = b"tiny state".to_vec();
        let frags = state_datagrams(Frame::Snapshot, 3, &small);
        assert_eq!(frags.len(), 1);
        assert_eq!(feed(&mut asm, &frags[0]), Some(small));

        // A multi-fragment body reassembles exactly, even arriving out of order.
        let big: Vec<u8> = (0..2 * DGRAM_FRAG_PAYLOAD + 100).map(|i| i as u8).collect();
        let frags = state_datagrams(Frame::Snapshot, 4, &big);
        assert_eq!(frags.len(), 3);
        assert_eq!(feed(&mut asm, &frags[2]), None);
        assert_eq!(feed(&mut asm, &frags[0]), None);
        assert_eq!(feed(&mut asm, &frags[1]), Some(big));
    }

    #[test]
    fn assembler_drops_stale_and_duplicate_ticks() {
        let mut asm = StateAssembler::default();
        let newer = state_datagrams(Frame::Snapshot, 10, b"ten");
        let stale = state_datagrams(Frame::Snapshot, 9, b"nine");
        assert_eq!(feed(&mut asm, &newer[0]), Some(b"ten".to_vec()));
        assert_eq!(feed(&mut asm, &stale[0]), None, "older than delivered");
        assert_eq!(feed(&mut asm, &newer[0]), None, "duplicate of delivered");
    }

    #[test]
    fn assembler_newer_tick_preempts_incomplete_older() {
        // Tick 5 loses a fragment (the wire may drop any datagram); tick 6 must still deliver —
        // loss costs one frame, never a stall. A late tick-5 straggler after 6 is stale.
        let mut asm = StateAssembler::default();
        let body5: Vec<u8> = vec![5; DGRAM_FRAG_PAYLOAD + 1];
        let body6: Vec<u8> = vec![6; DGRAM_FRAG_PAYLOAD + 1];
        let f5 = state_datagrams(Frame::Articulation, 5, &body5);
        let f6 = state_datagrams(Frame::Articulation, 6, &body6);
        assert_eq!(feed(&mut asm, &f5[0]), None);
        assert_eq!(
            feed(&mut asm, &f6[1]),
            None,
            "preempts the incomplete tick 5"
        );
        assert_eq!(feed(&mut asm, &f5[1]), None, "tick-5 straggler dropped");
        assert_eq!(feed(&mut asm, &f6[0]), Some(body6));
    }

    #[test]
    fn malformed_datagrams_are_rejected() {
        // Header too short.
        assert!(parse_state_datagram(&[Frame::Snapshot as u8; DGRAM_HDR_LEN - 1]).is_err());
        // A control kind must never arrive as a datagram (and vice versa on the stream —
        // Frame::via_datagram is the single routing source both receive paths enforce).
        let mut d = state_datagrams(Frame::Snapshot, 1, b"x")[0].to_vec();
        d[0] = Frame::Tick as u8;
        assert!(parse_state_datagram(&d).is_err());
        // Fragment index out of range.
        let mut d = state_datagrams(Frame::Snapshot, 1, b"x")[0].to_vec();
        d[9] = 1; // idx == count
        assert!(parse_state_datagram(&d).is_err());
        // A non-final fragment must carry a full payload (else reassembly is ambiguous).
        let big: Vec<u8> = vec![0; DGRAM_FRAG_PAYLOAD + 1];
        let mut d = state_datagrams(Frame::Snapshot, 1, &big)[0].to_vec();
        d.pop();
        assert!(parse_state_datagram(&d).is_err());
        // Fragment count over the frame cap.
        let mut d = state_datagrams(Frame::Snapshot, 1, &big)[0].to_vec();
        d[10] = u8::MAX;
        assert!(parse_state_datagram(&d).is_err());
        // The same tick re-announced with a different fragment count is a protocol violation.
        let mut asm = StateAssembler::default();
        let frags = state_datagrams(Frame::Snapshot, 7, &big);
        assert_eq!(feed(&mut asm, &frags[0]), None);
        let conflicting = state_datagrams(Frame::Snapshot, 7, b"short");
        assert!(
            asm.accept(&parse_state_datagram(&conflicting[0]).unwrap())
                .is_err()
        );
    }

    #[test]
    fn state_frames_and_only_state_frames_ride_datagrams() {
        for kind in [
            Frame::Tick,
            Frame::Beat,
            Frame::JoinRequest,
            Frame::Refuse,
            Frame::Welcome,
        ] {
            assert!(!kind.via_datagram(), "{kind:?} is control traffic");
        }
        for kind in [Frame::Snapshot, Frame::Articulation] {
            assert!(kind.via_datagram(), "{kind:?} is tick-stamped full state");
        }
    }

    #[test]
    fn frame_kind_byte_roundtrips() {
        assert_eq!(Frame::from_byte(0), Some(Frame::Tick));
        assert_eq!(Frame::from_byte(1), Some(Frame::Beat));
        assert_eq!(Frame::from_byte(2), None);
        assert_eq!(Frame::from_byte(3), Some(Frame::JoinRequest));
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
        use crate::articulation::{CrabArticulation, CrabFrame, PartTransform, VehiclePoseWire};
        let art = CrabArticulation {
            tick: 909,
            crabs: vec![CrabFrame {
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
                brain_label: "mlp512x3 @deadbeef".to_string(),
            }],
            vehicles: vec![VehiclePoseWire {
                pilot: 1,
                kind: crab_world::vehicle::VehicleKind::Ship,
                pos: [2.0, 5.5, -1.0],
                rot: [
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                ],
            }],
        };
        let body = CrabArticulation::encode(&art);
        assert_eq!(CrabArticulation::decode(&body).unwrap(), art);
        assert!(CrabArticulation::decode(&body[..body.len() - 1]).is_err());
    }

    #[test]
    fn snapshot_wire_roundtrips() {
        use std::collections::BTreeMap;

        use crate::sim::{CrabPose, Externals, Input, PlayerId, Pos, Sim};
        let mut sim = Sim::new(3, &[PlayerId(0), PlayerId(1)]);
        let posed = vec![CrabPose {
            pos: Pos { x: 77, z: -88 },
            yaw: 5,
            claws: Vec::new(),
        }];
        let inputs = BTreeMap::from([
            (PlayerId(0), Input::default()),
            (PlayerId(1), Input::default()),
        ]);
        sim.step(&inputs, Externals::crabs_only(&posed));
        let snap = sim.core_snapshot();
        let body = CoreSnapshot::encode(&snap);
        assert_eq!(CoreSnapshot::decode(&body).unwrap(), snap);
        assert!(CoreSnapshot::decode(&body[..body.len() - 1]).is_err());
    }

    #[test]
    fn join_request_wire_roundtrips() {
        let req = JoinRequest {
            stamp: crate::SyncStamp {
                body_digest: 0xdead_beef_cafe_f00d,
                plant_digest: 0x7e44_a100_0bad_5eed,
                crab_count: 3,
            },
        };
        assert_eq!(
            JoinRequest::decode(JoinRequest::encode(&req).as_ref()).unwrap(),
            req
        );
        assert!(JoinRequest::decode(&[0u8; 4]).is_err());
        // Trailing bytes too — notably a pre-rl#286 9-byte (body|count) frame must be a loud
        // version-skew error, never silently defaulted to plant digest 0.
        assert!(JoinRequest::decode(&[0u8; 9]).is_err());
        assert!(JoinRequest::decode(&[0u8; 18]).is_err());
    }

    #[test]
    fn welcome_wire_roundtrips() {
        use crate::sim::PlayerId;
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
        let mut body = Admission::encode(&Admission {
            pid: PlayerId(0),
            effective_tick: 7,
            roster: vec![PlayerId(0), PlayerId(1)],
        });
        body.pop();
        assert!(Admission::decode(&body).is_err());
    }
}
