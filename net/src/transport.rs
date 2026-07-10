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

// "lockstep" is stale vocabulary (rl#151 deleted that model) but the string is
// compat-breaking, so rename it (e.g. `hostauth/15`) with the next protocol bump
// instead of burning a rev on it (rl#244).
pub const ALPN: &[u8] = b"bddap/rl-game/lockstep/14";

pub const SERVICE_NAME: &str = "bddap-rl-game";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

fn vehicle_kind_byte(kind: crab_world::vehicle::VehicleKind) -> u8 {
    match kind {
        crab_world::vehicle::VehicleKind::Plane => 1,
        crab_world::vehicle::VehicleKind::Ship => 2,
    }
}

fn vehicle_kind_from_byte(b: u8) -> Option<crab_world::vehicle::VehicleKind> {
    match b {
        1 => Some(crab_world::vehicle::VehicleKind::Plane),
        2 => Some(crab_world::vehicle::VehicleKind::Ship),
        _ => None,
    }
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
            w[0] = vehicle_kind_byte(p.kind);
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
                let kind = vehicle_kind_from_byte(b)
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
    type Bytes = [u8; 9];

    fn encode(&self) -> [u8; 9] {
        let mut out = [0u8; 9];
        out[0..8].copy_from_slice(&self.asset_digest.to_le_bytes());
        out[8] = self.crab_count;
        out
    }

    fn decode(body: &[u8]) -> Result<Self> {
        let mut r = body;
        let asset_digest = u64::from_le_bytes(take(&mut r, 8, "asset_digest")?.try_into().unwrap());
        let crab_count = take(&mut r, 1, "crab_count")?[0];
        anyhow::ensure!(
            r.is_empty(),
            "join-request frame has {} trailing bytes",
            r.len()
        );
        Ok(JoinRequest {
            asset_digest,
            crab_count,
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
            Refusal::Admission(AdmissionRefusal::AssetsMismatch { host, joiner }) => {
                let mut b = vec![0u8];
                b.extend_from_slice(&host.to_le_bytes());
                b.extend_from_slice(&joiner.to_le_bytes());
                b
            }
            Refusal::Admission(AdmissionRefusal::CrabCountMismatch { host, joiner }) => {
                vec![1, *host, *joiner]
            }
            Refusal::Departed => vec![2],
        }
    }

    fn decode(body: &[u8]) -> Result<Self> {
        let mut r = body;
        let verdict = match take(&mut r, 1, "refusal tag")?[0] {
            0 => Refusal::Admission(AdmissionRefusal::AssetsMismatch {
                host: u64::from_le_bytes(take(&mut r, 8, "host digest")?.try_into().unwrap()),
                joiner: u64::from_le_bytes(take(&mut r, 8, "joiner digest")?.try_into().unwrap()),
            }),
            1 => Refusal::Admission(AdmissionRefusal::CrabCountMismatch {
                host: take(&mut r, 1, "host crab count")?[0],
                joiner: take(&mut r, 1, "joiner crab count")?[0],
            }),
            2 => Refusal::Departed,
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
        )
        .await
    }

    pub(crate) async fn send<M: Codec>(&self, peer: EndpointId, msg: &M) {
        let bytes = msg.encode();
        self.send_frame(peer, M::KIND, bytes.as_ref().into()).await;
    }

    pub(crate) async fn broadcast<M: Codec>(&self, msg: &M) {
        let bytes = msg.encode();
        self.broadcast_frame(M::KIND, bytes.as_ref().into()).await;
    }

    pub async fn send_to(&self, peer: EndpointId, msg: &TickMsg) {
        self.send(peer, msg).await;
    }

    pub async fn broadcast_snapshot(&self, snapshot: &CoreSnapshot) {
        self.broadcast(snapshot).await;
    }

    pub async fn broadcast_articulation(&self, articulation: &CrabArticulation) {
        self.broadcast(articulation).await;
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
        )
        .await
        {
            tracing::warn!("accepting game connection failed: {e:#}");
        }
        Ok(())
    }
}

const HELLO: u8 = 0xA5;

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
    links.lock().await.insert(peer, PeerLink { send: tx });

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
        drop_if_same(&links_for_reader, peer, &link_id).await;
    });
    Ok(())
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
        let body = &buf[1..];
        let msg = match kind {
            Frame::Tick => PeerWire::Tick(TickMsg::decode(body)?),
            Frame::Beat => PeerWire::Beat(Beat::decode(body)?),
            Frame::JoinRequest => PeerWire::JoinRequest(JoinRequest::decode(body)?),
            Frame::Refuse => PeerWire::Refuse(Refusal::decode(body)?),
            Frame::Welcome => PeerWire::Welcome(Admission::decode(body)?),
            Frame::Snapshot => PeerWire::Snapshot(CoreSnapshot::decode(body)?),
            Frame::Articulation => PeerWire::Articulation(CrabArticulation::decode(body)?),
        };
        if inbox.send(FromPeer { from: peer, msg }).await.is_err() {
            return Ok(());
        }
    }
}

async fn drop_if_same(links: &Links, id: EndpointId, failed: &LinkId) {
    let mut links = links.lock().await;
    if links
        .get(&id)
        .is_some_and(|l| std::sync::Weak::ptr_eq(&Arc::downgrade(&l.send), failed))
    {
        links.remove(&id);
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
            Refusal::Admission(AdmissionRefusal::AssetsMismatch {
                host: 0x1122_3344_5566_7788,
                joiner: 0x99aa_bbcc_ddee_ff00,
            }),
            Refusal::Admission(AdmissionRefusal::CrabCountMismatch { host: 3, joiner: 1 }),
            Refusal::Departed,
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
        use crate::articulation::{
            CrabArticulation, CrabFrame, PartTransform, ReposeWire, VehiclePoseWire,
        };
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
                repose: Some(ReposeWire {
                    shift: [1.0, 0.0, -2.0],
                }),
                brain_label: "mlp512x3 @deadbeef".to_string(),
            }],
            arena_anchor: [4.25, 0.0, -0.5],
            vehicles: vec![VehiclePoseWire {
                pilot: 1,
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
        use crate::sim::{PlayerId, Pos, Sim};
        let mut sim = Sim::new(3, &[PlayerId(0), PlayerId(1)]);
        sim.set_external_crab_pose(0, Pos { x: 77, z: -88 }, 5);
        let snap = sim.core_snapshot();
        let body = CoreSnapshot::encode(&snap);
        assert_eq!(CoreSnapshot::decode(&body).unwrap(), snap);
        assert!(CoreSnapshot::decode(&body[..body.len() - 1]).is_err());
    }

    #[test]
    fn join_request_wire_roundtrips() {
        let req = JoinRequest {
            asset_digest: 0xdead_beef_cafe_f00d,
            crab_count: 3,
        };
        assert_eq!(
            JoinRequest::decode(JoinRequest::encode(&req).as_ref()).unwrap(),
            req
        );
        assert!(JoinRequest::decode(&[0u8; 4]).is_err());
        // Trailing bytes too — notably a pre-rl#206 16-byte (weights|assets) frame must be a loud
        // version-skew error, never silently truncated to just its first field.
        assert!(JoinRequest::decode(&[0u8; 16]).is_err());
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
