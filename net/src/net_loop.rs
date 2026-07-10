use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::Result;
use iroh::EndpointId;

use crate::articulation::CrabArticulation;
use crate::client::{ClientSim, PeerMsg, TickMsg};
use crate::formation::{Formation, LobbyControl, early_peer_msgs, form_match};
use crate::server::{Admission, JoinRequest, Refusal, Server, may_admit_joiner};
use crate::sim::PlayerId;
use crate::snapshot::CoreSnapshot;
use crate::telemetry::{TelemetryEvent, TelemetrySender};
use crate::transport::{self, PeerWire, Session};

pub struct NetDriver {
    rt: tokio::runtime::Runtime,
    session: Session,
    me: PlayerId,
    server_eid: EndpointId,
    early: Vec<PeerMsg>,
    id_map: BTreeMap<EndpointId, PlayerId>,
    departed: std::collections::BTreeSet<EndpointId>,
    telemetry: Option<TelemetrySender>,
    /// The formation barrier's shared-asset verdict — see [`NetDriver::sync_verdict`].
    sync: crate::SyncVerdict,
    asset_digest: u64,
    crab_count: u8,
}

#[derive(Default)]
pub struct Exchanged {
    /// Host-authoritative game states this remote client drained, in ARRIVAL order: the driver
    /// adopts every one via [`ClientSim::adopt_snapshots`] (the one shared adopt policy — see its
    /// doc). Empty on the solo/host arm (its client reads the server it runs).
    pub snapshots: Vec<CoreSnapshot>,
    pub articulations: Vec<CrabArticulation>,
}

#[derive(Debug)]
pub enum ServerDown {
    LinkLost,
    Refused(Refusal),
}

impl std::fmt::Display for ServerDown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerDown::LinkLost => write!(
                f,
                "Connection to the host was lost — the host quit, crashed, or the link died."
            ),
            ServerDown::Refused(reason) => {
                write!(f, "The host dropped us from the match: {reason}")
            }
        }
    }
}

impl NetDriver {
    /// The live-telemetry handle, if this client is streaming to a collector (`None` when
    /// launched without `--telemetry`).
    pub fn telemetry(&self) -> Option<&TelemetrySender> {
        self.telemetry.as_ref()
    }

    pub fn sync_verdict(&self) -> crate::SyncVerdict {
        self.sync
    }

    pub fn is_host(&self) -> bool {
        self.me == PlayerId(0)
    }

    pub fn roster(&self) -> Vec<PlayerId> {
        self.id_map.values().copied().collect()
    }

    fn take_early(&mut self) -> Vec<PeerMsg> {
        std::mem::take(&mut self.early)
    }

    pub fn send_to_server(&self, msg: &TickMsg) {
        self.rt.block_on(self.session.send(self.server_eid, msg));
    }

    pub fn drain_client_inputs(&mut self) -> (Vec<PeerMsg>, Vec<(EndpointId, JoinRequest)>) {
        let mut inputs = Vec::new();
        let mut joins = Vec::new();
        while let Some(from) = self.session.try_recv() {
            match from.msg {
                // A rostered client's input → file it; a not-yet-rostered endpoint's stray input
                // is dropped (the server's `record_remote` would drop it anyway — it isn't
                // rostered yet), so a joiner's pre-admit frame never blocks the round.
                PeerWire::Tick(msg) => {
                    if let Some(&pid) = self.id_map.get(&from.from) {
                        inputs.push(PeerMsg { pid, msg });
                    } else if self.departed.remove(&from.from) {
                        self.refuse_joiner(from.from, Refusal::Departed);
                    }
                }
                PeerWire::JoinRequest(req) => joins.push((from.from, req)),
                _ => {}
            }
        }
        (inputs, joins)
    }

    pub fn local_asset_digest(&self) -> u64 {
        self.asset_digest
    }

    pub fn local_crab_count(&self) -> u8 {
        self.crab_count
    }

    fn admit_endpoint(&mut self, eid: EndpointId, pid: PlayerId) {
        self.id_map.insert(eid, pid);
    }

    pub fn is_rostered(&self, eid: EndpointId) -> bool {
        self.id_map.contains_key(&eid)
    }

    fn welcome_joiner(&self, eid: EndpointId, adm: &Admission) {
        self.rt.block_on(self.session.send(eid, adm));
    }

    pub fn refuse_joiner(&self, eid: EndpointId, verdict: Refusal) {
        self.rt.block_on(self.session.send(eid, &verdict));
    }

    /// (Host) Broadcast the host-authoritative [`CoreSnapshot`] DOWN to every client — the full
    /// game STATE, so a remote client renders it instead of re-stepping. Non-blocking: enqueued to
    /// each link's writer task, so a dead peer can never hold this (main-thread) call.
    pub fn broadcast_snapshot(&self, snapshot: &CoreSnapshot) {
        self.rt.block_on(self.session.broadcast_snapshot(snapshot));
    }

    /// (Host) Broadcast the render-only [`CrabArticulation`] DOWN to every client, beside the
    /// snapshot, so a remote client renders the host's exact crab pose without simulating
    /// physics. Non-blocking, like [`Self::broadcast_snapshot`].
    pub fn broadcast_articulation(&self, articulation: &CrabArticulation) {
        self.rt
            .block_on(self.session.broadcast_articulation(articulation));
    }

    pub fn connected_peers_now(&self) -> Vec<EndpointId> {
        self.rt.block_on(self.session.connected_peers())
    }

    fn server_linked(&self) -> bool {
        self.rt
            .block_on(self.session.connected_peers())
            .contains(&self.server_eid)
    }

    pub fn server_endpoint_id(&self) -> EndpointId {
        self.server_eid
    }

    pub fn drain_server_down(&mut self) -> Result<Exchanged, ServerDown> {
        let mut down = Exchanged::default();
        while let Some(from) = self.session.try_recv() {
            match from.msg {
                PeerWire::Snapshot(snap) => down.snapshots.push(snap),
                PeerWire::Articulation(art) => down.articulations.push(art),
                PeerWire::Refuse(verdict) if from.from == self.server_eid => {
                    tracing::error!("server refused us mid-match: {verdict}");
                    return Err(ServerDown::Refused(verdict));
                }
                _ => {}
            }
        }
        Ok(down)
    }
}

/// One peer's per-tick input coordination. Either we run the [`Server`] (solo: a roster of one,
/// no transport; host: the whole roster + the transport to remote clients) or we are a remote
/// client of a server peer. Solo and host are the SAME [`Coordinator::Server`] arm — that is the
/// SP=MP-uniformity proof: there is no separate single-player code path, only the server with one
/// client.
pub enum Coordinator {
    Server {
        // Boxed: the [`Server`] owns the authoritative [`crate::sim::Sim`], so it dwarfs the
        // `Client` variant's lone `NetDriver` — heap it to keep the enum balanced.
        server: Box<Server>,
        net: Option<NetDriver>,
    },
    Client {
        net: NetDriver,
    },
}

impl Coordinator {
    /// Build the coordinator for a freshly-formed round. `me` is the LOCAL player (the server it
    /// builds stores it as the pacing host — see [`Server::new`]); `peers` is the sim's
    /// participant set (solo ⇒ just `me`); `sim` is the tick-0 authoritative world the server
    /// steps (a clone of the client's freshly-built sim, so the two start byte-identical). The
    /// carrier stays `Option<NetDriver>` so the arming + determinism-pin decisions upstream
    /// (which key off `net.is_some()`) are untouched: `None` ⇒ a solo server; a host driver ⇒ a
    /// server over the roster (seeded with any early inputs); a client driver ⇒ a remote client
    /// (`sim` unused — it adopts the host's snapshots).
    pub fn for_round(
        net: Option<NetDriver>,
        peers: &[PlayerId],
        me: PlayerId,
        sim: crate::sim::Sim,
    ) -> Self {
        match net {
            None => Coordinator::Server {
                server: Box::new(Server::new(me, peers, sim)),
                net: None,
            },
            Some(mut d) if d.is_host() => {
                debug_assert_eq!(me, d.me, "the host driver's id is the local player");
                let mut srv = Server::new(me, &d.roster(), sim);
                srv.seed_early(&d.take_early());
                Coordinator::Server {
                    server: Box::new(srv),
                    net: Some(d),
                }
            }
            // A remote client ADOPTS the host's per-tick snapshot into its OWN `client` (no re-sim),
            // so the Coordinator holds no authoritative server and this tick-0 `sim` goes unused
            // (the client's `GameState.client` is what the snapshots advance).
            Some(d) => {
                let _ = sim;
                Coordinator::Client { net: d }
            }
        }
    }

    pub fn exchange(&mut self, msg: TickMsg) -> Result<Exchanged, ServerDown> {
        match self {
            Coordinator::Server { server, net } => {
                let (remote, joins) = net
                    .as_mut()
                    .map(NetDriver::drain_client_inputs)
                    .unwrap_or_default();
                if let Some(net) = net.as_mut() {
                    admit_joiners(server, net, joins);
                }
                // File the drained remote inputs into their per-player streams, then handle
                // departures: a gone peer's stream is dropped and the roster shrinks — pure
                // bookkeeping, since host pacing means nothing ever WAITED on it.
                for pm in remote {
                    server.record_remote(pm.pid, pm.msg);
                }
                if let Some(net) = net.as_mut() {
                    let connected = net.connected_peers_now();
                    let gone = depart_gone_peers(
                        server,
                        &mut net.id_map,
                        net.me,
                        &connected,
                        net.telemetry.as_ref(),
                    );
                    net.departed.extend(gone);
                }
                // Assemble THIS tick from our own input + each remote stream's next queued input
                // (or a starved hold) — the host paces the match; a remote can delay nothing
                // (rl#193/#194/#195). The windowed driver pumps the tick's crab physics, then
                // steps it (`step_next`) and broadcasts the snapshot (`broadcast_step`), so a
                // client renders state rather than re-stepping inputs.
                server.advance(msg);
                Ok(Exchanged::default())
            }
            Coordinator::Client { net } => {
                // Ship our input UP and drain the host's STATE down (snapshots + render
                // articulation), never an input set to re-step. The driver adopts every drained
                // snapshot ([`ClientSim::adopt_snapshots`]) and renders the last-arrived
                // articulation.
                net.send_to_server(&msg);
                let down = net.drain_server_down()?;
                if !net.server_linked() {
                    return Err(ServerDown::LinkLost);
                }
                Ok(down)
            }
        }
    }

    pub fn broadcast_step(&self, snapshot: &CoreSnapshot, articulation: Option<&CrabArticulation>) {
        if let Coordinator::Server { net: Some(net), .. } = self {
            net.broadcast_snapshot(snapshot);
            if let Some(art) = articulation {
                net.broadcast_articulation(art);
            }
        }
    }

    /// Whether THIS peer is a REMOTE client of another peer's server: it adopts the host's
    /// snapshots + renders its articulation, never pumping its own crab physics or stepping the
    /// sim. Distinct from the scripted screenshot harness, which self-sims.
    pub fn is_remote_client(&self) -> bool {
        matches!(self, Coordinator::Client { .. })
    }

    pub fn server_endpoint(&self) -> Option<EndpointId> {
        match self {
            Coordinator::Server { .. } => None,
            Coordinator::Client { net } => Some(net.server_endpoint_id()),
        }
    }

    pub fn server_mut(&mut self) -> Option<&mut Server> {
        match self {
            Coordinator::Server { server, .. } => Some(&mut **server),
            Coordinator::Client { .. } => None,
        }
    }

    pub fn server(&self) -> Option<&Server> {
        match self {
            Coordinator::Server { server, .. } => Some(&**server),
            Coordinator::Client { .. } => None,
        }
    }

    fn net(&self) -> Option<&NetDriver> {
        match self {
            Coordinator::Server { net, .. } => net.as_ref(),
            Coordinator::Client { net } => Some(net),
        }
    }

    pub fn telemetry(&self) -> Option<&TelemetrySender> {
        self.net().and_then(NetDriver::telemetry)
    }
}

pub fn depart_gone_peers(
    server: &mut Server,
    id_map: &mut BTreeMap<EndpointId, PlayerId>,
    me: PlayerId,
    connected: &[EndpointId],
    telemetry: Option<&TelemetrySender>,
) -> Vec<EndpointId> {
    let gone: Vec<(EndpointId, PlayerId)> = id_map
        .iter()
        .filter(|(eid, pid)| **pid != me && !connected.contains(eid))
        .map(|(eid, pid)| (*eid, *pid))
        .collect();
    let mut eids = Vec::new();
    for (eid, pid) in gone {
        id_map.remove(&eid);
        tracing::info!(
            "player {pid:?} ({}) departed — continuing without them",
            eid.fmt_short()
        );
        if let Some(t) = telemetry {
            t.send(TelemetryEvent::Departed {
                player: pid.0,
                endpoint: eid.fmt_short().to_string(),
            });
        }
        server.depart(pid);
        eids.push(eid);
    }
    eids
}

fn admit_joiners(server: &mut Server, net: &mut NetDriver, joins: Vec<(EndpointId, JoinRequest)>) {
    let host_assets = net.local_asset_digest();
    let host_crabs = net.local_crab_count();
    for (eid, req) in joins {
        if net.is_rostered(eid) {
            continue;
        }
        match may_admit_joiner(host_assets, host_crabs, &req) {
            Ok(()) => {
                let adm = server.admit();
                net.admit_endpoint(eid, adm.pid);
                net.welcome_joiner(eid, &adm);
                tracing::info!(
                    "admitted joiner {} as {:?}, roster change effective at tick {}",
                    eid.fmt_short(),
                    adm.pid,
                    adm.effective_tick
                );
                if let Some(t) = net.telemetry() {
                    t.send(TelemetryEvent::Admitted {
                        player: adm.pid.0,
                        endpoint: eid.fmt_short().to_string(),
                        effective_tick: adm.effective_tick,
                    });
                }
            }
            Err(refusal) => {
                tracing::error!("refused mid-game joiner {}: {refusal}", eid.fmt_short());
                net.refuse_joiner(eid, Refusal::Admission(refusal));
                if let Some(t) = net.telemetry() {
                    t.send(TelemetryEvent::RosterFailed {
                        reason: format!("join refused: {refusal}"),
                    });
                }
            }
        }
    }
}

pub enum MatchResult {
    Joined(Box<(ClientSim, NetDriver)>),
    /// Discovery completed with only us on the LAN — the caller starts a deterministic
    /// solo round (see [`crate::formation`]'s solo fallback).
    Alone,
    Cancelled,
}

#[allow(clippy::too_many_arguments)]
pub fn connect_and_form_dialing(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    dial: Option<iroh::EndpointId>,
    collector: Option<iroh::EndpointId>,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    local_asset_digest: u64,
    local_crab_count: u8,
) -> Result<MatchResult> {
    connect_and_form_inner(
        seed,
        discover_secs,
        expect,
        dial,
        collector,
        on_bound,
        None,
        local_asset_digest,
        local_crab_count,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn connect_and_form_lobby(
    seed: u64,
    expect: usize,
    dial: Option<iroh::EndpointId>,
    collector: Option<iroh::EndpointId>,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    control: LobbyControl,
    local_asset_digest: u64,
    local_crab_count: u8,
) -> Result<MatchResult> {
    connect_and_form_inner(
        seed,
        0,
        expect,
        dial,
        collector,
        on_bound,
        Some(control),
        local_asset_digest,
        local_crab_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn connect_and_form_inner(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    dial: Option<iroh::EndpointId>,
    collector: Option<iroh::EndpointId>,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    lobby: Option<LobbyControl>,
    local_asset_digest: u64,
    local_crab_count: u8,
) -> Result<MatchResult> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (session, formation, telemetry) = rt.block_on(async {
        let mut session = transport::start_session().await?;
        let my_eid = session.endpoint_id();
        println!("fp client endpoint id: {my_eid}");
        if let Some(tx) = &on_bound {
            let _ = tx.send(my_eid);
        }
        if let Some(host) = dial {
            if host == my_eid {
                tracing::warn!("join code is our own endpoint id — ignoring the self-dial");
            } else if let Err(e) = session.connect_direct(host).await {
                tracing::warn!("dialing host {} failed: {e:#}", host.fmt_short());
            }
        }
        let telemetry = connect_telemetry(collector, my_eid).await;
        let formation = form_match(
            &mut session,
            discover_secs,
            expect,
            telemetry.as_ref(),
            lobby.as_ref(),
            local_asset_digest,
            local_crab_count,
        )
        .await?;
        anyhow::Ok((session, formation, telemetry))
    })?;

    let frozen = match formation {
        Formation::Agreed(frozen) => frozen,
        Formation::Alone => return Ok(MatchResult::Alone),
        Formation::Cancelled => {
            drop(telemetry);
            rt.block_on(session.shutdown());
            return Ok(MatchResult::Cancelled);
        }
    };

    let all_ids: Vec<PlayerId> = frozen.id_map.values().copied().collect();
    println!(
        "starting round: {} player(s), I am {:?}",
        all_ids.len(),
        frozen.me
    );
    // Every peer spawns the byte-identical foot-only round. Early inputs ride the driver to
    // seed the host's server (see [`Coordinator::for_round`]) — never replayed into the client
    // sim, which would bypass the server's input streams.
    let client = ClientSim::new(seed, &all_ids, frozen.me);
    let server_eid = server_endpoint(&frozen.id_map);
    let early = early_peer_msgs(&frozen);
    let driver = NetDriver {
        rt,
        session,
        me: frozen.me,
        server_eid,
        early,
        id_map: frozen.id_map,
        departed: Default::default(),
        telemetry,
        sync: frozen.sync,
        asset_digest: local_asset_digest,
        crab_count: local_crab_count,
    };
    Ok(MatchResult::Joined(Box::new((client, driver))))
}

const JOIN_WELCOME_TIMEOUT: Duration = Duration::from_secs(10);

pub enum JoinResult {
    Joined(Box<(ClientSim, NetDriver)>),
    Refused(Refusal),
    Unreachable,
}

enum AdmissionVerdict {
    Admitted(Admission),
    Refused(Refusal),
    Timeout,
}

async fn await_admission(session: &mut Session, host: EndpointId) -> AdmissionVerdict {
    let deadline = tokio::time::timeout(JOIN_WELCOME_TIMEOUT, async {
        loop {
            let Some(from) = session.recv().await else {
                return AdmissionVerdict::Timeout;
            };
            if from.from != host {
                continue;
            }
            match from.msg {
                PeerWire::Welcome(adm) => return AdmissionVerdict::Admitted(adm),
                PeerWire::Refuse(verdict) => return AdmissionVerdict::Refused(verdict),
                _ => continue,
            }
        }
    })
    .await;
    deadline.unwrap_or(AdmissionVerdict::Timeout)
}

/// Dial INTO a live match as a host-authoritative mid-game joiner — the mid-game analogue of
/// forming via [`connect_and_form_dialing`]. Connect to `host`, send our collider digest as a [`JoinRequest`], and
/// await the host's verdict: admitted (become a remote-adopt [`Coordinator::Client`] that boots
/// from the host's next authoritative snapshot — the host spawns us into its LIVE round at
/// `effective_tick`, so we drop into the ongoing match rather than resetting it), refused (the
/// host's [`Refusal`] verdict relayed LOUDLY — never a silent wrong/fake-crab), or unreachable.
/// `seed` is the shared [`crate::sim`] match constant every peer holds.
pub fn connect_and_join(
    seed: u64,
    host: EndpointId,
    collector: Option<EndpointId>,
    local_asset_digest: u64,
    local_crab_count: u8,
) -> Result<JoinResult> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (session, verdict, telemetry) = rt.block_on(async {
        let mut session = transport::start_session().await?;
        let my_eid = session.endpoint_id();
        println!("joining as endpoint id: {my_eid}");
        anyhow::ensure!(host != my_eid, "cannot join our own endpoint id");
        match tokio::time::timeout(JOIN_WELCOME_TIMEOUT, session.connect_direct(host)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::warn!("dialing host {} failed: {e:#}", host.fmt_short());
                return anyhow::Ok((session, AdmissionVerdict::Timeout, None));
            }
            Err(_) => {
                tracing::warn!("dialing host {} timed out", host.fmt_short());
                return anyhow::Ok((session, AdmissionVerdict::Timeout, None));
            }
        }
        let telemetry = connect_telemetry(collector, my_eid).await;
        session
            .send(
                host,
                &JoinRequest {
                    asset_digest: local_asset_digest,
                    crab_count: local_crab_count,
                },
            )
            .await;
        let verdict = await_admission(&mut session, host).await;
        anyhow::Ok((session, verdict, telemetry))
    })?;

    match verdict {
        AdmissionVerdict::Refused(verdict) => {
            tracing::error!("host refused our join: {verdict}");
            rt.block_on(session.shutdown());
            Ok(JoinResult::Refused(verdict))
        }
        AdmissionVerdict::Timeout => {
            drop(telemetry);
            rt.block_on(session.shutdown());
            Ok(JoinResult::Unreachable)
        }
        AdmissionVerdict::Admitted(adm) => {
            let me = adm.pid;
            println!(
                "admitted as {me:?}; joining at tick {} over roster {:?}",
                adm.effective_tick, adm.roster
            );
            // This `client` is only a placeholder the remote-adopt client boots from — the driver is
            // a CLIENT (not host), so `for_round` makes this a `Coordinator::Client` that ADOPTS
            // the host's per-tick snapshots and never steps a sim of its own. The host spawns us
            // into its LIVE authoritative round at `effective_tick` (`Server::step_next` →
            // `Sim::spawn_joining_player`), so we render the host's output, never re-simming its
            // warm rapier world. `join_at` seeds the placeholder cursors/roster; the adopted
            // snapshot supersedes.
            let client = ClientSim::join_at(seed, &adm.roster, me, adm.effective_tick);
            let my_eid = session.endpoint_id();
            debug_assert!(
                adm.roster.contains(&PlayerId(0)),
                "the host (PlayerId 0) must be in the roster we were admitted into"
            );
            let mut id_map = BTreeMap::new();
            id_map.insert(host, PlayerId(0));
            id_map.insert(my_eid, me);
            let driver = NetDriver {
                rt,
                session,
                me,
                server_eid: host,
                early: Vec::new(),
                id_map,
                departed: Default::default(),
                telemetry,
                sync: crate::SyncVerdict {
                    assets: true,
                    crabs: true,
                },
                asset_digest: local_asset_digest,
                crab_count: local_crab_count,
            };
            Ok(JoinResult::Joined(Box::new((client, driver))))
        }
    }
}

pub async fn connect_telemetry(
    collector: Option<iroh::EndpointId>,
    my_eid: iroh::EndpointId,
) -> Option<TelemetrySender> {
    let collector = collector?;
    match TelemetrySender::connect(collector, *my_eid.as_bytes()).await {
        Ok(t) => Some(t),
        Err(e) => {
            tracing::warn!("telemetry disabled (endpoint bind failed): {e:#}");
            None
        }
    }
}

fn server_endpoint(id_map: &BTreeMap<EndpointId, PlayerId>) -> EndpointId {
    id_map
        .iter()
        .find(|(_, pid)| **pid == PlayerId(0))
        .map(|(&eid, _)| eid)
        .expect("a frozen roster always contains PlayerId(0)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solo_round_advances_through_the_coordinator() {
        use crate::sim::Input;
        let me = PlayerId(0);
        let mut client = ClientSim::new(0x5A11, &[me], me);
        let mut coord = Coordinator::for_round(None, client.peers(), me, client.sim().clone());
        assert!(
            matches!(coord, Coordinator::Server { net: None, .. }),
            "no driver ⇒ a solo internal-server coordinator"
        );
        let submits = 5u64;
        for _ in 0..submits {
            let msg = client.submit_local_input(Input::from_axes(1.0, 0.0), None);
            let exch = coord
                .exchange(msg)
                .expect("the solo/host arm can never lose its in-process server (rl#203)");
            assert!(
                exch.snapshots.is_empty(),
                "the solo/host arm returns state empty"
            );
            let server = coord.server_mut().expect("solo runs an internal server");
            while server.next_tick_ready() {
                let bytes = server.step_next(&[]).snapshot;
                let snap =
                    crate::snapshot::CoreSnapshot::from_bytes(&bytes).expect("snapshot decodes");
                client.apply_core_snapshot(snap);
            }
        }
        assert_eq!(
            client.sim().tick(),
            submits,
            "solo advances one tick per submit through the host-authoritative path"
        );
    }
}
