//! The per-round network coordination layer: [`Coordinator`] (solo/host server vs remote
//! client) and [`NetDriver`], the sync façade a windowed frame drives over the async
//! [`Session`]. A remote client ships inputs UP and adopts the host's authoritative
//! snapshots DOWN ([`ClientSim::adopt_snapshots`]) — it never re-steps the sim; that
//! contract is stated once, here, and every arm below leans on it.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::Result;
use iroh::EndpointId;

use crate::articulation::CrabArticulation;
use crate::client::{ClientSim, PeerMsg, TickMsg};
use crate::formation::{Formation, LobbyControl, early_peer_msgs, form_match};
use crate::server::{JoinRequest, Refusal, Server, may_admit_joiner};
use crate::sim::PlayerId;
use crate::snapshot::CoreSnapshot;
use crate::telemetry::{TelemetryEvent, TelemetrySender};
use crate::transport::{self, PeerWire, Session};

/// Most departed-endpoint ids remembered for the courtesy [`Refusal::Departed`] reply
/// (rl#350): under join/depart churn the set would otherwise grow one endpoint id per
/// cycle for the life of the round. Overflow clears the set — the reply is best-effort
/// (a forgotten departee's zombie link is closed by the unrostered grace instead, and
/// its client surfaces the loss as LinkLost).
const DEPARTED_CAP: usize = crate::membership::MAX_MEMBERS;

/// Sustained mid-game join-attempt rate the host processes, and the burst headroom over
/// it (a party joining at once). Joins are rare human actions; anything past this is a
/// flood — excess requests are dropped BEFORE admission (rl#350), bounding the
/// spawn/despawn, log, and telemetry work a sybil dialer can force. Per-source limits
/// don't hold here (a fresh keypair per attempt is free), so the budget is global.
const JOIN_RATE_PER_SEC: f64 = 2.0;
const JOIN_BURST: f64 = 8.0;

/// How long the host lets a connected-but-unrostered endpoint linger before closing its
/// link (rl#350) — comfortably past the joiner-side `JOIN_WELCOME_TIMEOUT`, so a pending
/// admission is never cut. Without this the transport's link cap is a one-shot fuse: a
/// burst of silent connections (our keep-alives hold them open) would pin the link table
/// full for the life of the round, denying every later legitimate join.
const UNROSTERED_GRACE: Duration = Duration::from_secs(30);

/// Token bucket over wall-clock time for [`JOIN_RATE_PER_SEC`]/[`JOIN_BURST`].
struct JoinBudget {
    tokens: f64,
    last: std::time::Instant,
}

impl Default for JoinBudget {
    fn default() -> Self {
        Self {
            tokens: JOIN_BURST,
            last: std::time::Instant::now(),
        }
    }
}

impl JoinBudget {
    fn allow(&mut self, now: std::time::Instant) -> bool {
        let dt = now.saturating_duration_since(self.last).as_secs_f64();
        self.last = now;
        self.tokens = (self.tokens + dt * JOIN_RATE_PER_SEC).min(JOIN_BURST);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// One peer's transport-side round state: the tokio runtime the sync façade blocks on,
/// the live [`Session`], and the roster bookkeeping (endpoint↔player map, departures,
/// telemetry, the formation sync verdict). Built only by [`connect_and_form_dialing`] /
/// [`connect_and_form_lobby`] / [`connect_and_join`]; driven only through
/// [`Coordinator`].
pub struct NetDriver {
    rt: tokio::runtime::Runtime,
    session: Session,
    me: PlayerId,
    server_eid: EndpointId,
    early: Vec<PeerMsg>,
    id_map: BTreeMap<EndpointId, PlayerId>,
    departed: std::collections::BTreeSet<EndpointId>,
    join_budget: JoinBudget,
    /// (Host) First-seen times of connected endpoints not (yet) in `id_map`, for the
    /// [`UNROSTERED_GRACE`] eviction. Keyed off the live connection set each pump, so it
    /// is bounded by the transport's link cap.
    unrostered_since: BTreeMap<EndpointId, std::time::Instant>,
    telemetry: Option<TelemetrySender>,
    /// The shared-asset verdict this round was formed (or admitted) under — computed by
    /// the ONE arbiter, [`crate::SyncVerdict::between`].
    sync: crate::SyncVerdict,
    stamp: crate::SyncStamp,
}

#[derive(Default)]
pub struct Exchanged {
    /// Host-authoritative game states this remote client drained, in ARRIVAL order, for
    /// [`ClientSim::adopt_snapshots`]. Empty on the solo/host arm (its client reads the
    /// server it runs).
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

impl Drop for NetDriver {
    fn drop(&mut self) {
        // Close both iroh endpoints (telemetry sender's + session's) before `rt` — the FIRST
        // field, so it drops before them — shuts down and aborts their tasks: an endpoint
        // dropped unclosed makes iroh log "Endpoint dropped without calling `Endpoint::close`"
        // at ERROR on every round teardown (fleet-error 2026-08-01).
        let telemetry = self.telemetry.take();
        let session = &self.session;
        self.rt.block_on(close_session(session, telemetry));
    }
}

/// The one graceful teardown for a bound session and its optional telemetry stream — every
/// pre-round exit path and [`NetDriver`]'s `Drop` end through here.
pub async fn close_session(session: &Session, telemetry: Option<TelemetrySender>) {
    if let Some(t) = telemetry {
        t.close().await;
    }
    session.shutdown().await;
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

    fn refuse(&self, eid: EndpointId, verdict: Refusal) {
        self.session.send(eid, &verdict);
    }

    /// (Host) One tick's whole inbound pump: drain every queued frame, filing rostered
    /// clients' inputs into the server's per-player streams, admitting or refusing
    /// joiners, and departing rostered peers whose links are gone. THE host-side
    /// transport↔server boundary — [`Coordinator::exchange`]'s server arm is this call
    /// plus [`Server::advance`].
    fn pump_host(&mut self, server: &mut Server) {
        let mut joins: Vec<(EndpointId, JoinRequest)> = Vec::new();
        while let Some(from) = self.session.try_recv() {
            match from.msg {
                // A rostered client's input → file it; a not-yet-rostered endpoint's stray input
                // is dropped (the server's `record_remote` would drop it anyway — it isn't
                // rostered yet), so a joiner's pre-admit frame never blocks the round.
                PeerWire::Tick(msg) => {
                    if let Some(&pid) = self.id_map.get(&from.from) {
                        server.record_remote(pid, msg);
                    } else if self.departed.remove(&from.from) {
                        self.refuse(from.from, Refusal::Departed);
                    }
                }
                PeerWire::JoinRequest(req) => joins.push((from.from, req)),
                // Down-stream state and formation traffic carry no host-side meaning: only
                // the server's own word is game state, and formation ended at the barrier.
                // Explicit arms so a NEW wire kind must decide its host-side handling here.
                PeerWire::Snapshot(_)
                | PeerWire::Articulation(_)
                | PeerWire::Beat(_)
                | PeerWire::Welcome(_)
                | PeerWire::Refuse(_) => {}
            }
        }
        for (eid, req) in joins {
            // Already-admitted first, so a duplicate JoinRequest can neither spend budget
            // nor get its LIVE link disconnected below.
            if self.id_map.contains_key(&eid) {
                continue;
            }
            // Over-budget requests are dropped, not refused: a reply per attempt would hand
            // a flooder attacker-rate outbound work. A rate-limited legitimate joiner falls
            // into its JOIN_WELCOME_TIMEOUT and can retry.
            if !self.join_budget.allow(std::time::Instant::now()) {
                tracing::debug!(
                    "join budget exhausted — dropping join from {}",
                    eid.fmt_short()
                );
                // Close, don't just ignore: an ignored flood connection would sit in the
                // link table (kept alive by our keep-alives) eating a capacity slot. A
                // QUIC close is not attacker-directed outbound work the way a refusal
                // frame would be. A rate-limited legitimate joiner sees "unreachable"
                // and retries.
                self.session.disconnect(eid);
                continue;
            }
            self.admit_joiner(server, eid, req);
        }
        let connected = self.session.connected_peers();
        let gone = depart_gone_peers(
            server,
            &mut self.id_map,
            self.me,
            &connected,
            self.telemetry.as_ref(),
        );
        self.departed.extend(gone);
        if self.departed.len() > DEPARTED_CAP {
            // Reset rather than evict by set order — pop_first would evict the SMALLEST
            // endpoint id, uncorrelated with recency, silently biasing the courtesy away
            // from exactly the just-departed peers it exists for. A rare full reset is
            // honest, and an evicted departee's zombie link still surfaces as LinkLost
            // via the unrostered grace below.
            self.departed.clear();
        }
        // Unrostered-link liveness (rl#350): close any connected endpoint that has held a
        // link past UNROSTERED_GRACE without becoming rostered — a sybil squatting a
        // capacity slot, or an evicted departee's zombie. A pending joiner is rostered at
        // admission, well inside the grace.
        let now = std::time::Instant::now();
        let id_map = &self.id_map;
        self.unrostered_since
            .retain(|eid, _| connected.contains(eid) && !id_map.contains_key(eid));
        for eid in connected {
            if self.id_map.contains_key(&eid) {
                continue;
            }
            let since = *self.unrostered_since.entry(eid).or_insert(now);
            if now.duration_since(since) > UNROSTERED_GRACE {
                tracing::info!(
                    "closing link to {} — connected {UNROSTERED_GRACE:?} without joining",
                    eid.fmt_short()
                );
                self.session.disconnect(eid);
                self.unrostered_since.remove(&eid);
            }
        }
    }

    /// Caller ([`NetDriver::pump_host`]'s join loop) has already screened duplicates
    /// against `id_map` and charged the join budget.
    fn admit_joiner(&mut self, server: &mut Server, eid: EndpointId, req: JoinRequest) {
        match may_admit_joiner(self.stamp, &req, server.roster().len()) {
            Ok(()) => {
                let adm = server.admit(self.stamp);
                self.id_map.insert(eid, adm.pid);
                self.session.send(eid, &adm);
                tracing::info!(
                    "admitted joiner {} as {:?}, roster change effective at tick {}",
                    eid.fmt_short(),
                    adm.pid,
                    adm.effective_tick
                );
                if let Some(t) = &self.telemetry {
                    t.send(TelemetryEvent::Admitted {
                        player: adm.pid.0,
                        endpoint: eid.fmt_short().to_string(),
                        effective_tick: adm.effective_tick,
                    });
                }
            }
            Err(refusal) => {
                // warn, not error: a refused join is a protocol outcome, not a host fault
                // (the joiner surfaces its own loud error), and error-level would feed
                // fleet alerting at joiner-controlled rate (rl#350).
                tracing::warn!("refused mid-game joiner {}: {refusal}", eid.fmt_short());
                self.refuse(eid, Refusal::Admission(refusal));
                if let Some(t) = &self.telemetry {
                    t.send(TelemetryEvent::RosterFailed {
                        reason: format!("join refused: {refusal}"),
                    });
                }
            }
        }
    }

    /// (Client) One tick's whole exchange: ship our input UP, drain the host's STATE down
    /// (snapshots + render articulation — see the module doc: adopted, never re-stepped),
    /// and fail loudly the tick the host link dies or refuses us.
    fn exchange_client(&mut self, msg: &TickMsg) -> Result<Exchanged, ServerDown> {
        self.session.send(self.server_eid, msg);
        let mut down = Exchanged::default();
        while let Some(from) = self.session.try_recv() {
            match from.msg {
                // Only the SERVER's word is game state: formation is a mesh (and discovery
                // dials every LAN peer), so a non-host peer can hold a live authenticated
                // link — its snapshots must never be adopted as authority.
                PeerWire::Snapshot(snap) if from.from == self.server_eid => {
                    down.snapshots.push(snap)
                }
                PeerWire::Articulation(art) if from.from == self.server_eid => {
                    down.articulations.push(art)
                }
                PeerWire::Refuse(verdict) if from.from == self.server_eid => {
                    tracing::error!("server refused us mid-match: {verdict}");
                    return Err(ServerDown::Refused(verdict));
                }
                // Non-server or non-state traffic carries no client-side meaning — explicit
                // arms so a NEW wire kind must decide its client-side handling here.
                PeerWire::Snapshot(_)
                | PeerWire::Articulation(_)
                | PeerWire::Refuse(_)
                | PeerWire::Tick(_)
                | PeerWire::Beat(_)
                | PeerWire::JoinRequest(_)
                | PeerWire::Welcome(_) => {}
            }
        }
        let connected = self.session.connected_peers();
        if !connected.contains(&self.server_eid) {
            return Err(ServerDown::LinkLost);
        }
        Ok(down)
    }

    /// (Host) Broadcast the authoritative [`CoreSnapshot`] (and, armed, the render-only
    /// [`CrabArticulation`]) DOWN to every client. Non-blocking: fire-and-forget datagrams
    /// (rl#259), so a dead peer can never hold this (main-thread) call.
    fn broadcast_step(&self, snapshot: &CoreSnapshot, articulation: Option<&CrabArticulation>) {
        self.session.broadcast_state(snapshot);
        if let Some(art) = articulation {
            self.session.broadcast_state(art);
        }
    }
}

/// One peer's per-tick input coordination. Either we run the [`Server`] (solo: a roster of one,
/// no transport; host: the whole roster + the transport to remote clients) or we are a remote
/// client of a server peer. Solo and host are the SAME server arm — that is the
/// SP=MP-uniformity proof: there is no separate single-player code path, only the server with one
/// client. The mode enum is private, [`Coordinator::for_round`] the only constructor, so an
/// illegal pairing (a server arm carrying a non-host driver, a client arm carrying none) is
/// unrepresentable outside this module.
pub struct Coordinator(Mode);

enum Mode {
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
    /// steps (a clone of the client's freshly-built sim, so the two start byte-identical).
    /// `None` ⇒ a solo server; a host driver ⇒ a server over the roster (seeded with any early
    /// inputs); a client driver ⇒ a remote client (`sim` unused — it adopts the host's
    /// snapshots, per the module doc).
    pub fn for_round(
        net: Option<NetDriver>,
        peers: &[PlayerId],
        me: PlayerId,
        sim: crate::sim::Sim,
    ) -> Self {
        Coordinator(match net {
            None => Mode::Server {
                server: Box::new(Server::new(me, peers, sim)),
                net: None,
            },
            Some(mut d) if d.is_host() => {
                debug_assert_eq!(me, d.me, "the host driver's id is the local player");
                let mut srv = Server::new(me, &d.id_map.values().copied().collect::<Vec<_>>(), sim);
                srv.seed_early(&std::mem::take(&mut d.early));
                Mode::Server {
                    server: Box::new(srv),
                    net: Some(d),
                }
            }
            // A remote client adopts the host's snapshots into its OWN `client`, so the
            // Coordinator holds no authoritative server and this tick-0 `sim` goes unused.
            Some(d) => {
                let _ = sim;
                Mode::Client { net: d }
            }
        })
    }

    pub fn exchange(&mut self, msg: TickMsg) -> Result<Exchanged, ServerDown> {
        match &mut self.0 {
            Mode::Server { server, net } => {
                if let Some(net) = net {
                    net.pump_host(server);
                }
                // Assemble THIS tick from our own input + each remote stream's next queued input
                // (or a starved hold) — the host paces the match; a remote can delay nothing
                // (rl#193/#194/#195). The windowed driver pumps the tick's crab physics, then
                // steps it (`step_next`) and broadcasts the snapshot (`broadcast_step`).
                server.advance(msg);
                Ok(Exchanged::default())
            }
            Mode::Client { net } => net.exchange_client(&msg),
        }
    }

    pub fn broadcast_step(&self, snapshot: &CoreSnapshot, articulation: Option<&CrabArticulation>) {
        if let Mode::Server { net: Some(net), .. } = &self.0 {
            net.broadcast_step(snapshot, articulation);
        }
    }

    /// Whether THIS peer is a REMOTE client of another peer's server (see the module doc).
    /// Distinct from the scripted screenshot harness, which self-sims.
    pub fn is_remote_client(&self) -> bool {
        matches!(self.0, Mode::Client { .. })
    }

    pub fn server_endpoint(&self) -> Option<EndpointId> {
        match &self.0 {
            Mode::Server { .. } => None,
            Mode::Client { net } => Some(net.server_eid),
        }
    }

    pub fn server_mut(&mut self) -> Option<&mut Server> {
        match &mut self.0 {
            Mode::Server { server, .. } => Some(&mut **server),
            Mode::Client { .. } => None,
        }
    }

    pub fn server(&self) -> Option<&Server> {
        match &self.0 {
            Mode::Server { server, .. } => Some(&**server),
            Mode::Client { .. } => None,
        }
    }

    pub fn telemetry(&self) -> Option<&TelemetrySender> {
        match &self.0 {
            Mode::Server { net, .. } => net.as_ref(),
            Mode::Client { net } => Some(net),
        }
        .and_then(NetDriver::telemetry)
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

pub enum MatchResult {
    Joined(Box<(ClientSim, NetDriver)>),
    /// Discovery completed with only us on the LAN — the caller starts a deterministic
    /// solo round (see [`crate::formation`]'s solo fallback).
    Alone,
    Cancelled,
}

/// The optional peers a formation/join launch dials — named fields so the two same-typed
/// endpoint ids cannot be transposed at a call site (dialing the collector as the host
/// compiles fine and fails only at runtime).
#[derive(Default, Clone, Copy)]
pub struct DialTargets {
    /// A known host/peer to dial directly (a join code), besides LAN discovery.
    pub host: Option<EndpointId>,
    /// The live-telemetry collector to stream to.
    pub collector: Option<EndpointId>,
}

pub fn connect_and_form_dialing(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    targets: DialTargets,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    stamp: crate::SyncStamp,
) -> Result<MatchResult> {
    connect_and_form_inner(seed, discover_secs, expect, targets, on_bound, None, stamp)
}

pub fn connect_and_form_lobby(
    seed: u64,
    expect: usize,
    targets: DialTargets,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    control: LobbyControl,
    stamp: crate::SyncStamp,
) -> Result<MatchResult> {
    connect_and_form_inner(seed, 0, expect, targets, on_bound, Some(control), stamp)
}

fn net_runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

fn connect_and_form_inner(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    targets: DialTargets,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    lobby: Option<LobbyControl>,
    stamp: crate::SyncStamp,
) -> Result<MatchResult> {
    let rt = net_runtime()?;

    let (session, formation, telemetry, dial_failed) = rt.block_on(async {
        let mut session = transport::start_session().await?;
        let my_eid = session.endpoint_id();
        println!("fp client endpoint id: {my_eid}");
        if let Some(tx) = &on_bound {
            let _ = tx.send(my_eid);
        }
        let mut dial_failed: Option<anyhow::Error> = None;
        if let Some(host) = targets.host {
            if host == my_eid {
                tracing::warn!("join code is our own endpoint id — ignoring the self-dial");
            } else if let Err(e) = session.connect_direct(host).await {
                // Not yet fatal: discovery may still find the host on the LAN. If it
                // doesn't, the Alone arm below turns this into a hard error rather than
                // silently starting a solo round the player didn't ask for.
                tracing::warn!("dialing host {} failed: {e:#}", host.fmt_short());
                dial_failed = Some(e);
            }
        }
        let telemetry = connect_telemetry(targets.collector, my_eid).await;
        let formation = form_match(
            &mut session,
            discover_secs,
            expect,
            telemetry.as_ref(),
            lobby.as_ref(),
            stamp,
        )
        .await?;
        anyhow::Ok((session, formation, telemetry, dial_failed))
    })?;

    let frozen = match formation {
        Formation::Agreed(frozen) => frozen,
        Formation::Alone => {
            rt.block_on(close_session(&session, telemetry));
            if let Some(e) = dial_failed {
                return Err(e.context(
                    "could not reach the host you asked to join, and LAN discovery found no one",
                ));
            }
            return Ok(MatchResult::Alone);
        }
        Formation::Cancelled => {
            rt.block_on(close_session(&session, telemetry));
            return Ok(MatchResult::Cancelled);
        }
    };

    let all_ids: Vec<PlayerId> = frozen.id_map.values().copied().collect();
    println!(
        "starting round: {} player(s), I am {:?}",
        all_ids.len(),
        frozen.me
    );
    // Every peer spawns a foot-only round from its own seed; since rl#305 the layout is
    // seed-derived, so a non-host peer's round is a PLACEHOLDER the host's first snapshot
    // (players, crabs, extraction) supersedes. Early inputs ride the driver to
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
        join_budget: Default::default(),
        unrostered_since: Default::default(),
        telemetry,
        sync: frozen.sync,
        stamp,
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
    Admitted(crate::server::Admission),
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
/// forming via [`connect_and_form_dialing`]. Connect to `host`, send our [`crate::SyncStamp`]
/// as a [`JoinRequest`], and await the host's verdict: admitted (become a remote-adopt client
/// booting from the host's next authoritative snapshot — the host spawns us into its LIVE
/// round at `effective_tick`), refused (the host's [`Refusal`] relayed LOUDLY — never a
/// silent wrong/fake-crab), or unreachable. `seed` is the shared [`crate::sim`] match
/// constant every peer holds.
pub fn connect_and_join(
    seed: u64,
    host: EndpointId,
    collector: Option<EndpointId>,
    stamp: crate::SyncStamp,
) -> Result<JoinResult> {
    let rt = net_runtime()?;

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
        session.send(host, &JoinRequest { stamp });
        let verdict = await_admission(&mut session, host).await;
        anyhow::Ok((session, verdict, telemetry))
    })?;

    match verdict {
        AdmissionVerdict::Refused(verdict) => {
            tracing::error!("host refused our join: {verdict}");
            rt.block_on(close_session(&session, telemetry));
            Ok(JoinResult::Refused(verdict))
        }
        AdmissionVerdict::Timeout => {
            rt.block_on(close_session(&session, telemetry));
            Ok(JoinResult::Unreachable)
        }
        AdmissionVerdict::Admitted(adm) => {
            let me = adm.pid;
            println!(
                "admitted as {me:?}; joining at tick {} over roster {:?}",
                adm.effective_tick, adm.roster
            );
            // This `client` is only a placeholder the remote-adopt client boots from — the
            // host spawns us into its LIVE round at `effective_tick` (`Server::step_next` →
            // `Sim::spawn_joining_player`); the adopted snapshot supersedes (module doc).
            let client = ClientSim::join_at(seed, &adm.roster, me, adm.effective_tick);
            let my_eid = session.endpoint_id();
            // Hard checks, not debug_asserts: the Admission came off the WIRE, so a
            // malicious/buggy host could otherwise hand us pid 0 and flip this joiner
            // into hosting the round it meant to join (NetDriver::is_host keys on it).
            anyhow::ensure!(
                me != PlayerId(0),
                "host admitted us as PlayerId(0) — refusing to become the host of a round we joined"
            );
            anyhow::ensure!(
                adm.roster.contains(&PlayerId(0)),
                "the host (PlayerId 0) must be in the roster we were admitted into"
            );
            let mut id_map = BTreeMap::new();
            id_map.insert(host, PlayerId(0));
            id_map.insert(my_eid, me);
            // The ONE arbiter judges our round verdict, from the host stamp the Welcome
            // carried — the same comparison the host's admission gate just passed, so an
            // admitted joiner's verdict is all-true by construction, computed rather than
            // assumed.
            let sync = crate::SyncVerdict::between(adm.host_stamp, stamp);
            let driver = NetDriver {
                rt,
                session,
                me,
                server_eid: host,
                early: Vec::new(),
                id_map,
                departed: Default::default(),
                join_budget: Default::default(),
                unrostered_since: Default::default(),
                telemetry,
                sync,
                stamp,
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
    fn join_budget_allows_a_burst_then_refills_at_the_sustained_rate() {
        let t0 = std::time::Instant::now();
        let mut b = JoinBudget {
            tokens: JOIN_BURST,
            last: t0,
        };
        let burst = (0..20).filter(|_| b.allow(t0)).count();
        assert_eq!(
            burst as f64, JOIN_BURST,
            "a same-instant flood gets the burst, no more"
        );
        assert!(
            !b.allow(t0 + Duration::from_millis(100)),
            "0.1s refills only 0.2 tokens"
        );
        // +0.5s refills another 1.0 token onto the 0.2: one allow, then dry again.
        assert!(
            b.allow(t0 + Duration::from_millis(600)),
            "a whole token accrued"
        );
        assert!(!b.allow(t0 + Duration::from_millis(600)), "…exactly one");
    }

    #[test]
    fn join_budget_caps_refill_at_the_burst() {
        let t0 = std::time::Instant::now();
        let mut b = JoinBudget {
            tokens: JOIN_BURST,
            last: t0,
        };
        let later = t0 + Duration::from_secs(3600);
        let allowed = (0..20).filter(|_| b.allow(later)).count();
        assert_eq!(
            allowed as f64, JOIN_BURST,
            "an idle hour banks no extra tokens"
        );
    }

    #[test]
    fn solo_round_advances_through_the_coordinator() {
        use crate::sim::Input;
        let me = PlayerId(0);
        let mut client = ClientSim::new(0x5A11, &[me], me);
        let mut coord = Coordinator::for_round(None, client.peers(), me, client.sim().clone());
        assert!(
            !coord.is_remote_client() && coord.server().is_some(),
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
                let poses = crate::sim::hold_poses(server.sim());
                let bytes = server.step_next(&poses, Default::default()).snapshot;
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
