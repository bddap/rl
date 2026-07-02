//! Synchronous bridge from the async iroh [`transport`] to a Bevy/game main loop.
//!
//! [`transport::Session`] is async (tokio); the deterministic lockstep driver and
//! the Bevy render loop are sync and own the main thread. [`NetDriver`] bridges the
//! two: it holds a tokio runtime + the session and exposes non-blocking calls a per-frame
//! system uses to play either role — the host drains clients' inputs into its [`Server`]
//! and broadcasts the authoritative [`CoreSnapshot`] it steps, a client ships its input up
//! and adopts the snapshots down. [`Coordinator`] wraps that into the single per-tick [`Coordinator::exchange`]
//! every driver calls, so solo / host / client are one path (rl#151). No determinism lives
//! here; it is pure I/O plumbing (the same split the netcode draws between [`transport`] and
//! [`crate::lockstep`]/[`crate::server`]).
//!
//! The LAN cold-start itself (the membership barrier, id assignment, and the solo
//! auto-fallback) lives in [`crate::formation`]; the entrypoints here
//! ([`connect_and_form`] and friends) bind the endpoint, run it, and wrap the agreed
//! match in a [`NetDriver`].

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::Result;
use iroh::EndpointId;

use crate::articulation::CrabArticulation;
use crate::formation::{Formation, LobbyControl, early_peer_msgs, form_match};
use crate::lockstep::{Lockstep, PeerMsg, TickMsg};
use crate::server::{self, Admission, JoinRequest, Server, may_admit_joiner};
use crate::sim::PlayerId;
use crate::snapshot::CoreSnapshot;
use crate::telemetry::{TelemetryEvent, TelemetrySender};
use crate::transport::{self, PeerWire, Session};

/// Owns the tokio runtime + iroh session and bridges them to a synchronous caller — the
/// TRANSPORT half of the server-coordinated model (rl#151). Held by the game loop as a non-send
/// resource (the runtime/session aren't `Sync`).
///
/// Post-formation the peer with the lowest endpoint id ([`PlayerId(0)`]) runs the match
/// [`Server`]; every other peer is a remote client of it. A `NetDriver` carries enough to play
/// either role: on the host it relays clients' inputs into its server and broadcasts the
/// authoritative STATE ([`NetDriver::drain_client_inputs`]/[`NetDriver::broadcast_snapshot`] +
/// [`NetDriver::broadcast_articulation`]); on a client it ships its input UP to the server and
/// drains the state DOWN ([`NetDriver::send_to_server`]/[`NetDriver::drain_server_down`]) to adopt,
/// never re-stepping (rl#151 increment 2 windowed). The role is read off the id alone — no negotiation — so both
/// ends agree by construction. The [`Server`] itself lives in the [`Coordinator`], not here, so
/// the host's server and the solo server are the one type down one path.
pub struct NetDriver {
    rt: tokio::runtime::Runtime,
    session: Session,
    /// This peer's [`PlayerId`]. `me == PlayerId(0)` ⇒ we run the server (the host).
    me: PlayerId,
    /// The server peer's endpoint — the lowest-id endpoint in the roster. A client routes its
    /// input here; on the host this is our own id and is unused (the local input never crosses the
    /// wire — it goes straight into the in-process server).
    server_eid: EndpointId,
    /// Inputs that arrived during formation (a peer that finished the barrier first may already be
    /// sending), mapped to their author's [`PlayerId`]. Drained once by the host into its server at
    /// round start (a client discards them — only the server holds the ledger).
    early: Vec<PeerMsg>,
    /// Live endpoint→PlayerId map (us + peers), seeded by formation (sorted agreed set), grown by
    /// admission ([`NetDriver::admit_endpoint`]) and SHRUNK by departure ([`depart_gone_peers`],
    /// rl#198). Used to tag inbound messages with their author's id.
    id_map: BTreeMap<EndpointId, PlayerId>,
    /// Endpoints DEPARTED from the live match (rl#198) — kept so a departed-but-alive peer that
    /// re-links (the mDNS dialer runs all match) and keeps sending inputs is REFUSED once, loudly,
    /// instead of silently spectating with dead controls ([[silent-fallback-antipattern]]). An
    /// entry is consumed by the refusal; couch-scale, so never more than a handful.
    departed: std::collections::BTreeSet<EndpointId>,
    /// Optional live-telemetry stream (set iff the client was launched with a
    /// collector). Best-effort and read-only — see [`crate::telemetry`]; the
    /// windowed driver pushes Tick/Input/RoundDecided/Fault through it.
    telemetry: Option<TelemetrySender>,
    /// The formation barrier's shared-asset verdict (rl#82 weights + rl#100 crab asset, GCR —
    /// [`crate::membership::Membership::sync_verdict`]); read by the arm sites via
    /// [`crate::may_arm_external_crab`]. Each half is `false` without a supplied
    /// checkpoint/resolvable model.
    sync: crate::SyncVerdict,
    /// OUR policy-weights digest and crab-asset digest (the values, not just the synced bools).
    /// The host gates a mid-game joiner on these — [`crate::server::may_admit_joiner`] requires a
    /// non-zero `weights_digest` of the HOST itself (the self-gate; the joiner's brain is never
    /// executed, rl#206) and the joiner's asset digest to equal ours, else a LOUD refusal (Stage 3,
    /// rl#151). A client never reads them (only the host admits).
    weights_digest: u64,
    asset_digest: u64,
}

/// The result of [`Coordinator::exchange`]: on a remote client, the host-authoritative game STATE it
/// drained this tick (snapshots + render articulation) to adopt — grouped so the single inbox drain
/// ([`NetDriver::drain_server_down`]) yields them together without one frame kind starving another.
/// The solo/host arm returns these empty — its own server is the source of truth and it steps +
/// broadcasts state itself. Roster changes ride each [`CoreSnapshot`]'s `roster`, not a separate
/// channel.
#[derive(Default)]
pub struct Exchanged {
    /// Host-authoritative game states this remote client drained (rl#151 increment 2 windowed),
    /// in ARRIVAL order: the driver adopts every one via [`Lockstep::adopt_snapshots`] (the one
    /// shared adopt policy — see its doc) instead of stepping its own sim. Empty on the solo/host
    /// arm (its client reads the server it runs).
    pub snapshots: Vec<CoreSnapshot>,
    /// Render-only crab poses this remote client drained, beside the snapshots (rl#151 increment 2
    /// windowed): the driver stashes the LAST-ARRIVED (= newest on the reliable ordered stream)
    /// for the render-side apply. Empty off the remote-client arm.
    pub articulations: Vec<CrabArticulation>,
}

impl NetDriver {
    /// The live-telemetry handle, if this client is streaming to a collector. The
    /// render loop reads it to push events (Tick/Input/RoundDecided/Fault). `None` when
    /// launched without `--telemetry`.
    pub fn telemetry(&self) -> Option<&TelemetrySender> {
        self.telemetry.as_ref()
    }

    /// The formation barrier's shared-asset verdict (rl#82 weights + rl#100 crab asset, GCR —
    /// [`crate::membership::Membership::sync_verdict`]). The arm sites gate the float NN crab
    /// on it via [`crate::may_arm_external_crab`]; an unsynced half means the round can't arm
    /// Sally and is refused (rl#114, no integer fallback). A half is always `false` without a
    /// supplied checkpoint / resolvable model (that digest exchanged as `0`).
    pub fn sync_verdict(&self) -> crate::SyncVerdict {
        self.sync
    }

    /// Whether this peer runs the match server (the host) — true iff it holds [`PlayerId(0)`], the
    /// lowest endpoint id. The server-vs-client role with no negotiation.
    pub fn is_host(&self) -> bool {
        self.me == PlayerId(0)
    }

    /// The frozen roster (every player, incl. us) — what the host builds its [`Server`] over.
    pub fn roster(&self) -> Vec<PlayerId> {
        self.id_map.values().copied().collect()
    }

    /// Take the formation-time early inputs (drained once by the host into its server).
    pub fn take_early(&mut self) -> Vec<PeerMsg> {
        std::mem::take(&mut self.early)
    }

    /// (Client) Ship our input UP to the server. Non-blocking: `send` only ENQUEUES to the link's
    /// writer task (see [`transport`]), so the `block_on` returns immediately. Losing the server
    /// link stalls us — the correct visible failure.
    pub fn send_to_server(&self, msg: &TickMsg) {
        self.rt.block_on(self.session.send(self.server_eid, msg));
    }

    /// (Host) Drain every client INPUT received so far, tagged with the sender's [`PlayerId`].
    /// Non-blocking. Messages from an endpoint not in the frozen set, and any stray non-input frame
    /// (a barrier beat from a peer still winding down formation), are dropped — the server only
    /// ledgers rostered clients' inputs.
    pub fn drain_client_inputs(&mut self) -> (Vec<PeerMsg>, Vec<(EndpointId, JoinRequest)>) {
        let mut inputs = Vec::new();
        let mut joins = Vec::new();
        while let Some(from) = self.session.try_recv() {
            match from.msg {
                // A rostered client's input → ledger it; a not-yet-rostered endpoint's stray input
                // is dropped (the server's `record` would drop it anyway — it isn't rostered at that
                // tick yet), so a joiner's pre-admit frame never blocks the round (333 stays fixed).
                PeerWire::Tick(msg) => {
                    if let Some(&pid) = self.id_map.get(&from.from) {
                        inputs.push(PeerMsg { pid, msg });
                    } else if self.departed.remove(&from.from) {
                        // A DEPARTED endpoint re-linked (the mDNS dialer runs all match) and is
                        // still sending inputs — it doesn't know it was dropped. Tell it ONCE,
                        // loudly (its client error-logs a mid-match Refuse), rather than let it
                        // spectate with dead controls (rl#198). A fresh joiner is untouched: it
                        // was never in `departed`.
                        self.refuse_joiner(
                            from.from,
                            "you were dropped from the match (connection lost) — rejoin",
                        );
                    }
                }
                // A would-be joiner dialing the live match (Stage 3) — surfaced for the coordinator
                // to gate + admit (it holds the `Server`; this driver only holds the transport).
                PeerWire::JoinRequest(req) => joins.push((from.from, req)),
                // Beats (a peer winding down formation), our own broadcasts echoed, etc. — not the
                // server's concern.
                _ => {}
            }
        }
        (inputs, joins)
    }

    /// (Host) The host's OWN digests, feeding the mid-game admission gate: the weights digest is
    /// the host self-gate, the asset digest the equality a joiner must match
    /// ([`crate::server::may_admit_joiner`]).
    pub fn local_digests(&self) -> (u64, u64) {
        (self.weights_digest, self.asset_digest)
    }

    /// (Host) Record an admitted joiner's endpoint→[`PlayerId`] in the live id_map — append-only,
    /// NEVER renumbering an incumbent (the determinism-stability guarantee: a positional renumber on
    /// join would instantly desync). The pid is the Server's lowest-free allocation
    /// ([`Server::admit`]). Now the joiner's inbound [`TickMsg`]s tag with this pid.
    pub fn admit_endpoint(&mut self, eid: EndpointId, pid: PlayerId) {
        self.id_map.insert(eid, pid);
    }

    /// (Host) Whether `eid` is already a rostered player — so a repeated [`JoinRequest`] (a joiner
    /// re-dialing, or its frames racing the admit) is admitted at most once.
    pub fn is_rostered(&self, eid: EndpointId) -> bool {
        self.id_map.contains_key(&eid)
    }

    /// (Host) UNICAST a just-admitted joiner its OWN [`Admission`] (Stage 3) — the welcome it builds
    /// [`Lockstep::join_at`] from. Unicast so a joiner never adopts a concurrent joiner's PlayerId.
    /// Incumbents learn the new roster from the next [`CoreSnapshot`] (it carries the roster), so no
    /// broadcast notice is needed.
    pub fn welcome_joiner(&self, eid: EndpointId, adm: &Admission) {
        self.rt.block_on(self.session.send(eid, adm));
    }

    /// (Host) LOUDLY refuse a would-be joiner `eid` with `reason` (a digest mismatch) — a typed
    /// turn-away, never a silent drop onto a wrong crab. Stage 3.
    pub fn refuse_joiner(&self, eid: EndpointId, reason: &str) {
        self.rt.block_on(
            self.session
                .send(eid, &transport::Refuse(reason.to_string())),
        );
    }

    /// (Host) Broadcast the host-authoritative [`CoreSnapshot`] DOWN to every client (rl#151
    /// increment 2 windowed): the windowed host ships the full game STATE its client just adopted,
    /// so a remote client renders it instead of re-stepping. Non-blocking: enqueued to each link's
    /// writer task, so a dead peer can never hold this (main-thread) call (rl#198).
    pub fn broadcast_snapshot(&self, snapshot: &CoreSnapshot) {
        self.rt.block_on(self.session.broadcast_snapshot(snapshot));
    }

    /// (Host) Broadcast the render-only [`CrabArticulation`] DOWN to every client (rl#151 increment
    /// 2 windowed), beside the snapshot, so a remote client renders the host's exact crab pose
    /// without simulating physics. Non-blocking, like [`Self::broadcast_snapshot`].
    pub fn broadcast_articulation(&self, articulation: &CrabArticulation) {
        self.rt
            .block_on(self.session.broadcast_articulation(articulation));
    }

    /// (Host) The endpoints we currently hold a link to — the sync view of
    /// [`Session::connected_peers`] that [`depart_gone_peers`] diffs the roster against.
    pub fn connected_peers_now(&self) -> Vec<EndpointId> {
        self.rt.block_on(self.session.connected_peers())
    }

    /// (Client) Drain everything the server sent DOWN this tick: host-authoritative
    /// [`CoreSnapshot`]s (rl#151 increment 2 windowed — the client ADOPTS them, never re-steps an
    /// input set) and the render-only [`CrabArticulation`]s beside them. A [`PeerWire::Refuse`]
    /// aimed at us is logged LOUD (an established client should never get one), never silently
    /// eaten. Drained ONCE so no frame kind starves another; snapshots are handed over in ARRIVAL
    /// order for [`Lockstep::adopt_snapshots`] (the one shared adopt policy) to apply.
    pub fn drain_server_down(&mut self) -> Exchanged {
        let mut down = Exchanged::default();
        while let Some(from) = self.session.try_recv() {
            match from.msg {
                PeerWire::Snapshot(snap) => down.snapshots.push(snap),
                PeerWire::Articulation(art) => down.articulations.push(art),
                PeerWire::Refuse(reason) => {
                    tracing::error!("server refused us mid-match: {reason}");
                }
                _ => {}
            }
        }
        down
    }
}

/// One peer's per-tick input coordination — the server-coordinated replacement for the deleted P2P
/// mesh (rl#151). Either we run the [`Server`] (solo: a roster of one, no transport; host: the
/// whole roster + the transport to remote clients) or we are a remote client of a server peer. Solo
/// and host are the SAME [`Coordinator::Server`] arm — that is the SP=MP-uniformity proof: there is
/// no separate single-player code path, only the server with one client.
pub enum Coordinator {
    /// We run the server. `net` is `None` for solo (no remote clients, so no iroh at all — solo
    /// stays network-free) and `Some` for a hosted match (relay to the remote clients).
    Server {
        // Boxed: the [`Server`] now owns the authoritative [`crate::sim::Sim`] (rl#151 increment 1),
        // so it dwarfs the `Client` variant's lone `NetDriver` — heap it to keep the enum balanced.
        server: Box<Server>,
        net: Option<NetDriver>,
    },
    /// We are a remote client of another peer's server.
    Client { net: NetDriver },
}

impl Coordinator {
    /// Build the coordinator for a freshly-formed round. `peers` is the sim's participant set (solo
    /// ⇒ just `me`); `sim` is the tick-0 authoritative world the server steps (a clone of the
    /// client's freshly-built sim, so the two start byte-identical). The carrier stays
    /// `Option<NetDriver>` so the arming + determinism-pin decisions upstream (which key off
    /// `net.is_some()`) are untouched: `None` ⇒ a solo server; a host driver ⇒ a server over the
    /// roster (seeded with any early inputs); a client driver ⇒ a remote client (steps its own
    /// lockstep until increment 2, so `sim` is unused there).
    pub fn for_round(net: Option<NetDriver>, peers: &[PlayerId], sim: crate::sim::Sim) -> Self {
        match net {
            None => Coordinator::Server {
                server: Box::new(Server::new(peers, sim)),
                net: None,
            },
            Some(mut d) if d.is_host() => {
                let mut srv = Server::new(&d.roster(), sim);
                srv.seed_early(&d.take_early());
                Coordinator::Server {
                    server: Box::new(srv),
                    net: Some(d),
                }
            }
            // A remote client ADOPTS the host's per-tick snapshot into its OWN `ls` (rl#151 increment
            // 2 windowed — no re-sim), so the Coordinator holds no authoritative server and this
            // tick-0 `sim` goes unused (the client's `GameState.ls` is what the snapshots advance).
            Some(d) => {
                let _ = sim;
                Coordinator::Client { net: d }
            }
        }
    }

    /// Submit our input for this tick. On the solo/host arm this returns the OTHER players' inputs to
    /// record and steps the authoritative server behind the scenes; on the remote-client arm it ships
    /// our input UP and returns the host's STATE drained down (snapshots + articulation) for the
    /// driver to adopt — no peer inputs, no re-sim (rl#151 increment 2 windowed).
    pub fn exchange(&mut self, me: PlayerId, msg: TickMsg) -> Exchanged {
        match self {
            Coordinator::Server { server, net } => {
                // Drain remote clients' inputs AND any mid-game join requests (none for solo).
                let (remote, joins) = net
                    .as_mut()
                    .map(NetDriver::drain_client_inputs)
                    .unwrap_or_default();
                // Gate + admit each joiner BEFORE assembling this tick, so its roster change is
                // scheduled on the server the same tick it dialed (incumbents learn it from the
                // next snapshot's roster).
                if let Some(net) = net.as_mut() {
                    admit_joiners(server, net, joins);
                }
                let mut sets = server::host_assemble(server, me, msg, remote);
                // Departures LAST (bddap/rl#198), after this tick's drained inputs are recorded —
                // a leaver's inputs for ticks before the departure boundary are still honored;
                // anything at/after it is scrubbed by `depart`. Without this the ledger waits
                // forever on the departed player and the match hard-freezes for everyone who
                // stayed.
                if let Some(net) = net.as_mut() {
                    let connected = net.connected_peers_now();
                    let (dsets, gone) =
                        depart_gone_peers(server, &mut net.id_map, net.me, &connected);
                    sets.extend(dsets);
                    net.departed.extend(gone);
                }
                // Queue the assembled sets for THIS server's authoritative step (rl#151 increment 1):
                // the windowed driver pumps each tick's crab physics then calls `step_next`.
                server.enqueue_for_step(&sets);
                // The host broadcasts the authoritative snapshot + articulation from the driver, the
                // moment it steps each tick (see `Coordinator::broadcast_step`), so a client renders
                // state rather than re-stepping inputs. `sets` feed THIS server's own step queue.
                Exchanged::default()
            }
            Coordinator::Client { net } => {
                // Host-authoritative (rl#151 increment 2 windowed): ship our input UP and drain the
                // host's STATE down (snapshots + render articulation), never an input set to re-step.
                // The driver adopts every drained snapshot ([`Lockstep::adopt_snapshots`]) and
                // renders the last-arrived articulation.
                net.send_to_server(&msg);
                net.drain_server_down()
            }
        }
    }

    /// (Host) Broadcast the authoritative game STATE for the tick the server just stepped: the
    /// `snapshot` bytes it emitted plus the render-only crab `articulation` (rl#151 increment 2
    /// windowed). Called by the windowed driver right after `Server::step_next`, so a remote client
    /// adopts exactly the state the host holds and renders its exact crab pose. A no-op for solo (no
    /// transport) and for a remote client (it broadcasts nothing). `snapshot` is the already-encoded
    /// bytes the driver decoded to apply locally, reused so the wire and the local render agree.
    pub fn broadcast_step(&self, snapshot: &CoreSnapshot, articulation: Option<&CrabArticulation>) {
        if let Coordinator::Server { net: Some(net), .. } = self {
            net.broadcast_snapshot(snapshot);
            if let Some(art) = articulation {
                net.broadcast_articulation(art);
            }
        }
    }

    /// Whether THIS peer is a REMOTE client of another peer's server (rl#151 increment 2 windowed):
    /// it adopts the host's snapshots + renders its articulation, never pumping its own crab physics
    /// or stepping the sim. Distinct from the scripted screenshot harness, which self-sims. The
    /// authoritative solo/host peer returns `false` (it runs [`Self::is_server_authoritative`]).
    pub fn is_remote_client(&self) -> bool {
        matches!(self, Coordinator::Client { .. })
    }

    /// Whether THIS peer runs the authoritative server for the round (solo or host) — so its local
    /// client renders the per-tick [`CoreSnapshot`] the server emits instead of stepping its own
    /// sim (rl#151 increment 1). A remote client returns `false` and instead ADOPTS the host's
    /// snapshots ([`Self::is_remote_client`], rl#151 increment 2 windowed). This is the
    /// Minecraft-model server/client role, NOT an SP/MP split — SP and host take the SAME arm
    /// ([[sp-is-mp-special-case]]).
    pub fn is_server_authoritative(&self) -> bool {
        matches!(self, Coordinator::Server { .. })
    }

    /// The authoritative server, if THIS peer runs one (solo or host); `None` for a remote client.
    pub fn server_mut(&mut self) -> Option<&mut Server> {
        match self {
            Coordinator::Server { server, .. } => Some(&mut **server),
            Coordinator::Client { .. } => None,
        }
    }

    /// Read-only view of the authoritative server, if THIS peer runs one.
    pub fn server(&self) -> Option<&Server> {
        match self {
            Coordinator::Server { server, .. } => Some(&**server),
            Coordinator::Client { .. } => None,
        }
    }

    /// The transport, if any (host or client). `None` for solo.
    fn net(&self) -> Option<&NetDriver> {
        match self {
            Coordinator::Server { net, .. } => net.as_ref(),
            Coordinator::Client { net } => Some(net),
        }
    }

    /// The live-telemetry handle, if this round streams to a collector (`None` solo / no collector).
    pub fn telemetry(&self) -> Option<&TelemetrySender> {
        self.net().and_then(NetDriver::telemetry)
    }
}

/// Detect and CONSUME client departures on a host (bddap/rl#198) — the ONE departure-handling
/// home, shared by the windowed [`Coordinator::exchange`] and the headless `game net` driver
/// (the same sync/async split precedent as [`host_assemble`]; the caller supplies `connected`
/// because only it knows whether to `block_on` or `await` the session).
///
/// A rostered endpoint (excluding `me` — the host holds no link to itself) with no entry in
/// `connected` has DEPARTED: a link exists for every rostered remote by construction (formation
/// and admission both ran over it), so its absence means the connection CLOSED — a clean peer
/// exit, a dead-connection write failure, or a wedged-peer eviction; never merely "not yet
/// connected". Each departed player is removed from `id_map` (which is what lets the same person
/// REJOIN: a fresh process dials as a new endpoint through the normal [`JoinRequest`] admission)
/// and [`Server::depart`]ed; the released [`TickSet`]s are returned for
/// [`Server::enqueue_for_step`] beside the tick's assembled sets, and the departed endpoints are
/// returned for the caller's departed-endpoint bookkeeping ([`NetDriver::drain_client_inputs`]'s
/// refuse-once). Level-triggered off the link table (no event to lose): a departure missed this
/// tick is caught the next.
pub fn depart_gone_peers(
    server: &mut Server,
    id_map: &mut BTreeMap<EndpointId, PlayerId>,
    me: PlayerId,
    connected: &[EndpointId],
) -> (Vec<server::TickSet>, Vec<EndpointId>) {
    let gone: Vec<(EndpointId, PlayerId)> = id_map
        .iter()
        .filter(|(eid, pid)| **pid != me && !connected.contains(eid))
        .map(|(eid, pid)| (*eid, *pid))
        .collect();
    let mut sets = Vec::new();
    let mut eids = Vec::new();
    for (eid, pid) in gone {
        id_map.remove(&eid);
        println!(
            "player {pid:?} ({}) departed — continuing without them",
            eid.fmt_short()
        );
        sets.extend(server.depart(pid));
        eids.push(eid);
    }
    (sets, eids)
}

/// Gate + admit each mid-game `joins` request on the host (Stage 3, rl#151). For each joiner:
/// verify the host is armed and the joiner's collider digest matches ([`may_admit_joiner`]); then,
/// allocate the stable lowest-free [`PlayerId`] ([`Server::admit`] — which schedules the roster
/// change on the authoritative server), record its endpoint→pid append-only, and UNICAST the
/// joiner its [`Admission`] (incumbents learn the new roster from the next snapshot). On a
/// mismatch, REFUSE LOUDLY — a wire refusal + an error log + telemetry — never a silent drop onto
/// a wrong crab ([[real-sally-definition]], rl#114). An endpoint already rostered (a re-dial or a
/// racing duplicate) is admitted at most once.
fn admit_joiners(server: &mut Server, net: &mut NetDriver, joins: Vec<(EndpointId, JoinRequest)>) {
    let (host_weights, host_assets) = net.local_digests();
    for (eid, req) in joins {
        if net.is_rostered(eid) {
            continue; // already in the match — a duplicate/racing JoinRequest
        }
        match may_admit_joiner(host_weights, host_assets, &req) {
            Ok(()) => {
                let adm = server.admit();
                net.admit_endpoint(eid, adm.pid);
                net.welcome_joiner(eid, &adm);
                println!(
                    "admitted joiner {} as {:?}, roster change effective at tick {}",
                    eid.fmt_short(),
                    adm.pid,
                    adm.effective_tick
                );
            }
            Err(refusal) => {
                let reason = refusal.to_string();
                tracing::error!("refused mid-game joiner {}: {reason}", eid.fmt_short());
                net.refuse_joiner(eid, &reason);
                if let Some(t) = net.telemetry() {
                    t.send(TelemetryEvent::RosterFailed {
                        reason: format!("join refused: {reason}"),
                    });
                }
            }
        }
    }
}

/// The outcome of [`connect_and_form`]: either we joined a networked match (a ready
/// [`Lockstep`] + the [`NetDriver`] pumping its transport) or the discovery window
/// elapsed with no other peer present, so the caller should run solo. An enum, not a
/// bare `Option<NetDriver>`, so "no driver" can't be confused with "telemetry off" or
/// any other absence — `Alone` means exactly "play offline" and nothing else.
pub enum MatchResult {
    /// A networked match formed: the agreed [`Lockstep`] and the driver for its peers.
    /// Boxed because both payloads are large and `Alone` is empty: without the box the
    /// enum is sized to `Joined`, so every `Alone` carries that dead weight too.
    Joined(Box<(Lockstep, NetDriver)>),
    /// Discovery completed with only us on the LAN (no peer ever beat us within the
    /// window). The caller starts a deterministic solo round instead of awaiting an
    /// empty networked match — see the module-level solo-fallback note.
    Alone,
    /// The user cancelled out of the host-triggered lobby before a match formed. The
    /// session is torn down before returning, so no LAN phantom lingers. The caller drops
    /// back to the menu — distinct from [`MatchResult::Alone`], which is a round to play;
    /// `Cancelled` is "play nothing, go back".
    Cancelled,
}

/// Bind the LAN endpoint, run the shared [`form_match`] cold-start (the membership
/// barrier), and either return a ready [`Lockstep`] + the [`NetDriver`] that pumps its
/// transport ([`MatchResult::Joined`]) — the windowed client's entry into a match — or,
/// when the discovery window elapses with no other peer present, [`MatchResult::Alone`]
/// so the caller falls back to a solo round. The match `seed` must be identical on every
/// peer (the caller passes the shared constant). `expect` is the minimum participant count
/// to close on (see [`form_match`]); `discover_secs` bounds how long we wait for a peer
/// before concluding we are alone.
///
/// Pure mDNS discovery (no explicit dial). The boot-menu Join-by-code path uses
/// [`connect_and_form_dialing`] to additionally direct-dial a host's endpoint id; this is
/// that with `dial == None`.
pub fn connect_and_form(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    collector: Option<iroh::EndpointId>,
) -> Result<MatchResult> {
    // No Policy is loaded on the scripted/headless path (weights digest `0` ⇒ a round HOSTED
    // by this peer can never arm the NN crab), but advertise our REAL crab-asset digest
    // (rl#100) so the value is honest if this peer ever forms with a rendered peer that does
    // arm.
    connect_and_form_dialing(
        seed,
        discover_secs,
        expect,
        None,
        collector,
        None,
        0,
        crab_world::bot::meshfit::crab_asset_digest(),
    )
}

/// [`connect_and_form`] plus an optional direct dial of a host's endpoint id before the
/// barrier runs — the boot-menu Join-by-code path. `dial == Some(host)` opens a QUIC link
/// to `host` (its LAN address resolved via the endpoint's registered mDNS lookup, so a bare
/// id is enough on the local network) so formation has a peer even when mDNS discovery is
/// slow/missed; `dial == None` is the plain mDNS path ([`connect_and_form`]).
///
/// Determinism is untouched by the dial: it only establishes a connection. The roster
/// still comes wholly from the [`form_match`] barrier — every peer freezes the identical
/// sorted set — so dialing the wrong/typo'd code simply fails to form a match (the
/// barrier never hears an agreeing peer and falls back to [`MatchResult::Alone`] or
/// errors), it can NEVER form a divergent roster. If the dial itself fails (bad code, host
/// gone) we log and proceed to the barrier anyway, which then resolves alone/failed — a
/// dial error must not be more fatal than an absent peer.
///
/// `on_bound` (if any) is sent our own endpoint id the instant the session binds — before
/// the slow barrier — so a Host UI can display the join code to share while waiting. A
/// closed receiver is ignored (the caller stopped caring); it never gates formation.
///
/// `local_weights_digest` is OUR policy-checkpoint digest (rl#82, GCR), `0` for none, and
/// `local_asset_digest` OUR crab-model-asset digest (rl#100, GCR), `0` for none. Both are
/// advertised in the formation beats; the agreed [`NetDriver::sync_verdict`] tells the
/// caller whether the round may arm the NN crab (the HOST's brain verified + the crab asset
/// matched on every peer — the upstream shared-asset guard).
#[allow(clippy::too_many_arguments)] // each arg is a distinct formation knob.
pub fn connect_and_form_dialing(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    dial: Option<iroh::EndpointId>,
    collector: Option<iroh::EndpointId>,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    local_weights_digest: u64,
    local_asset_digest: u64,
) -> Result<MatchResult> {
    // The scripted/headless path: no interactive lobby (`None`), so the default
    // (timer-closed) barrier.
    connect_and_form_inner(
        seed,
        discover_secs,
        expect,
        dial,
        collector,
        on_bound,
        None,
        local_weights_digest,
        local_asset_digest,
    )
}

/// The boot-menu networked entry: [`connect_and_form_dialing`] plus a [`LobbyControl`] for
/// the host-triggered lobby — the [`Role`], a host's Start signal, a Cancel that detaches
/// without a LAN phantom, and a live roster feed. Passing the control IS what selects the
/// lobby barrier (a joiner lobbies until the host clicks Start) over the timer-closed one; the
/// roster it freezes is identical either way (the membership core's guarantee), so determinism
/// is untouched. No `discover_secs` — the lobby is open-ended until the host starts or someone
/// cancels.
#[allow(clippy::too_many_arguments)] // each arg is a distinct formation knob.
pub fn connect_and_form_lobby(
    seed: u64,
    expect: usize,
    dial: Option<iroh::EndpointId>,
    collector: Option<iroh::EndpointId>,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    control: LobbyControl,
    local_weights_digest: u64,
    local_asset_digest: u64,
) -> Result<MatchResult> {
    connect_and_form_inner(
        seed,
        0,
        expect,
        dial,
        collector,
        on_bound,
        Some(control),
        local_weights_digest,
        local_asset_digest,
    )
}

/// The shared body of both networked entrypoints: bind, optionally dial, run the barrier
/// (timer-closed when `lobby` is `None`, host-triggered when `Some`), and build the
/// [`Lockstep`] + driver. One definition so the scripted and lobby paths can't drift on the
/// cold-start.
#[allow(clippy::too_many_arguments)] // every arg is a distinct formation knob; bundling further would obscure them.
fn connect_and_form_inner(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    dial: Option<iroh::EndpointId>,
    collector: Option<iroh::EndpointId>,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
    lobby: Option<LobbyControl>,
    local_weights_digest: u64,
    local_asset_digest: u64,
) -> Result<MatchResult> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (session, formation, telemetry) = rt.block_on(async {
        let mut session = transport::start_session().await?;
        let my_eid = session.endpoint_id();
        println!("fp client endpoint id: {my_eid}");
        // Report our bound id up front (best-effort) so a Host lobby can show the join code
        // immediately, without waiting out the barrier.
        if let Some(tx) = &on_bound {
            let _ = tx.send(my_eid);
        }
        // Join-by-code: direct-dial the host so the barrier has a peer without waiting on
        // mDNS. Best-effort — a failed dial (bad code / host absent) is logged and we fall
        // through to the barrier, which resolves the absence as Alone/Failed just like an
        // unreachable peer. Never read a roster from the dial; it only opens the link.
        if let Some(host) = dial {
            if host == my_eid {
                tracing::warn!("join code is our own endpoint id — ignoring the self-dial");
            } else if let Err(e) = session.connect_direct(host).await {
                tracing::warn!("dialing host {} failed: {e:#}", host.fmt_short());
            }
        }
        // Open the telemetry side-channel BEFORE forming the match, so the collector
        // sees the roster fill (RosterForming/Agreed). Best-effort: a failure to bind
        // the telemetry endpoint just runs the game without it.
        let telemetry = connect_telemetry(collector, my_eid).await;
        let formation = form_match(
            &mut session,
            discover_secs,
            expect,
            telemetry.as_ref(),
            lobby.as_ref(),
            local_weights_digest,
            local_asset_digest,
        )
        .await?;
        anyhow::Ok((session, formation, telemetry))
    })?;

    let frozen = match formation {
        Formation::Agreed(frozen) => frozen,
        // No peer showed up: drop the session/telemetry (runtime + endpoint tear down on
        // drop) and tell the caller to play offline.
        Formation::Alone => return Ok(MatchResult::Alone),
        // The player cancelled the lobby: tear the session down NOW (not on a lazy drop) so
        // no LAN phantom lingers, then report Cancelled so the caller returns to the menu.
        Formation::Cancelled => {
            drop(telemetry);
            rt.block_on(session.shutdown());
            return Ok(MatchResult::Cancelled);
        }
    };

    let all_ids: Vec<PlayerId> = frozen.id_map.values().copied().collect();
    println!(
        "starting lockstep: {} player(s), I am {:?}",
        all_ids.len(),
        frozen.me
    );
    // Every peer spawns the byte-identical foot-only round. The early inputs are NOT
    // replayed into the client sim anymore (that would bypass the server's ledger) — they ride the
    // driver to seed the host's server instead (see [`Coordinator::for_round`]).
    let ls = Lockstep::new(seed, &all_ids, frozen.me);
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
        weights_digest: local_weights_digest,
        asset_digest: local_asset_digest,
    };
    Ok(MatchResult::Joined(Box::new((ls, driver))))
}

/// How long a joiner waits for the host's admission verdict after sending its [`JoinRequest`]
/// before giving up (the host unreachable, or not running a joinable match). Generous — it spans a
/// QUIC handshake plus the host noticing the request on its next tick drain.
const JOIN_WELCOME_TIMEOUT: Duration = Duration::from_secs(10);

/// The outcome of [`connect_and_join`]: a mid-game join either took (a ready joiner [`Lockstep`] +
/// its client [`NetDriver`]), was REFUSED by the host (an unarmed host or a collider-digest
/// mismatch — surfaced, not silent), or the host was UNREACHABLE / never answered.
pub enum JoinResult {
    Joined(Box<(Lockstep, NetDriver)>),
    Refused(String),
    Unreachable,
}

/// The host's verdict on our [`JoinRequest`], read off the wire.
enum AdmissionVerdict {
    Admitted(Admission),
    Refused(String),
    Timeout,
}

/// Read the host's verdict after we send a [`JoinRequest`]: our UNICAST [`PeerWire::Welcome`] (our
/// own [`Admission`]) or a [`PeerWire::Refuse`]. Frames from anyone but the host, and any other
/// kind, are ignored. Bounded by [`JOIN_WELCOME_TIMEOUT`].
async fn await_admission(session: &mut Session, host: EndpointId) -> AdmissionVerdict {
    let deadline = tokio::time::timeout(JOIN_WELCOME_TIMEOUT, async {
        loop {
            let Some(from) = session.recv().await else {
                return AdmissionVerdict::Timeout; // session closed
            };
            if from.from != host {
                continue;
            }
            match from.msg {
                PeerWire::Welcome(adm) => return AdmissionVerdict::Admitted(adm),
                PeerWire::Refuse(reason) => return AdmissionVerdict::Refused(reason),
                _ => continue,
            }
        }
    })
    .await;
    deadline.unwrap_or(AdmissionVerdict::Timeout)
}

/// Dial INTO a live match as a host-authoritative mid-game joiner (GCR MP incr 4, rl#151) — the
/// dialing analogue of [`connect_and_form`]. Connect to `host`, send our collider digest as
/// a [`JoinRequest`], and await the host's verdict: admitted (become a remote-adopt
/// [`Coordinator::Client`] that boots from the host's next authoritative snapshot — the host spawns
/// us into its LIVE round at `effective_tick`, so we drop into the ongoing match rather than
/// resetting it; the `join_at` [`Lockstep`] is only the placeholder cursors the adopted snapshot
/// supersedes), refused (a collider mismatch OR a zero-digest host the gate turned away LOUDLY —
/// relayed, never a silent wrong/fake-crab), or unreachable. `seed` is the shared [`crate::sim`]
/// match constant every peer holds.
pub fn connect_and_join(
    seed: u64,
    host: EndpointId,
    collector: Option<EndpointId>,
    local_asset_digest: u64,
) -> Result<JoinResult> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (session, verdict, telemetry) = rt.block_on(async {
        let mut session = transport::start_session().await?;
        let my_eid = session.endpoint_id();
        println!("joining as endpoint id: {my_eid}");
        anyhow::ensure!(host != my_eid, "cannot join our own endpoint id");
        if let Err(e) = session.connect_direct(host).await {
            tracing::warn!("dialing host {} failed: {e:#}", host.fmt_short());
            return anyhow::Ok((session, AdmissionVerdict::Timeout, None));
        }
        let telemetry = connect_telemetry(collector, my_eid).await;
        session
            .send(
                host,
                &JoinRequest {
                    asset_digest: local_asset_digest,
                },
            )
            .await;
        let verdict = await_admission(&mut session, host).await;
        anyhow::Ok((session, verdict, telemetry))
    })?;

    match verdict {
        AdmissionVerdict::Refused(reason) => {
            tracing::error!("host refused our join: {reason}");
            rt.block_on(session.shutdown());
            Ok(JoinResult::Refused(reason))
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
            // Host-authoritative mid-game join (rl#151 incr 4): this `ls` is only a placeholder the
            // remote-adopt client boots from — the driver is a CLIENT (not host), so `for_round`
            // makes this a `Coordinator::Client` that ADOPTS the host's per-tick snapshots and never
            // steps a sim of its own. The host spawns us into its LIVE authoritative round at
            // `effective_tick` (`Server::step_next` → `Sim::spawn_joining_player`), so the first
            // snapshot we adopt carries us at the ongoing tick with the crab at its live pose — the
            // 509 fix by construction (we render the host's output, never re-sim its warm rapier
            // world). `join_at` seeds the placeholder cursors/roster; the adopted snapshot supersedes.
            let ls = Lockstep::join_at(seed, &adm.roster, me, adm.effective_tick);
            let my_eid = session.endpoint_id();
            // A client's id_map has NO determinism-path reader (inputs route by `server_eid`, the
            // server's sets carry their own pids, the roster count comes from the lockstep) — it is
            // inert bookkeeping. We know only our own endpoint and the host's, so map those two. The
            // host is PlayerId(0) by formation (the lowest endpoint id runs the server), which the
            // admitted roster must include.
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
                // Admitted ⇒ the host proved a non-zero brain (`HostNotArmed` is checked first
                // in the admission gate) and our asset digest matched the host's — both verdict halves
                // hold by the gate we just passed.
                sync: crate::SyncVerdict {
                    host_brain: true,
                    assets: true,
                },
                // A join-constructed driver is always a Client — it can never reach the Server
                // arm that reads `local_digests` (a joiner never admits) — so no weights digest is
                // computed or threaded here at all (rl#206): 0 marks it deliberately absent.
                weights_digest: 0,
                asset_digest: local_asset_digest,
            };
            Ok(JoinResult::Joined(Box::new((ls, driver))))
        }
    }
}

/// Open a [`TelemetrySender`] to `collector` if one was configured, tagging events with
/// our game endpoint id. Best-effort: any bind failure logs and yields `None`, so the
/// game runs unchanged without telemetry — telemetry can never gate a match.
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

/// The server peer's endpoint: the one holding [`PlayerId(0)`] (the lowest endpoint id). Every
/// peer computes the same answer from the frozen map, so the star agrees on its center with no
/// negotiation. The map is non-empty (we are always in it), so the lookup always resolves.
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

    /// END-TO-END solo through the host-authoritative path: a solo [`Coordinator`] (an internal
    /// server with a roster of one, no transport) — inputs go UP, the server steps its OWN sim, and
    /// the local [`Lockstep`] ADOPTS the emitted snapshot. Proves solo runs the SAME `exchange` +
    /// step machinery as a hosted match (SP=MP uniformity, rl#151), with no special-case solo path.
    #[test]
    fn solo_round_advances_through_the_coordinator() {
        use crate::sim::Input;
        let me = PlayerId(0);
        let mut ls = Lockstep::new(0x5A11, &[me], me);
        let mut coord = Coordinator::for_round(None, ls.peers(), ls.sim().clone());
        assert!(
            matches!(coord, Coordinator::Server { net: None, .. }),
            "no driver ⇒ a solo internal-server coordinator"
        );
        let submits = 5u64;
        for _ in 0..submits {
            let msg = ls.submit_local_input(Input::from_axes(1.0, 0.0));
            // The input goes UP to the internal server; with a roster of one there are no OTHER
            // players' inputs and no joins.
            let exch = coord.exchange(me, msg);
            assert!(
                exch.snapshots.is_empty(),
                "the solo/host arm returns state empty"
            );
            // The server steps its OWN sim and the local client ADOPTS each emitted snapshot — the
            // windowed ServerAuth arm's flow, minus the bevy crab pump (no rapier body here).
            let server = coord.server_mut().expect("solo runs an internal server");
            while server.next_tick_ready() {
                let bytes = server.step_next(None);
                let snap =
                    crate::snapshot::CoreSnapshot::from_bytes(&bytes).expect("snapshot decodes");
                ls.apply_core_snapshot(snap);
            }
        }
        assert_eq!(
            ls.sim().tick(),
            submits,
            "solo advances one tick per submit through the host-authoritative path"
        );
    }
}
