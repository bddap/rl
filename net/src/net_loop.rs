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
//! The LAN cold-start ([`form_match`]) — run the membership barrier, [`assign_player_ids`]
//! over the agreed set by sorted id, and collect tick inputs that arrived mid-formation —
//! is shared verbatim by the windowed client ([`connect_and_form`]) and the headless
//! `game net` driver. Same code on every peer on purpose: the agreed set + id assignment
//! MUST be byte-identical or the sims silently desync. The callers differ only in what
//! they do with the session after (wrap it in a [`NetDriver`], or drive it raw).
//!
//! Solo auto-fallback: the common launch is ALONE (one process, no LAN peer). The barrier
//! rightly refuses to freeze a guessed roster, so awaiting a match would leave a frozen,
//! unplayable round. So the cold-start has a SECOND outcome ([`Formation::Alone`]) — when
//! discovery elapses with no peer ever heard, the caller plays a deterministic solo round.
//! It fires ONLY in the genuinely-alone case (see [`run_barrier`]); the moment any peer is
//! present the full agreement barrier is back in force, so multiplayer is never weakened.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

use crate::articulation::CrabArticulation;
use crate::lockstep::{Lockstep, TickMsg};
use crate::snapshot::CoreSnapshot;
use crate::membership::{BEAT_EVERY, Membership, Role, Status};
use crate::server::{self, Admission, JoinRequest, Server, may_admit_joiner};
use crate::sim::PlayerId;
use crate::telemetry::{self, TelemetryEvent, TelemetrySender};
use crate::transport::{self, PeerWire, Session};

/// A peer tick message tagged with the sender's already-resolved [`PlayerId`]. The
/// id is mapped from the QUIC-authenticated endpoint id via the frozen
/// participant set (never read from the message body — see [`transport::FromPeer`]),
/// so the lockstep driver can trust it as the input's true author.
#[derive(Debug, Clone, Copy)]
pub struct PeerMsg {
    pub pid: PlayerId,
    pub msg: TickMsg,
}

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
    /// Frozen endpoint→PlayerId map (us + peers), agreed across peers by sorting the
    /// agreed participant set. Used to tag inbound messages with their author's id.
    id_map: BTreeMap<EndpointId, PlayerId>,
    /// Optional live-telemetry stream (set iff the client was launched with a
    /// collector). Best-effort and read-only — see [`crate::telemetry`]; the
    /// windowed driver pushes Tick/Input/RoundDecided/Fault through it.
    telemetry: Option<TelemetrySender>,
    /// Whether the barrier agreed every peer loaded the same non-zero weights (rl#82, GCR —
    /// [`Membership::weights_synced`]); read by the arm sites via
    /// [`crate::may_arm_external_crab`]. `false` without a supplied checkpoint.
    weights_synced: bool,
    /// Whether the barrier agreed every peer resolved the same non-zero crab-model asset
    /// (rl#100, GCR — [`Membership::assets_synced`]); ANDed with `weights_synced` at the arm
    /// sites. `false` without a resolvable model.
    assets_synced: bool,
    /// OUR policy-weights digest and crab-asset digest (the values, not just the synced bools).
    /// The host gates a mid-game joiner on these — [`crate::server::may_admit_joiner`] requires the
    /// joiner's digests to equal ours, else a LOUD refusal (Stage 3, rl#151). A client never reads
    /// them (only the host admits).
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

    /// Whether the formation barrier agreed every peer loaded the SAME non-zero policy
    /// weights (rl#82, GCR — [`Membership::weights_synced`]). The arm sites gate the float
    /// NN crab on this via [`crate::may_arm_external_crab`]; `false` means the round can't
    /// arm Sally and is refused (rl#114, no integer fallback). Always `false` without a supplied
    /// checkpoint (the digest exchanged was `0`).
    pub fn weights_synced(&self) -> bool {
        self.weights_synced
    }

    /// Whether the formation barrier agreed every peer resolved the SAME non-zero crab-model
    /// asset (rl#100, GCR — [`Membership::assets_synced`]). The arm sites AND this with
    /// [`NetDriver::weights_synced`] via [`crate::may_arm_external_crab`]; a mismatch means
    /// the round can't arm Sally and is refused (rl#114, no integer fallback). Always `false`
    /// without a resolvable model (asset digest exchanged `0`).
    pub fn assets_synced(&self) -> bool {
        self.assets_synced
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

    /// (Client) Ship our input UP to the server. Non-blocking from the caller's view: it drives the
    /// async `send` to completion on the runtime (a buffered QUIC write). Losing the server link
    /// stalls us — the correct visible failure.
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

    /// (Host) The host's OWN digests, the gate a mid-game joiner must match
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
        self.rt
            .block_on(self.session.send(eid, &transport::Refuse(reason.to_string())));
    }

    /// (Host) Broadcast the host-authoritative [`CoreSnapshot`] DOWN to every client (rl#151
    /// increment 2 windowed): the windowed host ships the full game STATE its client just adopted,
    /// so a remote client renders it instead of re-stepping. Non-blocking buffered QUIC writes.
    pub fn broadcast_snapshot(&self, snapshot: &CoreSnapshot) {
        self.rt.block_on(self.session.broadcast_snapshot(snapshot));
    }

    /// (Host) Broadcast the render-only [`CrabArticulation`] DOWN to every client (rl#151 increment
    /// 2 windowed), beside the snapshot, so a remote client renders the host's exact crab pose
    /// without simulating physics. Non-blocking buffered QUIC writes.
    pub fn broadcast_articulation(&self, articulation: &CrabArticulation) {
        self.rt.block_on(self.session.broadcast_articulation(articulation));
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
                let sets = server::host_assemble(server, me, msg, remote);
                // Queue the assembled sets for THIS server's authoritative step (rl#151 increment 1):
                // the windowed driver pumps each tick's crab physics then calls `step_next`. Only the
                // `Coordinator` path enqueues — the headless `game net` host steps straight off
                // `next_tick_ready`, so it never grows this queue.
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

/// Gate + admit each mid-game `joins` request on the host (Stage 3, rl#151). For each joiner:
/// verify its weight/collider digests against the host's ([`may_admit_joiner`]); on a match,
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

/// The host-triggered lobby: the [`Role`] (which decides the close trigger) plus the live
/// control + observation channels. Passing `Some(LobbyControl)` to the formation core IS
/// what selects the lobby barrier; `None` is the default timer-closed barrier. So the
/// determinism-critical mode is an explicit sum (`Option<LobbyControl>`), never inferred from
/// "did some channel happen to be wired".
pub struct LobbyControl {
    /// Host vs joiner. The host commands the synchronized start; a joiner waits for the GO.
    /// Threaded into [`Membership::host_triggered`] so the role is enforced by the type.
    pub role: Role,
    /// A signal that the player clicked **Start**. On the first signal the barrier calls
    /// [`Membership::set_starting`] — which is a structural no-op for a [`Role::Joiner`], so a
    /// joiner can't command a start even if this fires.
    pub start_rx: std::sync::mpsc::Receiver<()>,
    /// A signal that the player clicked **Cancel** (Host or Join). The barrier returns
    /// [`MatchResult::Cancelled`] promptly and tears the session down, so leaving the lobby
    /// doesn't strand a ~12 s LAN phantom.
    pub cancel_rx: std::sync::mpsc::Receiver<()>,
    /// A feed of the current live roster (us + peers, sorted) emitted each beat, for the
    /// lobby's live player list. Best-effort: a full/closed channel just drops the update.
    pub roster_tx: std::sync::mpsc::Sender<Vec<EndpointId>>,
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
    // No Policy is loaded on the scripted/headless path (weights digest `0` ⇒ the NN crab never
    // arms here), but advertise our REAL crab-asset digest (rl#100) so the value is honest if
    // this peer ever forms with a rendered peer that does arm.
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
/// advertised in the formation beats; the agreed [`NetDriver::weights_synced`]/
/// [`NetDriver::assets_synced`] tell the caller whether every peer matched them (the upstream
/// shared-asset guard — the NN crab arms only when both hold).
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
        telemetry,
        weights_synced: frozen.weights_synced,
        assets_synced: frozen.assets_synced,
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
/// its client [`NetDriver`]), was REFUSED by the host (a weight/collider digest mismatch — surfaced,
/// not silent), or the host was UNREACHABLE / never answered.
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
/// dialing analogue of [`connect_and_form`]. Connect to `host`, send our weight/collider digests as
/// a [`JoinRequest`], and await the host's verdict: admitted (become a remote-adopt
/// [`Coordinator::Client`] that boots from the host's next authoritative snapshot — the host spawns
/// us into its LIVE round at `effective_tick`, so we drop into the ongoing match rather than
/// resetting it; the `join_at` [`Lockstep`] is only the placeholder cursors the adopted snapshot
/// supersedes), refused (a digest mismatch OR a zero-digest host the gate turned away LOUDLY —
/// relayed, never a silent wrong/fake-crab), or unreachable. `seed` is the shared [`crate::sim`]
/// match constant every peer holds.
pub fn connect_and_join(
    seed: u64,
    host: EndpointId,
    collector: Option<EndpointId>,
    local_weights_digest: u64,
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
                    weights_digest: local_weights_digest,
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
                telemetry,
                // Admitted ⇒ our weight + asset digests matched the host's (the admission gate), so
                // the round is armable on the same Sally as everyone else.
                weights_synced: true,
                assets_synced: true,
                weights_digest: local_weights_digest,
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

/// The outcome of the LAN cold-start: the frozen participant→[`PlayerId`] map (agreed
/// identical on every peer by the barrier), which id is us, and the tick messages that
/// arrived mid-formation (carried to the host to seed its server ledger — see
/// [`Coordinator::for_round`] — since only the server holds the ledger now).
pub struct Frozen {
    pub id_map: BTreeMap<EndpointId, PlayerId>,
    pub me: PlayerId,
    pub early: Vec<(EndpointId, TickMsg)>,
    /// Carried out of the barrier so [`NetDriver::weights_synced`] can expose it; see
    /// [`Membership::weights_synced`].
    pub weights_synced: bool,
    /// Likewise the crab-asset verdict (rl#100), for [`NetDriver::assets_synced`]; see
    /// [`Membership::assets_synced`].
    pub assets_synced: bool,
}

/// What [`form_match`] resolves to: a real agreed match, the genuinely-alone case, or a
/// lobby cancel. [`Formation::Alone`] is returned ONLY when the discovery window elapsed
/// with no other peer ever heard — never when peers are present-but-not-yet-agreed (that
/// still drives the full barrier to agreement or errors).
pub enum Formation {
    /// The membership barrier agreed on a roster; play networked over it.
    Agreed(Frozen),
    /// Formation ended with only us live — play solo. Both routes are genuinely-alone (a
    /// peer present-and-live at the moment of decision keeps us on the barrier, never here):
    /// the `discover_secs` deadline elapsed never having heard a peer, or the JOIN_WINDOW
    /// expired with `live == 1`. See the [`run_barrier`] fallback notes.
    Alone,
    /// The player cancelled the host-triggered lobby before a match formed. The caller tears
    /// the session down and reports [`MatchResult::Cancelled`].
    Cancelled,
}

/// Run the LAN match-formation barrier ([`run_barrier`]), then freeze the AGREED
/// participant set and assign deterministic [`PlayerId`]s by sorted endpoint id. The
/// ONE cold-start both the windowed client ([`connect_and_form`]) and the headless
/// `game net` driver run — shared verbatim (not just described as shared) because it
/// must stay behaviorally identical on every peer for the sims to agree, and a drift
/// here would silently desync.
///
/// `expect` is the minimum participant count (incl. us) the barrier requires before it
/// will close — it stops a peer from freezing a lone `{self}` match before LAN discovery
/// has even found the others, and makes a too-small turnout time out ([`Status::Failed`])
/// rather than form a short match. `discover_secs` is the alone-fallback deadline: if it
/// elapses with no other peer ever heard, this returns [`Formation::Alone`] for a
/// solo round. It does NOT bound a formation that has peers — once any peer is present the
/// barrier waits for agreement within its own [`crate::membership::JOIN_WINDOW`].
pub async fn form_match(
    session: &mut transport::Session,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
    lobby: Option<&LobbyControl>,
    local_weights_digest: u64,
    local_asset_digest: u64,
) -> Result<Formation> {
    let my_eid = session.endpoint_id();
    println!(
        "forming match on the LAN (need {expect} player(s), solo if alone after {discover_secs}s)…"
    );

    let outcome = match run_barrier(
        session,
        my_eid,
        discover_secs,
        expect,
        telemetry,
        lobby,
        local_weights_digest,
        local_asset_digest,
    )
    .await
    {
        Ok(BarrierResult::Agreed(o)) => o,
        Ok(BarrierResult::Alone) => {
            // Discovery elapsed with only us live — fall back to solo. No
            // RosterAgreed/Failed event: no networked match formed, so the collector
            // shows neither; the caller runs an offline round.
            println!("no other peer found within {discover_secs}s — starting a solo round");
            return Ok(Formation::Alone);
        }
        Ok(BarrierResult::Cancelled) => {
            // The player left the lobby. No telemetry event — no match formed and no
            // failure; the caller tears down the session and returns to the menu.
            println!("lobby cancelled by the player");
            return Ok(Formation::Cancelled);
        }
        Err(e) => {
            // Mirror the formation failure to the collector (best-effort) before
            // surfacing it — the operator sees WHY a deck never started its round.
            if let Some(t) = telemetry {
                t.send(TelemetryEvent::RosterFailed {
                    reason: format!("{e:#}"),
                });
            }
            return Err(e);
        }
    };
    let id_map = assign_player_ids(my_eid, &outcome.roster)?;
    let me = id_map[&my_eid];
    println!(
        "match formed: {} participant(s), barrier agreed in {:.1}s",
        id_map.len(),
        outcome.elapsed.as_secs_f64()
    );
    if let Some(t) = telemetry {
        t.send(TelemetryEvent::RosterAgreed {
            members: telemetry::short_ids(&outcome.roster),
            roster_hash: crate::membership::roster_hash(&outcome.roster),
            me: me.0,
        });
    }
    // GCR shared-asset guard (rl#82 weights + rl#100 crab asset): make the verdict LOUD. With a
    // checkpoint loaded (`local_weights_digest != 0`) the operator needs to know whether the NN
    // crab will arm — it needs BOTH synced weights AND a synced crab asset; a mismatch on either
    // means it WON'T, and with no integer fallback (rl#114) the windowed client REFUSES the round
    // rather than substituting a fake crab. The asset verdict is reported whenever weights are
    // synced (so an asset-only mismatch — the rl#100 hole this closes — is diagnosable, never
    // silent).
    if local_weights_digest != 0 {
        if !outcome.weights_synced {
            tracing::warn!(
                "GCR: weights NOT synced across peers (digest mismatch or a peer has no \
                 checkpoint) — cannot arm the NN crab; the windowed client will REFUSE this round \
                 (rl#114, no integer fallback). Run rl-update on every device so all carry the \
                 identical brain."
            );
        } else if !outcome.assets_synced {
            tracing::warn!(
                "GCR: weights synced but crab MODEL ASSET NOT synced across peers (a peer has a \
                 different sally.glb / no model — different colliders would desync) — cannot arm \
                 the NN crab; the windowed client will REFUSE this round (rl#114, no integer \
                 fallback). Run rl-update on every device so all carry the identical crab model."
            );
        } else {
            println!(
                "GCR: policy weights AND crab asset synced across all {} peer(s) — NN crab \
                 eligible for lockstep",
                id_map.len()
            );
        }
    }
    Ok(Formation::Agreed(Frozen {
        id_map,
        me,
        early: outcome.early,
        weights_synced: outcome.weights_synced,
        assets_synced: outcome.assets_synced,
    }))
}

/// What a successful [`run_barrier`] yields: the agreed roster (sorted endpoint ids,
/// identical on every peer), the tick messages that arrived mid-formation (a peer that
/// finished the barrier first may already be broadcasting inputs), and how long agreement
/// took.
struct BarrierOutcome {
    roster: Vec<EndpointId>,
    early: Vec<(EndpointId, TickMsg)>,
    elapsed: Duration,
    /// [`Membership::weights_synced`] sampled at the close instant (rl#82, GCR).
    weights_synced: bool,
    /// [`Membership::assets_synced`] sampled at the close instant (rl#100, GCR).
    assets_synced: bool,
}

/// The non-error outcomes of [`run_barrier`]: a real agreement, the alone fallback, or a
/// user cancel. All three are distinct from the `Err` path on purpose — being alone is a
/// normal launch (play solo), cancelling is a deliberate exit, and only a genuine formation
/// failure is an error (relaunch).
enum BarrierResult {
    Agreed(BarrierOutcome),
    Alone,
    /// The player clicked Cancel in the host-triggered lobby; leave the barrier promptly so
    /// the session can be torn down before a LAN phantom forms.
    Cancelled,
}

/// Drive the membership barrier to agreement: beat our view every [`BEAT_EVERY`],
/// ingest peers' beats (and stash any early ticks), and poll the [`Membership`] state
/// machine until it returns [`Status::Agreed`] (freeze) or [`Status::Failed`] (give up
/// and error — never freeze a guessed set). `expect` is the minimum participant count to
/// close. The pure agreement logic lives in [`crate::membership`]; this is only the
/// I/O around it, plus the solo and lobby policies layered ON TOP (so `Membership` stays
/// pure and never itself freezes a guessed roster or aborts a formation that has peers):
///
/// - Solo fallback: when defaulted-networked and genuinely alone past the discovery
///   window, return [`BarrierResult::Alone`] so the caller plays solo. The exact predicate
///   (and why "alone" must mean never-heard-a-peer, not lost-a-peer) is [`is_alone_now`]
///   for the `Forming` deadline and [`is_alone_at_timeout`] for the `Failed` window expiry;
///   both read `live` AFTER `poll`, so a peer mid-handshake shows `live >= 2` and holds us
///   on the real barrier. A `live >= 2` failure stays the loud `Failed` ("relaunch
///   together") — real peers that never agreed is a genuine multi-peer fault.
/// - Lobby (`lobby == Some`): the barrier closes on the host's GO, not the timer. Each beat
///   we call [`Membership::set_starting`] once `start_rx` fires, return
///   [`BarrierResult::Cancelled`] the instant `cancel_rx` fires (so the session tears down
///   before a LAN phantom forms), and push the live roster to `roster_tx`. The solo
///   fallback is SKIPPED here — a lone host starts via the UI's instant-solo path and never
///   reaches this barrier, so being here means the host means to wait (open-ended).
///
/// Residual (a product call, not a bug): two co-launched peers that NEVER hear each other
/// within their windows both solo independently — inherent to a unilateral solo decision
/// with no "we agree nobody's here" exchange. `discover_secs` shrinks the window.
#[allow(clippy::too_many_arguments)] // each arg is a distinct formation knob.
async fn run_barrier(
    session: &mut Session,
    me: EndpointId,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
    lobby: Option<&LobbyControl>,
    local_weights_digest: u64,
    local_asset_digest: u64,
) -> Result<BarrierResult> {
    let start = Instant::now();
    // `Some(control)` is the interactive lobby (host-triggered close per its `role`); `None`
    // is the default timer barrier. The mode is this explicit choice, never inferred. Our
    // weights digest (rl#82) and crab-asset digest (rl#100) ride every beat so peers can agree
    // on a shared checkpoint AND a shared collider asset before arming the float NN crab.
    let mut m = match lobby {
        Some(c) => Membership::host_triggered(c.role, me, expect, start),
        None => Membership::new(me, expect, start),
    }
    .with_weights_digest(local_weights_digest)
    .with_asset_digest(local_asset_digest);
    let mut early: Vec<(EndpointId, TickMsg)> = Vec::new();
    let mut ticker = tokio::time::interval(BEAT_EVERY);
    let mut last_live = 0usize;
    let mut last_roster: Vec<EndpointId> = Vec::new();
    // Whether we've EVER received a direct beat from any peer this formation. Gates the solo
    // fallback: heard-then-lost is a link failure (loud `Failed`), only never-heard is solo.
    let mut ever_heard_peer = false;
    // Deadline past which "still only us" means play solo. `.max(1)` so a `discover_secs`
    // of 0 can't declare us alone before discovery has run a single beat. (The `expect > 1`
    // and never-heard-a-peer gates that keep this from preempting a deliberate solo-over-
    // network run or silently dropping a lost peer live in `is_alone_now`.)
    let alone_deadline = start + Duration::from_secs(discover_secs.max(1));

    loop {
        ticker.tick().await;
        let now = Instant::now();

        // Lobby controls, checked before forming: a Cancel ends the barrier NOW; a
        // Start latches the host into commanding the GO (a no-op for a joiner — `set_starting`
        // is structurally inert off [`Role::Host`]). `try_recv` is non-blocking; a closed
        // cancel channel (the UI dropped its sender) is treated as a cancel — the lobby is gone.
        if let Some(c) = lobby {
            match c.cancel_rx.try_recv() {
                Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Ok(BarrierResult::Cancelled);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if c.start_rx.try_recv().is_ok() {
                m.set_starting();
            }
        }

        // Ingest everything the transport has: beats feed the membership machine;
        // ticks are from a peer that already finished forming — hold them to replay
        // once we have a Lockstep (the transport doesn't resend, so dropping them would
        // stall us on that peer's first inputs).
        while let Some(from) = session.try_recv() {
            match from.msg {
                PeerWire::Beat(beat) => {
                    // A direct beat from a peer (never ourselves) — we are not alone, now
                    // or retroactively. Latch it so a peer that later goes silent fails
                    // loud instead of falling back to solo.
                    if from.from != me {
                        ever_heard_peer = true;
                    }
                    m.on_beat(from.from, &beat, now);
                }
                PeerWire::Tick(msg) => early.push((from.from, msg)),
                // No server exists yet during formation, so a snapshot / admission /
                // refusal can't legitimately arrive here; ignore a stray one (a peer racing ahead,
                // or a mid-game join frame on the wrong phase) rather than mishandle it. A real
                // mid-game join is handled by the running coordinator, not the formation barrier.
                PeerWire::Snapshot(_)
                | PeerWire::Articulation(_)
                | PeerWire::JoinRequest(_)
                | PeerWire::Refuse(_)
                | PeerWire::Welcome(_) => {}
            }
        }

        // Poll FIRST (expires stale peers + advances the verdict), so the beat we then
        // advertise reflects the freshly-pruned set — we never gossip a just-expired
        // phantom for an extra round.
        let status = m.poll(now);
        session.broadcast(&m.beat()).await;

        // Push the live roster to the lobby UI when it changes. Best-effort — a
        // closed/full channel just drops it. The set is the membership's own `live_set`,
        // so the lobby shows exactly what would freeze.
        if let Some(c) = lobby {
            let roster = m.live_set();
            if roster != last_roster {
                let _ = c.roster_tx.send(roster.clone());
                last_roster = roster;
            }
        }

        match status {
            Status::Agreed { roster } => {
                // Sample the weights verdict at the close instant — `poll` (above) just expired
                // the dead, so the live set this reflects is exactly the frozen `roster`.
                return Ok(BarrierResult::Agreed(BarrierOutcome {
                    roster,
                    early,
                    elapsed: now.duration_since(start),
                    weights_synced: m.weights_synced(),
                    assets_synced: m.assets_synced(),
                }));
            }
            Status::Failed => {
                // The JOIN_WINDOW elapsed without a closed roster. On the DEFAULT
                // (non-lobby) path this is the last-resort alone-fallback, not always an
                // error: if we are alone right now ([`is_alone_at_timeout`], `live` fresh
                // from the `poll` above) play solo rather than strand the player; `live >=
                // 2` stays the loud failure. The lobby path has no JOIN_WINDOW, so this is
                // the scripted/headless timer barrier only.
                if lobby.is_none() && is_alone_at_timeout(expect, m.live_set().len()) {
                    return Ok(BarrierResult::Alone);
                }
                anyhow::bail!(
                    "match formation failed: peers never agreed on one roster within the join window \
                     (too few players showed up, or a peer kept appearing/disappearing, or a link is \
                     one-way). Relaunch together."
                );
            }
            Status::Forming { live } => {
                // Solo fallback — DEFAULT (non-lobby) path only; a lone host never reaches
                // this barrier (the UI's instant-solo path handles it). `live` is fresh
                // from the `poll` above, so any peer that ever beat us makes `live >= 2`
                // AND latches `ever_heard_peer` — see [`is_alone_now`] for the full guard.
                if lobby.is_none()
                    && is_alone_now(expect, live, ever_heard_peer, now >= alone_deadline)
                {
                    return Ok(BarrierResult::Alone);
                }
                if live != last_live {
                    println!("forming: {live}/{expect} player(s) live, waiting for agreement…");
                    last_live = live;
                    if let Some(t) = telemetry {
                        t.send(TelemetryEvent::RosterForming { live, expect });
                    }
                }
            }
        }
    }
}

/// The solo-fallback policy as a pure predicate, so the (race-sensitive) decision is
/// unit-testable without a socket or the async clock. Returns true iff we should stop
/// awaiting a networked match and play solo. ALL must hold:
/// - `expect > 1` — a defaulted networked launch, not a deliberate `expect == 1`
///   solo-over-network (which forms `{self}` via the barrier and must not be preempted).
/// - `!ever_heard_peer` — we have NEVER heard a beat from any peer this formation. A peer
///   heard then lost is a link FAILURE (the loud `Failed`/relaunch), not the alone case —
///   the guard that stops a one-way link silently splitting into single-player.
/// - `live == 1` — only us is live right now (a fresh post-`poll` count; any present peer
///   makes this `>= 2` and keeps the full membership barrier in force).
/// - `past_deadline` — the discovery window has elapsed (the caller passes the comparison
///   result, not two same-typed `Instant`s, removing a silently-invertible transposition).
fn is_alone_now(expect: usize, live: usize, ever_heard_peer: bool, past_deadline: bool) -> bool {
    expect > 1 && !ever_heard_peer && live == 1 && past_deadline
}

/// The LAST-RESORT alone-fallback, applied when the barrier already reached
/// [`crate::membership::Status::Failed`] (the JOIN_WINDOW elapsed without a closed
/// roster) on the default (non-lobby) path. Returns true iff we should solo rather than
/// error. Unlike [`is_alone_now`] it does NOT consult `ever_heard_peer`: at the window's
/// END a once-heard peer that has since expired leaves us at `live == 1`, and stranding a
/// lone player with an error helps no one — so a phantom flicker solos here (just later
/// than a never-heard launch). No deadline either: reaching `Failed` already means the
/// window expired. The `live >= 2` case is excluded so it stays the loud `Failed` — real
/// peers present that never agreed is a genuine multi-peer fault, not the alone case.
fn is_alone_at_timeout(expect: usize, live: usize) -> bool {
    expect > 1 && live == 1
}

/// Build the single-peer lockstep for an OFFLINE round (just us). The one definition shared
/// by every offline entry into the windowed client — the boot-menu Host-alone Start, the
/// scripted `--host`/`--join` alone outcome, and the discovery-found-no-peer fallback — so
/// they all play the byte-identical deterministic solo round with no second construction to
/// drift. `seed` is the shared match seed (the caller passes the one constant every peer
/// uses). The player always starts ON FOOT; piloting a plane is reached in-game via the
/// client's enter/exit toggle ([`crate::render`]), not an env flag at spawn.
pub fn solo_lockstep_for(seed: u64) -> Lockstep {
    let me = PlayerId(0);
    Lockstep::new(seed, &[me], me)
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

/// The inputs that arrived during formation, mapped to their author's [`PlayerId`] (senders not in
/// the agreed set dropped). The host seeds its server ledger with these (via [`Server::seed_early`])
/// so a fast client's pre-serve inputs aren't lost; everyone else discards them (only the server
/// holds the ledger). `pub` so the headless `game net` driver builds the same set from its `Frozen`.
pub fn early_peer_msgs(frozen: &Frozen) -> Vec<PeerMsg> {
    frozen
        .early
        .iter()
        .filter_map(|(from, msg)| frozen.id_map.get(from).map(|&pid| PeerMsg { pid, msg: *msg }))
        .collect()
}

/// Map endpoint ids → [`PlayerId`]s by sorting the full agreed set (us + peers). Every
/// peer sorts the identical set, so a given endpoint is the same `PlayerId`
/// everywhere — the precondition lockstep needs to apply inputs in an agreed order.
/// Errors past [`PlayerId`]'s `u8` range rather than wrapping two endpoints onto one
/// id (this game is couch-scale, never close). Called from [`form_match`], so both
/// entrypoints assign ids identically.
pub fn assign_player_ids(
    me: EndpointId,
    roster: &[EndpointId],
) -> Result<BTreeMap<EndpointId, PlayerId>> {
    let mut all: Vec<EndpointId> = roster.to_vec();
    all.push(me);
    all.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    all.dedup();
    anyhow::ensure!(
        all.len() <= u8::MAX as usize + 1,
        "too many players: {}",
        all.len()
    );
    Ok(all
        .into_iter()
        .enumerate()
        .map(|(i, eid)| (eid, PlayerId(i as u8)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic distinct endpoint ids: derived from a secret-key seed because a raw
    /// byte pattern isn't a valid public key (see the same helper in `membership`).
    fn eid(i: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[i; 32]).public()
    }

    #[test]
    fn assign_player_ids_is_identical_regardless_of_roster_order() {
        // The determinism crux at the net_loop layer: two peers handed the SAME agreed
        // set in DIFFERENT order must produce the identical endpoint→PlayerId map (the
        // barrier guarantees the same set; this guarantees the same assignment over it).
        let me = eid(2);
        let a = assign_player_ids(me, &[eid(1), eid(3), eid(2)]).unwrap();
        let b = assign_player_ids(me, &[eid(3), eid(2), eid(1)]).unwrap();
        assert_eq!(a, b, "id assignment must not depend on input order");
        // PlayerIds are 0..n in endpoint-id BYTE order (not seed order — real public
        // keys sort arbitrarily). Verify the assignment IS that sorted order.
        let mut ids = [eid(1), eid(2), eid(3)];
        ids.sort_by(|x, y| x.as_bytes().cmp(y.as_bytes()));
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(a[id], PlayerId(i as u8), "id at sort position {i}");
        }
    }

    #[test]
    fn assign_player_ids_dedups_self_in_roster() {
        // The roster from the barrier already includes us; assign must not double-count
        // when `me` appears in it (it pushes `me` then dedups).
        let me = eid(5);
        let map = assign_player_ids(me, &[eid(5), eid(7)]).unwrap();
        assert_eq!(map.len(), 2, "self must not be double-counted");
        // The two distinct ids get the two distinct PlayerIds 0 and 1 (in byte order).
        let mut got: Vec<PlayerId> = vec![map[&eid(5)], map[&eid(7)]];
        got.sort();
        assert_eq!(got, vec![PlayerId(0), PlayerId(1)]);
    }

    /// Live multi-endpoint formation over REAL iroh QUIC (not the pure state machine):
    /// three sessions on the loopback, fully meshed by direct dial, run the actual
    /// [`form_match`] barrier concurrently and MUST all freeze the byte-identical
    /// participant→PlayerId map. The end-to-end proof of the determinism-critical guarantee
    /// that every peer agrees on the same roster — exercised through the genuine wire (beats
    /// encoded/decoded, heartbeats, the stability barrier), not just the unit-tested core.
    ///
    /// A THIRD peer is started a beat late (a staggered join) to prove the barrier waits
    /// it in rather than freezing {A,B} first. `#[ignore]` because it binds real UDP
    /// sockets and takes a couple seconds (the STABLE_FOR settle); run with
    /// `cargo test --lib net::net_loop -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "binds real iroh UDP endpoints; run explicitly with --ignored"]
    async fn three_endpoints_form_the_identical_match_over_iroh() {
        use std::collections::BTreeMap;

        // Start three real sessions.
        let mut s0 = transport::start_session().await.expect("start s0");
        let mut s1 = transport::start_session().await.expect("start s1");
        let mut s2 = transport::start_session().await.expect("start s2");
        let (a0, a1) = (s0.local_addr(), s1.local_addr());
        let (e0, e1, e2) = (s0.endpoint_id(), s1.endpoint_id(), s2.endpoint_id());

        // Mesh them by DIRECT dial (bypass mDNS, which is flaky/slow in CI). Each pair
        // needs exactly one dial; wire_connection orients the stream by the id tie-break
        // internally, so dialing either direction yields one correct link. We dial A→B
        // for each unordered pair. s2 joins LATE (after the others have started beating)
        // to exercise the staggered-join path.
        s0.connect_direct(a1.clone()).await.expect("s0->s1");

        // Run all three barriers concurrently. s2's dials are issued from inside its
        // future after a short delay, so it shows up mid-formation. `None` lobby — this
        // exercises the unchanged timer-closed barrier.
        let f0 = form_match(&mut s0, 1, 3, None, None, 0, 0);
        let f1 = form_match(&mut s1, 1, 3, None, None, 0, 0);
        let f2 = async {
            // Stagger: let s0/s1 form their partial view first, then s2 meshes in.
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
            s2.connect_direct(a0.clone()).await.expect("s2->s0");
            s2.connect_direct(a1.clone()).await.expect("s2->s1");
            form_match(&mut s2, 1, 3, None, None, 0, 0).await
        };
        let (r0, r1, r2) = tokio::join!(f0, f1, f2);
        // Each peer must AGREE (not fall back to solo — they all see each other), so unwrap
        // the `Formation::Agreed`; an `Alone` here would be a barrier bug (peers present).
        let unwrap_agreed = |r: Result<Formation>, who: &str| match r.expect(who) {
            Formation::Agreed(f) => f,
            Formation::Alone => panic!("{who}: fell back to solo despite peers being present"),
            Formation::Cancelled => panic!("{who}: cancelled despite no lobby control"),
        };
        let (r0, r1, r2) = (
            unwrap_agreed(r0, "s0 forms"),
            unwrap_agreed(r1, "s1 forms"),
            unwrap_agreed(r2, "s2 forms"),
        );

        // All three froze the SAME endpoint→PlayerId map — the whole point.
        assert_eq!(r0.id_map, r1.id_map, "s0 and s1 must agree on the roster");
        assert_eq!(r1.id_map, r2.id_map, "s1 and s2 must agree on the roster");
        assert_eq!(r0.id_map.len(), 3, "all three endpoints in the match");

        // And the map is exactly the three ids sorted → PlayerId 0,1,2 (the canonical
        // assignment every peer computes), so each peer also knows itself correctly.
        let mut ids = [e0, e1, e2];
        ids.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let expected: BTreeMap<EndpointId, PlayerId> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, PlayerId(i as u8)))
            .collect();
        assert_eq!(
            r0.id_map, expected,
            "roster must be the sorted-id assignment"
        );
        // Each peer's `me` is its own id's position.
        assert_eq!(r0.me, expected[&e0]);
        assert_eq!(r1.me, expected[&e1]);
        assert_eq!(r2.me, expected[&e2]);

        s0.shutdown().await;
        s1.shutdown().await;
        s2.shutdown().await;
    }

    /// The solo-fallback decision, exhaustively and deterministically (no socket, no real
    /// clock — the policy is the pure [`is_alone_now`] predicate). Each of the four
    /// conditions that must ALL hold is falsified in isolation to prove it's load-bearing.
    /// The cross-peer timing the predicate can't see (two real endpoints discovering each
    /// other) is left to the live `#[ignore]`d formation test; here we pin the policy.
    #[test]
    fn alone_fallback_fires_only_when_defaulted_networked_never_heard_and_truly_alone() {
        // (expect, live, ever_heard_peer, past_deadline).
        // The fallback case: defaulted networked (expect>1), never heard a peer, alone
        // (live==1), past the deadline.
        assert!(
            is_alone_now(2, 1, false, true),
            "defaulted-networked + never-heard + alone + past the window ⇒ solo"
        );

        // Each guard removed in turn must SUPPRESS the fallback:
        assert!(
            !is_alone_now(2, 1, false, false),
            "before the discovery window we keep waiting, never solo early"
        );
        assert!(
            !is_alone_now(2, 2, false, true),
            "a peer is present (live>=2) ⇒ the real barrier stays in force, never solo"
        );
        assert!(
            !is_alone_now(1, 1, false, true),
            "expect==1 is a deliberate solo-over-network — the barrier forms {{self}}, not a fallback"
        );
        assert!(
            !is_alone_now(2, 1, true, true),
            "heard a peer then lost it ⇒ a link FAILURE (loud Failed/relaunch), not a silent solo"
        );
        // And it stays suppressed for any larger live count.
        assert!(!is_alone_now(4, 3, false, true));
    }

    /// The LAST-RESORT timeout fallback (`is_alone_at_timeout`), applied when the barrier
    /// already hit `Status::Failed` (the JOIN_WINDOW expired). It solos iff we defaulted to
    /// networked (`expect > 1`) and are ALONE at expiry (`live == 1`) — regardless of
    /// `ever_heard_peer`, so a phantom that flickered then expired solos rather than
    /// erroring. A genuine multi-peer failure (`live >= 2`) stays the loud error, and a
    /// deliberate `expect == 1` solo-over-network is unaffected.
    #[test]
    fn timeout_fallback_solos_when_alone_at_window_expiry_else_stays_loud() {
        assert!(
            is_alone_at_timeout(2, 1),
            "defaulted-networked + alone at JOIN_WINDOW expiry ⇒ solo, not error (incl. the \
             phantom-flicker and discover_secs>=JOIN_WINDOW cases)"
        );
        assert!(
            !is_alone_at_timeout(2, 2),
            "peers present at expiry that never agreed ⇒ a real multi-peer fault, stay loud"
        );
        assert!(
            !is_alone_at_timeout(2, 5),
            "any live>=2 at expiry stays the loud Failed, never a silent solo"
        );
        assert!(
            !is_alone_at_timeout(1, 1),
            "expect==1 is a deliberate solo-over-network — the barrier forms {{self}} and \
             never reaches a JOIN_WINDOW Failed for this to catch"
        );
    }

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
            assert!(exch.snapshots.is_empty(), "the solo/host arm returns state empty");
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
