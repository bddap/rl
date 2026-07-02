//! The match server: the AUTHORITATIVE per-tick stepper + per-client input streams + roster
//! coordinator.
//!
//! Minecraft-style separation: server code is separate from the client even in one process,
//! and every client (the server's own local one included) dials a server. The server is
//! AUTHORITATIVE over world state: it owns the integer [`Sim`] and steps it once per tick,
//! emitting a [`CoreSnapshot`](crate::snapshot::CoreSnapshot) the local client APPLIES rather
//! than stepping a sim of its own. "Solo" is the same path with a roster of one — an internal
//! server + the single local client — so there is no separate single-player code path; SP
//! always serializes the hand-off too (build → `to_bytes` → `from_bytes` → apply), no
//! by-reference shortcut.
//!
//! **The HOST paces the match; a remote input can DELAY nothing** (rl#193/#194/#195). Each
//! authoritative tick is assembled by [`Server::advance`] the moment the host's own input for
//! it exists: the host's input applies at that tick, and every remote player contributes the
//! next queued input from its per-player [`InputStream`] — or, when the stream is starved by
//! transit lag, a HOLD of its last consumed move axes ([`Input::hold`]). Gating a tick on
//! every rostered client's input would instead put the host's whole world — its own avatar,
//! the crab pump, the piloted ship — at remote-input latency, applying ticks in RTT-jittered
//! bursts against the render's wall-clock interpolation (the MP wiggle/stutter), and make a
//! silent remote a match-wide freeze; under host pacing both failure classes are
//! unrepresentable — there is nothing to wait on.
//!
//! Inputs flow UP; the authoritative [`CoreSnapshot`](crate::snapshot::CoreSnapshot) flows
//! DOWN to every client, which ADOPTS it whole (no re-sim, no peer cross-check — the host is
//! the source of truth). Each snapshot carries `input_next` — per player, the first
//! [`TickMsg::issue_tick`] NOT yet consumed — so a remote client replays exactly its
//! still-unconsumed inputs when predicting its own avatar
//! ([`crate::lockstep::Lockstep::reconcile_local_prediction`]).
//!
//! The stream/roster core is pure and transport-agnostic: [`crate::transport`] /
//! [`crate::net_loop`] move the bytes (loopback for the co-located client, QUIC for a remote
//! one). The one impurity is the owned [`Sim`] the authoritative step advances.

use std::collections::{BTreeMap, VecDeque};

use crate::lockstep::TickMsg;
use crate::roster::RosterSchedule;
use crate::sim::{Input, PlayerId, Pos, Sim};

/// Ticks of lead between admitting a joiner and its roster change taking effect — headroom for
/// the welcome to reach the joiner so it can start issuing input near its entry tick (its
/// stream just holds neutral until the first input arrives; nothing stalls on it). Also keeps
/// the roster change strictly ahead of the assemble cursor.
pub const JOIN_LEAD: u64 = 3;

/// A remote player's queued (recorded, not yet consumed) inputs are FOLDED down to this depth
/// at record time, bounding that player's standing input latency at ~2 ticks (66 ms). Depth
/// only exceeds it transiently — a burst after a network stall, or clock drift between the
/// host's and the client's 30 Hz pacers — and consumption alone can never drain it (both sides
/// run at 1/tick), so without the fold a one-off stall would become that player's PERMANENT
/// added latency. Folded entries keep their edges: button taps OR and look deltas SUM into the
/// entry that survives ahead of them, so a tap or a turn is never lost — only move axes skip
/// ahead. Folding also bounds queue memory at the record seam.
const TARGET_BACKLOG: usize = 2;

/// The outcome of [`Server::admit`]: the stable [`PlayerId`] allocated to the joiner, the tick its
/// roster change takes effect, and the complete new roster from that tick. The caller UNICASTS
/// this to the joiner alone, which builds its session via [`crate::lockstep::Lockstep::join_at`];
/// incumbents learn the new roster from the next [`crate::snapshot::CoreSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admission {
    pub pid: PlayerId,
    pub effective_tick: u64,
    pub roster: Vec<PlayerId>,
}

/// A would-be joiner's credentials, sent UP to the host the moment it dials a live match. The
/// host gates admission on these BEFORE allocating a [`PlayerId`] and REFUSES LOUDLY, never
/// silently dropping a joiner onto a mismatched body ([`may_admit_joiner`]). Only the ASSET
/// digest travels: the joiner renders the crab it builds from its own model, so a different
/// sally.glb is a visibly wrong Sally. The joiner's WEIGHTS are deliberately absent — a joiner
/// never executes the brain (it adopts host snapshots and renders host articulation), so the
/// one weights guard is the host-side [`AdmissionRefusal::HostNotArmed`] self-gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinRequest {
    /// Digest of the joiner's crab/collider assets — must equal the host's, or the joiner builds
    /// and renders a different-shaped crab.
    pub asset_digest: u64,
}

/// Why a [`JoinRequest`] was refused — a loud, typed verdict the host sends back to the joiner and
/// logs to telemetry (never a silent drop). Carries the offending side so the refusal message is
/// actionable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionRefusal {
    /// The HOST is not running the real Sally — its own policy-weights digest is `0` (a failed or
    /// absent checkpoint drives the zero-action rest pose, not a trained crab). The host serves
    /// the crab POSE to every client, so a zero-digest host would feed a FAKE Sally to whoever
    /// joins — and two both-missing peers (`0 == 0`) would otherwise slip past the equality check
    /// and admit each other into a fake-crab match. Refused before the equality checks so a
    /// zero-digest host reports THIS, not a spurious mismatch. A unit variant — the digest is `0`
    /// by definition of the branch.
    HostNotArmed,
    /// The joiner's crab-asset/collider digest differs from the host's.
    AssetsMismatch { host: u64, joiner: u64 },
}

impl std::fmt::Display for AdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmissionRefusal::HostNotArmed => write!(
                f,
                "host is not running the real trained Sally (its weights digest is 0 — a \
                 failed/absent checkpoint); refusing to admit a joiner into a fake-crab match"
            ),
            AdmissionRefusal::AssetsMismatch { host, joiner } => write!(
                f,
                "crab-asset digest mismatch (host {host:#018x}, joiner {joiner:#018x}) — different Sally/colliders"
            ),
        }
    }
}

/// The admission gate: may a joiner advertising `req` enter a match the host runs on
/// `(host_weights, host_assets)`? `Ok(())` only when the HOST itself runs the real Sally
/// (`host_weights != 0`) AND the asset digests match exactly. The host self-gate is checked
/// FIRST (see [`AdmissionRefusal::HostNotArmed`]); the joiner's brain is deliberately ungated —
/// a joiner never executes it. The host→joiner analogue of the formation-time
/// `may_arm_external_crab` shared-asset gate, but per-joiner and fail-LOUD rather than a silent
/// disarm.
pub fn may_admit_joiner(
    host_weights: u64,
    host_assets: u64,
    req: &JoinRequest,
) -> Result<(), AdmissionRefusal> {
    if host_weights == 0 {
        return Err(AdmissionRefusal::HostNotArmed);
    }
    if req.asset_digest != host_assets {
        return Err(AdmissionRefusal::AssetsMismatch {
            host: host_assets,
            joiner: req.asset_digest,
        });
    }
    Ok(())
}

/// One remote player's ordered input queue + hold state. The transport's reliable per-peer
/// stream delivers [`TickMsg`]s in issue order, so the queue IS the issue sequence; each
/// assembled tick consumes the front ([`InputStream::consume`]) — exactly once, in order — or
/// holds when starved.
struct InputStream {
    queue: VecDeque<TickMsg>,
    /// The last input actually consumed — the HOLD source while the queue is starved. Held as
    /// [`Input::hold`] (move axes only): `look_yaw` is a per-tick DELTA (re-applying it would
    /// spin the avatar) and a re-fired button tap would double a grab/restart.
    held: Input,
    /// The first [`TickMsg::issue_tick`] NOT yet consumed — this player's
    /// [`CoreSnapshot::input_next`](crate::snapshot::CoreSnapshot::input_next) watermark, which
    /// its own client prunes + replays its prediction window against. `0` = nothing consumed
    /// yet (omitted from the snapshot; the client treats the two identically).
    input_next: u64,
}

impl InputStream {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            held: Input::default(),
            input_next: 0,
        }
    }

    /// File one input, then fold the queue down to [`TARGET_BACKLOG`] (see the constant). A
    /// stale or duplicate `issue_tick` (≤ anything already queued or consumed) is dropped —
    /// the live stream can re-deliver what [`Server::seed_early`] already queued.
    fn record(&mut self, msg: TickMsg) {
        let next_fresh = self
            .queue
            .back()
            .map_or(self.input_next, |m| m.issue_tick + 1);
        if msg.issue_tick < next_fresh {
            return;
        }
        self.queue.push_back(msg);
        while self.queue.len() > TARGET_BACKLOG {
            let dropped = self.queue.pop_front().expect("len > target");
            let next = &mut self
                .queue
                .front_mut()
                .expect("len > target ⇒ still nonempty")
                .input;
            next.buttons |= dropped.input.buttons;
            next.look_yaw = next.look_yaw.saturating_add(dropped.input.look_yaw);
        }
    }

    /// The input this player contributes to the tick being assembled: the next queued input
    /// (exactly once, in issue order), or a hold of the last consumed move axes when starved.
    fn consume(&mut self) -> Input {
        match self.queue.pop_front() {
            Some(m) => {
                self.held = m.input;
                self.input_next = m.issue_tick + 1;
                m.input
            }
            None => self.held.hold(),
        }
    }
}

/// One assembled-but-not-yet-stepped authoritative tick — [`Server::advance`]'s product,
/// [`Server::step_next`]'s input. Carries its OWN `input_next` watermark map (captured at
/// assemble), so the snapshot the step emits reports exactly the consumption that produced it —
/// stamping from live server state instead would attribute a later assemble's consumption to an
/// earlier tick's snapshot.
struct PendingTick {
    tick: u64,
    inputs: BTreeMap<PlayerId, Input>,
    /// Per rostered player, the first issue tick NOT consumed into this tick (players with
    /// nothing consumed yet are omitted — the client treats absent and `0` identically).
    input_next: BTreeMap<PlayerId, u64>,
}

/// The authoritative match server: per-remote-player [`InputStream`]s + the roster schedule +
/// the owned [`Sim`] its host-paced step advances. [`Server::advance`] assembles one tick from
/// the host's own input the moment that input exists; [`Server::step_next`] applies it.
pub struct Server {
    /// The player whose client co-locates with this server — the HOST (solo: the only player).
    /// Its input arrives per [`Server::advance`] call and paces the match; it never streams, and
    /// [`Server::record_remote`] rejects it, so a host input stream is unrepresentable.
    me: PlayerId,
    /// The participant set over time (sorted, deduped per change-point) — which players
    /// contribute to a tick's assembled input set, and the join/departure boundaries
    /// [`Server::step_next`] mirrors into the sim.
    roster: RosterSchedule,
    /// Per-REMOTE-player input streams. Created on a player's first recorded input, dropped on
    /// its departure — so stream presence implies currently rostered.
    streams: BTreeMap<PlayerId, InputStream>,
    /// The AUTHORITATIVE world this server owns and steps. Its per-tick
    /// [`CoreSnapshot`](crate::snapshot::CoreSnapshot) is what every client renders. Stepped one
    /// tick at a time by [`Server::step_next`], gated by [`Server::next_tick_ready`].
    sim: Sim,
    /// The assembled tick awaiting the authoritative step, if any. At most ONE by construction
    /// ([`Server::advance`] refuses to assemble past an unstepped tick), which is what lets each
    /// [`PendingTick`] carry the watermark map for exactly its own snapshot. Held (rather than
    /// stepping inside `advance`) so the driver can pump the tick's crab physics between the
    /// two: the rapier pump needs the bevy `World`, which this pure core can't hold. The
    /// headless `game net` host (no bevy world, no crab pump) assembles and steps in the same
    /// pass.
    pending: Option<PendingTick>,
}

/// The product of one authoritative step ([`Server::step_next`]): the tick's
/// [`CoreSnapshot`](crate::snapshot::CoreSnapshot) already serialized to wire bytes, plus
/// whether the tick was a RESTART edge. The edge rides the return value (an event, not
/// state) because the tick counter can no longer signal it: a restart is a state-reset at
/// the current tick, never a tick rewind (rl#204) — the driver hangs its per-round resets
/// (physics cadence, crab-body respawn) off this exact edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SteppedTick {
    /// The tick's authoritative snapshot, serialized — the always-serialized hand-off the
    /// local client decodes and applies (no by-reference shortcut even in SP).
    pub snapshot: Vec<u8>,
    /// Whether this tick was a RESTART edge (the sim rebuilt to spawn state at this tick).
    pub restarted: bool,
}

/// The freshly-pumped NN crab pose the authoritative server injects before stepping a tick.
/// The driver owns the bevy `World`, so it pumps the rapier crab body and hands the resulting
/// pose here; [`Server::step_next`] injects it via [`Sim::set_external_crab_pose`] at the one
/// snapshot construction site, so the integer crab pose and the (later) render articulation
/// can't drift. `None` to [`step_next`] leaves the crab untouched (an unarmed round behind the
/// menu, where no body steps).
#[derive(Debug, Clone, Copy)]
pub struct CrabPose {
    pub pos: Pos,
    pub yaw: i32,
    pub digest: u64,
}

impl Server {
    /// Start a server for `roster` (the frozen participant set, us + any remote clients) over the
    /// authoritative `sim` it will step. `me` is the co-located host player, which must be
    /// rostered; for solo `roster` is just `[me]`, for a hosted match the whole agreed roster.
    /// `sim` is the tick-0 world (the caller hands a clone of the client's freshly-built sim, so
    /// the two start byte-identical and the snapshots keep the client in sync).
    pub fn new(me: PlayerId, roster: &[PlayerId], sim: Sim) -> Self {
        debug_assert!(
            roster.contains(&me),
            "the host player must be in the roster it serves"
        );
        Self {
            me,
            roster: RosterSchedule::frozen(roster),
            streams: BTreeMap::new(),
            sim,
            pending: None,
        }
    }

    /// The next tick [`Server::advance`] will assemble — derived, not stored: the sim's tick
    /// plus the pending (assembled, unstepped) one, so it can't drift from either. Strictly
    /// monotone (a RESTART is a state-reset at the current tick, never a rewind — rl#204).
    fn next_assemble(&self) -> u64 {
        self.sim.tick() + u64::from(self.pending.is_some())
    }

    /// Read-only view of the authoritative world (for the driver's hunt-target read + restart-edge
    /// detection, and tests). The client renders from the SNAPSHOT this sim emits, never this sim
    /// directly.
    pub fn sim(&self) -> &Sim {
        &self.sim
    }

    /// Whether the authoritative sim's next tick can be stepped NOW: [`Server::advance`] has
    /// assembled it. Host-paced, so this is `false` only before the host's own input for the
    /// tick has been filed — never because a remote is late.
    pub fn next_tick_ready(&self) -> bool {
        self.pending.is_some()
    }

    /// File one remote client's input into its per-player stream. `from` MUST be the
    /// authenticated sender (the transport binds it to the QUIC peer id), never read from a
    /// body — otherwise a client could file input as someone else. An input from a player
    /// outside the current roster schedule is dropped; a just-admitted joiner's early inputs
    /// queue here and are consumed in order from its effective tick. The host's own player
    /// never streams (its input rides [`Server::advance`]); the authenticated transport can't
    /// produce one, so it is rejected as a caller bug.
    pub fn record_remote(&mut self, from: PlayerId, msg: TickMsg) {
        debug_assert_ne!(from, self.me, "the host's own input never streams");
        if from == self.me || !self.roster.current().contains(&from) {
            return;
        }
        self.streams
            .entry(from)
            .or_insert_with(InputStream::new)
            .record(msg);
    }

    /// Assemble the next authoritative tick, NOW — host-paced: `local` is the host's own input
    /// (issued exactly once per wall-clock tick, which is what paces the match), and every other
    /// player rostered at this tick contributes its stream's next queued input or a starved
    /// HOLD ([`InputStream::consume`]). Nothing waits on a remote. The assembled tick is held
    /// for [`Server::step_next`]; call it after the tick's crab physics is pumped — and before
    /// the next `advance`: assembling past an unstepped tick would mis-stamp its watermarks, so
    /// it is refused loudly.
    pub fn advance(&mut self, local: TickMsg) {
        assert!(
            self.pending.is_none(),
            "advance called again before the assembled tick was stepped"
        );
        let tick = self.next_assemble();
        debug_assert_eq!(
            local.issue_tick, tick,
            "the host issues exactly one input per assembled tick"
        );
        let mut inputs = BTreeMap::new();
        let mut input_next = BTreeMap::new();
        for &pid in self.roster.at(tick) {
            let input = if pid == self.me {
                input_next.insert(pid, local.issue_tick + 1);
                local.input
            } else if let Some(s) = self.streams.get_mut(&pid) {
                let input = s.consume();
                if s.input_next > 0 {
                    input_next.insert(pid, s.input_next);
                }
                input
            } else {
                // No stream: this player never sent an input, or it DEPARTED with its roster
                // removal still pending ([`Server::depart`] drops the stream immediately but the
                // shrink can land after an already-scheduled change). Play neutral, and create
                // no stream/watermark state — resurrecting a departed player's bookkeeping here
                // would leak it into every later snapshot, since `depart` never runs twice.
                Input::default()
            };
            inputs.insert(pid, input);
        }
        self.pending = Some(PendingTick {
            tick,
            inputs,
            input_next,
        });
    }

    /// Advance the authoritative sim by exactly one tick and return the resulting
    /// [`SteppedTick`]: the [`CoreSnapshot`](crate::snapshot::CoreSnapshot) already SERIALIZED to
    /// bytes, plus whether the tick was a RESTART edge. Injects `crab` (the freshly-pumped NN crab
    /// pose; `None` ⇒ crab unchanged) FIRST so this tick's grab/extraction resolve against the real
    /// body, then steps the sim with this tick's assembled input set, then builds the snapshot at
    /// this ONE site — stamping the per-player `input_next` watermarks — and returns its wire
    /// bytes. Returning bytes (not the struct) makes the hand-off ALWAYS serialized — the client
    /// decodes and applies, no by-reference shortcut even in SP. Must be called only when
    /// [`next_tick_ready`](Server::next_tick_ready) is `true`.
    pub fn step_next(&mut self, crab: Option<CrabPose>) -> SteppedTick {
        let tick = self.sim.tick();
        // Mid-game join: a pid rostered at THIS tick but absent from the authoritative sim is a
        // joiner whose admission ([`Server::admit`]) takes effect now — spawn it into the LIVE
        // round so THIS tick's snapshot carries it. Derived from the roster schedule (one source),
        // so there's no separate pending-join table to drift; idempotent past the effective tick
        // via `has_player`.
        for &pid in self.roster.at(tick) {
            if !self.sim.has_player(pid) {
                self.sim.spawn_joining_player(pid);
            }
        }
        // The departure mirror: a sim player NOT rostered at THIS tick has left
        // ([`Server::depart`] shrank the roster when its link died) — remove it from the live world
        // before stepping, so this tick's input set (which no longer carries it) matches the sim's
        // participant set and the snapshot broadcasts the removal to every client.
        let departed: Vec<PlayerId> = self
            .sim
            .players()
            .map(|(pid, _)| pid)
            .filter(|pid| !self.roster.at(tick).contains(pid))
            .collect();
        for pid in departed {
            self.sim.despawn_departed_player(pid);
        }
        let pending = self
            .pending
            .take()
            .expect("step_next called with no assembled tick — guard on next_tick_ready");
        debug_assert_eq!(
            pending.tick, tick,
            "the assembled tick is out of step with the sim tick"
        );
        if let Some(c) = crab {
            self.sim.set_external_crab_pose(c.pos, c.yaw, c.digest);
        }
        let restarted = self.sim.step(&pending.inputs);
        let mut snapshot = self.sim.core_snapshot();
        snapshot.input_next = pending.input_next;
        SteppedTick {
            snapshot: snapshot.to_bytes(),
            restarted,
        }
    }

    /// The current roster (sorted) — the latest scheduled set. Grows on [`Server::admit`], shrinks
    /// on [`Server::depart`].
    pub fn roster(&self) -> &[PlayerId] {
        self.roster.current()
    }

    /// Admit a new client mid-match and return the [`Admission`] for the caller to UNICAST to the
    /// joiner. Allocates the LOWEST [`PlayerId`] not currently in the roster — append-only, so
    /// every existing player KEEPS its id (a positional renumber would desync every peer).
    /// Schedules the new roster to take effect at `effective_tick` ([`JOIN_LEAD`] past the
    /// assemble cursor) so the welcome can reach the joiner around its entry; its stream holds
    /// neutral until its first input arrives, so a slow welcome delays nobody.
    ///
    /// Admission control ([`may_admit_joiner`]) is the CALLER's gate BEFORE this; `admit` is the
    /// bookkeeping once that gate has passed.
    pub fn admit(&mut self) -> Admission {
        let pid = self.lowest_free_pid();
        // Past the assemble cursor by JOIN_LEAD, but always strictly after any change already
        // scheduled (two joins admitted before a tick assembles would otherwise collide on the
        // same tick).
        let effective_tick = self
            .roster
            .earliest_change_at(self.next_assemble() + JOIN_LEAD);
        let mut roster = self.roster.current().to_vec();
        roster.push(pid);
        self.roster.schedule_change(effective_tick, &roster);
        Admission {
            pid,
            effective_tick,
            roster: self.roster.at(effective_tick).to_vec(),
        }
    }

    /// Remove a departed client from the match — the inverse of [`Server::admit`], called by the
    /// host when a rostered peer's link dies: schedule the shrunk roster at the assemble cursor
    /// (or just past any already-scheduled change) and drop the player's input stream. An
    /// already-assembled tick keeps the input captured at assemble; a later tick that still
    /// rosters it (the gap before a later-landing shrink) plays it neutral without recreating
    /// stream state (see [`Server::advance`]); the sim-side
    /// removal happens in [`step_next`](Server::step_next), derived from the roster schedule:
    /// one source, same as the join spawn. Nothing ever waits on a player under host pacing, so
    /// departure is pure bookkeeping — there is no stalled tick to release. A pid not in the
    /// current roster is a no-op (a double report).
    pub fn depart(&mut self, pid: PlayerId) {
        let current = self.roster.current();
        if !current.contains(&pid) {
            return;
        }
        let remaining: Vec<PlayerId> = current.iter().copied().filter(|p| *p != pid).collect();
        let effective_tick = self.roster.earliest_change_at(self.next_assemble());
        self.roster.schedule_change(effective_tick, &remaining);
        self.streams.remove(&pid);
    }

    /// The lowest [`PlayerId`] not in the current roster — the stable allocation `admit` hands a
    /// joiner. Couch-scale, so a free id always exists well inside `u8`.
    fn lowest_free_pid(&self) -> PlayerId {
        let in_use: std::collections::BTreeSet<PlayerId> =
            self.roster.current().iter().copied().collect();
        (0..=u8::MAX)
            .map(PlayerId)
            .find(|p| !in_use.contains(p))
            .expect("couch-scale: a free PlayerId always exists")
    }

    /// Queue inputs that arrived before this server started serving (a fast client sending
    /// during formation) into their senders' streams. Nothing is discarded: the first assembled
    /// ticks consume them in issue order, and the live stream's re-delivery of the same issues
    /// is deduped by the stream itself.
    pub fn seed_early(&mut self, early: &[crate::lockstep::PeerMsg]) {
        for pm in early {
            self.record_remote(pm.pid, pm.msg);
        }
    }

    /// Test-only view of a player's queued (not yet consumed) inputs.
    #[cfg(test)]
    fn queued(&self, pid: PlayerId) -> Vec<TickMsg> {
        self.streams
            .get(&pid)
            .map(|s| s.queue.iter().copied().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lockstep::Lockstep;
    use crate::sim::buttons;
    use crate::snapshot::CoreSnapshot;

    fn ids(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    /// A server over `roster` with a tick-0 authoritative sim on `seed` (the stream/roster tests
    /// don't read the sim; the join/lockstep tests match their clients' seed so the authoritative
    /// world agrees).
    fn srv(seed: u64, roster: &[PlayerId]) -> Server {
        Server::new(PlayerId(0), roster, Sim::new(seed, roster))
    }

    fn input(s: f32) -> Input {
        Input::from_axes(s, 0.0)
    }

    fn tickmsg(issue_tick: u64, s: f32) -> TickMsg {
        TickMsg {
            issue_tick,
            input: input(s),
        }
    }

    /// Step every tick `advance` has assembled and return the decoded snapshots.
    fn step_ready(s: &mut Server) -> Vec<CoreSnapshot> {
        let mut out = Vec::new();
        while s.next_tick_ready() {
            let bytes = s.step_next(None).snapshot;
            out.push(CoreSnapshot::from_bytes(&bytes).expect("snapshot decodes"));
        }
        out
    }

    /// Solo: every host input assembles + steps its tick immediately. The SP=MP-uniformity
    /// core — solo is the server with a roster of one.
    #[test]
    fn solo_advances_one_tick_per_input() {
        let mut s = srv(42, &ids(1));
        for t in 0..5 {
            s.advance(tickmsg(t, 1.0));
            let snaps = step_ready(&mut s);
            assert_eq!(snaps.len(), 1, "each host input steps exactly one tick");
            assert_eq!(snaps[0].tick, t + 1);
            assert_eq!(
                snaps[0].input_next[&PlayerId(0)],
                t + 1,
                "the host's own watermark tracks its issue cursor"
            );
        }
    }

    /// THE rl#193/#194/#195 root, distilled: a completely SILENT remote delays nothing. The
    /// host ticks at its own pace; the silent player just plays neutral holds (it stands still)
    /// until its inputs arrive. Under the old every-input gate this froze the match.
    #[test]
    fn host_never_waits_on_a_remote() {
        let mut s = srv(42, &ids(2));
        let spawn = s.sim().player(PlayerId(1)).expect("rostered").pos();
        for t in 0..10 {
            s.advance(tickmsg(t, 1.0));
            assert!(s.next_tick_ready(), "tick {t} steps with no remote input");
            let _ = s.step_next(None);
        }
        assert_eq!(s.sim().tick(), 10, "the match ran at host pace");
        assert_eq!(
            s.sim().player(PlayerId(1)).expect("rostered").pos(),
            spawn,
            "the silent remote held neutral — it never moved"
        );
    }

    /// Remote inputs are consumed in issue order, one per tick, and the snapshot's
    /// `input_next` watermark tracks the consumption so the remote's own client knows exactly
    /// which inputs to replay. A pre-consume burst past [`TARGET_BACKLOG`] is FOLDED at record
    /// (never re-lived tick by tick — that would be standing latency), so the watermark jumps
    /// over the folded issues and then advances one per tick.
    #[test]
    fn remote_inputs_consume_in_order_one_per_tick() {
        let mut s = srv(42, &ids(2));
        // A steady stream: one arrival per tick, starting one tick early — the queue never
        // exceeds the fold target, so every input is consumed exactly once, in order.
        s.record_remote(PlayerId(1), tickmsg(0, 1.0));
        for t in 0..4u64 {
            s.record_remote(PlayerId(1), tickmsg(t + 1, 1.0));
            s.advance(tickmsg(t, 0.0));
            let snaps = step_ready(&mut s);
            assert_eq!(
                snaps[0].input_next[&PlayerId(1)],
                t + 1,
                "tick {t}: watermark advances one consumed input per tick"
            );
        }
        // A 3-input burst into an idle stream folds to the target depth at record, so the
        // watermark skips the folded issue and then parks once the queue drains.
        let mut s = srv(42, &ids(2));
        for i in 0..3 {
            s.record_remote(PlayerId(1), tickmsg(i, 1.0));
        }
        for (t, expect) in [(0, 2), (1, 3), (2, 3), (3, 3)] {
            s.advance(tickmsg(t, 0.0));
            let snaps = step_ready(&mut s);
            assert_eq!(
                snaps[0].input_next[&PlayerId(1)],
                expect,
                "tick {t}: the burst folds, then consumption parks at the newest issue"
            );
        }
        assert!(s.queued(PlayerId(1)).is_empty(), "the burst fully consumed");
    }

    /// A starved stream HOLDS the last consumed move axes but zeroes the per-tick look delta
    /// (else the avatar spins) and the button taps (else a grab/restart re-fires).
    #[test]
    fn starved_stream_holds_axes_and_strips_edges() {
        let mut stream = InputStream::new();
        stream.record(TickMsg {
            issue_tick: 0,
            input: Input::new(0.5, 1.0, 0.7, buttons::ACTION),
        });
        let consumed = stream.consume();
        assert_eq!(consumed, Input::new(0.5, 1.0, 0.7, buttons::ACTION));
        let held = stream.consume(); // starved
        assert_eq!(
            held,
            Input::new(0.5, 1.0, 0.0, 0),
            "hold keeps move axes, zeroes look delta + buttons"
        );
        assert_eq!(stream.input_next, 1, "a hold consumes nothing");
    }

    /// A held stream keeps MOVING the player: the last consumed move axes stay applied on every
    /// starved tick, so a walking remote doesn't freeze on a network hiccup. (The client's
    /// prediction replay does NOT include these holds — the next adopt corrects the residue.)
    #[test]
    fn hold_keeps_a_moving_remote_moving() {
        let mut s = srv(42, &ids(2));
        s.record_remote(PlayerId(1), tickmsg(0, 1.0)); // walk, then go silent
        let mut positions = vec![s.sim().player(PlayerId(1)).expect("rostered").pos()];
        for t in 0..3 {
            s.advance(tickmsg(t, 0.0));
            let _ = s.step_next(None);
            positions.push(s.sim().player(PlayerId(1)).expect("rostered").pos());
        }
        let step0 = positions[1].x - positions[0].x;
        assert_ne!(step0, 0, "the consumed input moves the player");
        for w in positions.windows(2) {
            assert_eq!(
                w[1].x - w[0].x,
                step0,
                "each starved tick holds the same move axes — no freeze, no drift"
            );
        }
    }

    /// A queue past [`TARGET_BACKLOG`] folds forward at record time: move axes skip ahead, but
    /// the folded entries' button taps (OR) and look deltas (sum) ride the surviving entry — a
    /// tap or a turn is never lost, and a burst can't become standing input latency.
    #[test]
    fn backlog_folds_to_target_keeping_taps_and_look() {
        let mut stream = InputStream::new();
        let n = TARGET_BACKLOG as u64 + 3;
        for i in 0..n {
            let btns = if i == 0 { buttons::RESTART } else { 0 };
            stream.record(TickMsg {
                issue_tick: i,
                input: Input::new(0.0, 0.1, 0.1, btns),
            });
        }
        assert_eq!(
            stream.queue.len(),
            TARGET_BACKLOG,
            "a burst folds down to the target depth at record time"
        );
        let consumed = stream.consume();
        assert!(
            consumed.pressed(buttons::RESTART),
            "the folded oldest input's tap rode forward to the consumed input"
        );
        let folded = n as usize - TARGET_BACKLOG;
        assert_eq!(
            consumed.look_yaw,
            Input::new(0.0, 0.0, 0.1, 0).look_yaw * (folded + 1) as i16,
            "folded look deltas summed into the consumed input"
        );
        assert_eq!(
            stream.input_next,
            n - 1,
            "consumption skipped to the fold survivor's issue"
        );
    }

    /// Stale/duplicate deliveries (the live stream re-sending what `seed_early` queued) are
    /// dropped; fresh issues append.
    #[test]
    fn stale_and_duplicate_remote_inputs_are_dropped() {
        let mut s = srv(42, &ids(2));
        s.record_remote(PlayerId(1), tickmsg(0, 1.0));
        s.record_remote(PlayerId(1), tickmsg(0, -1.0)); // duplicate issue
        s.record_remote(PlayerId(1), tickmsg(1, 0.5));
        assert_eq!(
            s.queued(PlayerId(1))
                .iter()
                .map(|m| m.issue_tick)
                .collect::<Vec<_>>(),
            vec![0, 1],
            "one entry per issue tick"
        );
        assert_eq!(
            s.queued(PlayerId(1))[0].input,
            input(1.0),
            "the FIRST delivery of an issue wins"
        );
    }

    /// An input from a player outside the roster is dropped — it can never be consumed and must
    /// not grow a stream.
    #[test]
    fn non_rostered_input_is_dropped() {
        let mut s = srv(42, &ids(1));
        s.record_remote(PlayerId(9), tickmsg(0, 1.0));
        assert!(s.queued(PlayerId(9)).is_empty(), "no stranger stream");
    }

    /// Departure is pure bookkeeping under host pacing: the roster shrinks, the leaver's stream
    /// drops, the sim despawns it at the boundary, and the match just keeps ticking (there was
    /// never anything to stall). Idempotent, and the freed pid goes to the next joiner.
    #[test]
    fn departure_shrinks_the_roster_and_the_match_ticks_on() {
        let mut s = srv(42, &ids(2));
        // One mutual tick so both players exist in the stepped world.
        s.record_remote(PlayerId(1), tickmsg(0, 1.0));
        s.advance(tickmsg(0, 1.0));
        let _ = step_ready(&mut s);
        s.record_remote(PlayerId(1), tickmsg(1, 1.0)); // queued, never consumed — scrubbed below
        s.depart(PlayerId(1));
        assert_eq!(s.roster(), &[PlayerId(0)], "the roster shrank");
        assert!(s.queued(PlayerId(1)).is_empty(), "the stream dropped");
        s.depart(PlayerId(1)); // double report: no-op
        s.depart(PlayerId(9)); // stranger: no-op
        for t in 1..5 {
            s.advance(tickmsg(t, 1.0));
            let snaps = step_ready(&mut s);
            assert_eq!(snaps.len(), 1, "post-departure ticks flow at host pace");
            assert!(
                !snaps[0].players.contains_key(&PlayerId(1)),
                "the leaver is despawned from the stepped world"
            );
            assert!(
                !snaps[0].input_next.contains_key(&PlayerId(1)),
                "no stray watermark for the leaver"
            );
        }
        assert_eq!(
            s.admit().pid,
            PlayerId(1),
            "the departed id is free for the next joiner"
        );
    }

    /// A departure whose roster shrink lands PAST the assemble cursor (another change is
    /// already scheduled — here a second leaver in the same sweep) still stays departed: the
    /// gap ticks that roster the leaver play it neutral WITHOUT resurrecting its stream or
    /// watermark, which would otherwise leak into every later snapshot forever (`depart` never
    /// runs twice for a pid).
    #[test]
    fn departure_gap_does_not_resurrect_the_leaver() {
        let mut s = srv(42, &ids(3));
        // Both remotes have consumed input (watermarks live), then drop in one sweep — the
        // second shrink is forced past the first, opening a still-rostered gap tick for P2.
        s.record_remote(PlayerId(1), tickmsg(0, 1.0));
        s.record_remote(PlayerId(2), tickmsg(0, 1.0));
        s.advance(tickmsg(0, 0.0));
        let _ = step_ready(&mut s);
        s.depart(PlayerId(1));
        s.depart(PlayerId(2));
        assert_eq!(s.roster(), &[PlayerId(0)], "both shrinks scheduled");
        for t in 1..6 {
            s.advance(tickmsg(t, 0.0));
            let snaps = step_ready(&mut s);
            for pid in [PlayerId(1), PlayerId(2)] {
                assert!(
                    !snaps[0].input_next.contains_key(&pid),
                    "tick {t}: a gap tick must not resurrect the leaver's watermark"
                );
                assert!(s.queued(pid).is_empty(), "tick {t}: no resurrected stream");
            }
        }
        assert!(
            !s.sim().has_player(PlayerId(1)) && !s.sim().has_player(PlayerId(2)),
            "both leavers despawned once their boundaries passed"
        );
    }

    /// `admit` allocates the lowest free [`PlayerId`] and NEVER renumbers an existing one
    /// (a positional renumber on join would desync every peer).
    #[test]
    fn admit_allocates_lowest_free_id_without_renumbering() {
        let mut s = srv(42, &ids(2)); // [P0, P1]
        let a = s.admit();
        assert_eq!(a.pid, PlayerId(2), "the lowest id not already in use");
        assert_eq!(a.roster, vec![PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert!(
            a.effective_tick >= JOIN_LEAD,
            "a join lands at least JOIN_LEAD ahead"
        );
        assert_eq!(
            s.roster(),
            &[PlayerId(0), PlayerId(1), PlayerId(2)],
            "roster grew, ids 0/1 kept"
        );
        // A second admit before any tick assembles still gets a distinct, strictly-later
        // effective tick (the append-only invariant) and the next free id — the incumbents are
        // never renumbered.
        let b = s.admit();
        assert_eq!(b.pid, PlayerId(3));
        assert!(
            b.effective_tick > a.effective_tick,
            "back-to-back joins don't collide"
        );
    }

    /// A joiner's inputs sent between its admission and its effective tick QUEUE (never drop),
    /// and are consumed in issue order from the effective tick on — its watermark starts moving
    /// exactly at entry.
    #[test]
    fn joiner_inputs_queue_until_the_effective_tick() {
        let mut s = srv(42, &ids(1));
        let adm = s.admit();
        // The joiner starts issuing at its effective tick (`Lockstep::join_at`) the moment its
        // welcome lands — these arrive while the host is still assembling pre-join ticks.
        s.record_remote(
            adm.pid,
            TickMsg {
                issue_tick: adm.effective_tick,
                input: input(1.0),
            },
        );
        for t in 0..adm.effective_tick + 2 {
            s.advance(tickmsg(t, 0.0));
            let snaps = step_ready(&mut s);
            let snap = &snaps[0];
            if snap.tick <= adm.effective_tick {
                assert!(
                    !snap.players.contains_key(&adm.pid),
                    "no joiner before its effective tick"
                );
            } else {
                assert!(
                    snap.players.contains_key(&adm.pid),
                    "the joiner is spawned from its effective tick"
                );
            }
            if snap.tick == adm.effective_tick + 1 {
                assert_eq!(
                    snap.input_next[&adm.pid],
                    adm.effective_tick + 1,
                    "the joiner's queued input was consumed at its entry tick"
                );
            }
        }
    }

    /// The SP-IDENTITY invariant. Over a scripted input + crab-pose log, the
    /// host-authoritative path (the server steps its OWN sim and emits a snapshot the client applies)
    /// produces the EXACT same per-tick `state_hash` as stepping a bare [`Sim`] by hand, tick-for-tick
    /// — proving the authoritative step is behavior-preserving. It also checks the always-serialized
    /// hand-off: the client (which only ever calls `apply_core_snapshot`) re-emits byte-identical
    /// snapshots, so it faithfully mirrors the authoritative carried state.
    #[test]
    fn server_authoritative_path_is_sp_identical() {
        const SEED: u64 = 0x00C0FFEE;
        const SUBMITS: u64 = 40;
        let me = PlayerId(0);
        let roster = vec![me];

        // A deterministic, tick-varied crab pose. The rapier body is external to the sim; here a
        // synthetic stand-in is fed IDENTICALLY to both paths, so the comparison isolates the
        // sim-stepping equivalence, independent of the physics engine the live driver actually
        // pumps (which feeds the SAME pose into both arms — see `drive_lockstep`).
        let pose_at = |tick: u64| CrabPose {
            pos: Pos {
                x: 700 + tick as i64 * 11,
                z: -300 - tick as i64 * 7,
            },
            yaw: (tick as i32).wrapping_mul(4096),
            digest: tick.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xABCD,
        };
        // Scripted local input that actually moves the player (forward + a strafe wiggle + an
        // occasional ACTION), so the round evolves and a dropped field would diverge the hash.
        let input_at = |i: u64| {
            let strafe = ((i % 3) as f32 - 1.0) * 0.6;
            let btns = if i % 5 == 4 { buttons::ACTION } else { 0 };
            Input::new(strafe, 1.0, 0.0, btns)
        };

        // --- Path A: the reference. Step a bare [`Sim`] directly, by hand, exactly as the server
        // does internally — the issued input each tick, with the crab pose injected BEFORE each
        // step. This is what `Server::step_next` must reproduce.
        let mut baseline = Vec::new();
        {
            let mut sim = Sim::new(SEED, &roster);
            for i in 0..SUBMITS {
                let p = pose_at(i);
                sim.set_external_crab_pose(p.pos, p.yaw, p.digest);
                sim.step(&BTreeMap::from([(me, input_at(i))]));
                baseline.push((sim.tick(), sim.state_hash()));
            }
        }

        // --- Path B: the server steps its OWN sim and emits a serialized snapshot; a SEPARATE
        // client sim is advanced SOLELY by `apply_core_snapshot`. The client [`Lockstep`] is
        // exactly what the real driver holds (submit_local_input UP, apply snapshot DOWN).
        let mut server_hashes = Vec::new();
        {
            let mut client = Lockstep::new(SEED, &roster, me);
            let mut server = Server::new(me, &roster, Sim::new(SEED, &roster));
            for i in 0..SUBMITS {
                let msg = client.submit_local_input(input_at(i));
                server.advance(msg);
                while server.next_tick_ready() {
                    let bytes = server
                        .step_next(Some(pose_at(server.sim().tick())))
                        .snapshot;
                    let snap =
                        CoreSnapshot::from_bytes(&bytes).expect("the server snapshot decodes");
                    client.apply_core_snapshot(snap);
                    server_hashes.push((server.sim().tick(), server.sim().state_hash()));
                    // The client faithfully mirrors the authoritative CARRIED state. It can't fold
                    // the intentionally-dropped `external_crab_digest`, so its `state_hash` differs by
                    // design — compare the snapshot it RE-EMITS, which must be byte-identical.
                    assert_eq!(
                        client.core_snapshot().to_bytes(),
                        bytes,
                        "the client re-emits the server's exact snapshot (carried state mirrored)"
                    );
                    // The apply cursor tracks the adopted snapshot (the client doesn't step itself),
                    // so next_tick stays honest rather than frozen at 0 while the sim advances.
                    assert_eq!(
                        client.next_tick(),
                        client.sim().tick(),
                        "applying a snapshot advances the client's apply cursor"
                    );
                }
            }
        }

        assert_eq!(
            baseline, server_hashes,
            "host-authoritative server-steps == hand-stepped sim, tick-for-tick (SP identical)"
        );
        assert_eq!(baseline.len() as u64, SUBMITS);
        assert_eq!(baseline[0].0, 1);
    }

    /// The rl#204 wedge, distilled at the authoritative-server seam: a RESTART press used to
    /// rewind `Sim::tick` to 0 while the assemble-tick space stayed monotone and the match
    /// wedged permanently. Now a restart is a state-reset at the CURRENT tick: press R
    /// mid-round and assert the match keeps ticking — every subsequent input still assembles,
    /// steps, and emits a monotone snapshot, and the restart tick's own snapshot carries the
    /// spawn-state world.
    #[test]
    fn restart_does_not_wedge_the_match() {
        // Two rostered players — the MP shape from the issue; solo hits the same mismatch and
        // is covered by every other test's roster-of-one.
        let mut s = srv(42, &ids(2));
        let fresh_players: Vec<_> = Sim::new(42, &ids(2)).players().collect();
        const RESTART_AT: u64 = 10;
        const TOTAL: u64 = 25;
        let mut last_snap_tick = 0;
        for t in 0..TOTAL {
            let btns = if t == RESTART_AT { buttons::RESTART } else { 0 };
            s.record_remote(
                PlayerId(1),
                TickMsg {
                    issue_tick: t,
                    input: Input::new(0.0, 1.0, 0.0, btns),
                },
            );
            s.advance(tickmsg(t, 1.0));
            assert!(s.next_tick_ready(), "tick {t} is steppable — no wedge");
            let stepped = s.step_next(None);
            assert_eq!(
                stepped.restarted,
                t == RESTART_AT,
                "the restart edge fires exactly on the RESTART tick"
            );
            let snap = CoreSnapshot::from_bytes(&stepped.snapshot).expect("snapshot decodes");
            assert_eq!(
                snap.tick,
                t + 1,
                "snapshot ticks stay monotone and gap-free"
            );
            assert!(snap.tick > last_snap_tick || t == 0);
            last_snap_tick = snap.tick;
            if t == RESTART_AT {
                assert_eq!(
                    snap.players.values().map(|p| p.pos()).collect::<Vec<_>>(),
                    fresh_players
                        .iter()
                        .map(|(_, p)| p.pos())
                        .collect::<Vec<_>>(),
                    "the restart tick's snapshot carries the spawn-state world"
                );
            }
        }
        assert_eq!(
            s.sim().tick(),
            TOTAL,
            "the match ran to the end — no freeze"
        );
    }

    /// Mid-game join via snapshot transfer, proven at the authoritative-server seam. A solo host
    /// runs an armed round, stepping its OWN sim and walking the crab far off its spawn; then a
    /// joiner is admitted (`admit`). The moment the sim steps THROUGH the admission's
    /// `effective_tick`, the emitted snapshot must (a) carry the joiner as a fresh `Alive` player,
    /// (b) preserve the LIVE crab pose — NOT reset to spawn — and (c) keep the live, monotonic
    /// tick. That is the whole point: the joiner drops INTO the ongoing match, so a client
    /// adopting the snapshot renders the real Sally at the host's exact pose. A round-boundary
    /// reset would instead zero the tick and respawn the crab at spawn; these assertions catch
    /// that regression.
    #[test]
    fn mid_game_join_transfers_live_state_without_reset() {
        const SEED: u64 = 0x105E_F00D;
        const JOIN_AT: u64 = 12; // ticks before the join, so it genuinely lands mid-round
        const SUBMITS: u64 = 60;
        let host = PlayerId(0);
        let mut server = srv(SEED, &[host]);
        let crab_spawn = server.sim().crab().pos();

        // A crab pose walked steadily away from spawn, so "crab is at its live pose, not reset" is
        // unambiguous. Fed to `step_next` as the authoritative body pose each tick.
        let pose_at = |tick: u64| CrabPose {
            pos: Pos {
                x: 5000 + tick as i64 * 13,
                z: -4000 - tick as i64 * 9,
            },
            yaw: (tick as i32).wrapping_mul(2048),
            digest: tick.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234,
        };
        let input_at = |i: u64| Input::from_axes(((i % 3) as f32 - 1.0) * 0.5, 1.0);

        let mut joiner: Option<(PlayerId, u64, u64)> = None; // (pid, effective_tick, next issue)
        let mut snaps: Vec<CoreSnapshot> = Vec::new();

        for i in 0..SUBMITS {
            // Admit ONE joiner mid-round: the server schedules its roster change at
            // `effective_tick`, and from then its stream is consumed each tick.
            if i == JOIN_AT {
                let adm = server.admit();
                joiner = Some((adm.pid, adm.effective_tick, adm.effective_tick));
            }
            if let Some((jpid, _, next_issue)) = joiner.as_mut() {
                server.record_remote(
                    *jpid,
                    TickMsg {
                        issue_tick: *next_issue,
                        input: input_at(i),
                    },
                );
                *next_issue += 1;
            }
            server.advance(TickMsg {
                issue_tick: i,
                input: input_at(i),
            });
            while server.next_tick_ready() {
                let tick = server.sim().tick();
                let bytes = server.step_next(Some(pose_at(tick))).snapshot;
                snaps.push(CoreSnapshot::from_bytes(&bytes).expect("the server snapshot decodes"));
            }
        }

        let (jpid, eff, _) = joiner.expect("a joiner was admitted");
        // Snapshot ticks are strictly increasing — no reset to 0 anywhere across the join.
        for w in snaps.windows(2) {
            assert!(
                w[1].tick > w[0].tick,
                "the tick is monotonic — the join never resets the round"
            );
        }
        // The pre-join snapshot AT the effective tick (produced by stepping tick eff-1, when the
        // roster is still host-only) does NOT carry the joiner.
        let before = snaps
            .iter()
            .find(|s| s.tick == eff)
            .expect("the sim stepped up to eff");
        assert!(
            !before.players.contains_key(&jpid),
            "no joiner before its effective tick"
        );
        // Stepping tick `eff` spawns the joiner into the LIVE sim, so the next snapshot carries it.
        let after = snaps
            .iter()
            .find(|s| s.tick == eff + 1)
            .expect("the sim stepped through eff");
        assert!(
            after.players.contains_key(&jpid),
            "the joiner appears in the snapshot at its entry"
        );
        assert_eq!(
            after.players[&jpid].status(),
            crate::sim::PlayerStatus::Alive,
            "the joiner spawns Alive"
        );
        assert!(
            after.roster.contains(&jpid),
            "the snapshot roster carries the joiner"
        );
        // The crab is at its LIVE walked pose for that tick — the round kept running, NOT reset to
        // spawn: the joiner adopts an ONGOING crab, no round-boundary rebuild.
        assert_eq!(
            after.crab.pos(),
            pose_at(eff).pos,
            "the crab is at its live pose at tick eff"
        );
        assert_ne!(
            after.crab.pos(),
            crab_spawn,
            "the crab did NOT reset to spawn (no lockstep rebuild)"
        );
        assert!(eff > JOIN_LEAD, "the join genuinely lands mid-round");
    }

    /// The admission gate: an armed host admits an asset-matched joiner regardless of the joiner's
    /// brain (the joiner never executes one) and refuses an asset mismatch loudly, typed —
    /// the host→joiner analogue of the formation shared-asset gate, fail-LOUD per joiner.
    #[test]
    fn admission_gate_admits_asset_match_and_refuses_mismatch() {
        let (hw, ha) = (0xB7A1u64, 0x5A11_2233u64);
        assert_eq!(
            may_admit_joiner(hw, ha, &JoinRequest { asset_digest: ha }),
            Ok(()),
            "an armed host admits a matching-asset joiner"
        );
        assert_eq!(
            may_admit_joiner(
                hw,
                ha,
                &JoinRequest {
                    asset_digest: ha ^ 1
                }
            ),
            Err(AdmissionRefusal::AssetsMismatch {
                host: ha,
                joiner: ha ^ 1
            }),
            "a different Sally/colliders is refused"
        );
    }

    /// The real-Sally host self-gate: a host whose OWN weights digest is 0
    /// (a failed/absent checkpoint — the fake rest-pose crab) refuses every joiner as
    /// [`HostNotArmed`], even an asset-matched one. Under host-auth the host serves the crab pose
    /// to everyone, so this single gate is what keeps a fake-crab match from ever admitting anyone.
    #[test]
    fn zero_digest_host_refuses_every_joiner() {
        let ha = 0x5A11_2233u64;
        assert_eq!(
            may_admit_joiner(0, ha, &JoinRequest { asset_digest: ha }),
            Err(AdmissionRefusal::HostNotArmed),
            "an unarmed host can't admit anyone into its fake-crab match"
        );
        // Self-gate FIRST (the documented order): even an asset-MISMATCHED joiner is reported as
        // HostNotArmed — the host's own failure, never a spurious asset verdict.
        assert_eq!(
            may_admit_joiner(
                0,
                ha,
                &JoinRequest {
                    asset_digest: ha ^ 1
                }
            ),
            Err(AdmissionRefusal::HostNotArmed),
            "the self-gate is checked before the asset equality"
        );
    }
}
