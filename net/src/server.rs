use std::collections::{BTreeMap, VecDeque};

use crate::client::TickMsg;
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

/// Starvation-observability window, in assembled ticks (1 s). [`Server::advance`] counts each
/// remote player's starved fills — a stream hold, or the no-stream neutral of a player that
/// never delivered ONE input (the maximal starvation) — per window, keyed off the ROSTER, and
/// turns a chronic count into a [`StarvationReport`] at each window boundary. A WINDOWED count
/// (not a consecutive-run counter) on purpose: the chronic case rl#213 names — a late client
/// permanently one-tick-held — alternates hold/consume and would never form a long run.
const STARVATION_WINDOW: u64 = crate::sim::TICK_HZ;

/// Starved fills within one [`STARVATION_WINDOW`] at which a player counts as chronically
/// starved (half the window ≈ 0.5 s of holds per second — rl#213's magnitude). Below this,
/// occasional holds are normal jitter absorption and stay quiet.
const STARVATION_REPORT_THRESHOLD: u64 = STARVATION_WINDOW / 2;

/// Minimum ticks between two reports for the SAME player (10 s), so a chronically bad link
/// re-reports at a human cadence instead of once per window — the flood bound rl#213 requires.
const STARVATION_REPORT_COOLDOWN: u64 = 10 * crate::sim::TICK_HZ;

/// At most this many un-drained [`StarvationReport`]s are held on the [`Server`]; beyond it,
/// new ones are dropped WITHOUT arming the player's cooldown, so a capped-out player retries
/// at the next boundary once the driver drains. Only a driver that never drains could hit it —
/// the bound exists so telemetry bookkeeping can never grow without limit.
const STARVATION_REPORT_CAP: usize = 16;

/// One chronic input-starvation observation (rl#213): `pid` filled `starved` of the `window`
/// assembled ticks closing at `tick` with holds/neutral. Produced by [`Server::advance`] at
/// window boundaries (rate-limited per player), drained by the driver via
/// [`Server::take_starvation_reports`] and surfaced as telemetry
/// ([`crate::telemetry::surface_starvation`]). Pure observability — the hold itself already
/// keeps the sim correct and bounded (95d3c7b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StarvationReport {
    pub pid: PlayerId,
    /// Starved fills within the window — ≥ half the window by construction.
    pub starved: u64,
    /// The window length in ticks, so the consumer renders a rate without re-deriving it.
    pub window: u64,
    /// The assemble tick at which the window closed (the window covers the `window` fills
    /// before it).
    pub tick: u64,
}

impl std::fmt::Display for StarvationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "input starvation: player {} held on {}/{} ticks in the window closing at tick {} \
             — its input stream is chronically late (rl#213)",
            self.pid.0, self.starved, self.window, self.tick
        )
    }
}

/// Per-remote-player starvation bookkeeping (rl#213), keyed off the ROSTER — deliberately NOT
/// stored on [`InputStream`], whose lifetime starts at the first delivered input: the
/// never-sent player (no stream at all) is the maximal starvation and must count too. Entries
/// persist across windows so the report cooldown survives a clean window; a departed player's
/// entry is pruned at the window boundary once its roster shrink lands.
#[derive(Default)]
struct StarvationTally {
    /// Starved fills in the current [`STARVATION_WINDOW`]; reset at each boundary.
    starved_in_window: u64,
    /// The boundary tick of this player's last [`StarvationReport`], for the per-player
    /// [`STARVATION_REPORT_COOLDOWN`]. `None` = never reported.
    last_report: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admission {
    pub pid: PlayerId,
    pub effective_tick: u64,
    pub roster: Vec<PlayerId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinRequest {
    pub asset_digest: u64,
    pub crab_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionRefusal {
    AssetsMismatch { host: u64, joiner: u64 },
    CrabCountMismatch { host: u8, joiner: u8 },
}

impl std::fmt::Display for AdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmissionRefusal::AssetsMismatch { host, joiner } => write!(
                f,
                "crab-asset digest mismatch (host {host:#018x}, joiner {joiner:#018x}) — different Sally/colliders"
            ),
            AdmissionRefusal::CrabCountMismatch { host, joiner } => write!(
                f,
                "crab-count mismatch (host serves {host}, joiner renders {joiner}) — the joiner \
                 would show the wrong number of crabs; launch it with the host's binding list"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    Admission(AdmissionRefusal),
    Departed,
    /// The host is mid-formation (pre-round barrier) and not admitting joiners yet — a
    /// correct "busy" diagnosis for the dialer, instead of the silent drop that decayed
    /// into a misattributed "host unreachable" timeout (rl#245).
    Forming,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Admission(r) => r.fmt(f),
            Refusal::Departed => {
                write!(
                    f,
                    "you were dropped from the match (connection lost) — rejoin"
                )
            }
            Refusal::Forming => {
                write!(
                    f,
                    "the host is still forming its match — try again in a few seconds"
                )
            }
        }
    }
}

pub fn may_admit_joiner(
    host_assets: u64,
    host_crabs: u8,
    req: &JoinRequest,
) -> Result<(), AdmissionRefusal> {
    if req.asset_digest != host_assets {
        return Err(AdmissionRefusal::AssetsMismatch {
            host: host_assets,
            joiner: req.asset_digest,
        });
    }
    if req.crab_count != 0 && req.crab_count != host_crabs {
        return Err(AdmissionRefusal::CrabCountMismatch {
            host: host_crabs,
            joiner: req.crab_count,
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

    fn record(&mut self, msg: TickMsg) -> bool {
        let next_fresh = self
            .queue
            .back()
            .map_or(self.input_next, |m| m.issue_tick + 1);
        if msg.issue_tick < next_fresh {
            return false;
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
        true
    }

    /// The input this player contributes to the tick being assembled: the next queued input
    /// (exactly once, in issue order), or a hold of the last consumed move axes when starved.
    /// The returned flag is the ONE source of "this fill was starved" — the caller's rl#213
    /// tally counts it rather than re-deriving queue emptiness.
    fn consume(&mut self) -> (Input, bool) {
        match self.queue.pop_front() {
            Some(m) => {
                self.held = m.input;
                self.input_next = m.issue_tick + 1;
                (m.input, false)
            }
            None => (self.held.hold(), true),
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
    pilot_intents: BTreeMap<PlayerId, crate::client::PilotIntent>,
    /// The AUTHORITATIVE world this server owns and steps. Its per-tick
    /// [`CoreSnapshot`](crate::snapshot::CoreSnapshot) is what every client renders. Stepped one
    /// tick at a time by [`Server::step_next`], gated by [`Server::next_tick_ready`].
    sim: Sim,
    /// Per-remote-player starvation tallies (rl#213), roster-keyed (see [`StarvationTally`]).
    starvation: BTreeMap<PlayerId, StarvationTally>,
    /// Pending chronic-starvation observations (rl#213), pushed by [`Server::advance`] at
    /// window boundaries and drained by [`Server::take_starvation_reports`]. Bounded at
    /// [`STARVATION_REPORT_CAP`].
    starvation_reports: Vec<StarvationReport>,
    /// The assembled tick awaiting the authoritative step, if any. At most ONE by construction
    /// ([`Server::advance`] refuses to assemble past an unstepped tick), which is what lets each
    /// [`PendingTick`] carry the watermark map for exactly its own snapshot. Held (rather than
    /// stepping inside `advance`) so the driver can pump the tick's crab physics between the
    /// two: the rapier pump needs the bevy `World`, which this pure core can't hold. The
    /// headless `game net` host (no bevy world, no crab pump) assembles and steps in the same
    /// pass.
    pending: Option<PendingTick>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SteppedTick {
    pub snapshot: Vec<u8>,
    pub restarted: bool,
}

#[derive(Debug, Clone)]
pub struct CrabPose {
    pub pos: Pos,
    pub yaw: i32,
    /// This crab's claw colliders, bridged into sim space (rl#249) — empty on the paths
    /// with no physics crab (they pass no poses at all).
    pub claws: Vec<crate::sim::ClawPose>,
}

impl Server {
    pub fn new(me: PlayerId, roster: &[PlayerId], sim: Sim) -> Self {
        debug_assert!(
            roster.contains(&me),
            "the host player must be in the roster it serves"
        );
        Self {
            me,
            roster: RosterSchedule::frozen(roster),
            streams: BTreeMap::new(),
            pilot_intents: BTreeMap::new(),
            sim,
            pending: None,
            starvation: BTreeMap::new(),
            starvation_reports: Vec::new(),
        }
    }

    /// The next tick [`Server::advance`] will assemble — derived, not stored: the sim's tick
    /// plus the pending (assembled, unstepped) one, so it can't drift from either. Strictly
    /// monotone (a RESTART is a state-reset at the current tick, never a rewind — rl#204).
    fn next_assemble(&self) -> u64 {
        self.sim.tick() + u64::from(self.pending.is_some())
    }

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
        let accepted = self
            .streams
            .entry(from)
            .or_insert_with(InputStream::new)
            .record(msg);
        if accepted {
            self.file_intent(from, msg.pilot);
        }
    }

    fn file_intent(&mut self, pid: PlayerId, pilot: Option<crate::client::PilotIntent>) {
        match pilot {
            Some(intent) => {
                let may_board = self.pilot_intents.contains_key(&pid)
                    || self.sim.player(pid).is_some_and(|p| p.status().may_board());
                if may_board {
                    self.pilot_intents.insert(pid, intent);
                }
            }
            None => {
                self.pilot_intents.remove(&pid);
            }
        }
    }

    /// Drain the pending chronic-starvation observations (rl#213) for the driver to surface
    /// (telemetry + log). Empty in the healthy case at zero cost; un-drained reports are
    /// bounded at [`STARVATION_REPORT_CAP`], so a driver that never calls this leaks nothing.
    pub fn take_starvation_reports(&mut self) -> Vec<StarvationReport> {
        std::mem::take(&mut self.starvation_reports)
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
        self.file_intent(self.me, local.pilot);
        {
            let rostered = self.roster.at(tick);
            self.pilot_intents.retain(|pid, _| rostered.contains(pid));
        }
        // Starvation-window boundary (rl#213): close the window of the STARVATION_WINDOW fills
        // before `tick` (this runs before this tick's consumes), turning a chronic starved
        // count into a report — per-player cooldown bounds the rate, the cap bounds the
        // memory. Pruning first (level-triggered against the roster, like the sim's departure
        // mirror) retires a departed player's tally without relying on `depart` bookkeeping.
        // At tick 0 the tally map is empty, so the boundary there is a no-op.
        if tick.is_multiple_of(STARVATION_WINDOW) {
            let rostered = self.roster.at(tick);
            self.starvation.retain(|pid, _| rostered.contains(pid));
            for (&pid, tally) in &mut self.starvation {
                if tally.starved_in_window >= STARVATION_REPORT_THRESHOLD
                    && tally
                        .last_report
                        .is_none_or(|t0| tick - t0 >= STARVATION_REPORT_COOLDOWN)
                    && self.starvation_reports.len() < STARVATION_REPORT_CAP
                {
                    tally.last_report = Some(tick);
                    self.starvation_reports.push(StarvationReport {
                        pid,
                        starved: tally.starved_in_window,
                        window: STARVATION_WINDOW,
                        tick,
                    });
                }
                tally.starved_in_window = 0;
            }
        }
        let mut inputs = BTreeMap::new();
        let mut input_next = BTreeMap::new();
        for &pid in self.roster.at(tick) {
            let input = if pid == self.me {
                input_next.insert(pid, local.issue_tick + 1);
                local.input
            } else {
                let (input, starved) = if let Some(s) = self.streams.get_mut(&pid) {
                    let (input, starved) = s.consume();
                    if s.input_next > 0 {
                        input_next.insert(pid, s.input_next);
                    }
                    (input, starved)
                } else {
                    // No stream: this player never sent an input, or it DEPARTED with its roster
                    // removal still pending ([`Server::depart`] drops the stream immediately but
                    // the shrink can land after an already-scheduled change). Play neutral, and
                    // create no stream/watermark state — resurrecting a departed player's
                    // bookkeeping here would leak it into every later snapshot, since `depart`
                    // never runs twice. It still COUNTS as starved: the never-sent uplink is
                    // rl#213's maximal starvation (the tally prunes on the roster, not here).
                    (Input::default(), true)
                };
                if starved {
                    self.starvation.entry(pid).or_default().starved_in_window += 1;
                }
                input
            };
            // A piloting player walks nowhere: re-apply the pilot mask at assembly. The
            // client already masks (`LocalControl::sim_input`), but the server doesn't
            // TRUST it to (rl#191) — and a starved HOLD would otherwise replay the last
            // pre-boarding walk input underneath the craft. `Input::pilot_masked` owns
            // what survives (RESTART, rl#261) and why.
            let input = if self.pilot_intents.contains_key(&pid) {
                input.pilot_masked()
            } else {
                input
            };
            inputs.insert(pid, input);
        }
        self.pending = Some(PendingTick {
            tick,
            inputs,
            input_next,
        });
    }

    /// `pilots` is the driver's per-tick bridge of every spawned craft's pose back into
    /// sim space (rl#258) — filtered here against the filed intents, so only a player
    /// actually piloting can ride (or be down-exempted by) a craft pose.
    pub fn step_next(
        &mut self,
        crabs: &[CrabPose],
        mut pilots: BTreeMap<PlayerId, crate::sim::PilotPose>,
    ) -> SteppedTick {
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
        if !crabs.is_empty() {
            assert_eq!(
                crabs.len(),
                self.sim.crabs().len(),
                "the driver's crab poses must cover every sim crab"
            );
            for (idx, c) in crabs.iter().enumerate() {
                self.sim.set_external_crab_pose(idx, c.pos, c.yaw);
            }
            self.sim
                .set_external_claws(crabs.iter().flat_map(|c| c.claws.iter().copied()).collect());
        }
        pilots.retain(|pid, _| self.pilot_intents.contains_key(pid));
        self.sim.set_external_pilots(pilots);
        let restarted = self.sim.step(&pending.inputs);
        let mut snapshot = self.sim.core_snapshot();
        snapshot.input_next = pending.input_next;
        SteppedTick {
            snapshot: snapshot.to_bytes(),
            restarted,
        }
    }

    pub fn roster(&self) -> &[PlayerId] {
        self.roster.current()
    }

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
        self.pilot_intents.remove(&pid);
    }

    pub fn pilot_intents(&self) -> &BTreeMap<PlayerId, crate::client::PilotIntent> {
        &self.pilot_intents
    }

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
    pub fn seed_early(&mut self, early: &[crate::client::PeerMsg]) {
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
    use crate::client::ClientSim;
    use crate::sim::buttons;
    use crate::snapshot::CoreSnapshot;

    fn ids(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

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
            pilot: None,
        }
    }

    fn intent(kind: crab_world::vehicle::VehicleKind) -> crate::client::PilotIntent {
        crate::client::PilotIntent {
            kind,
            throttle_trim: 0.0,
            thrust: [0.0; 3],
            pitch: 0.0,
            roll: 0.0,
            yaw: 0.0,
            match_velocity: false,
        }
    }

    #[test]
    fn pilot_intents_follow_latest_message_and_roster() {
        use crab_world::vehicle::VehicleKind;
        let p1 = PlayerId(1);
        let mut s = srv(42, &ids(2));

        s.record_remote(
            p1,
            TickMsg {
                pilot: Some(intent(VehicleKind::Plane)),
                ..tickmsg(0, 0.0)
            },
        );
        assert_eq!(s.pilot_intents()[&p1].kind, VehicleKind::Plane);
        s.record_remote(p1, tickmsg(0, 0.0));
        assert_eq!(
            s.pilot_intents()[&p1].kind,
            VehicleKind::Plane,
            "a stale duplicate is rejected by the stream and must not clear the intent"
        );

        s.record_remote(
            p1,
            TickMsg {
                pilot: Some(intent(VehicleKind::Ship)),
                ..tickmsg(1, 0.0)
            },
        );
        assert_eq!(s.pilot_intents()[&p1].kind, VehicleKind::Ship);
        s.record_remote(p1, tickmsg(2, 0.0));
        assert!(
            !s.pilot_intents().contains_key(&p1),
            "pilot: None is on foot — the entry is removed, not zeroed"
        );

        s.advance(TickMsg {
            pilot: Some(intent(VehicleKind::Plane)),
            ..tickmsg(0, 0.0)
        });
        assert_eq!(s.pilot_intents()[&PlayerId(0)].kind, VehicleKind::Plane);
        step_ready(&mut s);

        s.record_remote(
            p1,
            TickMsg {
                pilot: Some(intent(VehicleKind::Ship)),
                ..tickmsg(3, 0.0)
            },
        );
        assert!(s.pilot_intents().contains_key(&p1));
        s.depart(p1);
        assert!(
            !s.pilot_intents().contains_key(&p1),
            "departure drops the craft"
        );
        for t in 1..=4 {
            s.advance(tickmsg(t, 0.0));
            step_ready(&mut s);
        }
        let adm = s.admit();
        assert_eq!(adm.pid, p1, "the departed pid is reused");
        assert!(
            !s.pilot_intents().contains_key(&p1),
            "the joiner reusing the pid starts intent-free"
        );
    }

    #[test]
    fn boarding_is_gated_by_may_board_at_the_intent_insert() {
        use crate::sim::{Player, PlayerStatus};
        use crab_world::vehicle::VehicleKind;
        let p1 = PlayerId(1);
        let mut sim = Sim::new(42, &ids(2));
        let mut snap = sim.core_snapshot();
        let p = snap.players[&p1];
        snap.players.insert(
            p1,
            Player::from_parts(p.pos(), p.yaw(), PlayerStatus::Extracted),
        );
        sim.apply_core_snapshot(snap);
        let mut s = Server::new(PlayerId(0), &ids(2), sim);

        s.record_remote(
            p1,
            TickMsg {
                pilot: Some(intent(VehicleKind::Plane)),
                ..tickmsg(0, 0.0)
            },
        );
        assert!(
            !s.pilot_intents().contains_key(&p1),
            "an extracted player's boarding intent is dropped, not parked"
        );

        s.file_intent(p1, Some(intent(VehicleKind::Plane)));
        assert!(
            !s.pilot_intents().contains_key(&p1),
            "file_intent applies the same gate"
        );
        let mut snap = s.sim().core_snapshot();
        let p = snap.players[&p1];
        snap.players.insert(
            p1,
            Player::from_parts(p.pos(), p.yaw(), PlayerStatus::Downed),
        );
        s.sim.apply_core_snapshot(snap);
        s.record_remote(
            p1,
            TickMsg {
                pilot: Some(intent(VehicleKind::Plane)),
                ..tickmsg(1, 0.0)
            },
        );
        assert!(
            s.pilot_intents().contains_key(&p1),
            "downed ⇒ boards (rl#262, provisional)"
        );
        let mut snap = s.sim().core_snapshot();
        let p = snap.players[&p1];
        snap.players.insert(
            p1,
            Player::from_parts(p.pos(), p.yaw(), PlayerStatus::Extracted),
        );
        s.sim.apply_core_snapshot(snap);
        s.record_remote(
            p1,
            TickMsg {
                pilot: Some(intent(VehicleKind::Ship)),
                ..tickmsg(2, 0.0)
            },
        );
        assert_eq!(
            s.pilot_intents()[&p1].kind,
            VehicleKind::Ship,
            "extraction mid-flight does not eject — the flying pilot's intent still applies"
        );
    }

    /// The assembly-side trust boundary (rl#191): a piloting player's foot input is neutralized
    /// by the SERVER, whatever the client sent — a lying client can't walk and fly at once, and
    /// a starved HOLD can't replay a stale walk underneath the craft.
    #[test]
    fn a_piloting_players_foot_input_is_neutralized_at_assembly() {
        use crab_world::vehicle::VehicleKind;
        let p1 = PlayerId(1);
        let mut s = srv(42, &ids(2));

        // Claims to walk AND pilot in the same message.
        s.record_remote(
            p1,
            TickMsg {
                pilot: Some(intent(VehicleKind::Plane)),
                ..tickmsg(0, 1.0)
            },
        );
        s.advance(TickMsg {
            pilot: Some(intent(VehicleKind::Ship)),
            ..tickmsg(0, -1.0)
        });
        let assembled = s
            .pending
            .as_ref()
            .expect("advance assembled")
            .inputs
            .clone();
        assert_eq!(
            assembled[&p1],
            Input::default(),
            "the remote's walk axis is overridden while its intent is filed"
        );
        assert_eq!(
            assembled[&PlayerId(0)],
            Input::default(),
            "the same rule covers the host's own seat"
        );
        step_ready(&mut s);

        // The remote goes silent while still piloting: the starved HOLD would replay its
        // last (walk) input — the intent persists through starvation, so it stays neutral.
        s.advance(tickmsg(1, 0.0));
        let assembled = s
            .pending
            .as_ref()
            .expect("advance assembled")
            .inputs
            .clone();
        assert_eq!(
            assembled[&p1],
            Input::default(),
            "a starved HOLD never replays a walk input underneath the craft"
        );
        step_ready(&mut s);

        // Off the craft, the very next walk input applies again.
        s.record_remote(p1, tickmsg(1, 1.0));
        s.advance(tickmsg(2, 0.5));
        let assembled = s
            .pending
            .as_ref()
            .expect("advance assembled")
            .inputs
            .clone();
        assert_eq!(
            assembled[&p1],
            input(1.0),
            "on foot again ⇒ input passes through"
        );
        assert_eq!(assembled[&PlayerId(0)], input(0.5));
    }

    /// RESTART works from every seat (rl#261): the pilot mask strips walk axes and ACTION
    /// (the ship's brake shares physical inputs with Extract) but passes RESTART through,
    /// and the round actually restarts on the piloting player's press.
    #[test]
    fn a_piloting_players_restart_survives_the_assembly_mask() {
        use crab_world::vehicle::VehicleKind;
        let p1 = PlayerId(1);
        let mut s = srv(42, &ids(2));

        s.record_remote(
            p1,
            TickMsg {
                issue_tick: 0,
                input: Input::new(1.0, 1.0, 0.0, buttons::RESTART | buttons::ACTION),
                pilot: Some(intent(VehicleKind::Plane)),
            },
        );
        s.advance(TickMsg {
            pilot: Some(intent(VehicleKind::Ship)),
            ..tickmsg(0, -1.0)
        });
        let assembled = s
            .pending
            .as_ref()
            .expect("advance assembled")
            .inputs
            .clone();
        assert_eq!(
            assembled[&p1],
            Input::new(0.0, 0.0, 0.0, buttons::RESTART),
            "walk axes and ACTION are masked; RESTART passes"
        );
        assert!(
            s.step_next(&[], Default::default()).restarted,
            "the piloting player's RESTART press restarts the round"
        );
    }

    /// Step every tick `advance` has assembled and return the decoded snapshots.
    fn step_ready(s: &mut Server) -> Vec<CoreSnapshot> {
        let mut out = Vec::new();
        while s.next_tick_ready() {
            let bytes = s.step_next(&[], Default::default()).snapshot;
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
            let _ = s.step_next(&[], Default::default());
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

    /// Drive `s` through ticks `0..=last`, recording one real input for player 1 whenever
    /// `send(t)` says so (issue ticks count up independently of `t`), and collect every
    /// [`StarvationReport`] as it surfaces. The starvation-observability test harness (rl#213).
    fn run_with_sender(
        s: &mut Server,
        last: u64,
        send: impl Fn(u64) -> bool,
    ) -> Vec<StarvationReport> {
        let mut reports = Vec::new();
        let mut issue = 0u64;
        for t in 0..=last {
            if send(t) {
                s.record_remote(PlayerId(1), tickmsg(issue, 1.0));
                issue += 1;
            }
            s.advance(tickmsg(t, 0.0));
            let _ = s.step_next(&[], Default::default());
            reports.extend(s.take_starvation_reports());
        }
        reports
    }

    /// rl#213: a stream that goes SILENT reports at the first window boundary — naming the
    /// player and the starved count — and then re-reports at the cooldown cadence, never once
    /// per window (the flood bound).
    #[test]
    fn silent_stream_reports_starvation_once_per_cooldown() {
        let mut s = srv(42, &ids(2));
        // One real input creates the stream, then silence — permanently held.
        let reports = run_with_sender(
            &mut s,
            STARVATION_REPORT_COOLDOWN + STARVATION_WINDOW,
            |t| t == 0,
        );
        assert_eq!(
            reports.len(),
            2,
            "one report at the first window boundary, the next only after the cooldown: {reports:?}"
        );
        assert_eq!(
            (reports[0].pid, reports[0].tick, reports[0].starved),
            (PlayerId(1), STARVATION_WINDOW, STARVATION_WINDOW - 1),
            "first window: every fill after the one consumed input was a hold"
        );
        assert_eq!(
            (reports[1].pid, reports[1].tick, reports[1].starved),
            (
                PlayerId(1),
                STARVATION_WINDOW + STARVATION_REPORT_COOLDOWN,
                STARVATION_WINDOW
            ),
            "re-report exactly one cooldown after the first"
        );
    }

    /// The maximal starvation — a rostered player that never delivered ONE input, so it has no
    /// stream at all — counts too: the tally keys off the ROSTER, not stream existence.
    #[test]
    fn never_sending_player_reports_full_window_starvation() {
        let mut s = srv(42, &ids(2));
        let reports = run_with_sender(&mut s, STARVATION_WINDOW, |_| false);
        assert_eq!(reports.len(), 1, "one report at the boundary: {reports:?}");
        assert_eq!(
            (reports[0].pid, reports[0].starved),
            (PlayerId(1), STARVATION_WINDOW),
            "every fill of the window was the no-stream neutral"
        );
    }

    /// rl#213's chronic case: a late client running at HALF rate alternates hold/consume —
    /// no long consecutive run ever forms, but half of every window is starved. The WINDOWED
    /// count catches it (a consecutive-run counter would stay silent forever).
    #[test]
    fn chronic_half_rate_stream_reports_starvation() {
        let mut s = srv(42, &ids(2));
        let reports = run_with_sender(&mut s, STARVATION_WINDOW, |t| t % 2 == 0);
        assert_eq!(reports.len(), 1, "half-rate is chronic: {reports:?}");
        assert_eq!(
            reports[0].starved,
            STARVATION_WINDOW / 2,
            "every other fill was a hold"
        );
    }

    /// A healthy full-rate stream never reports — occasional-holds-only noise would train
    /// operators to ignore the signal.
    #[test]
    fn healthy_stream_reports_no_starvation() {
        let mut s = srv(42, &ids(2));
        let reports = run_with_sender(&mut s, 2 * STARVATION_WINDOW, |_| true);
        assert!(
            reports.is_empty(),
            "full-rate stream is healthy: {reports:?}"
        );
    }

    /// A starved stream HOLDS the last consumed move axes but zeroes the per-tick look delta
    /// (else the avatar spins) and the button taps (else a grab/restart re-fires).
    #[test]
    fn starved_stream_holds_axes_and_strips_edges() {
        let mut stream = InputStream::new();
        stream.record(TickMsg {
            issue_tick: 0,
            input: Input::new(0.5, 1.0, 0.7, buttons::ACTION),
            pilot: None,
        });
        let (consumed, starved) = stream.consume();
        assert_eq!(consumed, Input::new(0.5, 1.0, 0.7, buttons::ACTION));
        assert!(!starved, "a real consume is not starved");
        let (held, starved) = stream.consume();
        assert!(starved, "an empty queue is a starved hold");
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
            let _ = s.step_next(&[], Default::default());
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
                pilot: None,
            });
        }
        assert_eq!(
            stream.queue.len(),
            TARGET_BACKLOG,
            "a burst folds down to the target depth at record time"
        );
        let (consumed, _) = stream.consume();
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
        let mut s = srv(42, &ids(2));
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
        // The joiner starts issuing at its effective tick (`ClientSim::join_at`) the moment its
        // welcome lands — these arrive while the host is still assembling pre-join ticks.
        s.record_remote(
            adm.pid,
            TickMsg {
                issue_tick: adm.effective_tick,
                input: input(1.0),
                pilot: None,
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

    #[test]
    fn server_authoritative_path_is_sp_identical() {
        const SEED: u64 = 0x00C0FFEE;
        const SUBMITS: u64 = 40;
        let me = PlayerId(0);
        let roster = vec![me];

        // Off past CRAB_GRAB_RADIUS and receding — the equivalence claim needs a LIVE world
        // every tick, so the scripted crab must never down the walking player (rl#236).
        let pose_at = |tick: u64| CrabPose {
            pos: Pos {
                x: 15_000 + tick as i64 * 11,
                z: -12_000 - tick as i64 * 7,
            },
            yaw: (tick as i32).wrapping_mul(4096),
            claws: Vec::new(),
        };
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
                sim.set_external_crab_pose(0, p.pos, p.yaw);
                sim.step(&BTreeMap::from([(me, input_at(i))]));
                baseline.push((sim.tick(), sim.state_hash()));
            }
        }

        // --- Path B: the server steps its OWN sim and emits a serialized snapshot; a SEPARATE
        // client sim is advanced SOLELY by `apply_core_snapshot`. The client [`ClientSim`] is
        // exactly what the real driver holds (submit_local_input UP, apply snapshot DOWN).
        let mut server_hashes = Vec::new();
        {
            let mut client = ClientSim::new(SEED, &roster, me);
            let mut server = Server::new(me, &roster, Sim::new(SEED, &roster));
            for i in 0..SUBMITS {
                let msg = client.submit_local_input(input_at(i), None);
                server.advance(msg);
                while server.next_tick_ready() {
                    let bytes = server
                        .step_next(&[pose_at(server.sim().tick())], Default::default())
                        .snapshot;
                    let snap =
                        CoreSnapshot::from_bytes(&bytes).expect("the server snapshot decodes");
                    client.apply_core_snapshot(snap);
                    server_hashes.push((server.sim().tick(), server.sim().state_hash()));
                    assert_eq!(
                        client.core_snapshot().to_bytes(),
                        bytes,
                        "the client re-emits the server's exact snapshot (carried state mirrored)"
                    );
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

    #[test]
    fn restart_does_not_wedge_the_match() {
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
                    pilot: None,
                },
            );
            s.advance(tickmsg(t, 1.0));
            assert!(s.next_tick_ready(), "tick {t} is steppable — no wedge");
            let stepped = s.step_next(&[], Default::default());
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
        let crab_spawn = server.sim().crabs()[0].pos();

        let pose_at = |tick: u64| CrabPose {
            pos: Pos {
                // Past CRAB_GRAB_RADIUS of both the host line and the join slots, and
                // receding — a mid-test down would freeze the very state this test
                // transfers (rl#236).
                x: 16_000 + tick as i64 * 13,
                z: -13_000 - tick as i64 * 9,
            },
            yaw: (tick as i32).wrapping_mul(2048),
            claws: Vec::new(),
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
                        pilot: None,
                    },
                );
                *next_issue += 1;
            }
            server.advance(TickMsg {
                issue_tick: i,
                input: input_at(i),
                pilot: None,
            });
            while server.next_tick_ready() {
                let tick = server.sim().tick();
                let bytes = server
                    .step_next(&[pose_at(tick)], Default::default())
                    .snapshot;
                snaps.push(CoreSnapshot::from_bytes(&bytes).expect("the server snapshot decodes"));
            }
        }

        let (jpid, eff, _) = joiner.expect("a joiner was admitted");
        for w in snaps.windows(2) {
            assert!(
                w[1].tick > w[0].tick,
                "the tick is monotonic — the join never resets the round"
            );
        }
        let before = snaps
            .iter()
            .find(|s| s.tick == eff)
            .expect("the sim stepped up to eff");
        assert!(
            !before.players.contains_key(&jpid),
            "no joiner before its effective tick"
        );
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
        assert_eq!(
            after.crabs[0].pos(),
            pose_at(eff).pos,
            "the crab is at its live pose at tick eff"
        );
        assert_ne!(
            after.crabs[0].pos(),
            crab_spawn,
            "the crab did NOT reset to spawn (no rebuild-on-join)"
        );
        assert!(eff > JOIN_LEAD, "the join genuinely lands mid-round");
    }

    #[test]
    fn admission_gate_admits_match_and_refuses_mismatches() {
        let ha = 0x5A11_2233u64;
        let req = |asset_digest, crab_count| JoinRequest {
            asset_digest,
            crab_count,
        };
        assert_eq!(
            may_admit_joiner(ha, 2, &req(ha, 2)),
            Ok(()),
            "the host admits a matching joiner"
        );
        assert_eq!(
            may_admit_joiner(ha, 2, &req(ha, 0)),
            Ok(()),
            "a headless joiner (rig count 0) renders nothing — always count-admissible"
        );
        assert_eq!(
            may_admit_joiner(ha, 2, &req(ha ^ 1, 2)),
            Err(AdmissionRefusal::AssetsMismatch {
                host: ha,
                joiner: ha ^ 1
            }),
            "a different Sally/colliders is refused"
        );
        assert_eq!(
            may_admit_joiner(ha, 2, &req(ha, 1)),
            Err(AdmissionRefusal::CrabCountMismatch { host: 2, joiner: 1 }),
            "a rendering joiner with the wrong rig count is refused (it would show the wrong \
             number of crabs)"
        );
    }
}
