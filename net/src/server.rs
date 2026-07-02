//! The match server: the AUTHORITATIVE per-tick stepper + input ledger + roster coordinator.
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
//! The server ALSO owns the input LEDGER + roster coordinator: it collects each client's input for a
//! tick (up-channel [`TickMsg`]) and, once every rostered client's input for that tick is in, feeds
//! that assembled set ([`TickSet`]) to its authoritative step. Inputs flow UP; the authoritative
//! [`CoreSnapshot`](crate::snapshot::CoreSnapshot) flows DOWN to every client, which ADOPTS it whole
//! (no re-sim, no peer cross-check — the host is the source of truth).
//!
//! The ledger/roster core is pure and transport-agnostic:
//! [`crate::transport`] / [`crate::net_loop`] move the bytes (loopback for the co-located client,
//! QUIC for a remote one). The one impurity is the owned [`Sim`] the authoritative step advances.

use std::collections::{BTreeMap, VecDeque};

use crate::lockstep::TickMsg;
use crate::roster::RosterSchedule;
use crate::sim::{Input, PlayerId, Pos, Sim};

/// Ticks of lead between admitting a joiner and its roster change taking effect. Past the
/// emit cursor so the joiner can receive its welcome and issue its first input before the tick is
/// due — plus a tick of slack so a late packet doesn't strand the boundary. The joiner builds its
/// [`crate::lockstep::Lockstep`] via `join_at(effective_tick)`, issuing input for `effective_tick`
/// onward. (Unverified under real QUIC latency.)
pub const JOIN_LEAD: u64 = 3;

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

/// The complete input set for ONE tick, assembled by the server once every rostered client's
/// input for that tick is in — the unit its own authoritative step consumes (clients ship inputs
/// UP; only [`crate::snapshot::CoreSnapshot`] state goes DOWN, never a set to re-step). Complete
/// by construction — a step never runs mid-set waiting on a straggler, because the server absorbed
/// that wait before assembling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TickSet {
    /// The tick these inputs apply at.
    pub apply_tick: u64,
    /// Every rostered player's input for `apply_tick`, in `PlayerId` order (the sim's apply
    /// order). Holds an entry for every roster member — that completeness is the invariant the
    /// server enforces before emitting.
    pub inputs: BTreeMap<PlayerId, Input>,
}

/// The input ledger + roster coordinator for one match. Pure: [`Server::record`] files a client's
/// message and returns the sets it completed; the caller broadcasts them. Memory is bounded by play
/// (a complete tick is removed from the ledger the instant it's emitted).
pub struct Server {
    /// The participant set over time (sorted, deduped per change-point). A tick is "complete" when
    /// it holds an input from every member of [`RosterSchedule::at`] that tick — so a mid-match join
    /// ([`Server::admit`]) shifts the required set on the agreed tick, and ticks before it still
    /// complete on the old set.
    roster: RosterSchedule,
    /// Per-tick input table awaiting completion: `ledger[tick][player]`. A tick leaves the ledger
    /// (into a [`TickSet`]) the moment every roster member's input for it is present.
    ledger: BTreeMap<u64, BTreeMap<PlayerId, Input>>,
    /// The next tick to emit. Sets are emitted strictly in order (from tick 0), so a client receives a
    /// gap-free run it can apply directly.
    next_emit: u64,
    /// The AUTHORITATIVE world this server owns and steps. Its per-tick
    /// [`CoreSnapshot`](crate::snapshot::CoreSnapshot) is what every client renders. Stepped one
    /// tick at a time by [`Server::step_next`], gated by [`Server::next_tick_ready`].
    sim: Sim,
    /// Assembled, in-order input sets awaiting the authoritative step — `(tick, inputs)`.
    /// [`Coordinator::exchange`](crate::net_loop::Coordinator::exchange) files each emitted tick here
    /// (via [`enqueue_for_step`](Server::enqueue_for_step)) so [`step_next`](Server::step_next) can
    /// advance the sim AFTER the driver has pumped that tick's crab physics: the rapier pump needs the
    /// bevy `World` (which this pure core can't hold), so the authoritative step can't run at drain
    /// time. The headless `game net` host (no bevy world, no crab pump) enqueues and steps in the
    /// same pass, so the queue never dwells there.
    pending_step: VecDeque<(u64, BTreeMap<PlayerId, Input>)>,
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
    /// authoritative `sim` it will step. For solo `roster` is just `[me]`; for a hosted match it is
    /// the whole agreed roster. `sim` is the tick-0 world (the caller hands a clone of the client's
    /// freshly-built sim, so the two start byte-identical and the snapshots keep the client in sync).
    pub fn new(roster: &[PlayerId], sim: Sim) -> Self {
        Self {
            roster: RosterSchedule::frozen(roster),
            ledger: BTreeMap::new(),
            next_emit: 0,
            sim,
            pending_step: VecDeque::new(),
        }
    }

    /// Read-only view of the authoritative world (for the driver's hunt-target read + restart-edge
    /// detection, and tests). The client renders from the SNAPSHOT this sim emits, never this sim
    /// directly.
    pub fn sim(&self) -> &Sim {
        &self.sim
    }

    /// Whether the authoritative sim's next tick can be stepped NOW: the next tick's complete input
    /// set has been assembled and queued by [`drain_complete`](Server::drain_complete). `false` means
    /// the server is stalled waiting on a client's input for this tick.
    pub fn next_tick_ready(&self) -> bool {
        let tick = self.sim.tick();
        self.pending_step.front().is_some_and(|(t, _)| *t == tick)
    }

    /// Advance the authoritative sim by exactly one tick and return the resulting
    /// [`SteppedTick`]: the [`CoreSnapshot`](crate::snapshot::CoreSnapshot) already SERIALIZED to
    /// bytes, plus whether the tick was a RESTART edge. Injects `crab` (the freshly-pumped NN crab
    /// pose; `None` ⇒ crab unchanged) FIRST so this tick's grab/extraction resolve against the real
    /// body, then steps the sim with this tick's assembled input set, then builds the snapshot at
    /// this ONE site and returns its wire bytes. Returning bytes (not the struct) makes the
    /// hand-off ALWAYS serialized — the client decodes and applies, no by-reference shortcut even
    /// in SP. Must be called only when [`next_tick_ready`](Server::next_tick_ready) is `true`.
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
        let (queued_tick, inputs) = self
            .pending_step
            .pop_front()
            .expect("step_next called with no ready tick — guard on next_tick_ready");
        debug_assert_eq!(
            queued_tick, tick,
            "authoritative step queue out of order with the sim tick"
        );
        if let Some(c) = crab {
            self.sim.set_external_crab_pose(c.pos, c.yaw, c.digest);
        }
        let restarted = self.sim.step(&inputs);
        SteppedTick {
            snapshot: self.sim.core_snapshot().to_bytes(),
            restarted,
        }
    }

    /// Queue freshly-assembled [`TickSet`]s (from [`host_assemble`] or [`Server::depart`]) for the
    /// authoritative step, in emit order. Kept OFF [`drain_complete`](Server::drain_complete) so
    /// the driver controls exactly when the deferred step runs (see `pending_step`).
    pub fn enqueue_for_step(&mut self, sets: &[TickSet]) {
        for set in sets {
            self.pending_step
                .push_back((set.apply_tick, set.inputs.clone()));
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
    /// Schedules the new roster to take effect at `effective_tick` ([`JOIN_LEAD`] past the emit
    /// cursor) so the joiner can issue its first input before that tick is due. From
    /// `effective_tick` a tick completes only once the joiner's input is in too.
    ///
    /// Admission control ([`may_admit_joiner`]) is the CALLER's gate BEFORE this; `admit` is the
    /// bookkeeping once that gate has passed.
    pub fn admit(&mut self) -> Admission {
        let pid = self.lowest_free_pid();
        // Past the emit cursor by JOIN_LEAD, but always strictly after any change already scheduled
        // (two joins admitted before a tick emits would otherwise collide on the same tick).
        let effective_tick = self.roster.earliest_change_at(self.next_emit + JOIN_LEAD);
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
    /// host when a rostered peer's link dies. Without this the ledger waits FOREVER on the
    /// departed player's next input and the authoritative sim never ticks again: the
    /// host-freezes-on-player-exit hang. Schedules the shrunk roster at the earliest legal tick
    /// (the emit cursor, or just past any already-scheduled change), then:
    ///
    /// - BACKFILLS a neutral input for the departed player on any still-pending tick before the
    ///   boundary (its real input, if already ledgered, wins). Under host-authority this cannot
    ///   desync anyone — the server is the one stepper and clients adopt its snapshots — it just
    ///   lets the ticks the departed player still gates complete instead of stalling the match.
    /// - SCRUBS any input it pre-filed at/after the boundary, so no stray non-rostered entry rides
    ///   an emitted [`TickSet`].
    ///
    /// Returns the [`TickSet`]s this releases (the departed player was usually exactly what the
    /// cursor tick was waiting on), for [`enqueue_for_step`](Server::enqueue_for_step) like any
    /// `record`. The sim-side removal happens in [`step_next`](Server::step_next), derived from the
    /// roster schedule — one source, same as the join spawn. A pid not in the current roster is a
    /// no-op (a double departure report).
    #[must_use = "the released sets feed this server's own authoritative step (enqueue_for_step), or ticks never advance"]
    pub fn depart(&mut self, pid: PlayerId) -> Vec<TickSet> {
        let current = self.roster.current();
        if !current.contains(&pid) {
            return Vec::new();
        }
        let remaining: Vec<PlayerId> = current.iter().copied().filter(|p| *p != pid).collect();
        let effective_tick = self.roster.earliest_change_at(self.next_emit);
        self.roster.schedule_change(effective_tick, &remaining);
        for t in self.next_emit..effective_tick {
            if self.roster.at(t).contains(&pid) {
                self.ledger.entry(t).or_default().entry(pid).or_default();
            }
        }
        for (_, inputs) in self.ledger.range_mut(effective_tick..) {
            inputs.remove(&pid);
        }
        self.drain_complete()
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

    /// Record one client's tick message — its input for `msg.apply_tick` — and return every
    /// [`TickSet`] this completes: the consecutive run of fully-inputted ticks from the emit cursor.
    /// `from` MUST be the authenticated sender (the transport binds it to the QUIC peer id, or it is
    /// the local client's own id), never read from a body — otherwise a client could file input as
    /// someone else. An input from a non-rostered player, or for an already-emitted tick, is dropped.
    #[must_use = "the returned sets feed this server's own authoritative step (enqueue_for_step), or ticks never advance"]
    pub fn record(&mut self, from: PlayerId, msg: TickMsg) -> Vec<TickSet> {
        // A client may only file input for a tick at which it is rostered: this drops a stranger
        // always AND a joiner's input for ticks before its join takes effect (it isn't required
        // there, so buffering it would be dead weight the ledger never consumes).
        if !self.roster.at(msg.apply_tick).contains(&from) {
            return Vec::new();
        }
        // Only buffer inputs for ticks not yet emitted; an input for an already-broadcast tick is a
        // late duplicate the ledger would never consume.
        if msg.apply_tick >= self.next_emit {
            self.ledger
                .entry(msg.apply_tick)
                .or_default()
                .insert(from, msg.input);
        }
        self.drain_complete()
    }

    /// Emit every consecutively-complete tick from the cursor. A tick is complete when the ledger
    /// holds an input from every roster member; stops at the first incomplete tick (the server is
    /// now waiting on that tick's missing client, exactly the wait the clients are spared).
    fn drain_complete(&mut self) -> Vec<TickSet> {
        let mut out = Vec::new();
        while self.ledger.get(&self.next_emit).is_some_and(|t| {
            self.roster
                .at(self.next_emit)
                .iter()
                .all(|p| t.contains_key(p))
        }) {
            let tick = self.next_emit;
            let inputs = self.ledger.remove(&tick).expect("just checked present");
            out.push(TickSet {
                apply_tick: tick,
                inputs,
            });
            self.next_emit += 1;
        }
        out
    }

    /// Seed the ledger with inputs that arrived before this server started serving (a fast client
    /// sending during formation). The completed sets are intentionally discarded: every such input
    /// is for a tick the clients re-drive through the live exchange anyway (a re-record is
    /// idempotent), and an input below the emit cursor is dropped outright — so nothing downstream
    /// needs the sets here. ONE home for this "dropping the sets is safe" reasoning, shared by every
    /// driver that builds a server.
    pub fn seed_early(&mut self, early: &[crate::lockstep::PeerMsg]) {
        for pm in early {
            let _ = self.record(pm.pid, pm.msg);
        }
    }
}

/// The role-agnostic core of one SERVER tick, shared by the sync windowed driver
/// ([`crate::net_loop::Coordinator::exchange`]) and the async headless driver (`game net`): record
/// the drained remote-client inputs and this peer's own local input into `server`, and return the
/// completed [`TickSet`]s to feed the authoritative step ([`Server::enqueue_for_step`]). The ONE
/// implementation of the assemble so the two transports — which must differ (sync `block_on` vs async
/// `await`) — can't drift on the coordination logic. `remote` is empty for solo (a roster of one), so
/// solo flows through this same function.
pub fn host_assemble(
    server: &mut Server,
    me: PlayerId,
    local: TickMsg,
    remote: Vec<crate::lockstep::PeerMsg>,
) -> Vec<TickSet> {
    let mut sets = Vec::new();
    for pm in remote {
        sets.extend(server.record(pm.pid, pm.msg));
    }
    sets.extend(server.record(me, local));
    sets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lockstep::Lockstep;

    fn ids(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    /// A server over `roster` with a tick-0 authoritative sim on `seed` (the ledger/roster tests
    /// don't read the sim; the join/lockstep tests match their clients' seed so the authoritative
    /// world agrees, though they assert on the lockstep clients, not the server's sim).
    fn srv(seed: u64, roster: &[PlayerId]) -> Server {
        Server::new(roster, Sim::new(seed, roster))
    }

    fn input(s: f32) -> Input {
        Input::from_axes(s, 0.0)
    }

    fn tickmsg(apply_tick: u64, s: f32) -> TickMsg {
        TickMsg {
            apply_tick,
            input: input(s),
        }
    }

    /// One client (solo): every input completes its tick immediately, emitted in order from the
    /// warmup boundary. The SP=MP-uniformity core — solo is the server with a roster of one.
    #[test]
    fn solo_roster_completes_every_tick() {
        let mut s = srv(42, &ids(1));
        for t in 0..5 {
            let sets = s.record(
                PlayerId(0),
                TickMsg {
                    apply_tick: t,
                    input: input(1.0),
                },
            );
            assert_eq!(sets.len(), 1, "a 1-player tick completes at once");
            assert_eq!(sets[0].apply_tick, t);
            assert_eq!(sets[0].inputs, BTreeMap::from([(PlayerId(0), input(1.0))]));
        }
    }

    /// Two clients: a tick is held until BOTH inputs are in, then emitted complete. The server
    /// absorbs the wait so the clients never stall mid-set.
    #[test]
    fn tick_emitted_only_when_every_client_is_in() {
        let mut s = srv(42, &ids(2));
        let t = 0;
        let none = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: t,
                input: input(0.5),
            },
        );
        assert!(none.is_empty(), "one of two clients in ⇒ not complete");
        let sets = s.record(
            PlayerId(1),
            TickMsg {
                apply_tick: t,
                input: input(-0.5),
            },
        );
        assert_eq!(sets.len(), 1);
        assert_eq!(
            sets[0].inputs,
            BTreeMap::from([(PlayerId(0), input(0.5)), (PlayerId(1), input(-0.5))]),
            "the emitted set holds every roster member's input"
        );
    }

    /// Inputs arriving out of tick order are buffered and released in order once the gap fills, so a
    /// client always receives a gap-free run.
    #[test]
    fn sets_emit_in_order_after_a_gap_fills() {
        let mut s = srv(42, &ids(1));
        let a = 0;
        let b = 1;
        // Future tick first: buffered, nothing emitted (the cursor tick is still missing).
        let none = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: b,
                input: input(1.0),
            },
        );
        assert!(none.is_empty());
        // The cursor tick arrives: BOTH release, in order.
        let sets = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: a,
                input: input(0.0),
            },
        );
        assert_eq!(
            sets.iter().map(|s| s.apply_tick).collect::<Vec<_>>(),
            vec![a, b],
            "buffered future tick releases in order behind the filled cursor tick"
        );
    }

    /// An input from a player outside the roster is dropped (it can never complete a tick and must
    /// not grow the ledger), and never blocks the rostered players.
    #[test]
    fn non_rostered_input_is_dropped() {
        let mut s = srv(42, &ids(1));
        let none = s.record(
            PlayerId(9),
            TickMsg {
                apply_tick: 0,
                input: input(1.0),
            },
        );
        assert!(none.is_empty(), "a stranger's input completes nothing");
        let sets = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: 0,
                input: input(1.0),
            },
        );
        assert_eq!(sets.len(), 1, "the roster still completes on its own input");
        assert_eq!(sets[0].inputs.len(), 1, "no stranger leaked into the set");
    }

    /// The host-freeze-on-departure hang, distilled: two players, one departs mid-round with the cursor tick waiting
    /// on its input. Before `depart` the tick can never complete (the host freezes forever);
    /// `depart` releases it and every later tick completes on the remaining player alone.
    #[test]
    fn depart_releases_the_stalled_tick_and_the_match_continues() {
        let mut s = srv(42, &ids(2));
        // P0 (the host) files tick 0; P1 never will — the exact freeze state.
        let none = s.record(PlayerId(0), tickmsg(0, 1.0));
        assert!(
            none.is_empty(),
            "tick 0 stalls on the departed player's input"
        );
        // P1's link dies. Departure releases tick 0 with a neutral backfill for P1.
        let sets = s.depart(PlayerId(1));
        assert_eq!(sets.len(), 1, "the stalled tick releases on departure");
        assert_eq!(sets[0].apply_tick, 0);
        assert_eq!(
            sets[0].inputs[&PlayerId(1)],
            Input::default(),
            "missing input backfilled neutral"
        );
        assert_eq!(s.roster(), &[PlayerId(0)], "the roster shrank");
        // From here the match ticks on the host alone — no trace of the departed player.
        for t in 1..5 {
            let sets = s.record(PlayerId(0), tickmsg(t, 1.0));
            assert_eq!(
                sets.len(),
                1,
                "post-departure ticks complete on the survivor alone"
            );
            assert!(
                !sets[0].inputs.contains_key(&PlayerId(1)),
                "no stray departed entry"
            );
        }
    }

    /// The departure boundary is the emit cursor: a departed player's PRE-FILED not-yet-emitted
    /// inputs are scrubbed, and every tick from the cursor on completes (and emits) without its
    /// entry — no stray non-rostered input ever rides a [`TickSet`].
    #[test]
    fn depart_scrubs_prefiled_future_inputs() {
        let mut s = srv(42, &ids(2));
        // Both file tick 0 (completes; cursor moves to 1). P1 pre-files ticks 1 and 3, then leaves.
        let _ = s.record(PlayerId(0), tickmsg(0, 0.0));
        let _ = s.record(PlayerId(1), tickmsg(0, 0.0));
        let _ = s.record(PlayerId(1), tickmsg(1, -1.0));
        let _ = s.record(PlayerId(1), tickmsg(3, 1.0));
        let sets = s.depart(PlayerId(1));
        assert!(
            sets.is_empty(),
            "the cursor tick still awaits the survivor's input"
        );
        for t in 1..5 {
            let sets = s.record(PlayerId(0), tickmsg(t, 0.5));
            assert_eq!(sets.len(), 1, "each tick completes on the survivor alone");
            assert_eq!(
                sets[0].inputs.keys().collect::<Vec<_>>(),
                vec![&PlayerId(0)],
                "the departed player's pre-filed input was scrubbed, not emitted"
            );
        }
    }

    /// A departure while a JOIN is still pending lands past the join boundary (roster changes are
    /// append-only), and the gap ticks — which still roster the leaver — are neutral-backfilled so
    /// the match crosses the boundary instead of stalling on a player who will never file again.
    #[test]
    fn depart_with_a_pending_join_backfills_the_gap() {
        let mut s = srv(42, &ids(2));
        let adm = s.admit(); // P2 joins at adm.effective_tick (= JOIN_LEAD, nothing emitted yet)
        // P1's REAL input for gap tick 1 is already ledgered when it departs — it must win over
        // the neutral backfill (only genuinely missing ticks are fabricated).
        let _ = s.record(PlayerId(1), tickmsg(1, 0.7));
        let sets = s.depart(PlayerId(1));
        assert!(
            sets.is_empty(),
            "every tick still awaits P0 (and later the joiner)"
        );
        assert_eq!(
            s.roster(),
            &[PlayerId(0), PlayerId(2)],
            "the final roster is survivor + joiner"
        );
        // Ticks before the join boundary complete on P0 alone (P1 backfilled neutral, except the
        // tick it really filed).
        for t in 0..adm.effective_tick {
            let expect = if t == 1 { input(0.7) } else { Input::default() };
            let sets = s.record(PlayerId(0), tickmsg(t, 0.0));
            assert!(
                sets.iter()
                    .any(|set| set.apply_tick == t && set.inputs[&PlayerId(1)] == expect),
                "a pre-boundary tick emits with the leaver's real input where filed, neutral elsewhere"
            );
        }
        // The join tick (still pre-departure-boundary: the removal lands just past the pending
        // join) needs the joiner, and carries the leaver's neutral backfill one last time.
        let _ = s.record(PlayerId(0), tickmsg(adm.effective_tick, 0.0));
        let sets = s.record(PlayerId(2), tickmsg(adm.effective_tick, 0.0));
        assert!(
            sets.iter().any(|set| set.apply_tick == adm.effective_tick
                && set.inputs.len() == 3
                && set.inputs[&PlayerId(1)] == Input::default()),
            "the join tick emits with all three (the leaver backfilled)"
        );
        // From the departure boundary the leaver gates (and rides) nothing.
        let t = adm.effective_tick + 1;
        let _ = s.record(PlayerId(0), tickmsg(t, 0.0));
        let sets = s.record(PlayerId(2), tickmsg(t, 0.0));
        assert!(
            sets.iter().any(|set| set.apply_tick == t
                && set.inputs.len() == 2
                && !set.inputs.contains_key(&PlayerId(1))),
            "past the boundary ticks emit on survivor + joiner alone"
        );
    }

    /// Departing frees the [`PlayerId`] for a later joiner (`admit` allocates the lowest free id),
    /// and a double departure report is a harmless no-op.
    #[test]
    fn depart_is_idempotent_and_frees_the_pid() {
        let mut s = srv(42, &ids(2));
        let _ = s.record(PlayerId(0), tickmsg(0, 0.0));
        let first = s.depart(PlayerId(1));
        assert_eq!(first.len(), 1, "departure releases the stalled tick");
        assert!(
            s.depart(PlayerId(1)).is_empty(),
            "a repeat departure is a no-op"
        );
        assert!(
            s.depart(PlayerId(9)).is_empty(),
            "departing a stranger is a no-op"
        );
        assert_eq!(s.roster(), &[PlayerId(0)]);
        let adm = s.admit();
        assert_eq!(
            adm.pid,
            PlayerId(1),
            "the departed id is free for the next joiner"
        );
    }

    /// The sim side of departure: stepping through the departure boundary DESPAWNS the
    /// leaver from the authoritative world — the snapshot no longer carries it (clients adopt the
    /// removal), the round keeps its live tick (no reset), and the survivor plays on.
    #[test]
    fn stepping_through_a_departure_despawns_the_player() {
        use crate::snapshot::CoreSnapshot;
        let mut s = srv(42, &ids(2));
        // Tick 0 completes with both players; step it.
        let _ = s.record(PlayerId(0), tickmsg(0, 0.0));
        let sets = s.record(PlayerId(1), tickmsg(0, 0.0));
        s.enqueue_for_step(&sets);
        let snap = CoreSnapshot::from_bytes(&s.step_next(None).snapshot).expect("snapshot decodes");
        assert!(snap.players.contains_key(&PlayerId(1)), "both play tick 0");
        // P1 departs; the survivor drives the next ticks alone.
        let sets = s.depart(PlayerId(1));
        assert!(sets.is_empty(), "tick 1 awaits the survivor");
        let sets = s.record(PlayerId(0), tickmsg(1, 1.0));
        s.enqueue_for_step(&sets);
        let snap = CoreSnapshot::from_bytes(&s.step_next(None).snapshot).expect("snapshot decodes");
        assert_eq!(
            snap.tick, 2,
            "the round kept its live tick — departure never resets"
        );
        assert!(
            !snap.players.contains_key(&PlayerId(1)),
            "the leaver is despawned"
        );
        assert!(
            !snap.roster.contains(&PlayerId(1)),
            "the wire roster shrank"
        );
        assert!(
            !s.sim().has_player(PlayerId(1)),
            "the authoritative sim dropped it"
        );
        assert!(s.sim().has_player(PlayerId(0)), "the survivor plays on");
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
        // A second admit before any tick emits still gets a distinct, strictly-later effective tick
        // (the append-only invariant) and the next free id — the incumbents are never renumbered.
        let b = s.admit();
        assert_eq!(b.pid, PlayerId(3));
        assert!(
            b.effective_tick > a.effective_tick,
            "back-to-back joins don't collide"
        );
    }

    /// A scheduled join shifts the completeness requirement on its tick: ticks before it complete on
    /// the old roster; the join tick stalls until the joiner's input is in too.
    #[test]
    fn a_scheduled_join_gates_completeness_from_its_tick() {
        let mut s = srv(42, &ids(2));
        let adm = s.admit(); // P2 joins at adm.effective_tick
        let t = adm.effective_tick;
        // Fill every tick up to the join boundary on P0+P1 — they complete WITHOUT the joiner (it
        // isn't rostered there yet), advancing the emit cursor right to the boundary.
        for pre in 0..t {
            let _ = s.record(PlayerId(0), tickmsg(pre, 0.0));
            let sets = s.record(PlayerId(1), tickmsg(pre, 0.0));
            assert!(
                sets.iter()
                    .any(|set| set.apply_tick == pre && set.inputs.len() == 2),
                "a pre-join tick completes on the two incumbents alone"
            );
        }
        // The join tick needs all THREE: P0+P1 alone no longer complete it (the joiner is required
        // from here — there is no gap left, so this isolates the roster requirement).
        let _ = s.record(PlayerId(0), tickmsg(t, 0.0));
        let none = s.record(PlayerId(1), tickmsg(t, 0.0));
        assert!(
            none.is_empty(),
            "the join tick stalls until the joiner's input arrives"
        );
        let sets = s.record(PlayerId(2), tickmsg(t, 0.0));
        assert!(
            sets.iter()
                .any(|set| set.apply_tick == t && set.inputs.len() == 3),
            "the join tick emits complete with all three players once the joiner is in"
        );
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

    /// The SP-IDENTITY invariant. Over a scripted input + crab-pose log, the
    /// host-authoritative path (the server steps its OWN sim and emits a snapshot the client applies)
    /// produces the EXACT same per-tick `state_hash` as stepping a bare [`Sim`] by hand, tick-for-tick
    /// — proving the authoritative step is behavior-preserving. It also checks the always-serialized
    /// hand-off: the client (which only ever calls `apply_core_snapshot`) re-emits byte-identical
    /// snapshots, so it faithfully mirrors the authoritative carried state.
    #[test]
    fn server_authoritative_path_is_sp_identical() {
        use crate::sim::buttons;
        use crate::snapshot::CoreSnapshot;

        const SEED: u64 = 0x00C0FFEE;
        const SUBMITS: u64 = 40;
        let me = PlayerId(0);
        let roster = vec![me];

        // A deterministic, tick-varied crab pose. The rapier body is external to the sim; here a
        // synthetic stand-in is fed IDENTICALLY to both paths, so the comparison isolates the
        // sim-stepping equivalence the increment changes, independent of the physics engine the
        // live driver actually pumps (which feeds the SAME pose into both arms — see `drive_lockstep`).
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

        // --- Path A: the reference. Step a bare [`Sim`] directly, by hand, exactly as the server does
        // internally — the scheduled input each tick (from tick 0, no input-delay barrier), with the
        // crab pose injected BEFORE each step. This is what `Server::step_next` must reproduce.
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

        // --- Path B: increment-1. The server steps its OWN sim and emits a serialized snapshot; a
        // SEPARATE client sim is advanced SOLELY by `apply_core_snapshot`. A throwaway lockstep
        // stands in for the client's input scheduler (its `submit_local_input` is exactly what the
        // real driver feeds the server via `exchange`).
        let mut server_hashes = Vec::new();
        {
            // `client` is a real [`Lockstep`] (what the driver holds) that only ever APPLIES the
            // server's snapshot. It doubles as the input scheduler, exactly as the windowed client's
            // lockstep does (submit_local_input UP, apply snapshot DOWN).
            let mut client = Lockstep::new(SEED, &roster, me);
            let mut server = Server::new(&roster, Sim::new(SEED, &roster));
            for i in 0..SUBMITS {
                let msg = client.submit_local_input(input_at(i));
                // Assemble + enqueue for the authoritative step, exactly as `Coordinator::exchange`
                // does (record → enqueue_for_step) — a roster of one completes each tick at once.
                let sets = server.record(me, msg);
                server.enqueue_for_step(&sets);
                while server.next_tick_ready() {
                    let bytes = server.step_next(Some(pose_at(server.sim().tick()))).snapshot;
                    let snap =
                        CoreSnapshot::from_bytes(&bytes).expect("the server snapshot decodes");
                    client.apply_core_snapshot(snap);
                    server_hashes.push((server.sim().tick(), server.sim().state_hash()));
                    // The client faithfully mirrors the authoritative CARRIED state. It can't fold
                    // the intentionally-dropped `external_crab_digest`, so its `state_hash` differs by
                    // design — compare the snapshot it RE-EMITS, which must be byte-identical.
                    assert_eq!(
                        client.sim().core_snapshot().to_bytes(),
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
            "host-authoritative server-steps == lockstep self-steps, tick-for-tick (SP identical)"
        );
        // The warmup + real ticks were actually exercised, first applied tick brings the sim to 1.
        assert_eq!(baseline.len() as u64, SUBMITS);
        assert_eq!(baseline[0].0, 1);
    }

    /// The rl#204 wedge, distilled at the authoritative-server seam: a RESTART press used to
    /// rewind `Sim::tick` to 0 while the ledger-tick space (`next_emit`, the pending-step queue)
    /// stayed monotone — `next_tick_ready` compared the pending front tick against a sim frozen
    /// at 0 and the match wedged permanently, in MP and solo alike. Now a restart is a
    /// state-reset at the CURRENT tick: press R mid-round and assert the match keeps ticking —
    /// every subsequent input still completes, steps, and emits a monotone snapshot, and the
    /// restart tick's own snapshot carries the spawn-state world.
    #[test]
    fn restart_does_not_wedge_the_ledger() {
        use crate::snapshot::CoreSnapshot;
        use crate::sim::buttons;

        // Two rostered players — the MP shape from the issue; solo hits the same mismatch and
        // is covered by every other test's roster-of-one.
        let mut s = srv(42, &ids(2));
        let fresh_players: Vec<_> = Sim::new(42, &ids(2)).players().collect();
        const RESTART_AT: u64 = 10;
        const TOTAL: u64 = 25;
        let mut last_snap_tick = 0;
        for t in 0..TOTAL {
            let btns = if t == RESTART_AT { buttons::RESTART } else { 0 };
            let _ = s.record(PlayerId(0), tickmsg(t, 1.0));
            let sets = s.record(
                PlayerId(1),
                TickMsg {
                    apply_tick: t,
                    input: Input::new(0.0, 1.0, 0.0, btns),
                },
            );
            assert_eq!(sets.len(), 1, "tick {t} completes (the ledger never stalls)");
            s.enqueue_for_step(&sets);
            assert!(s.next_tick_ready(), "tick {t} is steppable — no wedge");
            let stepped = s.step_next(None);
            assert_eq!(
                stepped.restarted,
                t == RESTART_AT,
                "the restart edge fires exactly on the RESTART tick"
            );
            let snap = CoreSnapshot::from_bytes(&stepped.snapshot).expect("snapshot decodes");
            assert_eq!(snap.tick, t + 1, "snapshot ticks stay monotone and gap-free");
            assert!(snap.tick > last_snap_tick || t == 0);
            last_snap_tick = snap.tick;
            if t == RESTART_AT {
                assert_eq!(
                    snap.players.values().map(|p| p.pos()).collect::<Vec<_>>(),
                    fresh_players.iter().map(|(_, p)| p.pos()).collect::<Vec<_>>(),
                    "the restart tick's snapshot carries the spawn-state world"
                );
            }
        }
        assert_eq!(s.sim().tick(), TOTAL, "the match ran to the end — no freeze");
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
        use crate::snapshot::CoreSnapshot;

        const SEED: u64 = 0x105E_F00D;
        const JOIN_AT: u64 = 12; // submits before the join (well past warmup)
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

        // The host's input scheduler (a lockstep used ONLY for `submit_local_input`, exactly as the
        // windowed client feeds the server via `exchange`); the joiner's is created on admission.
        let mut host_sched = Lockstep::new(SEED, &[host], host);
        let mut joiner: Option<(PlayerId, u64, Lockstep)> = None; // (pid, effective_tick, scheduler)
        let mut snaps: Vec<CoreSnapshot> = Vec::new();

        for i in 0..SUBMITS {
            // Admit ONE joiner mid-round: the server schedules its roster change at `effective_tick`,
            // and from then a tick completes only once the joiner's input is in too.
            if i == JOIN_AT {
                let adm = server.admit();
                let sched = Lockstep::join_at(SEED, &adm.roster, adm.pid, adm.effective_tick);
                joiner = Some((adm.pid, adm.effective_tick, sched));
            }
            let mut sets = server.record(host, host_sched.submit_local_input(input_at(i)));
            if let Some((jpid, _, sched)) = joiner.as_mut() {
                sets.extend(server.record(*jpid, sched.submit_local_input(input_at(i))));
            }
            server.enqueue_for_step(&sets);
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
}
