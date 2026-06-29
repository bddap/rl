//! The match server: the per-tick input ledger + roster coordinator (the GCR MP rewrite,
//! Stage 1+2, bddap/rl#151).
//!
//! Minecraft-style separation: server code is separate from the client even in one process,
//! and every client (the server's own local one included) dials a server. The server is NOT
//! authoritative over world state — lockstep is preserved (each client runs the full
//! deterministic [`Sim`] from the shared input ledger). The server's ONE job is to be the
//! input LEDGER + roster coordinator: it collects each client's input for a tick (up-channel
//! [`TickMsg`]) and, once every rostered client's input for that tick is in, broadcasts the
//! COMPLETE input set ([`TickSet`]) back down. Inputs flow UP, assembled input-sets flow DOWN;
//! world state never crosses the wire (strict lockstep, no rewind/rollback).
//!
//! "Solo" is the same path with a roster of one: an internal server + the single local client.
//! That is the SP=MP-uniformity proof — there is no separate single-player code path, only the
//! server with one client (see [`crate::render::driver`]).
//!
//! This core is pure and transport-agnostic, exactly like [`crate::lockstep`]: it consumes
//! recorded client messages and emits sets to broadcast. [`crate::transport`] / [`crate::net_loop`]
//! move the bytes (loopback for the co-located client, QUIC for a remote one).

use std::collections::BTreeMap;

use crate::lockstep::{Confirmed, INPUT_DELAY, TickMsg};
use crate::roster::RosterSchedule;
use crate::sim::{Input, PlayerId};

/// Ticks of lead between admitting a joiner and its roster change taking effect (Stage 3). Past the
/// emit cursor so every client learns of the change, and the joiner can issue its first input,
/// before the tick is due. The `+ 2`: one tick for the [`Admission`] broadcast to reach every
/// client past the [`INPUT_DELAY`] input pipeline, one of slack so a late packet doesn't strand the
/// boundary — the same lead-time role [`INPUT_DELAY`] plays for ordinary input. The joiner builds
/// its [`crate::lockstep::Lockstep`] via `join_at(effective_tick)`, issuing input for
/// `effective_tick` onward. (Whether this margin suffices under real QUIC latency is the transport
/// increment's to verify; a too-short lead can't silently desync — the hash oracle catches it.)
pub const JOIN_LEAD: u64 = INPUT_DELAY + 2;

/// The outcome of [`Server::admit`]: the stable [`PlayerId`] allocated to the joiner, the tick its
/// roster change takes effect on every client (the round-boundary rebuild tick), and the complete
/// new roster from that tick. The caller broadcasts these so every client schedules the identical
/// change at the identical tick (and the joiner builds its session via
/// [`crate::lockstep::Lockstep::join_at`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admission {
    pub pid: PlayerId,
    pub effective_tick: u64,
    pub roster: Vec<PlayerId>,
}

/// A would-be joiner's credentials, sent UP to the host the moment it dials a live match (Stage 3,
/// rl#151). The host gates admission on these BEFORE allocating a [`PlayerId`]: a joiner whose
/// policy-weights or crab-asset digest disagrees with the host's is running a different brain or a
/// different Sally, so admitting it would put a wrong crab into the round (or desync the moment its
/// external-crab digest folds into the state hash). Such a joiner is REFUSED LOUDLY, never silently
/// dropped onto a mismatched body ([`may_admit_joiner`]; rl#114 refuse-rather-than-fake).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinRequest {
    /// FNV digest of the joiner's policy weights — must equal the host's, or the two run different
    /// brains and the external-crab digest (weights Xored in) desyncs on the first post-join tick.
    pub weights_digest: u64,
    /// Digest of the joiner's crab/collider assets — must equal the host's, or the joiner builds a
    /// different-shaped rapier body and diverges.
    pub asset_digest: u64,
}

/// Why a [`JoinRequest`] was refused — a loud, typed verdict the host sends back to the joiner and
/// logs to telemetry (never a silent drop). Carries the offending side so the refusal message is
/// actionable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionRefusal {
    /// The joiner's policy-weights digest differs from the host's.
    WeightsMismatch { host: u64, joiner: u64 },
    /// The joiner's crab-asset/collider digest differs from the host's.
    AssetsMismatch { host: u64, joiner: u64 },
}

impl std::fmt::Display for AdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmissionRefusal::WeightsMismatch { host, joiner } => write!(
                f,
                "policy-weights digest mismatch (host {host:#018x}, joiner {joiner:#018x}) — different brain"
            ),
            AdmissionRefusal::AssetsMismatch { host, joiner } => write!(
                f,
                "crab-asset digest mismatch (host {host:#018x}, joiner {joiner:#018x}) — different Sally/colliders"
            ),
        }
    }
}

/// The admission gate: may a joiner advertising `req` enter a match the host runs on
/// `(host_weights, host_assets)`? `Ok(())` only when BOTH digests match exactly — a single
/// mismatch is a typed [`AdmissionRefusal`] the caller surfaces loudly. Weights are checked first
/// so a double mismatch reports the brain (the more fundamental disagreement). The host→joiner
/// analogue of the formation-time `may_arm_external_crab` shared-asset gate, but per-joiner and
/// fail-LOUD rather than a silent disarm.
pub fn may_admit_joiner(
    host_weights: u64,
    host_assets: u64,
    req: &JoinRequest,
) -> Result<(), AdmissionRefusal> {
    if req.weights_digest != host_weights {
        return Err(AdmissionRefusal::WeightsMismatch {
            host: host_weights,
            joiner: req.weights_digest,
        });
    }
    if req.asset_digest != host_assets {
        return Err(AdmissionRefusal::AssetsMismatch {
            host: host_assets,
            joiner: req.asset_digest,
        });
    }
    Ok(())
}

/// The complete input set for ONE tick, broadcast by the server to every client once every
/// rostered client's input for that tick is in. The down-channel counterpart to the client's
/// up-channel [`TickMsg`]: clients ship inputs UP, the server ships the assembled set DOWN.
/// Complete by construction — a client never stalls mid-set waiting on a straggler, because the
/// server absorbed that wait before emitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TickSet {
    /// The tick these inputs apply at.
    pub apply_tick: u64,
    /// Every rostered player's input for `apply_tick`, in `PlayerId` order (the sim's apply
    /// order). Holds an entry for every roster member — that completeness is the invariant the
    /// server enforces before emitting.
    pub inputs: BTreeMap<PlayerId, Input>,
    /// Freshly-advanced confirmed (tick, hash) per client since the last emitted set — the
    /// relayed desync cross-check. Deduped at the server (only a client whose confirmed advanced
    /// appears), so feeding these straight into [`crate::lockstep::Lockstep::record_remote`]
    /// can't spam a stale `Unverifiable`. A subset of the roster; often empty.
    pub confirmed: BTreeMap<PlayerId, Confirmed>,
}

/// The input ledger + roster coordinator for one match. Pure: [`Server::record`] files a client's
/// message and returns the sets it completed; the caller broadcasts them. Memory is bounded by play
/// (a complete tick is removed from the ledger the instant it's emitted).
pub struct Server {
    /// The participant set over time (sorted, deduped per change-point). A tick is "complete" when
    /// it holds an input from every member of [`RosterSchedule::at`] that tick — so a mid-match join
    /// ([`Server::admit`], Stage 3 rl#151) shifts the required set on the agreed tick, and ticks
    /// before it still complete on the old set. With no join scheduled it is the frozen Stage-1+2
    /// roster.
    roster: RosterSchedule,
    /// Per-tick input table awaiting completion: `ledger[tick][player]`. A tick leaves the ledger
    /// (into a [`TickSet`]) the moment every roster member's input for it is present.
    ledger: BTreeMap<u64, BTreeMap<PlayerId, Input>>,
    /// The latest confirmed (tick, hash) heard from each client — the source of the relayed
    /// cross-check. Kept newest-wins.
    confirmed: BTreeMap<PlayerId, Confirmed>,
    /// The newest confirmed tick already relayed in a [`TickSet`] for each client, so a confirmed
    /// is relayed exactly once (the dedup behind [`TickSet::confirmed`]).
    emitted_confirmed: BTreeMap<PlayerId, u64>,
    /// The next tick to emit. Sets are emitted strictly in order, so a client receives a gap-free
    /// run it can apply directly. Starts at [`INPUT_DELAY`]: ticks `[0, INPUT_DELAY)` are the
    /// lockstep warmup, which each client runs on neutral input WITHOUT a server set (see
    /// [`crate::lockstep::Lockstep::advance_one`]), so the server never coordinates them.
    next_emit: u64,
}

impl Server {
    /// Start a server for `roster` (the frozen participant set, us + any remote clients). For solo
    /// this is just `[me]`; for a hosted match it is the whole agreed roster.
    pub fn new(roster: &[PlayerId]) -> Self {
        Self {
            roster: RosterSchedule::frozen(roster),
            ledger: BTreeMap::new(),
            confirmed: BTreeMap::new(),
            emitted_confirmed: BTreeMap::new(),
            next_emit: INPUT_DELAY,
        }
    }

    /// The current roster (sorted) — the latest scheduled set. Grows on [`Server::admit`].
    pub fn roster(&self) -> &[PlayerId] {
        self.roster.current()
    }

    /// Admit a new client mid-match (Stage 3, rl#151) and return the [`Admission`] for the caller to
    /// broadcast to every client. Allocates the LOWEST [`PlayerId`] not currently in the roster —
    /// append-only, so every existing player KEEPS its id (a positional renumber would desync every
    /// peer; determinism risk #1). Schedules the new roster to take effect at `effective_tick`
    /// ([`JOIN_LEAD`] past the emit cursor) so every client learns of it and the joiner can issue its
    /// first input before that tick is due. From `effective_tick` a tick completes only once the
    /// joiner's input is in too.
    ///
    /// Admission control (the joiner's weight/collider digests must match) is the CALLER's gate
    /// BEFORE this — a server must refuse a mismatched joiner loudly rather than admit it onto a
    /// wrong crab (rl#114 / [[silent-fallback-antipattern]]); `admit` is the bookkeeping once that
    /// gate has passed.
    pub fn admit(&mut self) -> Admission {
        let pid = self.lowest_free_pid();
        // Past the emit cursor by JOIN_LEAD, but always strictly after any change already scheduled
        // (two joins admitted before a tick emits would otherwise collide on the same tick).
        let effective_tick = (self.next_emit + JOIN_LEAD).max(self.roster.latest_change_tick() + 1);
        let mut roster = self.roster.current().to_vec();
        roster.push(pid);
        self.roster.schedule_change(effective_tick, &roster);
        Admission {
            pid,
            effective_tick,
            roster: self.roster.at(effective_tick).to_vec(),
        }
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

    /// Record one client's tick message — its input for `msg.apply_tick` plus its latest confirmed
    /// (tick, hash) — and return every [`TickSet`] this completes: the consecutive run of
    /// fully-inputted ticks from the emit cursor. `from` MUST be the authenticated sender (the
    /// transport binds it to the QUIC peer id, or it is the local client's own id), never read from
    /// a body — otherwise a client could file input as someone else. An input from a non-rostered
    /// player, or for an already-emitted tick, is dropped.
    #[must_use = "the returned sets must be broadcast to every client (incl. the local one), or ticks never advance"]
    pub fn record(&mut self, from: PlayerId, msg: TickMsg) -> Vec<TickSet> {
        // A client may only file input for a tick at which it is rostered: this drops a stranger
        // always AND a joiner's input for ticks before its join takes effect (it isn't required
        // there, so buffering it would be dead weight the ledger never consumes).
        if !self.roster.at(msg.apply_tick).contains(&from) {
            return Vec::new();
        }
        if let Some(c) = msg.confirmed {
            // Newest-wins: a client only ever advances its confirmed, but an out-of-order packet
            // mustn't roll it back.
            let newer = self.confirmed.get(&from).is_none_or(|prev| c.tick >= prev.tick);
            if newer {
                self.confirmed.insert(from, c);
            }
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
        while self
            .ledger
            .get(&self.next_emit)
            .is_some_and(|t| self.roster.at(self.next_emit).iter().all(|p| t.contains_key(p)))
        {
            let tick = self.next_emit;
            let inputs = self.ledger.remove(&tick).expect("just checked present");
            let confirmed = self.take_fresh_confirmed();
            out.push(TickSet {
                apply_tick: tick,
                inputs,
                confirmed,
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
    pub fn seed_early(&mut self, early: &[crate::net_loop::PeerMsg]) {
        for pm in early {
            let _ = self.record(pm.pid, pm.msg);
        }
    }

    /// The confirmeds that advanced since each client's last relayed one, marking them relayed. The
    /// dedup that keeps [`TickSet::confirmed`] from re-sending a stale hash (which would spam
    /// `Unverifiable` once it aged out of a client's history window).
    fn take_fresh_confirmed(&mut self) -> BTreeMap<PlayerId, Confirmed> {
        let fresh: BTreeMap<PlayerId, Confirmed> = self
            .confirmed
            .iter()
            .filter(|(pid, c)| {
                self.emitted_confirmed
                    .get(*pid)
                    .is_none_or(|&t| c.tick > t)
            })
            .map(|(&pid, &c)| (pid, c))
            .collect();
        for (&pid, &c) in &fresh {
            self.emitted_confirmed.insert(pid, c.tick);
        }
        fresh
    }
}

/// The role-agnostic core of one SERVER tick, shared by the sync windowed driver
/// ([`crate::net_loop::Coordinator::exchange`]) and the async headless driver (`game net`): record
/// the drained remote-client inputs and this peer's own local input into `server`, and return both
/// the completed sets to broadcast to every client AND the OTHER players' messages to apply to the
/// local sim. The ONE implementation of the assemble+unpack so the two transports — which must
/// differ (sync `block_on` vs async `await`) — can't drift on the coordination logic itself.
/// `remote` is empty for solo (a roster of one), so solo flows through this same function.
pub fn host_assemble(
    server: &mut Server,
    me: PlayerId,
    local: TickMsg,
    remote: Vec<crate::net_loop::PeerMsg>,
) -> (Vec<TickSet>, Vec<crate::net_loop::PeerMsg>) {
    let mut sets = Vec::new();
    for pm in remote {
        sets.extend(server.record(pm.pid, pm.msg));
    }
    sets.extend(server.record(me, local));
    let peer_msgs = sets.iter().flat_map(|s| unpack_tickset(s, me)).collect();
    (sets, peer_msgs)
}

/// Unpack a server [`TickSet`] into the per-player messages the client records — one [`PeerMsg`]
/// per OTHER rostered player (the local player's own input was already filed by
/// [`crate::lockstep::Lockstep::submit_local_input`]), carrying that player's input for the set's
/// tick plus any freshly relayed confirmed hash for the cross-check. The local sim then advances
/// exactly as it did off the old mesh `drain_inbox`, so the lockstep driver above the link is
/// unchanged by the server topology.
pub fn unpack_tickset(set: &TickSet, me: PlayerId) -> Vec<crate::net_loop::PeerMsg> {
    set.inputs
        .iter()
        .filter(|(pid, _)| **pid != me)
        .map(|(&pid, &input)| crate::net_loop::PeerMsg {
            pid,
            msg: TickMsg {
                apply_tick: set.apply_tick,
                input,
                confirmed: set.confirmed.get(&pid).copied(),
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lockstep::Lockstep;
    use crate::net_loop::PeerMsg;

    fn ids(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    fn input(s: f32) -> Input {
        Input::from_axes(s, 0.0)
    }

    fn tickmsg(apply_tick: u64, s: f32) -> TickMsg {
        TickMsg {
            apply_tick,
            input: input(s),
            confirmed: None,
        }
    }

    /// One client (solo): every input completes its tick immediately, emitted in order from the
    /// warmup boundary. The SP=MP-uniformity core — solo is the server with a roster of one.
    #[test]
    fn solo_roster_completes_every_tick() {
        let mut s = Server::new(&ids(1));
        for t in INPUT_DELAY..INPUT_DELAY + 5 {
            let sets = s.record(
                PlayerId(0),
                TickMsg {
                    apply_tick: t,
                    input: input(1.0),
                    confirmed: None,
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
        let mut s = Server::new(&ids(2));
        let t = INPUT_DELAY;
        let none = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: t,
                input: input(0.5),
                confirmed: None,
            },
        );
        assert!(none.is_empty(), "one of two clients in ⇒ not complete");
        let sets = s.record(
            PlayerId(1),
            TickMsg {
                apply_tick: t,
                input: input(-0.5),
                confirmed: None,
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
        let mut s = Server::new(&ids(1));
        let a = INPUT_DELAY;
        let b = INPUT_DELAY + 1;
        // Future tick first: buffered, nothing emitted (the cursor tick is still missing).
        let none = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: b,
                input: input(1.0),
                confirmed: None,
            },
        );
        assert!(none.is_empty());
        // The cursor tick arrives: BOTH release, in order.
        let sets = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: a,
                input: input(0.0),
                confirmed: None,
            },
        );
        assert_eq!(
            sets.iter().map(|s| s.apply_tick).collect::<Vec<_>>(),
            vec![a, b],
            "buffered future tick releases in order behind the filled cursor tick"
        );
    }

    /// A confirmed hash is relayed exactly once (deduped), and only when it advances — so a client
    /// can't spam a stale `Unverifiable`.
    #[test]
    fn confirmed_is_relayed_once_and_only_when_advancing() {
        let mut s = Server::new(&ids(2));
        let c = Confirmed { tick: 0, hash: 0xabc };
        // Player 1's confirmed rides its input; player 0 has none yet.
        let _ = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: INPUT_DELAY,
                input: input(0.0),
                confirmed: None,
            },
        );
        let sets = s.record(
            PlayerId(1),
            TickMsg {
                apply_tick: INPUT_DELAY,
                input: input(0.0),
                confirmed: Some(c),
            },
        );
        assert_eq!(
            sets[0].confirmed,
            BTreeMap::from([(PlayerId(1), c)]),
            "the fresh confirmed is relayed in the completing set"
        );
        // Re-sending the SAME confirmed must not relay it again.
        let _ = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: INPUT_DELAY + 1,
                input: input(0.0),
                confirmed: None,
            },
        );
        let sets = s.record(
            PlayerId(1),
            TickMsg {
                apply_tick: INPUT_DELAY + 1,
                input: input(0.0),
                confirmed: Some(c),
            },
        );
        assert!(
            sets[0].confirmed.is_empty(),
            "an unchanged confirmed is not relayed twice"
        );
    }

    /// An input from a player outside the roster is dropped (it can never complete a tick and must
    /// not grow the ledger), and never blocks the rostered players.
    #[test]
    fn non_rostered_input_is_dropped() {
        let mut s = Server::new(&ids(1));
        let none = s.record(
            PlayerId(9),
            TickMsg {
                apply_tick: INPUT_DELAY,
                input: input(1.0),
                confirmed: None,
            },
        );
        assert!(none.is_empty(), "a stranger's input completes nothing");
        let sets = s.record(
            PlayerId(0),
            TickMsg {
                apply_tick: INPUT_DELAY,
                input: input(1.0),
                confirmed: None,
            },
        );
        assert_eq!(sets.len(), 1, "the roster still completes on its own input");
        assert_eq!(sets[0].inputs.len(), 1, "no stranger leaked into the set");
    }

    /// END-TO-END: two clients, each running the full deterministic [`Lockstep`] sim, kept
    /// bit-identical purely through ONE shared [`Server`] — inputs UP, the assembled set DOWN, no
    /// P2P mesh. The core proof that the server-coordinated path preserves lockstep (and the same
    /// machinery a solo round runs with a roster of one — SP=MP uniformity, rl#151).
    #[test]
    fn two_clients_stay_in_lockstep_through_the_server() {
        // Player 0 is the host (owns the server + runs a local client); player 1 is a remote client.
        // Drive them through the real `host_assemble`/`unpack_tickset` split, no transport.
        let roster = ids(2);
        let mut server = Server::new(&roster);
        let mut a = Lockstep::new(42, &roster, PlayerId(0)); // host's local client
        let mut b = Lockstep::new(42, &roster, PlayerId(1)); // remote client
        for t in 0..30u64 {
            let ma = a.submit_local_input(Input::from_axes((t % 3) as f32 - 1.0, 0.5));
            let mb = b.submit_local_input(Input::from_axes(0.0, (t % 2) as f32));
            // The remote client's input arrives at the host (drained off the transport in real life).
            let remote = vec![PeerMsg {
                pid: PlayerId(1),
                msg: mb,
            }];
            let (sets, a_peers) = host_assemble(&mut server, PlayerId(0), ma, remote);
            // Host applies the others' inputs to its local sim...
            for pm in a_peers {
                assert!(a.record_remote(pm.pid, pm.msg).is_none(), "host: no fault");
            }
            // ...and broadcasts the same complete sets DOWN; the remote client applies the others'.
            for s in &sets {
                for pm in unpack_tickset(s, PlayerId(1)) {
                    assert!(b.record_remote(pm.pid, pm.msg).is_none(), "client: no fault");
                }
            }
            assert!(a.try_advance().is_empty());
            assert!(b.try_advance().is_empty());
        }
        assert_eq!(a.sim().tick(), b.sim().tick());
        assert_eq!(
            a.sim().state_hash(),
            b.sim().state_hash(),
            "host + remote client through one server stay bit-identical (the lockstep digest oracle)"
        );
        assert!(
            a.sim().tick() > INPUT_DELAY,
            "the round advanced past the warmup window"
        );
    }

    /// `admit` allocates the lowest free [`PlayerId`] and NEVER renumbers an existing one — the
    /// Stage-3 determinism foundation (a positional renumber on join would desync every peer).
    #[test]
    fn admit_allocates_lowest_free_id_without_renumbering() {
        let mut s = Server::new(&ids(2)); // [P0, P1]
        let a = s.admit();
        assert_eq!(a.pid, PlayerId(2), "the lowest id not already in use");
        assert_eq!(a.roster, vec![PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert!(a.effective_tick >= INPUT_DELAY, "a join lands past the warmup window");
        assert_eq!(s.roster(), &[PlayerId(0), PlayerId(1), PlayerId(2)], "roster grew, ids 0/1 kept");
        // A second admit before any tick emits still gets a distinct, strictly-later effective tick
        // (the append-only invariant) and the next free id — the incumbents are never renumbered.
        let b = s.admit();
        assert_eq!(b.pid, PlayerId(3));
        assert!(b.effective_tick > a.effective_tick, "back-to-back joins don't collide");
    }

    /// A scheduled join shifts the completeness requirement on its tick: ticks before it complete on
    /// the old roster; the join tick stalls until the joiner's input is in too.
    #[test]
    fn a_scheduled_join_gates_completeness_from_its_tick() {
        let mut s = Server::new(&ids(2));
        let adm = s.admit(); // P2 joins at adm.effective_tick
        let t = adm.effective_tick;
        // Fill every tick up to the join boundary on P0+P1 — they complete WITHOUT the joiner (it
        // isn't rostered there yet), advancing the emit cursor right to the boundary.
        for pre in INPUT_DELAY..t {
            let _ = s.record(PlayerId(0), tickmsg(pre, 0.0));
            let sets = s.record(PlayerId(1), tickmsg(pre, 0.0));
            assert!(
                sets.iter().any(|set| set.apply_tick == pre && set.inputs.len() == 2),
                "a pre-join tick completes on the two incumbents alone"
            );
        }
        // The join tick needs all THREE: P0+P1 alone no longer complete it (the joiner is required
        // from here — there is no gap left, so this isolates the roster requirement).
        let _ = s.record(PlayerId(0), tickmsg(t, 0.0));
        let none = s.record(PlayerId(1), tickmsg(t, 0.0));
        assert!(none.is_empty(), "the join tick stalls until the joiner's input arrives");
        let sets = s.record(PlayerId(2), tickmsg(t, 0.0));
        assert!(
            sets.iter().any(|set| set.apply_tick == t && set.inputs.len() == 3),
            "the join tick emits complete with all three players once the joiner is in"
        );
    }

    /// Drive a full server-coordinated lockstep round starting from 2 clients (host P0 + remote P1),
    /// admitting ONE new client at each iteration in `join_iters`, and assert NO
    /// [`crate::lockstep::Fault`] on any peer at any tick. Returns the host plus every client
    /// (incumbents + joiners) so a caller can assert final convergence. The ONE driver every join
    /// test shares: a join is wired exactly as the transport increment will (the server `admit`s, the
    /// allocation+effective-tick is broadcast to all current clients via `schedule_roster_change`,
    /// and the joiner enters via `join_at`), all through the real `host_assemble`/`unpack_tickset`.
    fn run_round_with_joins(seed: u64, iters: u64, join_iters: &[u64]) -> (Lockstep, Vec<Lockstep>) {
        let roster0 = ids(2);
        let mut server = Server::new(&roster0);
        let mut host = Lockstep::new(seed, &roster0, PlayerId(0));
        let mut clients = vec![Lockstep::new(seed, &roster0, PlayerId(1))];
        // Deterministic, player- and tick-varied inputs so the round actually evolves (a constant
        // input would hide an apply-order bug).
        let inp =
            |p: u8, t: u64| Input::from_axes(((u64::from(p) + t) % 3) as f32 - 1.0, (t % 2) as f32);

        for it in 0..iters {
            let host_msg = host.submit_local_input(inp(0, it));
            let client_msgs: Vec<PeerMsg> = clients
                .iter_mut()
                .map(|c| {
                    let pid = c.me();
                    PeerMsg { pid, msg: c.submit_local_input(inp(pid.0, it)) }
                })
                .collect();

            if join_iters.contains(&it) {
                // The server admits the joiner and the allocation is broadcast to EVERY current
                // client (incumbents AND earlier joiners — so a prior joiner rebuilds at this newer
                // boundary too), then the new client enters at the live tick.
                let adm = server.admit();
                host.schedule_roster_change(adm.effective_tick, &adm.roster);
                for c in &mut clients {
                    c.schedule_roster_change(adm.effective_tick, &adm.roster);
                }
                clients.push(Lockstep::join_at(seed, &adm.roster, adm.pid, adm.effective_tick));
            }

            let (sets, host_peers) = host_assemble(&mut server, PlayerId(0), host_msg, client_msgs);
            for pm in host_peers {
                assert!(host.record_remote(pm.pid, pm.msg).is_none(), "host: no fault");
            }
            for set in &sets {
                for c in &mut clients {
                    for pm in unpack_tickset(set, c.me()) {
                        assert!(c.record_remote(pm.pid, pm.msg).is_none(), "client: no fault");
                    }
                }
            }
            assert!(host.try_advance().is_empty(), "host advances with no desync");
            for c in &mut clients {
                assert!(c.try_advance().is_empty(), "client advances with no desync");
            }
        }
        (host, clients)
    }

    /// THE Stage-3 determinism oracle: two clients run a server-coordinated lockstep round; a THIRD
    /// client JOINS mid-match; at the agreed tick every peer rebuilds the round over the new roster
    /// (the round-boundary join), and from there all three stay BIT-IDENTICAL — proven by the
    /// per-tick `state_hash` agreeing tick-for-tick with no [`crate::lockstep::Fault`] raised. The
    /// determinism risk Stage 3 exists to retire (stable ids + tick-aligned roster change + the
    /// join mechanism), all the way through the real `Server`/`host_assemble`/`unpack_tickset` split.
    #[test]
    fn a_mid_game_join_keeps_every_peer_in_lockstep() {
        let (host, clients) = run_round_with_joins(0xC0FFEE, 40, &[5]);
        assert_eq!(clients.len(), 2, "the remote client plus the one joiner");
        let joiner = &clients[1];
        assert_eq!(host.next_tick(), clients[0].next_tick(), "host + client agree on the tick");
        assert_eq!(host.next_tick(), joiner.next_tick(), "the joiner caught up to the same tick");
        for c in &clients {
            assert_eq!(
                host.sim().state_hash(),
                c.sim().state_hash(),
                "every peer (incl. the joiner) is bit-identical after the round-boundary join"
            );
        }
        assert_eq!(host.peers(), &[PlayerId(0), PlayerId(1), PlayerId(2)], "the roster grew to three");
        assert_eq!(host.peers(), joiner.peers(), "every peer agrees on the new roster");
        // The joiner genuinely played a stretch of the post-join round (not a degenerate 0-tick pass).
        assert!(
            joiner.next_tick() > JOIN_LEAD + INPUT_DELAY + 10,
            "the joiner ran well past its entry tick"
        );
    }

    /// TWO sequential joins: a third then a fourth client join at different ticks. The first joiner,
    /// now an INCUMBENT, must itself rebuild at the second join's boundary — the back-to-back-admit
    /// path (a second roster change scheduled strictly after the first). All four stay bit-identical,
    /// proving the round-boundary join generalizes past a single join with stable, never-renumbered ids.
    #[test]
    fn two_sequential_joins_keep_every_peer_in_lockstep() {
        let (host, clients) = run_round_with_joins(0x5EED, 60, &[5, 20]);
        assert_eq!(clients.len(), 3, "two incumbents (P1, the P2 joiner) plus the P3 joiner");
        for c in &clients {
            assert_eq!(host.next_tick(), c.next_tick(), "all four agree on the tick");
            assert_eq!(
                host.sim().state_hash(),
                c.sim().state_hash(),
                "all four (both joiners included) are bit-identical after two round-boundary joins"
            );
        }
        assert_eq!(
            host.peers(),
            &[PlayerId(0), PlayerId(1), PlayerId(2), PlayerId(3)],
            "stable lowest-free allocation grew the roster to four with no renumbering"
        );
    }

    /// `unpack_tickset` yields one message per OTHER player (never the local one — its input was
    /// already filed locally) and pairs each with that player's relayed confirmed.
    #[test]
    fn unpack_skips_self_and_pairs_confirmed() {
        let set = TickSet {
            apply_tick: 7,
            inputs: BTreeMap::from([
                (PlayerId(0), input(0.1)),
                (PlayerId(1), input(0.2)),
                (PlayerId(2), input(0.3)),
            ]),
            confirmed: BTreeMap::from([(PlayerId(2), Confirmed { tick: 3, hash: 9 })]),
        };
        let msgs = unpack_tickset(&set, PlayerId(1));
        let got: BTreeMap<PlayerId, (Input, Option<Confirmed>)> =
            msgs.iter().map(|m| (m.pid, (m.msg.input, m.msg.confirmed))).collect();
        assert_eq!(got.len(), 2, "self is skipped");
        assert!(!got.contains_key(&PlayerId(1)));
        assert_eq!(got[&PlayerId(0)], (input(0.1), None));
        assert_eq!(
            got[&PlayerId(2)],
            (input(0.3), Some(Confirmed { tick: 3, hash: 9 })),
            "a player's relayed confirmed rides its input message"
        );
        for m in &msgs {
            assert_eq!(m.msg.apply_tick, 7);
        }
    }

    /// The admission gate admits only a digest-matched joiner and refuses (loudly, typed) on either
    /// mismatch — the host→joiner analogue of the formation shared-asset gate, fail-LOUD per joiner.
    #[test]
    fn admission_gate_admits_match_and_refuses_each_mismatch() {
        let (hw, ha) = (0xB7A1u64, 0x5A11_2233u64);
        assert_eq!(
            may_admit_joiner(hw, ha, &JoinRequest { weights_digest: hw, asset_digest: ha }),
            Ok(()),
            "matching weights AND assets are admitted"
        );
        assert_eq!(
            may_admit_joiner(hw, ha, &JoinRequest { weights_digest: hw ^ 1, asset_digest: ha }),
            Err(AdmissionRefusal::WeightsMismatch { host: hw, joiner: hw ^ 1 }),
            "a different brain is refused"
        );
        assert_eq!(
            may_admit_joiner(hw, ha, &JoinRequest { weights_digest: hw, asset_digest: ha ^ 1 }),
            Err(AdmissionRefusal::AssetsMismatch { host: ha, joiner: ha ^ 1 }),
            "a different Sally/colliders is refused"
        );
        // A double mismatch reports the brain first (the more fundamental disagreement).
        assert!(matches!(
            may_admit_joiner(hw, ha, &JoinRequest { weights_digest: hw ^ 1, asset_digest: ha ^ 1 }),
            Err(AdmissionRefusal::WeightsMismatch { .. })
        ));
    }
}
