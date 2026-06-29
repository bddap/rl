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
use crate::sim::{Input, PlayerId};

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
    /// The frozen participant set (sorted, deduped). Static for Stage 1+2 — dynamic join is a
    /// later stage (rl#151) — so a tick is "complete" exactly when it holds an input from every
    /// member of this set.
    roster: Vec<PlayerId>,
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
        let mut roster = roster.to_vec();
        roster.sort();
        roster.dedup();
        Self {
            roster,
            ledger: BTreeMap::new(),
            confirmed: BTreeMap::new(),
            emitted_confirmed: BTreeMap::new(),
            next_emit: INPUT_DELAY,
        }
    }

    /// The frozen roster (sorted).
    pub fn roster(&self) -> &[PlayerId] {
        &self.roster
    }

    /// Record one client's tick message — its input for `msg.apply_tick` plus its latest confirmed
    /// (tick, hash) — and return every [`TickSet`] this completes: the consecutive run of
    /// fully-inputted ticks from the emit cursor. `from` MUST be the authenticated sender (the
    /// transport binds it to the QUIC peer id, or it is the local client's own id), never read from
    /// a body — otherwise a client could file input as someone else. An input from a non-rostered
    /// player, or for an already-emitted tick, is dropped.
    #[must_use = "the returned sets must be broadcast to every client (incl. the local one), or ticks never advance"]
    pub fn record(&mut self, from: PlayerId, msg: TickMsg) -> Vec<TickSet> {
        if !self.roster.contains(&from) {
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
            .is_some_and(|t| self.roster.iter().all(|p| t.contains_key(p)))
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

    fn ids(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    fn input(s: f32) -> Input {
        Input::from_axes(s, 0.0)
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
        use crate::lockstep::Lockstep;
        let roster = ids(2);
        let mut server = Server::new(&roster);
        let mut a = Lockstep::new(42, &roster, PlayerId(0));
        let mut b = Lockstep::new(42, &roster, PlayerId(1));
        for t in 0..30u64 {
            // Each client issues its input and ships it UP to the server.
            let ma = a.submit_local_input(Input::from_axes((t % 3) as f32 - 1.0, 0.5));
            let mb = b.submit_local_input(Input::from_axes(0.0, (t % 2) as f32));
            let mut sets = server.record(PlayerId(0), ma);
            sets.extend(server.record(PlayerId(1), mb));
            // The server broadcasts each COMPLETE set DOWN to both clients; each records the
            // OTHERS' inputs (its own was filed by `submit_local_input`).
            for s in &sets {
                for pm in unpack_tickset(s, PlayerId(0)) {
                    assert!(a.record_remote(pm.pid, pm.msg).is_none(), "no fault recording");
                }
                for pm in unpack_tickset(s, PlayerId(1)) {
                    assert!(b.record_remote(pm.pid, pm.msg).is_none(), "no fault recording");
                }
            }
            assert!(a.try_advance().is_empty());
            assert!(b.try_advance().is_empty());
        }
        assert_eq!(a.sim().tick(), b.sim().tick());
        assert_eq!(
            a.sim().state_hash(),
            b.sim().state_hash(),
            "two clients through one server stay bit-identical (the lockstep digest oracle)"
        );
        assert!(
            a.sim().tick() > INPUT_DELAY,
            "the round advanced past the warmup window"
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
}
