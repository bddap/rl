//! Deterministic fixed-tick lockstep driver (Factorio/RTS model).
//!
//! Each peer runs an identical [`Sim`]. Only inputs cross the wire. A tick advances
//! only once EVERY peer's input for that tick is in hand, so all peers compute the
//! same sequence of states — that's what makes a one-`u64`-per-tick state-hash
//! comparison a complete desync check.
//!
//! Input delay: a player's input issued at tick `T` is scheduled to APPLY at tick
//! `T + INPUT_DELAY`. That lead time is the window for the input to reach peers
//! before its tick is due, so a peer rarely has to stall waiting for it. The whole
//! mechanism here is transport-agnostic: it consumes already-received inputs and
//! emits the local input + hash to send. [`crate::net::transport`] moves the bytes.

use std::collections::BTreeMap;

use crate::net::sim::{Input, PlayerId, Sim};

/// Ticks between issuing an input and applying it. One tick of slack covers LAN
/// round-trips at a 60 Hz tick (~16 ms/tick) without stalling; raise it for higher
/// latency at the cost of input lag.
pub const INPUT_DELAY: u64 = 2;

/// How many recent per-tick local hashes to retain for comparing against peer hashes
/// that arrive after we've already applied the tick. It bounds how far a peer's hash
/// may lag ours and still be cross-checked; a hash older than this can no longer be
/// verified and is reported as a fault (see [`Lockstep::record_remote`]) rather than
/// silently dropped, so the "no desync goes unnoticed" guarantee holds. Far exceeds
/// [`INPUT_DELAY`] (the lag under healthy play), so honest peers never hit the edge.
const HASH_HISTORY: u64 = 256;

/// Largest gap, in ticks ahead of what we've applied, that we'll buffer a peer's
/// scheduled input or advertised hash for. Inputs/hashes referencing a tick further
/// out than this are rejected: under honest play a peer leads by ≈[`INPUT_DELAY`], so
/// anything wildly ahead is a bug or a hostile peer trying to grow our buffers
/// without bound. This caps [`Lockstep`]'s memory at O(`FUTURE_TICK_BOUND` × peers).
const FUTURE_TICK_BOUND: u64 = 1024;

/// The latest tick a peer has fully applied and the state hash it computed right
/// after. Paired because a hash is meaningless without the tick it belongs to;
/// carried as `Option` (see [`TickMsg::confirmed`]) so "nothing applied yet" is a
/// distinct, unmistakable state rather than a sentinel tick value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Confirmed {
    pub tick: u64,
    pub hash: u64,
}

/// What one peer publishes for a single tick: the input it wants applied at
/// `apply_tick`, plus its latest [`Confirmed`] state so peers can cross-check hashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickMsg {
    /// The tick at which `input` should be applied (issuing tick + [`INPUT_DELAY`]).
    pub apply_tick: u64,
    pub input: Input,
    /// This peer's latest confirmed (tick, hash), or `None` before its first tick.
    pub confirmed: Option<Confirmed>,
}

/// A cross-check fault the caller must surface — never silently swallowed, since an
/// undetected divergence is the failure mode lockstep exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// Hard divergence: at `tick`, our state hash and the peer's disagree. The sims
    /// have diverged; lockstep can't recover, so this is fatal to the session.
    Desync {
        tick: u64,
        peer: PlayerId,
        local_hash: u64,
        peer_hash: u64,
    },
    /// The peer advertised a hash for a `tick` we applied so long ago it's no longer
    /// in our history window, so we can't confirm it matched. Distinct from `Desync`
    /// (it may well have agreed) — it means our verification window was outrun, which
    /// under healthy play never happens, so it flags a sick link, not a sure divergence.
    Unverifiable {
        tick: u64,
        peer: PlayerId,
        peer_hash: u64,
    },
}

/// Drives one peer's deterministic sim: schedules inputs, advances only when every
/// peer's input for the next tick is present, and cross-checks state hashes.
pub struct Lockstep {
    sim: Sim,
    me: PlayerId,
    /// All peers in the session, including `me`. Fixed for the match; the input set
    /// required to advance a tick is exactly this set.
    peers: Vec<PlayerId>,
    /// Per-tick input table: `inputs[tick][player]`. A tick is ready to apply when it
    /// holds an entry for every peer. `BTreeMap` so a tick's inputs iterate in
    /// `PlayerId` order, matching the sim's apply order.
    inputs: BTreeMap<u64, BTreeMap<PlayerId, Input>>,
    /// Our latest confirmed (tick, hash) — what we advertise for peers to check.
    confirmed: Option<Confirmed>,
    /// Our own state hash per recently-applied tick, so a peer hash arriving AFTER we
    /// applied that tick can still be compared. Pruned to the newest [`HASH_HISTORY`]
    /// ticks.
    applied_hashes: BTreeMap<u64, u64>,
    /// Peer hashes that arrived for ticks we hadn't applied yet, checked when we reach
    /// the tick. The forward mirror of `applied_hashes`: between the two, a desync is
    /// caught whichever side's hash lands second.
    pending_peer_hashes: BTreeMap<(u64, PlayerId), u64>,
    /// The tick the NEXT local input will apply at. Incremented once per
    /// [`submit_local_input`], independent of how many ticks [`try_advance`] applies
    /// in a batch, so consecutive submits schedule consecutive ticks with no gaps.
    /// Starts at [`INPUT_DELAY`]: the first real input lands on the tick right after
    /// the warmup window (ticks `[0, INPUT_DELAY)` run on neutral input).
    next_issue_tick: u64,
}

impl Lockstep {
    /// Start a session. `seed` is the shared match seed (identical on every peer);
    /// `peers` is the full participant set; `me` is this peer's id and must be in it.
    pub fn new(seed: u64, peers: &[PlayerId], me: PlayerId) -> Self {
        Self::new_with_pilots(seed, peers, me, &[])
    }

    /// Like [`Lockstep::new`], but `pilots` spawn flying planes instead of on foot (the
    /// rl#38 vehicle first cut). Forwarded to [`Sim::new_with_pilots`]; every peer must
    /// pass the SAME `pilots` set or their sims diverge (pilot assignment is a launch
    /// flag here, not yet wire-negotiated — and the networked driver currently passes
    /// none, so this is solo/screenshot-only for now). With `pilots` empty this is
    /// [`Lockstep::new`].
    pub fn new_with_pilots(
        seed: u64,
        peers: &[PlayerId],
        me: PlayerId,
        pilots: &[PlayerId],
    ) -> Self {
        let mut peers = peers.to_vec();
        peers.sort();
        peers.dedup();
        debug_assert!(peers.contains(&me), "local player must be in the peer set");
        Self {
            sim: Sim::new_with_pilots(seed, &peers, pilots),
            me,
            peers,
            inputs: BTreeMap::new(),
            confirmed: None,
            applied_hashes: BTreeMap::new(),
            pending_peer_hashes: BTreeMap::new(),
            next_issue_tick: INPUT_DELAY,
        }
    }

    /// The next tick to be applied (0-based; equals the count already applied).
    pub fn next_tick(&self) -> u64 {
        self.confirmed.map_or(0, |c| c.tick + 1)
    }

    /// Submit THIS peer's input for the next issuing tick and get the message to
    /// broadcast. The input applies [`INPUT_DELAY`] ticks ahead of the apply cursor;
    /// the message also carries our latest confirmed state so peers can cross-check us.
    ///
    /// Call exactly once per tick before [`Lockstep::try_advance`] — the scheduled
    /// tick advances by one each call, so a missed or doubled call would gap or
    /// collide the input stream. We record our own input locally too, so a
    /// single-peer (offline) session still advances.
    pub fn submit_local_input(&mut self, input: Input) -> TickMsg {
        let apply_tick = self.next_issue_tick;
        self.next_issue_tick += 1;
        self.inputs
            .entry(apply_tick)
            .or_default()
            .insert(self.me, input);
        TickMsg {
            apply_tick,
            input,
            confirmed: self.confirmed,
        }
    }

    /// Ingest a peer's [`TickMsg`]: file its scheduled input and check its advertised
    /// hash. `from` must be the authenticated sender (the transport binds it to the
    /// QUIC peer id), never a value read from the message body — a peer could
    /// otherwise spoof another's input.
    ///
    /// Returns `Some(Fault)` if the peer's hash disagrees with ours for a tick we've
    /// already applied (a [`Fault::Desync`]) or references a tick too old to still be
    /// in our history (a [`Fault::Unverifiable`]). Otherwise the hash is stashed for
    /// [`try_advance`] to check when we reach that tick. Inputs/hashes referencing a
    /// tick outside the sane window `next_tick()..=next_tick()+FUTURE_TICK_BOUND` are
    /// dropped, bounding memory against a stale or misbehaving peer.
    #[must_use = "a returned Fault means the peer has diverged or fallen out of sync; surface it"]
    pub fn record_remote(&mut self, from: PlayerId, msg: TickMsg) -> Option<Fault> {
        // Buffer the input only within the sane forward window. Below `next_tick()` it
        // is an already-applied tick `try_advance` will never consume; far above it is
        // a bug/attack — either would grow `inputs` without bound.
        if self.in_window(msg.apply_tick) {
            self.inputs
                .entry(msg.apply_tick)
                .or_default()
                .insert(from, msg.input);
        }

        let c = msg.confirmed?;
        match self.applied_hashes.get(&c.tick) {
            Some(&local) => check(c.tick, from, local, c.hash), // already applied → compare now
            None if c.tick < self.next_tick() => {
                // Applied but pruned from history — we can no longer verify it matched.
                Some(Fault::Unverifiable {
                    tick: c.tick,
                    peer: from,
                    peer_hash: c.hash,
                })
            }
            None if self.in_window(c.tick) => {
                self.pending_peer_hashes.insert((c.tick, from), c.hash);
                None
            }
            None => None, // absurdly-future hash: drop, don't buffer
        }
    }

    /// Whether `tick` is in the window we'll buffer for: from the next tick to apply
    /// up to [`FUTURE_TICK_BOUND`] ahead. Outside it, a value is either already
    /// applied (never consumed) or implausibly far out (a misbehaving peer).
    fn in_window(&self, tick: u64) -> bool {
        let next = self.next_tick();
        (next..=next + FUTURE_TICK_BOUND).contains(&tick)
    }

    /// Advance as many ticks as are fully ready (every peer's input present),
    /// returning any faults detected against peer-advertised hashes.
    ///
    /// An empty vec means "in sync so far". A non-empty vec is the caller's cue to
    /// flag/halt; lockstep can't recover a diverged sim, so continuing past a desync
    /// only produces garbage. Stops at the first tick whose inputs are incomplete,
    /// leaving us stalled until the missing input arrives.
    pub fn try_advance(&mut self) -> Vec<Fault> {
        let mut faults = Vec::new();
        loop {
            let tick = self.next_tick();
            // Warmup window: the first INPUT_DELAY ticks have no scheduled input (the
            // earliest input any peer issues is for tick INPUT_DELAY). They apply with
            // neutral input on every peer, filling the input pipeline — the standard
            // lockstep cold-start. Without it the driver would stall at tick 0 forever
            // waiting for inputs that, by design, were never scheduled.
            let tick_inputs = if tick < INPUT_DELAY {
                BTreeMap::new()
            } else {
                let Some(tick_inputs) = self.inputs.get(&tick) else {
                    break;
                };
                if !self.peers.iter().all(|p| tick_inputs.contains_key(p)) {
                    break; // not everyone's input is here yet — stall this tick.
                }
                self.inputs.remove(&tick).expect("just checked present")
            };
            self.sim.step(&tick_inputs);
            let hash = self.sim.state_hash();
            self.confirmed = Some(Confirmed { tick, hash });
            self.applied_hashes.insert(tick, hash);
            while self.applied_hashes.len() as u64 > HASH_HISTORY {
                self.applied_hashes.pop_first();
            }
            // Compare against any peer hashes that arrived for this tick before we
            // reached it (the late-hash case is in record_remote).
            for &peer in &self.peers {
                if peer == self.me {
                    continue;
                }
                if let Some(peer_hash) = self.pending_peer_hashes.remove(&(tick, peer)) {
                    faults.extend(check(tick, peer, hash, peer_hash));
                }
            }
        }
        faults
    }

    /// Read-only sim view for rendering/inspection.
    pub fn sim(&self) -> &Sim {
        &self.sim
    }

    /// SOLO ONLY: hand the crab to external control (the rapier NN crab) — forwards to
    /// [`Sim::enable_external_crab`]. Call once at solo setup; the integer pursuit then stops
    /// and the caller drives the crab with [`Self::set_external_crab_pose`] each tick. Never
    /// on the networked path (a float crab desyncs peers).
    pub fn enable_external_crab(&mut self, external: bool) {
        self.sim.enable_external_crab(external);
    }

    /// SOLO ONLY: set the crab's ground position + yaw from the real NN crab body, BEFORE
    /// the next [`Self::try_advance`], so the grab/extraction checks resolve against it.
    /// Forwards to [`Sim::set_external_crab_pose`]; a no-op on behaviour unless
    /// [`Self::enable_external_crab`] was set.
    pub fn set_external_crab_pose(&mut self, pos: crate::net::sim::Pos, yaw: i32) {
        self.sim.set_external_crab_pose(pos, yaw);
    }

    /// This peer's id.
    pub fn me(&self) -> PlayerId {
        self.me
    }
}

/// One side of the hash cross-check: a [`Fault::Desync`] iff the two hashes for
/// `tick` disagree. Single definition so the early- and late-arrival sites can't drift.
fn check(tick: u64, peer: PlayerId, local: u64, remote: u64) -> Option<Fault> {
    (local != remote).then_some(Fault::Desync {
        tick,
        peer,
        local_hash: local,
        peer_hash: remote,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    #[test]
    fn single_peer_advances_each_submitted_tick() {
        // With one peer, our own input completes every tick. After the run we should
        // have applied the INPUT_DELAY warmup ticks plus one tick per submit.
        let mut ls = Lockstep::new(7, &ids(1), PlayerId(0));
        let submits = 5u64;
        for _ in 0..submits {
            ls.submit_local_input(Input::from_axes(1.0, 0.0));
            assert!(ls.try_advance().is_empty());
        }
        assert_eq!(ls.sim().tick(), INPUT_DELAY + submits);
    }

    #[test]
    fn stalls_after_warmup_until_remote_input_arrives() {
        let mut a = Lockstep::new(1, &ids(2), PlayerId(0));
        // A submits its own inputs but hears nothing from B. It can run the warmup
        // ticks (neutral input, no peer needed) but then stalls: the first real tick
        // needs B's input too, which never comes.
        for _ in 0..10 {
            a.submit_local_input(Input::default());
            a.try_advance();
        }
        assert_eq!(
            a.sim().tick(),
            INPUT_DELAY,
            "must stall at the first non-warmup tick without the remote peer's input"
        );
    }

    #[test]
    fn two_peers_stay_in_lockstep() {
        // Drive two drivers in tandem, feeding each the other's messages, and assert
        // their confirmed hashes match tick-for-tick (no desync reported).
        let mut a = Lockstep::new(99, &ids(2), PlayerId(0));
        let mut b = Lockstep::new(99, &ids(2), PlayerId(1));
        for t in 0..30u64 {
            let ma = a.submit_local_input(Input::from_axes((t % 3) as f32 - 1.0, 0.5));
            let mb = b.submit_local_input(Input::from_axes(0.0, (t % 2) as f32));
            assert!(a.record_remote(PlayerId(1), mb).is_none());
            assert!(b.record_remote(PlayerId(0), ma).is_none());
            assert!(a.try_advance().is_empty());
            assert!(b.try_advance().is_empty());
        }
        assert_eq!(a.sim().tick(), b.sim().tick());
        assert_eq!(a.sim().state_hash(), b.sim().state_hash());
    }

    #[test]
    fn desync_is_detected() {
        // Feed B a tampered input for A so the two sims diverge; the hash cross-check
        // must catch it rather than letting them silently drift.
        let mut a = Lockstep::new(5, &ids(2), PlayerId(0));
        let mut b = Lockstep::new(5, &ids(2), PlayerId(1));
        let mut saw_desync = false;
        for _ in 0..20u64 {
            let ma = a.submit_local_input(Input::from_axes(1.0, 0.0));
            let mb = b.submit_local_input(Input::from_axes(0.0, 0.0));
            // B receives a CORRUPTED version of A's input (the wire byte flipped).
            let mut tampered = ma;
            tampered.input = Input::from_axes(-1.0, 0.0);
            // A desync can surface either when a hash arrives for an already-applied
            // tick (record_remote) or when applying a tick whose peer hash is pending
            // (try_advance); check both sites for a Desync specifically.
            let is_desync = |f: &Fault| matches!(f, Fault::Desync { .. });
            saw_desync |= a
                .record_remote(PlayerId(1), mb)
                .as_ref()
                .is_some_and(is_desync);
            saw_desync |= b
                .record_remote(PlayerId(0), tampered)
                .as_ref()
                .is_some_and(is_desync);
            saw_desync |= a.try_advance().iter().any(is_desync);
            saw_desync |= b.try_advance().iter().any(is_desync);
        }
        assert!(saw_desync, "diverging inputs must surface a Desync");
    }

    #[test]
    fn out_of_window_input_does_not_grow_buffers() {
        // Inputs outside next_tick()..=+FUTURE_TICK_BOUND must not be buffered (they'd
        // grow `inputs` without bound): an absurd-future tick AND an already-applied
        // (too-old) tick are both dropped; an honest near-future input is kept.
        // Advance past the warmup first so next_tick() > 0 and the too-old case is a
        // genuine already-applied tick, not another huge value.
        let mut ls = Lockstep::new(0, &ids(1), PlayerId(0));
        for _ in 0..(INPUT_DELAY + 3) {
            ls.submit_local_input(Input::default());
            ls.try_advance();
        }
        let next = ls.next_tick();
        assert!(next > 0);
        let drop_cases = [u64::MAX, next - 1]; // implausibly future, and already applied
        let buffered_before = ls.buffered_input_ticks();
        for apply_tick in drop_cases {
            // confirmed: None → never a fault; we're only asserting buffer growth.
            let _ = ls.record_remote(
                PlayerId(1),
                TickMsg {
                    apply_tick,
                    input: Input::default(),
                    confirmed: None,
                },
            );
        }
        assert_eq!(
            ls.buffered_input_ticks(),
            buffered_before,
            "out-of-window inputs (too-future and too-old) must be dropped"
        );
        let _ = ls.record_remote(
            PlayerId(1),
            TickMsg {
                apply_tick: next,
                input: Input::default(),
                confirmed: None,
            },
        );
        assert_eq!(
            ls.buffered_input_ticks(),
            buffered_before + 1,
            "an in-window input must be kept"
        );
    }

    #[test]
    fn unverifiable_old_hash_is_reported_not_dropped() {
        // A peer advertising a hash for a tick we applied long ago (beyond the history
        // window) can't be verified; that must surface as Fault::Unverifiable, never
        // silently accepted — the "no desync goes unnoticed" guarantee.
        let mut ls = Lockstep::new(0, &ids(1), PlayerId(0));
        // Advance well past the history window (single peer auto-advances).
        for _ in 0..(HASH_HISTORY + INPUT_DELAY + 10) {
            ls.submit_local_input(Input::default());
            ls.try_advance();
        }
        let stale_tick = 1; // long since pruned from applied_hashes
        let fault = ls.record_remote(
            PlayerId(0),
            TickMsg {
                apply_tick: ls.next_tick(),
                input: Input::default(),
                confirmed: Some(Confirmed {
                    tick: stale_tick,
                    hash: 0xdead,
                }),
            },
        );
        assert!(
            matches!(fault, Some(Fault::Unverifiable { tick, .. }) if tick == stale_tick),
            "an unverifiable stale hash must be reported as Unverifiable, got {fault:?}"
        );
    }

    impl Lockstep {
        /// Test-only: how many distinct ticks currently have buffered input.
        fn buffered_input_ticks(&self) -> usize {
            self.inputs.len()
        }
    }
}

/// Test-only accessor reachable from the sibling [`crate::net`] desync/invariant suite
/// (rl#63's MP byte-identical guard), which constructs networked `Lockstep`s and must read
/// back whether the crab was handed to external control. `pub(crate)` + `#[cfg(test)]` so it
/// exists only in test builds and never widens the production surface.
#[cfg(test)]
impl Lockstep {
    pub(crate) fn crab_is_external(&self) -> bool {
        self.sim.crab_is_external()
    }
}
