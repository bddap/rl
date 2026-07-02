//! The local client's per-round state on the host-authoritative path.
//!
//! Under host-authoritative state-resync the client does NOT step a sim of its own: the authoritative
//! [`crate::server::Server`] steps the world at the HOST's own pace and emits a [`CoreSnapshot`] per
//! tick, and this type ADOPTS it ([`Lockstep::apply_core_snapshot`]) as the rendered state. Its
//! remaining jobs are: schedule the LOCAL player's input UP to the server
//! ([`Lockstep::submit_local_input`]), adopt the server's snapshot DOWN, and — on a remote client —
//! re-predict the local avatar over its still-unconsumed inputs
//! ([`Lockstep::reconcile_local_prediction`]) so WASD responds at input latency, not round-trip. It
//! owns the client's sim only as the surface the snapshot writes into and the render reads from.
//! Inputs cross the wire; state crosses it as full snapshots.

use std::collections::BTreeMap;

use crate::sim::{Input, PlayerId, Sim};
use crate::snapshot::CoreSnapshot;

/// What one client publishes UP to the server for a single tick: one input in its issue
/// sequence. The HOST's own input for tick T is applied exactly at tick T (it paces the match);
/// a REMOTE client's inputs are consumed by the server in issue order as they arrive — typically
/// a transit-lag of ticks later — with the consumption reported back per snapshot
/// ([`CoreSnapshot::input_next`]) so the client's prediction replays exactly the unconsumed tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickMsg {
    /// This input's position in the sender's issue sequence (one per wall-clock tick, strictly
    /// increasing). NOT a promise of the tick it applies at — the server never waits on a remote
    /// (rl#193/#194/#195).
    pub issue_tick: u64,
    pub input: Input,
}

/// A peer [`TickMsg`] tagged with the sender's already-resolved [`PlayerId`]. The
/// id is mapped from the QUIC-authenticated endpoint id via the frozen
/// participant set (never read from the message body — see [`crate::transport::FromPeer`]),
/// so the server can trust it as the input's true author.
#[derive(Debug, Clone, Copy)]
pub struct PeerMsg {
    pub pid: PlayerId,
    pub msg: TickMsg,
}

/// The local client's per-round state: schedules the local input UP, adopts the authoritative
/// snapshot DOWN, and predicts/reconciles the local avatar. Owns the client's sim as the surface the
/// snapshot writes into — it never steps that sim itself (the server is authoritative).
pub struct Lockstep {
    sim: Sim,
    me: PlayerId,
    /// The participant set at ROUND START (sorted, deduped, including `me`) — what the authoritative
    /// server is built over ([`Self::peers`]). Read only at round start; the live roster of record is
    /// the server's, carried on every adopted [`CoreSnapshot`] (a mid-game join grows the SIM's player
    /// set, never this).
    peers: Vec<PlayerId>,
    /// The local player's own submitted inputs, keyed by issue tick — kept for
    /// [`Self::reconcile_local_prediction`] to replay the still-unconsumed window after adopting a
    /// snapshot. Pruned by [`Self::apply_core_snapshot`] to the inputs the snapshot's
    /// [`CoreSnapshot::input_next`] watermark says the server has not yet consumed, so it can't
    /// grow unbounded while the server keeps serving.
    inputs: BTreeMap<u64, Input>,
    /// The last adopted snapshot's per-player input watermarks — our own entry drives the prune
    /// above, and [`Self::core_snapshot`] re-stamps the whole map so the client's re-emit mirrors
    /// the server's bytes exactly.
    input_next: BTreeMap<PlayerId, u64>,
    /// The tick the NEXT local input will be issued as. Incremented once per
    /// [`Self::submit_local_input`]. Starts at 0.
    next_issue_tick: u64,
    /// The next tick to APPLY (0-based; equals the count already applied). Advanced only by
    /// [`Self::apply_core_snapshot`] — the client no longer steps, so adopting the server's snapshot
    /// is the one place the apply cursor moves. Kept explicitly so a JOINER ([`Self::join_at`]) can
    /// enter at the live tick.
    next_apply_tick: u64,
}

impl Lockstep {
    /// Start a round. `seed` is the shared match seed (identical on the server); `peers` is the full
    /// participant set; `me` is this client's id and must be in it.
    pub fn new(seed: u64, peers: &[PlayerId], me: PlayerId) -> Self {
        let mut peers = peers.to_vec();
        peers.sort();
        peers.dedup();
        debug_assert!(peers.contains(&me), "local player must be in the peer set");
        Self {
            sim: Sim::new(seed, &peers),
            me,
            peers,
            inputs: BTreeMap::new(),
            input_next: BTreeMap::new(),
            next_issue_tick: 0,
            next_apply_tick: 0,
        }
    }

    /// Start a client JOINING an in-progress match at `at_tick`. The joiner does NOT replay ticks
    /// `[0, at_tick)`: it boots host-authoritatively from the server's snapshot for the live
    /// round, so this only seeds the placeholder cursors + roster the adopted snapshot then
    /// supersedes. Its apply + issue cursors begin at `at_tick`.
    pub fn join_at(seed: u64, roster: &[PlayerId], me: PlayerId, at_tick: u64) -> Self {
        let mut roster_set = roster.to_vec();
        roster_set.sort();
        roster_set.dedup();
        debug_assert!(
            roster_set.contains(&me),
            "joining player must be in the new roster"
        );
        Self {
            sim: Sim::new(seed, &roster_set),
            me,
            peers: roster_set,
            inputs: BTreeMap::new(),
            input_next: BTreeMap::new(),
            next_issue_tick: at_tick,
            next_apply_tick: at_tick,
        }
    }

    /// The next tick to be applied (0-based; equals the count already applied).
    pub fn next_tick(&self) -> u64 {
        self.next_apply_tick
    }

    /// Submit THIS client's input as the next issue in its sequence and get the message to ship UP
    /// to the server (which consumes it — immediately for the host's own input, in arrival order
    /// for a remote's).
    ///
    /// Call exactly once per tick — the issue cursor advances by one each call, so a missed or
    /// doubled call would gap or collide the input stream. The input is also recorded locally so
    /// [`Self::reconcile_local_prediction`] can replay the still-unconsumed window after an adopt.
    pub fn submit_local_input(&mut self, input: Input) -> TickMsg {
        let issue_tick = self.next_issue_tick;
        self.next_issue_tick += 1;
        self.inputs.insert(issue_tick, input);
        TickMsg { issue_tick, input }
    }

    /// Read-only sim view for rendering/inspection.
    pub fn sim(&self) -> &Sim {
        &self.sim
    }

    /// The local client's read seam onto authoritative game state ([`crate::snapshot`]). SP
    /// funnels through the SAME serialized [`CoreSnapshot`] a wire client consumes — one
    /// state-read path, no by-reference-in-SP fork ([[sp-is-mp-special-case]]). Byte-identical to
    /// the server's emitted snapshot (the carried sim state plus the stashed `input_next`
    /// watermarks); the round-trip through bytes just proves the seam end to end (the copy is
    /// ~hundreds of bytes/tick).
    pub fn core_snapshot(&self) -> CoreSnapshot {
        let mut snap = self.sim.core_snapshot();
        snap.input_next = self.input_next.clone();
        let bytes = snap.to_bytes();
        CoreSnapshot::from_bytes(&bytes).expect("a freshly-built snapshot must round-trip")
    }

    /// Adopt the authoritative server's [`CoreSnapshot`] as this client's rendered state: the
    /// client never steps its own sim — the server steps and emits a snapshot per tick, and this
    /// overwrites the carried game state ([`Sim::apply_core_snapshot`]).
    /// `scene::apply_transforms` then reads [`sim`](Lockstep::sim) exactly as before, tweening from
    /// `prev`. SP funnels the server's bytes through the SAME decode a wire client does
    /// ([[sp-is-mp-special-case]]), so there is no by-reference SP fork.
    ///
    /// Also stashes the snapshot's [`CoreSnapshot::input_next`] watermarks and prunes our own
    /// submitted inputs to the still-unconsumed window `issue_tick >= input_next[me]` (we file
    /// inputs UP but never step to consume them, so without the prune the map grows unbounded).
    /// The survivors are exactly what [`reconcile_local_prediction`] replays on the local avatar —
    /// the server consumes a remote's inputs as they arrive, so its watermark (not the snapshot
    /// tick) is the one true consumption cursor.
    pub fn apply_core_snapshot(&mut self, snapshot: CoreSnapshot) {
        let applied_tick = snapshot.tick;
        let next_unconsumed = snapshot.input_next.get(&self.me).copied().unwrap_or(0);
        self.input_next = snapshot.input_next.clone();
        self.sim.apply_core_snapshot(snapshot);
        // Track the apply cursor to the snapshot we just adopted — the client no longer steps, so this
        // is the one place it moves. `snapshot.tick` is the POST-step tick count (the sim is now AT
        // it), so the next tick to apply equals it. Keeps `next_tick()` honest (it would otherwise sit
        // at 0 forever while the sim advances, mis-stamping the `issue_tick` telemetry).
        self.next_apply_tick = applied_tick;
        // Prune to the still-unconsumed window — see the doc above.
        self.inputs = self.inputs.split_off(&next_unconsumed);
    }

    /// THE remote-client adopt policy — the one shared answer to "which drained snapshots does a
    /// client apply, in what order?" for both remote clients (the windowed driver's `RemoteAdopt`
    /// arm and headless `game net`'s client arm): apply EVERY snapshot, in ARRIVAL order, with NO
    /// tick gate. The reliable ordered per-peer stream delivers snapshots in the host's send
    /// (= step) order, so arrival order IS authoritative — a `tick <=`/`max` gate or a
    /// sort-by-tick would add nothing but a place to go wrong (and historically froze clients
    /// across a host RESTART, back when a restart rewound the tick; ticks are monotone across a
    /// restart now — rl#204 — so there is nothing to gate). Each snapshot is a FULL
    /// state overwrite, so applying intermediates is cheap (~hundreds of bytes/tick) and keeps
    /// per-adopt observers exact.
    ///
    /// `on_adopt` runs after each apply (the `--hash-log` observer; pass `|_| ()` to skip). Returns
    /// how many snapshots were adopted; callers gate POST-adopt work
    /// ([`reconcile_local_prediction`](Lockstep::reconcile_local_prediction)) on it having been at
    /// least one. PRE-adopt work (the driver's `prev` interpolation-source refresh) must instead be
    /// gated on the batch being nonempty BEFORE this call — the count comes back too late.
    ///
    /// Scope: this polices a DRAINED REMOTE batch. Host/solo self-mirrors of the single snapshot
    /// their own in-process server just stepped stay on raw
    /// [`apply_core_snapshot`](Lockstep::apply_core_snapshot) — no batch, no ordering question.
    pub fn adopt_snapshots(
        &mut self,
        snapshots: impl IntoIterator<Item = CoreSnapshot>,
        mut on_adopt: impl FnMut(&Self),
    ) -> usize {
        let mut adopted = 0;
        for snap in snapshots {
            self.apply_core_snapshot(snap);
            adopted += 1;
            on_adopt(self);
        }
        adopted
    }

    /// Re-predict the LOCAL player over its still-unconsumed inputs after adopting an authoritative
    /// snapshot, so a remote client's own avatar responds at input latency instead of round-trip
    /// latency. Call ONCE, right after
    /// [`apply_core_snapshot`](Lockstep::apply_core_snapshot), on the remote-adopt arm only: that call
    /// re-seats every entity to the host's authoritative state (our avatar included, at its
    /// round-trip-old position) and prunes `self.inputs` to the inputs the snapshot's watermark
    /// says the server hasn't consumed; this replays those, in issue order, on the local player
    /// alone — the same sequence the server will consume in order as they arrive. (Not exact
    /// under starvation: a stream gap makes the server interleave HOLDS of our last move axes,
    /// which the replay doesn't include — the residue is corrected at the next adopt.)
    /// Remote players and the crab stay authoritative and are interpolated, never predicted
    /// ([[render-matches-physics]] — the crab is the host's, not guessed). MUST NOT run on the
    /// server-authoritative (solo/host) arm: there the host already applied the local input in the
    /// same tick it emitted the snapshot, so replaying it would double-apply and run the avatar ahead.
    pub(crate) fn reconcile_local_prediction(&mut self) {
        let me = self.me;
        // Ascending `BTreeMap` order = issue order, which the facing-relative mover requires
        // (each tick's yaw feeds the next tick's translation).
        for &inp in self.inputs.values() {
            self.sim.predict_player(me, inp);
        }
    }

    /// Drive the crab's ground position + yaw + physics digest from the real NN crab body — forwards
    /// to [`Sim::set_external_crab_pose`], the ONLY way the crab moves. Used by the headless
    /// screenshot/probe seed path and the round-setup crab-pose seed.
    pub fn set_external_crab_pose(&mut self, pos: crate::sim::Pos, yaw: i32, phys_digest: u64) {
        self.sim.set_external_crab_pose(pos, yaw, phys_digest);
    }

    /// This client's id.
    pub fn me(&self) -> PlayerId {
        self.me
    }

    /// The ROUND-START participant set (sorted, incl. `me`) — the set the authoritative server is
    /// built over (solo ⇒ just `me`), so the client and its server agree on the roster by
    /// construction. Read only at round start; the live roster is the sim's (snapshot-carried).
    pub fn peers(&self) -> &[PlayerId] {
        &self.peers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::Server;

    fn ids(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    /// The reconciliation property under host pacing: a REMOTE client whose inputs reach the host
    /// a few ticks late (and whose snapshots come back just as late) must, after adopt +
    /// reconcile, render its own avatar exactly where replaying ALL of its issued inputs lands —
    /// input latency fully hidden, no residual snap. And once the wire drains (the host catches up
    /// on the client's queued inputs), the client's avatar must sit exactly where the HOST has it.
    #[test]
    fn local_prediction_hides_input_transit_latency() {
        let host = PlayerId(0);
        let me = PlayerId(1);
        let roster = ids(2);
        let mut server = Server::new(host, &roster, Sim::new(42, &roster));
        let mut host_sched = Lockstep::new(42, &roster, host); // the host's own input scheduler
        let mut client = Lockstep::new(42, &roster, me);
        // The pure-replay reference: our avatar driven by every issued input, in order — what a
        // zero-latency link would render. Prediction must match it exactly.
        let mut reference = Lockstep::new(42, &roster, me);

        const LATENCY: usize = 4; // ticks of one-way transit, both directions
        let mut wire_up: std::collections::VecDeque<TickMsg> = Default::default();
        let mut wire_down: std::collections::VecDeque<CoreSnapshot> = Default::default();

        // Stay within STARTUP_GRACE_TICKS (30): the crab stays disarmed (no grabs) and the round
        // stays Ongoing — pure movement, so the convergence is EXACT.
        const FRAMES: u64 = 20;
        for f in 0..FRAMES {
            // Vary move AND turn every tick (never neutral), so a dropped or mis-ordered replay
            // moves the avatar somewhere the reference isn't.
            let inp = Input::new(
                ((f % 3) as f32 - 1.0) * 0.7,
                ((f % 5) as f32) / 4.0 - 0.5,
                ((f % 4) as f32 - 1.5) / 1.5,
                0,
            );
            wire_up.push_back(client.submit_local_input(inp));
            reference.sim.predict_player(me, inp); // zero-latency ideal, applied at issue
            if wire_up.len() > LATENCY {
                let msg = wire_up.pop_front().expect("len checked");
                server.record_remote(me, msg);
            }
            server.advance(host_sched.submit_local_input(Input::default()));
            while server.next_tick_ready() {
                let bytes = server.step_next(None).snapshot;
                wire_down.push_back(CoreSnapshot::from_bytes(&bytes).expect("snapshot decodes"));
            }
            if wire_down.len() > LATENCY {
                let snap = wire_down.pop_front().expect("len checked");
                client.apply_core_snapshot(snap);
                client.reconcile_local_prediction();
                let cp = client.sim().player(me).expect("local player present");
                let rp = reference.sim().player(me).expect("local player present");
                assert_eq!(
                    (cp.pos(), cp.yaw()),
                    (rp.pos(), rp.yaw()),
                    "frame {f}: reconciled avatar == all issued inputs replayed \
                     (transit latency fully hidden, no snap)"
                );
            }
        }

        // Drain the wire: no new client inputs; the host consumes the queued tail, and the client
        // adopts everything. With nothing left to replay, the client's avatar must sit EXACTLY on
        // the host's authoritative state — prediction converged, not diverged.
        for _ in 0..(2 * LATENCY as u64 + 2) {
            if let Some(msg) = wire_up.pop_front() {
                server.record_remote(me, msg);
            }
            server.advance(host_sched.submit_local_input(Input::default()));
            while server.next_tick_ready() {
                let bytes = server.step_next(None).snapshot;
                wire_down.push_back(CoreSnapshot::from_bytes(&bytes).expect("snapshot decodes"));
            }
            let mut adopted = false;
            while let Some(snap) = wire_down.pop_front() {
                client.apply_core_snapshot(snap);
                adopted = true;
            }
            // Reconcile only right after an adopt — replaying onto an already-predicted state
            // would double-apply (the same rule the driver's RemoteAdopt arm follows).
            if adopted {
                client.reconcile_local_prediction();
            }
        }
        assert!(
            client.inputs.is_empty(),
            "every issued input was consumed by the host and pruned by its watermark"
        );
        let cp = client.sim().player(me).expect("local player present");
        let hp = server.sim().player(me).expect("local player present");
        assert_eq!(
            (cp.pos(), cp.yaw()),
            (hp.pos(), hp.yaw()),
            "with the wire drained, the client sits exactly on the authoritative state"
        );
    }

    /// A remote client follows a host RESTART (rl#204): the restart is a state-reset at the
    /// current tick, so the wire carries a gap-free MONOTONE tick stream whose restart-tick
    /// snapshot holds the spawn-state world. The client adopts every arrival in order and ends
    /// on the fresh round — no gate, no freeze, no tick regression anywhere.
    #[test]
    fn adopt_snapshots_follows_a_host_restart() {
        use crate::sim::buttons;

        let me = PlayerId(0);
        let roster = ids(1);
        let mut client = Lockstep::new(7, &roster, me);
        let spawn = Sim::new(7, &roster).player(me).expect("rostered").pos();

        // One host: walk away from spawn for 5 ticks, press RESTART, then one neutral tick.
        let mut sched = Lockstep::new(7, &roster, me);
        let mut host = Server::new(me, &roster, Sim::new(7, &roster));
        let mut arrivals = Vec::new();
        for t in 0..7u64 {
            let btns = if t == 5 { buttons::RESTART } else { 0 };
            let input = Input::new(0.0, if t < 5 { 1.0 } else { 0.0 }, 0.0, btns);
            host.advance(sched.submit_local_input(input));
            while host.next_tick_ready() {
                let bytes = host.step_next(None).snapshot;
                arrivals.push(CoreSnapshot::from_bytes(&bytes).expect("snapshot decodes"));
            }
        }

        let mut seen = Vec::new();
        let adopted = client.adopt_snapshots(arrivals, |c| seen.push(c.sim().tick()));
        assert_eq!(adopted, 7, "every arrival adopted — none gated away");
        assert_eq!(
            seen,
            [1, 2, 3, 4, 5, 6, 7],
            "the wire tick stream is monotone and gap-free across the restart"
        );
        assert_eq!(
            client.sim().player(me).expect("rostered").pos(),
            spawn,
            "the client ends on the restarted round's spawn state"
        );
    }

    /// The prune is watermark-driven, not tick-driven: a snapshot whose tick has advanced far
    /// past our issue cursor but whose watermark says nothing of ours was consumed must keep the
    /// whole in-flight window for replay.
    #[test]
    fn prune_follows_the_watermark_not_the_tick() {
        let me = PlayerId(1);
        let roster = ids(2);
        let mut client = Lockstep::new(9, &roster, me);
        for _ in 0..3 {
            let _ = client.submit_local_input(Input::from_axes(1.0, 0.0));
        }
        // A tick-10 snapshot that consumed NONE of our inputs (watermark 0 — e.g. they are all
        // still in flight).
        let mut snap = Sim::new(9, &roster).core_snapshot();
        snap.tick = 10;
        snap.input_next = BTreeMap::from([(PlayerId(0), 10), (me, 0)]);
        client.apply_core_snapshot(snap);
        assert_eq!(
            client.inputs.len(),
            3,
            "nothing consumed ⇒ nothing pruned, despite the far-ahead tick"
        );
        // Now one that consumed the first two.
        let mut snap = Sim::new(9, &roster).core_snapshot();
        snap.tick = 11;
        snap.input_next = BTreeMap::from([(PlayerId(0), 11), (me, 2)]);
        client.apply_core_snapshot(snap);
        assert_eq!(
            client.inputs.keys().copied().collect::<Vec<_>>(),
            vec![2],
            "consumed inputs pruned, the unconsumed tail kept"
        );
    }
}
