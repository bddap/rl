//! The local client's per-round state on the host-authoritative path (GCR MP rewrite, bddap/rl#151).
//!
//! Under host-authoritative state-resync the client does NOT step a sim of its own: the authoritative
//! [`crate::server::Server`] steps the world and emits a [`CoreSnapshot`] per tick, and this type
//! ADOPTS it ([`Lockstep::apply_core_snapshot`]) as the rendered state. Its remaining jobs are:
//! schedule the LOCAL player's input UP to the server ([`Lockstep::submit_local_input`]), adopt the
//! server's snapshot DOWN, and — on a remote client — re-predict the local avatar over its
//! still-in-flight inputs ([`Lockstep::reconcile_local_prediction`]) so WASD responds at input
//! latency, not round-trip. It owns the client's sim only as the surface the snapshot writes into and
//! the render reads from. Inputs cross the wire; state crosses it as full snapshots.

use std::collections::BTreeMap;

use crate::sim::{Input, PlayerId, Sim};
use crate::snapshot::CoreSnapshot;

/// What one client publishes UP to the server for a single tick: the input it wants applied at
/// `apply_tick`. The server records it into its ledger and, once every rostered client's input for
/// the tick is in, steps its authoritative sim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickMsg {
    /// The tick at which `input` should be applied (the client's current issue cursor — there is no
    /// input-delay barrier under host-authority; the server applies what it has and steps).
    pub apply_tick: u64,
    pub input: Input,
}

/// A peer [`TickMsg`] tagged with the sender's already-resolved [`PlayerId`]. The
/// id is mapped from the QUIC-authenticated endpoint id via the frozen
/// participant set (never read from the message body — see [`crate::transport::FromPeer`]),
/// so the lockstep driver can trust it as the input's true author.
#[derive(Debug, Clone, Copy)]
pub struct PeerMsg {
    pub pid: PlayerId,
    pub msg: TickMsg,
}

/// The local client's per-round state: schedules the local input UP, adopts the authoritative
/// snapshot DOWN, and predicts/reconciles the local avatar. Owns the client's sim as the surface the
/// snapshot writes into — it never steps that sim itself (the server is authoritative, rl#151).
pub struct Lockstep {
    sim: Sim,
    me: PlayerId,
    /// The participant set at ROUND START (sorted, deduped, including `me`) — what the authoritative
    /// server is built over ([`Self::peers`]). Read only at round start; the live roster of record is
    /// the server's, carried on every adopted [`CoreSnapshot`] (a mid-game join grows the SIM's player
    /// set, never this).
    peers: Vec<PlayerId>,
    /// The local player's own submitted inputs, keyed by `apply_tick` — kept for
    /// [`Self::reconcile_local_prediction`] to replay the still-in-flight window after adopting a
    /// snapshot. Pruned by [`Self::apply_core_snapshot`] to the inputs the snapshot doesn't yet
    /// reflect, so it can't grow unbounded (the client never steps to consume it).
    inputs: BTreeMap<u64, BTreeMap<PlayerId, Input>>,
    /// The tick the NEXT local input will apply at. Incremented once per [`Self::submit_local_input`].
    /// Starts at 0 — the first input applies at tick 0 (no input-delay barrier under host-authority).
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
            next_issue_tick: 0,
            next_apply_tick: 0,
        }
    }

    /// Start a client JOINING an in-progress match at `at_tick` (GCR MP Stage 3). The joiner does NOT
    /// replay ticks `[0, at_tick)`: it boots host-authoritatively from the server's snapshot for the
    /// live round (the incr-4 mid-game-join fix), so this only seeds the placeholder cursors + roster
    /// the adopted snapshot then supersedes. Its apply + issue cursors begin at `at_tick`.
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
            next_issue_tick: at_tick,
            next_apply_tick: at_tick,
        }
    }

    /// The next tick to be applied (0-based; equals the count already applied).
    pub fn next_tick(&self) -> u64 {
        self.next_apply_tick
    }

    /// Submit THIS client's input for the next issuing tick and get the message to ship UP to the
    /// server (which records it and steps when every rostered client's input for the tick is in).
    ///
    /// Call exactly once per tick — the scheduled tick advances by one each call, so a missed or
    /// doubled call would gap or collide the input stream. The input is also recorded locally so
    /// [`Self::reconcile_local_prediction`] can replay the still-in-flight window after an adopt.
    pub fn submit_local_input(&mut self, input: Input) -> TickMsg {
        let apply_tick = self.next_issue_tick;
        self.next_issue_tick += 1;
        self.inputs
            .entry(apply_tick)
            .or_default()
            .insert(self.me, input);
        TickMsg { apply_tick, input }
    }

    /// Read-only sim view for rendering/inspection.
    pub fn sim(&self) -> &Sim {
        &self.sim
    }

    /// The local client's read seam onto authoritative game state (bddap/rl#151 increment 0,
    /// [`crate::snapshot`]). SP funnels through the SAME serialized [`CoreSnapshot`] a wire client
    /// consumes — built, encoded, and decoded here so SP and MP share ONE state-read path with no
    /// by-reference-in-SP fork ([[sp-is-mp-special-case]], [[silent-fallback-antipattern]]).
    /// Byte-identical to reading [`sim`](Lockstep::sim) directly; the round-trip through bytes just
    /// proves the seam end to end (the copy is ~hundreds of bytes/tick).
    pub fn core_snapshot(&self) -> CoreSnapshot {
        let bytes = self.sim.core_snapshot().to_bytes();
        CoreSnapshot::from_bytes(&bytes).expect("a freshly-built snapshot must round-trip")
    }

    /// Adopt the authoritative server's [`CoreSnapshot`] as this client's rendered state (rl#151
    /// increment 1): the client never steps its own sim — the server steps and emits a snapshot per
    /// tick, and this overwrites the carried game state ([`Sim::apply_core_snapshot`]).
    /// `scene::apply_transforms` then reads [`sim`](Lockstep::sim) exactly as before, tweening from
    /// `prev`. SP funnels the server's bytes through the SAME decode a wire client does
    /// ([[sp-is-mp-special-case]]), so there is no by-reference SP fork.
    ///
    /// Also drops our own now-stale submitted inputs: we still call
    /// [`submit_local_input`](Lockstep::submit_local_input) to file each tick's input UP to the server,
    /// but we never step to consume `self.inputs`, so
    /// without this prune it would grow unbounded. The prune keeps the still-in-flight window
    /// `apply_tick >= snapshot.tick` — the boundary tick is RETAINED, since a tick-T snapshot reflects
    /// inputs only up through `apply_tick == T-1`, so `apply_tick == T` hasn't landed yet and is
    /// exactly what [`reconcile_local_prediction`] replays.
    pub fn apply_core_snapshot(&mut self, snapshot: CoreSnapshot) {
        let applied_tick = snapshot.tick;
        self.sim.apply_core_snapshot(snapshot);
        // Track the apply cursor to the snapshot we just adopted — the client no longer steps, so this
        // is the one place it moves. `snapshot.tick` is the POST-step tick count (the sim is now AT
        // it), so the next tick to apply equals it. Keeps `next_tick()` honest (it would otherwise sit
        // at 0 forever while the sim advances, mis-stamping the `issue_tick` telemetry).
        self.next_apply_tick = applied_tick;
        // Drop submitted inputs the snapshot ALREADY reflects, keeping only the still-in-flight window
        // `apply_tick >= applied_tick`. The boundary tick is deliberately KEPT: a snapshot at tick T is
        // the POST-step state that consumed inputs up through `apply_tick == T-1`, so the input filed
        // for `apply_tick == T` has NOT reached this state yet — it is in flight. Those survivors are
        // exactly the set `reconcile_local_prediction` replays on the local avatar (and, on the
        // server-authoritative arm that never reconciles, a couple of harmless entries the next
        // snapshot prunes). Without a prune the map would grow unbounded.
        self.inputs = self.inputs.split_off(&applied_tick);
    }

    /// THE remote-client adopt policy — the one shared answer to "which drained snapshots does a
    /// client apply, in what order?" for both remote clients (the windowed driver's `RemoteAdopt`
    /// arm and headless `game net`'s client arm): apply EVERY snapshot, in ARRIVAL order, with NO
    /// tick gate. The reliable ordered per-peer stream delivers snapshots in the host's send
    /// (= step) order, so arrival order IS authoritative — including the tick-0 snapshots a host
    /// RESTART rebroadcasts, which a `tick <=`/`max` gate (or a sort-by-tick) would silently
    /// reject/reorder, freezing the client on stale pre-restart state. Each snapshot is a FULL
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

    /// Re-predict the LOCAL player over its still-in-flight inputs after adopting an authoritative
    /// snapshot, so a remote client's own avatar responds at input latency instead of round-trip
    /// latency (rl#151 incr 3). Call ONCE, right after
    /// [`apply_core_snapshot`](Lockstep::apply_core_snapshot), on the remote-adopt arm only: that call
    /// re-seats every entity to the host's authoritative state (our avatar included, at its
    /// round-trip-old position) and prunes `self.inputs` to the inputs the snapshot doesn't yet
    /// reflect (`apply_tick >= snapshot.tick`); this replays those, in tick order, on the local player
    /// alone. Remote players and the crab stay authoritative and are interpolated, never predicted
    /// ([[render-matches-physics]] — the crab is the host's, not guessed). MUST NOT run on the
    /// server-authoritative (solo/host) arm: there the host already applied the local input in the
    /// same tick it emitted the snapshot, so replaying it would double-apply and run the avatar ahead.
    pub(crate) fn reconcile_local_prediction(&mut self) {
        let me = self.me;
        // `self.inputs` on a remote-adopt client holds ONLY our own inputs, so this is exactly the
        // local in-flight window. Ascending `BTreeMap` order = tick order, which the facing-relative
        // mover requires (each tick's yaw feeds the next tick's translation).
        for inputs in self.inputs.values() {
            if let Some(&inp) = inputs.get(&me) {
                self.sim.predict_player(me, inp);
            }
        }
    }

    /// Drive the crab's ground position + yaw + physics digest from the real NN crab body — forwards
    /// to [`Sim::set_external_crab_pose`], the ONLY way the crab moves (rl#114). Used by the headless
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

    /// The reconciliation property (rl#151 incr 3): a remote-adopt client that adopts the host's
    /// snapshot several ticks LATE, then reconciles, must render its own avatar exactly where a
    /// fully-caught-up host has it — hiding the round-trip lag with no residual snap. The
    /// authoritative reference is a real [`Server`] (the one stepper); the client only ever
    /// `apply_core_snapshot`s + `reconcile_local_prediction`s, never stepping a sim of its own.
    #[test]
    fn local_prediction_hides_round_trip_latency() {
        let me = PlayerId(0);
        let roster = ids(1);
        // The authoritative reference: a solo server, always fully caught up.
        let mut sched = Lockstep::new(42, &roster, me); // input scheduler for the host's own client
        let mut server = Server::new(&roster, Sim::new(42, &roster));
        // The remote client: files inputs UP and adopts snapshots DOWN, never steps its own sim.
        let mut client = Lockstep::new(42, &roster, me);
        const LATENCY: usize = 4; // snapshots the client trails the host by
        let mut wire: std::collections::VecDeque<CoreSnapshot> = std::collections::VecDeque::new();

        // Stay within STARTUP_GRACE_TICKS (30): the tick reaches only FRAMES, so the
        // crab stays disarmed (no grabs) and the round stays Ongoing — pure movement, so the
        // convergence is EXACT.
        const FRAMES: u64 = 24;
        for f in 0..FRAMES {
            // Vary move AND turn every tick (never neutral), so a dropped or mis-ordered replay moves
            // the avatar somewhere the host isn't.
            let inp = Input::new(
                ((f % 3) as f32 - 1.0) * 0.7,
                ((f % 5) as f32) / 4.0 - 0.5,
                ((f % 4) as f32 - 1.5) / 1.5,
                0,
            );
            client.submit_local_input(inp); // filed UP; the client never advances
            let msg = sched.submit_local_input(inp);
            let sets = server.record(me, msg);
            server.enqueue_for_step(&sets);
            while server.next_tick_ready() {
                let _ = server.step_next(None);
            }
            // Ship the host's NEWEST authoritative state — one snapshot per frame (latest-wins), the
            // full state the client adopts. Byte-identical to what the server broadcasts.
            wire.push_back(server.sim().core_snapshot());

            if wire.len() > LATENCY {
                let delayed = wire.pop_front().expect("len just checked > LATENCY");
                client.apply_core_snapshot(delayed);
                client.reconcile_local_prediction();
                let cp = client.sim().player(me).expect("local player present");
                let hp = server.sim().player(me).expect("local player present");
                assert_eq!(
                    (cp.pos(), cp.yaw()),
                    (hp.pos(), hp.yaw()),
                    "frame {f}: reconciled local avatar must match the caught-up host \
                     (latency fully hidden, no snap)"
                );
            }
        }
    }

    /// The restart-freeze regression ([`Lockstep::adopt_snapshots`]): a host RESTART rebroadcasts
    /// from tick 0, so the client's drained batch contains a tick REGRESSION. The policy must adopt
    /// it (arrival order is authoritative) — the old per-caller `tick <=` gate rejected it and froze
    /// the client on stale pre-restart state forever.
    #[test]
    fn adopt_snapshots_follows_a_host_restart_tick_regression() {
        let me = PlayerId(0);
        let roster = ids(1);
        let mut client = Lockstep::new(7, &roster, me);

        // Pre-restart host: stepped several ticks.
        let mut sched = Lockstep::new(7, &roster, me);
        let mut old_host = Server::new(&roster, Sim::new(7, &roster));
        let mut arrivals = Vec::new();
        for _ in 0..5 {
            let sets = old_host.record(me, sched.submit_local_input(Input::from_axes(1.0, 0.0)));
            old_host.enqueue_for_step(&sets);
            while old_host.next_tick_ready() {
                let _ = old_host.step_next(None);
            }
            arrivals.push(old_host.sim().core_snapshot());
        }
        // The host restarts: a fresh sim rebroadcasts from tick 0 — a regression on the wire.
        let mut sched2 = Lockstep::new(7, &roster, me);
        let mut new_host = Server::new(&roster, Sim::new(7, &roster));
        let sets = new_host.record(me, sched2.submit_local_input(Input::from_axes(0.0, 1.0)));
        new_host.enqueue_for_step(&sets);
        while new_host.next_tick_ready() {
            let _ = new_host.step_next(None);
        }
        arrivals.push(new_host.sim().core_snapshot());

        let mut seen = Vec::new();
        let adopted = client.adopt_snapshots(arrivals, |c| seen.push(c.sim().tick()));
        assert_eq!(adopted, 6, "every arrival adopted — none gated away");
        // Ticks are POST-step counts: five pre-restart arrivals (1..=5), then the restarted
        // host's tick-1 — the regression a `tick <=` gate would reject, freezing the client at 5.
        assert_eq!(
            seen,
            [1, 2, 3, 4, 5, 1],
            "adopted in arrival order, regression included"
        );
        assert_eq!(
            client.sim().tick(),
            1,
            "the client ends on the post-restart state, not frozen pre-restart"
        );
    }
}
