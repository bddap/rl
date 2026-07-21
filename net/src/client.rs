use std::collections::BTreeMap;

use crab_world::vehicle::{PilotCommand, VehicleKind};

use crate::sim::{Input, PlayerId, Sim};
use crate::snapshot::CoreSnapshot;

/// What one client publishes UP to the server for a single tick: one input in its issue
/// sequence. The HOST's own input for tick T is applied exactly at tick T (it paces the match);
/// a REMOTE client's inputs are consumed by the server in issue order as they arrive — typically
/// a transit-lag of ticks later — with the consumption reported back per snapshot
/// ([`CoreSnapshot::input_next`]) so the client's prediction replays exactly the unconsumed tail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickMsg {
    /// This input's position in the sender's issue sequence (one per wall-clock tick, strictly
    /// increasing). NOT a promise of the tick it applies at — the server never waits on a remote
    /// (rl#193/#194/#195).
    pub issue_tick: u64,
    pub input: Input,
    pub pilot: Option<PilotIntent>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PilotIntent {
    pub kind: VehicleKind,
    pub throttle_trim: f32,
    pub thrust: [f32; 3],
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub match_velocity: bool,
}

impl PilotIntent {
    /// `boarding` is host-authored from the authoritative sim (never the wire): where the
    /// pilot's walker is, for the craft to materialise at on the spawn edge (rl#258).
    pub fn to_command(&self, boarding: crab_world::vehicle::Boarding) -> PilotCommand {
        let s = |v: f32| {
            if v.is_finite() {
                v.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        };
        PilotCommand {
            kind: self.kind,
            boarding,
            throttle_trim: s(self.throttle_trim),
            thrust: bevy::math::Vec3::new(s(self.thrust[0]), s(self.thrust[1]), s(self.thrust[2])),
            pitch: s(self.pitch),
            roll: s(self.roll),
            yaw: s(self.yaw),
            match_velocity: self.match_velocity,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PeerMsg {
    pub pid: PlayerId,
    pub msg: TickMsg,
}

pub struct ClientSim {
    sim: Sim,
    me: PlayerId,
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
    next_apply_tick: u64,
}

impl ClientSim {
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

    pub fn next_tick(&self) -> u64 {
        self.next_apply_tick
    }

    pub fn submit_local_input(&mut self, input: Input, pilot: Option<PilotIntent>) -> TickMsg {
        let issue_tick = self.next_issue_tick;
        self.next_issue_tick += 1;
        self.inputs.insert(issue_tick, input);
        TickMsg {
            issue_tick,
            input,
            pilot,
        }
    }

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

    pub fn apply_core_snapshot(&mut self, snapshot: CoreSnapshot) {
        let applied_tick = snapshot.tick;
        let next_unconsumed = snapshot.input_next.get(&self.me).copied().unwrap_or(0);
        self.input_next = snapshot.input_next.clone();
        self.sim.apply_core_snapshot(snapshot);
        self.next_apply_tick = applied_tick;
        // Prune to the still-unconsumed window — see the doc above.
        self.inputs = self.inputs.split_off(&next_unconsumed);
    }

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

    // Live only via the render driver outside tests — dead render-off (rl#248).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn reconcile_local_prediction(&mut self) {
        let me = self.me;
        // Ascending `BTreeMap` order = issue order, which the facing-relative mover requires
        // (each tick's yaw feeds the next tick's translation).
        for &inp in self.inputs.values() {
            self.sim.predict_player(me, inp);
        }
    }

    pub fn configure_crabs(&mut self, crabs: usize) {
        self.sim.configure_crabs(crabs);
    }

    pub fn me(&self) -> PlayerId {
        self.me
    }

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

    #[test]
    fn intent_to_command_sanitizes_untrusted_axes() {
        let cmd = PilotIntent {
            kind: VehicleKind::Ship,
            throttle_trim: f32::NAN,
            thrust: [55.0, f32::NEG_INFINITY, -0.5],
            pitch: -9.0,
            roll: 0.25,
            yaw: f32::INFINITY,
            match_velocity: true,
        }
        .to_command(crab_world::vehicle::Boarding {
            pos: bevy::math::Vec3::ZERO,
            yaw: 0.0,
            velocity: bevy::math::Vec3::ZERO,
        });
        assert_eq!(cmd.kind, VehicleKind::Ship);
        assert_eq!(cmd.throttle_trim, 0.0, "NaN → neutral");
        assert_eq!(cmd.thrust.to_array(), [1.0, 0.0, -0.5], "clamped / zeroed");
        assert_eq!(cmd.pitch, -1.0, "clamped");
        assert_eq!(cmd.roll, 0.25, "an in-range axis passes through exactly");
        assert_eq!(cmd.yaw, 0.0, "∞ → neutral");
        assert!(cmd.match_velocity);
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
        let mut host_sched = ClientSim::new(42, &roster, host); // the host's own input scheduler
        let mut client = ClientSim::new(42, &roster, me);
        // The pure-replay reference: our avatar driven by every issued input, in order — what a
        // zero-latency link would render. Prediction must match it exactly.
        let mut reference = ClientSim::new(42, &roster, me);

        const LATENCY: usize = 4; // ticks of one-way transit, both directions
        let mut wire_up: std::collections::VecDeque<TickMsg> = Default::default();
        let mut wire_down: std::collections::VecDeque<CoreSnapshot> = Default::default();

        // Stay within STARTUP_GRACE_TICKS (30): the crab stays disarmed (no claw downs) and the round
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
            wire_up.push_back(client.submit_local_input(inp, None));
            reference.sim.predict_player(me, inp); // zero-latency ideal, applied at issue
            if wire_up.len() > LATENCY {
                let msg = wire_up.pop_front().expect("len checked");
                server.record_remote(me, msg);
            }
            server.advance(host_sched.submit_local_input(Input::default(), None));
            while server.next_tick_ready() {
                let poses = crate::sim::hold_poses(server.sim());
                let bytes = server.step_next(&poses, Default::default()).snapshot;
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
            server.advance(host_sched.submit_local_input(Input::default(), None));
            while server.next_tick_ready() {
                let poses = crate::sim::hold_poses(server.sim());
                let bytes = server.step_next(&poses, Default::default()).snapshot;
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

    #[test]
    fn adopt_snapshots_follows_a_host_restart() {
        use crate::sim::buttons;

        let me = PlayerId(0);
        let roster = ids(1);
        let mut client = ClientSim::new(7, &roster, me);

        let mut sched = ClientSim::new(7, &roster, me);
        let mut host = Server::new(me, &roster, Sim::new(7, &roster));
        let mut arrivals = Vec::new();
        for t in 0..7u64 {
            let btns = if t == 5 { buttons::RESTART } else { 0 };
            let input = Input::new(0.0, if t < 5 { 1.0 } else { 0.0 }, 0.0, btns);
            host.advance(sched.submit_local_input(input, None));
            while host.next_tick_ready() {
                let poses = crate::sim::hold_poses(host.sim());
                let bytes = host.step_next(&poses, Default::default()).snapshot;
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
            host.sim().player(me).expect("rostered").pos(),
            "the client ends on the host's restarted spawn state (the restart drew a \
             fresh layout, rl#305 — adoption must land exactly there)"
        );
        assert_eq!(
            client.sim().extraction(),
            host.sim().extraction(),
            "the re-rolled extraction rides the snapshot to the client"
        );
    }

    /// The prune is watermark-driven, not tick-driven: a snapshot whose tick has advanced far
    /// past our issue cursor but whose watermark says nothing of ours was consumed must keep the
    /// whole in-flight window for replay.
    #[test]
    fn prune_follows_the_watermark_not_the_tick() {
        let me = PlayerId(1);
        let roster = ids(2);
        let mut client = ClientSim::new(9, &roster, me);
        for _ in 0..3 {
            let _ = client.submit_local_input(Input::from_axes(1.0, 0.0), None);
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
