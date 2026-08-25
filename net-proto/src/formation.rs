//! The formation barrier as a poll-driven step machine (rl#411 stage 2).
//!
//! The platform-free core of match formation: feed it incoming beats/early ticks as
//! they arrive, call [`FormationCore::step`] on any cadence, act on the returned
//! [`Step`]. The core owns the beat cadence ([`BEAT_EVERY_MS`]) and every timeout, all
//! on the injected-ms clock, so the same machine runs under a tokio ticker today and a
//! browser frame loop later. The driver keeps only platform concerns: transport I/O,
//! lobby channel plumbing, cancellation, and user-facing prints.

use std::collections::BTreeMap;

use anyhow::Result;
use iroh_base::EndpointId;

use crate::SyncVerdict;
use crate::client::{ClientSim, PeerMsg, TickMsg};
use crate::membership::{BEAT_EVERY_MS, Beat, Membership, Role, Status};
use crate::sim::PlayerId;

pub struct Frozen {
    pub id_map: BTreeMap<EndpointId, PlayerId>,
    pub me: PlayerId,
    pub early: Vec<(EndpointId, TickMsg)>,
    pub sync: SyncVerdict,
}

pub struct FormationCore {
    me: EndpointId,
    expect: usize,
    m: Membership,
    early: Vec<(EndpointId, TickMsg)>,
    /// Whether we've EVER received a direct beat from any peer this formation — gates
    /// the solo fallback ([`is_alone_now`]).
    ever_heard_peer: bool,
    last_live: usize,
    last_roster: Vec<EndpointId>,
    last_beat_ms: Option<u64>,
    /// `None` in lobby (host-triggered) mode, where the solo fallbacks never fire.
    alone_deadline_ms: Option<u64>,
}

/// What one [`FormationCore::step`] asks the driver to do, in field order: broadcast
/// `beat`, surface the roster/live changes, then act on `outcome` — so the agreeing
/// step still broadcasts its final beat first, exactly like the old loop.
pub struct Step {
    /// Broadcast to all peers when `Some` — the core owns the [`BEAT_EVERY_MS`] cadence,
    /// so an over-eager driver (a 60Hz frame loop) still beats at the protocol rate.
    pub beat: Option<Beat>,
    /// The live roster, when it changed this step (feeds the lobby UI).
    pub roster_changed: Option<Vec<EndpointId>>,
    /// The live count, when it changed this step (feeds progress prints/telemetry).
    pub live_changed: Option<usize>,
    pub outcome: Option<Outcome>,
}

pub enum Outcome {
    Agreed(Agreement),
    /// Only us live at the window — play solo. Fires only in the genuinely-alone case:
    /// see [`is_alone_now`] / [`is_alone_at_timeout`]; never in lobby mode.
    Alone,
    /// Peers were present but never agreed within the join window.
    Failed,
}

pub struct Agreement {
    pub roster: Vec<EndpointId>,
    pub early: Vec<(EndpointId, TickMsg)>,
    /// ms since the core's epoch (the driver anchors it at formation start).
    pub elapsed_ms: u64,
    /// [`Membership::sync_verdict`] sampled at the close instant.
    pub sync: SyncVerdict,
}

impl FormationCore {
    /// A timer-closed (non-lobby) formation: closes on [`Status::Agreed`] stability,
    /// falls back to solo when genuinely alone past `discover_secs`.
    pub fn new(
        me: EndpointId,
        expect: usize,
        discover_secs: u64,
        stamp: crate::SyncStamp,
        now_ms: u64,
    ) -> Self {
        Self::build(
            me,
            expect,
            Membership::new(me, expect, now_ms).with_stamp(stamp),
            Some(now_ms + discover_secs.max(1) * 1000),
        )
    }

    /// A host-triggered (interactive lobby) formation: closes only on the host's GO
    /// ([`FormationCore::set_starting`]); the solo fallbacks never fire.
    pub fn host_triggered(
        role: Role,
        me: EndpointId,
        expect: usize,
        stamp: crate::SyncStamp,
        now_ms: u64,
    ) -> Self {
        Self::build(
            me,
            expect,
            Membership::host_triggered(role, me, expect, now_ms).with_stamp(stamp),
            None,
        )
    }

    fn build(me: EndpointId, expect: usize, m: Membership, alone_deadline_ms: Option<u64>) -> Self {
        Self {
            me,
            expect,
            m,
            early: Vec::new(),
            ever_heard_peer: false,
            last_live: 0,
            last_roster: Vec::new(),
            last_beat_ms: None,
            alone_deadline_ms,
        }
    }

    pub fn set_starting(&mut self) {
        self.m.set_starting();
    }

    pub fn on_beat(&mut self, from: EndpointId, beat: &Beat, now_ms: u64) {
        if from != self.me {
            self.ever_heard_peer = true;
        }
        self.m.on_beat(from, beat, now_ms);
    }

    /// A game tick that raced ahead of the barrier close — buffered and handed to the
    /// sim through [`Agreement::early`].
    pub fn on_early_tick(&mut self, from: EndpointId, msg: TickMsg) {
        self.early.push((from, msg));
    }

    pub fn step(&mut self, now_ms: u64) -> Step {
        let status = self.m.poll(now_ms);
        let beat = self
            .last_beat_ms
            .is_none_or(|t| now_ms.saturating_sub(t) >= BEAT_EVERY_MS)
            .then(|| {
                self.last_beat_ms = Some(now_ms);
                self.m.beat()
            });
        let roster = self.m.live_set();
        let roster_changed = (roster != self.last_roster).then(|| {
            self.last_roster = roster.clone();
            roster
        });

        let mut live_changed = None;
        let outcome = match status {
            Status::Agreed { roster } => Some(Outcome::Agreed(Agreement {
                roster,
                early: std::mem::take(&mut self.early),
                elapsed_ms: now_ms,
                sync: self.m.sync_verdict(),
            })),
            Status::Failed => {
                if self.alone_deadline_ms.is_some()
                    && is_alone_at_timeout(self.expect, self.m.live_set().len())
                {
                    Some(Outcome::Alone)
                } else {
                    Some(Outcome::Failed)
                }
            }
            Status::Forming { live } => {
                if live != self.last_live {
                    self.last_live = live;
                    live_changed = Some(live);
                }
                self.alone_deadline_ms
                    .is_some_and(|deadline| {
                        is_alone_now(self.expect, live, self.ever_heard_peer, now_ms >= deadline)
                    })
                    .then_some(Outcome::Alone)
            }
        };
        Step {
            beat,
            roster_changed,
            live_changed,
            outcome,
        }
    }
}

fn is_alone_now(expect: usize, live: usize, ever_heard_peer: bool, past_deadline: bool) -> bool {
    expect > 1 && !ever_heard_peer && live == 1 && past_deadline
}

fn is_alone_at_timeout(expect: usize, live: usize) -> bool {
    expect > 1 && live == 1
}

pub fn solo_client_for(seed: u64) -> ClientSim {
    let me = PlayerId(0);
    ClientSim::new(seed, &[me], me)
}

pub fn early_peer_msgs(frozen: &Frozen) -> Vec<PeerMsg> {
    frozen
        .early
        .iter()
        .filter_map(|(from, msg)| {
            frozen
                .id_map
                .get(from)
                .map(|&pid| PeerMsg { pid, msg: *msg })
        })
        .collect()
}

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
    use crate::SyncStamp;

    fn eid(i: u8) -> EndpointId {
        iroh_base::SecretKey::from_bytes(&[i; 32]).public()
    }

    #[test]
    fn assign_player_ids_is_identical_regardless_of_roster_order() {
        let me = eid(2);
        let a = assign_player_ids(me, &[eid(1), eid(3), eid(2)]).unwrap();
        let b = assign_player_ids(me, &[eid(3), eid(2), eid(1)]).unwrap();
        assert_eq!(a, b, "id assignment must not depend on input order");
        let mut ids = [eid(1), eid(2), eid(3)];
        ids.sort_by(|x, y| x.as_bytes().cmp(y.as_bytes()));
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(a[id], PlayerId(i as u8), "id at sort position {i}");
        }
    }

    #[test]
    fn assign_player_ids_dedups_self_in_roster() {
        let me = eid(5);
        let map = assign_player_ids(me, &[eid(5), eid(7)]).unwrap();
        assert_eq!(map.len(), 2, "self must not be double-counted");
        let mut got: Vec<PlayerId> = vec![map[&eid(5)], map[&eid(7)]];
        got.sort();
        assert_eq!(got, vec![PlayerId(0), PlayerId(1)]);
    }

    #[test]
    fn player_zero_is_host_of() {
        let roster = [eid(9), eid(3), eid(6)];
        let map = assign_player_ids(eid(3), &roster).unwrap();
        let host = crate::membership::host_of(&roster);
        assert_eq!(map[&host], PlayerId(0), "host_of must hold PlayerId(0)");
    }

    #[test]
    fn alone_fallback_fires_only_when_defaulted_networked_never_heard_and_truly_alone() {
        assert!(
            is_alone_now(2, 1, false, true),
            "defaulted-networked + never-heard + alone + past the window ⇒ solo"
        );

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
        assert!(!is_alone_now(4, 3, false, true));
    }

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

    #[test]
    fn core_beats_at_protocol_cadence_even_when_stepped_faster() {
        let mut c = FormationCore::new(eid(1), 1, 1, SyncStamp::ZERO, 0);
        assert!(c.step(0).beat.is_some(), "first step beats immediately");
        assert!(
            c.step(16).beat.is_none(),
            "a frame-rate step inside the beat interval stays silent"
        );
        assert!(
            c.step(BEAT_EVERY_MS).beat.is_some(),
            "due again at the interval"
        );
    }

    #[test]
    fn lone_core_with_expect_one_agrees_on_itself() {
        let me = eid(1);
        let mut c = FormationCore::new(me, 1, 1, SyncStamp::ZERO, 0);
        let mut t = 0;
        loop {
            match c.step(t).outcome {
                Some(Outcome::Agreed(a)) => {
                    assert_eq!(a.roster, vec![me]);
                    assert_eq!(a.elapsed_ms, t);
                    break;
                }
                Some(_) => panic!("expect==1 must agree on itself, never solo/fail"),
                None => {}
            }
            t += BEAT_EVERY_MS;
            assert!(t < 10_000, "must agree within the stability window");
        }
    }

    #[test]
    fn two_cores_converge_through_exchanged_beats() {
        let (ida, idb) = (eid(1), eid(2));
        let mut a = FormationCore::new(ida, 2, 5, SyncStamp::ZERO, 0);
        let mut b = FormationCore::new(idb, 2, 5, SyncStamp::ZERO, 0);
        let mut agreed = (None, None);
        let mut t = 0;
        while t <= 5000 && (agreed.0.is_none() || agreed.1.is_none()) {
            let sa = a.step(t);
            let sb = b.step(t);
            if let Some(beat) = &sa.beat {
                b.on_beat(ida, beat, t);
            }
            if let Some(beat) = &sb.beat {
                a.on_beat(idb, beat, t);
            }
            if let Some(Outcome::Agreed(ag)) = sa.outcome {
                agreed.0 = Some(ag.roster);
            }
            if let Some(Outcome::Agreed(ag)) = sb.outcome {
                agreed.1 = Some(ag.roster);
            }
            t += BEAT_EVERY_MS;
        }
        let ra = agreed.0.expect("A must agree");
        let rb = agreed.1.expect("B must agree");
        assert_eq!(ra, rb, "both cores must freeze the identical set");
        assert_eq!(ra.len(), 2);
    }

    #[test]
    fn alone_core_solos_at_the_discover_deadline_not_before() {
        let mut c = FormationCore::new(eid(1), 2, 2, SyncStamp::ZERO, 0);
        let mut t = 0;
        while t < 2000 {
            assert!(
                !matches!(c.step(t).outcome, Some(Outcome::Alone)),
                "no solo fallback before the discover deadline (t={t})"
            );
            t += BEAT_EVERY_MS;
        }
        assert!(
            matches!(c.step(2000).outcome, Some(Outcome::Alone)),
            "genuinely alone past the deadline ⇒ solo"
        );
    }

    #[test]
    fn early_ticks_ride_the_agreement_out() {
        let me = eid(1);
        let mut c = FormationCore::new(me, 1, 1, SyncStamp::ZERO, 0);
        let msg = TickMsg {
            issue_tick: 7,
            input: crate::sim::Input::default(),
            pilot: None,
        };
        c.on_early_tick(eid(2), msg);
        let mut t = 0;
        loop {
            if let Some(Outcome::Agreed(a)) = c.step(t).outcome {
                assert_eq!(a.early.len(), 1, "the buffered early tick must ride out");
                assert_eq!(a.early[0].0, eid(2));
                break;
            }
            t += BEAT_EVERY_MS;
            assert!(t < 10_000);
        }
    }
}
