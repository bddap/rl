//! The native poll-driven driver around the platform-free formation core
//! ([`net_proto::formation::FormationCore`], rl#411 stage 2): transport I/O, telemetry,
//! and user-facing prints live here; every protocol decision — beats, timeouts,
//! agreement, the solo fallbacks — is the core's. No thread, no async: the owner of a
//! [`FormationDriver`] pumps it from its own loop (the render frame, or a paced CLI
//! loop) until it yields.

use anyhow::Result;
use iroh::EndpointId;

use crate::membership::Role;
use crate::telemetry::{self, TelemetryEvent, TelemetrySender};
use crate::transport::{self, PeerWire};

use net_proto::formation::{FormationCore, Outcome};
pub use net_proto::formation::{Frozen, assign_player_ids, early_peer_msgs, solo_client_for};

/// The cadence at which BLOCKING callers (CLI forming/joining, with no frame loop of
/// their own) pump the poll-driven session.
#[cfg(not(target_family = "wasm"))]
pub(crate) const FORM_POLL: std::time::Duration = std::time::Duration::from_millis(10);

pub enum Formation {
    Agreed(Frozen),
    /// Formation ended with only us live — play solo. Fires only in the genuinely-alone
    /// case: see the core's `is_alone_now` / `is_alone_at_timeout`.
    Alone,
}

/// One formation attempt, pump-to-completion. The core's clock is injected millis —
/// the SESSION's axis ([`Session::now_ms`](transport::Session::now_ms)), so the same
/// driver runs wherever the link does (native tokio clock, browser performance.now).
pub struct FormationDriver {
    core: FormationCore,
    /// Session-axis ms at construction — for the human-facing "agreed in N s" print.
    born_ms: u64,
    expect: usize,
    stamp: crate::SyncStamp,
    /// Roster as of the last change the core reported — for lobby UI polling.
    roster: Vec<EndpointId>,
}

impl FormationDriver {
    /// Timed LAN discovery (the scripted/CLI path): solo fallback after `discover_secs`
    /// if nobody shows.
    pub fn discovering(
        session: &transport::Session,
        discover_secs: u64,
        expect: usize,
        stamp: crate::SyncStamp,
    ) -> Self {
        println!(
            "forming match on the LAN (need {expect} player(s), solo if alone after {discover_secs}s)…"
        );
        let now_ms = session.now_ms();
        Self {
            core: FormationCore::new(session.endpoint_id(), expect, discover_secs, stamp, now_ms),
            born_ms: now_ms,
            expect,
            stamp,
            roster: Vec::new(),
        }
    }

    /// A lobby (the menu path): no discovery timeout; the HOST's explicit
    /// [`Self::set_starting`] arms agreement (rl#94 liveness — a joiner-role core's is
    /// inert by construction, see `Membership::set_starting`).
    pub fn lobby(
        session: &transport::Session,
        role: Role,
        expect: usize,
        stamp: crate::SyncStamp,
    ) -> Self {
        let now_ms = session.now_ms();
        Self {
            core: FormationCore::host_triggered(role, session.endpoint_id(), expect, stamp, now_ms),
            born_ms: now_ms,
            expect,
            stamp,
            roster: Vec::new(),
        }
    }

    pub fn set_starting(&mut self) {
        self.core.set_starting();
    }

    /// Pump to completion on the calling thread — the pacer for callers with no frame
    /// loop of their own (the CLI paths). The windowed lobby polls [`Self::pump`] per
    /// frame instead: one driver, two pacers.
    #[cfg(not(target_family = "wasm"))]
    pub fn pump_blocking(
        mut self,
        session: &mut transport::Session,
        tel: Option<&TelemetrySender>,
    ) -> Result<Formation> {
        loop {
            if let Some(outcome) = self.pump(session, tel) {
                return outcome;
            }
            std::thread::sleep(FORM_POLL);
        }
    }

    pub fn roster(&self) -> &[EndpointId] {
        &self.roster
    }

    /// One pump: drain the session's inbox into the core, step it, send what it emits.
    /// Call at any cadence at or above the beat rate (frame rate is fine — the core's
    /// beat deadline is epoch-aligned, so faster polling never over-beats).
    pub fn pump(
        &mut self,
        session: &mut transport::Session,
        tel: Option<&TelemetrySender>,
    ) -> Option<Result<Formation>> {
        let now_ms = session.now_ms();

        while let Some(from) = session.try_recv() {
            match from.msg {
                PeerWire::Beat(beat) => self.core.on_beat(from.from, &beat, now_ms),
                PeerWire::Tick(msg) => self.core.on_early_tick(from.from, msg),
                // A dialer that catches us mid-formation would otherwise get silence and
                // misdiagnose "host unreachable" (rl#245) — tell it we're busy instead.
                PeerWire::JoinRequest(_) => {
                    tracing::warn!(
                        "refusing mid-formation join from {}: still forming",
                        from.from.fmt_short()
                    );
                    session.send(from.from, &crate::server::Refusal::Forming);
                }
                PeerWire::Snapshot(_)
                | PeerWire::Articulation(_)
                | PeerWire::Refuse(_)
                | PeerWire::Welcome(_) => {}
            }
        }

        let step = self.core.step(now_ms);
        if let Some(beat) = &step.beat {
            session.broadcast(beat);
        }
        if let Some(roster) = step.roster_changed {
            let live = roster.len();
            self.roster = roster;
            println!(
                "forming: {live}/{} player(s) live, waiting for agreement…",
                self.expect
            );
            if let Some(t) = tel {
                t.send(TelemetryEvent::RosterForming {
                    live,
                    expect: self.expect,
                });
            }
        }
        match step.outcome {
            Some(Outcome::Agreed(a)) => Some(self.agreed(session.endpoint_id(), a, tel, now_ms)),
            Some(Outcome::Alone) => {
                println!("no other peer found — starting a solo round");
                Some(Ok(Formation::Alone))
            }
            Some(Outcome::Failed) => {
                let e = anyhow::anyhow!(
                    "match formation failed: peers never agreed on one roster within the join window \
                     (too few players showed up, or a peer kept appearing/disappearing, or a link is \
                     one-way). Relaunch together."
                );
                if let Some(t) = tel {
                    t.send(TelemetryEvent::RosterFailed {
                        reason: format!("{e:#}"),
                    });
                }
                Some(Err(e))
            }
            None => None,
        }
    }

    fn agreed(
        &self,
        my_eid: EndpointId,
        agreement: net_proto::formation::Agreement,
        tel: Option<&TelemetrySender>,
        now_ms: u64,
    ) -> Result<Formation> {
        let id_map = assign_player_ids(my_eid, &agreement.roster)?;
        let me = id_map[&my_eid];
        println!(
            "match formed: {} participant(s), barrier agreed in {:.1}s",
            id_map.len(),
            now_ms.saturating_sub(self.born_ms) as f64 / 1000.0
        );
        if let Some(t) = tel {
            t.send(TelemetryEvent::RosterAgreed {
                members: telemetry::short_ids(&agreement.roster),
                roster_hash: crate::membership::roster_hash(&agreement.roster),
                me: me.0,
            });
        }
        if self.stamp.body_digest() != 0 {
            if !agreement.sync.body {
                tracing::warn!(
                    "GCR: crab BODY NOT synced across peers (a peer has a different sally.glb \
                     / no model / a different baked collider table or binary version — it \
                     would build and render a different crab) — cannot arm the NN crabs; the \
                     windowed client will REFUSE this round (rl#114, no integer fallback). \
                     Run rl-update on every device so all carry the identical build + model."
                );
            } else {
                println!(
                    "GCR: the crab body is synced across all {} peer(s) — NN crabs eligible",
                    id_map.len()
                );
            }
        }
        if self.stamp.plant_digest() != 0 {
            if !agreement.sync.plant {
                tracing::warn!(
                    "GCR: the PLANT is NOT synced across peers (a peer resolves a different arena, \
                     terrain bake, or joint-friction cap — its ground would disagree with the poses \
                     it adopts, floating/burying every crab and craft on that screen); the windowed \
                     client will REFUSE this round (rl#286, same policy as rl#114). Run rl-update on \
                     every device so all carry the identical build + checkpoint."
                );
            } else {
                println!(
                    "GCR: the plant (arena/bake/friction) is synced across all {} peer(s)",
                    id_map.len()
                );
            }
        }
        Ok(Formation::Agreed(Frozen {
            id_map,
            me,
            early: agreement.early,
            sync: agreement.sync,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SyncStamp;
    use crate::sim::PlayerId;

    #[test]
    #[ignore = "binds real iroh UDP endpoints; run explicitly with --ignored"]
    fn three_endpoints_form_the_identical_match_over_iroh() {
        let _serial = crate::real_net_serial();
        use std::collections::BTreeMap;

        let mut s0 = transport::start_session().expect("start s0");
        let mut s1 = transport::start_session().expect("start s1");
        let mut s2 = transport::start_session().expect("start s2");
        let (a0, a1) = (s0.local_addr(), s1.local_addr());
        let (e0, e1, e2) = (s0.endpoint_id(), s1.endpoint_id(), s2.endpoint_id());

        // A dial can LOSE the crossed-dial dedup to the peer's concurrent discovery-dial
        // (the kept link still serves — transport closes only the duplicate), so a dial
        // error is benign here exactly as in production (`connect_and_form_inner` warns
        // and proceeds); the formation assertions below are the real check. Fire-and-
        // forget: the verdict channels go unpolled for the same reason.
        let _d0 = s0.dial(a1.clone());

        let mut d0 = FormationDriver::discovering(&s0, 1, 3, SyncStamp::ZERO);
        let mut d1 = FormationDriver::discovering(&s1, 1, 3, SyncStamp::ZERO);
        // s2 joins late, by explicit dial, like a code-join straggler — its discovery
        // window is wider so the 600ms-late dial can't race its own solo fallback on a
        // loaded runner.
        let late_at = std::time::Instant::now() + std::time::Duration::from_millis(600);
        let mut s2_dialed = false;
        let mut d2 = FormationDriver::discovering(&s2, 2, 3, SyncStamp::ZERO);

        // Round-robin pump all three from one thread — the poll-driven interface needs
        // no per-peer runtime or thread, which is the point of the seam.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (mut r0, mut r1, mut r2) = (None, None, None);
        while r0.is_none() || r1.is_none() || r2.is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "formation did not converge in 30s"
            );
            if !s2_dialed && std::time::Instant::now() >= late_at {
                s2_dialed = true;
                let _ = s2.dial(a0.clone());
                let _ = s2.dial(a1.clone());
            }
            for (r, d, s) in [
                (&mut r0, &mut d0, &mut s0),
                (&mut r1, &mut d1, &mut s1),
                (&mut r2, &mut d2, &mut s2),
            ] {
                if r.is_none() {
                    *r = d.pump(s, None);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let unwrap_agreed =
            |r: Option<Result<Formation>>, who: &str| match r.expect("resolved").expect(who) {
                Formation::Agreed(f) => f,
                Formation::Alone => panic!("{who}: fell back to solo despite peers being present"),
            };
        let (r0, r1, r2) = (
            unwrap_agreed(r0, "s0 forms"),
            unwrap_agreed(r1, "s1 forms"),
            unwrap_agreed(r2, "s2 forms"),
        );

        assert_eq!(r0.id_map, r1.id_map, "s0 and s1 must agree on the roster");
        assert_eq!(r1.id_map, r2.id_map, "s1 and s2 must agree on the roster");
        assert_eq!(r0.id_map.len(), 3, "all three endpoints in the match");

        let mut ids = [e0, e1, e2];
        ids.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let expected: BTreeMap<EndpointId, PlayerId> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, PlayerId(i as u8)))
            .collect();
        assert_eq!(
            r0.id_map, expected,
            "roster must be the sorted-id assignment"
        );
        assert_eq!(r0.me, expected[&e0]);
        assert_eq!(r1.me, expected[&e1]);
        assert_eq!(r2.me, expected[&e2]);

        s0.close();
        s1.close();
        s2.close();
    }
}
