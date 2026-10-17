//! The native poll-driven driver around the platform-free formation core
//! ([`net_proto::formation::FormationCore`], rl#411 stage 2): transport I/O
//! and user-facing prints live here; every protocol decision — beats, timeouts,
//! agreement, the solo fallbacks — is the core's. No thread, no async: the owner of a
//! [`FormationDriver`] pumps it from its own loop (the render frame, or a paced CLI
//! loop) until it yields.

use anyhow::Result;
use iroh::EndpointId;

use crate::membership::Role;
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

/// One formation attempt, pump-to-completion. The core's clock is injected millis from
/// the caller — in production the session's axis
/// ([`Session::now_ms`](transport::Session::now_ms)), so the same driver runs wherever
/// the link does (native tokio clock, browser performance.now).
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
        now_ms: u64,
    ) -> Self {
        println!(
            "forming match on the LAN (need {expect} player(s), solo if alone after {discover_secs}s)…"
        );
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
        now_ms: u64,
    ) -> Self {
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
    pub fn pump_blocking(mut self, session: &mut transport::Session) -> Result<Formation> {
        loop {
            let now_ms = session.now_ms();
            if let Some(outcome) = self.pump(session, now_ms) {
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
        now_ms: u64,
    ) -> Option<Result<Formation>> {
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
        }
        match step.outcome {
            Some(Outcome::Agreed(a)) => Some(self.agreed(session.endpoint_id(), a, now_ms)),
            Some(Outcome::Alone) => {
                println!("no other peer found — starting a solo round");
                Some(Ok(Formation::Alone))
            }
            Some(Outcome::Failed) => Some(Err(anyhow::anyhow!(
                "match formation failed: peers never agreed on one roster within the join window \
                     (too few players showed up, or a peer kept appearing/disappearing, or a link is \
                     one-way). Relaunch together."
            ))),
            None => None,
        }
    }

    fn agreed(
        &self,
        my_eid: EndpointId,
        agreement: net_proto::formation::Agreement,
        now_ms: u64,
    ) -> Result<Formation> {
        let id_map = assign_player_ids(agreement.server, &agreement.roster)?;
        let me = id_map[&my_eid];
        println!(
            "match formed: {} participant(s), barrier agreed in {:.1}s",
            id_map.len(),
            now_ms.saturating_sub(self.born_ms) as f64 / 1000.0
        );
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
    fn three_endpoints_form_the_identical_match_over_iroh() {
        let _serial = crate::real_net_serial();
        use net_proto::membership::BEAT_EVERY_MS;
        use std::collections::BTreeMap;
        use std::time::Instant;

        let mut sessions = [(); 3].map(|()| transport::start_session().expect("start session"));
        let ids = sessions.each_ref().map(|s| s.endpoint_id());
        let (a0, a1) = (sessions[0].local_addr(), sessions[1].local_addr());
        let form =
            |s: &transport::Session, t| FormationDriver::discovering(s, 1, 3, SyncStamp::ZERO, t);

        // A dial can LOSE the crossed-dial dedup to the peer's concurrent discovery-dial
        // (the kept link still serves), so a dial verdict is no signal.
        let _ = sessions[0].dial(a1.clone());

        // The formation clock is the test's own: it steps a beat only once every forming
        // driver holds every other's current beat, arrived at the current step, so host
        // speed can never fire a protocol timer. Until then each re-sends its beat: one sent
        // before its link was up, or on a link a crossed dial replaced, is gone.
        let hang_guard = Instant::now() + crate::HANG_GUARD;
        let mut t = 0;
        let mut drivers = [
            Some(form(&sessions[0], t)),
            Some(form(&sessions[1], t)),
            None,
        ];
        let mut results: [Option<Result<Formation>>; 3] = [None, None, None];
        while results.iter().any(Option::is_none) {
            assert!(Instant::now() < hang_guard, "formation hung");
            // s2 joins late, by explicit dial, like a code-join straggler: only once s0 and
            // s1 already count each other live.
            if drivers[2].is_none()
                && drivers[..2]
                    .iter()
                    .all(|d| d.as_ref().is_some_and(|d| d.roster().len() == 2))
            {
                let _ = sessions[2].dial(a0.clone());
                let _ = sessions[2].dial(a1.clone());
                drivers[2] = Some(form(&sessions[2], t));
            }

            let forming: Vec<usize> = (0..3)
                .filter(|&i| drivers[i].is_some() && results[i].is_none())
                .collect();
            for &i in &forming {
                if let Some(d) = drivers[i].as_mut() {
                    results[i] = d.pump(&mut sessions[i], t);
                }
            }
            let stepped = forming.iter().all(|&r| {
                let core = &drivers[r].as_ref().expect("forming").core;
                forming.iter().filter(|&&p| p != r).all(|&p| {
                    let current = drivers[p].as_ref().expect("forming").core.beat();
                    core.last_direct_beat(&ids[p]) == Some((t, current.roster_hash()))
                })
            });
            if stepped {
                t += BEAT_EVERY_MS;
            } else {
                for &i in forming.iter().filter(|&&i| results[i].is_none()) {
                    let core = &drivers[i].as_ref().expect("forming").core;
                    sessions[i].broadcast(&core.beat());
                }
            }
            std::thread::sleep(FORM_POLL);
        }
        let [r0, r1, r2] = results;
        let [s0, s1, s2] = sessions;
        let [e0, e1, e2] = ids;
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
