use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

use crate::client::{ClientSim, PeerMsg, TickMsg};
use crate::membership::{BEAT_EVERY, Membership, Role, Status};
use crate::sim::PlayerId;
use crate::telemetry::{self, TelemetryEvent, TelemetrySender};
use crate::transport::{self, PeerWire, Session};

pub struct LobbyControl {
    pub role: Role,
    pub start_rx: std::sync::mpsc::Receiver<()>,
    pub cancel_rx: std::sync::mpsc::Receiver<()>,
    pub roster_tx: std::sync::mpsc::Sender<Vec<EndpointId>>,
}

pub struct Frozen {
    pub id_map: BTreeMap<EndpointId, PlayerId>,
    pub me: PlayerId,
    pub early: Vec<(EndpointId, TickMsg)>,
    pub sync: crate::SyncVerdict,
}

pub enum Formation {
    Agreed(Frozen),
    /// Formation ended with only us live — play solo. Fires only in the genuinely-alone
    /// case: see [`is_alone_now`] / [`is_alone_at_timeout`].
    Alone,
    Cancelled,
}

pub async fn form_match(
    session: &mut transport::Session,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
    lobby: Option<&LobbyControl>,
    local_asset_digest: u64,
    local_crab_count: u8,
) -> Result<Formation> {
    let my_eid = session.endpoint_id();
    println!(
        "forming match on the LAN (need {expect} player(s), solo if alone after {discover_secs}s)…"
    );

    let outcome = match run_barrier(
        session,
        my_eid,
        discover_secs,
        expect,
        telemetry,
        lobby,
        local_asset_digest,
        local_crab_count,
    )
    .await
    {
        Ok(BarrierResult::Agreed(o)) => o,
        Ok(BarrierResult::Alone) => {
            println!("no other peer found within {discover_secs}s — starting a solo round");
            return Ok(Formation::Alone);
        }
        Ok(BarrierResult::Cancelled) => {
            println!("lobby cancelled by the player");
            return Ok(Formation::Cancelled);
        }
        Err(e) => {
            if let Some(t) = telemetry {
                t.send(TelemetryEvent::RosterFailed {
                    reason: format!("{e:#}"),
                });
            }
            return Err(e);
        }
    };
    let id_map = assign_player_ids(my_eid, &outcome.roster)?;
    let me = id_map[&my_eid];
    println!(
        "match formed: {} participant(s), barrier agreed in {:.1}s",
        id_map.len(),
        outcome.elapsed.as_secs_f64()
    );
    if let Some(t) = telemetry {
        t.send(TelemetryEvent::RosterAgreed {
            members: telemetry::short_ids(&outcome.roster),
            roster_hash: crate::membership::roster_hash(&outcome.roster),
            me: me.0,
        });
    }
    if local_asset_digest != 0 {
        if !outcome.sync.assets {
            tracing::warn!(
                "GCR: crab MODEL ASSET NOT synced across peers (a peer has a different \
                 sally.glb / no model — it would build and render a different crab) — cannot \
                 arm the NN crabs; the windowed client will REFUSE this round (rl#114, no \
                 integer fallback). Run rl-update on every device so all carry the identical \
                 crab model."
            );
        } else {
            println!(
                "GCR: the crab asset is synced across all {} peer(s) — NN crabs eligible",
                id_map.len()
            );
        }
    }
    Ok(Formation::Agreed(Frozen {
        id_map,
        me,
        early: outcome.early,
        sync: outcome.sync,
    }))
}

struct BarrierOutcome {
    roster: Vec<EndpointId>,
    early: Vec<(EndpointId, TickMsg)>,
    elapsed: Duration,
    /// [`Membership::sync_verdict`] sampled at the close instant.
    sync: crate::SyncVerdict,
}

enum BarrierResult {
    Agreed(BarrierOutcome),
    Alone,
    Cancelled,
}

#[allow(clippy::too_many_arguments)]
async fn run_barrier(
    session: &mut Session,
    me: EndpointId,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
    lobby: Option<&LobbyControl>,
    local_asset_digest: u64,
    local_crab_count: u8,
) -> Result<BarrierResult> {
    let start = Instant::now();
    let mut m = match lobby {
        Some(c) => Membership::host_triggered(c.role, me, expect, start),
        None => Membership::new(me, expect, start),
    }
    .with_asset_digest(local_asset_digest)
    .with_crab_count(local_crab_count);
    let mut early: Vec<(EndpointId, TickMsg)> = Vec::new();
    let mut ticker = tokio::time::interval(BEAT_EVERY);
    let mut last_live = 0usize;
    let mut last_roster: Vec<EndpointId> = Vec::new();
    // Whether we've EVER received a direct beat from any peer this formation — gates the
    // solo fallback ([`is_alone_now`]).
    let mut ever_heard_peer = false;
    let alone_deadline = start + Duration::from_secs(discover_secs.max(1));

    loop {
        ticker.tick().await;
        let now = Instant::now();

        if let Some(c) = lobby {
            match c.cancel_rx.try_recv() {
                Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Ok(BarrierResult::Cancelled);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if c.start_rx.try_recv().is_ok() {
                m.set_starting();
            }
        }

        while let Some(from) = session.try_recv() {
            match from.msg {
                PeerWire::Beat(beat) => {
                    // A direct beat from a peer — latch it (gates the solo fallback).
                    if from.from != me {
                        ever_heard_peer = true;
                    }
                    m.on_beat(from.from, &beat, now);
                }
                PeerWire::Tick(msg) => early.push((from.from, msg)),
                // A dialer that catches us mid-formation would otherwise get silence and
                // misdiagnose "host unreachable" (rl#245) — tell it we're busy instead.
                PeerWire::JoinRequest(_) => {
                    tracing::warn!(
                        "refusing mid-formation join from {}: still forming",
                        from.from.fmt_short()
                    );
                    session
                        .send(from.from, &crate::server::Refusal::Forming)
                        .await;
                }
                PeerWire::Snapshot(_)
                | PeerWire::Articulation(_)
                | PeerWire::Refuse(_)
                | PeerWire::Welcome(_) => {}
            }
        }

        let status = m.poll(now);
        session.broadcast(&m.beat()).await;

        if let Some(c) = lobby {
            let roster = m.live_set();
            if roster != last_roster {
                let _ = c.roster_tx.send(roster.clone());
                last_roster = roster;
            }
        }

        match status {
            Status::Agreed { roster } => {
                return Ok(BarrierResult::Agreed(BarrierOutcome {
                    roster,
                    early,
                    elapsed: now.duration_since(start),
                    sync: m.sync_verdict(),
                }));
            }
            Status::Failed => {
                if lobby.is_none() && is_alone_at_timeout(expect, m.live_set().len()) {
                    return Ok(BarrierResult::Alone);
                }
                anyhow::bail!(
                    "match formation failed: peers never agreed on one roster within the join window \
                     (too few players showed up, or a peer kept appearing/disappearing, or a link is \
                     one-way). Relaunch together."
                );
            }
            Status::Forming { live } => {
                // Solo fallback — DEFAULT (non-lobby) path only ([`is_alone_now`]; `live`
                // is fresh from the `poll` above).
                if lobby.is_none()
                    && is_alone_now(expect, live, ever_heard_peer, now >= alone_deadline)
                {
                    return Ok(BarrierResult::Alone);
                }
                if live != last_live {
                    println!("forming: {live}/{expect} player(s) live, waiting for agreement…");
                    last_live = live;
                    if let Some(t) = telemetry {
                        t.send(TelemetryEvent::RosterForming { live, expect });
                    }
                }
            }
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

    fn eid(i: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[i; 32]).public()
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

    #[tokio::test]
    #[ignore = "binds real iroh UDP endpoints; run explicitly with --ignored"]
    async fn three_endpoints_form_the_identical_match_over_iroh() {
        use std::collections::BTreeMap;

        let _serial = crate::real_net_serial();
        let mut s0 = transport::start_session().await.expect("start s0");
        let mut s1 = transport::start_session().await.expect("start s1");
        let mut s2 = transport::start_session().await.expect("start s2");
        let (a0, a1) = (s0.local_addr(), s1.local_addr());
        let (e0, e1, e2) = (s0.endpoint_id(), s1.endpoint_id(), s2.endpoint_id());

        s0.connect_direct(a1.clone()).await.expect("s0->s1");

        let f0 = form_match(&mut s0, 1, 3, None, None, 0, 0);
        let f1 = form_match(&mut s1, 1, 3, None, None, 0, 0);
        let f2 = async {
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
            s2.connect_direct(a0.clone()).await.expect("s2->s0");
            s2.connect_direct(a1.clone()).await.expect("s2->s1");
            form_match(&mut s2, 1, 3, None, None, 0, 0).await
        };
        let (r0, r1, r2) = tokio::join!(f0, f1, f2);
        let unwrap_agreed = |r: Result<Formation>, who: &str| match r.expect(who) {
            Formation::Agreed(f) => f,
            Formation::Alone => panic!("{who}: fell back to solo despite peers being present"),
            Formation::Cancelled => panic!("{who}: cancelled despite no lobby control"),
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

        s0.shutdown().await;
        s1.shutdown().await;
        s2.shutdown().await;
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
}
