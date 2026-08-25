//! The native async driver around the platform-free formation core
//! ([`net_proto::formation::FormationCore`], rl#411 stage 2): transport I/O, lobby
//! channel plumbing, cancellation, and user-facing prints live here; every protocol
//! decision — beats, timeouts, agreement, the solo fallbacks — is the core's.

use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

use crate::membership::{BEAT_EVERY_MS, Role};
use crate::telemetry::{self, TelemetryEvent, TelemetrySender};
use crate::transport::{self, PeerWire};

use net_proto::formation::{FormationCore, Outcome};
pub use net_proto::formation::{Frozen, assign_player_ids, early_peer_msgs, solo_client_for};

pub struct LobbyControl {
    pub role: Role,
    pub start_rx: std::sync::mpsc::Receiver<()>,
    pub cancel_rx: std::sync::mpsc::Receiver<()>,
    pub roster_tx: std::sync::mpsc::Sender<Vec<EndpointId>>,
}

pub enum Formation {
    Agreed(Frozen),
    /// Formation ended with only us live — play solo. Fires only in the genuinely-alone
    /// case: see the core's `is_alone_now` / `is_alone_at_timeout`.
    Alone,
    Cancelled,
}

pub async fn form_match(
    session: &mut transport::Session,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
    lobby: Option<&LobbyControl>,
    stamp: crate::SyncStamp,
) -> Result<Formation> {
    let my_eid = session.endpoint_id();
    println!(
        "forming match on the LAN (need {expect} player(s), solo if alone after {discover_secs}s)…"
    );

    let agreement = match run_barrier(
        session,
        my_eid,
        discover_secs,
        expect,
        telemetry,
        lobby,
        stamp,
    )
    .await
    {
        Ok(BarrierResult::Agreed(a)) => a,
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
    let id_map = assign_player_ids(my_eid, &agreement.roster)?;
    let me = id_map[&my_eid];
    println!(
        "match formed: {} participant(s), barrier agreed in {:.1}s",
        id_map.len(),
        Duration::from_millis(agreement.elapsed_ms).as_secs_f64()
    );
    if let Some(t) = telemetry {
        t.send(TelemetryEvent::RosterAgreed {
            members: telemetry::short_ids(&agreement.roster),
            roster_hash: crate::membership::roster_hash(&agreement.roster),
            me: me.0,
        });
    }
    if stamp.body_digest() != 0 {
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
    if stamp.plant_digest() != 0 {
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
    let sync = agreement.sync;
    let early = agreement.early;
    Ok(Formation::Agreed(Frozen {
        id_map,
        me,
        early,
        sync,
    }))
}

enum BarrierResult {
    Agreed(net_proto::formation::Agreement),
    Alone,
    Cancelled,
}

async fn run_barrier(
    session: &mut transport::Session,
    me: EndpointId,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
    lobby: Option<&LobbyControl>,
    stamp: crate::SyncStamp,
) -> Result<BarrierResult> {
    // The core's clock is injected millis; this driver anchors the axis here.
    let start = Instant::now();
    let mut core = match lobby {
        Some(c) => FormationCore::host_triggered(c.role, me, expect, stamp, 0),
        None => FormationCore::new(me, expect, discover_secs, stamp, 0),
    };
    let mut ticker = tokio::time::interval(Duration::from_millis(BEAT_EVERY_MS));

    loop {
        ticker.tick().await;
        let now_ms = start.elapsed().as_millis() as u64;

        if let Some(c) = lobby {
            match c.cancel_rx.try_recv() {
                Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Ok(BarrierResult::Cancelled);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if c.start_rx.try_recv().is_ok() {
                core.set_starting();
            }
        }

        while let Some(from) = session.try_recv() {
            match from.msg {
                PeerWire::Beat(beat) => core.on_beat(from.from, &beat, now_ms),
                PeerWire::Tick(msg) => core.on_early_tick(from.from, msg),
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

        let step = core.step(now_ms);
        if let Some(beat) = &step.beat {
            session.broadcast(beat).await;
        }
        if let (Some(c), Some(roster)) = (lobby, step.roster_changed) {
            let _ = c.roster_tx.send(roster);
        }
        if let Some(live) = step.live_changed {
            println!("forming: {live}/{expect} player(s) live, waiting for agreement…");
            if let Some(t) = telemetry {
                t.send(TelemetryEvent::RosterForming { live, expect });
            }
        }
        match step.outcome {
            Some(Outcome::Agreed(a)) => return Ok(BarrierResult::Agreed(a)),
            Some(Outcome::Alone) => return Ok(BarrierResult::Alone),
            Some(Outcome::Failed) => anyhow::bail!(
                "match formation failed: peers never agreed on one roster within the join window \
                 (too few players showed up, or a peer kept appearing/disappearing, or a link is \
                 one-way). Relaunch together."
            ),
            None => {}
        }
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
        // A sync test so the serialization guard is held across the whole async body
        // without tripping `await_holding_lock` — it excludes sibling TESTS, not tasks.
        let _serial = crate::real_net_serial();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(three_endpoints_body());
    }

    async fn three_endpoints_body() {
        use std::collections::BTreeMap;

        let mut s0 = transport::start_session().await.expect("start s0");
        let mut s1 = transport::start_session().await.expect("start s1");
        let mut s2 = transport::start_session().await.expect("start s2");
        let (a0, a1) = (s0.local_addr(), s1.local_addr());
        let (e0, e1, e2) = (s0.endpoint_id(), s1.endpoint_id(), s2.endpoint_id());

        // A dial can LOSE the crossed-dial dedup to the peer's concurrent discovery-dial
        // (the kept link still serves — transport closes only the duplicate), so a dial
        // error is benign here exactly as in production (`connect_and_form_inner` warns
        // and proceeds); the formation assertions below are the real check.
        let dial = |r: Result<()>, leg: &str| {
            if let Err(e) = r {
                eprintln!("{leg} dial lost benignly: {e:#}");
            }
        };
        dial(s0.connect_direct(a1.clone()).await, "s0->s1");

        let f0 = form_match(&mut s0, 1, 3, None, None, SyncStamp::ZERO);
        let f1 = form_match(&mut s1, 1, 3, None, None, SyncStamp::ZERO);
        let f2 = async {
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
            dial(s2.connect_direct(a0.clone()).await, "s2->s0");
            dial(s2.connect_direct(a1.clone()).await, "s2->s1");
            form_match(&mut s2, 1, 3, None, None, SyncStamp::ZERO).await
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
}
