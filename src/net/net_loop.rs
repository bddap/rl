//! Synchronous bridge from the async iroh [`transport`] to a Bevy/game main loop.
//!
//! [`transport::Session`] is async (tokio); the deterministic lockstep driver and
//! the Bevy render loop are sync and own the main thread. [`NetDriver`] bridges the
//! two: it holds a tokio runtime + the session and exposes two non-blocking calls a
//! per-frame system can use — [`NetDriver::broadcast`] to fan our tick message out
//! to peers, and [`NetDriver::drain_inbox`] to pull whatever peer tick messages have
//! arrived. No determinism lives here; it is pure I/O plumbing (the same split the
//! netcode already draws between [`transport`] and [`crate::net::lockstep`]).
//!
//! The LAN cold-start — form ONE agreed match (the real membership barrier of
//! [`crate::net::membership`]), [`assign_player_ids`] over the agreed set by sorted id,
//! and collect any tick inputs that arrived mid-formation — is [`form_match`], shared by
//! BOTH the windowed client ([`connect_and_assign`]) and the headless `game net` driver.
//! It's the same code on every peer on purpose: the agreed set + id assignment must be
//! byte-identical or the sims silently desync. The two callers differ only in what they
//! do with the session afterward — wrap it in a [`NetDriver`] for the Bevy loop, or
//! drive it raw in the headless async loop — not in the cold-start itself.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

use crate::net::lockstep::{Lockstep, TickMsg};
use crate::net::membership::{BEAT_EVERY, Membership, Status};
use crate::net::sim::PlayerId;
use crate::net::telemetry::{self, TelemetryEvent, TelemetrySender};
use crate::net::transport::{self, PeerWire, Session};

/// A peer tick message tagged with the sender's already-resolved [`PlayerId`]. The
/// id is mapped from the QUIC-authenticated endpoint id via the frozen
/// participant set (never read from the message body — see [`transport::FromPeer`]),
/// so the lockstep driver can trust it as the input's true author.
#[derive(Debug, Clone, Copy)]
pub struct PeerMsg {
    pub pid: PlayerId,
    pub msg: TickMsg,
}

/// Owns the tokio runtime + iroh session and bridges them to a synchronous caller.
/// Held by the game loop as a non-send resource (the runtime/session aren't `Sync`).
pub struct NetDriver {
    rt: tokio::runtime::Runtime,
    session: Session,
    /// Frozen endpoint→PlayerId map (us + peers), agreed across peers by sorting the
    /// agreed participant set. Used to tag inbound messages with their author's id.
    id_map: BTreeMap<EndpointId, PlayerId>,
    /// Optional live-telemetry stream (set iff the client was launched with a
    /// collector). Best-effort and read-only — see [`crate::net::telemetry`]; the
    /// windowed driver pushes Tick/Input/RoundDecided/Fault through it.
    telemetry: Option<TelemetrySender>,
}

impl NetDriver {
    /// The live-telemetry handle, if this client is streaming to a collector. The
    /// render loop reads it to push events (Tick/Input/RoundDecided/Fault). `None` when
    /// launched without `--telemetry`.
    pub fn telemetry(&self) -> Option<&TelemetrySender> {
        self.telemetry.as_ref()
    }

    /// Size of the frozen roster (us + peers) — the match's player count. A sync,
    /// always-correct figure for telemetry's `peers` field (the live link count is async
    /// via the session; the agreed roster size is what the operator wants anyway).
    pub fn roster_len(&self) -> usize {
        self.id_map.len()
    }

    /// Broadcast our tick message to every connected peer. Non-blocking from the
    /// caller's view: it drives the async `broadcast` to completion on the runtime,
    /// which only does buffered QUIC writes (a dead peer is dropped inside, not
    /// awaited — the lockstep stall on the missing input is the visible failure).
    pub fn broadcast(&self, msg: &TickMsg) {
        self.rt.block_on(self.session.broadcast(msg));
    }

    /// Drain every peer TICK message received so far, tagged with the sender's
    /// [`PlayerId`]. Non-blocking: `try_recv` returns immediately when the inbox is
    /// empty. Messages from an endpoint not in the frozen set are dropped (a peer not
    /// in the agreed match isn't part of lockstep); a stray barrier beat that arrives
    /// after formation (a peer still winding down its beat loop) is ignored here, since
    /// the lockstep channel is the only one play cares about.
    pub fn drain_inbox(&mut self) -> Vec<PeerMsg> {
        let mut out = Vec::new();
        while let Some(from) = self.session.try_recv() {
            if let (PeerWire::Tick(msg), Some(&pid)) = (&from.msg, self.id_map.get(&from.from)) {
                out.push(PeerMsg { pid, msg: *msg });
            }
        }
        out
    }
}

/// Bind the LAN endpoint, run the shared [`form_match`] cold-start (the membership
/// barrier), replay the inputs that arrived during formation, and return a ready
/// [`Lockstep`] + the [`NetDriver`] that pumps its transport — the windowed client's
/// entry into a match. The match `seed` must be identical on every peer (the caller
/// passes the shared constant). `expect` is the minimum participant count to close on
/// (see [`form_match`]); `discover_secs` is an advisory UX hint only.
pub fn connect_and_assign(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    collector: Option<iroh::EndpointId>,
) -> Result<(Lockstep, NetDriver)> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (session, frozen, telemetry) = rt.block_on(async {
        let mut session = transport::start_session().await?;
        let my_eid = session.endpoint_id();
        println!("fp client endpoint id: {my_eid}");
        // Open the telemetry side-channel BEFORE forming the match, so the collector
        // sees the roster fill (RosterForming/Agreed). Best-effort: a failure to bind
        // the telemetry endpoint just runs the game without it.
        let telemetry = connect_telemetry(collector, my_eid).await;
        let frozen = form_match(&mut session, discover_secs, expect, telemetry.as_ref()).await?;
        anyhow::Ok((session, frozen, telemetry))
    })?;

    let all_ids: Vec<PlayerId> = frozen.id_map.values().copied().collect();
    println!(
        "starting lockstep: {} player(s), I am {:?}",
        all_ids.len(),
        frozen.me
    );
    let mut ls = Lockstep::new(seed, &all_ids, frozen.me);
    replay_early(&mut ls, &frozen);
    let driver = NetDriver {
        rt,
        session,
        id_map: frozen.id_map,
        telemetry,
    };
    Ok((ls, driver))
}

/// Open a [`TelemetrySender`] to `collector` if one was configured, tagging events with
/// our game endpoint id. Best-effort: any bind failure logs and yields `None`, so the
/// game runs unchanged without telemetry — telemetry can never gate a match.
pub async fn connect_telemetry(
    collector: Option<iroh::EndpointId>,
    my_eid: iroh::EndpointId,
) -> Option<TelemetrySender> {
    let collector = collector?;
    match TelemetrySender::connect(collector, *my_eid.as_bytes()).await {
        Ok(t) => Some(t),
        Err(e) => {
            tracing::warn!("telemetry disabled (endpoint bind failed): {e:#}");
            None
        }
    }
}

/// The outcome of the LAN cold-start: the frozen participant→[`PlayerId`] map (agreed
/// identical on every peer by the barrier), which id is us, and the tick messages that
/// arrived mid-formation (to be replayed once a [`Lockstep`] exists — see
/// [`replay_early`]).
pub struct Frozen {
    pub id_map: BTreeMap<EndpointId, PlayerId>,
    pub me: PlayerId,
    pub early: Vec<(EndpointId, TickMsg)>,
}

/// Run the LAN match-formation barrier ([`run_barrier`]), then freeze the AGREED
/// participant set and assign deterministic [`PlayerId`]s by sorted endpoint id. The
/// ONE cold-start both the windowed client ([`connect_and_assign`]) and the headless
/// `game net` driver run — shared verbatim (not just described as shared) because it
/// must stay behaviorally identical on every peer for the sims to agree, and a drift
/// here would silently desync.
///
/// `expect` is the minimum participant count (incl. us) the barrier requires before it
/// will close — it stops a peer from freezing a lone `{self}` match before LAN discovery
/// has even found the others, and makes a too-small turnout time out ([`Status::Failed`])
/// rather than form a short match. `discover_secs` is advisory (the barrier waits as long
/// as it needs within its own [`crate::net::membership::JOIN_WINDOW`]); it's kept only as
/// a UX hint in the log line.
pub async fn form_match(
    session: &mut transport::Session,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
) -> Result<Frozen> {
    let my_eid = session.endpoint_id();
    println!(
        "forming match on the LAN (need {expect} player(s), discovery hint {discover_secs}s)…"
    );

    let outcome = match run_barrier(session, my_eid, expect, telemetry).await {
        Ok(o) => o,
        Err(e) => {
            // Mirror the formation failure to the collector (best-effort) before
            // surfacing it — the operator sees WHY a deck never started its round.
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
            roster_hash: crate::net::membership::roster_hash(&outcome.roster),
            me: me.0,
        });
    }
    Ok(Frozen {
        id_map,
        me,
        early: outcome.early,
    })
}

/// What [`run_barrier`] returns: the agreed roster (sorted endpoint ids, identical on
/// every peer), the tick messages that arrived mid-formation (a peer that finished the
/// barrier first may already be broadcasting inputs), and how long agreement took.
struct BarrierOutcome {
    roster: Vec<EndpointId>,
    early: Vec<(EndpointId, TickMsg)>,
    elapsed: Duration,
}

/// Drive the membership barrier to agreement: beat our view every [`BEAT_EVERY`],
/// ingest peers' beats (and stash any early ticks), and poll the [`Membership`] state
/// machine until it returns [`Status::Agreed`] (freeze) or [`Status::Failed`] (give up
/// and error — never freeze a guessed set). `expect` is the minimum participant count to
/// close. The pure agreement logic lives in [`crate::net::membership`]; this is only the
/// I/O around it.
async fn run_barrier(
    session: &mut Session,
    me: EndpointId,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
) -> Result<BarrierOutcome> {
    let start = Instant::now();
    let mut m = Membership::new(me, expect, start);
    let mut early: Vec<(EndpointId, TickMsg)> = Vec::new();
    let mut ticker = tokio::time::interval(BEAT_EVERY);
    let mut last_live = 0usize;

    loop {
        ticker.tick().await;
        let now = Instant::now();

        // Ingest everything the transport has: beats feed the membership machine;
        // ticks are from a peer that already finished forming — hold them to replay
        // once we have a Lockstep (the transport doesn't resend, so dropping them would
        // stall us on that peer's first inputs).
        while let Some(from) = session.try_recv() {
            match from.msg {
                PeerWire::Beat(beat) => m.on_beat(from.from, &beat, now),
                PeerWire::Tick(msg) => early.push((from.from, msg)),
            }
        }

        // Poll FIRST (expires stale peers + advances the verdict), so the beat we then
        // advertise reflects the freshly-pruned set — we never gossip a just-expired
        // phantom for an extra round.
        let status = m.poll(now);
        session.broadcast_beat(&m.beat()).await;

        match status {
            Status::Agreed { roster } => {
                return Ok(BarrierOutcome {
                    roster,
                    early,
                    elapsed: now.duration_since(start),
                });
            }
            Status::Failed => {
                anyhow::bail!(
                    "match formation failed: peers never agreed on one roster within the join window \
                     (too few players showed up, or a peer kept appearing/disappearing, or a link is \
                     one-way). Relaunch together."
                );
            }
            Status::Forming { live } => {
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

/// Replay the inputs that arrived during formation into a freshly-built [`Lockstep`],
/// now that ids are assigned. They predate any applied tick, so `record_remote`
/// stashes them rather than comparing a hash — no fault is possible, hence the
/// discard. Senders not in the agreed set are dropped (the same filter the live loop
/// applies).
pub fn replay_early(ls: &mut Lockstep, frozen: &Frozen) {
    for (from, msg) in &frozen.early {
        if let Some(&pid) = frozen.id_map.get(from) {
            let _ = ls.record_remote(pid, *msg);
        }
    }
}

/// Map endpoint ids → [`PlayerId`]s by sorting the full agreed set (us + peers). Every
/// peer sorts the identical set, so a given endpoint is the same `PlayerId`
/// everywhere — the precondition lockstep needs to apply inputs in an agreed order.
/// Errors past [`PlayerId`]'s `u8` range rather than wrapping two endpoints onto one
/// id (this game is couch-scale, never close). Called from [`form_match`], so both
/// entrypoints assign ids identically.
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

    /// Deterministic distinct endpoint ids: derived from a secret-key seed because a raw
    /// byte pattern isn't a valid public key (see the same helper in `membership`).
    fn eid(i: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[i; 32]).public()
    }

    #[test]
    fn assign_player_ids_is_identical_regardless_of_roster_order() {
        // The determinism crux at the net_loop layer: two peers handed the SAME agreed
        // set in DIFFERENT order must produce the identical endpoint→PlayerId map (the
        // barrier guarantees the same set; this guarantees the same assignment over it).
        let me = eid(2);
        let a = assign_player_ids(me, &[eid(1), eid(3), eid(2)]).unwrap();
        let b = assign_player_ids(me, &[eid(3), eid(2), eid(1)]).unwrap();
        assert_eq!(a, b, "id assignment must not depend on input order");
        // PlayerIds are 0..n in endpoint-id BYTE order (not seed order — real public
        // keys sort arbitrarily). Verify the assignment IS that sorted order.
        let mut ids = vec![eid(1), eid(2), eid(3)];
        ids.sort_by(|x, y| x.as_bytes().cmp(y.as_bytes()));
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(a[id], PlayerId(i as u8), "id at sort position {i}");
        }
    }

    #[test]
    fn assign_player_ids_dedups_self_in_roster() {
        // The roster from the barrier already includes us; assign must not double-count
        // when `me` appears in it (it pushes `me` then dedups).
        let me = eid(5);
        let map = assign_player_ids(me, &[eid(5), eid(7)]).unwrap();
        assert_eq!(map.len(), 2, "self must not be double-counted");
        // The two distinct ids get the two distinct PlayerIds 0 and 1 (in byte order).
        let mut got: Vec<PlayerId> = vec![map[&eid(5)], map[&eid(7)]];
        got.sort();
        assert_eq!(got, vec![PlayerId(0), PlayerId(1)]);
    }

    /// Live multi-endpoint formation over REAL iroh QUIC (not the pure state machine):
    /// three sessions on the loopback, fully meshed by direct dial, run the actual
    /// [`form_match`] barrier concurrently and MUST all freeze the byte-identical
    /// participant→PlayerId map. This is the end-to-end proof of the rl#44 invariant —
    /// the determinism-critical guarantee that every peer agrees on the same roster —
    /// exercised through the genuine wire (beats encoded/decoded, heartbeats, the
    /// stability barrier), not just the unit-tested core.
    ///
    /// A THIRD peer is started a beat late (a staggered join) to prove the barrier waits
    /// it in rather than freezing {A,B} first. `#[ignore]` because it binds real UDP
    /// sockets and takes a couple seconds (the STABLE_FOR settle); run with
    /// `cargo test --lib net::net_loop -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "binds real iroh UDP endpoints; run explicitly with --ignored"]
    async fn three_endpoints_form_the_identical_match_over_iroh() {
        use std::collections::BTreeMap;

        // Start three real sessions.
        let mut s0 = transport::start_session().await.expect("start s0");
        let mut s1 = transport::start_session().await.expect("start s1");
        let mut s2 = transport::start_session().await.expect("start s2");
        let (a0, a1, a2) = (s0.local_addr(), s1.local_addr(), s2.local_addr());
        let (e0, e1, e2) = (s0.endpoint_id(), s1.endpoint_id(), s2.endpoint_id());

        // Mesh them by DIRECT dial (bypass mDNS, which is flaky/slow in CI). Each pair
        // needs exactly one dial; wire_connection orients the stream by the id tie-break
        // internally, so dialing either direction yields one correct link. We dial A→B
        // for each unordered pair. s2 joins LATE (after the others have started beating)
        // to exercise the staggered-join path.
        s0.connect_direct(a1.clone()).await.expect("s0->s1");

        // Run all three barriers concurrently. s2's dials are issued from inside its
        // future after a short delay, so it shows up mid-formation.
        let f0 = form_match(&mut s0, 1, 3, None);
        let f1 = form_match(&mut s1, 1, 3, None);
        let f2 = async {
            // Stagger: let s0/s1 form their partial view first, then s2 meshes in.
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
            s2.connect_direct(a0.clone()).await.expect("s2->s0");
            s2.connect_direct(a1.clone()).await.expect("s2->s1");
            form_match(&mut s2, 1, 3, None).await
        };
        let (r0, r1, r2) = tokio::join!(f0, f1, f2);
        let (r0, r1, r2) = (
            r0.expect("s0 forms"),
            r1.expect("s1 forms"),
            r2.expect("s2 forms"),
        );

        // All three froze the SAME endpoint→PlayerId map — the whole point.
        assert_eq!(r0.id_map, r1.id_map, "s0 and s1 must agree on the roster");
        assert_eq!(r1.id_map, r2.id_map, "s1 and s2 must agree on the roster");
        assert_eq!(r0.id_map.len(), 3, "all three endpoints in the match");

        // And the map is exactly the three ids sorted → PlayerId 0,1,2 (the canonical
        // assignment every peer computes), so each peer also knows itself correctly.
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
        // Each peer's `me` is its own id's position.
        assert_eq!(r0.me, expected[&e0]);
        assert_eq!(r1.me, expected[&e1]);
        assert_eq!(r2.me, expected[&e2]);

        s0.shutdown().await;
        s1.shutdown().await;
        s2.shutdown().await;
    }
}
