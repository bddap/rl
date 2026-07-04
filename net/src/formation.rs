//! The LAN cold-start: match formation, the membership barrier I/O, and the solo fallback.
//!
//! The cold-start ([`form_match`]) — run the membership barrier, [`assign_player_ids`]
//! over the agreed set by sorted id, and collect tick inputs that arrived mid-formation —
//! is shared verbatim by the windowed client ([`crate::net_loop::connect_and_form`]) and the
//! headless `game net` driver. Same code on every peer on purpose: the agreed set + id
//! assignment MUST be byte-identical or inputs and snapshots are silently mis-attributed
//! across peers. The callers differ only in
//! what they do with the session after (wrap it in a [`crate::net_loop::NetDriver`], or drive
//! it raw).
//!
//! Solo auto-fallback: the common launch is ALONE (one process, no LAN peer), and the
//! barrier rightly refuses to freeze a guessed roster — so the cold-start has a SECOND
//! outcome ([`Formation::Alone`]): play a deterministic solo round. The exact policy is
//! the [`is_alone_now`] / [`is_alone_at_timeout`] predicates; the moment any peer is
//! present the full agreement barrier is in force, so multiplayer is never weakened.
//!
//! The pure agreement logic lives in [`crate::membership`]; this module is the I/O around
//! it, plus the solo and lobby policies layered on top. The [`crate::net_loop::NetDriver`]
//! transport bridge that pumps the formed match lives in [`crate::net_loop`].

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

use crate::lockstep::{Lockstep, PeerMsg, TickMsg};
use crate::membership::{BEAT_EVERY, Membership, Role, Status};
use crate::sim::PlayerId;
use crate::telemetry::{self, TelemetryEvent, TelemetrySender};
use crate::transport::{self, PeerWire, Session};

/// The host-triggered lobby: the [`Role`] (which decides the close trigger) plus the live
/// control + observation channels. Passing `Some(LobbyControl)` to the formation core IS
/// what selects the lobby barrier; `None` is the default timer-closed barrier. So the
/// determinism-critical mode is an explicit sum (`Option<LobbyControl>`), never inferred from
/// "did some channel happen to be wired".
pub struct LobbyControl {
    /// Host vs joiner. The host commands the synchronized start; a joiner waits for the GO.
    /// Threaded into [`Membership::host_triggered`] so the role is enforced by the type.
    pub role: Role,
    /// A signal that the player clicked **Start**. On the first signal the barrier calls
    /// [`Membership::set_starting`] — which is a structural no-op for a [`Role::Joiner`], so a
    /// joiner can't command a start even if this fires.
    pub start_rx: std::sync::mpsc::Receiver<()>,
    /// A signal that the player clicked **Cancel** (Host or Join). The barrier returns
    /// [`MatchResult::Cancelled`] promptly and tears the session down, so leaving the lobby
    /// doesn't strand a ~12 s LAN phantom.
    pub cancel_rx: std::sync::mpsc::Receiver<()>,
    /// A feed of the current live roster (us + peers, sorted) emitted each beat, for the
    /// lobby's live player list. Best-effort: a full/closed channel just drops the update.
    pub roster_tx: std::sync::mpsc::Sender<Vec<EndpointId>>,
}

/// The outcome of the LAN cold-start: the frozen participant→[`PlayerId`] map (agreed
/// identical on every peer by the barrier), which id is us, and the tick messages that
/// arrived mid-formation (carried to the host to seed its server input streams — see
/// [`crate::net_loop::Coordinator::for_round`] — since only the server holds them).
pub struct Frozen {
    pub id_map: BTreeMap<EndpointId, PlayerId>,
    pub me: PlayerId,
    pub early: Vec<(EndpointId, TickMsg)>,
    /// The barrier's shared-asset verdict, carried out so
    /// [`crate::net_loop::NetDriver::sync_verdict`] can expose it to the arm sites; see
    /// [`Membership::sync_verdict`].
    pub sync: crate::SyncVerdict,
}

/// What [`form_match`] resolves to: a real agreed match, the genuinely-alone case, or a
/// lobby cancel.
pub enum Formation {
    /// The membership barrier agreed on a roster; play networked over it.
    Agreed(Frozen),
    /// Formation ended with only us live — play solo. Fires only in the genuinely-alone
    /// case: see [`is_alone_now`] / [`is_alone_at_timeout`].
    Alone,
    /// The player cancelled the host-triggered lobby before a match formed. The caller tears
    /// the session down and reports [`MatchResult::Cancelled`].
    Cancelled,
}

/// Run the LAN match-formation barrier ([`run_barrier`]), then freeze the AGREED
/// participant set and assign deterministic [`PlayerId`]s by sorted endpoint id — the
/// ONE cold-start both the windowed client ([`connect_and_form`]) and the headless
/// `game net` driver run (shared verbatim; see the module docs for why).
///
/// `expect` is the minimum participant count (incl. us) the barrier requires before it
/// will close — it stops a peer from freezing a lone `{self}` match before LAN discovery
/// has even found the others, and makes a too-small turnout time out ([`Status::Failed`])
/// rather than form a short match. `discover_secs` is the alone-fallback deadline; it
/// does NOT bound a formation that has peers — once any peer is present the barrier
/// waits for agreement within its own [`crate::membership::JOIN_WINDOW`].
pub async fn form_match(
    session: &mut transport::Session,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
    lobby: Option<&LobbyControl>,
    local_weights_digest: u64,
    local_asset_digest: u64,
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
        local_weights_digest,
        local_asset_digest,
    )
    .await
    {
        Ok(BarrierResult::Agreed(o)) => o,
        Ok(BarrierResult::Alone) => {
            // Discovery elapsed with only us live — fall back to solo. No
            // RosterAgreed/Failed event: no networked match formed, so the collector
            // shows neither; the caller runs an offline round.
            println!("no other peer found within {discover_secs}s — starting a solo round");
            return Ok(Formation::Alone);
        }
        Ok(BarrierResult::Cancelled) => {
            // The player left the lobby. No telemetry event — no match formed and no
            // failure; the caller tears down the session and returns to the menu.
            println!("lobby cancelled by the player");
            return Ok(Formation::Cancelled);
        }
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
            roster_hash: crate::membership::roster_hash(&outcome.roster),
            me: me.0,
        });
    }
    // GCR shared-asset guard: make the verdict LOUD. With a checkpoint loaded
    // (`local_weights_digest != 0`) the operator needs to know whether the NN crab will arm —
    // the HOST must verifiably run a real brain AND the crab asset must match on every peer;
    // failing either means it WON'T, and with no integer fallback the windowed client REFUSES
    // the round rather than substituting a fake crab. The asset verdict is reported whenever
    // the host gate passes (so an asset-only mismatch is diagnosable, never silent).
    if local_weights_digest != 0 {
        if !outcome.sync.host_brain {
            tracing::warn!(
                "GCR: the HOST is not verifiably running the real Sally (its advertised weights \
                 digest is 0 — a failed/absent checkpoint — or it was never heard directly) — \
                 cannot arm the NN crab; the windowed client will REFUSE this round (rl#114, no \
                 integer fallback). Run rl-update on the host device, or relaunch together."
            );
        } else if !outcome.sync.assets {
            tracing::warn!(
                "GCR: host brain verified but crab MODEL ASSET NOT synced across peers (a peer \
                 has a different sally.glb / no model — it would build and render a different \
                 crab) — cannot arm the NN crab; the windowed client will REFUSE this round \
                 (rl#114, no integer fallback). Run rl-update on every device so all carry the \
                 identical crab model."
            );
        } else {
            println!(
                "GCR: host runs the real Sally AND the crab asset is synced across all {} \
                 peer(s) — NN crab eligible for lockstep",
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

/// What a successful [`run_barrier`] yields: the agreed roster (sorted endpoint ids,
/// identical on every peer), the tick messages that arrived mid-formation (a peer that
/// finished the barrier first may already be broadcasting inputs), and how long agreement
/// took.
struct BarrierOutcome {
    roster: Vec<EndpointId>,
    early: Vec<(EndpointId, TickMsg)>,
    elapsed: Duration,
    /// [`Membership::sync_verdict`] sampled at the close instant.
    sync: crate::SyncVerdict,
}

/// The non-error outcomes of [`run_barrier`]: a real agreement, the alone fallback, or a
/// user cancel. All three are distinct from the `Err` path on purpose — being alone is a
/// normal launch (play solo), cancelling is a deliberate exit, and only a genuine formation
/// failure is an error (relaunch).
enum BarrierResult {
    Agreed(BarrierOutcome),
    Alone,
    /// The player clicked Cancel in the host-triggered lobby; leave the barrier promptly so
    /// the session can be torn down before a LAN phantom forms.
    Cancelled,
}

/// Drive the membership barrier to agreement: beat our view every [`BEAT_EVERY`],
/// ingest peers' beats (and stash any early ticks), and poll the [`Membership`] state
/// machine until it returns [`Status::Agreed`] (freeze) or [`Status::Failed`] (give up
/// and error — never freeze a guessed set). `expect` is the minimum participant count to
/// close. The pure agreement logic lives in [`crate::membership`]; this is only the
/// I/O around it, plus the solo and lobby policies layered ON TOP (so `Membership` stays
/// pure and never itself freezes a guessed roster or aborts a formation that has peers):
///
/// - Solo fallback: when defaulted-networked and genuinely alone, return
///   [`BarrierResult::Alone`] so the caller plays solo. The exact predicate (and why
///   "alone" must mean never-heard-a-peer, not lost-a-peer) is [`is_alone_now`] for the
///   `Forming` deadline and [`is_alone_at_timeout`] for the `Failed` window expiry; both
///   read `live` AFTER `poll`, so a peer mid-handshake holds us on the real barrier.
/// - Lobby (`lobby == Some`): the barrier closes on the host's GO, not the timer. Each beat
///   we call [`Membership::set_starting`] once `start_rx` fires, return
///   [`BarrierResult::Cancelled`] the instant `cancel_rx` fires (so the session tears down
///   before a LAN phantom forms), and push the live roster to `roster_tx`. The solo
///   fallback is SKIPPED here — a lone host starts via the UI's instant-solo path and never
///   reaches this barrier, so being here means the host means to wait (open-ended).
///
/// Residual (a product call, not a bug): two co-launched peers that NEVER hear each other
/// within their windows both solo independently — inherent to a unilateral solo decision
/// with no "we agree nobody's here" exchange. `discover_secs` shrinks the window.
#[allow(clippy::too_many_arguments)] // each arg is a distinct formation knob.
async fn run_barrier(
    session: &mut Session,
    me: EndpointId,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
    lobby: Option<&LobbyControl>,
    local_weights_digest: u64,
    local_asset_digest: u64,
) -> Result<BarrierResult> {
    let start = Instant::now();
    // `Some(control)` is the interactive lobby (host-triggered close per its `role`); `None`
    // is the default timer barrier. The mode is this explicit choice, never inferred. Our
    // weights and crab-asset digests ride every beat so peers can agree on a shared
    // checkpoint AND a shared collider asset before arming the float NN crab.
    let mut m = match lobby {
        Some(c) => Membership::host_triggered(c.role, me, expect, start),
        None => Membership::new(me, expect, start),
    }
    .with_weights_digest(local_weights_digest)
    .with_asset_digest(local_asset_digest);
    let mut early: Vec<(EndpointId, TickMsg)> = Vec::new();
    let mut ticker = tokio::time::interval(BEAT_EVERY);
    let mut last_live = 0usize;
    let mut last_roster: Vec<EndpointId> = Vec::new();
    // Whether we've EVER received a direct beat from any peer this formation — gates the
    // solo fallback ([`is_alone_now`]).
    let mut ever_heard_peer = false;
    // Deadline past which "still only us" means play solo. `.max(1)` so a `discover_secs`
    // of 0 can't declare us alone before discovery has run a single beat.
    let alone_deadline = start + Duration::from_secs(discover_secs.max(1));

    loop {
        ticker.tick().await;
        let now = Instant::now();

        // Lobby controls, checked before forming: a Cancel ends the barrier NOW; a
        // Start latches the host into commanding the GO (a no-op for a joiner — `set_starting`
        // is structurally inert off [`Role::Host`]). `try_recv` is non-blocking; a closed
        // cancel channel (the UI dropped its sender) is treated as a cancel — the lobby is gone.
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

        // Ingest everything the transport has: beats feed the membership machine;
        // ticks are from a peer that already finished forming — hold them to replay
        // once we have a Lockstep (the transport doesn't resend, so dropping them would
        // stall us on that peer's first inputs).
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
                // No server exists yet during formation, so a snapshot / admission /
                // refusal can't legitimately arrive here; ignore a stray one (a peer racing ahead,
                // or a mid-game join frame on the wrong phase) rather than mishandle it. A real
                // mid-game join is handled by the running coordinator, not the formation barrier.
                PeerWire::Snapshot(_)
                | PeerWire::Articulation(_)
                | PeerWire::JoinRequest(_)
                | PeerWire::Refuse(_)
                | PeerWire::Welcome(_) => {}
            }
        }

        // Poll FIRST (expires stale peers + advances the verdict), so the beat we then
        // advertise reflects the freshly-pruned set — we never gossip a just-expired
        // phantom for an extra round.
        let status = m.poll(now);
        session.broadcast(&m.beat()).await;

        // Push the live roster to the lobby UI when it changes. Best-effort — a
        // closed/full channel just drops it. The set is the membership's own `live_set`,
        // so the lobby shows exactly what would freeze.
        if let Some(c) = lobby {
            let roster = m.live_set();
            if roster != last_roster {
                let _ = c.roster_tx.send(roster.clone());
                last_roster = roster;
            }
        }

        match status {
            Status::Agreed { roster } => {
                // Sample the weights verdict at the close instant — `poll` (above) just expired
                // the dead, so the live set this reflects is exactly the frozen `roster`.
                return Ok(BarrierResult::Agreed(BarrierOutcome {
                    roster,
                    early,
                    elapsed: now.duration_since(start),
                    sync: m.sync_verdict(),
                }));
            }
            Status::Failed => {
                // The JOIN_WINDOW elapsed without a closed roster. On the DEFAULT
                // (non-lobby) path, alone-at-expiry plays solo rather than stranding the
                // player ([`is_alone_at_timeout`]); `live >= 2` stays the loud failure.
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

/// The solo-fallback policy as a pure predicate, so the (race-sensitive) decision is
/// unit-testable without a socket or the async clock. Returns true iff we should stop
/// awaiting a networked match and play solo. ALL must hold:
/// - `expect > 1` — a defaulted networked launch, not a deliberate `expect == 1`
///   solo-over-network (which forms `{self}` via the barrier and must not be preempted).
/// - `!ever_heard_peer` — we have NEVER heard a beat from any peer this formation. A peer
///   heard then lost is a link FAILURE (the loud `Failed`/relaunch), not the alone case —
///   the guard that stops a one-way link silently splitting into single-player.
/// - `live == 1` — only us is live right now (a fresh post-`poll` count; any present peer
///   makes this `>= 2` and keeps the full membership barrier in force).
/// - `past_deadline` — the discovery window has elapsed (the caller passes the comparison
///   result, not two same-typed `Instant`s, removing a silently-invertible transposition).
fn is_alone_now(expect: usize, live: usize, ever_heard_peer: bool, past_deadline: bool) -> bool {
    expect > 1 && !ever_heard_peer && live == 1 && past_deadline
}

/// The LAST-RESORT alone-fallback, applied when the barrier already reached
/// [`crate::membership::Status::Failed`] (the JOIN_WINDOW elapsed without a closed
/// roster) on the default (non-lobby) path. Returns true iff we should solo rather than
/// error. Unlike [`is_alone_now`] it does NOT consult `ever_heard_peer`: at the window's
/// END a once-heard peer that has since expired leaves us at `live == 1`, and stranding a
/// lone player with an error helps no one — so a phantom flicker solos here (just later
/// than a never-heard launch). No deadline either: reaching `Failed` already means the
/// window expired. The `live >= 2` case is excluded so it stays the loud `Failed` — real
/// peers present that never agreed is a genuine multi-peer fault, not the alone case.
fn is_alone_at_timeout(expect: usize, live: usize) -> bool {
    expect > 1 && live == 1
}

/// Build the single-peer lockstep for an OFFLINE round (just us). The one definition shared
/// by every offline entry into the windowed client — the boot-menu Host-alone Start, the
/// scripted `--host`/`--join` alone outcome, and the discovery-found-no-peer fallback — so
/// they all play the byte-identical deterministic solo round with no second construction to
/// drift. `seed` is the shared match seed (the caller passes the one constant every peer
/// uses). The player always starts ON FOOT; piloting a plane is reached in-game via the
/// client's enter/exit toggle ([`crate::render`]).
pub fn solo_lockstep_for(seed: u64) -> Lockstep {
    let me = PlayerId(0);
    Lockstep::new(seed, &[me], me)
}

/// The inputs that arrived during formation, mapped to their author's [`PlayerId`] (senders not in
/// the agreed set dropped). The host seeds its server input streams with these (via
/// [`Server::seed_early`]) so a fast client's pre-serve inputs aren't lost; everyone else discards
/// them (only the server holds the streams). `pub` so the headless `game net` driver builds the
/// same set from its `Frozen`.
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
        let mut ids = [eid(1), eid(2), eid(3)];
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

    #[test]
    fn player_zero_is_host_of() {
        // Cross-lock the two spellings of the host rule: `assign_player_ids`' PlayerId(0)
        // (the sorted assignment's first slot) must be exactly the peer
        // `membership::host_of` picks, or "host" and "server" could name different peers.
        let roster = [eid(9), eid(3), eid(6)];
        let map = assign_player_ids(eid(3), &roster).unwrap();
        let host = crate::membership::host_of(&roster);
        assert_eq!(map[&host], PlayerId(0), "host_of must hold PlayerId(0)");
    }

    /// Live multi-endpoint formation over REAL iroh QUIC (not the pure state machine):
    /// three sessions on the loopback, fully meshed by direct dial, run the actual
    /// [`form_match`] barrier concurrently and MUST all freeze the byte-identical
    /// participant→PlayerId map. The end-to-end proof of the determinism-critical guarantee
    /// that every peer agrees on the same roster — exercised through the genuine wire (beats
    /// encoded/decoded, heartbeats, the stability barrier), not just the unit-tested core.
    ///
    /// A THIRD peer is started a beat late (a staggered join) to prove the barrier waits
    /// it in rather than freezing {A,B} first. `#[ignore]` because it binds real UDP
    /// sockets and takes a couple seconds (the STABLE_FOR settle); run with
    /// `cargo test --lib formation -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "binds real iroh UDP endpoints; run explicitly with --ignored"]
    async fn three_endpoints_form_the_identical_match_over_iroh() {
        use std::collections::BTreeMap;

        // Start three real sessions.
        let mut s0 = transport::start_session().await.expect("start s0");
        let mut s1 = transport::start_session().await.expect("start s1");
        let mut s2 = transport::start_session().await.expect("start s2");
        let (a0, a1) = (s0.local_addr(), s1.local_addr());
        let (e0, e1, e2) = (s0.endpoint_id(), s1.endpoint_id(), s2.endpoint_id());

        // Mesh them by DIRECT dial (bypass mDNS, which is flaky/slow in CI). Each pair
        // needs exactly one dial; wire_connection orients the stream by the id tie-break
        // internally, so dialing either direction yields one correct link. We dial A→B
        // for each unordered pair. s2 joins LATE (after the others have started beating)
        // to exercise the staggered-join path.
        s0.connect_direct(a1.clone()).await.expect("s0->s1");

        // Run all three barriers concurrently. s2's dials are issued from inside its
        // future after a short delay, so it shows up mid-formation. `None` lobby — this
        // exercises the unchanged timer-closed barrier.
        let f0 = form_match(&mut s0, 1, 3, None, None, 0, 0);
        let f1 = form_match(&mut s1, 1, 3, None, None, 0, 0);
        let f2 = async {
            // Stagger: let s0/s1 form their partial view first, then s2 meshes in.
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
            s2.connect_direct(a0.clone()).await.expect("s2->s0");
            s2.connect_direct(a1.clone()).await.expect("s2->s1");
            form_match(&mut s2, 1, 3, None, None, 0, 0).await
        };
        let (r0, r1, r2) = tokio::join!(f0, f1, f2);
        // Each peer must AGREE (not fall back to solo — they all see each other), so unwrap
        // the `Formation::Agreed`; an `Alone` here would be a barrier bug (peers present).
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

    /// The solo-fallback decision, exhaustively and deterministically (no socket, no real
    /// clock — the policy is the pure [`is_alone_now`] predicate). Each of the four
    /// conditions that must ALL hold is falsified in isolation to prove it's load-bearing.
    /// The cross-peer timing the predicate can't see (two real endpoints discovering each
    /// other) is left to the live `#[ignore]`d formation test; here we pin the policy.
    #[test]
    fn alone_fallback_fires_only_when_defaulted_networked_never_heard_and_truly_alone() {
        // (expect, live, ever_heard_peer, past_deadline).
        // The fallback case: defaulted networked (expect>1), never heard a peer, alone
        // (live==1), past the deadline.
        assert!(
            is_alone_now(2, 1, false, true),
            "defaulted-networked + never-heard + alone + past the window ⇒ solo"
        );

        // Each guard removed in turn must SUPPRESS the fallback:
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
        // And it stays suppressed for any larger live count.
        assert!(!is_alone_now(4, 3, false, true));
    }

    /// The LAST-RESORT timeout fallback (`is_alone_at_timeout`), applied when the barrier
    /// already hit `Status::Failed` (the JOIN_WINDOW expired). It solos iff we defaulted to
    /// networked (`expect > 1`) and are ALONE at expiry (`live == 1`) — regardless of
    /// `ever_heard_peer`, so a phantom that flickered then expired solos rather than
    /// erroring. A genuine multi-peer failure (`live >= 2`) stays the loud error, and a
    /// deliberate `expect == 1` solo-over-network is unaffected.
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
