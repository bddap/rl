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
//! BOTH the windowed client ([`connect_and_form`]) and the headless `game net` driver.
//! It's the same code on every peer on purpose: the agreed set + id assignment must be
//! byte-identical or the sims silently desync. The two callers differ only in what they
//! do with the session afterward — wrap it in a [`NetDriver`] for the Bevy loop, or
//! drive it raw in the headless async loop — not in the cold-start itself.
//!
//! Solo auto-fallback (rl#47): the common launch is ALONE — one process, no other peer
//! on the LAN (a player opens the Steam shortcut). Forming a networked match then never
//! reaches agreement (the barrier rightly refuses to freeze a guessed roster), leaving a
//! frozen, unplayable round. So the cold-start has a SECOND outcome: if the discovery
//! window elapses with no other peer ever heard ([`Formation::Alone`]), the caller starts
//! a deterministic [`crate::net::render::InputSource::Solo`] round instead of awaiting an
//! empty match. The fallback fires ONLY in the genuinely-alone case (see [`run_barrier`]);
//! the moment ANY peer is present the full membership-agreement barrier is back in force,
//! so real multiplayer is never weakened into a premature solo.

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

/// The outcome of [`connect_and_form`]: either we joined a networked match (a ready
/// [`Lockstep`] + the [`NetDriver`] pumping its transport) or the discovery window
/// elapsed with no other peer present, so the caller should run solo. An enum, not a
/// bare `Option<NetDriver>`, so "no driver" can't be confused with "telemetry off" or
/// any other absence — `Alone` means exactly "play offline" and nothing else.
pub enum MatchResult {
    /// A networked match formed: the agreed [`Lockstep`] and the driver for its peers.
    /// Boxed because both payloads are large and `Alone` is empty: without the box the
    /// enum is sized to `Joined`, so every `Alone` carries that dead weight too.
    Joined(Box<(Lockstep, NetDriver)>),
    /// Discovery completed with only us on the LAN (no peer ever beat us within the
    /// window). The caller starts a deterministic solo round instead of awaiting an
    /// empty networked match — see the module-level rl#47 note.
    Alone,
}

/// Bind the LAN endpoint, run the shared [`form_match`] cold-start (the membership
/// barrier), and either return a ready [`Lockstep`] + the [`NetDriver`] that pumps its
/// transport ([`MatchResult::Joined`]) — the windowed client's entry into a match — or,
/// when the discovery window elapses with no other peer present, [`MatchResult::Alone`]
/// so the caller falls back to a solo round (rl#47). The match `seed` must be identical
/// on every peer (the caller passes the shared constant). `expect` is the minimum
/// participant count to close on (see [`form_match`]); `discover_secs` bounds how long we
/// wait for a peer before concluding we are alone.
///
/// Pure mDNS discovery (no explicit dial). The boot-menu Join-by-code path uses
/// [`connect_and_form_dialing`] to additionally direct-dial a host's endpoint id; this is
/// that with `dial == None`, kept as the name every existing caller already uses.
pub fn connect_and_form(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    collector: Option<iroh::EndpointId>,
) -> Result<MatchResult> {
    connect_and_form_dialing(seed, discover_secs, expect, None, collector, None)
}

/// [`connect_and_form`] plus an optional direct dial of a host's endpoint id before the
/// barrier runs — the boot-menu (rl#56) Join-by-code path. `dial == Some(host)` opens a
/// QUIC link to `host` (its LAN address resolved via the endpoint's registered mDNS
/// lookup, so a bare id is enough on the local network) so formation has a peer even when
/// mDNS discovery is slow/missed; `dial == None` is the plain mDNS path
/// ([`connect_and_form`]).
///
/// Determinism is untouched by the dial: it only establishes a connection. The roster
/// still comes wholly from the [`form_match`] barrier — every peer freezes the identical
/// sorted set — so dialing the wrong/typo'd code simply fails to form a match (the
/// barrier never hears an agreeing peer and falls back to [`MatchResult::Alone`] or
/// errors), it can NEVER form a divergent roster. If the dial itself fails (bad code, host
/// gone) we log and proceed to the barrier anyway, which then resolves alone/failed — a
/// dial error must not be more fatal than an absent peer.
///
/// `on_bound` (if any) is sent our own endpoint id the instant the session binds — before
/// the slow barrier — so a Host UI can display the join code to share while waiting. A
/// closed receiver is ignored (the caller stopped caring); it never gates formation.
pub fn connect_and_form_dialing(
    seed: u64,
    discover_secs: u64,
    expect: usize,
    dial: Option<iroh::EndpointId>,
    collector: Option<iroh::EndpointId>,
    on_bound: Option<std::sync::mpsc::Sender<iroh::EndpointId>>,
) -> Result<MatchResult> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (session, formation, telemetry) = rt.block_on(async {
        let mut session = transport::start_session().await?;
        let my_eid = session.endpoint_id();
        println!("fp client endpoint id: {my_eid}");
        // Report our bound id up front (best-effort) so a Host lobby can show the join code
        // immediately, without waiting out the barrier.
        if let Some(tx) = &on_bound {
            let _ = tx.send(my_eid);
        }
        // Join-by-code: direct-dial the host so the barrier has a peer without waiting on
        // mDNS. Best-effort — a failed dial (bad code / host absent) is logged and we fall
        // through to the barrier, which resolves the absence as Alone/Failed just like an
        // unreachable peer. Never read a roster from the dial; it only opens the link.
        if let Some(host) = dial {
            if host == my_eid {
                tracing::warn!("join code is our own endpoint id — ignoring the self-dial");
            } else if let Err(e) = session.connect_direct(host).await {
                tracing::warn!("dialing host {} failed: {e:#}", host.fmt_short());
            }
        }
        // Open the telemetry side-channel BEFORE forming the match, so the collector
        // sees the roster fill (RosterForming/Agreed). Best-effort: a failure to bind
        // the telemetry endpoint just runs the game without it.
        let telemetry = connect_telemetry(collector, my_eid).await;
        let formation = form_match(&mut session, discover_secs, expect, telemetry.as_ref()).await?;
        anyhow::Ok((session, formation, telemetry))
    })?;

    let frozen = match formation {
        Formation::Agreed(frozen) => frozen,
        // No peer showed up: drop the session/telemetry (the runtime + endpoint tear down
        // on drop) and tell the caller to play offline.
        Formation::Alone => return Ok(MatchResult::Alone),
    };

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
    Ok(MatchResult::Joined(Box::new((ls, driver))))
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

/// What [`form_match`] resolves to: a real agreed match, or the genuinely-alone case
/// (rl#47). [`Formation::Alone`] is returned ONLY when the discovery window elapsed with
/// no other peer ever heard — never when peers are present-but-not-yet-agreed (that still
/// drives the full barrier to [`crate::net::membership::Status::Agreed`] or errors). The
/// caller maps `Alone` to a solo offline round.
pub enum Formation {
    /// The membership barrier agreed on a roster; play networked over it.
    Agreed(Frozen),
    /// Discovery completed with only us live; play solo (see the module rl#47 note).
    Alone,
}

/// Run the LAN match-formation barrier ([`run_barrier`]), then freeze the AGREED
/// participant set and assign deterministic [`PlayerId`]s by sorted endpoint id. The
/// ONE cold-start both the windowed client ([`connect_and_form`]) and the headless
/// `game net` driver run — shared verbatim (not just described as shared) because it
/// must stay behaviorally identical on every peer for the sims to agree, and a drift
/// here would silently desync.
///
/// `expect` is the minimum participant count (incl. us) the barrier requires before it
/// will close — it stops a peer from freezing a lone `{self}` match before LAN discovery
/// has even found the others, and makes a too-small turnout time out ([`Status::Failed`])
/// rather than form a short match. `discover_secs` is the alone-fallback deadline (rl#47):
/// if it elapses with no other peer ever heard, this returns [`Formation::Alone`] for a
/// solo round. It does NOT bound a formation that has peers — once any peer is present the
/// barrier waits for agreement within its own [`crate::net::membership::JOIN_WINDOW`].
pub async fn form_match(
    session: &mut transport::Session,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
) -> Result<Formation> {
    let my_eid = session.endpoint_id();
    println!(
        "forming match on the LAN (need {expect} player(s), solo if alone after {discover_secs}s)…"
    );

    let outcome = match run_barrier(session, my_eid, discover_secs, expect, telemetry).await {
        Ok(BarrierResult::Agreed(o)) => o,
        Ok(BarrierResult::Alone) => {
            // Discovery elapsed with only us live — fall back to solo (rl#47). No
            // RosterAgreed/Failed event: no networked match formed, so the collector
            // shows neither; the caller runs an offline round.
            println!("no other peer found within {discover_secs}s — starting a solo round");
            return Ok(Formation::Alone);
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
            roster_hash: crate::net::membership::roster_hash(&outcome.roster),
            me: me.0,
        });
    }
    Ok(Formation::Agreed(Frozen {
        id_map,
        me,
        early: outcome.early,
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
}

/// The non-error outcomes of [`run_barrier`]: a real agreement, or the alone fallback
/// (rl#47). `Alone` is distinct from the `Err` path on purpose — being alone is a normal,
/// expected launch (play solo), not a formation failure (relaunch).
enum BarrierResult {
    Agreed(BarrierOutcome),
    Alone,
}

/// Drive the membership barrier to agreement: beat our view every [`BEAT_EVERY`],
/// ingest peers' beats (and stash any early ticks), and poll the [`Membership`] state
/// machine until it returns [`Status::Agreed`] (freeze) or [`Status::Failed`] (give up
/// and error — never freeze a guessed set). `expect` is the minimum participant count to
/// close. The pure agreement logic lives in [`crate::net::membership`]; this is only the
/// I/O around it.
///
/// rl#47 solo fallback: when `expect > 1` (the default networked launch), we have NEVER
/// heard a beat from any peer, and `discover_secs` elapses while we are STILL the only live
/// member (`live == 1`), return [`BarrierResult::Alone`] so the caller plays solo. This is
/// deliberately layered ON TOP of the agreement core, not inside it: `Membership` stays
/// pure and never freezes a guessed roster, and the fallback can NEVER abort a formation
/// that has peers. The check reads `live` AFTER `poll` (which expires the stale and
/// recomputes the live set), so a peer mid-handshake whose beat has landed already shows as
/// `live >= 2` and holds us on the real barrier.
///
/// Two guards make "alone" mean genuinely alone, not "lost a peer":
/// - `expect == 1` is exempt — that's a DELIBERATE solo-over-network run, where the barrier
///   itself agrees on `{self}` after [`crate::net::membership::STABLE_FOR`]; the fallback
///   must not preempt it (the alone deadline can be shorter than STABLE_FOR). So it's gated
///   to `expect > 1`, exactly the "defaulted to networked but launched alone" case.
/// - We solo ONLY if `ever_heard_peer` is false. If we DID hear a peer this formation but
///   it then went silent (an asymmetric/one-way link, a peer that crashed mid-handshake),
///   that's a connection FAILURE, not the alone case: we stay on the barrier and surface
///   the loud `Status::Failed` ("relaunch together") rather than silently splitting into a
///   solo round. This closes the skewed-two-peer regression a reviewer flagged — a peer we
///   ever reached must not be silently dropped into single-player.
///
/// What this does NOT solve (product calls, not barrier bugs):
/// - If two co-launched peers NEVER hear each other within their windows (both genuinely
///   see an empty LAN — slow/no mDNS), both solo independently. That residual is inherent
///   to a unilateral solo decision with no "we agree nobody's here" exchange; `discover_secs`
///   is the knob that shrinks it, and rl#47's intent ("one launcher, always playable")
///   favors playing solo over a hard fail when discovery genuinely finds nobody.
/// - A stale endpoint that beats ONCE then dies latches `ever_heard_peer`, so a genuinely-
///   alone launch next to a lingering phantom gets the loud `Failed` (relaunch) instead of
///   a solo round. Conservative by design — the same "heard ⇒ not alone" stance that closes
///   the heard-then-lost split — and rare (it needs another endpoint to have been on the LAN
///   recently); distinguishing a phantom from a real lost peer would reopen that split.
///
/// `discover_secs` MUST stay below [`crate::net::membership::JOIN_WINDOW`]: the fallback can
/// only fire while `poll` still returns `Forming`, so a `discover_secs >= JOIN_WINDOW` would
/// let a genuinely-alone peer hit `Failed` first and never solo. The defaults (4s vs 20s)
/// hold this with wide margin.
async fn run_barrier(
    session: &mut Session,
    me: EndpointId,
    discover_secs: u64,
    expect: usize,
    telemetry: Option<&TelemetrySender>,
) -> Result<BarrierResult> {
    let start = Instant::now();
    let mut m = Membership::new(me, expect, start);
    let mut early: Vec<(EndpointId, TickMsg)> = Vec::new();
    let mut ticker = tokio::time::interval(BEAT_EVERY);
    let mut last_live = 0usize;
    // Whether we've EVER received a direct beat from any peer this formation. Gates the solo
    // fallback: heard-then-lost is a link failure (loud `Failed`), only never-heard is solo.
    let mut ever_heard_peer = false;
    // Deadline past which "still only us" means play solo. `.max(1)` so a `discover_secs`
    // of 0 can't declare us alone before discovery has run a single beat. (The `expect > 1`
    // and never-heard-a-peer gates that keep this from preempting a deliberate solo-over-
    // network run or silently dropping a lost peer live in `is_alone_now`.)
    let alone_deadline = start + Duration::from_secs(discover_secs.max(1));

    loop {
        ticker.tick().await;
        let now = Instant::now();

        // Ingest everything the transport has: beats feed the membership machine;
        // ticks are from a peer that already finished forming — hold them to replay
        // once we have a Lockstep (the transport doesn't resend, so dropping them would
        // stall us on that peer's first inputs).
        while let Some(from) = session.try_recv() {
            match from.msg {
                PeerWire::Beat(beat) => {
                    // A direct beat from a peer (never ourselves) — we are not alone, now
                    // or retroactively. Latch it so a peer that later goes silent fails
                    // loud instead of falling back to solo.
                    if from.from != me {
                        ever_heard_peer = true;
                    }
                    m.on_beat(from.from, &beat, now);
                }
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
                return Ok(BarrierResult::Agreed(BarrierOutcome {
                    roster,
                    early,
                    elapsed: now.duration_since(start),
                }));
            }
            Status::Failed => {
                anyhow::bail!(
                    "match formation failed: peers never agreed on one roster within the join window \
                     (too few players showed up, or a peer kept appearing/disappearing, or a link is \
                     one-way). Relaunch together."
                );
            }
            Status::Forming { live } => {
                // Solo fallback (rl#47): the window elapsed and we are STILL alone and have
                // never heard a peer. `live` is fresh from the `poll` above (post-expiry),
                // and any peer that ever beat us directly makes `live >= 2` AND latches
                // `ever_heard_peer`, so this is the genuinely-empty-LAN case — never a peer
                // mid-handshake or a peer we lost. Real multiplayer keeps the full barrier.
                if is_alone_now(expect, live, ever_heard_peer, now >= alone_deadline) {
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

/// The rl#47 solo-fallback policy as a pure predicate, so the (race-sensitive) decision is
/// unit-testable without a socket or the async clock. Returns true iff we should stop
/// awaiting a networked match and play solo. ALL must hold:
/// - `expect > 1` — a defaulted networked launch, not a deliberate `expect == 1`
///   solo-over-network (which forms `{self}` via the barrier and must not be preempted).
/// - `!ever_heard_peer` — we have NEVER heard a beat from any peer this formation. A peer
///   heard then lost is a link failure (the loud `Failed`/relaunch), not the alone case.
/// - `live == 1` — only us is live right now (a fresh post-`poll` count; any present peer
///   makes this `>= 2` and keeps the full membership barrier in force).
/// - `past_deadline` — the discovery window has elapsed (the caller computes `now >=
///   deadline`; passing the comparison result, not two same-typed `Instant`s, removes the
///   silently-invertible transposition of `now`/`deadline`).
fn is_alone_now(expect: usize, live: usize, ever_heard_peer: bool, past_deadline: bool) -> bool {
    expect > 1 && !ever_heard_peer && live == 1 && past_deadline
}

/// Build the single-peer lockstep for an OFFLINE round (just us), honoring the
/// `RL_VEHICLE` pilot flag. The one definition shared by every offline entry into the
/// windowed client — the explicit `play --solo`, the boot-menu Solo / Host-Start-alone
/// buttons (rl#56), and the rl#47 discovery-found-no-peer fallback — so they all play the
/// byte-identical deterministic solo round with no second construction to drift. `seed`
/// is the shared match seed (the caller passes the one constant every peer uses).
pub fn solo_lockstep_for(seed: u64) -> Lockstep {
    let me = PlayerId(0);
    let pilots = pilots_from_env(me);
    Lockstep::new_with_pilots(seed, &[me], me, &pilots)
}

/// Which players spawn PILOTING a plane rather than on foot, from the `RL_VEHICLE` env
/// flag (rl#38 vehicle first cut). `RL_VEHICLE=plane` makes the LOCAL player (`me`) a
/// pilot; anything else (incl. unset) is the unchanged foot game (empty ⇒ byte-identical
/// sim). Offline paths only: the networked path ([`form_match`]) builds the session with
/// no pilots, so it ignores `RL_VEHICLE` entirely — wiring pilots over the wire needs the
/// peers to agree on the pilot set (a wire negotiation), which is future work.
fn pilots_from_env(me: PlayerId) -> Vec<PlayerId> {
    match std::env::var("RL_VEHICLE").as_deref() {
        Ok("plane") => vec![me],
        _ => Vec::new(),
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
        let (a0, a1) = (s0.local_addr(), s1.local_addr());
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
        // Each peer must AGREE (not fall back to solo — they all see each other), so unwrap
        // the `Formation::Agreed`; an `Alone` here would be a barrier bug (peers present).
        let unwrap_agreed = |r: Result<Formation>, who: &str| match r.expect(who) {
            Formation::Agreed(f) => f,
            Formation::Alone => panic!("{who}: fell back to solo despite peers being present"),
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

    /// The rl#47 solo-fallback decision, exhaustively and deterministically (no socket, no
    /// real clock — the policy is the pure [`is_alone_now`] predicate). Each of the four
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
}
