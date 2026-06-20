//! Synchronous bridge from the async iroh [`transport`] to a Bevy/game main loop.
//!
//! [`transport::Session`] is async (tokio); the deterministic lockstep driver and
//! the Bevy render loop are sync and own the main thread. [`NetDriver`] bridges the
//! two: it holds a tokio runtime + the session and exposes two non-blocking calls a
//! per-frame system can use — [`NetDriver::broadcast`] to fan our tick message out
//! to peers, and [`NetDriver::drain_inbox`] to pull whatever peer messages have
//! arrived. No determinism lives here; it is pure I/O plumbing (the same split the
//! netcode already draws between [`transport`] and [`crate::net::lockstep`]).
//!
//! The LAN cold-start — discover peers for a bit, freeze the participant set,
//! [`assign_player_ids`] by sorted id, collect any inputs that arrived mid-discovery —
//! is [`discover_and_freeze`], shared by BOTH the windowed client
//! ([`connect_and_assign`]) and the headless `game net` driver. It's the same code on
//! every peer on purpose: the freeze + id assignment must be byte-identical or the
//! sims silently desync. The two callers differ only in what they do with the session
//! afterward — wrap it in a [`NetDriver`] for the Bevy loop, or drive it raw in the
//! headless async loop — not in the cold-start itself.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

use crate::net::lockstep::{Lockstep, TickMsg};
use crate::net::sim::PlayerId;
use crate::net::transport::{self, Session};

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
    /// id set. Used to tag inbound messages with their author's id.
    id_map: BTreeMap<EndpointId, PlayerId>,
}

impl NetDriver {
    /// Broadcast our tick message to every connected peer. Non-blocking from the
    /// caller's view: it drives the async `broadcast` to completion on the runtime,
    /// which only does buffered QUIC writes (a dead peer is dropped inside, not
    /// awaited — the lockstep stall on the missing input is the visible failure).
    pub fn broadcast(&self, msg: &TickMsg) {
        self.rt.block_on(self.session.broadcast(msg));
    }

    /// Drain every peer message received so far, tagged with the sender's
    /// [`PlayerId`]. Non-blocking: `try_recv` returns immediately when the inbox is
    /// empty. Messages from an endpoint not in the frozen set are dropped (a peer
    /// that joined after the freeze isn't part of this match).
    pub fn drain_inbox(&mut self) -> Vec<PeerMsg> {
        let mut out = Vec::new();
        while let Some(from) = self.session.try_recv() {
            if let Some(&pid) = self.id_map.get(&from.from) {
                out.push(PeerMsg { pid, msg: from.msg });
            }
        }
        out
    }
}

/// Bind the LAN endpoint, run the shared [`discover_and_freeze`] cold-start, replay
/// the inputs that arrived during discovery, and return a ready [`Lockstep`] + the
/// [`NetDriver`] that pumps its transport — the windowed client's entry into a match.
/// The match `seed` must be identical on every peer (the caller passes the shared
/// constant).
pub fn connect_and_assign(
    seed: u64,
    discover_secs: u64,
    expect: usize,
) -> Result<(Lockstep, NetDriver)> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let (session, frozen) = rt.block_on(async {
        let mut session = transport::start_session().await?;
        println!("fp client endpoint id: {}", session.endpoint_id());
        let frozen = discover_and_freeze(&mut session, discover_secs, expect).await?;
        anyhow::Ok((session, frozen))
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
    };
    Ok((ls, driver))
}

/// The outcome of the LAN cold-start: the frozen participant→[`PlayerId`] map, which
/// id is us, and the tick messages that arrived mid-discovery (to be replayed once a
/// [`Lockstep`] exists — see [`replay_early`]).
pub struct Frozen {
    pub id_map: BTreeMap<EndpointId, PlayerId>,
    pub me: PlayerId,
    pub early: Vec<transport::FromPeer>,
}

/// Discover LAN peers for `discover_secs`, freeze the participant set, and assign
/// deterministic [`PlayerId`]s. The ONE cold-start both the windowed client
/// ([`connect_and_assign`]) and the headless `game net` driver run — shared verbatim
/// (not just described as shared) because it must stay behaviorally identical on every
/// peer for the sims to agree, and a drift here would silently desync. Both callers
/// `.await` it; they differ only in what they do with the `session` afterward (wrap it
/// in a [`NetDriver`] vs drive it raw), which is why the cold-start is the function and
/// the post-freeze handling is not.
pub async fn discover_and_freeze(
    session: &mut transport::Session,
    discover_secs: u64,
    expect: usize,
) -> Result<Frozen> {
    let my_eid = session.endpoint_id();
    println!("discovering peers on the LAN for {discover_secs}s…");

    // Poll the connected-peer set until we reach `expect` or time out. COLLECT early
    // tick messages into a holding buffer (don't drop them): a peer that finished its
    // cold-start first may already be broadcasting inputs for future ticks while we're
    // still forming the set, and the transport doesn't resend — so they must be replayed
    // into the lockstep once ids exist, or that peer stalls us waiting for inputs we
    // threw away.
    let mut early: Vec<transport::FromPeer> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(discover_secs);
    loop {
        while let Some(m) = session.try_recv() {
            early.push(m);
        }
        let n = session.connected_peers().await.len() + 1; // +1 = us
        if n >= expect || Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Brief settle so a peer that reached the set a beat later registers its link too.
    // Best-effort couch cold-start, NOT a real membership barrier (a peer joining after
    // the freeze is ignored; a positional PlayerId over divergent sets would desync —
    // Phase 1 must add a real barrier).
    tokio::time::sleep(Duration::from_millis(500)).await;
    while let Some(m) = session.try_recv() {
        early.push(m);
    }

    // Freeze the set: us + everyone we connected to, sorted by endpoint id. Every peer
    // computes the SAME sorted order over the SAME id set, so a given endpoint is the
    // same PlayerId everywhere — the precondition for the sims to agree.
    let peer_eids = session.connected_peers().await;
    let id_map = assign_player_ids(my_eid, &peer_eids)?;
    let me = id_map[&my_eid];
    Ok(Frozen { id_map, me, early })
}

/// Replay the inputs that arrived during discovery into a freshly-built [`Lockstep`],
/// now that ids are assigned. They predate any applied tick, so `record_remote`
/// stashes them rather than comparing a hash — no fault is possible, hence the
/// discard. Unknown senders are dropped (the same filter the live loop applies).
pub fn replay_early(ls: &mut Lockstep, frozen: &Frozen) {
    for m in &frozen.early {
        if let Some(&pid) = frozen.id_map.get(&m.from) {
            let _ = ls.record_remote(pid, m.msg);
        }
    }
}

/// Map endpoint ids → [`PlayerId`]s by sorting the full id set (us + peers). Every
/// peer sorts the identical set, so a given endpoint is the same `PlayerId`
/// everywhere — the precondition lockstep needs to apply inputs in an agreed order.
/// Errors past [`PlayerId`]'s `u8` range rather than wrapping two endpoints onto one
/// id (this game is couch-scale, never close). Called from [`discover_and_freeze`], so
/// both entrypoints assign ids identically.
pub fn assign_player_ids(
    me: EndpointId,
    peers: &[EndpointId],
) -> Result<BTreeMap<EndpointId, PlayerId>> {
    let mut all: Vec<EndpointId> = peers.to_vec();
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
