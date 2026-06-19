//! `game` — the giant-crab rescue game (rl#38), built multiplayer-first on the
//! deterministic-lockstep + iroh netcode foundation (rl#39).
//!
//! Phase 0 (this binary) is the skeleton: a trivial sim (player dots on a plane)
//! driven by [`rl::net::lockstep`] over [`rl::net::transport`] (iroh LAN discovery).
//! It exists to prove the netcode — discovery, input exchange, deterministic tick,
//! desync detection — before any game content. Phase 1 (world + giant crab,
//! first-person player, plane/heli) replaces the trivial sim behind the same
//! interfaces.
//!
//! Modes:
//! - `net` (default headless): bind an iroh endpoint, discover peers on the LAN,
//!   and run the lockstep loop for a fixed duration, printing per-second sync state.
//!   Run two copies on a LAN to see them find each other and stay in sync.
//! - `solo`: run the lockstep+sim loop with no network (one peer), for a quick
//!   smoke of the tick machinery.
//!
//! The windowed/first-person Bevy client is Phase 1; keeping Phase 0 headless makes
//! the foundation testable without a GPU (this box renders headlessly at best).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, Subcommand};
use rl::net::lockstep::{Lockstep, INPUT_DELAY};
use rl::net::sim::{Input, PlayerId};
use rl::net::transport;

/// Tick rate of the deterministic sim. 30 Hz is plenty for Phase 0's dots and keeps
/// the lockstep stall window forgiving on a LAN; Phase 1 can raise it.
const TICK_HZ: u64 = 30;

#[derive(Parser)]
#[command(about = "Giant-crab rescue — Phase 0 netcode skeleton (rl#39)")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Networked headless run: discover LAN peers over iroh and run lockstep.
    Net(NetArgs),
    /// Single-peer headless smoke of the tick machinery (no network).
    Solo(SoloArgs),
}

#[derive(Parser)]
struct NetArgs {
    /// Wait this long for peers to be discovered before starting the tick loop.
    /// Discovery is mDNS, so a couple seconds covers a quiet LAN.
    #[arg(long, default_value_t = 4)]
    discover_secs: u64,
    /// Run the lockstep loop for this many seconds, then report and exit.
    #[arg(long, default_value_t = 10)]
    run_secs: u64,
    /// Expected peer count (including us). The loop waits up to `discover_secs` to
    /// reach it; if fewer are found it proceeds with whoever showed up (and a single
    /// peer simply runs solo over the network stack).
    #[arg(long, default_value_t = 2)]
    expect: usize,
}

#[derive(Parser)]
struct SoloArgs {
    #[arg(long, default_value_t = 5)]
    run_secs: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_writer(std::io::stderr).init();
    // No subcommand → the networked mode with its own defaults (parsed from an empty
    // arg list so the `default_value_t`s are the single source, not duplicated here).
    let command = Cli::parse()
        .command
        .unwrap_or_else(|| Command::Net(NetArgs::parse_from(["game"])));
    match command {
        Command::Net(args) => run_net(args).await,
        Command::Solo(args) => run_solo(args),
    }
}

/// Deterministic match seed. In Phase 0 it's a constant so independently-launched
/// peers agree without a handshake; Phase 1's session setup will negotiate it (the
/// lower-id peer proposes, say) — the sim already takes it as a parameter.
const MATCH_SEED: u64 = 0x6372_6162; // "crab"

/// Drive the lockstep sim from a constant local input, ticking at [`TICK_HZ`]. Pure
/// machinery check: no peers, so our own input completes every tick.
fn run_solo(args: SoloArgs) -> Result<()> {
    let me = PlayerId(0);
    let mut ls = Lockstep::new(MATCH_SEED, &[me], me);
    let tick_dt = Duration::from_secs_f64(1.0 / TICK_HZ as f64);
    let end = Instant::now() + Duration::from_secs(args.run_secs);
    let mut next = Instant::now();
    while Instant::now() < end {
        // A lazy circular stir so the dot visibly moves.
        let t = ls.next_tick() as f32 * 0.1;
        ls.submit_local_input(Input::from_axes(t.cos(), t.sin()));
        let desyncs = ls.try_advance();
        debug_assert!(desyncs.is_empty(), "solo can't desync");
        next += tick_dt;
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    let dot = ls.sim().dot(me).unwrap();
    println!(
        "solo: {} ticks, dot=({}, {}), hash={:#018x}",
        ls.sim().tick(),
        dot.x,
        dot.y,
        ls.sim().state_hash()
    );
    Ok(())
}

/// Networked run: bind, discover, assign deterministic player ids from the sorted
/// endpoint-id set, then tick lockstep — broadcasting our input and ingesting peers'
/// each tick — and report whether we stayed in sync.
async fn run_net(args: NetArgs) -> Result<()> {
    let mut session = transport::start_session().await?;
    let my_eid = session.endpoint_id();
    println!("game endpoint id: {my_eid}");
    println!("discovering peers on the LAN for {}s…", args.discover_secs);

    // Wait for discovery: poll the connected-peer set until we reach `expect` or time
    // out. Drain any early tick messages into a holding buffer so none are lost while
    // we're still forming the peer set (they're for future ticks anyway).
    let deadline = Instant::now() + Duration::from_secs(args.discover_secs);
    let mut early: Vec<transport::FromPeer> = Vec::new();
    loop {
        while let Some(m) = session.try_recv() {
            early.push(m);
        }
        let n = session.connected_peers().await.len() + 1; // +1 = us
        if n >= args.expect || Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Brief settle once we've seen our peers: the connection is mutual only after BOTH
    // sides finish the stream handshake, and they may reach `expect` a beat apart. A
    // short pause lets the lagging side register its link too, so both freeze the same
    // set. This is a best-effort cold-start for Phase 0's small couch sessions — NOT a
    // real membership barrier. Phase 1 MUST add one (all peers agree on an identical
    // frozen set + seed before tick 0): a positional PlayerId over divergent sets
    // silently desyncs, and a peer joining after the freeze is currently ignored.
    tokio::time::sleep(Duration::from_millis(500)).await;
    while let Some(m) = session.try_recv() {
        early.push(m);
    }

    // Freeze the participant set: us + everyone we connected to, sorted by endpoint
    // id. Every peer computes the SAME sorted order over the SAME id set, so the
    // PlayerId each endpoint maps to is identical on all peers — the precondition for
    // the sims to agree (see lockstep).
    let peer_eids = session.connected_peers().await;
    let id_map = assign_player_ids(my_eid, &peer_eids)?;
    let me = id_map[&my_eid];
    let all_ids: Vec<PlayerId> = id_map.values().copied().collect();
    println!(
        "starting lockstep: {} player(s), I am {:?} ({})",
        all_ids.len(),
        me,
        my_eid.fmt_short()
    );

    let mut ls = Lockstep::new(MATCH_SEED, &all_ids, me);

    // Replay any messages that arrived during discovery now that ids are assigned.
    // (These predate any applied tick, so record_remote stashes rather than compares.)
    for m in early {
        if let Some(&pid) = id_map.get(&m.from) {
            let _ = ls.record_remote(pid, m.msg);
        }
    }

    let tick_dt = Duration::from_secs_f64(1.0 / TICK_HZ as f64);
    let mut ticker = tokio::time::interval(tick_dt);
    let end = Instant::now() + Duration::from_secs(args.run_secs);
    let mut total_desyncs = 0usize;
    // Report at fixed TICK boundaries (not wall-clock) so both peers print the SAME
    // ticks — the `(tick, hash)` lines are then directly comparable across peers, an
    // external check on the internal desync cross-check.
    let mut next_report_tick = TICK_HZ;

    while Instant::now() < end {
        ticker.tick().await;

        // Ingest everything the transport has for us this tick. A late-arriving hash
        // for an already-applied tick can surface a fault right here.
        while let Some(m) = session.try_recv() {
            if let Some(&pid) = id_map.get(&m.from)
                && let Some(f) = ls.record_remote(pid, m.msg)
            {
                report_fault(&mut total_desyncs, f);
            }
        }

        // Issue our input for this tick and tell every peer.
        let t = ls.next_tick() as f32 * 0.1;
        let msg = ls.submit_local_input(Input::from_axes(t.cos(), t.sin()));
        session.broadcast(&msg).await;

        // Advance every ready tick; surface faults found as we apply.
        for f in ls.try_advance() {
            report_fault(&mut total_desyncs, f);
        }

        // Report once the sim crosses each TICK_HZ boundary. The label is the actual
        // current tick and the hash is that same tick's state, so the pair is exact;
        // both peers cross the same boundaries, making the lines comparable.
        if ls.sim().tick() >= next_report_tick {
            next_report_tick = (ls.sim().tick() / TICK_HZ + 1) * TICK_HZ;
            println!(
                "tick={:>5} peers={} statehash={:#018x} desyncs={}",
                ls.sim().tick(),
                session.connected_peers().await.len(),
                ls.sim().state_hash(),
                total_desyncs,
            );
        }
    }

    println!(
        "done: {} ticks applied, {} desyncs, final hash {:#018x}",
        ls.sim().tick(),
        total_desyncs,
        ls.sim().state_hash()
    );
    if all_ids.len() > 1 && ls.sim().tick() < (args.run_secs * TICK_HZ).saturating_sub(INPUT_DELAY + TICK_HZ) {
        // We applied far fewer ticks than wall time allowed → we spent the run
        // stalled waiting for a peer's input. Flag it; a healthy link keeps pace.
        eprintln!(
            "WARNING: only {} ticks in {}s — peer link stalled (missing inputs)",
            ls.sim().tick(),
            args.run_secs
        );
    }
    session.shutdown().await;
    Ok(())
}

/// Count and log a cross-check fault. A desync is unrecoverable in lockstep, but
/// Phase 0 keeps running so the test harness can observe how many ticks faulted
/// rather than aborting on the first.
fn report_fault(total: &mut usize, f: rl::net::lockstep::Fault) {
    use rl::net::lockstep::Fault;
    *total += 1;
    match f {
        Fault::Desync { tick, peer, local_hash, peer_hash } => eprintln!(
            "DESYNC at tick {tick}: peer {peer:?} hash {peer_hash:#018x} != local {local_hash:#018x}"
        ),
        Fault::Unverifiable { tick, peer, peer_hash } => eprintln!(
            "UNVERIFIABLE at tick {tick}: peer {peer:?} hash {peer_hash:#018x} fell out of our history window"
        ),
    }
}

/// Map endpoint ids → [`PlayerId`]s by sorting the full id set (us + peers). Because
/// every peer sorts the identical set, the mapping is identical everywhere, so a
/// given endpoint is the same `PlayerId` on all peers — exactly what lockstep needs
/// to apply inputs in an agreed order. Errors past [`PlayerId`]'s `u8` range rather
/// than wrapping two endpoints onto one id (this game is couch-scale, never close).
fn assign_player_ids(
    me: iroh::EndpointId,
    peers: &[iroh::EndpointId],
) -> Result<BTreeMap<iroh::EndpointId, PlayerId>> {
    let mut all: Vec<iroh::EndpointId> = peers.to_vec();
    all.push(me);
    all.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    all.dedup();
    anyhow::ensure!(all.len() <= u8::MAX as usize + 1, "too many players: {}", all.len());
    Ok(all
        .into_iter()
        .enumerate()
        .map(|(i, eid)| (eid, PlayerId(i as u8)))
        .collect())
}
