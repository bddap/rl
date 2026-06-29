//! `net`: networked headless run — discover peers over iroh and run the lockstep loop.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use iroh::EndpointId;
use net::lockstep::{INPUT_DELAY, Lockstep};
use net::sim::{Input, PlayerId, TICK_HZ};
use net::telemetry::{TELEMETRY_TICK_EVERY, TelemetryEvent, TelemetrySender};
use net::{net_loop, transport};

use super::shared::run_solo_round;

#[derive(Parser)]
pub(crate) struct Args {
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
    /// Stream live telemetry to the collector with this endpoint id (from
    /// `game telemetry-collector`). Opens a SEPARATE iroh connection on a distinct ALPN
    /// — the lockstep transport/determinism is untouched, and a telemetry failure never
    /// affects the match. Omit to run with no telemetry.
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,
    /// Write a full per-tick `<tick> <state_hash>` log to this file (every applied tick,
    /// keyed by the true tick — unlike the coarse stdout report cadence). Two peers (or two
    /// machines) running the SAME match must produce logs that `diff` byte-identically over
    /// their overlapping tick range: the cross-peer (and cross-machine) determinism proof
    /// (rl#94). Omit for no log.
    #[arg(long, value_name = "FILE")]
    hash_log: Option<std::path::PathBuf>,
}

/// Build the tokio runtime the networked path needs and run it. `main` stays a plain `fn` and
/// each async mode owns its runtime explicitly — see the runtime note in `main`.
pub(crate) fn run(args: Args) -> Result<()> {
    tokio::runtime::Runtime::new()?.block_on(run_net(args))
}

/// Networked run: bind, discover, assign deterministic player ids from the sorted
/// endpoint-id set, then tick lockstep — broadcasting our input and ingesting peers'
/// each tick — and report whether we stayed in sync.
async fn run_net(args: Args) -> Result<()> {
    let mut session = transport::start_session().await?;
    let my_eid = session.endpoint_id();
    println!("game endpoint id: {my_eid}");

    // Open the telemetry side-channel (if configured) BEFORE forming the match, so the
    // collector sees the roster fill. Best-effort + isolated: separate iroh endpoint,
    // separate ALPN — see `net::telemetry`. A failure yields `None` and the run is
    // byte-for-byte the no-telemetry run.
    let tel = net_loop::connect_telemetry(args.telemetry, my_eid).await;

    // Form one agreed match via the shared cold-start barrier (same code the windowed
    // client runs, so the two can't drift apart and desync). Replay any inputs that
    // arrived during formation into the fresh sim. If discovery finds no peer, tear down
    // the network side and run a solo round instead of awaiting an empty match.
    let frozen = match net_loop::form_match(
        &mut session,
        args.discover_secs,
        args.expect,
        tel.as_ref(),
        None, // headless: timer-closed barrier, no interactive lobby
        0, // headless has no rapier-NN crab stack → 0 weights digest; the crab holds spawn (rl#114)
        crab_world::bot::meshfit::crab_asset_digest(), // honest crab-asset digest (rl#100)
    )
    .await?
    {
        net_loop::Formation::Agreed(frozen) => frozen,
        net_loop::Formation::Alone => {
            drop(tel);
            session.shutdown().await;
            return run_solo_round(args.run_secs);
        }
        // No cancel channel on the headless path, so a Cancel can never be signalled.
        net_loop::Formation::Cancelled => unreachable!("headless net has no lobby to cancel"),
    };
    let me = frozen.me;
    let id_map = &frozen.id_map;
    let all_ids: Vec<PlayerId> = id_map.values().copied().collect();
    println!(
        "starting lockstep: {} player(s), I am {:?} ({})",
        all_ids.len(),
        me,
        my_eid.fmt_short()
    );

    // Every peer spawns the identical foot-only round.
    let mut ls = Lockstep::new(super::shared::MATCH_SEED, &all_ids, me);

    // Server-coordinated play (rl#151): the lowest-id peer (PlayerId 0) runs the match server; the
    // rest are remote clients of it. Solo (a single peer) is the same path with a roster of one. The
    // Server core (the input ledger + completeness gating) is the SAME type the windowed client runs
    // — only the async-vs-sync transport plumbing differs (headless awaits the session directly; the
    // Bevy client drives it through `NetDriver`/`Coordinator`). Inputs flow UP as `TickMsg`s, the
    // server broadcasts the complete `TickSet` DOWN; world state never crosses the wire.
    let am_host = me == PlayerId(0);
    let server_eid = *id_map
        .iter()
        .find(|(_, pid)| **pid == PlayerId(0))
        .map(|(eid, _)| eid)
        .expect("a frozen roster always contains PlayerId(0)");
    let mut server = am_host.then(|| {
        let mut s = net::server::Server::new(&all_ids);
        s.seed_early(&net_loop::early_peer_msgs(&frozen));
        s
    });

    let tick_dt = Duration::from_secs_f64(1.0 / TICK_HZ as f64);
    let mut ticker = tokio::time::interval(tick_dt);
    let end = Instant::now() + Duration::from_secs(args.run_secs);
    let mut total_desyncs = 0usize;
    // Coarse human progress: print roughly once per second of sim. This samples the FIRST
    // tick at/after each boundary, which a batched `try_advance` can overshoot by a tick or
    // two — so these lines are a liveness/hash eyeball, NOT a byte-exact cross-peer compare.
    // The authoritative cross-peer determinism proofs are the internal desync cross-check
    // (peer-advertised hashes) and the per-tick `--hash-log` (keyed by the true tick).
    let mut next_report_tick = TICK_HZ;
    // Telemetry-side sampling cursor (independent of the stdout report cadence) and a
    // one-shot latch so RoundDecided is reported exactly once.
    let mut next_tel_tick = TELEMETRY_TICK_EVERY;
    let mut reported_outcome = false;
    // Optional per-tick hash log (Args::hash_log): every applied tick keyed by its true
    // tick, so two peers' logs diff byte-identically over their overlap — the cross-peer
    // (and cross-machine) determinism proof. Written one line per `advance_one` below.
    let mut hash_log = args
        .hash_log
        .as_ref()
        .map(|p| {
            std::fs::File::create(p)
                .map(std::io::BufWriter::new)
                .with_context(|| format!("creating hash log {}", p.display()))
        })
        .transpose()?;

    while Instant::now() < end {
        ticker.tick().await;

        // Ingest everything the transport has for us this tick. As the host: remote clients' input
        // `TickMsg`s. As a client: the server's assembled `TickSet`s. A stray barrier beat from a
        // peer still winding down formation is ignored either way.
        let mut remote_inputs: Vec<net_loop::PeerMsg> = Vec::new();
        let mut sets: Vec<net::server::TickSet> = Vec::new();
        while let Some(m) = session.try_recv() {
            match m.msg {
                transport::PeerWire::Tick(msg) => {
                    if let Some(&pid) = id_map.get(&m.from) {
                        remote_inputs.push(net_loop::PeerMsg { pid, msg });
                    }
                }
                transport::PeerWire::TickSet(set) => {
                    if !am_host {
                        sets.push(set);
                    }
                }
                transport::PeerWire::Beat(_) => {}
                // This is a FIXED-roster run: the peer set is frozen at discovery and never
                // grows, so the Stage 3 live-join frames (a joiner's credentials, a roster
                // change, a refusal) can't legitimately arrive here. Ignore a stray one rather
                // than mishandle it — the same stance `net_loop`'s formation barrier takes; a
                // real mid-match join is the running coordinator's job, not this harness.
                transport::PeerWire::JoinRequest(_)
                | transport::PeerWire::RosterChange(_)
                | transport::PeerWire::Welcome(_)
                | transport::PeerWire::Refuse(_) => {}
            }
        }

        // Issue our input and route it through the server. The host assembles via the SAME
        // `host_assemble` the windowed `Coordinator` uses (one coordination impl, two transports);
        // a client ships its input up and unpacks the sets it drained.
        let t = ls.next_tick() as f32 * 0.1;
        let issue_tick = ls.next_tick();
        let input = Input::from_axes(t.cos(), t.sin());
        let msg = ls.submit_local_input(input);
        let peer_msgs: Vec<net_loop::PeerMsg> = if let Some(srv) = server.as_mut() {
            let (out_sets, peer_msgs) = net::server::host_assemble(srv, me, msg, remote_inputs);
            for s in &out_sets {
                session.broadcast_tickset(s).await;
            }
            peer_msgs
        } else {
            session.send_to(server_eid, &msg).await;
            sets.iter()
                .flat_map(|s| net::server::unpack_tickset(s, me))
                .collect()
        };

        // Record the OTHER players' inputs — the same `record_remote` entry a mesh peer used to take,
        // so the cross-check + advance below are unchanged. A late hash for an already-applied tick
        // can surface a fault here.
        for pm in peer_msgs {
            if let Some(f) = ls.record_remote(pm.pid, pm.msg) {
                report_fault(&mut total_desyncs, f, tel.as_ref());
            }
        }

        // Advance every ready tick ONE AT A TIME so the hash log can record each tick's
        // closing hash at the instant it's applied — `try_advance` is exactly this loop, but
        // logging from its post-batch snapshot could miss a tick the batch already pruned.
        // Logging per `advance_one` writes every applied tick exactly once, regardless of how
        // many a single iteration catches up.
        while let Some(faults) = ls.advance_one() {
            for f in faults {
                report_fault(&mut total_desyncs, f, tel.as_ref());
            }
            if let Some((w, c)) = hash_log.as_mut().zip(ls.last_applied()) {
                use std::io::Write as _;
                writeln!(w, "{} {:#018x}", c.tick, c.hash).context("writing hash log")?;
            }
        }

        // Coarse progress print once the sim crosses each TICK_HZ boundary (see the
        // cadence note above — a batched advance can overshoot the boundary tick, so these
        // are not byte-comparable across peers; the `--hash-log` is).
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

        // Sampled telemetry: a Tick snapshot (+ the input we just issued) every
        // TELEMETRY_TICK_EVERY ticks, and a one-shot RoundDecided when the round ends.
        // All read-only on the sim; all best-effort (a send that can't keep up drops).
        if let Some(t) = tel.as_ref() {
            if ls.sim().tick() >= next_tel_tick {
                next_tel_tick = (ls.sim().tick() / TELEMETRY_TICK_EVERY + 1) * TELEMETRY_TICK_EVERY;
                // Agreed roster size (us + peers) — the same quantity render.rs and the
                // final snapshot report, so the feed's `roster` field means one thing
                // across every driver.
                t.send(TelemetryEvent::tick(ls.sim(), total_desyncs, all_ids.len()));
                t.send(TelemetryEvent::input(issue_tick, input));
            }
            if !reported_outcome && ls.sim().outcome() != net::sim::Outcome::Ongoing {
                reported_outcome = true;
                t.send(TelemetryEvent::round_decided(ls.sim()));
            }
        }
    }

    println!(
        "done: {} ticks applied, {} desyncs, final hash {:#018x}",
        ls.sim().tick(),
        total_desyncs,
        ls.sim().state_hash()
    );
    // A final snapshot so the collector records where this deck ended even if the round
    // never "decided" within run_secs (the common case for a short headless run).
    if let Some(t) = tel.as_ref() {
        t.send(TelemetryEvent::tick(ls.sim(), total_desyncs, all_ids.len()));
    }
    if all_ids.len() > 1
        && ls.sim().tick() < (args.run_secs * TICK_HZ).saturating_sub(INPUT_DELAY + TICK_HZ)
    {
        // We applied far fewer ticks than wall time allowed → we spent the run
        // stalled waiting for a peer's input. Flag it; a healthy link keeps pace.
        eprintln!(
            "WARNING: only {} ticks in {}s — peer link stalled (missing inputs)",
            ls.sim().tick(),
            args.run_secs
        );
    }
    // Give the best-effort telemetry queue a moment to flush its tail before the
    // process tears down the endpoint (the sender task drains on its own runtime). A
    // no-op when telemetry is off.
    if tel.is_some() {
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    if let Some(mut w) = hash_log.take() {
        use std::io::Write as _;
        w.flush().context("flushing hash log")?; // surface a write error, don't swallow it on drop
    }
    drop(tel); // close the telemetry channel so its task finishes the stream cleanly
    session.shutdown().await;
    Ok(())
}

/// Count and log a cross-check fault. A desync is unrecoverable in lockstep, but we keep
/// running so the test harness can observe how many ticks faulted rather than aborting on
/// the first. Also mirrored to telemetry (best-effort) so a remote operator sees the
/// divergence the instant a deck does.
fn report_fault(total: &mut usize, f: net::lockstep::Fault, telemetry: Option<&TelemetrySender>) {
    use net::lockstep::Fault;
    *total += 1;
    if let Some(t) = telemetry {
        t.send(TelemetryEvent::fault(&f));
    }
    match f {
        Fault::Desync {
            tick,
            peer,
            local_hash,
            peer_hash,
        } => eprintln!(
            "DESYNC at tick {tick}: peer {peer:?} hash {peer_hash:#018x} != local {local_hash:#018x}"
        ),
        Fault::Unverifiable {
            tick,
            peer,
            peer_hash,
        } => eprintln!(
            "UNVERIFIABLE at tick {tick}: peer {peer:?} hash {peer_hash:#018x} fell out of our history window"
        ),
    }
}
