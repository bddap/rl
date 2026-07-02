//! `net`: networked headless run — discover peers over iroh and run the host-authoritative tick loop.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use iroh::EndpointId;
use net::lockstep::Lockstep;
use net::sim::{Input, PlayerId, TICK_DT, TICK_HZ};
use net::telemetry::{TELEMETRY_TICK_EVERY, TelemetryEvent};
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
/// endpoint-id set, then tick the host-authoritative loop — the host steps its sim and
/// broadcasts a snapshot each tick; clients ship input up and adopt it — and report progress.
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
    // Mutable: a departed peer (link gone, rl#198) is removed so the host stops requiring its
    // input — the live roster of record past this point is the server's schedule.
    let mut id_map = frozen.id_map.clone();
    let all_ids: Vec<PlayerId> = id_map.values().copied().collect();
    println!(
        "starting lockstep: {} player(s), I am {:?} ({})",
        all_ids.len(),
        me,
        my_eid.fmt_short()
    );

    // Every peer spawns the identical foot-only round.
    let mut ls = Lockstep::new(super::shared::MATCH_SEED, &all_ids, me);

    // Host-authoritative play (rl#151): the lowest-id peer (PlayerId 0) runs the AUTHORITATIVE match
    // server; the rest are remote clients that ADOPT its snapshots. Solo (a single peer) is the same
    // path with a roster of one ([[sp-is-mp-special-case]]). The `Server` is the SAME type the
    // windowed driver runs — only the async-vs-sync transport plumbing differs (headless awaits the
    // session directly; the Bevy client drives it through `NetDriver`/`Coordinator`). Inputs flow UP
    // as `TickMsg`s; the server steps its own sim and broadcasts the authoritative `CoreSnapshot`
    // DOWN — the only world state on the wire, adopted whole (no re-sim, no peer cross-check).
    let am_host = me == PlayerId(0);
    let server_eid = *id_map
        .iter()
        .find(|(_, pid)| **pid == PlayerId(0))
        .map(|(eid, _)| eid)
        .expect("a frozen roster always contains PlayerId(0)");
    let mut server = am_host.then(|| {
        // The host owns and steps the authoritative sim (rl#151): seed it from the identical tick-0
        // world `ls` was built from so host and clients start byte-identical. The local `ls` is the
        // host's own client — it files input UP and adopts the snapshots, never stepping itself.
        let mut s = net::server::Server::new(&all_ids, ls.sim().clone());
        s.seed_early(&net_loop::early_peer_msgs(&frozen));
        s
    });

    let mut ticker = tokio::time::interval(Duration::from_secs_f64(TICK_DT));
    let end = Instant::now() + Duration::from_secs(args.run_secs);
    // How many authoritative snapshots this peer put on / took off the wire (rl#151 increment 2):
    // the host counts what it BROADCAST, a client what it ADOPTED. Surfaced in the `done:` line so a
    // 2-process proof can see the client is state-fed by the host, never re-stepping a sim.
    let mut snapshots_io = 0usize;
    // Coarse human progress: print roughly once per second of sim. This samples the FIRST
    // tick at/after each boundary, which a batched catch-up can overshoot by a tick or two — so
    // these lines are a liveness/hash eyeball, NOT a byte-exact cross-peer compare. The
    // authoritative cross-machine determinism proof is the per-tick `--hash-log` (keyed by the
    // true tick): host and client log the identical tick→hash line for every tick both applied.
    let mut next_report_tick = TICK_HZ;
    // Telemetry-side sampling cursor (independent of the stdout report cadence) and a
    // one-shot latch so RoundDecided is reported exactly once.
    let mut next_tel_tick = TELEMETRY_TICK_EVERY;
    let mut reported_outcome = false;
    // Optional per-tick hash log (Args::hash_log): every applied tick keyed by its true
    // tick, so two peers' logs diff byte-identically over their overlap — the cross-machine
    // determinism proof. Written one line per stepped/adopted tick below.
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
        // `TickMsg`s. As a client: the host's authoritative `CoreSnapshot`s (rl#151 increment 2 —
        // STATE down, not the input set: the client adopts it whole and never re-steps). A stray
        // barrier beat from a peer still winding down formation is ignored either way.
        let mut remote_inputs: Vec<net_loop::PeerMsg> = Vec::new();
        let mut snapshots: Vec<net::snapshot::CoreSnapshot> = Vec::new();
        while let Some(m) = session.try_recv() {
            match m.msg {
                transport::PeerWire::Tick(msg) => {
                    if let Some(&pid) = id_map.get(&m.from) {
                        remote_inputs.push(net_loop::PeerMsg { pid, msg });
                    }
                }
                transport::PeerWire::Snapshot(snap) => {
                    if !am_host {
                        snapshots.push(snap);
                    }
                }
                // Render-only crab pose (rl#151 increment 2 windowed): this HEADLESS harness renders
                // nothing, so it decodes and drops it. Only the windowed client applies it.
                transport::PeerWire::Articulation(_) => {}
                transport::PeerWire::Beat(_) => {}
                // This is a FIXED-roster run: the peer set is frozen at discovery and never
                // grows, so the Stage 3 live-join frames (a joiner's credentials, a welcome,
                // a refusal) can't legitimately arrive here. Ignore a stray one rather
                // than mishandle it — the same stance `net_loop`'s formation barrier takes; a
                // real mid-match join is the running coordinator's job, not this harness.
                transport::PeerWire::JoinRequest(_)
                | transport::PeerWire::Welcome(_)
                | transport::PeerWire::Refuse(_) => {}
            }
        }

        // Issue our input and route it through the server (rl#151 increment 2 — host-authoritative:
        // inputs go UP, STATE comes DOWN). The host assembles via the SAME `host_assemble` the
        // windowed `Coordinator` uses (one coordination impl, two transports), steps its own
        // authoritative sim, and broadcasts a `CoreSnapshot` per applied tick; a client ships its
        // input up and ADOPTS the host's snapshots without ever re-stepping the sim.
        let t = ls.next_tick() as f32 * 0.1;
        let issue_tick = ls.next_tick();
        let input = Input::from_axes(t.cos(), t.sin());
        let msg = ls.submit_local_input(input);

        if let Some(srv) = server.as_mut() {
            // HOST: fold in remote clients' inputs into the authoritative server's ledger, then step
            // its OWN sim once per ready tick and broadcast that tick's snapshot the instant it is
            // stepped — the SAME host-authoritative path the windowed driver runs (one stepper). A
            // client adopts exactly the state the host holds; there is no peer-symmetric self-step or
            // desync cross-check any more (the host IS the source of truth).
            let mut sets = net::server::host_assemble(srv, me, msg, remote_inputs);
            // Departures (rl#198), after this tick's drained inputs are recorded — the same
            // predicate the windowed host runs (`departed_players`): a rostered peer whose link is
            // gone left the match; drop it and keep ticking rather than waiting on its input
            // forever (the host-freeze hang).
            let connected = session.connected_peers().await;
            for (eid, pid) in net_loop::departed_players(&id_map, me, &connected) {
                id_map.remove(&eid);
                println!(
                    "player {pid:?} ({}) departed — continuing without them",
                    eid.fmt_short()
                );
                sets.extend(srv.depart(pid));
            }
            srv.enqueue_for_step(&sets);
            // Headless: weights digest 0, no rapier body → the crab holds spawn, so no pose to inject.
            while srv.next_tick_ready() {
                let bytes = srv.step_next(None);
                let snap = net::snapshot::CoreSnapshot::from_bytes(&bytes)
                    .expect("the authoritative server's snapshot must decode");
                session.broadcast_snapshot(&snap).await;
                snapshots_io += 1;
                // The local `ls` adopts the same snapshot so `ls.sim()` mirrors the authoritative
                // world for the report + telemetry below (exactly as the windowed local client does).
                ls.apply_core_snapshot(snap);
                if let Some(w) = hash_log.as_mut() {
                    use std::io::Write as _;
                    // Log the just-stepped tick (post-step count minus one) + its closing hash, so a
                    // host and client diff byte-identically — the same keying the client arm uses.
                    let applied = srv.sim().tick().saturating_sub(1);
                    writeln!(w, "{} {:#018x}", applied, srv.sim().state_hash())
                        .context("writing hash log")?;
                }
            }
        } else {
            // CLIENT: ship our input up, then ADOPT the host's authoritative snapshots — no re-sim,
            // no cross-check (the host IS the source of truth) — via the ONE shared client adopt
            // policy ([`net::lockstep::Lockstep::adopt_snapshots`]: arrival order, no tick gate/sort
            // — see its doc for the restart-freeze rationale).
            session.send_to(server_eid, &msg).await;
            // The adopt callback can't `?`; collect the per-adopt observations and write them after,
            // where the error propagates through normal control flow.
            let mut adopted_hashes: Vec<(u64, u64)> = Vec::new();
            snapshots_io += ls.adopt_snapshots(snapshots, |ls| {
                if hash_log.is_some() {
                    // The host logs the just-stepped tick index (its `sim().tick() - 1`); the snapshot
                    // we just adopted carries the POST-step count (`sim.tick()`), one higher — so log
                    // `tick - 1` to line up the two peers' logs for a byte-identical `diff`.
                    adopted_hashes.push((ls.sim().tick().saturating_sub(1), ls.sim().state_hash()));
                }
            });
            if let Some(w) = hash_log.as_mut() {
                use std::io::Write as _;
                for (applied, hash) in adopted_hashes {
                    writeln!(w, "{} {:#018x}", applied, hash).context("writing hash log")?;
                }
            }
        }

        // Coarse progress print once the sim crosses each TICK_HZ boundary (see the
        // cadence note above — a batched advance can overshoot the boundary tick, so these
        // are not byte-comparable across peers; the `--hash-log` is).
        if ls.sim().tick() >= next_report_tick {
            next_report_tick = (ls.sim().tick() / TICK_HZ + 1) * TICK_HZ;
            println!(
                "tick={:>5} peers={} statehash={:#018x}",
                ls.sim().tick(),
                session.connected_peers().await.len(),
                ls.sim().state_hash(),
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
                t.send(TelemetryEvent::tick(ls.sim(), all_ids.len()));
                t.send(TelemetryEvent::input(issue_tick, input));
            }
            if !reported_outcome && ls.sim().outcome() != net::sim::Outcome::Ongoing {
                reported_outcome = true;
                t.send(TelemetryEvent::round_decided(ls.sim()));
            }
        }
    }

    println!(
        "done: {} ticks applied, {} snapshots {}, final hash {:#018x}",
        ls.sim().tick(),
        snapshots_io,
        if am_host { "broadcast" } else { "adopted" },
        ls.sim().state_hash()
    );
    // A final snapshot so the collector records where this deck ended even if the round
    // never "decided" within run_secs (the common case for a short headless run).
    if let Some(t) = tel.as_ref() {
        t.send(TelemetryEvent::tick(ls.sim(), all_ids.len()));
    }
    if all_ids.len() > 1
        && ls.sim().tick() < (args.run_secs * TICK_HZ).saturating_sub(TICK_HZ)
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
