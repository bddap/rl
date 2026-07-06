
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use iroh::EndpointId;
use net::lockstep::Lockstep;
use net::sim::{Input, PlayerId, TICK_DT, TICK_HZ};
use net::telemetry::{TelemetryEvent, next_sample_tick};
use net::{formation, net_loop, transport};
use tracing::{info, warn};

use super::shared::run_solo_round;

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, default_value_t = super::shared::DEFAULT_DISCOVER_SECS)]
    discover_secs: u64,
    #[arg(long, default_value_t = 10)]
    run_secs: u64,
    #[arg(long, default_value_t = super::shared::DEFAULT_EXPECT)]
    expect: usize,
    #[arg(long, value_name = "COLLECTOR_ENDPOINT_ID")]
    telemetry: Option<EndpointId>,
    #[arg(long, value_name = "FILE")]
    hash_log: Option<std::path::PathBuf>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    tokio::runtime::Runtime::new()?.block_on(run_net(args))
}

async fn run_net(args: Args) -> Result<()> {
    let mut session = transport::start_session().await?;
    let my_eid = session.endpoint_id();
    info!("game endpoint id: {my_eid}");

    let tel = net_loop::connect_telemetry(args.telemetry, my_eid).await;

    let frozen = match formation::form_match(
        &mut session,
        args.discover_secs,
        args.expect,
        tel.as_ref(),
        None,
        crab_world::mesh_fallback::constructed_body_digest(), // honest crab-asset digest (rl#100)
        0,
    )
    .await?
    {
        formation::Formation::Agreed(frozen) => frozen,
        formation::Formation::Alone => {
            drop(tel);
            session.shutdown().await;
            return run_solo_round(args.run_secs);
        }
        formation::Formation::Cancelled => unreachable!("headless net has no lobby to cancel"),
    };
    let me = frozen.me;
    // Mutable: a departed peer (link gone, rl#198) is removed so its stale entry can't tag
    // inbound frames — the live roster of record past this point is the server's schedule.
    let mut id_map = frozen.id_map.clone();
    let all_ids: Vec<PlayerId> = id_map.values().copied().collect();
    info!(
        "starting round: {} player(s), I am {:?} ({})",
        all_ids.len(),
        me,
        my_eid.fmt_short()
    );

    let mut ls = Lockstep::new(super::shared::MATCH_SEED, &all_ids, me);

    let am_host = me == PlayerId(0);
    let server_eid = *id_map
        .iter()
        .find(|(_, pid)| **pid == PlayerId(0))
        .map(|(eid, _)| eid)
        .expect("a frozen roster always contains PlayerId(0)");
    let mut server = am_host.then(|| {
        let mut s = net::server::Server::new(me, &all_ids, ls.sim().clone());
        s.seed_early(&formation::early_peer_msgs(&frozen));
        s
    });

    let mut ticker = tokio::time::interval(Duration::from_secs_f64(TICK_DT));
    let end = Instant::now() + Duration::from_secs(args.run_secs);
    let mut snapshots_io = 0usize;
    let mut next_report_tick = TICK_HZ;
    let mut next_tel_tick = next_sample_tick(0);
    let mut reported_outcome = false;
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

        let mut server_down: Option<net_loop::ServerDown> = None;

        let mut remote_inputs: Vec<net::lockstep::PeerMsg> = Vec::new();
        let mut snapshots: Vec<net::snapshot::CoreSnapshot> = Vec::new();
        while let Some(m) = session.try_recv() {
            match m.msg {
                transport::PeerWire::Tick(msg) => {
                    if let Some(&pid) = id_map.get(&m.from) {
                        remote_inputs.push(net::lockstep::PeerMsg { pid, msg });
                    }
                }
                transport::PeerWire::Snapshot(snap) => {
                    if !am_host {
                        snapshots.push(snap);
                    }
                }
                transport::PeerWire::Articulation(_) => {}
                transport::PeerWire::Beat(_) => {}
                transport::PeerWire::Refuse(reason) => {
                    if !am_host && m.from == server_eid {
                        server_down = Some(net_loop::ServerDown::Refused(reason));
                    }
                }
                transport::PeerWire::JoinRequest(_) | transport::PeerWire::Welcome(_) => {}
            }
        }

        let t = ls.next_tick() as f32 * 0.1;
        let input = Input::from_axes(t.cos(), t.sin());
        let msg = ls.submit_local_input(input, None);
        // The telemetry stamp is the ISSUE tick — on a remote client `ls.next_tick()` (the
        // snapshot-apply cursor) trails it by the transit lag.
        let issue_tick = msg.issue_tick;

        if let Some(srv) = server.as_mut() {
            // HOST: file remote clients' inputs into their per-player streams, then assemble +
            // step THIS tick at our own pace (a remote can delay nothing — rl#193/#194/#195) and
            // broadcast the snapshot the instant it is stepped — the SAME host-authoritative path
            // the windowed driver runs (one stepper). A client adopts exactly the state the host
            // holds; there is no peer-symmetric self-step or desync cross-check (the
            // host IS the source of truth).
            for pm in remote_inputs {
                srv.record_remote(pm.pid, pm.msg);
            }
            // Departures (rl#198) — the SAME handling the windowed host runs
            // (`depart_gone_peers`): a rostered peer whose link is gone left the match; drop its
            // stream + roster entry (nothing ever waits on it). This harness sends no refusals,
            // so the returned departed endpoints go unused.
            let connected = session.connected_peers().await;
            let _ = net_loop::depart_gone_peers(srv, &mut id_map, me, &connected, tel.as_ref());
            srv.advance(msg);
            while srv.next_tick_ready() {
                let bytes = srv.step_next(&[]).snapshot;
                let snap = net::snapshot::CoreSnapshot::from_bytes(&bytes)
                    .expect("the authoritative server's snapshot must decode");
                session.broadcast_snapshot(&snap).await;
                snapshots_io += 1;
                ls.apply_core_snapshot(snap);
                if let Some(w) = hash_log.as_mut() {
                    use std::io::Write as _;
                    let applied = srv.sim().tick().saturating_sub(1);
                    let line = super::shared::tick_hash_line(applied, srv.sim().state_hash());
                    writeln!(w, "{line}").context("writing hash log")?;
                }
            }
            // Chronic input-starvation surface (rl#213) — the one shared drain policy.
            net::telemetry::surface_starvation(Some(srv), tel.as_ref());
        } else {
            session.send_to(server_eid, &msg).await;
            // The adopt callback can't `?`; collect the per-adopt observations and write them after,
            // where the error propagates through normal control flow.
            let mut adopted_hashes: Vec<(u64, u64)> = Vec::new();
            snapshots_io += ls.adopt_snapshots(snapshots, |ls| {
                if hash_log.is_some() {
                    adopted_hashes.push((ls.sim().tick().saturating_sub(1), ls.sim().state_hash()));
                }
            });
            if let Some(w) = hash_log.as_mut() {
                use std::io::Write as _;
                for (applied, hash) in adopted_hashes {
                    let line = super::shared::tick_hash_line(applied, hash);
                    writeln!(w, "{line}").context("writing hash log")?;
                }
            }
            if server_down.is_none() && !session.connected_peers().await.contains(&server_eid) {
                server_down = Some(net_loop::ServerDown::LinkLost);
            }
        }

        if let Some(down) = server_down {
            warn!("ending run at tick {}: {down}", ls.sim().tick());
            break;
        }

        if ls.sim().tick() >= next_report_tick {
            next_report_tick = (ls.sim().tick() / TICK_HZ + 1) * TICK_HZ;
            info!(
                "tick={:>5} peers={} statehash={:#018x}",
                ls.sim().tick(),
                session.connected_peers().await.len(),
                ls.sim().state_hash(),
            );
        }

        if let Some(t) = tel.as_ref() {
            if ls.sim().tick() >= next_tel_tick {
                next_tel_tick = next_sample_tick(ls.sim().tick());
                t.send(TelemetryEvent::tick(ls.sim(), ls.sim().players().count()));
                t.send(TelemetryEvent::input(issue_tick, input));
            }
            if !reported_outcome && ls.sim().outcome() != net::sim::Outcome::Ongoing {
                reported_outcome = true;
                t.send(TelemetryEvent::round_decided(ls.sim()));
            }
        }
    }

    info!(
        "done: {} ticks applied, {} snapshots {}, final hash {:#018x}",
        ls.sim().tick(),
        snapshots_io,
        if am_host { "broadcast" } else { "adopted" },
        ls.sim().state_hash()
    );
    if let Some(t) = tel.as_ref() {
        t.send(TelemetryEvent::tick(ls.sim(), ls.sim().players().count()));
    }
    if all_ids.len() > 1 && ls.sim().tick() < (args.run_secs * TICK_HZ).saturating_sub(TICK_HZ) {
        warn!(
            "only {} ticks in {}s — peer link stalled (missing inputs)",
            ls.sim().tick(),
            args.run_secs
        );
    }
    if tel.is_some() {
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    if let Some(mut w) = hash_log.take() {
        use std::io::Write as _;
        w.flush().context("flushing hash log")?;
    }
    drop(tel);
    session.shutdown().await;
    Ok(())
}
