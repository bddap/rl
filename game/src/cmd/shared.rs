use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use net::client::ClientSim;
use net::sim::{Input, PlayerId, TICK_DT};

/// The PINNED seed for byte-stable tooling — screenshots and the determinism/behavior
/// probes, where the same seed must reproduce the same run (the spawn layout derives
/// from the seed, rl#305). Real play draws [`net::sim::random_match_seed`] instead.
pub(crate) const MATCH_SEED: u64 = 0x6372_6162;

/// The view this binary boots in — every GCR surface is [`Surface::Game`], so the
/// surface is named once rather than at each entrypoint.
pub(crate) fn boot_view(args: crab_world::RenderArgs) -> crab_world::BootView {
    args.resolve(crab_world::mesh_fallback::Surface::Game)
}

/// The controls-overlay force-knobs resolved against GCR's control scheme — an unknown
/// context id dies here, at t=0, naming the valid ids.
pub(crate) fn gcr_controls(
    args: &crab_world::controls::ControlsOverlayArgs,
) -> Result<crab_world::controls::ControlsOverrides<net::controls::GcrControls>> {
    args.resolve().map_err(anyhow::Error::msg)
}

/// One determinism-log line, `<tick> <hash>` (zero-padded 16-hex) — the format two
/// peers/runs `diff` to prove byte-identical sims. The line IS the cross-peer diff
/// contract, so every writer (this file's whole-log writer and `game net`'s streaming
/// host/client writers) formats through here (#133). The FORMAT is shared; the hashed
/// QUANTITY is not: `game net` logs the bare sim hash, while the probe folds the crab
/// body digest in (rl#223) — diff only logs written by the same writer.
pub(crate) fn tick_hash_line(tick: u64, hash: u64) -> String {
    format!("{tick} {hash:#018x}")
}

/// Write per-tick [`tick_hash_line`]s to a file — the whole-log form the `nn-crab-probe`
/// gate diffs.
pub(crate) fn write_tick_hash_log(
    path: &std::path::Path,
    entries: impl Iterator<Item = (u64, u64)>,
) -> Result<()> {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (tick, hash) in entries {
        writeln!(out, "{}", tick_hash_line(tick, hash)).unwrap();
    }
    std::fs::write(path, out).with_context(|| format!("writing hash log to {}", path.display()))
}

pub(crate) fn parse_join_dial(join: Option<&str>) -> Result<Option<iroh::EndpointId>> {
    Ok(match join {
        Some(code) if !code.trim().is_empty() => Some(code.trim().parse()?),
        _ => None,
    })
}

pub(crate) const DEFAULT_EXPECT: usize = 2;
pub(crate) const DEFAULT_DISCOVER_SECS: u64 = 4;

pub(crate) fn run_solo_round(run_secs: u64) -> Result<()> {
    use net::crab_slot::HeadlessHostWorld;
    use net::server::Server;
    use net::snapshot::CoreSnapshot;

    let me = PlayerId(0);
    let mut client = ClientSim::new(net::sim::random_match_seed(), &[me], me);
    let mut server = Server::new(me, &[me], client.sim().clone());
    // Crab poses are mandatory (rl#298 stage 5): even this headless harness runs the
    // one crab world — a rest-pose brain, no checkpoint bound.
    let mut host_world =
        HeadlessHostWorld::new(vec![crab_world::policy::Policy::rest()], server.sim());
    let tick_dt = Duration::from_secs_f64(TICK_DT);
    let end = Instant::now() + Duration::from_secs(run_secs);
    let mut next = Instant::now();
    while Instant::now() < end {
        let t = client.next_tick() as f32 * 0.1;
        let msg = client.submit_local_input(Input::from_axes(t.cos(), t.sin()), None);
        server.advance(msg);
        for stepped in host_world.step_ready_ticks(&mut server) {
            client.apply_core_snapshot(
                CoreSnapshot::from_bytes(&stepped.snapshot)
                    .expect("the server's snapshot must decode"),
            );
        }
        next += tick_dt;
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    let p = client.sim().player(me).unwrap();
    let pos = p.pos();
    let crab = client.sim().crabs()[0].pos();
    println!(
        "solo: {} ticks, player=({}, {}) yaw={} status={:?}, crab=({}, {}), outcome={:?}, hash={:#018x}",
        client.sim().tick(),
        pos.x,
        pos.z,
        p.yaw(),
        p.status(),
        crab.x,
        crab.z,
        client.sim().outcome(),
        client.sim().state_hash()
    );
    Ok(())
}

/// Scripted chord entry + map save for evidence shots (rl#330/rl#358), shared by the
/// solo and two-peer screenshot surfaces (rl#398 needs the chord map driven on a
/// JOINED peer) — one flag set, one validation, one wiring.
#[derive(clap::Args)]
pub(crate) struct ChordScriptArgs {
    /// Scripted chord entry (rl#330 evidence): hold the kb chord modifier
    /// (right mouse) from this frame — the combo map (rl#358) opens.
    #[arg(long)]
    chord_hold_at: Option<u64>,
    /// Release the scripted chord modifier at this frame, executing the typed code.
    /// Default: never — the menu stays open into the shot.
    #[arg(long)]
    chord_release_at: Option<u64>,
    /// Scripted code taps while held: `frame:dir`, comma-separated, dir ∈ U|D|L|R
    /// (e.g. `45:U,55:L`).
    #[arg(long, value_delimiter = ',')]
    chord_taps: Vec<String>,
    /// Extra modifier hold windows beyond --chord-hold-at/--chord-release-at:
    /// `from:to` pairs, `to` empty for never-release (e.g. `120:180,220:`) — lets one
    /// clip enter several codes (rl#358 map-growth evidence).
    #[arg(long, value_delimiter = ',')]
    chord_holds: Vec<String>,
    /// Load/persist the shot's discovered-code set here (default: unpersisted,
    /// empty — an evidence shot must never touch the real save).
    #[arg(long)]
    chord_map_file: Option<std::path::PathBuf>,
}

impl ChordScriptArgs {
    /// Validate and wire onto a built app: the [`net::render::ChordScript`] resource
    /// (if a hold is scripted) and the discovered-set save override.
    pub(crate) fn apply(self, app: &mut net::render::App) -> Result<()> {
        if let Some(hold_at) = self.chord_hold_at {
            let taps = self
                .chord_taps
                .iter()
                .map(|spec| parse_chord_tap(spec))
                .collect::<Result<Vec<_>>>()?;
            // The script's frame counter starts at 1 and the modifier must be down before
            // a tap can buffer — a spec that can't fire must fail loud, not shoot a blank
            // evidence frame.
            if hold_at == 0 || taps.iter().any(|&(at, _)| at <= hold_at) {
                anyhow::bail!("chord frames must be ≥1 and taps after --chord-hold-at");
            }
            if self.chord_release_at.is_some_and(|r| r <= hold_at) {
                anyhow::bail!("--chord-release-at must be after --chord-hold-at");
            }
            let mut seen = std::collections::HashSet::new();
            if let Some(dup) = taps.iter().find(|&&t| !seen.insert(t)) {
                anyhow::bail!(
                    "duplicate chord tap {}:{:?} (one edge, one tap)",
                    dup.0,
                    dup.1
                );
            }
            let holds = self
                .chord_holds
                .iter()
                .map(|spec| parse_hold(spec))
                .collect::<Result<Vec<_>>>()?;
            // The script holds the modifier on ANY covering window, so windows that
            // touch or overlap silently merge into one capture and the release edge —
            // the thing that executes the code — never fires between them. Demand a
            // ≥1-frame gap, in order, and a released primary window before extras.
            if !holds.is_empty() && self.chord_release_at.is_none() {
                anyhow::bail!("--chord-holds needs --chord-release-at to close the first window");
            }
            let mut windows = vec![(hold_at, self.chord_release_at)];
            windows.extend(holds.iter().copied());
            for pair in windows.windows(2) {
                let (prev, next) = (pair[0], pair[1]);
                match prev.1 {
                    Some(end) if next.0 > end => {}
                    _ => anyhow::bail!(
                        "chord hold windows must be in order with a ≥1-frame gap \
                         (got {:?} then {:?}) — touching windows merge into one capture",
                        prev,
                        next
                    ),
                }
            }
            app.insert_resource(
                net::render::ChordScript::new(hold_at, self.chord_release_at, taps)
                    .with_holds(holds),
            );
        } else if !self.chord_taps.is_empty()
            || self.chord_release_at.is_some()
            || !self.chord_holds.is_empty()
        {
            anyhow::bail!("--chord-taps/--chord-release-at/--chord-holds need --chord-hold-at");
        }
        if let Some(file) = self.chord_map_file {
            app.insert_resource(net::render::chord_map::DiscoveredCodes::load(Some(file)));
        }
        Ok(())
    }
}

pub(crate) fn parse_hold(spec: &str) -> Result<(u64, Option<u64>)> {
    let (from, to) = spec
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("chord hold {spec:?} is not from:to"))?;
    let from: u64 = from.parse()?;
    let to = if to.is_empty() {
        None
    } else {
        let to: u64 = to.parse()?;
        if to <= from {
            anyhow::bail!("chord hold {spec:?} releases before it holds");
        }
        Some(to)
    };
    Ok((from, to))
}

fn parse_chord_tap(spec: &str) -> Result<(u64, crab_world::chord::ChordDir)> {
    let (frame, dir) = spec
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("chord tap {spec:?} is not frame:dir"))?;
    // One UDLR alphabet — the chord map's codec, not a second letter match.
    let dir = match net::render::chord_map::code_from_str(dir).as_deref() {
        Some([d]) => *d,
        _ => anyhow::bail!("chord tap dir {dir:?} is not U|D|L|R"),
    };
    Ok((frame.parse()?, dir))
}
