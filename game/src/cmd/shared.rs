
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use net::lockstep::Lockstep;
use net::sim::{Input, PlayerId, TICK_DT};

pub(crate) const MATCH_SEED: u64 = 0x6372_6162;

pub(crate) fn nn_crab_checkpoint_dir(
    flag: Option<std::path::PathBuf>,
) -> Result<std::path::PathBuf> {
    use crab_world::play::{RigDims, RigFit};
    let dir = flag
        .or_else(|| std::env::var_os("RL_CRAB_CHECKPOINT_DIR").map(std::path::PathBuf::from))
        .unwrap_or_else(|| {
            crab_world::assets::asset_root()
                .join("assets")
                .join("weights")
        });
    match crab_world::play::checkpoint_fits_rig(&dir) {
        RigFit::Ok => Ok(dir),
        RigFit::Missing => anyhow::bail!(
            "rl#114: no trained crab brain (brain.bin) under {} — the giant crab IS the trained NN \
             body (\"Sally\"), and there is no integer stand-in. Point --nn-crab-checkpoint or the \
             RL_CRAB_CHECKPOINT_DIR env var at a trained checkpoint dir (deploy/rl-update must set \
             it, and EVERY device needs the IDENTICAL brain + crab model), then relaunch.",
            dir.display()
        ),
        RigFit::Refused(why) => anyhow::bail!(
            "checkpoint under {} was REFUSED — {why}. Fix the checkpoint, then relaunch.",
            dir.display()
        ),
        RigFit::Mismatch(RigDims { obs, action }) => {
            let RigDims {
                obs: rig_obs,
                action: rig_act,
            } = crab_world::play::rig_dims();
            anyhow::bail!(
                "rl#199: checkpoint under {} was built for a DIFFERENT rig — its brain wants \
                 {obs} obs / {action} act but this binary's crab rig is {rig_obs} obs / \
                 {rig_act} act. Sally would launch as an inert rest-pose statue, so refusing to \
                 launch instead. Retrain/redeploy a checkpoint for this rig, or run a binary \
                 whose rig matches the checkpoint.",
                dir.display()
            )
        }
    }
}

pub(crate) fn nn_crab_checkpoint_dirs(
    flags: Vec<std::path::PathBuf>,
) -> Result<Vec<std::path::PathBuf>> {
    if flags.is_empty() {
        return Ok(vec![nn_crab_checkpoint_dir(None)?]);
    }
    flags
        .into_iter()
        .enumerate()
        .map(|(idx, dir)| {
            nn_crab_checkpoint_dir(Some(dir))
                .with_context(|| format!("crab {idx}'s brain binding is unusable (rl#200)"))
        })
        .collect()
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

pub(crate) fn resolve_render_mode(flag: Option<&str>) -> Result<net::render::RenderMode> {
    let flag = flag
        .map(|s| {
            net::render::RenderMode::parse(s).ok_or_else(|| {
                anyhow::anyhow!("--render-mode must be one of mesh|mesh+colliders|colliders")
            })
        })
        .transpose()?;
    Ok(crab_world::mesh_fallback::initial_render_mode(
        flag,
        crab_world::mesh_fallback::Surface::Game,
    ))
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
    use net::server::Server;
    use net::snapshot::CoreSnapshot;

    let me = PlayerId(0);
    let mut ls = Lockstep::new(MATCH_SEED, &[me], me);
    let mut server = Server::new(me, &[me], ls.sim().clone());
    let tick_dt = Duration::from_secs_f64(TICK_DT);
    let end = Instant::now() + Duration::from_secs(run_secs);
    let mut next = Instant::now();
    while Instant::now() < end {
        let t = ls.next_tick() as f32 * 0.1;
        let msg = ls.submit_local_input(Input::from_axes(t.cos(), t.sin()), None);
        server.advance(msg);
        while server.next_tick_ready() {
            let bytes = server.step_next(&[]).snapshot;
            ls.apply_core_snapshot(
                CoreSnapshot::from_bytes(&bytes).expect("the server's snapshot must decode"),
            );
        }
        next += tick_dt;
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    let p = ls.sim().player(me).unwrap();
    let pos = p.pos();
    let crab = ls.sim().crabs()[0].pos();
    println!(
        "solo: {} ticks, player=({}, {}) yaw={} status={:?}, crab=({}, {}), outcome={:?}, hash={:#018x}",
        ls.sim().tick(),
        pos.x,
        pos.z,
        p.yaw(),
        p.status(),
        crab.x,
        crab.z,
        ls.sim().outcome(),
        ls.sim().state_hash()
    );
    Ok(())
}
