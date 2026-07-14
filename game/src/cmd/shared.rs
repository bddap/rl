use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use net::client::ClientSim;
use net::sim::{Input, PlayerId, TICK_DT};

pub(crate) const MATCH_SEED: u64 = 0x6372_6162;

/// The checkpoint-dir env fallback the deploy scripts export (deploy/rl-update sets it).
/// Named ONCE here and referenced by every `--nn-crab-checkpoint` / `--checkpoint` flag that
/// honors it, so the flags and the error prose below can't drift apart.
pub(crate) const CHECKPOINT_ENV: &str = "RL_CRAB_CHECKPOINT_DIR";

/// The render mode this binary boots in — every GCR surface is [`Surface::Game`], so the
/// surface is named once rather than at each entrypoint.
pub(crate) fn render_mode(args: crab_world::RenderArgs) -> net::render::RenderMode {
    args.resolve(crab_world::mesh_fallback::Surface::Game)
}

/// The controls-overlay force-knobs resolved against GCR's control scheme — an unknown
/// context id dies here, at t=0, naming the valid ids.
pub(crate) fn gcr_controls(
    args: &crab_world::controls::ControlsOverlayArgs,
) -> Result<crab_world::controls::ControlsOverrides<net::controls::GcrControls>> {
    args.resolve().map_err(anyhow::Error::msg)
}

/// The launch gate: resolve the checkpoint dir and load it in ONE read — the returned
/// [`Policy`] is armed by construction, never re-read by the plugin (rl#241: a
/// classify-then-reload gate can straddle a checkpoint swap and arm a rest-pose statue
/// it never vetted). Returns the resolved dir alongside for operator-facing labels.
pub(crate) fn nn_crab_policy(
    flag: Option<std::path::PathBuf>,
) -> Result<(std::path::PathBuf, crab_world::policy::Policy)> {
    use crab_world::policy::{CheckpointUnusable, RigDims};
    // The env fallback is clap's, declared on each subcommand's checkpoint flag ([`CHECKPOINT_ENV`]).
    // `fp-screenshot` deliberately opts OUT of it: there the flag ARMS a crab at all, so the env
    // would seed one into a shot meant to have none.
    let dir = flag.unwrap_or_else(|| {
        crab_world::assets::asset_root()
            .join("assets")
            .join("weights")
    });
    match crab_world::policy::load_armed(&dir) {
        Ok(policy) => Ok((dir, policy)),
        Err(CheckpointUnusable::Missing) => anyhow::bail!(
            "rl#114: no trained crab brain (brain.bin) under {} — the giant crab IS the trained NN \
             body (\"Sally\"), and there is no integer stand-in. Point this command's checkpoint \
             flag or {CHECKPOINT_ENV} at a trained checkpoint dir (deploy/rl-update must set it, \
             and EVERY device needs the IDENTICAL brain + crab model), then relaunch.",
            dir.display()
        ),
        Err(CheckpointUnusable::Refused(why)) => anyhow::bail!(
            "checkpoint under {} was REFUSED — {why}. Fix the checkpoint, then relaunch.",
            dir.display()
        ),
        Err(CheckpointUnusable::Mismatch(RigDims { obs, action })) => {
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

/// [`nn_crab_policy`] over every `--nn-crab-checkpoint` binding (default binding when
/// none given) — the armed policies, one per crab.
pub(crate) fn nn_crab_policies(
    flags: Vec<std::path::PathBuf>,
) -> Result<Vec<crab_world::policy::Policy>> {
    if flags.is_empty() {
        return Ok(vec![nn_crab_policy(None)?.1]);
    }
    flags
        .into_iter()
        .enumerate()
        .map(|(idx, dir)| {
            nn_crab_policy(Some(dir))
                .map(|(_, policy)| policy)
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
    let mut client = ClientSim::new(MATCH_SEED, &[me], me);
    let mut server = Server::new(me, &[me], client.sim().clone());
    let tick_dt = Duration::from_secs_f64(TICK_DT);
    let end = Instant::now() + Duration::from_secs(run_secs);
    let mut next = Instant::now();
    while Instant::now() < end {
        let t = client.next_tick() as f32 * 0.1;
        let msg = client.submit_local_input(Input::from_axes(t.cos(), t.sin()), None);
        server.advance(msg);
        while server.next_tick_ready() {
            let bytes = server.step_next(&[], Default::default()).snapshot;
            client.apply_core_snapshot(
                CoreSnapshot::from_bytes(&bytes).expect("the server's snapshot must decode"),
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
