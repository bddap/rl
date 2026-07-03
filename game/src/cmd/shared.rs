//! Cross-subcommand helpers: the match seed, checkpoint/render-mode resolution shared by
//! `play`/`fp-screenshot`/the NN-crab gates, the one per-tick hash-log writer both gates use,
//! and the single offline lockstep round shared by `solo` and the headless `net` no-peer
//! fallback. Kept in one place so there is exactly one source for each — no second `30`-style
//! constant or a forked solo loop that could drift.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use net::lockstep::Lockstep;
use net::sim::{Input, PlayerId, TICK_DT};

/// Deterministic match seed: a constant so independently-launched peers agree without a
/// handshake. The sim takes it as a parameter, so a future session setup can negotiate it
/// (the lower-id peer proposes, say) without touching the sim.
pub(crate) const MATCH_SEED: u64 = 0x6372_6162; // "crab"

/// Resolve the REQUIRED NN-crab checkpoint dir: the `--nn-crab-checkpoint` flag (`flag`), else
/// the `RL_CRAB_CHECKPOINT_DIR` env var (deploy sets this), else `assets/weights` under the ONE
/// asset root ([`crab_world::assets::asset_root`] — `BEVY_ASSET_ROOT`, else the crab-world crate
/// dir; the SAME root the mesh + control glyphs resolve against, bddap/rl#146, so the weights
/// can't live somewhere the mesh doesn't) — so a checkpoint can be chosen at runtime, no recompile.
/// An unusable checkpoint is a HARD, ACTIONABLE launch failure: the one giant crab IS the trained
/// NN body ("Sally"), and there is no integer point-pursuer to fall back to, so rather than silently
/// substituting a fake crab we refuse to launch with a message naming the dir and how to fix it.
/// Three ways a checkpoint is unusable, all refused here: no `brain.bin` (rl#114), an
/// envelope-refused one (corrupt/legacy/wrong-arch/mis-paired, rl#200), and a brain built for a
/// DIFFERENT rig (rl#199) — a wrong-rig brain passes an existence check but the runtime loader
/// refuses to arm it, so letting launch proceed would ship an inert rest-pose Sally that looks
/// frozen-but-fine. Same [`crab_world::play::checkpoint_fits_rig`] verdict the release/deploy
/// gate (`checkpoint-check`) acts on, so launch and the gate can't disagree. This hard gate
/// outranks `RL_RANDOM_POLICY` in the game binary — that diagnostic belongs to rl-demo, which
/// loads checkpoints ungated.
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

/// One determinism-log line, `<tick> <hash>` (zero-padded 16-hex) — the format two
/// peers/runs `diff` to prove byte-identical sims. The line IS the cross-peer diff
/// contract, so every writer (this file's whole-log writer and `game net`'s streaming
/// host/client writers) formats through here (#133).
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

/// Resolve the `--render-mode` flag into a [`net::render::RenderMode`]: parse an explicit value
/// (rejecting an unknown token with an actionable error), then hand the precedence decision —
/// flag beats env beats the unusable-glb fallback — to the ONE shared
/// [`crab_world::mesh_fallback::initial_render_mode`], the same decision rl-demo boots from, so
/// the two player-facing surfaces can't diverge on a broken-glb + env-set launch. An unusable
/// mesh is logged LOUDLY in there; the on-screen banner companion is spawned on the windowed
/// surface in `net::render::scene` (this headless-resolvable check decides only the render MODE;
/// the banner needs the live window).
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

/// Parse a `--join` code into the endpoint to dial: `Some(code)` with a non-blank code dials
/// it; a bare/blank `--join` (or none) falls back to LAN discovery. One parser for every
/// joining command (`play`, `net-screenshot`) so the blank-means-discover rule can't drift.
pub(crate) fn parse_join_dial(join: Option<&str>) -> Result<Option<iroh::EndpointId>> {
    Ok(match join {
        Some(code) if !code.trim().is_empty() => Some(code.trim().parse()?),
        _ => None,
    })
}

/// Default peer-formation knobs shared by every networked command (`net`, `play`,
/// `net-screenshot`): expected peer count including us, and how long discovery waits for
/// them (mDNS on a quiet LAN — a couple of seconds covers it).
pub(crate) const DEFAULT_EXPECT: usize = 2;
pub(crate) const DEFAULT_DISCOVER_SECS: u64 = 4;

/// One offline lockstep round for `run_secs`: a single peer whose own input completes
/// every tick (no network), ticking at [`net::sim::TICK_HZ`] and printing a final summary. Shared by
/// the `solo` command and the headless `net` no-peer fallback, so the alone case runs the
/// SAME deterministic solo path — no second sim loop to drift.
pub(crate) fn run_solo_round(run_secs: u64) -> Result<()> {
    use net::server::Server;
    use net::snapshot::CoreSnapshot;

    let me = PlayerId(0);
    // Solo is the server with a roster of one — the SAME host-authoritative stepper the windowed
    // and networked paths run ([[sp-is-mp-special-case]]): the local `ls` files inputs UP and adopts
    // the snapshot the server emits, never stepping a sim of its own.
    let mut ls = Lockstep::new(MATCH_SEED, &[me], me);
    let mut server = Server::new(me, &[me], ls.sim().clone());
    let tick_dt = Duration::from_secs_f64(TICK_DT);
    let end = Instant::now() + Duration::from_secs(run_secs);
    let mut next = Instant::now();
    while Instant::now() < end {
        // A lazy circular stir so the dot visibly moves.
        let t = ls.next_tick() as f32 * 0.1;
        let msg = ls.submit_local_input(Input::from_axes(t.cos(), t.sin()));
        server.advance(msg);
        while server.next_tick_ready() {
            // Headless smoke: no rapier crab body, so the crab holds spawn (no pose to inject).
            let bytes = server.step_next(None).snapshot;
            ls.apply_core_snapshot(
                CoreSnapshot::from_bytes(&bytes).expect("the server's snapshot must decode"),
            );
        }
        next += tick_dt;
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    let p = ls.sim().player(me).unwrap();
    let pos = p.pos();
    let crab = ls.sim().crab().pos();
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
