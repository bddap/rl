//! `rl-train` — the HEADLESS trainer. Links `crab-world` with render OFF, so it pulls NO
//! bevy_render/bevy_pbr/wgpu27: the crab machinery (bot / physics / training) and the
//! shared `TrainConfig` come from the library, this binary is a thin entry that parses
//! its modes and dispatches. It DOES link burn-wgpu (the GPU PPO update is the sole
//! update path, rl#49) — that is wgpu 26 for compute, still no bevy_render/wgpu 27.
//!
//! Modes (all headless, no window, no GPU renderer):
//! - `learn` — the trainer: K rollout threads (CPU inference) + the GPU PPO update.
//! - `--verify-colliders` / `--verify-pivots` / `--check-rest-colliders` — DEV rig
//!   audits that build a windowless physics world, print a report, and exit.
//!
//! The windowed demo + screenshot live in the separate `rl-demo` binary (render on);
//! the multiplayer game in `game`. Splitting them off is what lets THIS binary link no
//! graphics crate (rl#51).

use std::path::PathBuf;

use bevy::prelude::*;
use clap::{Parser, Subcommand};
use crab_world::{TrainConfig, bot, training};

use training::systems::STEPS_PER_ROLLOUT;

/// Crab Combat — RL-trained crab bots learn to stand, walk, and fight.
///
/// Training is the `learn` subcommand (the sole trainer). With no subcommand the
/// binary runs one of the DEV rig audits (`--verify-colliders` / `--verify-pivots` /
/// `--check-rest-colliders`) from the flags below, else errors. The training knobs
/// live on the `learn` subcommand, so a stray `--workers` without it is a parse error
/// rather than a silent no-op. (The windowed demo/screenshot moved to `rl-demo`.)
#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Cli {
    #[command(flatten)]
    dev: DevArgs,

    #[command(subcommand)]
    command: Option<Command>,
}

/// The headless DEV rig audits — no subcommand, no window. Each loads the crab model,
/// runs a containment/agreement check, prints a table, and exits with a pass/fail
/// code, so each doubles as a regression gate on rig changes.
#[derive(Parser, Debug, Clone)]
struct DevArgs {
    /// DEV: score every live collider against the mesh it stands in for and print a
    /// per-part agreement table (signed surface distance, in model units), then exit.
    /// Exits nonzero if any part fails. Model is `CRAB_MODEL_PATH`, else the dev
    /// `sally.glb`.
    #[arg(long)]
    verify_colliders: bool,

    /// DEV: test whether every joint pivot and collider endpoint lies INSIDE the
    /// bind-pose mesh, via the generalized winding number against the model's triangle
    /// soup, then exit. Reports per-point winding number + signed nearest-surface
    /// distance and ranks the worst out-of-mesh offenders. Model is `CRAB_MODEL_PATH`,
    /// else the dev `sally.glb`.
    #[arg(long)]
    verify_pivots: bool,

    /// DEV: spawn the crab, settle it to rest, then test every pair of body colliders
    /// for interpenetration at the settled pose and flag any overlap the solver is
    /// actively fighting. Expected overlaps (jointed anchors, group-filtered nested
    /// links) are reported but never failed. Exits nonzero on any illegal one, so it
    /// gates rig changes.
    #[arg(long)]
    check_rest_colliders: bool,
}

/// The trainer. One learner (the main thread) owns the policy + optimizer +
/// normalizer; K rollout THREADS each step their own rapier world on their own core
/// and feed buffers back over a channel — wall-clock-parallel rollouts, crash-
/// isolated per worker, with no multiprocess IPC. See `training::inproc`.
#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Run the trainer: spawn K rollout threads, snapshot the policy to each, collect
    /// their rollouts, and run the PPO update. Resumes from `--checkpoint-dir` and
    /// stops at the `--ticks` budget.
    Learn(LearnArgs),

    /// Judge a checkpoint's TRAINING SUCCESS directly (not by reward): headless, load the
    /// checkpoint, place a ball at a FIXED far distance, drive the policy DETERMINISTICALLY, and
    /// print two honest numbers — real metres of progress the crab closed toward the ball, and
    /// the total applied joint torque (lower = done with minimal torque). Deterministic + fast,
    /// so the daemon can run it against the live checkpoint on demand to plot progress-toward-ball
    /// over training instead of watching the reward curve.
    Eval(EvalArgs),
}

/// Learner orchestration: the shared training config plus how many rollout threads
/// to fan out.
#[derive(Parser, Debug, Clone)]
struct LearnArgs {
    #[command(flatten)]
    train: TrainConfig,

    /// Number of rollout threads K, each stepping its own world on its own core.
    /// Default is PHYSICAL cores minus 2 (floored at one) — physical, not logical,
    /// so it never oversubscribes a hyperthreaded pair onto one core — leaving the
    /// rest of the machine a couple of cores. Pass an explicit value to use more.
    /// Clamped to 1..=64. Concurrent per-arch runs (bddap/rl#200) MUST each pass an
    /// explicit value partitioning the physical cores: two default-workers processes
    /// would EACH take physical−2 and oversubscribe the machine.
    #[arg(long)]
    workers: Option<usize>,

    /// Policy architecture for a FRESH start (an empty --checkpoint-dir); default
    /// mlp256. On a RESUME the checkpoint's arch tag is authoritative and this flag
    /// is only a cross-check — a value that disagrees with the tag ABORTS (never a
    /// cold start over the trained policy, never a silently ignored flag).
    #[arg(long, value_parser = parse_arch)]
    arch: Option<bot::arch::ArchId>,

    /// Rollout horizon H: physics ticks each thread rolls per iteration before
    /// handing its buffers back. Per-iteration sample count is K·(--envs)·H.
    #[arg(long, default_value_t = STEPS_PER_ROLLOUT as u64)]
    horizon: u64,

    /// Stop after this many PPO iterations (0 = unbounded). A benchmark / A-B knob;
    /// the production budget is `--ticks` (total physics ticks). Whichever limit is
    /// hit first stops the run.
    #[arg(long, default_value_t = 0)]
    iters: u64,

    /// Niceness applied to the whole process — the learner and its rollout threads
    /// share it (POSIX priority is per-process; higher = yields more CPU). Positive
    /// so a foreground game always preempts training even when the threads saturate
    /// their cores. Clamped to 0..=19 (0 disables; a negative nice would raise
    /// priority and needs privilege, so it is floored to 0 rather than attempted).
    #[arg(long, default_value_t = 10)]
    nice: i32,

    /// DEV: train the procedural fallback body when no usable `sally.glb` resolves,
    /// instead of refusing to start (bddap/rl#214). The checkpoints it writes carry
    /// body digest 0 and will be REFUSED by every canonical-body surface — a policy
    /// trained on the fallback is not Sally.
    #[arg(long)]
    allow_fallback_body: bool,
}

/// The `eval` subcommand: which checkpoint to judge, how long to roll, and how far to place the
/// ball. Deterministic — no seed knob, because the eval fixes the spawn, the target, and takes the
/// policy mean, so the report reproduces from these args alone.
#[derive(Parser, Debug, Clone)]
struct EvalArgs {
    /// Checkpoint directory to load (`brain.bin` + `normalizer.bin`). The daemon points this at
    /// the LIVE training checkpoint to judge the run in flight.
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,

    /// Physics ticks to run the policy for (after a short settle drop). Defaults to one
    /// training episode horizon ([`crab_world::training::systems::MAX_EPISODE_TICKS`],
    /// ~23 s of crab time at 64 Hz) — enough for a working gait to traverse a far target.
    #[arg(long, default_value_t = crab_world::training::systems::MAX_EPISODE_TICKS as u64)]
    ticks: u64,

    /// Fixed planar distance (m) to place the ball from the crab's spawn. Default is the far edge
    /// of the training band ([`crab_world::eval::DEFAULT_TARGET_DISTANCE_M`]) — the hardest
    /// in-distribution target, challenging but reachable.
    #[arg(long)]
    distance: Option<f32>,

    /// DEV: judge on the procedural fallback body when no usable `sally.glb` resolves,
    /// instead of refusing to start (bddap/rl#214). Only meaningful against a
    /// checkpoint trained on the fallback (body digest 0) or an unstamped pre-#214
    /// one; a Sally-stamped checkpoint still refuses (wrong body).
    #[arg(long)]
    allow_fallback_body: bool,
}

/// clap value-parser for `--arch`: delegates to the registry's `TryFrom<String>`, whose
/// error already names the unknown arch and lists the known ones.
fn parse_arch(s: &str) -> Result<bot::arch::ArchId, String> {
    bot::arch::ArchId::try_from(s.to_string())
}

/// The bddap/rl#214 body preflight, MANDATORY for the two subcommands that build rollout
/// worlds: without it, `learn`/`eval` reached the body's SILENT fallback and trained or
/// judged the procedural body as if it were Sally. The lib half logs the verdict (loud
/// refusal / latched fallback error / positive body line); a refusal exits nonzero here,
/// before any world is built. Placed on the subcommands, not in the lib entry points,
/// so the glb-less lib tests (which call `headless_stack`/`run_eval` directly on the
/// fallback body on purpose) stay hermetic.
fn require_canonical_body_or_exit(
    context: &str,
    allow_fallback: bool,
) -> crab_world::mesh_fallback::BodyGate {
    match crab_world::mesh_fallback::require_canonical_body(context, allow_fallback) {
        Ok(gate) => gate,
        Err(_) => std::process::exit(1), // refusal already logged loudly by the preflight
    }
}

fn main() {
    // Installs the tracing subscriber (stderr fmt, so the trainer's `info!`/`warn!`/`error!`
    // still surface headless where there is no bevy `LogPlugin` — a fail-loud guard that
    // never speaks is no guard). Also exports OTLP traces/metrics/logs when a telemetry
    // endpoint is configured (off otherwise). The guard flushes on drop, so it must live
    // for all of `main`, including the early-return `Learn` branch below. `RUST_LOG`
    // overrides the default `info` level.
    let _otel = otel::init("rl-train");
    let cli = Cli::parse();

    // The subcommands. `learn` steps no world itself (it owns the policy and runs PPO) and spawns
    // K rollout threads that each drive their own headless app; `eval` judges a checkpoint's
    // training success headlessly. Both return; a bare invocation falls through to the DEV audits.
    match cli.command {
        Some(Command::Learn(l)) => {
            let body_gate = require_canonical_body_or_exit("learn", l.allow_fallback_body);
            // run_learner owns nicing (it lowers process priority before building any
            // world) so a foreground game preempts training.
            training::inproc::run_learner(
                body_gate,
                &l.train,
                l.arch,
                training::inproc::default_workers(l.workers),
                l.horizon,
                l.iters,
                l.nice,
            );
            return;
        }
        Some(Command::Eval(e)) => {
            let body_gate = require_canonical_body_or_exit("eval", e.allow_fallback_body);
            let distance = e
                .distance
                .unwrap_or(crab_world::eval::DEFAULT_TARGET_DISTANCE_M);
            // A refused/mismatched checkpoint is a hard exit-1 with NO `EVAL_RESULT` line
            // (the daemon greps that prefix; wrong-body baseline numbers plotted as
            // training progress would be the eval-side rl#214). Absent stays the
            // legitimate zero-action baseline below.
            let r = match crab_world::eval::run_eval(body_gate, &e.checkpoint_dir, e.ticks, distance)
            {
                Ok(r) => r,
                Err(refusal) => {
                    eprintln!("eval: {refusal}");
                    std::process::exit(1);
                }
            };
            // The two headline numbers first (progress_m, total_torque), then the context that
            // lets them be trusted at face value. `EVAL_RESULT` is a stable, greppable prefix the
            // daemon parses to plot progress-toward-ball over training.
            println!(
                "EVAL_RESULT progress_m={:.4} total_torque={:.2} mean_torque_per_tick={:.4} \
                 initial_m={:.4} closest_m={:.4} final_m={:.4} target_m={:.2} reached={} \
                 ticks={} policy_loaded={}",
                r.progress_m,
                r.total_torque,
                r.mean_torque_per_tick,
                r.initial_distance_m,
                r.closest_distance_m,
                r.final_distance_m,
                r.target_distance_m,
                r.reached,
                r.active_ticks,
                r.policy_loaded,
            );
            if !r.policy_loaded {
                eprintln!(
                    "eval: no usable checkpoint at {} — the numbers above are the zero-action \
                     rest-pose baseline, NOT a trained policy",
                    e.checkpoint_dir.display()
                );
            }
            return;
        }
        None => {}
    }

    let dev = cli.dev;

    // DEV verify: score the live colliders against the mesh, print, exit.
    if dev.verify_colliders {
        std::process::exit(verify_colliders());
    }

    // DEV verify: test joint pivots + collider endpoints for mesh containment, exit.
    if dev.verify_pivots {
        std::process::exit(verify_pivots());
    }

    // The rest-collider check spawns the rig-derived body, so preflight a PRESENT model
    // first: a broken asset then fails fast with the real reason instead of panicking
    // deep in Startup (or blaming CRAB_MODEL_PATH for a parse error in a model that was
    // present). NO model is NOT an error: the body falls back to the procedural
    // stand-in (built in `CrabAssets::from_world`).
    if let Some(p) = bot::meshfit::model_path() {
        match bot::meshfit::LoadedModel::load(&p) {
            Err(e) => {
                eprintln!("crab model {p:?}: {e}");
                std::process::exit(1);
            }
            // A model that loads but lacks the expected crab bones builds no recipe.
            // Reject it here, not as `spawn_crab`'s expect deep in Startup with a
            // message that wrongly blames a missing/corrupt file.
            Ok(model) => {
                if bot::rig::build_recipe(&model).is_none() {
                    eprintln!(
                        "crab model {p:?}: loaded but has none of the expected crab bones (e.g. Def_leg_01.000.L)"
                    );
                    std::process::exit(1);
                }
            }
        }
    }

    // DEV check: settle the crab and audit its rest-pose colliders for illegal
    // interpenetration, then exit. After the model preflight so a missing model fails
    // with the message above, not a spawn panic deep in the check.
    if dev.check_rest_colliders {
        std::process::exit(bot::collider_check::run());
    }

    // No mode: this binary trains (`rl-train learn`) or runs a DEV audit; the windowed
    // demo/screenshot live in `rl-demo`.
    eprintln!(
        "no mode selected. Train with `rl-train learn` (the sole trainer), or run a DEV \
         rig audit (--verify-colliders / --verify-pivots / --check-rest-colliders). The \
         windowed demo + screenshot are the `rl-demo` binary."
    );
    std::process::exit(2);
}

/// DEV `--verify-colliders`: load the model, reconstruct every live collider in
/// bind-pose world, and score it against the mesh vertices it stands in for. Prints
/// a per-part agreement table (signed surface distance, model units) + a worst-
/// offender ranking, and returns a process exit code (0 = all pass, 1 = a part
/// fails or the model is unavailable) so it serves as both a diagnostic and a
/// regression gate.
fn verify_colliders() -> i32 {
    use bot::meshfit::{score_box, score_capsule};
    use bot::rig::RestShape;

    let Some(model_path) = bot::meshfit::model_path() else {
        eprintln!(
            "verify-colliders: no model — set CRAB_MODEL_PATH or place sally.glb at the dev path"
        );
        return 1;
    };
    let model = match bot::meshfit::LoadedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("verify-colliders: load {model_path:?}: {e}");
            return 1;
        }
    };
    let Some(recipe) = bot::rig::build_recipe(&model) else {
        eprintln!("verify-colliders: model built no rig recipe");
        return 1;
    };
    let clouds = model.vertices_by_part();
    let trunk = model.vertices_for_bones(&bot::rig::TRUNK_BONES);

    println!("collider<->mesh agreement (model units; +out = mesh pokes OUT of collider):");
    println!(
        "  {:<22} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>6}  {:>7}",
        "part", "n", "r", "fOut%", "pk95", "pkMax", "bulge", "skew", "rRat", "verdict"
    );

    // A capsule-only diagnostic prints as a number, or "-" when it doesn't apply
    // (a box, or a cloud too small to have a principal axis).
    let fmt = |x: Option<f32>| x.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
    // (label, severity = pk95/r, failed) for the worst-offender ranking.
    let mut ranking: Vec<(String, f32, bool)> = Vec::new();
    let mut any_fail = false;

    for rc in bot::rig::rest_colliders(&model, &recipe) {
        let label = format!("{:?}", rc.part);
        let (score, rnorm, fail) = match rc.shape {
            RestShape::Capsule { a, b, radius } => {
                let pts = clouds.get(&rc.part).map(|p| p.as_slice()).unwrap_or(&[]);
                let s = score_capsule(pts, a, b, radius);
                // Pass: little flesh escapes, the worst poke is shallow vs the part's
                // own radius, the collider isn't grossly oversized, the axis tracks
                // the limb, and the radius isn't starved/ballooned. The axis/radius
                // diagnostics only exist for a fittable cloud (`Some`); absent, those
                // two checks simply don't apply.
                let fail = s.frac_outside > 0.05
                    || s.poke_out_p95 > (0.15 * radius).max(0.005)
                    || s.bulge_p95 > 0.5 * radius
                    || s.capsule.is_some_and(|c| {
                        c.axis_skew_deg > 15.0 || !(0.85..=1.4).contains(&c.radius_ratio)
                    });
                (s, radius.max(1e-3), fail)
            }
            RestShape::Cuboid { center, half } => {
                let s = score_box(&trunk, center, half);
                // A box over-covering the shell is cosmetically fine; only flag flesh
                // escaping it (absolute, since a box has no single radius).
                let fail = s.frac_outside > 0.03 || s.poke_out_p95 > 0.02;
                (s, half.min_element().max(1e-3), fail)
            }
        };
        any_fail |= fail;
        ranking.push((label.clone(), score.poke_out_p95 / rnorm, fail));
        println!(
            "  {:<22} {:>5} {:>6.3} {:>6.1} {:>6.3} {:>6.3} {:>6.3} {:>5} {:>6}  {}",
            label,
            score.n,
            rnorm,
            score.frac_outside * 100.0,
            score.poke_out_p95,
            score.poke_out_max,
            score.bulge_p95,
            fmt(score.capsule.map(|c| c.axis_skew_deg)),
            fmt(score.capsule.map(|c| c.radius_ratio)),
            if fail { "FAIL" } else { "pass" },
        );
    }

    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst: Vec<String> = ranking
        .iter()
        .take(6)
        .map(|(l, s, f)| format!("{l} {:.2}{}", s, if *f { "!" } else { "" }))
        .collect();
    println!("worst (pk95/r): {}", worst.join(", "));
    println!(
        "{}",
        if any_fail {
            "VERDICT: FAIL — some colliders sit off the mesh"
        } else {
            "VERDICT: pass"
        }
    );
    i32::from(any_fail)
}

/// DEV `--verify-pivots`: empirically test whether each joint pivot and each fitted
/// collider endpoint lies INSIDE the crab's bind-pose visual mesh. Loads the model's
/// triangle soup (bind-world-skinned, same frame as the bone origins + clouds), then
/// for every query point computes the generalized winding number (inside/outside,
/// robust to a non-watertight mesh) and the signed nearest-surface distance (how far
/// in/out). Prints a per-link table + a worst-offender ranking, and exits 0/1.
fn verify_pivots() -> i32 {
    use bot::rig::RestShape;

    let Some(model_path) = bot::meshfit::model_path() else {
        eprintln!(
            "verify-pivots: no model — set CRAB_MODEL_PATH or place sally.glb at the dev path"
        );
        return 1;
    };
    let model = match bot::meshfit::LoadedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("verify-pivots: load {model_path:?}: {e}");
            return 1;
        }
    };
    let mesh = match bot::meshfit::load_bind_mesh(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("verify-pivots: load mesh {model_path:?}: {e}");
            return 1;
        }
    };
    let Some(recipe) = bot::rig::build_recipe(&model) else {
        eprintln!("verify-pivots: model built no rig recipe");
        return 1;
    };
    let pos = &mesh.positions;
    let tris = &mesh.triangles;

    let (lo, hi) = bot::meshfit::aabb(pos);

    // Mesh-containment probe over the bind soup — the same single path the skin-diag
    // audit uses. Its `orient` is the global winding sign from the soup's signed
    // volume: makes interior points read +1 whatever the triangle order, without
    // trusting any single "is this inside?" probe. The crab's vertex centroid sits in
    // a cavity (legs splayed, hollow shell), so it reads ~0 and is useless as the
    // orientation reference — the earlier bug. The carapace pivot (leg-hub centroid,
    // deep in the thorax) is the honest interior probe, used below only to *report*
    // the self-check, not to set the sign.
    let soup = bot::meshfit::MeshContainment::new(pos, tris);
    let signed_vol = soup.signed_vol();
    let orient = soup.orient();
    // Adapt the verdict to this reporter's `(wn, signed_dist, inside)` table layout.
    let probe = |p: Vec3| {
        let c = soup.probe(p);
        (c.wn, c.signed_dist, c.inside)
    };

    // Self-checks. Interior reference = the leg-hub centroid (the carapace pivot the
    // rig anchors every limb to), which is solidly inside the body shell; it must
    // read ~+1. A point 10 units past the bbox must read ~0.
    let hub = bot::rig::rest_colliders(&model, &recipe)
        .iter()
        .find(|rc| rc.part == bot::meshfit::PartId::Carapace)
        .map(|rc| rc.pivot)
        .unwrap_or((lo + hi) * 0.5);
    let centroid = pos.iter().copied().sum::<Vec3>() / pos.len().max(1) as f32;
    let far = hi + (hi - lo).max(Vec3::splat(1.0)) + Vec3::splat(10.0);
    let (hub_wn, _, _) = probe(hub);
    let (cen_wn, _, _) = probe(centroid);
    let (far_wn, _, _) = probe(far);

    println!(
        "mesh: {} verts, {} triangles, bbox {:.3}..{:.3}, signed_vol={:.4}",
        pos.len(),
        tris.len(),
        lo,
        hi,
        signed_vol
    );
    println!(
        "self-check: hub(interior) wn={:+.3} (expect ~+1), vertex-centroid wn={:+.3} (in a cavity → ~0 ok), far-point wn={:+.3} (expect ~0){}",
        hub_wn,
        cen_wn,
        far_wn,
        if orient < 0.0 {
            "  [triangle winding is CW/flipped — normalised via signed volume]"
        } else {
            ""
        }
    );

    println!();
    println!("per-link containment (signed dist: + = OUTSIDE mesh, - = inside):");
    println!(
        "  {:<24} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4}",
        "link", "piv.wn", "piv.dist", "in?", "a.wn", "a.dist", "in?", "b.wn", "b.dist", "in?"
    );

    // (label, signed outside distance) for the worst-offender ranking; only OUTSIDE
    // points (positive signed distance) are offenders. `windings` collects every
    // query point's winding so the watertight verdict can measure how tightly they
    // cluster at integers (clean) vs scatter fractionally (open/non-manifold).
    let mut pivots_out = 0usize;
    let mut endpoints_out = 0usize;
    let mut offenders: Vec<(String, f32)> = Vec::new();
    let mut windings: Vec<f32> = Vec::new();
    let yn = |b: bool| if b { "IN" } else { "OUT" };

    for rc in bot::rig::rest_colliders(&model, &recipe) {
        let label = format!("{:?}", rc.part);
        let (pwn, pdist, pin) = probe(rc.pivot);
        windings.push(pwn);
        if !pin {
            pivots_out += 1;
            offenders.push((format!("{label} pivot"), pdist));
        }
        match rc.shape {
            RestShape::Capsule { a, b, .. } => {
                let (awn, adist, ain) = probe(a);
                let (bwn, bdist, bin) = probe(b);
                windings.push(awn);
                windings.push(bwn);
                for (tag, inside, dist) in [("a", ain, adist), ("b", bin, bdist)] {
                    if !inside {
                        endpoints_out += 1;
                        offenders.push((format!("{label} {tag}"), dist));
                    }
                }
                println!(
                    "  {:<24} | {:>+7.3} {:>+8.4} {:>4} | {:>+7.3} {:>+8.4} {:>4} | {:>+7.3} {:>+8.4} {:>4}",
                    label,
                    pwn,
                    pdist,
                    yn(pin),
                    awn,
                    adist,
                    yn(ain),
                    bwn,
                    bdist,
                    yn(bin)
                );
            }
            RestShape::Cuboid { center, half } => {
                // The carapace box has no segment endpoints; test its 8 corners + the
                // center so we still learn whether the box surface escapes the shell.
                println!(
                    "  {:<24} | {:>+7.3} {:>+8.4} {:>4} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4}",
                    label,
                    pwn,
                    pdist,
                    yn(pin),
                    "(box)",
                    "corners",
                    "↓",
                    "",
                    "",
                    ""
                );
                for sx in [-1.0f32, 1.0] {
                    for sy in [-1.0f32, 1.0] {
                        for sz in [-1.0f32, 1.0] {
                            let corner = center + half * Vec3::new(sx, sy, sz);
                            let (cwn, cdist, cin) = probe(corner);
                            windings.push(cwn);
                            if !cin {
                                endpoints_out += 1;
                                offenders.push((
                                    format!("{label} corner({sx:+.0},{sy:+.0},{sz:+.0})"),
                                    cdist,
                                ));
                            }
                            println!(
                                "      corner ({:+.0},{:+.0},{:+.0})         | {:>+7.3} {:>+8.4} {:>4}",
                                sx,
                                sy,
                                sz,
                                cwn,
                                cdist,
                                yn(cin)
                            );
                        }
                    }
                }
                let (ccwn, ccdist, ccin) = probe(center);
                println!(
                    "      center                       | {:>+7.3} {:>+8.4} {:>4}",
                    ccwn,
                    ccdist,
                    yn(ccin)
                );
            }
        }
    }

    // Watertight verdict: a clean closed mesh makes every winding land near an
    // integer (0 outside, ±1 inside). Count query points whose winding is clearly
    // fractional (off the nearest integer by >0.1) — many ⇒ the surface is open or
    // non-manifold and the IN/OUT calls near the boundary are soft.
    let fractional = windings
        .iter()
        .filter(|&&w| (w - w.round()).abs() > 0.1)
        .count();
    let clean = (hub_wn > 0.9) && (far_wn.abs() < 0.1) && fractional == 0;

    offenders.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    println!();
    println!(
        "watertight: {} — {}/{} query windings are fractional (off integer by >0.1); interior wn={:+.3}, exterior wn={:+.3}",
        if clean { "CLEAN/closed" } else { "MESSY/open" },
        fractional,
        windings.len(),
        hub_wn,
        far_wn
    );
    println!(
        "SUMMARY: {pivots_out} pivot(s) OUTSIDE mesh, {endpoints_out} endpoint/corner(s) OUTSIDE mesh"
    );
    println!("worst offenders (model units outside the surface):");
    for (label, d) in offenders.iter().take(12) {
        println!("  {:<34} {:+.4}", label, d);
    }
    if offenders.is_empty() {
        println!("  (none — every query point is inside the mesh)");
    }

    let pass = pivots_out == 0;
    println!(
        "VERDICT: {}",
        if pass {
            "all joint pivots lie INSIDE the mesh"
        } else {
            "some joint pivots lie OUTSIDE the mesh — see ranking"
        }
    );
    i32::from(!pass)
}
