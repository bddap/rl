

use bevy::prelude::*;
use clap::{Parser, Subcommand};
use crab_world::{CheckpointArgs, TrainConfig, bot, training};

use training::systems::STEPS_PER_ROLLOUT;

#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Cli {
    #[command(flatten)]
    dev: DevArgs,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Parser, Debug, Clone)]
struct DevArgs {
    #[arg(long)]
    verify_colliders: bool,

    #[arg(long)]
    verify_pivots: bool,

    #[arg(long)]
    check_rest_colliders: bool,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Learn(LearnArgs),

    Eval(EvalArgs),
}

#[derive(Parser, Debug, Clone)]
struct LearnArgs {
    #[command(flatten)]
    train: TrainConfig,

    #[arg(long)]
    workers: Option<usize>,

    /// Policy architecture for a FRESH start (an empty --checkpoint-dir); default
    /// mlp512x3. On a RESUME the checkpoint's arch tag is authoritative and this flag
    /// is only a cross-check — a value that disagrees with the tag ABORTS (never a
    /// cold start over the trained policy, never a silently ignored flag).
    #[arg(long, value_parser = parse_arch)]
    arch: Option<bot::arch::ArchId>,

    #[arg(long, default_value_t = STEPS_PER_ROLLOUT as u64)]
    horizon: u64,

    #[arg(long, default_value_t = 0)]
    iters: u64,

    #[arg(long, default_value_t = 10)]
    nice: i32,

    /// DEV: train the procedural fallback body when no usable `sally.glb` resolves,
    /// instead of refusing to start (bddap/rl#214). The checkpoints it writes carry
    /// body digest 0 and will be REFUSED by every canonical-body surface — a policy
    /// trained on the fallback is not Sally.
    #[arg(long)]
    allow_fallback_body: bool,
}

#[derive(Parser, Debug, Clone)]
struct EvalArgs {
    // The daemon points `--checkpoint-dir` at the LIVE training checkpoint to judge
    // the run in flight.
    #[command(flatten)]
    checkpoint: CheckpointArgs,

    /// Physics ticks to run the policy for (after a short settle drop). Defaults to one
    /// training episode horizon ([`crab_world::training::systems::MAX_EPISODE_TICKS`],
    /// ~23 s of crab time at 64 Hz) — enough for a working gait to traverse a far target.
    #[arg(long, default_value_t = crab_world::training::systems::MAX_EPISODE_TICKS as u64)]
    ticks: u64,

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
    let _otel = otel::init("rl-train");
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Learn(l)) => {
            let body_gate = require_canonical_body_or_exit("learn", l.allow_fallback_body);
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
            let r = match crab_world::eval::run_eval(body_gate, &e.checkpoint.checkpoint_dir, e.ticks, distance)
            {
                Ok(r) => r,
                Err(refusal) => {
                    eprintln!("eval: {refusal}");
                    std::process::exit(1);
                }
            };
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
                    e.checkpoint.checkpoint_dir.display()
                );
            }
            return;
        }
        None => {}
    }

    let dev = cli.dev;

    if dev.verify_colliders {
        std::process::exit(verify_colliders());
    }

    if dev.verify_pivots {
        std::process::exit(verify_pivots());
    }

    if let Some(p) = bot::meshfit::model_path() {
        match bot::meshfit::LoadedModel::load(&p) {
            Err(e) => {
                eprintln!("crab model {p:?}: {e}");
                std::process::exit(1);
            }
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

    if dev.check_rest_colliders {
        std::process::exit(bot::collider_check::run());
    }

    eprintln!(
        "no mode selected. Train with `rl-train learn` (the sole trainer), or run a DEV \
         rig audit (--verify-colliders / --verify-pivots / --check-rest-colliders). The \
         windowed demo + screenshot are the `rl-demo` binary."
    );
    std::process::exit(2);
}

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

    let fmt = |x: Option<f32>| x.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
    let mut ranking: Vec<(String, f32, bool)> = Vec::new();
    let mut any_fail = false;

    for rc in bot::rig::rest_colliders(&recipe) {
        let label = format!("{:?}", rc.part);
        let (score, rnorm, fail) = match rc.shape {
            RestShape::Capsule { a, b, radius } => {
                let pts = clouds.get(&rc.part).map(|p| p.as_slice()).unwrap_or(&[]);
                let s = score_capsule(pts, a, b, radius);
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

    let soup = bot::meshfit::MeshContainment::new(pos, tris);
    let signed_vol = soup.signed_vol();
    let orient = soup.orient();
    let probe = |p: Vec3| {
        let c = soup.probe(p);
        (c.wn, c.signed_dist, c.inside)
    };

    let hub = bot::rig::rest_colliders(&recipe)
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

    let mut pivots_out = 0usize;
    let mut endpoints_out = 0usize;
    let mut offenders: Vec<(String, f32)> = Vec::new();
    let mut windings: Vec<f32> = Vec::new();
    let yn = |b: bool| if b { "IN" } else { "OUT" };

    for rc in bot::rig::rest_colliders(&recipe) {
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
