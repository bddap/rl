use std::process::ExitCode;

use clap::{Parser, Subcommand};
use crab_world::{CheckpointArgs, TrainConfig, bot, training};

use training::systems::STEPS_PER_ROLLOUT;

/// Train and evaluate the crab policy.
#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Cli {
    #[command(flatten)]
    otel: otel::OtelArgs,

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
    /// Run PPO against the crab world, checkpointing as it goes.
    Learn(LearnArgs),

    /// The chase eval: drive a checkpoint at a far ball and report metres closed.
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

    /// Physics ticks to run the policy for PER COMPASS BEARING (after a short settle
    /// drop each). The default is [`crab_world::eval::DEFAULT_EVAL_TICKS`] — the one
    /// place the chase-eval episode is defined, shared with the trainer's keep-best
    /// gate (bddap/rl#233); the bearing compass is [`crab_world::eval::EVAL_BEARINGS`]
    /// (bddap/rl#239).
    #[arg(long, default_value_t = crab_world::eval::DEFAULT_EVAL_TICKS)]
    ticks: u64,

    #[arg(long)]
    distance: Option<f32>,

    /// Gate mode: exit nonzero (after printing `EVAL_RESULT`) unless a real policy
    /// loaded AND it closed at least this many meters toward the target. Binds the
    /// pass/fail verdict to THIS eval — the one chase metric shared with the demo and
    /// GCR — so a release/promotion gate delegates here instead of growing a second
    /// behavior probe that drifts (bddap/bothouse#134). Without the flag, a missing
    /// checkpoint stays the legitimate exit-0 zero-action baseline the training
    /// monitor plots.
    #[arg(long)]
    min_progress: Option<f32>,

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

/// The bddap/rl#214 body preflight, MANDATORY for every mode that builds a crab world
/// (`learn`/`eval`/`--check-rest-colliders`): without it, they reached the body's SILENT
/// fallback and trained, judged, or audited the procedural body as if it were Sally.
/// The lib half logs the verdict (loud refusal / latched fallback error / positive body
/// line). Placed on the subcommands, not in the lib entry points, so the glb-less lib
/// tests (which call `headless_stack`/`run_eval` directly on the fallback body on
/// purpose) stay hermetic.
fn canonical_body(
    context: &str,
    allow_fallback: bool,
) -> Result<crab_world::mesh_fallback::BodyGate, String> {
    crab_world::mesh_fallback::require_canonical_body(context, allow_fallback)
        .map_err(|_| format!("{context}: refusing to run on a non-canonical body (see the preflight verdict above)"))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let _otel = otel::init("rl-train", cli.otel);
    // The one exit spine: every mode returns through here instead of calling
    // `process::exit` mid-match, so failures print one way and `_otel` always drops
    // (a scattered exit skipped the telemetry flush).
    match run(cli) {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("rl-train: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, String> {
    match cli.command {
        Some(Command::Learn(l)) => {
            let body_gate = canonical_body("learn", l.allow_fallback_body)?;
            training::inproc::run_learner(
                body_gate,
                &l.train,
                l.arch,
                training::inproc::default_workers(l.workers),
                l.horizon,
                l.iters,
                l.nice,
            );
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Eval(e)) => eval(e),
        None => dev_audit(cli.dev),
    }
}

fn eval(e: EvalArgs) -> Result<ExitCode, String> {
    let body_gate = canonical_body("eval", e.allow_fallback_body)?;
    let distance = e
        .distance
        .unwrap_or(crab_world::eval::DEFAULT_TARGET_DISTANCE_M);
    // A refused/mismatched checkpoint is a hard failure with NO `EVAL_RESULT` line
    // (the daemon greps that prefix; wrong-body baseline numbers plotted as training
    // progress would be the eval-side rl#214). Absent stays the legitimate
    // zero-action baseline below.
    let r = crab_world::eval::run_eval(body_gate, &e.checkpoint.checkpoint_dir, e.ticks, distance)
        .map_err(|refusal| format!("eval: {refusal}"))?;
    // The wire lines and their schema live with the report type (rl#270).
    print!("{}", r.wire_report());
    if !r.policy_loaded {
        eprintln!(
            "eval: no usable checkpoint at {} — the numbers above are the zero-action \
             rest-pose baseline, NOT a trained policy",
            e.checkpoint.checkpoint_dir.display()
        );
    }
    if let Some(min) = e.min_progress {
        // The literal `eval: FAIL` stderr prefix is the release gate's
        // refusal-vs-machinery seam (bothouse#148) — these verdicts keep their own
        // eprintln instead of riding the `rl-train:`-prefixed error spine.
        if !r.policy_loaded {
            eprintln!(
                "eval: FAIL — --min-progress {min} demands a loaded policy, but no \
                 usable checkpoint loaded from {}",
                e.checkpoint.checkpoint_dir.display()
            );
            return Ok(ExitCode::FAILURE);
        }
        if r.progress_m() < min {
            eprintln!(
                "eval: FAIL — policy closed {:.4} m toward the {:.2} m target at its \
                 worst bearing ({:.0}°), below the required --min-progress {min} m \
                 (dead/collapsed policy, or a dead bearing)",
                r.progress_m(),
                r.far.target_distance_m,
                r.far.worst().bearing_rad.to_degrees()
            );
            return Ok(ExitCode::FAILURE);
        }
        println!(
            "eval: PASS — worst-bearing progress {:.4} m ≥ --min-progress {min} m",
            r.progress_m()
        );
    }
    Ok(ExitCode::SUCCESS)
}

/// The no-subcommand DEV modes: rig-geometry audits, each a thin dispatch into
/// `crab-world` (rl#270). The audit prints its own report and verdict; a can't-run
/// error rides the main spine.
fn dev_audit(dev: DevArgs) -> Result<ExitCode, String> {
    if dev.verify_colliders {
        return audit(bot::rig_audit::verify_colliders());
    }
    if dev.verify_pivots {
        return audit(bot::rig_audit::verify_pivots());
    }
    if dev.check_rest_colliders {
        // The audit answers questions about SALLY's rest pose; on the fallback body it
        // prints an identical-format report with unrelated numbers (rl#234 was first
        // "measured" that way in a glb-less checkout). No fallback opt-in on purpose.
        canonical_body("check-rest-colliders", false)?;
        return audit(bot::collider_check::run());
    }

    eprintln!(
        "no mode selected. Train with `rl-train learn` (the sole trainer), or run a DEV \
         rig audit (--verify-colliders / --verify-pivots / --check-rest-colliders). The \
         windowed demo + screenshot are the `rl-demo` binary."
    );
    Ok(ExitCode::from(2))
}

fn audit(verdict: Result<bot::AuditVerdict, String>) -> Result<ExitCode, String> {
    verdict.map(|v| match v {
        bot::AuditVerdict::Pass => ExitCode::SUCCESS,
        bot::AuditVerdict::Fail => ExitCode::FAILURE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// See `game`'s twin: clap's own validity checks only run when the command is built.
    #[test]
    fn cli_is_well_formed() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
