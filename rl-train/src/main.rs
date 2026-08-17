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
}

#[derive(Parser, Debug, Clone)]
struct EvalArgs {
    // The daemon points `--checkpoint-dir` at the LIVE training checkpoint to judge
    // the run in flight.
    #[command(flatten)]
    checkpoint: CheckpointArgs,

    /// Physics ticks to run the policy for PER (heading, start) PAIR (after a short
    /// settle drop each). The default is [`crab_world::eval::DEFAULT_EVAL_TICKS`] —
    /// the one place the chase-eval episode is defined, shared with the trainer's
    /// keep-best gate (bddap/rl#233). 0 would read a default-constructed episode as
    /// a plausible hard zero, so it refuses at parse (rl#341 S1-3).
    #[arg(long, default_value_t = crab_world::eval::DEFAULT_EVAL_TICKS,
          value_parser = clap::value_parser!(u64).range(1..))]
    ticks: u64,

    /// DIAGNOSTIC: far-ball distance in metres, a finite in-band length (validated —
    /// NaN/∞/negative/out-of-band used to panic, hang, or silently rescale the gate,
    /// rl#341 S1-3). Non-default values also move the deterministic start set (starts
    /// are drawn progressable at THIS distance) — the wire's `target_m=` and
    /// provenance keys record it. Refused in gate mode.
    #[arg(long, value_parser = parse_distance)]
    distance: Option<f32>,

    /// Terrain relief amplitude: scales the committed bake's datum-shifted heights by
    /// this ONE scalar (1 = the canonical bake bit-identically; 0 = a plane). The
    /// whole eval — start derivation, episodes, probes — runs on the scaled grid
    /// (owner 08-03 / rl#341). Ground too rough to seat progressable starts refuses
    /// loudly. Refused in gate mode (the gate judges the pinned instrument).
    #[arg(long, default_value_t = 1.0, value_parser = parse_amplitude)]
    terrain_amplitude: f32,

    /// Gate mode: exit nonzero (after printing `EVAL_RESULT`) unless a real policy
    /// loaded AND its mean pair progress is at least this many meters. Binds the
    /// pass/fail verdict to THIS eval — the one chase metric shared with the demo and
    /// GCR — so a release/promotion gate delegates here instead of growing a second
    /// behavior probe that drifts (bddap/bothouse#134). Demands the PINNED instrument
    /// (default ticks/distance/amplitude): a resized episode or rescaled arena would
    /// silently rescale the bar it enforces (rl#341 S1-3). Without the flag, a
    /// missing checkpoint stays the legitimate exit-0 zero-action baseline the
    /// training monitor plots.
    #[arg(long)]
    min_progress: Option<f32>,
}

/// clap value-parser for `--arch`: delegates to the registry's `TryFrom<String>`, whose
/// error already names the unknown arch and lists the known ones.
fn parse_arch(s: &str) -> Result<bot::arch::ArchId, String> {
    bot::arch::ArchId::try_from(s.to_string())
}

/// clap value-parser for `--distance`: the eval owns its own domain
/// ([`crab_world::eval::validate_target_distance`], rl#341 S1-3).
fn parse_distance(s: &str) -> Result<f32, String> {
    let d: f32 = s.parse().map_err(|e| format!("{e}"))?;
    crab_world::eval::validate_target_distance(d)
}

/// clap value-parser for `--terrain-amplitude`: same domain the grid constructor
/// enforces ([`crab_world::terrain::TerrainGrid::gcr_with_amplitude`]).
fn parse_amplitude(s: &str) -> Result<f32, String> {
    let a: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if a.is_finite() && a >= 0.0 {
        Ok(a)
    } else {
        Err(format!(
            "terrain amplitude must be a finite non-negative scalar, got {a}"
        ))
    }
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
            training::inproc::run_learner(
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
    // Gate mode judges the PINNED instrument only (rl#341 S1-3): a shorter episode,
    // a nearer ball, or a flattened arena would silently rescale the bar — e.g.
    // `--distance 1 --min-progress 0.3` passed a 24 m-chase gate on a 0.3 m twitch.
    if e.min_progress.is_some()
        && (e.ticks != crab_world::eval::DEFAULT_EVAL_TICKS
            || e.distance.is_some()
            || e.terrain_amplitude != 1.0)
    {
        return Err(
            "--min-progress judges the pinned instrument; drop --ticks/--distance/\
             --terrain-amplitude (a resized episode or rescaled arena would silently \
             rescale the gate)"
                .to_string(),
        );
    }
    let distance = e
        .distance
        .unwrap_or(crab_world::eval::DEFAULT_TARGET_DISTANCE_M);
    // A refused/mismatched checkpoint is a hard failure with NO `EVAL_RESULT` line
    // (the daemon greps that prefix; wrong-body baseline numbers plotted as training
    // progress would be the eval-side rl#214). Absent stays the legitimate
    // zero-action baseline below.
    let r = crab_world::eval::run_eval(
        &e.checkpoint.checkpoint_dir,
        e.ticks,
        distance,
        e.terrain_amplitude,
    )
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
    if r.plant_unbounded() {
        // Unconditional (not just gate mode): an exploding plant is a plant bug, and
        // every consumer — the eval monitor, a hand run — must see it as a hard
        // fault, never as a slow eval with weird numbers (bddap/rl#315: the
        // rigid-contact explosion surfaced as release unit timeouts for days before
        // anyone read the magnitudes).
        eprintln!(
            "eval: FAIL — plant unbounded: a carapace strayed more than {:.0} m from \
             its spawn (or went non-finite unhealed) mid-episode; the plant is \
             injecting energy (bddap/rl#315). Exploded episodes were cut and forfeited.",
            crab_world::eval::PLANT_POSITION_BOUND_M
        );
        return Ok(ExitCode::FAILURE);
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
            let min_pair = r.far.min_pair();
            eprintln!(
                "eval: FAIL — policy closed a mean {:.4} m toward the {:.2} m target \
                 over {} (heading, start) pairs (worst pair {:.4} m @ {:.0}°, {} \
                 rescued), below the required --min-progress {min} m (dead/collapsed \
                 policy, or a dead heading dragging the mean)",
                r.progress_m(),
                r.far.target_distance_m,
                r.far.pairs.len(),
                min_pair.progress_m,
                min_pair.bearing_rad.to_degrees(),
                r.far.rescued_pairs(),
            );
            return Ok(ExitCode::FAILURE);
        }
        println!(
            "eval: PASS — mean pair progress {:.4} m ≥ --min-progress {min} m",
            r.progress_m()
        );
    }
    Ok(ExitCode::SUCCESS)
}

/// The no-subcommand DEV mode: the rest-pose collider audit, a thin dispatch into
/// `crab-world` (rl#270). The audit prints its own report and verdict; a can't-run
/// error rides the main spine. (The collider<->mesh fit audits moved to the offline
/// fitter with the rest of the fitting code: `cargo run -p meshfit -- verify-colliders`
/// / `verify-pivots`, bddap/rl#20.)
fn dev_audit(dev: DevArgs) -> Result<ExitCode, String> {
    if dev.check_rest_colliders {
        return audit(bot::collider_check::run());
    }

    eprintln!(
        "no mode selected. Train with `rl-train learn` (the sole trainer), or run the DEV \
         rest-pose audit (--check-rest-colliders; the mesh-fit audits live in the offline \
         `meshfit` tool). The windowed demo + screenshot are the `rl-demo` binary."
    );
    Ok(ExitCode::from(2))
}

fn audit(verdict: Result<bot::AuditVerdict, String>) -> Result<ExitCode, String> {
    verdict.map(ExitCode::from)
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
