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

    #[command(subcommand)]
    command: Option<Command>,
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

    /// DIAGNOSTIC: far-ball distance in metres, a finite in-band length. Non-default
    /// values also move the deterministic start set (starts are drawn progressable at
    /// THIS distance) — the wire's `target_m=` and provenance keys record it.
    #[arg(long, value_parser = parse_distance)]
    distance: Option<f32>,

    /// Terrain relief amplitude: scales the committed bake's datum-shifted heights by
    /// this ONE scalar (1 = the canonical bake bit-identically; 0 = a plane). The
    /// whole eval — start derivation, episodes, probes — runs on the scaled grid
    /// (rl#341). Ground too rough to seat progressable starts refuses loudly.
    #[arg(long, default_value_t = 1.0, value_parser = parse_amplitude)]
    terrain_amplitude: f32,
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
        None => {
            eprintln!(
                "no mode selected. Train with `rl-train learn` (the sole trainer); the mesh-fit \
                 audits live in the offline `meshfit` tool. The windowed demo + screenshot are \
                 the `rl-demo` binary."
            );
            Ok(ExitCode::from(2))
        }
    }
}

fn eval(e: EvalArgs) -> Result<ExitCode, String> {
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
        // An exploding plant is a plant bug: every consumer — the eval monitor, a hand
        // run — must see a hard fault, never a slow eval with weird numbers (rl#315).
        eprintln!(
            "eval: FAIL — plant unbounded: a carapace strayed more than {:.0} m from \
             its spawn (or went non-finite unhealed) mid-episode; the plant is \
             injecting energy (bddap/rl#315). Exploded episodes were cut and forfeited.",
            crab_world::eval::PLANT_POSITION_BOUND_M
        );
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
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
