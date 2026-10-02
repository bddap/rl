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

    /// Roll a frozen checkpoint through the training worlds until a tick budget is
    /// spent or an rl#343 integrity violation trips (rl#351). Exit 3 = reproduced.
    Repro(ReproArgs),
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
}

#[derive(Parser, Debug, Clone)]
struct ReproArgs {
    /// `--ticks` is the budget PER WORKER, so a worker's stream is the same for any
    /// `--workers`.
    #[command(flatten)]
    train: TrainConfig,

    #[arg(long)]
    workers: Option<usize>,

    #[arg(long, default_value_t = STEPS_PER_ROLLOUT as u64)]
    horizon: u64,

    /// Roll only this worker index (its seed stream), e.g. to replay a hit.
    #[arg(long)]
    only_worker: Option<usize>,

    /// Save a plant snapshot of this env every tick from `--capture-from-tick`.
    #[arg(long, requires_all = ["only_worker", "capture_from_tick", "capture_dir"])]
    capture_env: Option<usize>,

    #[arg(long)]
    capture_from_tick: Option<u64>,

    #[arg(long)]
    capture_dir: Option<std::path::PathBuf>,
}

#[derive(Parser, Debug, Clone)]
struct EvalArgs {
    // The daemon points `--checkpoint-dir` at the LIVE training checkpoint to judge
    // the run in flight.
    #[command(flatten)]
    checkpoint: CheckpointArgs,

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
            );
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Eval(e)) => eval(e),
        Some(Command::Repro(r)) => repro(r),
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

fn repro(r: ReproArgs) -> Result<ExitCode, String> {
    let capture = match (r.capture_env, r.capture_from_tick, r.capture_dir) {
        (Some(env), Some(from_tick), Some(dir)) => {
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            Some(training::inproc::repro::Capture {
                env,
                from_tick,
                dir,
            })
        }
        _ => None,
    };
    let outcomes = training::inproc::repro::run_repro(
        &r.train,
        training::inproc::default_workers(r.workers),
        r.only_worker,
        r.horizon,
        r.train.ticks,
        capture,
    )?;
    let mut hits = 0;
    for o in &outcomes {
        match &o.violation {
            None => println!("REPRO_WORKER {} ticks {} clean", o.worker, o.ticks),
            Some(msg) => {
                hits += 1;
                let head = msg.lines().next().unwrap_or_default();
                println!("REPRO_WORKER {} VIOLATION {head}", o.worker);
            }
        }
    }
    println!(
        "REPRO_RESULT workers {} violations {hits}",
        outcomes.len()
    );
    Ok(if hits > 0 {
        ExitCode::from(3)
    } else {
        ExitCode::SUCCESS
    })
}

fn eval(e: EvalArgs) -> Result<ExitCode, String> {
    // A refused/mismatched checkpoint is a hard failure with NO `EVAL_RESULT` line
    // (the daemon greps that prefix; wrong-body baseline numbers plotted as training
    // progress would be the eval-side rl#214). Absent stays the legitimate
    // zero-action baseline below.
    let r = crab_world::eval::run_eval(
        &e.checkpoint.checkpoint_dir,
        crab_world::eval::DEFAULT_EVAL_TICKS,
        crab_world::eval::DEFAULT_TARGET_DISTANCE_M,
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
