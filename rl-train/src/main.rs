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

    /// Roll a frozen checkpoint through the training worlds until each worker has
    /// spent `--ticks`, counting rl#343 integrity trips; a trip rebuilds that
    /// worker's world on a fresh seed (rl#351). Exit 3 = at least one trip.
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
    #[command(flatten)]
    train: TrainConfig,

    #[arg(long)]
    workers: Option<usize>,

    /// The first worker id: a worker's id sets its seed stream, so N processes of
    /// `--workers 1 --first-worker i` roll the same streams as one `--workers N`
    /// process, without the cross-worker contention that slows a shared process ~4×.
    #[arg(long, default_value_t = 0)]
    first_worker: usize,

    #[arg(long, default_value_t = STEPS_PER_ROLLOUT as u64)]
    horizon: u64,

    /// Save each near miss's pre-step plant snapshot here, for `sally-replay`.
    #[arg(long)]
    capture_dir: Option<std::path::PathBuf>,

    /// Solver counts for the rolled ticks, `OUTER,PGS,STAB[xSUBSTEPS]`; default shipped.
    #[arg(long, default_value_t = crab_world::physics::SHIPPED_SOLVER)]
    solver: crab_world::physics::SolverCounts,
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
    if let Some(dir) = &r.capture_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let started = std::time::Instant::now();
    let report = training::inproc::repro::run_repro(
        &r.train,
        r.first_worker..r.first_worker + training::inproc::default_workers(r.workers),
        r.horizon,
        r.train.ticks,
        r.capture_dir,
        r.solver,
    )?;
    let wall = started.elapsed().as_secs_f64();
    for o in &report.workers {
        println!(
            "REPRO_WORKER {} ticks {} trips {}",
            o.worker,
            o.ticks,
            o.trips.len()
        );
        for t in &o.trips {
            println!(
                "REPRO_TRIP worker {} {}",
                o.worker,
                t.lines().next().unwrap_or_default()
            );
        }
    }
    let ticks: u64 = report.workers.iter().map(|o| o.ticks).sum();
    let trips: usize = report.workers.iter().map(|o| o.trips.len()).sum();
    println!(
        "REPRO_RESULT solver {} workers {} ticks {ticks} trips {trips} per_M_ticks {:.3} \
         wall_s {wall:.0} cpu_s {:.0} ticks_per_wall_s {:.1} ticks_per_cpu_s {:.1}",
        r.solver,
        report.workers.len(),
        trips as f64 * 1e6 / ticks.max(1) as f64,
        report.cpu_secs,
        ticks as f64 / wall,
        ticks as f64 / report.cpu_secs,
    );
    Ok(if trips > 0 {
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
