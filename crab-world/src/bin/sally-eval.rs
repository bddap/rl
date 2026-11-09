//! Score a baseline or trained policy on the crab evals (`docs/evals.md`) with the rigged Sally
//! `.glb` that `scripts/sally/sally.sh rig` bakes, and print one `SALLY_EVAL <json>` line.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use crab_world::eval::{Policy, RandomPolicy, Task, ZeroPolicy, joint_torque_max, score};
use crab_world::policy::{Actor, Trained};
use crab_world::sally::Skeleton;

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Kind {
    Zero,
    Random,
    /// The `policy.json` that `sally-train` writes, named by `--weights`, acting its mean.
    Trained,
    /// The same, acting samples around its mean, as in training; one episode per seed.
    Sampled,
}

#[derive(Parser)]
struct Args {
    rig: PathBuf,
    #[arg(long, value_enum)]
    policy: Kind,
    #[arg(long)]
    weights: Option<PathBuf>,
    /// One episode per seed per task, seeds 0 up; the deterministic policies run once.
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u64).range(1..))]
    seeds: u64,
    /// The tasks to run; every task by default.
    #[arg(long, value_enum)]
    task: Vec<Task>,
    /// Writes every episode's bone poses into this directory as `<task>-<seed>.json`.
    #[arg(long)]
    record: Option<PathBuf>,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let glb = std::fs::read(&args.rig).map_err(|e| format!("{}: {e}", args.rig.display()))?;
    let skeleton = Skeleton::from_glb(&glb)?;
    let trained = match (args.policy, &args.weights) {
        (Kind::Trained | Kind::Sampled, Some(path)) => {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let actor: Actor =
                serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            Some(Trained::new(&skeleton, actor)?)
        }
        (Kind::Trained | Kind::Sampled, None) => {
            return Err("a trained policy needs --weights".into());
        }
        (_, Some(_)) => return Err("--weights names a trained policy".into()),
        (_, None) => None,
    };
    let seeds: Vec<u64> = match args.policy {
        Kind::Zero | Kind::Trained => vec![0],
        Kind::Random | Kind::Sampled => (0..args.seeds).collect(),
    };
    let tasks: Vec<Task> = Task::ALL
        .into_iter()
        .filter(|t| args.task.is_empty() || args.task.contains(t))
        .collect();
    if let Some(dir) = &args.record {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let make = |seed: u64| -> Box<dyn Policy> {
        match (args.policy, &trained) {
            (Kind::Random, _) => Box::new(RandomPolicy::new(seed)),
            (Kind::Sampled, Some(trained)) => Box::new(trained.clone().sampled(seed)),
            (_, Some(trained)) => Box::new(trained.clone()),
            (_, None) => Box::new(ZeroPolicy),
        }
    };
    let tasks = score(&skeleton, &tasks, &seeds, &make, args.record.as_deref())?;
    let report = serde_json::json!({
        "policy": args.policy.to_possible_value().map(|v| v.get_name().to_owned()),
        "weights": args.weights,
        "seeds": seeds,
        "joint_torque_max_nm": joint_torque_max(&skeleton)?,
        "tasks": tasks,
    });
    println!("SALLY_EVAL {report}");
    Ok(())
}
