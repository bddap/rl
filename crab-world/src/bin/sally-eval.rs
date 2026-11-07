//! Score a baseline policy on the crab evals (`docs/evals.md`) with the rigged Sally
//! `.glb` that `scripts/sally/sally.sh rig` bakes, and print one `SALLY_EVAL <json>` line.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use crab_world::eval::{Policy, RandomPolicy, ZeroPolicy, joint_torque_max, score};
use crab_world::sally::Skeleton;

#[derive(Clone, Copy, ValueEnum)]
enum Baseline {
    Zero,
    Random,
}

impl Baseline {
    fn policy(self, seed: u64) -> Box<dyn Policy> {
        match self {
            Baseline::Zero => Box::new(ZeroPolicy),
            Baseline::Random => Box::new(RandomPolicy::new(seed)),
        }
    }
}

#[derive(Parser)]
struct Args {
    rig: PathBuf,
    #[arg(long, value_enum)]
    policy: Baseline,
    /// One episode per seed per task, seeds 0 up; the zero policy runs once.
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u64).range(1..))]
    seeds: u64,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let glb = std::fs::read(&args.rig).map_err(|e| format!("{}: {e}", args.rig.display()))?;
    let skeleton = Skeleton::from_glb(&glb)?;
    let seeds: Vec<u64> = match args.policy {
        Baseline::Zero => vec![0],
        Baseline::Random => (0..args.seeds).collect(),
    };
    let tasks = score(&skeleton, &seeds, &|seed| args.policy.policy(seed))?;
    let report = serde_json::json!({
        "policy": args.policy.to_possible_value().map(|v| v.get_name().to_owned()),
        "seeds": seeds,
        "joint_torque_max_nm": joint_torque_max(&skeleton)?,
        "tasks": tasks,
    });
    println!("SALLY_EVAL {report}");
    Ok(())
}
