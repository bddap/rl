//! Train the crab policy (`docs/policy.md`) from a seeded random initialization on
//! the rigged Sally `.glb`, writing `policy.json` and one `train.jsonl` line per
//! iteration into the output directory.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use crab_world::sally::Skeleton;
use crab_world::train::{Config, Trainer};

#[derive(Parser)]
struct Args {
    rig: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 320)]
    iterations: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = Config::default().slots as u64, value_parser = clap::value_parser!(u64).range(1..))]
    slots: u64,
    /// Writes `policy.json` every this many iterations, and after the last.
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u64).range(1..))]
    save_every: u64,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let glb = std::fs::read(&args.rig).map_err(|e| format!("{}: {e}", args.rig.display()))?;
    let skeleton = Skeleton::from_glb(&glb)?;
    std::fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let config = Config {
        seed: args.seed,
        slots: args.slots as usize,
        ..Config::default()
    };
    let mut log = std::fs::File::create(args.out.join("train.jsonl")).map_err(|e| e.to_string())?;
    writeln!(log, "{}", serde_json::json!({ "config": config })).map_err(|e| e.to_string())?;
    let mut trainer = Trainer::new(&skeleton, config)?;
    let save = |actor: &crab_world::policy::Actor| {
        if !actor
            .net
            .params
            .iter()
            .chain(&actor.log_std)
            .all(|p| p.is_finite())
        {
            return Err("training diverged: the policy is not finite".to_string());
        }
        let path = args.out.join("policy.json");
        let tmp = path.with_extension("json.part");
        std::fs::write(&tmp, serde_json::to_vec(actor).map_err(|e| e.to_string())?)
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    let mut ticks = 0;
    trainer.run(args.iterations, &mut |report, actor| {
        ticks += report.ticks;
        writeln!(
            log,
            "{}",
            serde_json::to_string(report).map_err(|e| e.to_string())?
        )
        .map_err(|e| e.to_string())?;
        let finished = report.iteration + 1;
        if (finished as u64).is_multiple_of(args.save_every) || finished == args.iterations {
            save(actor)?;
        }
        eprintln!(
            "iteration {finished}: {ticks} ticks, mean reward {:.4}, kl {:.4}, std {:.3}",
            report.mean_reward,
            report.approx_kl,
            report.std.iter().sum::<f32>() / report.std.len() as f32
        );
        Ok(())
    })
}
