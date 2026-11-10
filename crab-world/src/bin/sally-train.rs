//! Search for the crab policy (`docs/policy.md`) from a seeded random initialization on
//! the rigged Sally `.glb`, writing `policy.json` and one `train.jsonl` line per
//! generation into the output directory.

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
    /// Simulated ticks to spend: the first actor's episodes, then the whole
    /// generations the rest covers.
    #[arg(long, default_value_t = 1_250_000)]
    ticks: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let glb = std::fs::read(&args.rig).map_err(|e| format!("{}: {e}", args.rig.display()))?;
    let skeleton = Skeleton::from_glb(&glb)?;
    std::fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let config = Config {
        seed: args.seed,
        ..Config::default()
    };
    let generations = config.generations(args.ticks);
    if generations == 0 {
        return Err(format!(
            "{} ticks do not cover the first actor and one generation",
            args.ticks
        ));
    }
    let mut trainer = Trainer::new(&skeleton, &config)?;
    let mut log = std::fs::File::create(args.out.join("train.jsonl")).map_err(|e| e.to_string())?;
    let mut line = |value: serde_json::Value| writeln!(log, "{value}").map_err(|e| e.to_string());
    line(serde_json::json!({ "config": config, "first": trainer.kept() }))?;
    let save = |actor: &crab_world::policy::Actor| {
        let path = args.out.join("policy.json");
        let tmp = path.with_extension("json.part");
        std::fs::write(&tmp, serde_json::to_vec(actor).map_err(|e| e.to_string())?)
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    save(trainer.actor())?;
    for _ in 0..generations {
        let report = trainer.generation()?;
        line(serde_json::to_value(&report).map_err(|e| e.to_string())?)?;
        if report.improved {
            save(trainer.actor())?;
        }
        eprintln!(
            "generation {} of {generations}: fitness {:.4}",
            report.generation + 1,
            report.kept.fitness()
        );
    }
    Ok(())
}
