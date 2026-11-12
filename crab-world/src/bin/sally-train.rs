//! Search for the crab policy (`docs/policy.md`) from the rest-holding actor on the
//! rigged Sally `.glb`, writing `policy.json` and one `train.jsonl` line per generation
//! into the output directory.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use crab_world::sally::Skeleton;
use crab_world::train::{Config, Fitnesses, Report, Scored, Trainer};
use serde::Serialize;

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

/// The log's first line: what the run searches with and what its first actor scored.
#[derive(Serialize)]
struct Start<'a> {
    config: &'a Config,
    first: &'a Scored,
}

/// What an earlier run of this search left in its log: the log's whole lines, a run
/// killed while writing leaving its last line cut short, and every generation's
/// fitnesses in them. Refuses a log this run cannot go on from.
fn logged<'a>(
    text: &'a str,
    start: &str,
    budget: usize,
) -> Result<(&'a str, Vec<Fitnesses>), String> {
    let whole = &text[..text.rfind('\n').map_or(0, |at| at + 1)];
    let mut lines = whole.lines();
    let generations = match lines.next() {
        None => Vec::new(),
        Some(first) if first == start => lines
            .enumerate()
            .map(|(generation, line)| Report::logged(line, generation))
            .collect::<Result<Vec<_>, String>>()?,
        Some(_) => return Err("it does not start as this run does".into()),
    };
    if generations.len() > budget {
        return Err(format!(
            "it holds {} generations and {budget} are asked for",
            generations.len()
        ));
    }
    Ok((whole, generations))
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
    let budget = config.generations(args.ticks);
    if budget == 0 {
        return Err(format!(
            "{} ticks do not cover the first actor and one generation",
            args.ticks
        ));
    }
    let trainer = Trainer::new(&skeleton, &config)?;
    let start = Start {
        config: &config,
        first: trainer.kept(),
    };
    let start = serde_json::to_string(&start).map_err(|e| e.to_string())?;

    let log_path = args.out.join("train.jsonl");
    let in_log = |e: String| format!("{}: {e}; remove it to start over", log_path.display());
    let text = match std::fs::read_to_string(&log_path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("{}: {e}", log_path.display())),
    };
    let (whole, logged) = logged(&text, &start, budget).map_err(in_log)?;
    let mut trainer = trainer.replay(&logged).map_err(in_log)?;

    let mut log = std::fs::File::options()
        .create(true)
        .append(true)
        .open(&log_path)
        .and_then(|log| log.set_len(whole.len() as u64).map(|()| log))
        .map_err(|e| format!("{}: {e}", log_path.display()))?;
    let mut line = |json: String| {
        log.write_all((json + "\n").as_bytes())
            .map_err(|e| e.to_string())
    };
    if whole.is_empty() {
        line(start)?;
    }
    let save = |actor: &crab_world::policy::Actor| {
        let path = args.out.join("policy.json");
        let tmp = path.with_extension("json.part");
        std::fs::write(&tmp, serde_json::to_vec(actor).map_err(|e| e.to_string())?)
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    save(trainer.actor())?;
    for _ in logged.len()..budget {
        let report = trainer.generation()?;
        line(serde_json::to_string(&report).map_err(|e| e.to_string())?)?;
        if report.improved {
            save(trainer.actor())?;
        }
        eprintln!(
            "generation {} of {budget}: fitness {:.4}",
            report.generation + 1,
            report.kept.fitness()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::logged;
    use crab_world::train::Fitnesses;

    const START: &str = r#"{"config":1}"#;

    fn generation(generation: usize) -> String {
        format!(
            r#"{{"generation":{generation},"candidates":[{generation}.5,null],"refinements":[2.0]}}"#
        )
    }

    #[test]
    fn a_log_goes_on_from_its_whole_lines() {
        let two = format!("{START}\n{}\n{}\n", generation(0), generation(1));
        let of = |candidate: f32| Fitnesses {
            candidates: vec![candidate, f32::NEG_INFINITY],
            refinements: vec![2.0],
        };
        let fitnesses = vec![of(0.5), of(1.5)];
        assert_eq!(
            logged(&two, START, 2),
            Ok((two.as_str(), fitnesses.clone()))
        );

        let cut = format!("{two}{}", &generation(2)[..40]);
        assert_eq!(logged(&cut, START, 3), Ok((two.as_str(), fitnesses)));

        let start_only = format!("{START}\n");
        assert_eq!(
            logged(&start_only, START, 1),
            Ok((start_only.as_str(), vec![]))
        );
        for nothing in ["", &START[..5]] {
            assert_eq!(logged(nothing, START, 1), Ok(("", vec![])));
        }
    }

    #[test]
    fn a_log_this_run_cannot_go_on_from_is_refused() {
        let two = format!("{START}\n{}\n{}\n", generation(0), generation(1));
        let refused =
            |text: &str, start: &str, budget: usize| logged(text, start, budget).unwrap_err();
        assert!(refused(&two, r#"{"config":2}"#, 2).contains("does not start"));
        assert!(refused(&two, START, 1).contains("holds 2 generations and 1 are asked for"));
        let skipped = format!("{START}\n{}\n", generation(1));
        assert!(refused(&skipped, START, 2).contains("generation 1"));
        let garbled = format!("{START}\nnot json\n");
        assert!(logged(&garbled, START, 2).is_err());
    }
}
