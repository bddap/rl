use anyhow::Result;
use clap::Parser;

use super::shared::{nn_crab_policy, write_tick_hash_log};

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 1200)]
    ticks: u64,
    #[arg(long, default_value_t = 100)]
    log_every: u64,
    #[arg(long, default_value_t = super::shared::MATCH_SEED)]
    seed: u64,
    #[arg(long, value_name = "FILE")]
    hash_log: Option<std::path::PathBuf>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    use net::external_crab::run_headless_probe;

    let (dir, policy) = nn_crab_policy(args.checkpoint.clone())?;
    println!("nn-crab-probe: checkpoint={}", dir.display());
    println!("nn-crab-probe: seed={:#x} ticks={}", args.seed, args.ticks);

    let log_every = if args.hash_log.is_some() {
        1
    } else {
        args.log_every
    };
    let samples = run_headless_probe(policy, args.seed, args.ticks, log_every);
    if samples.is_empty() {
        anyhow::bail!("nn-crab-probe: no samples — the crab never stepped");
    }

    if let Some(path) = &args.hash_log {
        write_tick_hash_log(path, samples.iter().map(|s| (s.tick, s.state_hash)))?;
        println!(
            "nn-crab-probe: wrote {} per-tick hashes to {}",
            samples.len(),
            path.display()
        );
    }

    println!("\n  tick   crab_x   crab_z   dist  | carapace x/y/z (walks?)  | claw→tgt");
    for s in &samples {
        println!(
            "  {:>5}  {:>7.2}  {:>7.2}  {:>6.2} | {:>7.2} {:>5.2} {:>7.2}  | {:>7.3}",
            s.tick,
            s.crab_x_m,
            s.crab_z_m,
            s.dist_to_prey_m,
            s.carapace_arena_x,
            s.carapace_y,
            s.carapace_arena_z,
            s.min_claw_to_target_m,
        );
    }

    let first = samples.first().unwrap().dist_to_prey_m;
    let last = samples.last().unwrap().dist_to_prey_m;
    let closed = first - last;
    println!(
        "\nnn-crab-probe: distance to player {first:.3} m → {last:.3} m  (closed {closed:.3} m)"
    );

    // The determinism re-run loads its own policy (Policy is arm-on-load, rl#241). A
    // checkpoint swap landing between the two loads fails the diff LOUDLY — fine for an
    // offline gate; what must never happen is a silent statue run.
    let (_, policy_b) = nn_crab_policy(args.checkpoint)?;
    let again = run_headless_probe(policy_b, args.seed, args.ticks, log_every);
    let hash_a = samples.last().unwrap().state_hash;
    let hash_b = again.last().map(|s| s.state_hash).unwrap_or(0);
    let traj_match = samples.len() == again.len()
        && samples
            .iter()
            .zip(&again)
            .all(|(a, b)| a.state_hash == b.state_hash);
    println!(
        "nn-crab-probe: determinism — final hash A={hash_a:#018x} B={hash_b:#018x} ({}), \
         full trajectory {}",
        if hash_a == hash_b { "MATCH" } else { "DIFFER" },
        if traj_match { "MATCHES" } else { "DIFFERS" },
    );

    // Verdict binds to reproducibility ONLY. The distance table above is a diagnostic:
    // this scenario (a ~21 m player spawn) is outside the trained chase domain, so a
    // distance threshold here fails known-good policies (bddap/rl#144). Behavioral
    // pass/fail lives in the one shared chase metric, `rl-train eval --min-progress`
    // (bddap/bothouse#134) — never a second gate here that drifts from it.
    if traj_match {
        println!(
            "nn-crab-probe: PASS — trajectory reproducible (closed {closed:.3} m, diagnostic only)"
        );
        Ok(())
    } else {
        anyhow::bail!("nn-crab-probe: FAIL — trajectory not reproducible (closed {closed:.3} m)")
    }
}
