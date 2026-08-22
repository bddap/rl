use anyhow::Result;
use clap::Parser;

use super::shared::nn_crab_policy;

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "DIR", env = super::shared::CHECKPOINT_ENV)]
    checkpoint: Option<std::path::PathBuf>,
    /// Measured window, in net ticks (each owing 2–3 steps on the 64:30 staircase).
    #[arg(long, default_value_t = 512)]
    ticks: u64,
    /// Settle + spin-up ticks excluded from the stats.
    #[arg(long, default_value_t = 256)]
    warmup: u64,
    #[arg(long, default_value_t = super::shared::MATCH_SEED)]
    seed: u64,
    /// Per-step samples as CSV, for offline plots.
    #[arg(long, value_name = "FILE")]
    csv: Option<std::path::PathBuf>,
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn dist(label: &str, vals: &mut [f64]) {
    vals.sort_by(f64::total_cmp);
    println!(
        "  {label:<26} p10 {:>6.2}  p50 {:>6.2}  p90 {:>6.2}  p99 {:>6.2}  max {:>6.2}",
        pct(vals, 0.10),
        pct(vals, 0.50),
        pct(vals, 0.90),
        pct(vals, 0.99),
        vals.last().unwrap(),
    );
}

pub(crate) fn run(args: Args) -> Result<()> {
    use net::probe::run_step_profile;

    let (dir, policy) = nn_crab_policy(args.checkpoint)?;
    println!("step-profile: checkpoint={}", dir.display());
    println!(
        "step-profile: seed={:#x} warmup={} ticks={} substeps={} solver={:?}",
        args.seed,
        args.warmup,
        args.ticks,
        crab_world::physics::PHYSICS_SUBSTEPS,
        crab_world::physics::SOLVER_ITERATIONS,
    );

    let profile = run_step_profile(policy, args.seed, args.warmup, args.ticks);
    if profile.steps.is_empty() {
        anyhow::bail!("step-profile: no steps measured");
    }

    if let Some(path) = &args.csv {
        let mut s =
            String::from("tick,wall_ms,substep_ms,solver_ms,collision_ms,vel_res_ms,vel_asm_ms\n");
        for x in &profile.steps {
            s.push_str(&format!(
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                x.tick,
                x.wall_ms,
                x.substep_ms,
                x.solver_ms,
                x.collision_ms,
                x.vel_res_ms,
                x.vel_asm_ms,
            ));
        }
        std::fs::write(path, s)?;
        println!(
            "step-profile: wrote {} per-step samples to {}",
            profile.steps.len(),
            path.display()
        );
    }

    // The ruler discipline (rl#396 stage-4 correction): print the hunt so a settled
    // window is visible as such — these numbers are only claims about the ACTIVE scene.
    println!("\n  tick   crab_x   crab_z   dist_to_prey");
    for s in &profile.activity {
        println!(
            "  {:>5}  {:>8.2} {:>8.2}  {:>6.2}",
            s.tick, s.crab_x_m, s.crab_z_m, s.dist_to_prey_m
        );
    }
    let walked: f64 = profile
        .activity
        .windows(2)
        .map(|w| {
            (((w[1].crab_x_m - w[0].crab_x_m).powi(2) + (w[1].crab_z_m - w[0].crab_z_m).powi(2))
                as f64)
                .sqrt()
        })
        .sum();
    println!("  crab walked {walked:.1} m across the measured window");

    println!(
        "\nper-step cost over {} steps (wall = whole FixedMain pass; counter rows = LAST substep of {}):",
        profile.steps.len(),
        crab_world::physics::PHYSICS_SUBSTEPS,
    );
    let col = |f: fn(&net::probe::StepSample) -> f64| -> Vec<f64> {
        profile.steps.iter().map(f).collect()
    };
    dist("step wall ms", &mut col(|s| s.wall_ms));
    dist("substep ms", &mut col(|s| s.substep_ms));
    dist("  solver ms", &mut col(|s| s.solver_ms));
    dist("    vel-resolution ms", &mut col(|s| s.vel_res_ms));
    dist("    vel-assembly ms", &mut col(|s| s.vel_asm_ms));
    dist("  collision-detect ms", &mut col(|s| s.collision_ms));

    println!("\nper-tick cost (the crossing frame pays finalize once per tick):");
    dist("finalize ms", &mut profile.finalize_ms.clone());
    dist("update-schedule ms", &mut profile.update_ms.clone());

    let mut wall: Vec<f64> = profile.steps.iter().map(|s| s.wall_ms).collect();
    wall.sort_by(f64::total_cmp);
    let wall_p50 = pct(&wall, 0.50);
    let mut sub: Vec<f64> = profile.steps.iter().map(|s| s.substep_ms).collect();
    sub.sort_by(f64::total_cmp);
    let physics = crab_world::physics::PHYSICS_SUBSTEPS as f64 * pct(&sub, 0.50);
    println!(
        "\nstep p50 split: wall {:.2} ms ≈ physics {:.2} + NN forward {:.3} + other {:.2} \
         (sense/act/schedule)\nheadless numbers are a FLOOR for the windowed host's \
         per-step cost (no render contention, lighter FixedMain, broadcast tail \
         unmeasured) — a floor that busts the budget is conclusive, one that fits is not",
        wall_p50,
        physics,
        profile.policy_forward_ms,
        (wall_p50 - physics - profile.policy_forward_ms).max(0.0),
    );
    Ok(())
}
