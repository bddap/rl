use anyhow::Result;
use clap::Parser;

use super::shared::nn_crab_policy;

/// rl#332: long headless soak hunting Sally's "flight" — sustained altitude /
/// vertical velocity beyond legitimate locomotion. Dumps a rolling state window
/// around each onset as JSONL evidence and reports whole-run extrema either way
/// (an empty run is a CENSORED negative at the printed tick budget).
#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "DIR", env = super::shared::CHECKPOINT_ENV)]
    checkpoint: Option<std::path::PathBuf>,
    /// Sim ticks to soak (64 ticks ≈ 1 s of sim time at the probe's 1:1 cadence).
    #[arg(long, default_value_t = 500_000)]
    ticks: u64,
    #[arg(long, default_value_t = super::shared::MATCH_SEED)]
    seed: u64,
    /// Evidence output dir (JSONL window per detected event).
    #[arg(long, value_name = "DIR")]
    out: std::path::PathBuf,
    #[arg(long, default_value_t = 50_000)]
    progress_every: u64,
    /// rl#332 ablation: force every actuator drive to zero from this tick on. A
    /// passive body that keeps accelerating indicts the solver; one that tumbles
    /// to rest indicts actuator-sourced energy.
    #[arg(long)]
    zero_drive_after: Option<u64>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let (dir, policy) = nn_crab_policy(args.checkpoint)?;
    println!("sally-soak: checkpoint={}", dir.display());
    println!(
        "sally-soak: seed={:#x} ticks={} (~{:.1} sim-min) out={}",
        args.seed,
        args.ticks,
        args.ticks as f64 / 64.0 / 60.0,
        args.out.display()
    );

    let report = net::probe::run_flight_soak(
        policy,
        args.seed,
        args.ticks,
        &args.out,
        args.progress_every,
        args.zero_drive_after,
    )?;

    println!(
        "\nsally-soak: {} ticks (~{:.1} sim-min) — {} events, {} teleports, \
         max_above_ground={:.3} m, max|vy|={:.3} m/s, max_up_vy={:.3} m/s, \
         airborne stretches ≥0.25s: {} (longest {:.2} s)",
        report.ticks_run,
        report.ticks_run as f64 / 64.0 / 60.0,
        report.events.len(),
        report.teleports,
        report.max_above_ground,
        report.max_abs_vy,
        report.max_up_vy,
        report.airborne_stretches,
        report.longest_airborne_ticks as f64 / 64.0
    );
    for (i, e) in report.events.iter().enumerate() {
        println!(
            "  event {i}: tick {} kind={} peak_above={:.2} m peak_vy={:.2} m/s — {}",
            e.onset_tick,
            e.kind,
            e.peak_above_ground,
            e.peak_vy,
            e.evidence_path.display()
        );
    }
    if report.events.is_empty() {
        println!(
            "sally-soak: NO flight events — a censored negative at this budget, not proof of absence"
        );
    }
    Ok(())
}
