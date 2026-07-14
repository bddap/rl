use anyhow::Result;
use clap::Parser;

use super::shared::nn_crab_policy;

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "DIR", env = super::shared::CHECKPOINT_ENV)]
    checkpoint: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 200)]
    warmup: u64,
    #[arg(long, default_value_t = 600)]
    post: u64,
    /// Match seed. Defaults to the round's own [`super::shared::MATCH_SEED`].
    #[arg(long, default_value_t = super::shared::MATCH_SEED)]
    seed: u64,
}

pub(crate) fn run(args: Args) -> Result<()> {
    use net::external_crab::run_vehicle_stability_probe;

    let (dir, policy) = nn_crab_policy(args.checkpoint)?;
    println!("nn-crab-vehicle-stability: checkpoint={}", dir.display());
    println!(
        "nn-crab-vehicle-stability: seed={:#x} warmup={} post={}",
        args.seed, args.warmup, args.post
    );

    let result = run_vehicle_stability_probe(policy, args.seed, args.warmup, args.post);
    if result.samples.is_empty() {
        anyhow::bail!("nn-crab-vehicle-stability: no samples — the crab never stepped");
    }
    let ram_tick = result.ram_tick;

    let mean_y = |from: u64, to: u64| -> Option<f32> {
        let ys: Vec<f32> = result
            .samples
            .iter()
            .filter(|s| s.tick >= from && s.tick < to && s.carapace_y.is_finite())
            .map(|s| s.carapace_y)
            .collect();
        (!ys.is_empty()).then(|| ys.iter().sum::<f32>() / ys.len() as f32)
    };
    let y_before = mean_y(ram_tick / 2, ram_tick).unwrap_or(f32::NAN);
    let recover_from = ram_tick + args.post / 2;
    let y_after = mean_y(recover_from, u64::MAX).unwrap_or(f32::NAN);

    let max_y = result
        .samples
        .iter()
        .map(|s| s.carapace_y.abs())
        .fold(0.0_f32, f32::max);
    let reach_after = result
        .samples
        .iter()
        .filter(|s| s.tick >= recover_from)
        .map(|s| s.min_claw_to_target_m)
        .filter(|d| d.is_finite())
        .fold(f32::INFINITY, f32::min);

    println!("\n  tick   dist   carapace_y   claw→tgt   (ram at tick {ram_tick})");
    for s in result
        .samples
        .iter()
        .filter(|s| s.tick.is_multiple_of(50) || (s.tick >= ram_tick && s.tick < ram_tick + 20))
    {
        let mark = if s.tick == ram_tick { " <<< RAM" } else { "" };
        println!(
            "  {:>5}  {:>5.2}  {:>9.3}  {:>9.3}{}",
            s.tick, s.dist_to_prey_m, s.carapace_y, s.min_claw_to_target_m, mark
        );
    }

    let finite = result.carapace_stayed_finite();
    let bounded = max_y < 5.0; // the ram never launched her skyward
    let stood_back_up = y_before.is_finite()
        && y_after.is_finite()
        && y_after > 0.4 * y_before
        && y_after < 1.8 * y_before;
    let still_reaching = reach_after.is_finite() && reach_after < 6.0;

    println!(
        "\nnn-crab-vehicle-stability: carapace_y before={y_before:.3} → after={y_after:.3} m \
         (max |y|={max_y:.2}); post-ram reach={reach_after:.3} m"
    );
    println!(
        "  finite={finite} bounded={bounded} stood_back_up={stood_back_up} still_reaching={still_reaching}"
    );

    if finite && bounded && stood_back_up && still_reaching {
        println!(
            "nn-crab-vehicle-stability: PASS — the vehicle struck Sally and her trained walking \
             recovered (finite, bounded, stood back up, still reaching)"
        );
        Ok(())
    } else {
        anyhow::bail!(
            "nn-crab-vehicle-stability: FAIL — the crab did not cleanly survive the vehicle \
             collision (finite={finite} bounded={bounded} stood_back_up={stood_back_up} \
             still_reaching={still_reaching})"
        )
    }
}
