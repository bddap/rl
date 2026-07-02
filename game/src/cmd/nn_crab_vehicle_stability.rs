//! `nn-crab-vehicle-stability`: headless crab-policy-stability gate for the SP-vehicle→rapier
//! migration — ram the trained crab with a real vehicle rigidbody and confirm its walking recovers.

use anyhow::Result;
use clap::Parser;

use super::shared::nn_crab_checkpoint_dir;

#[derive(Parser)]
pub(crate) struct Args {
    /// Trained crab checkpoint dir (`brain.bin` + `normalizer.bin`). Same resolution as
    /// `nn-crab-probe`: this flag, else `RL_CRAB_CHECKPOINT_DIR`, else `assets/weights`.
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<std::path::PathBuf>,
    /// Ticks to walk the crab BEFORE the ram (let it settle + start its policy behaviour).
    #[arg(long, default_value_t = 200)]
    warmup: u64,
    /// Ticks to watch the crab AFTER the ram (it absorbs the hit, then must recover).
    #[arg(long, default_value_t = 600)]
    post: u64,
    /// Match seed. Defaults to the round's own [`super::shared::MATCH_SEED`].
    #[arg(long, default_value_t = super::shared::MATCH_SEED)]
    seed: u64,
}

/// Crab-policy-stability gate (see [`crate::cmd::Command::NnCrabVehicleStability`]): run the trained
/// crab, ram it with a real vehicle rigidbody, and confirm the trained walking survives — the
/// migration's DONE bar #2. Three checks, all headless (physics + inference, no GPU): the carapace
/// stays FINITE throughout (no NaN/explosion), it never LAUNCHES out of the arena (bounded height),
/// and it RECOVERS its standing posture + reach after the hit. Exits nonzero on any failure.
pub(crate) fn run(args: Args) -> Result<()> {
    use net::external_crab::run_vehicle_stability_probe;

    let dir = nn_crab_checkpoint_dir(args.checkpoint)?;
    println!("nn-crab-vehicle-stability: checkpoint={}", dir.display());
    println!(
        "nn-crab-vehicle-stability: seed={:#x} warmup={} post={}",
        args.seed, args.warmup, args.post
    );

    let result = run_vehicle_stability_probe(&dir, args.seed, args.warmup, args.post);
    if result.samples.is_empty() {
        anyhow::bail!("nn-crab-vehicle-stability: no samples — the crab never stepped");
    }
    let ram_tick = result.ram_tick;

    // Mean carapace standing height over a window — the "is it standing?" signal, robust to the
    // slow locomotion checkpoint (which barely strides, so distance-closed is a weak signal).
    let mean_y = |from: u64, to: u64| -> Option<f32> {
        let ys: Vec<f32> = result
            .samples
            .iter()
            .filter(|s| s.tick >= from && s.tick < to && s.carapace_y.is_finite())
            .map(|s| s.carapace_y)
            .collect();
        (!ys.is_empty()).then(|| ys.iter().sum::<f32>() / ys.len() as f32)
    };
    // Baseline: the back half of warmup (past the settle transient). Recovery: the back half of
    // post (past the hit + re-stabilise transient).
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

    // A skimmable trace around the ram so the recovery is legible.
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

    // Gate. Finite is the hard floor; bounded height rules out a launch-to-infinity; the standing
    // height recovering to near its pre-ram baseline is "the trained walking came back" (not
    // collapsed flat or flung up); the reach staying finite means the policy still acts.
    let finite = result.carapace_stayed_finite();
    let bounded = max_y < 5.0; // never flew over the 2 m walls and away
    let stood_back_up = y_before.is_finite()
        && y_after.is_finite()
        && y_after > 0.4 * y_before
        && y_after < 1.8 * y_before;
    // The policy still acts: its claw stays near the target (a bounded reach, not just finite — a
    // crab knocked flat or flung off keeps finite claw coords, so finiteness alone is too weak).
    // The arena is ±10 m, so a reach within a few metres means the trained reaching survived.
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
