//! Microbenchmarks of the PPO **update** phase in isolation, on CPU or GPU, over the
//! production [`super::update::ppo_update_core`]. Driven by `rl-train`'s `bench-update`
//! subcommand to compare backends at a fixed K/M/H without the rollout/physics cost.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use burn::tensor::backend::AutodiffBackend;

use super::algorithm::{
    NormalizedValue, PpoConfig, ReturnNormalizer, RolloutBuffer, StepEnd, Transition,
};
use super::checkpoint::{CrabOpt, crab_optimizer};
use super::update::ppo_update_core;
use crate::bot::actuator::ACTION_SIZE;
use crate::bot::brain::CrabBrain;
use crate::bot::sensor::OBS_SIZE;

/// Microbenchmark of the PPO **update** phase in isolation. Builds a fresh
/// `CrabBrain` + Adam optimizer exactly as the real learner does, synthesizes
/// `workers*envs` rollout buffers of `horizon` transitions each (so the per-update
/// transition count and per-env GAE segment structure match a live `learn` iter at
/// the same K/M/H), then calls the production [`ppo_update_core`] `reps` times. The
/// first call is a warmup (page-in / first-alloc) and is excluded; the rest are
/// timed individually and reported as min / median / max plus the per-rep PPO
/// metrics (so NaN/garbage shows up as non-finite loss).
///
/// Generic over the autodiff backend `B` and parameterised by `device`: the caller picks
/// CPU (`TrainBackend` + `NdArrayDevice::Cpu`) or GPU (`Autodiff<Wgpu>` + a `WgpuDevice`),
/// so both exercise this one harness over the production [`ppo_update_core`].
/// `backend_label` is just printed. `batch_override` replaces [`PpoConfig`]'s default
/// minibatch size for the larger-batch sweep (does the GPU stay cheap as the per-step
/// matmul grows?); `None` keeps the live batch of 64.
pub fn bench_ppo_update<B: AutodiffBackend>(
    device: &B::Device,
    backend_label: &str,
    workers: usize,
    envs: usize,
    horizon: usize,
    reps: usize,
    batch_override: Option<usize>,
) {
    let mut brain: CrabBrain<B> = CrabBrain::new(device);
    // The production optimizer (shared `crab_optimizer`, grad-norm clip 0.5), so the
    // step we time is the real one, not a bare Adam.
    let mut optimizer: CrabOpt<B> = crab_optimizer();
    let mut config = PpoConfig::default();
    if let Some(bs) = batch_override {
        config.batch_size = bs;
    }

    // Synthetic rollouts: one buffer per env, `horizon` transitions each, filled with
    // small random obs/actions/rewards. The values matter only for numerical realism
    // (the matmul shapes — driven by OBS_SIZE/HIDDEN/ACTION_SIZE/batch_size — are what
    // we measure); a fixed-seed RNG keeps the two backend builds comparing the same
    // data, and continues into the per-epoch minibatch shuffle below. A non-terminal
    // tail per buffer forces the trailing value bootstrap (one extra forward), matching
    // a truncated live horizon.
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let n_envs = (workers * envs).max(1);
    let mut rollouts: Vec<RolloutBuffer> = Vec::with_capacity(n_envs);
    for _ in 0..n_envs {
        let mut buf = RolloutBuffer::new();
        for _step in 0..horizon {
            let mut obs = [0.0f32; OBS_SIZE];
            for o in obs.iter_mut() {
                *o = rng.gen_range(-1.0..1.0);
            }
            let mut action = [0.0f32; ACTION_SIZE];
            for a in action.iter_mut() {
                *a = rng.gen_range(-1.0..1.0);
            }
            buf.push(Transition {
                obs,
                action,
                reward: rng.gen_range(-1.0..1.0),
                value: NormalizedValue(rng.gen_range(-1.0..1.0)),
                log_prob: rng.gen_range(-5.0..0.0),
                // Every step Continues, so each buffer's tail is non-terminal and
                // `ppo_update_core` bootstraps its trailing value off the brain (one
                // extra forward per buffer) — matching a live horizon cut by the step
                // cap rather than a real episode end.
                end: StepEnd::Continues,
            });
        }
        rollouts.push(buf);
    }

    let n: usize = rollouts.iter().map(|b| b.len()).sum();
    let minibatches_per_epoch = n.div_ceil(config.batch_size);
    let opt_steps = minibatches_per_epoch * config.epochs_per_update as usize;
    eprintln!(
        "[bench-update] backend: {backend_label} | K={workers} × M={envs} × H={horizon} → \
         {n} transitions/update | batch_size {} epochs {} → {minibatches_per_epoch} \
         minibatches/epoch × {} epochs = {opt_steps} optimizer steps/update | OBS {OBS_SIZE} \
         HIDDEN 256 ACTION {ACTION_SIZE} | reps {reps} (1 warmup, {} timed)",
        config.batch_size,
        config.epochs_per_update,
        config.epochs_per_update,
        reps.saturating_sub(1),
    );
    eprintln!(
        "[bench-update] cpu gemm threads: MATMUL_NUM_THREADS={} RAYON_NUM_THREADS={} (no effect on the GPU backend)",
        std::env::var("MATMUL_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
    );

    let mut times_ms: Vec<f64> = Vec::with_capacity(reps);
    for rep in 0..reps {
        // Fresh return normalizer per rep so each rep is the same starting condition
        // (the normalizer's running stats would otherwise drift the value-loss targets
        // rep to rep). The brain/optimizer DO carry over — that mirrors the live loop,
        // where Adam moments persist across updates, and keeps the timing realistic.
        let mut ret_norm = ReturnNormalizer::new();
        let t0 = std::time::Instant::now();
        let metrics = ppo_update_core(
            &mut brain,
            &mut optimizer,
            &config,
            &rollouts,
            device,
            &mut ret_norm,
            &mut rng,
        );
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let finite = metrics.policy_loss.is_finite()
            && metrics.value_loss.is_finite()
            && metrics.entropy.is_finite();
        eprintln!(
            "[bench-update] rep {rep:>2}{}: {dt:8.1} ms | ploss {:+.5} vloss {:+.5} ent {:+.5}{}",
            if rep == 0 { " (warmup)" } else { "        " },
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            if finite { "" } else { "  <<< NON-FINITE!" },
        );
        if rep > 0 {
            times_ms.push(dt);
        }
    }

    if times_ms.is_empty() {
        eprintln!("[bench-update] no timed reps (need reps > 1)");
        return;
    }
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times_ms[times_ms.len() / 2];
    let min = times_ms[0];
    let max = *times_ms.last().unwrap();
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let per_step_ms = median / opt_steps as f64;
    eprintln!(
        "[bench-update] RESULT update: median {median:.1} ms (min {min:.1}, max {max:.1}, mean {mean:.1}) over {} timed reps | {per_step_ms:.3} ms/optimizer-step",
        times_ms.len(),
    );
}

/// GPU bench entry point: bring up the wgpu/Vulkan backend on the discrete GPU (proving
/// the adapter is real hardware, [`super::gpu::init_gpu_backend`]), then run the
/// [`bench_ppo_update`] harness on [`super::gpu::GpuBackend`]. `pub` because the
/// `bench-update --backend gpu` caller lives in the separate `rl-train` crate.
#[cfg(feature = "wgpu")]
pub fn bench_ppo_update_gpu(
    workers: usize,
    envs: usize,
    horizon: usize,
    reps: usize,
    batch_override: Option<usize>,
) {
    use super::gpu::{GpuBackend, init_gpu_backend};

    let device = init_gpu_backend("bench-update");
    bench_ppo_update::<GpuBackend>(
        &device,
        "GPU wgpu/Vulkan (Autodiff<Wgpu>)",
        workers,
        envs,
        horizon,
        reps,
        batch_override,
    );
}
