//! The learner's PPO update over all rollout buffers — the ONE update implementation,
//! generic over the autodiff backend so the live GPU learner ([`super::gpu::GpuLearner`])
//! and the CPU/GPU `bench-update` harness ([`super::bench`]) share it.

use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::algorithm::{
    NormalizedValue, PpoConfig, PpoMetrics, ReturnNormalizer, RolloutBuffer, StepEnd, Transition,
    compute_gae, gaussian_log_prob_rows,
};
use super::checkpoint::CrabOpt;
use crate::bot::actuator::ACTION_SIZE;
use crate::bot::brain::CrabBrain;
use crate::bot::sensor::OBS_SIZE;

/// The learner's PPO update over all K·M rollout buffers.
///
/// `rollouts` is one buffer per env (GAE is computed strictly per env, never
/// across a buffer boundary). The per-env trailing bootstrap — V of each buffer's
/// non-`Terminal` tail observation — is computed HERE from `brain`: the learner
/// holds the brain it snapshotted to the threads (which is what they rolled with),
/// so no precomputed value is needed. Mutating `brain`/`optimizer` in place keeps
/// Adam's moment estimates persistent across updates.
///
/// `ret_norm` is the learner's running return scale (see [`ReturnNormalizer`]): the
/// value head's outputs (stored per-step values and the trailing bootstrap) are
/// de-normalized by it so GAE/advantages stay in real reward units, then this
/// update's real-unit returns are folded in and the value-loss targets normalized by
/// the refreshed scale. It is `&mut` because the update advances it; the learner owns
/// the one copy, so passing it in keeps a single source of truth.
///
/// `rng` drives the per-epoch minibatch shuffle. The caller owns it (seeded from the
/// run's master seed) so the shuffle order is reproducible across a run and varies
/// across iterations as the stream advances.
///
/// Free function rather than a `TrainingState` method so the K=1 parity test
/// ([`super::inproc`] tests) can call the exact production update over hand-built buffers.
/// Generic over the autodiff backend `B` so the live GPU learner and the CPU/GPU
/// `bench-update` run the one implementation — same update, one backend parameter.
pub(crate) fn ppo_update_core<B: AutodiffBackend>(
    brain: &mut CrabBrain<B>,
    optimizer: &mut CrabOpt<B>,
    config: &PpoConfig,
    rollouts: &[RolloutBuffer],
    device: &B::Device,
    ret_norm: &mut ReturnNormalizer,
    rng: &mut StdRng,
) -> PpoMetrics {
    {
        let n: usize = rollouts.iter().map(|b| b.len()).sum();
        if n == 0 {
            return PpoMetrics::default();
        }

        // Return-normalization stats from BEFORE this update (PopArt ordering): GAE
        // de-normalizes the value head's outputs with the scale the head was trained
        // against, computes advantages/returns in REAL reward units, and only after
        // does `ret_norm.update` fold THIS update's returns in. The first update sees
        // the identity (no returns yet), so it is byte-identical to un-normalized PPO.
        let ret_norm_pre = ret_norm.clone();

        // GAE strictly per env: each buffer is one env's contiguous trajectory
        // segment, bootstrapped from ITS last observation. Advantages/returns are
        // then concatenated in the same env-major order as the transitions below.
        let mut advantages = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        for buf in rollouts.iter() {
            let Some(last_t) = buf.transitions.last() else {
                continue;
            };
            // A `Terminal` tail genuinely ended → 0 future return; `compute_gae`
            // hardcodes that step's bootstrap to `RealReturn(0.0)` and never reads
            // `last_value`, so the value passed here is inert (any `NormalizedValue`
            // would do). A non-terminal tail bootstraps V(s_tail) from the brain (the
            // trailing obs continues into the next horizon's buffer, so its value
            // carries the cut-off return); that output is in normalized units, which
            // `compute_gae` de-normalizes like every stored value.
            let last_value = if matches!(last_t.end, StepEnd::Terminal) {
                NormalizedValue(0.0)
            } else {
                let obs =
                    Tensor::<B, 1>::from_floats(last_t.obs.as_slice(), device).unsqueeze::<2>();
                NormalizedValue(
                    brain
                        .value(obs)
                        .flatten::<1>(0, 1)
                        .into_scalar()
                        .elem::<f32>(),
                )
            };
            let (a, r) = compute_gae(buf, last_value, config.gamma, config.lambda, &ret_norm_pre);
            advantages.extend(a);
            returns.extend(r);
        }

        // Fold this update's REAL-unit returns into the running scale, then normalize the
        // value-loss targets by the refreshed scale (the residual is then in σ-units — see
        // the value-loss site below).
        let real_returns: Vec<f32> = returns.iter().map(|r| r.0).collect();
        ret_norm.update(&real_returns);
        let returns: Vec<f32> = returns.iter().map(|&r| ret_norm.normalize(r).0).collect();

        // Env-major transition view matching the advantages/returns order.
        let transitions: Vec<&Transition> =
            rollouts.iter().flat_map(|b| b.transitions.iter()).collect();

        // Batch-normalizing the advantages strips their reward unit (centered and
        // scaled to a unitless gradient signal), so they leave `RealReturn` here.
        let advantages: Vec<f32> = advantages.iter().map(|a| a.0).collect();
        let adv_mean: f32 = advantages.iter().sum::<f32>() / n as f32;
        let adv_var: f32 = advantages
            .iter()
            .map(|a| (a - adv_mean).powi(2))
            .sum::<f32>()
            / n as f32;
        let adv_std = adv_var.sqrt().max(1e-8);
        let advantages_norm: Vec<f32> = advantages
            .iter()
            .map(|a| (a - adv_mean) / adv_std)
            .collect();

        let obs_data: Vec<f32> = transitions
            .iter()
            .flat_map(|t| t.obs.iter().copied())
            .collect();
        let actions_data: Vec<f32> = transitions
            .iter()
            .flat_map(|t| t.action.iter().copied())
            .collect();

        let obs_all = Tensor::<B, 2>::from_data(TensorData::new(obs_data, [n, OBS_SIZE]), device);
        let actions_all =
            Tensor::<B, 2>::from_data(TensorData::new(actions_data, [n, ACTION_SIZE]), device);

        // π_old: the behavior policy's log-prob of each stored action, recomputed HERE on
        // the UPDATE backend from the pre-update brain — NOT the `t.log_prob` the rollout
        // recorded. The rollout runs on the CPU (ndarray) backend and this update on the
        // GPU (wgpu); the two forwards disagree enough that the CPU-recorded log-probs put
        // the importance ratio far from 1 at update start (~0.7 KL measured on a frozen
        // brain) — a corrupt PPO ratio and a meaningless trust-region signal. π_old is by
        // definition the policy at update start, so a backend-consistent recompute is the
        // correct old log-prob: the ratio starts at exactly 1 and the target-KL guard then
        // measures true on-backend policy drift. Detached — π_old is a fixed reference, no
        // gradient flows back through it.
        let old_log_probs_all = {
            let (means, log_std) = brain.policy(obs_all.clone());
            gaussian_log_prob_rows(means, log_std, actions_all.clone()).detach()
        };

        // Diagnostic only: how far the rollout's CPU-recorded behavior log-prob sits from
        // the on-backend recompute above. The update IGNORES `t.log_prob` (it uses
        // `old_log_probs_all`); this just monitors the backend gap that recompute exists
        // to close, so a regression that re-opens it is visible on the learner line.
        let behavior_backend_div = {
            let cpu_old: Vec<f32> = transitions.iter().map(|t| t.log_prob).collect();
            let cpu_old = Tensor::<B, 1>::from_data(TensorData::new(cpu_old, [n]), device);
            (old_log_probs_all.clone() - cpu_old)
                .abs()
                .mean()
                .into_scalar()
                .elem::<f32>()
        };
        let advantages_all =
            Tensor::<B, 1>::from_data(TensorData::new(advantages_norm, [n]), device);
        let returns_all = Tensor::<B, 1>::from_data(TensorData::new(returns, [n]), device);

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut update_count = 0u32;
        let mut last_kl = 0.0f32;

        let bs = config.batch_size;

        // Target-KL trust region: the update stops the instant the policy has drifted
        // `1.5 × target_kl` from the behavior policy `π_old` above. The ratio clip only zeroes
        // out-of-band sample gradients — it does not bound total KL across the epochs ×
        // minibatches, so a sharpened policy or a cold-resumed Adam can still step off a
        // cliff in one iteration. This bounds each iteration's movement to ~`target_kl`
        // regardless (see `PpoConfig::target_kl`). Labeled so an over-KL minibatch
        // breaks BOTH loops, not just the inner one.
        'update: for _epoch in 0..config.epochs_per_update {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(rng);

            let num_batches = n.div_ceil(bs);

            for batch_idx in 0..num_batches {
                let start = batch_idx * bs;
                let end = (start + bs).min(n);
                let batch_n = end - start;
                let batch_indices = &indices[start..end];

                let idx_tensor = Tensor::<B, 1, Int>::from_data(
                    TensorData::new(
                        batch_indices.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                        [batch_n],
                    ),
                    device,
                );

                let obs = obs_all.clone().select(0, idx_tensor.clone());
                let actions = actions_all.clone().select(0, idx_tensor.clone());
                let old_lp = old_log_probs_all.clone().select(0, idx_tensor.clone());
                let advs = advantages_all.clone().select(0, idx_tensor.clone());
                let rets = returns_all.clone().select(0, idx_tensor);

                let (means, log_std) = brain.policy(obs.clone());

                // π_new for this minibatch — same Gaussian log-prob as π_old above (one
                // formula, `gaussian_log_prob_rows`), so the ratio can't drift on a
                // formula mismatch. log_std is pre-clamped by policy (single source).
                let new_lp = gaussian_log_prob_rows(means, log_std.clone(), actions);

                let log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0);
                let ratio = log_ratio.clone().exp();

                // Schulman's unbiased, non-negative KL estimate `mean((r-1) - ln r)`,
                // measured on this minibatch's forward — which already reflects every
                // step applied so far this iteration, so it is the cumulative
                // π_old→now drift (and starts at ~0, since π_old was recomputed on this
                // backend above). Crossing the ceiling ends the update before the step
                // that would walk the policy off the cliff. Checked BEFORE this minibatch
                // is applied, so a breaking minibatch contributes no step.
                let approx_kl =
                    ((ratio.clone() - 1.0) - log_ratio).mean().into_scalar().elem::<f32>();
                last_kl = approx_kl;
                if approx_kl > 1.5 * config.target_kl {
                    break 'update;
                }

                let entropy_per_dim = log_std.clone()
                    + (0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln());
                let entropy = entropy_per_dim.mean();

                let surr1 = ratio.clone() * advs.clone();
                let surr2 =
                    ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advs;
                let policy_loss = surr1.min_pair(surr2).mean().neg();

                // The value head's raw output is in NORMALIZED units, and `rets` was
                // normalized by the same running scale above, so this residual is in
                // σ-units and `value_loss_clip` is a σ-count. The head therefore fits
                // unit-scale targets regardless of the reward magnitude — the whole
                // point of return normalization.
                let values: Tensor<B, 1> = brain.value(obs).flatten(0, 1);
                let value_diff =
                    (values - rets).clamp(-config.value_loss_clip, config.value_loss_clip);
                let value_loss = value_diff.powf_scalar(2.0).mean();

                let loss = policy_loss.clone() + value_loss.clone() * config.value_coeff
                    - entropy.clone() * config.entropy_coeff;

                total_policy_loss += policy_loss.clone().into_scalar().elem::<f32>();
                total_value_loss += value_loss.clone().into_scalar().elem::<f32>();
                total_entropy += entropy.clone().into_scalar().elem::<f32>();
                update_count += 1;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, brain);
                *brain = optimizer.step(config.learning_rate, brain.clone(), grads);
            }
        }

        // `update_count == 0` only if the very FIRST minibatch already exceeded the
        // ceiling (the policy was handed in already past the trust region) — leave the
        // brain untouched and report the drift; `max(1)` guards the average against /0.
        let denom = update_count.max(1) as f32;
        PpoMetrics {
            policy_loss: total_policy_loss / denom,
            value_loss: total_value_loss / denom,
            entropy: total_entropy / denom,
            kl: last_kl,
            steps: update_count,
            behavior_backend_div,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::TrainBackend;
    use crate::training::checkpoint::crab_optimizer;
    use burn::backend::ndarray::NdArrayDevice;
    use rand::{Rng, SeedableRng};

    /// Synthetic rollouts (one buffer per env, `horizon` non-terminal transitions) from a
    /// fixed-seed RNG, so the update's INPUT is identical run-to-run and only the shuffle
    /// seed varies. More than one minibatch per epoch (256 transitions, batch 64) so the
    /// shuffle order actually changes the sequence of Adam steps.
    fn make_rollouts() -> Vec<RolloutBuffer> {
        let mut rng = StdRng::seed_from_u64(0xDA7A);
        (0..4)
            .map(|_| {
                let mut buf = RolloutBuffer::new();
                for _ in 0..64 {
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
                        end: StepEnd::Continues,
                    });
                }
                buf
            })
            .collect()
    }

    /// The minibatch shuffle is seeded: two updates with the same shuffle seed (over the
    /// SAME initial brain and the SAME data) produce a bit-identical result, and a
    /// different seed reorders the minibatches enough to change it. Self-contained — a CPU
    /// update over hand-built buffers, no learner thread or GPU. The brain is built once and
    /// CLONED into each run so the comparison can't be perturbed by the process-global
    /// weight-init RNG a parallel test may touch.
    #[test]
    fn ppo_update_is_deterministic_under_equal_shuffle_seed() {
        let device = NdArrayDevice::Cpu;
        let brain = CrabBrain::<TrainBackend>::new(&device);
        let rollouts = make_rollouts();

        let run = |shuffle_seed: u64| -> PpoMetrics {
            let mut brain = brain.clone();
            let mut optimizer = crab_optimizer::<TrainBackend>();
            // Disable the target-KL guard here: this test proves the minibatch SHUFFLE
            // is seeded, which needs the full epochs × minibatches to actually run (a
            // guard early-stop would mask a shuffle that secretly didn't reorder). The
            // guard itself is covered by `target_kl_guard_stops_an_over_kl_update`.
            let config = PpoConfig {
                target_kl: f32::INFINITY,
                ..PpoConfig::default()
            };
            let mut ret_norm = ReturnNormalizer::new();
            let mut rng = StdRng::seed_from_u64(shuffle_seed);
            ppo_update_core(
                &mut brain,
                &mut optimizer,
                &config,
                &rollouts,
                &device,
                &mut ret_norm,
                &mut rng,
            )
        };

        let a = run(0x5417);
        let b = run(0x5417);
        assert_eq!(
            a.policy_loss.to_bits(),
            b.policy_loss.to_bits(),
            "equal shuffle seed must give an identical policy loss"
        );
        assert_eq!(a.value_loss.to_bits(), b.value_loss.to_bits(), "value loss");
        assert_eq!(a.entropy.to_bits(), b.entropy.to_bits(), "entropy");

        // A different shuffle seed reorders the minibatches, so the sequential Adam steps
        // and the averaged losses differ — proving the shuffle is actually seeded, not
        // accidentally constant.
        let c = run(0x9999);
        let differs = a.policy_loss.to_bits() != c.policy_loss.to_bits()
            || a.value_loss.to_bits() != c.value_loss.to_bits()
            || a.entropy.to_bits() != c.entropy.to_bits();
        assert!(
            differs,
            "a different shuffle seed must change the update result"
        );
    }

    /// The target-KL trust region must STOP an update once the policy has drifted past
    /// the ceiling, and run every minibatch when the ceiling is infinite. Because
    /// `π_old` is recomputed on the update backend, the drift starts at ~0 and GROWS
    /// with each applied step — so the FIRST (zero-drift) minibatch always passes, and
    /// a tight ceiling then stops as soon as the first step's movement exceeds it. This
    /// is the guard that bounds each iteration's policy movement.
    #[test]
    fn target_kl_guard_stops_an_over_kl_update() {
        let device = NdArrayDevice::Cpu;
        let brain = CrabBrain::<TrainBackend>::new(&device);
        let rollouts = make_rollouts();

        let run = |target_kl: f32| -> PpoMetrics {
            let mut brain = brain.clone();
            let mut optimizer = crab_optimizer::<TrainBackend>();
            let config = PpoConfig {
                target_kl,
                ..PpoConfig::default()
            };
            let mut ret_norm = ReturnNormalizer::new();
            let mut rng = StdRng::seed_from_u64(0x5417);
            ppo_update_core(
                &mut brain, &mut optimizer, &config, &rollouts, &device, &mut ret_norm, &mut rng,
            )
        };

        // n=256, batch 64, 4 epochs ⇒ 16 minibatches when nothing early-stops.
        let unguarded = run(f32::INFINITY);
        assert_eq!(unguarded.steps, 16, "an infinite ceiling runs every minibatch");

        // A tiny ceiling: the first (zero-drift) minibatch is applied, then the very
        // next minibatch sees that step's drift exceed the ceiling and stops — so the
        // update moves the brain by at most a step or two, never the full 16.
        let guarded = run(1e-6);
        assert!(
            guarded.steps >= 1 && guarded.steps < unguarded.steps,
            "a tight ceiling early-stops after the first drift (1..16 steps), got {}",
            guarded.steps
        );
        assert!(
            guarded.kl > 1.5 * 1e-6,
            "the reported KL is the over-threshold drift that triggered the stop, got {}",
            guarded.kl
        );
    }
}
