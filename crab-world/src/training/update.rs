use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use tracing::error;

use super::algorithm::{
    PpoConfig, PpoMetrics, ReturnNormalizer, RolloutBuffer, Transition, compute_gae,
};
use super::checkpoint::CrabOpt;
use crate::bot::actuator::ACTION_SIZE;
use crate::bot::arch::{AnyBrain, GaussianHead};
use crate::bot::sensor::OBS_SIZE;

fn huber_value_loss<B: AutodiffBackend>(residual: Tensor<B, 1>, delta: f32) -> Tensor<B, 1> {
    let quad = residual.clone().clamp(-delta, delta).powf_scalar(2.0);
    let lin = residual
        .abs()
        .sub_scalar(delta)
        .clamp_min(0.0)
        .mul_scalar(2.0 * delta);
    quad + lin
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ppo_update_core<B: AutodiffBackend>(
    brain: &mut AnyBrain<B>,
    optimizer: &mut CrabOpt<B>,
    config: &PpoConfig,
    rollouts: &[RolloutBuffer],
    device: &B::Device,
    ret_norm: &mut ReturnNormalizer,
    rng: &mut StdRng,
    log_std_floor: f32,
) -> PpoMetrics {
    {
        let n: usize = rollouts.iter().map(|b| b.len()).sum();
        if n == 0 {
            return PpoMetrics::default();
        }

        let ret_norm_pre = ret_norm.clone();

        let mut advantages = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        for buf in rollouts.iter() {
            if buf.transitions.is_empty() {
                continue;
            }
            let (a, r) = compute_gae(buf, config.gamma, config.lambda, &ret_norm_pre);
            advantages.extend(a);
            returns.extend(r);
        }

        let real_returns: Vec<f32> = returns.iter().map(|r| r.0).collect();
        let nonfinite_returns = ret_norm.update(&real_returns);
        if nonfinite_returns > 0 {
            error!(
                "ppo_update: {nonfinite_returns}/{} returns were non-finite and skipped from the \
                 return scale — an env diverged this iteration",
                real_returns.len()
            );
        }
        let returns: Vec<f32> = returns.iter().map(|&r| ret_norm.normalize(r).0).collect();

        let transitions: Vec<&Transition> =
            rollouts.iter().flat_map(|b| b.transitions.iter()).collect();

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

        let old_log_probs_all = GaussianHead::new(brain.policy(obs_all.clone()), log_std_floor)
            .log_prob_rows(actions_all.clone())
            .detach();

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

        'update: for epoch in 0..config.epochs_per_update {
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

                let head = GaussianHead::new(brain.policy(obs.clone()), log_std_floor);

                let new_lp = head.log_prob_rows(actions);

                let log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0);
                let ratio = log_ratio.clone().exp();

                let approx_kl = ((ratio.clone() - 1.0) - log_ratio)
                    .mean()
                    .into_scalar()
                    .elem::<f32>();
                last_kl = approx_kl;
                if !approx_kl.is_finite() {
                    error!(
                        "ppo_update: non-finite KL ({approx_kl}) at epoch {epoch} batch \
                         {batch_idx}; aborting the update to protect the policy (no further step)"
                    );
                    break 'update;
                }
                if approx_kl > 1.5 * config.target_kl {
                    break 'update;
                }

                let entropy = head.entropy();

                let surr1 = ratio.clone() * advs.clone();
                let surr2 =
                    ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advs;
                let policy_loss = surr1.min_pair(surr2).mean().neg();

                let values: Tensor<B, 1> = brain.value(obs).flatten(0, 1);
                let value_loss = huber_value_loss(values - rets, config.value_loss_clip).mean();

                let loss = policy_loss.clone() + value_loss.clone() * config.value_coeff
                    - entropy.clone() * config.entropy_coeff;

                let policy_loss_v = policy_loss.clone().into_scalar().elem::<f32>();
                let value_loss_v = value_loss.clone().into_scalar().elem::<f32>();
                let entropy_v = entropy.clone().into_scalar().elem::<f32>();
                if !(policy_loss_v.is_finite() && value_loss_v.is_finite() && entropy_v.is_finite())
                {
                    error!(
                        "ppo_update: non-finite loss (policy={policy_loss_v}, value={value_loss_v}, \
                         entropy={entropy_v}) at epoch {epoch} batch {batch_idx}; refusing the Adam \
                         step and aborting the update to protect the policy"
                    );
                    break 'update;
                }
                total_policy_loss += policy_loss_v;
                total_value_loss += value_loss_v;
                total_entropy += entropy_v;
                update_count += 1;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, brain);
                *brain = optimizer.step(config.learning_rate, brain.clone(), grads);
            }
        }

        let denom = update_count.max(1) as f32;
        PpoMetrics {
            policy_loss: total_policy_loss / denom,
            value_loss: total_value_loss / denom,
            entropy: total_entropy / denom,
            kl: last_kl,
            steps: update_count,
            behavior_backend_div,
            nonfinite_returns: nonfinite_returns as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::algorithm::{NormalizedValue, StepEnd};
    use super::*;
    use crate::bot::arch::ArchId;
    use crate::training::TrainBackend;
    use crate::training::checkpoint::crab_optimizer;
    use burn::backend::ndarray::NdArrayDevice;
    use rand::{Rng, SeedableRng};

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

    #[test]
    fn ppo_update_is_deterministic_under_equal_shuffle_seed() {
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let rollouts = make_rollouts();

        let run = |shuffle_seed: u64| -> PpoMetrics {
            let mut brain = brain.clone();
            let mut optimizer = crab_optimizer::<TrainBackend>();
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
                crate::bot::arch::LOG_STD_MIN,
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

        let c = run(0x9999);
        let differs = a.policy_loss.to_bits() != c.policy_loss.to_bits()
            || a.value_loss.to_bits() != c.value_loss.to_bits()
            || a.entropy.to_bits() != c.entropy.to_bits();
        assert!(
            differs,
            "a different shuffle seed must change the update result"
        );
    }

    #[test]
    fn non_finite_loss_aborts_the_update_without_corrupting_the_policy() {
        let device = NdArrayDevice::Cpu;
        let mut brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);

        let probe = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(vec![0.1f32; OBS_SIZE], [1, OBS_SIZE]),
            &device,
        );
        let before: Vec<f32> = brain
            .policy(probe.clone())
            .0
            .flatten::<1>(0, 1)
            .to_data()
            .to_vec()
            .unwrap();

        let mut rollouts = make_rollouts();
        rollouts[0].transitions[0].reward = f32::NAN;

        let mut optimizer = crab_optimizer::<TrainBackend>();
        let config = PpoConfig {
            target_kl: f32::INFINITY,
            ..PpoConfig::default()
        };
        let mut ret_norm = ReturnNormalizer::new();
        let mut rng = StdRng::seed_from_u64(0x1267);
        let metrics = ppo_update_core(
            &mut brain,
            &mut optimizer,
            &config,
            &rollouts,
            &device,
            &mut ret_norm,
            &mut rng,
            crate::bot::arch::LOG_STD_MIN,
        );

        assert_eq!(
            metrics.steps, 0,
            "a non-finite loss must abort before any Adam step (got {} steps)",
            metrics.steps
        );
        let after: Vec<f32> = brain
            .policy(probe)
            .0
            .flatten::<1>(0, 1)
            .to_data()
            .to_vec()
            .unwrap();
        assert_eq!(
            before.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            after.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            "the policy must be bit-identical after a refused NaN update — the brain was stepped"
        );
    }

    #[test]
    fn target_kl_guard_stops_an_over_kl_update() {
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
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
                &mut brain,
                &mut optimizer,
                &config,
                &rollouts,
                &device,
                &mut ret_norm,
                &mut rng,
                crate::bot::arch::LOG_STD_MIN,
            )
        };

        let unguarded = run(f32::INFINITY);
        assert_eq!(
            unguarded.steps, 16,
            "an infinite ceiling runs every minibatch"
        );

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

    #[test]
    fn huber_value_loss_square_core_linear_tail_with_gradient() {
        let device = NdArrayDevice::Cpu;
        let delta = 3.0f32;
        let rs = [0.5f32, -2.0, 5.0, -8.0];
        let residual =
            Tensor::<TrainBackend, 1>::from_data(TensorData::new(rs.to_vec(), [4]), &device)
                .require_grad();

        let per_sample = huber_value_loss(residual.clone(), delta);
        let vals: Vec<f32> = per_sample.clone().to_data().to_vec().unwrap();
        let want_val = |r: f32| {
            if r.abs() <= delta {
                r * r
            } else {
                2.0 * delta * r.abs() - delta * delta
            }
        };
        for (v, r) in vals.iter().zip(rs) {
            assert!(
                (v - want_val(r)).abs() < 1e-3,
                "huber({r}) = {v}, want {}",
                want_val(r)
            );
        }

        let grads = per_sample.mean().backward();
        let g: Vec<f32> = residual.grad(&grads).unwrap().to_data().to_vec().unwrap();
        let n = rs.len() as f32;
        let want_g = |r: f32| {
            if r.abs() <= delta {
                2.0 * r / n
            } else {
                2.0 * delta * r.signum() / n
            }
        };
        for (gi, r) in g.iter().zip(rs) {
            assert!(
                (gi - want_g(r)).abs() < 1e-3,
                "grad huber({r}) = {gi}, want {}",
                want_g(r)
            );
            if r.abs() > delta {
                assert!(
                    gi.abs() > 1e-3,
                    "outlier r={r} must keep a nonzero gradient, got {gi}"
                );
            }
        }
    }
}
