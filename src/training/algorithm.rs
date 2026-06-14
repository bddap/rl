//! PPO (Proximal Policy Optimization) support functions.

use burn::prelude::*;

use crate::bot::brain::ACTION_SIZE;
use crate::bot::sensor::OBS_SIZE;

/// PPO hyperparameters.
pub struct PpoConfig {
    pub gamma: f32,
    pub lambda: f32,
    pub clip_epsilon: f32,
    pub entropy_coeff: f32,
    pub value_coeff: f32,
    pub learning_rate: f64,
    pub epochs_per_update: u32,
    pub batch_size: usize,
    pub value_loss_clip: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coeff: 0.01, // Per-dim mean entropy (not sum), so this is scale-invariant
            value_coeff: 0.5,
            learning_rate: 3e-4,
            epochs_per_update: 4,
            batch_size: 64,
            value_loss_clip: 10.0,
        }
    }
}

/// A single transition.
#[derive(Clone)]
pub struct Transition {
    pub obs: [f32; OBS_SIZE],
    pub action: [f32; ACTION_SIZE],
    pub reward: f32,
    pub value: f32,
    pub log_prob: f32,
    /// True terminal: the trajectory genuinely ended here (the crab failed a
    /// survival guard / the sim died), so the future return is 0.
    pub done: bool,
    /// Truncation: the episode was cut by the step cap, not ended — the crab was
    /// still standing. The value must be bootstrapped (see [`compute_gae`]), or
    /// the policy is taught that surviving to the cap is worth nothing. Never set
    /// together with `done`: the one site that builds a real transition computes
    /// `truncated = !done && over_cap`. (A rollout-window boundary mid-episode is
    /// neither: the episode continues into the next buffer, bootstrapped via
    /// `last_value`.)
    pub truncated: bool,
}

/// Rollout buffer.
pub struct RolloutBuffer {
    pub transitions: Vec<Transition>,
}

impl RolloutBuffer {
    pub fn new() -> Self {
        Self {
            transitions: Vec::with_capacity(2048),
        }
    }

    pub fn push(&mut self, t: Transition) {
        self.transitions.push(t);
    }

    pub fn clear(&mut self) {
        self.transitions.clear();
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }
}

/// Compute Generalized Advantage Estimation.
pub fn compute_gae(
    buffer: &RolloutBuffer,
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = buffer.len();
    let mut advantages = vec![0.0f32; n];
    let mut returns = vec![0.0f32; n];
    let mut last_gae = 0.0f32;
    let mut next_value = last_value;

    for i in (0..n).rev() {
        let t = &buffer.transitions[i];
        // Bootstrap target V(s_{i+1}):
        //  - done (true terminal): 0 — the episode genuinely ended.
        //  - truncated (cut by the step cap, env then reset): the real next
        //    state was discarded at reset, and the next buffer entry belongs to
        //    a *different* episode. Bootstrap from V(s_i) (this step's own value
        //    ≈ V of the cut continuation for a slowly-changing pose) so the cap
        //    isn't taught as a dead end.
        //  - otherwise: the next entry's value (an in-episode step, or a
        //    rollout-boundary cut bootstrapped via `last_value`).
        let bootstrap = if t.done {
            0.0
        } else if t.truncated {
            t.value
        } else {
            next_value
        };
        let delta = t.reward + gamma * bootstrap - t.value;
        // The GAE trace cannot cross an episode boundary: a done or a truncation
        // ends this trajectory segment, so the future (next-episode) advantage
        // must not fold back across it.
        last_gae = if t.done || t.truncated {
            delta
        } else {
            delta + gamma * lambda * last_gae
        };
        advantages[i] = last_gae;
        returns[i] = last_gae + t.value;
        next_value = t.value;
    }

    (advantages, returns)
}

/// Compute log probability of actions under Gaussian policy.
/// Uses log-space computation to avoid division by tiny variance.
pub fn compute_log_prob<B: Backend>(
    means: &Tensor<B, 1>,
    log_std: &Tensor<B, 1>,
    actions: &Tensor<B, 1>,
) -> f32 {
    // log_std arrives pre-clamped from CrabBrain::policy (single source of truth).
    let diff = actions.clone() - means.clone();
    // log p = -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
    let scaled_diff = diff / log_std.clone().exp();
    let log_probs = scaled_diff.powf_scalar(2.0).neg() * 0.5
        - log_std.clone()
        - 0.5 * (2.0 * std::f32::consts::PI).ln();
    log_probs.sum().into_scalar().elem::<f32>()
}

/// Sample actions from Gaussian policy.
pub fn sample_action<B: Backend>(
    means: &Tensor<B, 1>,
    log_std: &Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let std = log_std.clone().exp();
    let noise = Tensor::<B, 1>::random(
        [ACTION_SIZE],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let action = means.clone() + noise * std;
    action.clamp(-1.0, 1.0)
}

/// Metrics from a PPO update.
#[derive(Debug, Default, Clone)]
pub struct PpoMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(reward: f32, value: f32, done: bool) -> Transition {
        Transition {
            obs: [0.0; OBS_SIZE],
            action: [0.0; ACTION_SIZE],
            reward,
            value,
            log_prob: 0.0,
            done,
            truncated: false,
        }
    }

    /// GAE must be computed per env. Sweeping one concatenated buffer lets env
    /// A's last step bootstrap from env B's first value — silent advantage
    /// corruption that still "trains". This pins the per-env results and proves
    /// the concatenated sweep actually diverges (i.e. the split is load-bearing).
    #[test]
    fn gae_per_env_differs_from_concatenated_sweep() {
        let gamma = 0.5;
        let lambda = 0.5;

        let mut env_a = RolloutBuffer::new();
        env_a.push(t(1.0, 0.5, false));
        env_a.push(t(1.0, 0.5, false));
        let mut env_b = RolloutBuffer::new();
        env_b.push(t(0.0, 1.0, false));
        env_b.push(t(0.0, 1.0, true));

        // Per-env, hand-computed: A bootstraps from ITS next value (2.0).
        let (adv_a, _) = compute_gae(&env_a, 2.0, gamma, lambda);
        assert!((adv_a[1] - 1.5).abs() < 1e-6, "A[1]: {}", adv_a[1]);
        assert!((adv_a[0] - 1.125).abs() < 1e-6, "A[0]: {}", adv_a[0]);

        // Naive concatenated sweep: A's last step bootstraps from B's value.
        let mut concat = RolloutBuffer::new();
        for tr in env_a.transitions.iter().chain(env_b.transitions.iter()) {
            concat.push(tr.clone());
        }
        let (adv_concat, _) = compute_gae(&concat, 0.0, gamma, lambda);
        assert!(
            (adv_concat[1] - adv_a[1]).abs() > 1e-3,
            "concatenated sweep should corrupt A's advantages (got {} vs {})",
            adv_concat[1],
            adv_a[1]
        );
    }

    /// A truncated final step (episode cut by the step cap while still standing)
    /// must bootstrap its value, unlike a true terminal which has zero future
    /// return. Conflating the two — the original bug — taught the policy that
    /// surviving to the cap is worthless. Same reward and value, opposite
    /// bootstrap: the advantages must differ by exactly `gamma * value`.
    #[test]
    fn truncation_bootstraps_unlike_true_terminal() {
        let gamma = 0.99;
        let lambda = 0.95;
        let (reward, value) = (1.0, 5.0);

        let mut terminal = RolloutBuffer::new();
        terminal.push(t(reward, value, true));
        let (adv_term, _) = compute_gae(&terminal, 0.0, gamma, lambda);

        let mut truncated = RolloutBuffer::new();
        truncated.push(Transition {
            done: false,
            truncated: true,
            ..t(reward, value, false)
        });
        let (adv_trunc, ret_trunc) = compute_gae(&truncated, 0.0, gamma, lambda);

        // Terminal: advantage = reward - value (no bootstrap).
        assert!(
            (adv_term[0] - (reward - value)).abs() < 1e-6,
            "term: {}",
            adv_term[0]
        );
        // Truncated: advantage = reward + gamma*value - value (bootstrap own value).
        assert!(
            (adv_trunc[0] - (reward + gamma * value - value)).abs() < 1e-6,
            "trunc: {}",
            adv_trunc[0]
        );
        // The whole point: the cap is not a dead end.
        assert!(
            (adv_trunc[0] - adv_term[0] - gamma * value).abs() < 1e-6,
            "truncation must bootstrap gamma*value more than a true terminal"
        );
        // Return = bootstrapped one-step target.
        assert!(
            (ret_trunc[0] - (reward + gamma * value)).abs() < 1e-6,
            "ret: {}",
            ret_trunc[0]
        );
    }
}
