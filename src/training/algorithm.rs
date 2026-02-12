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
            entropy_coeff: 0.001,  // Reduced: was 0.01, overwhelmed action cost signal
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
    pub done: bool,
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
        let mask = if t.done { 0.0 } else { 1.0 };
        let delta = t.reward + gamma * next_value * mask - t.value;
        last_gae = delta + gamma * lambda * mask * last_gae;
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
    let log_std_clamped = log_std.clone().clamp(-5.0, 2.0);
    let diff = actions.clone() - means.clone();
    // log p = -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
    let scaled_diff = diff / log_std_clamped.clone().exp();
    let log_probs = scaled_diff.powf_scalar(2.0).neg() * 0.5
        - log_std_clamped
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
