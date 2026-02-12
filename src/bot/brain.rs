//! Neural network that maps observations to actions.
//!
//! Architecture for v1: simple MLP with shared trunk and separate
//! policy (actor) and value (critic) heads.
//!
//! We'll upgrade to a transformer once the pipeline is proven.

use burn::module::Param;
use burn::nn;
use burn::prelude::*;

use super::body::CrabJointId;
use super::sensor::OBS_SIZE;

pub const ACTION_SIZE: usize = CrabJointId::COUNT;
const HIDDEN_SIZE: usize = 256;

/// Actor-Critic network for PPO.
#[derive(Module, Debug)]
pub struct CrabBrain<B: Backend> {
    // Shared trunk
    trunk_fc1: nn::Linear<B>,
    trunk_fc2: nn::Linear<B>,
    trunk_ln1: nn::LayerNorm<B>,
    trunk_ln2: nn::LayerNorm<B>,

    // Policy head (actor): outputs action means
    policy_fc: nn::Linear<B>,

    // Value head (critic): outputs scalar value estimate
    value_fc1: nn::Linear<B>,
    value_fc2: nn::Linear<B>,

    // Log standard deviation for the policy (learnable, not state-dependent)
    log_std: Param<Tensor<B, 1>>,
}

impl<B: Backend> CrabBrain<B> {
    pub fn new(device: &B::Device) -> Self {
        let trunk_fc1 = nn::LinearConfig::new(OBS_SIZE, HIDDEN_SIZE)
            .with_bias(true)
            .init(device);
        let trunk_fc2 = nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
            .with_bias(true)
            .init(device);
        let trunk_ln1 = nn::LayerNormConfig::new(HIDDEN_SIZE).init(device);
        let trunk_ln2 = nn::LayerNormConfig::new(HIDDEN_SIZE).init(device);

        let policy_fc = nn::LinearConfig::new(HIDDEN_SIZE, ACTION_SIZE)
            .with_bias(true)
            .init(device);

        let value_fc1 = nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE / 2)
            .with_bias(true)
            .init(device);
        let value_fc2 = nn::LinearConfig::new(HIDDEN_SIZE / 2, 1)
            .with_bias(true)
            .init(device);

        // Initialize log_std to -0.5 (std ≈ 0.6, moderate exploration)
        let log_std = Param::from_tensor(Tensor::full([ACTION_SIZE], -0.5, device));

        Self {
            trunk_fc1,
            trunk_fc2,
            trunk_ln1,
            trunk_ln2,
            policy_fc,
            value_fc1,
            value_fc2,
            log_std,
        }
    }

    /// Forward pass through the shared trunk.
    fn trunk(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.trunk_fc1.forward(obs);
        let x = self.trunk_ln1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.trunk_fc2.forward(x);
        let x = self.trunk_ln2.forward(x);
        burn::tensor::activation::relu(x)
    }

    /// Returns (action_means, action_log_std) for the policy.
    /// Input shape: [batch, OBS_SIZE]
    /// Output: means [batch, ACTION_SIZE], log_std [ACTION_SIZE] (clamped to [-2, 0.5])
    pub fn policy(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let trunk = self.trunk(obs);
        let means = self.policy_fc.forward(trunk);
        // Tanh to bound action means to [-1, 1]
        let means = burn::tensor::activation::tanh(means);
        // Clamp log_std to prevent entropy divergence.
        // exp(-2) ≈ 0.14 (focused), exp(0.5) ≈ 1.65 (exploratory).
        let log_std = self.log_std.val().clamp(-2.0, 0.5);
        (means, log_std)
    }

    /// Returns the value estimate.
    /// Input shape: [batch, OBS_SIZE]
    /// Output: [batch, 1]
    pub fn value(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let trunk = self.trunk(obs);
        let x = self.value_fc1.forward(trunk);
        let x = burn::tensor::activation::relu(x);
        self.value_fc2.forward(x)
    }
}
