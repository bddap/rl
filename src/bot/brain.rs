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

/// Initial policy log-std (std ≈ 0.2): start with low exploration, see `new`.
const LOG_STD_INIT: f32 = -1.6;
/// Bounds the learnable log-std so entropy can't diverge or collapse. Single
/// source of truth: `policy` clamps to this range, so downstream log-prob /
/// entropy never re-clamp. exp(-2) ≈ 0.14 (focused), exp(0.5) ≈ 1.65 (wide).
const LOG_STD_MIN: f32 = -2.0;
const LOG_STD_MAX: f32 = 0.5;

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

        // Small-gain init on the policy head. The trunk ends in a LayerNorm
        // (unit-scale output), so a default-initialised head emits means ~±1 —
        // and since the action IS the joint torque now, an untrained crab would
        // command near-max torque on every joint and launch itself. With tiny
        // weights a fresh policy outputs ~zero torque (a near-limp crab) and
        // learns to push up from there. (Position control hid this: ±1 was just
        // a target angle the servo eased toward, not a torque.)
        let policy_fc = nn::LinearConfig::new(HIDDEN_SIZE, ACTION_SIZE)
            .with_bias(true)
            .with_initializer(nn::Initializer::Normal {
                mean: 0.0,
                std: 0.01,
            })
            .init(device);

        let value_fc1 = nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE / 2)
            .with_bias(true)
            .init(device);
        let value_fc2 = nn::LinearConfig::new(HIDDEN_SIZE / 2, 1)
            .with_bias(true)
            .init(device);

        // Actions are absolute joint-position targets applied every physics
        // step, so per-step noise is per-step jitter. std ≈ 0.6 (log_std -0.5)
        // resampled each of 35 joints at 60 Hz makes the body convulse and fall
        // before any pose-holding reward accrues — it can never learn to stand.
        // Start near-deterministic (std ≈ 0.2) so a stable pose persists long
        // enough to earn the height/uprightness signal; the policy can widen
        // exploration itself via the learnable log_std if it pays off.
        let log_std = Param::from_tensor(Tensor::full([ACTION_SIZE], LOG_STD_INIT, device));

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
        let log_std = self
            .log_std
            .val()
            .clamp(LOG_STD_MIN as f64, LOG_STD_MAX as f64);
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
