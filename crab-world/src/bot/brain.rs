//! Actor-critic network for PPO: a shared MLP trunk feeding separate policy
//! (actor) and value (critic) heads.

use burn::module::Param;
use burn::nn;
use burn::prelude::*;

use super::actuator::ACTION_SIZE;
use super::sensor::OBS_SIZE;

const HIDDEN_SIZE: usize = 256;

/// Initial policy log-std (std ≈ 0.2): start with low exploration, see `new`.
const LOG_STD_INIT: f32 = -1.6;
/// Bounds the learnable log-std so entropy can't diverge or collapse. Single
/// source of truth: `policy` clamps to this range, so downstream log-prob /
/// entropy never re-clamp. exp(-2) ≈ 0.14 (focused), exp(0.5) ≈ 1.65 (wide).
/// `LOG_STD_MIN` is also the resting/refine floor the exploration schedule anneals
/// back down to once the wide-early window elapses (see `PpoConfig::log_std_floor`).
pub(crate) const LOG_STD_MIN: f32 = -2.0;
const LOG_STD_MAX: f32 = 0.5;

/// Actor-Critic network for PPO.
#[derive(Module, Debug)]
pub struct CrabBrain<B: Backend> {
    trunk_fc1: nn::Linear<B>,
    trunk_fc2: nn::Linear<B>,
    trunk_ln1: nn::LayerNorm<B>,
    trunk_ln2: nn::LayerNorm<B>,

    policy_fc: nn::Linear<B>,

    value_fc1: nn::Linear<B>,
    value_fc2: nn::Linear<B>,

    // A free learnable parameter, not a head off the trunk: exploration spread is
    // state-independent, so the policy can widen/narrow it globally as training pays.
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

        // The action is a per-step joint torque, so per-step noise is per-step
        // jitter. A wide std (≈0.6, log_std -0.5) resampled on every joint each
        // tick convulses the body and topples it before it can hold a pose and walk
        // — it can never set off. Start near-deterministic (std ≈ 0.2) so a stable
        // gait can persist long enough to make progress toward the target (the dense
        // reward); the policy can widen exploration itself via the learnable log_std.
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

    fn trunk(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.trunk_fc1.forward(obs);
        let x = self.trunk_ln1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.trunk_fc2.forward(x);
        let x = self.trunk_ln2.forward(x);
        burn::tensor::activation::relu(x)
    }

    /// Action means (tanh-bounded to [-1, 1]) and the log-std, clamped to
    /// `[log_std_floor, LOG_STD_MAX]` so downstream log-prob/entropy need not re-clamp.
    ///
    /// `log_std_floor` is the LOWER clamp bound — the minimum exploration spread for this
    /// forward. It is the lever the training schedule raises early to FORCE σ wide (the
    /// learned `log_std` param sits below it, so the clamp overrides it) and anneals back
    /// down to `LOG_STD_MIN` for refinement (see `PpoConfig::log_std_floor`). Pass
    /// [`LOG_STD_MIN`] for the unforced/default bound; eval takes the policy MEAN and
    /// discards `log_std`, so the floor never reaches a deployed action — exploration
    /// widening is training-only. The floor is itself clamped into `[LOG_STD_MIN,
    /// LOG_STD_MAX]` so a misconfigured schedule can't widen past the architectural bound.
    pub fn policy(&self, obs: Tensor<B, 2>, log_std_floor: f32) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let trunk = self.trunk(obs);
        let means = self.policy_fc.forward(trunk);
        let means = burn::tensor::activation::tanh(means);
        let lo = log_std_floor.clamp(LOG_STD_MIN, LOG_STD_MAX);
        let log_std = self.log_std.val().clamp(lo as f64, LOG_STD_MAX as f64);
        (means, log_std)
    }

    /// Critic value estimate, one scalar per batch row.
    pub fn value(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let trunk = self.trunk(obs);
        let x = self.value_fc1.forward(trunk);
        let x = burn::tensor::activation::relu(x);
        self.value_fc2.forward(x)
    }

    /// The `(obs_dim, action_dim)` this brain's weights were built for, read off the
    /// first trunk layer's input and the policy head's output (`Linear` weight is
    /// `[d_input, d_output]`). After [`load_record`](burn::module::Module::load_record)
    /// these reflect the *loaded* checkpoint, not [`OBS_SIZE`]/[`ACTION_SIZE`] — so a
    /// caller can reject a checkpoint trained against a different rig before its
    /// mismatched weights reach a forward pass and panic in the matmul.
    pub fn io_dims(&self) -> (usize, usize) {
        let [obs_dim, _hidden] = self.trunk_fc1.weight.shape().dims();
        let [_hidden, action_dim] = self.policy_fc.weight.shape().dims();
        (obs_dim, action_dim)
    }
}
