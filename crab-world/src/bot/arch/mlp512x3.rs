//! `mlp512x3` — the second architecture leaf (bddap/rl#200 increment 5): a 512-wide
//! THREE-layer MLP trunk feeding the same separate policy/value heads as `mlp256`, with
//! the same state-independent learnable per-dim `log_std`. ~5.5× the parameters of
//! `mlp256` (~730k vs ~130k at the 92/38 rig) — a deliberate single-variable capacity
//! A/B: everything except trunk depth×width (and the value head scaling with it) is
//! copied from `mlp256`, including the small-gain head init and `LOG_STD_INIT`, so an
//! equal-env-step curve difference is attributable to capacity, not init policy.

use burn::module::Param;
use burn::nn;
use burn::prelude::*;

use super::super::actuator::ACTION_SIZE;
use super::super::sensor::OBS_SIZE;

const HIDDEN_SIZE: usize = 512;

/// Initial policy log-std (std ≈ 0.2) — same value and rationale as `mlp256`
/// (start near-deterministic so a gait can persist; the policy widens itself): the
/// A/B must not vary exploration init alongside capacity.
const LOG_STD_INIT: f32 = -1.6;

/// Actor-Critic network for PPO — `mlp256`'s shape scaled to a 3×512 trunk.
#[derive(Module, Debug)]
pub struct Mlp512x3<B: Backend> {
    trunk_fc1: nn::Linear<B>,
    trunk_fc2: nn::Linear<B>,
    trunk_fc3: nn::Linear<B>,
    trunk_ln1: nn::LayerNorm<B>,
    trunk_ln2: nn::LayerNorm<B>,
    trunk_ln3: nn::LayerNorm<B>,

    policy_fc: nn::Linear<B>,

    value_fc1: nn::Linear<B>,
    value_fc2: nn::Linear<B>,

    // A free learnable parameter, not a head off the trunk: exploration spread is
    // state-independent, so the policy can widen/narrow it globally as training pays.
    // LAST field by the registry convention (record bincode layout follows field
    // order; every leaf keeps log_std last so the layout rule stays uniform).
    log_std: Param<Tensor<B, 1>>,
}

impl<B: Backend> Mlp512x3<B> {
    pub fn new(device: &B::Device) -> Self {
        let trunk_fc1 = nn::LinearConfig::new(OBS_SIZE, HIDDEN_SIZE)
            .with_bias(true)
            .init(device);
        let trunk_fc2 = nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
            .with_bias(true)
            .init(device);
        let trunk_fc3 = nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
            .with_bias(true)
            .init(device);
        let trunk_ln1 = nn::LayerNormConfig::new(HIDDEN_SIZE).init(device);
        let trunk_ln2 = nn::LayerNormConfig::new(HIDDEN_SIZE).init(device);
        let trunk_ln3 = nn::LayerNormConfig::new(HIDDEN_SIZE).init(device);

        // Small-gain init on the policy head, same rationale as mlp256: the trunk ends
        // in a LayerNorm (unit-scale output), and the action IS the joint torque — a
        // default-init head would command near-max torque everywhere and launch the
        // crab. Tiny weights start it near-limp.
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

        let log_std = Param::from_tensor(Tensor::full([ACTION_SIZE], LOG_STD_INIT, device));

        Self {
            trunk_fc1,
            trunk_fc2,
            trunk_fc3,
            trunk_ln1,
            trunk_ln2,
            trunk_ln3,
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
        let x = burn::tensor::activation::relu(x);
        let x = self.trunk_fc3.forward(x);
        let x = self.trunk_ln3.forward(x);
        burn::tensor::activation::relu(x)
    }

    /// RAW heads per the seam contract (see [`super::AnyBrain::policy`]): tanh-bounded
    /// action means and the learned `log_std` broadcast per-row to `[rows, ACTION_SIZE]`
    /// — UN-floored and UN-clamped; only [`super::GaussianHead`] may clamp.
    pub fn policy(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let trunk = self.trunk(obs);
        let means = self.policy_fc.forward(trunk);
        let means = burn::tensor::activation::tanh(means);
        let rows = means.dims()[0];
        let log_std = self
            .log_std
            .val()
            .unsqueeze_dim::<2>(0)
            .expand([rows, ACTION_SIZE]);
        (means, log_std)
    }

    /// Critic value estimate, one scalar per batch row.
    pub fn value(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let trunk = self.trunk(obs);
        let x = self.value_fc1.forward(trunk);
        let x = burn::tensor::activation::relu(x);
        self.value_fc2.forward(x)
    }

    /// The `(obs_dim, action_dim)` this brain's weights were built for — same
    /// read-off-the-weights contract as `mlp256` (see its doc for why: rejects a
    /// wrong-rig checkpoint before a forward pass panics in the matmul).
    pub fn io_dims(&self) -> (usize, usize) {
        let [obs_dim, _hidden] = self.trunk_fc1.weight.shape().dims();
        let [_hidden, action_dim] = self.policy_fc.weight.shape().dims();
        (obs_dim, action_dim)
    }
}
