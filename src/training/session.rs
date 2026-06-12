//! Training session management.
//!
//! Manages the RL training loop integrated with the Bevy game loop.

use std::io::Write;
use std::path::{Path, PathBuf};

use bevy::app::AppExit;
use bevy::prelude::*;
use burn::backend::Autodiff;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use crate::Args;
use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabAssets, CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId};
use crate::bot::{CrabRescued, CrabSpawns, respawn_crab};
use crate::bot::brain::{ACTION_SIZE, CrabBrain};
use crate::bot::sensor::{CrabObservation, OBS_SIZE};

use super::algorithm::{
    PpoConfig, PpoMetrics, RolloutBuffer, Transition, compute_gae, compute_log_prob, sample_action,
};

/// Running observation normalizer using Welford's online algorithm.
/// Normalizes observations to zero mean, unit variance.
pub(crate) struct ObsNormalizer {
    mean: [f64; OBS_SIZE],
    var: [f64; OBS_SIZE],   // running variance (M2 / count)
    m2: [f64; OBS_SIZE],    // sum of squared differences from mean
    count: [u64; OBS_SIZE], // per-element count (NaN-skipped elements don't inflate others)
    clip: f32,              // max absolute normalized value
}

/// Serde-friendly mirror of `ObsNormalizer` (arrays > 32 don't auto-derive).
#[derive(Serialize, Deserialize)]
struct ObsNormalizerData {
    mean: Vec<f64>,
    var: Vec<f64>,
    m2: Vec<f64>,
    count: Vec<u64>,
    clip: f32,
}

impl ObsNormalizer {
    fn to_data(&self) -> ObsNormalizerData {
        ObsNormalizerData {
            mean: self.mean.to_vec(),
            var: self.var.to_vec(),
            m2: self.m2.to_vec(),
            count: self.count.to_vec(),
            clip: self.clip,
        }
    }

    fn from_data(d: ObsNormalizerData) -> Option<Self> {
        if d.mean.len() != OBS_SIZE
            || d.var.len() != OBS_SIZE
            || d.m2.len() != OBS_SIZE
            || d.count.len() != OBS_SIZE
        {
            warn!(
                "Normalizer size mismatch: expected {OBS_SIZE}, got {}",
                d.mean.len()
            );
            return None;
        }
        if d.clip <= 0.0 || d.var.iter().any(|&v| v < 0.0) {
            warn!("Normalizer contains invalid values (clip <= 0 or negative variance)");
            return None;
        }
        let mut n = Self::new(d.clip);
        n.mean.copy_from_slice(&d.mean);
        n.var.copy_from_slice(&d.var);
        n.m2.copy_from_slice(&d.m2);
        n.count.copy_from_slice(&d.count);
        Some(n)
    }
}

impl ObsNormalizer {
    pub(crate) fn new(clip: f32) -> Self {
        Self {
            mean: [0.0; OBS_SIZE],
            var: [1.0; OBS_SIZE],
            m2: [0.0; OBS_SIZE],
            count: [0; OBS_SIZE],
            clip,
        }
    }

    /// Update running stats, then return the normalized observation.
    pub(crate) fn normalize(&mut self, obs: &[f32; OBS_SIZE]) -> [f32; OBS_SIZE] {
        for (i, &raw) in obs.iter().enumerate() {
            if !raw.is_finite() {
                continue;
            }
            self.count[i] += 1;
            let n = self.count[i] as f64;
            let x = raw as f64;
            let delta = x - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = x - self.mean[i];
            self.m2[i] += delta * delta2;

            if self.count[i] > 1 {
                self.var[i] = (self.m2[i] / (n - 1.0)).max(0.0);
            }
        }
        self.normalize_frozen(obs)
    }

    /// Normalize against the current statistics WITHOUT updating them. Inference
    /// (play/demo) uses this so the running mean/var stay fixed at the values
    /// learned during training rather than drifting toward demo observations.
    pub(crate) fn normalize_frozen(&self, obs: &[f32; OBS_SIZE]) -> [f32; OBS_SIZE] {
        let clip = self.clip;
        let mut normalized = [0.0f32; OBS_SIZE];
        for i in 0..OBS_SIZE {
            let raw = obs[i];
            if !raw.is_finite() {
                normalized[i] = 0.0;
                continue;
            }
            let std = (self.var[i] as f32).sqrt().max(1e-6);
            let val = (raw - self.mean[i] as f32) / std;
            normalized[i] = if val.is_nan() {
                0.0
            } else {
                val.clamp(-clip, clip)
            };
        }
        normalized
    }

    fn save(&self, path: &Path) {
        let data = self.to_data();
        let bytes = match bincode::serialize(&data) {
            Ok(b) => b,
            Err(e) => {
                warn!("Failed to serialize normalizer: {e}");
                return;
            }
        };
        if let Err(e) = std::fs::write(path, bytes) {
            warn!("Failed to write normalizer to {}: {e}", path.display());
        }
    }

    pub(crate) fn load(path: &Path) -> Option<Self> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                warn!("Failed to read normalizer from {}: {e}", path.display());
                return None;
            }
        };
        let data: ObsNormalizerData = match bincode::deserialize(&bytes) {
            Ok(d) => d,
            Err(e) => {
                warn!(
                    "Failed to deserialize normalizer from {}: {e}",
                    path.display()
                );
                return None;
            }
        };
        Self::from_data(data)
    }
}

/// Backend type aliases. Training carries autodiff; inference (play/demo) uses
/// the bare inner backend — `AutodiffModule::valid()` converts one to the other.
pub type TrainBackend = Autodiff<NdArray>;
pub type InferBackend = NdArray;

/// Concrete optimizer type.
type CrabOptimizer = OptimizerAdaptor<Adam, CrabBrain<TrainBackend>, TrainBackend>;

/// CSV logger for training metrics.
struct MetricsLogger {
    episode_file: std::fs::File,
    update_file: std::fs::File,
    update_count: u32,
}

impl MetricsLogger {
    fn new() -> Self {
        std::fs::create_dir_all("tmp").expect("failed to create tmp/");

        let mut episode_file =
            std::fs::File::create("tmp/episodes.csv").expect("failed to create tmp/episodes.csv");
        writeln!(
            episode_file,
            "episode,reward,steps,avg_reward_10,mean_height,mean_upright,mean_energy"
        )
        .expect("failed to write header");

        let mut update_file = std::fs::File::create("tmp/ppo_updates.csv")
            .expect("failed to create tmp/ppo_updates.csv");
        writeln!(
            update_file,
            "update,policy_loss,value_loss,entropy,avg_reward,buffer_size"
        )
        .expect("failed to write header");

        Self {
            episode_file,
            update_file,
            update_count: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn log_episode(
        &mut self,
        episode: u32,
        reward: f32,
        steps: u32,
        avg_reward: f32,
        mean_height: f32,
        mean_upright: f32,
        mean_energy: f32,
    ) {
        writeln!(
            self.episode_file,
            "{},{:.4},{},{},{:.4},{:.4},{:.4}",
            episode, reward, steps, avg_reward, mean_height, mean_upright, mean_energy
        )
        .ok();
        if episode.is_multiple_of(10) {
            self.episode_file.flush().ok();
        }
    }

    fn log_update(&mut self, metrics: &PpoMetrics, avg_reward: f32, buffer_size: usize) {
        self.update_count += 1;
        writeln!(
            self.update_file,
            "{},{:.6},{:.6},{:.6},{:.4},{}",
            self.update_count,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            avg_reward,
            buffer_size,
        )
        .ok();
        self.update_file.flush().ok();
    }
}

/// Stem for brain checkpoint files. `BinFileRecorder` appends `.bin` automatically,
/// so the actual file on disk is `brain.bin`.
pub(crate) const BRAIN_STEM: &str = "brain";
pub(crate) const NORMALIZER_FILENAME: &str = "normalizer.bin";

/// The RL training state. Stored as a non-send resource because burn
/// tensors use `OnceCell` which is not `Sync`.
/// Per-env episode accumulators. Each env's episode runs and resets
/// independently; pose sums (carapace height, up·Y) are averaged at episode
/// end to quantify stance quality.
#[derive(Clone, Default)]
pub struct EnvEpisode {
    pub reward: f32,
    pub steps: u32,
    pub height_sum: f32,
    pub upright_sum: f32,
    pub energy_sum: f32,
    pub needs_reset: bool,
    /// Settle ticks remaining before the episode starts recording. A reset
    /// respawns a fresh crab in the rest pose ([`crate::bot::respawn_crab`]);
    /// during grace no transitions/termination are evaluated while it drops
    /// from spawn height and the motors take the load.
    pub grace: u32,
}

pub struct TrainingState {
    pub brain: CrabBrain<TrainBackend>,
    pub config: PpoConfig,
    /// One rollout buffer per env. Kept separate so GAE never sweeps across an
    /// env boundary — interleaving transitions from different envs in one
    /// buffer would bootstrap each env's advantages from another env's values.
    pub rollouts: Vec<RolloutBuffer>,
    pub device: NdArrayDevice,
    optimizer: CrabOptimizer,

    pub envs: Vec<EnvEpisode>,
    pub episode_count: u32,

    pub steps_per_rollout: u32,
    pub current_rollout_steps: u32,

    pub recent_rewards: Vec<f32>,
    pub recent_metrics: Option<PpoMetrics>,

    logger: MetricsLogger,

    pub total_steps: u64,
    last_log_time: std::time::Instant,

    obs_normalizer: ObsNormalizer,

    checkpoint_dir: PathBuf,
    checkpoint_interval: u32,
    saved_on_exit: bool,

    /// Stop after this many physics ticks (0 = unlimited). See `Args::ticks`.
    tick_budget: u64,
    /// Benchmark: skip NN inference to measure the physics/overhead floor.
    skip_nn: bool,
}

impl TrainingState {
    pub fn new(args: &Args) -> Self {
        let device = NdArrayDevice::Cpu;
        let mut brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);
        let optimizer: CrabOptimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
            .init();

        let mut obs_normalizer = ObsNormalizer::new(5.0);

        let brain_path = args.checkpoint_dir.join(BRAIN_STEM);
        let norm_path = args.checkpoint_dir.join(NORMALIZER_FILENAME);

        // BinFileRecorder appends .bin to the stem, so check for that.
        if brain_path.with_extension("bin").exists() {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            match recorder.load(brain_path.clone(), &device) {
                Ok(record) => {
                    brain = brain.load_record(record);
                    info!("Loaded brain weights from {}", brain_path.display());
                }
                Err(e) => {
                    warn!(
                        "Failed to load brain from {}: {e} — starting fresh",
                        brain_path.display()
                    );
                }
            }
        }

        if norm_path.exists()
            && let Some(loaded) = ObsNormalizer::load(&norm_path)
        {
            info!("Loaded normalizer state from {}", norm_path.display());
            obs_normalizer = loaded;
        }

        let n = args.envs.max(1) as usize;
        Self {
            brain,
            config: PpoConfig::default(),
            rollouts: (0..n).map(|_| RolloutBuffer::new()).collect(),
            device,
            optimizer,
            envs: vec![EnvEpisode::default(); n],
            episode_count: 0,
            steps_per_rollout: 1024,
            current_rollout_steps: 0,
            recent_rewards: Vec::new(),
            recent_metrics: None,
            logger: MetricsLogger::new(),
            total_steps: 0,
            last_log_time: std::time::Instant::now(),
            obs_normalizer,
            checkpoint_dir: args.checkpoint_dir.clone(),
            checkpoint_interval: args.save_interval,
            saved_on_exit: false,
            tick_budget: args.ticks,
            skip_nn: args.bench_skip_nn,
        }
    }

    fn save_checkpoint(&self) {
        if let Err(e) = std::fs::create_dir_all(&self.checkpoint_dir) {
            warn!(
                "Failed to create checkpoint dir {}: {e}",
                self.checkpoint_dir.display()
            );
            return;
        }

        let brain_path = self.checkpoint_dir.join(BRAIN_STEM);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        match recorder.record(self.brain.clone().into_record(), brain_path.clone()) {
            Ok(()) => info!("Saved brain to {}", brain_path.display()),
            Err(e) => warn!("Failed to save brain: {e}"),
        }

        let norm_path = self.checkpoint_dir.join(NORMALIZER_FILENAME);
        self.obs_normalizer.save(&norm_path);
    }

    fn avg_reward(&self, window: usize) -> f32 {
        if self.recent_rewards.is_empty() {
            return 0.0;
        }
        let n = self.recent_rewards.len();
        let start = n.saturating_sub(window);
        let slice = &self.recent_rewards[start..];
        slice.iter().sum::<f32>() / slice.len() as f32
    }

    /// Run PPO update using the persistent Adam optimizer.
    fn ppo_update(&mut self) -> PpoMetrics {
        let n: usize = self.rollouts.iter().map(|b| b.len()).sum();
        if n == 0 {
            return PpoMetrics::default();
        }

        let device = &self.device;

        // GAE strictly per env: each buffer is one env's contiguous trajectory
        // segment, bootstrapped from ITS last observation. Advantages/returns are
        // then concatenated in the same env-major order as the transitions below.
        let mut advantages = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        for buf in &self.rollouts {
            let Some(last_t) = buf.transitions.last() else {
                continue;
            };
            let last_value = if last_t.done {
                0.0
            } else {
                let obs = Tensor::<TrainBackend, 1>::from_floats(last_t.obs.as_slice(), device)
                    .unsqueeze::<2>();
                self.brain
                    .value(obs)
                    .flatten::<1>(0, 1)
                    .into_scalar()
                    .elem::<f32>()
            };
            let (a, r) = compute_gae(buf, last_value, self.config.gamma, self.config.lambda);
            advantages.extend(a);
            returns.extend(r);
        }

        // Env-major transition view matching the advantages/returns order.
        let transitions: Vec<&Transition> = self
            .rollouts
            .iter()
            .flat_map(|b| b.transitions.iter())
            .collect();

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
        let old_log_probs_data: Vec<f32> = transitions.iter().map(|t| t.log_prob).collect();

        let obs_all =
            Tensor::<TrainBackend, 2>::from_data(TensorData::new(obs_data, [n, OBS_SIZE]), device);
        let actions_all = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(actions_data, [n, ACTION_SIZE]),
            device,
        );
        let old_log_probs_all =
            Tensor::<TrainBackend, 1>::from_data(TensorData::new(old_log_probs_data, [n]), device);
        let advantages_all =
            Tensor::<TrainBackend, 1>::from_data(TensorData::new(advantages_norm, [n]), device);
        let returns_all =
            Tensor::<TrainBackend, 1>::from_data(TensorData::new(returns, [n]), device);

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut update_count = 0u32;

        let bs = self.config.batch_size;
        let half_log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();

        let mut rng = thread_rng();

        for _epoch in 0..self.config.epochs_per_update {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);

            let num_batches = n.div_ceil(bs);

            for batch_idx in 0..num_batches {
                let start = batch_idx * bs;
                let end = (start + bs).min(n);
                let batch_n = end - start;
                let batch_indices = &indices[start..end];

                let idx_tensor = Tensor::<TrainBackend, 1, Int>::from_data(
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

                let (means, log_std) = self.brain.policy(obs.clone());

                // log_std is pre-clamped by policy (single source of truth).
                let diff = actions - means;
                let log_std_2d = log_std
                    .clone()
                    .unsqueeze_dim::<2>(0)
                    .expand([batch_n, ACTION_SIZE]);
                let scaled_diff = diff / log_std_2d.clone().exp();
                let log_probs_per_dim =
                    scaled_diff.powf_scalar(2.0).neg() * 0.5 - log_std_2d - half_log_2pi;
                let new_lp: Tensor<TrainBackend, 1> = log_probs_per_dim.sum_dim(1).flatten(0, 1);

                let entropy_per_dim = log_std.clone()
                    + (0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln());
                let entropy = entropy_per_dim.mean();

                let log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0);
                let ratio = log_ratio.exp();
                let surr1 = ratio.clone() * advs.clone();
                let surr2 = ratio.clamp(
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advs;
                let policy_loss = surr1.min_pair(surr2).mean().neg();

                let values: Tensor<TrainBackend, 1> = self.brain.value(obs).flatten(0, 1);
                let value_diff = (values - rets)
                    .clamp(-self.config.value_loss_clip, self.config.value_loss_clip);
                let value_loss = value_diff.powf_scalar(2.0).mean();

                let loss = policy_loss.clone() + value_loss.clone() * self.config.value_coeff
                    - entropy.clone() * self.config.entropy_coeff;

                total_policy_loss += policy_loss.clone().into_scalar().elem::<f32>();
                total_value_loss += value_loss.clone().into_scalar().elem::<f32>();
                total_entropy += entropy.clone().into_scalar().elem::<f32>();
                update_count += 1;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.brain);
                self.brain =
                    self.optimizer
                        .step(self.config.learning_rate, self.brain.clone(), grads);
            }
        }

        PpoMetrics {
            policy_loss: total_policy_loss / update_count as f32,
            value_loss: total_value_loss / update_count as f32,
            entropy: total_entropy / update_count as f32,
        }
    }
}

/// Energy penalty coefficient. Quadratic in angular speed, so a calm stance or
/// a slow step costs ~nothing (Σω² ≈ 10 → penalty 0.001) while a violent launch
/// spike (limbs at 30+ rad/s, Σω² ≈ 30k) costs more than the height it buys.
/// First cut — tune from the per-episode Energy log, not by eye.
const ENERGY_COST: f32 = 1e-4;

/// Reward: mean eye height minus an energy tax. Two signals, both global —
/// behaviour still EMERGES rather than being hand-specified (owner's call:
/// mechanical terms like "feet on the ground" don't scale).
///
/// The eyes sit at the top of the kinematic chain (carapace → stalk → eye), so a
/// high eye reading needs the carapace both LEVEL (stalks point up) and HIGH (legs
/// extended underneath) — i.e. standing. Over a long (~25 s) episode the summed
/// return favours SUSTAINED height.
///
/// `energy` is Σ|ω|² over every body part (world angular speed). It is a proxy
/// for actuation effort, not true joint work — rapier 0.32 exposes neither joint
/// torques nor multibody coordinates (coords() lands in 0.33), so segment
/// angular speed is the most honest observable stand-in. Owner's goal is
/// believable, economical movement ("flying is cool, but…"); the tax prices
/// flailing without prescribing any gait.
fn compute_reward(mean_eye_height: f32, energy: f32) -> f32 {
    mean_eye_height - ENERGY_COST * energy
}

/// System: runs the brain to produce actions each physics step.
pub fn brain_step(
    mut training: NonSendMut<TrainingState>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: Query<(&CrabEnvId, &bevy_rapier3d::prelude::Velocity), With<CrabBodyPart>>,
    eyes_q: Query<(&CrabJoint, &CrabEnvId, &Transform)>,
    mut exit: MessageWriter<AppExit>,
    mut rescued: MessageReader<CrabRescued>,
) {
    let n = training.envs.len();
    // Envs whose crab went non-finite and was force-respawned this tick: the
    // pose this step reads is already the fresh crab back at spawn, so the
    // episode must end here or the teleport bleeds into the reward stream.
    let rescued_envs: Vec<usize> = rescued.read().map(|m| m.env).collect();
    if obs.envs.len() != n || actions.envs.len() != n {
        // Resources are sized by the spawn system; skip the tick(s) before that.
        return;
    }
    let device = training.device;

    // Normalize every env's observation (each row also updates the shared
    // running stats — N envs feed the same normalizer, just more samples).
    let mut obs_arrays: Vec<[f32; OBS_SIZE]> = Vec::with_capacity(n);
    for e in 0..n {
        let normalized = training.obs_normalizer.normalize(&obs.envs[e]);
        obs_arrays.push(normalized);
    }

    // ONE batched forward pass for all envs: [n, OBS_SIZE] through the trunk
    // once — this is what makes N crabs cheaper than N apps.
    let (means_rows, log_std, values): (Vec<Tensor<NdArray, 1>>, Tensor<NdArray, 1>, Vec<f32>) =
        if training.skip_nn {
            // Bench mode: no forward pass. The sampling below is cheap and what
            // it produces is irrelevant — we are isolating physics + overhead.
            let z = Tensor::<NdArray, 1>::zeros([ACTION_SIZE], &device);
            (vec![z.clone(); n], z, vec![0.0; n])
        } else {
            let inference_brain = training.brain.valid();
            let flat: Vec<f32> = obs_arrays.iter().flat_map(|a| a.iter().copied()).collect();
            let obs_batch = Tensor::<NdArray, 2>::from_data(
                burn::tensor::TensorData::new(flat, [n, OBS_SIZE]),
                &device,
            );
            let (means_batch, log_std) = inference_brain.policy(obs_batch.clone());
            let values: Vec<f32> = inference_brain
                .value(obs_batch)
                .flatten::<1>(0, 1)
                .to_data()
                .to_vec()
                .unwrap();
            let means_rows = (0..n)
                .map(|e| {
                    means_batch
                        .clone()
                        .slice([e..e + 1, 0..ACTION_SIZE])
                        .flatten(0, 1)
                })
                .collect();
            (means_rows, log_std, values)
        };

    // Per-env action sampling + NaN guards.
    let mut action_arrays: Vec<[f32; ACTION_SIZE]> = Vec::with_capacity(n);
    let mut log_probs: Vec<f32> = Vec::with_capacity(n);
    for means in &means_rows {
        let action_tensor = sample_action(means, &log_std, &device);
        let log_prob = compute_log_prob(means, &log_std, &action_tensor);
        log_probs.push(if log_prob.is_nan() || log_prob.is_infinite() {
            0.0
        } else {
            log_prob.clamp(-20.0, 20.0)
        });

        let action_data: Vec<f32> = action_tensor.to_data().to_vec().unwrap();
        let mut action_array = [0.0f32; ACTION_SIZE];
        let mut has_nan = false;
        for (i, &v) in action_data.iter().enumerate().take(ACTION_SIZE) {
            if v.is_nan() || v.is_infinite() {
                has_nan = true;
                action_array[i] = 0.0;
            } else {
                action_array[i] = v.clamp(-1.0, 1.0);
            }
        }
        if has_nan {
            warn!("NaN/Inf detected in NN output, clamping to zero");
        }
        action_arrays.push(action_array);
    }
    actions.envs.copy_from_slice(&action_arrays);
    // Settling envs hold the rest pose (action 0); the policy takes over at
    // step 0 of the new episode.
    for (e, ep) in training.envs.iter().enumerate() {
        if ep.grace > 0
            && let Some(v) = actions.envs.get_mut(e)
        {
            *v = [0.0; ACTION_SIZE];
        }
    }

    // Gather per-env body state: carapace pose + mean eye-tip height.
    let mut poses: Vec<Option<(f32, f32)>> = vec![None; n]; // (height, upright)
    for (env, transform) in carapace_q.iter() {
        if let Some(p) = poses.get_mut(env.0) {
            let up = transform.rotation * Vec3::Y;
            *p = Some((transform.translation.y, up.dot(Vec3::Y)));
        }
    }
    // Fastest body part per env — limbs, not the carapace, blow up first (tiny
    // eye-stalk balls + acceleration motors), so the blowup guard must watch
    // every body. NaN poisons the max, so fold it in as +inf.
    let mut max_speeds: Vec<f32> = vec![0.0; n];
    let mut energies: Vec<f32> = vec![0.0; n];
    for (env, vel) in parts_q.iter() {
        if let Some(m) = max_speeds.get_mut(env.0) {
            let lin = vel.linear.length();
            let ang = vel.angular.length();
            let s = if lin.is_finite() && ang.is_finite() {
                // Angular blowups (rad/s) run ~3x the linear scale before the
                // solver NaNs; fold both into one number on the linear scale.
                lin.max(ang / 3.0)
            } else {
                f32::INFINITY
            };
            *m = m.max(s);
            // Non-finite ω is the blowup guard's problem, not the tax's.
            if ang.is_finite() {
                energies[env.0] += ang * ang;
            }
        }
    }
    let mut eye_sums: Vec<(f32, u32)> = vec![(0.0, 0); n];
    for (joint, env, eye) in eyes_q.iter() {
        if matches!(joint.id, CrabJointId::EyeStalk(_))
            && let Some(s) = eye_sums.get_mut(env.0)
        {
            s.0 += eye.translation.y;
            s.1 += 1;
        }
    }

    // Per-env reward, termination, rollout push, and episode bookkeeping.
    for e in 0..n {
        // No transitions while the env is settling into its start pose (or, as
        // a guard, if its crab is somehow absent this tick).
        if training.envs[e].grace > 0 || poses[e].is_none() {
            continue;
        }
        let (reward, done, height, upright) = if let Some((height, upright)) = poses[e] {
            // Mean world-space height of the eye tips, taxed by Σω².
            let (sum, cnt) = eye_sums[e];
            let mean_eye_height = if cnt > 0 { sum / cnt as f32 } else { height };
            let r = compute_reward(mean_eye_height, energies[e]);
            // Termination is survival guards only — jumping, flipping, and
            // any other strategy the policy invents are legitimate solutions
            // (owner call: emergent behavior is the point). The height band
            // is sim sanity (clipped through the floor / left the playfield),
            // not a behavior bound. The blowup check isn't shaping either:
            // the acceleration-based motors can pump in energy until the
            // solver NaNs and Rapier panics the whole app — ending the
            // episode resets velocities before it gets there.
            let blowing_up = max_speeds[e] > 30.0 || !height.is_finite();
            let done = !(0.02..=50.0).contains(&height)
                || blowing_up
                || training.envs[e].steps > 1500
                || rescued_envs.contains(&e);
            (r, done, height, upright)
        } else {
            (0.0, false, 0.0, 0.0)
        };

        training.rollouts[e].push(Transition {
            obs: obs_arrays[e],
            action: action_arrays[e],
            reward,
            value: values[e],
            log_prob: log_probs[e],
            done,
        });

        let ep = &mut training.envs[e];
        ep.reward += reward;
        ep.steps += 1;
        ep.height_sum += height;
        ep.upright_sum += upright;
        ep.energy_sum += energies[e];

        if done {
            let ep_reward = ep.reward;
            let ep_steps = ep.steps;
            let ep_height = ep.height_sum / ep_steps.max(1) as f32;
            let ep_upright = ep.upright_sum / ep_steps.max(1) as f32;
            let ep_energy = ep.energy_sum / ep_steps.max(1) as f32;
            *ep = EnvEpisode {
                needs_reset: true,
                ..EnvEpisode::default()
            };

            training.recent_rewards.push(ep_reward);
            training.episode_count += 1;
            let avg = training.avg_reward(10);
            let ep_count = training.episode_count;

            training.logger.log_episode(
                ep_count, ep_reward, ep_steps, avg, ep_height, ep_upright, ep_energy,
            );

            if training.episode_count.is_multiple_of(10) {
                let elapsed = training.last_log_time.elapsed().as_secs_f32();
                let total_transitions =
                    training.total_steps * n as u64 + (e as u64 + 1).min(n as u64);
                let sps = if elapsed > 0.0 {
                    total_transitions as f32 / elapsed
                } else {
                    0.0
                };
                let buffered: usize = training.rollouts.iter().map(|b| b.len()).sum();
                info!(
                    "Ep {} | Avg reward: {:.2} | Steps: {} | Height: {:.2} | Upright: {:.2} | Energy: {:.0} | Buffer: {} | {:.0} steps/s",
                    training.episode_count, avg, ep_steps, ep_height, ep_upright, ep_energy, buffered, sps,
                );
            }
        }
    }

    training.current_rollout_steps += n as u32;
    training.total_steps += 1;

    // Fixed-tick stop: exactly `tick_budget` physics ticks, then save+exit. Tick
    // count, never wall-clock, so the run is reproducible across machines/load.
    // `==` (steps increment by one) so the catch-up burst that crosses the budget
    // logs/requests exit once, not once per remaining tick in the burst.
    if training.tick_budget != 0 && training.total_steps == training.tick_budget {
        info!(
            "Tick budget reached ({} ticks) — stopping training.",
            training.tick_budget
        );
        exit.write(AppExit::Success);
    }

    if training.current_rollout_steps >= training.steps_per_rollout {
        let buffer_size: usize = training.rollouts.iter().map(|b| b.len()).sum();
        let avg = training.avg_reward(10);

        info!("Running PPO update on {} transitions...", buffer_size);
        let metrics = training.ppo_update();

        info!(
            "PPO | Policy loss: {:.4} | Value loss: {:.4} | Entropy: {:.4}",
            metrics.policy_loss, metrics.value_loss, metrics.entropy,
        );

        training.logger.log_update(&metrics, avg, buffer_size);

        training.recent_metrics = Some(metrics);
        for buf in training.rollouts.iter_mut() {
            buf.clear();
        }
        training.current_rollout_steps = 0;

        if training.checkpoint_interval > 0
            && training
                .logger
                .update_count
                .is_multiple_of(training.checkpoint_interval)
        {
            training.save_checkpoint();
        }
    }
}

/// Settle ticks after a respawn: the fresh crab spawns in the rest pose with
/// the builder motors already holding it, so this only covers the drop from
/// spawn height onto the ground and the motors taking the load (0.5 s).
const RESET_GRACE_TICKS: u32 = 32;

/// System: rebuilds each env's crab when that env's episode ends.
///
/// A reset is a full despawn + respawn ([`respawn_crab`]): teleporting bodies
/// cannot repair a multibody whose joint state went non-finite — rapier 0.32
/// offers no way to rewrite multibody joint coordinates in place — and one
/// crab that tunnels through the floor would otherwise wedge its env forever.
/// The respawned crab starts in the overlap-free rest pose, so no unfold or
/// collision-group dance is needed; the grace just skips recording while it
/// takes load (see [`EnvEpisode::grace`]).
pub fn reset_crab(
    mut commands: Commands,
    mut training: NonSendMut<TrainingState>,
    mut actions: ResMut<CrabActions>,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>,
) {
    for (e, ep) in training.envs.iter_mut().enumerate() {
        if std::mem::take(&mut ep.needs_reset) {
            ep.grace = RESET_GRACE_TICKS;
            if let Some(v) = actions.envs.get_mut(e) {
                *v = [0.0; CrabJointId::COUNT];
            }
            let origin = spawns.0.get(e).copied().unwrap_or(Vec3::ZERO);
            respawn_crab(
                &mut commands,
                &assets,
                parts.iter().filter(|(_, id)| id.0 == e).map(|(ent, _)| ent),
                origin,
                e,
            );
        }
    }

    for ep in training.envs.iter_mut() {
        if ep.grace > 0 {
            ep.grace -= 1;
        }
    }
}

/// System: saves a final checkpoint when the app is about to exit.
pub fn save_on_exit(
    mut training: NonSendMut<TrainingState>,
    mut exit_events: bevy::prelude::MessageReader<AppExit>,
) {
    if training.saved_on_exit {
        return;
    }
    if exit_events.read().next().is_some() {
        info!("App exiting — saving final checkpoint...");
        training.save_checkpoint();
        training.saved_on_exit = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_increases_with_eye_height() {
        // Higher eyes (standing/reared) must score strictly above low eyes
        // (collapsed/flat) at equal effort.
        assert!(
            compute_reward(1.2, 0.0) > compute_reward(0.2, 0.0),
            "reward must increase with eye height"
        );
    }

    #[test]
    fn energy_tax_calibration() {
        // A calm stance (Σω² ~ 10) must cost a rounding error relative to its
        // height signal, while launch-grade violence (limbs near the 30 rad/s
        // blowup guard, Σω² ~ 30k) must cost more than the height it buys —
        // otherwise the tax either distorts standing or fails to price
        // flailing. Pins ENERGY_COST's order of magnitude.
        let calm = compute_reward(0.5, 10.0);
        assert!((calm - 0.5).abs() < 0.01, "standing must be ~untaxed: {calm}");
        let launch = compute_reward(3.0, 30_000.0);
        assert!(launch < 0.5, "launch spike must not out-earn a stand: {launch}");
    }

    #[test]
    fn welford_per_element_count_correctness() {
        let mut norm = ObsNormalizer::new(5.0);

        for i in 0..100 {
            let mut obs = [0.0f32; OBS_SIZE];
            obs[0] = 1.0;
            obs[1] = if i % 2 == 0 { 1.0 } else { f32::NAN };
            norm.normalize(&obs);
        }

        assert!(
            (norm.mean[0] - 1.0).abs() < 0.01,
            "element 0 mean should be ~1.0, got {}",
            norm.mean[0]
        );
        assert_eq!(norm.count[0], 100);

        assert!(
            (norm.mean[1] - 1.0).abs() < 0.01,
            "element 1 mean should be ~1.0, got {}",
            norm.mean[1]
        );
        assert_eq!(norm.count[1], 50);
    }

    #[test]
    fn log_prob_clamp_does_not_destroy_valid_values() {
        let log_prob: f32 = 18.6;
        let clamped = log_prob.clamp(-20.0, 20.0);
        assert!(
            (clamped - 18.6).abs() < 1e-6,
            "log_prob {log_prob} was clipped to {clamped} — symmetric clamp should preserve it"
        );
    }

    #[test]
    fn brain_checkpoint_round_trips() {
        let dir = std::env::temp_dir().join("rl_test_brain_checkpoint");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let device = NdArrayDevice::Cpu;
        let brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);

        let stem = dir.join(BRAIN_STEM);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .record(brain.clone().into_record(), stem.clone())
            .expect("save brain");

        assert!(
            stem.with_extension("bin").exists(),
            "brain.bin should exist"
        );

        let loaded_record = recorder.load(stem, &device).expect("load brain");
        let loaded = CrabBrain::<TrainBackend>::new(&device).load_record(loaded_record);

        let test_obs = Tensor::<TrainBackend, 2>::zeros([1, OBS_SIZE], &device);
        let (orig_means, orig_log_std) = brain.policy(test_obs.clone());
        let (loaded_means, loaded_log_std) = loaded.policy(test_obs);

        let orig_m: Vec<f32> = orig_means.to_data().to_vec().unwrap();
        let loaded_m: Vec<f32> = loaded_means.to_data().to_vec().unwrap();
        for (i, (a, b)) in orig_m.iter().zip(loaded_m.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "policy mean[{i}] diverged: {a} vs {b}"
            );
        }

        let orig_s: Vec<f32> = orig_log_std.to_data().to_vec().unwrap();
        let loaded_s: Vec<f32> = loaded_log_std.to_data().to_vec().unwrap();
        for (i, (a, b)) in orig_s.iter().zip(loaded_s.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "log_std[{i}] diverged: {a} vs {b}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn normalizer_round_trips_through_bincode() {
        let mut norm = ObsNormalizer::new(5.0);
        for i in 0..50 {
            let mut obs = [0.0f32; OBS_SIZE];
            obs[0] = i as f32;
            obs[1] = (i as f32) * 0.5;
            norm.normalize(&obs);
        }

        let data = norm.to_data();
        let bytes = bincode::serialize(&data).expect("serialize");
        let loaded_data: ObsNormalizerData = bincode::deserialize(&bytes).expect("deserialize");
        let loaded = ObsNormalizer::from_data(loaded_data).expect("from_data");

        assert_eq!(norm.count, loaded.count);
        for i in 0..OBS_SIZE {
            assert!(
                (norm.mean[i] - loaded.mean[i]).abs() < 1e-10,
                "mean[{i}] mismatch"
            );
            assert!(
                (norm.var[i] - loaded.var[i]).abs() < 1e-10,
                "var[{i}] mismatch"
            );
            assert!(
                (norm.m2[i] - loaded.m2[i]).abs() < 1e-10,
                "m2[{i}] mismatch"
            );
        }
        assert_eq!(norm.clip, loaded.clip);
    }
}
