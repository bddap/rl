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
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabJoint, CrabJointId};
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
            "episode,reward,steps,avg_reward_10,mean_height,mean_upright"
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
    ) {
        writeln!(
            self.episode_file,
            "{},{:.4},{},{},{:.4},{:.4}",
            episode, reward, steps, avg_reward, mean_height, mean_upright
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
pub struct TrainingState {
    pub brain: CrabBrain<TrainBackend>,
    pub config: PpoConfig,
    pub rollout: RolloutBuffer,
    pub device: NdArrayDevice,
    optimizer: CrabOptimizer,

    pub episode_reward: f32,
    pub episode_steps: u32,
    // Per-episode pose accumulators (averaged at episode end) — quantify stance
    // quality: mean carapace height and mean uprightness (up·Y, 1 = level).
    pub episode_height_sum: f32,
    pub episode_upright_sum: f32,
    pub episode_count: u32,
    pub needs_reset: bool,

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

        Self {
            brain,
            config: PpoConfig::default(),
            rollout: RolloutBuffer::new(),
            device,
            optimizer,
            episode_reward: 0.0,
            episode_steps: 0,
            episode_height_sum: 0.0,
            episode_upright_sum: 0.0,
            episode_count: 0,
            needs_reset: false,
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
        let n = self.rollout.len();
        if n == 0 {
            return PpoMetrics::default();
        }

        let device = &self.device;

        let last_t = &self.rollout.transitions[n - 1];
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

        let (advantages, returns) = compute_gae(
            &self.rollout,
            last_value,
            self.config.gamma,
            self.config.lambda,
        );

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

        let obs_data: Vec<f32> = self
            .rollout
            .transitions
            .iter()
            .flat_map(|t| t.obs.iter().copied())
            .collect();
        let actions_data: Vec<f32> = self
            .rollout
            .transitions
            .iter()
            .flat_map(|t| t.action.iter().copied())
            .collect();
        let old_log_probs_data: Vec<f32> = self
            .rollout
            .transitions
            .iter()
            .map(|t| t.log_prob)
            .collect();

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

/// Phase 1 reward: mean eye height. ONE signal — kept deliberately minimal so the
/// behaviour EMERGES rather than being hand-specified (owner's call: mechanical
/// terms like "feet on the ground" don't scale to complex emergent behaviour).
///
/// The eyes sit at the top of the kinematic chain (carapace → stalk → eye), so a
/// high eye reading needs the carapace both LEVEL (stalks point up) and HIGH (legs
/// extended underneath) — i.e. standing. Over a long (~25 s) episode the summed
/// return favours SUSTAINED height: a launch-and-crash spikes briefly then scores
/// low for the rest of the episode, while a steady stand scores high throughout.
/// The fall-over episode termination is the only structural guard; no shaped stance
/// terms, no drift/velocity/foot penalties.
fn compute_reward(mean_eye_height: f32) -> f32 {
    mean_eye_height
}

/// System: runs the brain to produce actions each physics step.
pub fn brain_step(
    mut training: NonSendMut<TrainingState>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    carapace_q: Query<(&Transform, &bevy_rapier3d::prelude::Velocity), With<CrabCarapace>>,
    eyes_q: Query<(&CrabJoint, &Transform)>,
) {
    let raw_obs = obs.values;
    let device = training.device;

    let obs_array = training.obs_normalizer.normalize(&raw_obs);

    let inference_brain = training.brain.valid();

    let obs_tensor = Tensor::<NdArray, 1>::from_floats(obs_array.as_slice(), &device);
    let obs_batch = obs_tensor.clone().unsqueeze::<2>();

    let (means_batch, log_std) = inference_brain.policy(obs_batch);
    let means: Tensor<NdArray, 1> = means_batch.flatten(0, 1);

    let obs_batch2 = obs_tensor.clone().unsqueeze::<2>();
    let value = inference_brain
        .value(obs_batch2)
        .flatten::<1>(0, 1)
        .into_scalar()
        .elem::<f32>();

    let action_tensor = sample_action(&means, &log_std, &device);
    let log_prob = compute_log_prob(&means, &log_std, &action_tensor);
    let log_prob = if log_prob.is_nan() || log_prob.is_infinite() {
        0.0
    } else {
        log_prob.clamp(-20.0, 20.0)
    };
    let value = if value.is_nan() || value.is_infinite() {
        0.0
    } else {
        value
    };

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
    actions.values = action_array;

    let (reward, done, height, upright) = if let Ok((transform, _vel)) = carapace_q.single() {
        let up = transform.rotation * Vec3::Y;
        let height = transform.translation.y;
        let upright = up.dot(Vec3::Y);
        // Mean world-space height of the eye tips — the entire reward signal.
        let mean_eye_height = {
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for (joint, eye) in eyes_q.iter() {
                if matches!(joint.id, CrabJointId::EyeStalk(_)) {
                    sum += eye.translation.y;
                    n += 1;
                }
            }
            if n > 0 { sum / n as f32 } else { transform.translation.y }
        };
        let r = compute_reward(mean_eye_height);
        // Long ~25 s episodes mean sustained eye height (a real stand, or a held
        // rear) outscores a brief launch-and-crash. End early only on a fall —
        // tipping past horizontal or leaving a sane height band — the one
        // structural guard, not a shaped reward term.
        let done = !(0.1..=5.0).contains(&height) || upright < 0.0 || training.episode_steps > 1500;
        (r, done, height, upright)
    } else {
        (0.0, false, 0.0, 0.0)
    };

    training.rollout.push(Transition {
        obs: obs_array,
        action: action_array,
        reward,
        value,
        log_prob,
        done,
    });

    training.episode_reward += reward;
    training.episode_steps += 1;
    training.episode_height_sum += height;
    training.episode_upright_sum += upright;
    training.current_rollout_steps += 1;
    training.total_steps += 1;

    if done {
        let ep_reward = training.episode_reward;
        let ep_steps = training.episode_steps;
        let ep_height = training.episode_height_sum / ep_steps.max(1) as f32;
        let ep_upright = training.episode_upright_sum / ep_steps.max(1) as f32;
        training.recent_rewards.push(ep_reward);
        training.episode_count += 1;
        training.needs_reset = true;

        let avg = training.avg_reward(10);
        let ep_count = training.episode_count;

        training
            .logger
            .log_episode(ep_count, ep_reward, ep_steps, avg, ep_height, ep_upright);

        if training.episode_count.is_multiple_of(10) {
            let elapsed = training.last_log_time.elapsed().as_secs_f32();
            let sps = if elapsed > 0.0 {
                training.total_steps as f32 / elapsed
            } else {
                0.0
            };
            info!(
                "Ep {} | Avg reward: {:.2} | Steps: {} | Height: {:.2} | Upright: {:.2} | Buffer: {} | {:.0} steps/s",
                training.episode_count,
                avg,
                ep_steps,
                ep_height,
                ep_upright,
                training.rollout.len(),
                sps,
            );
        }

        training.episode_reward = 0.0;
        training.episode_steps = 0;
        training.episode_height_sum = 0.0;
        training.episode_upright_sum = 0.0;
    }

    if training.current_rollout_steps >= training.steps_per_rollout {
        let buffer_size = training.rollout.len();
        let avg = training.avg_reward(10);

        info!("Running PPO update on {} transitions...", buffer_size);
        let metrics = training.ppo_update();

        info!(
            "PPO | Policy loss: {:.4} | Value loss: {:.4} | Entropy: {:.4}",
            metrics.policy_loss, metrics.value_loss, metrics.entropy,
        );

        training.logger.log_update(&metrics, avg, buffer_size);

        training.recent_metrics = Some(metrics);
        training.rollout.clear();
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

/// System: resets the crab when an episode ends.
pub fn reset_crab(
    mut training: NonSendMut<TrainingState>,
    mut actions: ResMut<CrabActions>,
    mut carapace_q: Query<
        (
            &mut Transform,
            &mut bevy_rapier3d::prelude::Velocity,
            &mut bevy_rapier3d::prelude::ExternalForce,
        ),
        With<CrabCarapace>,
    >,
    mut body_parts: Query<
        &mut bevy_rapier3d::prelude::Velocity,
        (With<CrabBodyPart>, Without<CrabCarapace>),
    >,
    mut joints: Query<(&CrabJoint, &mut bevy_rapier3d::prelude::MultibodyJoint)>,
) {
    if !training.needs_reset {
        return;
    }
    training.needs_reset = false;

    if let Ok((mut transform, mut vel, mut ext_force)) = carapace_q.single_mut() {
        transform.translation = Vec3::new(0.0, 1.0, 0.0);
        transform.rotation = Quat::IDENTITY;
        vel.linvel = Vec3::ZERO;
        vel.angvel = Vec3::ZERO;
        ext_force.force = Vec3::ZERO;
        ext_force.torque = Vec3::ZERO;
    }

    for mut vel in body_parts.iter_mut() {
        vel.linvel = Vec3::ZERO;
        vel.angvel = Vec3::ZERO;
    }

    for (crab_joint, mut mj) in joints.iter_mut() {
        let default_pos = crab_joint.id.default_position();
        let (stiffness, damping) = crab_joint.id.motor_stiffness_damping();
        let axis = crab_joint.id.joint_axis();
        let generic: &mut bevy_rapier3d::prelude::GenericJoint = mj.data.as_mut();
        generic.set_motor_position(axis, default_pos, stiffness, damping);
        generic.set_motor_max_force(axis, crab_joint.id.motor_max_force());
    }

    actions.values = [0.0; CrabJointId::COUNT];
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
        // The whole reward is mean eye height: higher eyes (standing/reared) must
        // score strictly above low eyes (collapsed/flat).
        assert!(
            compute_reward(1.2) > compute_reward(0.2),
            "reward must increase with eye height"
        );
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
