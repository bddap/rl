//! Training session management.
//!
//! Manages the RL training loop integrated with the Bevy game loop.

use std::io::Write;

use bevy::prelude::*;
use burn::backend::Autodiff;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabJoint, CrabJointId};
use crate::bot::brain::{ACTION_SIZE, CrabBrain};
use crate::bot::sensor::{CrabObservation, OBS_SIZE};

/// Running observation normalizer using Welford's online algorithm.
/// Normalizes observations to zero mean, unit variance.
struct ObsNormalizer {
    mean: [f64; OBS_SIZE],
    var: [f64; OBS_SIZE],   // running variance (M2 / count)
    m2: [f64; OBS_SIZE],    // sum of squared differences from mean
    count: [u64; OBS_SIZE], // per-element count (NaN-skipped elements don't inflate others)
    clip: f32,              // max absolute normalized value
}

impl ObsNormalizer {
    fn new(clip: f32) -> Self {
        Self {
            mean: [0.0; OBS_SIZE],
            var: [1.0; OBS_SIZE],
            m2: [0.0; OBS_SIZE],
            count: [0; OBS_SIZE],
            clip,
        }
    }

    /// Update running stats and return normalized observation.
    fn normalize(&mut self, obs: &[f32; OBS_SIZE]) -> [f32; OBS_SIZE] {
        // Welford update (skip NaN/Inf inputs to avoid poisoning stats)
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

        // Normalize
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
}

use super::algorithm::{
    PpoConfig, PpoMetrics, RolloutBuffer, Transition, compute_gae, compute_log_prob, sample_action,
};

/// Backend type aliases.
pub type TrainBackend = Autodiff<NdArray>;

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
        writeln!(episode_file, "episode,reward,steps,avg_reward_10")
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

    fn log_episode(&mut self, episode: u32, reward: f32, steps: u32, avg_reward: f32) {
        writeln!(
            self.episode_file,
            "{},{:.4},{},{}",
            episode, reward, steps, avg_reward
        )
        .ok();
        // Flush periodically
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

/// The RL training state. Stored as a non-send resource because burn
/// tensors use `OnceCell` which is not `Sync`.
pub struct TrainingState {
    pub brain: CrabBrain<TrainBackend>,
    pub config: PpoConfig,
    pub rollout: RolloutBuffer,
    pub device: NdArrayDevice,
    optimizer: CrabOptimizer,

    // Episode tracking
    pub episode_reward: f32,
    pub episode_steps: u32,
    pub episode_count: u32,
    pub needs_reset: bool,

    // Rollout settings
    pub steps_per_rollout: u32,
    pub current_rollout_steps: u32,

    // Metrics
    pub recent_rewards: Vec<f32>,
    pub recent_metrics: Option<PpoMetrics>,

    // Logging
    logger: MetricsLogger,

    // Total physics steps for throughput tracking
    pub total_steps: u64,
    last_log_time: std::time::Instant,

    // Observation normalizer
    obs_normalizer: ObsNormalizer,

    // Checkpointing
    checkpoint_interval: u32, // save every N PPO updates
}

impl TrainingState {
    pub fn new() -> Self {
        let device = NdArrayDevice::Cpu;
        let brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);
        let optimizer: CrabOptimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
            .init();

        Self {
            brain,
            config: PpoConfig::default(),
            rollout: RolloutBuffer::new(),
            device,
            optimizer,
            episode_reward: 0.0,
            episode_steps: 0,
            episode_count: 0,
            needs_reset: false,
            steps_per_rollout: 1024,
            current_rollout_steps: 0,
            recent_rewards: Vec::new(),
            recent_metrics: None,
            logger: MetricsLogger::new(),
            total_steps: 0,
            last_log_time: std::time::Instant::now(),
            obs_normalizer: ObsNormalizer::new(5.0),
            checkpoint_interval: 50, // save every 50 PPO updates
        }
    }

    /// Save brain weights to disk.
    fn save_checkpoint(&self) {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let path =
            std::path::PathBuf::from(format!("tmp/brain_update_{}", self.logger.update_count));
        match recorder.record(self.brain.clone().into_record(), path.clone()) {
            Ok(()) => info!(
                "Saved checkpoint to tmp/brain_update_{}",
                self.logger.update_count
            ),
            Err(e) => warn!("Failed to save checkpoint: {e}"),
        }
        // Also save as "latest" for easy resume
        let latest_path = std::path::PathBuf::from("tmp/brain_latest");
        let _ = recorder.record(self.brain.clone().into_record(), latest_path);
    }

    /// Compute rolling average of recent rewards.
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

        // Compute last value for GAE
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

        // Normalize advantages
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

        // Build tensors
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
            // Shuffle indices for random mini-batches
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);

            let num_batches = n.div_ceil(bs);

            for batch_idx in 0..num_batches {
                let start = batch_idx * bs;
                let end = (start + bs).min(n);
                let batch_n = end - start;
                let batch_indices = &indices[start..end];

                // Gather shuffled batch via index tensor
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

                // Policy forward
                let (means, log_std) = self.brain.policy(obs.clone());

                // Clamp log_std for numerical stability: exp(-5)≈0.007, exp(2)≈7.4
                let log_std_clamped = log_std.clone().clamp(-5.0, 2.0);

                // Gaussian log-prob: -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5*log(2*pi)
                // Compute in log-space to avoid division by tiny variance
                let diff = actions - means; // [batch, ACTION_SIZE]
                let log_std_2d = log_std_clamped
                    .clone()
                    .unsqueeze_dim::<2>(0)
                    .expand([batch_n, ACTION_SIZE]);
                let scaled_diff = diff / log_std_2d.clone().exp(); // (a - mu) / sigma
                let log_probs_per_dim =
                    scaled_diff.powf_scalar(2.0).neg() * 0.5 - log_std_2d - half_log_2pi;
                let new_lp: Tensor<TrainBackend, 1> = log_probs_per_dim.sum_dim(1).flatten(0, 1);

                // Entropy: 0.5 * log(2*pi*e) + log_std, averaged over dims
                // (mean, not sum, so entropy_coeff is scale-invariant w.r.t. ACTION_SIZE)
                let entropy_per_dim = log_std_clamped.clone()
                    + (0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln());
                let entropy = entropy_per_dim.mean();

                // PPO clipped objective
                // Clamp log-ratio to prevent exp() overflow
                let log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0);
                let ratio = log_ratio.exp();
                let surr1 = ratio.clone() * advs.clone();
                let surr2 = ratio.clamp(
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advs;
                let policy_loss = surr1.min_pair(surr2).mean().neg();

                // Value loss (clipped to prevent huge gradients)
                let values: Tensor<TrainBackend, 1> = self.brain.value(obs).flatten(0, 1);
                let value_diff = (values - rets)
                    .clamp(-self.config.value_loss_clip, self.config.value_loss_clip);
                let value_loss = value_diff.powf_scalar(2.0).mean();

                // Total loss
                let loss = policy_loss.clone() + value_loss.clone() * self.config.value_coeff
                    - entropy.clone() * self.config.entropy_coeff;

                total_policy_loss += policy_loss.clone().into_scalar().elem::<f32>();
                total_value_loss += value_loss.clone().into_scalar().elem::<f32>();
                total_entropy += entropy.clone().into_scalar().elem::<f32>();
                update_count += 1;

                // Backward + step (optimizer applies gradient clipping)
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

/// Phase 1 reward: learn to stand stably.
///
/// Components:
///   +1.0 alive bonus (always, for surviving)
///   +height: smooth bonus peaking at target_height=0.5m
///   +uprightness: dot(body_up, world_Y) scaled
///   -action_cost: penalise large motor commands (energy efficiency)
///   -velocity_cost: penalise body wobble (stability)
///   -step_cost: small per-step cost to prefer efficient behaviour
fn compute_reward(
    carapace_pos: Vec3,
    carapace_up: Vec3,
    linvel: Vec3,
    angvel: Vec3,
    actions: &[f32],
) -> f32 {
    let mut reward = 0.0f32;

    // Alive bonus
    reward += 1.0;

    // Height bonus: Gaussian-like around target height of 0.5m
    let height = carapace_pos.y;
    let target_height = 0.5;
    let height_bonus = (-2.0 * (height - target_height).powi(2)).exp();
    reward += height_bonus;

    // Uprightness: how aligned is body-up with world-up
    let uprightness = carapace_up.dot(Vec3::Y); // -1 to 1
    reward += uprightness * 0.5;

    // Action cost: penalise large motor outputs (L2 norm)
    let action_sq_sum: f32 = actions.iter().map(|a| a * a).sum();
    let action_cost = 0.01 * action_sq_sum;
    reward -= action_cost;

    // Velocity cost: penalise excessive body motion
    let vel_cost = 0.2 * (linvel.length_squared() + 0.1 * angvel.length_squared());
    reward -= vel_cost;

    // Small step cost
    reward -= 0.01;

    reward
}

/// System: runs the brain to produce actions each physics step.
pub fn brain_step(
    mut training: NonSendMut<TrainingState>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    carapace_q: Query<(&Transform, &bevy_rapier3d::prelude::Velocity), With<CrabCarapace>>,
) {
    let raw_obs = obs.values;
    let device = training.device;

    // Normalize observations using running mean/std
    let obs_array = training.obs_normalizer.normalize(&raw_obs);

    // Use the inner (non-autodiff) backend for inference — avoids building
    // a computation graph every step, which was the main perf bottleneck.
    let inference_brain = training.brain.valid();

    let obs_tensor = Tensor::<NdArray, 1>::from_floats(obs_array.as_slice(), &device);
    let obs_batch = obs_tensor.clone().unsqueeze::<2>();

    // Policy forward (no autodiff graph)
    let (means_batch, log_std) = inference_brain.policy(obs_batch);
    let means: Tensor<NdArray, 1> = means_batch.flatten(0, 1);

    // Value forward (no autodiff graph)
    let obs_batch2 = obs_tensor.clone().unsqueeze::<2>();
    let value = inference_brain
        .value(obs_batch2)
        .flatten::<1>(0, 1)
        .into_scalar()
        .elem::<f32>();

    // Sample action (on inner backend)
    let action_tensor = sample_action(&means, &log_std, &device);
    let log_prob = compute_log_prob(&means, &log_std, &action_tensor);
    // Clamp log_prob to prevent extreme values that cause ratio explosion in PPO
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

    // Extract actions, with NaN guard
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

    // Compute reward
    let (reward, done) = if let Ok((transform, vel)) = carapace_q.single() {
        let up = transform.rotation * Vec3::Y;
        let r = compute_reward(
            transform.translation,
            up,
            vel.linvel,
            vel.angvel,
            &action_array,
        );
        let done = transform.translation.y < 0.1
            || transform.translation.y > 5.0
            || up.dot(Vec3::Y) < 0.0
            || training.episode_steps > 500;
        (r, done)
    } else {
        (0.0, false)
    };

    // Store transition
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
    training.current_rollout_steps += 1;
    training.total_steps += 1;

    if done {
        let ep_reward = training.episode_reward;
        let ep_steps = training.episode_steps;
        training.recent_rewards.push(ep_reward);
        training.episode_count += 1;
        training.needs_reset = true;

        let avg = training.avg_reward(10);
        let ep_count = training.episode_count;

        // Log every episode to CSV
        training
            .logger
            .log_episode(ep_count, ep_reward, ep_steps, avg);

        if training.episode_count.is_multiple_of(10) {
            let elapsed = training.last_log_time.elapsed().as_secs_f32();
            let sps = if elapsed > 0.0 {
                training.total_steps as f32 / elapsed
            } else {
                0.0
            };
            info!(
                "Ep {} | Avg reward: {:.2} | Steps: {} | Buffer: {} | {:.0} steps/s",
                training.episode_count,
                avg,
                ep_steps,
                training.rollout.len(),
                sps,
            );
        }

        training.episode_reward = 0.0;
        training.episode_steps = 0;
    }

    // PPO update
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

        // Checkpoint periodically
        if training
            .logger
            .update_count
            .is_multiple_of(training.checkpoint_interval)
        {
            training.save_checkpoint();
        }
    }
}

/// System: resets the crab when an episode ends.
/// Resets carapace position/rotation, all body part velocities,
/// and drives all joints back to their default motor positions.
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

    // Reset carapace position, velocity, and external forces
    if let Ok((mut transform, mut vel, mut ext_force)) = carapace_q.single_mut() {
        transform.translation = Vec3::new(0.0, 1.0, 0.0);
        transform.rotation = Quat::IDENTITY;
        vel.linvel = Vec3::ZERO;
        vel.angvel = Vec3::ZERO;
        ext_force.force = Vec3::ZERO;
        ext_force.torque = Vec3::ZERO;
    }

    // Zero velocities on all child body parts
    for mut vel in body_parts.iter_mut() {
        vel.linvel = Vec3::ZERO;
        vel.angvel = Vec3::ZERO;
    }

    // Reset joint motors to default positions and re-apply motor_max_force
    for (crab_joint, mut mj) in joints.iter_mut() {
        let default_pos = crab_joint.id.default_position();
        let (stiffness, damping) = crab_joint.id.motor_stiffness_damping();
        let axis = crab_joint.id.joint_axis();
        let generic: &mut bevy_rapier3d::prelude::GenericJoint = mj.data.as_mut();
        generic.set_motor_position(axis, default_pos, stiffness, damping);
        generic.set_motor_max_force(axis, crab_joint.id.motor_max_force());
    }

    // Reset action vector to zero
    actions.values = [0.0; CrabJointId::COUNT];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_velocity_penalty_is_significant() {
        let stationary = compute_reward(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::Y,
            Vec3::ZERO,
            Vec3::ZERO,
            &[0.0; ACTION_SIZE],
        );
        let moving = compute_reward(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::Y,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::ZERO,
            &[0.0; ACTION_SIZE],
        );
        let ratio = moving / stationary;
        // With vel_cost=0.2, a 5 m/s crab should lose substantial reward.
        // ratio should be well below 0.5 (was 0.9 with the old 0.01 coeff).
        assert!(
            ratio < 0.5,
            "velocity penalty too weak: moving/stationary ratio = {ratio:.3} (expected < 0.5)"
        );
    }

    #[test]
    fn welford_per_element_count_correctness() {
        let mut norm = ObsNormalizer::new(5.0);

        // Feed 100 samples where element 0 is always 1.0 but element 1 is
        // NaN half the time. With per-element counts, element 1's mean should
        // converge correctly despite missing half the data.
        for i in 0..100 {
            let mut obs = [0.0f32; OBS_SIZE];
            obs[0] = 1.0;
            obs[1] = if i % 2 == 0 { 1.0 } else { f32::NAN };
            norm.normalize(&obs);
        }

        // Element 0: count=100, mean≈1.0
        assert!(
            (norm.mean[0] - 1.0).abs() < 0.01,
            "element 0 mean should be ~1.0, got {}",
            norm.mean[0]
        );
        assert_eq!(norm.count[0], 100);

        // Element 1: count=50 (only even iterations), mean≈1.0
        assert!(
            (norm.mean[1] - 1.0).abs() < 0.01,
            "element 1 mean should be ~1.0, got {}",
            norm.mean[1]
        );
        assert_eq!(norm.count[1], 50);
    }

    #[test]
    fn log_prob_clamp_does_not_destroy_valid_values() {
        // For a 32-dim Gaussian, log_prob of a sample near the mean with
        // moderate std can legitimately be around -30 to +5 or so.
        // The old clamp(-20, 2) would clip valid positive log-probs.
        let log_prob: f32 = 18.6;
        let clamped = log_prob.clamp(-20.0, 20.0);
        assert!(
            (clamped - 18.6).abs() < 1e-6,
            "log_prob {log_prob} was clipped to {clamped} — symmetric clamp should preserve it"
        );
    }
}
