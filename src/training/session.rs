//! Training session management.
//!
//! Manages the RL training loop integrated with the Bevy game loop.

use std::io::Write;

use bevy::prelude::*;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::grad_clipping::GradientClippingConfig;
use burn::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabJoint, CrabJointId};
use crate::bot::brain::{CrabBrain, ACTION_SIZE};
use crate::bot::sensor::{CrabObservation, OBS_SIZE};

use super::algorithm::{
    compute_log_prob, sample_action, PpoConfig, PpoMetrics, RolloutBuffer, Transition,
    compute_gae,
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

        let mut episode_file = std::fs::File::create("tmp/episodes.csv")
            .expect("failed to create tmp/episodes.csv");
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
        if episode % 10 == 0 {
            self.episode_file.flush().ok();
        }
    }

    fn log_update(
        &mut self,
        metrics: &PpoMetrics,
        avg_reward: f32,
        buffer_size: usize,
    ) {
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
            steps_per_rollout: 1024,
            current_rollout_steps: 0,
            recent_rewards: Vec::new(),
            recent_metrics: None,
            logger: MetricsLogger::new(),
            total_steps: 0,
            last_log_time: std::time::Instant::now(),
        }
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
            self.brain.value(obs).flatten::<1>(0, 1).into_scalar().elem::<f32>()
        };

        let (advantages, returns) =
            compute_gae(&self.rollout, last_value, self.config.gamma, self.config.lambda);

        // Normalize advantages
        let adv_mean: f32 = advantages.iter().sum::<f32>() / n as f32;
        let adv_var: f32 =
            advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>() / n as f32;
        let adv_std = adv_var.sqrt().max(1e-8);
        let advantages_norm: Vec<f32> = advantages.iter().map(|a| (a - adv_mean) / adv_std).collect();

        // Build tensors
        let obs_data: Vec<f32> = self.rollout.transitions.iter()
            .flat_map(|t| t.obs.iter().copied()).collect();
        let actions_data: Vec<f32> = self.rollout.transitions.iter()
            .flat_map(|t| t.action.iter().copied()).collect();
        let old_log_probs_data: Vec<f32> = self.rollout.transitions.iter()
            .map(|t| t.log_prob).collect();

        let obs_all = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(obs_data, [n, OBS_SIZE]), device);
        let actions_all = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(actions_data, [n, ACTION_SIZE]), device);
        let old_log_probs_all = Tensor::<TrainBackend, 1>::from_data(
            TensorData::new(old_log_probs_data, [n]), device);
        let advantages_all = Tensor::<TrainBackend, 1>::from_data(
            TensorData::new(advantages_norm, [n]), device);
        let returns_all = Tensor::<TrainBackend, 1>::from_data(
            TensorData::new(returns, [n]), device);

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut update_count = 0u32;

        let bs = self.config.batch_size;
        let half_log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();

        for _epoch in 0..self.config.epochs_per_update {
            let num_batches = n.div_ceil(bs);

            for batch_idx in 0..num_batches {
                let start = batch_idx * bs;
                let end = (start + bs).min(n);
                let batch_n = end - start;

                let obs = obs_all.clone().slice([start..end, 0..OBS_SIZE]);
                let actions = actions_all.clone().slice([start..end, 0..ACTION_SIZE]);
                let old_lp = old_log_probs_all.clone().slice([start..end]);
                let advs = advantages_all.clone().slice([start..end]);
                let rets = returns_all.clone().slice([start..end]);

                // Policy forward
                let (means, log_std) = self.brain.policy(obs.clone());

                // Clamp log_std for numerical stability: exp(-5)≈0.007, exp(2)≈7.4
                let log_std_clamped = log_std.clone().clamp(-5.0, 2.0);

                // Gaussian log-prob: -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5*log(2*pi)
                // Compute in log-space to avoid division by tiny variance
                let diff = actions - means; // [batch, ACTION_SIZE]
                let log_std_2d = log_std_clamped.clone()
                    .unsqueeze_dim::<2>(0)
                    .expand([batch_n, ACTION_SIZE]);
                let scaled_diff = diff / log_std_2d.clone().exp(); // (a - mu) / sigma
                let log_probs_per_dim = scaled_diff.powf_scalar(2.0).neg() * 0.5
                    - log_std_2d
                    - half_log_2pi;
                let new_lp: Tensor<TrainBackend, 1> = log_probs_per_dim.sum_dim(1).flatten(0, 1);

                // Entropy: 0.5 * log(2*pi*e) + log_std, summed over dims
                let entropy_per_dim = log_std_clamped.clone() + (0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln());
                let entropy = entropy_per_dim.sum();

                // PPO clipped objective
                // Clamp log-ratio to prevent exp() overflow
                let log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0);
                let ratio = log_ratio.exp();
                let surr1 = ratio.clone() * advs.clone();
                let surr2 = ratio
                    .clamp(1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
                    * advs;
                let policy_loss = surr1.min_pair(surr2).mean().neg();

                // Value loss (clipped to prevent huge gradients)
                let values: Tensor<TrainBackend, 1> = self.brain.value(obs).flatten(0, 1);
                let value_diff = (values - rets).clamp(
                    -self.config.value_loss_clip,
                    self.config.value_loss_clip,
                );
                let value_loss = value_diff.powf_scalar(2.0).mean();

                // Total loss
                let loss = policy_loss.clone()
                    + value_loss.clone() * self.config.value_coeff
                    - entropy.clone() * self.config.entropy_coeff;

                total_policy_loss += policy_loss.clone().into_scalar().elem::<f32>();
                total_value_loss += value_loss.clone().into_scalar().elem::<f32>();
                total_entropy += entropy.clone().into_scalar().elem::<f32>();
                update_count += 1;

                // Backward + step (optimizer applies gradient clipping)
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.brain);
                self.brain = self.optimizer.step(self.config.learning_rate, self.brain.clone(), grads);
            }
        }

        PpoMetrics {
            policy_loss: total_policy_loss / update_count as f32,
            value_loss: total_value_loss / update_count as f32,
            entropy: total_entropy / update_count as f32,
        }
    }
}

/// Phase 1 reward: learn to stand.
fn compute_reward(carapace_pos: Vec3, carapace_up: Vec3) -> f32 {
    let mut reward = 0.0f32;

    let height = carapace_pos.y;
    if height > 0.3 {
        reward += 1.0;
        reward += (1.0 - (height - 0.5).abs()).max(0.0) * 0.5;
    } else {
        reward -= 1.0;
    }

    let uprightness = carapace_up.dot(Vec3::Y);
    reward += uprightness * 0.5;
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
    let obs_array = obs.values;
    let device = training.device;

    let obs_tensor = Tensor::<TrainBackend, 1>::from_floats(obs_array.as_slice(), &device);
    let obs_batch = obs_tensor.clone().unsqueeze::<2>();

    // Policy forward
    let (means_batch, log_std) = training.brain.policy(obs_batch);
    let means: Tensor<TrainBackend, 1> = means_batch.flatten(0, 1);

    // Value forward
    let obs_batch2 = obs_tensor.clone().unsqueeze::<2>();
    let value = training.brain.value(obs_batch2)
        .flatten::<1>(0, 1)
        .into_scalar()
        .elem::<f32>();

    // Sample action
    let action_tensor = sample_action(&means, &log_std, &device);
    let log_prob = compute_log_prob(&means, &log_std, &action_tensor);
    // Clamp log_prob to prevent extreme values that cause ratio explosion in PPO
    let log_prob = if log_prob.is_nan() || log_prob.is_infinite() {
        0.0
    } else {
        log_prob.clamp(-20.0, 2.0)
    };
    let value = if value.is_nan() || value.is_infinite() { 0.0 } else { value };

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
    let (reward, done) = if let Ok((transform, _vel)) = carapace_q.single() {
        let up = transform.rotation * Vec3::Y;
        let r = compute_reward(transform.translation, up);
        let done = transform.translation.y < 0.1
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

        let avg = training.avg_reward(10);
        let ep_count = training.episode_count;

        // Log every episode to CSV
        training.logger.log_episode(ep_count, ep_reward, ep_steps, avg);

        if training.episode_count % 10 == 0 {
            let elapsed = training.last_log_time.elapsed().as_secs_f32();
            let sps = if elapsed > 0.0 {
                training.total_steps as f32 / elapsed
            } else {
                0.0
            };
            info!(
                "Ep {} | Avg reward: {:.2} | Steps: {} | Buffer: {} | {:.0} steps/s",
                training.episode_count, avg, ep_steps, training.rollout.len(), sps,
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
    }
}

/// System: resets the crab when an episode ends.
/// Resets carapace position/rotation, all body part velocities,
/// and drives all joints back to their default motor positions.
pub fn reset_crab(
    training: NonSend<TrainingState>,
    mut actions: ResMut<CrabActions>,
    mut carapace_q: Query<
        (&mut Transform, &mut bevy_rapier3d::prelude::Velocity),
        With<CrabCarapace>,
    >,
    mut body_parts: Query<
        &mut bevy_rapier3d::prelude::Velocity,
        (With<CrabBodyPart>, Without<CrabCarapace>),
    >,
    mut joints: Query<(&CrabJoint, &mut bevy_rapier3d::prelude::MultibodyJoint)>,
) {
    if let Some(last) = training.rollout.transitions.last() {
        if last.done {
            // Reset carapace position and velocity
            if let Ok((mut transform, mut vel)) = carapace_q.single_mut() {
                transform.translation = Vec3::new(0.0, 1.0, 0.0);
                transform.rotation = Quat::IDENTITY;
                vel.linvel = Vec3::ZERO;
                vel.angvel = Vec3::ZERO;
            }

            // Zero velocities on all child body parts
            for mut vel in body_parts.iter_mut() {
                vel.linvel = Vec3::ZERO;
                vel.angvel = Vec3::ZERO;
            }

            // Reset joint motors to default positions
            for (crab_joint, mut mj) in joints.iter_mut() {
                let default_pos = crab_joint.id.default_position();
                let generic: &mut bevy_rapier3d::prelude::GenericJoint = mj.data.as_mut();

                let axis = match &crab_joint.id {
                    crate::bot::body::CrabJointId::ClawPincer(_) => {
                        bevy_rapier3d::prelude::JointAxis::LinX
                    }
                    _ => bevy_rapier3d::prelude::JointAxis::AngX,
                };

                // Use the same stiffness/damping as initial spawn
                let (stiffness, damping) = match &crab_joint.id {
                    crate::bot::body::CrabJointId::EyeStalk(_) => (25.0, 5.0),
                    crate::bot::body::CrabJointId::ClawUpper(_)
                    | crate::bot::body::CrabJointId::ClawFore(_)
                    | crate::bot::body::CrabJointId::ClawPincer(_) => (250.0, 25.0),
                    _ => (200.0, 20.0), // legs
                };

                generic.set_motor_position(axis, default_pos, stiffness, damping);
            }

            // Reset action vector to zero
            actions.values = [0.0; CrabJointId::COUNT];
        }
    }
}
