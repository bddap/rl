//! Proximal policy optimization of the [`Actor`] on the eval tasks, per
//! `docs/policy.md`: parallel episodes, one per slot, each in its own world.

use std::sync::{Arc, mpsc};

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use serde::Serialize;

use crate::eval::{Env, Episode, Task};
use crate::physics::PHYSICS_DT;
use crate::policy::{Actor, Anatomy, Mlp, critic, uniform};
use crate::sally::Skeleton;

/// *Grapsus* routine walking pace (Martinez et al. 1998, Table 1); a walk's reward is
/// its speed in these units.
const ROUTINE_PACE_M_S: f32 = 0.097;

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub seed: u64,
    pub slots: usize,
    pub rollout_ticks: usize,
    pub gamma: f32,
    pub lambda: f32,
    pub epochs: usize,
    pub minibatch: usize,
    pub learning_rate: f32,
    pub clip: f32,
    pub max_grad_norm: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            slots: 16,
            rollout_ticks: 256,
            gamma: 0.99,
            lambda: 0.95,
            epochs: 4,
            minibatch: 1024,
            learning_rate: 3e-4,
            clip: 0.2,
            max_grad_norm: 0.5,
        }
    }
}

enum End {
    Running,
    /// The task's length ran out: the critic values what would have followed.
    Cut(Vec<f32>),
    Diverged,
}

struct Sample {
    views: Vec<f32>,
    critic: Vec<f32>,
    targets: Vec<f32>,
    log_prob: f32,
    reward: f32,
    end: End,
}

struct Rollout {
    slot: usize,
    samples: Vec<Sample>,
    /// The critic input after the last sample, to bootstrap a running episode.
    tail: Vec<f32>,
    episodes: Vec<Episode>,
}

fn reward(task: Task, supported: bool, left_m: f32) -> f32 {
    match task {
        Task::Stand | Task::Righting => supported as u8 as f32,
        Task::WalkLeft | Task::WalkRight if supported => {
            task.walk_sign() * left_m / (ROUTINE_PACE_M_S * PHYSICS_DT)
        }
        Task::WalkLeft | Task::WalkRight => 0.0,
    }
}

/// Generalized advantage estimates for one slot's consecutive ticks, each its reward,
/// its value and, on an episode's last tick, the value of what would have followed.
fn gae(ticks: &[(f32, f32, Option<f32>)], tail: f32, gamma: f32, lambda: f32) -> Vec<f32> {
    let mut out = vec![0.0; ticks.len()];
    let (mut next_value, mut next_advantage) = (tail, 0.0);
    for (t, &(reward, value, end)) in ticks.iter().enumerate().rev() {
        let (bootstrap, continues) = end.map_or((next_value, 1.0), |v| (v, 0.0));
        out[t] = reward + gamma * bootstrap - value + gamma * lambda * continues * next_advantage;
        (next_value, next_advantage) = (value, out[t]);
    }
    out
}

fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    let u = 1.0 - uniform(rng);
    let v = uniform(rng);
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
}

fn log_prob(targets: &[f32], means: &[f32], log_std: &[f32]) -> f32 {
    let n = log_std.len();
    targets
        .iter()
        .zip(means)
        .enumerate()
        .map(|(i, (a, m))| {
            let s = log_std[i % n];
            let z = (a - m) / s.exp();
            -0.5 * z * z - s - 0.5 * (std::f32::consts::TAU).ln()
        })
        .sum()
}

fn draw_task(rng: &mut ChaCha8Rng) -> Task {
    Task::ALL[rng.next_u32() as usize % Task::ALL.len()]
}

/// One slot's episodes, stepped `rollout_ticks` per rollout under the actor it is
/// sent. A panic is reported as an error, so the learner never waits on a dead slot.
fn slot(
    index: usize,
    skeleton: &Skeleton,
    anatomy: &Anatomy,
    rollout_ticks: usize,
    rng: ChaCha8Rng,
    actors: mpsc::Receiver<Arc<Actor>>,
    rollouts: mpsc::Sender<Result<Rollout, String>>,
) {
    let run = std::panic::AssertUnwindSafe(|| {
        rollout_slot(
            index,
            skeleton,
            anatomy,
            rollout_ticks,
            rng,
            &actors,
            &rollouts,
        )
    });
    if let Err(panic) = std::panic::catch_unwind(run) {
        let why = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        let _ = rollouts.send(Err(format!("rollout slot {index} panicked: {why}")));
    }
}

fn rollout_slot(
    index: usize,
    skeleton: &Skeleton,
    anatomy: &Anatomy,
    rollout_ticks: usize,
    mut rng: ChaCha8Rng,
    actors: &mpsc::Receiver<Arc<Actor>>,
    rollouts: &mpsc::Sender<Result<Rollout, String>>,
) {
    let fresh = |rng: &mut ChaCha8Rng| Env::new(skeleton, draw_task(rng));
    let mut env = match fresh(&mut rng) {
        Ok(env) => env,
        Err(e) => {
            let _ = rollouts.send(Err(e));
            return;
        }
    };
    let mut action = vec![0.0; anatomy.hinges()];
    for actor in actors {
        let mut samples = Vec::with_capacity(rollout_ticks);
        let mut episodes = Vec::new();
        for _ in 0..rollout_ticks {
            let obs = env.observe();
            let views = anatomy.views(&obs);
            let means = actor.means(&views);
            let n = actor.log_std.len();
            let targets: Vec<f32> = means
                .iter()
                .enumerate()
                .map(|(i, m)| m + actor.log_std[i % n].exp() * gaussian(&mut rng))
                .collect();
            anatomy.scatter(&targets, &mut action);
            let tick = env.step(&action);
            let mut sample = Sample {
                log_prob: log_prob(&targets, &means, &actor.log_std),
                reward: reward(obs.task, tick.supported, tick.left_m),
                critic: anatomy.critic_input(&obs),
                views,
                targets,
                end: End::Running,
            };
            if env.done() {
                let last = env.observe();
                let replaced = match fresh(&mut rng) {
                    Ok(env) => env,
                    Err(e) => {
                        let _ = rollouts.send(Err(e));
                        return;
                    }
                };
                let episode = std::mem::replace(&mut env, replaced).finish();
                sample.end = if episode.finite {
                    End::Cut(anatomy.critic_input(&last))
                } else {
                    End::Diverged
                };
                episodes.push(episode);
            }
            samples.push(sample);
        }
        let tail = anatomy.critic_input(&env.observe());
        if rollouts
            .send(Ok(Rollout {
                slot: index,
                samples,
                tail,
                episodes,
            }))
            .is_err()
        {
            return;
        }
    }
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    fn step<'a>(&mut self, params: impl Iterator<Item = &'a mut f32>, grad: &[f32], lr: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        self.t += 1;
        let (c1, c2) = (1.0 - B1.powi(self.t), 1.0 - B2.powi(self.t));
        for (((p, g), m), v) in params.zip(grad).zip(&mut self.m).zip(&mut self.v) {
            *m = B1 * *m + (1.0 - B1) * g;
            *v = B2 * *v + (1.0 - B2) * g * g;
            *p -= lr * (*m / c1) / ((*v / c2).sqrt() + 1e-8);
        }
    }
}

fn clip_norm(grad: &mut [f32], max: f32) {
    let norm = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max {
        grad.iter_mut().for_each(|g| *g *= max / norm);
    }
}

/// What one iteration did, for the training log.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub iteration: usize,
    pub ticks: usize,
    pub mean_reward: f32,
    /// Every episode that ended this iteration, scored by its task's eval metric.
    pub episodes: Vec<Episode>,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub approx_kl: f32,
    pub clip_fraction: f32,
    pub std: Vec<f32>,
}

/// The PPO learner: the actor being trained, its critic and their optimizers.
pub struct Trainer<'a> {
    skeleton: &'a Skeleton,
    config: Config,
    anatomy: Anatomy,
    actor: Actor,
    critic: Mlp,
    actor_adam: Adam,
    critic_adam: Adam,
    rng: ChaCha8Rng,
}

impl<'a> Trainer<'a> {
    /// A seeded random actor and critic for `skeleton`.
    pub fn new(skeleton: &'a Skeleton, config: Config) -> Result<Self, String> {
        let anatomy = Anatomy::new(skeleton)?;
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        rng.set_stream(u64::MAX);
        let actor = Actor::new(&anatomy, &mut rng);
        let critic = critic(&anatomy, &mut rng);
        Ok(Self {
            actor_adam: Adam::new(actor.net.params.len() + actor.log_std.len()),
            critic_adam: Adam::new(critic.params.len()),
            skeleton,
            config,
            anatomy,
            actor,
            critic,
            rng,
        })
    }

    pub fn actor(&self) -> &Actor {
        &self.actor
    }

    fn value(&self, input: &[f32]) -> f32 {
        self.critic.forward(input).output()[0]
    }

    /// Trains for `iterations`, calling `report` after each with the actor as it
    /// stands.
    pub fn run(
        &mut self,
        iterations: usize,
        report: &mut dyn FnMut(&Report, &Actor) -> Result<(), String>,
    ) -> Result<(), String> {
        let (skeleton, config, anatomy) =
            (self.skeleton, self.config.clone(), self.anatomy.clone());
        std::thread::scope(|scope| {
            let (rollout_tx, rollout_rx) = mpsc::channel();
            let slots: Vec<mpsc::Sender<Arc<Actor>>> = (0..config.slots)
                .map(|s| {
                    let (tx, rx) = mpsc::channel();
                    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
                    rng.set_stream(s as u64);
                    let rollouts = rollout_tx.clone();
                    let anatomy = &anatomy;
                    scope.spawn(move || {
                        slot(
                            s,
                            skeleton,
                            anatomy,
                            config.rollout_ticks,
                            rng,
                            rx,
                            rollouts,
                        )
                    });
                    tx
                })
                .collect();
            drop(rollout_tx);
            let stopped = |rx: &mpsc::Receiver<Result<Rollout, String>>| {
                rx.try_iter()
                    .find_map(Result::err)
                    .unwrap_or_else(|| "a rollout slot stopped".into())
            };
            for iteration in 0..iterations {
                let actor = Arc::new(self.actor.clone());
                for tx in &slots {
                    tx.send(actor.clone()).map_err(|_| stopped(&rollout_rx))?;
                }
                let mut rollouts: Vec<Rollout> = (0..config.slots)
                    .map(|_| rollout_rx.recv().map_err(|_| stopped(&rollout_rx))?)
                    .collect::<Result<_, String>>()?;
                rollouts.sort_by_key(|r| r.slot);
                let r = self.update(iteration, rollouts);
                report(&r, &self.actor)?;
            }
            drop(slots);
            Ok(())
        })
    }

    fn update(&mut self, iteration: usize, rollouts: Vec<Rollout>) -> Report {
        let c = self.config.clone();
        let mut samples = Vec::new();
        let mut advantages = Vec::new();
        let mut returns = Vec::new();
        let mut episodes = Vec::new();
        for rollout in rollouts {
            let values: Vec<f32> = rollout
                .samples
                .iter()
                .map(|s| self.value(&s.critic))
                .collect();
            let ticks: Vec<(f32, f32, Option<f32>)> = rollout
                .samples
                .iter()
                .zip(&values)
                .map(|(s, &v)| {
                    let end = match &s.end {
                        End::Running => None,
                        End::Cut(last) => Some(self.value(last)),
                        End::Diverged => Some(0.0),
                    };
                    (s.reward, v, end)
                })
                .collect();
            let adv = gae(&ticks, self.value(&rollout.tail), c.gamma, c.lambda);
            returns.extend(adv.iter().zip(&values).map(|(a, v)| a + v));
            advantages.extend(adv);
            samples.extend(rollout.samples);
            episodes.extend(rollout.episodes);
        }
        let n = samples.len() as f32;
        let mean = advantages.iter().sum::<f32>() / n;
        let std = (advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n).sqrt();
        for a in &mut advantages {
            *a = (*a - mean) / (std + 1e-8);
        }

        let mut order: Vec<usize> = (0..samples.len()).collect();
        let (mut policy_loss, mut value_loss, mut kl, mut clipped, mut counted) =
            (0.0, 0.0, 0.0, 0.0, 0.0);
        let n_actor = self.actor.net.params.len();
        let hinges = self.actor.log_std.len();
        let view = self.actor.net.sizes[0];
        for _ in 0..c.epochs {
            for i in (1..order.len()).rev() {
                order.swap(i, self.rng.next_u32() as usize % (i + 1));
            }
            for batch in order.chunks(c.minibatch) {
                let mut actor_grad = vec![0.0; n_actor + hinges];
                let mut critic_grad = vec![0.0; self.critic.params.len()];
                let scale = 1.0 / batch.len() as f32;
                for &i in batch {
                    let s = &samples[i];
                    let traces: Vec<_> = s
                        .views
                        .chunks_exact(view)
                        .map(|v| self.actor.net.forward(v))
                        .collect();
                    let means: Vec<f32> = traces.iter().flat_map(|t| t.output().to_vec()).collect();
                    let lp = log_prob(&s.targets, &means, &self.actor.log_std);
                    let ratio = (lp - s.log_prob).exp();
                    let a = advantages[i];
                    let unclipped = ratio * a;
                    let bounded = ratio.clamp(1.0 - c.clip, 1.0 + c.clip) * a;
                    policy_loss -= unclipped.min(bounded);
                    kl += s.log_prob - lp;
                    let active = unclipped <= bounded;
                    clipped += (!active) as u8 as f32;
                    counted += 1.0;
                    if active {
                        let d_lp = -a * ratio * scale;
                        for (k, trace) in traces.iter().enumerate() {
                            let d_mean: Vec<f32> = (0..hinges)
                                .map(|j| {
                                    let var = (2.0 * self.actor.log_std[j]).exp();
                                    d_lp * (s.targets[k * hinges + j] - means[k * hinges + j]) / var
                                })
                                .collect();
                            self.actor
                                .net
                                .backward(trace, &d_mean, &mut actor_grad[..n_actor]);
                        }
                        for (idx, (t, m)) in s.targets.iter().zip(&means).enumerate() {
                            let j = idx % hinges;
                            let z2 = (t - m).powi(2) / (2.0 * self.actor.log_std[j]).exp();
                            actor_grad[n_actor + j] += d_lp * (z2 - 1.0);
                        }
                    }
                    let trace = self.critic.forward(&s.critic);
                    let err = trace.output()[0] - returns[i];
                    value_loss += 0.5 * err * err;
                    self.critic
                        .backward(&trace, &[err * scale], &mut critic_grad);
                }
                clip_norm(&mut actor_grad, c.max_grad_norm);
                clip_norm(&mut critic_grad, c.max_grad_norm);
                let Actor { net, log_std } = &mut self.actor;
                self.actor_adam.step(
                    net.params.iter_mut().chain(log_std.iter_mut()),
                    &actor_grad,
                    c.learning_rate,
                );
                self.critic_adam
                    .step(self.critic.params.iter_mut(), &critic_grad, c.learning_rate);
            }
        }
        Report {
            iteration,
            ticks: samples.len(),
            mean_reward: samples.iter().map(|s| s.reward).sum::<f32>() / n,
            episodes,
            policy_loss: policy_loss / counted,
            value_loss: value_loss / counted,
            approx_kl: kl / counted,
            clip_fraction: clipped / counted,
            std: self.actor.log_std.iter().map(|s| s.exp()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sally::test_rig::biped;

    fn small(seed: u64) -> Config {
        Config {
            seed,
            slots: 3,
            rollout_ticks: 48,
            minibatch: 32,
            ..Config::default()
        }
    }

    fn trained(seed: u64, iterations: usize) -> (Actor, Vec<Report>) {
        let skeleton = biped();
        let mut trainer = Trainer::new(&skeleton, small(seed)).unwrap();
        let mut reports = Vec::new();
        trainer
            .run(iterations, &mut |r, _| {
                reports.push(r.clone());
                Ok(())
            })
            .unwrap();
        (trainer.actor, reports)
    }

    #[test]
    fn a_run_replays_exactly_from_its_seed() {
        let (a, reports) = trained(5, 2);
        let (b, _) = trained(5, 2);
        let (c, _) = trained(6, 2);
        assert_eq!(a.net.params, b.net.params);
        assert_eq!(a.log_std, b.log_std);
        assert_ne!(a.net.params, c.net.params);
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|r| r.ticks == 3 * 48));
    }

    #[test]
    fn an_update_moves_the_mean_toward_the_rewarded_action() {
        let skeleton = biped();
        let mut trainer = Trainer::new(&skeleton, small(3)).unwrap();
        let obs = Env::new(&skeleton, Task::Stand).unwrap().observe();
        let views = trainer.anatomy.views(&obs);
        let critic = trainer.anatomy.critic_input(&obs);
        let means = trainer.actor.means(&views);
        let sample = |shift: f32, reward: f32| {
            let targets: Vec<f32> = means.iter().map(|m| m + shift).collect();
            Sample {
                log_prob: log_prob(&targets, &means, &trainer.actor.log_std),
                views: views.clone(),
                critic: critic.clone(),
                targets,
                reward,
                end: End::Diverged,
            }
        };
        let rollout = Rollout {
            slot: 0,
            samples: vec![sample(0.1, 1.0), sample(-0.1, 0.0)],
            tail: critic.clone(),
            episodes: Vec::new(),
        };
        let value_before = trainer.value(&critic);
        trainer.update(0, vec![rollout]);
        let after = trainer.actor.means(&views);
        assert!(
            after.iter().zip(&means).all(|(a, m)| a > m),
            "{means:?} -> {after:?}"
        );
        assert!(trainer.value(&critic) > value_before);
    }

    #[test]
    fn advantages_stop_at_an_episode_s_end() {
        let (g, l) = (0.9, 0.5);
        let ticks = [(1.0, 0.5, None), (2.0, 0.25, Some(4.0)), (3.0, 1.0, None)];
        let adv = gae(&ticks, 2.0, g, l);
        let d2 = 3.0 + g * 2.0 - 1.0;
        let d1 = 2.0 + g * 4.0 - 0.25;
        let d0 = 1.0 + g * 0.25 - 0.5;
        let want = [d0 + g * l * d1, d1, d2];
        assert!(
            adv.iter().zip(want).all(|(a, w)| (a - w).abs() < 1e-6),
            "{adv:?} vs {want:?}"
        );
    }

    #[test]
    fn a_walk_pays_its_speed_in_routine_paces_only_while_supported() {
        let pace = ROUTINE_PACE_M_S * PHYSICS_DT;
        assert!((reward(Task::WalkLeft, true, pace) - 1.0).abs() < 1e-6);
        assert!((reward(Task::WalkRight, true, pace) + 1.0).abs() < 1e-6);
        assert_eq!(reward(Task::WalkLeft, false, pace), 0.0);
        assert_eq!(reward(Task::Stand, true, pace), 1.0);
        assert_eq!(reward(Task::Righting, false, 0.0), 0.0);
    }

    #[test]
    fn log_prob_is_the_gaussian_density() {
        let stds = [0.5f32, 2.0];
        let (targets, means) = ([0.5, -1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 1.0]);
        let want: f32 = targets
            .iter()
            .zip(means)
            .enumerate()
            .map(|(i, (t, m))| {
                let s = stds[i % 2];
                (-(t - m) * (t - m) / (2.0 * s * s)).exp() / (s * std::f32::consts::TAU.sqrt())
            })
            .map(f32::ln)
            .sum();
        let got = log_prob(&targets, &means, &stds.map(f32::ln));
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
}
