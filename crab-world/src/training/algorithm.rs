use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::bot::actuator::ACTION_SIZE;
use crate::bot::sensor::OBS_SIZE;

pub(crate) struct PpoConfig {
    pub(crate) gamma: f32,
    pub(crate) lambda: f32,
    pub(crate) clip_epsilon: f32,
    pub(crate) entropy_coeff: f32,
    pub(crate) value_coeff: f32,
    pub(crate) learning_rate: f64,
    pub(crate) epochs_per_update: u32,
    pub(crate) batch_size: usize,
    pub(crate) value_loss_clip: f32,
    pub(crate) target_kl: f32,
    /// Hard cap on minibatch steps per update; `None` = uncapped (rl#276).
    pub(crate) steps_cap: Option<std::num::NonZeroU32>,
    pub(crate) log_std_floor_start: f32,
    pub(crate) log_std_floor_end: f32,
    pub(crate) log_std_anneal_ticks: u64,
}

impl PpoConfig {
    pub(crate) fn log_std_floor(&self, ticks_into_anneal: u64) -> f32 {
        if self.log_std_anneal_ticks == 0 {
            return self.log_std_floor_end;
        }
        let frac = (ticks_into_anneal as f32 / self.log_std_anneal_ticks as f32).clamp(0.0, 1.0);
        self.log_std_floor_start + (self.log_std_floor_end - self.log_std_floor_start) * frac
    }
}

fn env_or<T: std::str::FromStr>(var: &str, default: T) -> T {
    match std::env::var(var) {
        Ok(s) => s.trim().parse().unwrap_or_else(|_| {
            eprintln!("[config] {var}={s:?} did not parse; using default");
            default
        }),
        Err(_) => default,
    }
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coeff: 0.001,
            value_coeff: 0.5,
            learning_rate: 3e-4,
            epochs_per_update: 4,
            batch_size: 64,
            value_loss_clip: 3.0,
            target_kl: 0.03,
            steps_cap: None,
            log_std_floor_start: env_or("RL_LOG_STD_FLOOR_START", -0.7),
            log_std_floor_end: env_or("RL_LOG_STD_FLOOR_END", crate::bot::arch::LOG_STD_MIN),
            log_std_anneal_ticks: env_or("RL_LOG_STD_ANNEAL_TICKS", 5_000_000),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StepEnd {
    Continues,
    Terminal,
    Truncated,
}

impl StepEnd {
    pub(crate) fn ends_segment(self) -> bool {
        matches!(self, StepEnd::Terminal | StepEnd::Truncated)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct NormalizedValue(pub(crate) f32);

#[derive(Clone, Copy)]
pub(crate) struct RealReturn(pub(crate) f32);

impl std::ops::Add for RealReturn {
    type Output = RealReturn;
    fn add(self, rhs: RealReturn) -> RealReturn {
        RealReturn(self.0 + rhs.0)
    }
}

impl std::ops::Sub for RealReturn {
    type Output = RealReturn;
    fn sub(self, rhs: RealReturn) -> RealReturn {
        RealReturn(self.0 - rhs.0)
    }
}

impl std::ops::Mul<f32> for RealReturn {
    type Output = RealReturn;
    fn mul(self, scalar: f32) -> RealReturn {
        RealReturn(self.0 * scalar)
    }
}

#[derive(Clone)]
pub(crate) struct Transition {
    pub(crate) obs: [f32; OBS_SIZE],
    pub(crate) action: [f32; ACTION_SIZE],
    pub(crate) reward: f32,
    pub(crate) value: NormalizedValue,
    pub(crate) log_prob: f32,
    pub(crate) end: StepEnd,
}

pub(crate) struct RolloutBuffer {
    pub(crate) transitions: Vec<Transition>,
    pub(crate) bootstrap: Option<NormalizedValue>,
}

impl RolloutBuffer {
    pub fn new() -> Self {
        Self {
            transitions: Vec::with_capacity(2048),
            bootstrap: None,
        }
    }

    pub fn push(&mut self, t: Transition) {
        self.transitions.push(t);
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }
}

impl Default for RolloutBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub(crate) struct ReturnNormalizer {
    mean: f64,
    m2: f64,
    count: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ReturnNormalizerData {
    mean: f64,
    m2: f64,
    count: u64,
}

impl Default for ReturnNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ReturnNormalizer {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
        }
    }

    fn std(&self) -> f32 {
        if self.count > 1 {
            ((self.m2 / (self.count as f64 - 1.0)).max(0.0).sqrt() as f32).max(1e-6)
        } else {
            1.0
        }
    }

    fn mean(&self) -> f32 {
        self.mean as f32
    }

    pub(crate) fn normalize(&self, ret: RealReturn) -> NormalizedValue {
        NormalizedValue((ret.0 - self.mean()) / self.std())
    }

    pub(crate) fn denormalize(&self, value: NormalizedValue) -> RealReturn {
        RealReturn(value.0 * self.std() + self.mean())
    }

    #[must_use = "a nonzero non-finite-return count signals a diverging env; surface it"]
    pub fn update(&mut self, returns: &[f32]) -> usize {
        let mut skipped = 0usize;
        for &r in returns {
            if !r.is_finite() {
                skipped += 1;
                continue;
            }
            self.count += 1;
            let x = r as f64;
            let delta = x - self.mean;
            self.mean += delta / self.count as f64;
            let delta2 = x - self.mean;
            self.m2 += delta * delta2;
        }
        skipped
    }

    pub fn to_data(&self) -> ReturnNormalizerData {
        ReturnNormalizerData {
            mean: self.mean,
            m2: self.m2,
            count: self.count,
        }
    }

    pub fn from_data(d: ReturnNormalizerData) -> Option<Self> {
        if d.m2 < 0.0 || !d.m2.is_finite() || !d.mean.is_finite() {
            return None;
        }
        Some(Self {
            mean: d.mean,
            m2: d.m2,
            count: d.count,
        })
    }
}

pub(crate) fn compute_gae(
    buffer: &RolloutBuffer,
    gamma: f32,
    lambda: f32,
    ret_norm: &ReturnNormalizer,
) -> (Vec<RealReturn>, Vec<RealReturn>) {
    let n = buffer.len();
    let mut advantages = vec![RealReturn(0.0); n];
    let mut returns = vec![RealReturn(0.0); n];
    let mut last_gae = RealReturn(0.0);
    let mut next_value = ret_norm.denormalize(buffer.bootstrap.unwrap_or(NormalizedValue(0.0)));

    for i in (0..n).rev() {
        let t = &buffer.transitions[i];
        let value = ret_norm.denormalize(t.value);
        let bootstrap = match t.end {
            StepEnd::Terminal => RealReturn(0.0),
            StepEnd::Truncated => value,
            StepEnd::Continues => next_value,
        };
        let delta = RealReturn(t.reward) + bootstrap * gamma - value;
        last_gae = if t.end.ends_segment() {
            delta
        } else {
            delta + last_gae * (gamma * lambda)
        };
        advantages[i] = last_gae;
        returns[i] = last_gae + value;
        next_value = value;
    }

    (advantages, returns)
}

fn next_standard_normal(rng: &mut StdRng) -> f32 {
    let u1: f32 = 1.0 - rng.r#gen::<f32>();
    let u2: f32 = rng.r#gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

const EXPLORE_CORRELATION: f32 = 0.95;

pub(crate) struct OuNoise {
    state: Vec<[f32; ACTION_SIZE]>,
}

impl OuNoise {
    pub(crate) fn new(n_envs: usize) -> Self {
        Self {
            state: vec![[0.0; ACTION_SIZE]; n_envs],
        }
    }

    pub(crate) fn reset(&mut self, e: usize, rng: &mut StdRng) {
        for s in &mut self.state[e] {
            *s = next_standard_normal(rng);
        }
    }

    pub(crate) fn next(&mut self, e: usize, rng: &mut StdRng) -> [f32; ACTION_SIZE] {
        let a = EXPLORE_CORRELATION;
        let b = (1.0 - a * a).sqrt();
        for x in &mut self.state[e] {
            *x = a * *x + b * next_standard_normal(rng);
        }
        self.state[e]
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct PpoMetrics {
    pub(crate) policy_loss: f32,
    pub(crate) value_loss: f32,
    pub(crate) entropy: f32,
    pub(crate) kl: f32,
    pub(crate) steps: u32,
    pub(crate) behavior_backend_div: f32,
    pub(crate) nonfinite_returns: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_std_floor_anneals_start_to_end_then_holds() {
        let config = PpoConfig {
            log_std_floor_start: -0.7,
            log_std_floor_end: -2.0,
            log_std_anneal_ticks: 1000,
            ..PpoConfig::default()
        };
        assert!(
            (config.log_std_floor(0) - (-0.7)).abs() < 1e-6,
            "wide at the epoch"
        );
        assert!(
            (config.log_std_floor(500) - (-1.35)).abs() < 1e-6,
            "linear midpoint"
        );
        assert!(
            (config.log_std_floor(1000) - (-2.0)).abs() < 1e-6,
            "refine at horizon"
        );
        assert!(
            (config.log_std_floor(10_000) - (-2.0)).abs() < 1e-6,
            "holds at the refine floor past the horizon"
        );

        let off = PpoConfig {
            log_std_anneal_ticks: 0,
            ..config
        };
        assert!(
            (off.log_std_floor(0) - (-2.0)).abs() < 1e-6,
            "a zero horizon is the refine floor from tick 0 (schedule off)"
        );
    }

    fn t(reward: f32, value: f32, end: StepEnd) -> Transition {
        Transition {
            obs: [0.0; OBS_SIZE],
            action: [0.0; ACTION_SIZE],
            reward,
            value: NormalizedValue(value),
            log_prob: 0.0,
            end,
        }
    }

    #[test]
    fn gae_per_env_differs_from_concatenated_sweep() {
        let gamma = 0.5;
        let lambda = 0.5;

        let mut env_a = RolloutBuffer::new();
        env_a.push(t(1.0, 0.5, StepEnd::Continues));
        env_a.push(t(1.0, 0.5, StepEnd::Continues));
        env_a.bootstrap = Some(NormalizedValue(2.0));
        let mut env_b = RolloutBuffer::new();
        env_b.push(t(0.0, 1.0, StepEnd::Continues));
        env_b.push(t(0.0, 1.0, StepEnd::Terminal));

        let id = ReturnNormalizer::new();
        let (adv_a, _) = compute_gae(&env_a, gamma, lambda, &id);
        assert!((adv_a[1].0 - 1.5).abs() < 1e-6, "A[1]: {}", adv_a[1].0);
        assert!((adv_a[0].0 - 1.125).abs() < 1e-6, "A[0]: {}", adv_a[0].0);

        let mut concat = RolloutBuffer::new();
        for tr in env_a.transitions.iter().chain(env_b.transitions.iter()) {
            concat.push(tr.clone());
        }
        let (adv_concat, _) = compute_gae(&concat, gamma, lambda, &id);
        assert!(
            (adv_concat[1].0 - adv_a[1].0).abs() > 1e-3,
            "concatenated sweep should corrupt A's advantages (got {} vs {})",
            adv_concat[1].0,
            adv_a[1].0
        );
    }

    #[test]
    fn truncation_bootstraps_unlike_true_terminal() {
        let gamma = 0.99;
        let lambda = 0.95;
        let (reward, value) = (1.0, 5.0);

        let id = ReturnNormalizer::new();
        let mut terminal = RolloutBuffer::new();
        terminal.push(t(reward, value, StepEnd::Terminal));
        let (adv_term, _) = compute_gae(&terminal, gamma, lambda, &id);

        let mut truncated = RolloutBuffer::new();
        truncated.push(Transition {
            end: StepEnd::Truncated,
            ..t(reward, value, StepEnd::Continues)
        });
        let (adv_trunc, ret_trunc) = compute_gae(&truncated, gamma, lambda, &id);

        assert!(
            (adv_term[0].0 - (reward - value)).abs() < 1e-6,
            "term: {}",
            adv_term[0].0
        );
        assert!(
            (adv_trunc[0].0 - (reward + gamma * value - value)).abs() < 1e-6,
            "trunc: {}",
            adv_trunc[0].0
        );
        assert!(
            (adv_trunc[0].0 - adv_term[0].0 - gamma * value).abs() < 1e-6,
            "truncation must bootstrap gamma*value more than a true terminal"
        );
        assert!(
            (ret_trunc[0].0 - (reward + gamma * value)).abs() < 1e-6,
            "ret: {}",
            ret_trunc[0].0
        );
    }

    #[test]
    fn continues_tail_bootstraps_from_successor_not_own_value() {
        let gamma = 0.9;
        let lambda = 0.95;
        let id = ReturnNormalizer::new();
        let (reward, own_v, succ_v) = (1.0f32, 5.0f32, 11.0f32);

        let mut buf = RolloutBuffer::new();
        buf.push(t(reward, own_v, StepEnd::Continues));
        buf.bootstrap = Some(NormalizedValue(succ_v));
        let (adv, _) = compute_gae(&buf, gamma, lambda, &id);

        let correct = reward + gamma * succ_v - own_v;
        let buggy = reward + gamma * own_v - own_v;
        assert!(
            (adv[0].0 - correct).abs() < 1e-6,
            "tail adv {}, want {correct}",
            adv[0].0
        );
        assert!(
            (adv[0].0 - buggy).abs() > 1e-3,
            "tail must NOT bootstrap from its own value"
        );
    }

    #[test]
    fn return_norm_preserves_advantage_sign_and_ordering() {
        let gamma = 0.99;
        let lambda = 0.95;

        let raw = [
            (-30.0f32, -120.0f32),
            (10.0, -90.0),
            (-5.0, -200.0),
            (40.0, -60.0),
            (-15.0, -150.0),
        ];
        let mut buf_raw = RolloutBuffer::new();
        for &(reward, value) in &raw {
            buf_raw.push(t(reward, value, StepEnd::Continues));
        }
        let id = ReturnNormalizer::new();
        let last_value_raw = -100.0f32;
        buf_raw.bootstrap = Some(NormalizedValue(last_value_raw));
        let (adv_raw, ret_raw) = compute_gae(&buf_raw, gamma, lambda, &id);

        let mut ret_norm = ReturnNormalizer::new();
        let skipped = ret_norm.update(&[-300.0, -100.0, 50.0, -250.0, 0.0, -180.0]);
        assert_eq!(skipped, 0, "all returns here are finite — none skipped");
        let mu = ret_norm.mean();
        let sigma = ret_norm.std();
        assert!(sigma > 1.0, "test needs a non-trivial scale, got σ={sigma}");
        assert!(mu.abs() > 1.0, "test needs a non-zero shift, got μ={mu}");

        let mut buf_norm = RolloutBuffer::new();
        for &(reward, value) in &raw {
            buf_norm.push(t(reward, (value - mu) / sigma, StepEnd::Continues));
        }
        buf_norm.bootstrap = Some(NormalizedValue((last_value_raw - mu) / sigma));
        let (adv_norm, ret_norm_out) = compute_gae(&buf_norm, gamma, lambda, &ret_norm);

        for (i, (a, b)) in adv_raw.iter().zip(adv_norm.iter()).enumerate() {
            assert!(
                (a.0 - b.0).abs() < 1e-3,
                "advantage[{i}] changed under return normalization: {} vs {}",
                a.0,
                b.0
            );
        }
        for i in 0..adv_raw.len() {
            assert_eq!(
                adv_raw[i].0.signum(),
                adv_norm[i].0.signum(),
                "advantage[{i}] sign flipped"
            );
            for j in (i + 1)..adv_raw.len() {
                assert_eq!(
                    adv_raw[i].0 < adv_raw[j].0,
                    adv_norm[i].0 < adv_norm[j].0,
                    "advantage ordering of {i} vs {j} changed"
                );
            }
        }

        for (i, (a, b)) in ret_raw.iter().zip(ret_norm_out.iter()).enumerate() {
            assert!(
                (a.0 - b.0).abs() < 1e-2,
                "return[{i}] changed under return normalization: {} vs {}",
                a.0,
                b.0
            );
        }

        let targets: Vec<f32> = ret_raw.iter().map(|&r| ret_norm.normalize(r).0).collect();
        for i in 0..ret_raw.len() {
            for j in (i + 1)..ret_raw.len() {
                assert_eq!(
                    ret_raw[i].0 < ret_raw[j].0,
                    targets[i] < targets[j],
                    "value-target ordering of {i} vs {j} changed"
                );
            }
        }
    }

    #[test]
    fn return_normalizer_reports_skipped_nonfinite_returns() {
        let mut norm = ReturnNormalizer::new();
        let skipped = norm.update(&[1.0, f32::NAN, 2.0, f32::INFINITY, 3.0]);
        assert_eq!(
            skipped, 2,
            "both non-finite returns must be reported as skipped"
        );

        let mut clean = ReturnNormalizer::new();
        let none = clean.update(&[1.0, 2.0, 3.0]);
        assert_eq!(none, 0, "all-finite returns skip nothing");
        assert!(
            (norm.mean() - clean.mean()).abs() < 1e-9,
            "the skip must not perturb the scale"
        );
        assert!(
            (norm.std() - clean.std()).abs() < 1e-9,
            "the skip must not perturb the std"
        );
    }

    #[test]
    fn return_normalizer_checkpoint_round_trips() {
        let mut norm = ReturnNormalizer::new();
        let returns: Vec<f32> = (0..200).map(|i| -2000.0 + (i as f32) * 17.3).collect();
        let _ = norm.update(&returns);

        let bytes = bincode::serialize(&norm.to_data()).expect("serialize");
        let data: ReturnNormalizerData = bincode::deserialize(&bytes).expect("deserialize");
        let loaded = ReturnNormalizer::from_data(data).expect("from_data");

        assert!(
            (norm.mean() - loaded.mean()).abs() < 1e-6,
            "mean: {} vs {}",
            norm.mean(),
            loaded.mean()
        );
        assert!(
            (norm.std() - loaded.std()).abs() < 1e-6,
            "std: {} vs {}",
            norm.std(),
            loaded.std()
        );
        for &r in &[-2000.0f32, -1000.0, 0.0, 500.0] {
            assert!(
                (norm.normalize(RealReturn(r)).0 - loaded.normalize(RealReturn(r)).0).abs() < 1e-6,
                "normalize({r}) diverged across checkpoint"
            );
        }

        let corrupt = ReturnNormalizerData {
            mean: 0.0,
            m2: -1.0,
            count: 10,
        };
        assert!(
            ReturnNormalizer::from_data(corrupt).is_none(),
            "a negative-M2 record must be rejected"
        );
    }
}
