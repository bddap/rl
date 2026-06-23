//! PPO (Proximal Policy Optimization) support functions.

use std::cell::RefCell;

use burn::prelude::*;
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::bot::actuator::ACTION_SIZE;
use crate::bot::sensor::OBS_SIZE;

pub struct PpoConfig {
    pub gamma: f32,
    pub lambda: f32,
    pub clip_epsilon: f32,
    pub entropy_coeff: f32,
    pub value_coeff: f32,
    pub learning_rate: f64,
    pub epochs_per_update: u32,
    pub batch_size: usize,
    /// Max absolute per-sample value-prediction ERROR `|V' - R'|` admitted into the
    /// value loss, in NORMALIZED units (the value head fits `R' = (R-μ)/σ`, so the
    /// unit is one standard deviation of the return). This is a plain RESIDUAL clamp
    /// — a Huber-style cap that bounds the squared loss's gradient so one mispredicted
    /// outlier can't dominate the update — NOT SB3's `clip_range_vf`, which trust-
    /// region-clips how far the new value may move from the OLD prediction. A residual
    /// that small (0.2σ) would clip almost every sample early in training and freeze
    /// the head, which is exactly the failure the old 10.0 produced against ~-2700
    /// returns (10 ≪ 2700, so every residual was clamped). A few σ is the right order:
    /// it passes honest predictions through and only tames genuine outliers.
    pub value_loss_clip: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            clip_epsilon: 0.2,
            // Per-dim mean entropy (not sum), so scale-invariant. Kept small: near a
            // converged pose the reward gradient goes flat, and a larger bonus then
            // overpowers it and inflates the action distribution until the stand
            // dissolves into noise (observed at 0.01 — entropy ran away late-training,
            // reward eroded). 0.003 still explores early but won't run away.
            entropy_coeff: 0.003,
            value_coeff: 0.5,
            learning_rate: 3e-4,
            epochs_per_update: 4,
            batch_size: 64,
            // ~3σ of the (now unit-scale) return target: loose enough not to clip
            // honest predictions, tight enough that a single outlier can't dominate
            // the value gradient. See the field doc for why this is σ, not the SB3
            // value-trust-region's 0.2.
            value_loss_clip: 3.0,
        }
    }
}

/// How a stored transition ends. Replaces the old `done`/`truncated` bool pair so
/// the "never both set" invariant is structural rather than a comment the producer
/// has to uphold by hand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepEnd {
    /// The trajectory continues past this step — an in-episode step, or a
    /// rollout-window boundary mid-episode (bootstrapped via `last_value`).
    Continues,
    /// True terminal: the episode genuinely ended here (a survival guard failed /
    /// the sim died), so the future return is 0.
    Terminal,
    /// Truncation: the episode was cut by the step cap while the crab was still
    /// standing. Its value must be bootstrapped (see [`compute_gae`]), or the policy
    /// is taught that surviving to the cap is worth nothing.
    Truncated,
}

impl StepEnd {
    /// A terminal or a truncation ends the trajectory segment, so the GAE trace must
    /// not fold future (next-episode) advantage back across it.
    fn ends_segment(self) -> bool {
        matches!(self, StepEnd::Terminal | StepEnd::Truncated)
    }
}

/// A value in the value head's NORMALIZED space — `(R-μ)/σ`, what the head emits and
/// regresses against. The ONLY way to reach [`RealReturn`] is [`ReturnNormalizer::denormalize`],
/// which is what makes "a value reaches GAE un-denormalized" a compile error rather
/// than a silent training corruption.
#[derive(Clone, Copy)]
pub(crate) struct NormalizedValue(pub(crate) f32);

/// A value in REAL reward units — the space GAE runs in (deltas, advantages,
/// returns). Crossing back to [`NormalizedValue`] is only [`ReturnNormalizer::normalize`].
/// Arithmetic is closed over real units (sum/difference of reals, real scaled by a
/// scalar discount); there is deliberately no op that mixes it with `NormalizedValue`.
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
    /// Discounting (`γ`, `γλ`) is a unitless scalar, so it stays in real units.
    fn mul(self, scalar: f32) -> RealReturn {
        RealReturn(self.0 * scalar)
    }
}

#[derive(Clone)]
pub struct Transition {
    pub obs: [f32; OBS_SIZE],
    pub action: [f32; ACTION_SIZE],
    pub reward: f32,
    /// The value head's raw output for this step — always NORMALIZED units, never
    /// pre-de-normalized (GAE de-normalizes it).
    pub(crate) value: NormalizedValue,
    pub log_prob: f32,
    pub end: StepEnd,
}

pub struct RolloutBuffer {
    pub transitions: Vec<Transition>,
}

impl RolloutBuffer {
    pub fn new() -> Self {
        Self {
            transitions: Vec::with_capacity(2048),
        }
    }

    pub fn push(&mut self, t: Transition) {
        self.transitions.push(t);
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

impl Default for RolloutBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Running mean/std of the value targets (GAE returns), used to fit the value head
/// against UNIT-SCALE targets regardless of reward magnitude.
///
/// # Why this exists
/// The advantages PPO's policy gradient uses are already batch-normalized
/// ([`ppo_update_core`]), so the policy is scale-invariant. The value head is not:
/// it regresses raw returns, and when the reward magnitude is large (the pose+reach
/// reward accumulates to ~+1500 over a full 1500-step episode) the squared value loss
/// and its gradient blow up, the bounded value head can't track the target, advantages
/// derived from `R - V` become noise, and training diverges. Normalizing the value
/// TARGET to unit scale fixes the value head's conditioning without touching the
/// reward or the policy.
///
/// # Why the optimum is preserved
/// Let `R` be a return and `R' = (R - μ)/σ` with `σ > 0`. The value head fits `R'`;
/// wherever a value re-enters the algorithm as a real quantity (GAE bootstrap,
/// `R - V` deltas) it is DE-normalized first (`V = V'·σ + μ`), so GAE, the returns,
/// and the advantages are computed entirely in real reward units — identical to no
/// normalization up to f32 rounding in the de-normalize. Normalization is a positive
/// affine map applied ONLY to the scalar the value head regresses, which moves
/// neither the argmax policy nor the sign/ordering of advantages. The mean shift is
/// safe BECAUSE of the de-normalize: it is subtracted only from the value head's
/// regression target and added straight back before the value enters GAE, so it
/// never shifts the reward or the return the way subtracting a mean from the reward
/// itself would. (Standard PPO return normalization — cf. SB3
/// `VecNormalize` returns / PopArt, here PopArt-lite: stats track the target but the
/// head's last layer is not analytically rescaled, because the running stats are
/// applied OUTSIDE the head, on its raw output, so a rescale is unnecessary.)
///
/// # Single source of truth
/// Mean/M2/count are the only stored fields; variance/std are derived on demand
/// (mirrors [`ObsNormalizer`](crate::training::session)). The learner owns ONE of
/// these; rollout threads never update it (they only emit value PREDICTIONS, which
/// the learner de-normalizes with its own stats), so there is no second copy to
/// drift. The stats lag the policy by exactly one update — they are refreshed from
/// each update's returns AFTER that update's GAE used the previous stats — which is
/// the standard PopArt ordering and unbiased in expectation; advantage
/// normalization makes the one-update lag irrelevant to the policy gradient.
#[derive(Clone)]
pub struct ReturnNormalizer {
    mean: f64,
    m2: f64,
    count: u64,
}

/// Serde mirror of [`ReturnNormalizer`] for the checkpoint (persisted beside the obs
/// normalizer so a resumed run de-normalizes against the same scale it trained with).
/// No `var`/`std` field: both derive from `m2`/`count`, so storing them would be a
/// second source of truth that can drift.
#[derive(Clone, Serialize, Deserialize)]
pub struct ReturnNormalizerData {
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
        // count 0 ⇒ identity transform (μ=0, σ=1): the first update, before any
        // return has been observed, normalizes/de-normalizes to a no-op, so it is
        // byte-identical to un-normalized PPO until stats exist.
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
        }
    }

    /// Running std `sqrt(m2 / (count-1))`, floored at 1e-6 so the divide in
    /// [`Self::normalize`] never explodes. Returns 1.0 until at least two returns
    /// have been seen (no spread estimate yet ⇒ identity scale), so early updates
    /// are unaffected.
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

    /// Map a real return to the unit-scale target the value head regresses:
    /// `(R - μ)/σ`. The sole crossing from real units into the value head's space.
    pub(crate) fn normalize(&self, ret: RealReturn) -> NormalizedValue {
        NormalizedValue((ret.0 - self.mean()) / self.std())
    }

    /// Inverse of [`Self::normalize`]: map a value-head output back to real reward
    /// units, `V'·σ + μ`. The sole crossing from the head's space into real units, so
    /// every predicted value re-entering the algorithm (GAE bootstrap, stored per-step
    /// values) is forced through here and GAE stays in real units by construction.
    pub(crate) fn denormalize(&self, value: NormalizedValue) -> RealReturn {
        RealReturn(value.0 * self.std() + self.mean())
    }

    /// Fold a batch of real returns into the running (count, mean, M2) via Welford.
    /// Non-finite returns are skipped (a blown-up env must not poison the scale).
    pub fn update(&mut self, returns: &[f32]) {
        for &r in returns {
            if !r.is_finite() {
                continue;
            }
            self.count += 1;
            let x = r as f64;
            let delta = x - self.mean;
            self.mean += delta / self.count as f64;
            let delta2 = x - self.mean;
            self.m2 += delta * delta2;
        }
    }

    pub fn to_data(&self) -> ReturnNormalizerData {
        ReturnNormalizerData {
            mean: self.mean,
            m2: self.m2,
            count: self.count,
        }
    }

    /// Rebuild from the checkpoint mirror, rejecting a corrupt record — negative or
    /// non-finite M2, or a non-finite mean — so a bad checkpoint can't hand the
    /// trainer a NaN/Inf scale (`std`/`mean` would propagate it into every value).
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

/// `ret_norm` is the running return scale that de-normalizes the value-head outputs
/// ([`ReturnNormalizer::denormalize`]) — the only path from [`NormalizedValue`] to the
/// [`RealReturn`] arithmetic here, so GAE is in real reward units by construction.
/// Until any return has been observed the scale is the identity, so this matches
/// un-normalized GAE bit-for-bit before the head is trained against a normalized target.
pub(crate) fn compute_gae(
    buffer: &RolloutBuffer,
    last_value: NormalizedValue,
    gamma: f32,
    lambda: f32,
    ret_norm: &ReturnNormalizer,
) -> (Vec<RealReturn>, Vec<RealReturn>) {
    let n = buffer.len();
    let mut advantages = vec![RealReturn(0.0); n];
    let mut returns = vec![RealReturn(0.0); n];
    let mut last_gae = RealReturn(0.0);
    // The trailing bootstrap is a value-head output (normalized) → real units.
    let mut next_value = ret_norm.denormalize(last_value);

    for i in (0..n).rev() {
        let t = &buffer.transitions[i];
        // The stored per-step value is the value head's normalized prediction;
        // de-normalize to real reward units so the delta/return below match the
        // real-unit reward.
        let value = ret_norm.denormalize(t.value);
        // Bootstrap target V(s_{i+1}):
        //  - done (true terminal): 0 — the episode genuinely ended.
        //  - truncated (cut by the step cap, env then reset): the real next
        //    state was discarded at reset, and the next buffer entry belongs to
        //    a *different* episode. Bootstrap from V(s_i) (this step's own value
        //    ≈ V of the cut continuation for a slowly-changing pose) so the cap
        //    isn't taught as a dead end.
        //  - otherwise: the next entry's value (an in-episode step, or a
        //    rollout-boundary cut bootstrapped via `last_value`).
        let bootstrap = match t.end {
            StepEnd::Terminal => RealReturn(0.0),
            StepEnd::Truncated => value,
            StepEnd::Continues => next_value,
        };
        // Reward is already in real units, so the whole delta stays in real units.
        let delta = RealReturn(t.reward) + bootstrap * gamma - value;
        // The GAE trace cannot cross an episode boundary: a done or a truncation
        // ends this trajectory segment, so the future (next-episode) advantage
        // must not fold back across it.
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

/// Log-space throughout to avoid dividing by a tiny variance.
pub fn compute_log_prob<B: Backend>(
    means: &Tensor<B, 1>,
    log_std: &Tensor<B, 1>,
    actions: &Tensor<B, 1>,
) -> f32 {
    // log_std arrives pre-clamped from CrabBrain::policy (single source of truth).
    let diff = actions.clone() - means.clone();
    // log p = -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
    let scaled_diff = diff / log_std.clone().exp();
    let log_probs = scaled_diff.powf_scalar(2.0).neg() * 0.5
        - log_std.clone()
        - 0.5 * (2.0 * std::f32::consts::PI).ln();
    log_probs.sum().into_scalar().elem::<f32>()
}

thread_local! {
    /// Per-thread RNG for action-noise sampling. The NdArray backend's
    /// `Tensor::random` locks a process-global `static SEED: Mutex<NdArrayRng>` on
    /// every draw, so with K rollout threads sampling every env every tick they all
    /// serialize on that one mutex — a hot-path global lock that throttles the
    /// near-linear scaling this module exists for. Drawing the Gaussian noise from a
    /// thread-local stream instead removes the lock entirely while preserving the
    /// sampling distribution (standard-normal noise, then `mean + std·noise`).
    ///
    /// Seeded from OS entropy mixed with the thread's id: entropy alone already gives
    /// each thread an independent stream (it is drawn fresh per thread-local init),
    /// and folding in the id guarantees distinctness even in the astronomically
    /// unlikely event two threads' entropy draws coincide.
    static ACTION_RNG: RefCell<StdRng> = RefCell::new(seed_action_rng());
}

/// Seed a thread-local action RNG from OS entropy XORed with the current thread's
/// id, so each rollout thread draws an independent noise stream off the global lock.
fn seed_action_rng() -> StdRng {
    use rand::{RngCore, SeedableRng};
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::thread::current().id().hash(&mut hasher);
    let tid_mix = hasher.finish().to_le_bytes();

    // Fill the full ChaCha seed from the OS CSPRNG (already independent per init),
    // then XOR the thread-id hash into the first 8 bytes as a belt-and-suspenders
    // guarantee that no two threads can share a stream.
    let mut seed = <StdRng as SeedableRng>::Seed::default();
    rand::rngs::OsRng.fill_bytes(&mut seed);
    for (b, m) in seed.iter_mut().zip(tid_mix.iter()) {
        *b ^= *m;
    }
    StdRng::from_seed(seed)
}

/// Draw one standard-normal sample (N(0,1)) from `rng` via the Box–Muller transform.
/// `rand 0.8` has no normal distribution in core and the project's `rand_distr` is a
/// later-`rand` release (incompatible RNG traits), so this keeps the thread-local
/// noise source on the same `rand 0.8` `StdRng` the rest of the crate uses. The
/// distribution is exactly standard-normal — identical to the backend RNG's
/// `Distribution::Normal(0.0, 1.0)` this replaced, only off the global lock.
fn next_standard_normal(rng: &mut StdRng) -> f32 {
    // u1 in (0, 1] so ln(u1) is finite (open at 0); u2 in [0, 1) for the angle.
    let u1: f32 = 1.0 - rng.r#gen::<f32>();
    let u2: f32 = rng.r#gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Sample actions from Gaussian policy.
///
/// The noise comes from a THREAD-LOCAL RNG ([`ACTION_RNG`]) rather than the backend's
/// global-mutex-locked `Tensor::random`, so K rollout threads don't serialize on a
/// hot-path lock. Swapping the RNG source leaves the distribution unchanged:
/// standard-normal noise, then `mean + std·noise`, clamped.
pub fn sample_action<B: Backend>(
    means: &Tensor<B, 1>,
    log_std: &Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let std = log_std.clone().exp();
    let noise_vals: [f32; ACTION_SIZE] =
        ACTION_RNG.with(|rng| std::array::from_fn(|_| next_standard_normal(&mut rng.borrow_mut())));
    let noise = Tensor::<B, 1>::from_floats(noise_vals, device);
    let action = means.clone() + noise * std;
    action.clamp(-1.0, 1.0)
}

#[derive(Debug, Default, Clone)]
pub struct PpoMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// GAE must be computed per env. Sweeping one concatenated buffer lets env
    /// A's last step bootstrap from env B's first value — silent advantage
    /// corruption that still "trains". This pins the per-env results and proves
    /// the concatenated sweep actually diverges (i.e. the split is load-bearing).
    #[test]
    fn gae_per_env_differs_from_concatenated_sweep() {
        let gamma = 0.5;
        let lambda = 0.5;

        let mut env_a = RolloutBuffer::new();
        env_a.push(t(1.0, 0.5, StepEnd::Continues));
        env_a.push(t(1.0, 0.5, StepEnd::Continues));
        let mut env_b = RolloutBuffer::new();
        env_b.push(t(0.0, 1.0, StepEnd::Continues));
        env_b.push(t(0.0, 1.0, StepEnd::Terminal));

        // Identity scale (no returns observed ⇒ μ=0, σ=1): GAE is in raw units, so
        // these hand-computed expectations are the un-normalized values.
        let id = ReturnNormalizer::new();
        // Per-env, hand-computed: A bootstraps from ITS next value (2.0).
        let (adv_a, _) = compute_gae(&env_a, NormalizedValue(2.0), gamma, lambda, &id);
        assert!((adv_a[1].0 - 1.5).abs() < 1e-6, "A[1]: {}", adv_a[1].0);
        assert!((adv_a[0].0 - 1.125).abs() < 1e-6, "A[0]: {}", adv_a[0].0);

        // Naive concatenated sweep: A's last step bootstraps from B's value.
        let mut concat = RolloutBuffer::new();
        for tr in env_a.transitions.iter().chain(env_b.transitions.iter()) {
            concat.push(tr.clone());
        }
        let (adv_concat, _) = compute_gae(&concat, NormalizedValue(0.0), gamma, lambda, &id);
        assert!(
            (adv_concat[1].0 - adv_a[1].0).abs() > 1e-3,
            "concatenated sweep should corrupt A's advantages (got {} vs {})",
            adv_concat[1].0,
            adv_a[1].0
        );
    }

    /// A truncated final step (episode cut by the step cap while still standing)
    /// must bootstrap its value, unlike a true terminal which has zero future
    /// return. Conflating the two — the original bug — taught the policy that
    /// surviving to the cap is worthless. Same reward and value, opposite
    /// bootstrap: the advantages must differ by exactly `gamma * value`.
    #[test]
    fn truncation_bootstraps_unlike_true_terminal() {
        let gamma = 0.99;
        let lambda = 0.95;
        let (reward, value) = (1.0, 5.0);

        // Identity scale: GAE runs in raw units (see the sibling test).
        let id = ReturnNormalizer::new();
        let mut terminal = RolloutBuffer::new();
        terminal.push(t(reward, value, StepEnd::Terminal));
        let (adv_term, _) = compute_gae(&terminal, NormalizedValue(0.0), gamma, lambda, &id);

        let mut truncated = RolloutBuffer::new();
        truncated.push(Transition {
            end: StepEnd::Truncated,
            ..t(reward, value, StepEnd::Continues)
        });
        let (adv_trunc, ret_trunc) =
            compute_gae(&truncated, NormalizedValue(0.0), gamma, lambda, &id);

        // Terminal: advantage = reward - value (no bootstrap).
        assert!(
            (adv_term[0].0 - (reward - value)).abs() < 1e-6,
            "term: {}",
            adv_term[0].0
        );
        // Truncated: advantage = reward + gamma*value - value (bootstrap own value).
        assert!(
            (adv_trunc[0].0 - (reward + gamma * value - value)).abs() < 1e-6,
            "trunc: {}",
            adv_trunc[0].0
        );
        // The whole point: the cap is not a dead end.
        assert!(
            (adv_trunc[0].0 - adv_term[0].0 - gamma * value).abs() < 1e-6,
            "truncation must bootstrap gamma*value more than a true terminal"
        );
        // Return = bootstrapped one-step target.
        assert!(
            (ret_trunc[0].0 - (reward + gamma * value)).abs() < 1e-6,
            "ret: {}",
            ret_trunc[0].0
        );
    }

    /// Return normalization must not change the policy's learning signal: with the
    /// value head's outputs de-normalized inside GAE, a normalized run's advantages
    /// are an exact AFFINE transform of an un-normalized run's — here the identity
    /// (slope 1, offset 0), so sign and ordering are preserved bit-for-bit. This is
    /// the core correctness claim: normalizing the value TARGET leaves GAE in real
    /// units, so the argmax policy and every advantage's sign are untouched.
    ///
    /// Setup: a raw value function `v` over a multi-step episode. A normalized value
    /// head would emit `v' = (v - μ)/σ`; feeding those normalized values to
    /// `compute_gae` WITH the matching `ReturnNormalizer{μ,σ}` must reproduce exactly
    /// the advantages the raw values produced with the identity scale.
    #[test]
    fn return_norm_preserves_advantage_sign_and_ordering() {
        let gamma = 0.99;
        let lambda = 0.95;

        // A real-unit episode: large-magnitude values/rewards (the regime that
        // diverges) with sign changes so the ordering claim has something to bite on.
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
        // Un-normalized baseline: identity scale, raw values, raw trailing bootstrap.
        let id = ReturnNormalizer::new();
        let last_value_raw = -100.0f32;
        let (adv_raw, ret_raw) = compute_gae(
            &buf_raw,
            NormalizedValue(last_value_raw),
            gamma,
            lambda,
            &id,
        );

        // A normalizer with a non-trivial, positive affine scale (built from a spread
        // of returns so μ and σ are both non-zero).
        let mut ret_norm = ReturnNormalizer::new();
        ret_norm.update(&[-300.0, -100.0, 50.0, -250.0, 0.0, -180.0]);
        let mu = ret_norm.mean();
        let sigma = ret_norm.std();
        assert!(sigma > 1.0, "test needs a non-trivial scale, got σ={sigma}");
        assert!(mu.abs() > 1.0, "test needs a non-zero shift, got μ={mu}");

        // The value head, trained against this scale, would emit normalized values.
        let mut buf_norm = RolloutBuffer::new();
        for &(reward, value) in &raw {
            buf_norm.push(t(reward, (value - mu) / sigma, StepEnd::Continues));
        }
        let last_value_norm = NormalizedValue((last_value_raw - mu) / sigma);
        let (adv_norm, ret_norm_out) =
            compute_gae(&buf_norm, last_value_norm, gamma, lambda, &ret_norm);

        // Advantages are IDENTICAL (the affine map with slope 1, offset 0): GAE saw
        // the same real-unit values either way, so the policy gradient is unchanged.
        for (i, (a, b)) in adv_raw.iter().zip(adv_norm.iter()).enumerate() {
            assert!(
                (a.0 - b.0).abs() < 1e-3,
                "advantage[{i}] changed under return normalization: {} vs {}",
                a.0,
                b.0
            );
        }
        // Hence sign and strict ordering are preserved (the weaker properties the
        // affine invariance implies, asserted directly so a regression is legible).
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

        // The returns GAE reports are likewise in real units (the value head's loss
        // target is `normalize(return)`, applied later in `ppo_update_core`).
        for (i, (a, b)) in ret_raw.iter().zip(ret_norm_out.iter()).enumerate() {
            assert!(
                (a.0 - b.0).abs() < 1e-2,
                "return[{i}] changed under return normalization: {} vs {}",
                a.0,
                b.0
            );
        }

        // And the value-loss TARGET itself — `normalize(return)` — is a positive
        // affine image of the raw return, so it preserves order/sign too (this is the
        // only place the scale actually enters the regression).
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

    /// The running return scale must survive a checkpoint: what `to_data` serializes,
    /// `from_data` restores — same mean/std/normalize — so a resumed run de-normalizes
    /// the value head against the scale it trained with (a cold scale would mis-scale
    /// every value prediction on the first updates after a resume). Also pins that a
    /// corrupt (negative-M2) record is rejected rather than yielding a NaN std.
    #[test]
    fn return_normalizer_checkpoint_round_trips() {
        let mut norm = ReturnNormalizer::new();
        // A spread of returns spanning the large-magnitude regime.
        let returns: Vec<f32> = (0..200).map(|i| -2000.0 + (i as f32) * 17.3).collect();
        norm.update(&returns);

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
        // The user-visible behavior (normalize) must match across the round-trip.
        for &r in &[-2000.0f32, -1000.0, 0.0, 500.0] {
            assert!(
                (norm.normalize(RealReturn(r)).0 - loaded.normalize(RealReturn(r)).0).abs() < 1e-6,
                "normalize({r}) diverged across checkpoint"
            );
        }

        // A negative M2 is corrupt and must be refused (it would give a NaN std).
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
