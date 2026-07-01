//! PPO (Proximal Policy Optimization) support functions.

use burn::prelude::*;
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
    /// Huber knee δ for the value loss (see `huber_value_loss` for the squared-core/
    /// linear-tail math and why it beats a hard clamp), on the value-prediction
    /// ERROR `|V' - R'|` in NORMALIZED units — the value head fits `R' = (R-μ)/σ`, so the unit
    /// is one standard deviation of the return. NOT SB3's `clip_range_vf`, which trust-region-
    /// clips how far the new value may move from the OLD prediction. A knee that small (0.2σ)
    /// would push almost every sample into the linear tail and under-fit the head — the failure
    /// the old 10.0 produced against ~-2700 returns (10 ≪ 2700); a few σ keeps honest
    /// predictions in the quadratic band and only outliers in the tail.
    pub(crate) value_loss_clip: f32,
    /// Trust-region ceiling on how far ONE update may move the policy, as an
    /// approximate KL divergence (`π_old` → current policy, the unbiased
    /// `mean((r-1) - ln r)` estimator, both forwards on the update backend). The PPO
    /// ratio clip only zeroes the gradient for samples outside the band — it does NOT
    /// bound total KL across the 4 epochs × many minibatches, so a sharpened
    /// (near-deterministic) policy can still walk off a cliff in a single iteration
    /// (observed as the reach-1.0 → reach-0.0 collapse around iter ~4000). Once
    /// cumulative KL crosses `1.5 × target_kl` the update STOPS for this iteration, so
    /// each iteration moves the policy by at most ~`target_kl`. 0.03 is generous:
    /// healthy updates run ~0.01, so it never throttles normal learning, but it
    /// hard-stops the 10×+ over-steps of a collapse. Only meaningful because `π_old` is
    /// recomputed on the update backend (see the update); against the rollout's
    /// CPU-recorded log-probs the backend mismatch alone reads as ~0.7 KL.
    pub(crate) target_kl: f32,
    /// Exploration-σ schedule (bddap/rl#161): the annealed LOWER bound on the policy's
    /// `log_std`, starting HIGH to force a wide exploration amplitude early — bypassing the
    /// target-KL trust region that otherwise pins σ at its cautious init (the entropy bonus
    /// can't widen it inside the per-update KL budget, falsified job 616) — then decaying to
    /// `log_std_floor_end` for refinement. Composed with the OU temporally-correlated noise
    /// ([`OuNoise`]), early exploration is both WIDE and coherent: the best shot at stumbling
    /// into a coordinated 38-DOF gait. The schedule clamps `log_std` from below (see
    /// [`CrabBrain::policy`]); the learned param sits beneath the early floor, so the floor —
    /// not the (KL-throttled) entropy gradient — does the widening. TRAINING exploration only:
    /// eval/demo takes the policy MEAN, so the floor never reaches a deployed action.
    ///
    /// `log_std_floor_start`/`_end` are the wide-early and refine `log_std` floors;
    /// `log_std_anneal_ticks` is the physics-tick horizon over which it linearly anneals from
    /// start to end. Env-overridable (`RL_LOG_STD_FLOOR_START` / `_END` /
    /// `RL_LOG_STD_ANNEAL_TICKS`) so the window is tunable without a rebuild.
    pub(crate) log_std_floor_start: f32,
    pub(crate) log_std_floor_end: f32,
    pub(crate) log_std_anneal_ticks: u64,
}

impl PpoConfig {
    /// The exploration-σ floor (lower `log_std` clamp) for a point `ticks_into_anneal` physics
    /// ticks into the schedule's horizon: a linear ramp from `log_std_floor_start` (wide) down
    /// to `log_std_floor_end` (refine), holding at the end value past the horizon. The caller
    /// passes ticks measured from the schedule's epoch (the warm-resume point, or a cold reset),
    /// not absolute training ticks, so "wide early" means early in THIS experiment regardless of
    /// how much prior training the resumed checkpoint carries. A zero horizon yields the end
    /// value from the first tick (schedule effectively off).
    pub(crate) fn log_std_floor(&self, ticks_into_anneal: u64) -> f32 {
        if self.log_std_anneal_ticks == 0 {
            return self.log_std_floor_end;
        }
        let frac = (ticks_into_anneal as f32 / self.log_std_anneal_ticks as f32).clamp(0.0, 1.0);
        self.log_std_floor_start + (self.log_std_floor_end - self.log_std_floor_start) * frac
    }
}

/// Read an `f32`/`u64` tuning knob from the environment, falling back to `default` when the var
/// is unset or unparseable. Keeps the σ-schedule window tunable without a rebuild (the trainer's
/// launcher exports the var); a typo'd value loudly falls back rather than silently changing the
/// schedule.
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
            // Per-dim mean entropy (not sum), so scale-invariant. It's a constant upward
            // force on the policy's log_std, opposed only by the reach-reward gradient —
            // which weakens as the curriculum moves the target farther (~⅓ less pull from
            // band 1.5-3.0m to 2.5-4.0m), so the coeff must sit below what the FARTHEST
            // band's reach can contain, else log_std inflates unbounded and the policy
            // dissolves into noise (ran away at 0.01; 0.003 still crept up in band 1, then
            // diffused once band 2 weakened the reach signal). 0.001 leaves margin.
            entropy_coeff: 0.001,
            value_coeff: 0.5,
            learning_rate: 3e-4,
            epochs_per_update: 4,
            batch_size: 64,
            value_loss_clip: 3.0,
            target_kl: 0.03,
            // Wide-early → refine exploration-σ floor (bddap/rl#161). Start log_std -0.7
            // (σ≈0.50 — the amplitude that earlier found the most reach, now refinable thanks
            // to OU correlation) and anneal to LOG_STD_MIN (σ≈0.135), the resting clamp, over
            // ~5M ticks (~1200 iters at 4096 ticks/iter). Tunable without a rebuild.
            log_std_floor_start: env_or("RL_LOG_STD_FLOOR_START", -0.7),
            log_std_floor_end: env_or("RL_LOG_STD_FLOOR_END", crate::bot::brain::LOG_STD_MIN),
            log_std_anneal_ticks: env_or("RL_LOG_STD_ANNEAL_TICKS", 5_000_000),
        }
    }
}

/// How a stored transition ends. Replaces the old `done`/`truncated` bool pair so
/// the "never both set" invariant is structural rather than a comment the producer
/// has to uphold by hand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StepEnd {
    /// The trajectory continues past this step — an in-episode step, or a
    /// rollout-window boundary mid-episode (the tail's successor value comes from
    /// [`RolloutBuffer::bootstrap`]).
    Continues,
    /// True terminal: the episode genuinely ended here (the crab GRABBED the target, a
    /// survival guard failed, or the sim died), so the future return is 0.
    Terminal,
    /// Truncation: the episode was cut by the step cap while the crab was still
    /// standing. Its value must be bootstrapped (see [`compute_gae`]), or the policy
    /// is taught that surviving to the cap is worth nothing.
    Truncated,
}

impl StepEnd {
    /// A terminal or a truncation ends the trajectory segment, so the GAE trace must
    /// not fold future (next-episode) advantage back across it (and the rollout resets
    /// the env on it).
    pub(crate) fn ends_segment(self) -> bool {
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
pub(crate) struct Transition {
    pub(crate) obs: [f32; OBS_SIZE],
    /// The policy's unbounded pre-clamp DRIVE `μ + σ·ε` (the sim ran `drive.clamp(±1)`). The
    /// PPO update recomputes its log-prob over THIS, so it must be the sample the stored
    /// `log_prob` was taken on — the drive, not the clamped command (see `sample_actions`).
    pub(crate) action: [f32; ACTION_SIZE],
    pub(crate) reward: f32,
    /// The value head's raw output for this step — always NORMALIZED units, never
    /// pre-de-normalized (GAE de-normalizes it).
    pub(crate) value: NormalizedValue,
    pub(crate) log_prob: f32,
    pub(crate) end: StepEnd,
}

pub(crate) struct RolloutBuffer {
    pub(crate) transitions: Vec<Transition>,
    /// GAE bootstrap `V(s_{last+1})` for a non-terminal (`Continues`) tail: the value of
    /// the successor state this buffer was cut just before. Sourced from the rollout's
    /// dangling `Pending` (`systems.rs`) — the un-finalized next action whose stored value
    /// IS that successor's value, computed on the SAME (CPU rollout) backend as every body
    /// value. ONLY consumed when the tail is `Continues`; a `Terminal`/`Truncated` tail
    /// self-bootstraps in [`compute_gae`] and ignores this seed (it may still be `Some` —
    /// the first `Pending` of a fresh episode that started after the terminal — harmlessly).
    /// `None` when no action is pending (a settling/reset env) or the buffer is empty.
    /// Binding the bootstrap to the buffer is what makes it structurally the right state on
    /// the right backend — the alternative (recomputing `V(last_obs)` on the update backend)
    /// read the wrong state (off-by-one from the one-tick `Pending` phasing, rl#174) on the
    /// wrong backend (rl#173 tail).
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

/// Running mean/std of the value targets (GAE returns), used to fit the value head
/// against UNIT-SCALE targets regardless of reward magnitude.
///
/// # Why this exists
/// The advantages PPO's policy gradient uses are already batch-normalized
/// ([`ppo_update_core`]), so the policy is scale-invariant. The value head is not:
/// it regresses raw returns, and when the reward magnitude is large (a reward that
/// accumulates to hundreds or thousands over a full episode) the squared value loss
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
/// (mirrors [`ObsNormalizer`](crate::training::normalizer)). The learner owns ONE of
/// these; rollout threads never update it (they only emit value PREDICTIONS, which
/// the learner de-normalizes with its own stats), so there is no second copy to
/// drift. The stats lag the policy by exactly one update — they are refreshed from
/// each update's returns AFTER that update's GAE used the previous stats — which is
/// the standard PopArt ordering and unbiased in expectation; advantage
/// normalization makes the one-update lag irrelevant to the policy gradient.
#[derive(Clone)]
pub(crate) struct ReturnNormalizer {
    mean: f64,
    m2: f64,
    count: u64,
}

/// Serde mirror of [`ReturnNormalizer`] for the checkpoint (persisted beside the obs
/// normalizer so a resumed run de-normalizes against the same scale it trained with).
/// No `var`/`std` field: both derive from `m2`/`count`, so storing them would be a
/// second source of truth that can drift.
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

    /// Fold a batch of real returns into the running (count, mean, M2) via Welford. A non-finite
    /// return is skipped (a blown-up env must not poison the scale) — but the skip is no longer
    /// silent: the number skipped is RETURNED so the caller can surface it loudly (bddap/rl#167),
    /// turning "a diverging env quietly corrupts the normalizer" into a visible boundary signal.
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
    gamma: f32,
    lambda: f32,
    ret_norm: &ReturnNormalizer,
) -> (Vec<RealReturn>, Vec<RealReturn>) {
    let n = buffer.len();
    let mut advantages = vec![RealReturn(0.0); n];
    let mut returns = vec![RealReturn(0.0); n];
    let mut last_gae = RealReturn(0.0);
    // Trailing bootstrap V(s_{last+1}): the buffer carries the successor value of its
    // `Continues` tail (`RolloutBuffer::bootstrap`), already on the CPU rollout backend.
    // `None` ⟹ a `Terminal`/`Truncated` tail (the per-step `match` below ignores this
    // seed) or an empty buffer; the 0 fallback is then inert.
    let mut next_value =
        ret_norm.denormalize(buffer.bootstrap.unwrap_or(NormalizedValue(0.0)));

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
        //  - otherwise: the next entry's value (an in-episode step, or the
        //    rollout-boundary cut bootstrapped via `buffer.bootstrap`).
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
pub(crate) fn compute_log_prob<B: Backend>(
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

/// Diagonal-Gaussian log-prob of each action ROW under a shared per-dim `log_std`:
/// `Σ_d [ -0.5·((aᵈ-μᵈ)/σᵈ)² - ln σᵈ - 0.5·ln(2π) ]`. The ONE batched form of
/// [`compute_log_prob`], used by the PPO update for BOTH the pre-update behavior
/// log-prob (`π_old`, recomputed on the update backend so the importance ratio starts
/// at 1) and each minibatch's `π_new` — one formula, so the two can't drift. Generic
/// over the backend; the autodiff graph flows through `means`/`log_std` for `π_new`
/// and is detached by the caller for `π_old`.
pub(crate) fn gaussian_log_prob_rows<B: Backend>(
    means: Tensor<B, 2>,
    log_std: Tensor<B, 1>,
    actions: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let rows = means.dims()[0];
    let log_std_2d = log_std.unsqueeze_dim::<2>(0).expand([rows, ACTION_SIZE]);
    let scaled_diff = (actions - means) / log_std_2d.clone().exp();
    let half_log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
    (scaled_diff.powf_scalar(2.0).neg() * 0.5 - log_std_2d - half_log_2pi)
        .sum_dim(1)
        .flatten::<1>(0, 1)
}

/// Draw one standard-normal sample (N(0,1)) from `rng` via the Box–Muller transform.
/// `rand 0.8` has no normal distribution in core and `rand_distr` tracks a later `rand`
/// (incompatible RNG traits) — so rather than carry that dep, this keeps the noise source
/// on the same `rand 0.8` `StdRng` the rest of the crate uses. The distribution is exactly
/// standard-normal.
fn next_standard_normal(rng: &mut StdRng) -> f32 {
    // u1 in (0, 1] so ln(u1) is finite (open at 0); u2 in [0, 1) for the angle.
    let u1: f32 = 1.0 - rng.r#gen::<f32>();
    let u2: f32 = rng.r#gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Sample one DRIVE per joint from the Gaussian policy: `dᵢ = μᵢ + σ·εᵢ`, UN-clamped.
///
/// Returns the raw pre-clamp drive — the random variable the policy actually drew, and the
/// quantity the reward's metabolic tax and the PPO log-prob are both taken over (see
/// `sample_actions`). The ±1 clamp that bounds the sim's torque command is applied by the
/// caller, so the unbounded drive survives for the tax to bite on saturation (a `|d|≫1` drive
/// that slams a joint onto its rail). Clamping here would erase that overshoot — the tax would
/// see only the bounded command and lose its pull off the rail.
///
/// `noise` is the standard-normal exploration `ε` the caller drew — per-tick-independent in the
/// old design, now the temporally-correlated draw from [`OuNoise`] (bddap/rl#161). Either way it
/// is marginally N(0,1), so the per-step Gaussian log-prob the PPO update recomputes is unchanged
/// — only the temporal correlation of successive `ε`s differs, which is what makes a coordinated
/// gait discoverable.
pub(crate) fn sample_action<B: Backend>(
    means: &Tensor<B, 1>,
    log_std: &Tensor<B, 1>,
    noise: [f32; ACTION_SIZE],
    device: &B::Device,
) -> Tensor<B, 1> {
    let std = log_std.clone().exp();
    let noise = Tensor::<B, 1>::from_floats(noise, device);
    means.clone() + noise * std
}

/// AR(1) retention coefficient α ∈ [0, 1) for the exploration noise — the single tunable for
/// how far over time exploration is correlated, and THE lever this experiment exists to turn
/// (bddap/rl#161). The crab explored with per-tick-INDEPENDENT Gaussian noise on ~38 joints at
/// 64 Hz, which can only jitter — it never holds a coordinated, periodic limb motion long
/// enough to earn progress reward, so a walking gait never bootstraps. Correlating the noise
/// over time makes sustained, gait-like limb sweeps the DEFAULT exploration, so the policy can
/// stumble into one and be rewarded.
///
/// 0.95 ⇒ correlation time constant −1/ln α ≈ 19.5 ticks ≈ 0.3 s at 64 Hz — a coherent push
/// lasting a meaningful fraction of a ~0.5–1 s stride, long enough for a chance-aligned
/// multi-joint sweep to persist and earn progress (α=0.9's ~0.15 s decorrelates inside one
/// stride). Tune by editing this constant and warm-resuming: α→1 lengthens the sweep (toward a
/// slow drift), α=0 recovers the old per-tick-independent noise. It is variance-preserving
/// (`x ← α·x + √(1−α²)·ε`, see [`OuNoise::next`]), so it changes only the temporal SMOOTHNESS of
/// exploration, never its MAGNITUDE (that stays the policy's learnable `log_std`) — keeping this
/// lever orthogonal to the entropy/sigma knobs 616 falsified.
const EXPLORE_CORRELATION: f32 = 0.95;

/// Temporally-correlated exploration noise: one variance-preserving AR(1) (discrete
/// Ornstein–Uhlenbeck) process per env, per joint, supplying the `ε` in the policy's
/// exploration draw `μ + σ·ε` (see [`sample_action`]). Successive draws are correlated over
/// ~[`EXPLORE_CORRELATION`] ticks, so exploration is a sustained limb sweep rather than per-tick
/// jitter — the lever for bootstrapping a gait (bddap/rl#161).
///
/// TRAINING-ROLLOUT only, by construction: it is owned by `TrainingState` and the eval/demo
/// policy path takes the policy MEAN (no `ε` at all — see `play::policy::PolicyController::act`),
/// so exploration noise cannot leak into the deployed/greedy action.
pub(crate) struct OuNoise {
    /// AR(1) state per env (`state[e]`), per joint — env `e`'s current correlated `ε`.
    state: Vec<[f32; ACTION_SIZE]>,
}

impl OuNoise {
    pub(crate) fn new(n_envs: usize) -> Self {
        Self {
            state: vec![[0.0; ACTION_SIZE]; n_envs],
        }
    }

    /// Re-seed env `e`'s noise to a fresh standard-normal draw — called when an episode
    /// (re)starts so the new episode's exploration begins uncorrelated with the last, with the
    /// correct N(0,1) marginal from its first tick.
    pub(crate) fn reset(&mut self, e: usize, rng: &mut StdRng) {
        for s in &mut self.state[e] {
            *s = next_standard_normal(rng);
        }
    }

    /// Advance env `e`'s AR(1) noise one tick and return the new `ε`: `x ← α·x + √(1−α²)·ε`,
    /// `ε ~ N(0,1)`, `α =` [`EXPLORE_CORRELATION`]. Variance-preserving — at stationarity
    /// `Var(x) = α²·1 + (1−α²)·1 = 1` — so the marginal is standard-normal at any α (keeping the
    /// per-step Gaussian log-prob the PPO update recomputes valid) while successive draws stay
    /// correlated.
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
    /// Approximate KL the policy moved this update (rollout → final policy). The
    /// target-KL guard ([`PpoConfig::target_kl`]) stops the update once this crosses
    /// the ceiling, so it should track ~`target_kl` on a healthy run and reveals a
    /// throttled (early-stopped) iteration when it sits at the ceiling.
    pub(crate) kl: f32,
    /// Optimizer steps actually applied this update. Equals
    /// `epochs × ceil(n/batch)` on a full update; fewer when the target-KL guard
    /// early-stopped the iteration (a visible signal the policy hit the trust region).
    pub(crate) steps: u32,
    /// Mean `|π_old_backend − π_old_cpu|`: how far the rollout's CPU-recorded behavior
    /// log-prob sits from the on-backend recompute the update actually uses. ~0 means
    /// the rollout (CPU) and update (GPU) backends agree; a large value is the
    /// importance-ratio-corrupting divergence the recompute neutralizes — so watching
    /// it catches a regression (a backend/precision change re-opening the gap).
    pub(crate) behavior_backend_div: f32,
    /// Non-finite returns SKIPPED by [`ReturnNormalizer::update`] this update (bddap/rl#167). A
    /// blown-up env feeding a NaN/Inf return is dropped from the running scale rather than
    /// poisoning it — but silently dropping it hid a diverging env behind a merely-wrong
    /// normalizer. Counted and surfaced here (loud when nonzero) so a divergence is caught at the
    /// boundary, not inferred later from drift. 0 on a healthy update.
    pub(crate) nonfinite_returns: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exploration-σ floor ramps linearly from the wide start to the refine end over the
    /// horizon, holds at the end past it, and degenerates to the end value for a zero horizon
    /// (schedule off) — the shape the gait-bootstrap lever (bddap/rl#161) depends on.
    #[test]
    fn log_std_floor_anneals_start_to_end_then_holds() {
        let config = PpoConfig {
            log_std_floor_start: -0.7,
            log_std_floor_end: -2.0,
            log_std_anneal_ticks: 1000,
            ..PpoConfig::default()
        };
        assert!((config.log_std_floor(0) - (-0.7)).abs() < 1e-6, "wide at the epoch");
        assert!(
            (config.log_std_floor(500) - (-1.35)).abs() < 1e-6,
            "linear midpoint"
        );
        assert!((config.log_std_floor(1000) - (-2.0)).abs() < 1e-6, "refine at horizon");
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
        env_a.bootstrap = Some(NormalizedValue(2.0));
        let mut env_b = RolloutBuffer::new();
        env_b.push(t(0.0, 1.0, StepEnd::Continues));
        env_b.push(t(0.0, 1.0, StepEnd::Terminal));

        // Identity scale (no returns observed ⇒ μ=0, σ=1): GAE is in raw units, so
        // these hand-computed expectations are the un-normalized values.
        let id = ReturnNormalizer::new();
        // Per-env, hand-computed: A bootstraps from ITS next value (2.0).
        let (adv_a, _) = compute_gae(&env_a, gamma, lambda, &id);
        assert!((adv_a[1].0 - 1.5).abs() < 1e-6, "A[1]: {}", adv_a[1].0);
        assert!((adv_a[0].0 - 1.125).abs() < 1e-6, "A[0]: {}", adv_a[0].0);

        // Naive concatenated sweep: A's last step bootstraps from B's value.
        let mut concat = RolloutBuffer::new();
        for tr in env_a.transitions.iter().chain(env_b.transitions.iter()) {
            concat.push(tr.clone());
        }
        // concat's tail is Terminal (env_b's last) → bootstrap self-zeroed, so the
        // default `None` is correct; the corruption it pins is in env_a's interior step.
        let (adv_concat, _) = compute_gae(&concat, gamma, lambda, &id);
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
        let (adv_term, _) = compute_gae(&terminal, gamma, lambda, &id);

        let mut truncated = RolloutBuffer::new();
        truncated.push(Transition {
            end: StepEnd::Truncated,
            ..t(reward, value, StepEnd::Continues)
        });
        let (adv_trunc, ret_trunc) = compute_gae(&truncated, gamma, lambda, &id);

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

    /// The non-terminal (`Continues`) tail must bootstrap from the SUCCESSOR value
    /// `V(s_{last+1})` the buffer carries (`RolloutBuffer::bootstrap`), NOT the tail
    /// step's own value. The one-tick `Pending` phasing once made the trailing bootstrap
    /// re-read the tail's own obs (rl#174), degenerating the tail delta to
    /// `r − (1−γ)·V(s_last)`; with the successor value bound to the buffer the tail delta
    /// is the textbook `r + γ·V(s_{last+1}) − V(s_last)`.
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
        let buggy = reward + gamma * own_v - own_v; // rl#174: bootstrapped its OWN value
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
        buf_raw.bootstrap = Some(NormalizedValue(last_value_raw));
        let (adv_raw, ret_raw) = compute_gae(&buf_raw, gamma, lambda, &id);

        // A normalizer with a non-trivial, positive affine scale (built from a spread
        // of returns so μ and σ are both non-zero).
        let mut ret_norm = ReturnNormalizer::new();
        let skipped = ret_norm.update(&[-300.0, -100.0, 50.0, -250.0, 0.0, -180.0]);
        assert_eq!(skipped, 0, "all returns here are finite — none skipped");
        let mu = ret_norm.mean();
        let sigma = ret_norm.std();
        assert!(sigma > 1.0, "test needs a non-trivial scale, got σ={sigma}");
        assert!(mu.abs() > 1.0, "test needs a non-zero shift, got μ={mu}");

        // The value head, trained against this scale, would emit normalized values.
        let mut buf_norm = RolloutBuffer::new();
        for &(reward, value) in &raw {
            buf_norm.push(t(reward, (value - mu) / sigma, StepEnd::Continues));
        }
        buf_norm.bootstrap = Some(NormalizedValue((last_value_raw - mu) / sigma));
        let (adv_norm, ret_norm_out) = compute_gae(&buf_norm, gamma, lambda, &ret_norm);

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
    /// bddap/rl#167: `update` must REPORT how many non-finite returns it skipped (so a diverging
    /// env surfaces loudly) while still folding the finite ones into the scale unperturbed — the
    /// skip is a fail-safe, but no longer a silent one.
    #[test]
    fn return_normalizer_reports_skipped_nonfinite_returns() {
        let mut norm = ReturnNormalizer::new();
        // Two NaN/Inf returns among the finite ones.
        let skipped = norm.update(&[1.0, f32::NAN, 2.0, f32::INFINITY, 3.0]);
        assert_eq!(skipped, 2, "both non-finite returns must be reported as skipped");

        // The scale reflects ONLY the finite returns (1,2,3 → mean 2), as if the bad ones never came.
        let mut clean = ReturnNormalizer::new();
        let none = clean.update(&[1.0, 2.0, 3.0]);
        assert_eq!(none, 0, "all-finite returns skip nothing");
        assert!((norm.mean() - clean.mean()).abs() < 1e-9, "the skip must not perturb the scale");
        assert!((norm.std() - clean.std()).abs() < 1e-9, "the skip must not perturb the std");
    }

    #[test]
    fn return_normalizer_checkpoint_round_trips() {
        let mut norm = ReturnNormalizer::new();
        // A spread of returns spanning the large-magnitude regime.
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
