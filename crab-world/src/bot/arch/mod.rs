//! The architecture registry (bddap/rl#200): ONE polymorphic policy seam, N architecture
//! leaves behind it. [`AnyBrain`] is the seam — an enum `Module`, one variant per living
//! architecture; adding an arch is a leaf module + a variant + an `init` arm, and culling
//! one is deleting the variant (the compiler walks you to every match site). Deliberately
//! NOT a trait: the arch is chosen at runtime (checkpoint tag), so every long-lived holder
//! would need an enum wrapper anyway, and the load-bearing same-arch-two-backends pairings
//! (`GpuLearner`'s GPU↔CPU mirror, `InferenceCachedBrain`'s train↔valid clone) are trivial
//! with `AnyBrain<_>` at two backends but inexpressible with a flat `Brain<B>` generic.
//!
//! Fixed v1 contract for every leaf: feedforward, single-frame obs, diagonal Gaussian with
//! PER-ROW `log_std` (`[rows, ACTION_SIZE]` — a state-independent leaf broadcasts, so a
//! state-dependent-σ arch is admissible without touching the contract). Recurrence /
//! obs-history / non-Gaussian heads change shared plumbing the seam pins, so they are a
//! future seam-extension epic, not a leaf.

pub mod mlp256;

use burn::module::Module;
use burn::prelude::*;
use burn::record::{Recorder, RecorderError};

pub use mlp256::Mlp256;

/// Bounds the learnable log-std so entropy can't diverge or collapse. Applied ONCE, in
/// [`GaussianHead::new`] — leaves emit RAW heads and never clamp. `LOG_STD_MIN` is also the
/// resting/refine floor the exploration schedule anneals back down to once the wide-early
/// window elapses (see `PpoConfig::log_std_floor`). exp(-2) ≈ 0.14 (focused),
/// exp(0.5) ≈ 1.65 (wide).
pub(crate) const LOG_STD_MIN: f32 = -2.0;
pub(crate) const LOG_STD_MAX: f32 = 0.5;

/// Architecture identity — the registry's key. In process it is this enum (an unregistered
/// arch is unrepresentable); on disk / on the wire it is ALWAYS its stable kebab-case
/// string via [`ArchId::name`]/[`TryFrom<String>`], NEVER a serde enum: bincode-1 (the
/// workspace encoder) encodes enum variants by index and ignores rename attrs, so a bare
/// enum field would re-map every tagged file when a variant is culled. Deliberately NO
/// serde derive — the checkpoint envelope encodes a plain `String` and validates through
/// `TryFrom` explicitly (so an unknown arch is attributed by name), and a derive would be
/// a second, unused encoding waiting to be reached for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchId {
    Mlp256,
}

impl ArchId {
    /// The stable on-disk / CLI name. Kebab-case, never reused after a cull. Keep
    /// [`TryFrom<String>`] below in lockstep — it is the inverse of this table.
    pub fn name(self) -> &'static str {
        match self {
            Self::Mlp256 => "mlp256",
        }
    }
}

impl std::fmt::Display for ArchId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl TryFrom<String> for ArchId {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.as_str() {
            "mlp256" => Ok(Self::Mlp256),
            other => Err(format!("unknown policy architecture {other:?}")),
        }
    }
}

impl From<ArchId> for String {
    fn from(arch: ArchId) -> String {
        arch.name().to_string()
    }
}

/// The registry: one variant per living architecture. Culling an architecture = deleting
/// its variant + leaf module; the compiler walks you to every match site. The derive makes
/// this a `Module`/`AutodiffModule` (so `.valid()`, optimizer adaptors, and gradient
/// collection all work over the enum), and every shared holder (`TrainingState`,
/// `GpuLearner`, `Policy`) stores this type concretely — no generics over the leaf.
#[derive(Module, Debug)]
pub enum AnyBrain<B: Backend> {
    Mlp256(Mlp256<B>),
}

impl<B: Backend> AnyBrain<B> {
    /// Fresh (untrained) weights for `arch` — the one construction chokepoint. Every
    /// registered architecture is an arm here.
    pub fn init(arch: ArchId, device: &B::Device) -> Self {
        match arch {
            ArchId::Mlp256 => Self::Mlp256(Mlp256::new(device)),
        }
    }

    pub fn arch(&self) -> ArchId {
        match self {
            Self::Mlp256(_) => ArchId::Mlp256,
        }
    }

    /// RAW policy heads: action means (leaf-bounded to [-1, 1]) and PER-ROW `log_std`
    /// (`[rows, ACTION_SIZE]`; a state-independent leaf broadcasts its learned vector).
    /// The `log_std` is UN-floored and UN-clamped — sampling/log-prob/entropy math accepts
    /// only a [`GaussianHead`], whose constructor is the single floor/clamp site, so an
    /// unfloored σ can't reach the math by construction.
    pub fn policy(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self {
            Self::Mlp256(m) => m.policy(obs),
        }
    }

    /// Critic value estimate, one scalar per batch row.
    pub fn value(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Self::Mlp256(m) => m.value(obs),
        }
    }

    /// The `(obs_dim, action_dim)` the loaded weights were built for (post-`load_record`
    /// these reflect the checkpoint, not the compiled rig — the rig-fit guard reads them).
    pub fn io_dims(&self) -> (usize, usize) {
        match self {
            Self::Mlp256(m) => m.io_dims(),
        }
    }

    /// Record this brain via `recorder` as its LEAF record — NEVER the derived
    /// `AnyBrainRecord`: that enum record bincode-encodes its variant by index (culling a
    /// variant would silently re-map every old file) and burn's generated `load_record`
    /// panics cross-variant. A naive `into_record()` save would prefix an enum index onto
    /// `brain.bin` while every in-repo save/load test stayed green — and every fleet
    /// checkpoint would stop loading via the quiet-rest-pose path. The golden-file test in
    /// `crate::policy` pins this. All brain I/O goes through this pair.
    pub fn record_leaf<R: Recorder<B>>(
        &self,
        recorder: &R,
        args: R::RecordArgs,
    ) -> Result<R::RecordOutput, RecorderError> {
        match self {
            Self::Mlp256(m) => recorder.record(m.clone().into_record(), args),
        }
    }

    /// Load THIS brain's architecture's leaf record via `recorder`, replacing the weights.
    /// Consumes `self` (callers that need the old weights on failure clone first — the
    /// tensors are refcounted, so a clone is cheap). Which record type to decode comes from
    /// the variant, so a checkpoint can never be blind-loaded into a guessed architecture;
    /// increment 2's envelope tag will pick the variant before this call.
    pub fn load_leaf_record<R: Recorder<B>>(
        self,
        recorder: &R,
        args: R::LoadArgs,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        match self {
            Self::Mlp256(m) => Ok(Self::Mlp256(m.load_record(recorder.load(args, device)?))),
        }
    }
}

/// A floored, clamped diagonal-Gaussian action distribution — the ONLY input the
/// sampling/log-prob/entropy math accepts. The σ-floor and the [`LOG_STD_MIN`]/
/// [`LOG_STD_MAX`] clamp are applied HERE, once, so a raw leaf head can't reach the math
/// unclamped at any call site (a missed clamp would silently bias the PPO ratio; the type
/// forbids it, not a convention).
///
/// `log_std_floor` is the LOWER clamp bound — the minimum exploration spread. It is the
/// lever the training schedule raises early to FORCE σ wide (the learned `log_std` sits
/// below it, so the clamp overrides it) and anneals back down to [`LOG_STD_MIN`] for
/// refinement (see `PpoConfig::log_std_floor`). Pass [`LOG_STD_MIN`] for the
/// unforced/default bound; eval takes the policy MEAN and never builds a head, so the
/// floor never reaches a deployed action. The floor is itself clamped into
/// `[LOG_STD_MIN, LOG_STD_MAX]` so a misconfigured schedule can't widen past the
/// architectural bound.
pub(crate) struct GaussianHead<B: Backend> {
    means: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
}

impl<B: Backend> GaussianHead<B> {
    /// Floor + clamp the raw heads from [`AnyBrain::policy`] (its `(means, log_std)`
    /// tuple is this argument, so the two compose without naming intermediates).
    pub(crate) fn new(raw: (Tensor<B, 2>, Tensor<B, 2>), log_std_floor: f32) -> Self {
        let (means, log_std) = raw;
        let lo = log_std_floor.clamp(LOG_STD_MIN, LOG_STD_MAX);
        let log_std = log_std.clamp(lo as f64, LOG_STD_MAX as f64);
        Self { means, log_std }
    }

    /// Sample one DRIVE per row from the Gaussian: `dᵢ = μᵢ + σᵢ·εᵢ`, UN-clamped.
    ///
    /// Returns the raw pre-clamp drives — the random variable the policy actually drew, and
    /// the quantity the reward's metabolic tax and the PPO log-prob are both taken over. The
    /// ±1 clamp that bounds the sim's torque command is applied by the caller, so the
    /// unbounded drive survives for the tax to bite on saturation (a `|d|≫1` drive that
    /// slams a joint onto its rail). Clamping here would erase that overshoot.
    ///
    /// `noise` is the standard-normal exploration `ε` the caller drew — the temporally-
    /// correlated draw from `OuNoise` (bddap/rl#161), one row per env. It is marginally
    /// N(0,1), so the per-step Gaussian log-prob the PPO update recomputes is unchanged —
    /// only the temporal correlation of successive `ε`s differs, which is what makes a
    /// coordinated gait discoverable.
    pub(crate) fn sample(&self, noise: Tensor<B, 2>) -> Tensor<B, 2> {
        self.means.clone() + noise * self.log_std.clone().exp()
    }

    /// Diagonal-Gaussian log-prob of each action ROW:
    /// `Σ_d [ -0.5·((aᵈ-μᵈ)/σᵈ)² - ln σᵈ - 0.5·ln(2π) ]`. THE one log-prob formula — the
    /// rollout's behavior log-prob and the PPO update's `π_old`/`π_new` all come from here,
    /// so the importance ratio can't drift on a formula mismatch. Log-space throughout to
    /// avoid dividing by a tiny variance. The autodiff graph flows through the heads for
    /// `π_new`; the caller detaches for `π_old`.
    pub(crate) fn log_prob_rows(&self, actions: Tensor<B, 2>) -> Tensor<B, 1> {
        let scaled_diff = (actions - self.means.clone()) / self.log_std.clone().exp();
        let half_log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
        (scaled_diff.powf_scalar(2.0).neg() * 0.5 - self.log_std.clone() - half_log_2pi)
            .sum_dim(1)
            .flatten::<1>(0, 1)
    }

    /// Mean per-dimension Gaussian entropy, `ln σ + ½·ln(2πe)` averaged over every
    /// row × dim — the PPO update's entropy bonus. For a state-independent σ every row is
    /// identical, so this equals the old per-dim mean (value and gradient); for a
    /// state-dependent-σ arch it becomes the batch-mean entropy, which is the estimator
    /// PPO wants.
    pub(crate) fn entropy(&self) -> Tensor<B, 1> {
        (self.log_std.clone() + 0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln())
            .mean()
    }
}
