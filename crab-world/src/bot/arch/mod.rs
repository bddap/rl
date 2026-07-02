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
pub mod mlp512x3;

use burn::module::Module;
use burn::prelude::*;
use burn::record::{Recorder, RecorderError};

pub use mlp256::Mlp256;
pub use mlp512x3::Mlp512x3;

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
    Mlp512x3,
}

impl ArchId {
    /// Every living architecture. `TryFrom<String>` inverts [`Self::name`] through this
    /// list, so the name table lives in exactly one place; a new variant that misses
    /// this list is caught by `roundtrips_every_arch` below.
    pub const ALL: &'static [ArchId] = &[Self::Mlp256, Self::Mlp512x3];

    /// The arch a fresh start gets when `--arch` is omitted (the founding run's; the
    /// trainer's CLI help states it in prose — keep that in sync).
    pub const DEFAULT: ArchId = Self::Mlp256;

    /// The stable on-disk / CLI name. Kebab-case, never reused after a cull.
    pub fn name(self) -> &'static str {
        match self {
            Self::Mlp256 => "mlp256",
            Self::Mlp512x3 => "mlp512x3",
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
        Self::ALL
            .iter()
            .copied()
            .find(|a| a.name() == s)
            .ok_or_else(|| {
                let known: Vec<_> = Self::ALL.iter().map(|a| a.name()).collect();
                format!(
                    "unknown policy architecture {s:?} (known: {})",
                    known.join(", ")
                )
            })
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
    Mlp512x3(Mlp512x3<B>),
}

impl<B: Backend> AnyBrain<B> {
    /// Fresh (untrained) weights for `arch` — the one construction chokepoint. Every
    /// registered architecture is an arm here.
    pub fn init(arch: ArchId, device: &B::Device) -> Self {
        match arch {
            ArchId::Mlp256 => Self::Mlp256(Mlp256::new(device)),
            ArchId::Mlp512x3 => Self::Mlp512x3(Mlp512x3::new(device)),
        }
    }

    pub fn arch(&self) -> ArchId {
        match self {
            Self::Mlp256(_) => ArchId::Mlp256,
            Self::Mlp512x3(_) => ArchId::Mlp512x3,
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
            Self::Mlp512x3(m) => m.policy(obs),
        }
    }

    /// Critic value estimate, one scalar per batch row.
    pub fn value(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Self::Mlp256(m) => m.value(obs),
            Self::Mlp512x3(m) => m.value(obs),
        }
    }

    /// The `(obs_dim, action_dim)` the loaded weights were built for (post-`load_record`
    /// these reflect the checkpoint, not the compiled rig — the rig-fit guard reads them).
    pub fn io_dims(&self) -> (usize, usize) {
        match self {
            Self::Mlp256(m) => m.io_dims(),
            Self::Mlp512x3(m) => m.io_dims(),
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
            Self::Mlp512x3(m) => recorder.record(m.clone().into_record(), args),
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
            Self::Mlp512x3(m) => Ok(Self::Mlp512x3(m.load_record(recorder.load(args, device)?))),
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

#[cfg(test)]
mod tests {
    use super::super::actuator::ACTION_SIZE;
    use super::super::sensor::OBS_SIZE;
    use super::{AnyBrain, ArchId};
    use crate::training::TrainBackend;
    use burn::prelude::*;

    /// Every registered arch's name parses back to itself — catches a new variant
    /// missing from `ArchId::ALL` (which would make its checkpoints unloadable) or a
    /// name collision between variants.
    #[test]
    fn roundtrips_every_arch() {
        for &arch in ArchId::ALL {
            assert_eq!(ArchId::try_from(arch.name().to_string()), Ok(arch));
        }
    }

    /// Every registered arch honors the v1 seam contract on a fresh init: rig-matched
    /// `io_dims`, `[rows, ACTION_SIZE]` heads with means in the tanh bound, per-row
    /// `log_std`, and a `[rows, 1]` value — so a leaf that broadcasts wrong or forgets
    /// the mean bound fails HERE, not as NaNs mid-run.
    #[test]
    fn every_arch_honors_the_seam_contract() {
        let device = Default::default();
        let rows = 3;
        let obs = Tensor::<TrainBackend, 2>::zeros([rows, OBS_SIZE], &device);
        for &arch in ArchId::ALL {
            let brain = AnyBrain::<TrainBackend>::init(arch, &device);
            assert_eq!(brain.arch(), arch);
            assert_eq!(brain.io_dims(), (OBS_SIZE, ACTION_SIZE), "{arch}");
            let (means, log_std) = brain.policy(obs.clone());
            assert_eq!(means.dims(), [rows, ACTION_SIZE], "{arch}");
            assert_eq!(log_std.dims(), [rows, ACTION_SIZE], "{arch}");
            let peak = means.abs().max().into_scalar();
            assert!(peak <= 1.0, "{arch}: means outside the tanh bound: {peak}");
            assert_eq!(brain.value(obs.clone()).dims(), [rows, 1], "{arch}");
        }
    }
}
