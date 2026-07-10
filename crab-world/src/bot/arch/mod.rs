pub mod mlp512x3;

use burn::module::Module;
use burn::prelude::*;
use burn::record::{Recorder, RecorderError};

pub use mlp512x3::Mlp512x3;

// These consts and `GaussianHead` are used only from `training`, whose liveness root
// is the wgpu-gated learner (see `training`'s module allow) — dead without `wgpu`.
// They stay HERE because the clamp contract couples to the policy nets' raw output
// (see `mlp512x3`).
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
pub(crate) const LOG_STD_MIN: f32 = -2.0;
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
pub(crate) const LOG_STD_MAX: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchId {
    Mlp512x3,
}

impl ArchId {
    /// Every living architecture. `TryFrom<String>` inverts [`Self::name`] through this
    /// list, so the name table lives in exactly one place; a new variant that misses
    /// this list is caught by `roundtrips_every_arch` below.
    pub const ALL: &'static [ArchId] = &[Self::Mlp512x3];

    /// The arch a fresh start gets when `--arch` is omitted (the sole survivor of the
    /// 5b cull; the trainer's CLI help states it in prose — keep that in sync).
    pub const DEFAULT: ArchId = Self::Mlp512x3;

    /// The stable on-disk / CLI name. Kebab-case, never reused after a cull.
    pub fn name(self) -> &'static str {
        match self {
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

#[derive(Module, Debug)]
pub enum AnyBrain<B: Backend> {
    Mlp512x3(Mlp512x3<B>),
}

impl<B: Backend> AnyBrain<B> {
    pub fn init(arch: ArchId, device: &B::Device) -> Self {
        match arch {
            ArchId::Mlp512x3 => Self::Mlp512x3(Mlp512x3::new(device)),
        }
    }

    pub fn arch(&self) -> ArchId {
        match self {
            Self::Mlp512x3(_) => ArchId::Mlp512x3,
        }
    }

    pub fn policy(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self {
            Self::Mlp512x3(m) => m.policy(obs),
        }
    }

    pub fn value(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Self::Mlp512x3(m) => m.value(obs),
        }
    }

    pub fn io_dims(&self) -> (usize, usize) {
        match self {
            Self::Mlp512x3(m) => m.io_dims(),
        }
    }

    pub fn record_leaf<R: Recorder<B>>(
        &self,
        recorder: &R,
        args: R::RecordArgs,
    ) -> Result<R::RecordOutput, RecorderError> {
        match self {
            Self::Mlp512x3(m) => recorder.record(m.clone().into_record(), args),
        }
    }

    pub fn load_leaf_record<R: Recorder<B>>(
        self,
        recorder: &R,
        args: R::LoadArgs,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        match self {
            Self::Mlp512x3(m) => Ok(Self::Mlp512x3(m.load_record(recorder.load(args, device)?))),
        }
    }
}

#[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
pub(crate) struct GaussianHead<B: Backend> {
    means: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
}

#[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
impl<B: Backend> GaussianHead<B> {
    pub(crate) fn new(raw: (Tensor<B, 2>, Tensor<B, 2>), log_std_floor: f32) -> Self {
        let (means, log_std) = raw;
        let lo = log_std_floor.clamp(LOG_STD_MIN, LOG_STD_MAX);
        let log_std = log_std.clamp(lo as f64, LOG_STD_MAX as f64);
        Self { means, log_std }
    }

    pub(crate) fn sample(&self, noise: Tensor<B, 2>) -> Tensor<B, 2> {
        self.means.clone() + noise * self.log_std.clone().exp()
    }

    pub(crate) fn log_prob_rows(&self, actions: Tensor<B, 2>) -> Tensor<B, 1> {
        let scaled_diff = (actions - self.means.clone()) / self.log_std.clone().exp();
        let half_log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
        (scaled_diff.powf_scalar(2.0).neg() * 0.5 - self.log_std.clone() - half_log_2pi)
            .sum_dim(1)
            .flatten::<1>(0, 1)
    }

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
