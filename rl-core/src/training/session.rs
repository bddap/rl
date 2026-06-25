//! The RL training loop, integrated with the Bevy game loop as ECS systems.

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
use burn::record::{BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
// The log macros come from `tracing` directly, not `bevy::prelude`: the headless
// trainer builds bevy without its `bevy_log` default feature (no renderer), so the
// prelude no longer re-exports `warn!`/`info!`. tracing (a direct dep) carries the same
// macros and is present in every build. An explicit import shadows the prelude glob, so
// this is unambiguous in render builds too.
use tracing::{info, warn};

use crate::TrainConfig;
use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{
    CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, random_spawn_rotation,
};
use crate::bot::brain::CrabBrain;
use crate::bot::sensor::{CrabObservation, CrabTargets, OBS_SIZE};
use crate::bot::{CrabRescued, CrabSpawns, respawn_crab_rotated};

use super::algorithm::{
    NormalizedValue, PpoConfig, PpoMetrics, ReturnNormalizer, ReturnNormalizerData, RolloutBuffer,
    StepEnd, Transition, compute_gae, compute_log_prob, sample_action,
};

/// Running observation normalizer using Welford's online algorithm.
/// Normalizes observations to zero mean, unit variance.
///
/// Variance is NOT stored: it is `m2 / (count-1)`, derived on demand in
/// [`Self::variance`]. Keeping a separate `var` array would be a second source
/// of truth that can silently drift from `m2`/`count` across save/merge.
pub(crate) struct ObsNormalizer {
    mean: [f64; OBS_SIZE],
    m2: [f64; OBS_SIZE],    // sum of squared differences from mean
    count: [u64; OBS_SIZE], // per-element count (NaN-skipped elements don't inflate others)
    clip: f32,              // max absolute normalized value
}

/// Serde-friendly mirror of `ObsNormalizer` (arrays > 32 don't auto-derive). Also
/// the form the learner snapshots to / merges from across rollout threads (passed
/// in-process, not over a wire) and the on-disk checkpoint format, so it is
/// `pub(crate)`. No `var` field, for the same reason `ObsNormalizer` stores none.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ObsNormalizerData {
    mean: Vec<f64>,
    m2: Vec<f64>,
    count: Vec<u64>,
    clip: f32,
}

impl ObsNormalizer {
    fn to_data(&self) -> ObsNormalizerData {
        ObsNormalizerData {
            mean: self.mean.to_vec(),
            m2: self.m2.to_vec(),
            count: self.count.to_vec(),
            clip: self.clip,
        }
    }

    fn from_data(d: ObsNormalizerData) -> Option<Self> {
        if d.mean.len() != OBS_SIZE || d.m2.len() != OBS_SIZE || d.count.len() != OBS_SIZE {
            warn!(
                "Normalizer size mismatch: expected {OBS_SIZE}, got {}",
                d.mean.len()
            );
            return None;
        }
        if d.clip <= 0.0 || d.m2.iter().any(|&v| v < 0.0) {
            warn!("Normalizer contains invalid values (clip <= 0 or negative M2)");
            return None;
        }
        let mut n = Self::new(d.clip);
        n.mean.copy_from_slice(&d.mean);
        n.m2.copy_from_slice(&d.m2);
        n.count.copy_from_slice(&d.count);
        Some(n)
    }
}

/// Max absolute normalized observation value (Welford clip). One source of truth
/// for every `ObsNormalizer::new`, so the learner's master and a rollout thread's
/// per-horizon increment share the same clip and can't drift.
pub(crate) const NORMALIZER_CLIP: f32 = 5.0;

impl ObsNormalizer {
    pub(crate) fn new(clip: f32) -> Self {
        Self {
            mean: [0.0; OBS_SIZE],
            m2: [0.0; OBS_SIZE],
            count: [0; OBS_SIZE],
            clip,
        }
    }

    /// Per-element variance `m2 / (count-1)`, the value `normalize_frozen` scales
    /// by. Defaults to 1.0 for an element seen at most once (no spread estimate
    /// yet), matching the unit-variance starting point a fresh normalizer used.
    fn variance(&self, i: usize) -> f64 {
        if self.count[i] > 1 {
            (self.m2[i] / (self.count[i] as f64 - 1.0)).max(0.0)
        } else {
            1.0
        }
    }

    /// Fold one finite sample of element `i` into the running (count, mean, m2)
    /// — the inner Welford step, shared by the full normalizer and the worker's
    /// per-horizon increment accumulator so they cannot compute it differently.
    fn observe_element(&mut self, i: usize, raw: f32) {
        if !raw.is_finite() {
            return;
        }
        self.count[i] += 1;
        let n = self.count[i] as f64;
        let x = raw as f64;
        let delta = x - self.mean[i];
        self.mean[i] += delta / n;
        let delta2 = x - self.mean[i];
        self.m2[i] += delta * delta2;
    }

    /// Update running stats, then return the normalized observation.
    pub(crate) fn normalize(&mut self, obs: &[f32; OBS_SIZE]) -> [f32; OBS_SIZE] {
        for (i, &raw) in obs.iter().enumerate() {
            self.observe_element(i, raw);
        }
        self.normalize_frozen(obs)
    }

    /// Fold one observation into the running stats WITHOUT normalizing it. The
    /// worker's per-horizon increment uses this: it must count exactly the same
    /// samples the master sees this horizon, but the normalized value is produced
    /// by the master (with the full baseline+horizon stats), not by the increment.
    pub(crate) fn observe(&mut self, obs: &[f32; OBS_SIZE]) {
        for (i, &raw) in obs.iter().enumerate() {
            self.observe_element(i, raw);
        }
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
            let std = (self.variance(i) as f32).sqrt().max(1e-6);
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
        if let Err(e) = atomic_write(path, &bytes) {
            warn!("Failed to write normalizer to {}: {e}", path.display());
        }
    }

    /// Snapshot the running stats as the serde mirror. Used to hand the master's
    /// stats to a rollout thread, ship a thread's increment back, and persist the
    /// checkpoint.
    pub(crate) fn snapshot(&self) -> ObsNormalizerData {
        self.to_data()
    }

    /// Replace this normalizer's stats with `data` (e.g. the learner's merged
    /// master, handed to a rollout thread before its next rollout). Returns false on
    /// a size/validity mismatch, leaving self unchanged.
    pub(crate) fn load_snapshot(&mut self, data: ObsNormalizerData) -> bool {
        match Self::from_data(data) {
            Some(n) => {
                *self = n;
                true
            }
            None => false,
        }
    }

    /// Parallel Welford merge: fold another accumulator's per-element
    /// (count, mean, M2) into this one. This is the exact combination of two
    /// INDEPENDENT streams — so it is only valid when `other` shares no samples
    /// with `self`. The in-process path upholds that by merging a per-horizon
    /// INCREMENT (only the samples this iteration added, never the snapshot baseline
    /// the master already counted); merging a cumulative snapshot would double-count
    /// the baseline. Per element because the NaN-skip lets counts differ across
    /// elements; variance is derived from the merged M2 on demand.
    pub(crate) fn merge(&mut self, other: &ObsNormalizerData) {
        if other.mean.len() != OBS_SIZE {
            warn!("normalizer merge: size mismatch, skipping");
            return;
        }
        for i in 0..OBS_SIZE {
            let na = self.count[i] as f64;
            let nb = other.count[i];
            if nb == 0 {
                continue;
            }
            let nb = nb as f64;
            let total = na + nb;
            let delta = other.mean[i] - self.mean[i];
            let mean = self.mean[i] + delta * nb / total;
            let m2 = self.m2[i] + other.m2[i] + delta * delta * na * nb / total;
            self.count[i] += other.count[i];
            self.mean[i] = mean;
            self.m2[i] = m2;
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

/// The Adam optimizer over a `CrabBrain` on backend `B`. Generic over the backend so
/// the one PPO update ([`ppo_update_core`]) serves the live GPU learner, the GPU/CPU
/// `bench-update` comparison, and the CPU-backed update test from one code path.
type CrabOpt<B> = OptimizerAdaptor<Adam, CrabBrain<B>, B>;

/// Build the learner's Adam optimizer (global grad-norm clip 0.5). The ONE source of
/// the optimizer construction — the live `GpuLearner`, the `bench-update` harness, and
/// the CPU-backed update test all call this, so the clip constant can't silently drift
/// between paths meant to be identical.
pub(crate) fn crab_optimizer<B: AutodiffBackend>() -> CrabOpt<B> {
    AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
        .init()
}

/// Format version of the [`OPTIMIZER_FILENAME`] envelope. Bumped whenever the serialized
/// layout changes; a file tagged with any other version is ignored on load (→ cold
/// moments), never deserialized blind — so both an older checkpoint (which has no such
/// file at all) and one from a future format resume safely instead of erroring. v1 is
/// burn's Adam record (per-param m/v + step `time`) under `FullPrecisionSettings`, wrapped
/// in this versioned bincode envelope.
#[cfg(any(feature = "wgpu", test))]
const OPTIMIZER_FORMAT_VERSION: u32 = 1;

/// On-disk envelope for the optimizer state: a version tag wrapping the burn optimizer
/// record's bytes. The bytes are device-independent — the `FullPrecisionSettings` recorder
/// reads the moment tensors off whatever device the optimizer lives on (the GPU, in
/// production) into host floats on save, and uploads them back on load — so one file
/// restores onto whichever device the next learner brings up.
#[cfg(any(feature = "wgpu", test))]
#[derive(Serialize, Deserialize)]
struct OptimizerCheckpoint {
    version: u32,
    record: Vec<u8>,
}

/// Persist an Adam optimizer's state (per-param first/second moments + step) to `path`,
/// atomically and version-tagged. Generic over the backend so the live GPU learner and the
/// CPU round-trip test serialize through ONE path — no save/load drift. The
/// `FullPrecisionSettings` recorder reads the moment tensors back off the optimizer's device
/// into host floats, exactly as the brain bridge does for weights. Best-effort: any failure
/// is logged, not fatal — the run continues, a resume just falls back to cold moments.
#[cfg(any(feature = "wgpu", test))]
fn save_optimizer<B: AutodiffBackend>(optimizer: &CrabOpt<B>, path: &Path) {
    let record = optimizer.to_record();
    let bytes = match BinBytesRecorder::<FullPrecisionSettings>::default().record(record, ()) {
        Ok(b) => b,
        Err(e) => {
            warn!("Failed to serialize Adam optimizer state: {e}");
            return;
        }
    };
    let envelope = OptimizerCheckpoint {
        version: OPTIMIZER_FORMAT_VERSION,
        record: bytes,
    };
    match bincode::serialize(&envelope) {
        Ok(encoded) => {
            if let Err(e) = atomic_write(path, &encoded) {
                warn!(
                    "Failed to write Adam optimizer state to {}: {e}",
                    path.display()
                );
            }
        }
        Err(e) => warn!("Failed to encode Adam optimizer state: {e}"),
    }
}

/// Load an Adam optimizer state saved by [`save_optimizer`] onto `device`, returning the
/// optimizer with its moments + step restored. Returns the `cold` optimizer UNCHANGED — no
/// error — when the file is absent (a pre-rl#60 checkpoint, or a fresh run), unreadable,
/// corrupt, or tagged with a version this build doesn't recognize; all of these resume cold,
/// which is correct, just without the warm-momentum head start. The per-parameter keys line
/// up across the round trip because the resumed brain restores the SAME `ParamId`s from its
/// own record, so each moment lands back on its parameter.
#[cfg(any(feature = "wgpu", test))]
fn load_optimizer<B: AutodiffBackend>(
    cold: CrabOpt<B>,
    path: &Path,
    device: &B::Device,
) -> CrabOpt<B> {
    let Ok(bytes) = std::fs::read(path) else {
        // Absent file is the EXPECTED case for a pre-rl#60 checkpoint, so info, not warn —
        // a warm-continue of an older policy with cold moments is normal, not a fault.
        info!(
            "No Adam optimizer state at {} — starting the optimizer cold",
            path.display()
        );
        return cold;
    };
    let envelope: OptimizerCheckpoint = match bincode::deserialize(&bytes) {
        Ok(e) => e,
        Err(e) => {
            warn!(
                "Corrupt Adam optimizer state at {}: {e} — starting the optimizer cold",
                path.display()
            );
            return cold;
        }
    };
    if envelope.version != OPTIMIZER_FORMAT_VERSION {
        warn!(
            "Adam optimizer state at {} is format v{}, this build writes v{} — starting cold",
            path.display(),
            envelope.version,
            OPTIMIZER_FORMAT_VERSION
        );
        return cold;
    }
    match BinBytesRecorder::<FullPrecisionSettings>::default().load(envelope.record, device) {
        Ok(record) => {
            info!("Restored Adam optimizer state from {}", path.display());
            cold.load_record(record)
        }
        Err(e) => {
            warn!(
                "Failed to decode Adam optimizer record at {}: {e} — starting cold",
                path.display()
            );
            cold
        }
    }
}

struct MetricsLogger {
    episode_file: std::fs::File,
}

impl MetricsLogger {
    /// `dir` is where `episodes.csv` lands. A rollout thread passes its own scratch
    /// dir so K threads don't clobber one shared CSV; the learner's host uses "tmp"
    /// (the established location the plotting scripts read).
    fn new(dir: &Path) -> Self {
        std::fs::create_dir_all(dir).expect("failed to create metrics dir");

        let ep_path = dir.join("episodes.csv");
        let mut episode_file =
            std::fs::File::create(&ep_path).expect("failed to create episodes.csv");
        writeln!(
            episode_file,
            "episode,reward,steps,avg_reward_10,mean_height,mean_upright,mean_sq_angvel"
        )
        .expect("failed to write header");

        Self { episode_file }
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
        mean_sq_angvel: f32,
    ) {
        writeln!(
            self.episode_file,
            "{},{:.4},{},{},{:.4},{:.4},{:.4}",
            episode, reward, steps, avg_reward, mean_height, mean_upright, mean_sq_angvel
        )
        .ok();
        if episode.is_multiple_of(10) {
            self.episode_file.flush().ok();
        }
    }
}

/// Stem for brain checkpoint files. `BinFileRecorder` appends `.bin` automatically,
/// so the actual file on disk is `brain.bin`.
pub(crate) const BRAIN_STEM: &str = "brain";
pub(crate) const NORMALIZER_FILENAME: &str = "normalizer.bin";
/// Return (value-target) normalizer checkpoint, beside the obs normalizer, so a
/// resumed run de-normalizes value predictions against the same scale it trained
/// with (a cold scale on resume would briefly mis-scale the value head).
pub(crate) const RETURN_NORMALIZER_FILENAME: &str = "return_normalizer.bin";
/// Curriculum band checkpoint, beside the brain, so a warm restart CONTINUES the
/// distance curriculum at the rung it reached rather than resetting to near targets
/// (which would re-teach what the policy already knows). Fallback on a missing/bad file
/// is rung 1 (see [`load_curriculum`]).
pub(crate) const CURRICULUM_FILENAME: &str = "curriculum.bin";
/// Persisted Adam optimizer state (rl#60): the per-parameter first/second moments and step
/// (`time`) the GPU learner carries across iterations. A resume restores these so the
/// optimizer continues with warm momentum instead of paying the brief self-correcting
/// transient a cold restart costs. Absent in pre-rl#60 checkpoints, which then resume cold
/// (see [`load_optimizer`]) rather than erroring — the format version inside the file
/// (see [`OPTIMIZER_FORMAT_VERSION`]) guards a layout change the same way.
#[cfg(any(feature = "wgpu", test))]
pub(crate) const OPTIMIZER_FILENAME: &str = "optimizer.bin";

/// Write `bytes` to `path` atomically: a sibling temp file then a rename, so a crash
/// mid-write leaves the previous file intact rather than a torn one. The overnight
/// trainer is killed and resumed, so a torn checkpoint would be silently discarded on
/// load and the run would resume from random weights.
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Persist the return normalizer's running stats (bincode, like the obs normalizer).
/// A write failure is logged, not fatal — the run continues, only resume loses the
/// scale.
fn save_return_normalizer(norm: &ReturnNormalizer, path: &Path) {
    match bincode::serialize(&norm.to_data()) {
        Ok(bytes) => {
            if let Err(e) = atomic_write(path, &bytes) {
                warn!(
                    "Failed to write return normalizer to {}: {e}",
                    path.display()
                );
            }
        }
        Err(e) => warn!("Failed to serialize return normalizer: {e}"),
    }
}

/// Load the return normalizer from a checkpoint, or `None` on a read/parse error or
/// a corrupt (negative-M2) record — a missing/bad file leaves a fresh identity scale.
fn load_return_normalizer(path: &Path) -> Option<ReturnNormalizer> {
    let bytes = std::fs::read(path)
        .map_err(|e| {
            warn!(
                "Failed to read return normalizer from {}: {e}",
                path.display()
            )
        })
        .ok()?;
    let data: ReturnNormalizerData = bincode::deserialize(&bytes)
        .map_err(|e| {
            warn!(
                "Failed to deserialize return normalizer from {}: {e}",
                path.display()
            )
        })
        .ok()?;
    ReturnNormalizer::from_data(data)
}

/// Serde mirror of [`Curriculum`] — the on-disk form. A plain `(min, max)` with no
/// invariant of its own; [`Curriculum::from_data`] re-validates on load, so a corrupt or
/// hand-edited file can never reconstitute an illegal band.
#[derive(Serialize, Deserialize)]
struct CurriculumData {
    min: f32,
    max: f32,
}

/// Persist the curriculum band beside the checkpoint (bincode, like the normalizers).
/// A write failure is logged, not fatal — the run continues; only a resume would lose
/// the rung and restart the curriculum from the start band.
pub(crate) fn save_curriculum(curriculum: Curriculum, path: &Path) {
    match bincode::serialize(&curriculum.to_data()) {
        Ok(bytes) => {
            if let Err(e) = atomic_write(path, &bytes) {
                warn!("Failed to write curriculum to {}: {e}", path.display());
            }
        }
        Err(e) => warn!("Failed to serialize curriculum: {e}"),
    }
}

/// Load the curriculum band from a checkpoint, defaulting to rung 1
/// ([`Curriculum::start`]) on ANY of: a missing file (fresh run, or a checkpoint that
/// predates the curriculum — the warm-continue case), a parse error, or a band that
/// fails the invariant (corrupt/hand-edited). Never returns an illegal band: the policy
/// simply resumes the curriculum from the start, which is safe because the start band is
/// learnable from any policy.
pub(crate) fn load_curriculum(path: &Path) -> Curriculum {
    let Ok(bytes) = std::fs::read(path) else {
        // Missing file is the EXPECTED case for a pre-curriculum checkpoint, so this is
        // info-level, not a warning — a warm-continue of an older policy is normal.
        info!(
            "No curriculum checkpoint at {} — starting the distance curriculum at rung 1",
            path.display()
        );
        return Curriculum::start();
    };
    match bincode::deserialize::<CurriculumData>(&bytes) {
        Ok(data) => Curriculum::from_data(data).unwrap_or_else(|| {
            warn!(
                "Curriculum checkpoint at {} is out of bounds — starting at rung 1",
                path.display()
            );
            Curriculum::start()
        }),
        Err(e) => {
            warn!(
                "Failed to deserialize curriculum from {} ({e}) — starting at rung 1",
                path.display()
            );
            Curriculum::start()
        }
    }
}

/// Default rollout horizon: the number of physics ticks each rollout thread rolls
/// per iteration before handing its buffers back, when `--horizon` is not given.
pub const STEPS_PER_ROLLOUT: u32 = 1024;

/// Where an env sits in the record → reset → settle lifecycle. One field, not a
/// `needs_reset: bool` + `grace: u32` pair, so an illegal combination (a respawn pending
/// *while* already settling) is unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EnvPhase {
    /// Live episode: transitions are recorded and termination is evaluated.
    #[default]
    Recording,
    /// Ended by a normal terminal/truncation; `reset_crab` will despawn+respawn
    /// this env's crab on its next run and move it to `Settling`. Set by
    /// `brain_step` and consumed by `reset_crab` in the same tick, so it is
    /// never observed across a tick boundary.
    AwaitingRespawn,
    /// Fresh crab dropping into the rest pose: `grace` ticks remain in which no
    /// transition is recorded and no termination is evaluated, while it lands
    /// and the motors take the load. Reached from `AwaitingRespawn` (a normal
    /// reset) or directly from a non-finite rescue, which respawns the crab itself
    /// and so skips `AwaitingRespawn`. `grace` is always ≥ 1 here — it returns to
    /// `Recording` the tick it would hit 0.
    Settling { grace: u32 },
}

/// A transition whose action has been chosen and obs/value/effort captured, but
/// whose reward and end need NEXT tick's post-physics pose to finalize.
///
/// The reach term must score the pose `aₜ` *produced* (the claw-tip distance at
/// `s_{t+1}`) — but the schedule is Sense → Think (`brain_step`) → Act → physics, so
/// when `brain_step` runs at tick `t` the carapace it can read is still `sₜ` (physics
/// hasn't integrated `aₜ` yet). So everything known at tick `t` is stashed here and the
/// transition completed at tick `t+1`, in phase with the pose that action caused.
#[derive(Clone)]
struct Pending {
    obs: [f32; OBS_SIZE],
    action: [f32; ACTION_SIZE],
    value: NormalizedValue,
    log_prob: f32,
    /// `Σ|aᵢ|^L` for this action — the effort summand over the RAW pre-clamp outputs
    /// (see [`action_effort`]), final at tick `t` and traveling with the action it
    /// priced. [`compute_reward`] scales it by [`EFFORT_WEIGHT`] at finalization.
    effort: f32,
}

/// Per-env episode accumulators. Each env's episode runs and resets
/// independently; pose sums (carapace height, up·Y) are averaged at episode
/// end to quantify stance quality.
#[derive(Clone, Default)]
pub struct EnvEpisode {
    pub reward: f32,
    pub steps: u32,
    pub height_sum: f32,
    pub upright_sum: f32,
    pub sq_angvel_sum: f32,
    pub phase: EnvPhase,
    /// Closest claw-tip→target 3D euclidean distance seen at any tick this episode — the
    /// curriculum's competence signal (an episode "reached" if this drops below
    /// [`CURRICULUM_REACH_RADIUS`]). `None` until the first finite tip reading. The MIN
    /// over the whole episode (not the final-tick distance) is the honest "did it get
    /// there", since the crab need only touch the target once, and the target then
    /// stays fixed for the rest of the episode.
    min_tip_dist: Option<f32>,
    /// Tick `t`'s chosen action awaiting tick `t+1`'s post-physics pose (see
    /// [`Pending`]). `None` outside a live recording stride — before the first action of
    /// an episode, and after its last is finalized or dropped at a reset/rescue boundary.
    pending: Option<Pending>,
}

/// Stored as a non-send resource because burn tensors use `OnceCell`, which is
/// not `Sync`.
pub struct TrainingState {
    pub brain: CrabBrain<TrainBackend>,
    pub config: PpoConfig,
    /// One rollout buffer per env. Kept separate so GAE never sweeps across an
    /// env boundary — interleaving transitions from different envs in one
    /// buffer would bootstrap each env's advantages from another env's values.
    pub rollouts: Vec<RolloutBuffer>,
    pub device: NdArrayDevice,

    pub envs: Vec<EnvEpisode>,
    pub episode_count: u32,

    pub recent_rewards: Vec<f32>,

    logger: MetricsLogger,

    pub total_steps: u64,
    last_log_time: std::time::Instant,

    obs_normalizer: ObsNormalizer,

    /// Running mean/std of the value targets (GAE returns), normalizing what the value
    /// head regresses to unit scale so it can track large-magnitude returns (see
    /// [`ReturnNormalizer`]). The LEARNER owns the single copy — rollout threads emit raw
    /// value predictions and never touch it, so there is no second instance to drift.
    /// Persisted in the checkpoint beside `obs_normalizer`.
    return_normalizer: ReturnNormalizer,

    checkpoint_dir: PathBuf,
    saved_on_exit: bool,

    /// Stop after this many physics ticks (0 = unlimited). See `Args::ticks`.
    tick_budget: u64,
    /// Benchmark: skip NN inference to measure the physics/overhead floor.
    skip_nn: bool,
    /// Count of `recent_rewards` already handed to the learner. The drain returns
    /// the tail past this, so each finished episode's reward reaches the learner's
    /// reward curve exactly once. The learner's own host (which steps no world) never
    /// records an episode, so its count stays 0.
    reported_episodes: usize,
    /// A fresh Welford accumulator over ONLY the observations seen since the last
    /// `reset_horizon_counter` (i.e. this horizon's samples). The rollout thread
    /// ships THIS — not the cumulative `obs_normalizer` — so the learner merges an
    /// increment the master hasn't already counted (the snapshot baseline lives in
    /// `obs_normalizer`, never re-merged). `None` on the learner's host, which never
    /// rolls and so ships nothing.
    normalizer_increment: Option<ObsNormalizer>,

    /// Running sum and count of the carapace planar (XZ) drift-from-spawn over
    /// recording envs, this horizon — the walking diagnostic shipped to the learner
    /// (drained per horizon, like the episode rewards) and logged as a mean. f64 sum so
    /// a long horizon can't lose precision.
    drift_sum: f64,
    drift_count: u64,

    /// The curriculum band a rollout thread samples targets from THIS horizon. The
    /// learner owns advancement and ships the current band down each horizon (set via
    /// [`Self::set_curriculum`]); the thread only reads it in [`seed_target`]. Defaults to
    /// the start band so a thread that somehow rolls before its first `set_curriculum`
    /// still samples a sane (rung-1) target rather than a garbage band.
    curriculum: Curriculum,
    /// This horizon's per-episode reach tally over FINISHED episodes (rollout thread →
    /// learner, drained per horizon like the rewards): `reached` of `finished` episodes
    /// came within [`CURRICULUM_REACH_RADIUS`] of the target. The learner pools these
    /// into the competence window that gates advancement. Counts only, so the learner
    /// aggregates across threads by summing.
    reach_reached: u64,
    reach_finished: u64,
}

impl TrainingState {
    /// The learner's policy host (and the test fixtures): logs to "tmp". The learner
    /// steps no world itself — it owns the policy and runs the PPO update from the
    /// threads' buffers, checkpointing every iteration directly.
    pub fn new(config: &TrainConfig) -> Self {
        Self::build(config, Path::new("tmp"), false)
    }

    /// In-process rollout thread: collects transitions but never runs the PPO update
    /// locally (the learner does), and logs to its own `metrics_dir` so K threads
    /// don't fight over one CSV. `worker_mode` turns on the per-horizon normalizer
    /// increment the thread ships back; everything else (env count, reset/grace/
    /// rescue, reward) is the shared per-env machinery.
    pub fn new_worker(config: &TrainConfig, metrics_dir: &Path) -> Self {
        Self::build(config, metrics_dir, true)
    }

    fn build(config: &TrainConfig, metrics_dir: &Path, worker_mode: bool) -> Self {
        let device = NdArrayDevice::Cpu;
        let mut brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);

        let mut obs_normalizer = ObsNormalizer::new(NORMALIZER_CLIP);
        let mut return_normalizer = ReturnNormalizer::new();

        let brain_path = config.checkpoint_dir.join(BRAIN_STEM);
        let norm_path = config.checkpoint_dir.join(NORMALIZER_FILENAME);
        let ret_norm_path = config.checkpoint_dir.join(RETURN_NORMALIZER_FILENAME);

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

        if ret_norm_path.exists()
            && let Some(loaded) = load_return_normalizer(&ret_norm_path)
        {
            info!(
                "Loaded return normalizer state from {}",
                ret_norm_path.display()
            );
            return_normalizer = loaded;
        }

        let n = config.envs.max(1) as usize;
        // A rollout thread accumulates a per-horizon increment over the same clip;
        // the learner's host steps no world and ships no normalizer, so it stays None.
        let normalizer_increment = worker_mode.then(|| ObsNormalizer::new(obs_normalizer.clip));
        Self {
            brain,
            config: PpoConfig::default(),
            rollouts: (0..n).map(|_| RolloutBuffer::new()).collect(),
            device,
            envs: vec![EnvEpisode::default(); n],
            episode_count: 0,
            recent_rewards: Vec::new(),
            logger: MetricsLogger::new(metrics_dir),
            total_steps: 0,
            last_log_time: std::time::Instant::now(),
            obs_normalizer,
            return_normalizer,
            checkpoint_dir: config.checkpoint_dir.clone(),
            saved_on_exit: false,
            tick_budget: config.ticks,
            skip_nn: config.bench_skip_nn,
            reported_episodes: 0,
            normalizer_increment,
            drift_sum: 0.0,
            drift_count: 0,
            curriculum: Curriculum::start(),
            reach_reached: 0,
            reach_finished: 0,
        }
    }

    pub(crate) fn save_checkpoint(&self) {
        if let Err(e) = std::fs::create_dir_all(&self.checkpoint_dir) {
            warn!(
                "Failed to create checkpoint dir {}: {e}",
                self.checkpoint_dir.display()
            );
            return;
        }

        let brain_path = self.checkpoint_dir.join(BRAIN_STEM);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        // Record to a temp stem then rename into place, so a crash mid-write can't
        // leave a torn brain.bin (silently discarded on load → resume from random
        // weights). The stem must be dot-free: BinFileRecorder sets the extension to
        // `.bin`, so a "brain.tmp" stem would become brain.bin and clobber the live
        // file before the rename.
        let brain_tmp_stem = self.checkpoint_dir.join("brain-tmp");
        match recorder.record(self.brain.clone().into_record(), brain_tmp_stem.clone()) {
            Ok(()) => {
                let tmp_file = brain_tmp_stem.with_extension("bin");
                let final_file = brain_path.with_extension("bin");
                match std::fs::rename(&tmp_file, &final_file) {
                    Ok(()) => info!("Saved brain to {}", final_file.display()),
                    Err(e) => warn!("Failed to finalize brain checkpoint: {e}"),
                }
            }
            Err(e) => warn!("Failed to save brain: {e}"),
        }

        let norm_path = self.checkpoint_dir.join(NORMALIZER_FILENAME);
        self.obs_normalizer.save(&norm_path);

        let ret_norm_path = self.checkpoint_dir.join(RETURN_NORMALIZER_FILENAME);
        save_return_normalizer(&self.return_normalizer, &ret_norm_path);
    }

    // ---- In-process rollout-thread / learner hooks ------------------------
    //
    // These let `training::inproc` drive a worker-mode TrainingState by hand on a
    // rollout thread: load the learner's snapshot weights + master normalizer, roll a
    // horizon (via the normal systems), then hand the buffers + per-horizon normalizer
    // increment + finished rewards back. One struct for both the learner host and the
    // rollout threads (not a parallel one) keeps their collection + update on the same code.

    /// Load brain weights from the learner's in-memory snapshot bytes (the same
    /// `FullPrecisionSettings` bincode the on-disk checkpoint uses, produced by the
    /// in-process learner once per iteration). Replaces a file load: weights move
    /// thread-to-thread as `Send` bytes, never as the `!Send` live tensors. Leaves
    /// the brain unchanged on a decode error (logged), the same fail-safe as the
    /// demo hot-reload against a torn write.
    pub fn load_brain_bytes(&mut self, bytes: &[u8]) {
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        match recorder.load(bytes.to_vec(), &self.device) {
            Ok(record) => self.brain = self.brain.clone().load_record(record),
            Err(e) => warn!("rollout thread: failed to load snapshot brain: {e}"),
        }
    }

    /// Overwrite this state's normalizer from the learner's master snapshot. The
    /// per-horizon increment is reset separately in `reset_horizon_counter`, so the
    /// increment always starts fresh each horizon regardless of this call.
    pub(crate) fn set_normalizer(&mut self, data: ObsNormalizerData) {
        self.obs_normalizer.load_snapshot(data);
    }

    /// Snapshot the master normalizer's full stats (learner → rollout threads), so
    /// each thread's policy normalizes observations against the same baseline the
    /// learner holds.
    pub(crate) fn normalizer_snapshot(&self) -> ObsNormalizerData {
        self.obs_normalizer.snapshot()
    }

    /// Snapshot the per-horizon normalizer INCREMENT to ship back (rollout thread →
    /// learner; see [`ObsNormalizer::merge`]). Empty (count 0) outside worker mode.
    pub(crate) fn normalizer_increment_snapshot(&self) -> ObsNormalizerData {
        match self.normalizer_increment.as_ref() {
            Some(inc) => inc.snapshot(),
            None => self.obs_normalizer.snapshot(),
        }
    }

    /// Merge a rollout thread's per-horizon normalizer increment into this (learner's)
    /// normalizer (see [`ObsNormalizer::merge`] for why an increment, not a snapshot).
    pub(crate) fn merge_normalizer(&mut self, data: &ObsNormalizerData) {
        self.obs_normalizer.merge(data);
    }

    /// Move the collected transitions out, leaving the buffers empty for the next
    /// horizon. The per-env episode accumulators (`envs`) are deliberately left
    /// untouched: an episode that spans a horizon boundary must continue cleanly
    /// across the cut rather than be force-terminated at the window edge.
    pub fn take_rollouts(&mut self) -> Vec<Vec<Transition>> {
        self.rollouts
            .iter_mut()
            .map(|buf| std::mem::take(&mut buf.transitions))
            .collect()
    }

    /// Reset the per-horizon normalizer increment (rollout thread, at the start of each
    /// horizon), so it always holds exactly this horizon's samples. `total_steps` stays
    /// monotonic — it is the thread's tick odometer the learner diffs for horizon length.
    pub fn reset_horizon_counter(&mut self) {
        if let Some(inc) = self.normalizer_increment.as_mut() {
            *inc = ObsNormalizer::new(self.obs_normalizer.clip);
        }
    }

    /// Drain the rewards of episodes that finished since the last drain, so the
    /// worker ships each finished episode's reward to the learner exactly once
    /// (the learner's reward-vs-samples curve aggregates all workers').
    pub fn drain_finished_episode_rewards(&mut self) -> Vec<f32> {
        let out = self.recent_rewards[self.reported_episodes..].to_vec();
        self.reported_episodes = self.recent_rewards.len();
        out
    }

    /// Drain this horizon's accumulated carapace drift-from-spawn as `(sum, count)`,
    /// resetting both. The learner sums these across rollout threads and divides for the
    /// mean planar drift it logs — the walking diagnostic. `(0.0, 0)` when nothing was
    /// recorded (a fully-settling horizon), which the learner treats as no sample.
    pub fn drain_drift(&mut self) -> (f64, u64) {
        let out = (self.drift_sum, self.drift_count);
        self.drift_sum = 0.0;
        self.drift_count = 0;
        out
    }

    /// Set the curriculum band a rollout thread samples targets from this horizon
    /// (learner → thread, once per horizon before the roll, like `set_normalizer`). The
    /// learner owns the single advancing curriculum; the thread only consumes the band.
    pub(crate) fn set_curriculum(&mut self, curriculum: Curriculum) {
        self.curriculum = curriculum;
    }

    /// Drain this horizon's per-episode reach tally as `(reached, finished)`, resetting
    /// both. The learner pools these across rollout threads into the competence window
    /// (see [`CurriculumProgress`]). `(0, 0)` when no episode finished this horizon,
    /// which records nothing.
    pub fn drain_reach(&mut self) -> (u64, u64) {
        let out = (self.reach_reached, self.reach_finished);
        self.reach_reached = 0;
        self.reach_finished = 0;
        out
    }

    /// Hand a CPU-backend PPO update its non-optimizer pieces (brain/config/device/
    /// return-normalizer; see `ppo_update_core`). The live learner updates on the GPU
    /// (see `learner_parts_for_gpu` + the `GpuLearner`); this CPU accessor backs only the
    /// `#[cfg(test)]` CPU update test, which exercises the shared update math without a
    /// GPU. The optimizer is not learner state: each CPU update site (this test and the
    /// `bench-update --backend cpu` harness) builds its own via [`crab_optimizer`], so the
    /// production learner carries no optimizer the GPU path never steps. The return
    /// normalizer is the single copy (rollout threads never touch it), handed out `&mut`
    /// to fold in the iteration's returns.
    pub fn learner_parts(
        &mut self,
    ) -> (
        &mut CrabBrain<TrainBackend>,
        &PpoConfig,
        &NdArrayDevice,
        &mut ReturnNormalizer,
    ) {
        (
            &mut self.brain,
            &self.config,
            &self.device,
            &mut self.return_normalizer,
        )
    }

    /// Learner-side accessor for the live GPU update: the CPU brain (mirrored to/from the
    /// GPU each update by [`GpuLearner`]), the PPO config, and the host return normalizer.
    /// Unlike [`Self::learner_parts`] it also omits the CPU device — the GPU update runs
    /// Adam on its own device-resident optimizer. The brain stays the single source of
    /// truth (rollout snapshots + checkpoints read it); the GPU learner only borrows it
    /// to load weights in and write back.
    #[cfg(feature = "wgpu")]
    pub fn learner_parts_for_gpu(
        &mut self,
    ) -> (
        &mut CrabBrain<TrainBackend>,
        &PpoConfig,
        &mut ReturnNormalizer,
    ) {
        (&mut self.brain, &self.config, &mut self.return_normalizer)
    }

    /// Record a finished episode's reward (the learner aggregates these from every
    /// rollout thread for the reward-vs-samples curve).
    pub fn record_episode_reward(&mut self, reward: f32) {
        self.recent_rewards.push(reward);
        self.episode_count += 1;
    }

    pub(crate) fn avg_reward(&self, window: usize) -> f32 {
        if self.recent_rewards.is_empty() {
            return 0.0;
        }
        let n = self.recent_rewards.len();
        let start = n.saturating_sub(window);
        let slice = &self.recent_rewards[start..];
        slice.iter().sum::<f32>() / slice.len() as f32
    }

    /// Per env, finalize the PREVIOUS tick's pending transition with this tick's post-physics
    /// pose (see [`Pending`] for the one-tick phasing), then stash this tick's action as the
    /// next pending. On an episode end (terminal/truncation/rescue) push the transition, tally
    /// stance + reach, log the episode, and reset the env (seeding its next target).
    ///
    /// The heart of `brain_step`: the only writer of [`Transition`]s and the per-env episode
    /// lifecycle. Termination is survival guards only — jumping, flipping, and any other
    /// strategy the policy invents are legitimate (owner call: emergent behaviour is the
    /// point); the height band is sim sanity, not a behaviour bound.
    #[allow(clippy::too_many_arguments)]
    fn finalize_transitions(
        &mut self,
        n: usize,
        body: &BodyState,
        min_tip_dists: &[Option<f32>],
        obs_arrays: &[[f32; OBS_SIZE]],
        action_arrays: &[[f32; ACTION_SIZE]],
        values: &[NormalizedValue],
        log_probs: &[f32],
        efforts: &[f32],
        rescued_envs: &[usize],
        targets: &mut CrabTargets,
        spawns: &CrabSpawns,
        curriculum: Curriculum,
    ) {
        for e in 0..n {
            // A pending exists only across a live recording stride, so an env that is
            // settling (or whose crab is momentarily absent) has none to finalize and
            // nothing to stash — the policy is holding the rest pose, not acting.
            if matches!(self.envs[e].phase, EnvPhase::Settling { .. }) || body.poses[e].is_none() {
                continue;
            }

            // Finalize the action chosen last tick using this tick's pose. Three
            // outcomes: rescued (the action drove the body non-finite — a failure),
            // true terminal (a survival guard tripped on the post-physics pose), or a
            // normal step (continues / truncated-at-cap).
            let episode_ended = if let Some(pending) = self.envs[e].pending.take() {
                if rescued_envs.contains(&e) {
                    // The pending action's result went non-finite and was force-
                    // respawned this tick (rescue runs .before(Sense)), so the pose
                    // read above is the FRESH spawn, not what the action produced.
                    // Finalize the action as the episode's terminal step with NO reach
                    // credit (`None`): the body teleported to spawn, so this tick's
                    // claw-tip distance isn't the action's doing. The effort tax still
                    // applies — it priced the COMMAND, not its result.
                    let reward = compute_reward(None, pending.effort);
                    self.rollouts[e].push(Transition {
                        obs: pending.obs,
                        action: pending.action,
                        reward,
                        value: pending.value,
                        log_prob: pending.log_prob,
                        end: StepEnd::Terminal,
                    });
                    let ep = &mut self.envs[e];
                    ep.reward += reward;
                    ep.steps += 1;
                    // No finite pose to fold into the stance averages.
                    true
                } else {
                    let (height, upright) =
                        body.poses[e].expect("poses[e].is_none() handled above");
                    // `height`/`upright` feed no reward (see `compute_reward`) — only the
                    // off-reward machinery: `height` the blow-up/fell-through guard below and
                    // the `mean_height` diagnostic, `upright` the stance diagnostics.
                    let reward = compute_reward(min_tip_dists[e], pending.effort);
                    // The blowup check only catches a genuine numerical explosion before the
                    // solver NaNs and Rapier panics the whole app; the threshold is high
                    // because direct torque is bounded (no acceleration-motor energy pump), so
                    // ordinary vigorous, limb-flinging motion is legal — only a part moving at
                    // clearly unphysical speed ends the episode. The height band is sim sanity
                    // (clipped through the floor / left the playfield).
                    let blowing_up = body.max_speeds[e] > 100.0 || !height.is_finite();
                    let done = !(0.02..=50.0).contains(&height) || blowing_up;
                    // The step cap is a TRUNCATION, not a failure: a crab still standing
                    // at the cap was cut short, so GAE must bootstrap its value rather
                    // than learn the cap is a dead end (see StepEnd::Truncated).
                    let truncated = !done && self.envs[e].steps > 1500;

                    let end = if done {
                        StepEnd::Terminal
                    } else if truncated {
                        StepEnd::Truncated
                    } else {
                        StepEnd::Continues
                    };
                    self.rollouts[e].push(Transition {
                        obs: pending.obs,
                        action: pending.action,
                        reward,
                        value: pending.value,
                        log_prob: pending.log_prob,
                        end,
                    });

                    let ep = &mut self.envs[e];
                    ep.reward += reward;
                    ep.steps += 1;
                    ep.height_sum += height;
                    ep.upright_sum += upright;
                    ep.sq_angvel_sum += body.sq_angvels[e];

                    done || truncated
                }
            } else {
                // No pending yet: the first recording tick of an episode only chooses
                // an action (stashed below); its result, and thus its transition,
                // arrives next tick.
                false
            };

            // Stash this tick's action to finalize next tick — but only if the env is
            // still recording (a just-ended env is resetting below, and a rescued env
            // is being respawned). Settling/absent envs already `continue`d above.
            if !episode_ended && matches!(self.envs[e].phase, EnvPhase::Recording) {
                self.envs[e].pending = Some(Pending {
                    obs: obs_arrays[e],
                    action: action_arrays[e],
                    value: values[e],
                    log_prob: log_probs[e],
                    effort: efforts[e],
                });
            }

            if episode_ended {
                let ep = &self.envs[e];
                let ep_reward = ep.reward;
                let ep_steps = ep.steps;
                let ep_height = ep.height_sum / ep_steps.max(1) as f32;
                let ep_upright = ep.upright_sum / ep_steps.max(1) as f32;
                let ep_sq_angvel = ep.sq_angvel_sum / ep_steps.max(1) as f32;
                // Did this episode reach the target — the curriculum's competence signal, read
                // off the episode's closest-ever tip distance before the reset clears it.
                // `None` (no finite tip all episode) and a blown-up episode that never got close
                // both count as honest misses. A 3D radius (see [`dist_3d`]): a tip on the floor
                // under a raised ball does not count.
                let reached = ep.min_tip_dist.is_some_and(|d| d < CURRICULUM_REACH_RADIUS);
                // A rescued env was already despawned+respawned this tick by
                // rescue_nonfinite_crabs (runs .before(Sense)); a second respawn from reset_crab
                // would tear down that zero-tick-old fresh crab and rebuild an identical one. So
                // the rescue path owns the reset: straight to `Settling` here, taking the grace
                // itself, while a normal end goes to `AwaitingRespawn` for reset_crab to respawn.
                self.envs[e] = EnvEpisode {
                    phase: if rescued_envs.contains(&e) {
                        EnvPhase::Settling {
                            grace: RESET_GRACE_TICKS,
                        }
                    } else {
                        EnvPhase::AwaitingRespawn
                    },
                    ..EnvEpisode::default()
                };

                // New episode → fresh target around this env's spawn slot from the current
                // band, so the next episode poses a new target. Done here (the one place both
                // the normal and rescue ends converge) so target life tracks episode life.
                seed_target(targets, spawns, e, curriculum);

                // Tally this finished episode's reach for the curriculum (drained per horizon
                // to the learner, like the rewards just below).
                self.reach_finished += 1;
                if reached {
                    self.reach_reached += 1;
                }

                self.recent_rewards.push(ep_reward);
                self.episode_count += 1;
                let avg = self.avg_reward(10);
                let ep_count = self.episode_count;

                self.logger.log_episode(
                    ep_count,
                    ep_reward,
                    ep_steps,
                    avg,
                    ep_height,
                    ep_upright,
                    ep_sq_angvel,
                );

                if self.episode_count.is_multiple_of(10) {
                    let elapsed = self.last_log_time.elapsed().as_secs_f32();
                    let total_transitions =
                        self.total_steps * n as u64 + (e as u64 + 1).min(n as u64);
                    let sps = if elapsed > 0.0 {
                        total_transitions as f32 / elapsed
                    } else {
                        0.0
                    };
                    let buffered: usize = self.rollouts.iter().map(|b| b.len()).sum();
                    // Σω² is telemetry only — never enters the reward. (The other
                    // labels spell out their scope inline.)
                    info!(
                        "Ep {} | avg reward(10): {:.2} | last ep (1 env): {} steps, height {:.2}, upright {:.2}, Σω² {:.0} | buffer {} | {:.0} steps/s (lifetime avg)",
                        self.episode_count,
                        avg,
                        ep_steps,
                        ep_height,
                        ep_upright,
                        ep_sq_angvel,
                        buffered,
                        sps,
                    );
                }
            }
        }
    }

    /// Accumulate this tick's carapace drift-from-spawn over RECORDING envs (one sample each)
    /// into the horizon's walking diagnostic (see `drift_sum`). Recording-only, so a settle
    /// pose can't masquerade as a cold policy's ~0 reach.
    fn accumulate_drift(&mut self, drifts: &[Option<f32>]) {
        for (e, drift) in drifts.iter().enumerate() {
            if matches!(self.envs[e].phase, EnvPhase::Recording)
                && let Some(d) = *drift
                && d.is_finite()
            {
                self.drift_sum += d as f64;
                self.drift_count += 1;
            }
        }
    }
}

/// Effort/tax/reach probe (RL_LOG_EFFORT only — inert otherwise): per tick, the mean raw-action
/// effort `Σ|a|³`, the resulting tax `EFFORT_WEIGHT·effort`, and the reach term, over the live
/// RECORDING envs. Lets a calibration run read how big a bite the tax takes out of the
/// (reach-only) positive reward at the current weight, without parsing rollouts.
///
/// Calibration diagnostic, ONE tick skewed: mean_tax is tick-`t`'s command while mean_reach is
/// tick-`t`'s pose (the result of the LAST action), so read the ratio as a magnitude check, not
/// as exactly-aligned same-action terms.
fn log_effort_probe(envs: &[EnvEpisode], efforts: &[f32], min_tip_dists: &[Option<f32>]) {
    if std::env::var_os("RL_LOG_EFFORT").is_none() {
        return;
    }
    let mut count = 0usize;
    let mut effort_sum = 0.0f32;
    let mut reach_sum = 0.0f32;
    for (e, ep) in envs.iter().enumerate() {
        if matches!(ep.phase, EnvPhase::Recording) {
            count += 1;
            effort_sum += efforts[e];
            reach_sum += reach_bonus(min_tip_dists[e]);
        }
    }
    if count > 0 {
        let mean_effort = effort_sum / count as f32;
        info!(
            "EFFORTLOG n={count} mean_effort={mean_effort:.3} \
             mean_tax={:.4} mean_reach={:.4}",
            EFFORT_WEIGHT * mean_effort,
            reach_sum / count as f32,
        );
    }
}

/// The learner's PPO update over all K·M rollout buffers.
///
/// `rollouts` is one buffer per env (GAE is computed strictly per env, never
/// across a buffer boundary). The per-env trailing bootstrap — V of each buffer's
/// non-`Terminal` tail observation — is computed HERE from `brain`: the learner
/// holds the brain it snapshotted to the threads (which is what they rolled with),
/// so no precomputed value is needed. Mutating `brain`/`optimizer` in place keeps
/// Adam's moment estimates persistent across updates.
///
/// `ret_norm` is the learner's running return scale (see [`ReturnNormalizer`]): the
/// value head's outputs (stored per-step values and the trailing bootstrap) are
/// de-normalized by it so GAE/advantages stay in real reward units, then this
/// update's real-unit returns are folded in and the value-loss targets normalized by
/// the refreshed scale. It is `&mut` because the update advances it; the learner owns
/// the one copy, so passing it in keeps a single source of truth.
///
/// Free function rather than a `TrainingState` method so the K=1 parity test
/// ([`inproc`] tests) can call the exact production update over hand-built buffers.
/// Generic over the autodiff backend `B` so the live GPU learner and the CPU/GPU
/// `bench-update` run the one implementation — same update, one backend parameter.
pub(crate) fn ppo_update_core<B: AutodiffBackend>(
    brain: &mut CrabBrain<B>,
    optimizer: &mut CrabOpt<B>,
    config: &PpoConfig,
    rollouts: &[RolloutBuffer],
    device: &B::Device,
    ret_norm: &mut ReturnNormalizer,
) -> PpoMetrics {
    {
        let n: usize = rollouts.iter().map(|b| b.len()).sum();
        if n == 0 {
            return PpoMetrics::default();
        }

        // Return-normalization stats from BEFORE this update (PopArt ordering): GAE
        // de-normalizes the value head's outputs with the scale the head was trained
        // against, computes advantages/returns in REAL reward units, and only after
        // does `ret_norm.update` fold THIS update's returns in. The first update sees
        // the identity (no returns yet), so it is byte-identical to un-normalized PPO.
        let ret_norm_pre = ret_norm.clone();

        // GAE strictly per env: each buffer is one env's contiguous trajectory
        // segment, bootstrapped from ITS last observation. Advantages/returns are
        // then concatenated in the same env-major order as the transitions below.
        let mut advantages = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        for buf in rollouts.iter() {
            let Some(last_t) = buf.transitions.last() else {
                continue;
            };
            // A `Terminal` tail genuinely ended → 0 future return; `compute_gae`
            // hardcodes that step's bootstrap to `RealReturn(0.0)` and never reads
            // `last_value`, so the value passed here is inert (any `NormalizedValue`
            // would do). A non-terminal tail bootstraps V(s_tail) from the brain (the
            // trailing obs continues into the next horizon's buffer, so its value
            // carries the cut-off return); that output is in normalized units, which
            // `compute_gae` de-normalizes like every stored value.
            let last_value = if matches!(last_t.end, StepEnd::Terminal) {
                NormalizedValue(0.0)
            } else {
                let obs =
                    Tensor::<B, 1>::from_floats(last_t.obs.as_slice(), device).unsqueeze::<2>();
                NormalizedValue(
                    brain
                        .value(obs)
                        .flatten::<1>(0, 1)
                        .into_scalar()
                        .elem::<f32>(),
                )
            };
            let (a, r) = compute_gae(buf, last_value, config.gamma, config.lambda, &ret_norm_pre);
            advantages.extend(a);
            returns.extend(r);
        }

        // Fold this update's REAL-unit returns into the running scale, then normalize the
        // value-loss targets by the refreshed scale (the residual is then in σ-units — see
        // the value-loss site below).
        let real_returns: Vec<f32> = returns.iter().map(|r| r.0).collect();
        ret_norm.update(&real_returns);
        let returns: Vec<f32> = returns.iter().map(|&r| ret_norm.normalize(r).0).collect();

        // Env-major transition view matching the advantages/returns order.
        let transitions: Vec<&Transition> =
            rollouts.iter().flat_map(|b| b.transitions.iter()).collect();

        // Batch-normalizing the advantages strips their reward unit (centered and
        // scaled to a unitless gradient signal), so they leave `RealReturn` here.
        let advantages: Vec<f32> = advantages.iter().map(|a| a.0).collect();
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

        let obs_data: Vec<f32> = transitions
            .iter()
            .flat_map(|t| t.obs.iter().copied())
            .collect();
        let actions_data: Vec<f32> = transitions
            .iter()
            .flat_map(|t| t.action.iter().copied())
            .collect();
        let old_log_probs_data: Vec<f32> = transitions.iter().map(|t| t.log_prob).collect();

        let obs_all = Tensor::<B, 2>::from_data(TensorData::new(obs_data, [n, OBS_SIZE]), device);
        let actions_all =
            Tensor::<B, 2>::from_data(TensorData::new(actions_data, [n, ACTION_SIZE]), device);
        let old_log_probs_all =
            Tensor::<B, 1>::from_data(TensorData::new(old_log_probs_data, [n]), device);
        let advantages_all =
            Tensor::<B, 1>::from_data(TensorData::new(advantages_norm, [n]), device);
        let returns_all = Tensor::<B, 1>::from_data(TensorData::new(returns, [n]), device);

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut update_count = 0u32;

        let bs = config.batch_size;
        let half_log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();

        let mut rng = thread_rng();

        for _epoch in 0..config.epochs_per_update {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);

            let num_batches = n.div_ceil(bs);

            for batch_idx in 0..num_batches {
                let start = batch_idx * bs;
                let end = (start + bs).min(n);
                let batch_n = end - start;
                let batch_indices = &indices[start..end];

                let idx_tensor = Tensor::<B, 1, Int>::from_data(
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

                let (means, log_std) = brain.policy(obs.clone());

                // log_std is pre-clamped by policy (single source of truth).
                let diff = actions - means;
                let log_std_2d = log_std
                    .clone()
                    .unsqueeze_dim::<2>(0)
                    .expand([batch_n, ACTION_SIZE]);
                let scaled_diff = diff / log_std_2d.clone().exp();
                let log_probs_per_dim =
                    scaled_diff.powf_scalar(2.0).neg() * 0.5 - log_std_2d - half_log_2pi;
                let new_lp: Tensor<B, 1> = log_probs_per_dim.sum_dim(1).flatten(0, 1);

                let entropy_per_dim = log_std.clone()
                    + (0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln());
                let entropy = entropy_per_dim.mean();

                let log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0);
                let ratio = log_ratio.exp();
                let surr1 = ratio.clone() * advs.clone();
                let surr2 =
                    ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advs;
                let policy_loss = surr1.min_pair(surr2).mean().neg();

                // The value head's raw output is in NORMALIZED units, and `rets` was
                // normalized by the same running scale above, so this residual is in
                // σ-units and `value_loss_clip` is a σ-count. The head therefore fits
                // unit-scale targets regardless of the reward magnitude — the whole
                // point of return normalization.
                let values: Tensor<B, 1> = brain.value(obs).flatten(0, 1);
                let value_diff =
                    (values - rets).clamp(-config.value_loss_clip, config.value_loss_clip);
                let value_loss = value_diff.powf_scalar(2.0).mean();

                let loss = policy_loss.clone() + value_loss.clone() * config.value_coeff
                    - entropy.clone() * config.entropy_coeff;

                total_policy_loss += policy_loss.clone().into_scalar().elem::<f32>();
                total_value_loss += value_loss.clone().into_scalar().elem::<f32>();
                total_entropy += entropy.clone().into_scalar().elem::<f32>();
                update_count += 1;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, brain);
                *brain = optimizer.step(config.learning_rate, brain.clone(), grads);
            }
        }

        PpoMetrics {
            policy_loss: total_policy_loss / update_count as f32,
            value_loss: total_value_loss / update_count as f32,
            entropy: total_entropy / update_count as f32,
        }
    }
}

/// Microbenchmark of the PPO **update** phase in isolation. Builds a fresh
/// `CrabBrain` + Adam optimizer exactly as the real learner does, synthesizes
/// `workers*envs` rollout buffers of `horizon` transitions each (so the per-update
/// transition count and per-env GAE segment structure match a live `learn` iter at
/// the same K/M/H), then calls the production [`ppo_update_core`] `reps` times. The
/// first call is a warmup (page-in / first-alloc) and is excluded; the rest are
/// timed individually and reported as min / median / max plus the per-rep PPO
/// metrics (so NaN/garbage shows up as non-finite loss).
///
/// Generic over the autodiff backend `B` and parameterised by `device`: the caller picks
/// CPU (`TrainBackend` + `NdArrayDevice::Cpu`) or GPU (`Autodiff<Wgpu>` + a `WgpuDevice`),
/// so both exercise this one harness over the production [`ppo_update_core`].
/// `backend_label` is just printed. `batch_override` replaces [`PpoConfig`]'s default
/// minibatch size for the larger-batch sweep (does the GPU stay cheap as the per-step
/// matmul grows?); `None` keeps the live batch of 64.
pub fn bench_ppo_update<B: AutodiffBackend>(
    device: &B::Device,
    backend_label: &str,
    workers: usize,
    envs: usize,
    horizon: usize,
    reps: usize,
    batch_override: Option<usize>,
) {
    use rand::{Rng, SeedableRng};

    let mut brain: CrabBrain<B> = CrabBrain::new(device);
    // The production optimizer (shared `crab_optimizer`, grad-norm clip 0.5), so the
    // step we time is the real one, not a bare Adam.
    let mut optimizer: CrabOpt<B> = crab_optimizer();
    let mut config = PpoConfig::default();
    if let Some(bs) = batch_override {
        config.batch_size = bs;
    }

    // Synthetic rollouts: one buffer per env, `horizon` transitions each, filled with
    // small random obs/actions/rewards. The values matter only for numerical realism
    // (the matmul shapes — driven by OBS_SIZE/HIDDEN/ACTION_SIZE/batch_size — are what
    // we measure); a fixed-seed RNG keeps the two backend builds comparing the same
    // data. A non-terminal tail per buffer forces the trailing value bootstrap (one
    // extra forward), matching a truncated live horizon.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
    let n_envs = (workers * envs).max(1);
    let mut rollouts: Vec<RolloutBuffer> = Vec::with_capacity(n_envs);
    for _ in 0..n_envs {
        let mut buf = RolloutBuffer::new();
        for _step in 0..horizon {
            let mut obs = [0.0f32; OBS_SIZE];
            for o in obs.iter_mut() {
                *o = rng.gen_range(-1.0..1.0);
            }
            let mut action = [0.0f32; ACTION_SIZE];
            for a in action.iter_mut() {
                *a = rng.gen_range(-1.0..1.0);
            }
            buf.push(Transition {
                obs,
                action,
                reward: rng.gen_range(-1.0..1.0),
                value: NormalizedValue(rng.gen_range(-1.0..1.0)),
                log_prob: rng.gen_range(-5.0..0.0),
                // Every step Continues, so each buffer's tail is non-terminal and
                // `ppo_update_core` bootstraps its trailing value off the brain (one
                // extra forward per buffer) — matching a live horizon cut by the step
                // cap rather than a real episode end.
                end: StepEnd::Continues,
            });
        }
        rollouts.push(buf);
    }

    let n: usize = rollouts.iter().map(|b| b.len()).sum();
    let minibatches_per_epoch = n.div_ceil(config.batch_size);
    let opt_steps = minibatches_per_epoch * config.epochs_per_update as usize;
    eprintln!(
        "[bench-update] backend: {backend_label} | K={workers} × M={envs} × H={horizon} → \
         {n} transitions/update | batch_size {} epochs {} → {minibatches_per_epoch} \
         minibatches/epoch × {} epochs = {opt_steps} optimizer steps/update | OBS {OBS_SIZE} \
         HIDDEN 256 ACTION {ACTION_SIZE} | reps {reps} (1 warmup, {} timed)",
        config.batch_size,
        config.epochs_per_update,
        config.epochs_per_update,
        reps.saturating_sub(1),
    );
    eprintln!(
        "[bench-update] cpu gemm threads: MATMUL_NUM_THREADS={} RAYON_NUM_THREADS={} (no effect on the GPU backend)",
        std::env::var("MATMUL_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
    );

    let mut times_ms: Vec<f64> = Vec::with_capacity(reps);
    for rep in 0..reps {
        // Fresh return normalizer per rep so each rep is the same starting condition
        // (the normalizer's running stats would otherwise drift the value-loss targets
        // rep to rep). The brain/optimizer DO carry over — that mirrors the live loop,
        // where Adam moments persist across updates, and keeps the timing realistic.
        let mut ret_norm = ReturnNormalizer::new();
        let t0 = std::time::Instant::now();
        let metrics = ppo_update_core(
            &mut brain,
            &mut optimizer,
            &config,
            &rollouts,
            device,
            &mut ret_norm,
        );
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let finite = metrics.policy_loss.is_finite()
            && metrics.value_loss.is_finite()
            && metrics.entropy.is_finite();
        eprintln!(
            "[bench-update] rep {rep:>2}{}: {dt:8.1} ms | ploss {:+.5} vloss {:+.5} ent {:+.5}{}",
            if rep == 0 { " (warmup)" } else { "        " },
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            if finite { "" } else { "  <<< NON-FINITE!" },
        );
        if rep > 0 {
            times_ms.push(dt);
        }
    }

    if times_ms.is_empty() {
        eprintln!("[bench-update] no timed reps (need reps > 1)");
        return;
    }
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times_ms[times_ms.len() / 2];
    let min = times_ms[0];
    let max = *times_ms.last().unwrap();
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let per_step_ms = median / opt_steps as f64;
    eprintln!(
        "[bench-update] RESULT update: median {median:.1} ms (min {min:.1}, max {max:.1}, mean {mean:.1}) over {} timed reps | {per_step_ms:.3} ms/optimizer-step",
        times_ms.len(),
    );
}

/// The GPU training backend: `Autodiff<Wgpu>` over Vulkan. The live learner (the SOLE
/// update path) and the `bench-update --backend gpu` comparison both run the one generic
/// [`ppo_update_core`] on this — same update, GPU device. WGSL→Vulkan (the SPIR-V
/// `burn/vulkan` path is a separate, untried lever). Rollout inference stays on
/// [`TrainBackend`] (CPU): rollouts are many tiny per-step obs forwards across the K
/// worker threads, where GPU dispatch overhead would dominate; only the one batched
/// update moves to the GPU.
#[cfg(feature = "wgpu")]
pub(crate) type GpuBackend = Autodiff<burn::backend::wgpu::Wgpu>;

/// Bring up the wgpu/Vulkan backend on the discrete GPU and PROVE the chosen adapter
/// is real hardware, returning the device to run on. The single source of the
/// adapter-selection + software-fallback guard, shared by `bench-update --backend gpu`
/// and the live `learn` learner (the SOLE update path), so there is one place that
/// decides "is this actually the GPU".
///
/// The guard is the load-bearing part. The box's Vulkan ICD set includes lavapipe
/// (`lvp` — a CPU software rasteriser, `DeviceType::Cpu`); if wgpu silently fell back
/// to it, the "GPU" update would run on the CPU. So we (a) request `DiscreteGpu(0)`,
/// which cubecl filters by `device_type == DiscreteGpu` (lavapipe, a CPU device, is
/// excluded before selection — and cubecl panics "No Discrete GPU device found" rather
/// than fall back), and (b) read the chosen adapter, PRINT it, and PANIC if it is a
/// `Cpu`/`Other` device or a known software-rasteriser name. Pair with
/// `VK_ICD_FILENAMES` pointing at only `nvidia_icd.json` to make the NVIDIA card the
/// only Vulkan device at all. `tag` prefixes the log lines (e.g. `bench-update` /
/// `learner`).
#[cfg(feature = "wgpu")]
pub(crate) fn init_gpu_backend(tag: &str) -> burn::backend::wgpu::WgpuDevice {
    use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::Vulkan, init_setup};

    // DiscreteGpu(0), not DefaultDevice, so cubecl's filter excludes lavapipe before
    // selection (see the guard rationale above).
    let device = WgpuDevice::DiscreteGpu(0);

    // init_setup::<Vulkan> forces the Vulkan API, registers this device, and hands back
    // the setup so we can inspect the real adapter.
    eprintln!("[{tag}] initialising wgpu/Vulkan on {device:?} …");
    let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
    let info = setup.adapter.get_info();
    eprintln!(
        "[{tag}] wgpu adapter: name={:?} backend={:?} device_type={:?} driver={:?} {:?}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info,
    );

    // Hard gate: refuse a software adapter (a CPU run mislabelled as GPU is worse than no
    // result). Check the device_type via its Debug form (to avoid pinning the wgpu crate
    // version) AND the adapter name against the known software-rasteriser names.
    let name_lc = info.name.to_lowercase();
    let type_str = format!("{:?}", info.device_type).to_lowercase();
    let is_software = type_str.contains("cpu")
        // Some software ICDs report `DeviceType::Other` rather than `Cpu`; reject those
        // too. A real discrete GPU reports `DiscreteGpu`, so this never rejects hardware.
        || type_str.contains("other")
        || name_lc.contains("llvmpipe")
        || name_lc.contains("lavapipe")
        || name_lc.contains("software")
        || name_lc.contains("swiftshader");
    assert!(
        !is_software,
        "wgpu selected a SOFTWARE adapter (name={:?}, type={:?}) — refusing to run the update on \
         it. Set VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json to expose \
         only the NVIDIA card.",
        info.name, info.device_type,
    );
    eprintln!(
        "[{tag}] adapter confirmed as hardware GPU ({}) — proceeding.",
        info.name
    );
    device
}

/// GPU bench entry point: bring up the wgpu/Vulkan backend on the discrete GPU (proving
/// the adapter is real hardware, [`init_gpu_backend`]), then run the [`bench_ppo_update`]
/// harness on [`GpuBackend`]. `pub` because the `bench-update --backend gpu` caller lives
/// in the separate `rl-train` crate.
#[cfg(feature = "wgpu")]
pub fn bench_ppo_update_gpu(
    workers: usize,
    envs: usize,
    horizon: usize,
    reps: usize,
    batch_override: Option<usize>,
) {
    let device = init_gpu_backend("bench-update");
    bench_ppo_update::<GpuBackend>(
        &device,
        "GPU wgpu/Vulkan (Autodiff<Wgpu>)",
        workers,
        envs,
        horizon,
        reps,
        batch_override,
    );
}

/// Wall-clock breakdown of one learner iteration's GPU update phase: the CPU→GPU weight
/// load, the GPU PPO update, and the GPU→CPU write-back, in milliseconds. The update pays
/// the two host↔device copies every iter, so the learner logs all three to show whether
/// the copies eat the GPU win — the make-or-break end-to-end number.
#[cfg(feature = "wgpu")]
#[derive(Clone, Copy)]
pub(crate) struct GpuUpdateTiming {
    /// CPU brain → bytes → GPU brain (load the policy onto the device for this update).
    pub load_ms: f64,
    /// The GPU [`ppo_update_core`] itself (autodiff backward + Adam steps).
    pub update_ms: f64,
    /// Updated GPU brain → bytes → CPU brain (the next rollout reads the CPU copy).
    pub store_ms: f64,
}

/// The GPU-resident learner — the SOLE PPO-update path (rl#49). Owns a GPU brain + GPU
/// Adam optimizer that PERSIST across iterations — the optimizer's moment estimates must
/// carry over across updates, so this is built once and reused, never per-iter.
///
/// The CPU `TrainingState.brain` stays the source of truth. Each iteration
/// [`Self::update`] mirrors the CPU policy onto the GPU, runs the one generic
/// [`ppo_update_core`] there, and mirrors the result back — no second update
/// implementation, only a device for the existing one. Weights cross the boundary as the
/// same `FullPrecisionSettings` bincode the snapshot/checkpoint uses (records are
/// backend-agnostic); no tensor is ever moved directly between backends.
#[cfg(feature = "wgpu")]
pub(crate) struct GpuLearner {
    device: burn::backend::wgpu::WgpuDevice,
    brain: CrabBrain<GpuBackend>,
    optimizer: CrabOpt<GpuBackend>,
}

#[cfg(feature = "wgpu")]
impl GpuLearner {
    /// Bring up the GPU backend and build a GPU brain + the shared [`crab_optimizer`].
    /// The brain's initial weights are irrelevant: [`Self::update`] loads the CPU policy
    /// onto it before every update, so the first update trains the real policy, not this
    /// fresh net.
    ///
    /// # Panics
    /// Via [`init_gpu_backend`], if no real discrete-GPU Vulkan adapter is available (a
    /// software lavapipe/llvmpipe adapter, or none at all). Deliberate: the GPU is the
    /// only update path, so it must fail loudly at boot, never silently run on the CPU.
    pub fn new() -> Self {
        let device = init_gpu_backend("learner");
        let brain: CrabBrain<GpuBackend> = CrabBrain::new(&device);
        let optimizer: CrabOpt<GpuBackend> = crab_optimizer();
        Self {
            device,
            brain,
            optimizer,
        }
    }

    /// Run one PPO update on the GPU and mirror the result back to the CPU brain. Loads
    /// `cpu_brain`'s current weights onto the GPU (so the GPU updates exactly the policy
    /// the threads rolled with), runs [`ppo_update_core`] on [`GpuBackend`], then writes
    /// the result back into `cpu_brain`. `ret_norm` (backend-independent f32 stats) is
    /// advanced in place as the CPU path does. Returns the metrics + the load/update/store
    /// wall-clock split.
    pub fn update(
        &mut self,
        cpu_brain: &mut CrabBrain<TrainBackend>,
        config: &PpoConfig,
        rollouts: &[RolloutBuffer],
        ret_norm: &mut ReturnNormalizer,
    ) -> (PpoMetrics, GpuUpdateTiming) {
        use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
        type Bridge = BinBytesRecorder<FullPrecisionSettings>;

        // CPU → GPU: serialize the CPU policy to bincode, then load it into the GPU brain.
        let t_load = std::time::Instant::now();
        let bytes = Bridge::default()
            .record(cpu_brain.clone().into_record(), ())
            .expect("serialize CPU brain for GPU update");
        let record = Bridge::default()
            .load(bytes, &self.device)
            .expect("load brain record onto GPU");
        self.brain = self.brain.clone().load_record(record);
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

        // GPU: the one production update, on the device. `ppo_update_core` reads each
        // minibatch's losses back via `into_scalar`, which forces that minibatch's
        // forward (and the prior step's backward+optimizer.step it depends on) to
        // complete — so almost all the work is genuinely synced inside this region. The
        // explicit `sync` after forces the FINAL minibatch's backward+step too, so
        // `update_ms` is the true compute time and `store_ms` below is purely the copy
        // (without it, the last step would leak into the store timing).
        let t_update = std::time::Instant::now();
        let metrics = ppo_update_core(
            &mut self.brain,
            &mut self.optimizer,
            config,
            rollouts,
            &self.device,
            ret_norm,
        );
        <GpuBackend as burn::tensor::backend::Backend>::sync(&self.device)
            .expect("GPU sync after PPO update");
        let update_ms = t_update.elapsed().as_secs_f64() * 1000.0;

        // GPU → CPU: mirror the updated weights back so the next rollout snapshot +
        // the checkpoint (both off the CPU brain) carry this iteration's update.
        let t_store = std::time::Instant::now();
        let bytes = Bridge::default()
            .record(self.brain.clone().into_record(), ())
            .expect("serialize GPU brain back to CPU");
        let cpu_device = NdArrayDevice::Cpu;
        let record = Bridge::default()
            .load(bytes, &cpu_device)
            .expect("load updated brain record onto CPU");
        *cpu_brain = cpu_brain.clone().load_record(record);
        let store_ms = t_store.elapsed().as_secs_f64() * 1000.0;

        (
            metrics,
            GpuUpdateTiming {
                load_ms,
                update_ms,
                store_ms,
            },
        )
    }

    /// Persist the optimizer's Adam state (per-param m/v + step) to `path`, reading the
    /// moment tensors back off the GPU. Called each iteration beside the brain checkpoint so
    /// a resume warm-starts the optimizer. Best-effort (see [`save_optimizer`]).
    pub fn save_adam_state(&self, path: &Path) {
        save_optimizer(&self.optimizer, path);
    }

    /// Restore the optimizer's Adam state from `path`, uploading the moments back onto this
    /// learner's GPU device. A missing file (pre-rl#60 checkpoint) or an unrecognized version
    /// leaves the optimizer cold without error (see [`load_optimizer`]).
    pub fn load_adam_state(&mut self, path: &Path) {
        let cold = std::mem::replace(&mut self.optimizer, crab_optimizer());
        self.optimizer = load_optimizer(cold, path, &self.device);
    }
}

/// The start band (rung 1): the planar (XZ) distance, in metres, at which a fresh
/// target spawns from the env's origin. NEAR on purpose. WHY a curriculum at all: a cold
/// policy cannot learn a FAR (3–6 m) target — that far out the reach term is both small
/// and flat (~0.115, slope ~0.05/m at 4.5 m), too weak for a stand to discover locomotion
/// from, so it stalls in the stand basin and never walks (verified: 150 iters pinned at
/// the stand floor, drift ~0.3 m). At the near band it is large and steep (~0.385, slope
/// ~0.13/m at 1.5 m), so the crab sets off immediately and a gait forms. The band then
/// WIDENS outward only as the policy masters the current rung (see [`Curriculum`]). Lower
/// bound clears the ~1.3 m reach shell so even the nearest target demands a step, not a lean.
const BAND_START_MIN: f32 = 1.5;
const BAND_START_MAX: f32 = 3.0;
/// How far the band slides outward per advancement (both bounds move, so the width is
/// invariant). 1 m is roughly one rung of reach-gradient difficulty: small enough that
/// the policy is already competent just inside the new far edge, large enough that the
/// curriculum reaches the arena cap in a handful of rungs rather than crawling.
const BAND_ADVANCE_STEP: f32 = 1.0;
/// Vertical band of the target (world Y). A modest claw-height span so a crab that
/// has walked up to the target still finishes with a real reach, not a foot-level
/// touch. Kept low and narrow — the reward is about getting THERE, so the target sits
/// just high enough to demand a genuine reach, no higher.
const TARGET_Y_MIN: f32 = 0.15;
const TARGET_Y_MAX: f32 = 0.7;
/// Half-extent the target's planar position is clamped within: a 1 m margin inside the
/// arena walls, DERIVED from the wall position so a wall move can't strand a far target
/// in or beyond a wall where the crab can't stand on it. The margin leaves room for the
/// crab's own body at the goal. It is also the curriculum's hard cap: the band's far
/// edge never advances past it, since a target the crab can't physically stand at is
/// not a rung worth training.
const TARGET_ARENA_HALF: f32 = crate::physics::world::ARENA_HALF_SIZE - 1.0;

/// Per-episode reach radius (m): the curriculum scores an episode "reached" if the
/// crab's claw tip came within this of the target at any tick. The CANONICAL reach
/// distance — the demo's ball-hop (`crate::play::DEMO_REACH_RADIUS`) derives from this one
/// constant, so "reached" means the same event a viewer sees the ball teleport on. Lives
/// in the always-compiled trainer (`pub(crate)` so the demo re-exports it) rather than the
/// render-only demo, so the headless build owns the source. A touch looser than zero so a
/// near-miss the policy clearly solved still counts.
pub(crate) const CURRICULUM_REACH_RADIUS: f32 = 0.8;
/// Reach-fraction over the competence window at or above which the band advances. 0.6,
/// not ~1.0: the goal is "the policy reliably gets there", not "every episode is
/// perfect" — targets near the arena edge clamp short and some spawns are awkward, so
/// demanding unanimity would stall the curriculum on noise it has effectively mastered.
const ADVANCE_REACH_FRACTION: f32 = 0.6;
/// Number of recent FINISHED episodes (pooled across all rollout threads) the
/// reach-fraction is measured over before an advance is considered. Wide enough that
/// one lucky streak can't trip an advance, narrow enough that the signal tracks the
/// CURRENT policy rather than ancient episodes from before the last advance. Episodes
/// from before an advance are dropped on advancing (see [`Curriculum::record_episode`])
/// so the window only ever judges the rung it currently sits on.
const COMPETENCE_WINDOW: usize = 200;

/// The target-distance curriculum: the single source of truth for the current planar
/// distance band, plus the competence window that decides when to widen it.
///
/// Invariant, upheld by construction (private fields, only [`Self::start`] and
/// [`Self::advanced`] build one): `BAND_START_MIN ≤ min < max ≤ TARGET_ARENA_HALF`, and
/// the width `max − min` is constant across rungs. So a `Curriculum` can never name an
/// empty or inverted band, nor one past the arena cap — illegal states are
/// unrepresentable rather than checked at every read.
///
/// The LEARNER owns the one instance: it pools every rollout thread's per-episode reach
/// outcomes into `window` and advances when the rung is mastered. Threads receive only
/// the band (`min`/`max`) for the horizon, sample targets from it, and ship reach counts
/// back — they never advance, so there is no second curriculum to drift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Curriculum {
    /// Near edge of the current band (m).
    min: f32,
    /// Far edge of the current band (m).
    max: f32,
}

impl Curriculum {
    /// Rung 1 — the near start band a cold policy can bootstrap from (see
    /// [`BAND_START_MIN`]). The only entry point for a fresh run or a checkpoint that
    /// predates the curriculum.
    pub(crate) const fn start() -> Self {
        Self {
            min: BAND_START_MIN,
            max: BAND_START_MAX,
        }
    }

    /// The current band `[min, max)` the thread samples a target distance from.
    pub(crate) fn band(self) -> (f32, f32) {
        (self.min, self.max)
    }

    /// The on-disk mirror (see [`save_curriculum`]).
    fn to_data(self) -> CurriculumData {
        CurriculumData {
            min: self.min,
            max: self.max,
        }
    }

    /// Reconstitute from the on-disk mirror, re-checking the invariant so a corrupt or
    /// hand-edited file cannot produce an illegal band: finite, `BAND_START_MIN ≤ min <
    /// max ≤ TARGET_ARENA_HALF`. `None` on any violation — the caller falls back to
    /// rung 1. (The width is NOT re-checked: only `start`/`advanced` build bands, both
    /// width-preserving, so an in-bounds persisted band is necessarily a real rung.)
    fn from_data(d: CurriculumData) -> Option<Self> {
        let ok = d.min.is_finite()
            && d.max.is_finite()
            && d.min >= BAND_START_MIN
            && d.min < d.max
            && d.max <= TARGET_ARENA_HALF;
        ok.then_some(Self {
            min: d.min,
            max: d.max,
        })
    }

    /// The next rung out: both edges slid by [`BAND_ADVANCE_STEP`] (width preserved),
    /// the far edge capped at [`TARGET_ARENA_HALF`]. `None` once the far edge is already
    /// at the cap — the curriculum is done.
    fn advanced(self) -> Option<Self> {
        if self.max >= TARGET_ARENA_HALF {
            return None;
        }
        let max = (self.max + BAND_ADVANCE_STEP).min(TARGET_ARENA_HALF);
        // Slide the near edge by the same amount the far edge actually moved (which the
        // cap may have shortened on the last rung), so the width stays exactly constant.
        let min = self.min + (max - self.max);
        Some(Self { min, max })
    }
}

/// The learner's competence tracker over the curriculum: the current rung plus a
/// sliding window of recent per-episode reach outcomes (pooled across rollout threads),
/// which gates advancement. Separate from [`Curriculum`] (the persisted band) because
/// the window is transient learner bookkeeping — it is rebuilt from live episodes after
/// a restart and deliberately NOT persisted (only the band itself survives a checkpoint).
pub(crate) struct CurriculumProgress {
    curriculum: Curriculum,
    /// `true` = that episode reached the target. Bounded to [`COMPETENCE_WINDOW`]
    /// (oldest dropped) so the fraction always reflects the current rung's recent
    /// performance, and cleared on every advance so a new rung starts judging fresh.
    window: std::collections::VecDeque<bool>,
}

impl CurriculumProgress {
    pub(crate) fn new(curriculum: Curriculum) -> Self {
        Self {
            curriculum,
            window: std::collections::VecDeque::with_capacity(COMPETENCE_WINDOW),
        }
    }

    pub(crate) fn curriculum(&self) -> Curriculum {
        self.curriculum
    }

    /// Fold one finished episode's reach outcome into the window and, if the rung is now
    /// mastered, advance the band. Mastery = a FULL window whose reach-fraction is at
    /// least [`ADVANCE_REACH_FRACTION`]. Requiring a full window stops a brand-new rung
    /// (or a fresh restart) from advancing on a handful of early episodes. On an advance
    /// the window is cleared so the next rung is judged only on episodes that actually
    /// faced it. Monotone: [`Curriculum::advanced`] only moves outward and returns `None`
    /// at the cap, so the band never regresses and never exceeds the arena. Returns `true`
    /// iff this episode triggered an advance, so a batch fold ([`Self::record_episodes`])
    /// can stop before seeding the cleared rung with episodes from the old band.
    pub(crate) fn record_episode(&mut self, reached: bool) -> bool {
        if self.window.len() == COMPETENCE_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(reached);

        if self.window.len() < COMPETENCE_WINDOW {
            return false;
        }
        let reached_count = self.window.iter().filter(|&&r| r).count();
        let fraction = reached_count as f32 / self.window.len() as f32;
        if fraction >= ADVANCE_REACH_FRACTION
            && let Some(next) = self.curriculum.advanced()
        {
            self.curriculum = next;
            self.window.clear();
            return true;
        }
        false
    }

    /// Fold a horizon's pooled reach tally (`reached` of `finished` episodes reached)
    /// into the window one episode at a time, so the window bound and the advance check
    /// run per episode exactly as if each had been recorded singly. Threads ship counts,
    /// not per-episode booleans, because the gate only uses the window's reach-fraction
    /// (order-free). STOP at an advance: the rest of this horizon's episodes were rolled
    /// against the now-superseded (nearer, easier) band, so folding them into the freshly
    /// cleared rung's window would bias the new rung optimistically — drop them; the next
    /// horizon faces the new band.
    pub(crate) fn record_episodes(&mut self, reached: u64, finished: u64) {
        let reached = reached.min(finished);
        for i in 0..finished {
            if self.record_episode(i < reached) {
                break;
            }
        }
    }
}

/// Sample a fresh target world position for a crab whose env spawns at `origin`, at a
/// planar distance drawn from the CURRENT curriculum band (see [`Curriculum`]). Picks a
/// uniform random heading and a distance in the band, places the target that far from
/// `origin` on the XZ plane, then CLAMPS it inside the arena (see [`TARGET_ARENA_HALF`])
/// so an edge spawn can't throw it into a wall. Y is an independent claw-height draw.
/// World-space (not carapace-relative) because the crab spawns at varied orientations
/// and walks: a point fixed in the world is an unambiguous goal the observation
/// re-expresses in body axes each tick. `pub(crate)` so the demo's red-ball marker
/// (`play::target_ball`) relocates its target through the very same rule training
/// samples — one sampling rule, so the demo can never pose a target training never saw.
pub(crate) fn sample_target(
    origin: Vec3,
    curriculum: Curriculum,
    rng: &mut impl rand::Rng,
) -> Vec3 {
    let (min, max) = curriculum.band();
    let theta = rng.gen_range(0.0..std::f32::consts::TAU);
    let dist = rng.gen_range(min..max);
    let x = (origin.x + dist * theta.cos()).clamp(-TARGET_ARENA_HALF, TARGET_ARENA_HALF);
    let z = (origin.z + dist * theta.sin()).clamp(-TARGET_ARENA_HALF, TARGET_ARENA_HALF);
    Vec3::new(x, rng.gen_range(TARGET_Y_MIN..TARGET_Y_MAX), z)
}

/// Install a fresh target for env `e`, sampled around its spawn slot from the current
/// curriculum `band`. The one home for "a new target is needed" — called to seed the
/// first episode (envs start target-less) and to refresh on every reset, so both callers
/// sample it identically. (Training holds the target fixed within an episode — no
/// resample on reach; see the reach-hover note in [`brain_step`].)
fn seed_target(targets: &mut CrabTargets, spawns: &CrabSpawns, e: usize, curriculum: Curriculum) {
    if let Some(slot) = targets.envs.get_mut(e) {
        let origin = spawns.0.get(e).copied().unwrap_or(Vec3::ZERO);
        *slot = Some(sample_target(origin, curriculum, &mut thread_rng()));
    }
}

/// Planar (XZ) distance between two world points. NOT the reach reward's `d` (that is
/// [`dist_3d`]); kept for the genuinely 2D diagnostics — the carapace's ground drift
/// from spawn and the curriculum band, both DEFINED on the floor plane.
fn planar_dist(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    (d.x * d.x + d.z * d.z).sqrt()
}

/// Full 3D euclidean distance between two world points — the reach reward's `d`. 3D (not
/// planar) so lowering a claw onto a low ball pays: a ground-only `d` would score a tip
/// hovering a metre above the target identically to one resting on it, leaving nothing to
/// pull the claw down the last stretch. `pub(crate)` so the demo's reached-test
/// (`play::target_ball`) measures the SAME `d` the reward does.
pub(crate) fn dist_3d(a: Vec3, b: Vec3) -> f32 {
    (a - b).length()
}

/// Weight of the effort term `− EFFORT_WEIGHT·Σ|aᵢ|^L` (see [`compute_reward`]): the
/// per-command actuation cost. Binding constraint — effort is the convex cube of the
/// RAW (unbounded) outputs, so the weight fixes a break-even command size (tax = reach)
/// under which exploring-and-reaching nets positive. That break-even must exceed a real
/// stride or a COLD stand can't explore a gait and stays stuck in the stand basin just
/// paying the tax. At 0.005 break-even is ~|a|≈1.6/joint (a gait is explorable) while
/// deep saturation (|a|=3) still costs ~4 ≫ the W=0.6 reach, so flailing is still reined
/// in. Convex `EFFORT_EXP`=`L` keeps the gentlest sufficient command the cheapest.
const EFFORT_WEIGHT: f32 = 0.005;
const EFFORT_EXP: f32 = 3.0;

/// The effort summand `Σ|aᵢ|^L` that [`compute_reward`] weights by [`EFFORT_WEIGHT`],
/// taken over the RAW network outputs (the sampled PRE-clamp actions — see
/// [`brain_step`]), NOT the ±1-clamped actions the sim runs. The point is a gradient
/// PAST the clamp: `|a|^L` keeps rising beyond ±1, so an output that overshoots the
/// usable range is taxed in proportion to the overshoot and the policy is pulled back
/// into range. Taxing the clamped value instead would flatten the gradient at ±1 — a
/// saturating logit would pay a fixed toll but feel no pull off the rail.
pub(crate) fn action_effort(raw_actions: &[f32; ACTION_SIZE]) -> f32 {
    raw_actions.iter().map(|a| a.abs().powf(EFFORT_EXP)).sum()
}

/// Weight `W` and length scale `S` of the reach term `W·(1 − tanh(d/S))` (see
/// [`reach_bonus`]): `W` is the bonus a claw tip earns by reaching the target
/// dead-on; `S` sets how the pull decays with distance.
///
/// **Why `1 − tanh(d/S)` and not `exp(−d/S)`:** targets spawn metres away (the
/// curriculum band runs 1.5–9 m), and `exp(−d/0.4)` is ~0 (≈1e-3 at 3 m) out there —
/// no gradient, nothing pulling the crab across the arena. `tanh` has a long
/// polynomial-ish tail: with `S = 4 m`, `1 − tanh(d/S)` is ~0.36 at 3 m and ~0.10 at
/// 6 m, with a clearly non-zero slope the whole way (proved numerically in
/// [`tests::new_reach_term_has_gradient_at_spawn_distance`]). That non-vanishing slope
/// at spawn distance IS the walking signal — descend it by getting closer, i.e. walk.
///
/// `W` is set well above the effort cost of the gentle motion that closes the
/// distance, so a whole walk to a far target nets positive and the policy is paid to
/// set off (the tradeoff is pinned in [`tests::effort_cost_calibration`]).
const REACH_WEIGHT: f32 = 0.6;
const REACH_SCALE: f32 = 4.0;

/// Shaped proximity bonus `W·(1 − tanh(d/S))` (weight and scale on [`REACH_WEIGHT`]),
/// where `d` is the minimum 3D euclidean distance over (claw tip, target) pairs. The
/// reward's only positive term: strictly POSITIVE, maxing at `W` when a tip reaches the
/// target (`d`→0). `None` (no target, no claw tip) yields 0 — the reward then degrades to
/// just the effort tax (the demo path and any tip-less tick).
fn reach_bonus(min_tip_dist: Option<f32>) -> f32 {
    match min_tip_dist {
        Some(d) if d.is_finite() => REACH_WEIGHT * (1.0 - (d / REACH_SCALE).tanh()),
        _ => 0.0,
    }
}

/// The reward: `W·(1 − tanh(d/S)) − EFFORT_WEIGHT·Σ|aᵢ|^L`, the reach pull
/// ([`reach_bonus`]) minus the cost of the commands that earn it ([`action_effort`]),
/// where `d` is the closest 3D euclidean claw-tip-to-target distance.
///
/// The reach signal is GLOBAL — a single distance, no gait term, no "feet on the
/// ground" — so locomotion EMERGES instead of being hand-specified (owner's call:
/// mechanical terms don't scale to emergent behaviour). Height and uprightness are
/// observations, not reward: this function literally can't see them, so no pose can be
/// gamed for free reward — only closing `d` pays.
fn compute_reward(min_tip_dist: Option<f32>, effort: f32) -> f32 {
    reach_bonus(min_tip_dist) - EFFORT_WEIGHT * effort
}

/// One env's sampled action for this tick: the ±1-clamped command the sim runs, the RAW
/// pre-clamp output the effort tax is taken over (see [`action_effort`]), and the sampling
/// log-prob (NaN/Inf-guarded and clamped). One row of [`sample_actions`].
struct SampledAction {
    /// Command sent to the sim — each output clamped to ±1.
    action: [f32; ACTION_SIZE],
    /// Pre-clamp output, kept only for the effort tax over the unbounded value.
    raw_action: [f32; ACTION_SIZE],
    log_prob: f32,
}

/// Per-env body state read off this tick's post-physics poses (the live `s_{t+1}`). Every
/// field is off-reward: poses/speeds feed the survival guards and stance diagnostics, drift
/// the walking diagnostic — none enters [`compute_reward`]. `None` for an env whose crab is
/// momentarily absent (mid-respawn).
struct BodyState {
    /// `(carapace height, up·Y uprightness)`, the survival-guard + stance inputs.
    poses: Vec<Option<(f32, f32)>>,
    /// Carapace planar (XZ) distance from spawn — the walking diagnostic.
    drifts: Vec<Option<f32>>,
    /// Fastest body part (limbs blow up first), linear-scaled — the blow-up guard input.
    max_speeds: Vec<f32>,
    /// Σω² over body parts — the angular-energy stance diagnostic.
    sq_angvels: Vec<f32>,
}

/// Normalize every env's observation, feeding the shared running stats (and, in worker mode,
/// the per-horizon increment the thread ships back — see [`TrainingState::normalizer_increment`]).
/// Returns one normalized row per env, the forward pass's input.
fn normalize_observations(
    training: &mut TrainingState,
    obs: &CrabObservation,
) -> Vec<[f32; OBS_SIZE]> {
    let n = training.envs.len();
    let mut obs_arrays: Vec<[f32; OBS_SIZE]> = Vec::with_capacity(n);
    for e in 0..n {
        let normalized = training.obs_normalizer.normalize(&obs.envs[e]);
        if let Some(inc) = training.normalizer_increment.as_mut() {
            inc.observe(&obs.envs[e]);
        }
        obs_arrays.push(normalized);
    }
    obs_arrays
}

/// ONE batched forward pass for all `n` envs: `[n, OBS_SIZE]` through the trunk once — this is
/// what makes N crabs cheaper than N apps. Returns each env's policy-mean row, the shared
/// `log_std`, and each value. Value-head outputs enter the type system as [`NormalizedValue`]
/// HERE (the single wrap point), so every stored value is in the head's normalized space.
///
/// `skip_nn` (bench mode) runs no network: the zeros it returns are irrelevant — the bench
/// isolates physics + overhead, and the cheap sampling below still runs on them.
fn forward_pass(
    training: &TrainingState,
    obs_arrays: &[[f32; OBS_SIZE]],
) -> (
    Vec<Tensor<NdArray, 1>>,
    Tensor<NdArray, 1>,
    Vec<NormalizedValue>,
) {
    let n = obs_arrays.len();
    let device = training.device;
    if training.skip_nn {
        let z = Tensor::<NdArray, 1>::zeros([ACTION_SIZE], &device);
        return (vec![z.clone(); n], z, vec![NormalizedValue(0.0); n]);
    }
    let inference_brain = training.brain.valid();
    let flat: Vec<f32> = obs_arrays.iter().flat_map(|a| a.iter().copied()).collect();
    let obs_batch = Tensor::<NdArray, 2>::from_data(
        burn::tensor::TensorData::new(flat, [n, OBS_SIZE]),
        &device,
    );
    let (means_batch, log_std) = inference_brain.policy(obs_batch.clone());
    let values: Vec<NormalizedValue> = inference_brain
        .value(obs_batch)
        .flatten::<1>(0, 1)
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(NormalizedValue)
        .collect();
    let means_rows = (0..n)
        .map(|e| {
            means_batch
                .clone()
                .slice([e..e + 1, 0..ACTION_SIZE])
                .flatten(0, 1)
        })
        .collect();
    (means_rows, log_std, values)
}

/// Sample one action per env from its policy mean and the shared `log_std`, with the NaN/Inf
/// guards the live solver needs: a non-finite log-prob becomes 0 (else clamped to ±20), and
/// any non-finite output element zeroes that element (warning once for the row). The kept RAW
/// output feeds the effort tax; the ±1 clamp is what the sim runs.
fn sample_actions(
    means_rows: &[Tensor<NdArray, 1>],
    log_std: &Tensor<NdArray, 1>,
    device: &NdArrayDevice,
) -> Vec<SampledAction> {
    means_rows
        .iter()
        .map(|means| {
            let action_tensor = sample_action(means, log_std, device);
            let log_prob = compute_log_prob(means, log_std, &action_tensor);
            let log_prob = if log_prob.is_nan() || log_prob.is_infinite() {
                0.0
            } else {
                log_prob.clamp(-20.0, 20.0)
            };

            let action_data: Vec<f32> = action_tensor.to_data().to_vec().unwrap();
            let mut action = [0.0f32; ACTION_SIZE];
            let mut raw_action = [0.0f32; ACTION_SIZE];
            let mut has_nan = false;
            for (i, &v) in action_data.iter().enumerate().take(ACTION_SIZE) {
                if v.is_nan() || v.is_infinite() {
                    has_nan = true;
                    action[i] = 0.0;
                    raw_action[i] = 0.0;
                } else {
                    raw_action[i] = v;
                    action[i] = v.clamp(-1.0, 1.0);
                }
            }
            if has_nan {
                warn!("NaN/Inf detected in NN output, clamping to zero");
            }
            SampledAction {
                action,
                raw_action,
                log_prob,
            }
        })
        .collect()
}

/// Gather each env's [`BodyState`] from this tick's post-physics poses/velocities. Computed
/// from queries already in hand (no extra reads). Rapier writes each parentless link's world
/// pose straight into `Transform` every FixedUpdate tick, so these are the live `s_{t+1}`
/// readings, in phase with the deferred transition (`GlobalTransform` would be PostUpdate-stale).
fn gather_body_state(
    n: usize,
    spawns: &CrabSpawns,
    carapace_q: &Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: &Query<(&CrabEnvId, &bevy_rapier3d::prelude::Velocity), With<CrabBodyPart>>,
) -> BodyState {
    let mut poses: Vec<Option<(f32, f32)>> = vec![None; n];
    let mut drifts: Vec<Option<f32>> = vec![None; n];
    for (env, transform) in carapace_q.iter() {
        if let Some(p) = poses.get_mut(env.0) {
            let up = transform.rotation * Vec3::Y;
            *p = Some((transform.translation.y, up.dot(Vec3::Y)));
        }
        if let Some(d) = drifts.get_mut(env.0) {
            let origin = spawns.0.get(env.0).copied().unwrap_or(Vec3::ZERO);
            *d = Some(planar_dist(transform.translation, origin));
        }
    }
    // Fastest body part per env — limbs, not the carapace, blow up first (tiny eye-stalk
    // balls + acceleration motors), so the blowup guard must watch every body. NaN poisons
    // the max, so fold it in as +inf.
    let mut max_speeds: Vec<f32> = vec![0.0; n];
    let mut sq_angvels: Vec<f32> = vec![0.0; n];
    for (env, vel) in parts_q.iter() {
        if let Some(m) = max_speeds.get_mut(env.0) {
            let lin = vel.linear.length();
            let ang = vel.angular.length();
            let s = if lin.is_finite() && ang.is_finite() {
                // Angular blowups (rad/s) run ~3x the linear scale before the solver NaNs;
                // fold both into one number on the linear scale.
                lin.max(ang / 3.0)
            } else {
                f32::INFINITY
            };
            *m = m.max(s);
            // Non-finite ω is the blowup guard's problem, not the tax's.
            if ang.is_finite() {
                sq_angvels[env.0] += ang * ang;
            }
        }
    }
    BodyState {
        poses,
        drifts,
        max_speeds,
        sq_angvels,
    }
}

/// Closest claw-tip→target 3D distance per env this tick (the reach reward's `d`, see
/// [`dist_3d`]), folded over both claw tips. `None` when the env has no target or no claw tip
/// this tick (mid-respawn); a non-finite tip is skipped, not folded as a spurious hit.
fn closest_tip_dists(
    n: usize,
    targets: &CrabTargets,
    claw_tips_q: &Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
) -> Vec<Option<f32>> {
    let mut min_tip_dists: Vec<Option<f32>> = vec![None; n];
    for (env, tip) in claw_tips_q.iter() {
        let Some(slot) = min_tip_dists.get_mut(env.0) else {
            continue;
        };
        let Some(target) = targets.get(env.0) else {
            continue;
        };
        if !tip.translation.is_finite() {
            continue;
        }
        let d = dist_3d(tip.translation, target);
        *slot = Some(slot.map_or(d, |cur| cur.min(d)));
    }
    min_tip_dists
}

/// System: runs the brain to produce actions each physics step.
#[allow(clippy::too_many_arguments)]
pub fn brain_step(
    mut training: NonSendMut<TrainingState>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    mut targets: ResMut<CrabTargets>,
    spawns: Res<CrabSpawns>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: Query<(&CrabEnvId, &bevy_rapier3d::prelude::Velocity), With<CrabBodyPart>>,
    claw_tips_q: Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
    mut exit: MessageWriter<AppExit>,
    mut rescued: MessageReader<CrabRescued>,
) {
    let n = training.envs.len();
    // Envs whose crab went non-finite and was force-respawned this tick: the
    // pose this step reads is already the fresh crab back at spawn, so the
    // episode must end here or the teleport bleeds into the reward stream.
    let rescued_envs: Vec<usize> = rescued.read().map(|m| m.env).collect();
    if obs.envs.len() != n || actions.envs.len() != n {
        // Resources are sized by the spawn system; skip the tick(s) before that.
        return;
    }
    let device = training.device;
    // The horizon's curriculum band (Copy), captured before the per-env borrows below so
    // both `seed_target` paths sample from the same band the learner set this horizon —
    // one band per horizon, identical for the lazy first-episode seed and every reset.
    let curriculum = training.curriculum;

    // Sense → Think: normalize, one batched forward pass, sample an action per env.
    let obs_arrays = normalize_observations(&mut training, &obs);
    let (means_rows, log_std, values) = forward_pass(&training, &obs_arrays);
    let sampled = sample_actions(&means_rows, &log_std, &device);

    let action_arrays: Vec<[f32; ACTION_SIZE]> = sampled.iter().map(|s| s.action).collect();
    let log_probs: Vec<f32> = sampled.iter().map(|s| s.log_prob).collect();
    // Effort tax (the reward's tax summand) is taken over the RAW pre-clamp outputs, per
    // env (see `action_effort`).
    let efforts: Vec<f32> = sampled
        .iter()
        .map(|s| action_effort(&s.raw_action))
        .collect();

    actions.envs.copy_from_slice(&action_arrays);
    // Settling envs hold the rest pose (action 0); the policy takes over at
    // step 0 of the new episode.
    for (e, ep) in training.envs.iter().enumerate() {
        if matches!(ep.phase, EnvPhase::Settling { .. })
            && let Some(v) = actions.envs.get_mut(e)
        {
            *v = [0.0; ACTION_SIZE];
        }
    }

    let body = gather_body_state(n, &spawns, &carapace_q, &parts_q);

    // Lazily seed the FIRST episode's target for any env still without one (envs
    // start Recording with no target; episode-end reset seeds every subsequent one).
    // Training-only by construction: only `brain_step` runs here, so the demo's
    // CrabTargets stays empty and its obs target vector stays zero.
    for e in 0..n {
        if targets.get(e).is_none() {
            seed_target(&mut targets, &spawns, e, curriculum);
        }
    }

    let min_tip_dists = closest_tip_dists(n, &targets, &claw_tips_q);
    // Fold this tick's closest tip distance into each RECORDING env's episode minimum —
    // the curriculum's competence signal. Recording-only: a Settling env already holds
    // the NEXT episode's target (seeded at reset), so crediting its settle-pose distances
    // would contaminate that episode's reach with poses the policy never chose.
    for (e, tip) in min_tip_dists.iter().enumerate() {
        if matches!(training.envs[e].phase, EnvPhase::Recording)
            && let Some(d) = *tip
        {
            let ep = &mut training.envs[e];
            ep.min_tip_dist = Some(ep.min_tip_dist.map_or(d, |cur| cur.min(d)));
        }
    }

    // ONE far target per episode: seeded at reset (and lazily above for the first episode)
    // and then held FIXED — no mid-episode resample on reach. The reward is a pure distance
    // field with no event bonus for the reach itself, so resampling-on-reach would make
    // reaching MOVE the reward away, and the optimal policy would hover just outside the
    // reach radius forever rather than touch. With a fixed goal the crab instead walks up
    // and settles at d≈0, where the reach term peaks — so full reaching is the optimum.

    // Act → record: finalize last tick's pending transition against this tick's pose, stash
    // this tick's, and roll over any episode that ended. The sole writer of `Transition`s.
    training.finalize_transitions(
        n,
        &body,
        &min_tip_dists,
        &obs_arrays,
        &action_arrays,
        &values,
        &log_probs,
        &efforts,
        &rescued_envs,
        &mut targets,
        &spawns,
        curriculum,
    );

    log_effort_probe(&training.envs, &efforts, &min_tip_dists);
    training.accumulate_drift(&body.drifts);

    training.total_steps += 1;

    // Fixed-tick stop: exactly `tick_budget` physics ticks, then save+exit. Tick
    // count, never wall-clock, so the run is reproducible across machines/load.
    // `==` (steps increment by one) so the catch-up burst that crosses the budget
    // logs/requests exit once, not once per remaining tick in the burst.
    if training.tick_budget != 0 && training.total_steps == training.tick_budget {
        info!(
            "Tick budget reached ({} ticks) — stopping training.",
            training.tick_budget
        );
        exit.write(AppExit::Success);
    }
}

/// Settle ticks after a respawn: the fresh crab spawns in the rest pose with
/// the builder motors already holding it, so this only covers the drop from
/// spawn height onto the ground and the motors taking the load (0.5 s).
///
/// This is the ONE settle window for the whole project: the demo's post-respawn
/// settle (`play::demo_settle`) reuses it via [`settle_countdown`] so a change to
/// the drop window keeps training and the streamed demo in lock-step — they must
/// hold zero actions for the same number of ticks or the demo stops mirroring the
/// crab the policy was actually trained to settle.
pub const RESET_GRACE_TICKS: u32 = 32;

/// Advance a settle countdown by one tick: returns the grace for the next tick, or 0
/// once the window is spent (the caller then resumes normal control — `Recording` for
/// training, policy drive for the demo). Sole source of the post-respawn settle
/// arithmetic so the training reset path ([`reset_crab`]) and the demo settle
/// (`play::demo_settle`) decrement the exact same way.
pub fn settle_countdown(grace: u32) -> u32 {
    grace.saturating_sub(1)
}

/// System: rebuilds each env's crab when that env's episode ends by a normal
/// terminal/truncation — `brain_step` leaves such an env in
/// [`EnvPhase::AwaitingRespawn`], which this system consumes. An episode ended
/// by a non-finite *rescue* is deliberately NOT handled here — that crab was
/// already respawned this tick by [`rescue_nonfinite_crabs`], so the rescue path
/// goes straight to [`EnvPhase::Settling`] and never enters `AwaitingRespawn`;
/// a respawn here would just rebuild the fresh crab a second time.
///
/// A reset is a full despawn + respawn ([`respawn_crab_rotated`]): teleporting bodies
/// cannot repair a multibody whose joint state went non-finite — rapier 0.32
/// offers no way to rewrite multibody joint coordinates in place — and one
/// crab that tunnels through the floor would otherwise wedge its env forever.
/// The respawned crab starts in the overlap-free rest pose, so no unfold or
/// collision-group dance is needed; the grace just skips recording while it
/// takes load (see [`EnvPhase::Settling`]).
pub fn reset_crab(
    mut commands: Commands,
    mut training: NonSendMut<TrainingState>,
    mut actions: ResMut<CrabActions>,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>,
) {
    // Randomized-start curriculum: each respawning env gets a fresh random orientation
    // so the policy learns to stand (and right itself) from varied, even inverted,
    // starts instead of memorising the one bind pose. This is training-only — reset_crab
    // never runs in the demo (no `TrainingState`), which respawns upright.
    for (e, ep) in training.envs.iter_mut().enumerate() {
        if matches!(ep.phase, EnvPhase::AwaitingRespawn) {
            ep.phase = EnvPhase::Settling {
                grace: RESET_GRACE_TICKS,
            };
            if let Some(v) = actions.envs.get_mut(e) {
                *v = [0.0; ACTION_SIZE];
            }
            let origin = spawns.0.get(e).copied().unwrap_or(Vec3::ZERO);
            let init_rotation = random_spawn_rotation(&mut thread_rng());
            respawn_crab_rotated(
                &mut commands,
                &assets,
                parts.iter().filter(|(_, id)| id.0 == e).map(|(ent, _)| ent),
                origin,
                e,
                init_rotation,
            );
        }
    }

    // Count the settle grace down on every settling env (including one just set
    // above, which lands at RESET_GRACE_TICKS-1 this tick); when the shared
    // countdown is spent it returns to Recording and the policy takes back over.
    for ep in training.envs.iter_mut() {
        if let EnvPhase::Settling { grace } = ep.phase {
            ep.phase = match settle_countdown(grace) {
                0 => EnvPhase::Recording,
                g => EnvPhase::Settling { grace: g },
            };
        }
    }
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
    fn height_does_not_change_the_reward() {
        // Guards against reintroducing a height arg while leaving the reach term inert:
        // `compute_reward` has no height argument, so tip distance must still move the
        // reward on its own.
        let effort = action_effort(&[0.3; ACTION_SIZE]);
        let near = compute_reward(Some(0.1), effort);
        let far = compute_reward(Some(2.0), effort);
        assert!(
            near > far,
            "reach still moves the reward (closer out-scores farther): {near} vs {far}"
        );
    }

    #[test]
    fn reach_bonus_rewards_reaching() {
        // The reach term is strictly positive, maxes at the target (d→0 ⇒ W), and
        // decreases monotonically with distance — a dense pull that stays alive across the
        // far band. Depends ONLY on the tip distance (uprightness is not in the reward).
        assert!(
            (reach_bonus(Some(0.0)) - REACH_WEIGHT).abs() < 1e-6,
            "a claw tip on the target earns the full reach weight"
        );
        assert!(
            reach_bonus(Some(0.1)) > reach_bonus(Some(0.5)),
            "closer to the target must out-reward farther"
        );
        // Still clearly positive at the FARTHEST the curriculum can push the target (the
        // arena cap) — the walking signal survives every rung.
        assert!(
            reach_bonus(Some(TARGET_ARENA_HALF)) > 0.0,
            "the reach bonus is strictly positive even at the arena-cap target distance"
        );
        assert!(
            reach_bonus(Some(BAND_START_MIN)) > reach_bonus(Some(TARGET_ARENA_HALF)),
            "the pull must slope downward across the whole curriculum span — the walking signal"
        );
        assert_eq!(
            reach_bonus(None),
            0.0,
            "no target (or no tip) contributes nothing — with no positive term left the \
             reward is just the effort tax"
        );
    }

    #[test]
    fn new_reach_term_has_gradient_at_spawn_distance() {
        // Numerically pins the why-tanh rationale (on [`REACH_WEIGHT`]): the `1 − tanh(d/S)`
        // term and its slope are clearly non-zero across the spawn band where an `exp` term
        // would be ~0, so a far target gives a real gradient to WALK down. Compares the two
        // at each distance via a finite-difference slope.
        const OLD_SCALE: f32 = 0.4; // an exp length scale, for the comparison
        let old_term = |d: f32| (-d / OLD_SCALE).exp();
        let new_term = |d: f32| 1.0 - (d / REACH_SCALE).tanh();
        let slope = |f: &dyn Fn(f32) -> f32, d: f32| {
            let h = 1e-3;
            (f(d + h) - f(d - h)) / (2.0 * h)
        };
        // Two distinct claims, because the curriculum band now spans a NEAR start edge
        // (where the cold policy bootstraps) out to the arena cap (the farthest rung) —
        // the old far-only band assumed the exp was ~0 everywhere, which is false at the
        // near edge (exp(−1.5/0.4)=exp(−3.75)≈0.024, small but not negligible).
        //
        // (a) NEAR edge — what makes bootstrapping possible is a strong ABSOLUTE gradient
        // at the spawn pose; the new term/slope just has to be clearly usable AND beat the
        // old (it does, ~3.7×), not dominate it 20×.
        assert!(
            new_term(BAND_START_MIN) > 0.3 && slope(&new_term, BAND_START_MIN).abs() > 0.1,
            "near edge must have a strong absolute reach gradient (the bootstrap signal): \
             term {} slope {}",
            new_term(BAND_START_MIN),
            slope(&new_term, BAND_START_MIN),
        );
        assert!(
            new_term(BAND_START_MIN) > old_term(BAND_START_MIN)
                && slope(&new_term, BAND_START_MIN).abs() > slope(&old_term, BAND_START_MIN).abs(),
            "even at the near edge the new term/slope must exceed the old",
        );
        // (b) FAR rungs (3 m out to the arena cap) — here an exp would have all but vanished
        // (exp(−7.5)…exp(−22.5)), so the tanh term/slope DOMINATE it. The term itself
        // shrinks toward the cap (1−tanh(9/4)≈0.022 at d=9), so the guarantees are: strictly
        // positive, a clearly non-zero SLOPE (the learning signal), and overwhelming
        // dominance of the exp — not an absolute term floor (which the curriculum's longer
        // reach can't assume).
        for &d in &[3.0, 4.5, TARGET_ARENA_HALF] {
            assert!(
                new_term(d) > 0.0,
                "new tanh term must stay strictly positive at far d={d}: {}",
                new_term(d)
            );
            assert!(
                new_term(d) > 20.0 * old_term(d),
                "new tanh term must dominate the old exp at far d={d}: new {} vs old {}",
                new_term(d),
                old_term(d)
            );
            assert!(
                slope(&new_term, d).abs() > 1e-3,
                "new tanh slope must be clearly non-zero at far d={d}: {}",
                slope(&new_term, d)
            );
            assert!(
                slope(&new_term, d).abs() > 20.0 * slope(&old_term, d).abs(),
                "new tanh slope must dominate the old exp slope at far d={d}: new {} vs old {}",
                slope(&new_term, d),
                slope(&old_term, d)
            );
        }
    }

    #[test]
    fn sampled_targets_lie_in_the_current_band_and_inside_the_arena() {
        // Every sampled target lies in the CURRENT curriculum band AND is clamped inside
        // the arena so a crab can always walk to and stand at it. The demo relocates its
        // target through this very `sample_target`, so the demo can never pose a goal
        // training never saw. Verified at BOTH the near start band and a far-advanced
        // band, since the curriculum moves the band outward over a run.
        let mut rng = rand::thread_rng();
        for curriculum in [Curriculum::start(), advanced_to_cap()] {
            let (min, max) = curriculum.band();
            // Worst-case CORNER origin (hard against two walls, where the clamp does the
            // most work — a target headed into the corner is pulled well inside the band).
            let origin = Vec3::new(8.0, 0.0, -8.0);
            let mut saw_clamped = false;
            for _ in 0..2000 {
                let t = sample_target(origin, curriculum, &mut rng);
                assert!(t.is_finite(), "a sampled target is always finite");
                // Inside the arena interior (the clamp guarantees this from any origin).
                assert!(
                    t.x.abs() <= TARGET_ARENA_HALF && t.z.abs() <= TARGET_ARENA_HALF,
                    "target {t:?} must stay inside ±{TARGET_ARENA_HALF} m"
                );
                assert!(t.y >= TARGET_Y_MIN && t.y <= TARGET_Y_MAX);
                // Pre-clamp distance is in the band; post-clamp can only shorten it. Track
                // that clamping actually engages at this edge origin (so the test
                // exercises the in-arena guarantee).
                let d = planar_dist(t, origin);
                if d + 1e-3 < min {
                    saw_clamped = true;
                }
            }
            // From a central origin, nothing is clamped and every target lies in the band.
            let center = Vec3::ZERO;
            for _ in 0..2000 {
                let t = sample_target(center, curriculum, &mut rng);
                let d = planar_dist(t, center);
                assert!(
                    (min..=max).contains(&d),
                    "from center, target distance {d} must lie in the current band \
                     [{min}, {max}]"
                );
            }
            assert!(
                saw_clamped,
                "an edge origin must sometimes clamp a target inward (in-arena guarantee active)"
            );
        }
    }

    /// A curriculum advanced repeatedly until it caps at the arena edge — the far end of
    /// the curriculum, used to verify sampling/advance behavior at the last rung.
    fn advanced_to_cap() -> Curriculum {
        let mut c = Curriculum::start();
        while let Some(next) = c.advanced() {
            c = next;
        }
        c
    }

    /// Drive `build_observation` over a single hand-placed carapace and return env 0's
    /// observation. No physics/rig — just the resources the system reads plus one
    /// carapace entity at the given world pose, so the body-state and target-local slots
    /// can be checked against an exact expected value (joint slots stay 0, no joints).
    fn observe_one_carapace(carapace: Transform, target: Option<Vec3>) -> [f32; OBS_SIZE] {
        use bevy::ecs::system::RunSystemOnce;
        use bevy_rapier3d::prelude::Velocity;

        let mut world = bevy::ecs::world::World::new();
        let mut obs = CrabObservation::default();
        obs.resize(1);
        let mut targets = CrabTargets::default();
        targets.resize(1);
        targets.envs[0] = target;
        world.insert_resource(obs);
        world.insert_resource(targets);
        world.insert_resource(CrabSpawns(vec![Vec3::ZERO]));
        world.spawn((CrabCarapace, CrabEnvId(0), carapace, Velocity::default()));
        world
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation");
        world.resource::<CrabObservation>().envs[0]
    }

    /// Index of the first target-local obs slot (the carapace-frame target vector lives in
    /// `[BASE, BASE+3)`), mirroring `build_observation`'s `body_base + 13`. Pinned here so
    /// the directional test reads the same slots the sensor writes.
    const TARGET_LOCAL_BASE: usize = crate::bot::body::CrabJointId::COUNT * 2 + 13;

    /// The reach goal must enter the observation as a vector that points TOWARD the
    /// target in the carapace's OWN frame (correct sign), and that body-local vector must
    /// be orientation-invariant: yaw the body and the world offset is unchanged, but its
    /// body-frame coordinates counter-rotate. This is the property the policy relies on to
    /// "walk toward the target vector" from any heading — a sign flip here would train it
    /// to walk directly away.
    #[test]
    fn target_obs_points_toward_target() {
        let base = TARGET_LOCAL_BASE;

        // Identity pose at origin: the body frame equals the world frame, so the
        // target-local vector must equal the raw world offset to the target — same
        // direction, same sign (it points AT the target, not away).
        let offset = Vec3::new(2.0, 0.5, -1.0);
        let obs = observe_one_carapace(Transform::IDENTITY, Some(offset));
        let local = Vec3::new(obs[base], obs[base + 1], obs[base + 2]);
        assert!(
            (local - offset).length() < 1e-5,
            "identity pose: target-local {local:?} must equal the world offset {offset:?} \
             (points toward the target with the right sign)"
        );

        // Yaw the carapace 180° about Y, target fixed in WORLD. The world offset is
        // unchanged, but in the rotated body frame "forward" now points the other way, so
        // the body-local X and Z must FLIP sign (Y, the spin axis, is unchanged). This is
        // the orientation-invariance the obs frame buys: same goal, body-relative reading.
        let yaw = Quat::from_rotation_y(std::f32::consts::PI);
        let obs_rot = observe_one_carapace(Transform::from_rotation(yaw), Some(offset));
        let local_rot = Vec3::new(obs_rot[base], obs_rot[base + 1], obs_rot[base + 2]);
        let expected_rot = yaw.inverse() * offset;
        assert!(
            (local_rot - expected_rot).length() < 1e-5,
            "180° yaw: target-local {local_rot:?} must be the offset rotated into the body \
             frame {expected_rot:?}"
        );
        assert!(
            (local_rot.x + offset.x).abs() < 1e-5 && (local_rot.z + offset.z).abs() < 1e-5,
            "a 180° yaw must flip the body-local forward/right components: got {local_rot:?} \
             vs world offset {offset:?}"
        );
        assert!(
            (local_rot.y - offset.y).abs() < 1e-5,
            "yaw about Y must leave the body-local Y (height) component unchanged"
        );

        // Off-origin too: target-local is the offset FROM the carapace, not the absolute
        // target — translating the body by the same vector as the target leaves it zero.
        let pos = Vec3::new(3.0, 0.0, 4.0);
        let obs_at = observe_one_carapace(Transform::from_translation(pos), Some(pos));
        let local_at = Vec3::new(obs_at[base], obs_at[base + 1], obs_at[base + 2]);
        assert!(
            local_at.length() < 1e-5,
            "carapace sitting on the target reads a zero target-local vector, got {local_at:?}"
        );
    }

    /// The reach reward must pull the tip toward the target IN 3D: a smaller 3D tip→target
    /// distance scores strictly higher, including when the only difference is HEIGHT — a
    /// claw lowered onto a low ball must beat one hovering above it at the same ground spot.
    #[test]
    fn closer_tip_in_3d_raises_reward() {
        let target = Vec3::new(1.0, 0.3, 0.0);
        // Two tips at the SAME ground position, differing only in height: one resting on
        // the ball, one a metre above it. A planar `d` ties these; the 3D `d` must rank
        // the on-target tip strictly closer.
        let on_ball = Vec3::new(1.0, 0.3, 0.0);
        let hovering = Vec3::new(1.0, 1.3, 0.0);
        let d_on = dist_3d(on_ball, target);
        let d_hover = dist_3d(hovering, target);
        assert!(
            d_on < d_hover,
            "3D distance must distinguish height: on-ball {d_on} should be < hovering {d_hover}"
        );
        // Same command effort both poses, so the reach term alone decides — closer ⇒ higher.
        let effort = action_effort(&[0.2; ACTION_SIZE]);
        assert!(
            compute_reward(Some(d_on), effort) > compute_reward(Some(d_hover), effort),
            "a tip resting on the ball must out-score one hovering a metre above it at the \
             same ground spot — the 3D reach pulls the claw DOWN, not just across"
        );
        // And monotone in general: any strictly smaller 3D distance scores strictly higher.
        for (near, far) in [(0.0_f32, 0.5_f32), (0.5, 2.0), (2.0, 6.0)] {
            assert!(
                reach_bonus(Some(near)) > reach_bonus(Some(far)),
                "reach reward must strictly increase as 3D distance shrinks: \
                 d={near} should beat d={far}"
            );
        }
    }

    #[test]
    fn reward_is_reach_minus_effort() {
        // Reward is EXACTLY `reach_bonus(d) − K·Σ|a|^L` — two terms, no height, no
        // uprightness, no hidden term. With no target and no command it is exactly zero
        // (nothing to reach, nothing to tax); a target adds the (ungated) reach term; a
        // command subtracts the tax.
        assert!(
            compute_reward(None, 0.0).abs() < 1e-6,
            "with no target and no effort, reward is exactly zero"
        );
        let expected = reach_bonus(Some(0.3)) - EFFORT_WEIGHT * action_effort(&[0.2; ACTION_SIZE]);
        assert!(
            (compute_reward(Some(0.3), action_effort(&[0.2; ACTION_SIZE])) - expected).abs() < 1e-6,
            "reward is exactly reach_bonus − K·effort"
        );
    }

    #[test]
    fn uprightness_does_not_change_the_reward() {
        // Uprightness lives in the observation, not the reward — `compute_reward` has no
        // uprightness argument, so a flat crab and a level one at the same tip distance and
        // command earn IDENTICAL reward. The consequence pinned here: a claw dangled onto
        // the target collects the FULL reach bonus ungated by pose, adding exactly
        // `reach_bonus(0) = W` over not reaching, at any pose.
        let effort = action_effort(&[0.3; ACTION_SIZE]);
        let no_reach = compute_reward(None, effort);
        let on_target = compute_reward(Some(0.0), effort);
        assert!(
            (on_target - no_reach - REACH_WEIGHT).abs() < 1e-6,
            "a claw on the target adds the full reach weight {REACH_WEIGHT} with no pose gate: \
             {on_target} − {no_reach}"
        );
    }

    #[test]
    fn higher_effort_lowers_the_reward() {
        // The tax is strictly increasing in command size, so a harder command always
        // scores below a gentler one — the lever that should make the crab economical
        // ("tired af"): it spends actuation only where reach pays for it.
        let still = compute_reward(None, action_effort(&[0.0; ACTION_SIZE]));
        let gentle = compute_reward(None, action_effort(&[0.3; ACTION_SIZE]));
        let hard = compute_reward(None, action_effort(&[0.9; ACTION_SIZE]));
        assert!(
            still > gentle && gentle > hard,
            "reward must fall as commanded effort rises: still {still} > gentle {gentle} > hard {hard}"
        );
        // With no target, a still policy pays NO tax and earns nothing — reward is zero.
        assert!(
            still.abs() < 1e-6,
            "a still policy with no target is untaxed and unrewarded: {still} should be zero"
        );
    }

    #[test]
    fn effort_cost_calibration() {
        // Pin the ordering that matters — reach is the ONLY positive term, so the tradeoff
        // is reach vs the cost of the motion that earns it (the tax is over the RAW
        // pre-clamp outputs, see `action_effort`):
        // 1. A still policy with no target pays no tax and earns nothing — reward is zero.
        let still = compute_reward(None, action_effort(&[0.0; ACTION_SIZE]));
        assert!(
            still.abs() < 1e-6,
            "a still policy with no target is zero: {still}"
        );
        // 2. Reaching the target with a MODERATE in-range command (|a| < 1) must still net
        //    POSITIVE — the reach payoff has to exceed the cost of the gentle motion that
        //    closes the distance, or the policy would rather lie still than walk. At weight
        //    0.005 a |a|=0.4 command across all 30 joints costs 0.005·30·0.4³ ≈ 0.0096, well under
        //    the W=0.6 reach payoff, so honest moderate motion that reaches stays worthwhile.
        let moderate_reach = compute_reward(Some(0.0), action_effort(&[0.4; ACTION_SIZE]));
        assert!(
            moderate_reach > 0.0,
            "reaching the target with a moderate command must net positive: {moderate_reach}"
        );
        // 3. A saturation-seeking command (raw outputs driven far past the ±1 the sim
        //    clamps to) is taxed BELOW that moderate reach even when it lands on the target —
        //    because the tax reads the raw outputs, |a|^L keeps climbing past the clamp, so
        //    the gradient pushes the policy OUT of saturation rather than letting it sit
        //    pinned at the rail for a flat toll. At |a|=3 the cost (0.005·30·27 ≈ 4.05)
        //    swamps any reach payoff, driving the reward deeply negative.
        let oversaturated = compute_reward(Some(0.0), action_effort(&[3.0; ACTION_SIZE]));
        assert!(
            oversaturated < moderate_reach,
            "saturation-seeking must be taxed below a moderate reach: {oversaturated} vs {moderate_reach}"
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

    /// One deterministic Adam step on a tiny constant-gradient loss, returning the updated
    /// brain + optimizer. Warms the Adam moments for the round-trip test without the full
    /// PPO machinery: a fixed input + sum loss puts non-zero grads on every parameter, so
    /// the moments + step advance as a real update would.
    fn adam_test_step(
        brain: CrabBrain<TrainBackend>,
        mut optimizer: CrabOpt<TrainBackend>,
        device: &NdArrayDevice,
    ) -> (CrabBrain<TrainBackend>, CrabOpt<TrainBackend>) {
        let obs = Tensor::<TrainBackend, 2>::ones([4, OBS_SIZE], device);
        let (means, log_std) = brain.policy(obs);
        let loss = means.sum() + log_std.sum();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &brain);
        let brain = optimizer.step(1e-2, brain, grads);
        (brain, optimizer)
    }

    fn policy_means(brain: &CrabBrain<TrainBackend>, device: &NdArrayDevice) -> Vec<f32> {
        let obs = Tensor::<TrainBackend, 2>::zeros([1, OBS_SIZE], device);
        brain.policy(obs).0.to_data().to_vec().unwrap()
    }

    #[test]
    fn adam_optimizer_state_round_trips_through_checkpoint() {
        let dir = std::env::temp_dir().join("rl_test_adam_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;

        // Warm an optimizer over several steps so its Adam moments + step are non-trivial,
        // then snapshot brain + optimizer mid-run.
        let mut brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);
        let mut warm = crab_optimizer::<TrainBackend>();
        for _ in 0..5 {
            let (b, o) = adam_test_step(brain, warm, &device);
            brain = b;
            warm = o;
        }
        assert!(
            !warm.to_record().is_empty(),
            "the warmed optimizer should hold per-parameter moments"
        );

        let path = dir.join(OPTIMIZER_FILENAME);
        save_optimizer(&warm, &path);
        assert!(path.exists(), "optimizer.bin should be written");

        // A fresh cold optimizer loaded from the snapshot must take the NEXT step
        // identically to the warm one (same momentum + step/bias-correction), and
        // DIFFERENTLY from a truly cold optimizer — that difference is exactly the
        // self-correcting transient a warm resume avoids.
        let restored = load_optimizer(crab_optimizer::<TrainBackend>(), &path, &device);
        assert_eq!(
            restored.to_record().len(),
            warm.to_record().len(),
            "restored optimizer should hold the same per-parameter moment count"
        );

        let (warm_next, _) = adam_test_step(brain.clone(), warm, &device);
        let (restored_next, _) = adam_test_step(brain.clone(), restored, &device);
        let (cold_next, _) =
            adam_test_step(brain.clone(), crab_optimizer::<TrainBackend>(), &device);

        let warm_m = policy_means(&warm_next, &device);
        let restored_m = policy_means(&restored_next, &device);
        let cold_m = policy_means(&cold_next, &device);

        for (i, (a, b)) in warm_m.iter().zip(restored_m.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "restored step diverged from warm at mean[{i}]: {a} vs {b}"
            );
        }
        // The moments must actually matter: a cold optimizer's step differs from the warm
        // one. If not, the round-trip wouldn't be exercising the persisted state.
        let differs = warm_m
            .iter()
            .zip(cold_m.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "cold and warm steps were identical — the test isn't exercising the moments"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_or_unknown_optimizer_state_loads_cold_without_error() {
        let dir = std::env::temp_dir().join("rl_test_adam_compat");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let path = dir.join(OPTIMIZER_FILENAME);

        // (a) Absent file — a pre-rl#60 checkpoint. Loads cold, no panic/error.
        assert!(!path.exists());
        let cold = load_optimizer(crab_optimizer::<TrainBackend>(), &path, &device);
        assert!(
            cold.to_record().is_empty(),
            "an absent optimizer file must leave the optimizer cold"
        );

        // (b) A file tagged with a version this build doesn't recognize (a future format)
        //     is ignored rather than deserialized blind → cold, no panic.
        let bogus = OptimizerCheckpoint {
            version: OPTIMIZER_FORMAT_VERSION + 1,
            record: vec![0xde, 0xad, 0xbe, 0xef],
        };
        std::fs::write(&path, bincode::serialize(&bogus).unwrap()).unwrap();
        let cold2 = load_optimizer(crab_optimizer::<TrainBackend>(), &path, &device);
        assert!(
            cold2.to_record().is_empty(),
            "an unknown-version optimizer file must leave the optimizer cold"
        );

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
                (norm.variance(i) - loaded.variance(i)).abs() < 1e-10,
                "var[{i}] mismatch"
            );
            assert!(
                (norm.m2[i] - loaded.m2[i]).abs() < 1e-10,
                "m2[{i}] mismatch"
            );
        }
        assert_eq!(norm.clip, loaded.clip);
    }

    /// The normalizer merge must be exact: K rollout threads each normalizing their
    /// own slice of samples, then merged on the learner, must give the same running
    /// stats as one single-threaded stream that saw every sample. That equivalence is
    /// the load-bearing correctness check for the multi-threaded rollout. Includes a
    /// NaN-skipped element to exercise the per-element count bookkeeping.
    #[test]
    fn parallel_normalizer_merge_matches_single_stream() {
        let sample = |i: usize| {
            let mut o = [0.0f32; OBS_SIZE];
            o[0] = i as f32;
            o[1] = (i as f32) * 0.5 - 3.0;
            o[2] = ((i * 7) % 11) as f32;
            // Element 3 is present only on even i: counts diverge across elements,
            // so the merge must combine them per element, not with a shared count.
            o[3] = if i.is_multiple_of(2) {
                i as f32
            } else {
                f32::NAN
            };
            o
        };

        // One stream over all 80 samples.
        let mut whole = ObsNormalizer::new(5.0);
        for i in 0..80 {
            whole.normalize(&sample(i));
        }

        // Two independent half-streams, then merge B's stats into A.
        let mut a = ObsNormalizer::new(5.0);
        for i in 0..40 {
            a.normalize(&sample(i));
        }
        let mut b = ObsNormalizer::new(5.0);
        for i in 40..80 {
            b.normalize(&sample(i));
        }
        a.merge(&b.to_data());

        for i in 0..OBS_SIZE {
            assert_eq!(a.count[i], whole.count[i], "count[{i}]");
            assert!(
                (a.mean[i] - whole.mean[i]).abs() < 1e-9,
                "mean[{i}]: merged {} vs whole {}",
                a.mean[i],
                whole.mean[i]
            );
            // M2 (and hence variance) is the part a naive mean-only merge gets
            // wrong; assert it directly. Relative tolerance because M2 grows large.
            let scale = whole.m2[i].abs().max(1.0);
            assert!(
                (a.m2[i] - whole.m2[i]).abs() / scale < 1e-9,
                "m2[{i}]: merged {} vs whole {}",
                a.m2[i],
                whole.m2[i]
            );
        }
    }

    /// CRITICAL regression: the snapshot→roll→merge LOOP must not double-count the
    /// re-handed baseline. Each iteration the learner's master is snapshotted to the
    /// rollout thread, the thread rolls (mutating its copy with this horizon's
    /// samples) and hands back ONLY the per-horizon increment, which the master
    /// merges. After N iterations the master must equal a single stream over every
    /// sample — no doubling. The classic bug ships the cumulative snapshot, so the
    /// master re-merges its own baseline every iteration (C → 2C+S → 4C+3S …); this
    /// models that exact loop with one thread and pins it.
    #[test]
    fn snapshot_roll_merge_loop_matches_single_stream() {
        let sample = |i: usize| {
            let mut o = [0.0f32; OBS_SIZE];
            o[0] = i as f32;
            o[1] = (i as f32) * 0.5 - 2.0;
            o[2] = ((i * 5) % 7) as f32;
            o
        };

        // Ground truth: one normalizer that sees every sample exactly once.
        let mut whole = ObsNormalizer::new(5.0);
        // The learner's master, updated only by merging per-horizon increments.
        let mut master = ObsNormalizer::new(5.0);

        let iters = 5;
        let per_iter = 8;
        let mut next = 0usize;
        for _ in 0..iters {
            // Snapshot: the thread loads the master, and starts a fresh increment
            // over only the samples it is about to see this horizon. The thread's
            // full copy keeps normalizing against baseline+horizon, but only the
            // increment is handed back.
            let mut worker_full = ObsNormalizer::from_data(master.to_data()).expect("snapshot");
            let mut increment = ObsNormalizer::new(master.clip);
            for _ in 0..per_iter {
                let obs = sample(next);
                next += 1;
                whole.normalize(&obs);
                worker_full.normalize(&obs); // policy normalizes with full stats
                increment.observe(&obs); // but only this horizon's samples ship
            }
            // Ship the increment; the master merges samples it has not counted.
            master.merge(&increment.to_data());
        }

        for i in 0..OBS_SIZE {
            assert_eq!(
                master.count[i], whole.count[i],
                "count[{i}] diverged — baseline double-counted?"
            );
            assert!(
                (master.mean[i] - whole.mean[i]).abs() < 1e-9,
                "mean[{i}]: master {} vs single-stream {}",
                master.mean[i],
                whole.mean[i]
            );
            let scale = whole.m2[i].abs().max(1.0);
            assert!(
                (master.m2[i] - whole.m2[i]).abs() / scale < 1e-9,
                "m2[{i}]: master {} vs single-stream {}",
                master.m2[i],
                whole.m2[i]
            );
        }
    }

    use bevy::ecs::system::RunSystemOnce;

    /// Headless training app (physics + bot + training), one fixed tick per
    /// `update()`, one env. The windowless physics+bot stack is the shared
    /// [`crate::bot::test_util::headless_stack`] (same builder `headless_app` and the
    /// rollout workers use); this adds the training systems on top, so these tests
    /// exercise the exact stack the sole trainer runs. Unlike the rollout worker it
    /// keeps the single-world default pool (no K-thread scaling fix needed for one app).
    fn headless_training_app(checkpoint_dir: &std::path::Path) -> App {
        use crate::bot::test_util::{HeadlessStack, WorldRole, headless_stack};
        use clap::Parser;

        // Point the checkpoint dir at an empty scratch path so no real checkpoint
        // loads; every other field keeps its default (tick budget 0 = unlimited,
        // so brain_step never writes AppExit during the test).
        let config = TrainConfig::try_parse_from([
            "rl",
            "--checkpoint-dir",
            checkpoint_dir.to_str().expect("utf-8 checkpoint dir"),
        ])
        .expect("parse default TrainConfig");

        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
        });

        // Wire the training world the same way the `rl learn` rollout worlds do
        // (see inproc::build_rollout_app): worker-mode TrainingState + the Sense→
        // Think→Act systems, so these tests exercise the brain_step / reset_crab /
        // rescue semantics the sole trainer runs. The metrics dir is the per-test
        // scratch checkpoint dir (no shared CSV to clobber).
        let state = TrainingState::new_worker(&config, checkpoint_dir);
        app.insert_non_send_resource(state)
            .add_systems(
                FixedUpdate,
                (brain_step, reset_crab)
                    .chain()
                    .in_set(crate::bot::BotSet::Think),
            )
            .add_systems(Last, save_on_exit);
        app
    }

    fn body_part_entities(app: &mut App) -> std::collections::HashSet<Entity> {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<CrabBodyPart>>();
        q.iter(app.world()).collect()
    }

    /// A crab that goes non-finite is rescued (despawn+respawn) by `rescue_nonfinite_crabs`
    /// BEFORE Sense; the same tick, `brain_step` ends the episode. A rescued env must
    /// respawn EXACTLY ONCE (the rescue's) — `reset_crab` must leave it alone — yet the
    /// episode must still terminate for training.
    ///
    /// The post-tick episode state is identical whether or not reset_crab also respawns
    /// (both end at `Settling { grace: RESET_GRACE_TICKS - 1 }`), so the only discriminator
    /// is ENTITY IDENTITY: the crab after the tick must be the exact set the rescue spawned.
    /// So we drive the rescue tick by hand, capture the rescued entity set, then run
    /// brain_step + reset_crab and assert the set is untouched.
    #[test]
    fn rescued_env_respawns_exactly_once() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_rescue_once_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir);

        // Settle past grace (RESET_GRACE_TICKS) and record a few real steps, so
        // the rescued branch has a recorded step to mark terminal (steps > 0).
        for _ in 0..(RESET_GRACE_TICKS + 8) {
            app.update();
        }
        {
            let st = app.world().non_send_resource::<TrainingState>();
            assert!(
                matches!(st.envs[0].phase, EnvPhase::Recording),
                "settle grace elapsed and no reset pending — env is recording"
            );
            assert!(st.envs[0].steps > 0, "episode should have recorded steps");
        }
        let episodes_before = app
            .world()
            .non_send_resource::<TrainingState>()
            .episode_count;

        // Poison the multibody the way a tunneling blowup does: a non-finite root
        // pose (the path rescue_nonfinite_crabs detects).
        {
            let mut q = app
                .world_mut()
                .query_filtered::<&mut Transform, With<CrabCarapace>>();
            let mut t = q.single_mut(app.world_mut()).expect("carapace");
            t.translation = Vec3::splat(f32::NAN);
        }

        // --- Drive the rescue tick by hand, all within one frame (no update() in
        // between, so the CrabRescued message survives for brain_step to read). ---

        // Phase A: rescue runs (.before(Sense) in the real schedule) — despawns the
        // NaN crab, spawns a fresh one, emits CrabRescued. Capture the fresh set.
        app.world_mut()
            .run_system_once(crate::bot::rescue_nonfinite_crabs)
            .expect("rescue system");
        let rescued_set = body_part_entities(&mut app);
        assert!(
            rescued_set.iter().all(|&e| {
                app.world()
                    .get::<Transform>(e)
                    .is_some_and(|t| t.translation.is_finite())
            }),
            "rescue must leave a finite crab"
        );

        // Phase B: Sense → brain_step → reset_crab, the rest of the tick.
        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step");

        // After brain_step the rescued env must be in Settling (the rescue took the
        // grace itself), NOT AwaitingRespawn — that is what stops reset_crab from
        // respawning it again.
        {
            let st = app.world().non_send_resource::<TrainingState>();
            assert!(
                matches!(st.envs[0].phase, EnvPhase::Settling { grace } if grace == RESET_GRACE_TICKS),
                "rescue path takes the settle grace itself (Settling, not AwaitingRespawn) — \
                 being in Settling and not AwaitingRespawn is what stops reset_crab respawning again"
            );
            assert_eq!(
                st.episode_count,
                episodes_before + 1,
                "the rescue must still terminate the episode for training"
            );
        }

        app.world_mut()
            .run_system_once(reset_crab)
            .expect("reset_crab");

        // The crux: reset_crab must NOT have torn the rescue's crab down and built a
        // third one. The body-part entities after the full tick are EXACTLY the set
        // the rescue spawned.
        let after_set = body_part_entities(&mut app);
        assert_eq!(
            after_set, rescued_set,
            "rescued env was respawned twice in one tick (issue #16): reset_crab \
             replaced the rescue's crab instead of leaving it alone"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    /// Each action's reward and the pose change it produced must occupy the SAME
    /// transition — the one-tick deferral [`Pending`] documents. This pins the phase at the
    /// unambiguous seam, a terminal: tick A chooses action `act_a` at a live height; then we
    /// drop the carapace below the kill floor so tick B reads `h < 0.02` and terminates. The
    /// terminal the kill-floor height produces must carry `act_a` (the action whose result
    /// that height IS), not tick B's action.
    #[test]
    fn height_reward_pairs_with_the_action_that_produced_it() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_phase15_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir);

        // Settle past grace and record a few real steps so the env is Recording
        // with a pending already primed (so tick A below is a steady-state tick).
        for _ in 0..(RESET_GRACE_TICKS + 8) {
            app.update();
        }
        assert!(
            matches!(
                app.world().non_send_resource::<TrainingState>().envs[0].phase,
                EnvPhase::Recording
            ),
            "env must be recording before the hand-driven ticks"
        );

        // Tick A (carapace at its live, above-floor height): Sense → brain_step.
        // This finalizes the pre-existing pending and stashes pending_A — whose
        // action is what brain_step just wrote to CrabActions.
        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation A");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step A");
        let act_a = app.world().resource::<CrabActions>().envs[0];

        // Drop the carapace below the kill floor (0.02 m) so tick B reads a
        // terminal height. With physics not stepped here, this is the exact pose
        // tick B's brain_step sees.
        {
            let mut q = app
                .world_mut()
                .query_filtered::<&mut Transform, With<CrabCarapace>>();
            let mut t = q.single_mut(app.world_mut()).expect("carapace");
            t.translation.y = -1.0;
        }

        let transitions_before = app.world().non_send_resource::<TrainingState>().rollouts[0].len();

        // Tick B: Sense → brain_step. h(s_B) = -1 < 0.02 finalizes pending_A as a
        // terminal. brain_step also writes tick B's own action — capture it to
        // prove the terminal carries act_a, not act_b.
        app.world_mut()
            .run_system_once(crate::bot::sensor::build_observation)
            .expect("build observation B");
        app.world_mut()
            .run_system_once(brain_step)
            .expect("brain_step B");
        let act_b = app.world().resource::<CrabActions>().envs[0];

        let st = app.world().non_send_resource::<TrainingState>();
        let last = st.rollouts[0]
            .transitions
            .last()
            .expect("a transition was pushed");

        // Exactly one transition pushed at tick B (the finalized pending_A).
        assert_eq!(
            st.rollouts[0].len(),
            transitions_before + 1,
            "tick B finalizes exactly the one pending transition"
        );
        assert_eq!(
            last.end,
            StepEnd::Terminal,
            "the sub-floor height read at tick B must terminate the transition"
        );
        // The discriminator only means something if the two actions differ; with
        // independent sampling on different observations they almost surely do.
        assert_ne!(
            act_a, act_b,
            "consecutive sampled actions differ, so the pairing below is decisive"
        );
        assert_eq!(
            last.action, act_a,
            "the terminal height (read at tick B) is paired with act_a — the tick-A \
             action whose physics result that height is — not tick B's action; this \
             is the one-tick phase the fix restores (issue #15)"
        );
        // The env resets after a terminal — no pending may straddle the reset.
        assert!(
            st.envs[0].pending.is_none(),
            "a terminated env carries no pending into its reset"
        );

        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    // ---- Distance curriculum ------------------------------------------------

    /// Feed `n` finished episodes at a fixed reach-fraction into a progress tracker.
    /// `reached` of every `COMPETENCE_WINDOW`-sized chunk are reaches — but the helper
    /// just streams `n` episodes whose reach pattern hits `fraction`.
    fn feed(progress: &mut CurriculumProgress, episodes: usize, fraction: f32) {
        for i in 0..episodes {
            // Deterministic pattern that converges to `fraction` reached: reach iff the
            // running reached-count is below the target ratio. Over a full window this
            // lands within one episode of `fraction`.
            let reached = ((i as f32 + 1.0) * fraction).floor() > (i as f32 * fraction).floor();
            progress.record_episode(reached);
        }
    }

    #[test]
    fn curriculum_starts_at_rung_one() {
        // A fresh curriculum is the near start band — the only band a cold policy can
        // bootstrap from.
        assert_eq!(Curriculum::start().band(), (BAND_START_MIN, BAND_START_MAX));
    }

    #[test]
    fn advances_one_step_when_competence_met() {
        // A full window at/above the threshold advances the band by exactly one STEP,
        // width preserved.
        let mut p = CurriculumProgress::new(Curriculum::start());
        feed(&mut p, COMPETENCE_WINDOW, ADVANCE_REACH_FRACTION);
        assert_eq!(
            p.curriculum().band(),
            (
                BAND_START_MIN + BAND_ADVANCE_STEP,
                BAND_START_MAX + BAND_ADVANCE_STEP
            ),
            "a mastered rung slides the whole band out by one STEP"
        );
    }

    #[test]
    fn does_not_advance_below_threshold_or_before_a_full_window() {
        // Below the reach threshold: no advance no matter how many episodes.
        let mut low = CurriculumProgress::new(Curriculum::start());
        feed(
            &mut low,
            COMPETENCE_WINDOW * 3,
            ADVANCE_REACH_FRACTION - 0.2,
        );
        assert_eq!(
            low.curriculum().band(),
            Curriculum::start().band(),
            "an under-competent policy never advances"
        );
        // At/above threshold but fewer than a full window: still no advance (a short
        // lucky streak must not trip it).
        let mut partial = CurriculumProgress::new(Curriculum::start());
        feed(&mut partial, COMPETENCE_WINDOW - 1, 1.0);
        assert_eq!(
            partial.curriculum().band(),
            Curriculum::start().band(),
            "a partial window cannot advance even at a perfect reach-fraction"
        );
    }

    #[test]
    fn never_regresses_and_caps_at_the_arena() {
        // Mastering rung after rung walks the band outward monotonically and stops at the
        // arena cap — the far edge never exceeds TARGET_ARENA_HALF, and once capped it
        // stays put no matter how much more competence arrives.
        let mut p = CurriculumProgress::new(Curriculum::start());
        let mut prev_min = BAND_START_MIN;
        // Far more competence than any finite number of rungs needs.
        for _ in 0..(2 * (TARGET_ARENA_HALF as usize) + 10) {
            feed(&mut p, COMPETENCE_WINDOW, 1.0);
            let (min, max) = p.curriculum().band();
            assert!(
                min >= prev_min,
                "the band must never slide inward (no regress)"
            );
            assert!(
                max <= TARGET_ARENA_HALF + 1e-6,
                "the far edge must never exceed the arena cap, got {max}"
            );
            prev_min = min;
        }
        let (min, max) = p.curriculum().band();
        assert!(
            (max - TARGET_ARENA_HALF).abs() < 1e-6,
            "enough mastery must drive the far edge to the arena cap, got {max}"
        );
        // Width is preserved across every advance (modulo the final cap clamp, which
        // shortens BOTH edges equally, so the width is identical to the start band's).
        assert!(
            (max - min - (BAND_START_MAX - BAND_START_MIN)).abs() < 1e-6,
            "the band width is invariant across rungs"
        );
        // Capped: `advanced()` yields nothing, so further mastery is a no-op.
        assert_eq!(
            p.curriculum().advanced(),
            None,
            "the capped band cannot advance"
        );
        feed(&mut p, COMPETENCE_WINDOW, 1.0);
        assert_eq!(
            p.curriculum().band(),
            (min, max),
            "a capped curriculum ignores further competence"
        );
    }

    #[test]
    fn advance_clears_the_window_so_the_new_rung_is_judged_fresh() {
        // After an advance the window is empty, so the next rung needs its own full
        // window before it can advance again — competence does not carry across rungs.
        let mut p = CurriculumProgress::new(Curriculum::start());
        feed(&mut p, COMPETENCE_WINDOW, 1.0); // advances to rung 2
        let rung2 = p.curriculum().band();
        // One episode short of a fresh full window on the new rung: must not advance.
        feed(&mut p, COMPETENCE_WINDOW - 1, 1.0);
        assert_eq!(
            p.curriculum().band(),
            rung2,
            "the new rung must accumulate its own full window before advancing"
        );
        // The episode that completes the fresh window advances again.
        feed(&mut p, 1, 1.0);
        assert_ne!(
            p.curriculum().band(),
            rung2,
            "completing a fresh full window at competence advances the new rung"
        );
    }

    #[test]
    fn record_episodes_matches_individual_records() {
        // The pooled-count path the learner uses (`record_episodes`) must advance
        // identically to recording episodes one at a time.
        let mut pooled = CurriculumProgress::new(Curriculum::start());
        pooled.record_episodes(COMPETENCE_WINDOW as u64, COMPETENCE_WINDOW as u64);
        let mut singly = CurriculumProgress::new(Curriculum::start());
        for _ in 0..COMPETENCE_WINDOW {
            singly.record_episode(true);
        }
        assert_eq!(
            pooled.curriculum().band(),
            singly.curriculum().band(),
            "pooled counts and individual records must advance the band identically"
        );
    }

    #[test]
    fn record_episodes_drops_leftovers_after_an_advance() {
        // A pooled batch that advances mid-fold must NOT seed the freshly cleared rung
        // with its remaining episodes — those were rolled against the old (nearer, easier)
        // band. With the window one short of full, the batch's first episode advances and
        // the other nine are leftovers that must be discarded, leaving the new rung empty.
        let mut p = CurriculumProgress::new(Curriculum::start());
        feed(&mut p, COMPETENCE_WINDOW - 1, 1.0);
        let rung1 = p.curriculum().band();
        p.record_episodes(10, 10);
        let rung2 = p.curriculum().band();
        assert_ne!(
            rung2, rung1,
            "the batch's first episode completes the window and advances"
        );
        // Had the nine leftovers seeded rung 2's window, a further WINDOW-1 reached
        // episodes would overfill it and advance again; dropped, WINDOW-1 leaves the new
        // window one short, so the band must stay at rung 2.
        feed(&mut p, COMPETENCE_WINDOW - 1, 1.0);
        assert_eq!(
            p.curriculum().band(),
            rung2,
            "leftover old-band episodes must not seed the freshly cleared rung's window"
        );
    }

    #[test]
    fn missing_or_corrupt_checkpoint_loads_rung_one() {
        let dir = std::env::temp_dir().join(format!("rl-curric-load-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(CURRICULUM_FILENAME);

        // No file at all (fresh run OR a checkpoint predating the curriculum) → rung 1.
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            load_curriculum(&path).band(),
            Curriculum::start().band(),
            "a missing curriculum checkpoint must start at rung 1 (warm-continue safety)"
        );

        // Garbage bytes (corrupt file) → rung 1, not a panic or an illegal band.
        std::fs::write(&path, b"not a curriculum").expect("write garbage");
        assert_eq!(
            load_curriculum(&path).band(),
            Curriculum::start().band(),
            "a corrupt curriculum checkpoint must fall back to rung 1"
        );

        // An in-bounds advanced band round-trips exactly.
        let advanced = advanced_to_cap();
        save_curriculum(advanced, &path);
        assert_eq!(
            load_curriculum(&path).band(),
            advanced.band(),
            "a saved band must reload to the same rung (warm restart continues the curriculum)"
        );

        // A persisted band that violates the invariant (e.g. an out-of-arena far edge)
        // is rejected on load → rung 1.
        let bad = bincode::serialize(&CurriculumData {
            min: 1.5,
            max: TARGET_ARENA_HALF + 5.0,
        })
        .expect("serialize");
        std::fs::write(&path, &bad).expect("write bad band");
        assert_eq!(
            load_curriculum(&path).band(),
            Curriculum::start().band(),
            "an out-of-bounds persisted band must be rejected and fall back to rung 1"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
