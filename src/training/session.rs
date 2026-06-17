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
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use crate::TrainConfig;
use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{CrabAssets, CrabBodyPart, CrabCarapace, CrabEnvId, CrabEyeTip};
use crate::bot::brain::CrabBrain;
use crate::bot::sensor::{CrabObservation, OBS_SIZE};
use crate::bot::{CrabRescued, CrabSpawns, respawn_crab};

use super::algorithm::{
    PpoConfig, PpoMetrics, ReturnNormalizer, ReturnNormalizerData, RolloutBuffer, StepEnd,
    Transition, compute_gae, compute_log_prob, sample_action,
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
/// `pub(crate)`.
///
/// No `var` field: variance is recomputed from `m2`/`count` on load, so carrying it
/// would be OBS_SIZE redundant f64s per snapshot/checkpoint and a drift hazard.
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
    /// the master already counted); merging a cumulative snapshot that re-included
    /// the baseline would double-count it. (The per-element NaN-skip
    /// means counts can differ across elements, which is why the merge is per
    /// element; variance is derived from the merged M2 on demand.)
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

type CrabOptimizer = OptimizerAdaptor<Adam, CrabBrain<TrainBackend>, TrainBackend>;

struct MetricsLogger {
    episode_file: std::fs::File,
    update_file: std::fs::File,
    update_count: u32,
}

impl MetricsLogger {
    /// `dir` is where the two CSVs land. The single-process trainer and the
    /// in-process learner use "tmp" (the established location the plotting scripts
    /// read); a rollout thread passes its own scratch dir so K threads don't clobber
    /// one shared CSV (and the learner keeps owning "tmp").
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

        let up_path = dir.join("ppo_updates.csv");
        let mut update_file =
            std::fs::File::create(&up_path).expect("failed to create ppo_updates.csv");
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

/// Stem for brain checkpoint files. `BinFileRecorder` appends `.bin` automatically,
/// so the actual file on disk is `brain.bin`.
pub(crate) const BRAIN_STEM: &str = "brain";
pub(crate) const NORMALIZER_FILENAME: &str = "normalizer.bin";
/// Return (value-target) normalizer checkpoint, beside the obs normalizer, so a
/// resumed run de-normalizes value predictions against the same scale it trained
/// with (a cold scale on resume would briefly mis-scale the value head).
pub(crate) const RETURN_NORMALIZER_FILENAME: &str = "return_normalizer.bin";

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

/// Physics ticks per PPO update in single-process / windowed training: the
/// rollout window `brain_step` fills before running an update. Also the default
/// for the in-process learner's `--horizon` (one source, so the two can't silently
/// diverge). Worker mode deliberately ignores `steps_per_rollout` — the learner's
/// `--horizon` sets the window and reads the buffers out itself.
pub(crate) const STEPS_PER_ROLLOUT: u32 = 1024;

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
    pub needs_reset: bool,
    /// Settle ticks remaining before the episode starts recording. A reset
    /// respawns a fresh crab in the rest pose ([`crate::bot::respawn_crab`]);
    /// during grace no transitions/termination are evaluated while it drops
    /// from spawn height and the motors take the load.
    pub grace: u32,
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
    optimizer: CrabOptimizer,

    pub envs: Vec<EnvEpisode>,
    pub episode_count: u32,

    /// Ticks per PPO update ([`STEPS_PER_ROLLOUT`]). Worker mode bypasses this:
    /// the learner's `--horizon` bounds the window and the driver reads the
    /// buffers out, so `brain_step`'s rollout-boundary update never fires there.
    pub steps_per_rollout: u32,
    pub current_rollout_steps: u32,

    pub recent_rewards: Vec<f32>,
    pub recent_metrics: Option<PpoMetrics>,

    logger: MetricsLogger,

    pub total_steps: u64,
    last_log_time: std::time::Instant,

    obs_normalizer: ObsNormalizer,

    /// Running mean/std of the value targets (GAE returns), normalizing what the
    /// value head regresses to unit scale so it can track large-magnitude returns
    /// (see [`ReturnNormalizer`]). The LEARNER owns the single copy: rollout threads
    /// never update it — they emit raw value predictions, which the learner
    /// de-normalizes with this scale in `ppo_update_core` — so there is no second
    /// instance to drift. Persisted in the checkpoint beside `obs_normalizer`.
    return_normalizer: ReturnNormalizer,

    checkpoint_dir: PathBuf,
    checkpoint_interval: u32,
    saved_on_exit: bool,

    /// Stop after this many physics ticks (0 = unlimited). See `Args::ticks`.
    tick_budget: u64,
    /// Benchmark: skip NN inference to measure the physics/overhead floor.
    skip_nn: bool,
    /// Rollout-thread (worker) mode: `brain_step` collects transitions exactly as
    /// in single-process but does NOT run the PPO update at the rollout boundary —
    /// the learner owns the update. The in-process rollout thread (`training::inproc`)
    /// reads the buffers out directly each horizon. Default false, so a no-flag run
    /// is byte-for-byte the original loop.
    worker_mode: bool,
    /// Count of `recent_rewards` already handed to the learner (worker mode). The
    /// drain returns the tail past this, so each finished episode's reward reaches
    /// the learner's reward curve exactly once. Stays 0 in single-process.
    reported_episodes: usize,
    /// Worker mode only: a fresh Welford accumulator over ONLY the observations
    /// seen since the last `reset_horizon_counter` (i.e. this horizon's samples).
    /// The thread ships THIS — not the cumulative `obs_normalizer` — so the learner
    /// merges an increment the master hasn't already counted (the snapshot baseline
    /// lives in `obs_normalizer`, never re-merged). `None` in
    /// single-process, where the normalizer is never shipped or merged.
    normalizer_increment: Option<ObsNormalizer>,
}

impl TrainingState {
    /// The in-process learner's policy host (and the test fixtures): logs to "tmp".
    /// The learner steps no world, so the rollout-boundary periodic checkpoint never
    /// fires here — it checkpoints every iteration directly — hence interval 0.
    pub fn new(config: &TrainConfig) -> Self {
        Self::build(config, Path::new("tmp"), false, 0)
    }

    /// Single-process trainer: logs to "tmp", runs the PPO update at each rollout
    /// boundary and checkpoints there every `save_interval` updates.
    pub fn new_single_process(config: &TrainConfig, save_interval: u32) -> Self {
        Self::build(config, Path::new("tmp"), false, save_interval)
    }

    /// In-process rollout thread: collects transitions but never runs the PPO
    /// update locally, and logs to its own `metrics_dir` so K threads don't fight
    /// over one CSV. Everything else — env count, reset/grace/rescue, reward,
    /// normalizer — is identical to single-process, which is what makes a K=1
    /// rollout match an `--envs M` rollout. Never reaches the rollout boundary, so
    /// the periodic-checkpoint interval is irrelevant (0).
    pub fn new_worker(config: &TrainConfig, metrics_dir: &Path) -> Self {
        Self::build(config, metrics_dir, true, 0)
    }

    fn build(
        config: &TrainConfig,
        metrics_dir: &Path,
        worker_mode: bool,
        checkpoint_interval: u32,
    ) -> Self {
        let device = NdArrayDevice::Cpu;
        let mut brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);
        let optimizer: CrabOptimizer = AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
            .init();

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
        // A worker accumulates a per-horizon increment over the same clip; the
        // learner/single-process host never ships a normalizer, so it stays None.
        let normalizer_increment = worker_mode.then(|| ObsNormalizer::new(obs_normalizer.clip));
        Self {
            brain,
            config: PpoConfig::default(),
            rollouts: (0..n).map(|_| RolloutBuffer::new()).collect(),
            device,
            optimizer,
            envs: vec![EnvEpisode::default(); n],
            episode_count: 0,
            steps_per_rollout: STEPS_PER_ROLLOUT,
            current_rollout_steps: 0,
            recent_rewards: Vec::new(),
            recent_metrics: None,
            logger: MetricsLogger::new(metrics_dir),
            total_steps: 0,
            last_log_time: std::time::Instant::now(),
            obs_normalizer,
            return_normalizer,
            checkpoint_dir: config.checkpoint_dir.clone(),
            checkpoint_interval,
            saved_on_exit: false,
            tick_budget: config.ticks,
            skip_nn: config.bench_skip_nn,
            worker_mode,
            reported_episodes: 0,
            normalizer_increment,
        }
    }

    // FOLLOW-UP (out of scope): the checkpoint persists brain + normalizer but NOT
    // the Adam optimizer moments, so a restart resumes the policy with cold moment
    // estimates (a brief, self-correcting transient on the first updates after a
    // resume). Persisting optimizer state would remove it.
    //
    // `pub(crate)` so the in-process learner can persist the latest weights each
    // iteration (for a live demo hot-reload + restart resume); the single-process
    // path calls it internally at the rollout boundary.
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
    // rollout thread: load the learner's snapshot weights + master normalizer, roll
    // a horizon (via the normal systems), then hand the buffers + per-horizon
    // normalizer increment + finished rewards back. The learner side reuses
    // `brain`/`optimizer`/`config` through accessors. Reusing this struct (rather
    // than a parallel one) is what guarantees a rollout thread's collection is the
    // *same* code as single-process — the K=1 == single-process parity anchor.

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
    pub fn set_normalizer(&mut self, data: ObsNormalizerData) {
        self.obs_normalizer.load_snapshot(data);
    }

    /// Snapshot the master normalizer's full stats (learner → rollout threads), so
    /// each thread's policy normalizes observations against the same baseline the
    /// learner holds.
    pub fn normalizer_snapshot(&self) -> ObsNormalizerData {
        self.obs_normalizer.snapshot()
    }

    /// Snapshot the per-horizon normalizer INCREMENT (rollout thread → learner):
    /// only the samples this horizon added, so merging it into the learner's master
    /// — which already holds the snapshot baseline handed back to the thread —
    /// counts every sample exactly once. Empty (count 0) outside worker mode.
    pub fn normalizer_increment_snapshot(&self) -> ObsNormalizerData {
        match self.normalizer_increment.as_ref() {
            Some(inc) => inc.snapshot(),
            None => self.obs_normalizer.snapshot(),
        }
    }

    /// Merge a rollout thread's normalizer increment into this (learner's)
    /// normalizer. The data merged here is ONLY samples the master has not already
    /// counted (the thread ships a per-horizon increment, never the snapshot
    /// baseline).
    pub fn merge_normalizer(&mut self, data: &ObsNormalizerData) {
        self.obs_normalizer.merge(data);
    }

    /// Move the collected transitions out, leaving the buffers empty for the next
    /// horizon. The per-env episode accumulators (`envs`) are deliberately left
    /// untouched: an episode that spans a horizon boundary must continue, exactly
    /// as the single-process 1024-step window cuts mid-episode and keeps going.
    pub fn take_rollouts(&mut self) -> Vec<Vec<Transition>> {
        self.rollouts
            .iter_mut()
            .map(|buf| std::mem::take(&mut buf.transitions))
            .collect()
    }

    /// Reset the per-horizon rollout-step counter AND the normalizer increment
    /// (rollout thread, called at the start of each horizon). `total_steps` stays
    /// monotonic — it is the thread's tick odometer the learner diffs to measure the
    /// horizon length. Resetting the increment here, separate from `set_normalizer`,
    /// keeps the two concerns independent and guarantees the increment is always
    /// exactly this horizon's samples.
    pub fn reset_horizon_counter(&mut self) {
        self.current_rollout_steps = 0;
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

    /// Learner-side accessors: hand the PPO update its pieces (see
    /// `ppo_update_core`). The learner builds rollouts from the threads' returned
    /// buffers rather than stepping any world, so it reaches into the same
    /// brain/optimizer/config/return-normalizer. The return normalizer is the
    /// learner's single copy (rollout threads never touch it), so it is handed out
    /// `&mut` here for the update to fold the iteration's returns into.
    pub fn learner_parts(
        &mut self,
    ) -> (
        &mut CrabBrain<TrainBackend>,
        &mut CrabOptimizer,
        &PpoConfig,
        &NdArrayDevice,
        &mut ReturnNormalizer,
    ) {
        (
            &mut self.brain,
            &mut self.optimizer,
            &self.config,
            &self.device,
            &mut self.return_normalizer,
        )
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

    fn ppo_update(&mut self) -> PpoMetrics {
        ppo_update_core(
            &mut self.brain,
            &mut self.optimizer,
            &self.config,
            &self.rollouts,
            &self.device,
            &mut self.return_normalizer,
        )
    }
}

/// PPO update shared by the single-process trainer and the in-process learner.
///
/// `rollouts` is one buffer per env (GAE is computed strictly per env, never
/// across a buffer boundary). The per-env trailing bootstrap — V of each buffer's
/// non-`done` tail observation — is computed HERE from `brain`, the single owner
/// of that logic: the single-process path already holds the brain it rolled with,
/// and the in-process learner holds the brain it snapshotted to the threads (which
/// is what they rolled with), so neither needs a precomputed value. Mutating
/// `brain`/`optimizer` in place keeps Adam's moment estimates persistent across
/// updates.
///
/// `ret_norm` is the learner's running return scale (see [`ReturnNormalizer`]): the
/// value head's outputs (stored per-step values and the trailing bootstrap) are
/// de-normalized by it so GAE/advantages stay in real reward units, then this
/// update's real-unit returns are folded in and the value-loss targets normalized by
/// the refreshed scale. It is `&mut` because the update advances it; the single
/// learner owns the one copy, so passing it in keeps a single source of truth.
///
/// Factored out (rather than duplicated) so the two callers can never drift: the
/// correctness claim "K=1 in-process == single-process" rests on this being the
/// *same* code, byte for byte.
pub(crate) fn ppo_update_core(
    brain: &mut CrabBrain<TrainBackend>,
    optimizer: &mut CrabOptimizer,
    config: &PpoConfig,
    rollouts: &[RolloutBuffer],
    device: &NdArrayDevice,
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
            // A `done` tail genuinely ended → 0 future return; otherwise bootstrap
            // V(s_tail) with the brain (the trailing obs continues into the next
            // horizon's buffer, so its value carries the cut-off return). The head
            // outputs a NORMALIZED value; `compute_gae` de-normalizes it (and every
            // stored value) so GAE runs in real units. A `done` tail's 0 is a true
            // zero return, not a normalized value — pass it through `normalize` so
            // `compute_gae`'s `denormalize` recovers 0.0 (up to f32 rounding)
            // regardless of μ/σ.
            let last_value = if matches!(last_t.end, StepEnd::Terminal) {
                ret_norm_pre.normalize(0.0)
            } else {
                let obs = Tensor::<TrainBackend, 1>::from_floats(last_t.obs.as_slice(), device)
                    .unsqueeze::<2>();
                brain
                    .value(obs)
                    .flatten::<1>(0, 1)
                    .into_scalar()
                    .elem::<f32>()
            };
            let (a, r) = compute_gae(buf, last_value, config.gamma, config.lambda, &ret_norm_pre);
            advantages.extend(a);
            returns.extend(r);
        }

        // Fold this update's REAL-unit returns into the running scale, then normalize
        // the value-loss targets by the refreshed scale. The value head's raw output
        // is in the same normalized space, so the loss `(V' - R')²` below is unit-
        // scale and `value_loss_clip` is a σ-count (see PpoConfig::value_loss_clip).
        ret_norm.update(&returns);
        let returns: Vec<f32> = returns.iter().map(|&r| ret_norm.normalize(r)).collect();

        // Env-major transition view matching the advantages/returns order.
        let transitions: Vec<&Transition> =
            rollouts.iter().flat_map(|b| b.transitions.iter()).collect();

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
                let new_lp: Tensor<TrainBackend, 1> = log_probs_per_dim.sum_dim(1).flatten(0, 1);

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
                let values: Tensor<TrainBackend, 1> = brain.value(obs).flatten(0, 1);
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

/// Effort tax weight `K` and exponent `L` in `reward = h − K·Σ|aᵢ|^L` (owner's
/// form): carapace pose (height × uprightness) rewarded directly, commanded effort
/// taxed. `K` is small so a
/// calm stand is barely taxed; `L`=2 is cheap for honest moderate commands and
/// steep for max-torque flailing. `K` is provisional — tune from the per-episode
/// effort log so a moderate stand clears a lie-down while saturation-seeking stays
/// punished. The rig body holds itself up by ACTIVE leg actuation (no penetration
/// prop welding the legs into the carapace, as the old hand-coded body had), so a
/// true stand's Σ|aᵢ|² is large — too steep a `K` makes a low-effort list
/// out-reward standing, and the policy correctly learns to lie there.
const EFFORT_COST: f32 = 0.007;
const EFFORT_EXP: f32 = 2.0;

/// Per-output effort tax `f(a) = |a|^L`, summed over the RAW network outputs
/// (the sampled pre-clamp actions — see [`brain_step`]). The point is the gradient
/// past the clamp: the sim clamps actions to ±1, but `|a|^L` keeps rising beyond
/// ±1, so an action that overshoots the usable range is taxed in proportion to the
/// overshoot. The old `e^|clamp(a)|−1` taxed the *clamped* value, so its gradient
/// went flat at ±1 — the policy paid a fixed toll but felt no pull back into range,
/// and the toll was steep enough to make lying down out-reward standing. Quadratic
/// (L=2) is cheap for honest moderate commands and steep for the max-torque flailing.
pub(crate) fn action_effort(raw_actions: &[f32; ACTION_SIZE]) -> f32 {
    raw_actions.iter().map(|a| a.abs().powf(EFFORT_EXP)).sum()
}

/// Reward: carapace pose rewarded, commanded effort taxed —
/// `reward = (h·u) − K·Σ|aᵢ|^L`, where `h` is the carapace's world height and `u`
/// its uprightness (carapace up · world Y). Both signals are global, so behaviour
/// still EMERGES rather than being hand-specified (owner's call: mechanical terms
/// like "feet on the ground" don't scale).
///
/// `h·u` is high only when the carapace is BOTH elevated (legs extended underneath)
/// and LEVEL (up·Y → 1) — i.e. standing. An earlier eye-tip-height proxy was gamed:
/// the policy reared the body to point the long eye-stalks up (cheap height) without
/// standing level. Reading the carapace instead, gated by uprightness, removes that
/// stalk lever. Over a long episode the summed return favours a SUSTAINED pose.
/// `effort` is [`action_effort`]: the policy is charged for how hard it commands, so
/// flailing costs whatever motion it buys.
fn compute_reward(pose_height: f32, effort: f32) -> f32 {
    pose_height - EFFORT_COST * effort
}

/// System: runs the brain to produce actions each physics step.
#[allow(clippy::too_many_arguments)]
pub fn brain_step(
    mut training: NonSendMut<TrainingState>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    parts_q: Query<(&CrabEnvId, &bevy_rapier3d::prelude::Velocity), With<CrabBodyPart>>,
    eyes_q: Query<(&CrabEnvId, &Transform), With<CrabEyeTip>>,
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

    // Normalize every env's observation (each row also updates the shared
    // running stats — N envs feed the same normalizer, just more samples). In
    // worker mode the SAME raw rows feed a per-horizon increment accumulator, so
    // the thread ships only this horizon's samples (the master's baseline, already
    // on the learner, is never re-merged — see `normalizer_increment`).
    let mut obs_arrays: Vec<[f32; OBS_SIZE]> = Vec::with_capacity(n);
    for e in 0..n {
        let normalized = training.obs_normalizer.normalize(&obs.envs[e]);
        if let Some(inc) = training.normalizer_increment.as_mut() {
            inc.observe(&obs.envs[e]);
        }
        obs_arrays.push(normalized);
    }

    // ONE batched forward pass for all envs: [n, OBS_SIZE] through the trunk
    // once — this is what makes N crabs cheaper than N apps.
    let (means_rows, log_std, values): (Vec<Tensor<NdArray, 1>>, Tensor<NdArray, 1>, Vec<f32>) =
        if training.skip_nn {
            // Bench mode: no forward pass. The sampling below is cheap and what
            // it produces is irrelevant — we are isolating physics + overhead.
            let z = Tensor::<NdArray, 1>::zeros([ACTION_SIZE], &device);
            (vec![z.clone(); n], z, vec![0.0; n])
        } else {
            let inference_brain = training.brain.valid();
            let flat: Vec<f32> = obs_arrays.iter().flat_map(|a| a.iter().copied()).collect();
            let obs_batch = Tensor::<NdArray, 2>::from_data(
                burn::tensor::TensorData::new(flat, [n, OBS_SIZE]),
                &device,
            );
            let (means_batch, log_std) = inference_brain.policy(obs_batch.clone());
            let values: Vec<f32> = inference_brain
                .value(obs_batch)
                .flatten::<1>(0, 1)
                .to_data()
                .to_vec()
                .unwrap();
            let means_rows = (0..n)
                .map(|e| {
                    means_batch
                        .clone()
                        .slice([e..e + 1, 0..ACTION_SIZE])
                        .flatten(0, 1)
                })
                .collect();
            (means_rows, log_std, values)
        };

    // Per-env action sampling + NaN guards.
    let mut action_arrays: Vec<[f32; ACTION_SIZE]> = Vec::with_capacity(n);
    // Raw (pre-clamp) actions, kept only to compute the effort tax: the tax must
    // see the unbounded network output so it can push a saturating logit back, not
    // the ±1-clamped value the sim is fed.
    let mut raw_action_arrays: Vec<[f32; ACTION_SIZE]> = Vec::with_capacity(n);
    let mut log_probs: Vec<f32> = Vec::with_capacity(n);
    for means in &means_rows {
        let action_tensor = sample_action(means, &log_std, &device);
        let log_prob = compute_log_prob(means, &log_std, &action_tensor);
        log_probs.push(if log_prob.is_nan() || log_prob.is_infinite() {
            0.0
        } else {
            log_prob.clamp(-20.0, 20.0)
        });

        let action_data: Vec<f32> = action_tensor.to_data().to_vec().unwrap();
        let mut action_array = [0.0f32; ACTION_SIZE];
        let mut raw_action_array = [0.0f32; ACTION_SIZE];
        let mut has_nan = false;
        for (i, &v) in action_data.iter().enumerate().take(ACTION_SIZE) {
            if v.is_nan() || v.is_infinite() {
                has_nan = true;
                action_array[i] = 0.0;
                raw_action_array[i] = 0.0;
            } else {
                raw_action_array[i] = v;
                action_array[i] = v.clamp(-1.0, 1.0);
            }
        }
        if has_nan {
            warn!("NaN/Inf detected in NN output, clamping to zero");
        }
        action_arrays.push(action_array);
        raw_action_arrays.push(raw_action_array);
    }
    actions.envs.copy_from_slice(&action_arrays);
    // Settling envs hold the rest pose (action 0); the policy takes over at
    // step 0 of the new episode.
    for (e, ep) in training.envs.iter().enumerate() {
        if ep.grace > 0
            && let Some(v) = actions.envs.get_mut(e)
        {
            *v = [0.0; ACTION_SIZE];
        }
    }

    // Gather per-env body state: carapace pose + mean eye-tip height.
    let mut poses: Vec<Option<(f32, f32)>> = vec![None; n]; // (height, upright)
    for (env, transform) in carapace_q.iter() {
        if let Some(p) = poses.get_mut(env.0) {
            let up = transform.rotation * Vec3::Y;
            *p = Some((transform.translation.y, up.dot(Vec3::Y)));
        }
    }
    // Fastest body part per env — limbs, not the carapace, blow up first (tiny
    // eye-stalk balls + acceleration motors), so the blowup guard must watch
    // every body. NaN poisons the max, so fold it in as +inf.
    let mut max_speeds: Vec<f32> = vec![0.0; n];
    let mut sq_angvels: Vec<f32> = vec![0.0; n];
    for (env, vel) in parts_q.iter() {
        if let Some(m) = max_speeds.get_mut(env.0) {
            let lin = vel.linear.length();
            let ang = vel.angular.length();
            let s = if lin.is_finite() && ang.is_finite() {
                // Angular blowups (rad/s) run ~3x the linear scale before the
                // solver NaNs; fold both into one number on the linear scale.
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
    let mut eye_sums: Vec<(f32, u32)> = vec![(0.0, 0); n];
    for (env, eye) in eyes_q.iter() {
        if let Some(s) = eye_sums.get_mut(env.0) {
            s.0 += eye.translation.y;
            s.1 += 1;
        }
    }
    // Commanded effort this step (the reward's tax term), per env — taxed on the
    // RAW outputs, not the clamped actions the sim ran.
    let efforts: Vec<f32> = raw_action_arrays.iter().map(action_effort).collect();

    // Per-env reward, termination, rollout push, and episode bookkeeping.
    for e in 0..n {
        // No transitions while the env is settling into its start pose (or, as
        // a guard, if its crab is somehow absent this tick).
        if training.envs[e].grace > 0 || poses[e].is_none() {
            continue;
        }
        // Does the episode end this tick? Three paths: rescued (no transition),
        // true terminal (done), truncation (cut by the step cap).
        let episode_ended = if rescued_envs.contains(&e) {
            // Rescued: the crab went non-finite and was force-respawned this tick
            // (rescue runs .before(Sense)), so every pose read above is the FRESH
            // crab back at spawn, not the state the last action produced.
            // Recording it would credit a blow-up with the spawn pose's height —
            // a positive reward on a failure — so push nothing. Instead mark the
            // previously recorded step terminal: it was the real final step
            // before the blow-up, and leaving it done=false would let GAE
            // bootstrap it across the reset seam from the NEXT episode's value.
            // If this episode recorded nothing yet (rescued during settle), there
            // is no step to mark and no episode to log.
            if training.envs[e].steps > 0 {
                if let Some(last) = training.rollouts[e].transitions.last_mut() {
                    last.end = StepEnd::Terminal;
                }
                true
            } else {
                false
            }
        } else {
            let (height, upright) = poses[e].expect("poses[e].is_none() handled above");
            // Carapace pose reward: world height scaled by uprightness. The earlier
            // eye-tip-height proxy was gamed — the policy reared the body up to lift
            // the long eye-stalks (cheap height) instead of standing level — so reward
            // the carapace pose directly: only a LEVEL (up·Y → 1) and HIGH carapace
            // scores, and a tilted rear-brace is discounted by its low up·Y. (eye_sums
            // is gathered above but currently unused.)
            // NOTE: the height term reads s_t (this tick is pre-physics), so it
            // is one tick out of phase with the action it pairs with; the effort
            // term is correctly phased. Small, deliberately deferred — see
            // https://github.com/bddap/rl/issues/15.
            let reward = compute_reward(height * upright.max(0.0), efforts[e]);
            // Termination is survival guards only — jumping, flipping, and
            // any other strategy the policy invents are legitimate solutions
            // (owner call: emergent behavior is the point). The height band
            // is sim sanity (clipped through the floor / left the playfield),
            // not a behavior bound. The blowup check only catches a genuine
            // numerical explosion before the solver NaNs and Rapier panics the
            // whole app; the threshold is high because direct torque is bounded
            // (no acceleration-motor energy pump), so ordinary vigorous,
            // limb-flinging motion is legal — only a part moving at clearly
            // unphysical speed ends the episode.
            let blowing_up = max_speeds[e] > 100.0 || !height.is_finite();
            let done = !(0.02..=50.0).contains(&height) || blowing_up;
            // The step cap is a TRUNCATION, not a failure: a crab still standing
            // at the cap was cut short, so GAE must bootstrap its value rather
            // than learn the cap is a dead end (see Transition::truncated).
            let truncated = !done && training.envs[e].steps > 1500;

            let end = if done {
                StepEnd::Terminal
            } else if truncated {
                StepEnd::Truncated
            } else {
                StepEnd::Continues
            };
            training.rollouts[e].push(Transition {
                obs: obs_arrays[e],
                action: action_arrays[e],
                reward,
                value: values[e],
                log_prob: log_probs[e],
                end,
            });

            let ep = &mut training.envs[e];
            ep.reward += reward;
            ep.steps += 1;
            ep.height_sum += height;
            ep.upright_sum += upright;
            ep.sq_angvel_sum += sq_angvels[e];

            done || truncated
        };

        if episode_ended {
            let ep = &training.envs[e];
            let ep_reward = ep.reward;
            let ep_steps = ep.steps;
            let ep_height = ep.height_sum / ep_steps.max(1) as f32;
            let ep_upright = ep.upright_sum / ep_steps.max(1) as f32;
            let ep_sq_angvel = ep.sq_angvel_sum / ep_steps.max(1) as f32;
            // A rescued env was already despawned+respawned this tick by
            // rescue_nonfinite_crabs (runs .before(Sense)); asking reset_crab for
            // a second respawn would tear down that fresh crab — which has lived
            // zero ticks — and rebuild an identical one (issue #16). So the rescue
            // path owns the reset: take its grace here and leave needs_reset clear,
            // so reset_crab respawns only envs that ended a NORMAL episode.
            training.envs[e] = if rescued_envs.contains(&e) {
                EnvEpisode {
                    grace: RESET_GRACE_TICKS,
                    ..EnvEpisode::default()
                }
            } else {
                EnvEpisode {
                    needs_reset: true,
                    ..EnvEpisode::default()
                }
            };

            training.recent_rewards.push(ep_reward);
            training.episode_count += 1;
            let avg = training.avg_reward(10);
            let ep_count = training.episode_count;

            training.logger.log_episode(
                ep_count,
                ep_reward,
                ep_steps,
                avg,
                ep_height,
                ep_upright,
                ep_sq_angvel,
            );

            if training.episode_count.is_multiple_of(10) {
                let elapsed = training.last_log_time.elapsed().as_secs_f32();
                let total_transitions =
                    training.total_steps * n as u64 + (e as u64 + 1).min(n as u64);
                let sps = if elapsed > 0.0 {
                    total_transitions as f32 / elapsed
                } else {
                    0.0
                };
                let buffered: usize = training.rollouts.iter().map(|b| b.len()).sum();
                // Σω² is telemetry only — never enters the reward. (The other
                // labels spell out their scope inline.)
                info!(
                    "Ep {} | avg reward(10): {:.2} | last ep (1 env): {} steps, height {:.2}, upright {:.2}, Σω² {:.0} | buffer {} | {:.0} steps/s (lifetime avg)",
                    training.episode_count,
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

    training.current_rollout_steps += n as u32;
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

    // Worker mode never runs a local PPO update: the learner owns it. The rollout
    // thread (`training::inproc`) reads the buffers out after its fixed horizon and
    // clears them, so this boundary must not fire (it would consume the rollout the
    // learner is about to collect).
    if !training.worker_mode && training.current_rollout_steps >= training.steps_per_rollout {
        let buffer_size: usize = training.rollouts.iter().map(|b| b.len()).sum();
        let avg = training.avg_reward(10);

        info!("Running PPO update on {} transitions...", buffer_size);
        let metrics = training.ppo_update();

        info!(
            "PPO | Policy loss: {:.4} | Value loss: {:.4} | Entropy: {:.4}",
            metrics.policy_loss, metrics.value_loss, metrics.entropy,
        );

        training.logger.log_update(&metrics, avg, buffer_size);

        training.recent_metrics = Some(metrics);
        for buf in training.rollouts.iter_mut() {
            buf.clear();
        }
        training.current_rollout_steps = 0;

        if training.checkpoint_interval > 0
            && training
                .logger
                .update_count
                .is_multiple_of(training.checkpoint_interval)
        {
            training.save_checkpoint();
        }
    }
}

/// Settle ticks after a respawn: the fresh crab spawns in the rest pose with
/// the builder motors already holding it, so this only covers the drop from
/// spawn height onto the ground and the motors taking the load (0.5 s).
const RESET_GRACE_TICKS: u32 = 32;

/// System: rebuilds each env's crab when that env's episode ends by a normal
/// terminal/truncation (the `needs_reset` flag [`brain_step`] sets). An episode
/// ended by a non-finite *rescue* is deliberately NOT handled here — that crab
/// was already respawned this tick by [`rescue_nonfinite_crabs`], so the rescue
/// path takes the grace itself and leaves `needs_reset` clear (issue #16); a
/// respawn here would just rebuild the fresh crab a second time.
///
/// A reset is a full despawn + respawn ([`respawn_crab`]): teleporting bodies
/// cannot repair a multibody whose joint state went non-finite — rapier 0.32
/// offers no way to rewrite multibody joint coordinates in place — and one
/// crab that tunnels through the floor would otherwise wedge its env forever.
/// The respawned crab starts in the overlap-free rest pose, so no unfold or
/// collision-group dance is needed; the grace just skips recording while it
/// takes load (see [`EnvEpisode::grace`]).
pub fn reset_crab(
    mut commands: Commands,
    mut training: NonSendMut<TrainingState>,
    mut actions: ResMut<CrabActions>,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>,
) {
    for (e, ep) in training.envs.iter_mut().enumerate() {
        if std::mem::take(&mut ep.needs_reset) {
            ep.grace = RESET_GRACE_TICKS;
            if let Some(v) = actions.envs.get_mut(e) {
                *v = [0.0; ACTION_SIZE];
            }
            let origin = spawns.0.get(e).copied().unwrap_or(Vec3::ZERO);
            respawn_crab(
                &mut commands,
                &assets,
                parts.iter().filter(|(_, id)| id.0 == e).map(|(ent, _)| ent),
                origin,
                e,
            );
        }
    }

    for ep in training.envs.iter_mut() {
        if ep.grace > 0 {
            ep.grace -= 1;
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
    fn reward_increases_with_pose_height() {
        // A higher carapace pose score `h·u` (standing) must score strictly above a
        // low one (collapsed/flat/tilted) at equal effort.
        assert!(
            compute_reward(1.2, 0.0) > compute_reward(0.2, 0.0),
            "reward must increase with carapace pose height"
        );
    }

    #[test]
    fn effort_cost_calibration() {
        // The tax `K·Σ|a|^L` must leave the optimum at a standing pose, not a
        // lie-down, while still punishing a policy that drives its outputs into
        // saturation. Three checkpoints (exact K is tuned from the effort log; the
        // ordering is what must hold):
        // 1. A still policy (zero command) pays no tax — reward is pure height.
        let still = compute_reward(0.5, action_effort(&[0.0; ACTION_SIZE]));
        assert!(
            (still - 0.5).abs() < 1e-6,
            "a still policy is untaxed: {still}"
        );

        // 2. A moderate stand (raw |a| well inside the usable ±1) at stand height
        //    out-rewards a low, still crouch — so standing, not lying down, wins.
        let moderate_stand = compute_reward(1.2, action_effort(&[0.4; ACTION_SIZE]));
        assert!(
            moderate_stand > still,
            "moderate stand must beat a still crouch: {moderate_stand} vs {still}"
        );

        // 3. A saturation-seeking command (raw outputs driven far past the ±1 the
        //    sim clamps to) is taxed below the moderate stand — the |a|^L gradient
        //    pushes the policy OUT of saturation, where the old flat-at-clamp tax
        //    let it sit pinned for free.
        let oversaturated = compute_reward(1.2, action_effort(&[3.0; ACTION_SIZE]));
        assert!(
            oversaturated < moderate_stand,
            "saturation-seeking must be taxed below a moderate stand: {oversaturated}"
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
    /// stats as one stream that saw every sample. This is what lets a K>1 run claim
    /// the same observation normalization as single-process — so it is the
    /// load-bearing correctness check for the multi-threaded path. Includes a
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
    use bevy_rapier3d::prelude::*;

    /// Headless training app (physics + bot + training), one fixed tick per
    /// `update()`, one env. Mirrors `bot::test_util::headless_app` plus the
    /// training systems; that helper is private to `bot`, so we rebuild it here.
    fn headless_training_app(checkpoint_dir: &std::path::Path) -> App {
        use crate::Visuals;
        use crate::bot::{BotPlugin, NumEnvs};
        use crate::physics::PhysicsWorldPlugin;
        use crate::training::TrainingPlugin;
        use clap::Parser;
        use std::time::Duration;

        // Point the checkpoint dir at an empty scratch path so no real checkpoint
        // loads; every other field keeps its default (tick budget 0 = unlimited,
        // so brain_step never writes AppExit during the test).
        let config = TrainConfig::try_parse_from([
            "rl",
            "--checkpoint-dir",
            checkpoint_dir.to_str().expect("utf-8 checkpoint dir"),
        ])
        .expect("parse default TrainConfig");

        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(bevy::window::WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    ..default()
                })
                .set(bevy::render::RenderPlugin {
                    render_creation: bevy::render::settings::RenderCreation::Automatic(
                        bevy::render::settings::WgpuSettings {
                            backends: None,
                            ..default()
                        },
                    ),
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>()
                .disable::<bevy::log::LogPlugin>(),
        );
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f64(1.0 / 64.0),
        ));
        app.insert_resource(Visuals(false))
            .insert_resource(NumEnvs(1))
            // CrabAssets builds the rig-derived one model (no body-source switch).
            // Same fixed timestep as production (one source — see physics::fixed_timestep)
            // so this test runs the physics the demo/training loop actually uses.
            .insert_resource(crate::physics::fixed_timestep())
            .insert_resource(crate::physics::rapier_context_init())
            .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
            .add_plugins(PhysicsWorldPlugin)
            .add_plugins(BotPlugin)
            .add_plugins(TrainingPlugin::new(config, 0));
        app
    }

    fn body_part_entities(app: &mut App) -> std::collections::HashSet<Entity> {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<CrabBodyPart>>();
        q.iter(app.world()).collect()
    }

    /// Issue #16: a crab that goes non-finite is rescued (despawn+respawn) by
    /// `rescue_nonfinite_crabs` BEFORE Sense; the same tick, `brain_step` ends the
    /// episode and `reset_crab` used to honor `needs_reset` and respawn a SECOND
    /// time — so the rescue's fresh crab lived zero ticks. This pins the fix: a
    /// rescued env respawns EXACTLY ONCE (the rescue's), reset_crab leaves it
    /// alone, and the episode still terminates for training.
    ///
    /// The post-tick *episode state* is identical with or without the bug (both
    /// end at grace = RESET_GRACE_TICKS-1, needs_reset = false), so the only thing
    /// that distinguishes "respawned once" from "twice" is ENTITY IDENTITY: the
    /// crab present after the tick must be the exact set the rescue spawned. We
    /// therefore drive the rescue tick by hand, capture the rescued entity set,
    /// then run brain_step + reset_crab and assert the set is untouched.
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
            assert_eq!(st.envs[0].grace, 0, "grace should have elapsed");
            assert!(st.envs[0].steps > 0, "episode should have recorded steps");
            assert!(!st.envs[0].needs_reset, "no pending reset before NaN");
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

        // After brain_step the rescued env must be set up for the rescue to own the
        // reset: grace taken, needs_reset NOT set (that is what stops reset_crab
        // from respawning again).
        {
            let st = app.world().non_send_resource::<TrainingState>();
            assert!(
                !st.envs[0].needs_reset,
                "rescued env must not request a second respawn (needs_reset stays clear)"
            );
            assert_eq!(
                st.envs[0].grace, RESET_GRACE_TICKS,
                "rescue path must take the settle grace itself"
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
}
