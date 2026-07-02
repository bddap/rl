//! [`TrainingState`] — the learner/rollout-thread state — and the per-horizon protocol
//! ([`HorizonRequest`]/[`HorizonOutput`], [`TrainingState::begin_horizon`]/[`end_horizon`])
//! plus checkpoint save/load. The per-tick step lives in [`super::step`], the episode
//! lifecycle in [`super::lifecycle`].

use std::cell::RefCell;
use std::path::PathBuf;

use burn::backend::ndarray::NdArrayDevice;
use burn::module::AutodiffModule;
use burn::record::{BinBytesRecorder, FullPrecisionSettings};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tracing::{error, info, warn};

use crate::TrainConfig;
use crate::bot::arch::{AnyBrain, ArchId};
use crate::training::algorithm::{OuNoise, PpoConfig, ReturnNormalizer, RolloutBuffer};
use crate::training::checkpoint::{
    BrainLoadError, CheckpointDir, load_brain_file, load_return_normalizer, save_brain,
    save_return_normalizer,
};
use crate::training::curriculum::TargetBand;
use crate::training::envelope::EnvelopeError;
use crate::training::normalizer::{
    IncrementAccumulator, NORMALIZER_CLIP, NormalizerIncrement, NormalizerSnapshot, ObsNormalizer,
};
use crate::training::{InferBackend, TrainBackend};

use super::lifecycle::EnvEpisode;

/// Default rollout horizon: the number of physics ticks each rollout thread rolls
/// per iteration before handing its buffers back, when `--horizon` is not given.
pub const STEPS_PER_ROLLOUT: u32 = 1024;

/// The live PPO brain plus a lazily-built inference clone on the inner backend.
/// A rollout thread changes the weights once per horizon (it loads the learner's
/// snapshot, then steps the whole window) yet `forward_pass` runs every tick;
/// `AutodiffModule::valid()` rebuilds the inference module from scratch, so running it
/// every tick between those rare changes is wasted work. The inference clone is built on
/// demand and reused until the train weights change — and the only way to change them is
/// a `&mut`/`set`, both of which drop the cache, so a stale clone can't be served.
pub(super) struct InferenceCachedBrain {
    pub(super) train: AnyBrain<TrainBackend>,
    /// Cleared whenever `train` is mutated, rebuilt on the next inference. `RefCell`
    /// because `forward_pass` reads the state through a shared `&TrainingState` (it runs
    /// as a Bevy system over a non-send resource) yet must refresh this cache; the state
    /// is single-threaded, so the borrow is never contended.
    inference: RefCell<Option<AnyBrain<InferBackend>>>,
}

impl InferenceCachedBrain {
    pub(super) fn new(train: AnyBrain<TrainBackend>) -> Self {
        Self {
            train,
            inference: RefCell::new(None),
        }
    }

    /// Read-only access to the live train brain (checkpoint save, snapshot bytes).
    fn train(&self) -> &AnyBrain<TrainBackend> {
        &self.train
    }

    /// Mutable access to the train brain, dropping the inference cache: the only reason
    /// to take `&mut` is to change the weights, so the clone is now stale.
    pub(super) fn train_mut(&mut self) -> &mut AnyBrain<TrainBackend> {
        *self.inference.get_mut() = None;
        &mut self.train
    }

    /// Replace the train brain wholesale (a snapshot/checkpoint load), dropping the cache.
    pub(super) fn set(&mut self, train: AnyBrain<TrainBackend>) {
        self.train = train;
        *self.inference.get_mut() = None;
    }

    /// Run `f` against the cached inference brain, building it from the current train
    /// weights first if the cache is cold. Bit-identical to calling `train.valid()` afresh
    /// each time, since `valid()` is a pure detach of the same weights.
    pub(super) fn with_inference<R>(&self, f: impl FnOnce(&AnyBrain<InferBackend>) -> R) -> R {
        let mut slot = self.inference.borrow_mut();
        let brain = slot.get_or_insert_with(|| self.train.valid());
        f(brain)
    }
}

/// Stored as a non-send resource because burn tensors use `OnceCell`, which is
/// not `Sync`.
pub(crate) struct TrainingState {
    pub(super) brain: InferenceCachedBrain,
    pub config: PpoConfig,
    /// One rollout buffer per env — kept separate so GAE never sweeps across an env boundary.
    pub rollouts: Vec<RolloutBuffer>,
    pub device: NdArrayDevice,

    pub envs: Vec<EnvEpisode>,
    pub episode_count: u32,

    pub recent_rewards: Vec<f32>,

    pub total_steps: u64,

    pub(super) obs_normalizer: ObsNormalizer,

    /// Running mean/std of the value targets (GAE returns), owned by the LEARNER and
    /// persisted in the checkpoint beside `obs_normalizer` (see [`ReturnNormalizer`]).
    pub(super) return_normalizer: ReturnNormalizer,

    /// The run's seeded RNG threaded to every stochastic choice (action noise, target
    /// placement, spawn rotation); per-state so each thread draws an independent stream.
    pub(super) rng: StdRng,

    /// Per-env temporally-correlated exploration noise (the `ε` in `μ + σ·ε`, bddap/rl#161);
    /// rollout-thread-local and reset per episode, so it is NOT checkpointed.
    pub(super) explore_noise: OuNoise,

    /// This horizon's exploration-σ floor — the lower `log_std` clamp the rollout samples
    /// under, shipped down by the learner each horizon (bddap/rl#161).
    pub(super) log_std_floor: f32,

    pub(super) checkpoint_dir: PathBuf,
    pub(super) saved_on_exit: bool,

    /// Whether the brain actually warm-started from the checkpoint dir at build time.
    /// The optimizer's warm/cold gate ([`super::super::inproc`]): Adam moments are
    /// per-parameter, so they resume iff the brain they belong to did.
    warm_started: bool,

    /// Stop after this many physics ticks (0 = unlimited). See `Args::ticks`.
    pub(super) tick_budget: u64,
    /// Count of `recent_rewards` already handed to the learner, so each finished episode's
    /// reward reaches the learner's reward curve exactly once.
    pub(super) reported_episodes: usize,
    /// This horizon's Welford increment over only its own observations (worker mode); the
    /// thread ships THIS, not the cumulative `obs_normalizer`, so the learner never
    /// double-counts. `None` on the learner's host, which never rolls.
    pub(super) normalizer_increment: Option<IncrementAccumulator>,

    /// Running `(sum, count)` of the carapace planar drift-from-spawn over recording envs
    /// this horizon — the walking diagnostic, drained per horizon and logged as a mean.
    pub(super) drift_sum: f64,
    pub(super) drift_count: u64,

    /// This horizon's target-distance band (learner → thread each horizon via
    /// [`Self::set_band`]); the thread only reads it in `seed_target`.
    pub(super) band: TargetBand,
    /// This horizon's per-episode reach tally over FINISHED episodes as `(reached,
    /// finished)` — the curriculum's competence signal, drained per horizon.
    pub(super) reach_reached: u64,
    pub(super) reach_finished: u64,

    /// Count of transitions this horizon whose progress term was DROPPED as non-physical,
    /// drained per horizon so a silent reward dropout surfaces on the learner line
    /// (bddap/rl#175); 0 on a healthy run.
    pub(super) progress_glitch_drops: u64,

    /// Count of observation ELEMENTS this horizon that were non-finite and so skipped from
    /// the normalizer, drained per horizon so a NaN reading surfaces (bddap/rl#181); 0 on a
    /// healthy horizon.
    pub(super) nonfinite_obs_elements: u64,
}

/// What the learner ships a rollout thread to OPEN one horizon (bddap/rl#165): the policy
/// weight snapshot, the master normalizer baseline, and this horizon's target band + σ-floor.
/// Borrowed weights (the learner's `Arc<Vec<u8>>` bytes) — [`TrainingState::begin_horizon`]
/// consumes it once and keeps nothing. One shape for the whole per-horizon setup, so the
/// load→set→reset sequence lives behind `begin_horizon` instead of leaking into `inproc`.
pub(crate) struct HorizonRequest<'a> {
    pub brain_bytes: &'a [u8],
    pub normalizer: NormalizerSnapshot,
    pub band: TargetBand,
    pub log_std_floor: f32,
}

/// Everything one horizon PRODUCES for the learner to merge, moved out in a single shot by
/// [`TrainingState::end_horizon`] (bddap/rl#165). Folding the per-metric `drain_*` accessors
/// into one output shape keeps the collection sequence inside `state` — the fixed order
/// no longer leaks across `state`↔`inproc.rs`, and a new per-horizon metric is one field
/// here, not a new accessor plus a matching drain call at the far call site. `ticks` is NOT
/// here: it is the thread's own tick-odometer diff, measured by `inproc`, not drained from state.
pub(crate) struct HorizonOutput {
    /// Per-env rollout buffers (one per env; GAE never sweeps across envs), each carrying
    /// its own GAE tail bootstrap.
    pub envs: Vec<RolloutBuffer>,
    /// Per-horizon normalizer increment — only this horizon's observations, so merging it
    /// into the master (which holds the baseline) never double-counts.
    pub increment: NormalizerIncrement,
    /// Rewards of episodes that finished during this horizon.
    pub rewards: Vec<f32>,
    /// Carapace planar drift-from-spawn this horizon as `(sum, count)` over recording ticks.
    pub drift: (f64, u64),
    /// This horizon's per-episode reach tally as `(reached, finished)`.
    pub reach: (u64, u64),
    /// Count of progress terms dropped as non-physical this horizon (bddap/rl#175).
    pub glitch_drops: u64,
    /// Count of non-finite observation elements skipped from the normalizer (bddap/rl#181).
    pub nonfinite_obs: u64,
}

impl TrainingState {
    /// The learner's policy host (and the test fixtures). The learner steps no world
    /// itself — it owns the policy and runs the PPO update from the threads' buffers,
    /// checkpointing every iteration directly.
    pub fn new(config: &TrainConfig) -> Self {
        Self::build(config, false, 0)
    }

    /// In-process rollout thread: collects transitions but never runs the PPO update
    /// locally (the learner does). `worker_mode` turns on the per-horizon normalizer
    /// increment the thread ships back; `worker_index` mixes into the RNG seed so each
    /// thread explores an independent stream even under a fixed `--seed` (see
    /// [`Self::build`]). Everything else (env count, reset/grace/rescue, reward) is the
    /// shared per-env machinery.
    pub fn new_worker(config: &TrainConfig, worker_index: usize) -> Self {
        Self::build(config, true, worker_index)
    }

    fn build(config: &TrainConfig, worker_mode: bool, worker_index: usize) -> Self {
        let device = NdArrayDevice::Cpu;

        // Resolve the run's RNG seed: an explicit `--seed`, else a fresh draw from
        // entropy. Each rollout thread mixes in its index so a fixed `--seed` still gives
        // every thread an INDEPENDENT exploration stream (identical streams would collapse
        // the K-way diversity the parallel rollout exists for); index 0 (the learner host
        // and the single-world / test path) uses the seed unmixed. LOGGED so a run can be
        // reproduced after the fact by passing the base seed back via `--seed`.
        let base_seed = config.seed.unwrap_or_else(rand::random);
        let seed = base_seed ^ (worker_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        info!(
            "Training RNG seed {seed} (base {base_seed}, worker {worker_index}) — pass \
             --seed {base_seed} to reproduce this run"
        );
        let rng = StdRng::seed_from_u64(seed);

        // Seed the backend's weight-init RNG too, so a fresh brain's initial weights are
        // reproducible from the same seed (init draws from a process-global RNG). Harmless
        // under K concurrent worker builds: each worker's init weights are immediately
        // overwritten by the learner's snapshot before its first rollout, so the racing
        // reseeds never reach training.
        <TrainBackend as burn::tensor::backend::Backend>::seed(&device, seed);

        let mut brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);

        let mut obs_normalizer = ObsNormalizer::new(NORMALIZER_CLIP);
        let mut return_normalizer = ReturnNormalizer::new();

        let paths = CheckpointDir::new(&config.checkpoint_dir);

        // Warm-start from the checkpoint's envelope tag — the tag is AUTHORITATIVE for
        // which architecture resumes (bddap/rl#200 §2/§3). The refusal policy here is
        // ABORT, not cold-start: a present-but-unusable brain (legacy, corrupt, unknown
        // arch/version, mis-copied) must never silently discard the trained policy of a
        // live run by cold-starting over it. The ONE deliberate cold start is a DOF
        // change (bddap/rl#31): same arch, but the checkpoint's obs/action widths — read
        // from the loaded record via `io_dims`, the post-load check that replaced the
        // `shape.txt` sidecar — no longer fit this build's rig, so the stale weights are
        // discarded on purpose, loudly.
        let mut warm_started = false;
        match load_brain_file::<TrainBackend>(&paths.brain_file(), &device) {
            Ok(loaded) => {
                let (obs, action) = loaded.io_dims();
                if crate::policy::dims_fit_rig(obs, action) {
                    info!(
                        "Loaded {} brain weights from {}",
                        loaded.arch(),
                        paths.brain_file().display()
                    );
                    brain = loaded;
                    warm_started = true;
                } else {
                    warn!(
                        "Checkpoint at {} has an incompatible obs/action shape ({obs}/{action}; \
                         the actuated DOF set changed) — starting the policy fresh",
                        paths.brain_file().display()
                    );
                }
            }
            // No brain file: a fresh checkpoint dir — the legitimate cold start.
            Err(BrainLoadError::Envelope(EnvelopeError::Absent)) => {}
            Err(e) => panic!(
                "REFUSING to train over checkpoint dir {}: brain checkpoint is unusable \
                 ({e}). Cold-starting here would silently discard the trained policy; fix \
                 or move the checkpoint before resuming.",
                config.checkpoint_dir.display()
            ),
        }

        // Normalizers are brain-PAIRED: they load iff the brain warm-started, tagged with
        // its arch (the dir-coherence check), and a warm brain with unusable normalizer
        // stats is an ABORT — training trained weights against cold or mis-paired scales
        // mis-normalizes every observation/value, the exact silent skew the envelope
        // exists to refuse. On a cold start they stay fresh with the fresh brain.
        if warm_started {
            let arch = brain.arch();
            let norm_path = paths.normalizer_path();
            match ObsNormalizer::load(&norm_path, arch) {
                Ok(loaded) => {
                    info!("Loaded normalizer state from {}", norm_path.display());
                    obs_normalizer = loaded;
                }
                Err(e) => panic!(
                    "REFUSING to resume {}: obs normalizer at {} is unusable ({e}) — a \
                     warm brain never trains against cold or mis-paired normalizer stats.",
                    config.checkpoint_dir.display(),
                    norm_path.display()
                ),
            }
            let ret_norm_path = paths.return_normalizer_path();
            match load_return_normalizer(&ret_norm_path, arch) {
                Ok(loaded) => {
                    info!(
                        "Loaded return normalizer state from {}",
                        ret_norm_path.display()
                    );
                    return_normalizer = loaded;
                }
                Err(e) => panic!(
                    "REFUSING to resume {}: return normalizer at {} is unusable ({e}) — a \
                     warm brain never trains against cold or mis-paired normalizer stats.",
                    config.checkpoint_dir.display(),
                    ret_norm_path.display()
                ),
            }
        }

        let n = config.envs.max(1) as usize;
        // A rollout thread accumulates a per-horizon increment; the learner's host
        // steps no world and ships no normalizer, so it stays None.
        let normalizer_increment = worker_mode.then(IncrementAccumulator::new);
        Self {
            brain: InferenceCachedBrain::new(brain),
            config: PpoConfig::default(),
            rollouts: (0..n).map(|_| RolloutBuffer::new()).collect(),
            device,
            envs: vec![EnvEpisode::default(); n],
            explore_noise: OuNoise::new(n),
            log_std_floor: crate::bot::arch::LOG_STD_MIN,
            episode_count: 0,
            recent_rewards: Vec::new(),
            total_steps: 0,
            obs_normalizer,
            return_normalizer,
            rng,
            checkpoint_dir: config.checkpoint_dir.clone(),
            saved_on_exit: false,
            warm_started,
            tick_budget: config.ticks,
            reported_episodes: 0,
            normalizer_increment,
            drift_sum: 0.0,
            drift_count: 0,
            band: TargetBand::start(),
            reach_reached: 0,
            reach_finished: 0,
            progress_glitch_drops: 0,
            nonfinite_obs_elements: 0,
        }
    }

    /// Whether the brain warm-started from the checkpoint dir at build time — the
    /// optimizer's warm/cold gate.
    pub(crate) fn warm_started(&self) -> bool {
        self.warm_started
    }

    pub(crate) fn save_checkpoint(&self) {
        if let Err(e) = std::fs::create_dir_all(&self.checkpoint_dir) {
            warn!(
                "Failed to create checkpoint dir {}: {e}",
                self.checkpoint_dir.display()
            );
            return;
        }

        // Every artifact is written atomically (temp + fsync-rename inside the envelope
        // writer), tagged with THIS brain's arch, so a crash can't tear a file and the
        // set can't cross-pair.
        let paths = CheckpointDir::new(&self.checkpoint_dir);
        let arch = self.brain.train().arch();
        match save_brain(self.brain.train(), &paths.brain_file()) {
            Ok(()) => info!("Saved brain to {}", paths.brain_file().display()),
            Err(e) => warn!("Failed to save brain: {e}"),
        }
        self.obs_normalizer.save(arch, &paths.normalizer_path());
        save_return_normalizer(
            &self.return_normalizer,
            arch,
            &paths.return_normalizer_path(),
        );
    }

    // ---- In-process rollout-thread / learner hooks ------------------------
    //
    // These let `training::inproc` drive a worker-mode TrainingState by hand on a
    // rollout thread: load the learner's snapshot weights + master normalizer, roll a
    // horizon (via the normal systems), then hand the buffers + per-horizon normalizer
    // increment + finished rewards back. One struct for both the learner host and the
    // rollout threads (not a parallel one) keeps their collection + update on the same code.
    //
    // The per-horizon protocol is hidden behind exactly two methods — `begin_horizon` /
    // `end_horizon` (bddap/rl#165) — so `inproc` never spells out the load→set→reset setup or
    // the fixed drain order; the private hooks below are their building blocks, not the interface.

    /// OPEN one horizon on a rollout thread: load the learner's snapshot (weights + normalizer
    /// baseline), set this horizon's target band + σ-floor, and reset the per-horizon increment.
    /// Returns `false` — having LOGGED which component failed — if the brain OR the normalizer
    /// snapshot did not load, in which case the caller MUST refuse the horizon (bddap/rl#177):
    /// rolling a stale/off-policy brain or mis-normalized observations ships data the learner
    /// can't tell from honest. Both loads are attempted (not short-circuited) so the log names
    /// every failing component. This is the single entry point for horizon setup — the fixed
    /// order lives here, not at the `inproc` call site.
    #[must_use = "a failed horizon open must be refused, not rolled on a stale/mis-normalized policy"]
    pub(crate) fn begin_horizon(&mut self, req: HorizonRequest) -> bool {
        let brain_ok = self.load_brain_bytes(req.brain_bytes);
        let normalizer_ok = self.set_normalizer(req.normalizer);
        if !brain_ok || !normalizer_ok {
            error!(
                "rollout thread: snapshot load failed (brain_ok={brain_ok}, normalizer_ok={normalizer_ok}); \
                 refusing this horizon rather than rolling a stale/off-policy or mis-normalized policy"
            );
            return false;
        }
        self.set_band(req.band);
        self.set_log_std_floor(req.log_std_floor);
        self.reset_horizon_counter();
        true
    }

    /// CLOSE the horizon: move out every per-horizon artifact the learner merges, in one shot
    /// (bddap/rl#165). Folds the dozen `drain_*`/`take_*` accessors so the collection sequence
    /// is owned here and `inproc` sees only begin → roll → end. Panics via `normalizer_increment`
    /// if called on a non-worker state — only a worker rolls a horizon.
    pub(crate) fn end_horizon(&mut self) -> HorizonOutput {
        HorizonOutput {
            envs: self.take_rollouts(),
            increment: self.normalizer_increment(),
            rewards: self.drain_finished_episode_rewards(),
            drift: self.drain_drift(),
            reach: self.drain_reach(),
            glitch_drops: self.drain_progress_glitches(),
            nonfinite_obs: self.drain_nonfinite_obs(),
        }
    }

    /// Load brain weights from the learner's in-memory snapshot bytes (the same
    /// `FullPrecisionSettings` bincode the on-disk checkpoint uses, produced by the
    /// in-process learner once per iteration). Replaces a file load: weights move
    /// thread-to-thread as `Send` bytes, never as the `!Send` live tensors.
    ///
    /// Returns whether the weights actually loaded. On a decode error the brain is left
    /// UNCHANGED (stale) and `false` is returned — the caller MUST then abort the horizon
    /// rather than roll the stale policy, since rolling on stale weights ships off-policy
    /// samples the learner can't tell apart from on-policy ones (bddap/rl#177). `#[must_use]`
    /// so that failure can never again be silently dropped.
    #[must_use = "a failed brain load must abort the horizon, not roll a stale/off-policy policy"]
    fn load_brain_bytes(&mut self, bytes: &[u8]) -> bool {
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        match self
            .brain
            .train()
            .clone()
            .load_leaf_record(&recorder, bytes.to_vec(), &self.device)
        {
            Ok(updated) => {
                self.brain.set(updated);
                true
            }
            Err(e) => {
                error!("rollout thread: failed to load snapshot brain: {e}");
                false
            }
        }
    }

    /// The live train brain — read-only, for snapshotting its weights to bytes
    /// (rollout-thread → learner) and the determinism test's seed clone.
    pub(crate) fn brain(&self) -> &AnyBrain<TrainBackend> {
        self.brain.train()
    }

    /// Overwrite this state's normalizer from the learner's master snapshot. The
    /// per-horizon increment is reset separately in `reset_horizon_counter`, so the
    /// increment always starts fresh each horizon regardless of this call.
    ///
    /// Returns whether the snapshot actually loaded (`load_snapshot`'s verdict). On a
    /// size/validity mismatch the normalizer is left UNCHANGED and `false` is returned — the
    /// caller MUST abort the horizon rather than roll mis-normalized observations through the
    /// policy (bddap/rl#177). `#[must_use]` so the failure can't be silently dropped again.
    #[must_use = "a failed normalizer load must abort the horizon, not roll mis-normalized obs"]
    fn set_normalizer(&mut self, snapshot: NormalizerSnapshot) -> bool {
        self.obs_normalizer.load_snapshot(snapshot)
    }

    /// Snapshot the master normalizer's full stats (learner → rollout threads), so
    /// each thread's policy normalizes observations against the same baseline the
    /// learner holds. A cumulative snapshot — by type, not mergeable.
    pub(crate) fn normalizer_snapshot(&self) -> NormalizerSnapshot {
        self.obs_normalizer.snapshot()
    }

    /// The per-horizon normalizer INCREMENT to ship back (rollout thread → learner; see
    /// [`ObsNormalizer::merge`]). Panics if called off a worker: only a worker rolls and
    /// accumulates an increment, and there is deliberately no full-snapshot fallback —
    /// shipping the cumulative snapshot here would double-count the baseline on merge.
    fn normalizer_increment(&self) -> NormalizerIncrement {
        self.normalizer_increment
            .as_ref()
            .expect("normalizer increment requested on a non-worker TrainingState")
            .increment()
    }

    /// Merge a rollout thread's per-horizon increment into this (learner's) normalizer.
    /// Only a [`NormalizerIncrement`] is accepted — a cumulative snapshot can't reach
    /// here (see [`ObsNormalizer::merge`]).
    pub(crate) fn merge_normalizer(&mut self, increment: &NormalizerIncrement) {
        self.obs_normalizer.merge(increment);
    }

    /// Move the collected transitions out, leaving the buffers empty for the next
    /// horizon. The per-env episode accumulators (`envs`) are deliberately left
    /// untouched: an episode that spans a horizon boundary must continue cleanly
    /// across the cut rather than be force-terminated at the window edge.
    fn take_rollouts(&mut self) -> Vec<RolloutBuffer> {
        // Each buffer carries the GAE bootstrap for its `Continues` tail: a dangling
        // `Pending` is the un-finalized successor action of the buffer's last `Continues`
        // transition (an episode-ending push re-seeds the env and stashes no pending), so
        // `Pending::value` IS V(s_{last+1}) — the correct bootstrap, on the CPU rollout
        // backend like every body value. `compute_gae` only reads it for a `Continues`
        // tail; a Terminal/Truncated tail self-bootstraps and ignores it (it may still be
        // `Some` from a fresh episode, harmlessly). This is the rl#174 / rl#173-tail fix:
        // the bootstrap is the successor's value, never a recompute of the tail's own obs
        // on the update backend. Index loop because the bootstrap reads `self.envs[e]`
        // while transitions are taken from `self.rollouts[e]`.
        (0..self.rollouts.len())
            .map(|e| RolloutBuffer {
                transitions: std::mem::take(&mut self.rollouts[e].transitions),
                bootstrap: self.envs[e].pending.as_ref().map(|p| p.value),
            })
            .collect()
    }

    /// Reset the per-horizon normalizer increment (rollout thread, at the start of each
    /// horizon), so it always holds exactly this horizon's samples. `total_steps` stays
    /// monotonic — it is the thread's tick odometer the learner diffs for horizon length.
    fn reset_horizon_counter(&mut self) {
        if let Some(inc) = self.normalizer_increment.as_mut() {
            *inc = IncrementAccumulator::new();
        }
    }

    /// Drain the rewards of episodes that finished since the last drain, so the
    /// worker ships each finished episode's reward to the learner exactly once
    /// (the learner's reward-vs-samples curve aggregates all workers').
    fn drain_finished_episode_rewards(&mut self) -> Vec<f32> {
        let out = self.recent_rewards[self.reported_episodes..].to_vec();
        self.reported_episodes = self.recent_rewards.len();
        out
    }

    /// Drain this horizon's accumulated carapace drift-from-spawn as `(sum, count)`,
    /// resetting both. The learner sums these across rollout threads and divides for the
    /// mean planar drift it logs — the walking diagnostic. `(0.0, 0)` when nothing was
    /// recorded (a fully-settling horizon), which the learner treats as no sample.
    fn drain_drift(&mut self) -> (f64, u64) {
        let out = (self.drift_sum, self.drift_count);
        self.drift_sum = 0.0;
        self.drift_count = 0;
        out
    }

    /// Set the target-distance band a rollout thread samples from this horizon (learner →
    /// thread, once per horizon before the roll, like `set_normalizer`). The band is the one
    /// fixed full-arena range; the thread only consumes it.
    pub(super) fn set_band(&mut self, band: TargetBand) {
        self.band = band;
    }

    /// Set this horizon's exploration-σ floor (learner → thread, once per horizon before the
    /// roll, like [`Self::set_band`]). The learner evaluates the anneal schedule from the
    /// durable tick odometer and ships the scalar so the rollout samples — and the update later
    /// recomputes log-probs — under the same lower `log_std` clamp (bddap/rl#161).
    fn set_log_std_floor(&mut self, log_std_floor: f32) {
        self.log_std_floor = log_std_floor;
    }

    /// Drain this horizon's per-episode reach tally as `(reached, finished)`, resetting
    /// both. The learner pools these across rollout threads for the log's reach fraction and
    /// the best-keeper's solid-reach floor ([`crate::training::curriculum::SOLID_REACH_FRACTION`]).
    /// `(0, 0)` when no episode finished this horizon.
    fn drain_reach(&mut self) -> (u64, u64) {
        let out = (self.reach_reached, self.reach_finished);
        self.reach_reached = 0;
        self.reach_finished = 0;
        out
    }

    /// Drain this horizon's count of progress-term drops (non-physical deltas the reward zeroed;
    /// see [`progress_glitch_drops`](Self::progress_glitch_drops)), resetting it. The learner sums
    /// these across rollout threads and surfaces the total so a silent reward dropout is visible
    /// (bddap/rl#175). `0` on a healthy horizon — the common case.
    fn drain_progress_glitches(&mut self) -> u64 {
        std::mem::take(&mut self.progress_glitch_drops)
    }

    /// Drain this horizon's count of non-finite observation elements skipped from the normalizer
    /// (see [`nonfinite_obs_elements`](Self::nonfinite_obs_elements)), resetting it. The learner
    /// sums these across rollout threads and surfaces the total so a NaN sensor/physics reading is
    /// visible (bddap/rl#181). `0` on a healthy horizon — the common case.
    fn drain_nonfinite_obs(&mut self) -> u64 {
        std::mem::take(&mut self.nonfinite_obs_elements)
    }

    /// Hand a CPU-backend PPO update its non-optimizer pieces (brain/config/device/
    /// return-normalizer/rng; see `crate::training::update::ppo_update_core`). The live learner
    /// updates on the GPU (see `learner_parts_for_gpu` + the `GpuLearner`); this CPU
    /// accessor backs only the `#[cfg(test)]` CPU update test, which exercises the shared
    /// update math without a GPU. The optimizer is not learner state: the CPU update test
    /// builds its own via [`crate::training::checkpoint::crab_optimizer`], so the production learner
    /// carries no optimizer the GPU path never steps. The return normalizer is handed out `&mut`
    /// to fold in the iteration's returns; `rng` drives the update's minibatch shuffle.
    #[cfg(test)]
    pub fn learner_parts(
        &mut self,
    ) -> (
        &mut AnyBrain<TrainBackend>,
        &PpoConfig,
        &NdArrayDevice,
        &mut ReturnNormalizer,
        &mut StdRng,
    ) {
        (
            self.brain.train_mut(),
            &self.config,
            &self.device,
            &mut self.return_normalizer,
            &mut self.rng,
        )
    }

    /// Learner-side accessor for the live GPU update: the CPU brain (mirrored to/from the
    /// GPU each update by [`crate::training::gpu::GpuLearner`]), the PPO config, the host return
    /// normalizer, and the run's seeded `rng` (drives the update's minibatch shuffle).
    /// Unlike [`Self::learner_parts`] it omits the CPU device — the GPU update runs Adam on
    /// its own device-resident optimizer. The brain stays the single source of truth
    /// (rollout snapshots + checkpoints read it); the GPU learner only borrows it to load
    /// weights in and write back.
    #[cfg(feature = "wgpu")]
    pub fn learner_parts_for_gpu(
        &mut self,
    ) -> (
        &mut AnyBrain<TrainBackend>,
        &PpoConfig,
        &mut ReturnNormalizer,
        &mut StdRng,
    ) {
        (
            self.brain.train_mut(),
            &self.config,
            &mut self.return_normalizer,
            &mut self.rng,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::sensor::OBS_SIZE;
    use burn::tensor::Tensor;

    /// The inference-cache contract `forward_pass` relies on: the cached inference brain
    /// is BIT-IDENTICAL to rebuilding `valid()` every call (so caching can't perturb the
    /// determinism hash when the NN crab is armed), and it REFRESHES on any weight change —
    /// through both `set` (snapshot load) and `train_mut` (a learner update) — so a stale
    /// clone is never served.
    // `with_inference` takes its closure by value (`impl FnOnce`); `policy_bits` is a
    // non-Copy closure (captures `obs`) reused across four calls, so each `&policy_bits`
    // borrow is load-bearing — dropping it would move the closure on the first call and fail
    // to compile on the next. The lint's "needless" verdict is a false positive here.
    #[allow(clippy::needless_borrows_for_generic_args)]
    #[test]
    fn inference_cache_is_bit_identical_and_refreshes_on_weight_change() {
        let device = NdArrayDevice::Cpu;
        let obs = Tensor::<InferBackend, 2>::zeros([3, OBS_SIZE], &device);
        // Bits of EVERY inference output forward_pass reads — policy means, the shared
        // log_std, and the value — so the cache is held to bit-identity on all three.
        let bits = |t: &[f32]| -> Vec<u32> { t.iter().map(|v| v.to_bits()).collect() };
        let policy_bits = |brain: &AnyBrain<InferBackend>| -> Vec<u32> {
            let (means, log_std) = brain.policy(obs.clone());
            let value = brain.value(obs.clone());
            let mut out = bits(&means.to_data().to_vec::<f32>().unwrap());
            out.extend(bits(&log_std.to_data().to_vec::<f32>().unwrap()));
            out.extend(bits(&value.to_data().to_vec::<f32>().unwrap()));
            out
        };

        let brain_a: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);
        let want_a = policy_bits(&brain_a.valid());
        let mut cached = InferenceCachedBrain::new(brain_a);

        // Building the cache, then reusing it, is bit-identical to a fresh `valid()`.
        assert_eq!(cached.with_inference(&policy_bits), want_a);
        assert_eq!(cached.with_inference(&policy_bits), want_a);

        // `set` (snapshot load) refreshes: the new weights are served, never the stale clone.
        let brain_b: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);
        let want_b = policy_bits(&brain_b.valid());
        assert_ne!(
            want_a, want_b,
            "fresh brains must differ for this assertion to bite"
        );
        cached.set(brain_b);
        assert_eq!(cached.with_inference(&policy_bits), want_b);

        // `train_mut` (a learner update) refreshes too.
        let brain_c: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);
        let want_c = policy_bits(&brain_c.valid());
        assert_ne!(
            want_b, want_c,
            "fresh brains must differ for this assertion to bite"
        );
        *cached.train_mut() = brain_c;
        assert_eq!(cached.with_inference(&policy_bits), want_c);
    }

    /// DETERMINISM (rl#139): a fresh brain's weight init is REPRODUCIBLE from the backend seed.
    /// `AnyBrain::init` draws its initial weights from the backend's RNG, which `TrainState::build`
    /// seeds per run (`Backend::seed`) precisely so a run can be replayed from its logged `--seed`.
    /// This pins the contract the opposite way from `inference_cache_…` (which proves two UNSEEDED
    /// brains differ): seed identically ⇒ bit-identical weights; seed differently ⇒ different. A
    /// reseed regression that quietly made init unseedable would otherwise make every run
    /// unrepeatable with no test to catch it.
    #[test]
    fn brain_init_is_reproducible_from_the_backend_seed() {
        let device = NdArrayDevice::Cpu;
        let obs = Tensor::<InferBackend, 2>::zeros([3, OBS_SIZE], &device);
        // Bits of the full forward output (policy means + log_std + value) — a faithful
        // fingerprint of the initial weights, the same projection `inference_cache_…` hashes.
        let weight_bits = |brain: &AnyBrain<InferBackend>| -> Vec<u32> {
            let (means, log_std) = brain.policy(obs.clone());
            let value = brain.value(obs.clone());
            let bits = |t: &[f32]| -> Vec<u32> { t.iter().map(|v| v.to_bits()).collect() };
            let mut out = bits(&means.to_data().to_vec::<f32>().unwrap());
            out.extend(bits(&log_std.to_data().to_vec::<f32>().unwrap()));
            out.extend(bits(&value.to_data().to_vec::<f32>().unwrap()));
            out
        };
        let init_with_seed = |seed: u64| -> Vec<u32> {
            <TrainBackend as burn::tensor::backend::Backend>::seed(&device, seed);
            weight_bits(&AnyBrain::<TrainBackend>::init(ArchId::Mlp256, &device).valid())
        };

        // The backend's init RNG is process-GLOBAL, so under a parallel test run a sibling
        // brain-building test can consume it between our seed and our build and perturb one
        // sample. That race only ever makes same-seed weights spuriously DIFFER (never makes two
        // different seeds collide), so the reproducibility check just needs ONE uncontended
        // same-seed window — retry to find it; a genuine non-determinism never produces one.
        let reproducible = (0..16).any(|_| init_with_seed(0x5EED) == init_with_seed(0x5EED));
        assert!(
            reproducible,
            "same backend seed never reproduced identical initial weights across 16 tries — init \
             is not seed-deterministic, so training can't replay a run from its logged --seed"
        );
        // Different seeds must give different weights (else seeding is a no-op). Race-immune: an
        // interleave can't make two genuinely-different seeds produce equal weights.
        assert_ne!(
            init_with_seed(0x5EED),
            init_with_seed(0xC0FFEE),
            "different seeds must give different initial weights, else seeding does nothing"
        );
    }
}
