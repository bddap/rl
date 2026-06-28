//! The RL training loop, integrated with the Bevy game loop as ECS systems: the per-env
//! episode lifecycle ([`TrainingState`], [`EnvEpisode`], [`EnvPhase`]) and the
//! Sense→Think→Act systems ([`brain_step`], [`reset_crab`], [`save_on_exit`]) the sole
//! trainer and each rollout thread run.

use std::path::PathBuf;

use bevy::app::AppExit;
use bevy::prelude::*;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::{AutodiffModule, Module};
use burn::record::{BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
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
    NormalizedValue, PpoConfig, ReturnNormalizer, RolloutBuffer, StepEnd, Transition,
    compute_log_prob, sample_action,
};
use super::TrainBackend;
use super::checkpoint::{CheckpointDir, load_return_normalizer, save_return_normalizer};
use super::curriculum::{CURRICULUM_REACH_RADIUS, Curriculum, seed_target};
use super::normalizer::{
    IncrementAccumulator, NORMALIZER_CLIP, NormalizerIncrement, NormalizerSnapshot, ObsNormalizer,
};
use super::reward::{
    EFFORT_WEIGHT, GRAB_REWARD, action_effort, compute_reward, dist_3d, planar_dist,
};

/// Default rollout horizon: the number of physics ticks each rollout thread rolls
/// per iteration before handing its buffers back, when `--horizon` is not given.
pub const STEPS_PER_ROLLOUT: u32 = 1024;

/// Episode length cap: a crab still alive after this many physics ticks is TRUNCATED (not
/// failed — GAE bootstraps its value; see `StepEnd::Truncated`). At 64 Hz this is ~23 s of
/// crab-time. The reward calibration is balanced over exactly this horizon (a full traverse's
/// progress must out-earn the integrated effort tax across these ticks — see
/// `reward::EFFORT_WEIGHT`), so it is named once here and shared, never duplicated as a magic
/// `1500`.
pub(crate) const MAX_EPISODE_TICKS: u32 = 1500;

/// Classify a recorded step's episode end from its three independent end conditions, in
/// PRIORITY order — the single place the terminal-vs-truncation contract is decided, kept pure
/// so it is unit-tested directly (not only through a full rollout):
///   * `grabbed` (a claw tip reached the target) or `fell` (a survival guard tripped) is a TRUE
///     [`StepEnd::Terminal`] — the episode genuinely ended, so GAE bootstraps ZERO past it. A
///     grab outranks the cap: a step that both grabs and crosses the cap is a success, not a
///     truncation.
///   * else `over_cap` (the step cap reached while still alive) is a [`StepEnd::Truncated`] — the
///     episode was cut short, so GAE must BOOTSTRAP the cut-short value (teaching the cap is a
///     dead end would be wrong).
///   * else the trajectory [`StepEnd::Continues`].
fn classify_step_end(grabbed: bool, fell: bool, over_cap: bool) -> StepEnd {
    if grabbed || fell {
        StepEnd::Terminal
    } else if over_cap {
        StepEnd::Truncated
    } else {
        StepEnd::Continues
    }
}

/// Where an env sits in the record → reset → settle lifecycle. One field, not a
/// `needs_reset: bool` + `grace: u32` pair, so an illegal combination (a respawn pending
/// *while* already settling) is unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum EnvPhase {
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
/// The reward and end must score the pose `aₜ` *produced* — the carapace's distance closed
/// (progress) and the claw-tip distance (the grab terminal) at `s_{t+1}` — but the schedule is
/// Sense → Think (`brain_step`) → Act → physics, so when `brain_step` runs at tick `t` the pose
/// it can read is still `sₜ` (physics hasn't integrated `aₜ` yet). So everything known at tick
/// `t` is stashed here and the transition completed at tick `t+1`, in phase with the pose that
/// action caused.
#[derive(Clone)]
struct Pending {
    obs: [f32; OBS_SIZE],
    /// The policy's unbounded DRIVE `μ + σ·ε` this tick — the RL action proper the PPO update
    /// recomputes its log-prob over (the sim ran `drive.clamp(±1)`). See [`SampledAction`].
    action: [f32; ACTION_SIZE],
    value: NormalizedValue,
    log_prob: f32,
    /// `Σ|dᵢ|^L` for this drive — the effort summand over the unbounded DRIVES (see
    /// [`action_effort`]), final at tick `t` and traveling with the drive it priced.
    /// [`compute_reward`] scales it by [`EFFORT_WEIGHT`] at finalization.
    effort: f32,
    /// Carapace planar distance to the target at `s_t` (the pose this action was chosen from).
    /// The progress reward is the REDUCTION in this distance to `s_{t+1}`, computed at
    /// finalization (`P·(target_dist − d_now)`). `None` if the carapace pose or target was
    /// absent at stash time, in which case the transition earns no progress credit.
    target_dist: Option<f32>,
}

/// Per-env episode accumulators. Each env's episode runs and resets
/// independently.
#[derive(Clone, Default)]
pub(crate) struct EnvEpisode {
    pub(crate) reward: f32,
    pub(crate) steps: u32,
    pub(crate) phase: EnvPhase,
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
pub(crate) struct TrainingState {
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

    pub total_steps: u64,

    obs_normalizer: ObsNormalizer,

    /// Running mean/std of the value targets (GAE returns), normalizing what the value
    /// head regresses to unit scale so it can track large-magnitude returns (see
    /// [`ReturnNormalizer`]). The LEARNER owns the single copy — rollout threads emit raw
    /// value predictions and never touch it, so there is no second instance to drift.
    /// Persisted in the checkpoint beside `obs_normalizer`.
    return_normalizer: ReturnNormalizer,

    /// The run's seeded RNG, threaded to every stochastic choice this state makes:
    /// action-noise sampling, target placement, and spawn rotation. Owned per state so
    /// each rollout thread draws an INDEPENDENT stream off any global lock (see the seed
    /// derivation in [`Self::build`]); the same seed reproduces the trajectory exactly,
    /// which the determinism regression test pins.
    rng: StdRng,

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
    normalizer_increment: Option<IncrementAccumulator>,

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

/// The per-env arrays captured this tick, bundled into one named borrow so
/// [`TrainingState::finalize_transitions`]'s call site can't transpose the several
/// same-typed slices (three `&[f32]`, the obs/action arrays). Every field is indexed by
/// env and has length `envs.len()`.
struct StepInputs<'a> {
    /// Post-physics body readings (poses/speeds/drift) — the survival-guard +
    /// walking-diagnostic inputs.
    body: &'a BodyState,
    /// Closest claw-tip→target 3D distance per env this tick — the grab terminal's `d` (and
    /// the curriculum "reached" signal's, via the per-episode minimum).
    min_tip_dists: &'a [Option<f32>],
    /// Normalized observation fed to the policy this tick (stashed in the pending).
    obs: &'a [[f32; OBS_SIZE]],
    /// The policy's unbounded DRIVE this tick — the RL action proper, carried into the
    /// transition so the PPO log-prob recomputes over the same quantity it was sampled on
    /// (the sim ran `drive.clamp(±1)`; that command is not stored — nothing reads it back).
    drives: &'a [[f32; ACTION_SIZE]],
    /// Value-head output for this tick's observation.
    values: &'a [NormalizedValue],
    /// Sampling log-prob of this tick's drive.
    log_probs: &'a [f32],
    /// `Σ|dᵢ|^L` effort summand over the unbounded DRIVES (the metabolic tax input).
    efforts: &'a [f32],
    /// Envs force-respawned this tick by the non-finite rescue (their pose is the fresh
    /// spawn, not the action's result — so the action ends the episode with no reach credit).
    rescued_envs: &'a [usize],
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

        let mut brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);

        let mut obs_normalizer = ObsNormalizer::new(NORMALIZER_CLIP);
        let mut return_normalizer = ReturnNormalizer::new();

        let paths = CheckpointDir::new(&config.checkpoint_dir);
        let norm_path = paths.normalizer_path();
        let ret_norm_path = paths.return_normalizer_path();

        // Only warm-start the brain when the checkpoint's persisted obs/action widths
        // match this build's. A DOF change (bddap/rl#31) re-shapes the policy net, and
        // loading an old-width record into it would silently misalign every weight; the
        // shape sidecar makes that a clean cold start instead (see `warm_start_compatible`).
        if paths.brain_file().exists() {
            if !paths.warm_start_compatible() {
                warn!(
                    "Checkpoint at {} has an incompatible obs/action shape (the actuated \
                     DOF set changed) — starting the policy fresh",
                    paths.brain_file().display()
                );
            } else {
                let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
                match recorder.load(paths.brain_stem(), &device) {
                    Ok(record) => {
                        brain = brain.load_record(record);
                        info!("Loaded brain weights from {}", paths.brain_file().display());
                    }
                    Err(e) => {
                        warn!(
                            "Failed to load brain from {}: {e} — starting fresh",
                            paths.brain_file().display()
                        );
                    }
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
        // A rollout thread accumulates a per-horizon increment; the learner's host
        // steps no world and ships no normalizer, so it stays None.
        let normalizer_increment = worker_mode.then(IncrementAccumulator::new);
        Self {
            brain,
            config: PpoConfig::default(),
            rollouts: (0..n).map(|_| RolloutBuffer::new()).collect(),
            device,
            envs: vec![EnvEpisode::default(); n],
            episode_count: 0,
            recent_rewards: Vec::new(),
            total_steps: 0,
            obs_normalizer,
            return_normalizer,
            rng,
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

        let paths = CheckpointDir::new(&self.checkpoint_dir);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        // Record to a temp stem then rename into place, so a crash mid-write can't
        // leave a torn brain.bin (silently discarded on load → resume from random
        // weights).
        let brain_tmp_stem = paths.brain_tmp_stem();
        match recorder.record(self.brain.clone().into_record(), brain_tmp_stem.clone()) {
            Ok(()) => {
                let tmp_file = brain_tmp_stem.with_extension("bin");
                let final_file = paths.brain_file();
                match std::fs::rename(&tmp_file, &final_file) {
                    Ok(()) => info!("Saved brain to {}", final_file.display()),
                    Err(e) => warn!("Failed to finalize brain checkpoint: {e}"),
                }
            }
            Err(e) => warn!("Failed to save brain: {e}"),
        }

        self.obs_normalizer.save(&paths.normalizer_path());
        save_return_normalizer(&self.return_normalizer, &paths.return_normalizer_path());
        // Stamp the obs/action widths beside the brain so a later resume can reject a
        // shape-incompatible warm start (bddap/rl#31) rather than load weights askew.
        paths.save_shape();
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
    pub(crate) fn set_normalizer(&mut self, snapshot: NormalizerSnapshot) {
        self.obs_normalizer.load_snapshot(snapshot);
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
    pub(crate) fn normalizer_increment(&self) -> NormalizerIncrement {
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
            *inc = IncrementAccumulator::new();
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
    /// (see [`super::curriculum::CurriculumProgress`]). `(0, 0)` when no episode finished
    /// this horizon, which records nothing.
    pub fn drain_reach(&mut self) -> (u64, u64) {
        let out = (self.reach_reached, self.reach_finished);
        self.reach_reached = 0;
        self.reach_finished = 0;
        out
    }

    /// Hand a CPU-backend PPO update its non-optimizer pieces (brain/config/device/
    /// return-normalizer/rng; see `super::update::ppo_update_core`). The live learner
    /// updates on the GPU (see `learner_parts_for_gpu` + the `GpuLearner`); this CPU
    /// accessor backs only the `#[cfg(test)]` CPU update test, which exercises the shared
    /// update math without a GPU. The optimizer is not learner state: each CPU update site
    /// (this test and the `bench-update --backend cpu` harness) builds its own via
    /// [`super::checkpoint::crab_optimizer`], so the production learner carries no optimizer
    /// the GPU path never steps. The return normalizer is the single copy (rollout threads
    /// never touch it), handed out `&mut` to fold in the iteration's returns; `rng` drives
    /// the update's minibatch shuffle.
    #[cfg(test)]
    pub fn learner_parts(
        &mut self,
    ) -> (
        &mut CrabBrain<TrainBackend>,
        &PpoConfig,
        &NdArrayDevice,
        &mut ReturnNormalizer,
        &mut StdRng,
    ) {
        (
            &mut self.brain,
            &self.config,
            &self.device,
            &mut self.return_normalizer,
            &mut self.rng,
        )
    }

    /// Learner-side accessor for the live GPU update: the CPU brain (mirrored to/from the
    /// GPU each update by [`super::gpu::GpuLearner`]), the PPO config, the host return
    /// normalizer, and the run's seeded `rng` (drives the update's minibatch shuffle).
    /// Unlike [`Self::learner_parts`] it omits the CPU device — the GPU update runs Adam on
    /// its own device-resident optimizer. The brain stays the single source of truth
    /// (rollout snapshots + checkpoints read it); the GPU learner only borrows it to load
    /// weights in and write back.
    #[cfg(feature = "wgpu")]
    pub fn learner_parts_for_gpu(
        &mut self,
    ) -> (
        &mut CrabBrain<TrainBackend>,
        &PpoConfig,
        &mut ReturnNormalizer,
        &mut StdRng,
    ) {
        (
            &mut self.brain,
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

    /// Per env, finalize the PREVIOUS tick's pending transition with this tick's post-physics
    /// pose (see [`Pending`] for the one-tick phasing), then stash this tick's action as the
    /// next pending. On an episode end (terminal/truncation/rescue) push the transition, tally
    /// the reach signal, record the reward, and reset the env (seeding its next target).
    ///
    /// The heart of `brain_step`: the only writer of [`Transition`]s and the per-env episode
    /// lifecycle. Termination is survival guards only — jumping, flipping, and any other
    /// strategy the policy invents are legitimate (owner call: emergent behaviour is the
    /// point); the height band is sim sanity, not a behaviour bound.
    fn finalize_transitions(
        &mut self,
        inputs: &StepInputs,
        targets: &mut CrabTargets,
        spawns: &CrabSpawns,
        curriculum: Curriculum,
    ) {
        let body = inputs.body;
        let min_tip_dists = inputs.min_tip_dists;
        let rescued_envs = inputs.rescued_envs;
        // Index loop, not a zip: each iteration reads several parallel per-env arrays
        // (body/min_tip_dists/the StepInputs slices) AND mutates `self.envs[e]` /
        // `self.rollouts[e]`, so there is no single slice to iterate over.
        #[allow(clippy::needless_range_loop)]
        for e in 0..self.envs.len() {
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
                    // credit AND no progress credit (both `None`): the body teleported to
                    // spawn, so neither this tick's claw-tip distance nor the carapace's
                    // change in distance-to-target is the action's doing (crediting the
                    // spawn jump would be a huge spurious progress delta). The effort tax
                    // still applies — it priced the DRIVE, not its result.
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
                    true
                } else {
                    let (height, _upright) =
                        body.poses[e].expect("poses[e].is_none() handled above");
                    // `height` feeds no reward (see `compute_reward`) — only the off-reward
                    // blow-up/fell-through guard below. The pose's second element (uprightness)
                    // is unused.
                    //
                    // World-frame progress toward the (fixed, per-episode) target: the metres
                    // the carapace's planar distance to the goal SHRANK from `s_t` (the pose
                    // the action was chosen at, `pending.target_dist`) to this tick's `s_{t+1}`.
                    // `None` if either distance is missing (carapace or target absent) — the
                    // reward then degrades to reach + tax, never a spurious progress credit.
                    let d_now = carapace_target_dist(body, targets, e);
                    let distance_closed = pending
                        .target_dist
                        .zip(d_now)
                        .map(|(prev, now)| prev - now);
                    let mut reward = compute_reward(distance_closed, pending.effort);
                    // The blowup check only catches a genuine numerical explosion before the
                    // solver NaNs and Rapier panics the whole app; the threshold is high
                    // because direct torque is bounded (no acceleration-motor energy pump), so
                    // ordinary vigorous, limb-flinging motion is legal — only a part moving at
                    // clearly unphysical speed ends the episode. The height band is sim sanity
                    // (clipped through the floor / left the playfield).
                    let blowing_up = body.max_speeds[e] > 100.0 || !height.is_finite();
                    let fell = !(0.02..=50.0).contains(&height) || blowing_up;
                    // SPARSE TERMINAL GRAB (rl#95 — see `reward` module header): a claw tip within
                    // the grab radius of the target THIS tick adds a one-shot `GRAB_REWARD` and
                    // ends the episode as a SUCCESS terminal (`done` ⇒ GAE bootstraps ZERO past
                    // it). Detected on this tick's post-physics tip distance (`min_tip_dists[e]` —
                    // the result of `pending`'s action), so the credit lands on the grabbing
                    // action, in phase with the pose it caused. `is_some_and` makes a missing/NaN
                    // distance a non-grab (fail-safe, no spurious terminal). The radius is the
                    // single shared `CURRICULUM_REACH_RADIUS` (also the curriculum "reached" signal
                    // and the demo ball-hop), so a grab implies a reached episode.
                    let grabbed = min_tip_dists[e].is_some_and(|d| d < CURRICULUM_REACH_RADIUS);
                    if grabbed {
                        reward += GRAB_REWARD;
                    }
                    // The step cap is a TRUNCATION, not a failure: a crab still standing at the
                    // cap was cut short, so GAE bootstraps its value rather than learning the cap
                    // is a dead end (see [`classify_step_end`] / StepEnd::Truncated).
                    let over_cap = self.envs[e].steps > MAX_EPISODE_TICKS;
                    let end = classify_step_end(grabbed, fell, over_cap);
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

                    end.ends_segment()
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
                // Carapace→target distance at THIS pose (`s_t` for the action chosen now); next
                // tick's finalize credits the reduction to `s_{t+1}` as the progress reward.
                let target_dist = carapace_target_dist(body, targets, e);
                self.envs[e].pending = Some(Pending {
                    obs: inputs.obs[e],
                    action: inputs.drives[e],
                    value: inputs.values[e],
                    log_prob: inputs.log_probs[e],
                    effort: inputs.efforts[e],
                    target_dist,
                });
            }

            if episode_ended {
                let ep = &self.envs[e];
                let ep_reward = ep.reward;
                // Did this episode reach the target — the curriculum's competence signal, read
                // off the episode's closest-ever tip distance before the reset clears it. Same
                // `CURRICULUM_REACH_RADIUS` as the grab terminal, but over the episode MINIMUM
                // (not just this tick), so a grab ⟹ reached: the grab fires the first finalized
                // tick the tip is inside the radius, which also drives the episode-min inside it.
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
                seed_target(targets, spawns, e, curriculum, &mut self.rng);

                // Tally this finished episode's reach for the curriculum (drained per horizon
                // to the learner, like the rewards just below).
                self.reach_finished += 1;
                if reached {
                    self.reach_reached += 1;
                }

                self.recent_rewards.push(ep_reward);
                self.episode_count += 1;
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

/// Effort/tax probe (RL_LOG_EFFORT only — inert otherwise): per tick, the mean drive effort
/// `Σ|d|²` and the resulting tax `EFFORT_WEIGHT·effort`, over the live RECORDING envs. Lets a
/// calibration run read how big a bite the tax takes out of the positive reward at the current
/// weight, without parsing rollouts.
fn log_effort_probe(envs: &[EnvEpisode], efforts: &[f32]) {
    if std::env::var_os("RL_LOG_EFFORT").is_none() {
        return;
    }
    let mut count = 0usize;
    let mut effort_sum = 0.0f32;
    for (e, ep) in envs.iter().enumerate() {
        if matches!(ep.phase, EnvPhase::Recording) {
            count += 1;
            effort_sum += efforts[e];
        }
    }
    if count > 0 {
        let mean_effort = effort_sum / count as f32;
        info!(
            "EFFORTLOG n={count} mean_effort={mean_effort:.3} mean_tax={:.4}",
            EFFORT_WEIGHT * mean_effort,
        );
    }
}

/// One env's sample for this tick: the policy's unbounded neural DRIVE `μ + σ·ε` (see
/// [`sample_action`]) and its sampling log-prob. The drive is the RL action proper — the PPO
/// log-prob and the metabolic tax are both over it. The ±1 torque bound the sim runs is the
/// actuator's job, not stored here (`apply_actions` clamps every command), so the unbounded
/// drive is the single quantity and a saturating `|d|≫1` overshoot stays visible to the tax.
/// One row of [`sample_actions`].
struct SampledAction {
    drive: [f32; ACTION_SIZE],
    /// Sampling log-prob of `drive` under the policy (NaN/Inf-guarded and clamped).
    log_prob: f32,
}

/// Per-env body state read off this tick's post-physics poses (the live `s_{t+1}`). Most
/// fields are off-reward (poses/speeds feed the survival guards, drift the walking diagnostic)
/// — the ONE that enters [`compute_reward`] is `carapace_pos`, from which the call site derives
/// the carapace→target distance whose per-tick REDUCTION is the progress reward. `None` for an
/// env whose crab is momentarily absent (mid-respawn).
struct BodyState {
    /// `(carapace height, up·Y uprightness)`, the survival-guard input (only height is read).
    poses: Vec<Option<(f32, f32)>>,
    /// Carapace world position — the progress-reward input: the call site computes its planar
    /// distance to the target and credits the per-tick reduction (the body's net ground
    /// covered). Measuring the ORIGIN's distance (not COM velocity) is what makes the progress
    /// term spin/limb-fling-proof (see [`super::reward`] module header).
    carapace_pos: Vec<Option<Vec3>>,
    /// Carapace planar (XZ) distance from spawn — the walking diagnostic.
    drifts: Vec<Option<f32>>,
    /// Fastest body part (limbs blow up first), linear-scaled — the blow-up guard input.
    max_speeds: Vec<f32>,
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

/// Sample one DRIVE per env from its policy mean and the shared `log_std`, drawing the
/// Gaussian noise from the run's seeded `rng` (so the trajectory is reproducible), with the
/// NaN/Inf guards the live solver needs: a non-finite log-prob becomes 0 (else clamped to ±20),
/// and any non-finite drive element zeroes that element (warning once for the row). The
/// log-prob and the effort tax are both over the unbounded `drive`; the ±1 torque bound is the
/// actuator's clamp, not applied here, so the drive stays a single un-truncated quantity.
fn sample_actions(
    means_rows: &[Tensor<NdArray, 1>],
    log_std: &Tensor<NdArray, 1>,
    device: &NdArrayDevice,
    rng: &mut StdRng,
) -> Vec<SampledAction> {
    means_rows
        .iter()
        .map(|means| {
            let drive_tensor = sample_action(means, log_std, device, rng);
            // Log-prob of the ACTUAL sample (the unbounded drive): it must be the quantity the
            // PPO update later recomputes its ratio over, and the drive — not a clamp of it —
            // is what the Gaussian drew.
            let log_prob = compute_log_prob(means, log_std, &drive_tensor);
            let log_prob = if log_prob.is_nan() || log_prob.is_infinite() {
                0.0
            } else {
                log_prob.clamp(-20.0, 20.0)
            };

            let drive_data: Vec<f32> = drive_tensor.to_data().to_vec().unwrap();
            let mut drive = [0.0f32; ACTION_SIZE];
            let mut has_nan = false;
            for (i, &v) in drive_data.iter().enumerate().take(ACTION_SIZE) {
                if v.is_nan() || v.is_infinite() {
                    has_nan = true;
                    drive[i] = 0.0;
                } else {
                    drive[i] = v;
                }
            }
            if has_nan {
                warn!("NaN/Inf detected in NN drive, zeroing the offending joints");
            }
            SampledAction { drive, log_prob }
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
    let mut carapace_pos: Vec<Option<Vec3>> = vec![None; n];
    let mut drifts: Vec<Option<f32>> = vec![None; n];
    for (env, transform) in carapace_q.iter() {
        if let Some(p) = poses.get_mut(env.0) {
            let up = transform.rotation * Vec3::Y;
            *p = Some((transform.translation.y, up.dot(Vec3::Y)));
        }
        // World position for the progress reward — the call site computes its planar distance
        // to the target and credits the per-tick reduction. Rapier writes the multibody root's
        // world pose into this `Transform` each tick (the live `s_{t+1}`).
        if let Some(c) = carapace_pos.get_mut(env.0) {
            *c = Some(transform.translation);
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
        }
    }
    BodyState {
        poses,
        carapace_pos,
        drifts,
        max_speeds,
    }
}

/// Env `e`'s carapace planar distance to its target this tick — the quantity whose per-tick
/// REDUCTION is the progress reward (see [`super::reward`]). `None` if the carapace pose or the
/// target is absent (mid-respawn / unseeded), so the transition earns no spurious progress.
fn carapace_target_dist(body: &BodyState, targets: &CrabTargets, e: usize) -> Option<f32> {
    body.carapace_pos[e]
        .zip(targets.get(e))
        .map(|(pos, target)| planar_dist(pos, target))
}

/// Closest claw-tip→target 3D distance per env this tick (the grab terminal's `d`, see
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
pub(crate) fn brain_step(
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
    let sampled = sample_actions(&means_rows, &log_std, &device, &mut training.rng);

    let drive_arrays: Vec<[f32; ACTION_SIZE]> = sampled.iter().map(|s| s.drive).collect();
    let log_probs: Vec<f32> = sampled.iter().map(|s| s.log_prob).collect();
    // Metabolic effort tax is taken over the unbounded DRIVE, per env (see `action_effort`) —
    // so a saturating `|d|≫1` drive pays for the overshoot the ±1 torque bound would hide.
    let efforts: Vec<f32> = sampled.iter().map(|s| action_effort(&s.drive)).collect();

    // The sim runs the drive through the actuator's ±1 clamp (`apply_actions`, the single
    // torque-bound source), so the unbounded drive is written here as-is — no second clamp.
    actions.envs.copy_from_slice(&drive_arrays);
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
            seed_target(&mut targets, &spawns, e, curriculum, &mut training.rng);
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
    // and then held FIXED — no mid-episode resample. A grab now ENDS the episode (the sparse
    // terminal in `finalize_transitions`), so a fixed goal makes touching it strictly optimal:
    // the dense progress field pulls the body in, and the one-shot grab bonus + done caps the
    // approach — there is no positive per-tick stream to farm by hovering (progress telescopes
    // to zero once arrived, and effort is a net cost), so the crab closes the last stretch and
    // grabs rather than loitering at the radius edge.

    // Act → record: finalize last tick's pending transition against this tick's pose, stash
    // this tick's, and roll over any episode that ended. The sole writer of `Transition`s.
    let inputs = StepInputs {
        body: &body,
        min_tip_dists: &min_tip_dists,
        obs: &obs_arrays,
        drives: &drive_arrays,
        values: &values,
        log_probs: &log_probs,
        efforts: &efforts,
        rescued_envs: &rescued_envs,
    };
    training.finalize_transitions(&inputs, &mut targets, &spawns, curriculum);

    log_effort_probe(&training.envs, &efforts);
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
pub(crate) fn reset_crab(
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
    // never runs in the demo (no `TrainingState`), which respawns upright. The rotation
    // is drawn from the run's seeded RNG so a resumed/replayed run reproduces it.
    for e in 0..training.envs.len() {
        if matches!(training.envs[e].phase, EnvPhase::AwaitingRespawn) {
            training.envs[e].phase = EnvPhase::Settling {
                grace: RESET_GRACE_TICKS,
            };
            if let Some(v) = actions.envs.get_mut(e) {
                *v = [0.0; ACTION_SIZE];
            }
            let origin = spawns.0.get(e).copied().unwrap_or(Vec3::ZERO);
            let init_rotation = random_spawn_rotation(&mut training.rng);
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
pub(crate) fn save_on_exit(
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
    use bevy::ecs::system::RunSystemOnce;

    /// The terminal-vs-truncation contract the value targets depend on (rl#95): a GRAB or a
    /// fall is a TRUE terminal (GAE bootstrap 0); the step cap is a TRUNCATION (bootstrap the
    /// cut-short value); otherwise the trajectory continues. A grab OUTRANKS the cap — a step
    /// that both grabs and crosses the cap is a success, not a truncation — so the success
    /// return is never silently bootstrapped past.
    #[test]
    fn classify_step_end_terminal_vs_truncation() {
        // Grab ⇒ true terminal (does not bootstrap). This is the new sparse-grab path.
        assert_eq!(classify_step_end(true, false, false), StepEnd::Terminal);
        // A grab on the very tick the cap is hit is still a SUCCESS terminal, not a truncation.
        assert_eq!(classify_step_end(true, false, true), StepEnd::Terminal);
        // A fall (survival guard) is a true terminal too.
        assert_eq!(classify_step_end(false, true, false), StepEnd::Terminal);
        // Alive at the cap ⇒ truncation (bootstraps the value — must differ from a terminal).
        assert_eq!(classify_step_end(false, false, true), StepEnd::Truncated);
        // Otherwise the episode continues.
        assert_eq!(classify_step_end(false, false, false), StepEnd::Continues);
        // The bootstrap contract those map to: terminal/truncation end the segment, continue
        // does not — so GAE bootstraps 0 on the grab terminal, the value on truncation.
        assert!(classify_step_end(true, false, false).ends_segment());
        assert!(classify_step_end(false, false, true).ends_segment());
        assert!(!classify_step_end(false, false, false).ends_segment());
    }

    /// Drive `build_observation` over a single hand-placed carapace and return env 0's
    /// observation. No physics/rig — just the resources the system reads plus one
    /// carapace entity at the given world pose, so the body-state and target-local slots
    /// can be checked against an exact expected value (joint slots stay 0, no joints).
    fn observe_one_carapace(carapace: Transform, target: Option<Vec3>) -> [f32; OBS_SIZE] {
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
    /// `[BASE, BASE+3)`). Taken from the observation layout's one home so this test can't
    /// drift from the slots the sensor actually writes.
    const TARGET_LOCAL_BASE: usize = crate::bot::sensor::TARGET_SLOT;

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

    /// Headless training app (physics + bot + training), one fixed tick per
    /// `update()`, one env, with a fixed RNG `seed` so the run is deterministic. The
    /// windowless physics+bot stack is the shared [`crate::bot::test_util::headless_stack`]
    /// (same builder the rollout workers use); this adds the training systems on top, so
    /// these tests exercise the exact stack the sole trainer runs. Unlike the rollout
    /// worker it keeps the single-world default pool (no K-thread scaling fix needed for
    /// one app).
    fn headless_training_app(checkpoint_dir: &std::path::Path, seed: u64) -> App {
        use crate::bot::test_util::{HeadlessStack, WorldRole, headless_stack};
        use clap::Parser;

        // Point the checkpoint dir at an empty scratch path so no real checkpoint
        // loads; every other field keeps its default (tick budget 0 = unlimited,
        // so brain_step never writes AppExit during the test).
        let config = TrainConfig::try_parse_from([
            "rl",
            "--checkpoint-dir",
            checkpoint_dir.to_str().expect("utf-8 checkpoint dir"),
            "--seed",
            &seed.to_string(),
        ])
        .expect("parse default TrainConfig");

        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
        });

        // Wire the training world the same way the `rl learn` rollout worlds do
        // (see inproc::build_rollout_app): worker-mode TrainingState + the Sense→
        // Think→Act systems, so these tests exercise the brain_step / reset_crab /
        // rescue semantics the sole trainer runs. Worker index 0 → the seed is used
        // unmixed, so a fixed `seed` reproduces the trajectory exactly.
        let state = TrainingState::new_worker(&config, 0);
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

    /// Same seed ⇒ identical rollout trajectory. The two runs are handed the SAME initial
    /// brain (cloned in), so the seeded `StdRng` that drives action noise, target placement,
    /// and spawn rotation is the ONLY thing steering the trajectory — a regression that
    /// drops the seed on any of those sites would desync the runs. Copying the brain also
    /// makes the test robust to the process-global weight-init RNG a parallel test may touch
    /// (the per-state `StdRng` is not shared). Self-contained: a CPU single-world app, no
    /// learner thread and no GPU.
    #[test]
    fn same_seed_reproduces_the_rollout_trajectory() {
        const SEED: u64 = 0x00D3_7E2A;
        const TICKS: u32 = RESET_GRACE_TICKS + 80;
        // Force an episode end partway through so the reset path's spawn rotation
        // (`random_spawn_rotation`, drawn from the seeded rng) is exercised — otherwise a
        // random initial policy may never trip a terminal within the window, leaving that
        // RNG site unchecked. Applied identically in every run, so it can't desync them.
        const FORCE_RESET_AT: u32 = RESET_GRACE_TICKS + 20;

        fn run(seed: u64, initial_brain: &CrabBrain<TrainBackend>) -> Vec<Transition> {
            let dir = std::env::temp_dir()
                .join(format!("rl_test_determinism_{seed}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let mut app = headless_training_app(&dir, seed);
            // Start from the SAME weights so only the seed differs across runs.
            app.world_mut()
                .non_send_resource_mut::<TrainingState>()
                .brain = initial_brain.clone();
            for t in 0..TICKS {
                if t == FORCE_RESET_AT {
                    // Drop the carapace below the kill floor so the next tick terminates the
                    // episode → reset → seeded spawn rotation.
                    let mut q = app
                        .world_mut()
                        .query_filtered::<&mut Transform, With<CrabCarapace>>();
                    if let Ok(mut tr) = q.single_mut(app.world_mut()) {
                        tr.translation.y = -1.0;
                    }
                }
                app.update();
            }
            let traj = app
                .world()
                .non_send_resource::<TrainingState>()
                .rollouts[0]
                .transitions
                .clone();
            let _ = std::fs::remove_dir_all(&dir);
            traj
        }

        // One fixed initial brain shared by every run (the determinism we test is the RNG
        // plumbing, not weight init).
        let seed_dir =
            std::env::temp_dir().join(format!("rl_test_determinism_seed_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&seed_dir);
        let brain = headless_training_app(&seed_dir, SEED)
            .world()
            .non_send_resource::<TrainingState>()
            .brain
            .clone();
        let _ = std::fs::remove_dir_all(&seed_dir);

        let a = run(SEED, &brain);
        let b = run(SEED, &brain);
        assert!(!a.is_empty(), "the run must record transitions to compare");
        assert_eq!(
            a.len(),
            b.len(),
            "the same seed must record the same number of transitions"
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.obs, y.obs, "transition {i} obs diverged across identical seeds");
            assert_eq!(
                x.action, y.action,
                "transition {i} action diverged across identical seeds"
            );
            assert_eq!(
                x.reward.to_bits(),
                y.reward.to_bits(),
                "transition {i} reward diverged across identical seeds"
            );
        }

        // A DIFFERENT seed must (almost surely) change the trajectory — otherwise the seed
        // isn't actually steering the run.
        let c = run(SEED ^ 0xABCD, &brain);
        let differs =
            a.len() != c.len() || a.iter().zip(c.iter()).any(|(x, y)| x.action != y.action);
        assert!(differs, "a different seed must change the trajectory");
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
        let mut app = headless_training_app(&checkpoint_dir, 0x1234);

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

    /// The sparse-grab path end-to-end (rl#95): a claw tip within the reach radius of the target
    /// ENDS the episode as a TRUE terminal carrying the one-shot grab bonus, and the env resets.
    /// We force the grab by moving the target ONTO a live claw tip of env 0, so this tick's
    /// minimum tip distance is ~0 (well under `CURRICULUM_REACH_RADIUS`).
    #[test]
    fn grab_within_radius_ends_episode_with_terminal_bonus() {
        let checkpoint_dir =
            std::env::temp_dir().join(format!("rl_test_grab_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let mut app = headless_training_app(&checkpoint_dir, 0x6AB);

        // Settle past grace and record a few real steps so env 0 has a pending action to
        // finalize against the grab pose this tick (steps > 0, Recording).
        for _ in 0..(RESET_GRACE_TICKS + 8) {
            app.update();
        }
        assert!(
            matches!(
                app.world().non_send_resource::<TrainingState>().envs[0].phase,
                EnvPhase::Recording
            ),
            "env 0 must be live-recording before the grab"
        );
        let episodes_before = app
            .world()
            .non_send_resource::<TrainingState>()
            .episode_count;

        // Place the target ON one of env 0's claw tips so the next finalize sees a tip distance
        // of ~0 → grab. The target then stays fixed until the grab-triggered reset reseeds it.
        let tip_pos = {
            let mut q = app
                .world_mut()
                .query_filtered::<(&CrabEnvId, &Transform), With<CrabClawTip>>();
            q.iter(app.world())
                .find(|(env, _)| env.0 == 0)
                .map(|(_, t)| t.translation)
                .expect("env 0 must have a claw tip")
        };
        app.world_mut().resource_mut::<CrabTargets>().envs[0] = Some(tip_pos);

        // One tick: brain_step finalizes the pending action against a pose whose claw tip is on
        // the target → grab → Terminal + bonus, then reset_crab respawns the env.
        app.update();

        let st = app.world().non_send_resource::<TrainingState>();
        let last = st.rollouts[0]
            .transitions
            .last()
            .expect("env 0 recorded a transition");
        assert_eq!(
            last.end,
            StepEnd::Terminal,
            "a grab must end the episode as a TRUE terminal (GAE bootstrap 0), not a truncation"
        );
        assert!(
            last.reward >= GRAB_REWARD - 1.0,
            "the grabbing transition must carry the one-shot grab bonus (~{GRAB_REWARD}): got {}",
            last.reward
        );
        assert_eq!(
            st.episode_count,
            episodes_before + 1,
            "the grab must end the episode and count it"
        );
        assert!(
            !matches!(st.envs[0].phase, EnvPhase::Recording),
            "env 0 must have left Recording (reset for the next episode) after the grab"
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
        let mut app = headless_training_app(&checkpoint_dir, 0x5678);

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
}
