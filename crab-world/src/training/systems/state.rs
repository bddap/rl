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
    BrainFile, BrainLoadError, CheckpointDir, StampIdentity, check_body_identity,
    check_channel_layout, load_brain_file, load_return_normalizer, save_brain,
    save_return_normalizer,
};
use crate::training::envelope::{EnvelopeError, SetKey};
use crate::training::normalizer::{
    IncrementAccumulator, NORMALIZER_CLIP, NormalizerIncrement, NormalizerSnapshot, ObsNormalizer,
};
use crate::training::{
    InferBackend, TrainBackend, hardlink_missing_entries, replace_dir_atomically,
};

use super::lifecycle::EnvEpisode;

pub const STEPS_PER_ROLLOUT: u32 = 1024;

pub(super) struct InferenceCachedBrain {
    pub(super) train: AnyBrain<TrainBackend>,
    inference: RefCell<Option<AnyBrain<InferBackend>>>,
}

impl InferenceCachedBrain {
    pub(super) fn new(train: AnyBrain<TrainBackend>) -> Self {
        Self {
            train,
            inference: RefCell::new(None),
        }
    }

    fn train(&self) -> &AnyBrain<TrainBackend> {
        &self.train
    }

    pub(super) fn train_mut(&mut self) -> &mut AnyBrain<TrainBackend> {
        *self.inference.get_mut() = None;
        &mut self.train
    }

    pub(super) fn set(&mut self, train: AnyBrain<TrainBackend>) {
        self.train = train;
        *self.inference.get_mut() = None;
    }

    pub(super) fn with_inference<R>(&self, f: impl FnOnce(&AnyBrain<InferBackend>) -> R) -> R {
        let mut slot = self.inference.borrow_mut();
        let brain = slot.get_or_insert_with(|| self.train.valid());
        f(brain)
    }
}

pub(crate) struct TrainingState {
    pub(super) brain: InferenceCachedBrain,
    pub config: PpoConfig,
    pub rollouts: Vec<RolloutBuffer>,
    pub device: NdArrayDevice,

    pub envs: Vec<EnvEpisode>,
    pub episode_count: u32,

    pub recent_rewards: Vec<f32>,

    pub total_steps: u64,

    pub(super) obs_normalizer: ObsNormalizer,

    pub(super) return_normalizer: ReturnNormalizer,

    pub(super) rng: StdRng,

    pub(super) explore_noise: OuNoise,

    pub(super) log_std_floor: f32,

    /// Fraction of episodes whose target samples the close disc instead of the
    /// chase band (rl#250) — `TrainConfig::target_close_frac`. Default 0.0 keeps
    /// the pure chase band, so raising it per-run is the deliberate TRAINING
    /// change, sequenced under one-change-at-a-time.
    pub(super) close_frac: f32,

    /// Effort-tax coefficient (rl#268) — `TrainConfig::effort_weight`.
    pub(super) effort_weight: f32,

    /// DIAGNOSTIC effort probe — `TrainConfig::log_effort`.
    pub(super) log_effort: bool,

    pub(super) checkpoint_dir: PathBuf,

    /// `Some` iff the brain warm-started from the checkpoint dir at build time,
    /// carrying that set's pairing key (arch + save stamp). The optimizer's warm/cold
    /// gate ([`super::super::inproc`]): Adam moments are per-parameter, so they resume
    /// iff the brain they belong to did — and only under this key (bddap/rl#215), since
    /// the optimizer is saved beside the set but loaded later. One field, not a
    /// `warm_started` bool beside a stamp, so "cold start with a leftover key" is
    /// unrepresentable.
    resumed: Option<SetKey>,

    pub(super) tick_budget: u64,
    pub(super) reported_episodes: usize,
    pub(super) normalizer_increment: Option<IncrementAccumulator>,

    pub(super) drift_sum: f64,
    pub(super) drift_count: u64,

    pub(super) reach_reached: u64,
    pub(super) reach_finished: u64,

    /// `(reached, finished)` per compass bearing bin (rl#276) — the aggregate reach
    /// split by [`crate::eval::bearing_bin`] of each episode's target, so a
    /// one-sector competence hole shows in train.log instead of hiding behind
    /// seven healthy bearings (the s2 90° collapse cost aggregate reach only
    /// 0.91→0.83). Sums to the aggregate tally above, minus episodes with no
    /// target at end (unrepresentable in practice).
    pub(super) reach_by_bearing: [(u64, u64); crate::eval::EVAL_BEARINGS],

    pub(super) progress_glitch_drops: u64,

    pub(super) nonfinite_obs_elements: u64,
}

pub(crate) struct HorizonRequest<'a> {
    pub brain_bytes: &'a [u8],
    pub normalizer: NormalizerSnapshot,
    pub log_std_floor: f32,
}

pub(crate) struct HorizonOutput {
    pub envs: Vec<RolloutBuffer>,
    pub increment: NormalizerIncrement,
    pub rewards: Vec<f32>,
    pub drift: (f64, u64),
    pub reach: (u64, u64),
    pub reach_by_bearing: [(u64, u64); crate::eval::EVAL_BEARINGS],
    pub glitch_drops: u64,
    pub nonfinite_obs: u64,
}

impl TrainingState {
    pub fn new(config: &TrainConfig, requested: Option<ArchId>) -> Self {
        Self::build(config, false, 0, requested)
    }

    pub fn new_worker(config: &TrainConfig, worker_index: usize, arch: ArchId) -> Self {
        Self::build(config, true, worker_index, Some(arch))
    }

    fn build(
        config: &TrainConfig,
        worker_mode: bool,
        worker_index: usize,
        requested: Option<ArchId>,
    ) -> Self {
        let device = NdArrayDevice::Cpu;

        let base_seed = config.seed.unwrap_or_else(rand::random);
        let seed = base_seed ^ (worker_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        info!(
            "Training RNG seed {seed} (base {base_seed}, worker {worker_index}) — pass \
             --seed {base_seed} to reproduce this run"
        );
        let rng = StdRng::seed_from_u64(seed);

        <TrainBackend as burn::tensor::backend::Backend>::seed(&device, seed);

        // Cold-start / worker arch: the `--arch` request (a worker's is the learner's
        // resolved arch), defaulting per the registry. A warm resume below replaces this
        // brain with the checkpoint's — whose TAG is authoritative — after the flag↔tag
        // check.
        let mut brain: AnyBrain<TrainBackend> =
            AnyBrain::init(requested.unwrap_or(ArchId::DEFAULT), &device);

        let mut obs_normalizer = ObsNormalizer::new(NORMALIZER_CLIP);
        let mut return_normalizer = ReturnNormalizer::new();

        let paths = CheckpointDir::new(&config.checkpoint.checkpoint_dir);

        // Warm-start from the checkpoint's envelope tag — the tag is AUTHORITATIVE for
        // which architecture resumes (bddap/rl#200 §2/§3). The refusal policy here is
        // ABORT, not cold-start: a present-but-unusable brain (legacy, corrupt, unknown
        // arch/version, mis-copied) must never silently discard the trained policy of a
        // live run by cold-starting over it. The ONE deliberate cold start is a DOF
        // change (bddap/rl#31): same arch, but the checkpoint's obs/action widths — read
        // from the loaded record via `io_dims`, the post-load check that replaced the
        // `shape.txt` sidecar — no longer fit this build's rig, so the stale weights are
        // discarded on purpose, loudly.
        //
        // LEARNER-ONLY: a rollout worker skips the load entirely. Its brain + obs
        // normalizer are replaced by the learner's snapshot at the first `begin_horizon`
        // (a failed load there refuses the horizon), and the return normalizer /
        // `resumed` key are learner-side state — so a worker read would be K redundant
        // disk loads, where this abort policy would misfire on a fresh dir.
        // `Some(key)` once the brain warm-starts: the resumed set's pairing key
        // (bddap/rl#200 §2 + #215), which every paired artifact must match. Stays `None`
        // on every cold start.
        let mut resumed: Option<SetKey> = None;
        match (!worker_mode).then(|| load_brain_file::<TrainBackend>(&paths.brain_file(), &device))
        {
            None => {} // worker: no load at all
            Some(Ok(file)) => {
                let key = file.set_key();
                let BrainFile {
                    brain: loaded,
                    body_digest,
                    layout_digest,
                    ..
                } = file;
                // Resume is TAG-authoritative; an explicit `--arch` that disagrees with
                // the tag is an operator error and ABORTS (bddap/rl#200 §3). Cold-starting
                // into the flag's arch here would silently discard the trained policy;
                // proceeding with the tag's would silently ignore the flag.
                if let Some(req) = requested {
                    assert_eq!(
                        req,
                        loaded.arch(),
                        "REFUSING to train over checkpoint dir {}: --arch {req} disagrees \
                         with the checkpoint's arch tag {} (the tag is authoritative on a \
                         resume). Drop the flag, or point --checkpoint-dir at a fresh dir \
                         for a {req} run.",
                        config.checkpoint.checkpoint_dir.display(),
                        loaded.arch(),
                    );
                }
                // Body↔policy identity (bddap/rl#214): training this checkpoint on a
                // different body than it was trained on silently retrains a not-Sally
                // policy — abort, same class as the arch refusals above. A pre-stamp
                // checkpoint is trusted on first use; `save_checkpoint`'s next write
                // stamps it (the one-time migration, never an invalidation).
                let constructed = crate::mesh_fallback::constructed_body_digest();
                match check_body_identity(body_digest, constructed) {
                    Ok(StampIdentity::Match) => {}
                    // TOFU exists to grandfather the fleet's live pre-stamp checkpoints
                    // ONTO SALLY — a pre-stamp checkpoint is almost certainly
                    // Sally-trained, so trusting it onto the FALLBACK body
                    // (--allow-fallback-body on a mesh-less box) would retrain it wrong
                    // AND stamp it `0` on the next save, irreversibly relabelling a
                    // Sally lineage as fallback-trained. Fallback runs get a fresh dir.
                    Ok(StampIdentity::TrustOnFirstUse) if constructed == 0 => panic!(
                        "REFUSING to train over checkpoint dir {}: this run constructs \
                         the procedural fallback body, but the checkpoint predates \
                         body-identity stamps (bddap/rl#214) and is presumably \
                         Sally-trained. Point --checkpoint-dir at a fresh dir for a \
                         fallback run.",
                        config.checkpoint.checkpoint_dir.display()
                    ),
                    Ok(StampIdentity::TrustOnFirstUse) => warn!(
                        "checkpoint at {} predates body-identity stamping (bddap/rl#214) — \
                         trusting on first use; the next save stamps body digest \
                         {constructed:#018x}",
                        paths.brain_file().display(),
                    ),
                    Err(why) => panic!(
                        "REFUSING to train over checkpoint dir {}: {why}",
                        config.checkpoint.checkpoint_dir.display()
                    ),
                }
                // Channel-layout identity (bddap/rl#271): same class as the body check —
                // a layout change under unchanged dims would pass the DOF gate below and
                // silently retrain with every channel remapped. Abort so the operator
                // picks a fresh dir instead of the run destroying the trained lineage.
                let built_layout = crate::bot::channel_layout_digest();
                match check_channel_layout(layout_digest, built_layout) {
                    Ok(StampIdentity::Match) => {}
                    // TOFU's residual risk: a pre-#271 brain trained on a DIFFERENT
                    // same-dims layout resumes remapped and the next save stamps the
                    // current digest onto that lineage. Undetectable (the digest didn't
                    // exist yet) — the same one-time migration window body TOFU accepted.
                    Ok(StampIdentity::TrustOnFirstUse) => warn!(
                        "checkpoint at {} predates channel-layout stamping (bddap/rl#271) — \
                         trusting on first use; the next save stamps layout digest \
                         {built_layout:#018x}",
                        paths.brain_file().display(),
                    ),
                    Err(why) => panic!(
                        "REFUSING to train over checkpoint dir {}: {why}",
                        config.checkpoint.checkpoint_dir.display()
                    ),
                }
                let (obs, action) = loaded.io_dims();
                if crate::policy::dims_fit_rig(obs, action) {
                    info!(
                        "Loaded {} brain weights from {}",
                        loaded.arch(),
                        paths.brain_file().display()
                    );
                    brain = loaded;
                    resumed = Some(key);
                } else {
                    warn!(
                        "Checkpoint at {} has an incompatible obs/action shape ({obs}/{action}; \
                         the actuated DOF set changed) — starting the policy fresh",
                        paths.brain_file().display()
                    );
                    // Reachable only for pre-#271 (layout-unstamped) brains: a DOF
                    // change also changes the layout digest, so a v4-stamped brain
                    // aborts at the layout check above instead — stricter by design
                    // (fresh dir over silently overwriting the lineage in place).
                    // The deliberate DOF cold start stays on the CHECKPOINT's arch (tag
                    // still authoritative — without a flag the default above could
                    // otherwise silently switch a non-default run's architecture).
                    // Guarded so the matching-arch case keeps the FIRST init (a second
                    // init consumes another backend-RNG draw, which would change a
                    // fixed-`--seed` cold start's initial weights for nothing).
                    if brain.arch() != loaded.arch() {
                        brain = AnyBrain::init(loaded.arch(), &device);
                    }
                }
            }
            // No brain file: a fresh checkpoint dir — the legitimate cold start.
            Some(Err(BrainLoadError::Envelope(EnvelopeError::Absent))) => {}
            Some(Err(e)) => panic!(
                "REFUSING to train over checkpoint dir {}: brain checkpoint is unusable \
                 ({e}). Cold-starting here would silently discard the trained policy; fix \
                 or move the checkpoint before resuming.",
                config.checkpoint.checkpoint_dir.display()
            ),
        }

        // Normalizers are brain-PAIRED: they load iff the brain warm-started, tagged with
        // its arch and stamped with its save stamp (the set-coherence checks,
        // bddap/rl#200 §2 + #215 — a partial save landing one member without the others
        // must refuse here, not load clean), and a warm brain with unusable normalizer
        // stats is an ABORT — training trained weights against cold or mis-paired scales
        // mis-normalizes every observation/value, the exact silent skew the envelope
        // exists to refuse. On a cold start they stay fresh with the fresh brain.
        if let Some(key) = resumed {
            let norm_path = paths.normalizer_path();
            match ObsNormalizer::load(&norm_path, key) {
                Ok(loaded) => {
                    info!("Loaded normalizer state from {}", norm_path.display());
                    obs_normalizer = loaded;
                }
                Err(e) => panic!(
                    "REFUSING to resume {}: obs normalizer at {} is unusable ({e}) — a \
                     warm brain never trains against cold or mis-paired normalizer stats.",
                    config.checkpoint.checkpoint_dir.display(),
                    norm_path.display()
                ),
            }
            let ret_norm_path = paths.return_normalizer_path();
            match load_return_normalizer(&ret_norm_path, key) {
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
                    config.checkpoint.checkpoint_dir.display(),
                    ret_norm_path.display()
                ),
            }
        }

        let n = config.envs.max(1) as usize;
        let normalizer_increment = worker_mode.then(IncrementAccumulator::new);
        let close_frac = config.target_close_frac;
        if close_frac > 0.0 {
            // train.log is multi-run; without this line nobody can later tell which
            // segments trained with the close-target mix.
            info!("Close-target curriculum active: --target-close-frac {close_frac} (rl#250)");
        }
        Self {
            brain: InferenceCachedBrain::new(brain),
            config: PpoConfig {
                log_std_floor_start: config.log_std_floor_start,
                log_std_floor_end: config.log_std_floor_end,
                log_std_anneal_ticks: config.log_std_anneal_ticks,
                ..PpoConfig::default()
            },
            rollouts: (0..n).map(|_| RolloutBuffer::new()).collect(),
            device,
            envs: vec![EnvEpisode::default(); n],
            explore_noise: OuNoise::new(n),
            log_std_floor: crate::bot::arch::LOG_STD_MIN,
            close_frac,
            effort_weight: config.effort_weight,
            log_effort: config.log_effort,
            episode_count: 0,
            recent_rewards: Vec::new(),
            total_steps: 0,
            obs_normalizer,
            return_normalizer,
            rng,
            checkpoint_dir: config.checkpoint.checkpoint_dir.clone(),
            resumed,
            tick_budget: config.ticks,
            reported_episodes: 0,
            normalizer_increment,
            drift_sum: 0.0,
            drift_count: 0,
            reach_reached: 0,
            reach_finished: 0,
            reach_by_bearing: [(0, 0); crate::eval::EVAL_BEARINGS],
            progress_glitch_drops: 0,
            nonfinite_obs_elements: 0,
        }
    }

    /// `Some(key)` iff the brain warm-started from the checkpoint dir at build time —
    /// the optimizer's warm/cold gate AND its pairing key in one: the moments resume
    /// only under the resumed set's key (bddap/rl#215), never onto a cold-started brain.
    pub(crate) fn resumed_set_key(&self) -> Option<SetKey> {
        self.resumed
    }

    /// Save the checkpoint SET (brain + obs normalizer + return normalizer), every
    /// member stamped with one freshly-drawn save_stamp (bddap/rl#215) so a loader can
    /// verify the set was written together. `extra` writes anything saved AS PART OF
    /// the set beyond the core members — the learner passes the optimizer here — into
    /// the same staged generation, stamped with the same save_stamp.
    ///
    /// The whole set lands in ONE atomic dir swap ([`replace_dir_atomically`],
    /// bddap/rl#238): members are staged in a sibling temp dir — the live dir's
    /// non-set entries (`best/`, the watermarks) hardlinked in beside them — then
    /// swapped into place, so a concurrent reader sees the old complete set or the new
    /// complete set, never a mix, and a member failure leaves the live dir untouched.
    /// The brain (the largest member, the one ENOSPC kills) is staged first so a full
    /// disk aborts before any cheap member is written. The #215 stamps stay as the
    /// load-time backstop for out-of-band copies.
    pub(crate) fn save_checkpoint(&self, extra: impl FnOnce(&CheckpointDir, u64)) {
        let arch = self.brain.train().arch();
        let save_stamp: u64 = rand::random();
        let staged = replace_dir_atomically(&self.checkpoint_dir, |staging| {
            let paths = CheckpointDir::new(staging);
            save_brain(self.brain.train(), &paths.brain_file(), save_stamp)?;
            self.obs_normalizer
                .save(arch, &paths.normalizer_path(), save_stamp)?;
            save_return_normalizer(
                &self.return_normalizer,
                arch,
                &paths.return_normalizer_path(),
                save_stamp,
            )?;
            extra(&paths, save_stamp);
            hardlink_missing_entries(&self.checkpoint_dir, staging)
        });
        match staged {
            Ok(()) => info!("Saved checkpoint set to {}", self.checkpoint_dir.display()),
            Err(e) => warn!(
                "Failed to save checkpoint set to {}: {e} — the previous set stays \
                 intact and coherent",
                self.checkpoint_dir.display()
            ),
        }
    }

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
        self.set_log_std_floor(req.log_std_floor);
        self.reset_horizon_counter();
        true
    }

    pub(crate) fn end_horizon(&mut self) -> HorizonOutput {
        HorizonOutput {
            envs: self.take_rollouts(),
            increment: self.normalizer_increment(),
            rewards: self.drain_finished_episode_rewards(),
            drift: self.drain_drift(),
            reach: self.drain_reach(),
            reach_by_bearing: std::mem::replace(
                &mut self.reach_by_bearing,
                [(0, 0); crate::eval::EVAL_BEARINGS],
            ),
            glitch_drops: self.drain_progress_glitches(),
            nonfinite_obs: self.drain_nonfinite_obs(),
        }
    }

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

    pub(crate) fn brain(&self) -> &AnyBrain<TrainBackend> {
        self.brain.train()
    }

    #[must_use = "a failed normalizer load must abort the horizon, not roll mis-normalized obs"]
    fn set_normalizer(&mut self, snapshot: NormalizerSnapshot) -> bool {
        self.obs_normalizer.load_snapshot(snapshot)
    }

    pub(crate) fn normalizer_snapshot(&self) -> NormalizerSnapshot {
        self.obs_normalizer.snapshot()
    }

    fn normalizer_increment(&self) -> NormalizerIncrement {
        self.normalizer_increment
            .as_ref()
            .expect("normalizer increment requested on a non-worker TrainingState")
            .increment()
    }

    pub(crate) fn merge_normalizer(&mut self, increment: &NormalizerIncrement) {
        self.obs_normalizer.merge(increment);
    }

    fn take_rollouts(&mut self) -> Vec<RolloutBuffer> {
        (0..self.rollouts.len())
            .map(|e| RolloutBuffer {
                transitions: std::mem::take(&mut self.rollouts[e].transitions),
                bootstrap: self.envs[e].pending.as_ref().map(|p| p.value),
            })
            .collect()
    }

    fn reset_horizon_counter(&mut self) {
        if let Some(inc) = self.normalizer_increment.as_mut() {
            *inc = IncrementAccumulator::new();
        }
    }

    fn drain_finished_episode_rewards(&mut self) -> Vec<f32> {
        let out = self.recent_rewards[self.reported_episodes..].to_vec();
        self.reported_episodes = self.recent_rewards.len();
        out
    }

    fn drain_drift(&mut self) -> (f64, u64) {
        let out = (self.drift_sum, self.drift_count);
        self.drift_sum = 0.0;
        self.drift_count = 0;
        out
    }

    fn set_log_std_floor(&mut self, log_std_floor: f32) {
        self.log_std_floor = log_std_floor;
    }

    fn drain_reach(&mut self) -> (u64, u64) {
        let out = (self.reach_reached, self.reach_finished);
        self.reach_reached = 0;
        self.reach_finished = 0;
        out
    }

    fn drain_progress_glitches(&mut self) -> u64 {
        std::mem::take(&mut self.progress_glitch_drops)
    }

    fn drain_nonfinite_obs(&mut self) -> u64 {
        std::mem::take(&mut self.nonfinite_obs_elements)
    }

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

    #[allow(clippy::needless_borrows_for_generic_args)]
    #[test]
    fn inference_cache_is_bit_identical_and_refreshes_on_weight_change() {
        let device = NdArrayDevice::Cpu;
        let obs = Tensor::<InferBackend, 2>::zeros([3, OBS_SIZE], &device);
        let bits = |t: &[f32]| -> Vec<u32> { t.iter().map(|v| v.to_bits()).collect() };
        let policy_bits = |brain: &AnyBrain<InferBackend>| -> Vec<u32> {
            let (means, log_std) = brain.policy(obs.clone());
            let value = brain.value(obs.clone());
            let mut out = bits(&means.to_data().to_vec::<f32>().unwrap());
            out.extend(bits(&log_std.to_data().to_vec::<f32>().unwrap()));
            out.extend(bits(&value.to_data().to_vec::<f32>().unwrap()));
            out
        };

        // The refresh assertions below need a fresh brain whose weights DIFFER from the
        // previous one. Fresh inits draw from the backend's process-GLOBAL RNG, so under a
        // parallel test run a sibling reseeding it (`brain_init_is_reproducible_…` below, or
        // any `TrainingState::new` caller) can make two draws straddling its reseed come back
        // bit-identical (rl#207). Retry across that transient window — the dual of the
        // sibling's same-seed retry: a race can only make fresh draws spuriously EQUAL, while
        // an init that stopped drawing from the RNG never yields a differing pair, so the
        // bound still bites.
        let fresh_brain_differing_from = |prev: &[u32]| -> (AnyBrain<TrainBackend>, Vec<u32>) {
            for _ in 0..16 {
                let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
                let bits = policy_bits(&brain.valid());
                if bits != prev {
                    return (brain, bits);
                }
            }
            panic!(
                "16 fresh inits all bit-identical to the previous brain — init is not drawing \
                 fresh weights, so the refresh assertions cannot bite"
            );
        };

        let brain_a: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
        let want_a = policy_bits(&brain_a.valid());
        let mut cached = InferenceCachedBrain::new(brain_a);

        assert_eq!(cached.with_inference(&policy_bits), want_a);
        assert_eq!(cached.with_inference(&policy_bits), want_a);

        let (brain_b, want_b) = fresh_brain_differing_from(&want_a);
        cached.set(brain_b);
        assert_eq!(cached.with_inference(&policy_bits), want_b);

        let (brain_c, want_c) = fresh_brain_differing_from(&want_b);
        *cached.train_mut() = brain_c;
        assert_eq!(cached.with_inference(&policy_bits), want_c);
    }

    /// `save_checkpoint` at its real call shape: the set lands complete under ONE
    /// pairing key (#215 ∘ #238 — a torn or mixed-generation dir cannot load), and the
    /// live dir's non-set entries (`best/`, the tick watermark) survive the swap — the
    /// carry is load-bearing, so a save must never silently destroy the best snapshot
    /// or reset the odometer.
    #[test]
    fn save_checkpoint_lands_one_coherent_set_and_preserves_non_set_entries() {
        let dir = std::env::temp_dir().join("rl_test_save_checkpoint_set");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("best")).unwrap();
        std::fs::write(dir.join("best/brain.bin"), b"incumbent-best").unwrap();
        std::fs::write(dir.join("ticks.txt"), b"777").unwrap();

        let config = crate::TrainConfig::scratch(&dir, 1, 42);
        let state = TrainingState::new(&config, None);
        let extra_saw_staged_dir = RefCell::new(false);
        state.save_checkpoint(|paths, _stamp| {
            assert_ne!(
                paths.brain_file().parent().unwrap(),
                dir,
                "extra members are written into the staged generation, never the live dir"
            );
            *extra_saw_staged_dir.borrow_mut() = true;
        });
        assert!(*extra_saw_staged_dir.borrow());

        let paths = CheckpointDir::new(&dir);
        let brain = load_brain_file::<TrainBackend>(&paths.brain_file(), &NdArrayDevice::Cpu)
            .expect("saved brain loads");
        let key = brain.set_key();
        assert!(key.save_stamp.is_some(), "the set is stamped");
        ObsNormalizer::load(&paths.normalizer_path(), key)
            .expect("obs normalizer pairs with the brain's key");
        load_return_normalizer(&paths.return_normalizer_path(), key)
            .expect("return normalizer pairs with the brain's key");
        assert_eq!(
            std::fs::read(dir.join("best/brain.bin")).unwrap(),
            b"incumbent-best",
            "best/ survives the swap"
        );
        assert_eq!(std::fs::read(dir.join("ticks.txt")).unwrap(), b"777");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn brain_init_is_reproducible_from_the_backend_seed() {
        let device = NdArrayDevice::Cpu;
        let obs = Tensor::<InferBackend, 2>::zeros([3, OBS_SIZE], &device);
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
            weight_bits(&AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device).valid())
        };

        let reproducible = (0..16).any(|_| init_with_seed(0x5EED) == init_with_seed(0x5EED));
        assert!(
            reproducible,
            "same backend seed never reproduced identical initial weights across 16 tries — init \
             is not seed-deterministic, so training can't replay a run from its logged --seed"
        );
        assert_ne!(
            init_with_seed(0x5EED),
            init_with_seed(0xC0FFEE),
            "different seeds must give different initial weights, else seeding does nothing"
        );
    }
}
