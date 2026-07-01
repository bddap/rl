//! Persisted training artifacts and the backend/optimizer types they serialize: the
//! atomic-write primitive, checkpoint file names, the Adam optimizer construction +
//! its versioned on-disk envelope, and the return-normalizer round trip. The brain and
//! obs-normalizer checkpoints are written by their owners ([`super::systems::TrainingState`]
//! and [`super::normalizer::ObsNormalizer`]); this module owns the shared plumbing.

use std::path::{Path, PathBuf};

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig};
use burn::tensor::backend::AutodiffBackend;
use tracing::warn;

use super::algorithm::{ReturnNormalizer, ReturnNormalizerData};
use super::atomic_write;
use crate::bot::actuator::ACTION_SIZE;
use crate::bot::arch::AnyBrain;
use crate::bot::sensor::OBS_SIZE;

/// The Adam optimizer over an [`AnyBrain`] on backend `B` — the moments key off leaf
/// `ParamId`s, so the enum wrapper changes nothing about the record. Generic over the backend so
/// the one PPO update ([`super::update::ppo_update_core`]) serves the live GPU learner
/// and the CPU-backed update test from one code path.
pub(crate) type CrabOpt<B> = OptimizerAdaptor<Adam, AnyBrain<B>, B>;

/// Write a brain's LEAF record to `stem` (the recorder appends `.bin`) — the ONE recipe
/// for how a brain lands on disk, shared by the trainer's checkpoint write and the test
/// fixtures. That sharing is the writer-side drift guard: the save→`Policy::load` tests
/// exercise this exact function, so a regression that made it round-trip the enum record
/// (see [`AnyBrain::record_leaf`]) turns those tests red instead of silently stranding
/// the fleet on the rest pose.
pub(crate) fn save_brain_record<B: burn::tensor::backend::Backend>(
    brain: &AnyBrain<B>,
    stem: std::path::PathBuf,
) -> Result<(), burn::record::RecorderError> {
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    brain
        .record_leaf(&BinFileRecorder::<FullPrecisionSettings>::default(), stem)
        .map(|_| ())
}

/// Build the learner's Adam optimizer (global grad-norm clip 0.5). The ONE source of
/// the optimizer construction — the live `GpuLearner` and the CPU-backed update test
/// both call this, so the clip constant can't silently drift between paths meant to be
/// identical.
pub(crate) fn crab_optimizer<B: AutodiffBackend>() -> CrabOpt<B> {
    AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
        .init()
}

/// Canonical checkpoint filenames. The single place each artifact's on-disk name lives;
/// every reader and writer (training, resume, the demo's load + hot-reload, and the
/// best-keeper's [`super::best`] snapshot) references these rather than re-typing the
/// string, so the names can't drift between modules.
///
/// Stem (no extension) for the brain record — `BinFileRecorder` appends `.bin`, so the
/// file on disk is [`BRAIN_FILENAME`].
const BRAIN_STEM: &str = "brain";
/// `brain.bin` as it lands on disk — [`BRAIN_STEM`] plus the `.bin` the recorder appends.
pub(crate) const BRAIN_FILENAME: &str = "brain.bin";
pub(crate) const NORMALIZER_FILENAME: &str = "normalizer.bin";
/// Return (value-target) normalizer checkpoint, beside the obs normalizer, so a
/// resumed run de-normalizes value predictions against the same scale it trained
/// with (a cold scale on resume would briefly mis-scale the value head).
pub(crate) const RETURN_NORMALIZER_FILENAME: &str = "return_normalizer.bin";
/// Persisted Adam optimizer state (rl#60): the per-parameter first/second moments and step
/// (`time`) the GPU learner carries across iterations. A resume restores these so the
/// optimizer continues with warm momentum instead of paying the brief self-correcting
/// transient a cold restart costs. Absent in pre-rl#60 checkpoints, which then resume cold
/// (see [`load_optimizer`]) rather than erroring — the format version inside the file
/// (see [`OPTIMIZER_FORMAT_VERSION`]) guards a layout change the same way.
pub(crate) const OPTIMIZER_FILENAME: &str = "optimizer.bin";
/// Tick-budget odometer, beside the checkpoint, so a restarted learner resumes the
/// `--ticks` budget rather than restarting it (the overnight loop makes restarts the
/// expected case). Read/written by [`super::inproc`]; policy-independent.
pub(crate) const TICK_WATERMARK_FILENAME: &str = "ticks.txt";

/// The on-disk layout of a checkpoint directory: the single place that knows which
/// filename each artifact uses and how its path is assembled. Callers ask for
/// [`Self::brain_stem`] / [`Self::normalizer_path`] / … instead of re-deriving
/// `dir.join(CONST)` by hand, so a layout change lands in one place and every reader and
/// writer (training, resume, the demo's load + hot-reload) stays in lockstep. Borrows the
/// dir, so it's free to construct at each use.
pub(crate) struct CheckpointDir<'a> {
    dir: &'a Path,
}

impl<'a> CheckpointDir<'a> {
    pub(crate) fn new(dir: &'a Path) -> Self {
        Self { dir }
    }

    /// Stem (no extension) for the brain record — pass this to the recorder's
    /// `record`/`load`, which append `.bin` themselves. Use [`Self::brain_file`] for the
    /// file as it lands on disk.
    pub(crate) fn brain_stem(&self) -> PathBuf {
        self.dir.join(BRAIN_STEM)
    }

    /// The brain file on disk (`brain.bin`) — for existence and mtime checks.
    pub(crate) fn brain_file(&self) -> PathBuf {
        self.dir.join(BRAIN_FILENAME)
    }

    /// Temp stem the brain is recorded to before an atomic rename onto [`Self::brain_file`],
    /// so a crash mid-write can't leave a torn `brain.bin` (silently discarded on load →
    /// resume from random weights). Dot-free for the same reason the stem is: the recorder
    /// forces the `.bin` extension, so a "brain.tmp" stem would clobber the live file.
    pub(crate) fn brain_tmp_stem(&self) -> PathBuf {
        self.dir.join("brain-tmp")
    }

    pub(crate) fn normalizer_path(&self) -> PathBuf {
        self.dir.join(NORMALIZER_FILENAME)
    }

    pub(crate) fn return_normalizer_path(&self) -> PathBuf {
        self.dir.join(RETURN_NORMALIZER_FILENAME)
    }

    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn optimizer_path(&self) -> PathBuf {
        self.dir.join(OPTIMIZER_FILENAME)
    }

    fn shape_path(&self) -> PathBuf {
        self.dir.join(SHAPE_FILENAME)
    }

    /// Whether this checkpoint's persisted obs/action widths match the running build's
    /// — the gate every shape-bound warm load (the brain, the Adam optimizer) consults
    /// before trusting the saved weights. `false` when the sidecar is absent (a fresh
    /// dir, or a pre-guard checkpoint) or its widths differ, so the caller starts that
    /// artifact cold instead of loading a shape-incompatible record. See [`SHAPE_FILENAME`].
    pub(crate) fn warm_start_compatible(&self) -> bool {
        match BrainShape::load(&self.shape_path()) {
            Some(saved) => saved == BrainShape::current(),
            None => false,
        }
    }

    /// Persist the current obs/action widths beside the brain. Called on every brain
    /// checkpoint so the sidecar can't lag the weights it guards.
    pub(crate) fn save_shape(&self) {
        let s = BrainShape::current();
        let body = format!("{} {}\n", s.obs, s.action);
        if let Err(e) = atomic_write(&self.shape_path(), body.as_bytes()) {
            warn!(
                "Failed to write brain shape sidecar to {}: {e}",
                self.shape_path().display()
            );
        }
    }
}

/// The observation/action vector widths a checkpoint's weights were trained at,
/// recorded beside the brain as [`SHAPE_FILENAME`]. Both derive from
/// [`crate::bot::body::CrabJointId::COUNT`] (the policy-actuated DOF set), so adding
/// articulation (bddap/rl#31) changes them — and a brain saved at the old width is
/// silently wrong in the new net (burn would load a `[30, …]` policy head into a
/// `[38, …]` one). Persisting and checking the widths makes that mismatch loud and
/// safe by construction: a resume refuses to warm-start an incompatible checkpoint and
/// starts cold instead. A pre-guard checkpoint has no sidecar and is likewise treated
/// as incompatible — exactly the behavior wanted across a DOF change (the stale weights
/// are discarded, not loaded askew).
pub(crate) const SHAPE_FILENAME: &str = "shape.txt";

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
struct BrainShape {
    obs: usize,
    action: usize,
}

impl BrainShape {
    fn current() -> Self {
        Self {
            obs: OBS_SIZE,
            action: ACTION_SIZE,
        }
    }

    /// Parse the two whitespace-separated widths; `None` on a missing, unreadable, or
    /// malformed sidecar (all of which the caller treats as "not warm-startable").
    fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut it = text.split_whitespace();
        let obs = it.next()?.parse().ok()?;
        let action = it.next()?.parse().ok()?;
        Some(Self { obs, action })
    }
}

/// Format version of the [`OPTIMIZER_FILENAME`] envelope. Bumped whenever the serialized
/// layout changes; a file tagged with any other version is ignored on load (→ cold
/// moments), never deserialized blind. v1 is burn's Adam record (per-param m/v + step
/// `time`) under `FullPrecisionSettings`, wrapped in this versioned bincode envelope.
#[cfg(any(feature = "wgpu", test))]
const OPTIMIZER_FORMAT_VERSION: u32 = 1;

/// On-disk envelope for the optimizer state: a version tag wrapping the burn optimizer
/// record's bytes. The bytes are device-independent — the `FullPrecisionSettings` recorder
/// reads the moment tensors off whatever device the optimizer lives on (the GPU, in
/// production) into host floats on save, and uploads them back on load — so one file
/// restores onto whichever device the next learner brings up.
#[cfg(any(feature = "wgpu", test))]
#[derive(serde::Serialize, serde::Deserialize)]
struct OptimizerCheckpoint {
    version: u32,
    record: Vec<u8>,
}

/// Persist an Adam optimizer's state (per-param first/second moments + step) to `path`,
/// atomically and version-tagged. Generic over the backend so the live GPU learner and the
/// CPU round-trip test serialize through ONE path — no save/load drift. The
/// `FullPrecisionSettings` recorder reads the moment tensors back off the optimizer's device
/// into host floats, exactly as the brain bridge does for weights. Best-effort: any failure
/// is logged, not fatal (a resume then loads cold — see [`OPTIMIZER_FILENAME`]).
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn save_optimizer<B: AutodiffBackend>(optimizer: &CrabOpt<B>, path: &Path) {
    use burn::optim::Optimizer;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

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
/// error — when the file is absent, unreadable, corrupt, or version-unrecognized (see
/// [`OPTIMIZER_FILENAME`] for why cold is the correct fallback). The per-parameter keys line
/// up across the round trip because the resumed brain restores the SAME `ParamId`s from its
/// own record, so each moment lands back on its parameter.
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn load_optimizer<B: AutodiffBackend>(
    cold: CrabOpt<B>,
    path: &Path,
    device: &B::Device,
) -> CrabOpt<B> {
    use burn::optim::Optimizer;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

    let Ok(bytes) = std::fs::read(path) else {
        // Absent file is the EXPECTED case for a pre-rl#60 checkpoint, so info, not warn —
        // a warm-continue of an older policy with cold moments is normal, not a fault.
        tracing::info!(
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
            tracing::info!("Restored Adam optimizer state from {}", path.display());
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

/// Persist the return normalizer's running stats (bincode, like the obs normalizer).
/// A write failure is logged, not fatal — the run continues, only resume loses the
/// scale.
pub(crate) fn save_return_normalizer(norm: &ReturnNormalizer, path: &Path) {
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
pub(crate) fn load_return_normalizer(path: &Path) -> Option<ReturnNormalizer> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::arch::ArchId;
    use crate::bot::sensor::OBS_SIZE;
    use crate::training::TrainBackend;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::optim::{GradientsParams, Optimizer};
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    use burn::tensor::Tensor;

    /// The shape sidecar gates warm-starts: absent ⇒ never warm-start (a fresh dir or a
    /// pre-guard checkpoint), matching widths ⇒ warm, differing widths ⇒ cold. This is the
    /// guard that keeps a DOF change (bddap/rl#31) from loading an old-width brain into the
    /// re-shaped net.
    #[test]
    fn shape_guard_gates_warm_start() {
        let dir = std::env::temp_dir().join(format!("rl_test_shape_guard_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let paths = CheckpointDir::new(&dir);

        // No sidecar yet ⇒ not warm-startable.
        assert!(
            !paths.warm_start_compatible(),
            "absent sidecar must block warm start"
        );

        // After stamping the current widths ⇒ warm-startable.
        paths.save_shape();
        assert!(
            paths.warm_start_compatible(),
            "a sidecar matching the current widths must allow warm start"
        );
        let parsed = BrainShape::load(&paths.shape_path()).expect("sidecar parses");
        assert_eq!(parsed, BrainShape::current());

        // A sidecar from a different DOF count ⇒ blocked again.
        std::fs::write(
            paths.shape_path(),
            format!("{} {}\n", OBS_SIZE + 1, ACTION_SIZE + 1),
        )
        .unwrap();
        assert!(
            !paths.warm_start_compatible(),
            "a sidecar with different widths must block warm start"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn brain_checkpoint_round_trips() {
        let dir = std::env::temp_dir().join("rl_test_brain_checkpoint");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let device = NdArrayDevice::Cpu;
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);

        let paths = CheckpointDir::new(&dir);
        let stem = paths.brain_stem();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        brain
            .record_leaf(&recorder, stem.clone())
            .expect("save brain");

        assert!(paths.brain_file().exists(), "brain.bin should exist");

        let loaded = AnyBrain::<TrainBackend>::init(ArchId::Mlp256, &device)
            .load_leaf_record(&recorder, stem, &device)
            .expect("load brain");

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
        brain: AnyBrain<TrainBackend>,
        mut optimizer: CrabOpt<TrainBackend>,
        device: &NdArrayDevice,
    ) -> (AnyBrain<TrainBackend>, CrabOpt<TrainBackend>) {
        let obs = Tensor::<TrainBackend, 2>::ones([4, OBS_SIZE], device);
        let (means, log_std) = brain.policy(obs);
        let loss = means.sum() + log_std.sum();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &brain);
        let brain = optimizer.step(1e-2, brain, grads);
        (brain, optimizer)
    }

    fn policy_means(brain: &AnyBrain<TrainBackend>, device: &NdArrayDevice) -> Vec<f32> {
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
        let mut brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);
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

        let path = CheckpointDir::new(&dir).optimizer_path();
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
        let path = CheckpointDir::new(&dir).optimizer_path();

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
}
