//! Persisted training artifacts and the backend/optimizer types they serialize: the
//! checkpoint file names/layout, the enveloped brain + Adam-optimizer + return-normalizer
//! round trips, and the Adam construction. Every artifact ships inside a
//! [`CheckpointEnvelope`](super::envelope) (kind + per-kind version + arch), so a loader
//! always knows WHAT it is decoding and for WHICH architecture — see [`load_brain_file`]
//! for the arch dispatch. The obs-normalizer round trip lives with its owner
//! ([`super::normalizer::ObsNormalizer`]); this module owns the shared plumbing.

use std::path::{Path, PathBuf};

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use tracing::warn;

use super::algorithm::{ReturnNormalizer, ReturnNormalizerData};
use super::envelope::{
    ArtifactKind, EnvelopeError, read_envelope, read_envelope_expecting, write_envelope,
};
use crate::bot::arch::{AnyBrain, ArchId};

/// The Adam optimizer over an [`AnyBrain`] on backend `B` — the moments key off leaf
/// `ParamId`s, so the enum wrapper changes nothing about the record. Generic over the backend so
/// the one PPO update ([`super::update::ppo_update_core`]) serves the live GPU learner
/// and the CPU-backed update test from one code path.
pub(crate) type CrabOpt<B> = OptimizerAdaptor<Adam, AnyBrain<B>, B>;

/// Write a brain to `path` as its LEAF record inside a [`ArtifactKind::Brain`] envelope
/// tagged with the brain's own arch — the ONE recipe for how a brain lands on disk,
/// shared by the trainer's checkpoint write and the test fixtures. That sharing is the
/// writer-side drift guard: the save→`Policy::load` tests exercise this exact function,
/// so a regression that made it round-trip the enum record (see [`AnyBrain::record_leaf`])
/// or drop the envelope turns those tests red instead of silently stranding the fleet.
/// Atomic (temp + fsync-rename), so a crash mid-write can't leave a torn `brain.bin`.
pub(crate) fn save_brain<B: Backend>(brain: &AnyBrain<B>, path: &Path) -> std::io::Result<()> {
    let bytes = brain
        .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
        .map_err(std::io::Error::other)?;
    write_envelope(path, ArtifactKind::Brain, brain.arch(), bytes)
}

/// Why a brain file did not yield a brain. `Envelope` wraps the tag-level refusals
/// (absent/legacy/corrupt/unknown-arch/…); `Record` means the envelope validated but the
/// payload didn't decode as that arch's leaf record.
#[derive(Debug)]
pub(crate) enum BrainLoadError {
    Envelope(EnvelopeError),
    Record(String),
}

impl std::fmt::Display for BrainLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Envelope(e) => e.fmt(f),
            Self::Record(e) => write!(f, "leaf record does not decode: {e}"),
        }
    }
}

/// A contained panic's message, for attribution in the refusal it becomes.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

/// Decode `payload` as `arch`'s leaf record on `device`. burn's bytes recorder PANICS on
/// malformed input (it unwraps its internal bincode decode), so the decode is contained
/// here — where brain record bytes from DISK meet the recorder — and a bad payload
/// surfaces as an `Err` the callers' refusal policies can act on, never a crash. (The
/// containment is control-flow only: `catch_unwind` doesn't suppress the panic hook, so
/// stderr still shows the panic + backtrace before the tidy refusal line. The in-process
/// snapshot/GPU bridges keep their bare recorder calls: they move a live brain's own
/// bytes, where a decode failure IS a bug worth the panic.)
pub(crate) fn decode_brain_payload<B: Backend>(
    arch: ArchId,
    payload: Vec<u8>,
    device: &B::Device,
) -> Result<AnyBrain<B>, String> {
    let device = device.clone();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        AnyBrain::<B>::init(arch, &device).load_leaf_record(
            &BinBytesRecorder::<FullPrecisionSettings>::default(),
            payload,
            &device,
        )
    }))
    .map_err(|p| format!("record bytes do not decode: {}", panic_message(p)))?
    .map_err(|e| e.to_string())
}

/// Load the brain at `path`, DISPATCHING on the envelope's arch tag: read the envelope,
/// `AnyBrain::init` the tagged architecture, then load the leaf record into it. The one
/// chokepoint where a checkpoint chooses its architecture — no code path blind-loads a
/// record into a guessed variant, and an unregistered arch is refused by name before any
/// payload decode. Callers apply their own refusal policy to the error (trainer aborts,
/// inference refuses loudly, see bddap/rl#200 §2).
pub(crate) fn load_brain_file<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<AnyBrain<B>, BrainLoadError> {
    let env = read_envelope(path, ArtifactKind::Brain).map_err(BrainLoadError::Envelope)?;
    decode_brain_payload::<B>(env.arch, env.payload, device).map_err(BrainLoadError::Record)
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
pub(crate) const BRAIN_FILENAME: &str = "brain.bin";
pub(crate) const NORMALIZER_FILENAME: &str = "normalizer.bin";
/// Return (value-target) normalizer checkpoint, beside the obs normalizer, so a
/// resumed run de-normalizes value predictions against the same scale it trained
/// with (a cold scale on resume would briefly mis-scale the value head).
pub(crate) const RETURN_NORMALIZER_FILENAME: &str = "return_normalizer.bin";
/// Persisted Adam optimizer state (rl#60): the per-parameter first/second moments and step
/// (`time`) the GPU learner carries across iterations. A resume restores these so the
/// optimizer continues with warm momentum instead of paying the brief self-correcting
/// transient a cold restart costs. Absent in older checkpoints, which then resume cold
/// (see [`load_optimizer`]) rather than erroring.
pub(crate) const OPTIMIZER_FILENAME: &str = "optimizer.bin";
/// Tick-budget odometer, beside the checkpoint, so a restarted learner resumes the
/// `--ticks` budget rather than restarting it (the overnight loop makes restarts the
/// expected case). Read/written by [`super::inproc`]; policy-independent plain text,
/// so it carries no envelope.
pub(crate) const TICK_WATERMARK_FILENAME: &str = "ticks.txt";

/// The on-disk layout of a checkpoint directory: the single place that knows which
/// filename each artifact uses and how its path is assembled. Callers ask for
/// [`Self::brain_file`] / [`Self::normalizer_path`] / … instead of re-deriving
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

    /// The brain file on disk (`brain.bin`).
    pub(crate) fn brain_file(&self) -> PathBuf {
        self.dir.join(BRAIN_FILENAME)
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
}

/// Persist an Adam optimizer's state (per-param first/second moments + step) to `path`,
/// atomically, inside an [`ArtifactKind::Optimizer`] envelope tagged with `arch` — the
/// architecture of the brain these moments belong to. The tag is the optimizer's SOLE
/// cross-arch guard: the record is a `HashMap<ParamId, …>`, so a wrong-arch load would
/// not even fail — every lookup would silently miss and the moments go cold. Generic over
/// the backend so the live GPU learner and the CPU round-trip test serialize through ONE
/// path — no save/load drift. Best-effort: any failure is logged, not fatal (a resume
/// then loads cold — see [`OPTIMIZER_FILENAME`]).
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn save_optimizer<B: AutodiffBackend>(
    optimizer: &CrabOpt<B>,
    arch: ArchId,
    path: &Path,
) {
    use burn::optim::Optimizer;

    let record = optimizer.to_record();
    let bytes = match BinBytesRecorder::<FullPrecisionSettings>::default().record(record, ()) {
        Ok(b) => b,
        Err(e) => {
            warn!("Failed to serialize Adam optimizer state: {e}");
            return;
        }
    };
    if let Err(e) = write_envelope(path, ArtifactKind::Optimizer, arch, bytes) {
        warn!(
            "Failed to write Adam optimizer state to {}: {e}",
            path.display()
        );
    }
}

/// Load an Adam optimizer state saved by [`save_optimizer`] onto `device`, returning the
/// optimizer with its moments + step restored. REFUSE-AND-COLD is this artifact's whole
/// refusal policy (bddap/rl#200 §2): the `cold` optimizer is returned UNCHANGED — logged,
/// no error — when the file is absent, legacy, corrupt, version-unrecognized, or tagged
/// with an arch other than `expected_arch` (the resumed brain's — the dir-coherence
/// check). Cold moments only cost a brief self-correcting transient; wrong moments would
/// silently miss every `ParamId` lookup. The per-parameter keys line up across an honest
/// round trip because the resumed brain restores the SAME `ParamId`s from its own record.
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn load_optimizer<B: AutodiffBackend>(
    cold: CrabOpt<B>,
    path: &Path,
    device: &B::Device,
    expected_arch: ArchId,
) -> CrabOpt<B> {
    use burn::optim::Optimizer;

    let env = match read_envelope_expecting(path, ArtifactKind::Optimizer, expected_arch) {
        Ok(env) => env,
        Err(EnvelopeError::Absent) => {
            // Absent is the EXPECTED case for an older checkpoint, so info, not warn — a
            // warm-continue of a policy with cold moments is normal, not a fault.
            tracing::info!(
                "No Adam optimizer state at {} — starting the optimizer cold",
                path.display()
            );
            return cold;
        }
        Err(e) => {
            warn!(
                "Refusing Adam optimizer state at {}: {e} — starting cold",
                path.display()
            );
            return cold;
        }
    };
    // Contained like `decode_brain_payload`: burn's bytes recorder PANICS on malformed
    // input, and this artifact's policy is refuse-and-cold, never a crash at resume.
    let device_owned = device.clone();
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        BinBytesRecorder::<FullPrecisionSettings>::default().load(env.payload, &device_owned)
    }));
    match loaded {
        Ok(Ok(record)) => {
            tracing::info!("Restored Adam optimizer state from {}", path.display());
            cold.load_record(record)
        }
        Ok(Err(e)) => {
            warn!(
                "Failed to decode Adam optimizer record at {}: {e} — starting cold",
                path.display()
            );
            cold
        }
        Err(p) => {
            warn!(
                "Failed to decode Adam optimizer record at {}: {} — starting cold",
                path.display(),
                panic_message(p)
            );
            cold
        }
    }
}

/// Persist the return normalizer's running stats inside a
/// [`ArtifactKind::ReturnNormalizer`] envelope tagged with `arch` — the brain these value
/// scales were trained against. A write failure is logged, not fatal — the run continues,
/// only resume loses the scale.
pub(crate) fn save_return_normalizer(norm: &ReturnNormalizer, arch: ArchId, path: &Path) {
    let bytes = match bincode::serialize(&norm.to_data()) {
        Ok(b) => b,
        Err(e) => {
            warn!("Failed to serialize return normalizer: {e}");
            return;
        }
    };
    if let Err(e) = write_envelope(path, ArtifactKind::ReturnNormalizer, arch, bytes) {
        warn!(
            "Failed to write return normalizer to {}: {e}",
            path.display()
        );
    }
}

/// Load the return normalizer from a checkpoint, refusing an envelope whose arch tag
/// differs from `expected_arch` (the resumed brain's — normalizers are brain-PAIRED and
/// must never load cross-arch, bddap/rl#200 §2). The caller applies the refusal policy:
/// the trainer aborts rather than train warm weights against cold/mis-paired scales.
pub(crate) fn load_return_normalizer(
    path: &Path,
    expected_arch: ArchId,
) -> Result<ReturnNormalizer, EnvelopeError> {
    let env = read_envelope_expecting(path, ArtifactKind::ReturnNormalizer, expected_arch)?;
    let data: ReturnNormalizerData = bincode::deserialize(&env.payload)
        .map_err(|e| EnvelopeError::Corrupt(format!("return normalizer payload: {e}")))?;
    ReturnNormalizer::from_data(data).ok_or_else(|| {
        EnvelopeError::Corrupt("return normalizer stats invalid (negative M2)".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::sensor::OBS_SIZE;
    use crate::training::TrainBackend;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::optim::{GradientsParams, Optimizer};
    use burn::tensor::Tensor;

    #[test]
    fn brain_checkpoint_round_trips() {
        let dir = std::env::temp_dir().join("rl_test_brain_checkpoint");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let device = NdArrayDevice::Cpu;
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);

        let paths = CheckpointDir::new(&dir);
        save_brain(&brain, &paths.brain_file()).expect("save brain");
        assert!(paths.brain_file().exists(), "brain.bin should exist");

        let loaded = load_brain_file::<TrainBackend>(&paths.brain_file(), &device)
            .expect("load brain via the arch-dispatching loader");
        assert_eq!(loaded.arch(), ArchId::Mlp256, "arch restored from the tag");

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

    /// A legacy (pre-envelope) brain file is REFUSED with the migrate-pointing verdict —
    /// the loader has no untagged-read path, so the old quiet-rest-pose degrade on a raw
    /// record is impossible by construction.
    #[test]
    fn legacy_brain_file_is_refused_not_blind_loaded() {
        let dir = std::env::temp_dir().join("rl_test_brain_legacy_refuse");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;

        // Exactly what a pre-envelope writer produced: the bare leaf record bytes.
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);
        let raw = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        let path = CheckpointDir::new(&dir).brain_file();
        std::fs::write(&path, raw).unwrap();

        match load_brain_file::<TrainBackend>(&path, &device) {
            Err(BrainLoadError::Envelope(EnvelopeError::Legacy)) => {}
            other => panic!("expected Legacy refusal, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn return_normalizer_round_trips_and_refuses_cross_arch() {
        let dir = std::env::temp_dir().join("rl_test_retnorm_envelope");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = CheckpointDir::new(&dir).return_normalizer_path();

        let norm = ReturnNormalizer::new();
        save_return_normalizer(&norm, ArchId::Mlp256, &path);
        assert!(
            load_return_normalizer(&path, ArchId::Mlp256).is_ok(),
            "matching arch loads"
        );
        // No second arch is registered yet, so the cross-arch refusal is exercised at the
        // envelope layer instead: a WRONG-KIND read of the same file must refuse.
        assert!(
            matches!(
                read_envelope(&path, ArtifactKind::ObsNormalizer),
                Err(EnvelopeError::WrongKind { .. })
            ),
            "a return-normalizer envelope read as an obs normalizer must refuse by kind"
        );
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
        save_optimizer(&warm, ArchId::Mlp256, &path);
        assert!(path.exists(), "optimizer.bin should be written");

        // A fresh cold optimizer loaded from the snapshot must take the NEXT step
        // identically to the warm one (same momentum + step/bias-correction), and
        // DIFFERENTLY from a truly cold optimizer — that difference is exactly the
        // self-correcting transient a warm resume avoids.
        let restored = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::Mlp256,
        );
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
    fn missing_legacy_or_miscopied_optimizer_state_loads_cold_without_error() {
        let dir = std::env::temp_dir().join("rl_test_adam_compat");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let path = CheckpointDir::new(&dir).optimizer_path();

        // (a) Absent file — an older checkpoint. Loads cold, no panic/error.
        assert!(!path.exists());
        let cold = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::Mlp256,
        );
        assert!(
            cold.to_record().is_empty(),
            "an absent optimizer file must leave the optimizer cold"
        );

        // (b) A legacy pre-envelope file (the old `OptimizerCheckpoint` bincode) has no
        //     magic → refused, cold, no panic. The migrate tool is its only parser.
        std::fs::write(&path, bincode::serialize(&(1u32, vec![0u8; 4])).unwrap()).unwrap();
        let cold2 = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::Mlp256,
        );
        assert!(
            cold2.to_record().is_empty(),
            "a legacy optimizer file must leave the optimizer cold"
        );

        // (c) A mis-copied file — a BRAIN envelope at the optimizer path — fails the kind
        //     check → cold, never decoded as moments.
        write_envelope(&path, ArtifactKind::Brain, ArchId::Mlp256, vec![1, 2, 3]).unwrap();
        let cold3 = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::Mlp256,
        );
        assert!(
            cold3.to_record().is_empty(),
            "a wrong-kind file must leave the optimizer cold"
        );

        // (d) A VALID envelope whose inner record bytes are corrupt: burn's bytes recorder
        //     panics on malformed input, so this pins the containment — refuse-and-cold,
        //     never a crash at resume.
        write_envelope(
            &path,
            ArtifactKind::Optimizer,
            ArchId::Mlp256,
            vec![0xde, 0xad, 0xbe, 0xef],
        )
        .unwrap();
        let cold4 = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::Mlp256,
        );
        assert!(
            cold4.to_record().is_empty(),
            "a corrupt optimizer payload must leave the optimizer cold, not panic"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
