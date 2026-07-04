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
///
/// The envelope is stamped with THIS process's constructed body digest
/// ([`crate::mesh_fallback::constructed_body_digest`], bddap/rl#214) — read here, not
/// taken as a parameter, so no caller can stamp a body the process didn't actually
/// build; the resume check ([`check_body_identity`]) aborts a mismatch BEFORE any save,
/// so the stamp can never launder a wrong-body resume either. `generation` is the
/// checkpoint-set stamp shared with the paired artifacts saved beside it (bddap/rl#215).
pub(crate) fn save_brain<B: Backend>(
    brain: &AnyBrain<B>,
    path: &Path,
    generation: u64,
) -> std::io::Result<()> {
    let bytes = brain
        .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
        .map_err(std::io::Error::other)?;
    write_envelope(
        path,
        ArtifactKind::Brain,
        brain.arch(),
        bytes,
        Some(crate::mesh_fallback::constructed_body_digest()),
        generation,
    )
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

/// A loaded brain plus the body identity its envelope carried — [`load_brain_file`]'s
/// result, so no caller can take the weights while dropping the body stamp on the floor.
pub(crate) struct BrainFile<B: Backend> {
    pub(crate) brain: AnyBrain<B>,
    /// See [`CheckpointEnvelope::body_digest`](super::envelope::CheckpointEnvelope):
    /// `None` = a pre-#214 v1 brain (trust-on-first-use).
    pub(crate) body_digest: Option<u64>,
    /// The checkpoint-set generation the paired artifacts must carry to load with this
    /// brain (bddap/rl#215); `None` = a pre-stamp brain, pairing only with unstamped
    /// partners. See [`CheckpointEnvelope::generation`](super::envelope::CheckpointEnvelope).
    pub(crate) generation: Option<u64>,
}

/// Load the brain at `path`, DISPATCHING on the envelope's arch tag: read the envelope,
/// `AnyBrain::init` the tagged architecture, then load the leaf record into it. The one
/// chokepoint where a checkpoint chooses its architecture — no code path blind-loads a
/// record into a guessed variant, and an unregistered arch is refused by name before any
/// payload decode. Callers apply their own refusal policy to the error (trainer aborts,
/// inference refuses loudly, see bddap/rl#200 §2) and to the body digest
/// ([`check_body_identity`]).
pub(crate) fn load_brain_file<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<BrainFile<B>, BrainLoadError> {
    let env = read_envelope(path, ArtifactKind::Brain).map_err(BrainLoadError::Envelope)?;
    let brain =
        decode_brain_payload::<B>(env.arch, env.payload, device).map_err(BrainLoadError::Record)?;
    Ok(BrainFile {
        brain,
        body_digest: env.body_digest,
        generation: env.generation,
    })
}

/// A [`check_body_identity`] pass: the checkpoint matches the constructed body, or
/// predates the stamp and is trusted on first use (the next save stamps it).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BodyIdentity {
    Match,
    TrustOnFirstUse,
}

/// THE body↔policy identity check (bddap/rl#214): a checkpoint stamped with one body
/// digest must never drive or train the body this process actually constructs if the two
/// differ — that policy is not this crab. Pure over (checkpoint stamp, constructed
/// digest) so the matrix is unit-testable; callers pass
/// [`crate::mesh_fallback::constructed_body_digest`] and apply their refusal policy to
/// the `Err` (the trainer aborts, inference refuses to arm).
pub(crate) fn check_body_identity(
    checkpoint: Option<u64>,
    constructed: u64,
) -> Result<BodyIdentity, String> {
    match checkpoint {
        None => Ok(BodyIdentity::TrustOnFirstUse),
        Some(d) if d == constructed => Ok(BodyIdentity::Match),
        Some(d) => Err(format!(
            "checkpoint is stamped body digest {d:#018x} but this process constructs body \
             digest {constructed:#018x} ({}) — the policy was trained on a DIFFERENT crab \
             body (the asset changed, or one side is the procedural fallback; digest 0 = \
             fallback). A policy is only Sally on the body it trained on (bddap/rl#214); \
             restore the matching sally.glb or use a checkpoint trained on this body.",
            if constructed == 0 {
                "the procedural fallback"
            } else {
                "the canonical mesh"
            },
        )),
    }
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
/// path — no save/load drift. `generation` is the set stamp of the checkpoint these
/// moments were saved beside (bddap/rl#215). Best-effort: any failure is logged, not
/// fatal (a resume then loads cold — see [`OPTIMIZER_FILENAME`]).
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn save_optimizer<B: AutodiffBackend>(
    optimizer: &CrabOpt<B>,
    arch: ArchId,
    path: &Path,
    generation: u64,
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
    if let Err(e) = write_envelope(path, ArtifactKind::Optimizer, arch, bytes, None, generation) {
        warn!(
            "Failed to write Adam optimizer state to {}: {e}",
            path.display()
        );
    }
}

/// Load an Adam optimizer state saved by [`save_optimizer`] onto `device`, returning the
/// optimizer with its moments + step restored. REFUSE-AND-COLD is this artifact's whole
/// refusal policy (bddap/rl#200 §2): the `cold` optimizer is returned UNCHANGED — logged,
/// no error — when the file is absent, legacy, corrupt, version-unrecognized, tagged
/// with an arch other than `expected_arch`, or stamped with a generation other than
/// `expected_generation` (both the resumed brain's — the set-coherence checks; a
/// mismatched generation means these moments were saved beside a DIFFERENT brain,
/// bddap/rl#215). Cold moments only cost a brief self-correcting transient; wrong moments
/// would silently miss every `ParamId` lookup. The per-parameter keys line up across an
/// honest round trip because the resumed brain restores the SAME `ParamId`s from its own
/// record.
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn load_optimizer<B: AutodiffBackend>(
    cold: CrabOpt<B>,
    path: &Path,
    device: &B::Device,
    expected_arch: ArchId,
    expected_generation: Option<u64>,
) -> CrabOpt<B> {
    use burn::optim::Optimizer;

    let env = match read_envelope_expecting(
        path,
        ArtifactKind::Optimizer,
        expected_arch,
        expected_generation,
    ) {
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
/// [`ArtifactKind::ReturnNormalizer`] envelope tagged with `arch` and stamped with
/// `generation` — the brain these value scales were trained against, and the save it
/// belongs to (bddap/rl#215). Returns the failure instead of swallowing it: the caller
/// (`save_checkpoint`) aborts the SET on a member failure so a partial save never poses
/// as a complete one.
pub(crate) fn save_return_normalizer(
    norm: &ReturnNormalizer,
    arch: ArchId,
    path: &Path,
    generation: u64,
) -> std::io::Result<()> {
    let bytes = bincode::serialize(&norm.to_data()).map_err(std::io::Error::other)?;
    write_envelope(
        path,
        ArtifactKind::ReturnNormalizer,
        arch,
        bytes,
        None,
        generation,
    )
}

/// Load the return normalizer from a checkpoint, refusing an envelope whose arch tag or
/// generation stamp differ from `expected_arch` / `expected_generation` (both the resumed
/// brain's — normalizers are brain-PAIRED and must never load cross-arch or cross-save,
/// bddap/rl#200 §2, #215). The caller applies the refusal policy: the trainer aborts
/// rather than train warm weights against cold/mis-paired scales.
pub(crate) fn load_return_normalizer(
    path: &Path,
    expected_arch: ArchId,
    expected_generation: Option<u64>,
) -> Result<ReturnNormalizer, EnvelopeError> {
    let env = read_envelope_expecting(
        path,
        ArtifactKind::ReturnNormalizer,
        expected_arch,
        expected_generation,
    )?;
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
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);

        let paths = CheckpointDir::new(&dir);
        save_brain(&brain, &paths.brain_file(), 1).expect("save brain");
        assert!(paths.brain_file().exists(), "brain.bin should exist");

        let loaded = load_brain_file::<TrainBackend>(&paths.brain_file(), &device)
            .expect("load brain via the arch-dispatching loader")
            .brain;
        assert_eq!(loaded.arch(), ArchId::DEFAULT, "arch restored from the tag");

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

    /// A legacy (pre-envelope) brain file is REFUSED with the legacy-naming verdict —
    /// the loader has no untagged-read path, so the old quiet-rest-pose degrade on a raw
    /// record is impossible by construction.
    #[test]
    fn legacy_brain_file_is_refused_not_blind_loaded() {
        let dir = std::env::temp_dir().join("rl_test_brain_legacy_refuse");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;

        // Exactly what a pre-envelope writer produced: the bare leaf record bytes.
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
        let raw = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        let path = CheckpointDir::new(&dir).brain_file();
        std::fs::write(&path, raw).unwrap();

        match load_brain_file::<TrainBackend>(&path, &device) {
            Err(BrainLoadError::Envelope(EnvelopeError::Legacy)) => {}
            Err(other) => panic!("expected Legacy refusal, got {other:?}"),
            Ok(_) => panic!("expected Legacy refusal, got a loaded brain"),
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
        save_return_normalizer(&norm, ArchId::DEFAULT, &path, 3).expect("save return normalizer");
        assert!(
            load_return_normalizer(&path, ArchId::DEFAULT, Some(3)).is_ok(),
            "matching arch + generation loads"
        );
        // A generation stamp from a different save refuses — the bddap/rl#215 mis-pair
        // (e.g. this normalizer landed but the brain write failed, so the dir's brain is
        // an older save's).
        assert!(
            matches!(
                load_return_normalizer(&path, ArchId::DEFAULT, Some(4)),
                Err(EnvelopeError::GenerationMismatch { .. })
            ),
            "a cross-save return normalizer must refuse by generation"
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

    /// One deterministic Adam step on a tiny quadratic target-pulling loss, returning the
    /// updated brain + optimizer. Warms the Adam moments for the round-trip test without
    /// the full PPO machinery. The loss MUST be non-linear in the outputs: gradients then
    /// change as the parameters move, so the warmed moments genuinely alter the next step.
    /// A plain `sum()` loss has constant gradients, and Adam's normalized step under a
    /// constant gradient is `lr·sign(g)` REGARDLESS of moment history — warm and cold
    /// steps come out identical and the round-trip test's divergence probe has nothing to
    /// measure (it only ever passed on trunk-path numerical residue). The 0.5 / +1.0
    /// targets sit away from the init values so every parameter, `log_std` included,
    /// starts with a non-zero gradient.
    fn adam_test_step(
        brain: AnyBrain<TrainBackend>,
        mut optimizer: CrabOpt<TrainBackend>,
        device: &NdArrayDevice,
    ) -> (AnyBrain<TrainBackend>, CrabOpt<TrainBackend>) {
        let obs = Tensor::<TrainBackend, 2>::ones([4, OBS_SIZE], device);
        let (means, log_std) = brain.policy(obs);
        let loss = (means - 0.5).powf_scalar(2.0).sum() + (log_std + 1.0).powf_scalar(2.0).sum();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &brain);
        let brain = optimizer.step(1e-2, brain, grads);
        (brain, optimizer)
    }

    /// Policy means CONCATENATED with the (broadcast) log_std row. The round-trip
    /// assertions probe both: means exercise the whole trunk, while log_std is a free
    /// parameter with a direct constant gradient in [`adam_test_step`]'s loss — so the
    /// warm-vs-cold moment difference registers here even for a deep LayerNorm trunk
    /// whose head-mean shift falls below the comparison tolerance (tanh + three LNs
    /// damp it under 1e-6 at the 3×512 arch).
    fn policy_probe(brain: &AnyBrain<TrainBackend>, device: &NdArrayDevice) -> Vec<f32> {
        let obs = Tensor::<TrainBackend, 2>::zeros([1, OBS_SIZE], device);
        let (means, log_std) = brain.policy(obs);
        let mut probe: Vec<f32> = means.to_data().to_vec().unwrap();
        probe.extend(log_std.to_data().to_vec::<f32>().unwrap());
        probe
    }

    #[test]
    fn adam_optimizer_state_round_trips_through_checkpoint() {
        let dir = std::env::temp_dir().join("rl_test_adam_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;

        // Warm an optimizer over several steps so its Adam moments + step are non-trivial,
        // then snapshot brain + optimizer mid-run.
        let mut brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
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
        save_optimizer(&warm, ArchId::DEFAULT, &path, 5);
        assert!(path.exists(), "optimizer.bin should be written");

        // A generation from a different save refuses to cold (bddap/rl#215): these
        // moments belong to a different brain than the one resuming.
        let cross_save = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::DEFAULT,
            Some(6),
        );
        assert!(
            cross_save.to_record().is_empty(),
            "a cross-save optimizer must load cold"
        );

        // A fresh cold optimizer loaded from the snapshot must take the NEXT step
        // identically to the warm one (same momentum + step/bias-correction), and
        // DIFFERENTLY from a truly cold optimizer — that difference is exactly the
        // self-correcting transient a warm resume avoids.
        let restored = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::DEFAULT,
            Some(5),
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

        let warm_m = policy_probe(&warm_next, &device);
        let restored_m = policy_probe(&restored_next, &device);
        let cold_m = policy_probe(&cold_next, &device);

        for (i, (a, b)) in warm_m.iter().zip(restored_m.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "restored step diverged from warm at probe[{i}]: {a} vs {b}"
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
            ArchId::DEFAULT,
            None,
        );
        assert!(
            cold.to_record().is_empty(),
            "an absent optimizer file must leave the optimizer cold"
        );

        // (b) A legacy pre-envelope file (the old `OptimizerCheckpoint` bincode) has no
        //     magic → refused, cold, no panic. Its only parser was the migration tool,
        //     deleted with the rl#200 fleet migration.
        std::fs::write(&path, bincode::serialize(&(1u32, vec![0u8; 4])).unwrap()).unwrap();
        let cold2 = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::DEFAULT,
            None,
        );
        assert!(
            cold2.to_record().is_empty(),
            "a legacy optimizer file must leave the optimizer cold"
        );

        // (c) A mis-copied file — a BRAIN envelope at the optimizer path — fails the kind
        //     check → cold, never decoded as moments.
        write_envelope(
            &path,
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            vec![1, 2, 3],
            Some(0),
            9,
        )
        .unwrap();
        let cold3 = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::DEFAULT,
            Some(9),
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
            ArchId::DEFAULT,
            vec![0xde, 0xad, 0xbe, 0xef],
            None,
            9,
        )
        .unwrap();
        let cold4 = load_optimizer(
            crab_optimizer::<TrainBackend>(),
            &path,
            &device,
            ArchId::DEFAULT,
            Some(9),
        );
        assert!(
            cold4.to_record().is_empty(),
            "a corrupt optimizer payload must leave the optimizer cold, not panic"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bddap/rl#214 body↔policy matrix: pre-stamp checkpoints are trusted on first
    /// use (never invalidated), a matching stamp passes, and a mismatched stamp is the
    /// loud refusal — in BOTH directions (Sally checkpoint on a fallback body, and a
    /// fallback-trained checkpoint on Sally).
    #[test]
    fn body_identity_matrix() {
        assert_eq!(
            check_body_identity(None, 0xabc),
            Ok(BodyIdentity::TrustOnFirstUse)
        );
        assert_eq!(
            check_body_identity(Some(0xabc), 0xabc),
            Ok(BodyIdentity::Match)
        );
        assert_eq!(check_body_identity(Some(0), 0), Ok(BodyIdentity::Match));
        let err = check_body_identity(Some(0xabc), 0).unwrap_err();
        assert!(err.contains("DIFFERENT crab body"), "{err}");
        assert!(check_body_identity(Some(0), 0xabc).is_err());
    }

    /// `save_brain` stamps the CONSTRUCTED body digest and `load_brain_file` hands it
    /// back — the stamp round-trips and, being read from the process-global verdict on
    /// both sides, always passes [`check_body_identity`] for a same-process round trip.
    #[test]
    fn save_brain_stamps_constructed_body_digest() {
        let dir = std::env::temp_dir().join("rl_test_body_stamp");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;

        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
        let path = CheckpointDir::new(&dir).brain_file();
        save_brain(&brain, &path, 11).unwrap();

        let loaded = load_brain_file::<TrainBackend>(&path, &device).unwrap();
        let constructed = crate::mesh_fallback::constructed_body_digest();
        assert_eq!(loaded.body_digest, Some(constructed));
        assert_eq!(loaded.generation, Some(11), "the set stamp round-trips");
        assert_eq!(
            check_body_identity(loaded.body_digest, constructed),
            Ok(BodyIdentity::Match)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
