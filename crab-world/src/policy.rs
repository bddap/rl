//! The trained policy: load a checkpoint's brain + obs-normalizer and run deterministic
//! inference (no learning). One implementation, three callers — the demo's `policy_step`
//! (`play::policy`), the game's solo NN-crab (`net::external_crab`), and the headless
//! training-success eval ([`crate::eval`]) — all drive [`Policy::act`].
//!
//! Lives at the crate root (not under the render-gated `play` module) because it is PURE
//! inference: burn tensors + a checkpoint reader, no bevy render types. Keeping it headless
//! is what lets the trainer-side eval reuse the SAME deterministic policy the rendered demo
//! runs, instead of a second copy that could drift (`play` re-exports it for the renderers).

use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::Tensor;

use crate::bot::actuator::ACTION_SIZE;
use crate::bot::arch::{AnyBrain, ArchId};
use crate::bot::sensor::OBS_SIZE;
use crate::training::InferBackend;
use crate::training::checkpoint::{BrainLoadError, CheckpointDir, load_brain_file};
use crate::training::envelope::EnvelopeError;
use crate::training::normalizer::{NORMALIZER_CLIP, ObsNormalizer};

/// A policy that maps observations to actions for inference (no learning). The
/// checkpoint-derived state — whether a brain drives the crab, and its weight-identity
/// digest — lives in [`PolicyState`], which makes the illegal combinations
/// unrepresentable (no `loaded` bool to keep in sync with the brain, no `0`-digest sentinel
/// that could ride alongside a real brain into the lockstep hash). The fields here are the
/// state-independent wiring: the inference device and the demo's hot-reload bookkeeping.
///
/// Non-send because the `ndarray` backend's tensors are not `Sync` (same reason
/// as `TrainingState`).
pub struct Policy {
    device: NdArrayDevice,
    /// Live training checkpoint dir the demo hot-reloads from while running (None
    /// disables). `last_loaded` is the mtime of the brain file last swapped in, so
    /// we reload only when training has written a newer one. See [`Self::try_hot_reload`].
    // Read only by the render-gated demo hot-reload; the headless eval/trainer never swaps
    // checkpoints mid-run, so these are legitimately unused there.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    live_dir: Option<PathBuf>,
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    last_loaded: Option<std::time::SystemTime>,
    /// Brain-file mtime whose REFUSAL was already logged, so a persistently-bad
    /// checkpoint logs once per distinct file while still being retried each poll —
    /// unlike `last_loaded`, a refusal never suppresses the retry itself, because it may
    /// be the transient mid-save window (see [`Self::try_hot_reload`], bddap/rl#215).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    last_refused: Option<std::time::SystemTime>,
    state: PolicyState,
}

/// What (if anything) drives the crab, and its weight-identity digest — the three states
/// a policy can be in, kept as ONE enum so an impossible combination (a brain with no digest
/// claiming to be loaded, or a "loaded" flag with no brain, or a `0`/stale digest beside a
/// real brain) can't be constructed.
// `Loaded`/`Diagnostic` carry the ~6KB brain inline while `Rest` is tiny; a `Policy` is a
// long-lived resource constructed once (or swapped on hot-reload), never passed by value on a
// hot path, so boxing the brain would only add an indirection to every `act` call.
#[allow(clippy::large_enum_variant)]
enum PolicyState {
    /// No brain drives the crab — [`Policy::act`] returns the zero-action rest pose (a
    /// neutral, deterministic view of the body geometry, not an untrained brain's noise).
    /// Two ways here, distinguished at load time by their logs (rl#121): a LOUDLY refused
    /// checkpoint (wrong-rig [`log_rig_mismatch`], or envelope-refused
    /// [`log_checkpoint_refusal`]) and the quiet, legitimate no-checkpoint-yet pose.
    Rest,
    /// `RL_RANDOM_POLICY`: an untrained current-rig brain drives the crab so an operator can
    /// see what a FRESH policy does (vs the zero-action rest pose). It has NO on-disk weights,
    /// hence no digest, hence — by construction, not by a runtime `!= 0` check — can never
    /// arm a networked round (`weights_digest()` is `None`).
    Diagnostic {
        brain: AnyBrain<InferBackend>,
        normalizer: ObsNormalizer,
    },
    /// A real checkpoint's brain drives the crab. `digest` is a stable hash of the checkpoint
    /// bytes (brain + normalizer): two peers running the SAME weights get the same digest,
    /// different weights get different ones. `NonZeroU64` because the GCR bridge folds it into
    /// the crab's per-tick lockstep hash — a `0` there used to mean "no real brain", so a
    /// zero/stale digest riding alongside a loaded brain was a silent-desync trap; the type now
    /// forbids it.
    Loaded {
        brain: AnyBrain<InferBackend>,
        normalizer: ObsNormalizer,
        digest: NonZeroU64,
    },
}

/// Digest of a checkpoint's on-disk weights (brain + normalizer bytes), or `0` if the brain
/// file is unreadable. The brain-identity check: identical files → identical digest — the
/// HOST's non-zero digest is what arms a networked round (`sync_verdict`), and telemetry
/// compares it across decks. Reads the raw bytes rather than the deserialized tensors so it needs no backend
/// and can't drift from how the weights are stored.
pub fn checkpoint_digest(dir: &Path) -> u64 {
    let paths = CheckpointDir::new(dir);
    let Ok(mut bytes) = std::fs::read(paths.brain_file()) else {
        return 0;
    };
    if let Ok(norm) = std::fs::read(paths.normalizer_path()) {
        bytes.extend_from_slice(&norm);
    }
    crate::fnv::fnv1a(&bytes)
}

/// Whether a brain's `(obs, action)` dims drive THIS binary's compiled crab rig. The ONE
/// place the fits-the-rig rule is spelled — the runtime loader (soft-fallback), the release
/// gate, the game's launch gate (both hard-fail), and the trainer's warm-start check
/// (`training::systems::state`) all ask through here, so they can't disagree on what
/// "fits" means.
pub(crate) fn dims_fit_rig(obs: usize, action: usize) -> bool {
    (obs, action) == (OBS_SIZE, ACTION_SIZE)
}

/// Whether the checkpoint at `dir` fits this binary's crab rig — the verdict the
/// release/deploy gate and the game's launch gate act on, produced by the SAME classifier
/// the runtime loader uses ([`load_brain_normalizer`]), so the gates can never pass a
/// checkpoint the runtime would refuse. The caller only turns the verdict into a message +
/// exit code. A dim-mismatched or envelope-refused checkpoint would degrade the NN crab to
/// its motionless rest pose, so the gates refuse it outright; the rig side of `Mismatch`
/// is `crate::play::rig_dims` (render-gated, so not linked here).
pub fn checkpoint_fits_rig(dir: &Path) -> RigFit {
    match load_brain_normalizer(dir, &NdArrayDevice::Cpu) {
        Loaded::Fit(..) => RigFit::Ok,
        Loaded::Absent => RigFit::Missing,
        Loaded::Mismatch(dims) => RigFit::Mismatch(dims),
        Loaded::Refused(why) => RigFit::Refused(why),
    }
}

/// A brain's `(obs, action)` input/output dimensions — the pair a checkpoint is checked
/// against the rig with. A named pair, not a bare `(usize, usize)`, so the two same-typed
/// fields can't be swapped at a call site or a struct field; the whole rig-fit machinery
/// (the [`RigFit`] verdict, the runtime loader's refusal)
/// speaks this one type rather than re-spelling the pair each way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RigDims {
    pub obs: usize,
    pub action: usize,
}

/// A checkpoint's fit against this binary's compiled crab rig (see [`checkpoint_fits_rig`]).
pub enum RigFit {
    /// The brain's dims match the rig — safe to drive the NN crab.
    Ok,
    /// No `brain.bin` in the dir — nothing to check.
    Missing,
    /// The checkpoint was refused before any dims existed to compare, with the reason
    /// preformatted: a truncated/corrupt file (distinct from `Missing` so the operator
    /// redeploys the file they can SEE instead of chasing path resolution), a legacy
    /// unmigrated file, an unregistered architecture named in the reason (the bddap/rl#200
    /// §2 arch arm), a mis-copied artifact, or a missing/mis-paired obs normalizer.
    Refused(String),
    /// The brain loads but its dims (carried here) differ from the rig.
    Mismatch(RigDims),
}

/// The result of trying to read a checkpoint's brain + normalizer for THIS binary's rig.
/// Three outcomes the caller must handle differently — the whole point of #121 is that
/// `Absent` and `Mismatch` are NOT the same thing and must not collapse to one silent
/// rest pose. The fit verdict comes from the shared [`dims_fit_rig`] predicate, so this
/// loader and the release gate can't disagree on what "fits" means.
// `Fit` carries the ~6KB brain inline while the other variants are tiny; this value is a
// transient, constructed and immediately destructured at its one call site (the brain moves
// straight into `Policy`), so boxing it would only add a heap alloc on the cold load path.
#[allow(clippy::large_enum_variant)]
enum Loaded {
    /// Brain + its paired normalizer parsed and the dims fit the rig — arm the NN crab.
    Fit(AnyBrain<InferBackend>, ObsNormalizer),
    /// No `brain.bin`. The LEGITIMATE "no brain yet" case — keep the current policy /
    /// hold the neutral rest pose. Not an error.
    Absent,
    /// The brain parsed but its dims don't fit the compiled rig (carried here so the caller
    /// can name both sides). Arming it would degrade the crab to an inert rest pose that
    /// looks frozen-but-fine, or panic in the first matmul (rl#36) — so the caller LOUDLY
    /// refuses to arm. An operator error (wrong checkpoint for this build), never a transient.
    Mismatch(RigDims),
    /// The envelope refused the checkpoint — corrupt/truncated, legacy (unmigrated), an
    /// unregistered arch, a mis-copied file, or a missing/mis-paired normalizer. Like
    /// `Mismatch`, an operator error refused LOUDLY with the reason; what used to degrade
    /// to the quiet rest pose misattributed as "no checkpoint" (bddap/rl#200 §2's
    /// silent-fallback class) is now this distinct verdict.
    Refused(String),
}

/// Read a brain + normalizer from `dir`, classifying the result for the caller — THE one
/// checkpoint classifier: the runtime loaders (initial + hot-reload) and the gates
/// ([`checkpoint_fits_rig`]) all speak this verdict, so they can't drift. The brain load
/// dispatches on the envelope's arch tag ([`load_brain_file`]); the normalizer is
/// brain-PAIRED — it must exist and carry the brain's own arch tag, because a real brain
/// normalizing against cold or mis-paired stats acts silently wrong, a worse failure
/// than not arming.
fn load_brain_normalizer(dir: &Path, device: &NdArrayDevice) -> Loaded {
    let paths = CheckpointDir::new(dir);
    let loaded = match load_brain_file::<InferBackend>(&paths.brain_file(), device) {
        Ok(loaded) => loaded,
        Err(BrainLoadError::Envelope(EnvelopeError::Absent)) => return Loaded::Absent,
        Err(e) => {
            return Loaded::Refused(format!(
                "{}: {e}",
                crate::training::checkpoint::BRAIN_FILENAME
            ));
        }
    };
    // Body↔policy identity (bddap/rl#214): a policy trained on a different body than the
    // one this process constructs is not this crab — refuse to arm it (same class as the
    // rig-dims mismatch below), never drive the wrong body "mostly fine". A pre-stamp
    // checkpoint passes on trust (read-only surfaces don't stamp; the trainer's next
    // save migrates it).
    if let Err(why) = crate::training::checkpoint::check_body_identity(
        loaded.body_digest,
        crate::mesh_fallback::constructed_body_digest(),
    ) {
        return Loaded::Refused(format!(
            "{}: {why}",
            crate::training::checkpoint::BRAIN_FILENAME
        ));
    }
    let key = loaded.set_key();
    let brain = loaded.brain;
    // A checkpoint from a different rig (e.g. a stale 77-dim brain against the current
    // OBS_SIZE) parses fine here but its mismatched first-layer weight would panic in the
    // matmul at the first `policy()` call. Surface it as a distinct `Mismatch` (carrying
    // the checkpoint's own dims) — the caller refuses to arm AND logs loudly, so a stale
    // `checkpoints/` is an obvious, attributable error rather than a silent statue (rl#36,
    // rl#121).
    let (obs, action) = brain.io_dims();
    if !dims_fit_rig(obs, action) {
        return Loaded::Mismatch(RigDims { obs, action });
    }
    // The set key (bddap/rl#215) keys the pairing to the brain's own SAVE, not
    // just its arch: a partial save (or a copy torn across one) leaves one member from an
    // older set, which would normalize this brain with another save's statistics —
    // refused here like any other mis-pair.
    match ObsNormalizer::load(&paths.normalizer_path(), key) {
        Ok(normalizer) => Loaded::Fit(brain, normalizer),
        Err(e) => Loaded::Refused(format!(
            "{}: {e} (a brain never arms without its paired obs normalizer)",
            crate::training::checkpoint::NORMALIZER_FILENAME
        )),
    }
}

/// The loud, actionable refusal logged when a checkpoint's dims don't fit this binary's
/// rig. One message for both arming sites (initial load + hot-reload) so they stay
/// consistent: names the surface, the path, and BOTH dim pairs, and states the
/// consequence (the crab will NOT arm). Distinct from the quiet `Absent` rest pose.
fn log_rig_mismatch(surface: &str, dir: &Path, dims: RigDims) {
    let RigDims { obs, action } = dims;
    error!(
        "{surface}: checkpoint at {} was built for a DIFFERENT rig — its brain wants \
         {obs} obs / {action} act but this binary's crab rig is {OBS_SIZE} obs / \
         {ACTION_SIZE} act. REFUSING to arm the NN crab: it would hold an inert rest \
         pose that looks frozen-but-fine. Rebuild the checkpoint for this rig, or run a \
         binary whose rig matches the checkpoint.",
        dir.display(),
    );
}

/// The loud refusal for an envelope-level rejection (corrupt/legacy/unknown-arch/
/// mis-paired) — `log_rig_mismatch`'s sibling, one message for both arming sites.
fn log_checkpoint_refusal(surface: &str, dir: &Path, why: &str) {
    error!(
        "{surface}: REFUSING checkpoint at {} — {why}. The NN crab will NOT arm (it \
         would hold an inert rest pose that looks frozen-but-fine).",
        dir.display(),
    );
}

/// Build the armed state for a brain read off disk, from the checkpoint's byte digest. A real
/// checkpoint's `NonZeroU64` digest arms [`PolicyState::Loaded`] (may enter lockstep); a `0`
/// digest — a degenerate hash, or a brain with no readable file behind it — falls to
/// [`PolicyState::Diagnostic`] instead, so a value that can't distinguish peers never rides
/// into the lockstep hash pretending to. Shared by the initial load and the hot-reload so the
/// two can't disagree on how a digest becomes a state.
fn loaded_state(
    brain: AnyBrain<InferBackend>,
    normalizer: ObsNormalizer,
    digest: u64,
) -> PolicyState {
    match NonZeroU64::new(digest) {
        Some(digest) => PolicyState::Loaded {
            brain,
            normalizer,
            digest,
        },
        None => PolicyState::Diagnostic { brain, normalizer },
    }
}

impl Policy {
    /// Load brain + normalizer from a checkpoint dir. A missing checkpoint falls back
    /// QUIETLY to the zero-action rest pose so the app still launches (useful before the
    /// first checkpoint exists, and to inspect the body's neutral pose); a present-but-
    /// unusable one (wrong rig, or envelope-refused — corrupt/legacy/wrong-arch) is
    /// refused LOUDLY and also rests.
    pub fn load(checkpoint_dir: &Path) -> Self {
        let device = NdArrayDevice::Cpu;
        // RL_RANDOM_POLICY drives the crab with an untrained random-init brain when there is
        // NO usable checkpoint, to see what a FRESH policy does (vs the zero-action rest pose)
        // — distinguishes a learned behaviour from one the dynamics produce on their own. It
        // never overrides a real checkpoint (a Fit always wins).
        let random_override = std::env::var("RL_RANDOM_POLICY").is_ok_and(|v| v == "1");
        let loaded = load_brain_normalizer(checkpoint_dir, &device);
        // The loud refusals fire FIRST, unconditionally — the diagnostic override below
        // must not swallow the reason (a corrupt/legacy/wrong-rig checkpoint is exactly
        // what an operator debugging with the random brain needs to see).
        match &loaded {
            Loaded::Mismatch(dims) => log_rig_mismatch("play", checkpoint_dir, *dims),
            Loaded::Refused(why) => log_checkpoint_refusal("play", checkpoint_dir, why),
            Loaded::Fit(..) | Loaded::Absent => {}
        }
        let state = match loaded {
            Loaded::Fit(brain, normalizer) => {
                info!("play: loaded checkpoint from {}", checkpoint_dir.display());
                loaded_state(brain, normalizer, checkpoint_digest(checkpoint_dir))
            }
            // No usable checkpoint, but the diagnostic override is on: drive with an untrained
            // current-rig brain. No on-disk weights ⇒ no digest ⇒ can't enter lockstep.
            Loaded::Absent | Loaded::Mismatch(_) | Loaded::Refused(_) if random_override => {
                warn!(
                    "play: RL_RANDOM_POLICY — driving with an untrained random brain \
                     (no usable checkpoint at {})",
                    checkpoint_dir.display()
                );
                PolicyState::Diagnostic {
                    brain: AnyBrain::<InferBackend>::init(ArchId::DEFAULT, &device),
                    normalizer: ObsNormalizer::new(NORMALIZER_CLIP),
                }
            }
            Loaded::Absent => {
                // The legitimate "no brain yet" rest pose — quiet, expected before the
                // first checkpoint exists or when inspecting the body's neutral pose.
                warn!(
                    "play: no usable checkpoint at {} — using zero-action pose",
                    checkpoint_dir.display()
                );
                PolicyState::Rest
            }
            // Refusal already logged above; the state is the (attributed) rest pose.
            Loaded::Mismatch(_) | Loaded::Refused(_) => PolicyState::Rest,
        };

        Self {
            device,
            live_dir: None,
            last_loaded: None,
            last_refused: None,
            state,
        }
    }

    /// Point this policy at a live training checkpoint dir to hot-reload from (see
    /// [`Self::try_hot_reload`]); `None` disables. The demo's inference wiring sets it after
    /// [`Self::load`]; the eval / networked crab leave it unset (they never hot-reload).
    // Demo-only wiring (render-gated); headless callers never set a live dir.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn set_live_dir(&mut self, dir: Option<PathBuf>) {
        self.live_dir = dir;
    }

    /// If the live training dir holds a brain file newer than the one we're
    /// running, swap it in; returns whether it did. Safe against a bad file appearing
    /// mid-run: any non-`Fit` verdict keeps the current policy rather than blanking the
    /// demo to a rest pose. Each FILE is written atomically, so a torn read of one can't
    /// occur — but the SET is not: the trainer renames `brain.bin` before the
    /// normalizers, so a poll landing in that window sees a set-torn dir (a save-stamp
    /// refusal, bddap/rl#215) whose trailing members arrive without touching the brain's
    /// mtime. A `Refused` verdict is therefore treated as possibly TRANSIENT — retried
    /// next poll — never cached against the mtime like `Fit`/`Mismatch` are.
    // Demo-only (render-gated) hot-reload; the headless eval/trainer never swaps mid-run.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn try_hot_reload(&mut self) -> bool {
        let Some(dir) = self.live_dir.clone() else {
            return false;
        };
        let brain_bin = CheckpointDir::new(&dir).brain_file();
        let Ok(mtime) = std::fs::metadata(&brain_bin).and_then(|m| m.modified()) else {
            return false; // no live brain file yet
        };
        if self.last_loaded == Some(mtime) {
            return false; // already handled this checkpoint (armed it, or refused it once)
        }
        match load_brain_normalizer(&dir, &self.device) {
            Loaded::Fit(brain, normalizer) => {
                self.state = loaded_state(brain, normalizer, checkpoint_digest(&dir));
                self.last_loaded = Some(mtime);
                true
            }
            Loaded::Absent => false, // no brain file yet — keep the current policy
            // A wrong-rig brain landed in the live dir: refuse it loudly but keep
            // whatever we're driving (a bad file must not blank a working demo; an
            // unarmed policy just stays at rest). A rig mismatch is a property of the
            // brain FILE itself, so stamp `last_loaded` — log once per distinct file
            // (mtime), no pointless re-reads.
            Loaded::Mismatch(dims) => {
                log_rig_mismatch("play (hot-reload)", &dir, dims);
                self.last_loaded = Some(mtime);
                false
            }
            // Same keep-driving policy, but do NOT stamp `last_loaded`: the refusal may
            // be the transient mid-save window above, and the completed set won't change
            // the brain's mtime — stamping here would refuse a run's FINAL save forever.
            // A genuinely bad checkpoint just gets re-read (and re-refused) each poll;
            // `last_refused` keeps the log at once per distinct file.
            Loaded::Refused(why) => {
                if self.last_refused != Some(mtime) {
                    log_checkpoint_refusal("play (hot-reload)", &dir, &why);
                    self.last_refused = Some(mtime);
                }
                false
            }
        }
    }

    /// Whether a brain drives the crab (vs the zero-action rest-pose fallback). Lets a
    /// caller fail loud when the body will only hold its rest pose ([`Self::act`] returns the
    /// neutral pose while this is false). True for both a real checkpoint and the
    /// `RL_RANDOM_POLICY` diagnostic brain — use [`Self::weights_digest`] to tell those apart.
    pub fn is_loaded(&self) -> bool {
        !matches!(self.state, PolicyState::Rest)
    }

    /// Stable digest of the loaded weights, or `None` when no real checkpoint is armed (the
    /// rest pose OR the `RL_RANDOM_POLICY` diagnostic brain). The GCR bridge folds this into
    /// each crab's per-tick physics digest on the peer that PUMPS the physics — under
    /// host-auth the host (rl#151); clients adopt its snapshots. `None` — not a `0`
    /// sentinel — is "no real brain": it XORs as a no-op, so a rest-pose/diagnostic body
    /// contributes no fake weight identity to `state_hash`. (The old formation/admission
    /// weights gates this fed are deleted — rl#200 increment 6; launch validation is the
    /// guard now.)
    pub fn weights_digest(&self) -> Option<NonZeroU64> {
        match &self.state {
            PolicyState::Loaded { digest, .. } => Some(*digest),
            PolicyState::Rest | PolicyState::Diagnostic { .. } => None,
        }
    }

    /// Deterministic action: the policy mean (no exploration noise), so the crab
    /// holds a steady pose instead of jittering. One policy implementation, three
    /// callers — the demo, the game's solo NN-crab, and the headless eval.
    ///
    /// Returns the RAW mean — unclamped like every `CrabActions` writer (the trainer
    /// writes unclamped μ+σ·ε too; the actuator's `apply_actions` is the sole ±1
    /// torque-bound) and, on this path, un-sanitized: a non-finite mean flows through
    /// to the actuator's latched `error!` (rl#145). Sanitizing here zeroed a
    /// NaN-spewing checkpoint upstream of that guard, degrading it to rest pose
    /// silently (rl#219). (The trainer alone pre-zeroes non-finite drive, with a
    /// `warn!` — a NaN there would poison the PPO buffer, not just one tick's torque.)
    pub fn act(&self, raw_obs: &[f32; OBS_SIZE]) -> [f32; ACTION_SIZE] {
        // No brain → hold the neutral (zero-action) pose: a deterministic view of the body
        // geometry, not an untrained brain's noise. Both driving states run the same inference.
        let (brain, normalizer) = match &self.state {
            PolicyState::Loaded {
                brain, normalizer, ..
            }
            | PolicyState::Diagnostic { brain, normalizer } => (brain, normalizer),
            PolicyState::Rest => return [0.0; ACTION_SIZE],
        };
        let obs = normalizer.normalize_frozen(raw_obs);
        let input =
            Tensor::<InferBackend, 1>::from_floats(obs.as_slice(), &self.device).unsqueeze();
        // Eval/demo is deterministic: it takes the policy MEAN and discards `log_std`
        // entirely — no `GaussianHead` is built, so the exploration-σ floor never
        // reaches a deployed action by construction.
        let (means, _log_std) = brain.policy(input);
        let flat: Vec<f32> = means.flatten::<1>(0, 1).to_data().to_vec().unwrap();

        flat.try_into()
            .expect("policy mean count == ACTION_SIZE (the rig gates refuse mismatched brains)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

    use crate::bot::arch::Mlp512x3;
    use crate::training::TrainBackend;
    use crate::training::envelope::{ArtifactKind, write_envelope};

    /// Save a freshly-initialised brain + its paired identity normalizer into `dir`
    /// through the PRODUCTION writers ([`save_brain`](crate::training::checkpoint::save_brain)
    /// / [`ObsNormalizer::save`], the same recipes `save_checkpoint` uses) — so every
    /// save→`Policy::load` test here is an end-to-end writer↔loader format guard,
    /// not a test-only round trip that would stay green through a writer drift.
    fn save_brain(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let paths = CheckpointDir::new(dir);
        crate::training::checkpoint::save_brain(&brain, &paths.brain_file(), 21).unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(brain.arch(), &paths.normalizer_path(), 21)
            .unwrap();
    }

    /// The fixture obs the golden `actions.hex` was generated over: deterministic,
    /// integer-derived (no libm), shared by the golden tests.
    fn golden_obs() -> [f32; OBS_SIZE] {
        let mut obs = [0.0f32; OBS_SIZE];
        for (i, o) in obs.iter_mut().enumerate() {
            *o = ((i * 37 % 101) as f32) / 50.5 - 1.0;
        }
        obs
    }

    /// The golden fixture dir: the enveloped golden brain plus `actions.hex`, its
    /// pinned action bits over [`golden_obs`].
    fn golden_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden-mlp512x3-env")
    }

    fn golden_action_bits() -> Vec<u32> {
        std::fs::read_to_string(golden_dir().join("actions.hex"))
            .expect("read golden actions.hex")
            .lines()
            .map(|l| u32::from_str_radix(l.trim(), 16).expect("parse golden bits"))
            .collect()
    }

    /// GOLDEN FILE (bddap/rl#200 increment 2): `tests/data/golden-mlp512x3-env/` holds
    /// an ENVELOPED golden brain (v1/TOFU shape, plus an identity obs normalizer) with
    /// its action bits over [`golden_obs`] pinned in `actions.hex` at generation time.
    /// Loading it through today's loader and reproducing those bits EXACTLY proves the
    /// tagged on-disk format did not drift (a reader/writer change that re-mapped
    /// envelope fields would strand every fleet checkpoint while all save/load-symmetric
    /// tests stayed green). Regenerate with `regenerate_enveloped_golden_fixture` below
    /// ONLY on a deliberate format version bump or an arch cull.
    #[test]
    fn golden_enveloped_checkpoint_loads_and_acts_bit_identically() {
        let policy = Policy::load(&golden_dir());
        assert!(
            policy.is_loaded(),
            "the enveloped golden checkpoint no longer loads — the tagged on-disk format \
             drifted (every migrated fleet checkpoint would refuse to arm)"
        );

        let expected = golden_action_bits();
        assert_eq!(
            expected.len(),
            ACTION_SIZE,
            "golden fixture is for this rig"
        );
        let act = policy.act(&golden_obs());
        let got: Vec<u32> = act.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            got, expected,
            "actions on the fixture obs are not bit-identical to the pinned golden bits"
        );
    }

    /// A legacy (pre-envelope) checkpoint must be REFUSED — a loud `Refused` verdict
    /// naming the legacy diagnosis, never the quiet rest pose or a blind load
    /// (bddap/rl#200 §2: the loader has no untagged-read path; the sole legacy parser,
    /// `migrate-checkpoint`, was deleted with the fleet migration). The legacy file is
    /// synthesized — exactly what a pre-envelope writer produced was the bare leaf
    /// record bytes, and the classifier is a magic-prefix check, so real historical
    /// bytes would add nothing.
    #[test]
    fn legacy_checkpoint_is_refused() {
        let dir = std::env::temp_dir().join(format!("rl-legacy-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
        let raw = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        std::fs::write(CheckpointDir::new(&dir).brain_file(), raw).unwrap();

        let policy = Policy::load(&dir);
        assert!(
            !policy.is_loaded(),
            "a legacy untagged brain.bin must not load — the loader has no untagged path"
        );
        // The non-arming is graceful: the rest pose, never a panic.
        assert_eq!(policy.act(&golden_obs()), [0.0; ACTION_SIZE]);

        // The gates speak the same classifier: a loud, attributable refusal naming the
        // diagnosis (pre-envelope, i.e. predates the rl#200 fleet migration).
        match checkpoint_fits_rig(&dir) {
            RigFit::Refused(why) => assert!(
                why.contains("pre-envelope"),
                "the refusal must name the legacy (pre-envelope) diagnosis, got: {why}"
            ),
            _ => panic!("a legacy checkpoint must classify as Refused, not Missing/Ok"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regenerates the golden fixture from a FRESH [`ArchId::DEFAULT`] brain — writes
    /// the enveloped brain + identity normalizer, then pins its action bits over
    /// [`golden_obs`] into `actions.hex` — then commit the result (run with
    /// `cargo test -- --ignored regenerate_enveloped_golden_fixture`). Only for a
    /// DELIBERATE format change or an arch cull; ignored because a golden fixture that
    /// silently regenerates guards nothing. Writes the LEGACY v1 shape — for BOTH
    /// members — NOT the production writers: the production writer stamps the
    /// regenerating machine's constructed body digest (bddap/rl#214) plus a save
    /// save_stamp (bddap/rl#215); the digest would make the committed fixture refuse to
    /// arm on every machine whose `sally.glb` differs or is absent, and a stamped
    /// normalizer refuses to pair with the unstamped v1 brain. The fixture must stay
    /// wholly v1 (trust-on-first-use) to remain machine-portable, which also keeps it
    /// the end-to-end guard of the fleet's v1-resume path.
    #[test]
    #[ignore = "fixture generator, run manually on a deliberate format change or arch cull"]
    fn regenerate_enveloped_golden_fixture() {
        let dir = golden_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let paths = CheckpointDir::new(&dir);
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let bytes = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        crate::training::envelope::write_v1_envelope(
            &paths.brain_file(),
            ArtifactKind::Brain,
            brain.arch(),
            bytes,
        )
        .unwrap();
        crate::training::envelope::write_v1_envelope(
            &paths.normalizer_path(),
            ArtifactKind::ObsNormalizer,
            brain.arch(),
            bincode::serialize(&ObsNormalizer::new(NORMALIZER_CLIP).snapshot()).unwrap(),
        )
        .unwrap();

        let policy = Policy::load(&dir);
        assert!(policy.is_loaded(), "the fixture just written must load");
        let bits: Vec<String> = policy
            .act(&golden_obs())
            .iter()
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect();
        std::fs::write(dir.join("actions.hex"), bits.join("\n") + "\n").unwrap();
    }

    /// bddap/rl#214 WIRING guard (the pure `check_body_identity` matrix is unit-tested
    /// beside it): a checkpoint stamped with a different body digest than this process
    /// constructs must refuse to arm through the one classifier — deleting or
    /// reordering the identity check in `load_brain_normalizer` turns this red.
    #[test]
    fn wrong_body_digest_checkpoint_refuses_to_arm() {
        let dir = std::env::temp_dir().join(format!("rl-bodydigest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;

        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let bytes = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        let paths = CheckpointDir::new(&dir);
        // Differs from the constructed digest whatever the test env's body is.
        let wrong = crate::mesh_fallback::constructed_body_digest() ^ 0xdead_beef;
        write_envelope(
            &paths.brain_file(),
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            bytes,
            Some(wrong),
            21,
        )
        .unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(ArchId::DEFAULT, &paths.normalizer_path(), 21)
            .unwrap();

        let policy = Policy::load(&dir);
        assert!(!policy.is_loaded(), "a wrong-body checkpoint must not arm");
        match checkpoint_fits_rig(&dir) {
            RigFit::Refused(why) => assert!(
                why.contains("DIFFERENT crab body"),
                "the refusal must name the body mismatch, got: {why}"
            ),
            _ => panic!("a wrong-body checkpoint must classify as Refused"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The demo's "always fresh" guarantee: when training writes a new checkpoint
    /// into the live dir, the running policy swaps it in (flipping to `loaded`),
    /// and it does NOT reload the same file twice. Also pins the safe no-ops: no
    /// `live_dir`, and a live dir with no brain yet, both leave the policy alone.
    #[test]
    fn hot_reload_swaps_in_a_new_checkpoint() {
        let tmp = std::env::temp_dir();
        let live = tmp.join(format!("rl-hotreload-live-{}", std::process::id()));
        let empty = tmp.join(format!("rl-hotreload-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
        std::fs::create_dir_all(&empty).unwrap();

        // No checkpoint anywhere → unloaded (holds the zero-action rest pose).
        let mut policy = Policy::load(&empty);
        assert!(
            !policy.is_loaded(),
            "empty checkpoint dir should give an unloaded policy"
        );
        assert_eq!(
            policy.weights_digest(),
            None,
            "an unloaded policy has no weight digest (no `0` sentinel)"
        );
        assert!(
            !policy.try_hot_reload(),
            "no live_dir set → nothing to reload"
        );

        // Point at a live dir that has no brain yet → still a no-op.
        policy.live_dir = Some(live.clone());
        assert!(
            !policy.try_hot_reload(),
            "live dir without a brain → no reload"
        );

        // Training writes a checkpoint → the policy picks it up exactly once.
        save_brain(&live);
        assert!(
            policy.try_hot_reload(),
            "a new brain in the live dir must reload"
        );
        assert!(
            policy.is_loaded(),
            "a successful hot-reload marks the policy loaded"
        );
        assert!(
            policy.weights_digest().is_some(),
            "a hot-reloaded real checkpoint carries a nonzero weight digest"
        );
        assert!(
            !policy.try_hot_reload(),
            "the same checkpoint must not reload again"
        );

        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// bddap/rl#215: the trainer renames `brain.bin` BEFORE the normalizers, so a poll
    /// can catch a set-torn dir (new brain, previous save's normalizer). The refusal
    /// must keep the current policy AND not cache the verdict against the brain's mtime:
    /// the trailing normalizer lands WITHOUT touching `brain.bin`, so the next poll must
    /// arm the completed set. (Stamping `last_loaded` on the refusal would refuse a
    /// run's FINAL save forever — there is no later save to bump the mtime.)
    #[test]
    fn hot_reload_retries_a_set_torn_refusal_once_the_set_completes() {
        let tmp = std::env::temp_dir();
        let live = tmp.join(format!("rl-hotreload-torn-{}", std::process::id()));
        let empty = tmp.join(format!("rl-hotreload-torn-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
        std::fs::create_dir_all(&live).unwrap();
        std::fs::create_dir_all(&empty).unwrap();

        // The mid-save window: brain from save 22, normalizer still from save 21.
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let paths = CheckpointDir::new(&live);
        crate::training::checkpoint::save_brain(&brain, &paths.brain_file(), 22).unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(brain.arch(), &paths.normalizer_path(), 21)
            .unwrap();

        let mut policy = Policy::load(&empty);
        policy.live_dir = Some(live.clone());
        assert!(
            !policy.try_hot_reload(),
            "a set-torn dir must refuse, keeping the current policy"
        );

        // The trailing normalizer lands; brain.bin is untouched (same mtime).
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(brain.arch(), &paths.normalizer_path(), 22)
            .unwrap();
        assert!(
            policy.try_hot_reload(),
            "the completed set must arm on the next poll despite the unchanged brain mtime"
        );
        assert!(policy.is_loaded());

        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// Write a hand-mutated `Mlp512x3` LEAF record inside a proper envelope — the same
    /// on-disk layout production writes — plus the paired identity normalizer, so each
    /// caller's test exercises ONLY the defect its record mutation planted. Concrete
    /// record type on purpose: the envelope is tagged `ArchId::DEFAULT`, so accepting any
    /// `Record` would let the tag lie about the bytes.
    fn save_brain_record(dir: &Path, record: crate::bot::arch::mlp512x3::Mlp512x3Record<TrainBackend>) {
        std::fs::create_dir_all(dir).unwrap();
        let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
            .record(record, ())
            .unwrap();
        let paths = CheckpointDir::new(dir);
        write_envelope(
            &paths.brain_file(),
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            bytes,
            Some(crate::mesh_fallback::constructed_body_digest()),
            21,
        )
        .unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(ArchId::DEFAULT, &paths.normalizer_path(), 21)
            .unwrap();
    }

    /// Save a brain whose first trunk layer expects `obs_dim` inputs instead of the
    /// current `OBS_SIZE` — the on-disk shape a checkpoint from an older rig has. We
    /// can't get one from `Mlp512x3::new` (it bakes in today's `OBS_SIZE`), so swap
    /// the `trunk_fc1` weight in the record for a `[obs_dim, HIDDEN]` tensor before
    /// recording. This is exactly the file that used to reach the matmul and panic.
    fn save_brain_with_obs_dim(dir: &Path, obs_dim: usize) {
        use burn::module::{Param, ParamId};
        let device = NdArrayDevice::Cpu;
        let mut record = Mlp512x3::<TrainBackend>::new(&device).into_record();
        let [_obs, hidden] = record.trunk_fc1.weight.shape().dims();
        let weight = Tensor::<TrainBackend, 2>::zeros([obs_dim, hidden], &device);
        record.trunk_fc1.weight = Param::initialized(ParamId::new(), weight);
        save_brain_record(dir, record);
    }

    /// rl#36: a checkpoint built for a different `OBS_SIZE` must NOT panic in the matmul —
    /// loading leaves the policy unloaded and `act` returns zeros without ever running the
    /// mismatched weights through a forward pass. (The mismatch-vs-missing distinction is
    /// rl#121, asserted on the [`checkpoint_fits_rig`] verdict in
    /// `checkpoint_fits_rig_classifies_the_on_disk_brain`; the game's launch gate acts on
    /// it, rl#199.)
    #[test]
    fn dim_mismatched_checkpoint_falls_back_instead_of_panicking() {
        let tmp = std::env::temp_dir();
        let dir = tmp.join(format!("rl-dimmismatch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // A stale brain expecting OBS_SIZE+4 inputs (mirrors the seen 77-vs-73 drift).
        save_brain_with_obs_dim(&dir, OBS_SIZE + 4);

        let policy = Policy::load(&dir);
        assert!(
            !policy.is_loaded(),
            "a dim-mismatched checkpoint must fall back to unloaded, not load"
        );
        // The real regression: this call hits the matmul for a loaded policy; with the
        // fallback it returns zeros and never touches the mismatched weights.
        assert_eq!(
            policy.act(&[0.0; OBS_SIZE]),
            [0.0; ACTION_SIZE],
            "an unloaded policy holds the zero-action pose"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A shape-valid checkpoint whose policy head is all-NaN — numerically broken in a
    /// way no load gate can see (the gates classify dims, not values).
    fn save_nan_brain(dir: &Path) {
        use burn::module::{Param, ParamId};
        let device = NdArrayDevice::Cpu;
        let mut record = Mlp512x3::<TrainBackend>::new(&device).into_record();
        let dims: [usize; 2] = record.policy_fc.weight.shape().dims();
        let weight = Tensor::<TrainBackend, 2>::full(dims, f32::NAN, &device);
        record.policy_fc.weight = Param::initialized(ParamId::new(), weight);
        save_brain_record(dir, record);
    }

    /// rl#219: a NaN-spewing brain must be VISIBLE in `act()`'s output. The raw non-finite
    /// mean flows into `CrabActions`, where the actuator — the sole sanitizer — zeroes it
    /// under its latched `error!` (rl#145). `act()` sanitizing upstream masked the fault
    /// as a legitimate rest pose and that guard could never fire.
    #[test]
    fn act_propagates_a_non_finite_brain_output() {
        let dir = std::env::temp_dir().join(format!("rl-nanbrain-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        save_nan_brain(&dir);

        let policy = Policy::load(&dir);
        assert!(
            policy.is_loaded(),
            "the NaN brain is shape-valid, so the dim gates must load it"
        );
        let act = policy.act(&golden_obs());
        assert!(
            act.iter().any(|v| !v.is_finite()),
            "a numerically-broken brain's non-finite mean must reach the caller \
             (the actuator owns zeroing + the loud latched error), got {act:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// rl#121 + rl#199 + rl#200: the gates classify a checkpoint against the rig — a
    /// current-rig brain is `Ok`, a stale one is `Mismatch` carrying its OWN dims, a dir
    /// with no brain is `Missing`, and a present-but-unusable `brain.bin` (corrupt,
    /// legacy, mis-copied, wrong arch, or missing its paired normalizer) is `Refused`
    /// with the reason — distinct from `Missing` so the operator fixes the file they can
    /// see instead of chasing path resolution. This verdict is what lets the release gate
    /// and the game's launch gate refuse a bad checkpoint loudly instead of shipping/arming
    /// an inert rest-pose crab.
    #[test]
    fn checkpoint_fits_rig_classifies_the_on_disk_brain() {
        let tmp = std::env::temp_dir();
        let dir = tmp.join(format!("rl-rigfit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // No brain yet → Missing.
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Missing));

        // A current-rig brain (with its paired normalizer) fits.
        save_brain(&dir);
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Ok));

        // A brain whose paired normalizer is MISSING is Refused — a real brain must never
        // arm against silently-cold normalizer stats.
        std::fs::remove_file(CheckpointDir::new(&dir).normalizer_path()).unwrap();
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Refused(_)));
        save_brain(&dir); // restore the pair

        // A stale brain is a Mismatch carrying its own obs dim (what the gate reports).
        save_brain_with_obs_dim(&dir, OBS_SIZE + 4);
        assert!(matches!(
            checkpoint_fits_rig(&dir),
            RigFit::Mismatch(RigDims { obs, .. }) if obs == OBS_SIZE + 4
        ));

        // A present-but-corrupt brain.bin is Refused, not Missing.
        std::fs::write(CheckpointDir::new(&dir).brain_file(), b"truncated garbage").unwrap();
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Refused(_)));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
