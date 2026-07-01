//! The trained policy: load a checkpoint's brain + obs-normalizer and run deterministic
//! inference (no learning). One implementation, three callers — the demo's `policy_step`
//! (`play::policy`), the game's solo NN-crab (`net::external_crab`), and the headless
//! training-success eval ([`crate::eval`]) — all drive [`Policy::act`].
//!
//! Lives at the crate root (not under the render-gated `play` module) because it is PURE
//! inference: burn tensors + a checkpoint reader, no bevy render types. Keeping it headless
//! is what lets the trainer-side eval reuse the SAME deterministic policy the rendered demo
//! runs, instead of a second copy that could drift (`play` re-exports it for the renderers).

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::{AutodiffModule, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;

use crate::bot::actuator::ACTION_SIZE;
use crate::bot::brain::CrabBrain;
use crate::bot::sensor::OBS_SIZE;
use crate::training::checkpoint::CheckpointDir;
use crate::training::normalizer::{NORMALIZER_CLIP, ObsNormalizer};
use crate::training::{InferBackend, TrainBackend};

/// A loaded policy that maps observations to actions for inference (no learning).
///
/// Non-send because the `ndarray` backend's tensors are not `Sync` (same reason
/// as `TrainingState`).
pub struct Policy {
    brain: CrabBrain<InferBackend>,
    normalizer: ObsNormalizer,
    device: NdArrayDevice,
    /// False when no checkpoint loaded — `act` then returns zero actions (a
    /// neutral, deterministic rest pose) instead of an untrained brain's noise,
    /// so a no-checkpoint render shows the body geometry cleanly.
    loaded: bool,
    /// `Some(dims)` when we LOUDLY refused to arm a wrong-rig checkpoint (its dims, for
    /// attribution); `None` whenever the policy is armed OR holding the legitimate
    /// no-checkpoint rest pose. The one bit that tells a silent-statue-because-no-brain-yet
    /// (`!loaded && mismatch.is_none()`) apart from an operator error
    /// (`!loaded && mismatch.is_some()`). See [`Self::rig_mismatch`].
    mismatch: Option<RigDims>,
    /// Live training checkpoint dir the demo hot-reloads from while running (None
    /// disables). `last_loaded` is the mtime of the brain file last swapped in, so
    /// we reload only when training has written a newer one. See [`Self::try_hot_reload`].
    // Read only by the render-gated demo hot-reload; the headless eval/trainer never swaps
    // checkpoints mid-run, so these are legitimately unused there.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    live_dir: Option<PathBuf>,
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    last_loaded: Option<std::time::SystemTime>,
    /// A stable digest of the loaded checkpoint's bytes (brain + normalizer), `0` when no
    /// checkpoint loaded. Two peers running the SAME weights get the same digest; different
    /// weights get different ones. The GCR bridge folds this into the crab's per-tick lockstep
    /// hash ([`Self::weights_digest`]), so two peers with mismatched brains desync by
    /// construction on tick 1 rather than diverging silently as the float bodies drift apart.
    weights_digest: u64,
}

/// Digest of a checkpoint's on-disk weights (brain + normalizer bytes), or `0` if the brain
/// file is unreadable. The cross-peer "same weights?" check: identical files → identical
/// digest. Reads the raw bytes rather than the deserialized tensors so it needs no backend
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

/// Load a checkpoint's brain record into an inference brain, or `None` if `brain.bin`
/// is absent or won't deserialize. The ONE way a brain is read off disk — shared by the
/// normalizer-paired loader and the rig-fit check ([`checkpoint_fits_rig`]) so the two
/// can't drift on how a checkpoint is parsed.
fn load_brain(dir: &Path, device: &NdArrayDevice) -> Option<CrabBrain<InferBackend>> {
    let paths = CheckpointDir::new(dir);
    if !paths.brain_file().exists() {
        return None;
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder.load(paths.brain_stem(), device).ok()?;
    Some(
        CrabBrain::<TrainBackend>::new(device)
            .load_record(record)
            .valid(),
    )
}

/// Whether a brain's `(obs, action)` dims drive THIS binary's compiled crab rig. The ONE
/// place the fits-the-rig rule is spelled — both the runtime loader (soft-fallback) and the
/// release gate (hard-fail) ask through here, so they can't disagree on what "fits" means.
fn dims_fit_rig(obs: usize, action: usize) -> bool {
    (obs, action) == (OBS_SIZE, ACTION_SIZE)
}

/// Whether the checkpoint at `dir` fits this binary's crab rig — the verdict the
/// release/deploy gate acts on. The fit RULE lives here in crab-world (the consts it
/// compares against are here); the caller only turns the verdict into a message + exit code.
/// A mismatched checkpoint loads "fine" but would degrade the NN crab to its motionless rest
/// pose (rl#36 catches it at load time, but only by going inert), so the gate refuses to ship
/// one at all. `Mismatch` carries the checkpoint's own dims so the operator sees both sides;
/// the rig side is `crate::play::rig_dims` (render-gated, so not linked here).
pub fn checkpoint_fits_rig(dir: &Path) -> RigFit {
    match load_brain(dir, &NdArrayDevice::Cpu) {
        None => RigFit::Missing,
        Some(brain) => {
            let (obs, action) = brain.io_dims();
            if dims_fit_rig(obs, action) {
                RigFit::Ok
            } else {
                RigFit::Mismatch(RigDims { obs, action })
            }
        }
    }
}

/// A brain's `(obs, action)` input/output dimensions — the pair a checkpoint is checked
/// against the rig with. A named pair, not a bare `(usize, usize)`, so the two same-typed
/// fields can't be swapped at a call site or a struct field; the whole rig-fit machinery
/// (the [`RigFit`] verdict, the runtime loader's refusal, the stored `Policy::mismatch`)
/// speaks this one type rather than re-spelling the pair three ways.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RigDims {
    pub obs: usize,
    pub action: usize,
}

/// A checkpoint's fit against this binary's compiled crab rig (see [`checkpoint_fits_rig`]).
pub enum RigFit {
    /// The brain's dims match the rig — safe to drive the NN crab.
    Ok,
    /// No readable `brain.bin` in the dir — nothing to check.
    Missing,
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
    /// Brain + normalizer parsed and the dims fit the rig — arm the NN crab with it.
    Fit(CrabBrain<InferBackend>, ObsNormalizer),
    /// No readable `brain.bin`: the file is absent, or a hot-reload raced a mid-save
    /// write and got a torn read. The LEGITIMATE "no brain yet" case — keep the current
    /// policy / hold the neutral rest pose. Not an error.
    Absent,
    /// The brain parsed but its dims don't fit the compiled rig (carried here so the caller
    /// can name both sides). Arming it would degrade the crab to an inert rest pose that
    /// looks frozen-but-fine, or panic in the first matmul (rl#36) — so the caller LOUDLY
    /// refuses to arm. An operator error (wrong checkpoint for this build), never a transient.
    Mismatch(RigDims),
}

/// Read a brain + normalizer from `dir`, classifying the result for the caller. A torn
/// mid-save read presents as `Absent` (load fails to parse), so a hot-reload keeps the
/// policy it has rather than blanking the running demo to a rest pose; a wrong-rig brain
/// presents as `Mismatch` so the caller can refuse it loudly instead of silently degrading.
fn load_brain_normalizer(dir: &Path, device: &NdArrayDevice) -> Loaded {
    let paths = CheckpointDir::new(dir);
    let Some(brain) = load_brain(dir, device) else {
        return Loaded::Absent;
    };
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
    // Same clip the trainer wrote the normalizer with, so the demo de-normalizes on the
    // exact scale training used — sourced from the one const, never a bare literal.
    let mut normalizer = ObsNormalizer::new(NORMALIZER_CLIP);
    let norm_path = paths.normalizer_path();
    if norm_path.exists()
        && let Some(loaded) = ObsNormalizer::load(&norm_path)
    {
        normalizer = loaded;
    }
    Loaded::Fit(brain, normalizer)
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

impl Policy {
    /// Load brain + normalizer from a checkpoint dir. Missing/corrupt files fall
    /// back to a zero-action policy so the app still launches (useful before the
    /// first checkpoint exists, and to inspect the body's neutral rest pose).
    pub fn load(checkpoint_dir: &Path) -> Self {
        let device = NdArrayDevice::Cpu;
        // A neutral rest-pose brain for the two non-arming outcomes (Absent, Mismatch):
        // a current-rig random-init brain that `act` ignores while `loaded` is false.
        let rest_pose = || {
            (
                CrabBrain::<TrainBackend>::new(&device).valid(),
                ObsNormalizer::new(NORMALIZER_CLIP),
            )
        };
        let (brain, normalizer, loaded, mismatch) =
            match load_brain_normalizer(checkpoint_dir, &device) {
                Loaded::Fit(brain, normalizer) => {
                    info!("play: loaded checkpoint from {}", checkpoint_dir.display());
                    (brain, normalizer, true, None)
                }
                Loaded::Absent => {
                    // The legitimate "no brain yet" rest pose — quiet, expected before the
                    // first checkpoint exists or when inspecting the body's neutral pose.
                    warn!(
                        "play: no usable checkpoint at {} — using zero-action pose",
                        checkpoint_dir.display()
                    );
                    let (brain, normalizer) = rest_pose();
                    (brain, normalizer, false, None)
                }
                Loaded::Mismatch(dims) => {
                    // Wrong checkpoint for this build: refuse to arm, loud + attributable.
                    log_rig_mismatch("play", checkpoint_dir, dims);
                    let (brain, normalizer) = rest_pose();
                    (brain, normalizer, false, Some(dims))
                }
            };

        // Diagnostic: RL_RANDOM_POLICY drives the crab with the untrained
        // random-init brain even without a checkpoint, to see what a FRESH
        // policy does (vs the zero-action rest pose) — distinguishes a learned
        // behaviour from one the dynamics produce on their own.
        let loaded = loaded || std::env::var("RL_RANDOM_POLICY").is_ok_and(|v| v == "1");
        // If the diagnostic override armed a (current-rig) random brain over a mismatched
        // checkpoint, the crab is NOT inert — drop the recorded mismatch so `rig_mismatch()`
        // never contradicts `is_loaded()`. A mismatch only stands while we hold the rest pose.
        let mismatch = if loaded { None } else { mismatch };

        // Digest the on-disk weights iff a real checkpoint loaded. A RANDOM_POLICY brain (no
        // file) gets `0` — it must never enter networked lockstep, and the bridge's
        // loaded-checkpoint guard refuses to arm it there.
        let weights_digest = if loaded {
            checkpoint_digest(checkpoint_dir)
        } else {
            0
        };

        Self {
            brain,
            normalizer,
            device,
            loaded,
            mismatch,
            live_dir: None,
            last_loaded: None,
            weights_digest,
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
    /// running, swap it in; returns whether it did. Safe against a mid-save race:
    /// a torn read makes [`load_brain_normalizer`] return [`Loaded::Absent`] and we keep
    /// the current policy rather than blanking the demo to a rest pose.
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
                self.brain = brain;
                self.normalizer = normalizer;
                self.loaded = true;
                self.mismatch = None;
                self.last_loaded = Some(mtime);
                self.weights_digest = checkpoint_digest(&dir);
                true
            }
            Loaded::Absent => false, // mid-save / unreadable — keep the current policy
            Loaded::Mismatch(dims) => {
                // A wrong-rig brain landed in the live dir: refuse it loudly and keep the
                // policy we have. Stamp `last_loaded` so we log once per distinct file
                // (mtime), not every tick, until a fitting checkpoint replaces it.
                log_rig_mismatch("play (hot-reload)", &dir, dims);
                self.mismatch = Some(dims);
                self.last_loaded = Some(mtime);
                false
            }
        }
    }

    /// Whether a usable checkpoint loaded (vs the zero-action rest-pose fallback). Lets a
    /// caller fail loud when the body will only hold its rest pose ([`Self::act`] returns the
    /// neutral pose while this is false).
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// `Some(dims)` when the last checkpoint we tried to load was a rig MISMATCH we refused
    /// to arm (and logged loudly) — the checkpoint's own dims, so a caller (HUD, the GCR
    /// bridge's arming log) can attribute the inert crab to a wrong-rig checkpoint, distinct
    /// from the legitimate "no checkpoint yet" rest pose where this is `None`. `is_loaded()`
    /// is false in both, but only a mismatch is an operator error.
    pub fn rig_mismatch(&self) -> Option<RigDims> {
        self.mismatch
    }

    /// Stable digest of the loaded weights (`0` if none) — see
    /// [`weights_digest`](Self::weights_digest). The GCR bridge folds it into the crab's
    /// per-tick lockstep hash so peers running different brains desync immediately.
    pub fn weights_digest(&self) -> u64 {
        self.weights_digest
    }

    /// Deterministic action: the policy mean (no exploration noise), so the crab
    /// holds a steady pose instead of jittering. One policy implementation, three
    /// callers — the demo, the game's solo NN-crab, and the headless eval.
    pub fn act(&self, raw_obs: &[f32; OBS_SIZE]) -> [f32; ACTION_SIZE] {
        // No checkpoint → hold the neutral (zero-action) pose: a deterministic
        // view of the body geometry, not an untrained brain's noise.
        if !self.loaded {
            return [0.0; ACTION_SIZE];
        }
        let obs = self.normalizer.normalize_frozen(raw_obs);
        let input =
            Tensor::<InferBackend, 1>::from_floats(obs.as_slice(), &self.device).unsqueeze();
        // Eval/demo is deterministic: it takes the policy MEAN and discards `log_std`, so the
        // exploration-σ floor never reaches a deployed action. Pass the resting floor.
        let (means, _log_std) = self.brain.policy(input, crate::bot::brain::LOG_STD_MIN);
        let flat: Vec<f32> = means.flatten::<1>(0, 1).to_data().to_vec().unwrap();

        let mut out = [0.0f32; ACTION_SIZE];
        for (o, &v) in out.iter_mut().zip(flat.iter()) {
            *o = if v.is_finite() {
                v.clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Module;

    /// Save a freshly-initialised brain into `dir` the way training does, so a
    /// hot-reload has a real checkpoint file to pick up.
    fn save_brain(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let brain = CrabBrain::<TrainBackend>::new(&device);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .record(brain.into_record(), CheckpointDir::new(dir).brain_stem())
            .unwrap();
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
            !policy.loaded,
            "empty checkpoint dir should give an unloaded policy"
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
            policy.loaded,
            "a successful hot-reload marks the policy loaded"
        );
        assert!(
            !policy.try_hot_reload(),
            "the same checkpoint must not reload again"
        );

        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// Save a brain whose first trunk layer expects `obs_dim` inputs instead of the
    /// current `OBS_SIZE` — the on-disk shape a checkpoint from an older rig has. We
    /// can't get one from `CrabBrain::new` (it bakes in today's `OBS_SIZE`), so swap
    /// the `trunk_fc1` weight in the record for a `[obs_dim, HIDDEN]` tensor before
    /// recording. This is exactly the file that used to reach the matmul and panic.
    fn save_brain_with_obs_dim(dir: &Path, obs_dim: usize) {
        use burn::module::{Param, ParamId};
        use burn::tensor::Tensor;
        std::fs::create_dir_all(dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let mut record = CrabBrain::<TrainBackend>::new(&device).into_record();
        let [_obs, hidden] = record.trunk_fc1.weight.shape().dims();
        let weight = Tensor::<TrainBackend, 2>::zeros([obs_dim, hidden], &device);
        record.trunk_fc1.weight = Param::initialized(ParamId::new(), weight);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .record(record, CheckpointDir::new(dir).brain_stem())
            .unwrap();
    }

    /// rl#36: a checkpoint built for a different `OBS_SIZE` must NOT panic in the matmul —
    /// loading leaves the policy unloaded and `act` returns zeros without ever running the
    /// mismatched weights through a forward pass. (The loud-vs-quiet distinction between this
    /// refused mismatch and a legitimate missing checkpoint is rl#121, asserted separately in
    /// `rig_mismatch_refuses_loudly_missing_rests_quietly`.)
    #[test]
    fn dim_mismatched_checkpoint_falls_back_instead_of_panicking() {
        let tmp = std::env::temp_dir();
        let dir = tmp.join(format!("rl-dimmismatch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // A stale brain expecting OBS_SIZE+4 inputs (mirrors the seen 77-vs-73 drift).
        save_brain_with_obs_dim(&dir, OBS_SIZE + 4);

        let policy = Policy::load(&dir);
        assert!(
            !policy.loaded,
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

    /// rl#121: the runtime loader must DISTINGUISH a rig MISMATCH from a MISSING checkpoint.
    /// Both hold the inert rest pose (`!loaded`, `act` returns zeros), but a mismatch is the
    /// loud-refuse path — it records the offending dims via `rig_mismatch()` (the loud
    /// `error!` rides alongside) so the inert crab is attributable, NOT a silent statue that
    /// looks like the legitimate "no brain yet" pose (where `rig_mismatch()` is `None`).
    #[test]
    fn rig_mismatch_refuses_loudly_missing_rests_quietly() {
        let tmp = std::env::temp_dir();
        let id = std::process::id();
        let miss = tmp.join(format!("rl-rt-missing-{id}"));
        let bad = tmp.join(format!("rl-rt-mismatch-{id}"));
        let _ = std::fs::remove_dir_all(&miss);
        let _ = std::fs::remove_dir_all(&bad);

        // MISSING: no checkpoint → the legitimate rest pose, NOT flagged as a mismatch.
        let missing = Policy::load(&miss);
        assert!(!missing.loaded, "a missing checkpoint must not arm the policy");
        assert_eq!(
            missing.rig_mismatch(),
            None,
            "a missing checkpoint is the quiet rest pose, not a rig mismatch"
        );

        // MISMATCH: a wrong-rig brain → refuse to arm AND record the dims for attribution.
        save_brain_with_obs_dim(&bad, OBS_SIZE + 4);
        let mismatched = Policy::load(&bad);
        assert!(
            !mismatched.loaded,
            "a mismatched checkpoint must refuse to arm (drive the rest pose, not the brain)"
        );
        assert_eq!(
            mismatched.rig_mismatch(),
            Some(RigDims {
                obs: OBS_SIZE + 4,
                action: ACTION_SIZE
            }),
            "a mismatch must be recorded with the checkpoint's own dims for the loud refusal"
        );
        // The non-arming is still graceful: it holds the rest pose, never panics in the matmul.
        assert_eq!(
            mismatched.act(&[0.0; OBS_SIZE]),
            [0.0; ACTION_SIZE],
            "a refused mismatch holds the zero-action pose without running the bad weights"
        );

        let _ = std::fs::remove_dir_all(&miss);
        let _ = std::fs::remove_dir_all(&bad);
    }

    /// The release gate classifies a checkpoint against the rig: a current-rig brain is
    /// `Ok`, a stale one is `Mismatch` carrying its OWN dims, and a dir with no brain is
    /// `Missing`. This is what lets the gate refuse a mismatch loudly instead of shipping a
    /// checkpoint that would go inert at runtime.
    #[test]
    fn checkpoint_fits_rig_classifies_the_on_disk_brain() {
        let tmp = std::env::temp_dir();
        let dir = tmp.join(format!("rl-rigfit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // No brain yet → Missing.
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Missing));

        // A current-rig brain fits.
        let device = NdArrayDevice::Cpu;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        std::fs::create_dir_all(&dir).unwrap();
        recorder
            .record(
                CrabBrain::<TrainBackend>::new(&device).into_record(),
                CheckpointDir::new(&dir).brain_stem(),
            )
            .unwrap();
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Ok));

        // A stale brain is a Mismatch carrying its own obs dim (what the gate reports).
        save_brain_with_obs_dim(&dir, OBS_SIZE + 4);
        assert!(matches!(
            checkpoint_fits_rig(&dir),
            RigFit::Mismatch(RigDims { obs, .. }) if obs == OBS_SIZE + 4
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
