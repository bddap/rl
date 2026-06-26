//! The trained policy: load a checkpoint's brain + obs-normalizer and run deterministic
//! inference (no learning). One implementation, two callers — the demo's `policy_step`
//! here and the game's solo NN-crab ([`crate::net::solo_crab`]) — both drive [`Policy::act`].

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::{AutodiffModule, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::brain::CrabBrain;
use crate::bot::sensor::{CrabObservation, OBS_SIZE};
use crate::training::checkpoint::CheckpointDir;
use crate::training::normalizer::{NORMALIZER_CLIP, ObsNormalizer};
use crate::training::{InferBackend, TrainBackend};

use super::manual_control::ManualControl;

/// A loaded policy that maps observations to actions for inference (no learning).
///
/// Non-send because the `ndarray` backend's tensors are not `Sync` (same reason
/// as `TrainingState`).
pub(crate) struct Policy {
    brain: CrabBrain<InferBackend>,
    normalizer: ObsNormalizer,
    device: NdArrayDevice,
    /// False when no checkpoint loaded — `act` then returns zero actions (a
    /// neutral, deterministic rest pose) instead of an untrained brain's noise,
    /// so a no-checkpoint render shows the body geometry cleanly.
    loaded: bool,
    /// Live training checkpoint dir the demo hot-reloads from while running (None
    /// disables). `last_loaded` is the mtime of the brain file last swapped in, so
    /// we reload only when training has written a newer one. See [`Self::try_hot_reload`].
    live_dir: Option<PathBuf>,
    last_loaded: Option<std::time::SystemTime>,
    /// A stable digest of the loaded checkpoint's bytes (brain + normalizer), `0` when no
    /// checkpoint loaded. Two peers running the SAME weights get the same digest; different
    /// weights get different ones. The GCR bridge folds this into the crab's per-tick lockstep
    /// hash ([`Self::weights_digest`]), so two peers with mismatched brains desync by
    /// construction on tick 1 rather than diverging silently as the float bodies drift apart.
    weights_digest: u64,
}

/// FNV-1a over `bytes` — a build-stable digest (unlike `std`'s randomized hashers) so two
/// same-binary peers hash an identical checkpoint to an identical value. Used for the policy
/// weights digest that gates lockstep (peers MUST run identical weights or they desync).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Digest of a checkpoint's on-disk weights (brain + normalizer bytes), or `0` if the brain
/// file is unreadable. The cross-peer "same weights?" check: identical files → identical
/// digest. Reads the raw bytes rather than the deserialized tensors so it needs no backend
/// and can't drift from how the weights are stored.
pub(crate) fn checkpoint_digest(dir: &Path) -> u64 {
    let paths = CheckpointDir::new(dir);
    let Ok(mut bytes) = std::fs::read(paths.brain_file()) else {
        return 0;
    };
    if let Ok(norm) = std::fs::read(paths.normalizer_path()) {
        bytes.extend_from_slice(&norm);
    }
    fnv1a(&bytes)
}

/// Load a brain + normalizer from `dir`, or `None` if the brain file is absent or
/// fails to parse. Returning `None` (rather than a zero-action fallback) lets a
/// hot-reload keep the policy it has when it races a mid-save write, instead of
/// blanking the running demo to a rest pose on a torn read.
fn load_brain_normalizer(
    dir: &Path,
    device: &NdArrayDevice,
) -> Option<(CrabBrain<InferBackend>, ObsNormalizer)> {
    let paths = CheckpointDir::new(dir);
    if !paths.brain_file().exists() {
        return None;
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder.load(paths.brain_stem(), device).ok()?;
    let brain = CrabBrain::<TrainBackend>::new(device)
        .load_record(record)
        .valid();
    // A checkpoint from a different rig (e.g. a stale 77-dim brain against the
    // current OBS_SIZE) loads fine here but its mismatched first-layer weight would
    // panic in the matmul at the first `policy()` call. Reject it as if it were
    // missing — None routes to the same zero-action / keep-current fallback a missing
    // brain takes — so a stale `checkpoints/` degrades to the rest pose instead of
    // crashing the demo/screenshot window on launch (rl#36).
    let (obs_dim, action_dim) = brain.io_dims();
    if obs_dim != OBS_SIZE || action_dim != ACTION_SIZE {
        warn!(
            "play: checkpoint dims ({obs_dim} obs, {action_dim} act) don't match the \
             current rig ({OBS_SIZE} obs, {ACTION_SIZE} act) — ignoring it",
        );
        return None;
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
    Some((brain, normalizer))
}

impl Policy {
    /// Load brain + normalizer from a checkpoint dir. Missing/corrupt files fall
    /// back to a zero-action policy so the app still launches (useful before the
    /// first checkpoint exists, and to inspect the body's neutral rest pose).
    pub(crate) fn load(checkpoint_dir: &Path) -> Self {
        let device = NdArrayDevice::Cpu;
        let (brain, normalizer, loaded) = match load_brain_normalizer(checkpoint_dir, &device) {
            Some((brain, normalizer)) => {
                info!("play: loaded checkpoint from {}", checkpoint_dir.display());
                (brain, normalizer, true)
            }
            None => {
                warn!(
                    "play: no usable checkpoint at {} — using zero-action pose",
                    checkpoint_dir.display()
                );
                // Random-init brain; `act` ignores it (returns the rest pose) while
                // `loaded` is false, unless RL_RANDOM_POLICY opts in below.
                let brain = CrabBrain::<TrainBackend>::new(&device).valid();
                (brain, ObsNormalizer::new(NORMALIZER_CLIP), false)
            }
        };

        // Diagnostic: RL_RANDOM_POLICY drives the crab with the untrained
        // random-init brain even without a checkpoint, to see what a FRESH
        // policy does (vs the zero-action rest pose) — distinguishes a learned
        // behaviour from one the dynamics produce on their own.
        let loaded = loaded || std::env::var("RL_RANDOM_POLICY").is_ok_and(|v| v == "1");

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
            live_dir: None,
            last_loaded: None,
            weights_digest,
        }
    }

    /// If the live training dir holds a brain file newer than the one we're
    /// running, swap it in; returns whether it did. Safe against a mid-save race:
    /// a torn read makes [`load_brain_normalizer`] return `None` and we keep the
    /// current policy rather than blanking the demo to a rest pose.
    pub(super) fn try_hot_reload(&mut self) -> bool {
        let Some(dir) = self.live_dir.clone() else {
            return false;
        };
        let brain_bin = CheckpointDir::new(&dir).brain_file();
        let Ok(mtime) = std::fs::metadata(&brain_bin).and_then(|m| m.modified()) else {
            return false; // no live brain file yet
        };
        if self.last_loaded == Some(mtime) {
            return false; // already running this checkpoint
        }
        let Some((brain, normalizer)) = load_brain_normalizer(&dir, &self.device) else {
            return false; // mid-save / unreadable — keep the current policy
        };
        self.brain = brain;
        self.normalizer = normalizer;
        self.loaded = true;
        self.last_loaded = Some(mtime);
        self.weights_digest = checkpoint_digest(&dir);
        true
    }

    /// Whether a usable checkpoint loaded (vs the zero-action rest-pose fallback). Lets a
    /// caller fail loud when the body will only hold its rest pose ([`Self::act`] returns the
    /// neutral pose while this is false).
    pub(crate) fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Stable digest of the loaded weights (`0` if none) — see
    /// [`weights_digest`](Self::weights_digest). The GCR bridge folds it into the crab's
    /// per-tick lockstep hash so peers running different brains desync immediately.
    pub(crate) fn weights_digest(&self) -> u64 {
        self.weights_digest
    }

    /// Deterministic action: the policy mean (no exploration noise), so the crab
    /// holds a steady pose instead of jittering. One policy implementation, two
    /// callers — the demo and the game's solo NN-crab.
    pub(crate) fn act(&self, raw_obs: &[f32; OBS_SIZE]) -> [f32; ACTION_SIZE] {
        // No checkpoint → hold the neutral (zero-action) pose: a deterministic
        // view of the body geometry, not an untrained brain's noise.
        if !self.loaded {
            return [0.0; ACTION_SIZE];
        }
        let obs = self.normalizer.normalize_frozen(raw_obs);
        let input =
            Tensor::<InferBackend, 1>::from_floats(obs.as_slice(), &self.device).unsqueeze();
        let (means, _log_std) = self.brain.policy(input);
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

/// System (BotSet::Think): run the policy and write the actions the actuator will
/// apply — unless manual control has taken over (then `manual_control_step` drives).
pub(super) fn policy_step(
    policy: NonSend<Policy>,
    manual: Option<Res<ManualControl>>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    mut warned_no_env: Local<bool>,
) {
    if manual.is_some_and(|m| m.active) {
        return;
    }
    let (Some(o), Some(a)) = (obs.envs.first(), actions.envs.first_mut()) else {
        // Env 0 is sized at startup and lives the whole run, so a missing slot here is a
        // wiring bug (the policy could never drive the crab), not a respawn transient —
        // surface it. Latched so a persistent miswire logs once, not every tick.
        if !*warned_no_env {
            error!("play: env-0 observation/action slot missing — policy cannot drive the crab");
            *warned_no_env = true;
        }
        return;
    };
    *a = policy.act(o);
}

/// Load the trained policy as a resource. The driver system that turns it into
/// actions (`policy_step`) is added by the caller, so `--manual-control` can swap
/// in [`super::manual_control::manual_control_step`] instead.
pub(super) fn add_inference(app: &mut App, checkpoint_dir: &Path, live_dir: Option<PathBuf>) {
    let mut policy = Policy::load(checkpoint_dir);
    policy.live_dir = live_dir;
    app.insert_non_send_resource(policy);
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

    /// rl#36: a checkpoint built for a different `OBS_SIZE` must degrade to the
    /// zero-action rest pose (as a missing checkpoint does), NOT panic in the matmul.
    /// Loading must leave the policy unloaded and `act` must return zeros without ever
    /// running the mismatched weights through a forward pass.
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
}
