//! Running observation normalizer (Welford) shared by the learner's master stats and
//! each rollout thread's per-horizon increment.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::bot::sensor::OBS_SIZE;

use super::atomic_write;

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
/// `pub(crate)`. No `var` field, for the same reason `ObsNormalizer` stores none.
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

    pub(crate) fn save(&self, path: &Path) {
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

    /// The Welford clip an increment must share with the master it merges into (see
    /// [`NORMALIZER_CLIP`]) — so the rollout thread's per-horizon increment is built over
    /// the same clip as the master stats it accumulates against.
    pub(crate) fn clip(&self) -> f32 {
        self.clip
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
    /// the master already counted); merging a cumulative snapshot would double-count
    /// the baseline. Per element because the NaN-skip lets counts differ across
    /// elements; variance is derived from the merged M2 on demand.
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

#[cfg(test)]
mod tests {
    use super::*;

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
    /// stats as one single-threaded stream that saw every sample. That equivalence is
    /// the load-bearing correctness check for the multi-threaded rollout. Includes a
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
}
