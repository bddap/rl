//! Running observation normalizer (Welford) shared by the learner's master stats and
//! each rollout thread's per-horizon increment.
//!
//! # The snapshot → increment → merge contract
//!
//! The master stats flow learner→worker as a [`NormalizerSnapshot`] (the full
//! cumulative baseline), and a worker's per-horizon samples flow worker→learner as a
//! [`NormalizerIncrement`] (only the observations THIS horizon added). [`ObsNormalizer::merge`]
//! accepts only the latter: the parallel-Welford combine is exact only for two
//! DISJOINT streams, so folding in a cumulative snapshot would re-count the baseline
//! the master already holds. That invariant is enforced by the type system here, not
//! by prose + a regression test — a snapshot and an increment are distinct types, and
//! only an increment can reach `merge`. The two are also produced by distinct types:
//! a baseline-carrying [`ObsNormalizer`] yields only snapshots, a zero-baseline
//! [`IncrementAccumulator`] yields only increments, so an increment can never smuggle
//! a baseline in.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::bot::sensor::OBS_SIZE;

use super::atomic_write;

/// Per-element Welford state: running `(count, mean, M2)` for each of the `OBS_SIZE`
/// observation elements. Both the master normalizer and a worker's per-horizon
/// increment ARE one of these over their own sample stream — they differ only in
/// whether a baseline was loaded first — so the fold and the parallel combine live
/// here once and the two cannot compute them differently.
///
/// Per-element counts (not one shared count) because the NaN-skip lets an element be
/// observed while its neighbours are not; variance is derived from `m2` on demand, so
/// there is no separate `var` to drift from `m2`/`count`.
#[derive(Clone)]
struct Welford {
    mean: [f64; OBS_SIZE],
    m2: [f64; OBS_SIZE],    // sum of squared differences from mean
    count: [u64; OBS_SIZE], // per-element count (NaN-skipped elements don't inflate others)
}

impl Welford {
    fn new() -> Self {
        Self {
            mean: [0.0; OBS_SIZE],
            m2: [0.0; OBS_SIZE],
            count: [0; OBS_SIZE],
        }
    }

    /// Per-element variance `m2 / (count-1)`. Defaults to 1.0 for an element seen at
    /// most once (no spread estimate yet), matching the unit-variance starting point a
    /// fresh normalizer uses.
    fn variance(&self, i: usize) -> f64 {
        if self.count[i] > 1 {
            (self.m2[i] / (self.count[i] as f64 - 1.0)).max(0.0)
        } else {
            1.0
        }
    }

    /// Fold one finite sample of element `i` into the running `(count, mean, m2)` — the
    /// inner Welford step.
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

    fn observe(&mut self, obs: &[f32; OBS_SIZE]) {
        for (i, &raw) in obs.iter().enumerate() {
            self.observe_element(i, raw);
        }
    }

    /// Parallel Welford combine: fold a DISJOINT `other` stream's per-element
    /// `(count, mean, M2)` into this one. Exact only when `other` shares no samples
    /// with `self` — the type wall around [`NormalizerIncrement`] is what upholds that.
    /// Per element because the NaN-skip lets counts differ across elements.
    fn merge(&mut self, other: &Welford) {
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
}

/// Running observation normalizer using Welford's online algorithm. Normalizes
/// observations to zero mean, unit variance, and is the only Welford that carries a
/// `clip` and a loaded baseline — so it produces a [`NormalizerSnapshot`], never an
/// increment (an increment must be baseline-free, see [`IncrementAccumulator`]).
pub(crate) struct ObsNormalizer {
    welford: Welford,
    clip: f32, // max absolute normalized value
}

/// The on-disk checkpoint format AND the cumulative baseline shipped learner→rollout
/// thread (in-process, not over a wire). Serde-friendly because arrays > 32 don't
/// auto-derive. A snapshot is the master's FULL stats, so it must never reach
/// [`ObsNormalizer::merge`] — the type, not a comment, enforces that. No `var` field:
/// variance is `m2/(count-1)`, derived on demand, so it can't drift from `m2`/`count`.
///
/// The field set and order are the bincode wire format on disk; changing them breaks
/// resuming existing checkpoints, so a format change must bump a version and cold-start.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct NormalizerSnapshot {
    mean: Vec<f64>,
    m2: Vec<f64>,
    count: Vec<u64>,
    clip: f32,
}

/// A worker's disjoint per-horizon delta: a Welford over ONLY the observations one
/// rollout thread saw since its last reset, counted from a zero baseline. The single
/// input [`ObsNormalizer::merge`] accepts — never a [`NormalizerSnapshot`]. In-process
/// only (worker→learner over a channel), so it carries no `clip` and no serde: the
/// merge reads only `(count, mean, M2)`, and the fixed-size arrays make the old
/// runtime size-mismatch check unrepresentable. Boxed because the `Welford` is ~2KB of
/// inline arrays; keeping it off the `RollOutcome` enum (whose other variant is empty)
/// avoids bloating every outcome that crosses the channel.
pub(crate) struct NormalizerIncrement(Box<Welford>);

/// Accumulates ONLY one horizon's observations from a zero baseline — the delta a
/// rollout thread ships back for the learner to [`ObsNormalizer::merge`]. Distinct
/// from [`ObsNormalizer`] precisely so it can never be loaded with a baseline or used
/// to normalize: it only `observe`s and yields a [`NormalizerIncrement`]. This is what
/// makes "merge only a disjoint increment, never a cumulative snapshot" a property of
/// the types rather than of a convention.
pub(crate) struct IncrementAccumulator {
    welford: Welford,
}

impl IncrementAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            welford: Welford::new(),
        }
    }

    /// Count one observation toward this horizon's increment (NaN-skipped per element),
    /// WITHOUT normalizing it — the policy's normalized value comes from the master copy
    /// (full baseline+horizon stats), so the increment only needs to tally the samples.
    pub(crate) fn observe(&mut self, obs: &[f32; OBS_SIZE]) {
        self.welford.observe(obs);
    }

    /// The disjoint delta to ship to the learner. A by-value `Welford` clone; the
    /// accumulator is reset (replaced) at the next horizon, so no aliasing. Named
    /// distinctly from [`ObsNormalizer::snapshot`] because it yields the opposite kind:
    /// a delta the master may merge, never a cumulative baseline.
    pub(crate) fn increment(&self) -> NormalizerIncrement {
        NormalizerIncrement(Box::new(self.welford.clone()))
    }
}

/// Max absolute normalized observation value (Welford clip). One source of truth for
/// every `ObsNormalizer::new`, so the learner's master and a checkpoint reload share
/// the same clip and can't drift.
pub(crate) const NORMALIZER_CLIP: f32 = 5.0;

impl ObsNormalizer {
    pub(crate) fn new(clip: f32) -> Self {
        Self {
            welford: Welford::new(),
            clip,
        }
    }

    /// Update running stats, then return the normalized observation.
    pub(crate) fn normalize(&mut self, obs: &[f32; OBS_SIZE]) -> [f32; OBS_SIZE] {
        self.welford.observe(obs);
        self.normalize_frozen(obs)
    }

    /// Normalize against the current statistics WITHOUT updating them. Inference
    /// (play/demo) uses this so the running mean/var stay fixed at the values learned
    /// during training rather than drifting toward demo observations.
    pub(crate) fn normalize_frozen(&self, obs: &[f32; OBS_SIZE]) -> [f32; OBS_SIZE] {
        let clip = self.clip;
        let mut normalized = [0.0f32; OBS_SIZE];
        for i in 0..OBS_SIZE {
            let raw = obs[i];
            if !raw.is_finite() {
                normalized[i] = 0.0;
                continue;
            }
            let std = (self.welford.variance(i) as f32).sqrt().max(1e-6);
            let val = (raw - self.welford.mean[i] as f32) / std;
            normalized[i] = if val.is_nan() {
                0.0
            } else {
                val.clamp(-clip, clip)
            };
        }
        normalized
    }

    pub(crate) fn save(&self, path: &Path) {
        let bytes = match bincode::serialize(&self.snapshot()) {
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

    /// The cumulative baseline as the serde mirror — handed to a rollout thread, and
    /// persisted as the checkpoint. A snapshot, never an increment, so it cannot be
    /// merged.
    pub(crate) fn snapshot(&self) -> NormalizerSnapshot {
        NormalizerSnapshot {
            mean: self.welford.mean.to_vec(),
            m2: self.welford.m2.to_vec(),
            count: self.welford.count.to_vec(),
            clip: self.clip,
        }
    }

    /// Replace this normalizer's stats with a master `snapshot` (e.g. handed to a
    /// rollout thread before its next rollout). Returns false on a size/validity
    /// mismatch, leaving self unchanged.
    pub(crate) fn load_snapshot(&mut self, snapshot: NormalizerSnapshot) -> bool {
        match Self::from_snapshot(snapshot) {
            Some(n) => {
                *self = n;
                true
            }
            None => false,
        }
    }

    fn from_snapshot(d: NormalizerSnapshot) -> Option<Self> {
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
        n.welford.mean.copy_from_slice(&d.mean);
        n.welford.m2.copy_from_slice(&d.m2);
        n.welford.count.copy_from_slice(&d.count);
        Some(n)
    }

    /// Fold a rollout thread's disjoint per-horizon [`NormalizerIncrement`] into the
    /// master. Only an increment is accepted: see the module-level contract for why a
    /// cumulative snapshot must never reach here.
    pub(crate) fn merge(&mut self, increment: &NormalizerIncrement) {
        self.welford.merge(&increment.0);
    }

    pub(crate) fn load(path: &Path) -> Option<Self> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                warn!("Failed to read normalizer from {}: {e}", path.display());
                return None;
            }
        };
        let data: NormalizerSnapshot = match bincode::deserialize(&bytes) {
            Ok(d) => d,
            Err(e) => {
                warn!(
                    "Failed to deserialize normalizer from {}: {e}",
                    path.display()
                );
                return None;
            }
        };
        Self::from_snapshot(data)
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
            (norm.welford.mean[0] - 1.0).abs() < 0.01,
            "element 0 mean should be ~1.0, got {}",
            norm.welford.mean[0]
        );
        assert_eq!(norm.welford.count[0], 100);

        assert!(
            (norm.welford.mean[1] - 1.0).abs() < 0.01,
            "element 1 mean should be ~1.0, got {}",
            norm.welford.mean[1]
        );
        assert_eq!(norm.welford.count[1], 50);
    }

    #[test]
    fn load_snapshot_rejects_a_malformed_snapshot_and_leaves_self_unchanged() {
        // bddap/rl#177: a failed snapshot load must report `false` (so the rollout thread can
        // REFUSE the horizon instead of rolling mis-normalized obs) AND leave the existing stats
        // intact — never a half-applied normalizer. Pin both: a wrong-width snapshot is rejected,
        // and the normalizer keeps the stats it had.
        let mut norm = ObsNormalizer::new(5.0);
        for i in 0..30 {
            let mut obs = [0.0f32; OBS_SIZE];
            obs[0] = i as f32;
            norm.normalize(&obs);
        }
        let before = norm.welford.mean[0];
        let bad = NormalizerSnapshot {
            mean: vec![0.0; OBS_SIZE + 1], // wrong width — must be rejected
            m2: vec![0.0; OBS_SIZE + 1],
            count: vec![0; OBS_SIZE + 1],
            clip: 5.0,
        };
        assert!(!norm.load_snapshot(bad), "a wrong-width snapshot must fail to load");
        assert_eq!(
            norm.welford.mean[0], before,
            "a rejected load must leave the normalizer's stats unchanged, not half-applied"
        );

        // A negative-M2 (otherwise correctly-sized) snapshot is also rejected.
        let mut neg = norm.snapshot();
        neg.m2[0] = -1.0;
        assert!(!norm.load_snapshot(neg), "a negative-M2 snapshot must fail to load");
        assert_eq!(norm.welford.mean[0], before, "still unchanged after the second rejection");
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

        let bytes = bincode::serialize(&norm.snapshot()).expect("serialize");
        let loaded_data: NormalizerSnapshot = bincode::deserialize(&bytes).expect("deserialize");
        let loaded = ObsNormalizer::from_snapshot(loaded_data).expect("from_snapshot");

        assert_eq!(norm.welford.count, loaded.welford.count);
        for i in 0..OBS_SIZE {
            assert!(
                (norm.welford.mean[i] - loaded.welford.mean[i]).abs() < 1e-10,
                "mean[{i}] mismatch"
            );
            assert!(
                (norm.welford.variance(i) - loaded.welford.variance(i)).abs() < 1e-10,
                "var[{i}] mismatch"
            );
            assert!(
                (norm.welford.m2[i] - loaded.welford.m2[i]).abs() < 1e-10,
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

        // Two independent half-streams: A normalizes its half, B accumulates the other
        // half as a disjoint increment, then A merges B in.
        let mut a = ObsNormalizer::new(5.0);
        for i in 0..40 {
            a.normalize(&sample(i));
        }
        let mut b = IncrementAccumulator::new();
        for i in 40..80 {
            b.observe(&sample(i));
        }
        a.merge(&b.increment());

        for i in 0..OBS_SIZE {
            assert_eq!(a.welford.count[i], whole.welford.count[i], "count[{i}]");
            assert!(
                (a.welford.mean[i] - whole.welford.mean[i]).abs() < 1e-9,
                "mean[{i}]: merged {} vs whole {}",
                a.welford.mean[i],
                whole.welford.mean[i]
            );
            // M2 (and hence variance) is the part a naive mean-only merge gets
            // wrong; assert it directly. Relative tolerance because M2 grows large.
            let scale = whole.welford.m2[i].abs().max(1.0);
            assert!(
                (a.welford.m2[i] - whole.welford.m2[i]).abs() / scale < 1e-9,
                "m2[{i}]: merged {} vs whole {}",
                a.welford.m2[i],
                whole.welford.m2[i]
            );
        }
    }

    /// CRITICAL regression: the snapshot→roll→merge LOOP must not double-count the
    /// re-handed baseline. Each iteration the learner's master is snapshotted to the
    /// rollout thread, the thread rolls (mutating its copy with this horizon's
    /// samples) and hands back ONLY the per-horizon increment, which the master
    /// merges. After N iterations the master must equal a single stream over every
    /// sample — no doubling. The classic bug ships the cumulative snapshot, so the
    /// master re-merges its own baseline every iteration (C → 2C+S → 4C+3S …); the
    /// type wall (`merge` takes a `NormalizerIncrement`, never a `NormalizerSnapshot`)
    /// now makes that bug a compile error, and this pins the loop end to end.
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
            // Snapshot: the thread loads the master, and starts a fresh increment over
            // only the samples it is about to see this horizon. The thread's full copy
            // keeps normalizing against baseline+horizon, but only the increment ships.
            let mut worker_full = ObsNormalizer::new(5.0);
            assert!(worker_full.load_snapshot(master.snapshot()), "load snapshot");
            let mut increment = IncrementAccumulator::new();
            for _ in 0..per_iter {
                let obs = sample(next);
                next += 1;
                whole.normalize(&obs);
                worker_full.normalize(&obs); // policy normalizes with full stats
                increment.observe(&obs); // but only this horizon's samples ship
            }
            // Ship the increment; the master merges samples it has not counted.
            master.merge(&increment.increment());
        }

        for i in 0..OBS_SIZE {
            assert_eq!(
                master.welford.count[i], whole.welford.count[i],
                "count[{i}] diverged — baseline double-counted?"
            );
            assert!(
                (master.welford.mean[i] - whole.welford.mean[i]).abs() < 1e-9,
                "mean[{i}]: master {} vs single-stream {}",
                master.welford.mean[i],
                whole.welford.mean[i]
            );
            let scale = whole.welford.m2[i].abs().max(1.0);
            assert!(
                (master.welford.m2[i] - whole.welford.m2[i]).abs() / scale < 1e-9,
                "m2[{i}]: master {} vs single-stream {}",
                master.welford.m2[i],
                whole.welford.m2[i]
            );
        }
    }
}
