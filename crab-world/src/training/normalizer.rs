use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::bot::arch::ArchId;
use crate::bot::sensor::OBS_SIZE;

use super::envelope::{
    ArtifactKind, EnvelopeError, SetKey, read_envelope_expecting, write_envelope,
};

#[derive(Clone)]
struct Welford {
    mean: [f64; OBS_SIZE],
    m2: [f64; OBS_SIZE],
    count: [u64; OBS_SIZE],
}

impl Welford {
    fn new() -> Self {
        Self {
            mean: [0.0; OBS_SIZE],
            m2: [0.0; OBS_SIZE],
            count: [0; OBS_SIZE],
        }
    }

    fn variance(&self, i: usize) -> f64 {
        if self.count[i] > 1 {
            (self.m2[i] / (self.count[i] as f64 - 1.0)).max(0.0)
        } else {
            1.0
        }
    }

    fn observe_element(&mut self, i: usize, raw: f32) -> bool {
        if !raw.is_finite() {
            return true;
        }
        self.count[i] += 1;
        let n = self.count[i] as f64;
        let x = raw as f64;
        let delta = x - self.mean[i];
        self.mean[i] += delta / n;
        let delta2 = x - self.mean[i];
        self.m2[i] += delta * delta2;
        false
    }

    fn observe(&mut self, obs: &[f32; OBS_SIZE]) -> u32 {
        let mut nonfinite = 0;
        for (i, &raw) in obs.iter().enumerate() {
            if self.observe_element(i, raw) {
                nonfinite += 1;
            }
        }
        nonfinite
    }

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

pub(crate) struct ObsNormalizer {
    welford: Welford,
    clip: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct NormalizerSnapshot {
    mean: Vec<f64>,
    m2: Vec<f64>,
    count: Vec<u64>,
    clip: f32,
}

pub(crate) struct NormalizerIncrement(Box<Welford>);

pub(crate) struct IncrementAccumulator {
    welford: Welford,
}

impl IncrementAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            welford: Welford::new(),
        }
    }

    pub(crate) fn observe(&mut self, obs: &[f32; OBS_SIZE]) -> u32 {
        self.welford.observe(obs)
    }

    pub(crate) fn increment(&self) -> NormalizerIncrement {
        NormalizerIncrement(Box::new(self.welford.clone()))
    }
}

pub(crate) const NORMALIZER_CLIP: f32 = 5.0;

impl ObsNormalizer {
    pub(crate) fn new(clip: f32) -> Self {
        Self {
            welford: Welford::new(),
            clip,
        }
    }

    pub(crate) fn normalize(&mut self, obs: &[f32; OBS_SIZE]) -> [f32; OBS_SIZE] {
        self.welford.observe(obs);
        self.normalize_frozen(obs)
    }

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

    /// Persist the running stats inside an [`ArtifactKind::ObsNormalizer`] envelope
    /// tagged with `arch` and stamped with `save_stamp` — the brain these obs scales
    /// were trained against, and the save it belongs to (bddap/rl#215). Returns the
    /// failure instead of swallowing it: the caller (`save_checkpoint`) aborts the SET
    /// on a member failure so a partial save never poses as a complete one.
    pub(crate) fn save(&self, arch: ArchId, path: &Path, save_stamp: u64) -> std::io::Result<()> {
        let bytes = bincode::serialize(&self.snapshot()).map_err(std::io::Error::other)?;
        write_envelope(
            path,
            ArtifactKind::ObsNormalizer,
            arch,
            bytes,
            None,
            save_stamp,
        )
    }

    pub(crate) fn snapshot(&self) -> NormalizerSnapshot {
        NormalizerSnapshot {
            mean: self.welford.mean.to_vec(),
            m2: self.welford.m2.to_vec(),
            count: self.welford.count.to_vec(),
            clip: self.clip,
        }
    }

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

    pub(crate) fn merge(&mut self, increment: &NormalizerIncrement) {
        self.welford.merge(&increment.0);
    }

    /// Load the checkpointed stats, refusing an envelope that fails either half of
    /// `key` — the loaded brain's [`SetKey`]: normalizers are brain-PAIRED and must
    /// never load cross-arch or cross-save (bddap/rl#200 §2, #215). The caller applies
    /// the refusal policy: the trainer aborts, inference refuses the whole checkpoint —
    /// never a warm brain normalizing against cold or mis-paired stats.
    pub(crate) fn load(path: &Path, key: SetKey) -> Result<Self, EnvelopeError> {
        let env = read_envelope_expecting(path, ArtifactKind::ObsNormalizer, key)?;
        let data: NormalizerSnapshot = bincode::deserialize(&env.payload)
            .map_err(|e| EnvelopeError::Corrupt(format!("obs normalizer payload: {e}")))?;
        Self::from_snapshot(data).ok_or_else(|| {
            EnvelopeError::Corrupt("obs normalizer stats invalid (see log for detail)".to_string())
        })
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
    fn observe_counts_nonfinite_elements_skipped() {
        let mut inc = IncrementAccumulator::new();

        let clean = [0.5f32; OBS_SIZE];
        assert_eq!(inc.observe(&clean), 0, "a fully-finite obs skips nothing");

        let mut dirty = [0.5f32; OBS_SIZE];
        dirty[0] = f32::NAN;
        dirty[1] = f32::INFINITY;
        dirty[2] = f32::NEG_INFINITY;
        assert_eq!(
            inc.observe(&dirty),
            3,
            "NaN and ±inf elements are each counted as one skipped sample"
        );

        assert_eq!(inc.welford.count[0], 1, "the NaN element was never folded");
        assert_eq!(
            inc.welford.count[3], 2,
            "a finite element is folded both times"
        );
    }

    #[test]
    fn load_snapshot_rejects_a_malformed_snapshot_and_leaves_self_unchanged() {
        let mut norm = ObsNormalizer::new(5.0);
        for i in 0..30 {
            let mut obs = [0.0f32; OBS_SIZE];
            obs[0] = i as f32;
            norm.normalize(&obs);
        }
        let before = norm.welford.mean[0];
        let bad = NormalizerSnapshot {
            mean: vec![0.0; OBS_SIZE + 1],
            m2: vec![0.0; OBS_SIZE + 1],
            count: vec![0; OBS_SIZE + 1],
            clip: 5.0,
        };
        assert!(
            !norm.load_snapshot(bad),
            "a wrong-width snapshot must fail to load"
        );
        assert_eq!(
            norm.welford.mean[0], before,
            "a rejected load must leave the normalizer's stats unchanged, not half-applied"
        );

        let mut neg = norm.snapshot();
        neg.m2[0] = -1.0;
        assert!(
            !norm.load_snapshot(neg),
            "a negative-M2 snapshot must fail to load"
        );
        assert_eq!(
            norm.welford.mean[0], before,
            "still unchanged after the second rejection"
        );
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

    #[test]
    fn parallel_normalizer_merge_matches_single_stream() {
        let sample = |i: usize| {
            let mut o = [0.0f32; OBS_SIZE];
            o[0] = i as f32;
            o[1] = (i as f32) * 0.5 - 3.0;
            o[2] = ((i * 7) % 11) as f32;
            o[3] = if i.is_multiple_of(2) {
                i as f32
            } else {
                f32::NAN
            };
            o
        };

        let mut whole = ObsNormalizer::new(5.0);
        for i in 0..80 {
            whole.normalize(&sample(i));
        }

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
            let scale = whole.welford.m2[i].abs().max(1.0);
            assert!(
                (a.welford.m2[i] - whole.welford.m2[i]).abs() / scale < 1e-9,
                "m2[{i}]: merged {} vs whole {}",
                a.welford.m2[i],
                whole.welford.m2[i]
            );
        }
    }

    #[test]
    fn snapshot_roll_merge_loop_matches_single_stream() {
        let sample = |i: usize| {
            let mut o = [0.0f32; OBS_SIZE];
            o[0] = i as f32;
            o[1] = (i as f32) * 0.5 - 2.0;
            o[2] = ((i * 5) % 7) as f32;
            o
        };

        let mut whole = ObsNormalizer::new(5.0);
        let mut master = ObsNormalizer::new(5.0);

        let iters = 5;
        let per_iter = 8;
        let mut next = 0usize;
        for _ in 0..iters {
            let mut worker_full = ObsNormalizer::new(5.0);
            assert!(
                worker_full.load_snapshot(master.snapshot()),
                "load snapshot"
            );
            let mut increment = IncrementAccumulator::new();
            for _ in 0..per_iter {
                let obs = sample(next);
                next += 1;
                whole.normalize(&obs);
                worker_full.normalize(&obs);
                increment.observe(&obs);
            }
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
