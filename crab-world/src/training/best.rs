//! Best-by-competence checkpoint keeping (rl#157 / job 556): mirror the full checkpoint
//! set into `<ckpt>/best/` whenever the live policy demonstrates NEW competence, so a
//! later training COLLAPSE can neither destroy the best policy nor ship it to the demo.
//! The release/demo pipeline mirrors `<ckpt>/best/`, not the latest `<ckpt>/`, so the
//! high-water-mark policy is what reaches the TV — a collapse stays confined to `<ckpt>/`
//! where the trainer resumes from it, while the demo holds the good gait.
//!
//! ONE source of truth, in the trainer: the learner already pools every rollout thread's
//! per-episode reach, so the competence signal lives here rather than in an external
//! log-scraper that would have to re-derive it and could drift from what the trainer
//! actually trained.
//!
//! Competence is `(band_max, reach)`, ordered "farther mastered band wins; at the same band a
//! higher smoothed reach wins". The target-distance band is now a FIXED full-arena range, so
//! `band_max` is constant across every iter and the ordering reduces to best-by-reach in
//! practice (the band dimension is retained but inert — a candidate simplification once the
//! collapse fight settles). A solid-reach floor ([`SOLID_REACH_FRACTION`]) gates every
//! promotion, so a collapse — which drops reach — can never become the best.

use std::path::{Path, PathBuf};

use tracing::{info, warn};

use super::curriculum::SOLID_REACH_FRACTION;

/// Subdirectory of the checkpoint dir holding the best-by-competence snapshot.
const BEST_SUBDIR: &str = "best";

/// Sidecar recording the competence the snapshot in `best/` was taken at, as
/// `"<band_max> <reach>"`. Read at startup to resume the running best across a trainer
/// restart (the overnight loop makes restarts the expected case), so a collapse after a
/// restart can't reset the bar and overwrite the good snapshot. Lives in `best/`, written
/// last on each snapshot.
const COMPETENCE_SIDECAR: &str = "competence.txt";

/// The checkpoint files copied into `best/`, in install order with `brain.bin` LAST: the
/// release poller and demo hot-reload key pairing on `brain.bin`'s mtime, so writing it
/// after the rest guarantees a reader never pairs a new brain with stale normalizers.
/// `optimizer.bin` is included so `best/` is a complete, warm-resumable checkpoint, not
/// only an inference set. `ticks.txt` (the tick odometer) is included so the set the demo
/// loads is complete, but it is the ONE policy-INDEPENDENT entry: the learner advances it
/// mid-iter (after the rollout, see `write_tick_watermark`), so the snapshot's odometer is
/// up to one iter ahead of the rest — immaterial, it names a budget count, not a policy.
/// Mirrors the on-disk names [`super::checkpoint::CheckpointDir`] owns; if a checkpoint
/// artifact is added there, add it here too.
const BEST_FILES: &[&str] = &[
    "normalizer.bin",
    "return_normalizer.bin",
    "optimizer.bin",
    "shape.txt",
    "ticks.txt",
    "brain.bin",
];

/// EMA smoothing factor for the per-iter reach fraction. Reach over one iter's finished
/// episodes is noisy (a handful of episodes); ~0.05 averages over roughly the last 20
/// iters, so a single lucky or unlucky iter neither promotes a snapshot nor masks a real
/// gain.
const REACH_EMA_ALPHA: f32 = 0.05;

/// A snapshot at the SAME band must beat the current best's reach by at least this, so
/// EMA jitter around a plateau doesn't churn the snapshot every iter.
const REACH_MARGIN: f32 = 0.02;

/// Bands compare equal within this (metres) — guards the `band_max` float comparison. With the
/// band now fixed at the full arena range, every iter's `band_max` is identical, so this only
/// absorbs serialization round-trip noise.
const BAND_EPS: f32 = 0.01;

/// Iters with at least one finished episode the EMA must accumulate before any snapshot is
/// eligible, so a freshly-seeded EMA (cold at restart) can't promote on early noise.
const MIN_EMA_UPDATES: u32 = 30;

/// A policy's demonstrated competence: the far edge of the target band it was rolled at and
/// its smoothed reach there. Ordered so "more capable" is unambiguous (see module docs). The
/// band is now fixed, so `band_max` is constant and the order reduces to best-by-reach.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Competence {
    /// Far edge of the target band the policy was rolled at (metres) — now a fixed constant.
    band_max: f32,
    /// Smoothed reach fraction at that band.
    reach: f32,
}

impl Competence {
    /// Does `self` demonstrate strictly more competence than `other`? A farther mastered
    /// band wins; at the same band a higher reach (by [`REACH_MARGIN`]) wins; a nearer band
    /// never wins. Gated by the solid-reach floor so a collapse can't promote at any band.
    fn beats(&self, other: &Competence) -> bool {
        if self.reach < SOLID_REACH_FRACTION {
            return false;
        }
        if self.band_max > other.band_max + BAND_EPS {
            return true;
        }
        if (self.band_max - other.band_max).abs() <= BAND_EPS {
            return self.reach > other.reach + REACH_MARGIN;
        }
        false
    }

    /// Parse the `best/competence.txt` sidecar (`"<band_max> <reach>"`); `None` on a
    /// missing, unreadable, or malformed file (a fresh `best/`, or none yet).
    fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut it = text.split_whitespace();
        let band_max = it.next()?.parse().ok()?;
        let reach = it.next()?.parse().ok()?;
        Some(Self { band_max, reach })
    }
}

/// Tracks the smoothed reach and the running best, and snapshots `<ckpt>/best/` when the
/// live policy beats it. Owned by the learner loop, fed one observation per iter. Backend-
/// agnostic (pure file I/O + arithmetic), so it is unit-tested without a GPU.
pub(crate) struct BestKeeper {
    checkpoint_dir: PathBuf,
    /// Smoothed per-iter reach; `None` until the first iter with a finished episode.
    smoothed_reach: Option<f32>,
    /// Count of EMA updates so far this run, for the warmup guard.
    ema_updates: u32,
    /// The competence `best/` currently holds; `None` if `best/` has no sidecar yet, in
    /// which case the first eligible solid-reach policy seeds it.
    best: Option<Competence>,
}

impl BestKeeper {
    /// Construct over a checkpoint dir, resuming the running best from `best/competence.txt`
    /// if present. Logs what it resumed so a restart's bar is visible in the trainer log.
    pub(crate) fn new(checkpoint_dir: &Path) -> Self {
        let best = Competence::load(&checkpoint_dir.join(BEST_SUBDIR).join(COMPETENCE_SIDECAR));
        match best {
            Some(c) => info!(
                "[best] keeping best-by-competence in {}/{BEST_SUBDIR} | resumed best band {:.1}m reach {:.3}",
                checkpoint_dir.display(),
                c.band_max,
                c.reach
            ),
            None => info!(
                "[best] keeping best-by-competence in {}/{BEST_SUBDIR} | no prior best — first solid policy seeds it",
                checkpoint_dir.display()
            ),
        }
        Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            smoothed_reach: None,
            ema_updates: 0,
            best,
        }
    }

    /// Observe one finished iter. `reach` is `Some(reached/finished)` when the iter
    /// finished at least one episode, else `None` (no reach signal — the EMA holds).
    /// `band_max` is the far edge of the band this iter was rolled at. Updates the EMA and,
    /// if the smoothed competence beats the best (past the warmup), snapshots `best/`.
    ///
    /// CORRECTNESS — the on-disk `<ckpt>/` POLICY set this copies must be the policy that
    /// produced THIS iter's reach. The learner persists the checkpoint at the TOP of each iter
    /// (the snapshot it then rolls) and does not rewrite the policy files until the next iter's
    /// top, so when the learner calls this after the iter completes, `<ckpt>/` on disk still
    /// holds this iter's policy — not the post-update one. (The lone exception is `ticks.txt`,
    /// the odometer the learner advances mid-iter; it is policy-independent, so its one-iter
    /// skew in the snapshot is immaterial — see [`BEST_FILES`].) Call it from that position.
    pub(crate) fn observe(&mut self, reach: Option<f32>, band_max: f32) {
        if let Some(r) = reach {
            self.smoothed_reach = Some(match self.smoothed_reach {
                Some(prev) => prev + REACH_EMA_ALPHA * (r - prev),
                None => r,
            });
            self.ema_updates += 1;
        }

        if self.ema_updates < MIN_EMA_UPDATES {
            return;
        }
        let Some(reach) = self.smoothed_reach else {
            return;
        };
        let candidate = Competence { band_max, reach };

        let beats = match self.best {
            Some(best) => candidate.beats(&best),
            // No prior best: seed on the first policy that clears the solid-reach floor.
            None => reach >= SOLID_REACH_FRACTION,
        };
        if !beats {
            return;
        }

        match self.snapshot(candidate) {
            Ok(()) => {
                info!(
                    "[best] new best snapshot → {}/{BEST_SUBDIR} | band {:.1}m reach {:.3} (was {})",
                    self.checkpoint_dir.display(),
                    candidate.band_max,
                    candidate.reach,
                    self.best
                        .map(|c| format!("band {:.1}m reach {:.3}", c.band_max, c.reach))
                        .unwrap_or_else(|| "none".to_string()),
                );
                self.best = Some(candidate);
            }
            Err(e) => warn!(
                "[best] failed to snapshot best checkpoint to {}/{BEST_SUBDIR}: {e} — keeping previous best",
                self.checkpoint_dir.display()
            ),
        }
    }

    /// Copy the live checkpoint set into `best/`, each file via a temp-then-rename so a
    /// crash mid-copy can't leave a torn file, `brain.bin` last (see [`BEST_FILES`]), then
    /// the competence sidecar last of all. A missing source file is skipped (e.g. a
    /// cold-resumed optimizer that hasn't been written yet) rather than failing the whole
    /// snapshot — `brain.bin` and the normalizers, the inference set the demo needs, are
    /// always present by the time this runs.
    fn snapshot(&self, competence: Competence) -> std::io::Result<()> {
        let best_dir = self.checkpoint_dir.join(BEST_SUBDIR);
        std::fs::create_dir_all(&best_dir)?;
        for name in BEST_FILES {
            let src = self.checkpoint_dir.join(name);
            if !src.exists() {
                continue;
            }
            let dst = best_dir.join(name);
            let tmp = best_dir.join(format!("{name}.tmp"));
            std::fs::copy(&src, &tmp)?;
            std::fs::rename(&tmp, &dst)?;
        }
        let sidecar = best_dir.join(COMPETENCE_SIDECAR);
        let tmp = best_dir.join(format!("{COMPETENCE_SIDECAR}.tmp"));
        std::fs::write(&tmp, format!("{} {}\n", competence.band_max, competence.reach))?;
        std::fs::rename(&tmp, &sidecar)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comp(band_max: f32, reach: f32) -> Competence {
        Competence { band_max, reach }
    }

    #[test]
    fn farther_band_with_solid_reach_beats_nearer() {
        // A reliably-reaching policy at a farther band is more capable even at lower reach.
        assert!(comp(9.0, 0.7).beats(&comp(3.0, 1.0)));
    }

    #[test]
    fn nearer_band_never_beats_farther() {
        // Even a perfect near-band policy must not regress the demo to easier targets.
        assert!(!comp(3.0, 1.0).beats(&comp(9.0, 0.7)));
    }

    #[test]
    fn same_band_needs_margin() {
        assert!(comp(9.0, 0.90).beats(&comp(9.0, 0.80)));
        assert!(!comp(9.0, 0.805).beats(&comp(9.0, 0.80))); // within REACH_MARGIN
    }

    #[test]
    fn collapse_never_promotes_at_any_band() {
        // Below the solid-reach floor, no band — however far — promotes. This is the
        // collapse-immunity invariant: a collapsed policy can never become the best.
        assert!(!comp(9.0, 0.3).beats(&comp(3.0, 1.0)));
        assert!(!comp(9.0, 0.59).beats(&comp(9.0, 0.10)));
    }

    /// A scratch checkpoint dir with the inference file set present, so `snapshot` has real
    /// files to copy. Returns the dir; the caller drives a `BestKeeper` over it.
    fn scratch_ckpt(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rl-best-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, body) in [
            ("brain.bin", b"brain-v1" as &[u8]),
            ("normalizer.bin", b"norm"),
            ("return_normalizer.bin", b"rnorm"),
            ("shape.txt", b"92 38"),
            ("ticks.txt", b"123"),
        ] {
            std::fs::write(dir.join(name), body).unwrap();
        }
        dir
    }

    #[test]
    fn warmup_then_snapshot_then_immune_to_collapse() {
        let dir = scratch_ckpt("lifecycle");
        let mut k = BestKeeper::new(&dir);
        let best_brain = dir.join(BEST_SUBDIR).join("brain.bin");

        // During warmup (no prior best), nothing is snapshotted even at solid reach. Seed
        // below the ceiling (0.7, not 1.0) so the same-band improvement step below has room
        // to beat it — a best already at reach 1.0 is unbeatable at its band by construction.
        for _ in 0..MIN_EMA_UPDATES - 1 {
            k.observe(Some(0.7), 9.0);
        }
        assert!(!best_brain.exists(), "no snapshot during warmup");

        // Past warmup with solid reach: best/ is seeded with the live brain + sidecar.
        k.observe(Some(0.7), 9.0);
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v1");
        let c = Competence::load(&dir.join(BEST_SUBDIR).join(COMPETENCE_SIDECAR)).unwrap();
        assert_eq!(c.band_max, 9.0);

        // The policy drifts on disk and then COLLAPSES: best/ must NOT be overwritten.
        std::fs::write(dir.join("brain.bin"), b"brain-collapsed").unwrap();
        for _ in 0..200 {
            k.observe(Some(0.0), 9.0);
        }
        assert_eq!(
            std::fs::read(&best_brain).unwrap(),
            b"brain-v1",
            "collapse must not overwrite the best snapshot"
        );

        // A genuine improvement at the same band DOES update best/.
        std::fs::write(dir.join("brain.bin"), b"brain-v2").unwrap();
        for _ in 0..200 {
            k.observe(Some(1.0), 9.0);
        }
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v2");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resumes_best_across_restart() {
        let dir = scratch_ckpt("resume");
        // First keeper seeds best/ at a high bar.
        let mut k1 = BestKeeper::new(&dir);
        for _ in 0..MIN_EMA_UPDATES {
            k1.observe(Some(1.0), 9.0);
        }
        assert!(dir.join(BEST_SUBDIR).join("competence.txt").exists());

        // A fresh keeper (restart) must resume that bar, so a post-restart collapse can't
        // reset it and overwrite the good snapshot via the seed-on-empty path.
        std::fs::write(dir.join("brain.bin"), b"brain-collapsed").unwrap();
        let mut k2 = BestKeeper::new(&dir);
        for _ in 0..MIN_EMA_UPDATES {
            k2.observe(Some(0.5), 9.0);
        }
        assert_eq!(
            std::fs::read(dir.join(BEST_SUBDIR).join("brain.bin")).unwrap(),
            b"brain-v1",
            "resumed bar must reject a post-restart sub-floor policy"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
