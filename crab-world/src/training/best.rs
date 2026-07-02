//! Best-by-reach checkpoint keeping (rl#157): mirror the full checkpoint
//! set into `<ckpt>/best/` whenever the live policy demonstrates a NEW high-water reach, so a
//! later training COLLAPSE can never destroy the best policy. Note the demo pipeline streams
//! the LATEST `<ckpt>/` (owner call: show the in-progress journey, warts and all) — `best/`
//! is the durable high-water archive a wrecked run recovers from, not the demo's source.
//!
//! ONE source of truth, in the trainer: the learner already pools every rollout thread's
//! per-episode reach, so the competence signal lives here rather than in an external
//! log-scraper that would have to re-derive it and could drift from what the trainer
//! actually trained.
//!
//! Competence is just the smoothed reach fraction (the target band is a FIXED full-arena range,
//! so every iter rolls at the same distance and "more capable" reduces to "reaches more often").
//! A solid-reach floor ([`SOLID_REACH_FRACTION`]) gates every promotion, so a collapse — which
//! drops reach — can never become the best.

use std::path::{Path, PathBuf};

use tracing::{info, warn};

use super::checkpoint::{
    BRAIN_FILENAME, NORMALIZER_FILENAME, OPTIMIZER_FILENAME, RETURN_NORMALIZER_FILENAME,
    TICK_WATERMARK_FILENAME,
};
use super::curriculum::SOLID_REACH_FRACTION;

/// Subdirectory of the checkpoint dir holding the best-by-reach snapshot.
const BEST_SUBDIR: &str = "best";

/// Sidecar recording the reach the snapshot in `best/` was taken at, as a single `"<reach>"`
/// float. Read at startup to resume the running best across a trainer restart (the overnight
/// loop makes restarts the expected case), so a collapse after a restart can't reset the bar
/// and overwrite the good snapshot. Lives in `best/`, written last on each snapshot.
const REACH_SIDECAR: &str = "reach.txt";

/// One file in a `best/` snapshot and whether the set is incomplete without it.
struct BestFile {
    /// On-disk name, from [`super::checkpoint`]'s canonical consts so it can't drift.
    name: &'static str,
    /// A `required` file missing means an incomplete inference set — [`BestKeeper::snapshot`]
    /// refuses to write a `best/` that would be marked valid yet missing it. The inference
    /// set the demo loads (brain + the two normalizers) is required; the optimizer
    /// (warm-resume only, absent on a cold-resumed run) and the tick odometer
    /// (policy-independent budget count) are optional and skipped if absent.
    required: bool,
}

/// The checkpoint files copied into `best/`, in install order with [`BRAIN_FILENAME`] LAST:
/// the release poller and demo hot-reload key pairing on the brain's mtime, so writing it
/// after the rest guarantees a reader never pairs a new brain with stale normalizers. Names
/// reference [`super::checkpoint`]'s consts (no re-typed literals), so the set stays in
/// lockstep with the writers; `required` makes a missing inference artifact fail loud rather
/// than yield a silently-incomplete best set (the case that bites hardest, since `best/` is
/// the demo's collapse-proof fallback).
const BEST_FILES: &[BestFile] = &[
    BestFile {
        name: NORMALIZER_FILENAME,
        required: true,
    },
    BestFile {
        name: RETURN_NORMALIZER_FILENAME,
        required: true,
    },
    BestFile {
        name: OPTIMIZER_FILENAME,
        required: false,
    },
    BestFile {
        name: TICK_WATERMARK_FILENAME,
        required: false,
    },
    BestFile {
        name: BRAIN_FILENAME,
        required: true,
    },
];

/// EMA smoothing factor for the per-iter reach fraction. Reach over one iter's finished
/// episodes is noisy (a handful of episodes); ~0.05 averages over roughly the last 20
/// iters, so a single lucky or unlucky iter neither promotes a snapshot nor masks a real
/// gain.
const REACH_EMA_ALPHA: f32 = 0.05;

/// A new snapshot must beat the current best's reach by at least this, so EMA jitter around a
/// plateau doesn't churn the snapshot every iter.
const REACH_MARGIN: f32 = 0.02;

/// Iters with at least one finished episode the EMA must accumulate before any snapshot is
/// eligible, so a freshly-seeded EMA (cold at restart) can't promote on early noise.
const MIN_EMA_UPDATES: u32 = 30;

/// A smoothed reach fraction, in `0..=1` and finite by construction. The newtype exists for
/// one load-bearing reason: a NaN reach must never reach [`Reach::beats`]'s floor comparison,
/// where `NaN < SOLID_REACH_FRACTION` is `false` and would let a corrupt-metric checkpoint pass
/// the collapse guard and be accepted as "best". Rejecting non-finite at construction makes that
/// unrepresentable. No `PartialOrd` derive on purpose: all ordering goes through [`Reach::beats`]'s
/// explicit `.get()` float comparisons, and an implicit `<` on the newtype would re-open the very
/// float-comparison footgun it exists to discipline.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Reach(f32);

impl Reach {
    /// `Some` iff `v` is finite and in `0..=1`; `None` for NaN/∞/out-of-range.
    fn new(v: f32) -> Option<Self> {
        (v.is_finite() && (0.0..=1.0).contains(&v)).then_some(Self(v))
    }

    fn get(self) -> f32 {
        self.0
    }

    /// Does this reach clear the solid-reach floor — the gate a policy must pass to seed or beat
    /// the best, so a collapse (reach below the floor) can never promote. Sound against a corrupt
    /// metric because [`Reach`] is finite by construction.
    fn is_solid(self) -> bool {
        self.0 >= SOLID_REACH_FRACTION
    }

    /// Does `self` demonstrate strictly more reach than `other` — clears the solid-reach floor AND
    /// improves on `other` by at least [`REACH_MARGIN`] (so plateau EMA jitter doesn't churn).
    fn beats(self, other: Reach) -> bool {
        self.is_solid() && self.0 > other.0 + REACH_MARGIN
    }

    /// Parse the `best/reach.txt` sidecar (a single `"<reach>"` float); `None` on a missing,
    /// unreadable, malformed, or out-of-range/non-finite file (a fresh `best/`, or a corrupt
    /// one — both treated as "no prior best", which the seed path then re-gates by reach).
    fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Reach::new(text.split_whitespace().next()?.parse().ok()?)
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
    /// The reach `best/` currently holds; `None` if `best/` has no sidecar yet, in
    /// which case the first eligible solid-reach policy seeds it.
    best: Option<Reach>,
}

impl BestKeeper {
    /// Construct over a checkpoint dir, resuming the running best from `best/reach.txt`
    /// if present. Logs what it resumed so a restart's bar is visible in the trainer log.
    pub(crate) fn new(checkpoint_dir: &Path) -> Self {
        let best = Reach::load(&checkpoint_dir.join(BEST_SUBDIR).join(REACH_SIDECAR));
        match best {
            Some(r) => info!(
                "[best] keeping best-by-reach in {}/{BEST_SUBDIR} | resumed best reach {:.3}",
                checkpoint_dir.display(),
                r.get()
            ),
            None => info!(
                "[best] keeping best-by-reach in {}/{BEST_SUBDIR} | no prior best — first solid policy seeds it",
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
    /// Updates the EMA and, if the smoothed reach beats the best (past the warmup),
    /// snapshots `best/`.
    ///
    /// CORRECTNESS — the on-disk `<ckpt>/` POLICY set this copies must be the policy that
    /// produced THIS iter's reach. The learner persists the checkpoint at the TOP of each iter
    /// (the snapshot it then rolls) and does not rewrite the policy files until the next iter's
    /// top, so when the learner calls this after the iter completes, `<ckpt>/` on disk still
    /// holds this iter's policy — not the post-update one. (The lone exception is `ticks.txt`,
    /// the odometer the learner advances mid-iter; it is policy-independent, so its one-iter
    /// skew in the snapshot is immaterial — see [`BEST_FILES`].) Call it from that position.
    pub(crate) fn observe(&mut self, reach: Option<f32>) {
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
        let Some(smoothed) = self.smoothed_reach else {
            return;
        };
        // Reject a non-finite/out-of-range metric here rather than let it slip past the floor
        // guard (NaN < FLOOR is false). A corrupt reach means no eligible candidate.
        let Some(candidate) = Reach::new(smoothed) else {
            warn!("[best] non-finite reach ({smoothed}) — not eligible for snapshot");
            return;
        };

        let beats = match self.best {
            Some(best) => candidate.beats(best),
            // No prior best: seed on the first policy that clears the solid-reach floor.
            None => candidate.is_solid(),
        };
        if !beats {
            return;
        }

        match self.snapshot(candidate) {
            Ok(()) => {
                info!(
                    "[best] new best snapshot → {}/{BEST_SUBDIR} | reach {:.3} (was {})",
                    self.checkpoint_dir.display(),
                    candidate.get(),
                    self.best
                        .map(|r| format!("reach {:.3}", r.get()))
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

    /// Copy the live checkpoint set into `best/`, each file via a temp-then-fsync-rename so
    /// no crash (process or power loss) can leave a torn file, [`BRAIN_FILENAME`] last (see
    /// [`BEST_FILES`]), then the reach sidecar last of all.
    ///
    /// FAIL-LOUD: a missing REQUIRED source (the brain, the two normalizers, the shape
    /// sidecar — see [`BestFile`]) aborts the whole snapshot with an error BEFORE any file is
    /// touched, so `best/` is never left marked-valid-but-incomplete (the trap when `best/`
    /// is the demo's collapse-proof fallback). Optional files (optimizer, tick odometer) are
    /// skipped if absent. The caller logs the error and keeps the previous `best/`.
    fn snapshot(&self, reach: Reach) -> std::io::Result<()> {
        // Pre-flight: every required source must exist before we mutate `best/`, so a missing
        // one can't leave a half-overwritten set (new normalizer + stale brain + stale sidecar).
        for f in BEST_FILES.iter().filter(|f| f.required) {
            let src = self.checkpoint_dir.join(f.name);
            if !src.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "required checkpoint file {} missing from {} — refusing to write an \
                         incomplete best set",
                        f.name,
                        self.checkpoint_dir.display()
                    ),
                ));
            }
        }

        let best_dir = self.checkpoint_dir.join(BEST_SUBDIR);
        std::fs::create_dir_all(&best_dir)?;
        for f in BEST_FILES {
            let src = self.checkpoint_dir.join(f.name);
            if !src.exists() {
                continue; // required files were pre-flighted above; this is an absent optional
            }
            let dst = best_dir.join(f.name);
            let tmp = best_dir.join(format!("{}.tmp", f.name));
            std::fs::copy(&src, &tmp)?;
            super::fsync_rename(&tmp, &dst)?;
        }
        let sidecar = best_dir.join(REACH_SIDECAR);
        super::atomic_write(&sidecar, format!("{}\n", reach.get()).as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reach(v: f32) -> Reach {
        Reach::new(v).expect("test reach in range")
    }

    #[test]
    fn reach_rejects_non_finite_and_out_of_range() {
        assert!(Reach::new(f32::NAN).is_none());
        assert!(Reach::new(f32::INFINITY).is_none());
        assert!(Reach::new(-0.1).is_none());
        assert!(Reach::new(1.1).is_none());
        assert!(Reach::new(0.6).is_some());
    }

    #[test]
    fn higher_reach_needs_margin() {
        assert!(reach(0.90).beats(reach(0.80)));
        assert!(!reach(0.805).beats(reach(0.80))); // within REACH_MARGIN
    }

    #[test]
    fn collapse_never_promotes() {
        // Below the solid-reach floor, nothing promotes — however much it "improves" on a low
        // prior. This is the collapse-immunity invariant: a collapsed policy can never become
        // the best.
        assert!(!reach(0.3).beats(reach(0.0)));
        assert!(!reach(0.59).beats(reach(0.10)));
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
        // below the ceiling (0.7, not 1.0) so the improvement step below has room to beat it —
        // a best already at reach 1.0 is unbeatable by construction.
        for _ in 0..MIN_EMA_UPDATES - 1 {
            k.observe(Some(0.7));
        }
        assert!(!best_brain.exists(), "no snapshot during warmup");

        // Past warmup with solid reach: best/ is seeded with the live brain + sidecar.
        k.observe(Some(0.7));
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v1");
        let r = Reach::load(&dir.join(BEST_SUBDIR).join(REACH_SIDECAR)).unwrap();
        assert!((r.get() - 0.7).abs() < 1e-6);

        // The policy drifts on disk and then COLLAPSES: best/ must NOT be overwritten.
        std::fs::write(dir.join("brain.bin"), b"brain-collapsed").unwrap();
        for _ in 0..200 {
            k.observe(Some(0.0));
        }
        assert_eq!(
            std::fs::read(&best_brain).unwrap(),
            b"brain-v1",
            "collapse must not overwrite the best snapshot"
        );

        // A genuine improvement DOES update best/.
        std::fs::write(dir.join("brain.bin"), b"brain-v2").unwrap();
        for _ in 0..200 {
            k.observe(Some(1.0));
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
            k1.observe(Some(1.0));
        }
        assert!(dir.join(BEST_SUBDIR).join(REACH_SIDECAR).exists());

        // A fresh keeper (restart) must resume that bar, so a post-restart collapse can't
        // reset it and overwrite the good snapshot via the seed-on-empty path.
        std::fs::write(dir.join("brain.bin"), b"brain-collapsed").unwrap();
        let mut k2 = BestKeeper::new(&dir);
        for _ in 0..MIN_EMA_UPDATES {
            k2.observe(Some(0.5));
        }
        assert_eq!(
            std::fs::read(dir.join(BEST_SUBDIR).join("brain.bin")).unwrap(),
            b"brain-v1",
            "resumed bar must reject a post-restart sub-floor policy"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nan_reach_never_snapshots() {
        // The bug this guards: a NaN reach passes `reach < FLOOR` (NaN comparisons are false),
        // so without the finite-range newtype a corrupt metric could be accepted as "best".
        let dir = scratch_ckpt("nan");
        let mut k = BestKeeper::new(&dir);
        for _ in 0..MIN_EMA_UPDATES + 5 {
            k.observe(Some(f32::NAN));
        }
        assert!(
            !dir.join(BEST_SUBDIR).join("brain.bin").exists(),
            "a NaN reach must never produce a best snapshot"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_required_file_fails_loud_and_keeps_previous() {
        // A complete set seeds best/, then a REQUIRED source goes missing: the next eligible
        // snapshot must FAIL (return Err → caller keeps the previous best) rather than write a
        // best/ marked valid yet missing the inference file. Verified directly via snapshot().
        let dir = scratch_ckpt("missing-required");
        let mut k = BestKeeper::new(&dir);
        for _ in 0..MIN_EMA_UPDATES {
            k.observe(Some(0.7));
        }
        let best_brain = dir.join(BEST_SUBDIR).join("brain.bin");
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v1", "seeded");

        // Drop a required source (a normalizer) and improve the brain on disk; a snapshot
        // attempt must error and leave best/ untouched (old brain, no torn set).
        std::fs::remove_file(dir.join("normalizer.bin")).unwrap();
        std::fs::write(dir.join("brain.bin"), b"brain-v2").unwrap();
        assert!(
            k.snapshot(reach(1.0)).is_err(),
            "a missing required source must fail the snapshot"
        );
        assert_eq!(
            std::fs::read(&best_brain).unwrap(),
            b"brain-v1",
            "failed snapshot must not overwrite the previous best brain"
        );

        // The optimizer is OPTIONAL: absent (scratch_ckpt never writes it), a snapshot of the
        // restored full set still succeeds.
        std::fs::write(dir.join("normalizer.bin"), b"norm").unwrap();
        assert!(
            k.snapshot(reach(1.0)).is_ok(),
            "an absent optional optimizer must not block a snapshot"
        );
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v2");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
