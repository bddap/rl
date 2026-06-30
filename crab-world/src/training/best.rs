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

use super::checkpoint::{
    BRAIN_FILENAME, NORMALIZER_FILENAME, OPTIMIZER_FILENAME, RETURN_NORMALIZER_FILENAME,
    SHAPE_FILENAME, TICK_WATERMARK_FILENAME,
};
use super::curriculum::SOLID_REACH_FRACTION;

/// Subdirectory of the checkpoint dir holding the best-by-competence snapshot.
const BEST_SUBDIR: &str = "best";

/// Sidecar recording the competence the snapshot in `best/` was taken at, as
/// `"<band_max> <reach>"`. Read at startup to resume the running best across a trainer
/// restart (the overnight loop makes restarts the expected case), so a collapse after a
/// restart can't reset the bar and overwrite the good snapshot. Lives in `best/`, written
/// last on each snapshot.
const COMPETENCE_SIDECAR: &str = "competence.txt";

/// One file in a `best/` snapshot and whether the set is incomplete without it.
struct BestFile {
    /// On-disk name, from [`super::checkpoint`]'s canonical consts so it can't drift.
    name: &'static str,
    /// A `required` file missing means an incomplete inference set — [`BestKeeper::snapshot`]
    /// refuses to write a `best/` that would be marked valid yet missing it. The inference
    /// set the demo loads (brain + the two normalizers) and the shape sidecar are required;
    /// the optimizer (warm-resume only, absent on a cold-resumed run) and the tick odometer
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
    BestFile { name: NORMALIZER_FILENAME, required: true },
    BestFile { name: RETURN_NORMALIZER_FILENAME, required: true },
    BestFile { name: SHAPE_FILENAME, required: true },
    BestFile { name: OPTIMIZER_FILENAME, required: false },
    BestFile { name: TICK_WATERMARK_FILENAME, required: false },
    BestFile { name: BRAIN_FILENAME, required: true },
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

/// A smoothed reach fraction, in `0..=1` and finite by construction. The newtype exists for
/// one load-bearing reason: a NaN reach must never reach [`Competence::beats`]'s floor
/// comparison, where `NaN < SOLID_REACH_FRACTION` is `false` and would let a corrupt-metric
/// checkpoint pass the collapse guard and be accepted as "best". Rejecting non-finite at
/// construction makes that unrepresentable. No `PartialOrd` derive on purpose: all ordering
/// goes through [`Competence::beats`]'s explicit `.get()` float comparisons, and an implicit
/// `<` on the newtype would re-open the very float-comparison footgun it exists to discipline.
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
}

/// The far edge of the target band a policy was rolled at (metres), finite and non-negative
/// by construction. The band is now a FIXED full-arena range, so this is constant across
/// iters and the competence order reduces to best-by-reach in practice — but a newtype still
/// rules out a NaN/inverted band corrupting [`Competence::beats`]. No `PartialOrd` derive,
/// for the same reason as [`Reach`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct Band(f32);

impl Band {
    /// `Some` iff `v` is finite and non-negative; `None` otherwise.
    fn new(v: f32) -> Option<Self> {
        (v.is_finite() && v >= 0.0).then_some(Self(v))
    }
    fn get(self) -> f32 {
        self.0
    }
}

/// A policy's demonstrated competence: the band it was rolled at and its smoothed reach
/// there. Ordered so "more capable" is unambiguous (see module docs). The band is now fixed,
/// so it is constant and the order reduces to best-by-reach. Both fields are finite-range
/// newtypes, so a corrupt metric can't construct a `Competence` that defeats the guards.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Competence {
    band: Band,
    reach: Reach,
}

impl Competence {
    /// Does `self` demonstrate strictly more competence than `other`? A farther mastered
    /// band wins; at the same band a higher reach (by [`REACH_MARGIN`]) wins; a nearer band
    /// never wins. Gated by the solid-reach floor so a collapse can't promote at any band.
    /// Sound against a corrupt metric because [`Reach`]/[`Band`] are finite by construction.
    fn beats(&self, other: &Competence) -> bool {
        if self.reach.get() < SOLID_REACH_FRACTION {
            return false;
        }
        if self.band.get() > other.band.get() + BAND_EPS {
            return true;
        }
        if (self.band.get() - other.band.get()).abs() <= BAND_EPS {
            return self.reach.get() > other.reach.get() + REACH_MARGIN;
        }
        false
    }

    /// Parse the `best/competence.txt` sidecar (`"<band_max> <reach>"`); `None` on a missing,
    /// unreadable, malformed, or out-of-range/non-finite file (a fresh `best/`, or a corrupt
    /// one — both treated as "no prior best", which the seed path then re-gates by reach).
    fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut it = text.split_whitespace();
        let band = Band::new(it.next()?.parse().ok()?)?;
        let reach = Reach::new(it.next()?.parse().ok()?)?;
        Some(Self { band, reach })
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
                c.band.get(),
                c.reach.get()
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
        let Some(smoothed) = self.smoothed_reach else {
            return;
        };
        // Reject a non-finite/out-of-range metric here rather than let it slip past the floor
        // guard (NaN < FLOOR is false). A corrupt reach or band means no eligible candidate.
        let (Some(reach), Some(band)) = (Reach::new(smoothed), Band::new(band_max)) else {
            warn!(
                "[best] non-finite competence (reach={smoothed}, band={band_max}) — not eligible for snapshot"
            );
            return;
        };
        let candidate = Competence { band, reach };

        let beats = match self.best {
            Some(best) => candidate.beats(&best),
            // No prior best: seed on the first policy that clears the solid-reach floor.
            None => reach.get() >= SOLID_REACH_FRACTION,
        };
        if !beats {
            return;
        }

        match self.snapshot(candidate) {
            Ok(()) => {
                info!(
                    "[best] new best snapshot → {}/{BEST_SUBDIR} | band {:.1}m reach {:.3} (was {})",
                    self.checkpoint_dir.display(),
                    candidate.band.get(),
                    candidate.reach.get(),
                    self.best
                        .map(|c| format!("band {:.1}m reach {:.3}", c.band.get(), c.reach.get()))
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
    /// [`BEST_FILES`]), then the competence sidecar last of all.
    ///
    /// FAIL-LOUD: a missing REQUIRED source (the brain, the two normalizers, the shape
    /// sidecar — see [`BestFile`]) aborts the whole snapshot with an error BEFORE any file is
    /// touched, so `best/` is never left marked-valid-but-incomplete (the trap when `best/`
    /// is the demo's collapse-proof fallback). Optional files (optimizer, tick odometer) are
    /// skipped if absent. The caller logs the error and keeps the previous `best/`.
    fn snapshot(&self, competence: Competence) -> std::io::Result<()> {
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
        let sidecar = best_dir.join(COMPETENCE_SIDECAR);
        let tmp = best_dir.join(format!("{COMPETENCE_SIDECAR}.tmp"));
        std::fs::write(
            &tmp,
            format!("{} {}\n", competence.band.get(), competence.reach.get()),
        )?;
        super::fsync_rename(&tmp, &sidecar)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comp(band: f32, reach: f32) -> Competence {
        Competence {
            band: Band::new(band).expect("test band in range"),
            reach: Reach::new(reach).expect("test reach in range"),
        }
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
        assert_eq!(c.band.get(), 9.0);

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

    #[test]
    fn nan_reach_never_snapshots() {
        // The bug this guards: a NaN reach passes `reach < FLOOR` (NaN comparisons are false),
        // so without the finite-range newtype a corrupt metric could be accepted as "best".
        let dir = scratch_ckpt("nan");
        let mut k = BestKeeper::new(&dir);
        for _ in 0..MIN_EMA_UPDATES + 5 {
            k.observe(Some(f32::NAN), 9.0);
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
            k.observe(Some(0.7), 9.0);
        }
        let best_brain = dir.join(BEST_SUBDIR).join("brain.bin");
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v1", "seeded");

        // Drop a required source (a normalizer) and improve the brain on disk; a snapshot
        // attempt must error and leave best/ untouched (old brain, no torn set).
        std::fs::remove_file(dir.join("normalizer.bin")).unwrap();
        std::fs::write(dir.join("brain.bin"), b"brain-v2").unwrap();
        let competence = Competence {
            band: Band::new(9.0).unwrap(),
            reach: Reach::new(1.0).unwrap(),
        };
        assert!(
            k.snapshot(competence).is_err(),
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
            k.snapshot(competence).is_ok(),
            "an absent optional optimizer must not block a snapshot"
        );
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v2");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
