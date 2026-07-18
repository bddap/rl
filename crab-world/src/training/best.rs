use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tracing::{error, info, warn};

use super::checkpoint::{
    BRAIN_FILENAME, NORMALIZER_FILENAME, OPTIMIZER_FILENAME, RETURN_NORMALIZER_FILENAME,
    TICK_WATERMARK_FILENAME,
};
use crate::eval::{DEFAULT_EVAL_TICKS, DEFAULT_TARGET_DISTANCE_M, EvalReport};
use crate::mesh_fallback::BodyGate;

const BEST_SUBDIR: &str = "best";

/// The min-over-bearings chase-progress bar (rl#239). The sidecar name IS the metric
/// version: a `best/` stamped under a retired metric carries none of THIS sidecar, so
/// the score-incumbent-first machinery re-baselines it under the current eval before
/// any candidate is considered — never a new-metric candidate judged against an
/// old-metric bar.
const PROGRESS_SIDECAR: &str = "min_progress.txt";

/// Sidecars of retired metrics — `reach.txt` (pre-#233 reach-keyed gate) and
/// `progress.txt` (the +X-only chase eval, rl#239) — removed whenever the bar is
/// (re)stamped so a `best/` never carries two competing scores.
const LEGACY_SIDECARS: &[&str] = &["reach.txt", "progress.txt"];

struct BestFile {
    name: &'static str,
    required: bool,
}

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
    // The plant the brain trained on (rl#268) — without it an eval of `best/` would
    // silently judge a damped policy on the default plant. Optional: absent means the
    // default plant (every pre-sidecar checkpoint).
    BestFile {
        name: crate::bot::body::PLANT_FILENAME,
        required: false,
    },
];

/// How often to spend one chase-eval scoring the live checkpoint. Periodic rather than
/// reach-triggered: a trigger derived from near-heavy TRAIN episodes is blind to
/// far-approach movement in either direction — exactly the divergence that let a
/// 6.92 m brain displace the 8.93 m one (bddap/rl#233). The compass eval runs one
/// episode per bearing at the far distance (rl#239, ~2 min measured on the training
/// box), the same compass again for the rl#252 close probe, plus the short rl#280
/// pace-probe compass (~10 s episodes; ~5 min total), so with the period at 3× the
/// pre-compass 600 s, eval spend stays under ~17% of training wall-clock (rollout
/// threads idle while the learner-thread eval runs).
const EVAL_PERIOD: Duration = Duration::from_secs(1800);

/// Meters of chase progress a candidate must add over the incumbent to displace it.
/// The eval is deterministic per brain, so this only suppresses churn from
/// meaningfully-equal policies, not noise.
const PROGRESS_MARGIN_M: f32 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Progress(f32);

impl Progress {
    fn new(v: f32) -> Option<Self> {
        (v.is_finite() && v >= 0.0).then_some(Self(v))
    }

    fn get(self) -> f32 {
        self.0
    }

    fn beats(self, other: Progress) -> bool {
        self.0 > other.0 + PROGRESS_MARGIN_M
    }

    fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Progress::new(text.split_whitespace().next()?.parse().ok()?)
    }
}

/// The chase-eval seam. Production is [`crate::eval::run_eval`] — THE far-ball metric
/// every gate shares (bddap/bothouse#134); tests inject canned reports so gate
/// decisions are testable without a bevy world per case.
type Evaluator = Box<dyn FnMut(&Path) -> Result<EvalReport, String>>;

pub(crate) struct BestKeeper {
    checkpoint_dir: PathBuf,
    evaluate: Evaluator,
    eval_period: Duration,
    /// `None` until the first eval so it fires on the first call: a restart-looping
    /// trainer must not accumulate unevaluated 10-minute dead zones, and a legacy
    /// (reach-era) `best/` gets its progress bar established promptly, not after a
    /// full period of exposure.
    last_eval: Option<Instant>,
    best: Option<Progress>,
}

impl BestKeeper {
    pub(crate) fn new(checkpoint_dir: &Path, body_gate: BodyGate) -> Self {
        Self::with_evaluator(
            checkpoint_dir,
            EVAL_PERIOD,
            Box::new(move |dir| {
                // The CLI's own defaults (`rl-train eval`), so the gate judges the
                // IDENTICAL episode the release gate and rl-eval-monitor judge. Safe
                // in-process only because the trainer already ran the identical
                // thread-pool pin at boot (`init_process_pools`), making run_eval's
                // own pin a pure read.
                crate::eval::run_eval(
                    body_gate,
                    dir,
                    DEFAULT_EVAL_TICKS,
                    DEFAULT_TARGET_DISTANCE_M,
                )
            }),
        )
    }

    fn with_evaluator(checkpoint_dir: &Path, eval_period: Duration, evaluate: Evaluator) -> Self {
        let best = Progress::load(&checkpoint_dir.join(BEST_SUBDIR).join(PROGRESS_SIDECAR));
        match best {
            Some(p) => info!(
                "[best] keeping best-by-chase-eval in {}/{BEST_SUBDIR} | resumed best progress {:.3} m",
                checkpoint_dir.display(),
                p.get()
            ),
            None => info!(
                "[best] keeping best-by-chase-eval in {}/{BEST_SUBDIR} | no progress bar yet — \
                 an existing best/ is scored before any candidate can displace it",
                checkpoint_dir.display()
            ),
        }
        Self {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            evaluate,
            eval_period,
            last_eval: None,
            best,
        }
    }

    /// Run the evaluator with a panic firewall: the eval builds a whole physics world
    /// in the learner thread, and a panic there must degrade to a skipped stamp — not
    /// take down the live training run every period.
    fn eval_dir(&mut self, dir: &Path) -> Result<EvalReport, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.evaluate)(dir)))
            .unwrap_or_else(|_| Err("chase-eval panicked".to_string()))
    }

    /// Once per [`EVAL_PERIOD`]: chase-eval the checkpoint on disk and mirror it into
    /// `best/` iff it beats the incumbent's progress. Runs between learner iterations
    /// (rollout threads idle for the eval's ~4 min far+close sweep), so it costs wall
    /// clock only — no training data or update is touched.
    ///
    /// Returns whether the eval period elapsed this call (an eval was spent, whatever
    /// its outcome) — the learner keys its per-bearing training-reach window (rl#276)
    /// off this so both readouts cover the same train.log stretch.
    pub(crate) fn maybe_snapshot(&mut self) -> bool {
        if let Some(last) = self.last_eval
            && last.elapsed() < self.eval_period
        {
            return false;
        }
        // Reset even when nothing promotes: the period paces eval SPEND, not successes.
        self.last_eval = Some(Instant::now());

        // A best/ scored under a retired metric (reach-keyed era, or the +X-only eval)
        // carries no current-metric bar; score the incumbent FIRST so it cannot be
        // displaced unscored (the bddap/rl#233 failure). Until the incumbent scores
        // cleanly, no candidate is considered.
        if self.best.is_none()
            && self
                .checkpoint_dir
                .join(BEST_SUBDIR)
                .join(BRAIN_FILENAME)
                .exists()
            && !self.score_incumbent()
        {
            return true;
        }

        let report = match self.eval_dir(&self.checkpoint_dir.clone()) {
            Ok(r) => r,
            Err(e) => {
                warn!("[best] chase-eval of candidate failed: {e} — keeping previous best");
                return true;
            }
        };
        if !report.policy_loaded {
            warn!("[best] chase-eval loaded no policy from the live checkpoint — not eligible");
            return true;
        }
        let Some(candidate) = Progress::new(report.progress_m()) else {
            warn!(
                "[best] non-finite chase progress ({}) — not eligible for snapshot",
                report.progress_m()
            );
            return true;
        };

        let beats = match self.best {
            Some(best) => candidate.beats(best),
            None => true,
        };
        let bar = self
            .best
            .map(|p| format!("{:.3} m", p.get()))
            .unwrap_or_else(|| "none".to_string());
        // The close-probe numbers ride every candidate log line so train.log carries a
        // periodic close-range time series (the rl#250 flip's upside readout, rl#252);
        // they never feed the promotion decision. The far per-bearing vector rides too
        // (rl#276): the min headline collapses a one-bearing hole into an ambiguous
        // scalar — the vector names the failing bearing the moment it opens.
        let close = format!(
            "close probe min {:.3} m, reached {}/{}",
            report.close.worst().progress_m,
            report.close.reached_count(),
            crate::eval::EVAL_BEARINGS
        );
        // Median locale's compass (rl#293) — the one the headline min collapsed.
        let far = format!("far bearings {}", report.median_far().progress_line());
        if !beats {
            info!(
                "[best] chase-eval: min-bearing progress {:.3} m (reached={}) vs bar {bar} — keeping incumbent | {far} | {close}",
                candidate.get(),
                report.reached()
            );
            return true;
        }

        match self.snapshot(candidate) {
            Ok(()) => {
                info!(
                    "[best] new best snapshot → {}/{BEST_SUBDIR} | min-bearing chase progress {:.3} m \
                     (reached={}) beats {bar} | {far} | {close}",
                    self.checkpoint_dir.display(),
                    candidate.get(),
                    report.reached()
                );
                self.best = Some(candidate);
            }
            Err(e) => warn!(
                "[best] failed to snapshot best checkpoint to {}/{BEST_SUBDIR}: {e} — keeping previous best",
                self.checkpoint_dir.display()
            ),
        }
        true
    }

    /// Establish the progress bar for a pre-existing `best/` by chase-evaling it in
    /// place. Returns whether the bar now exists; on any failure the incumbent stays
    /// protected (no candidate is scored this round, retried next period).
    fn score_incumbent(&mut self) -> bool {
        let best_dir = self.checkpoint_dir.join(BEST_SUBDIR);
        // error!, not warn!: a persistently unscorable incumbent (e.g. a rig-mismatched
        // brain after a rig change) wedges promotion entirely — that must surface on
        // fleet-error, not scroll by at warn level every period.
        let report = match self.eval_dir(&best_dir) {
            Ok(r) => r,
            Err(e) => {
                error!("[best] chase-eval of incumbent best/ failed: {e} — protecting it unscored");
                return false;
            }
        };
        let scored = Progress::new(report.progress_m()).filter(|_| report.policy_loaded);
        let Some(progress) = scored else {
            error!(
                "[best] incumbent best/ produced no usable score (policy_loaded={}, progress={}) \
                 — protecting it unscored",
                report.policy_loaded,
                report.progress_m()
            );
            return false;
        };
        // The DISK sidecar is the durable bar; if it can't be stamped, stay unscored and
        // retry next period (the eval is deterministic, so the re-score is free of drift).
        if let Err(e) = write_progress_sidecar(&best_dir, progress) {
            error!("[best] failed to stamp incumbent progress sidecar: {e}");
            return false;
        }
        info!(
            "[best] incumbent best/ scored: min-bearing chase progress {:.3} m (reached={}) — bar established",
            progress.get(),
            report.reached()
        );
        self.best = Some(progress);
        true
    }

    /// Mirror the live set into `best/` as ONE staged generation swapped in atomically
    /// ([`super::replace_dir_atomically`], bddap/rl#238): a concurrent reader of
    /// `best/` sees the old or the new complete set, never a mix; a missing REQUIRED
    /// source or any copy failure aborts the swap with the incumbent untouched; and an
    /// optional file from an earlier generation never survives into a new one (the
    /// staging starts empty), so `best/` is one generation, never e.g. an old
    /// optimizer paired with a new brain.
    fn snapshot(&self, progress: Progress) -> std::io::Result<()> {
        super::replace_dir_atomically(&self.checkpoint_dir.join(BEST_SUBDIR), |staging| {
            for f in BEST_FILES {
                let src = self.checkpoint_dir.join(f.name);
                match std::fs::copy(&src, staging.join(f.name)) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound && !f.required => {}
                    Err(e) => {
                        return Err(std::io::Error::new(
                            e.kind(),
                            format!("staging {} from {}: {e}", f.name, src.display()),
                        ));
                    }
                }
            }
            write_progress_sidecar(staging, progress)
        })
    }
}

fn write_progress_sidecar(best_dir: &Path, progress: Progress) -> std::io::Result<()> {
    for legacy in LEGACY_SIDECARS {
        let _ = std::fs::remove_file(best_dir.join(legacy));
    }
    super::atomic_write(
        &best_dir.join(PROGRESS_SIDECAR),
        format!("{}\n", progress.get()).as_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;

    fn progress(v: f32) -> Progress {
        Progress::new(v).expect("test progress in range")
    }

    fn report(progress_m: f32, policy_loaded: bool) -> EvalReport {
        let bearing = crate::eval::BearingReport {
            bearing_rad: 0.0,
            progress_m,
            total_torque: 0.0,
            mean_torque_per_tick: 0.0,
            saturation: 0.0,
            work_j: 0.0,
            initial_distance_m: DEFAULT_TARGET_DISTANCE_M,
            closest_distance_m: DEFAULT_TARGET_DISTANCE_M - progress_m,
            final_distance_m: DEFAULT_TARGET_DISTANCE_M - progress_m,
            closest_tip_distance_m: f32::INFINITY,
            reached: false,
            active_ticks: DEFAULT_EVAL_TICKS,
            sustained_pace_m_per_s: 0.0,
        };
        // The close probe is scripted to beat every bar in the suite (100 m — canned
        // reports needn't be physically plausible), so each no-promotion test below
        // doubles as proof the sidecar never rescues a candidate: if close numbers
        // ever leaked into the promotion decision, the regression/collapse/NaN cases
        // would start promoting and fail.
        let close_bearing = crate::eval::BearingReport {
            progress_m: 100.0,
            initial_distance_m: 100.0,
            closest_distance_m: 0.0,
            final_distance_m: 0.0,
            reached: true,
            ..bearing
        };
        EvalReport {
            policy_loaded,
            far: [crate::eval::CompassSweep {
                target_distance_m: DEFAULT_TARGET_DISTANCE_M,
                per_bearing: [bearing; crate::eval::EVAL_BEARINGS],
            }; crate::eval::EVAL_LOCALES],
            close: crate::eval::CompassSweep {
                target_distance_m: crate::eval::CLOSE_PROBE_DISTANCE_M,
                per_bearing: [close_bearing; crate::eval::EVAL_BEARINGS],
            },
            // Scripted to beat every bar for the same reason as the close probe: a
            // pace number leaking into promotion would flip the cases below.
            pace: crate::eval::CompassSweep {
                target_distance_m: crate::eval::PACE_PROBE_DISTANCE_M,
                per_bearing: [close_bearing; crate::eval::EVAL_BEARINGS],
            },
        }
    }

    /// Keeper whose evaluator returns the current value of `score` — `Err` for a
    /// negative sentinel — and counts calls, tagged by whether it scored `best/`.
    fn scripted_keeper(
        dir: &Path,
        score: Rc<RefCell<Result<EvalReport, String>>>,
        calls: Rc<RefCell<Vec<bool>>>,
    ) -> BestKeeper {
        BestKeeper::with_evaluator(
            dir,
            Duration::ZERO,
            Box::new(move |d| {
                calls.borrow_mut().push(d.ends_with(BEST_SUBDIR));
                score.borrow().clone()
            }),
        )
    }

    #[test]
    fn progress_rejects_non_finite_and_negative() {
        assert!(Progress::new(f32::NAN).is_none());
        assert!(Progress::new(f32::INFINITY).is_none());
        assert!(Progress::new(-0.1).is_none());
        assert!(Progress::new(0.0).is_some());
        assert!(Progress::new(8.93).is_some());
    }

    #[test]
    fn higher_progress_needs_margin() {
        assert!(progress(8.93).beats(progress(6.92)));
        assert!(!progress(6.93).beats(progress(6.92)));
        assert!(!progress(6.92).beats(progress(8.93)));
    }

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
    fn seeds_then_rejects_regression_then_ratchets() {
        let dir = scratch_ckpt("lifecycle");
        let score = Rc::new(RefCell::new(Ok(report(5.0, true))));
        let mut k = scripted_keeper(&dir, score.clone(), Rc::default());
        let best_brain = dir.join(BEST_SUBDIR).join("brain.bin");

        k.maybe_snapshot();
        assert_eq!(
            std::fs::read(&best_brain).unwrap(),
            b"brain-v1",
            "first score seeds"
        );
        let bar = Progress::load(&dir.join(BEST_SUBDIR).join(PROGRESS_SIDECAR)).unwrap();
        assert!((bar.get() - 5.0).abs() < 1e-6);

        std::fs::write(dir.join("brain.bin"), b"brain-worse").unwrap();
        *score.borrow_mut() = Ok(report(4.0, true));
        k.maybe_snapshot();
        assert_eq!(
            std::fs::read(&best_brain).unwrap(),
            b"brain-v1",
            "a lower chase-eval must not displace the best"
        );

        std::fs::write(dir.join("brain.bin"), b"brain-v2").unwrap();
        *score.borrow_mut() = Ok(report(6.0, true));
        k.maybe_snapshot();
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v2");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn collapse_and_errors_never_promote() {
        let dir = scratch_ckpt("collapse");
        let score = Rc::new(RefCell::new(Ok(report(5.0, true))));
        let mut k = scripted_keeper(&dir, score.clone(), Rc::default());
        k.maybe_snapshot();

        std::fs::write(dir.join("brain.bin"), b"brain-collapsed").unwrap();
        let best_brain = dir.join(BEST_SUBDIR).join("brain.bin");
        for bad in [
            Ok(report(0.0, true)),
            Ok(report(f32::NAN, true)),
            Ok(report(9.0, false)),
            Err("eval exploded".to_string()),
        ] {
            *score.borrow_mut() = bad;
            k.maybe_snapshot();
            assert_eq!(
                std::fs::read(&best_brain).unwrap(),
                b"brain-v1",
                "collapse/NaN/rest-baseline/error must not overwrite the best snapshot"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_policy_never_seeds() {
        let dir = scratch_ckpt("no-policy");
        let score = Rc::new(RefCell::new(Ok(report(9.0, false))));
        let mut k = scripted_keeper(&dir, score, Rc::default());
        k.maybe_snapshot();
        assert!(
            !dir.join(BEST_SUBDIR).join("brain.bin").exists(),
            "a zero-action rest baseline must never become best"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bddap/rl#233 and rl#239 migration scenario: a `best/` scored under a retired
    /// metric (a reach.txt AND a +X-era progress.txt, no min_progress.txt) must be
    /// re-scored in place under the CURRENT eval before a weaker candidate can be
    /// considered — and must survive it. This re-baseline is what keeps promotions
    /// apples-to-apples across a metric change.
    #[test]
    fn legacy_best_is_scored_before_any_candidate_displaces_it() {
        let dir = scratch_ckpt("migration");
        let best_dir = dir.join(BEST_SUBDIR);
        std::fs::create_dir_all(&best_dir).unwrap();
        for (name, body) in [
            ("brain.bin", b"brain-893" as &[u8]),
            ("normalizer.bin", b"norm"),
            ("return_normalizer.bin", b"rnorm"),
            ("reach.txt", b"0.729"),
            ("progress.txt", b"8.9315"),
        ] {
            std::fs::write(best_dir.join(name), body).unwrap();
        }

        let score = Rc::new(RefCell::new(Ok(report(8.93, true))));
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut k = scripted_keeper(&dir, score.clone(), calls.clone());
        assert!(
            k.best.is_none(),
            "retired-metric sidecars carry no current-metric bar"
        );

        // Incumbent scores 8.93; the candidate (same scripted score here) then fails
        // the margin — incumbent kept, bar stamped, legacy sidecars gone.
        k.maybe_snapshot();
        assert_eq!(*calls.borrow(), vec![true, false], "incumbent scored FIRST");
        assert_eq!(
            std::fs::read(best_dir.join("brain.bin")).unwrap(),
            b"brain-893"
        );
        let bar = Progress::load(&best_dir.join(PROGRESS_SIDECAR)).unwrap();
        assert!((bar.get() - 8.93).abs() < 1e-6);
        for legacy in LEGACY_SIDECARS {
            assert!(
                !best_dir.join(legacy).exists(),
                "retired sidecar {legacy} replaced"
            );
        }

        *score.borrow_mut() = Ok(report(6.92, true));
        k.maybe_snapshot();
        assert_eq!(
            std::fs::read(best_dir.join("brain.bin")).unwrap(),
            b"brain-893",
            "the 6.92 m candidate must not displace the 8.93 m incumbent"
        );

        *score.borrow_mut() = Ok(report(9.5, true));
        k.maybe_snapshot();
        assert_eq!(
            std::fs::read(best_dir.join("brain.bin")).unwrap(),
            b"brain-v1",
            "a genuinely better candidate (the LIVE ckpt brain) finally displaces the \
             migrated incumbent"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An incumbent that fails to score is protected: nothing is evaluated as a
    /// candidate and nothing overwrites `best/` until the incumbent scores cleanly.
    #[test]
    fn unscorable_incumbent_blocks_candidates() {
        let dir = scratch_ckpt("unscorable");
        let best_dir = dir.join(BEST_SUBDIR);
        std::fs::create_dir_all(&best_dir).unwrap();
        std::fs::write(best_dir.join("brain.bin"), b"brain-incumbent").unwrap();

        let score = Rc::new(RefCell::new(Err("refused".to_string())));
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut k = scripted_keeper(&dir, score, calls.clone());
        k.maybe_snapshot();
        assert_eq!(
            *calls.borrow(),
            vec![true],
            "only the incumbent was evaluated"
        );
        assert_eq!(
            std::fs::read(best_dir.join("brain.bin")).unwrap(),
            b"brain-incumbent",
            "an unscored incumbent must never be displaced"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resumes_bar_across_restart() {
        let dir = scratch_ckpt("resume");
        let score = Rc::new(RefCell::new(Ok(report(7.0, true))));
        let mut k1 = scripted_keeper(&dir, score.clone(), Rc::default());
        k1.maybe_snapshot();

        std::fs::write(dir.join("brain.bin"), b"brain-collapsed").unwrap();
        *score.borrow_mut() = Ok(report(3.0, true));
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut k2 = scripted_keeper(&dir, score, calls.clone());
        k2.maybe_snapshot();
        assert_eq!(
            *calls.borrow(),
            vec![false],
            "a stamped bar resumes without re-scoring the incumbent"
        );
        assert_eq!(
            std::fs::read(dir.join(BEST_SUBDIR).join("brain.bin")).unwrap(),
            b"brain-v1",
            "resumed bar must reject a post-restart regression"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn period_gates_eval_spend() {
        let dir = scratch_ckpt("period");
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut k = BestKeeper::with_evaluator(&dir, Duration::from_secs(3600), {
            let calls = calls.clone();
            Box::new(move |_| {
                calls.borrow_mut().push(false);
                Ok(report(5.0, true))
            })
        });
        for _ in 0..100 {
            k.maybe_snapshot();
        }
        assert_eq!(
            calls.borrow().len(),
            1,
            "the first call evals immediately (no boot dead zone); the rest wait out the period"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn evaluator_panic_is_contained() {
        let dir = scratch_ckpt("panic");
        let mut k = BestKeeper::with_evaluator(
            &dir,
            Duration::ZERO,
            Box::new(|_| panic!("physics world exploded")),
        );
        k.maybe_snapshot();
        assert!(
            !dir.join(BEST_SUBDIR).join("brain.bin").exists(),
            "a panicking eval must skip the stamp, not unwind into the learner"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_required_file_fails_loud_and_keeps_previous() {
        let dir = scratch_ckpt("missing-required");
        let score = Rc::new(RefCell::new(Ok(report(5.0, true))));
        let mut k = scripted_keeper(&dir, score.clone(), Rc::default());
        k.maybe_snapshot();
        let best_brain = dir.join(BEST_SUBDIR).join("brain.bin");
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v1", "seeded");

        std::fs::remove_file(dir.join("normalizer.bin")).unwrap();
        std::fs::write(dir.join("brain.bin"), b"brain-v2").unwrap();
        assert!(
            k.snapshot(progress(9.0)).is_err(),
            "a missing required source must fail the snapshot"
        );
        assert_eq!(
            std::fs::read(&best_brain).unwrap(),
            b"brain-v1",
            "failed snapshot must not overwrite the previous best brain"
        );

        std::fs::write(dir.join("normalizer.bin"), b"norm").unwrap();
        assert!(
            k.snapshot(progress(9.0)).is_ok(),
            "an absent optional optimizer must not block a snapshot"
        );
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v2");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
