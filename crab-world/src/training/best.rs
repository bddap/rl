
use std::path::{Path, PathBuf};

use tracing::{info, warn};

use super::checkpoint::{
    BRAIN_FILENAME, NORMALIZER_FILENAME, OPTIMIZER_FILENAME, RETURN_NORMALIZER_FILENAME,
    TICK_WATERMARK_FILENAME,
};
use super::targets::SOLID_REACH_FRACTION;

const BEST_SUBDIR: &str = "best";

const REACH_SIDECAR: &str = "reach.txt";

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
];

const REACH_EMA_ALPHA: f32 = 0.05;

const REACH_MARGIN: f32 = 0.02;

const MIN_EMA_UPDATES: u32 = 30;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Reach(f32);

impl Reach {
    fn new(v: f32) -> Option<Self> {
        (v.is_finite() && (0.0..=1.0).contains(&v)).then_some(Self(v))
    }

    fn get(self) -> f32 {
        self.0
    }

    fn is_solid(self) -> bool {
        self.0 >= SOLID_REACH_FRACTION
    }

    fn beats(self, other: Reach) -> bool {
        self.is_solid() && self.0 > other.0 + REACH_MARGIN
    }

    fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Reach::new(text.split_whitespace().next()?.parse().ok()?)
    }
}

pub(crate) struct BestKeeper {
    checkpoint_dir: PathBuf,
    smoothed_reach: Option<f32>,
    ema_updates: u32,
    best: Option<Reach>,
}

impl BestKeeper {
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
        let Some(candidate) = Reach::new(smoothed) else {
            warn!("[best] non-finite reach ({smoothed}) — not eligible for snapshot");
            return;
        };

        let beats = match self.best {
            Some(best) => candidate.beats(best),
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

    fn snapshot(&self, reach: Reach) -> std::io::Result<()> {
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
                continue;
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
        assert!(!reach(0.805).beats(reach(0.80)));
    }

    #[test]
    fn collapse_never_promotes() {
        assert!(!reach(0.3).beats(reach(0.0)));
        assert!(!reach(0.59).beats(reach(0.10)));
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
    fn warmup_then_snapshot_then_immune_to_collapse() {
        let dir = scratch_ckpt("lifecycle");
        let mut k = BestKeeper::new(&dir);
        let best_brain = dir.join(BEST_SUBDIR).join("brain.bin");

        for _ in 0..MIN_EMA_UPDATES - 1 {
            k.observe(Some(0.7));
        }
        assert!(!best_brain.exists(), "no snapshot during warmup");

        k.observe(Some(0.7));
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v1");
        let r = Reach::load(&dir.join(BEST_SUBDIR).join(REACH_SIDECAR)).unwrap();
        assert!((r.get() - 0.7).abs() < 1e-6);

        std::fs::write(dir.join("brain.bin"), b"brain-collapsed").unwrap();
        for _ in 0..200 {
            k.observe(Some(0.0));
        }
        assert_eq!(
            std::fs::read(&best_brain).unwrap(),
            b"brain-v1",
            "collapse must not overwrite the best snapshot"
        );

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
        let mut k1 = BestKeeper::new(&dir);
        for _ in 0..MIN_EMA_UPDATES {
            k1.observe(Some(1.0));
        }
        assert!(dir.join(BEST_SUBDIR).join(REACH_SIDECAR).exists());

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
        let dir = scratch_ckpt("missing-required");
        let mut k = BestKeeper::new(&dir);
        for _ in 0..MIN_EMA_UPDATES {
            k.observe(Some(0.7));
        }
        let best_brain = dir.join(BEST_SUBDIR).join("brain.bin");
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v1", "seeded");

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

        std::fs::write(dir.join("normalizer.bin"), b"norm").unwrap();
        assert!(
            k.snapshot(reach(1.0)).is_ok(),
            "an absent optional optimizer must not block a snapshot"
        );
        assert_eq!(std::fs::read(&best_brain).unwrap(), b"brain-v2");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
