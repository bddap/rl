//! Pre-registered read marks (bddap/rl#427). A run's `ckpt/` holds only its latest
//! set, so a read registered at a tick mark and taken after the mark passed reads a
//! later step. The learner instead keeps the set it holds when the odometer first
//! reaches each mark in `<ckpt>/marks/<mark>/`, a subdir every save carries forward
//! untouched; a reader takes that dir and gets the mark or nothing.

use std::path::{Path, PathBuf};

pub const MARKS_SUBDIR: &str = "marks";

pub fn mark_dir(checkpoint_dir: &Path, mark: u64) -> PathBuf {
    checkpoint_dir.join(MARKS_SUBDIR).join(mark.to_string())
}

pub(crate) struct ReadMarks {
    checkpoint_dir: PathBuf,
    pending: Vec<u64>,
}

impl ReadMarks {
    /// A mark the odometer already reached with no set kept refuses, even one
    /// iteration on: `ticks.txt` runs one iteration ahead of `brain.bin` between an
    /// iteration's end and the next save, so a resumed set may predate the mark.
    pub(crate) fn new(
        checkpoint_dir: &Path,
        marks: &[u64],
        total_ticks: u64,
    ) -> Result<Self, String> {
        let mut pending = Vec::new();
        for &mark in marks {
            if mark_dir(checkpoint_dir, mark).exists() || pending.contains(&mark) {
                continue;
            }
            if total_ticks >= mark {
                return Err(format!(
                    "read mark {mark} already passed (odometer {total_ticks}) with no set kept \
                     at {} — a later set is not the registered read",
                    mark_dir(checkpoint_dir, mark).display()
                ));
            }
            pending.push(mark);
        }
        Ok(Self {
            checkpoint_dir: checkpoint_dir.to_owned(),
            pending,
        })
    }

    /// Call right after a save, `total_ticks` being the odometer that set carries. A
    /// failed save or copy drops its due marks: a later save would keep a later step.
    pub(crate) fn keep_due(&mut self, total_ticks: u64, saved: bool) {
        let checkpoint_dir = &self.checkpoint_dir;
        self.pending.retain(|&mark| {
            if total_ticks < mark {
                return true;
            }
            let dir = mark_dir(checkpoint_dir, mark);
            let kept = if saved {
                super::replace_dir_atomically(&dir, |staging| {
                    super::best::stage_set(checkpoint_dir, staging)
                })
            } else {
                Err(std::io::Error::other("the checkpoint save failed"))
            };
            match kept {
                Ok(()) => eprintln!(
                    "[learner] read mark {mark}: kept the set at {total_ticks} ticks in {}",
                    dir.display()
                ),
                Err(e) => eprintln!(
                    "[learner] ERROR read mark {mark}: keeping the set at {total_ticks} ticks \
                     in {} failed: {e} — this mark has no set",
                    dir.display()
                ),
            }
            false
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::checkpoint::{
        BRAIN_FILENAME, NORMALIZER_FILENAME, RETURN_NORMALIZER_FILENAME, TICK_WATERMARK_FILENAME,
    };
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rl-marks-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn save(dir: &Path, ticks: u64) {
        for name in [
            BRAIN_FILENAME,
            NORMALIZER_FILENAME,
            RETURN_NORMALIZER_FILENAME,
        ] {
            std::fs::write(dir.join(name), format!("{name}@{ticks}")).unwrap();
        }
        std::fs::write(dir.join(TICK_WATERMARK_FILENAME), ticks.to_string()).unwrap();
    }

    fn kept_ticks(dir: &Path, mark: u64) -> String {
        std::fs::read_to_string(mark_dir(dir, mark).join(TICK_WATERMARK_FILENAME)).unwrap()
    }

    #[test]
    fn keeps_the_first_set_at_or_past_each_mark_and_never_rewrites_it() {
        let dir = scratch("keep");
        let mut marks = ReadMarks::new(&dir, &[40, 20], 0).unwrap();
        for ticks in [0, 15, 30, 45, 60] {
            save(&dir, ticks);
            marks.keep_due(ticks, true);
        }
        assert_eq!(kept_ticks(&dir, 20), "30");
        assert_eq!(kept_ticks(&dir, 40), "45");
        assert_eq!(
            std::fs::read_to_string(mark_dir(&dir, 40).join(BRAIN_FILENAME)).unwrap(),
            format!("{BRAIN_FILENAME}@45")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_resume_honours_kept_marks_and_refuses_a_passed_one() {
        let dir = scratch("resume");
        save(&dir, 30);
        ReadMarks::new(&dir, &[20], 0).unwrap().keep_due(30, true);

        let mut resumed = ReadMarks::new(&dir, &[20, 120], 100).unwrap();
        save(&dir, 130);
        resumed.keep_due(130, true);
        assert_eq!(
            kept_ticks(&dir, 20),
            "30",
            "a kept mark is never re-kept later"
        );
        assert_eq!(kept_ticks(&dir, 120), "130");

        let err = ReadMarks::new(&dir, &[20, 130], 130).err().unwrap();
        assert!(err.contains("read mark 130 already passed"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_save_or_copy_drops_the_mark() {
        let dir = scratch("fail");
        let mut marks = ReadMarks::new(&dir, &[10, 20], 0).unwrap();
        std::fs::write(dir.join(TICK_WATERMARK_FILENAME), "10").unwrap();
        marks.keep_due(10, true);
        assert!(
            !mark_dir(&dir, 10).exists(),
            "a set missing its brain is not kept"
        );
        save(&dir, 20);
        marks.keep_due(20, false);
        assert!(!mark_dir(&dir, 20).exists(), "a failed save keeps nothing");
        save(&dir, 25);
        marks.keep_due(25, true);
        for mark in [10, 20] {
            assert!(
                !mark_dir(&dir, mark).exists(),
                "a later set never fills mark {mark}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
