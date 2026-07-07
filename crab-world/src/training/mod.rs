#![cfg_attr(feature = "render", allow(dead_code, unused_imports, unused_variables))]

use std::path::{Path, PathBuf};

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;

pub mod algorithm;
pub mod best;
pub mod checkpoint;
pub(crate) mod envelope;
#[cfg(feature = "wgpu")]
pub mod gpu;
pub mod inproc;
pub mod normalizer;
pub mod reward;
pub mod systems;
pub mod targets;
pub mod update;

pub type TrainBackend = Autodiff<NdArray>;
pub type InferBackend = NdArray;

pub(crate) fn fsync_rename(tmp: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::File::open(tmp)?.sync_all()?;
    std::fs::rename(tmp, dst)?;
    let dir = match dst.parent() {
        Some(p) if !p.as_os_str().is_empty() => PathBuf::from(p),
        _ => PathBuf::from("."),
    };
    std::fs::File::open(dir)?.sync_all()
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    fsync_rename(&tmp, path)
}

/// THE checkpoint-SET writer (bddap/rl#238): stage the whole set in a fresh sibling
/// dir, fsync it, then swap it into place in ONE atomic step — so a concurrent reader
/// (the release poll, the eval monitor, a manual copy) sees either the old complete
/// set or the new complete set, never a mix. Per-file [`atomic_write`] cannot give
/// that: members land one rename at a time, and a reader between renames tears the
/// set (the 2026-07-07 rl-release / rl-eval-monitor incidents). Every writer of a
/// set-shaped dir routes through here; the #215 save-stamp check stays as the
/// backstop for out-of-band copies that read member files by separate path lookups.
///
/// `stage` receives the staging dir and writes the ENTIRE new generation into it —
/// anything it does not write does not survive the swap (a caller that wants old
/// entries carried forward hardlinks them in, see [`hardlink_missing_entries`]). On
/// any error the live dir is untouched and the staging dir is removed; a leftover
/// staging dir from a crash is discarded on the next call, never merged into a new
/// generation.
///
/// The swap is `renameat2(RENAME_EXCHANGE)` — no instant where `target` is absent —
/// falling back to rename-aside + rename-in only where the filesystem lacks EXCHANGE
/// support (that fallback has a reader-visible ENOENT window, the lesser evil).
pub(crate) fn replace_dir_atomically(
    target: &Path,
    stage: impl FnOnce(&Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let name = target.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("no file name in swap target {}", target.display()),
        )
    })?;
    let parent = match target.parent() {
        Some(p) if !p.as_os_str().is_empty() => PathBuf::from(p),
        _ => PathBuf::from("."),
    };
    let staging = parent.join(format!(".{}.staging", name.to_string_lossy()));
    // A leftover from a crash mid-save MUST go before staging anew: entries a crashed
    // generation wrote but this one doesn't would otherwise ride into the new set.
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&parent)?;
    std::fs::create_dir(&staging)?;

    let staged = stage(&staging).and_then(|()| fsync_dir_deep(&staging));
    if let Err(e) = staged {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }

    if target.exists() {
        exchange_paths(&staging, target)?;
        // The old generation now sits at the staging path; its removal is cleanup, not
        // part of the swap — a failure here leaves garbage the next call's leftover
        // sweep discards, so it must not fail a save that already landed.
        let _ = std::fs::remove_dir_all(&staging);
    } else {
        std::fs::rename(&staging, target)?;
    }
    std::fs::File::open(&parent)?.sync_all()
}

/// Atomically swap the directories at `a` and `b` (same filesystem), with a
/// rename-aside fallback where the filesystem lacks `RENAME_EXCHANGE`.
fn exchange_paths(a: &Path, b: &Path) -> std::io::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let ac = std::ffi::CString::new(a.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let bc = std::ffi::CString::new(b.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let rc = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            ac.as_ptr(),
            libc::AT_FDCWD,
            bc.as_ptr(),
            libc::RENAME_EXCHANGE,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        // Kernel or filesystem without EXCHANGE — the documented lesser-evil fallback:
        // a reader can see `b` absent between the two renames, but never a torn set.
        Some(libc::ENOSYS) | Some(libc::EINVAL) | Some(libc::ENOTSUP) => {
            let aside = b.with_extension("exchange-aside");
            std::fs::rename(b, &aside)?;
            std::fs::rename(a, b)?;
            std::fs::rename(&aside, a)?;
            Ok(())
        }
        _ => Err(err),
    }
}

/// Hardlink every entry of `from` that `to` lacks, recursing into subdirectories —
/// how a set writer carries the live dir's NON-set entries (`best/`, the tick
/// watermark, the σ-anneal epoch) into the staged generation without copying bytes.
/// Entries the stage already wrote are never overwritten.
pub(crate) fn hardlink_missing_entries(from: &Path, to: &Path) -> std::io::Result<()> {
    if !from.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let dst = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            if !dst.exists() {
                std::fs::create_dir(&dst)?;
            }
            hardlink_missing_entries(&entry.path(), &dst)?;
        } else if !dst.exists() {
            std::fs::hard_link(entry.path(), &dst)?;
        }
    }
    Ok(())
}

/// fsync every file and directory under `dir` (`dir` included), so the staged
/// generation is durable BEFORE the swap makes it the live one — a crash after the
/// swap must not resurrect a half-written set. Redundant for members written via
/// [`atomic_write`], necessary for plain copies/hardlinks staged beside them.
fn fsync_dir_deep(dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            fsync_dir_deep(&entry.path())?;
        } else {
            std::fs::File::open(entry.path())?.sync_all()?;
        }
    }
    std::fs::File::open(dir)?.sync_all()
}

#[cfg(test)]
mod swap_tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rl-swap-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_set(dir: &Path, generation: &str) {
        for name in ["brain.bin", "normalizer.bin"] {
            std::fs::write(dir.join(name), format!("{generation}-{name}")).unwrap();
        }
    }

    fn read_set(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// The bddap/rl#238 invariant: while the stage writes, the LIVE dir still holds the
    /// complete OLD set — no member is touched in place — and after the swap it holds
    /// exactly the complete NEW set. A reader can therefore never observe a mix.
    #[test]
    fn live_dir_holds_a_complete_set_at_every_observable_point() {
        let root = scratch("invariant");
        let target = root.join("ckpt");
        std::fs::create_dir(&target).unwrap();
        write_set(&target, "old");

        replace_dir_atomically(&target, |staging| {
            assert_ne!(staging, target, "stage must never receive the live dir");
            assert_eq!(
                std::fs::read(target.join("brain.bin")).unwrap(),
                b"old-brain.bin",
                "old set intact while the new one stages"
            );
            assert_eq!(
                std::fs::read(target.join("normalizer.bin")).unwrap(),
                b"old-normalizer.bin"
            );
            write_set(staging, "new");
            Ok(())
        })
        .unwrap();

        assert_eq!(read_set(&target), ["brain.bin", "normalizer.bin"]);
        assert_eq!(std::fs::read(target.join("brain.bin")).unwrap(), b"new-brain.bin");
        assert_eq!(
            std::fs::read(target.join("normalizer.bin")).unwrap(),
            b"new-normalizer.bin"
        );
        assert_eq!(
            read_set(&root),
            ["ckpt"],
            "no staging/retired debris after a clean swap"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failed_stage_leaves_the_old_set_untouched() {
        let root = scratch("failed-stage");
        let target = root.join("ckpt");
        std::fs::create_dir(&target).unwrap();
        write_set(&target, "old");

        let err = replace_dir_atomically(&target, |staging| {
            // Half a set staged, then the brain write "fails" (the ENOSPC shape).
            std::fs::write(staging.join("normalizer.bin"), b"new-normalizer.bin").unwrap();
            Err(std::io::Error::other("disk full"))
        });
        assert!(err.is_err());
        assert_eq!(std::fs::read(target.join("brain.bin")).unwrap(), b"old-brain.bin");
        assert_eq!(
            std::fs::read(target.join("normalizer.bin")).unwrap(),
            b"old-normalizer.bin"
        );
        assert_eq!(read_set(&root), ["ckpt"], "failed staging is cleaned up");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn first_save_creates_the_target() {
        let root = scratch("first-save");
        let target = root.join("ckpt");
        replace_dir_atomically(&target, |staging| {
            write_set(staging, "new");
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(target.join("brain.bin")).unwrap(), b"new-brain.bin");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A crash between staging and swapping leaves a `.ckpt.staging` behind; the next
    /// save must discard it wholesale — an entry the dead generation wrote but the new
    /// one doesn't must never ride into the new set.
    #[test]
    fn leftover_staging_from_a_crash_never_pollutes_the_next_save() {
        let root = scratch("leftover");
        let target = root.join("ckpt");
        std::fs::create_dir(&target).unwrap();
        write_set(&target, "old");
        let leftover = root.join(".ckpt.staging");
        std::fs::create_dir(&leftover).unwrap();
        std::fs::write(leftover.join("stale-orphan.bin"), b"from a dead save").unwrap();

        replace_dir_atomically(&target, |staging| {
            write_set(staging, "new");
            Ok(())
        })
        .unwrap();
        assert_eq!(
            read_set(&target),
            ["brain.bin", "normalizer.bin"],
            "the dead generation's orphan must not appear in the new set"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hardlink_missing_entries_carries_non_set_entries_without_overwriting() {
        let root = scratch("carry");
        let live = root.join("live");
        let staging = root.join("staging");
        std::fs::create_dir_all(live.join("best")).unwrap();
        std::fs::write(live.join("brain.bin"), b"old-brain").unwrap();
        std::fs::write(live.join("ticks.txt"), b"123").unwrap();
        std::fs::write(live.join("best/brain.bin"), b"best-brain").unwrap();
        std::fs::create_dir(&staging).unwrap();
        std::fs::write(staging.join("brain.bin"), b"new-brain").unwrap();

        hardlink_missing_entries(&live, &staging).unwrap();
        assert_eq!(
            std::fs::read(staging.join("brain.bin")).unwrap(),
            b"new-brain",
            "a staged member is never overwritten by the carry"
        );
        assert_eq!(std::fs::read(staging.join("ticks.txt")).unwrap(), b"123");
        assert_eq!(std::fs::read(staging.join("best/brain.bin")).unwrap(), b"best-brain");
        let _ = std::fs::remove_dir_all(&root);
    }
}
