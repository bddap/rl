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
    let aside = parent.join(format!(".{}.aside", name.to_string_lossy()));
    // Crash recovery, in dependency order. An aside dir exists only if a FALLBACK swap
    // (no RENAME_EXCHANGE) crashed mid-sequence: if the target vanished with it, the
    // aside IS the previous live generation — restore it rather than let this save
    // build a fresh target that silently drops the non-staged entries (`best/`, the
    // watermarks). Then discard any leftover staging: entries a crashed generation
    // wrote but this one doesn't must never ride into the new set.
    if aside.exists() {
        if target.exists() {
            std::fs::remove_dir_all(&aside)?;
        } else {
            std::fs::rename(&aside, target)?;
        }
    }
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
        exchange_paths(&staging, target, &aside)?;
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
/// rename-aside fallback (through the caller's `aside` path, so the caller's crash
/// sweep can recover it) where the filesystem lacks `RENAME_EXCHANGE`.
fn exchange_paths(a: &Path, b: &Path, aside: &Path) -> std::io::Result<()> {
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
        // a reader can see `b` absent between the two renames (and a crash there
        // leaves `b` absent until the next call's aside sweep restores it), but a
        // torn set is still impossible.
        Some(libc::ENOSYS) | Some(libc::EINVAL) | Some(libc::ENOTSUP) => {
            std::fs::rename(b, aside)?;
            std::fs::rename(a, b)?;
            std::fs::rename(aside, a)?;
            Ok(())
        }
        _ => Err(err),
    }
}

/// Hardlink every entry of `from` that `to` lacks, recursing into subdirectories —
/// how a set writer carries the live dir's NON-set entries (`best/`, the tick
/// watermark, the σ-anneal epoch) into the staged generation without copying bytes.
/// Entries the stage already wrote are never overwritten. Sound only because every
/// writer in this crate replaces files whole by rename ([`atomic_write`] / the
/// envelope writers), never mutates in place — an in-place write through one link
/// would mutate the carried generation through the shared inode.
///
/// Writer debris (`*.tmp` from an interrupted [`atomic_write`], a crashed swap's
/// staging/aside dirs) is skipped: carrying it would immortalize into every future
/// generation what used to be transient garbage. ONE exception: a `.X.aside` whose
/// `X` is ABSENT is the sole surviving copy of `X` — a NESTED fallback swap (e.g.
/// `best/` inside the checkpoint dir) crashed between its renames, and
/// [`replace_dir_atomically`]'s own sweep only recovers asides for its OWN target —
/// so it is carried AS `X`, or a parent-dir swap would destroy the incumbent with
/// the retired generation.
pub(crate) fn hardlink_missing_entries(from: &Path, to: &Path) -> std::io::Result<()> {
    if !from.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let name = entry.file_name();
        let lossy = name.to_string_lossy();
        let carry_as = match lossy
            .strip_prefix('.')
            .and_then(|s| s.strip_suffix(".aside"))
        {
            // An aside shadowed by a live `X` is debris; an unshadowed one IS `X`.
            Some(orig) if !from.join(orig).exists() => std::ffi::OsString::from(orig),
            Some(_) => continue,
            None if lossy.ends_with(".tmp") || lossy.ends_with(".staging") => continue,
            None => name,
        };
        let dst = to.join(carry_as);
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
        std::fs::write(live.join("optimizer.tmp"), b"interrupted write").unwrap();
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
        assert!(
            !staging.join("optimizer.tmp").exists(),
            "writer debris must stay transient, never immortalized into the next generation"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A NESTED dir's crashed fallback swap (e.g. `best/` inside the checkpoint dir)
    /// leaves `.best.aside` as the only copy of the incumbent; the parent-dir carry
    /// must resurrect it as `best/` — treating it as debris would let the parent swap
    /// destroy the incumbent with the retired generation. A shadowed aside (the
    /// original also present) IS debris and stays behind.
    #[test]
    fn unshadowed_nested_aside_is_carried_as_the_original() {
        let root = scratch("nested-aside");
        let live = root.join("live");
        let staging = root.join("staging");
        std::fs::create_dir_all(live.join(".best.aside")).unwrap();
        std::fs::write(live.join(".best.aside/brain.bin"), b"incumbent").unwrap();
        std::fs::write(live.join("ticks.txt"), b"1").unwrap();
        std::fs::write(live.join(".ticks.txt.aside"), b"stale").unwrap();
        std::fs::create_dir(&staging).unwrap();

        hardlink_missing_entries(&live, &staging).unwrap();
        assert_eq!(
            std::fs::read(staging.join("best/brain.bin")).unwrap(),
            b"incumbent",
            "the sole surviving copy of best/ is carried as best/"
        );
        assert!(!staging.join(".best.aside").exists());
        assert_eq!(std::fs::read(staging.join("ticks.txt")).unwrap(), b"1");
        assert!(
            !staging.join(".ticks.txt.aside").exists(),
            "an aside shadowed by its live original is debris"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The #238 headline property is that the live path never goes absent — pinned at
    /// the syscall seam: after an exchange BOTH paths still exist, contents swapped.
    /// A regression to remove-then-rename (which has an absent window by construction)
    /// leaves `a` gone and fails here.
    #[test]
    fn exchange_leaves_both_paths_present_with_contents_swapped() {
        let root = scratch("exchange");
        let a = root.join("a");
        let b = root.join("b");
        std::fs::create_dir(&a).unwrap();
        std::fs::create_dir(&b).unwrap();
        std::fs::write(a.join("who.txt"), b"was-a").unwrap();
        std::fs::write(b.join("who.txt"), b"was-b").unwrap();

        exchange_paths(&a, &b, &root.join(".b.aside")).unwrap();
        assert_eq!(std::fs::read(a.join("who.txt")).unwrap(), b"was-b");
        assert_eq!(std::fs::read(b.join("who.txt")).unwrap(), b"was-a");
        assert!(
            !root.join(".b.aside").exists(),
            "the aside is fallback-only scratch, empty after a completed swap"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A fallback swap that crashed between its renames leaves the live dir ABSENT and
    /// the old generation at the aside path; the next save must restore it first —
    /// otherwise the caller's carry (`best/`, watermarks) reads an empty live dir and
    /// silently drops them from the new generation.
    #[test]
    fn crashed_fallback_swap_is_recovered_from_the_aside() {
        let root = scratch("aside-recovery");
        let target = root.join("ckpt");
        let aside = root.join(".ckpt.aside");
        std::fs::create_dir(&aside).unwrap();
        write_set(&aside, "old");
        std::fs::write(aside.join("ticks.txt"), b"999").unwrap();

        replace_dir_atomically(&target, |staging| {
            write_set(staging, "new");
            assert_eq!(
                std::fs::read(target.join("ticks.txt")).unwrap(),
                b"999",
                "the restored generation is visible to the stage for carry-over"
            );
            hardlink_missing_entries(&target, staging)
        })
        .unwrap();
        assert_eq!(std::fs::read(target.join("brain.bin")).unwrap(), b"new-brain.bin");
        assert_eq!(std::fs::read(target.join("ticks.txt")).unwrap(), b"999");
        assert!(!aside.exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
