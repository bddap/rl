use std::path::{Path, PathBuf};

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;

pub mod algorithm;
pub mod best;
pub mod checkpoint;
pub mod curriculum;
#[cfg(feature = "wgpu")]
pub mod gpu;
pub mod inproc;
pub mod normalizer;
pub mod reward;
pub mod systems;
pub mod update;

/// Backend type aliases — the foundational training types every other module builds on.
/// Training carries autodiff; inference (play/demo) uses the bare inner backend, and
/// `AutodiffModule::valid()` converts one to the other.
pub type TrainBackend = Autodiff<NdArray>;
pub type InferBackend = NdArray;

/// Durably move `tmp` onto `dst`: fsync the file's data, rename, then fsync the parent
/// directory so the rename entry itself is on stable storage. Without the file fsync a
/// power loss (real on bothouse's solar supply) can make the rename durable while the
/// data blocks are not — yielding a torn/zero-length FINAL checkpoint, not just a torn
/// temp. The dir fsync persists the rename so the new name can't be lost either. The ONE
/// crash-safe finalize behind [`atomic_write`] and the brain/tick writers (which produce
/// their temp by other means), so every checkpoint artifact gets the same guarantee from
/// one place. A process crash (no fsync needed) was already safe; this adds the power-loss
/// case.
pub(crate) fn fsync_rename(tmp: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::File::open(tmp)?.sync_all()?;
    std::fs::rename(tmp, dst)?;
    let dir = match dst.parent() {
        Some(p) if !p.as_os_str().is_empty() => PathBuf::from(p),
        _ => PathBuf::from("."),
    };
    std::fs::File::open(dir)?.sync_all()
}

/// Write `bytes` to `path` atomically and durably: a sibling temp file, fsync, then a
/// rename (see [`fsync_rename`]). A crash — process OR power loss — leaves the previous
/// file intact rather than a torn one. The overnight trainer is killed and resumed, so a
/// torn checkpoint would be silently discarded on load and the run would resume from
/// random weights. A crate-wide primitive (the normalizer and optimizer-state writers all
/// share it), so it lives here rather than in any one persistence module.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    fsync_rename(&tmp, path)
}
