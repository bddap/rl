// The trainer subsystem is UNREACHABLE in an isolated `render` build of crab-world (rl-demo /
// game link only the inference slice — load a checkpoint, drive the policy — never the
// optimizer/rollout/best-keeper), so `cargo clippy -p crab-world --features render` flags all
// of `training::*` as dead_code. It is NOT dead: the headless trainer lives in the `rl-train`
// binary, which drives this subsystem via its pub API (`training::inproc::run_learner`, …).
//
// It can't be `cfg`-gated out under `render`: the workspace resolver unifies features, so the
// canonical `cargo clippy --all-targets` builds crab-world render-ON (from rl-demo/game/net)
// and rl-train links THAT same build — a `#[cfg(not(feature = "render"))]` on these modules
// would make rl-train fail to find them. So scope a render-only dead_code allow to exactly
// this subsystem (it propagates to every `training::*` submodule). Dead code anywhere else in
// a render build — play, crab_view, sky, the inference path — is still caught. In the unified
// workspace build the items are live (rl-train uses them), so the allow is a harmless no-op.
#![cfg_attr(feature = "render", allow(dead_code, unused_imports, unused_variables))]

use std::path::{Path, PathBuf};

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;

pub mod algorithm;
pub mod best;
pub mod checkpoint;
pub mod curriculum;
pub(crate) mod envelope;
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
