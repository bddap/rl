use std::path::Path;

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;

pub mod algorithm;
pub mod bench;
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

/// Write `bytes` to `path` atomically: a sibling temp file then a rename, so a crash
/// mid-write leaves the previous file intact rather than a torn one. The overnight
/// trainer is killed and resumed, so a torn checkpoint would be silently discarded on
/// load and the run would resume from random weights. A crate-wide primitive (the
/// normalizer, curriculum, and optimizer-state writers all share it), so it lives here
/// rather than in any one persistence module.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}
