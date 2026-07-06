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
