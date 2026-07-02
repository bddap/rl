//! One-shot legacy→envelope checkpoint migration (bddap/rl#200 §2), the `rl-train
//! migrate-checkpoint` subcommand. THE sole legacy parser in the tree: every loader reads
//! only [`super::envelope`]-tagged files, so there is no dual-read window to later close —
//! a legacy file either passes through here once or is refused everywhere. Deleted when
//! the fleet is migrated (epic increment 3).
//!
//! Each legacy artifact is VALIDATED (parsed as what it claims to be) before being
//! wrapped, tagged `arch: mlp256` — correct by definition, the only architecture that has
//! ever existed. The brain/normalizer payloads are the legacy bytes VERBATIM (the envelope
//! only wraps them), the old versioned `OptimizerCheckpoint` is UNWRAPPED (not nested),
//! and the superseded `shape.txt` sidecar is deleted. Already-tagged files are refused
//! (skipped, reported), so a rerun after a partial migration finishes the rest.

use std::path::Path;

use burn::record::{BinBytesRecorder, FullPrecisionSettings};

use crate::bot::arch::{AnyBrain, ArchId};
use crate::training::InferBackend;

use super::algorithm::ReturnNormalizerData;
use super::envelope::{ArtifactKind, is_enveloped, write_envelope};
use super::normalizer::NormalizerSnapshot;

/// The pre-envelope `optimizer.bin` format (`OptimizerCheckpoint` as it was in
/// `checkpoint.rs` before bddap/rl#200): a version tag wrapping the burn optimizer
/// record's bytes. Lives ONLY here, per the sole-legacy-parser rule.
#[derive(serde::Serialize, serde::Deserialize)]
struct LegacyOptimizerCheckpoint {
    version: u32,
    record: Vec<u8>,
}

/// The one legacy optimizer version ever written.
const LEGACY_OPTIMIZER_VERSION: u32 = 1;

/// The superseded obs/action-widths sidecar the envelope replaces (its consumers now read
/// the envelope + post-load `io_dims`); deleted by the migration.
const LEGACY_SHAPE_FILENAME: &str = "shape.txt";

/// What happened to one artifact file.
enum FileVerdict {
    Migrated,
    AlreadyTagged,
    Absent,
}

/// Human-readable per-file migration log plus the count that decides the exit code
/// (a run that migrated nothing is an operator error — wrong dir, or already done).
pub struct MigrationReport {
    pub lines: Vec<String>,
    pub migrated: usize,
}

/// Migrate every legacy checkpoint artifact in `dir` (and its `best/` snapshot, if
/// present) into tagged envelopes. Errors on the first artifact that neither carries an
/// envelope nor parses as its legacy format — wrapping unvalidated bytes would launder a
/// corrupt file into a valid-looking envelope.
pub fn migrate_dir(dir: &Path) -> Result<MigrationReport, String> {
    let mut report = MigrationReport {
        lines: Vec::new(),
        migrated: 0,
    };
    migrate_one_dir(dir, &mut report)?;
    let best = dir.join("best");
    if best.is_dir() {
        migrate_one_dir(&best, &mut report)?;
    }
    Ok(report)
}

fn migrate_one_dir(dir: &Path, report: &mut MigrationReport) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    for (name, kind) in [
        (super::checkpoint::BRAIN_FILENAME, ArtifactKind::Brain),
        (
            super::checkpoint::OPTIMIZER_FILENAME,
            ArtifactKind::Optimizer,
        ),
        (
            super::checkpoint::NORMALIZER_FILENAME,
            ArtifactKind::ObsNormalizer,
        ),
        (
            super::checkpoint::RETURN_NORMALIZER_FILENAME,
            ArtifactKind::ReturnNormalizer,
        ),
    ] {
        let path = dir.join(name);
        match migrate_file(&path, kind)? {
            FileVerdict::Migrated => {
                report.migrated += 1;
                report.lines.push(format!("{}: migrated", path.display()));
            }
            FileVerdict::AlreadyTagged => report.lines.push(format!(
                "{}: already tagged — refused (left as is)",
                path.display()
            )),
            FileVerdict::Absent => report.lines.push(format!("{}: absent", path.display())),
        }
    }
    let shape = dir.join(LEGACY_SHAPE_FILENAME);
    if shape.exists() {
        std::fs::remove_file(&shape)
            .map_err(|e| format!("failed to delete {}: {e}", shape.display()))?;
        report.lines.push(format!(
            "{}: deleted (superseded by the envelope)",
            shape.display()
        ));
    }
    Ok(())
}

fn migrate_file(path: &Path, kind: ArtifactKind) -> Result<FileVerdict, String> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(FileVerdict::Absent),
        Err(e) => return Err(format!("failed to read {}: {e}", path.display())),
    };
    if is_enveloped(&bytes) {
        return Ok(FileVerdict::AlreadyTagged);
    }

    // Validate-then-wrap. The payload for brain/normalizers is the legacy bytes VERBATIM
    // (bit-preserving — the weights the fleet trained stay exactly as they are); the
    // optimizer envelope replaces the legacy wrapper, so its payload is the inner record.
    let payload = match kind {
        ArtifactKind::Brain => {
            let device = burn::backend::ndarray::NdArrayDevice::Cpu;
            crate::training::checkpoint::decode_brain_payload::<InferBackend>(
                ArchId::Mlp256,
                bytes.clone(),
                &device,
            )
            .map_err(|e| {
                format!(
                    "{} does not parse as a legacy mlp256 brain record: {e}",
                    path.display()
                )
            })?;
            bytes
        }
        ArtifactKind::Optimizer => {
            let legacy: LegacyOptimizerCheckpoint = bincode::deserialize(&bytes).map_err(|e| {
                format!(
                    "{} does not parse as a legacy optimizer checkpoint: {e}",
                    path.display()
                )
            })?;
            if legacy.version != LEGACY_OPTIMIZER_VERSION {
                return Err(format!(
                    "{} is legacy optimizer format v{}, this tool migrates only v{LEGACY_OPTIMIZER_VERSION}",
                    path.display(),
                    legacy.version
                ));
            }
            legacy.record
        }
        ArtifactKind::ObsNormalizer => {
            bincode::deserialize::<NormalizerSnapshot>(&bytes).map_err(|e| {
                format!(
                    "{} does not parse as a legacy obs-normalizer snapshot: {e}",
                    path.display()
                )
            })?;
            bytes
        }
        ArtifactKind::ReturnNormalizer => {
            bincode::deserialize::<ReturnNormalizerData>(&bytes).map_err(|e| {
                format!(
                    "{} does not parse as a legacy return-normalizer record: {e}",
                    path.display()
                )
            })?;
            bytes
        }
    };
    write_envelope(path, kind, ArchId::Mlp256, payload)
        .map_err(|e| format!("failed to write envelope at {}: {e}", path.display()))?;
    Ok(FileVerdict::Migrated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::checkpoint::{CheckpointDir, load_brain_file, load_return_normalizer};
    use crate::training::normalizer::ObsNormalizer;
    use crate::training::{TrainBackend, algorithm::ReturnNormalizer};
    use burn::backend::ndarray::NdArrayDevice;

    /// Write a full LEGACY checkpoint set exactly as pre-envelope main did: bare leaf
    /// record, bare bincode snapshots, the versioned optimizer wrapper, and shape.txt.
    fn write_legacy_set(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::Mlp256, &device);
        let raw = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        std::fs::write(dir.join("brain.bin"), raw).unwrap();
        std::fs::write(
            dir.join("normalizer.bin"),
            bincode::serialize(&ObsNormalizer::new(5.0).snapshot()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("return_normalizer.bin"),
            bincode::serialize(&ReturnNormalizer::new().to_data()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("optimizer.bin"),
            bincode::serialize(&LegacyOptimizerCheckpoint {
                version: LEGACY_OPTIMIZER_VERSION,
                record: vec![],
            })
            .unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("shape.txt"), b"92 38\n").unwrap();
    }

    #[test]
    fn migrates_a_full_legacy_dir_that_then_loads_through_the_envelope_readers() {
        let dir = std::env::temp_dir().join(format!("rl-migrate-full-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_legacy_set(&dir);
        // A best/ snapshot rides along.
        write_legacy_set(&dir.join("best"));

        let report = migrate_dir(&dir).expect("migration succeeds");
        assert_eq!(report.migrated, 8, "four artifacts in each of the two dirs");
        assert!(!dir.join("shape.txt").exists(), "shape.txt deleted");
        assert!(!dir.join("best/shape.txt").exists());

        // Every migrated artifact loads through the production (envelope-only) readers.
        let device = NdArrayDevice::Cpu;
        for d in [&dir, &dir.join("best")] {
            let paths = CheckpointDir::new(d);
            let brain = load_brain_file::<TrainBackend>(&paths.brain_file(), &device)
                .expect("migrated brain loads");
            assert_eq!(brain.arch(), ArchId::Mlp256);
            ObsNormalizer::load(&paths.normalizer_path(), ArchId::Mlp256)
                .expect("migrated obs normalizer loads");
            load_return_normalizer(&paths.return_normalizer_path(), ArchId::Mlp256)
                .expect("migrated return normalizer loads");
        }

        // Rerunning refuses the now-tagged files and migrates nothing — idempotent.
        let rerun = migrate_dir(&dir).expect("rerun is not an error per-file");
        assert_eq!(rerun.migrated, 0, "already-tagged files are refused");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_to_launder_garbage_into_an_envelope() {
        let dir = std::env::temp_dir().join(format!("rl-migrate-garbage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("brain.bin"), b"not a burn record").unwrap();
        assert!(
            migrate_dir(&dir).is_err(),
            "an unparseable legacy brain must fail the migration, not be wrapped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
