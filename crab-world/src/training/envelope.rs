//! The tagged checkpoint envelope (bddap/rl#200 §2): every persisted training artifact —
//! brain, optimizer, obs normalizer, return normalizer — ships inside a
//! [`CheckpointEnvelope`] that names its artifact kind, per-kind format version, and
//! policy architecture. Identity lives IN the file, not a sidecar (the release packer
//! drops sidecars), so no code path can blind-load a record into a guessed architecture:
//! a mis-copied file fails the kind check, a wrong-arch file fails the arch check, and a
//! pre-envelope file fails the magic check — each a distinct, attributable refusal
//! instead of the old quiet-rest-pose degrade.
//!
//! On disk: [`MAGIC`] then bincode of the envelope, with `kind` and `arch` encoded as
//! their stable kebab-case STRINGS — never a bare serde enum, because bincode-1 encodes
//! enum variants by index and ignores rename attrs, so culling an architecture variant
//! would silently re-map every tagged file. String on the wire, enum in process.
//!
//! Legacy (pre-envelope) files are parsed by exactly ONE place — the
//! [`super::migrate`] tool — never by any loader here, so there is no dual-read window
//! to later close.

use std::path::Path;

use crate::bot::arch::ArchId;

use super::atomic_write;

/// File magic identifying an enveloped artifact. Its ONE job is to make "legacy
/// untagged file" a distinct, attributable verdict ([`EnvelopeError::Legacy`], pointing
/// the operator at `migrate-checkpoint`) instead of a probabilistic bincode decode
/// failure indistinguishable from corruption. Format evolution rides the per-kind
/// `version` field inside the envelope, not this constant.
pub(crate) const MAGIC: &[u8; 8] = b"CRABCKPT";

/// Which artifact an envelope carries. A mis-copied file (an optimizer renamed to
/// `brain.bin`) fails this check by name, before any payload decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArtifactKind {
    Brain,
    Optimizer,
    ObsNormalizer,
    ReturnNormalizer,
}

impl ArtifactKind {
    /// Stable on-disk name. Kebab-case, mirrors [`ArchId::name`]'s table style; keep
    /// [`Self::parse`] in lockstep — it is the inverse of this table.
    fn name(self) -> &'static str {
        match self {
            Self::Brain => "brain",
            Self::Optimizer => "optimizer",
            Self::ObsNormalizer => "obs-normalizer",
            Self::ReturnNormalizer => "return-normalizer",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "brain" => Some(Self::Brain),
            "optimizer" => Some(Self::Optimizer),
            "obs-normalizer" => Some(Self::ObsNormalizer),
            "return-normalizer" => Some(Self::ReturnNormalizer),
            _ => None,
        }
    }

    /// The format version this build writes and accepts for this kind — each kind owns
    /// its counter, so e.g. an optimizer-record layout change bumps only the optimizer's.
    /// A file tagged with any other version is refused, never deserialized blind
    /// (the `OptimizerCheckpoint` precedent this module generalizes).
    pub(crate) fn current_version(self) -> u32 {
        match self {
            Self::Brain | Self::Optimizer | Self::ObsNormalizer | Self::ReturnNormalizer => 1,
        }
    }
}

impl std::fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A validated, in-process envelope: reading one through [`read_envelope`] already
/// checked magic, kind, version, and that `arch` names a REGISTERED architecture — so
/// holding this type means the payload is safe to hand to that architecture's decoder.
#[derive(Debug)]
pub(crate) struct CheckpointEnvelope {
    pub(crate) arch: ArchId,
    pub(crate) payload: Vec<u8>,
}

/// The serde mirror actually encoded after [`MAGIC`] — `kind`/`arch` as plain strings so
/// a read can DECODE first and VALIDATE second, attributing an unknown arch by name
/// ([`EnvelopeError::UnknownArch`] carries the string) instead of failing opaquely inside
/// bincode. The one encoder and one decoder both go through this type, so the two can't
/// drift. Field set + order are the on-disk format; changing them is a version bump.
#[derive(serde::Serialize, serde::Deserialize)]
struct RawEnvelope {
    kind: String,
    version: u32,
    arch: String,
    payload: Vec<u8>,
}

/// Why an artifact read was refused. Every variant's `Display` names the fix, because
/// these strings are the loud refusals operators act on (trainer aborts, inference
/// refusal attribution, gate messages).
#[derive(Debug)]
pub(crate) enum EnvelopeError {
    /// No file — the one EXPECTED case (a fresh dir); callers treat it as "nothing
    /// saved yet", every other variant as a fault.
    Absent,
    Io(std::io::Error),
    /// No [`MAGIC`]: a pre-envelope legacy file. The loader has no untagged-read path;
    /// the fix is the one-shot migration tool.
    Legacy,
    /// Magic present but the envelope doesn't decode — a torn or corrupt file.
    Corrupt(String),
    WrongKind {
        found: String,
        expected: ArtifactKind,
    },
    /// The arch tag names no REGISTERED architecture — a checkpoint from a future or
    /// culled arch. Carries the string for attribution.
    UnknownArch(String),
    UnknownVersion {
        found: u32,
        expected: u32,
    },
    /// The tag is a registered arch, but not the one this checkpoint set's brain
    /// declares — a cross-paired dir (the coherence check).
    ArchMismatch {
        found: ArchId,
        expected: ArchId,
    },
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absent => write!(f, "no such file"),
            Self::Io(e) => write!(f, "read failed: {e}"),
            Self::Legacy => write!(
                f,
                "legacy pre-envelope file — migrate the checkpoint dir with \
                 `rl-train migrate-checkpoint <dir>`"
            ),
            Self::Corrupt(e) => write!(f, "corrupt envelope: {e}"),
            Self::WrongKind { found, expected } => write!(
                f,
                "file holds a {found:?} artifact, not the {expected} expected here — \
                 a mis-copied checkpoint file"
            ),
            Self::UnknownArch(s) => write!(
                f,
                "tagged with unregistered policy architecture {s:?} — this build knows \
                 no such arch (a checkpoint from a newer or culled architecture)"
            ),
            Self::UnknownVersion { found, expected } => {
                write!(f, "format v{found}, this build reads v{expected}")
            }
            Self::ArchMismatch { found, expected } => write!(
                f,
                "tagged arch {found} but this checkpoint set's brain is {expected} — \
                 a cross-paired checkpoint dir"
            ),
        }
    }
}

/// Wrap `payload` in an envelope for `kind`/`arch` and write it atomically. The ONE
/// writer — every artifact save goes through here, so a file without magic + tags can't
/// be produced by any production path.
pub(crate) fn write_envelope(
    path: &Path,
    kind: ArtifactKind,
    arch: ArchId,
    payload: Vec<u8>,
) -> std::io::Result<()> {
    let raw = RawEnvelope {
        kind: kind.name().to_string(),
        version: kind.current_version(),
        arch: arch.name().to_string(),
        payload,
    };
    let body = bincode::serialize(&raw).map_err(std::io::Error::other)?;
    let mut bytes = Vec::with_capacity(MAGIC.len() + body.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&body);
    atomic_write(path, &bytes)
}

/// Whether `bytes` are an enveloped artifact — the migrate tool's already-tagged check.
pub(crate) fn is_enveloped(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

/// Read and fully validate the envelope at `path`, expecting `expected` (kind and its
/// current version). The ONE reader: magic → decode → kind → version → arch, each
/// failure a distinct [`EnvelopeError`]. The payload's own decode belongs to the
/// artifact's owner; arch-vs-brain coherence to the caller (which knows the set's brain).
pub(crate) fn read_envelope(
    path: &Path,
    expected: ArtifactKind,
) -> Result<CheckpointEnvelope, EnvelopeError> {
    let bytes = std::fs::read(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => EnvelopeError::Absent,
        _ => EnvelopeError::Io(e),
    })?;
    let Some(body) = bytes.strip_prefix(MAGIC.as_slice()) else {
        return Err(EnvelopeError::Legacy);
    };
    let raw: RawEnvelope =
        bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
    if ArtifactKind::parse(&raw.kind) != Some(expected) {
        return Err(EnvelopeError::WrongKind {
            found: raw.kind,
            expected,
        });
    }
    if raw.version != expected.current_version() {
        return Err(EnvelopeError::UnknownVersion {
            found: raw.version,
            expected: expected.current_version(),
        });
    }
    let arch =
        ArchId::try_from(raw.arch.clone()).map_err(|_| EnvelopeError::UnknownArch(raw.arch))?;
    Ok(CheckpointEnvelope {
        arch,
        payload: raw.payload,
    })
}

/// [`read_envelope`] plus the arch-COHERENCE check for brain-PAIRED artifacts (optimizer,
/// normalizers): the envelope must be tagged with `expected_arch` — the checkpoint set's
/// brain's — or the read refuses with [`EnvelopeError::ArchMismatch`]. The one place the
/// coherence rule is spelled; the brain itself reads via the plain [`read_envelope`]
/// because its tag IS the authority the others are checked against.
pub(crate) fn read_envelope_expecting(
    path: &Path,
    kind: ArtifactKind,
    expected_arch: ArchId,
) -> Result<CheckpointEnvelope, EnvelopeError> {
    let env = read_envelope(path, kind)?;
    if env.arch != expected_arch {
        return Err(EnvelopeError::ArchMismatch {
            found: env.arch,
            expected: expected_arch,
        });
    }
    Ok(env)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rl-envelope-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn round_trips_payload_and_arch() {
        let dir = scratch("roundtrip");
        let path = dir.join("brain.bin");
        write_envelope(&path, ArtifactKind::Brain, ArchId::Mlp256, vec![1, 2, 3]).unwrap();
        let env = read_envelope(&path, ArtifactKind::Brain).unwrap();
        assert_eq!(env.arch, ArchId::Mlp256);
        assert_eq!(env.payload, vec![1, 2, 3]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Each refusal is its own attributable verdict — the properties the per-artifact
    /// refusal policies act on: absent vs legacy vs corrupt vs mis-copied vs
    /// future-arch are DIFFERENT failures with different fixes.
    #[test]
    fn refusals_are_distinct_and_attributed() {
        let dir = scratch("refusals");
        let path = dir.join("f.bin");

        assert!(matches!(
            read_envelope(&path, ArtifactKind::Brain),
            Err(EnvelopeError::Absent)
        ));

        // A pre-envelope file (no magic) is Legacy, not Corrupt — the migrate hint hangs
        // off this distinction.
        std::fs::write(&path, b"old raw burn record bytes").unwrap();
        assert!(matches!(
            read_envelope(&path, ArtifactKind::Brain),
            Err(EnvelopeError::Legacy)
        ));

        // Magic but garbage after it: Corrupt.
        std::fs::write(&path, [MAGIC.as_slice(), &[0xff; 4]].concat()).unwrap();
        assert!(matches!(
            read_envelope(&path, ArtifactKind::Brain),
            Err(EnvelopeError::Corrupt(_))
        ));

        // A mis-copied file: optimizer envelope read as the brain.
        write_envelope(&path, ArtifactKind::Optimizer, ArchId::Mlp256, vec![]).unwrap();
        assert!(matches!(
            read_envelope(&path, ArtifactKind::Brain),
            Err(EnvelopeError::WrongKind { .. })
        ));

        // An arch this build doesn't register is refused BY NAME (attribution), which
        // requires the two-step decode-then-validate — a bare serde-enum field would
        // fail opaquely inside bincode instead.
        let raw = RawEnvelope {
            kind: "brain".into(),
            version: 1,
            arch: "mlp9000".into(),
            payload: vec![],
        };
        std::fs::write(
            &path,
            [MAGIC.as_slice(), &bincode::serialize(&raw).unwrap()].concat(),
        )
        .unwrap();
        match read_envelope(&path, ArtifactKind::Brain) {
            Err(EnvelopeError::UnknownArch(s)) => assert_eq!(s, "mlp9000"),
            other => panic!("expected UnknownArch, got {other:?}"),
        }

        // A future format version is refused, never deserialized blind.
        let raw = RawEnvelope {
            kind: "brain".into(),
            version: ArtifactKind::Brain.current_version() + 1,
            arch: "mlp256".into(),
            payload: vec![],
        };
        std::fs::write(
            &path,
            [MAGIC.as_slice(), &bincode::serialize(&raw).unwrap()].concat(),
        )
        .unwrap();
        assert!(matches!(
            read_envelope(&path, ArtifactKind::Brain),
            Err(EnvelopeError::UnknownVersion { .. })
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
