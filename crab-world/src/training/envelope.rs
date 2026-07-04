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
//! Legacy (pre-envelope) files are parsed NOWHERE: their sole parser, the one-shot
//! `migrate-checkpoint` tool, was deleted once the fleet was migrated (rl#200
//! increment 3). A stray legacy file today is refused; its bytes are recoverable only
//! via that tool in git history.

use std::path::Path;

use crate::bot::arch::ArchId;

use super::atomic_write;

/// File magic identifying an enveloped artifact. Its ONE job is to make "legacy
/// untagged file" a distinct, attributable verdict ([`EnvelopeError::Legacy`]) instead
/// of a probabilistic bincode decode failure indistinguishable from corruption. Format
/// evolution rides the per-kind `version` field inside the envelope, not this constant.
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
    /// (the `OptimizerCheckpoint` precedent this module generalizes) — with ONE
    /// deliberate exception: a v1 BRAIN (predates the bddap/rl#214 body-digest field) is
    /// still read, as `body_digest: None`, so the fleet's live checkpoints resume
    /// trust-on-first-use instead of being invalidated; their next save writes v2.
    pub(crate) fn current_version(self) -> u32 {
        match self {
            Self::Brain => 2,
            Self::Optimizer | Self::ObsNormalizer | Self::ReturnNormalizer => 1,
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
    /// Body identity the artifact was trained against ([`crate::mesh_fallback::
    /// constructed_body_digest`]): `Some(0)` = the procedural fallback body, `Some(_)` =
    /// a mesh-fitted body. `None` = a v1 brain from before the stamp existed
    /// (bddap/rl#214) — resumed trust-on-first-use — or a paired kind, which never
    /// carries one (the BRAIN is the body authority, as its arch tag is the arch
    /// authority the paired artifacts are checked against).
    pub(crate) body_digest: Option<u64>,
}

/// The serde mirrors actually encoded after [`MAGIC`] — `kind`/`arch` as plain strings so
/// a read can DECODE first and VALIDATE second, attributing an unknown arch by name
/// ([`EnvelopeError::UnknownArch`] carries the string) instead of failing opaquely inside
/// bincode. ONE shape per format version, `V2` a strict append-only extension of `V1`;
/// each version has one encoder and one decoder, both through its shape, so the two can't
/// drift. Field set + order are the on-disk format; changing them is a version bump. The
/// reader picks the shape off [`RawHeader`], the common `kind`+`version` prefix (bincode
/// decodes fields sequentially and the crate's legacy config tolerates trailing bytes, so
/// the prefix decodes from either shape).
#[derive(serde::Serialize, serde::Deserialize)]
struct RawEnvelopeV1 {
    kind: String,
    version: u32,
    arch: String,
    payload: Vec<u8>,
}

/// V2 (brain-only): V1 plus the body-identity digest (bddap/rl#214).
#[derive(serde::Serialize, serde::Deserialize)]
struct RawEnvelopeV2 {
    kind: String,
    version: u32,
    arch: String,
    payload: Vec<u8>,
    body_digest: u64,
}

/// The shared decode prefix of every raw shape: enough to pick the full shape to decode
/// (and to attribute `WrongKind`/`UnknownVersion` before touching the payload).
#[derive(serde::Deserialize)]
struct RawHeader {
    kind: String,
    version: u32,
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
    /// the fix is re-copying from a migrated checkpoint (the fleet migrated in rl#200,
    /// then the one-shot migration tool was deleted).
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
                "legacy pre-envelope file — the fleet was migrated to tagged envelopes \
                 (rl#200); re-copy from a migrated checkpoint, or build the pre-deletion \
                 tree from git history for its `migrate-checkpoint` tool"
            ),
            Self::Corrupt(e) => write!(
                f,
                "corrupt envelope: {e} — a torn or truncated copy; re-copy or redeploy"
            ),
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
/// be produced by any production path. Always writes the kind's CURRENT version, so
/// `body_digest` is required exactly when the kind carries one (the brain, v2) and
/// forbidden otherwise — a call that disagrees is a programming error, caught by every
/// save-path test, never a silently mis-shaped file.
pub(crate) fn write_envelope(
    path: &Path,
    kind: ArtifactKind,
    arch: ArchId,
    payload: Vec<u8>,
    body_digest: Option<u64>,
) -> std::io::Result<()> {
    let kind_name = kind.name().to_string();
    let version = kind.current_version();
    let arch_name = arch.name().to_string();
    let body = match version {
        1 => {
            assert!(
                body_digest.is_none(),
                "{kind} envelopes (v1) carry no body digest"
            );
            bincode::serialize(&RawEnvelopeV1 {
                kind: kind_name,
                version,
                arch: arch_name,
                payload,
            })
        }
        2 => bincode::serialize(&RawEnvelopeV2 {
            kind: kind_name,
            version,
            arch: arch_name,
            payload,
            body_digest: body_digest
                .unwrap_or_else(|| panic!("{kind} envelopes (v2) require a body digest")),
        }),
        v => unreachable!("no writer for {kind} envelope version {v}"),
    }
    .map_err(std::io::Error::other)?;
    let mut bytes = Vec::with_capacity(MAGIC.len() + body.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&body);
    atomic_write(path, &bytes)
}

/// TEST-ONLY writer of a LEGACY v1 brain envelope. The golden checkpoint fixture must
/// stay v1 forever: the production writer stamps the writing machine's constructed body
/// digest, which would make a committed fixture refuse to arm on every machine whose
/// `sally.glb` differs or is absent — v1 reads as trust-on-first-use everywhere, i.e.
/// machine-portable. Also the writer for reader-side TOFU tests.
#[cfg(test)]
pub(crate) fn write_v1_brain_envelope(
    path: &Path,
    arch: ArchId,
    payload: Vec<u8>,
) -> std::io::Result<()> {
    let raw = RawEnvelopeV1 {
        kind: ArtifactKind::Brain.name().to_string(),
        version: 1,
        arch: arch.name().to_string(),
        payload,
    };
    let body = bincode::serialize(&raw).map_err(std::io::Error::other)?;
    let mut bytes = Vec::with_capacity(MAGIC.len() + body.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&body);
    atomic_write(path, &bytes)
}

/// Read and fully validate the envelope at `path`, expecting `expected` (kind and an
/// accepted version). The ONE reader: magic → header → kind → version-picked shape →
/// arch, each failure a distinct [`EnvelopeError`]. The payload's own decode belongs to
/// the artifact's owner; arch-vs-brain coherence to the caller (which knows the set's
/// brain); body-digest verification to the caller too (which knows the constructed body).
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
    let hdr: RawHeader =
        bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
    if ArtifactKind::parse(&hdr.kind) != Some(expected) {
        return Err(EnvelopeError::WrongKind {
            found: hdr.kind,
            expected,
        });
    }
    // Accepted versions per kind: the current one, plus v1 for the brain (pre-#214
    // files, trust-on-first-use — `body_digest: None`). Anything else is refused.
    let (arch, payload, body_digest) = match (expected, hdr.version) {
        (ArtifactKind::Brain, 2) => {
            let raw: RawEnvelopeV2 =
                bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
            (raw.arch, raw.payload, Some(raw.body_digest))
        }
        (_, 1) if expected.current_version() == 1 || expected == ArtifactKind::Brain => {
            let raw: RawEnvelopeV1 =
                bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
            (raw.arch, raw.payload, None)
        }
        (_, found) => {
            return Err(EnvelopeError::UnknownVersion {
                found,
                expected: expected.current_version(),
            });
        }
    };
    let arch = ArchId::try_from(arch.clone()).map_err(|_| EnvelopeError::UnknownArch(arch))?;
    Ok(CheckpointEnvelope {
        arch,
        payload,
        body_digest,
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
    fn round_trips_payload_arch_and_body_digest() {
        let dir = scratch("roundtrip");
        let path = dir.join("brain.bin");
        write_envelope(
            &path,
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            vec![1, 2, 3],
            Some(0xfeed_beef),
        )
        .unwrap();
        let env = read_envelope(&path, ArtifactKind::Brain).unwrap();
        assert_eq!(env.arch, ArchId::DEFAULT);
        assert_eq!(env.payload, vec![1, 2, 3]);
        assert_eq!(env.body_digest, Some(0xfeed_beef));

        // Paired kinds stay v1 and never carry a digest.
        let path = dir.join("optimizer.bin");
        write_envelope(&path, ArtifactKind::Optimizer, ArchId::DEFAULT, vec![9], None).unwrap();
        let env = read_envelope(&path, ArtifactKind::Optimizer).unwrap();
        assert_eq!(env.payload, vec![9]);
        assert_eq!(env.body_digest, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A v1 brain — every checkpoint written before bddap/rl#214 — still reads, as
    /// `body_digest: None` (trust-on-first-use), so shipping the digest does NOT
    /// invalidate the fleet's live checkpoints.
    #[test]
    fn v1_brain_reads_as_tofu() {
        let dir = scratch("tofu");
        let path = dir.join("brain.bin");
        write_v1_brain_envelope(&path, ArchId::DEFAULT, vec![4, 5]).unwrap();
        let env = read_envelope(&path, ArtifactKind::Brain).unwrap();
        assert_eq!(env.arch, ArchId::DEFAULT);
        assert_eq!(env.payload, vec![4, 5]);
        assert_eq!(env.body_digest, None);
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

        // A pre-envelope file (no magic) is Legacy, not Corrupt — "predates the fleet
        // migration" and "damaged" are different diagnoses.
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
        write_envelope(&path, ArtifactKind::Optimizer, ArchId::DEFAULT, vec![], None).unwrap();
        assert!(matches!(
            read_envelope(&path, ArtifactKind::Brain),
            Err(EnvelopeError::WrongKind { .. })
        ));

        // An arch this build doesn't register is refused BY NAME (attribution), which
        // requires the two-step decode-then-validate — a bare serde-enum field would
        // fail opaquely inside bincode instead.
        let raw = RawEnvelopeV1 {
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
        let raw = RawEnvelopeV1 {
            kind: "brain".into(),
            version: ArtifactKind::Brain.current_version() + 1,
            arch: ArchId::DEFAULT.name().into(),
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

        // v1-legacy acceptance is a BRAIN-only carve-out: a paired kind at v2 (even
        // with a well-formed V2 shape) is refused, never TOFU'd.
        let raw = RawEnvelopeV2 {
            kind: "optimizer".into(),
            version: 2,
            arch: ArchId::DEFAULT.name().into(),
            payload: vec![],
            body_digest: 7,
        };
        std::fs::write(
            &path,
            [MAGIC.as_slice(), &bincode::serialize(&raw).unwrap()].concat(),
        )
        .unwrap();
        assert!(matches!(
            read_envelope(&path, ArtifactKind::Optimizer),
            Err(EnvelopeError::UnknownVersion { found: 2, .. })
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
