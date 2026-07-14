use std::path::Path;

use crate::bot::arch::ArchId;

use super::atomic_write;

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
    /// (the `OptimizerCheckpoint` precedent this module generalizes) — with deliberate
    /// BACKWARD exceptions so the fleet's live checkpoints resume instead of being
    /// invalidated: every kind's v1 is still read (predates all stamps), and v2/v3
    /// BRAINS (v2 predates the bddap/rl#215 save stamp, v3 the bddap/rl#271 layout
    /// digest) are still read — each with the absent stamps as `None`
    /// (trust-on-first-use); the next save writes the current version.
    pub(crate) fn current_version(self) -> u32 {
        match self {
            Self::Brain => 4,
            Self::Optimizer | Self::ObsNormalizer | Self::ReturnNormalizer => 2,
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
    /// The obs/action channel-layout identity the brain was trained against
    /// ([`crate::bot::channel_layout_digest`], bddap/rl#271). `None` = a pre-v4 brain
    /// (trust-on-first-use) or a paired kind (the brain is the layout authority).
    pub(crate) layout_digest: Option<u64>,
    /// The checkpoint-SET save stamp (bddap/rl#215): one random value drawn per
    /// `save_checkpoint`, written into every member saved together, so a mixed set is
    /// DETECTABLE at load — a paired member whose stamp differs from the brain's comes
    /// from a different save and must not pair with it. The writers themselves can no
    /// longer put a mixed set on disk (the whole set lands in one atomic dir swap,
    /// bddap/rl#238); the stamp is the backstop for out-of-band copies that read
    /// member files by separate path lookups mid-swap, and for a stale optional
    /// member carried past a failed write. `None` = predates the stamp (brain ≤v2,
    /// paired v1), checked as a set: an all-`None` set is trusted on first use, a
    /// mixed set refuses.
    pub(crate) save_stamp: Option<u64>,
}

/// The serde mirrors actually encoded after [`MAGIC`] — `kind`/`arch` as plain strings so
/// a read can DECODE first and VALIDATE second, attributing an unknown arch by name
/// ([`EnvelopeError::UnknownArch`] carries the string) instead of failing opaquely inside
/// bincode. ONE shape per (kind, version), each a strict append-only extension of `V1`;
/// each has one encoder and one decoder, both through its shape, so the two can't
/// drift. Field set + order are the on-disk format; changing them is a version bump. The
/// reader picks the shape off [`RawHeader`], the common `kind`+`version` prefix (bincode
/// decodes fields sequentially and the crate's legacy config tolerates trailing bytes, so
/// the prefix decodes from every shape).
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

/// Brain V3: V2 plus the checkpoint-set save stamp (bddap/rl#215).
#[derive(serde::Serialize, serde::Deserialize)]
struct RawEnvelopeV3 {
    kind: String,
    version: u32,
    arch: String,
    payload: Vec<u8>,
    body_digest: u64,
    save_stamp: u64,
}

/// Brain V4: V3 plus the obs/action channel-layout digest (bddap/rl#271).
#[derive(serde::Serialize, serde::Deserialize)]
struct RawEnvelopeV4 {
    kind: String,
    version: u32,
    arch: String,
    payload: Vec<u8>,
    body_digest: u64,
    save_stamp: u64,
    layout_digest: u64,
}

/// Paired-kind V2 (optimizer, normalizers): V1 plus the save stamp (bddap/rl#215).
/// Bincode-identical layout to the brain's [`RawEnvelopeV2`] (both append one u64) — the
/// reader can't cross them anyway, since it dispatches on (kind, version) first — but a
/// distinct type because the u64 MEANS something different: with the field NAMED
/// `save_stamp`, no encode/decode site can conflate the two u64s in source.
#[derive(serde::Serialize, serde::Deserialize)]
struct RawEnvelopePairedV2 {
    kind: String,
    version: u32,
    arch: String,
    payload: Vec<u8>,
    save_stamp: u64,
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
    /// The save stamp differs from the one this checkpoint set's brain carries —
    /// this member and the brain were written by DIFFERENT saves, i.e. a partial/torn
    /// save or copy mis-paired the set (bddap/rl#215). `None` = unstamped (pre-#215).
    SaveStampMismatch {
        found: Option<u64>,
        expected: Option<u64>,
    },
}

/// Render a save stamp for refusal messages: the hex value, or its absence named.
fn save_stamp_str(g: Option<u64>) -> String {
    match g {
        Some(g) => format!("{g:#018x}"),
        None => "none (pre-rl#215, unstamped)".to_string(),
    }
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
            Self::SaveStampMismatch { found, expected } => write!(
                f,
                "carries save stamp {} but this checkpoint set's brain carries {} — \
                 this member and the brain come from DIFFERENT saves (a partial or torn \
                 save/copy mis-paired the set, bddap/rl#215); restore a coherent set \
                 (e.g. the best/ snapshot) or point at a fresh dir",
                save_stamp_str(*found),
                save_stamp_str(*expected),
            ),
        }
    }
}

/// The identity stamps only BRAIN envelopes carry — one type, so a writer can't stamp a
/// body without a channel layout or vice versa (the paired kinds carry neither: the
/// brain is the identity authority they are checked against).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BrainStamps {
    /// [`crate::mesh_fallback::constructed_body_digest`] (bddap/rl#214).
    pub(crate) body_digest: u64,
    /// [`crate::bot::channel_layout_digest`] (bddap/rl#271).
    pub(crate) layout_digest: u64,
}

/// Wrap `payload` in an envelope for `kind`/`arch` and write it atomically. The ONE
/// writer — every artifact save goes through here, so a file without magic + tags can't
/// be produced by any production path. Always writes the kind's CURRENT version, so
/// `stamps` is required exactly when the kind carries them (the brain) and forbidden
/// otherwise — a call that disagrees is a programming error, caught by every save-path
/// test, never a silently mis-shaped file. `save_stamp` is the checkpoint-set stamp
/// (bddap/rl#215), required on every kind: all members of one save share one value.
pub(crate) fn write_envelope(
    path: &Path,
    kind: ArtifactKind,
    arch: ArchId,
    payload: Vec<u8>,
    stamps: Option<BrainStamps>,
    save_stamp: u64,
) -> std::io::Result<()> {
    let kind_name = kind.name().to_string();
    let version = kind.current_version();
    let arch_name = arch.name().to_string();
    let body = match kind {
        ArtifactKind::Brain => {
            // Loud at the FIRST save, not at some later load: bumping `current_version`
            // without teaching this arm the new shape would otherwise write files that
            // claim the new version with the old shape — refused by every reader.
            assert_eq!(version, 4, "no writer arm for brain envelope v{version}");
            let stamps = stamps
                .unwrap_or_else(|| panic!("{kind} envelopes require the brain identity stamps"));
            bincode::serialize(&RawEnvelopeV4 {
                kind: kind_name,
                version,
                arch: arch_name,
                payload,
                body_digest: stamps.body_digest,
                save_stamp,
                layout_digest: stamps.layout_digest,
            })
        }
        ArtifactKind::Optimizer | ArtifactKind::ObsNormalizer | ArtifactKind::ReturnNormalizer => {
            assert_eq!(version, 2, "no writer arm for {kind} envelope v{version}");
            assert!(
                stamps.is_none(),
                "{kind} envelopes carry no identity stamps (the brain is the authority)"
            );
            bincode::serialize(&RawEnvelopePairedV2 {
                kind: kind_name,
                version,
                arch: arch_name,
                payload,
                save_stamp,
            })
        }
    }
    .map_err(std::io::Error::other)?;
    let mut bytes = Vec::with_capacity(MAGIC.len() + body.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&body);
    atomic_write(path, &bytes)
}

/// TEST-ONLY writer of a LEGACY v1 envelope (any kind). The golden checkpoint fixture
/// must stay v1 forever: the production writer stamps the writing machine's constructed
/// body digest and a per-save stamp, which would make a committed fixture refuse to
/// arm on every machine whose `sally.glb` differs or is absent — v1 reads as
/// trust-on-first-use everywhere, i.e. machine-portable (and the fixture's brain and
/// normalizer must BOTH be v1: an unstamped brain refuses a stamped partner,
/// bddap/rl#215). Also the writer for reader-side TOFU tests.
#[cfg(test)]
pub(crate) fn write_v1_envelope(
    path: &Path,
    kind: ArtifactKind,
    arch: ArchId,
    payload: Vec<u8>,
) -> std::io::Result<()> {
    let raw = RawEnvelopeV1 {
        kind: kind.name().to_string(),
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
    // Accepted versions per kind: the current one, plus the pre-stamp backward reads —
    // every kind's v1 (predates all stamps), the brain's v2 (pre-#215) and v3
    // (pre-#271) — each with the absent stamps as `None` (trust-on-first-use).
    // Anything else is refused.
    let (arch, payload, body_digest, save_stamp, layout_digest) = match (expected, hdr.version) {
        (ArtifactKind::Brain, 4) => {
            let raw: RawEnvelopeV4 =
                bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
            (
                raw.arch,
                raw.payload,
                Some(raw.body_digest),
                Some(raw.save_stamp),
                Some(raw.layout_digest),
            )
        }
        (ArtifactKind::Brain, 3) => {
            let raw: RawEnvelopeV3 =
                bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
            (
                raw.arch,
                raw.payload,
                Some(raw.body_digest),
                Some(raw.save_stamp),
                None,
            )
        }
        (ArtifactKind::Brain, 2) => {
            let raw: RawEnvelopeV2 =
                bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
            (raw.arch, raw.payload, Some(raw.body_digest), None, None)
        }
        (
            ArtifactKind::Optimizer | ArtifactKind::ObsNormalizer | ArtifactKind::ReturnNormalizer,
            2,
        ) => {
            let raw: RawEnvelopePairedV2 =
                bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
            (raw.arch, raw.payload, None, Some(raw.save_stamp), None)
        }
        (_, 1) => {
            let raw: RawEnvelopeV1 =
                bincode::deserialize(body).map_err(|e| EnvelopeError::Corrupt(e.to_string()))?;
            (raw.arch, raw.payload, None, None, None)
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
        save_stamp,
        layout_digest,
    })
}

/// The pairing key a checkpoint set's BRAIN establishes and every paired member must
/// match: the arch tag (bddap/rl#200 §2) and the save stamp (bddap/rl#215). One type,
/// not two parallel parameters, so a call site can't pair one brain's arch with another
/// brain's stamp; production constructs it only from the loaded brain
/// (`BrainFile::set_key` in [`super::checkpoint`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SetKey {
    pub(crate) arch: ArchId,
    /// `None` = a pre-stamp brain, pairing only with unstamped members.
    pub(crate) save_stamp: Option<u64>,
}

/// [`read_envelope`] plus the SET-COHERENCE checks for brain-PAIRED artifacts (optimizer,
/// normalizers): the envelope must carry the checkpoint set's brain's [`SetKey`] — its
/// arch tag AND its save stamp — or the read refuses with
/// [`EnvelopeError::ArchMismatch`] / [`EnvelopeError::SaveStampMismatch`]. The one place
/// the coherence rules are spelled; the brain itself reads via the plain
/// [`read_envelope`] because its tags ARE the authority the others are checked against.
/// The save-stamp check is EXACT over `Option` (bddap/rl#215): `None == None` passes (a
/// wholly pre-stamp set, trust-on-first-use), while a stamped member against an
/// unstamped brain — or vice versa — is a partial save straddling the upgrade and
/// refuses like any other mismatch.
pub(crate) fn read_envelope_expecting(
    path: &Path,
    kind: ArtifactKind,
    key: SetKey,
) -> Result<CheckpointEnvelope, EnvelopeError> {
    let env = read_envelope(path, kind)?;
    if env.arch != key.arch {
        return Err(EnvelopeError::ArchMismatch {
            found: env.arch,
            expected: key.arch,
        });
    }
    if env.save_stamp != key.save_stamp {
        return Err(EnvelopeError::SaveStampMismatch {
            found: env.save_stamp,
            expected: key.save_stamp,
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
    fn round_trips_payload_arch_stamps_and_save_stamp() {
        let dir = scratch("roundtrip");
        let path = dir.join("brain.bin");
        write_envelope(
            &path,
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            vec![1, 2, 3],
            Some(BrainStamps {
                body_digest: 0xfeed_beef,
                layout_digest: 0x1a_0e57,
            }),
            0x6e6e_7261_7469_6f6e,
        )
        .unwrap();
        let env = read_envelope(&path, ArtifactKind::Brain).unwrap();
        assert_eq!(env.arch, ArchId::DEFAULT);
        assert_eq!(env.payload, vec![1, 2, 3]);
        assert_eq!(env.body_digest, Some(0xfeed_beef));
        assert_eq!(env.save_stamp, Some(0x6e6e_7261_7469_6f6e));
        assert_eq!(env.layout_digest, Some(0x1a_0e57));

        // Paired kinds carry the save stamp but never the brain identity stamps.
        let path = dir.join("optimizer.bin");
        write_envelope(
            &path,
            ArtifactKind::Optimizer,
            ArchId::DEFAULT,
            vec![9],
            None,
            0x6e6e_7261_7469_6f6e,
        )
        .unwrap();
        let env = read_envelope(&path, ArtifactKind::Optimizer).unwrap();
        assert_eq!(env.payload, vec![9]);
        assert_eq!(env.body_digest, None);
        assert_eq!(env.layout_digest, None);
        assert_eq!(env.save_stamp, Some(0x6e6e_7261_7469_6f6e));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Pre-stamp files — every checkpoint written before bddap/rl#214/#215 — still read,
    /// with the absent stamps as `None` (trust-on-first-use), so shipping the stamps does
    /// NOT invalidate the fleet's live checkpoints: a v1 brain, a v2 brain, and a v1
    /// paired artifact each resume.
    #[test]
    fn pre_stamp_files_read_as_tofu() {
        let dir = scratch("tofu");
        let path = dir.join("brain.bin");
        write_v1_envelope(&path, ArtifactKind::Brain, ArchId::DEFAULT, vec![4, 5]).unwrap();
        let env = read_envelope(&path, ArtifactKind::Brain).unwrap();
        assert_eq!(env.arch, ArchId::DEFAULT);
        assert_eq!(env.payload, vec![4, 5]);
        assert_eq!(env.body_digest, None);
        assert_eq!(env.save_stamp, None);

        // A v2 brain (bddap/rl#214, pre-#215): body digest present, save_stamp None.
        let raw = RawEnvelopeV2 {
            kind: "brain".into(),
            version: 2,
            arch: ArchId::DEFAULT.name().into(),
            payload: vec![6],
            body_digest: 0xabc,
        };
        std::fs::write(
            &path,
            [MAGIC.as_slice(), &bincode::serialize(&raw).unwrap()].concat(),
        )
        .unwrap();
        let env = read_envelope(&path, ArtifactKind::Brain).unwrap();
        assert_eq!(env.body_digest, Some(0xabc));
        assert_eq!(env.save_stamp, None);
        assert_eq!(env.layout_digest, None);

        // A v3 brain (bddap/rl#215, pre-#271): body digest + save stamp present,
        // layout_digest None — the fleet's live brains at the #271 upgrade.
        let raw = RawEnvelopeV3 {
            kind: "brain".into(),
            version: 3,
            arch: ArchId::DEFAULT.name().into(),
            payload: vec![8],
            body_digest: 0xabc,
            save_stamp: 42,
        };
        std::fs::write(
            &path,
            [MAGIC.as_slice(), &bincode::serialize(&raw).unwrap()].concat(),
        )
        .unwrap();
        let env = read_envelope(&path, ArtifactKind::Brain).unwrap();
        assert_eq!(env.body_digest, Some(0xabc));
        assert_eq!(env.save_stamp, Some(42));
        assert_eq!(env.layout_digest, None);

        // A v1 paired artifact: no stamps at all.
        let path = dir.join("normalizer.bin");
        write_v1_envelope(&path, ArtifactKind::ObsNormalizer, ArchId::DEFAULT, vec![7]).unwrap();
        let env = read_envelope(&path, ArtifactKind::ObsNormalizer).unwrap();
        assert_eq!(env.body_digest, None);
        assert_eq!(env.save_stamp, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bddap/rl#215 coherence matrix on `read_envelope_expecting`: a paired member
    /// pairs only with the brain save_stamp it was saved with — equal stamps pass, a
    /// wholly pre-stamp set passes on trust, and every mixed case (different stamps, or
    /// stamped-vs-unstamped in either direction — a partial save straddling the upgrade)
    /// is the distinct `SaveStampMismatch` refusal.
    #[test]
    fn save_stamp_pairing_matrix() {
        let dir = scratch("genmatrix");
        let stamped = dir.join("stamped.bin");
        write_envelope(
            &stamped,
            ArtifactKind::ObsNormalizer,
            ArchId::DEFAULT,
            vec![1],
            None,
            77,
        )
        .unwrap();
        let unstamped = dir.join("unstamped.bin");
        write_v1_envelope(
            &unstamped,
            ArtifactKind::ObsNormalizer,
            ArchId::DEFAULT,
            vec![1],
        )
        .unwrap();

        let read = |path, save_stamp| {
            read_envelope_expecting(
                path,
                ArtifactKind::ObsNormalizer,
                SetKey {
                    arch: ArchId::DEFAULT,
                    save_stamp,
                },
            )
        };
        assert!(read(&stamped, Some(77)).is_ok(), "matching stamps pair");
        assert!(
            read(&unstamped, None).is_ok(),
            "a wholly pre-stamp set pairs on trust"
        );
        for (path, expected) in [
            (&stamped, Some(78)),   // different saves
            (&stamped, None),       // stamped member, unstamped brain (brain write failed)
            (&unstamped, Some(77)), // unstamped member, stamped brain (member write failed)
        ] {
            match read(path, expected) {
                Err(EnvelopeError::SaveStampMismatch { .. }) => {}
                other => panic!("expected SaveStampMismatch for {expected:?}, got {other:?}"),
            }
        }
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
        write_envelope(
            &path,
            ArtifactKind::Optimizer,
            ArchId::DEFAULT,
            vec![],
            None,
            1,
        )
        .unwrap();
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

        // Version acceptance is per KIND: a paired kind at the brain's v3 (even with a
        // well-formed V3 shape) is refused, never decoded through the brain's carve-outs.
        let raw = RawEnvelopeV3 {
            kind: "optimizer".into(),
            version: 3,
            arch: ArchId::DEFAULT.name().into(),
            payload: vec![],
            body_digest: 7,
            save_stamp: 7,
        };
        std::fs::write(
            &path,
            [MAGIC.as_slice(), &bincode::serialize(&raw).unwrap()].concat(),
        )
        .unwrap();
        assert!(matches!(
            read_envelope(&path, ArtifactKind::Optimizer),
            Err(EnvelopeError::UnknownVersion { found: 3, .. })
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
