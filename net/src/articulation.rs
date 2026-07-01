//! `CrabArticulation` — the render-only crab pose extension frame (bddap/rl#151, increment 2
//! windowed).
//!
//! The host-authoritative snapshot ([`CoreSnapshot`](crate::snapshot::CoreSnapshot)) carries the
//! integer game state and NOTHING render-only — a render type in it would break the no-`render`
//! trainer build ([[verify-all-bins-on-module-moves]]). But a windowed remote client no longer
//! runs the crab's rapier physics (host-authoritative: the host owns the one Sally, the client
//! renders the host's pose), so it has no per-part transforms to skin the mesh from. Those ride
//! HERE, in a SEPARATE frame the host broadcasts beside the snapshot.
//!
//! The wire type is plain POD (`f32` arrays + a `u8` part tag) with a hand-rolled little-endian
//! codec — deliberately NO bevy/glam type crosses this boundary, so this module compiles in the
//! trainer build unchanged and honors "the trainer never pulls a render type" without the
//! fragility of a `#[cfg]`-split wire enum. The bevy CAPTURE (querying the crab body parts) and
//! APPLY (writing them onto the client's render entities) are the render-only halves; they live in
//! the render modules, gated on `feature = "render"`. This carries only the bytes between them.
//!
//! Unlike [`CoreSnapshot`](crate::snapshot::CoreSnapshot), articulation is NOT authoritative and
//! NOT deterministic — it is float render garnish. A malformed one still fails LOUDLY (a client
//! must never render a half-decoded pose), but a dropped one is merely a skipped render frame, not
//! a correctness fault: it is superseded by the next tick's articulation.

/// One crab body part's render transform for a tick: its world-space position + orientation in the
/// crab's ARENA frame (the frame [`SkinRepose`](crate::articulation::ReposeWire) then relocates to
/// the giant game spot), exactly the input `crab_world::bot::skin::drive_bones` reads off each
/// physics link. No scale — the parts carry none (render==physics).
///
/// `part` is the wire tag for the part's identity (its `crab_world` `PartId`), mapped on the render
/// side so this module stays free of the render crate's types. The host and the client compute the
/// same tag from the same rig, so the client matches each transform to its own part entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartTransform {
    /// Wire tag of the part's `PartId` (`0` = carapace, `1 + joint.index()` = a joint link).
    pub part: u8,
    /// Arena-frame translation, `[x, y, z]`.
    pub pos: [f32; 3],
    /// Arena-frame rotation quaternion, `[x, y, z, w]`.
    pub rot: [f32; 4],
}

/// The giant-crab blow-up placement for a tick — the render-only rigid shift + scale
/// `crab_world::bot::skin::SkinRepose` applies to relocate the ~1 m arena rig to its game spot and
/// size it to the giant. The client can't recompute it (it doesn't run the bridge that integrates
/// the crab's game-world walk), so the host ships it here alongside the parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReposeWire {
    /// Arena→game-world translation added to each part before scaling.
    pub shift: [f32; 3],
    /// Ground pivot the rig is scaled up about (feet stay on the floor).
    pub pivot: [f32; 3],
    /// Uniform blow-up factor.
    pub scale: f32,
}

/// A whole crab's render pose for one tick — every body part plus the giant-blow-up placement. The
/// host captures it after stepping (post-`integrate_crab`/`publish_skin_repose`, so it is this
/// tick's settled pose) and broadcasts it; a windowed client writes it onto its own frozen crab's
/// render entities. `repose` is `None` only before the bridge has published one (transiently at
/// spawn) — the client then leaves its placement untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct CrabArticulation {
    /// The tick this pose is OF — the version, matching the [`CoreSnapshot`](crate::snapshot::CoreSnapshot)
    /// tick it rides beside. A client renders the highest it has seen and drops older arrivals.
    pub tick: u64,
    /// Every crab body part's arena-frame transform, in ascending `part`-tag order.
    pub parts: Vec<PartTransform>,
    /// The giant-blow-up placement, or `None` before the host has published one.
    pub repose: Option<ReposeWire>,
}

/// Why decoding a [`CrabArticulation`] failed. Like a snapshot, a client must never render a
/// half-decoded pose, so a malformed buffer is a hard typed error, not a best-effort partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArticulationDecodeError {
    /// The buffer ended before a field the format requires was fully read.
    Truncated,
    /// The 1-byte `repose-present` flag held a value other than 0 or 1.
    BadFlag,
    /// Bytes remained after a complete articulation decoded — a framing/length mismatch.
    TrailingBytes,
}

impl std::fmt::Display for ArticulationDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Truncated => "articulation buffer ended mid-field",
            Self::BadFlag => "articulation repose flag was neither 0 nor 1",
            Self::TrailingBytes => "trailing bytes after a complete articulation",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ArticulationDecodeError {}

impl CrabArticulation {
    /// Little-endian wire form: `tick(8) | n_parts(4) | part[ tag(1) pos(3×4) rot(4×4) ]… |
    /// repose_present(1) | [ shift(3×4) pivot(3×4) scale(4) ]`. [`from_bytes`](Self::from_bytes) is
    /// its exact inverse.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&(self.parts.len() as u32).to_le_bytes());
        for p in &self.parts {
            out.push(p.part);
            for v in p.pos {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for v in p.rot {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        match &self.repose {
            None => out.push(0),
            Some(r) => {
                out.push(1);
                for v in r.shift {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                for v in r.pivot {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out.extend_from_slice(&r.scale.to_le_bytes());
            }
        }
        out
    }

    /// Inverse of [`to_bytes`](Self::to_bytes). Rejects a buffer that is too short, carries a bad
    /// present-flag, or has bytes left over.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArticulationDecodeError> {
        let mut r = Reader::new(bytes);
        let tick = u64::from_le_bytes(r.take()?);
        let n_parts = u32::from_le_bytes(r.take::<4>()?) as usize;
        // Don't pre-allocate from the untrusted count — grow bounded by the buffer (each `take`
        // Truncates the moment the real bytes run out), never by the claimed length.
        let mut parts = Vec::new();
        for _ in 0..n_parts {
            let part = r.byte()?;
            let pos = read_vec3(&mut r)?;
            let rot = read_vec4(&mut r)?;
            parts.push(PartTransform { part, pos, rot });
        }
        let repose = match r.byte()? {
            0 => None,
            1 => {
                let shift = read_vec3(&mut r)?;
                let pivot = read_vec3(&mut r)?;
                let scale = f32::from_le_bytes(r.take()?);
                Some(ReposeWire { shift, pivot, scale })
            }
            _ => return Err(ArticulationDecodeError::BadFlag),
        };
        if !r.is_empty() {
            return Err(ArticulationDecodeError::TrailingBytes);
        }
        Ok(Self { tick, parts, repose })
    }
}

fn read_vec3(r: &mut Reader<'_>) -> Result<[f32; 3], ArticulationDecodeError> {
    Ok([
        f32::from_le_bytes(r.take()?),
        f32::from_le_bytes(r.take()?),
        f32::from_le_bytes(r.take()?),
    ])
}

fn read_vec4(r: &mut Reader<'_>) -> Result<[f32; 4], ArticulationDecodeError> {
    Ok([
        f32::from_le_bytes(r.take()?),
        f32::from_le_bytes(r.take()?),
        f32::from_le_bytes(r.take()?),
        f32::from_le_bytes(r.take()?),
    ])
}

/// A bounds-checked forward cursor — the same discipline [`crate::snapshot`]'s reader uses; every
/// read advances and returns [`ArticulationDecodeError::Truncated`] past the end.
struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], ArticulationDecodeError> {
        let end = self.at.checked_add(N).ok_or(ArticulationDecodeError::Truncated)?;
        let slice = self.buf.get(self.at..end).ok_or(ArticulationDecodeError::Truncated)?;
        self.at = end;
        Ok(slice.try_into().expect("slice length checked above"))
    }

    fn byte(&mut self) -> Result<u8, ArticulationDecodeError> {
        Ok(self.take::<1>()?[0])
    }

    fn is_empty(&self) -> bool {
        self.at == self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CrabArticulation {
        CrabArticulation {
            tick: 4242,
            parts: vec![
                PartTransform { part: 0, pos: [1.0, 2.0, 3.0], rot: [0.0, 0.0, 0.0, 1.0] },
                PartTransform { part: 7, pos: [-4.5, 0.25, 9.0], rot: [0.5, 0.5, 0.5, 0.5] },
            ],
            repose: Some(ReposeWire {
                shift: [10.0, 0.0, -20.0],
                pivot: [0.0, 1.0, 0.0],
                scale: 8.0,
            }),
        }
    }

    #[test]
    fn bytes_roundtrip_is_exact() {
        let a = sample();
        assert_eq!(CrabArticulation::from_bytes(&a.to_bytes()).unwrap(), a);
    }

    #[test]
    fn roundtrip_without_repose() {
        let mut a = sample();
        a.repose = None;
        assert_eq!(CrabArticulation::from_bytes(&a.to_bytes()).unwrap(), a);
    }

    #[test]
    fn empty_parts_roundtrip() {
        let a = CrabArticulation { tick: 0, parts: vec![], repose: None };
        assert_eq!(CrabArticulation::from_bytes(&a.to_bytes()).unwrap(), a);
    }

    #[test]
    fn truncated_buffer_is_rejected() {
        let bytes = sample().to_bytes();
        assert_eq!(
            CrabArticulation::from_bytes(&bytes[..bytes.len() - 1]),
            Err(ArticulationDecodeError::Truncated)
        );
        assert_eq!(CrabArticulation::from_bytes(&[]), Err(ArticulationDecodeError::Truncated));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = sample().to_bytes();
        bytes.push(0);
        assert_eq!(CrabArticulation::from_bytes(&bytes), Err(ArticulationDecodeError::TrailingBytes));
    }

    #[test]
    fn bad_present_flag_is_rejected() {
        // Corrupt the repose-present flag: it sits right after tick(8) + n_parts(4) + the two
        // parts (each 1 + 12 + 16 = 29 B).
        let flag_off = 8 + 4 + 2 * (1 + 12 + 16);
        let mut bytes = sample().to_bytes();
        bytes[flag_off] = 2;
        assert_eq!(CrabArticulation::from_bytes(&bytes), Err(ArticulationDecodeError::BadFlag));
    }
}
