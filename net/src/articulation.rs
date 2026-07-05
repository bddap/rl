//! `CrabArticulation` — the render-only crab pose extension frame.
//!
//! The host-authoritative snapshot ([`CoreSnapshot`](crate::snapshot::CoreSnapshot)) carries the
//! integer game state and NOTHING render-only — a render type in it would break the no-`render`
//! trainer build ([[verify-all-bins-on-module-moves]]). But a windowed remote client no longer
//! runs the crabs' rapier physics (host-authoritative: the host owns every Sally, the client
//! renders the host's poses), so it has no per-part transforms to skin the meshes from. Those
//! ride HERE, in a SEPARATE frame the host broadcasts beside the snapshot — one [`CrabFrame`]
//! per crab, in crab-index order (the index IS the crab's identity, matching the snapshot's
//! `crabs` and the host's crab-world env ids — rl#200 multi-brain rounds), along with the pose
//! of the host's piloted craft ([`VehiclePoseWire`]), the other host-only rapier body a remote
//! client must render without simulating (rl#192).
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
/// crab's ARENA frame (the frame [`ReposeWire`] then relocates to
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

/// ONE crab's render pose for one tick: every keyed body part plus that crab's giant-blow-up
/// placement. Its position in [`CrabArticulation::crabs`] IS its identity — the crab index the
/// client routes the frame to its own matching render rig by ([`crab_world`'s `CrabEnvId`]).
/// `repose` is `None` only before the bridge has published one for that crab (transiently at
/// spawn) — the client then leaves its placement untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct CrabFrame {
    /// Every keyed body part's arena-frame transform, in ascending `part`-tag order.
    pub parts: Vec<PartTransform>,
    /// The giant-blow-up placement, or `None` before the host has published one.
    pub repose: Option<ReposeWire>,
    /// This crab's brain label, exactly as the HOST formatted it (`Policy::brain_label` —
    /// `arch @shortdigest`, or an attributed failure state): a client renders the string
    /// verbatim, so who's-who can't be re-derived differently per peer (rl#200 increment 7).
    /// It rides the articulation stream — not a join-time table — because articulation
    /// already reaches BOTH client kinds (formation and mid-join) index-aligned with the
    /// crabs, and stays current if a future round ever rebinds; ~30 bytes/crab/frame is
    /// noise beside the part transforms. Capped at 255 bytes on the wire ([`clamp_label`]).
    pub brain_label: String,
}

/// Every crab's render pose for one tick — one [`CrabFrame`] per crab, in crab-index order. The
/// host captures it after stepping (post-`integrate_crab`/`publish_skin_repose`, so it is this
/// tick's settled pose) and broadcasts it; a windowed client writes each frame onto its own
/// frozen matching crab's render entities.
#[derive(Debug, Clone, PartialEq)]
pub struct CrabArticulation {
    /// The tick this pose is OF, matching the [`CoreSnapshot`](crate::snapshot::CoreSnapshot)
    /// tick it rides beside — what lets the windowed client pair the frame with the snapshot it
    /// ADOPTS from its jitter buffer (rl#194), instead of rendering the newest arrival raw.
    pub tick: u64,
    /// One frame per crab, in crab-index order (index = the snapshot's crab index = the host's
    /// crab-world env id).
    pub crabs: Vec<CrabFrame>,
    /// The host's piloted craft's pose, or `None` while the host is on foot (the body is
    /// despawned then, so absence IS the on-foot signal — a client clears its drawn craft on
    /// `None` rather than freezing a stale one).
    pub vehicle: Option<VehiclePoseWire>,
}

/// The host's piloted craft for one tick — its arena-frame rigidbody pose. Like the crabs, the
/// vehicle is rapier state only the HOST steps (host-authoritative, off the integer sim), so a
/// remote client can't compute where it is; this riding beside the crab poses is how the second
/// player sees the host fly (rl#192). The client draws the craft's collider wireframe — its one
/// visual (the craft has no mesh) — at this pose, under the same [`ReposeWire`] placement as the
/// crab cage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehiclePoseWire {
    /// Arena-frame translation, `[x, y, z]`.
    pub pos: [f32; 3],
    /// Arena-frame rotation quaternion, `[x, y, z, w]`.
    pub rot: [f32; 4],
}

/// Why decoding a [`CrabArticulation`] failed. Like a snapshot, a client must never render a
/// half-decoded pose, so a malformed buffer is a hard typed error, not a best-effort partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArticulationDecodeError {
    /// The buffer ended before a field the format requires was fully read.
    Truncated,
    /// A 1-byte present-flag (repose or vehicle) held a value other than 0 or 1.
    BadFlag,
    /// Bytes remained after a complete articulation decoded — a framing/length mismatch.
    TrailingBytes,
    /// A brain label's bytes were not valid UTF-8.
    BadLabel,
}

impl std::fmt::Display for ArticulationDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Truncated => "articulation buffer ended mid-field",
            Self::BadFlag => "articulation present-flag was neither 0 nor 1",
            Self::TrailingBytes => "trailing bytes after a complete articulation",
            Self::BadLabel => "articulation brain label was not valid UTF-8",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ArticulationDecodeError {}

impl CrabArticulation {
    /// Little-endian wire form: `tick(8) | n_crabs(4) | crab[ n_parts(4) part[ tag(1) pos(3×4)
    /// rot(4×4) ]… repose_present(1) [ shift(3×4) pivot(3×4) scale(4) ] label_len(1) label… ]… |
    /// vehicle_present(1) | [ pos(3×4) rot(4×4) ]`. [`from_bytes`](Self::from_bytes) is its
    /// exact inverse.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&(self.crabs.len() as u32).to_le_bytes());
        for crab in &self.crabs {
            out.extend_from_slice(&(crab.parts.len() as u32).to_le_bytes());
            for p in &crab.parts {
                out.push(p.part);
                for v in p.pos {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                for v in p.rot {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            match &crab.repose {
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
            let label = clamp_label(&crab.brain_label);
            out.push(label.len() as u8);
            out.extend_from_slice(label.as_bytes());
        }
        match &self.vehicle {
            None => out.push(0),
            Some(v) => {
                out.push(1);
                for c in v.pos {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                for c in v.rot {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        out
    }

    /// Inverse of [`to_bytes`](Self::to_bytes). Rejects a buffer that is too short, carries a bad
    /// present-flag, or has bytes left over.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArticulationDecodeError> {
        let mut r = Reader::new(bytes);
        let tick = u64::from_le_bytes(r.take()?);
        let n_crabs = u32::from_le_bytes(r.take::<4>()?) as usize;
        // Don't pre-allocate from the untrusted counts — grow bounded by the buffer (each `take`
        // Truncates the moment the real bytes run out), never by a claimed length.
        let mut crabs = Vec::new();
        for _ in 0..n_crabs {
            let n_parts = u32::from_le_bytes(r.take::<4>()?) as usize;
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
                    Some(ReposeWire {
                        shift,
                        pivot,
                        scale,
                    })
                }
                _ => return Err(ArticulationDecodeError::BadFlag),
            };
            let label_len = r.byte()? as usize;
            let brain_label = std::str::from_utf8(r.slice(label_len)?)
                .map_err(|_| ArticulationDecodeError::BadLabel)?
                .to_string();
            crabs.push(CrabFrame {
                parts,
                repose,
                brain_label,
            });
        }
        let vehicle = match r.byte()? {
            0 => None,
            1 => Some(VehiclePoseWire {
                pos: read_vec3(&mut r)?,
                rot: read_vec4(&mut r)?,
            }),
            _ => return Err(ArticulationDecodeError::BadFlag),
        };
        if !r.is_empty() {
            return Err(ArticulationDecodeError::TrailingBytes);
        }
        Ok(Self {
            tick,
            crabs,
            vehicle,
        })
    }
}

/// Bound a brain label to the wire's 1-byte length prefix, cutting on a char boundary. The
/// formatter (`Policy::brain_label`) already keeps labels far shorter; this is the codec's own
/// guarantee that `label.len() as u8` can't wrap regardless of what a caller hands it.
fn clamp_label(label: &str) -> &str {
    const MAX: usize = u8::MAX as usize;
    if label.len() <= MAX {
        return label;
    }
    let mut end = MAX;
    while !label.is_char_boundary(end) {
        end -= 1;
    }
    &label[..end]
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
        let end = self
            .at
            .checked_add(N)
            .ok_or(ArticulationDecodeError::Truncated)?;
        let slice = self
            .buf
            .get(self.at..end)
            .ok_or(ArticulationDecodeError::Truncated)?;
        self.at = end;
        Ok(slice.try_into().expect("slice length checked above"))
    }

    fn byte(&mut self) -> Result<u8, ArticulationDecodeError> {
        Ok(self.take::<1>()?[0])
    }

    /// A runtime-length read (the label bytes) — same bounds discipline as [`Self::take`].
    fn slice(&mut self, n: usize) -> Result<&'a [u8], ArticulationDecodeError> {
        let end = self
            .at
            .checked_add(n)
            .ok_or(ArticulationDecodeError::Truncated)?;
        let slice = self
            .buf
            .get(self.at..end)
            .ok_or(ArticulationDecodeError::Truncated)?;
        self.at = end;
        Ok(slice)
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
            crabs: vec![
                CrabFrame {
                    parts: vec![
                        PartTransform {
                            part: 0,
                            pos: [1.0, 2.0, 3.0],
                            rot: [0.0, 0.0, 0.0, 1.0],
                        },
                        PartTransform {
                            part: 7,
                            pos: [-4.5, 0.25, 9.0],
                            rot: [0.5, 0.5, 0.5, 0.5],
                        },
                    ],
                    repose: Some(ReposeWire {
                        shift: [10.0, 0.0, -20.0],
                        pivot: [0.0, 1.0, 0.0],
                        scale: 8.0,
                    }),
                    brain_label: "mlp512x3 @1a2b3c4d".to_string(),
                },
                // A second crab (rl#200 multi-brain) with a distinct pose and no repose yet,
                // so the per-crab framing + optional repose are both exercised.
                CrabFrame {
                    parts: vec![PartTransform {
                        part: 3,
                        pos: [7.0, 0.5, -2.0],
                        rot: [0.0, 1.0, 0.0, 0.0],
                    }],
                    repose: None,
                    // Distinct per-crab labels, and a failure state on the wire — the
                    // attribution channel is part of the format, not just the happy path.
                    brain_label: "REFUSED: wrong rig".to_string(),
                },
            ],
            vehicle: Some(VehiclePoseWire {
                pos: [2.0, 5.5, -1.0],
                rot: [
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                ],
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
        a.crabs[0].repose = None;
        assert_eq!(CrabArticulation::from_bytes(&a.to_bytes()).unwrap(), a);
    }

    #[test]
    fn roundtrip_without_vehicle() {
        let mut a = sample();
        a.vehicle = None;
        assert_eq!(CrabArticulation::from_bytes(&a.to_bytes()).unwrap(), a);
    }

    #[test]
    fn empty_crabs_roundtrip() {
        let a = CrabArticulation {
            tick: 0,
            crabs: vec![],
            vehicle: None,
        };
        assert_eq!(CrabArticulation::from_bytes(&a.to_bytes()).unwrap(), a);
    }

    #[test]
    fn empty_and_oversize_labels_roundtrip_bounded() {
        let mut a = sample();
        a.crabs[1].brain_label = String::new();
        // 300 bytes of 2-byte chars: the codec clamps to ≤255 on a char boundary (254 here)
        // rather than wrapping the length prefix or splitting a char.
        a.crabs[0].brain_label = "é".repeat(150);
        let back = CrabArticulation::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(back.crabs[1].brain_label, "");
        assert_eq!(back.crabs[0].brain_label, "é".repeat(127));
    }

    #[test]
    fn bad_label_utf8_is_rejected() {
        let mut a = sample();
        a.crabs[1].brain_label = "x".to_string();
        let mut bytes = a.to_bytes();
        // Corrupt the single label byte of the LAST crab: it sits right before the trailing
        // vehicle block (present-flag 1 + pos 12 + rot 16).
        let label_off = bytes.len() - (1 + 12 + 16) - 1;
        assert_eq!(bytes[label_off], b'x');
        bytes[label_off] = 0xFF;
        assert_eq!(
            CrabArticulation::from_bytes(&bytes),
            Err(ArticulationDecodeError::BadLabel)
        );
    }

    #[test]
    fn truncated_buffer_is_rejected() {
        let bytes = sample().to_bytes();
        assert_eq!(
            CrabArticulation::from_bytes(&bytes[..bytes.len() - 1]),
            Err(ArticulationDecodeError::Truncated)
        );
        assert_eq!(
            CrabArticulation::from_bytes(&[]),
            Err(ArticulationDecodeError::Truncated)
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = sample().to_bytes();
        bytes.push(0);
        assert_eq!(
            CrabArticulation::from_bytes(&bytes),
            Err(ArticulationDecodeError::TrailingBytes)
        );
    }

    #[test]
    fn bad_present_flag_is_rejected() {
        // Corrupt crab 0's repose-present flag: it sits after tick(8) + n_crabs(4) + crab 0's
        // n_parts(4) + its two parts (each 1 + 12 + 16 = 29 B).
        let flag_off = 8 + 4 + 4 + 2 * (1 + 12 + 16);
        let mut bytes = sample().to_bytes();
        bytes[flag_off] = 2;
        assert_eq!(
            CrabArticulation::from_bytes(&bytes),
            Err(ArticulationDecodeError::BadFlag)
        );
        // And the vehicle-present flag, after crab 0's repose block (flag + 3+3+1 f32s) + label
        // (len byte + bytes) and the whole of crab 1 (n_parts(4) + one part (29) + repose
        // flag(1) + its label).
        let label = |i: usize| 1 + sample().crabs[i].brain_label.len();
        let mut bytes = sample().to_bytes();
        bytes[flag_off + 1 + 7 * 4 + label(0) + 4 + 29 + 1 + label(1)] = 2;
        assert_eq!(
            CrabArticulation::from_bytes(&bytes),
            Err(ArticulationDecodeError::BadFlag)
        );
    }
}
