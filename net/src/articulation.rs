#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartTransform {
    pub part: u8,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrabFrame {
    pub parts: Vec<PartTransform>,
    /// This crab's brain label, exactly as the HOST formatted it (`Policy::brain_label` —
    /// `arch @shortdigest`, or an attributed failure state): a client renders the string
    /// verbatim, so who's-who can't be re-derived differently per peer (rl#200 increment 7).
    /// It rides the articulation stream — not a join-time table — because articulation
    /// already reaches BOTH client kinds (formation and mid-join) index-aligned with the
    /// crabs, and stays current if a future round ever rebinds; ~30 bytes/crab/frame is
    /// noise beside the part transforms. Capped at 255 bytes on the wire ([`clamp_label`]).
    pub brain_label: String,
}

/// Every pose on this wire is a WORLD pose (rl#298 stage 5: one frame — the world's
/// coordinates are the sim's meters), so a client renders parts and vehicles verbatim;
/// the per-round `arena_anchor` translate and the per-crab skin-repose shift this
/// message used to carry died with the bridge (both are identically zero in one frame).
#[derive(Debug, Clone, PartialEq)]
pub struct CrabArticulation {
    pub tick: u64,
    pub crabs: Vec<CrabFrame>,
    pub vehicles: Vec<VehiclePoseWire>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehiclePoseWire {
    pub pilot: u8,
    /// Which craft this is — a client renders the kind's silhouette and can't infer it
    /// from the pose (rl#260).
    pub kind: crab_world::vehicle::VehicleKind,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    /// Body-frame thrust command, per-axis fractions in [-1, 1] quantized ×127 (rl#308):
    /// what fires the craft's exhaust plumes on every peer. A byte per axis — this drives
    /// a visual, not physics.
    pub thrust: [i8; 3],
}

impl VehiclePoseWire {
    pub fn quantize_thrust(v: bevy::math::Vec3) -> [i8; 3] {
        v.to_array()
            .map(|c| (c.clamp(-1.0, 1.0) * 127.0).round() as i8)
    }

    pub fn thrust_fraction(&self) -> bevy::math::Vec3 {
        bevy::math::Vec3::from_array(self.thrust.map(|b| b as f32 / 127.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArticulationDecodeError {
    Truncated,
    TrailingBytes,
    /// A brain label's bytes were not valid UTF-8.
    BadLabel,
    UnorderedVehicles,
    /// A vehicle pose's kind byte mapped to no [`crab_world::vehicle::VehicleKind`].
    BadVehicleKind,
}

impl std::fmt::Display for ArticulationDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Truncated => "articulation buffer ended mid-field",
            Self::TrailingBytes => "trailing bytes after a complete articulation",
            Self::BadLabel => "articulation brain label was not valid UTF-8",
            Self::UnorderedVehicles => "articulation vehicles were not in ascending pilot order",
            Self::BadVehicleKind => "articulation vehicle carried an unknown kind byte",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ArticulationDecodeError {}

impl CrabArticulation {
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
            let label = clamp_label(&crab.brain_label);
            out.push(label.len() as u8);
            out.extend_from_slice(label.as_bytes());
        }
        out.extend_from_slice(&(self.vehicles.len() as u16).to_le_bytes());
        for v in &self.vehicles {
            out.push(v.pilot);
            out.push(v.kind.wire_byte());
            for c in v.pos {
                out.extend_from_slice(&c.to_le_bytes());
            }
            for c in v.rot {
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&v.thrust.map(|b| b as u8));
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArticulationDecodeError> {
        let mut r = Reader::new(bytes);
        let tick = u64::from_le_bytes(r.take()?);
        let n_crabs = u32::from_le_bytes(r.take::<4>()?) as usize;
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
            let label_len = r.byte()? as usize;
            let brain_label = std::str::from_utf8(r.slice(label_len)?)
                .map_err(|_| ArticulationDecodeError::BadLabel)?
                .to_string();
            crabs.push(CrabFrame { parts, brain_label });
        }
        let n_vehicles = u16::from_le_bytes(r.take::<2>()?) as usize;
        let mut vehicles: Vec<VehiclePoseWire> = Vec::new();
        for _ in 0..n_vehicles {
            let v = VehiclePoseWire {
                pilot: r.byte()?,
                kind: crab_world::vehicle::VehicleKind::from_wire_byte(r.byte()?)
                    .ok_or(ArticulationDecodeError::BadVehicleKind)?,
                pos: read_vec3(&mut r)?,
                rot: read_vec4(&mut r)?,
                thrust: r.take::<3>()?.map(|b| b as i8),
            };
            if vehicles.last().is_some_and(|prev| prev.pilot >= v.pilot) {
                return Err(ArticulationDecodeError::UnorderedVehicles);
            }
            vehicles.push(v);
        }
        if !r.is_empty() {
            return Err(ArticulationDecodeError::TrailingBytes);
        }
        Ok(Self {
            tick,
            crabs,
            vehicles,
        })
    }
}

/// Bound a brain label to the wire's 1-byte length prefix, cutting on a char boundary. The
/// formatter (`Policy::brain_label`) already keeps labels far shorter; this is the codec's own
/// guarantee that `label.len() as u8` can't wrap regardless of what a caller hands it.
fn clamp_label(label: &str) -> &str {
    crab_world::truncate_at_char_boundary(label, u8::MAX as usize)
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

    /// One serialized vehicle record: pilot, kind, pos, rot, thrust (rl#308).
    const VEHICLE_RECORD: usize = 1 + 1 + 12 + 16 + 3;

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
                    brain_label: "mlp512x3 @1a2b3c4d".to_string(),
                },
                CrabFrame {
                    parts: vec![PartTransform {
                        part: 3,
                        pos: [7.0, 0.5, -2.0],
                        rot: [0.0, 1.0, 0.0, 0.0],
                    }],
                    // Distinct per-crab labels, and a failure state on the wire — the
                    // attribution channel is part of the format, not just the happy path.
                    brain_label: "REFUSED: wrong rig".to_string(),
                },
            ],
            vehicles: vec![
                VehiclePoseWire {
                    pilot: 0,
                    kind: crab_world::vehicle::VehicleKind::Plane,
                    pos: [2.0, 5.5, -1.0],
                    rot: [
                        0.0,
                        std::f32::consts::FRAC_1_SQRT_2,
                        0.0,
                        std::f32::consts::FRAC_1_SQRT_2,
                    ],
                    thrust: [127, -64, 0],
                },
                VehiclePoseWire {
                    pilot: 2,
                    kind: crab_world::vehicle::VehicleKind::Ship,
                    pos: [-3.0, 1.5, 4.0],
                    rot: [0.0, 0.0, 0.0, 1.0],
                    thrust: [0, 13, -127],
                },
            ],
        }
    }

    #[test]
    fn bytes_roundtrip_is_exact() {
        let a = sample();
        assert_eq!(CrabArticulation::from_bytes(&a.to_bytes()).unwrap(), a);
    }

    #[test]
    fn roundtrip_without_vehicles() {
        let mut a = sample();
        a.vehicles.clear();
        assert_eq!(CrabArticulation::from_bytes(&a.to_bytes()).unwrap(), a);
    }

    #[test]
    fn empty_crabs_roundtrip() {
        let a = CrabArticulation {
            tick: 0,
            crabs: vec![],
            vehicles: vec![],
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
        let label_off = bytes.len() - (2 + 2 * VEHICLE_RECORD) - 1;
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
    fn vehicle_count_past_the_bytes_is_rejected() {
        let a = sample();
        let mut bytes = a.to_bytes();
        let count_off = bytes.len() - 2 * VEHICLE_RECORD - 2;
        assert_eq!(bytes[count_off], a.vehicles.len() as u8);
        bytes[count_off] += 1;
        assert_eq!(
            CrabArticulation::from_bytes(&bytes),
            Err(ArticulationDecodeError::Truncated)
        );
    }

    #[test]
    fn unknown_vehicle_kind_byte_is_rejected() {
        let a = sample();
        let mut bytes = a.to_bytes();
        // First vehicle record follows the u16 count: pilot byte, then the kind byte.
        let kind_off = bytes.len() - 2 * VEHICLE_RECORD + 1;
        assert_eq!(bytes[kind_off], a.vehicles[0].kind.wire_byte());
        bytes[kind_off] = 0xEE;
        assert_eq!(
            CrabArticulation::from_bytes(&bytes),
            Err(ArticulationDecodeError::BadVehicleKind)
        );
    }

    #[test]
    fn unordered_or_duplicate_vehicle_pilots_are_rejected() {
        let mut a = sample();
        a.vehicles.swap(0, 1);
        assert_eq!(
            CrabArticulation::from_bytes(&a.to_bytes()),
            Err(ArticulationDecodeError::UnorderedVehicles)
        );
        let mut a = sample();
        a.vehicles[1].pilot = a.vehicles[0].pilot;
        assert_eq!(
            CrabArticulation::from_bytes(&a.to_bytes()),
            Err(ArticulationDecodeError::UnorderedVehicles)
        );
    }
}
