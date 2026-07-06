
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartTransform {
    pub part: u8,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReposeWire {
    pub shift: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrabFrame {
    pub parts: Vec<PartTransform>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct CrabArticulation {
    pub tick: u64,
    pub crabs: Vec<CrabFrame>,
    pub vehicles: Vec<VehiclePoseWire>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehiclePoseWire {
    pub pilot: u8,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArticulationDecodeError {
    Truncated,
    BadFlag,
    TrailingBytes,
    /// A brain label's bytes were not valid UTF-8.
    BadLabel,
    UnorderedVehicles,
}

impl std::fmt::Display for ArticulationDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Truncated => "articulation buffer ended mid-field",
            Self::BadFlag => "articulation present-flag was neither 0 nor 1",
            Self::TrailingBytes => "trailing bytes after a complete articulation",
            Self::BadLabel => "articulation brain label was not valid UTF-8",
            Self::UnorderedVehicles => "articulation vehicles were not in ascending pilot order",
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
            match &crab.repose {
                None => out.push(0),
                Some(r) => {
                    out.push(1);
                    for v in r.shift {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            let label = clamp_label(&crab.brain_label);
            out.push(label.len() as u8);
            out.extend_from_slice(label.as_bytes());
        }
        out.extend_from_slice(&(self.vehicles.len() as u16).to_le_bytes());
        for v in &self.vehicles {
            out.push(v.pilot);
            for c in v.pos {
                out.extend_from_slice(&c.to_le_bytes());
            }
            for c in v.rot {
                out.extend_from_slice(&c.to_le_bytes());
            }
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
            let repose = match r.byte()? {
                0 => None,
                1 => Some(ReposeWire {
                    shift: read_vec3(&mut r)?,
                }),
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
        let n_vehicles = u16::from_le_bytes(r.take::<2>()?) as usize;
        let mut vehicles: Vec<VehiclePoseWire> = Vec::new();
        for _ in 0..n_vehicles {
            let v = VehiclePoseWire {
                pilot: r.byte()?,
                pos: read_vec3(&mut r)?,
                rot: read_vec4(&mut r)?,
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
                    }),
                    brain_label: "mlp512x3 @1a2b3c4d".to_string(),
                },
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
            vehicles: vec![
                VehiclePoseWire {
                    pilot: 0,
                    pos: [2.0, 5.5, -1.0],
                    rot: [
                        0.0,
                        std::f32::consts::FRAC_1_SQRT_2,
                        0.0,
                        std::f32::consts::FRAC_1_SQRT_2,
                    ],
                },
                VehiclePoseWire {
                    pilot: 2,
                    pos: [-3.0, 1.5, 4.0],
                    rot: [0.0, 0.0, 0.0, 1.0],
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
    fn roundtrip_without_repose() {
        let mut a = sample();
        a.crabs[0].repose = None;
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
        let label_off = bytes.len() - (2 + 2 * (1 + 12 + 16)) - 1;
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
        let flag_off = 8 + 4 + 4 + 2 * (1 + 12 + 16);
        let mut bytes = sample().to_bytes();
        bytes[flag_off] = 2;
        assert_eq!(
            CrabArticulation::from_bytes(&bytes),
            Err(ArticulationDecodeError::BadFlag)
        );
    }

    #[test]
    fn vehicle_count_past_the_bytes_is_rejected() {
        let a = sample();
        let mut bytes = a.to_bytes();
        let count_off = bytes.len() - 2 * (1 + 12 + 16) - 2;
        assert_eq!(bytes[count_off], a.vehicles.len() as u8);
        bytes[count_off] += 1;
        assert_eq!(
            CrabArticulation::from_bytes(&bytes),
            Err(ArticulationDecodeError::Truncated)
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
