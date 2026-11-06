//! The host's per-tick craft poses, the render state a client adopts verbatim: every
//! pose is a WORLD pose (the world's coordinates are the sim's meters).

use crab_world::vehicle::VehicleKind;

#[derive(Debug, Clone, PartialEq)]
pub struct VehicleFrame {
    pub tick: u64,
    /// Ascending, unique pilot order — the decoder enforces it.
    pub vehicles: Vec<VehiclePoseWire>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehiclePoseWire {
    pub pilot: u8,
    /// Which craft this is — a client renders the kind's silhouette and can't infer it
    /// from the pose (rl#260).
    pub kind: VehicleKind,
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
pub enum VehicleFrameDecodeError {
    Truncated,
    TrailingBytes,
    UnorderedVehicles,
    BadVehicleKind,
}

impl std::fmt::Display for VehicleFrameDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Truncated => "vehicle frame ended mid-field",
            Self::TrailingBytes => "trailing bytes after a complete vehicle frame",
            Self::UnorderedVehicles => "vehicle frame was not in ascending pilot order",
            Self::BadVehicleKind => "vehicle frame carried an unknown kind byte",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for VehicleFrameDecodeError {}

impl VehicleFrame {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(10 + RECORD_LEN * self.vehicles.len());
        out.extend_from_slice(&self.tick.to_le_bytes());
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

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VehicleFrameDecodeError> {
        use VehicleFrameDecodeError as E;
        let mut r = Reader { buf: bytes, at: 0 };
        let tick = u64::from_le_bytes(r.take()?);
        let n = u16::from_le_bytes(r.take()?) as usize;
        let mut vehicles: Vec<VehiclePoseWire> = Vec::new();
        for _ in 0..n {
            let v = VehiclePoseWire {
                pilot: r.take::<1>()?[0],
                kind: VehicleKind::from_wire_byte(r.take::<1>()?[0]).ok_or(E::BadVehicleKind)?,
                pos: r.floats()?,
                rot: r.floats()?,
                thrust: r.take::<3>()?.map(|b| b as i8),
            };
            if vehicles.last().is_some_and(|prev| prev.pilot >= v.pilot) {
                return Err(E::UnorderedVehicles);
            }
            vehicles.push(v);
        }
        if r.at != bytes.len() {
            return Err(E::TrailingBytes);
        }
        Ok(Self { tick, vehicles })
    }
}

const RECORD_LEN: usize = 1 + 1 + 12 + 16 + 3;

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], VehicleFrameDecodeError> {
        let slice = self
            .buf
            .get(self.at..self.at + N)
            .ok_or(VehicleFrameDecodeError::Truncated)?;
        self.at += N;
        Ok(slice.try_into().expect("slice length checked above"))
    }

    fn floats<const N: usize>(&mut self) -> Result<[f32; N], VehicleFrameDecodeError> {
        let mut out = [0.0; N];
        for c in &mut out {
            *c = f32::from_le_bytes(self.take()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> VehicleFrame {
        VehicleFrame {
            tick: 4242,
            vehicles: vec![
                VehiclePoseWire {
                    pilot: 0,
                    kind: VehicleKind::Plane,
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
                    kind: VehicleKind::Ship,
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
        assert_eq!(VehicleFrame::from_bytes(&a.to_bytes()).unwrap(), a);
        let empty = VehicleFrame {
            tick: 0,
            vehicles: vec![],
        };
        assert_eq!(VehicleFrame::from_bytes(&empty.to_bytes()).unwrap(), empty);
    }

    #[test]
    fn truncated_buffer_is_rejected() {
        let bytes = sample().to_bytes();
        assert_eq!(
            VehicleFrame::from_bytes(&bytes[..bytes.len() - 1]),
            Err(VehicleFrameDecodeError::Truncated)
        );
        assert_eq!(
            VehicleFrame::from_bytes(&[]),
            Err(VehicleFrameDecodeError::Truncated)
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = sample().to_bytes();
        bytes.push(0);
        assert_eq!(
            VehicleFrame::from_bytes(&bytes),
            Err(VehicleFrameDecodeError::TrailingBytes)
        );
    }

    #[test]
    fn vehicle_count_past_the_bytes_is_rejected() {
        let a = sample();
        let mut bytes = a.to_bytes();
        assert_eq!(bytes[8], a.vehicles.len() as u8);
        bytes[8] += 1;
        assert_eq!(
            VehicleFrame::from_bytes(&bytes),
            Err(VehicleFrameDecodeError::Truncated)
        );
    }

    #[test]
    fn unknown_vehicle_kind_byte_is_rejected() {
        let a = sample();
        let mut bytes = a.to_bytes();
        let kind_off = 10 + 1;
        assert_eq!(bytes[kind_off], a.vehicles[0].kind.wire_byte());
        bytes[kind_off] = 0xEE;
        assert_eq!(
            VehicleFrame::from_bytes(&bytes),
            Err(VehicleFrameDecodeError::BadVehicleKind)
        );
    }

    #[test]
    fn unordered_or_duplicate_vehicle_pilots_are_rejected() {
        let mut a = sample();
        a.vehicles.swap(0, 1);
        assert_eq!(
            VehicleFrame::from_bytes(&a.to_bytes()),
            Err(VehicleFrameDecodeError::UnorderedVehicles)
        );
        let mut a = sample();
        a.vehicles[1].pilot = a.vehicles[0].pilot;
        assert_eq!(
            VehicleFrame::from_bytes(&a.to_bytes()),
            Err(VehicleFrameDecodeError::UnorderedVehicles)
        );
    }
}
