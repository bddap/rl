//! Byte-exact encodings — the layouts that define cross-peer agreement.
//!
//! Two kinds live here, both pinned byte-for-byte by tests:
//! - the [`Input`] wire codec (the per-tick input frame [`crate::transport`] ships), and
//! - the canonical field encodings (`pos_bytes`, the status/outcome tags) shared by
//!   [`Sim::state_hash`](crate::sim::Sim::state_hash) and
//!   [`CoreSnapshot`](crate::snapshot::CoreSnapshot) — one statement of each layout, so
//!   the hash and the snapshot cannot drift apart.

use crate::sim::{Input, Outcome, PlayerStatus, Pos};

impl Input {
    pub const WIRE_LEN: usize = 7;

    pub fn to_bytes(self) -> [u8; Self::WIRE_LEN] {
        debug_assert!(
            [self.move_strafe, self.move_forward, self.look_yaw]
                .iter()
                .all(|a| (-Self::AXIS_SCALE..=Self::AXIS_SCALE).contains(a)),
            "encoding an out-of-range axis; construct via Input::new"
        );
        let mut b = [0u8; Self::WIRE_LEN];
        b[0..2].copy_from_slice(&self.move_strafe.to_le_bytes());
        b[2..4].copy_from_slice(&self.move_forward.to_le_bytes());
        b[4..6].copy_from_slice(&self.look_yaw.to_le_bytes());
        b[6] = self.buttons;
        b
    }

    /// Axes clamp to ±[`AXIS_SCALE`](Self::AXIS_SCALE) on decode: the wire carries a
    /// full `i16` per axis, so a forged frame could otherwise push one 32× past full
    /// deflection. Honest senders already clamp at construction ([`new`](Self::new)),
    /// so this bites only forged bytes — and only the forger's own prediction diverges
    /// from the authoritative snapshots. Unknown button bits pass through deliberately;
    /// readers mask via [`pressed`](Self::pressed).
    pub fn from_bytes(b: [u8; Self::WIRE_LEN]) -> Self {
        let axis = |lo, hi| i16::from_le_bytes([lo, hi]).clamp(-Self::AXIS_SCALE, Self::AXIS_SCALE);
        Self {
            move_strafe: axis(b[0], b[1]),
            move_forward: axis(b[2], b[3]),
            look_yaw: axis(b[4], b[5]),
            buttons: b[6],
        }
    }
}

/// The canonical byte encoding of a [`Pos`]: x then z, little-endian.
pub(crate) fn pos_bytes(p: Pos) -> [u8; 16] {
    let Pos { x, z } = p;
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&x.to_le_bytes());
    b[8..].copy_from_slice(&z.to_le_bytes());
    b
}

/// Inverse of [`pos_bytes`].
pub(crate) fn pos_from_bytes(b: [u8; 16]) -> Pos {
    Pos {
        x: i64::from_le_bytes(b[..8].try_into().expect("split of a fixed 16-byte array")),
        z: i64::from_le_bytes(b[8..].try_into().expect("split of a fixed 16-byte array")),
    }
}

impl PlayerStatus {
    pub(crate) fn tag(self) -> u8 {
        match self {
            PlayerStatus::Alive => 0,
            PlayerStatus::Downed => 1,
            PlayerStatus::Extracted => 2,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(PlayerStatus::Alive),
            1 => Some(PlayerStatus::Downed),
            2 => Some(PlayerStatus::Extracted),
            _ => None,
        }
    }
}

impl Outcome {
    pub(crate) fn tag(self) -> u8 {
        match self {
            Outcome::Ongoing => 0,
            Outcome::Extracted => 1,
            Outcome::Wiped => 2,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Outcome::Ongoing),
            1 => Some(Outcome::Extracted),
            2 => Some(Outcome::Wiped),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::buttons;

    #[test]
    fn input_bytes_roundtrip() {
        // Identity holds on the in-range axis domain; out-of-range bytes clamp on
        // decode (next test) and out-of-range structs fail to_bytes' debug_assert.
        for inp in [
            Input::default(),
            Input::from_axes(1.0, -1.0),
            Input::new(-0.5, 0.25, 1.0, buttons::ACTION),
            Input {
                move_strafe: -Input::AXIS_SCALE,
                move_forward: Input::AXIS_SCALE,
                look_yaw: -123,
                buttons: 0xFF,
            },
        ] {
            assert_eq!(Input::from_bytes(inp.to_bytes()), inp);
        }
    }

    #[test]
    fn input_from_bytes_clamps_out_of_range_axes() {
        let mut b = [0u8; Input::WIRE_LEN];
        b[0..2].copy_from_slice(&i16::MIN.to_le_bytes());
        b[2..4].copy_from_slice(&i16::MAX.to_le_bytes());
        b[4..6].copy_from_slice(&(-Input::AXIS_SCALE - 1).to_le_bytes());
        b[6] = 0xFF;
        assert_eq!(
            Input::from_bytes(b),
            Input {
                move_strafe: -Input::AXIS_SCALE,
                move_forward: Input::AXIS_SCALE,
                look_yaw: -Input::AXIS_SCALE,
                buttons: 0xFF,
            }
        );
    }

    #[test]
    fn pos_bytes_layout_is_x_then_z_little_endian() {
        // Absolute bytes, not just a roundtrip: a self-consistent layout change (both
        // sides swap fields or flip endianness) would pass a roundtrip while silently
        // changing state_hash and the snapshot wire.
        let p = Pos {
            x: 0x0102_0304_0506_0708,
            z: -0x1112_1314_1516_1718,
        };
        let b = pos_bytes(p);
        assert_eq!(b[..8], p.x.to_le_bytes());
        assert_eq!(b[8..], p.z.to_le_bytes());
        assert_eq!(pos_from_bytes(b), p);
    }
}
