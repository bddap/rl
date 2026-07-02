//! `CoreSnapshot` — the host-authoritative game-state seam.
//!
//! The host-authoritative MP design (`docs/gcr-mp-host-authoritative.md`) makes the integer
//! [`Sim`](crate::sim::Sim) the authoritative state and ships it whole each tick: the host
//! steps the sim and broadcasts a snapshot; every client — including the host's own local
//! client, and in single-player the *only* client — reads game state from that snapshot
//! rather than from a sim it stepped itself. SP is MP with zero remote clients, one path
//! ([[sp-is-mp-special-case]]), so SP **always serializes too** — no by-reference-in-SP
//! fork ([[silent-fallback-antipattern]]).
//!
//! `CoreSnapshot` carries the authoritative integer game state and **nothing render-only** —
//! the ~13 crab body-part transforms the skinned mesh needs ride a SEPARATE
//! `#[cfg(feature = "render")]` extension frame, never this struct, so the no-render trainer
//! build never pulls a render type through here.
//!
//! It reuses [`Sim`](crate::sim::Sim)'s own field types ([`Player`], [`Crab`], …) instead
//! of re-encoding into parallel structs, so completeness is caught at the one construction
//! site, [`Sim::core_snapshot`](crate::sim::Sim::core_snapshot): an exhaustive no-`..`
//! destructure stops the build until a newly-added authoritative field is carried, exactly
//! the discipline [`Sim::state_hash`](crate::sim::Sim::state_hash) uses. A hand-mirrored
//! copy struct would have dropped the field silently.
//!
//! The encoding is hand-rolled little-endian — like [`Input::to_bytes`](crate::sim::Input)
//! and the sim's own `state_hash` — which keeps the deterministic sim module free of serde
//! (the same split telemetry's `Outcome` shim keeps). The fields are POD integers, so the
//! wire is fully deterministic; serializing at the snapshot boundary is inert w.r.t. the
//! determinism firewall, which governs the *step* (no `HashMap` walk / `thread_rng` /
//! wall-clock), not the wire.

use std::collections::BTreeMap;

use crate::sim::{Crab, Outcome, Player, PlayerId, PlayerStatus, Pos};

/// The authoritative game state for one tick — everything game logic and rendering need,
/// and nothing else. Reuses [`Sim`](crate::sim::Sim)'s own entity types so the carried set
/// can't drift from the sim's (see the module docs). ~100–200 bytes for a whole match.
///
/// Built only by [`Sim::core_snapshot`](crate::sim::Sim::core_snapshot) and consumed by
/// [`Sim::apply_core_snapshot`](crate::sim::Sim::apply_core_snapshot); [`to_bytes`] /
/// [`from_bytes`] are the deterministic wire form between them.
///
/// [`to_bytes`]: CoreSnapshot::to_bytes
/// [`from_bytes`]: CoreSnapshot::from_bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreSnapshot {
    /// The tick this snapshot is OF. A snapshot is a FULL state, so it versions the roster
    /// too — no separate roster epoch. (Clients adopt snapshots in arrival order:
    /// [`crate::lockstep::Lockstep::adopt_snapshots`].)
    pub tick: u64,
    /// First-person players, in `PlayerId` order. Reuses [`Sim`](crate::sim::Sim)'s own
    /// `BTreeMap` so a new `Player` field is carried by construction, not re-encoded.
    pub players: BTreeMap<PlayerId, Player>,
    /// The giant crab's authoritative integer pose (`Pos` + yaw). The float rapier body and
    /// its per-tick physics digest are NOT carried — the client renders the pose, never the
    /// solver.
    pub crab: Crab,
    /// The round result the client reads to end the round.
    pub outcome: Outcome,
    /// The participant set the server owns (sorted, deduped) — the server ships the roster;
    /// peers never agree it tick-for-tick. Not in
    /// [`Sim::state_hash`](crate::sim::Sim::state_hash) (config-level, can't differ between
    /// in-sync peers), so the round-trip test asserts it separately.
    pub roster: Vec<PlayerId>,
}

/// Why decoding a [`CoreSnapshot`] from bytes failed. A snapshot is host-authored and a
/// client must never silently render a half-decoded state ([[silent-fallback-antipattern]]),
/// so every malformed buffer is a hard, typed error rather than a best-effort partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotDecodeError {
    /// The buffer ended before a field that the format requires was fully read.
    Truncated,
    /// A 1-byte enum tag (player status or outcome) held a value no variant maps to.
    BadTag,
    /// Bytes remained after a complete snapshot was decoded — a framing/length mismatch.
    TrailingBytes,
}

impl std::fmt::Display for SnapshotDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Truncated => "snapshot buffer ended mid-field",
            Self::BadTag => "snapshot held an unknown enum tag",
            Self::TrailingBytes => "trailing bytes after a complete snapshot",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for SnapshotDecodeError {}

impl CoreSnapshot {
    /// Deterministic little-endian wire form. The fields are POD integers, so the same
    /// `CoreSnapshot` always encodes to the same bytes on every target — no map-order or
    /// float nondeterminism (`players` iterates in `PlayerId` order; `roster` is already
    /// sorted). [`from_bytes`](CoreSnapshot::from_bytes) is its exact inverse.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.tick.to_le_bytes());

        out.extend_from_slice(&(self.players.len() as u32).to_le_bytes());
        for (id, player) in &self.players {
            out.push(id.0);
            write_pos(&mut out, player.pos());
            out.extend_from_slice(&player.yaw().to_le_bytes());
            out.push(player.status().tag());
        }

        write_pos(&mut out, self.crab.pos());
        out.extend_from_slice(&self.crab.yaw().to_le_bytes());

        out.push(self.outcome.tag());

        out.extend_from_slice(&(self.roster.len() as u32).to_le_bytes());
        for id in &self.roster {
            out.push(id.0);
        }
        out
    }

    /// Inverse of [`to_bytes`](CoreSnapshot::to_bytes). Rejects any buffer that is too
    /// short, carries an unknown enum tag, or has bytes left over — a host snapshot is
    /// trusted to be well-formed, so a malformed one is a hard error, never a partial apply.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotDecodeError> {
        let mut r = Reader::new(bytes);

        let tick = u64::from_le_bytes(r.take()?);

        let n_players = u32::from_le_bytes(r.take::<4>()?) as usize;
        let mut players = BTreeMap::new();
        for _ in 0..n_players {
            let id = PlayerId(r.byte()?);
            let pos = read_pos(&mut r)?;
            let yaw = i32::from_le_bytes(r.take()?);
            let status = PlayerStatus::from_tag(r.byte()?).ok_or(SnapshotDecodeError::BadTag)?;
            players.insert(id, Player::from_parts(pos, yaw, status));
        }

        let crab = {
            let pos = read_pos(&mut r)?;
            let yaw = i32::from_le_bytes(r.take()?);
            Crab::from_parts(pos, yaw)
        };

        let outcome = Outcome::from_tag(r.byte()?).ok_or(SnapshotDecodeError::BadTag)?;

        let n_roster = u32::from_le_bytes(r.take::<4>()?) as usize;
        // Don't pre-allocate from the untrusted count — a corrupt length would reserve
        // gigabytes before any bounds check. Each `byte()` Truncates the moment the real
        // bytes run out, so growth is bounded by the buffer, not the claimed count.
        let mut roster = Vec::new();
        for _ in 0..n_roster {
            roster.push(PlayerId(r.byte()?));
        }

        if !r.is_empty() {
            return Err(SnapshotDecodeError::TrailingBytes);
        }
        Ok(Self {
            tick,
            players,
            crab,
            outcome,
            roster,
        })
    }
}

/// Append a [`Pos`] as `x` then `z`, little-endian — the same order [`read_pos`] reads back.
fn write_pos(out: &mut Vec<u8>, p: Pos) {
    out.extend_from_slice(&p.x.to_le_bytes());
    out.extend_from_slice(&p.z.to_le_bytes());
}

fn read_pos(r: &mut Reader<'_>) -> Result<Pos, SnapshotDecodeError> {
    let x = i64::from_le_bytes(r.take()?);
    let z = i64::from_le_bytes(r.take()?);
    Ok(Pos { x, z })
}

/// A bounds-checked forward cursor over the snapshot bytes. Every read advances and returns
/// [`SnapshotDecodeError::Truncated`] past the end, so a short buffer can never read past it.
struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    /// Read a fixed `N`-byte array, advancing the cursor.
    fn take<const N: usize>(&mut self) -> Result<[u8; N], SnapshotDecodeError> {
        let end = self
            .at
            .checked_add(N)
            .ok_or(SnapshotDecodeError::Truncated)?;
        let slice = self
            .buf
            .get(self.at..end)
            .ok_or(SnapshotDecodeError::Truncated)?;
        self.at = end;
        Ok(slice.try_into().expect("slice length checked above"))
    }

    /// Read one byte, advancing the cursor.
    fn byte(&mut self) -> Result<u8, SnapshotDecodeError> {
        Ok(self.take::<1>()?[0])
    }

    /// Whether every byte has been consumed.
    fn is_empty(&self) -> bool {
        self.at == self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Sim;

    fn sample() -> CoreSnapshot {
        let mut sim = Sim::new(9, &[PlayerId(0), PlayerId(1)]);
        sim.set_external_crab_pose(Pos { x: 1234, z: -5678 }, 42, 0);
        sim.core_snapshot()
    }

    #[test]
    fn bytes_roundtrip_is_exact() {
        let snap = sample();
        assert_eq!(CoreSnapshot::from_bytes(&snap.to_bytes()).unwrap(), snap);
    }

    #[test]
    fn truncated_buffer_is_rejected() {
        let bytes = sample().to_bytes();
        assert_eq!(
            CoreSnapshot::from_bytes(&bytes[..bytes.len() - 1]),
            Err(SnapshotDecodeError::Truncated)
        );
        assert_eq!(
            CoreSnapshot::from_bytes(&[]),
            Err(SnapshotDecodeError::Truncated)
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = sample().to_bytes();
        bytes.push(0);
        assert_eq!(
            CoreSnapshot::from_bytes(&bytes),
            Err(SnapshotDecodeError::TrailingBytes)
        );
    }

    #[test]
    fn unknown_status_tag_is_rejected() {
        // Layout: tick(8) + player_count(4) + per player[ id(1) pos.x(8) pos.z(8) yaw(4)
        // status(1) ]… — so the first player's status byte sits at this offset. Corrupt it
        // to a value no variant maps to and decoding must fail loudly, not default.
        let status_off = 8 + 4 + 1 + 8 + 8 + 4;
        let mut bytes = sample().to_bytes();
        bytes[status_off] = 99;
        assert_eq!(
            CoreSnapshot::from_bytes(&bytes),
            Err(SnapshotDecodeError::BadTag)
        );
    }
}
