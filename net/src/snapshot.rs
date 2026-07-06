
use std::collections::BTreeMap;

use crate::sim::{Crab, Outcome, Player, PlayerId, PlayerStatus, Pos};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreSnapshot {
    /// The tick this snapshot is OF. A snapshot is a FULL state, so it versions the roster
    /// too — no separate roster epoch. (Clients adopt snapshots in arrival order:
    /// [`crate::lockstep::Lockstep::adopt_snapshots`].)
    pub tick: u64,
    pub players: BTreeMap<PlayerId, Player>,
    pub crabs: Vec<Crab>,
    pub outcome: Outcome,
    /// The participant set the server owns (sorted, deduped) — the server ships the roster;
    /// clients adopt it, never negotiate it. Not in
    /// [`Sim::state_hash`](crate::sim::Sim::state_hash) (config-level, not evolving state),
    /// so the round-trip test asserts it separately.
    pub roster: Vec<PlayerId>,
    /// Per player, the first [`TickMsg::issue_tick`](crate::lockstep::TickMsg::issue_tick) NOT
    /// yet consumed into this state — the input watermark a remote client prunes + replays its
    /// own prediction window against
    /// ([`Lockstep::reconcile_local_prediction`](crate::lockstep::Lockstep::reconcile_local_prediction)).
    /// The host steps at its own pace and consumes a remote's inputs as they arrive
    /// ([`crate::server`]), so consumption trails issuance by the transit lag — this map is
    /// what keeps the client's replay exact. SERVER-owned coordination metadata, not sim state:
    /// [`Sim::core_snapshot`](crate::sim::Sim::core_snapshot) leaves it empty, the server
    /// stamps it, and the client's `Lockstep` stashes + re-stamps it, so it is outside
    /// `state_hash` (like `roster`).
    pub input_next: BTreeMap<PlayerId, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotDecodeError {
    Truncated,
    BadTag,
    NoCrabs,
    TrailingBytes,
}

impl std::fmt::Display for SnapshotDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Truncated => "snapshot buffer ended mid-field",
            Self::BadTag => "snapshot held an unknown enum tag",
            Self::NoCrabs => "snapshot carried zero crabs (a round always has one — rl#114)",
            Self::TrailingBytes => "trailing bytes after a complete snapshot",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for SnapshotDecodeError {}

impl CoreSnapshot {
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

        out.extend_from_slice(&(self.crabs.len() as u32).to_le_bytes());
        for crab in &self.crabs {
            write_pos(&mut out, crab.pos());
            out.extend_from_slice(&crab.yaw().to_le_bytes());
        }

        out.push(self.outcome.tag());

        out.extend_from_slice(&(self.roster.len() as u32).to_le_bytes());
        for id in &self.roster {
            out.push(id.0);
        }

        out.extend_from_slice(&(self.input_next.len() as u32).to_le_bytes());
        for (id, next) in &self.input_next {
            out.push(id.0);
            out.extend_from_slice(&next.to_le_bytes());
        }
        out
    }

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

        let n_crabs = u32::from_le_bytes(r.take::<4>()?) as usize;
        if n_crabs == 0 {
            return Err(SnapshotDecodeError::NoCrabs);
        }
        let mut crabs = Vec::new();
        for _ in 0..n_crabs {
            let pos = read_pos(&mut r)?;
            let yaw = i32::from_le_bytes(r.take()?);
            crabs.push(Crab::from_parts(pos, yaw));
        }

        let outcome = Outcome::from_tag(r.byte()?).ok_or(SnapshotDecodeError::BadTag)?;

        let n_roster = u32::from_le_bytes(r.take::<4>()?) as usize;
        let mut roster = Vec::new();
        for _ in 0..n_roster {
            roster.push(PlayerId(r.byte()?));
        }

        let n_watermarks = u32::from_le_bytes(r.take::<4>()?) as usize;
        let mut input_next = BTreeMap::new();
        for _ in 0..n_watermarks {
            let id = PlayerId(r.byte()?);
            input_next.insert(id, u64::from_le_bytes(r.take()?));
        }

        if !r.is_empty() {
            return Err(SnapshotDecodeError::TrailingBytes);
        }
        Ok(Self {
            tick,
            players,
            crabs,
            outcome,
            roster,
            input_next,
        })
    }
}

fn write_pos(out: &mut Vec<u8>, p: Pos) {
    out.extend_from_slice(&p.x.to_le_bytes());
    out.extend_from_slice(&p.z.to_le_bytes());
}

fn read_pos(r: &mut Reader<'_>) -> Result<Pos, SnapshotDecodeError> {
    let x = i64::from_le_bytes(r.take()?);
    let z = i64::from_le_bytes(r.take()?);
    Ok(Pos { x, z })
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

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

    fn byte(&mut self) -> Result<u8, SnapshotDecodeError> {
        Ok(self.take::<1>()?[0])
    }

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
        sim.set_external_crab_pose(0, Pos { x: 1234, z: -5678 }, 42, 0);
        let mut snap = sim.core_snapshot();
        // Server-stamped watermarks (`Sim::core_snapshot` leaves them empty) — nonempty here so
        // the roundtrip exercises the map encoding.
        snap.input_next = BTreeMap::from([(PlayerId(0), 7), (PlayerId(1), 0)]);
        snap
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
        let status_off = 8 + 4 + 1 + 8 + 8 + 4;
        let mut bytes = sample().to_bytes();
        bytes[status_off] = 99;
        assert_eq!(
            CoreSnapshot::from_bytes(&bytes),
            Err(SnapshotDecodeError::BadTag)
        );
    }
}
