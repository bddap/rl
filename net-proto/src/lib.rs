//! The platform-free protocol core (rl#411 stage 2). No tokio, no iroh, no clock,
//! no IO — the link layer (`net-link`) feeds it bytes and injected time.
//! Enforced by construction: this crate checks on wasm32 (see test-map.json).

pub mod cadence;
pub mod client;
pub mod codec;
pub mod cordic;
pub mod formation;
pub mod membership;
mod reader;
pub mod roster;
pub mod server;
pub mod sim;
pub mod snapshot;
pub mod vehicles;
pub mod wire;

/// What a peer advertises about its local world during formation/admission: the digest
/// of the terrain bake it stands on. Two peers whose binaries embed different bakes stand
/// on different mountains, so [`SyncStamp::agrees`] refuses them. The field is
/// `pub(crate)`: [`SyncStamp::local`] is the honest constructor and
/// [`SyncStamp::from_wire`] the link layer's decode path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncStamp {
    pub(crate) terrain_digest: u64,
}

impl SyncStamp {
    /// Unset: [`membership::Membership`]'s pre-`with_stamp` start, and tests exercising
    /// formation mechanics rather than the sync check. A zero digest never agrees.
    pub const ZERO: SyncStamp = SyncStamp { terrain_digest: 0 };

    pub const WIRE_LEN: usize = 8;

    pub fn to_wire(self) -> [u8; Self::WIRE_LEN] {
        self.terrain_digest.to_le_bytes()
    }

    pub fn from_wire(b: [u8; Self::WIRE_LEN]) -> Self {
        Self {
            terrain_digest: u64::from_le_bytes(b),
        }
    }

    pub fn terrain_digest(self) -> u64 {
        self.terrain_digest
    }

    pub fn local() -> Self {
        Self {
            terrain_digest: crab_world::terrain::gcr_bake_digest(),
        }
    }

    /// THE shared-world arbiter (rl#336): formation ([`membership::Membership::synced`])
    /// and mid-game admission ([`server::may_admit_joiner`]) both derive from this, so a
    /// joiner a host would refuse is exactly a peer formation would not start with.
    pub fn agrees(host: SyncStamp, peer: SyncStamp) -> bool {
        host.terrain_digest != 0 && peer.terrain_digest == host.terrain_digest
    }
}

#[cfg(test)]
mod desync_test {

    use std::collections::BTreeMap;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::sim::{Input, PlayerId, Sim, buttons};

    fn input_log(seed: u64, players: &[PlayerId], ticks: usize) -> Vec<BTreeMap<PlayerId, Input>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..ticks)
            .map(|_| {
                players
                    .iter()
                    .map(|&p| {
                        let x: f32 = rng.gen_range(-1.0..=1.0);
                        let y: f32 = rng.gen_range(-1.0..=1.0);
                        let look: f32 = rng.gen_range(-1.0..=1.0);
                        let act = if rng.gen_range(0..8) == 0 {
                            buttons::ACTION
                        } else {
                            0
                        };
                        (p, Input::new(x, y, look, act))
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn two_sims_stay_in_lockstep_on_one_input_log() {
        let players: Vec<PlayerId> = (0..4).map(PlayerId).collect();
        let seed = 0xA11CE;
        let log = input_log(0xBEEF, &players, 240); // 8s at TICK_HZ=30

        let mut a = Sim::new(seed, &players);
        let mut b = Sim::new(seed, &players);
        assert_eq!(a.state_hash(), b.state_hash(), "initial state must match");

        for (t, inputs) in log.iter().enumerate() {
            a.step(inputs, &BTreeMap::new());
            b.step(inputs, &BTreeMap::new());
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "state hash diverged at tick {t} — sim is nondeterministic"
            );
        }
        assert_eq!(a.tick(), log.len() as u64);
    }

    #[test]
    fn replaying_the_same_log_reproduces_the_same_final_hash() {
        let players: Vec<PlayerId> = (0..3).map(PlayerId).collect();
        let log = input_log(0x1234, &players, 120);

        let run = || {
            let mut s = Sim::new(77, &players);
            for inputs in &log {
                s.step(inputs, &BTreeMap::new());
            }
            s.state_hash()
        };
        assert_eq!(
            run(),
            run(),
            "same inputs must yield the same final state hash"
        );
    }

    #[test]
    fn restart_edge_in_a_log_keeps_two_sims_in_lockstep() {
        let players: Vec<PlayerId> = (0..4).map(PlayerId).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(0x6A0D);
        let log: Vec<BTreeMap<PlayerId, Input>> = (0..300)
            .map(|t| {
                players
                    .iter()
                    .map(|&p| {
                        let x: f32 = rng.gen_range(-1.0..=1.0);
                        let y: f32 = rng.gen_range(-1.0..=1.0);
                        let look: f32 = rng.gen_range(-1.0..=1.0);
                        let btns = if t % 50 == 7 { buttons::RESTART } else { 0 };
                        (p, Input::new(x, y, look, btns))
                    })
                    .collect()
            })
            .collect();

        let mut a = Sim::new(0xC0FFEE, &players);
        let mut b = Sim::new(0xC0FFEE, &players);
        assert_eq!(a.state_hash(), b.state_hash(), "initial state must match");
        let mut restarts = 0u32;
        for (t, inputs) in log.iter().enumerate() {
            restarts += u32::from(a.step(inputs, &BTreeMap::new()));
            b.step(inputs, &BTreeMap::new());
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "sims diverged at tick {t} across a RESTART edge"
            );
        }
        assert_eq!(restarts, 6, "each periodic RESTART press fired once");
        assert_eq!(
            a.tick(),
            log.len() as u64,
            "restarts never rewind the tick counter"
        );
    }
}
