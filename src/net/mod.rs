//! Multiplayer netcode foundation (rl#39): deterministic lockstep simulation over
//! iroh LAN transport. Multiplayer-first, so the whole game (rl#38) is built on top
//! of these three pieces:
//!
//! - [`sim`] — the deterministic simulation core. The contract (pure step, complete
//!   state hash, no nondeterminism) is what every later game system must honor.
//!   Phase 1 swaps the trivial dot world here for the real world + giant crab.
//! - [`lockstep`] — the fixed-tick driver: exchange inputs (not state) with an
//!   input-delay buffer, apply in a fixed peer order, advance, and compare state
//!   hashes to catch desync. Transport-agnostic.
//! - [`transport`] — iroh mDNS LAN discovery + per-tick message exchange over QUIC.
//!
//! The determinism-critical code ([`sim`] + [`lockstep`]) is pure and sync; all the
//! async/IO lives in [`transport`]. That separation is deliberate: it keeps the part
//! that MUST be reproducible free of any source of nondeterminism, and it's why the
//! desync test below can prove determinism without touching the network.

pub mod lockstep;
pub mod sim;
pub mod transport;

#[cfg(test)]
mod desync_test {
    //! The headless determinism proof (rl#39): replay ONE input log through two
    //! independently-constructed sims and assert their state hashes match
    //! tick-for-tick. If the sim ever acquires a nondeterminism bug (a `HashMap`
    //! walk, a `thread_rng` draw, a wall-clock read), the two diverge and a tick's
    //! hashes disagree — this test goes red. It is the empirical guard the issue
    //! calls for: determinism is testable, so test it.

    use std::collections::BTreeMap;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::net::sim::{Input, PlayerId, Sim};

    /// Generate a deterministic pseudo-random input log: `ticks` ticks, each with one
    /// input per player. Seeded so the log itself is reproducible (the test must be
    /// deterministic too, or a failure couldn't be reproduced).
    fn input_log(seed: u64, players: &[PlayerId], ticks: usize) -> Vec<BTreeMap<PlayerId, Input>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..ticks)
            .map(|_| {
                players
                    .iter()
                    .map(|&p| {
                        let x: f32 = rng.gen_range(-1.0..=1.0);
                        let y: f32 = rng.gen_range(-1.0..=1.0);
                        (p, Input::from_axes(x, y))
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn two_sims_stay_in_lockstep_on_one_input_log() {
        let players: Vec<PlayerId> = (0..4).map(PlayerId).collect();
        let seed = 0xA11CE;
        let log = input_log(0xBEEF, &players, 240); // ~4s at 60Hz

        // Two sims built separately from the same seed — like two peers booting the
        // same match. They must agree from tick 0.
        let mut a = Sim::new(seed, &players);
        let mut b = Sim::new(seed, &players);
        assert_eq!(a.state_hash(), b.state_hash(), "initial state must match");

        for (t, inputs) in log.iter().enumerate() {
            a.step(inputs);
            b.step(inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "state hash diverged at tick {t} — sim is nondeterministic"
            );
        }
        // And both advanced exactly the log length.
        assert_eq!(a.tick(), log.len() as u64);
    }

    #[test]
    fn replaying_the_same_log_reproduces_the_same_final_hash() {
        // Determinism across SEPARATE runs (not just two sims in one process): the
        // final hash is a pure function of (seed, players, input log).
        let players: Vec<PlayerId> = (0..3).map(PlayerId).collect();
        let log = input_log(0x1234, &players, 120);

        let run = || {
            let mut s = Sim::new(77, &players);
            for inputs in &log {
                s.step(inputs);
            }
            s.state_hash()
        };
        assert_eq!(run(), run(), "same inputs must yield the same final state hash");
    }
}
