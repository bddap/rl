//! Multiplayer netcode foundation (rl#39): deterministic lockstep simulation over
//! iroh LAN transport. Multiplayer-first, so the whole game (rl#38) is built on top
//! of these three pieces:
//!
//! - [`sim`] — the deterministic simulation core: the Phase 1 gray-box Extraction
//!   loop (first-person players + one giant crab + an extraction point). The contract
//!   (pure step, complete state hash, no nondeterminism) is what every later game
//!   system must honor; the render/vehicle subs build on the interface documented at
//!   the top of [`sim`], they do not bypass it.
//! - [`lockstep`] — the fixed-tick driver: exchange inputs (not state) with an
//!   input-delay buffer, apply in a fixed peer order, advance, and compare state
//!   hashes to catch desync. Transport-agnostic.
//! - [`transport`] — iroh mDNS LAN discovery + per-tick message exchange over QUIC.
//!
//! Two client-side layers build ON those, consuming the sim read-only and producing
//! [`sim::Input`] — they add NO nondeterminism (the sim contract at the top of
//! [`sim`] spells out why this is the firewall it is):
//! - [`net_loop`] — a synchronous bridge from the async [`transport`] to a game main
//!   loop (pump broadcast/inbox without `.await`), plus the discover-and-assign
//!   cold-start shared with the headless driver.
//! - [`render`] — the windowed first-person Bevy client (rl#38): FP camera at the
//!   local player, the gray-box scene (players, the giant crab, the extraction
//!   point), WASD+mouse+gamepad → [`sim::Input`], tick interpolation, and a headless
//!   screenshot mode for evidence.
//!
//! The determinism-critical code ([`sim`] + [`lockstep`]) is pure and sync; all the
//! async/IO lives in [`transport`]/[`net_loop`] and all the rendering in [`render`].
//! That separation is deliberate: it keeps the part that MUST be reproducible free of
//! any source of nondeterminism, and it's why the desync test below can prove
//! determinism without touching the network or a GPU.

pub mod lockstep;
pub mod membership;
pub mod net_loop;
pub mod render;
pub mod sim;
pub mod transport;

#[cfg(test)]
mod desync_test {
    //! The headless determinism proof (rl#39): replay ONE input log through two
    //! independently-constructed sims and assert their state hashes match
    //! tick-for-tick. If the sim ever acquires a nondeterminism bug (a `HashMap`
    //! walk, a `thread_rng` draw, a wall-clock read, a raw `f32::sin`), the two
    //! diverge and a tick's hashes disagree — this test goes red. It is the empirical
    //! guard the issue calls for: determinism is testable, so test it.
    //!
    //! Phase 1 (rl#38) extends it past the old dot world: the log now exercises the
    //! FULL [`Input`] surface (move + yaw-look + the action bit), and a long replay
    //! drives the real gray-box — players turning and moving by facing, the giant crab
    //! pursuing and grabbing, the round resolving — so the hash equality proves
    //! determinism of the ACTUAL sim (player yaw, crab position, statuses, outcome),
    //! not a trivial placeholder.
    //!
    //! The vehicle first cut (rl#38) adds [`two_pilot_sims_stay_in_lockstep`]: it flies
    //! a plane through a multi-hundred-tick throttle/turn/climb/dive sequence on two
    //! independent sims and asserts their hashes agree every tick — the plane's full
    //! integer state (3D pos/velocity, heading, pitch) folding into the same hash, so a
    //! float creeping into the flight math would diverge the peers and fail here.

    use std::collections::BTreeMap;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::net::sim::{Input, Outcome, PlayerId, Sim, buttons};

    /// Generate a deterministic pseudo-random input log: `ticks` ticks, each with one
    /// input per player, spanning every input field — move axes, yaw-look delta, and
    /// the action button — so the replay drives turning, facing-relative movement,
    /// and extraction attempts, not just the two move axes. Seeded so the log itself
    /// is reproducible (the test must be deterministic too, or a failure couldn't be
    /// reproduced).
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
                        // Press action ~1/8 of ticks so extraction logic is exercised
                        // without the button being held constantly.
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
    fn two_pilot_sims_stay_in_lockstep() {
        // The vehicle determinism proof (rl#38 first cut): a plane flown through a
        // deliberate throttle / turn / climb / dive program on two independently-built
        // sims must hash identically EVERY tick. The plane's whole evolving state (3D
        // position + velocity, heading, pitch) is in the hash, so any nondeterminism in
        // the flight integrator — a stray f32, an unordered map — diverges the two and
        // fails this. A scripted program (not random) so the phases are legible and the
        // motion is non-trivial: it climbs, turns, then dives, exercising both angles
        // and all three velocity axes.
        let pilots: Vec<PlayerId> = (0..2).map(PlayerId).collect();
        let seed = 0xF1A11;
        // Build the input program: 600 ticks (~20s at 30Hz) cycling through flight
        // phases, the SAME input fed to every pilot each tick.
        let program: Vec<BTreeMap<PlayerId, Input>> = (0..600usize)
            .map(|t| {
                // throttle (forward), pitch (strafe: + climbs), yaw-look (turn).
                let (forward, strafe, look) = match t % 240 {
                    0..60 => (1.0, 0.6, 0.0),    // climb under full throttle
                    60..120 => (1.0, 0.0, 1.0),  // level off, hard turn
                    120..180 => (0.7, -0.7, 0.0), // nose down, dive
                    _ => (1.0, 0.2, -0.5),       // pull up, opposite turn
                };
                pilots
                    .iter()
                    .map(|&p| (p, Input::new(strafe, forward, look, 0)))
                    .collect()
            })
            .collect();

        let mut a = Sim::new_with_pilots(seed, &pilots, &pilots);
        let mut b = Sim::new_with_pilots(seed, &pilots, &pilots);
        assert_eq!(a.state_hash(), b.state_hash(), "initial pilot state must match");

        let start = a.plane(PlayerId(0)).expect("player 0 should be a pilot").pos();
        for (t, inputs) in program.iter().enumerate() {
            a.step(inputs);
            b.step(inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "plane state hash diverged at tick {t} — flight sim is nondeterministic"
            );
        }
        // Non-trivial proof: the plane actually flew somewhere (not a frozen no-op that
        // would "stay in lockstep" vacuously). It climbed (Y up from spawn) and moved
        // horizontally (turned + thrust), so pos changed on multiple axes.
        let end = a.plane(PlayerId(0)).unwrap().pos();
        assert_ne!(end, start, "the plane must have moved over 600 ticks");
        assert!(
            end.y != start.y && (end.x != start.x || end.z != start.z),
            "flight must change altitude AND ground position, got {start:?} -> {end:?}"
        );
    }

    #[test]
    fn long_replay_drives_the_real_loop_and_stays_in_lockstep() {
        // Two players that hold still while the crab hunts them: a DETERMINISTIC
        // scenario (neutral input every tick) that is guaranteed to resolve — the crab
        // closes in and wipes the round. Two sims must agree EVERY tick, including
        // across the outcome transition and the frozen-after-decided ticks, so
        // determinism is proven for the crab pursuit, the grab, and the freeze — not
        // just free movement. The guaranteed `Wiped` keeps the test honest (a no-op sim
        // would also "stay in lockstep") WITHOUT leaning on random inputs happening to
        // resolve, so it can't flake.
        let players: Vec<PlayerId> = (0..2).map(PlayerId).collect();
        let neutral: BTreeMap<PlayerId, Input> = BTreeMap::new();
        let mut a = Sim::new(0x5EED, &players);
        let mut b = Sim::new(0x5EED, &players);
        let mut resolved_at = None;
        for t in 0..1500u64 {
            a.step(&neutral);
            b.step(&neutral);
            assert_eq!(a.state_hash(), b.state_hash(), "diverged at tick {t}");
            if resolved_at.is_none() && a.outcome() != Outcome::Ongoing {
                resolved_at = Some(t);
            }
        }
        assert_eq!(
            a.outcome(),
            Outcome::Wiped,
            "still players must be wiped by the crab"
        );
        assert!(
            resolved_at.is_some(),
            "round must resolve mid-replay, then stay frozen"
        );
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
        assert_eq!(
            run(),
            run(),
            "same inputs must yield the same final state hash"
        );
    }
}
