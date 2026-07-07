pub mod articulation;
pub mod cadence;
pub mod controls;
pub mod cordic;
pub mod formation;
pub mod lockstep;
pub mod membership;
pub mod net_loop;
pub mod roster;
pub mod server;
pub mod sim;
pub mod snapshot;
pub mod telemetry;
pub mod transport;

#[cfg(feature = "render")]
pub mod external_crab;
#[cfg(feature = "render")]
pub mod menu;
#[cfg(feature = "render")]
pub mod render;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncVerdict {
    pub assets: bool,
    pub crabs: bool,
}

pub fn may_arm_external_crab(sync: Option<SyncVerdict>) -> bool {
    sync.is_none_or(|v| v.assets && v.crabs)
}

#[cfg(test)]
mod desync_test {

    use std::collections::BTreeMap;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::SyncVerdict;
    use crate::sim::{Input, Outcome, PlayerId, Sim, buttons};

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
            a.step(inputs);
            b.step(inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "state hash diverged at tick {t} — sim is nondeterministic"
            );
        }
        assert_eq!(a.tick(), log.len() as u64);
    }

    #[test]
    fn long_replay_drives_the_real_loop_and_stays_in_lockstep() {
        let players: Vec<PlayerId> = (0..2).map(PlayerId).collect();
        let neutral: BTreeMap<PlayerId, Input> =
            players.iter().map(|&p| (p, Input::default())).collect();
        let mut a = Sim::new(0x5EED, &players);
        let mut b = Sim::new(0x5EED, &players);
        let mut resolved_at = None;
        for t in 0..1500u64 {
            super::sim::drive_crab_toward_prey(&mut a);
            super::sim::drive_crab_toward_prey(&mut b);
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
            restarts += u32::from(a.step(inputs));
            b.step(inputs);
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

    fn would_arm_external_crab(sync: Option<SyncVerdict>, checkpoint: Option<()>) -> bool {
        checkpoint.is_some() && super::may_arm_external_crab(sync)
    }

    fn synced(assets: bool) -> Option<SyncVerdict> {
        Some(SyncVerdict {
            assets,
            crabs: true,
        })
    }

    #[test]
    fn arm_gate_keys_on_solo_or_synced_assets() {
        assert!(
            !would_arm_external_crab(synced(false), Some(())),
            "a networked round with mismatched crab ASSETS must NOT arm the NN crab (round refused)"
        );

        assert!(
            would_arm_external_crab(synced(true), Some(())),
            "a networked round with SYNCED assets must arm the NN crab"
        );

        assert!(would_arm_external_crab(None, Some(())));

        assert!(!would_arm_external_crab(None, None));
        assert!(!would_arm_external_crab(synced(true), None));
    }

    #[test]
    fn may_arm_external_crab_rules() {
        assert!(
            super::may_arm_external_crab(None),
            "solo (no formation ran, no verdict) always arms"
        );
        assert!(
            super::may_arm_external_crab(synced(true)),
            "networked + synced assets → may arm (the GCR path)"
        );
        assert!(
            !super::may_arm_external_crab(synced(false)),
            "networked + UNSYNCED crab assets → must NOT arm (different colliders render \
             different Sallys)"
        );
    }
}
