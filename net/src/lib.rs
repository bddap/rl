// rl#282: sibling sim suites have wedged (all threads futex_wait, 0% CPU) under
// trainer saturation; abort loudly on a process-wide CPU flatline instead.
#[cfg(test)]
test_watchdog::arm!();

pub mod articulation;
pub mod cadence;
pub mod client;
pub mod controls;
pub mod cordic;
pub mod formation;
pub mod membership;
pub mod net_loop;
pub mod roster;
pub mod server;
pub mod sim;
pub mod snapshot;
pub mod telemetry;
pub mod transport;
pub mod wire;

// Gated only because the bridge it drives is: the seam itself is render-free
// (crab-world's headless bot stack compiles without render) — lift the gate when the
// trainer's headless server world needs it (rl#298 stage 4) or the bridge dies (stage 5).
#[cfg(feature = "render")]
pub(crate) mod crab_slot;
#[cfg(feature = "render")]
pub mod external_crab;
#[cfg(feature = "render")]
pub mod menu;
#[cfg(feature = "render")]
pub mod render;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncVerdict {
    pub body: bool,
    pub crabs: bool,
    /// Every peer runs the same effective plant — arena, terrain bake, friction caps
    /// ([`crab_world::bot::body::constructed_plant_digest`], rl#286). Without it two
    /// peers arm DISAGREEING WORLDS: a flat-arena client adopts terrain-height poses,
    /// so Sally and every craft float or bury by the tile's relief on that screen.
    pub plant: bool,
}

pub fn may_arm_external_crab(sync: Option<SyncVerdict>) -> bool {
    // Destructure so a new verdict axis is a compile error here, not a silently
    // un-gated bool.
    sync.is_none_or(|v| {
        let SyncVerdict { body, crabs, plant } = v;
        body && crabs && plant
    })
}

/// What a peer advertises about its local world during formation/admission — the
/// digests the [`SyncVerdict`] is judged from. One value threaded from the launch
/// site to [`membership::Membership`], so a new identity axis rides the existing
/// plumbing instead of growing every signature by another scalar.
/// Fields are `pub(crate)` so [`SyncStamp::local`] is the only cross-crate
/// constructor BY CONSTRUCTION — a launcher cannot hand-roll a dishonest stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncStamp {
    /// [`crab_world::mesh_fallback::constructed_body_digest`] — 0 = no usable model.
    pub(crate) body_digest: u64,
    /// [`crab_world::bot::body::constructed_plant_digest`] — never legitimately 0.
    pub(crate) plant_digest: u64,
    /// NN crabs this peer would render; 0 = crabless viewer, host count rules.
    pub(crate) crab_count: u8,
}

impl SyncStamp {
    /// No-model, no-plant, crabless: [`membership::Membership`]'s pre-`with_stamp`
    /// start, and tests exercising formation mechanics rather than the sync verdict.
    pub(crate) const ZERO: SyncStamp = SyncStamp {
        body_digest: 0,
        plant_digest: 0,
        crab_count: 0,
    };

    /// The honest local stamp. Call AFTER any checkpoint plant is adopted
    /// (`adopt_recorded_plant`) — a pre-adopt call latches the wrong plant, which
    /// adoption then refuses loudly. Launch paths that load policies do so before
    /// forming a match; checkpoint-less ones (headless `game net`) advertise the
    /// env-default plant and are refused by checkpoint-bearing peers unless the
    /// plants genuinely agree — that refusal is the rl#286 guard, not a bug.
    pub fn local(crab_count: u8) -> Self {
        Self {
            body_digest: crab_world::mesh_fallback::constructed_body_digest(),
            plant_digest: crab_world::bot::body::constructed_plant_digest(),
            crab_count,
        }
    }
}

/// Serializes the `#[ignore]`d real-endpoint tests: every live iroh endpoint on the box
/// mDNS-discovers and dials every other, so two lobby tests running at once merge into
/// one oversized roster. A lock they all take beats a `--test-threads=1` flag someone
/// must remember to pass.
#[cfg(test)]
pub(crate) fn real_net_serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod desync_test {

    use std::collections::BTreeMap;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::SyncVerdict;
    use crate::sim::{Externals, Input, Outcome, PlayerId, Sim, buttons};

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
            a.step(inputs, Externals::NONE);
            b.step(inputs, Externals::NONE);
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
            let claws_a = super::sim::drive_crab_toward_prey(&mut a);
            let claws_b = super::sim::drive_crab_toward_prey(&mut b);
            a.step(&neutral, Externals::claws_only(&claws_a));
            b.step(&neutral, Externals::claws_only(&claws_b));
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
                s.step(inputs, Externals::NONE);
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
            restarts += u32::from(a.step(inputs, Externals::NONE));
            b.step(inputs, Externals::NONE);
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

    fn synced(body: bool) -> Option<SyncVerdict> {
        Some(SyncVerdict {
            body,
            crabs: true,
            plant: true,
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

        assert!(
            !would_arm_external_crab(
                Some(SyncVerdict {
                    body: true,
                    crabs: true,
                    plant: false,
                }),
                Some(())
            ),
            "a networked round with mismatched PLANTS (arena/bake/friction) must NOT arm — \
             the peers would render disagreeing worlds (rl#286)"
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
