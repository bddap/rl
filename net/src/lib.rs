//! Multiplayer netcode: a host-AUTHORITATIVE client/server model over
//! iroh LAN transport — inputs go UP, full game state comes DOWN
//! ([[mp-minecraft-model]]). Multiplayer-first, so the whole game is built on
//! these pieces:
//!
//! - [`sim`] — the deterministic simulation core: the gray-box Extraction
//!   loop (first-person players + one giant crab + an extraction point). The contract
//!   (pure step, complete state hash, no nondeterminism) is what every other game
//!   system must honor; the render/vehicle subs build on the interface documented at
//!   the top of [`sim`], they do not bypass it.
//! - [`server`] — the ONE authoritative peer: steps its own [`sim`] at the HOST's pace
//!   (remote inputs stream in per player and are consumed as they arrive — a remote can
//!   delay nothing) and emits a [`snapshot::CoreSnapshot`] per tick.
//! - [`lockstep`] — the client session (the name predates host-authority): files the
//!   local input UP, ADOPTS the server's snapshot DOWN (never stepping its own sim),
//!   and predicts/reconciles the local avatar.
//! - [`transport`] — iroh mDNS LAN discovery + framed message exchange over QUIC.
//!
//! Two client-side layers build ON those, consuming the sim read-only and producing
//! [`sim::Input`]:
//! - [`net_loop`] — a synchronous bridge from the async [`transport`] to a game main
//!   loop (the [`net_loop::Coordinator`]: solo and host are the SAME server arm,
//!   [[sp-is-mp-special-case]]), plus formation and the mid-game join path.
//! - [`render`] — the windowed first-person Bevy client: FP camera at the
//!   local player, the gray-box scene (players, the giant crab, the extraction
//!   point), WASD+mouse+gamepad → [`sim::Input`], tick interpolation, and a headless
//!   screenshot mode for evidence.
//!
//! [`controls`] is the data-driven binding table both [`render`]'s input handling AND its
//! on-screen legend derive from (one source, can't drift). Its pure core is NOT
//! gated on `render` so it unit-tests in the headless build like [`sim`]; only the
//! Bevy-input glue is render-only.
//!
//! The determinism-critical code ([`sim`] + [`server`]'s step) is pure and sync; all
//! the async/IO lives in [`transport`]/[`net_loop`] and all the rendering in
//! [`render`]. That separation keeps the part that must be reproducible free of any
//! source of nondeterminism, so the tests below can pin sim behavior without touching
//! the network or a GPU.

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

// Rendering-only client layers — gated out of the headless trainer build (they pull
// bevy's renderer + egui: the FP window, the boot menu, the solo NN-crab's render
// transforms). The headless netcode (sim/lockstep/transport) and the trainer don't
// need them.
#[cfg(feature = "render")]
pub mod external_crab;
#[cfg(feature = "render")]
pub mod menu;
#[cfg(feature = "render")]
pub mod render;

/// The formation barrier's verdict on the one input that must agree across peers for the float
/// NN crabs — computed ONCE by [`crate::membership::Membership::sync_verdict`] at the barrier's
/// close and carried as this single value through `Frozen` → `NetDriver` to the arm sites:
/// - `assets`: every peer advertised the SAME non-zero crab-model-asset digest. The
///   giant crabs' rapier colliders are derived from the crab MODEL asset
///   ([`crab_world::mesh_fallback::constructed_body_digest`]), so peers with different crab models
///   diverge client-side the moment a crab is built — this check is peer-symmetric.
///
/// There is deliberately NO brain/weights half any more (rl#200 increment 6 deleted the
/// all-peers weights-digest gate): under host-authority only the HOST executes brains — clients
/// adopt snapshots and render articulation, so a weights advertisement guards nothing — and the
/// host's own bindings are validated fail-loud at launch (every checkpoint must fit the rig or
/// the game refuses to start), which is strictly stronger than a digest handshake. Per-crab
/// brain identity still folds into [`crate::sim::Sim::state_hash`] via the bridge's physics
/// digests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncVerdict {
    pub assets: bool,
}

/// The shared-asset guard for handing the crabs to the float NN bodies:
/// a round may arm the external crabs only when it can't diverge peers on the
/// collider asset every peer builds its rendered crabs from. `None` means the round is
/// SOLO (no networked formation ran) — one peer, nothing to disagree, always arms. A NETWORKED
/// round arms ONLY on a synced [`SyncVerdict`].
///
/// On failure the round CANNOT arm the NN crabs and — with no integer fallback (rl#114) —
/// the production sites REFUSE it loudly rather than substituting a fake crab.
///
/// This is the SINGLE arm predicate — the [`render`] arming sites (the `Boot::Round` build, the
/// menu's `poll_formation` pre-gate, and `ensure_round_installed`) reach it through the one
/// `arm_round` gate (whose `ArmedRound` proof the install path consumes), and the tests
/// below call it directly, so the rule can't drift
/// between them. Each caller ANDs it with "a checkpoint/NN stack is present" (no brain ⇒ nothing to
/// arm). Deliberately NOT behind `cfg(render)`: the no-feature test build (like the headless
/// trainer) must exercise the REAL predicate, not a re-encoded copy.
pub fn may_arm_external_crab(sync: Option<SyncVerdict>) -> bool {
    sync.is_none_or(|v| v.assets)
}

#[cfg(test)]
mod desync_test {
    //! The headless determinism proof: replay ONE input log through two
    //! independently-constructed sims and assert their state hashes match
    //! tick-for-tick. If the sim ever acquires a nondeterminism bug (a `HashMap`
    //! walk, a `thread_rng` draw, a wall-clock read, a raw `f32::sin`), the two
    //! diverge and a tick's hashes disagree — this test goes red. Determinism is
    //! testable, so test it.
    //!
    //! The log exercises the FULL [`Input`] surface (move + yaw-look + the action
    //! bit), and a long replay drives the real gray-box — players turning and moving
    //! by facing, the giant crab pursuing and grabbing, the round resolving — so the
    //! hash equality proves determinism of the ACTUAL sim (player yaw, crab position,
    //! statuses, outcome), not a trivial placeholder.

    use std::collections::BTreeMap;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::SyncVerdict;
    use crate::sim::{Input, Outcome, PlayerId, Pos, Sim, buttons};

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
        let log = input_log(0xBEEF, &players, 240); // 8s at TICK_HZ=30

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
    fn long_replay_drives_the_real_loop_and_stays_in_lockstep() {
        // Two players that hold still while the crab hunts them: a DETERMINISTIC
        // scenario (neutral input every tick) that is guaranteed to resolve — the crab
        // (driven by the shared deterministic test driver; the sim has no integer
        // pursuit of its own) closes in and wipes the round. Two sims must agree EVERY tick, including
        // across the outcome transition and the frozen-after-decided ticks, so
        // determinism is proven for the crab drive, the grab, and the freeze — not
        // just free movement. The guaranteed `Wiped` keeps the test honest (a no-op sim
        // would also "stay in lockstep") WITHOUT leaning on random inputs happening to
        // resolve, so it can't flake.
        let players: Vec<PlayerId> = (0..2).map(PlayerId).collect();
        // Complete neutral map — `Sim::step` requires one input per participant,
        // never an empty map silently defaulted to all-neutral.
        let neutral: BTreeMap<PlayerId, Input> =
            players.iter().map(|&p| (p, Input::default())).collect();
        let mut a = Sim::new(0x5EED, &players);
        let mut b = Sim::new(0x5EED, &players);
        let mut resolved_at = None;
        for t in 0..1500u64 {
            // Drive both crabs identically (same integer math) so the wipe still happens and the
            // two sims stay byte-identical — the property under test.
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

    #[test]
    fn restart_edge_in_a_log_keeps_two_sims_in_lockstep() {
        // The RESTART bit is the one button (besides ACTION) that crosses into the sim,
        // and a periodic press is the only thing the gamepad's Start adds to the input
        // stream that the other desync logs don't already cover. Drive two independent
        // sims with a moving log that presses RESTART every ~50 ticks and assert they
        // hash-match EVERY tick, including across each mid-replay round rebuild — so
        // the edge-triggered restart (sim.rs `restart_held` latch) is proven to fire on
        // the SAME tick on every peer. (Determinism of the analog AXES is covered by
        // `two_sims_stay_in_lockstep_on_one_input_log`, which feeds full-range f32 axes
        // through `Input::new`; this test adds the restart edge, not a third axis path.)
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
                        // A clean periodic edge (held one tick), so the sim's edge-latch
                        // sees press→release and restarts once per press, not every tick.
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
        // Non-vacuous: every periodic press fired (t = 7, 57, ... 257), proving the RESTART
        // bit actually drove the sim — not a no-op that would "stay in lockstep" trivially.
        // The tick counter stays monotone through all of them (a restart is a state-reset at
        // the current tick, never a rewind — rl#204).
        assert_eq!(restarts, 6, "each periodic RESTART press fired once");
        assert_eq!(
            a.tick(),
            log.len() as u64,
            "restarts never rewind the tick counter"
        );
    }

    // -----------------------------------------------------------------------------
    // External NN-crab arm gate
    // -----------------------------------------------------------------------------
    //
    // The external NN crab (`net::external_crab`, render-only) drives a FLOAT rapier crab and writes
    // its pose into the integer `Sim` via `set_external_crab_pose`. A float crab desyncs peers
    // UNLESS they share the same brain and step it identically, so it may arm only when
    // [`super::may_arm_external_crab`] allows: a SOLO round always, a NETWORKED round only with
    // SYNCED weights+assets. There is NO integer fallback: a networked-UNSYNCED round
    // CANNOT arm and the production sites REFUSE it LOUDLY — GRACEFULLY: the scripted
    // `Boot::Round` build returns an `Err` (clean CLI exit) and the menu's `poll_formation` returns
    // to the chooser showing an actionable peer-mismatch message — rather than substituting a fake
    // crab or crashing. These tests pin that gate predicate.
    //
    // FAITHFULNESS / LIMITATION: `external_crab`/`render` are `#[cfg(feature = "render")]` (they pull
    // bevy's full GPU stack), so this suite — which builds with NO features, like the headless
    // trainer — cannot reference `ExternalCrabPlugin`/`ExternalCrabArmed`/`build_windowed_app`
    // directly, nor stand up a real iroh transport. These re-encode the SAME `may_arm_external_crab`
    // predicate the production sites use. The armed crab against the actual physics is exercised by
    // the headless NN-crab probes (`game nn-crab-probe` / `nn-crab-vehicle-stability`).

    /// Models the production arm decision exactly: a checkpoint must be present AND
    /// [`super::may_arm_external_crab`] must allow it (the SAME predicate the `Boot::Round` build,
    /// the menu's `poll_formation` gate, and `ensure_round_installed` all call via
    /// `arm_round`, so this can't drift from them). `checkpoint` is an `Option<()>`
    /// stand-in — only its `is_some()` feeds the gate; `sync` is the formation handshake's
    /// verdict (`None` = solo round). Returns whether the round WOULD arm the NN crab; on a
    /// networked round `false` means the production sites REFUSE the round (no integer
    /// fallback), not a silent downgrade.
    fn would_arm_external_crab(sync: Option<SyncVerdict>, checkpoint: Option<()>) -> bool {
        checkpoint.is_some() && super::may_arm_external_crab(sync)
    }

    /// Shorthand for a networked round's verdict in these tests.
    fn synced(assets: bool) -> Option<SyncVerdict> {
        Some(SyncVerdict { assets })
    }

    #[test]
    fn arm_gate_keys_on_solo_or_synced_assets() {
        // The invariant: a networked round arms ONLY with synced crab assets; solo always arms
        // (with a checkpoint); no checkpoint never arms. A networked round that does NOT arm is
        // REFUSED by the production sites — there is no integer crab. (The old all-peers
        // weights half is deliberately GONE — rl#200 increment 6: clients run no inference, and
        // the host's bindings are validated fail-loud at launch.)

        // Networked + UNSYNCED crab assets: must NOT arm — two peers with different crab models
        // build different colliders and render different Sallys.
        assert!(
            !would_arm_external_crab(synced(false), Some(())),
            "a networked round with mismatched crab ASSETS must NOT arm the NN crab (round refused)"
        );

        // Networked + SYNCED assets + checkpoint: DOES arm.
        assert!(
            would_arm_external_crab(synced(true), Some(())),
            "a networked round with SYNCED assets must arm the NN crab"
        );

        // Solo + checkpoint: always arms (one peer, nothing to disagree) — solo is `None`, so
        // there is no verdict to consult (no peer to be synced WITH).
        assert!(would_arm_external_crab(None, Some(())));

        // No checkpoint never arms — neither solo nor networked-synced. (Production rejects a
        // missing checkpoint even earlier, at `nn_crab_checkpoint_dir`.)
        assert!(!would_arm_external_crab(None, None));
        assert!(!would_arm_external_crab(synced(true), None));
    }

    /// The shared-checkpoint guard: two peers running the SAME float crab pose but
    /// DIFFERENT policy weights must desync, because the bridge folds the weights digest into
    /// the per-tick physics hash. Here we push an identical pose to both externally-driven
    /// sims but a different `phys_digest` (standing in for "different weights"), and require the
    /// state hashes to diverge — so a peer that loaded the wrong brain can't masquerade as
    /// in-sync.
    ///
    /// SCOPE: this steps two `Sim`s by HAND with a synthetic digest — it proves the FOLD has teeth
    /// (a digest mismatch surfaces as a `state_hash` divergence) in isolation. The armed crab against
    /// the REAL physics is exercised by the headless NN-crab probes; cross-MACHINE bit-identity is the
    /// 2-Deck gate's job.
    #[test]
    fn external_crab_with_mismatched_weights_desyncs() {
        let seed = 0x5151_8202;
        let players: Vec<PlayerId> = (0..2).map(PlayerId).collect();
        let neutral: BTreeMap<PlayerId, Input> =
            players.iter().map(|&p| (p, Input::default())).collect();
        let pose = Pos { x: 1234, z: -567 };
        // Two distinct weights digests fold into otherwise-identical external crab state.
        let (weights_a, weights_b) = (0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB);

        let mut a = Sim::new(seed, &players);
        let mut b = Sim::new(seed, &players);
        for _ in 0..10u64 {
            a.set_external_crab_pose(0, pose, 7, weights_a);
            b.set_external_crab_pose(0, pose, 7, weights_b);
            a.step(&neutral);
            b.step(&neutral);
        }
        assert_ne!(
            a.state_hash(),
            b.state_hash(),
            "identical pose + DIFFERENT weights digest must desync (shared-checkpoint guard)"
        );

        // Control: the SAME weights digest with the same pose stays in sync — the desync above is the
        // weights mismatch, not a spurious always-diverge.
        let mut c = Sim::new(seed, &players);
        let mut d = Sim::new(seed, &players);
        for _ in 0..10u64 {
            c.set_external_crab_pose(0, pose, 7, weights_a);
            d.set_external_crab_pose(0, pose, 7, weights_a);
            c.step(&neutral);
            d.step(&neutral);
        }
        assert_eq!(
            c.state_hash(),
            d.state_hash(),
            "identical pose + SAME weights digest must stay in sync"
        );
    }

    /// [`super::may_arm_external_crab`]: solo may always arm; networked may arm only with
    /// SYNCED crab assets (the upstream half of the shared-asset guard).
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
