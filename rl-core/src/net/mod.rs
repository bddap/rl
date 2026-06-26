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
//! [`controls`] is the data-driven control map both [`render`]'s input handling AND its
//! on-screen legend derive from (one source, can't drift — rl#57). Its pure core is NOT
//! gated on `render` so it unit-tests in the headless build like [`sim`]; only the
//! Bevy-input glue is render-only.
//!
//! The determinism-critical code ([`sim`] + [`lockstep`]) is pure and sync; all the
//! async/IO lives in [`transport`]/[`net_loop`] and all the rendering in [`render`].
//! That separation is deliberate: it keeps the part that MUST be reproducible free of
//! any source of nondeterminism, and it's why the desync test below can prove
//! determinism without touching the network or a GPU.

pub mod cadence;
pub mod controls;
pub mod lockstep;
pub mod membership;
pub mod net_loop;
pub mod sim;
pub mod telemetry;
pub mod transport;

// Rendering-only client layers — gated out of the headless trainer build (they pull
// bevy's renderer + egui: the FP window, the boot menu, the solo NN-crab's render
// transforms). The headless netcode (sim/lockstep/transport) and the trainer don't
// need them.
#[cfg(feature = "render")]
pub mod menu;
#[cfg(feature = "render")]
pub mod render;
#[cfg(feature = "render")]
pub mod external_crab;

/// The shared-checkpoint guard for handing the crab to the float NN body in LOCKSTEP (rl#82,
/// GCR): a round may arm the external crab only when it can't desync peers on the weights.
/// SOLO rounds always may — one peer, nothing to desync. A NETWORKED round may arm ONLY with
/// genuinely-SYNCED weights (`weights_synced`): a random-init or unloadable brain differs per
/// peer (burn seeds fresh weights from process entropy), so it would desync by construction —
/// it must stay on the deterministic integer crab. This is the UPSTREAM half of the guard: it
/// keeps an *unsynced* brain out of lockstep entirely; the policy-weights digest folded into
/// [`crate::net::sim::Sim::state_hash`] is the downstream half, catching a mismatch between
/// two *synced* brains.
///
/// This is the SINGLE arm predicate — the [`render`] arming sites (`Boot::Round` and
/// `ensure_round_installed`) and the rl#63 tests all call it, so the rule can't drift between
/// them. Each caller ANDs it with "a checkpoint/NN stack is present" (no brain ⇒ nothing to
/// arm). Deliberately NOT behind `cfg(render)`: the no-feature test build (like the headless
/// trainer) must exercise the REAL predicate, not a re-encoded copy.
///
/// CALLER CONTRACT: `weights_synced` must be `Policy::weights_digest() != 0`, **NOT**
/// `is_loaded()` — `RL_RANDOM_POLICY=1` forces `is_loaded()` true on a fresh random brain whose
/// digest is `0` (no checkpoint bytes), which would desync peers silently; a zero digest is
/// exactly "no shared checkpoint", so gating on a non-zero digest closes that trap.
pub fn may_arm_external_crab(net_is_none: bool, weights_synced: bool) -> bool {
    net_is_none || weights_synced
}

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

    use crate::net::lockstep::Lockstep;
    use crate::net::sim::{Input, Outcome, PlayerId, Pos, Sim, buttons};

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
                    0..60 => (1.0, 0.6, 0.0),     // climb under full throttle
                    60..120 => (1.0, 0.0, 1.0),   // level off, hard turn
                    120..180 => (0.7, -0.7, 0.0), // nose down, dive
                    _ => (1.0, 0.2, -0.5),        // pull up, opposite turn
                };
                pilots
                    .iter()
                    .map(|&p| (p, Input::new(strafe, forward, look, 0)))
                    .collect()
            })
            .collect();

        let mut a = Sim::new_with_pilots(seed, &pilots, &pilots);
        let mut b = Sim::new_with_pilots(seed, &pilots, &pilots);
        assert_eq!(
            a.state_hash(),
            b.state_hash(),
            "initial pilot state must match"
        );

        let start = a
            .plane(PlayerId(0))
            .expect("player 0 should be a pilot")
            .pos();
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

    #[test]
    fn restart_edge_in_a_log_keeps_two_sims_in_lockstep() {
        // The RESTART bit is the one button (besides ACTION) that crosses into the sim,
        // and a periodic press is the only thing the gamepad's Start adds to the input
        // stream that the other desync logs don't already cover. Drive two independent
        // sims with a moving log that presses RESTART every ~50 ticks and assert they
        // hash-match EVERY tick, including across each mid-replay rewind to tick 0 — so
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
        for (t, inputs) in log.iter().enumerate() {
            a.step(inputs);
            b.step(inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "sims diverged at tick {t} across a RESTART edge"
            );
        }
        // Non-vacuous: the last rewind (tick 257) leaves the counter well below the log
        // length, proving the RESTART bit actually drove the sim — not a no-op that would
        // "stay in lockstep" trivially.
        assert!(
            a.tick() < log.len() as u64,
            "restart edges must have rewound the sim (tick {} vs {} ticks)",
            a.tick(),
            log.len()
        );
    }

    // -----------------------------------------------------------------------------
    // External NN-crab arm gate (rl#63 + GCR rl#82)
    // -----------------------------------------------------------------------------
    //
    // The external NN crab (`net::external_crab`, render-only) drives a FLOAT rapier crab and writes
    // its pose into the integer `Sim` via `enable_external_crab(true)`/`set_external_crab_pose`.
    // A float crab desyncs peers UNLESS they share the same brain and step it identically, so it
    // may arm only when [`super::may_arm_external_crab`] allows: a SOLO round always, a NETWORKED
    // round only with SYNCED weights. A networked-UNSYNCED round MUST stay integer-driven
    // (`crab_external == false`). Production enforces this by routing every arm site through that
    // one predicate (the `Boot::Round` arm + the menu's `ensure_round_installed`); `drive_lockstep`
    // calls `sync_external_crab` only when the `ExternalCrabArmed` gate is set. These tests would
    // go red if a refactor let `crab_external` flip true on a networked-UNSYNCED round.
    //
    // FAITHFULNESS / LIMITATION: `external_crab`/`render` are `#[cfg(feature = "render")]` (they pull
    // bevy's full GPU stack), so this suite — which builds with NO features, like the headless
    // trainer — cannot reference `ExternalCrabPlugin`/`ExternalCrabArmed`/`build_windowed_app`
    // directly, nor stand up a real iroh transport. These re-encode the SAME `may_arm_external_crab`
    // predicate the production sites use and assert the networked-unsynced branch leaves the crab
    // integer-driven, and prove the only sim-level footprint of an inactive stack there is a
    // bit-for-bit no-op across a real multi-peer round. The armed-networked determinism (the synced
    // case) is proven at the real cadence in `bot::determinism_probe` against the actual physics.

    /// Models the production arm decision exactly: a checkpoint must be present AND
    /// [`super::may_arm_external_crab`] must allow it (the SAME predicate the `Boot::Round` and
    /// `ensure_round_installed` arming sites call, so this can't drift from them). On a hit it
    /// mirrors render's side effect (`enable_external_crab(true)`). `net`/`checkpoint` are
    /// `Option<()>` stand-ins — only their `is_none()`/`is_some()` feeds the gate; `weights_synced`
    /// is the formation handshake's verdict (irrelevant on a solo round). Returns whether the
    /// crab WOULD be externally driven.
    fn arm_external_crab_like_render(
        ls: &mut Lockstep,
        net: Option<()>,
        checkpoint: Option<()>,
        weights_synced: bool,
    ) -> bool {
        let arm = checkpoint.is_some() && super::may_arm_external_crab(net.is_none(), weights_synced);
        if arm {
            ls.enable_external_crab(true);
        }
        arm
    }

    #[test]
    fn arm_gate_keys_on_solo_or_synced_weights() {
        // The invariant: a networked round arms ONLY with synced weights; solo always arms (with
        // a checkpoint); no checkpoint never arms.
        let net2: Vec<PlayerId> = (0..2).map(PlayerId).collect();

        // Networked + UNSYNCED weights + checkpoint present: must STAY integer (a float crab on
        // mismatched brains would desync peers).
        let mut unsynced = Lockstep::new(0x3963, &net2, PlayerId(0));
        assert!(
            !arm_external_crab_like_render(&mut unsynced, Some(()), Some(()), false),
            "a networked round with UNSYNCED weights must NOT arm the external crab"
        );
        assert!(!unsynced.crab_is_external());

        // Networked + SYNCED weights + checkpoint: DOES arm (the GCR fold) — peers share the
        // brain and step it deterministically, so the float crab is safe in lockstep.
        let mut synced = Lockstep::new(0x3963, &net2, PlayerId(0));
        assert!(
            arm_external_crab_like_render(&mut synced, Some(()), Some(()), true),
            "a networked round with SYNCED weights must arm the external crab"
        );
        assert!(synced.crab_is_external());

        // Solo + checkpoint: always arms (one peer, nothing to desync), regardless of the
        // weights-synced flag (no peer to be synced WITH).
        let mut solo = Lockstep::new(0x3963, &[PlayerId(0)], PlayerId(0));
        assert!(arm_external_crab_like_render(&mut solo, None, Some(()), false));
        assert!(solo.crab_is_external());

        // No checkpoint never arms — neither solo nor networked-synced.
        let mut solo_no_ckpt = Lockstep::new(0x3963, &[PlayerId(0)], PlayerId(0));
        assert!(!arm_external_crab_like_render(&mut solo_no_ckpt, None, None, false));
        assert!(!solo_no_ckpt.crab_is_external());
        let mut net_no_ckpt = Lockstep::new(0x3963, &net2, PlayerId(0));
        assert!(!arm_external_crab_like_render(&mut net_no_ckpt, Some(()), None, true));
        assert!(!net_no_ckpt.crab_is_external());
    }

    #[test]
    fn inactive_crab_stack_is_a_noop_on_an_unsynced_networked_round() {
        // On a networked-UNSYNCED round the NN-crab stack may be PRESENT (the menu builds it in
        // before the round resolves) but stays INACTIVE: the gate is off, so `sync_external_crab`
        // is never called and `enable_external_crab(true)` never fires. The only sim-facing
        // footprint is the integer crab driving the round. Prove that footprint is a bit-for-bit
        // no-op: a real 2-peer lockstep round driven identically — once plain, once with the
        // inactive stack's sim-facing API exercised (`enable_external_crab(false)`, the no-op the
        // gate leaves) — must produce the SAME state-hash sequence tick-for-tick. If a refactor
        // made the "inactive" stack perturb the sim, this diverges.
        let players: Vec<PlayerId> = (0..2).map(PlayerId).collect();
        let seed = 0x0A0B;

        // Run a full 2-peer round (modeled on `two_peers_stay_in_lockstep`), optionally calling
        // the inactive stack's sim-facing API, and collect the per-tick hash sequence.
        let run = |touch_inactive_stack: bool| -> Vec<u64> {
            let mut a = Lockstep::new(seed, &players, PlayerId(0));
            let mut b = Lockstep::new(seed, &players, PlayerId(1));
            if touch_inactive_stack {
                // What the menu's NETWORKED branch does with the stack present: it does NOT arm
                // the gate. `enable_external_crab(false)` is the closest standalone call that
                // represents "the inactive stack touched the sim" — it must change nothing.
                a.enable_external_crab(false);
                b.enable_external_crab(false);
            }
            let mut hashes = Vec::new();
            for t in 0..200u64 {
                let ia = Input::from_axes((t % 3) as f32 - 1.0, 0.5);
                let ib = Input::from_axes(0.0, (t % 2) as f32);
                let ma = a.submit_local_input(ia);
                let mb = b.submit_local_input(ib);
                assert!(a.record_remote(PlayerId(1), mb).is_none());
                assert!(b.record_remote(PlayerId(0), ma).is_none());
                assert!(a.try_advance().is_empty());
                assert!(b.try_advance().is_empty());
                assert_eq!(
                    a.sim().state_hash(),
                    b.sim().state_hash(),
                    "peers diverged at tick {t}"
                );
                hashes.push(a.sim().state_hash());
            }
            hashes
        };

        let without_stack = run(false);
        let with_inactive_stack = run(true);
        assert_eq!(
            without_stack, with_inactive_stack,
            "the inactive solo-crab stack must be a bit-for-bit no-op on a networked round"
        );

        // Non-vacuity: the round must actually evolve, so the equality above isn't a trivial
        // match of two frozen sims. The integer crab moves under its pursuit, so the hash
        // changes over the run.
        assert!(
            without_stack.first() != without_stack.last(),
            "the round must change state (else the no-op equality is vacuous)"
        );

        // Teeth: prove the hash is SENSITIVE to the crab pose, so the no-op equality above is
        // meaningful — i.e. IF a refactor wrongly armed the external crab and pushed a pose on a
        // networked round, the hashes WOULD diverge (this is the desync the invariant prevents).
        let mut armed = Lockstep::new(seed, &players, PlayerId(0));
        let mut plain = Lockstep::new(seed, &players, PlayerId(1));
        armed.enable_external_crab(true);
        // Warm past the startup grace so the integer pursuit on `plain` has begun moving the
        // crab, then push a DIFFERENT pose into the externally-driven sim.
        for _ in 0..40u64 {
            let ma = armed.submit_local_input(Input::default());
            let mb = plain.submit_local_input(Input::default());
            let _ = armed.record_remote(PlayerId(1), mb);
            let _ = plain.record_remote(PlayerId(0), ma);
            armed.set_external_crab_pose(Pos { x: 0, z: 0 }, 0, 0);
            let _ = armed.try_advance();
            let _ = plain.try_advance();
        }
        assert_ne!(
            armed.sim().state_hash(),
            plain.sim().state_hash(),
            "an externally-driven crab pose MUST change the hash — else the no-op test has no teeth"
        );
    }

    /// The shared-checkpoint guard (rl#82): two peers running the SAME float crab pose but
    /// DIFFERENT policy weights must desync, because the bridge folds the weights digest into
    /// the per-tick physics hash. Here we push an identical pose to both externally-driven
    /// sims but a different `phys_digest` (standing in for "different weights"), and require the
    /// state hashes to diverge — so a peer that loaded the wrong brain can't masquerade as
    /// in-sync.
    ///
    /// SCOPE: this drives two `Lockstep`s by HAND with a synthetic digest — it proves the FOLD
    /// has teeth (a digest mismatch surfaces as a hash divergence) in isolation. The armed
    /// networked crab against the REAL physics, at the production 64:30 cadence, is proven in
    /// `bot::determinism_probe`; cross-MACHINE bit-identity is the 2-Deck gate's job.
    #[test]
    fn external_crab_with_mismatched_weights_desyncs() {
        let seed = 0x5151_8202;
        let players = [PlayerId(0), PlayerId(1)];
        let mut a = Lockstep::new(seed, &players, PlayerId(0));
        let mut b = Lockstep::new(seed, &players, PlayerId(1));
        a.enable_external_crab(true);
        b.enable_external_crab(true);
        let pose = Pos { x: 1234, z: -567 };
        // Two distinct weights digests fold into otherwise-identical external crab state.
        let (weights_a, weights_b) = (0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB);
        for _ in 0..10u64 {
            let ma = a.submit_local_input(Input::default());
            let mb = b.submit_local_input(Input::default());
            let _ = a.record_remote(PlayerId(1), mb);
            let _ = b.record_remote(PlayerId(0), ma);
            a.set_external_crab_pose(pose, 7, weights_a);
            b.set_external_crab_pose(pose, 7, weights_b);
            let _ = a.try_advance();
            let _ = b.try_advance();
        }
        assert_ne!(
            a.sim().state_hash(),
            b.sim().state_hash(),
            "identical pose + DIFFERENT weights digest must desync (shared-checkpoint guard)"
        );

        // Control: the SAME weights digest with the same pose stays in sync — the desync above
        // is the weights mismatch, not a spurious always-diverge.
        let mut c = Lockstep::new(seed, &players, PlayerId(0));
        let mut d = Lockstep::new(seed, &players, PlayerId(1));
        c.enable_external_crab(true);
        d.enable_external_crab(true);
        for _ in 0..10u64 {
            let mc = c.submit_local_input(Input::default());
            let md = d.submit_local_input(Input::default());
            let _ = c.record_remote(PlayerId(1), md);
            let _ = d.record_remote(PlayerId(0), mc);
            c.set_external_crab_pose(pose, 7, weights_a);
            d.set_external_crab_pose(pose, 7, weights_a);
            let _ = c.try_advance();
            let _ = d.try_advance();
        }
        assert_eq!(
            c.sim().state_hash(),
            d.sim().state_hash(),
            "identical pose + SAME weights digest must stay in sync"
        );
    }

    /// [`super::may_arm_external_crab`]: solo may always arm; networked may arm only with
    /// SYNCED weights (the upstream half of the shared-checkpoint guard).
    #[test]
    fn may_arm_external_crab_rules() {
        assert!(super::may_arm_external_crab(true, true), "solo + synced → arm");
        assert!(
            super::may_arm_external_crab(true, false),
            "solo may arm even unsynced (rest pose, single peer, nothing to desync)"
        );
        assert!(
            super::may_arm_external_crab(false, true),
            "networked + synced weights → may arm (the GCR path)"
        );
        assert!(
            !super::may_arm_external_crab(false, false),
            "networked + UNSYNCED weights → must NOT arm (a random-init brain desyncs peers)"
        );
    }
}
