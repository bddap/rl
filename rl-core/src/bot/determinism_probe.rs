//! Two-sim cross-determinism regression for the articulated NN-crab physics — the
//! standing guard for GCR avenue 1 (rl#82): putting the real trained crab into multiplayer
//! requires rapier's reduced-coordinate multibody solver to be bit-identical across two
//! independent same-binary sims.
//!
//! Builds TWO independently-constructed headless rapier worlds, drives both with ONE fixed
//! scripted torque sequence (no policy, to isolate physics), and after every tick hashes
//! each world's full dynamic state via the SHARED [`super::physics_digest`] layout (the
//! same bytes the production bridge folds into the lockstep desync hash). The two states
//! must be bit-for-bit equal every tick; the first mismatch reports the tick, body,
//! quantity, and magnitude.
//!
//! Run (avoid trainer contention — see CLAUDE.md), exercising the SHIPPED config:
//!   cargo test -p rl-core --features enhanced-determinism determinism_probe \
//!       -- --nocapture --test-threads=2
//! Tick count via `PROBE_TICKS` (default 1000). Without the feature it is the informative
//! contrast (does plain float already match same-binary, or is the feature what closes the
//! gap?) — the gonogo found same-binary already agrees, so both pass here.

use bevy::prelude::*;
use bevy_rapier3d::prelude::Velocity;

use super::actuator::{ACTION_SIZE, CrabActions};
use super::body::{CrabBodyPart, CrabCarapace, CrabJoint};
use super::physics_digest::{BODY_FIELDS, DIGEST_SEED, body_bits, body_key, fold_bodies};
use super::test_util::{headless_app, tick};

/// One rigid body's full dynamic state as raw IEEE-754 bits, keyed by the shared semantic
/// key (see [`body_key`]) so it pairs across two independently-built worlds.
#[derive(Clone, PartialEq, Eq)]
struct BodyState {
    key: usize,
    bits: [u32; BODY_FIELDS],
}

const FIELD: [&str; BODY_FIELDS] = [
    "pos.x", "pos.y", "pos.z", "rot.x", "rot.y", "rot.z", "rot.w", "linvel.x", "linvel.y",
    "linvel.z", "angvel.x", "angvel.y", "angvel.z",
];

fn snapshot(app: &mut App) -> Vec<BodyState> {
    let mut q = app.world_mut().query_filtered::<(
        &Transform,
        &Velocity,
        Option<&CrabJoint>,
        Option<&CrabCarapace>,
    ), With<CrabBodyPart>>();
    let mut v: Vec<BodyState> = q
        .iter(app.world())
        .filter_map(|(t, vel, joint, carapace)| {
            body_key(carapace.is_some(), joint).map(|key| BodyState {
                key,
                bits: body_bits(t, vel),
            })
        })
        .collect();
    v.sort_by_key(|b| b.key);
    v
}

/// A deterministic torque schedule, identical for both worlds. A per-tick LCG (computed
/// once, in this binary, so byte-identical for both sims by construction) drives every joint
/// over its full signed range, so the legs flail and load the ground contacts — exercising
/// the contact solver + the transcendental-heavy joint math each tick.
fn scripted_actions(n: usize) -> Vec<[f32; ACTION_SIZE]> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut seq = Vec::with_capacity(n);
    for _ in 0..n {
        let mut a = [0.0f32; ACTION_SIZE];
        for slot in a.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 40) as u32; // 24 bits — exactly representable in f32
            *slot = (u as f32 / ((1u32 << 24) as f32)) * 1.8 - 0.9; // [-0.9, 0.9]
        }
        seq.push(a);
    }
    seq
}

fn set_actions(app: &mut App, a: &[f32; ACTION_SIZE]) {
    app.world_mut().resource_mut::<CrabActions>().envs[0] = *a;
}

fn body_name(key: usize) -> String {
    if key == 0 {
        "carapace".to_string()
    } else {
        format!("joint[{}]", key - 1)
    }
}

/// First (key, field) where the two snapshots differ, plus the float magnitude of the gap.
/// `field == usize::MAX` flags a structural mismatch (different body count or pairing).
fn first_divergence(a: &[BodyState], b: &[BodyState]) -> Option<(usize, usize, f32)> {
    if a.len() != b.len() {
        return Some((usize::MAX, usize::MAX, f32::NAN));
    }
    for (x, y) in a.iter().zip(b) {
        if x.key != y.key {
            return Some((x.key, usize::MAX, f32::NAN));
        }
        for (i, (&p, &q)) in x.bits.iter().zip(y.bits.iter()).enumerate() {
            if p != q {
                let d = (f32::from_bits(p) - f32::from_bits(q)).abs();
                return Some((x.key, i, d));
            }
        }
    }
    None
}

/// The run's rolling digest over every tick's full state, via the SHARED fold — so a passing
/// digest here is the exact contract the bridge pushes into the lockstep hash.
fn digest(prev: u64, s: &[BodyState]) -> u64 {
    let bodies: Vec<(usize, [u32; BODY_FIELDS])> = s.iter().map(|b| (b.key, b.bits)).collect();
    fold_bodies(prev, &bodies)
}

#[test]
fn two_sims_bit_identical_under_scripted_torque() {
    let n: usize = std::env::var("PROBE_TICKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let seq = scripted_actions(n);

    // Two independently-built worlds with identical construction.
    let mut a = headless_app();
    let mut b = headless_app();
    // One tick so spawn_initial_crabs has built the crab and sized CrabActions.
    tick(&mut a, 1);
    tick(&mut b, 1);

    let (sa0, sb0) = (snapshot(&mut a), snapshot(&mut b));
    assert!(!sa0.is_empty(), "no crab bodies spawned — harness wired wrong");
    if let Some((key, field, mag)) = first_divergence(&sa0, &sb0) {
        let fname = FIELD.get(field).copied().unwrap_or("<structural>");
        panic!(
            "INITIAL state diverged before any step: body {} ({} of {}), {fname}, |Δ|={mag:e} \
             — spawn/init is not deterministic (rapier determinism init clause)",
            body_name(key),
            sa0.len(),
            sb0.len()
        );
    }

    let mut digest_a = DIGEST_SEED;
    let mut digest_b = DIGEST_SEED;
    for (t, act) in seq.iter().enumerate() {
        set_actions(&mut a, act);
        set_actions(&mut b, act);
        a.update();
        b.update();
        let sa = snapshot(&mut a);
        let sb = snapshot(&mut b);
        if let Some((key, field, mag)) = first_divergence(&sa, &sb) {
            let fname = FIELD.get(field).copied().unwrap_or("<structural>");
            panic!(
                "DIVERGED at tick {t} (of {n}): body {}, quantity {fname}, |Δ|={mag:e} \
                 — articulated physics is NOT cross-sim bit-identical",
                body_name(key)
            );
        }
        digest_a = digest(digest_a, &sa);
        digest_b = digest(digest_b, &sb);
    }

    assert_eq!(
        digest_a, digest_b,
        "in-process digests disagree (should be impossible after per-tick check)"
    );
    eprintln!(
        "PROBE PASS: {n} ticks, two independent sims bit-identical every tick \
         ({} rigid bodies hashed/tick)",
        sa0.len()
    );
    // CROSS-PROCESS marker: stable across separate invocations iff the run is
    // process-independent (run this binary twice, diff the digest → Deck↔Deck proxy).
    eprintln!("PROBE DIGEST: {digest_a:#018x} (ticks={n})");
}

// ---------------------------------------------------------------------------
// GCR: the NN crab folded INTO the networked lockstep tick (rl#82, job #62)
// ---------------------------------------------------------------------------
//
// The cross-determinism probe above proves the articulated crab physics is bit-identical
// across two same-binary sims. These tests close the loop the GCR fold actually needs: the
// per-tick bridge handshake — `crab_state_digest(bodies) ^ weights_digest` pushed into the
// lockstep via `set_external_crab_pose` BEFORE each `try_advance` (exactly what
// `external_crab::sync_external_crab` does in production) — fed through a REAL 2-peer
// [`Lockstep`] pair that cross-checks state hashes. So a synced pair stays bit-identical
// every tick, and a weights mismatch trips [`Fault::Desync`] at the tick it happens. The
// crab body is stepped at the REAL production cadence — [`PhysicsCadence`] doles out
// [`crate::physics::PHYSICS_HZ`] (64 Hz) physics steps across [`crate::net::sim::TICK_HZ`]
// (30 Hz) lockstep ticks (mostly 2, periodically 3), one pose pushed per APPLIED tick via
// [`Lockstep::advance_one`] — the exact coupling the windowed driver fold must preserve. This
// is the headless proof of that coupling, independent of the (GPU) windowed client.

use crate::net::cadence::PhysicsCadence;
use crate::net::lockstep::{Fault, INPUT_DELAY, Lockstep};
use crate::net::sim::{Input, PlayerId, Pos, TICK_HZ, UNIT, buttons};
use crate::physics::PHYSICS_HZ;

/// The match seed for the 2-peer GCR determinism tests. Arbitrary but fixed (reproducible).
const GCR_SEED: u64 = 0x6C20_5EED;

/// Read env 0's crab from `app` and produce what the production bridge pushes into the sim:
/// the carapace's game-world ground pose ([`Pos`]) and the full articulated physics digest
/// XORed with `weights_digest` — the SAME `crab_state_digest` fold the lockstep desync hash
/// folds in. One physics step must already have run (`app.update()`), so the bodies exist.
fn bridge_pose_and_digest(app: &mut App, weights_digest: u64) -> (Pos, u64) {
    let mut q = app.world_mut().query_filtered::<(
        &Transform,
        &Velocity,
        Option<&CrabJoint>,
        Option<&CrabCarapace>,
    ), With<CrabBodyPart>>();
    // The digest over every actuated body — identical to `external_crab::hash_crab_physics`.
    let digest = crate::bot::physics_digest::crab_state_digest(q.iter(app.world())) ^ weights_digest;
    // The carapace ground position → the sim's fixed-point pose. A SIMPLIFIED model of the
    // bridge's `world_pos` (no world-gain accumulation) — fine here because both peers apply
    // the identical float→i64 cast to identical inputs (same binary), so the pose path is
    // approximate-but-symmetric; the DIGEST path below is the exact production fold, and it's
    // what carries the determinism teeth.
    let mut cq = app
        .world_mut()
        .query_filtered::<&Transform, With<CrabCarapace>>();
    let c = cq
        .iter(app.world())
        .next()
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);
    let pose = Pos {
        x: (c.x * UNIT as f32) as i64,
        z: (c.z * UNIT as f32) as i64,
    };
    (pose, digest)
}

/// Outcome of a two-peer armed run (named fields over positional bools, so the asserts read).
struct ArmedRun {
    /// Any [`Fault::Desync`] surfaced on either peer across the whole run.
    saw_desync: bool,
    /// The two peers' `state_hash` agree at the END of the run.
    matched_end: bool,
    /// The two peers' `state_hash` agreed at EVERY applied tick (through any restart) — the
    /// stronger "never diverged for a moment" check the end-only compare can't make.
    matched_every_tick: bool,
    /// A drain tick actually rewound the sim (`tick() < before`) — proves a requested RESTART
    /// fired and wasn't a silent no-op.
    saw_restart: bool,
}

/// Drive two independently-built crab worlds + two networked [`Lockstep`]s in tandem, with
/// the external crab ARMED on both, at the REAL production cadence: each peer issues one
/// input per outer iteration, then drains every now-ready apply tick, stepping the crab body
/// [`PhysicsCadence::steps_for_next_tick`] physics steps and pushing ONE pose+digest per
/// APPLIED tick via [`Lockstep::advance_one`] (the windowed driver's contract). `(da, db)`
/// are the two peers' weights digests folded into their per-tick physics digest; `iterations`
/// is the number of input-issue rounds. `restart_tick`, if set, ORs [`buttons::RESTART`] into
/// that ONE iteration's input on BOTH peers (an edge-clean press→release, since every other
/// iteration is neutral) so the armed round restarts in lockstep — exercising the sim's
/// restart edge with the crab digest folded in.
fn run_two_peer_armed(
    da: u64,
    db: u64,
    iterations: usize,
    restart_tick: Option<usize>,
) -> ArmedRun {
    let players = [PlayerId(0), PlayerId(1)];
    let mut a = Lockstep::new(GCR_SEED, &players, PlayerId(0));
    let mut b = Lockstep::new(GCR_SEED, &players, PlayerId(1));

    let mut app_a = headless_app();
    let mut app_b = headless_app();
    // One tick so `spawn_initial_crabs` builds the body + sizes CrabActions on each peer.
    tick(&mut app_a, 1);
    tick(&mut app_b, 1);

    // Real production cadence: the body advances PHYSICS_HZ steps per TICK_HZ lockstep ticks
    // via the integer accumulator, NOT one step per tick. Both peers start from Default, so
    // they step the body the same number of times every applied tick — the determinism the
    // cadence guarantees, exercised here at the real 64:30 ratio.
    let mut cad_a = PhysicsCadence::default();
    let mut cad_b = PhysicsCadence::default();
    // Scripted torque per physics step. Worst case is the ceil ratio every applied tick, plus
    // the INPUT_DELAY warmup ticks the first drain also applies.
    let per_tick_ceil = (PHYSICS_HZ as usize).div_ceil(TICK_HZ as usize);
    let seq = scripted_actions((iterations + INPUT_DELAY as usize + 1) * per_tick_ceil);
    let mut step_idx = 0usize;

    let is_desync = |f: &Fault| matches!(f, Fault::Desync { .. });
    let mut saw_desync = false;
    let mut saw_restart = false;
    let mut matched_every_tick = true;
    for i in 0..iterations {
        // Each peer issues its own input for the next scheduled tick and exchanges it. On the
        // designated iteration both peers press RESTART (identical bits, same scheduled tick),
        // so the round restarts in lockstep; every other tick is neutral, making it a clean
        // press→release edge.
        let btns = if Some(i) == restart_tick {
            buttons::RESTART
        } else {
            0
        };
        let input = Input::new(0.0, 0.0, 0.0, btns);
        let ma = a.submit_local_input(input);
        let mb = b.submit_local_input(input);
        saw_desync |= a
            .record_remote(PlayerId(1), mb)
            .as_ref()
            .is_some_and(is_desync);
        saw_desync |= b
            .record_remote(PlayerId(0), ma)
            .as_ref()
            .is_some_and(is_desync);

        // Drain every now-ready apply tick, one at a time, stepping the crab's physics by the
        // deterministic cadence and pushing ONE pose+digest per APPLIED tick. Both peers are
        // symmetric, so they advance in lockstep — drive both off `a`'s readiness.
        while a.next_tick_ready() {
            debug_assert!(b.next_tick_ready(), "symmetric peers must advance in lockstep");
            let na = cad_a.steps_for_next_tick();
            let nb = cad_b.steps_for_next_tick();
            assert_eq!(na, nb, "the cadence must be identical across peers");
            for _ in 0..na {
                // Identical scripted torque per physics step on both peers → bit-identical
                // physics (the probe's standing invariant), so any divergence is the weights.
                let act = &seq[step_idx];
                step_idx += 1;
                set_actions(&mut app_a, act);
                set_actions(&mut app_b, act);
                app_a.update();
                app_b.update();
            }
            let (pose_a, dig_a) = bridge_pose_and_digest(&mut app_a, da);
            let (pose_b, dig_b) = bridge_pose_and_digest(&mut app_b, db);
            a.set_external_crab_pose(pose_a, 0, dig_a);
            b.set_external_crab_pose(pose_b, 0, dig_b);
            let before = a.sim().tick();
            saw_desync |= a.advance_one().expect("ready").iter().any(is_desync);
            saw_desync |= b.advance_one().expect("ready").iter().any(is_desync);
            // A RESTART rewinds the sim to tick 0 inside advance_one — record it so the test can
            // prove the press wasn't a no-op.
            saw_restart |= a.sim().tick() < before;
            // Compare per tick, not just at the end: a transient divergence at the restart tick
            // that re-converged would slip past an end-only check.
            matched_every_tick &= a.sim().state_hash() == b.sim().state_hash();
        }
    }
    ArmedRun {
        saw_desync,
        matched_end: a.sim().state_hash() == b.sim().state_hash(),
        matched_every_tick,
        saw_restart,
    }
}

#[test]
fn networked_nn_crab_synced_stays_bit_identical_every_tick() {
    // Two peers running the SAME brain (equal non-zero weights digest): the real articulated
    // crab is stepped at the production 64:30 cadence, its pose + digest folded into the desync
    // hash per applied tick — and the pair stays bit-identical every tick, no desync. This is
    // the headless proof that an ARMED networked NN crab is deterministic when weights are
    // synced (the cadence fold's core invariant), independent of the GPU windowed client.
    const BRAIN: u64 = 0x00C0_FFEE_1234_5678;
    let run = run_two_peer_armed(BRAIN, BRAIN, 200, None);
    assert!(
        !run.saw_desync,
        "synced weights + identical physics must never desync across the 2-peer round"
    );
    assert!(
        run.matched_end,
        "synced peers must hold identical state hashes through the whole round"
    );
}

#[test]
fn networked_nn_crab_restart_stays_in_lockstep() {
    // The armed-crab + RESTART regression (rl#101): mid-round, both peers press RESTART on the
    // same scheduled tick with the crab digest folded into the lockstep hash. The sim must
    // rewind to tick 0 on BOTH peers at the SAME applied tick and stay bit-identical through and
    // after the restart. This guards the restart edge that `drive_lockstep` also hangs the crab
    // bridge re-seed off (rl#101 part a): that re-seed is deterministic precisely because this
    // edge is — it fires identically on both peers from the shared input stream, never a
    // local-only signal. `saw_restart` makes the check non-vacuous: a press that silently no-op'd
    // would pass the hash asserts while testing nothing.
    const BRAIN: u64 = 0x00C0_FFEE_1234_5678;
    let run = run_two_peer_armed(BRAIN, BRAIN, 120, Some(60));
    assert!(
        run.saw_restart,
        "the RESTART press must actually rewind the sim (non-vacuous: a no-op would hide a bug)"
    );
    assert!(
        !run.saw_desync,
        "an armed-crab round that restarts in lockstep must never desync"
    );
    assert!(
        run.matched_every_tick,
        "peers must hold identical state hashes at EVERY tick through and after the restart"
    );
}

#[test]
fn networked_nn_crab_weights_mismatch_trips_desync() {
    // The downstream half of the shared-checkpoint guard with TEETH on the real body: two
    // peers with IDENTICAL physics but DIFFERENT weights digests must desync — the digest is
    // folded into the per-tick lockstep hash, so a peer that loaded the wrong brain can't
    // masquerade as in-sync. This is what makes arming gated on `weights_synced` load-bearing:
    // even if the upstream handshake were bypassed, the divergence is caught LOUDLY at the tick.
    const BRAIN_A: u64 = 0xAAAA_AAAA_AAAA_AAAA;
    const BRAIN_B: u64 = 0xBBBB_BBBB_BBBB_BBBB;
    let run = run_two_peer_armed(BRAIN_A, BRAIN_B, 60, None);
    assert!(
        run.saw_desync,
        "mismatched weights digests must trip Fault::Desync at the divergent tick"
    );
    assert!(
        !run.matched_end,
        "mismatched-brain peers must NOT end on equal state hashes"
    );
}
