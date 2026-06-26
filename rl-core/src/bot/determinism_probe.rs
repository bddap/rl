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
// `solo_crab::sync_external_crab` does in production) — fed through a REAL 2-peer
// [`Lockstep`] pair that cross-checks state hashes. So a synced pair stays bit-identical
// every tick, and a weights mismatch trips [`Fault::Desync`] at the tick it happens. The
// crab body is stepped EXACTLY ONCE per lockstep tick here (one `app.update()` : one
// `try_advance`), which is the cadence the windowed driver fold must preserve — this is the
// headless proof of that coupling, independent of the (GPU) windowed client.

use crate::net::lockstep::{Fault, Lockstep};
use crate::net::sim::{Input, PlayerId, Pos, UNIT};

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
    // The digest over every actuated body — identical to `solo_crab::hash_crab_physics`.
    let digest = crate::bot::physics_digest::crab_state_digest(
        q.iter(app.world()).map(|(t, v, j, c)| (t, v, j, c)),
    ) ^ weights_digest;
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

/// Drive two independently-built crab worlds + two networked [`Lockstep`]s in tandem, with
/// the external crab ARMED on both, stepping the crab exactly once per lockstep tick and
/// cross-checking hashes each tick. Returns `(any_desync, hashes_match_at_end)`. `(da, db)`
/// are the two peers' weights digests folded into their per-tick physics digest.
fn run_two_peer_armed(da: u64, db: u64, ticks: usize) -> (bool, bool) {
    let players = [PlayerId(0), PlayerId(1)];
    let mut a = Lockstep::new(GCR_SEED, &players, PlayerId(0));
    let mut b = Lockstep::new(GCR_SEED, &players, PlayerId(1));
    a.enable_external_crab(true);
    b.enable_external_crab(true);

    let mut app_a = headless_app();
    let mut app_b = headless_app();
    // One tick so `spawn_initial_crabs` builds the body + sizes CrabActions on each peer.
    tick(&mut app_a, 1);
    tick(&mut app_b, 1);

    let seq = scripted_actions(ticks);
    let is_desync = |f: &Fault| matches!(f, Fault::Desync { .. });
    let mut saw_desync = false;
    for act in &seq {
        // Same scripted torque on both peers → bit-identical physics (the probe's invariant).
        set_actions(&mut app_a, act);
        set_actions(&mut app_b, act);
        app_a.update();
        app_b.update();

        let (pose_a, dig_a) = bridge_pose_and_digest(&mut app_a, da);
        let (pose_b, dig_b) = bridge_pose_and_digest(&mut app_b, db);

        // Each peer submits its own input, exchanges the tick message, pushes its crab pose +
        // digest BEFORE advancing (the bridge contract), then advances one tick.
        let ma = a.submit_local_input(Input::default());
        let mb = b.submit_local_input(Input::default());
        a.set_external_crab_pose(pose_a, 0, dig_a);
        b.set_external_crab_pose(pose_b, 0, dig_b);
        saw_desync |= a
            .record_remote(PlayerId(1), mb)
            .as_ref()
            .is_some_and(is_desync);
        saw_desync |= b
            .record_remote(PlayerId(0), ma)
            .as_ref()
            .is_some_and(is_desync);
        saw_desync |= a.try_advance().iter().any(is_desync);
        saw_desync |= b.try_advance().iter().any(is_desync);
    }
    (saw_desync, a.sim().state_hash() == b.sim().state_hash())
}

#[test]
fn networked_nn_crab_synced_stays_bit_identical_every_tick() {
    // Two peers running the SAME brain (equal non-zero weights digest): the real articulated
    // crab is stepped once per lockstep tick, its pose + digest folded into the desync hash —
    // and the pair stays bit-identical every tick, no desync. This is the headless proof that
    // an ARMED networked NN crab is deterministic when weights are synced (the cadence fold's
    // core invariant), independent of the GPU windowed client.
    const BRAIN: u64 = 0xC0FFEE_1234_5678;
    let (saw_desync, matched) = run_two_peer_armed(BRAIN, BRAIN, 200);
    assert!(
        !saw_desync,
        "synced weights + identical physics must never desync across the 2-peer round"
    );
    assert!(
        matched,
        "synced peers must hold identical state hashes through the whole round"
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
    let (saw_desync, matched) = run_two_peer_armed(BRAIN_A, BRAIN_B, 60);
    assert!(
        saw_desync,
        "mismatched weights digests must trip Fault::Desync at the divergent tick"
    );
    assert!(
        !matched,
        "mismatched-brain peers must NOT end on equal state hashes"
    );
}
