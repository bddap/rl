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
