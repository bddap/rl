//! One shared digest of the crab's full rapier physics state — the per-tick number the
//! GCR lockstep desync check folds in so a float divergence between peers is caught on the
//! tick it happens (rl#82).
//!
//! For a reduced-coordinate multibody, the link poses + velocities fully determine the joint
//! DOFs, so hashing every actuated body's `(pos, quat, linvel, angvel)` as raw IEEE-754 bits
//! captures the complete dynamic state. Equality is an exact integer compare (no epsilon):
//! the determinism contract is bit-identity, not nearness. ONE definition, used by both the
//! production bridge ([`crate::net::external_crab`]) and the two-sim determinism regression
//! ([`super::determinism_probe`]) so the hashed layout can't drift between them.

use bevy::prelude::Transform;
use bevy_rapier3d::prelude::Velocity;

use super::body::{CrabCarapace, CrabJoint};

/// The body digest's start value, so several body sources seed one rolling digest identically.
/// Shares the FNV offset-basis *value* — but [`fold_bodies`] is word-wise, not byte-wise FNV
/// (see there), so this is its own constant aliased onto the basis, not an [`crate::fnv::Fnv`].
pub const DIGEST_SEED: u64 = crate::fnv::OFFSET_BASIS;

/// Per-body field count: pos(3) + quat(4) + linvel(3) + angvel(3).
pub const BODY_FIELDS: usize = 13;

/// The stable SEMANTIC key for a crab body part, identical across two independently-built
/// worlds (bevy `Entity` ids are NOT — bevy allocates internal entities in its own order, so
/// they can't pair bodies): `0` = carapace, `1 + joint.index()` = an actuated joint link.
/// `None` for fixed (non-actuated) parts — the eye-stalks — which rigidly follow the carapace
/// and so are redundant for the determinism check (and have no stable key to pair on).
pub fn body_key(carapace: bool, joint: Option<&CrabJoint>) -> Option<usize> {
    match (carapace, joint) {
        (true, _) => Some(0),
        (false, Some(j)) => Some(1 + j.id.index()),
        (false, None) => None,
    }
}

/// f32 → bits, CANONICALIZED so two peers that compute an equal-valued state hash to equal
/// bits: every NaN (any payload) collapses to one quiet-NaN pattern, and ±0.0 (equal values
/// with different bit patterns) collapse to +0.0. Without this a blow-up tick — hashed BEFORE
/// the deferred non-finite rescue runs (rescue is `.before(Sense)`, one tick late) — could
/// surface a spurious mismatch on NaN payload or zero sign even between honest peers. Matches
/// the canonicalization the determinism measurement harness used (determinism-report.md).
fn canon_bits(x: f32) -> u32 {
    if x.is_nan() {
        0x7fc0_0000 // one canonical quiet NaN
    } else if x == 0.0 {
        0 // collapse -0.0 and +0.0
    } else {
        x.to_bits()
    }
}

/// One body's 13 dynamic-state words as canonicalized f32 bits (see [`canon_bits`]), in the
/// field order pos, quat, linvel, angvel.
pub fn body_bits(transform: &Transform, vel: &Velocity) -> [u32; BODY_FIELDS] {
    [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
        vel.linear.x,
        vel.linear.y,
        vel.linear.z,
        vel.angular.x,
        vel.angular.y,
        vel.angular.z,
    ]
    .map(canon_bits)
}

/// Fold a sorted-by-key slice of `(key, bits)` bodies into a rolling digest. The caller MUST
/// sort by key first so two worlds whose ECS iteration order differs still produce the same
/// digest (the key order is the cross-peer-stable order). Returns the updated digest so several
/// body sets can be chained into one number.
///
/// This is an FNV-*style* word-wise fold (XOR each whole 32-bit word, then multiply), NOT
/// byte-wise FNV-1a: it borrows [`crate::fnv::PRIME`] as the multiplier from one source, but
/// folds at u32 granularity, so it is deliberately NOT routed through [`crate::fnv::Fnv`]
/// (whose `write` is byte-wise and would yield a different digest). Don't "unify" it onto
/// `Fnv` — that silently changes the value and desyncs peers.
pub fn fold_bodies(mut h: u64, sorted_bodies: &[(usize, [u32; BODY_FIELDS])]) -> u64 {
    for (_, bits) in sorted_bodies {
        for &w in bits {
            h ^= w as u64;
            h = h.wrapping_mul(crate::fnv::PRIME);
        }
    }
    h
}

/// Whole-state digest of one crab from its bodies: collect `(key, bits)`, sort by the
/// semantic key, and fold from [`DIGEST_SEED`]. `bodies` is every `CrabBodyPart`'s
/// `(transform, vel, joint, carapace)`; fixed parts (no key) are dropped. The single
/// production entry point — the bridge feeds it env 0's parts each tick.
pub fn crab_state_digest<'a>(
    bodies: impl Iterator<
        Item = (
            &'a Transform,
            &'a Velocity,
            Option<&'a CrabJoint>,
            Option<&'a CrabCarapace>,
        ),
    >,
) -> u64 {
    let mut v: Vec<(usize, [u32; BODY_FIELDS])> = bodies
        .filter_map(|(t, vel, joint, carapace)| {
            body_key(carapace.is_some(), joint).map(|key| (key, body_bits(t, vel)))
        })
        .collect();
    v.sort_by_key(|(k, _)| *k);
    fold_bodies(DIGEST_SEED, &v)
}
