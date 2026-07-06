
use bevy::prelude::Transform;
use bevy_rapier3d::prelude::Velocity;

use super::body::{CrabCarapace, CrabJoint};

pub(crate) const DIGEST_SEED: u64 = crate::fnv::OFFSET_BASIS;

pub(crate) const BODY_FIELDS: usize = 13;

pub fn body_key(carapace: bool, joint: Option<&CrabJoint>) -> Option<usize> {
    match (carapace, joint) {
        (true, _) => Some(0),
        (false, Some(j)) => Some(1 + j.id.index()),
        (false, None) => None,
    }
}

fn canon_bits(x: f32) -> u32 {
    if x.is_nan() {
        0x7fc0_0000
    } else if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}

pub(crate) fn body_bits(transform: &Transform, vel: &Velocity) -> [u32; BODY_FIELDS] {
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

pub(crate) fn fold_bodies(mut h: u64, mut bodies: Vec<(usize, [u32; BODY_FIELDS])>) -> u64 {
    bodies.sort_by_key(|(k, _)| *k);
    for (_, bits) in &bodies {
        for &w in bits {
            h ^= w as u64;
            h = h.wrapping_mul(crate::fnv::PRIME);
        }
    }
    h
}

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
    let v: Vec<(usize, [u32; BODY_FIELDS])> = bodies
        .filter_map(|(t, vel, joint, carapace)| {
            body_key(carapace.is_some(), joint).map(|key| (key, body_bits(t, vel)))
        })
        .collect();
    fold_bodies(DIGEST_SEED, v)
}
