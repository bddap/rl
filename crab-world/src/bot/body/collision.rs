//! Collision groups and membership: which crab parts (and the player vehicle) may
//! contact the arena, each other, and each per-env crab — plus the joint-adjacent
//! contact filter. Membership is what keeps each env a physically independent RL
//! problem on one shared arena.

use bevy_rapier3d::prelude::*;

/// Bit 0 (`GROUP_1`): the arena (ground + walls). Every crab — whichever env —
/// must contact it, so its filter is "every group except my own" ([`Group::ALL`]
/// minus arena): the arena collides with all per-env crab bits and the nested-link
/// bit without having to enumerate them. Collision is an AND of both directions, so
/// the arena naming a group is what lets a part on that group touch the ground at all.
pub const ARENA_COLLISION: CollisionGroups =
    CollisionGroups::new(Group::GROUP_1, Group::ALL.difference(Group::GROUP_1));

/// Bit reserved for carapace-NESTED links: links whose collider center falls inside
/// the carapace box — on this model the actuated coxa/claw-shoulder (and the
/// eye-stalks) that ride just under the shell. Membership is purely geometric (see
/// `spawn_crab`), so it catches actuated joints, not only locked links. They keep
/// their mass but collide only with the arena — never with the carapace or each
/// other. A link jammed inside the carapace collider just fights the solver every
/// tick (the near-massless pincers ring it as rest jitter, bddap/rl#20), and
/// `no_adjacent_contacts` can't filter it because its joint parent is another nested
/// link, not the carapace. One shared bit across all envs is fine: nested links only
/// ever touch the arena, never another crab's parts.
pub const NESTED_COLLISION: CollisionGroups = CollisionGroups::new(Group::GROUP_2, Group::GROUP_1);

/// Highest env bit a crab may occupy (`GROUP_3`..`GROUP_18` ⇒ envs 0..15), which is
/// why `--envs` is capped at 16: env `e` takes bit `1 << (e + 2)` and we must stay
/// inside `Group`'s 32 bits with room for the arena (bit 0) and nested (bit 1) bits.
pub const MAX_ENVS: usize = 16;

/// Bit reserved for the player's single-player VEHICLE rigidbody (the rapier plane/ship,
/// [`crate::vehicle`]). One bit above the env range so the vehicle collides with the arena AND
/// with every env's crab parts — the headline (owner 703): it bounces off Sally and shoves her
/// legs by mass. At TRAINING time no vehicle entity exists, so a crab filter naming this bit (see
/// [`crab_collision`]) matches nothing and the trained physics stays bit-identical — the vehicle
/// is policy-safe by construction. Only ever ONE vehicle, in a solo round.
const VEHICLE_GROUP: Group = Group::GROUP_19; // bit 18 = MAX_ENVS + 2

// Envs occupy bits 2..=MAX_ENVS+1 of `Group`'s 32; the vehicle takes the next bit
// (`VEHICLE_GROUP`, bit MAX_ENVS+2). Compile-time guarantees that both fit and that
// `VEHICLE_GROUP` really is that next bit — so raising MAX_ENVS past the budget, or letting the
// vehicle bit collide with an env bit, fails the build instead of silently truncating at runtime.
const _: () = assert!(MAX_ENVS + 2 < 32);
const _: () = assert!(VEHICLE_GROUP.bits() == 1 << (MAX_ENVS + 2));

/// Collision groups for the player's vehicle: its own [`VEHICLE_GROUP`] bit, filtered to hit the
/// arena (so the walls bounce it — owner Option A) and every env's crab parts (so it strikes
/// Sally). Reciprocity needs the crab filter to name `VEHICLE_GROUP` too — [`crab_collision`]
/// adds it. Excludes the nested bit (`GROUP_2`): those links hide inside the shell and only ever
/// touch the arena, so a vehicle contact there would just fight the solver.
pub fn vehicle_collision() -> CollisionGroups {
    // Every env's membership bit: bits 2..MAX_ENVS+2 — the same `1 << (e + 2)` `crab_collision`
    // hands each env, unioned so the one vehicle hits whichever crab is present.
    let mut env_bits = Group::empty();
    let mut e = 0;
    while e < MAX_ENVS {
        env_bits = env_bits.union(Group::from_bits_truncate(1 << (e + 2)));
        e += 1;
    }
    CollisionGroups::new(VEHICLE_GROUP, Group::GROUP_1.union(env_bits))
}

/// Collision membership for env `e`'s ordinary (non-nested, distal) crab parts.
///
/// **Each env gets its OWN bit**, so a crab's distal limbs collide with the arena
/// and with that SAME crab's other distal limbs — preserving self-collision (without
/// it the policy "tucks" legs through one another: free interpenetration is an
/// exploit, not a stance) — but NOT with any other env's crab, which keeps each env a
/// physically independent RL problem even as the M crabs walk across one shared arena
/// toward far targets and would otherwise plow into each other. Joint-adjacent
/// segments are separately contact-filtered (`no_adjacent_contacts`).
///
/// `e` must be `< MAX_ENVS`; the `--envs` clap range (1..=16) guarantees it.
pub fn crab_collision(env: usize) -> CollisionGroups {
    debug_assert!(
        env < MAX_ENVS,
        "env {env} exceeds the {MAX_ENVS}-env bit budget"
    );
    // Env 0 → GROUP_3 (bit 2); arena=bit 0, nested=bit 1 are reserved below it.
    let bit = Group::from_bits_truncate(1 << (env + 2));
    // Filter: the arena, this crab's own distal parts, AND the player's vehicle ([`VEHICLE_GROUP`])
    // so a vehicle strike is a real reciprocal contact. The vehicle bit only exists in a solo round
    // with a spawned vehicle; training never spawns one, so naming it here changes no trained
    // physics (no collider carries `VEHICLE_GROUP` to match) — the migration stays policy-safe.
    CollisionGroups::new(bit, Group::GROUP_1.union(bit).union(VEHICLE_GROUP))
}

/// Disable contacts between the two segments a joint connects. The joint
/// already constrains that pair, and their colliders overlap at the anchor by
/// construction — contacts there would only fight the articulation. All other
/// (non-adjacent) parts of the SAME crab DO collide; see [`crab_collision`].
pub(super) fn no_adjacent_contacts(joint: impl Into<TypedJoint>) -> TypedJoint {
    let mut joint = joint.into();
    let generic: &mut GenericJoint = joint.as_mut();
    generic.set_contacts_enabled(false);
    joint
}
