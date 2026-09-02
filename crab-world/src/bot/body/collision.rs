use bevy_rapier3d::prelude::*;

pub const ARENA_COLLISION: CollisionGroups =
    CollisionGroups::new(Group::GROUP_1, Group::ALL.difference(Group::GROUP_1));

pub const MAX_ENVS: usize = 16;

const CRAB_GROUP: Group = Group::GROUP_2;
const VEHICLE_GROUP: Group = Group::GROUP_3;

/// Every crab part, one group: solid to the arena and to vehicles, never to a crab —
/// not another crab, and not its own links. A same-crab link–link contact is a
/// constraint row inside ONE multibody whose two sides can barely separate along
/// the contact normal (the relative Jacobian nearly cancels, rl#332), so the PGS
/// pass moves both links together instead: the 4–9 m/s one-tick kicks on the feet.
pub const CRAB_COLLISION: CollisionGroups =
    CollisionGroups::new(CRAB_GROUP, Group::GROUP_1.union(VEHICLE_GROUP));

pub const VEHICLE_COLLISION: CollisionGroups =
    CollisionGroups::new(VEHICLE_GROUP, Group::GROUP_1.union(CRAB_GROUP));

// Rapier activates a pair only when EACH side's filter names the other's membership —
// one direction alone is silent non-contact (rl#235).
const _: () = assert!(
    VEHICLE_COLLISION
        .filters
        .intersects(CRAB_COLLISION.memberships)
        && CRAB_COLLISION
            .filters
            .intersects(VEHICLE_COLLISION.memberships)
);

pub(super) fn no_adjacent_contacts(joint: impl Into<TypedJoint>) -> TypedJoint {
    let mut joint = joint.into();
    let generic: &mut GenericJoint = joint.as_mut();
    generic.set_contacts_enabled(false);
    joint
}
