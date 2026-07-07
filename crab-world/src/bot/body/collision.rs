use bevy_rapier3d::prelude::*;

pub const ARENA_COLLISION: CollisionGroups =
    CollisionGroups::new(Group::GROUP_1, Group::ALL.difference(Group::GROUP_1));

pub const NESTED_COLLISION: CollisionGroups = CollisionGroups::new(Group::GROUP_2, Group::GROUP_1);

pub const MAX_ENVS: usize = 16;

const VEHICLE_GROUP: Group = Group::GROUP_19;

const _: () = assert!(MAX_ENVS + 2 < 32);
const _: () = assert!(VEHICLE_GROUP.bits() == 1 << (MAX_ENVS + 2));

pub fn vehicle_collision() -> CollisionGroups {
    let mut env_bits = Group::empty();
    let mut e = 0;
    while e < MAX_ENVS {
        env_bits = env_bits.union(Group::from_bits_truncate(1 << (e + 2)));
        e += 1;
    }
    CollisionGroups::new(VEHICLE_GROUP, Group::GROUP_1.union(env_bits))
}

pub fn crab_collision(env: usize) -> CollisionGroups {
    debug_assert!(
        env < MAX_ENVS,
        "env {env} exceeds the {MAX_ENVS}-env bit budget"
    );
    let bit = Group::from_bits_truncate(1 << (env + 2));
    CollisionGroups::new(bit, Group::GROUP_1.union(bit).union(VEHICLE_GROUP))
}

pub(super) fn no_adjacent_contacts(joint: impl Into<TypedJoint>) -> TypedJoint {
    let mut joint = joint.into();
    let generic: &mut GenericJoint = joint.as_mut();
    generic.set_contacts_enabled(false);
    joint
}
