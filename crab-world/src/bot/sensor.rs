use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::CrabSpawns;
use super::body::{CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, joint_angle};

#[derive(Clone, Copy, Default)]
struct JointObs {
    angle: f32,
    rate: f32,
}

impl JointObs {
    const LEN: usize = 2;
}

#[derive(Clone, Copy)]
struct BodyObs {
    pos: Vec3,
    rot: Quat,
    linvel: Vec3,
    angvel: Vec3,
}

impl Default for BodyObs {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            rot: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
        }
    }
}

impl BodyObs {
    const LEN: usize = 13;
}

const TARGET_LEN: usize = 3;

/// First slot of the spawn-relative body.pos triple (x, y, z) — the channel that is
/// bounded only by the training arena's walls, so GCR's open-field drift guard
/// (rl#240) watches it by name.
pub const BODY_POS_SLOT: usize = CrabJointId::COUNT * JointObs::LEN;

pub(crate) const TARGET_SLOT: usize = BODY_POS_SLOT + BodyObs::LEN;

#[derive(Clone, Copy)]
pub(crate) struct Observation {
    joints: [JointObs; CrabJointId::COUNT],
    body: BodyObs,
    target_local: Vec3,
}

impl Default for Observation {
    fn default() -> Self {
        Self {
            joints: [JointObs::default(); CrabJointId::COUNT],
            body: BodyObs::default(),
            target_local: Vec3::ZERO,
        }
    }
}

pub const OBS_SIZE: usize = TARGET_SLOT + TARGET_LEN;

impl Observation {
    pub(crate) fn serialize(&self) -> [f32; OBS_SIZE] {
        let mut v = [0.0f32; OBS_SIZE];
        let mut i = 0;
        for joint in &self.joints {
            v[i] = joint.angle;
            v[i + 1] = joint.rate;
            i += JointObs::LEN;
        }
        v[i..i + 3].copy_from_slice(&self.body.pos.to_array());
        v[i + 3..i + 7].copy_from_slice(&self.body.rot.to_array());
        v[i + 7..i + 10].copy_from_slice(&self.body.linvel.to_array());
        v[i + 10..i + 13].copy_from_slice(&self.body.angvel.to_array());
        i += BodyObs::LEN;
        v[i..i + TARGET_LEN].copy_from_slice(&self.target_local.to_array());
        v
    }
}

#[derive(Resource, Default)]
pub struct CrabObservation {
    pub envs: Vec<[f32; OBS_SIZE]>,
}

impl CrabObservation {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![[0.0; OBS_SIZE]; n];
    }
}

#[derive(Resource, Default)]
pub struct CrabTargets {
    pub envs: Vec<Option<Vec3>>,
}

impl CrabTargets {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![None; n];
    }

    pub fn get(&self, e: usize) -> Option<Vec3> {
        self.envs.get(e).copied().flatten()
    }
}

pub fn build_observation(
    spawns: Res<CrabSpawns>,
    targets: Res<CrabTargets>,
    mut obs: ResMut<CrabObservation>,
    carapace_q: Query<(Entity, &CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>,
    joint_q: Query<(
        Entity,
        &CrabJoint,
        &CrabEnvId,
        &MultibodyJoint,
        &Transform,
        &Velocity,
    )>,
) {
    let mut built = vec![Observation::default(); obs.envs.len()];

    let mut motion: HashMap<Entity, (Quat, Vec3, Vec3)> = HashMap::new();
    for (e, _, _, _, t, vel) in joint_q.iter() {
        motion.insert(e, (t.rotation, vel.linear, vel.angular));
    }
    for (e, _, t, vel) in carapace_q.iter() {
        motion.insert(e, (t.rotation, vel.linear, vel.angular));
    }

    for (_, crab_joint, env, mj, transform, vel) in joint_q.iter() {
        let Some(o) = built.get_mut(env.0) else {
            continue;
        };
        // Infallible by construction: every joint's parent is the carapace or another
        // joint part, and both queries feeding `motion` run in this same system. A miss
        // is a rig wiring bug — substituting identity would silently corrupt this env's
        // joint angles/rates every tick (rl#242), far worse than a crash.
        let (parent_rot, _parent_lin, parent_ang) = motion
            .get(&mj.parent)
            .copied()
            .expect("crab joint parent missing from motion map — rig wiring bug (rl#242)");

        let joint = &mut o.joints[crab_joint.id.index()];
        joint.angle = joint_angle(crab_joint.axis_local, parent_rot, transform.rotation);
        let axis_world = parent_rot * crab_joint.axis_local;
        joint.rate = (vel.angular - parent_ang).dot(axis_world);
    }

    for (_, env, transform, vel) in carapace_q.iter() {
        let Some(o) = built.get_mut(env.0) else {
            continue;
        };
        // Infallible by construction: `spawn_initial_crabs` rebuilds CrabSpawns and sizes
        // the obs slots from the same n. A miss would feed ABSOLUTE world coords into a
        // channel trained as spawn-relative — instantly OOD, never logged (rl#242).
        let origin = spawns
            .0
            .get(env.0)
            .copied()
            .expect("CrabSpawns has no origin for a live env — spawn wiring bug (rl#242)");
        o.body.pos = transform.translation - origin;
        o.body.rot = transform.rotation;
        o.body.linvel = vel.linear;
        o.body.angvel = vel.angular;

        o.target_local = targets
            .get(env.0)
            .map(|t| transform.rotation.inverse() * (t - transform.translation))
            .unwrap_or(Vec3::ZERO);
    }

    let mut nonfinite = 0usize;
    for (env, o) in built.iter().enumerate() {
        let mut slots = o.serialize();
        for val in slots.iter_mut() {
            debug_assert!(
                val.is_finite(),
                "non-finite observation slot in env {env}: corrupt physics reached Sense"
            );
            if !val.is_finite() {
                *val = 0.0;
                nonfinite += 1;
            }
        }
        obs.envs[env] = slots;
    }
    if nonfinite > 0 {
        error!(
            "build_observation sanitized {nonfinite} non-finite observation slot(s) \
             this tick — corrupt physics upstream of Sense (see rescue_nonfinite_crabs)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_is_slot_identical() {
        let mut o = Observation::default();
        for (idx, j) in o.joints.iter_mut().enumerate() {
            j.angle = idx as f32 + 0.5;
            j.rate = -(idx as f32) - 0.25;
        }
        o.body.pos = Vec3::new(1.0, 2.0, 3.0);
        o.body.rot = Quat::from_xyzw(0.1, 0.2, 0.3, 0.4);
        o.body.linvel = Vec3::new(4.0, 5.0, 6.0);
        o.body.angvel = Vec3::new(7.0, 8.0, 9.0);
        o.target_local = Vec3::new(-1.0, -2.0, -3.0);

        let got = o.serialize();

        let mut want = [0.0f32; OBS_SIZE];
        for idx in 0..CrabJointId::COUNT {
            want[idx * 2] = idx as f32 + 0.5;
            want[idx * 2 + 1] = -(idx as f32) - 0.25;
        }
        let body_base = BODY_POS_SLOT;
        want[body_base] = 1.0;
        want[body_base + 1] = 2.0;
        want[body_base + 2] = 3.0;
        want[body_base + 3] = 0.1;
        want[body_base + 4] = 0.2;
        want[body_base + 5] = 0.3;
        want[body_base + 6] = 0.4;
        want[body_base + 7] = 4.0;
        want[body_base + 8] = 5.0;
        want[body_base + 9] = 6.0;
        want[body_base + 10] = 7.0;
        want[body_base + 11] = 8.0;
        want[body_base + 12] = 9.0;
        want[body_base + 13] = -1.0;
        want[body_base + 14] = -2.0;
        want[body_base + 15] = -3.0;

        assert_eq!(got, want, "serialize() drifted from the pinned obs layout");
    }

    #[test]
    fn obs_size_is_unchanged() {
        assert_eq!(OBS_SIZE, CrabJointId::COUNT * 2 + 13 + 3);
    }

    #[test]
    fn default_serializes_to_zeros() {
        assert_eq!(Observation::default().serialize(), [0.0f32; OBS_SIZE]);
    }

    use bevy::ecs::system::RunSystemOnce;
    use bevy::ecs::world::World;
    use bevy_rapier3d::prelude::{MultibodyJoint, RevoluteJointBuilder, Velocity};

    use super::super::body::Side;

    fn obs_world(spawns: Vec<Vec3>) -> World {
        let mut world = World::new();
        let mut obs = CrabObservation::default();
        obs.resize(1);
        let mut targets = CrabTargets::default();
        targets.resize(1);
        world.insert_resource(obs);
        world.insert_resource(targets);
        world.insert_resource(CrabSpawns(spawns));
        world
    }

    /// rl#242 pin: for valid (fully-wired) inputs the obs builder's output is
    /// bit-identical to the by-hand formulas — the expect()s changed nothing on the
    /// healthy path shared by trainer and game.
    #[test]
    fn valid_inputs_build_bit_identical_obs() {
        let origin = Vec3::new(10.0, 0.0, -20.0);
        let target = Vec3::new(4.0, 0.5, -2.0);
        let mut world = obs_world(vec![origin]);
        world.resource_mut::<CrabTargets>().envs[0] = Some(target);

        let body_rot = Quat::from_axis_angle(Vec3::Y, 0.7);
        let body_t = Transform::from_translation(Vec3::new(11.5, 0.25, -19.0))
            .with_rotation(body_rot);
        let body_vel = Velocity {
            linear: Vec3::new(0.1, -0.2, 0.3),
            angular: Vec3::new(-0.4, 0.5, -0.6),
        };
        let carapace = world
            .spawn((CrabCarapace, CrabEnvId(0), body_t, body_vel))
            .id();

        let axis = Vec3::X;
        let joint_id = CrabJointId::LegCoxa(Side::Left, 0);
        let joint_rot = Quat::from_axis_angle(Vec3::X, 0.3) * body_rot;
        let joint_t = Transform::from_translation(Vec3::new(11.6, 0.2, -19.1))
            .with_rotation(joint_rot);
        let joint_vel = Velocity {
            linear: Vec3::new(0.7, 0.8, 0.9),
            angular: Vec3::new(1.0, -1.1, 1.2),
        };
        world.spawn((
            CrabJoint {
                id: joint_id,
                axis_local: axis,
            },
            CrabEnvId(0),
            MultibodyJoint::new(carapace, RevoluteJointBuilder::new(axis).build().into()),
            joint_t,
            joint_vel,
        ));

        world
            .run_system_once(build_observation)
            .expect("build observation");
        let got = world.resource::<CrabObservation>().envs[0];

        let mut want = Observation::default();
        want.joints[joint_id.index()].angle = joint_angle(axis, body_rot, joint_rot);
        want.joints[joint_id.index()].rate =
            (joint_vel.angular - body_vel.angular).dot(body_rot * axis);
        want.body.pos = body_t.translation - origin;
        want.body.rot = body_rot;
        want.body.linvel = body_vel.linear;
        want.body.angvel = body_vel.angular;
        want.target_local = body_rot.inverse() * (target - body_t.translation);
        assert_eq!(got, want.serialize(), "valid-input obs drifted (rl#242 pin)");
    }

    /// rl#242: a spawn-origin miss must be LOUD, never a silent Vec3::ZERO substitute
    /// feeding absolute coords into the spawn-relative channel.
    #[test]
    #[should_panic(expected = "spawn wiring bug")]
    fn missing_spawn_origin_panics_loud() {
        let mut world = obs_world(vec![]);
        world.spawn((
            CrabCarapace,
            CrabEnvId(0),
            Transform::default(),
            Velocity::default(),
        ));
        let _ = world.run_system_once(build_observation);
    }

    /// rl#242: a joint-parent motion miss must be LOUD, never a silent identity
    /// rotation corrupting joint angles/rates.
    #[test]
    #[should_panic(expected = "rig wiring bug")]
    fn missing_joint_parent_panics_loud() {
        let mut world = obs_world(vec![Vec3::ZERO]);
        let orphan_parent = world.spawn_empty().id();
        world.spawn((
            CrabJoint {
                id: CrabJointId::LegCoxa(Side::Left, 0),
                axis_local: Vec3::X,
            },
            CrabEnvId(0),
            MultibodyJoint::new(
                orphan_parent,
                RevoluteJointBuilder::new(Vec3::X).build().into(),
            ),
            Transform::default(),
            Velocity::default(),
        ));
        let _ = world.run_system_once(build_observation);
    }
}
