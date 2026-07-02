//! Builds the observation vector from physics state.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::CrabSpawns;
use super::body::{CrabCarapace, CrabEnvId, CrabJoint, CrabJointId, joint_angle};

/// One joint's observation block. Serializes to [`Self::LEN`] floats; the full
/// observation carries [`CrabJointId::COUNT`] of these.
#[derive(Clone, Copy, Default)]
struct JointObs {
    /// Current joint angle — the coordinate the policy drives by torque.
    angle: f32,
    /// Signed joint DOF rate d(angle)/dt: the link's angular velocity relative to
    /// its parent projected onto the joint axis. Signed (not a magnitude) so the
    /// policy can tell which way a joint moves and damp it.
    rate: f32,
}

impl JointObs {
    const LEN: usize = 2;
}

/// The carapace (body) observation block: trunk pose and velocity.
#[derive(Clone, Copy)]
struct BodyObs {
    /// Position relative to the env's spawn origin — "how far have I drifted",
    /// not the absolute grid slot, so x/z can't encode env identity.
    pos: Vec3,
    /// World orientation.
    rot: Quat,
    linvel: Vec3,
    angvel: Vec3,
}

impl Default for BodyObs {
    fn default() -> Self {
        // A ZERO quat (not the identity `Quat::default` would give): an env with no
        // carapace must serialize to an all-zero body block, since the policy reads
        // that as "no body signal" and the saved normalizer expects it. The rotation
        // is always overwritten with the real pose for any env that has a carapace.
        Self {
            pos: Vec3::ZERO,
            rot: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
        }
    }
}

impl BodyObs {
    const LEN: usize = 13; // pos(3) + quat(4) + linvel(3) + angvel(3)
}

/// Touch-target block width: the target's carapace-local xyz, always present so
/// [`OBS_SIZE`] is the same with or without a live target system.
const TARGET_LEN: usize = 3;

/// First slot of the target block in the serialized vector. The one home for that
/// offset: layout-aware readers (e.g. the directional obs test) take it from here
/// instead of re-deriving `COUNT*2 + 13`, so a layout change moves it in lockstep.
pub(crate) const TARGET_SLOT: usize = CrabJointId::COUNT * JointObs::LEN + BodyObs::LEN;

/// The full per-env observation — the SINGLE source of the obs memory layout.
/// Fields are written by name (a joint by [`CrabJointId::index`], the body block,
/// the target), so a write can't land in the wrong slot the way raw offset
/// arithmetic could. The flat `[f32; OBS_SIZE]` the policy network and normalizer
/// consume exists only as [`serialize`](Self::serialize)'s output: the slot
/// offsets live in that one function (and `TARGET_SLOT`), nowhere else.
#[derive(Clone, Copy)]
pub(crate) struct Observation {
    /// Per-joint blocks indexed by [`CrabJointId::index`].
    joints: [JointObs; CrabJointId::COUNT],
    body: BodyObs,
    /// The touch target in the carapace's LOCAL frame, or `Vec3::ZERO` when no
    /// target is set for the env. The policy can only reach a goal it perceives,
    /// so the reach target is part of the observation; see [`CrabTargets`].
    target_local: Vec3,
}

impl Default for Observation {
    fn default() -> Self {
        // Manual (not derived) so it holds for any `CrabJointId::COUNT`: the std
        // `[T; N]: Default` impl stops at N = 32, and the joint count is allowed to
        // grow past it (bddap/rl#31).
        Self {
            joints: [JointObs::default(); CrabJointId::COUNT],
            body: BodyObs::default(),
            target_local: Vec3::ZERO,
        }
    }
}

/// Total observation width, derived from [`Observation`]'s block layout — the one
/// place the slot count is defined. The policy net's input dim and the
/// normalizer's per-slot vector length both come from here.
///
/// The live checkpoint's policy weights AND the obs-normalizer's per-slot Welford
/// state are keyed to BOTH this value and [`Observation::serialize`]'s slot order;
/// changing either silently invalidates every saved policy + normalizer, so both
/// are pinned by the `serialize_is_slot_identical` / `obs_size_is_unchanged` tests.
pub const OBS_SIZE: usize = TARGET_SLOT + TARGET_LEN;

impl Observation {
    /// Pack into the flat slot vector the policy network and normalizer consume.
    /// The ONLY place obs fields become raw offsets.
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

/// Resource holding the current observation vector for each environment.
#[derive(Resource, Default)]
pub struct CrabObservation {
    /// `envs[e]` = env e's observation, in the serialized wire form the network reads.
    pub envs: Vec<[f32; OBS_SIZE]>,
}

impl CrabObservation {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![[0.0; OBS_SIZE]; n];
    }
}

/// The per-env touch target: a point in WORLD space the crab is rewarded for
/// reaching with a claw tip (see the target-touch reward in `training::reward`).
/// Owned by the training plugin, which respawns a fresh target each episode; the
/// observation reads it to tell the policy where to reach. Empty / `None` outside
/// training (the demo), where `build_observation` then writes a zero target vector
/// so the same `OBS_SIZE` is produced without a target system present.
#[derive(Resource, Default)]
pub struct CrabTargets {
    /// `envs[e]` = env e's target world position, or `None` if unset for that env.
    pub envs: Vec<Option<Vec3>>,
}

impl CrabTargets {
    pub fn resize(&mut self, n: usize) {
        self.envs = vec![None; n];
    }

    /// Env `e`'s target world position, or `None` if `e` is out of range or unset.
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

    // World rotation + velocity of every part, keyed by entity. A joint reads its
    // PARENT's motion from here: the velocity input is the signed DOF rate (the
    // joint coordinate's velocity), which is the part's motion RELATIVE to its
    // parent about the joint axis — that needs the parent's world rotation and
    // velocity. The carapace is included because it parents the coxae/claws/eyes.
    let mut motion: HashMap<Entity, (Quat, Vec3, Vec3)> = HashMap::new();
    for (e, _, _, _, t, vel) in joint_q.iter() {
        motion.insert(e, (t.rotation, vel.linear, vel.angular));
    }
    for (e, _, t, vel) in carapace_q.iter() {
        motion.insert(e, (t.rotation, vel.linear, vel.angular));
    }

    // -- Per-joint observations ------------------------------------------------
    for (_, crab_joint, env, mj, transform, vel) in joint_q.iter() {
        let Some(o) = built.get_mut(env.0) else {
            continue;
        };
        let (parent_rot, _parent_lin, parent_ang) =
            motion
                .get(&mj.parent)
                .copied()
                .unwrap_or((Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO));

        let joint = &mut o.joints[crab_joint.id.index()];
        joint.angle = joint_angle(crab_joint.axis_local, parent_rot, transform.rotation);
        // Every joint is revolute now (the pincer too), so the relative-to-parent
        // velocity projected on the world axis is always the angular DOF rate.
        let axis_world = parent_rot * crab_joint.axis_local;
        joint.rate = (vel.angular - parent_ang).dot(axis_world);
    }

    // -- Body state (carapace) -------------------------------------------------
    for (_, env, transform, vel) in carapace_q.iter() {
        let Some(o) = built.get_mut(env.0) else {
            continue;
        };
        let origin = spawns.0.get(env.0).copied().unwrap_or(Vec3::ZERO);
        o.body.pos = transform.translation - origin;
        o.body.rot = transform.rotation;
        o.body.linvel = vel.linear;
        o.body.angvel = vel.angular;

        // Touch target in the carapace's LOCAL frame: the displacement from the
        // carapace to the target, rotated into body axes so "target is to my
        // left / above my back" reads the same regardless of world yaw — the
        // orientation-invariance the policy needs to reach from any heading (the
        // randomized starts spawn it facing anywhere).
        o.target_local = targets
            .get(env.0)
            .map(|t| transform.rotation.inverse() * (t - transform.translation))
            .unwrap_or(Vec3::ZERO);
    }

    // -- Single serialization boundary: typed obs -> flat wire vector ----------
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
        // rescue_nonfinite_crabs runs .before(Sense) but only inspects part
        // TRANSFORMS, never velocities — a non-finite joint/body rate can still
        // reach here for the one tick before it integrates into a non-finite pose
        // that rescue then catches. Surface it loudly rather than letting a
        // plausible-but-wrong 0 silently train the policy; the substitution is a
        // last-resort guard so one bad rate can neither poison the whole network
        // (NaN spreads through every weight) nor crash the live trainer.
        error!(
            "build_observation sanitized {nonfinite} non-finite observation slot(s) \
             this tick — corrupt physics upstream of Sense (see rescue_nonfinite_crabs)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The serialized vector MUST stay slot-for-slot identical to the hand-packed
    /// layout the live checkpoint's policy weights + per-slot normalizer were
    /// trained against: joint `idx` -> `[idx*2]`=angle, `[idx*2+1]`=rate; body block
    /// at `COUNT*2` = pos(3)/quat-xyzw(4)/linvel(3)/angvel(3); target at `+13`. Any
    /// drift here silently invalidates every saved policy, so this pins the mapping
    /// against independent offset arithmetic.
    #[test]
    fn serialize_is_slot_identical() {
        let mut o = Observation::default();
        // Distinct per-field values so a transposed or off-by-one slot is caught.
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

        // Independent hand-pack with the CURRENT offsets.
        let mut want = [0.0f32; OBS_SIZE];
        for idx in 0..CrabJointId::COUNT {
            want[idx * 2] = idx as f32 + 0.5;
            want[idx * 2 + 1] = -(idx as f32) - 0.25;
        }
        let body_base = CrabJointId::COUNT * 2;
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

    /// `OBS_SIZE` must stay this exact numeric width — the on-disk checkpoint +
    /// normalizer vectors are this long; the block-width derivation must not move it.
    #[test]
    fn obs_size_is_unchanged() {
        assert_eq!(OBS_SIZE, CrabJointId::COUNT * 2 + 13 + 3);
    }

    /// A default observation (no carapace, target absent — the demo path) serializes
    /// to all zeros at the same width, so a missing target system can't change shape.
    #[test]
    fn default_serializes_to_zeros() {
        assert_eq!(Observation::default().serialize(), [0.0f32; OBS_SIZE]);
    }
}
