use bevy::prelude::*;
use bevy_rapier3d::prelude::{ExternalForce, Velocity};

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabCarapace, CrabJointId, Side};

#[derive(Resource, Clone)]
pub(super) struct RigPose {
    action: f32,
    joints: Vec<CrabJointId>,
}

/// Which joints `--rig-pose` drives (rl-demo screenshot mode).
#[derive(clap::ValueEnum, Debug, Clone, Copy, Default)]
pub enum RigPosePart {
    #[default]
    Shoulder,
    #[value(name = "legbasis")]
    LegBasis,
}

#[derive(Resource, Default)]
pub(super) struct RigPosePin {
    target: Option<Transform>,
}

impl RigPose {
    pub(super) fn new(action: f32, part: RigPosePart) -> Self {
        let joints = match part {
            RigPosePart::LegBasis => (0..4)
                .flat_map(|leg| {
                    [
                        CrabJointId::LegBasis(Side::Left, leg),
                        CrabJointId::LegBasis(Side::Right, leg),
                    ]
                })
                .collect(),
            RigPosePart::Shoulder => vec![
                CrabJointId::ClawShoulder(Side::Left),
                CrabJointId::ClawShoulder(Side::Right),
            ],
        };
        Self { action, joints }
    }
}

pub(super) fn rig_pose_drive(pose: Res<RigPose>, mut actions: ResMut<CrabActions>) {
    for &joint in &pose.joints {
        // Deliberate skip pre-spawn (env 0 not sized yet).
        let _ = actions.set_drive(0, joint, pose.action);
    }
}

pub(super) fn rig_pose_pin(
    mut pin: ResMut<RigPosePin>,
    mut carapace_q: Query<(&Transform, &Velocity, &mut ExternalForce), With<CrabCarapace>>,
) {
    let Ok((xform, vel, mut force)) = carapace_q.single_mut() else {
        return;
    };
    let target = *pin.target.get_or_insert(*xform);
    let (f, t) = pin_correction(&target, xform, vel);
    force.force += f;
    force.torque += t;
}

const PIN_ROT_KP: f32 = 20.0;
const PIN_ROT_KD: f32 = 12.0;
const PIN_POS_KP: f32 = 60.0;
const PIN_POS_KD: f32 = 30.0;
const PIN_MAX_TORQUE: f32 = 12.0;
const PIN_MAX_FORCE: f32 = 120.0;

fn pin_correction(target: &Transform, xform: &Transform, vel: &Velocity) -> (Vec3, Vec3) {
    let err_rot = target.rotation * xform.rotation.inverse();
    let (axis, angle) = err_rot.to_axis_angle();
    let angle = if angle > std::f32::consts::PI {
        angle - std::f32::consts::TAU
    } else {
        angle
    };
    let torque =
        (axis * angle * PIN_ROT_KP - vel.angular * PIN_ROT_KD).clamp_length_max(PIN_MAX_TORQUE);

    let err_pos = target.translation - xform.translation;
    let force = (err_pos * PIN_POS_KP - vel.linear * PIN_POS_KD).clamp_length_max(PIN_MAX_FORCE);
    (force, torque)
}
