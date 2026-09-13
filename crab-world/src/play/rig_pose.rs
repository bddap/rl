use bevy::prelude::*;
use bevy_rapier3d::plugin::context::{RapierContextJoints, RapierRigidBodySet};
use bevy_rapier3d::prelude::RapierMultibodyJointHandle;

use crate::bot::body::{CrabCarapace, CrabJoint, CrabJointId, CrabRestPose, Side};
use crate::bot::skin::CrabRenderPose;

#[derive(Resource, Clone)]
pub(super) struct RigPose {
    angle: f32,
    joints: Vec<CrabJointId>,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, Default)]
pub enum RigPosePart {
    #[default]
    Shoulder,
    #[value(name = "legbasis")]
    LegBasis,
    Wrist,
}

impl RigPose {
    pub(super) fn new(angle: f32, part: RigPosePart) -> Self {
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
            RigPosePart::Wrist => vec![
                CrabJointId::ClawWrist(Side::Left),
                CrabJointId::ClawWrist(Side::Right),
            ],
        };
        Self { angle, joints }
    }
}

pub(super) fn rig_pose_render(
    pose: Res<RigPose>,
    contexts: Query<(&RapierContextJoints, &RapierRigidBodySet)>,
    joints: Query<(Entity, &CrabJoint, &RapierMultibodyJointHandle)>,
    carapace: Query<(Entity, &CrabRestPose), With<CrabCarapace>>,
    mut rendered: ResMut<CrabRenderPose>,
) {
    let (Ok((context, bodies)), Ok((root_entity, rest)), Some((_, _, handle))) =
        (contexts.single(), carapace.single(), joints.iter().next())
    else {
        return;
    };
    let Some((multibody, _)) = context.multibody_joints.get(handle.0) else {
        return;
    };
    let mut displacement = vec![0.0; multibody.ndofs()];
    for (_, joint, handle) in &joints {
        let (_, index) = context.multibody_joints.get(handle.0).expect("joint link");
        let link = multibody.link(index).expect("link");
        let target = if pose.joints.contains(&joint.id) {
            pose.angle
        } else {
            0.0
        };
        displacement[link.assembly_id()] = target - link.joint().coords()[3];
    }
    let root =
        multibody.forward_kinematics_single_link(&bodies.bodies, 0, Some(&displacement), None);
    rendered.0.clear();
    rendered.0.insert(root_entity, rest.0);
    for (entity, _, handle) in &joints {
        let (_, index) = context.multibody_joints.get(handle.0).expect("joint link");
        let relative = root.inverse()
            * multibody.forward_kinematics_single_link(
                &bodies.bodies,
                index,
                Some(&displacement),
                None,
            );
        rendered.0.insert(
            entity,
            rest.0
                * Transform {
                    translation: relative.translation,
                    rotation: relative.rotation,
                    ..default()
                },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::headless::{flat_headless_app, tick};
    use bevy::ecs::system::RunSystemOnce;

    #[test]
    fn prescribed_wrists_render_equal_coordinates_without_moving_physics() {
        let mut app = flat_headless_app();
        tick(&mut app, 3);
        app.insert_resource(RigPose::new(0.6, RigPosePart::Wrist))
            .init_resource::<CrabRenderPose>();
        let before: Vec<_> = app
            .world_mut()
            .query::<(Entity, &CrabJoint, &Transform)>()
            .iter(app.world())
            .map(|(e, j, t)| (e, *j, *t))
            .collect();
        let root = app
            .world_mut()
            .query_filtered::<&CrabRestPose, With<CrabCarapace>>()
            .single(app.world())
            .expect("carapace")
            .0;
        app.world_mut().run_system_once(rig_pose_render).unwrap();
        let rendered = app.world().resource::<CrabRenderPose>();
        for (entity, joint, physical) in before {
            assert_eq!(*app.world().get::<Transform>(entity).unwrap(), physical);
            let angle = if matches!(joint.id, CrabJointId::ClawWrist(_)) {
                0.6
            } else if matches!(joint.id, CrabJointId::ClawShoulder(_)) {
                0.0
            } else {
                continue;
            };
            let expected =
                root.rotation * Quat::from_axis_angle(joint.axis_local.normalize(), angle);
            let actual = rendered.0[&entity].rotation;
            assert!(
                (actual - expected)
                    .length()
                    .min((actual + expected).length())
                    < 1e-4,
                "{:?}: {actual:?} != {expected:?}",
                joint.id
            );
        }
    }
}
