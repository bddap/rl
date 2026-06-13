//! Pins the policy→physics→render path under torque control.
//!
//! The crab is torque-controlled: each action is a signed torque applied about
//! the joint's free axis (no position servo). Two invariants worth pinning,
//! because the failure that hid here before — a command landing on a locked
//! axis and never reaching the joint — was silent and looked like good
//! training:
//!
//! 1. `commanded_torque_moves_the_joints`: commanding +1 vs −1 on the femurs
//!    must drive their angles to opposite ends of the range. Gravity is the
//!    same in both runs, so the *difference* isolates the commanded torque.
//! 2. Render honesty: every body part's bevy `Transform` equals rapier's pose,
//!    so the meshes render exactly where physics puts the bodies.

use bevy::prelude::*;

use super::actuator::CrabActions;
use super::body::{CrabJoint, CrabJointId, Side, joint_angle};
use super::test_util::{assert_transforms_match_rapier, headless_app, tick};

fn joint_entity(app: &mut App, id: CrabJointId) -> Entity {
    let mut q = app.world_mut().query::<(Entity, &CrabJoint)>();
    q.iter(app.world())
        .find(|(_, j)| j.id == id)
        .map(|(e, _)| e)
        .expect("crab joint entity")
}

/// Mean femur angle after holding a constant torque on every femur (and tibia,
/// to unlock the knee) for ~2.5 s on a fresh crab. `check_render` asserts the
/// transform-writeback path on the way out.
fn mean_femur_angle_under_torque(torque: f32, check_render: bool) -> f32 {
    let mut app = headless_app();
    // One tick so spawn_initial_crabs has sized CrabActions and built the crab.
    tick(&mut app, 1);

    {
        let mut actions = app.world_mut().resource_mut::<CrabActions>();
        for side in [Side::Left, Side::Right] {
            for leg in 0u8..4 {
                actions.envs[0][CrabJointId::LegFemur(side, leg).index()] = torque;
                actions.envs[0][CrabJointId::LegTibia(side, leg).index()] = torque;
            }
        }
    }
    tick(&mut app, 160);

    if check_render {
        assert_transforms_match_rapier(&mut app);
    }

    // Gather (femur, coxa, id) entities, then read their transforms.
    let mut pairs = Vec::new();
    for side in [Side::Left, Side::Right] {
        for leg in 0u8..4 {
            let id = CrabJointId::LegFemur(side, leg);
            let femur = joint_entity(&mut app, id);
            let coxa = joint_entity(&mut app, CrabJointId::LegCoxa(side, leg));
            pairs.push((id, femur, coxa));
        }
    }
    let sum: f32 = pairs
        .iter()
        .map(|&(id, femur, coxa)| {
            let cr = app.world().get::<Transform>(femur).unwrap().rotation;
            let pr = app.world().get::<Transform>(coxa).unwrap().rotation;
            joint_angle(id, pr, cr)
        })
        .sum();
    sum / pairs.len() as f32
}

#[test]
fn commanded_torque_moves_the_joints() {
    let plus = mean_femur_angle_under_torque(1.0, true);
    let minus = mean_femur_angle_under_torque(-1.0, false);
    println!("mean femur angle: +1 torque {plus:+.3}, -1 torque {minus:+.3}");
    // Opposite torques must drive the femurs to clearly different angles — that
    // is the proof the command reaches the physical joint. (The sign of the
    // angle convention vs the action is irrelevant; the policy learns it.)
    assert!(
        (plus - minus).abs() > 0.5,
        "commanded torque did not reach the femurs: +1 gave {plus:+.3}, -1 gave \
         {minus:+.3} — opposite commands should split the joint angle"
    );
}
