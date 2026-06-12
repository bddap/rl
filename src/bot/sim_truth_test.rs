//! Pins the policy→physics motor path and the physics→render transform path.
//!
//! The rapier semantics that make this worth pinning: a revolute joint's
//! single degree of freedom always lives on `JointAxis::AngX` *of the joint's
//! local frame* — `RevoluteJointBuilder::new(axis)` rotates that frame so its
//! X axis lines up with the axis you pass (`LinX` for prismatic). A motor or
//! limit written to any other slot targets a LOCKED axis, and rapier silently
//! drops those on both the multibody path (`multibody_joint.rs::
//! velocity_constraints` iterates free axes only) and the impulse path
//! (`motor_axes & !locked_axes`) — leaving whatever the spawn builder's
//! `.motor_position(..)` put on the live AngX slot in charge forever.
//!
//! 1. `actuator_motor_writes_land_on_the_live_axis` (data-level): the
//!    actuator's runtime motor writes must land on the free-axis slot.
//!
//! 2. `commanded_leg_motors_actually_move_the_legs` (sim-level): headless
//!    app, settle, then command every femur/tibia to its action=+1 target for
//!    2.5 s — an eternity for these motors, so the hinge angles must visibly
//!    move. Also asserts bevy `Transform` == rapier `rb.position()` for every
//!    body part: the meshes must render exactly where physics says bodies are.

#![cfg(test)]

use std::time::Duration;

use bevy::prelude::*;
use bevy_rapier3d::plugin::context::RapierRigidBodySet;
use bevy_rapier3d::prelude::*;

use super::actuator::CrabActions;
use super::body::{CrabBodyPart, CrabCarapace, CrabJoint, CrabJointId, Side};
use super::{BotPlugin, NumEnvs};
use crate::Visuals;
use crate::physics::PhysicsWorldPlugin;

// ---------------------------------------------------------------------------
// Data-level mechanism
// ---------------------------------------------------------------------------

#[test]
fn actuator_motor_writes_land_on_the_live_axis() {
    use bevy_rapier3d::rapier::dynamics::JointAxesMask;

    let femur_id = CrabJointId::LegFemur(Side::Left, 0);

    // The femur joint exactly as spawn_leg builds it (left side, s = 1).
    let built = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor1(Vec3::new(0.15, 0.0, 0.0))
        .local_anchor2(Vec3::new(0.0, 0.18, 0.0))
        .limits(femur_id.limits())
        .motor_position(femur_id.default_position(), 50.0, 6.0)
        .motor_max_force(100.0)
        .motor_model(MotorModel::AccelerationBased);

    let mut typed: TypedJoint = built.into();
    let generic: &mut GenericJoint = typed.as_mut();

    // What apply_actions does every tick, here for action = +1 (target = 1.8):
    generic.set_motor_position(femur_id.joint_axis(), 1.8, 50.0, 6.0);
    generic.set_motor_max_force(femur_id.joint_axis(), femur_id.motor_max_force());

    let raw = generic.raw;
    // Motor slot order: LinX=0, LinY=1, LinZ=2, AngX=3, AngY=4, AngZ=5.
    // A revolute joint's single free axis is AngX of the joint's local frame;
    // every other axis is locked, and motors on locked axes are dropped by the
    // solver (`motor_axes & !locked_axes`). So for the policy's command to have
    // ANY physical effect, it must land on the AngX slot.
    assert!(!raw.locked_axes.contains(JointAxesMask::ANG_X));
    let live = raw.motors[3];
    assert_eq!(
        live.target_pos, 1.8,
        "the actuator's motor target must land on the joint's one free axis \
         (AngX); it went to a locked slot instead, so the joint stays pinned \
         at the spawn rest target ({})",
        live.target_pos
    );

    // Same story for the prismatic pincer: its free axis is LinX, not the
    // world Z the builder was given.
    let pincer_id = CrabJointId::ClawPincer(Side::Left);
    let built = PrismaticJointBuilder::new(Vec3::Z)
        .local_anchor1(Vec3::new(0.0, 0.0, 0.1))
        .local_anchor2(Vec3::new(0.0, 0.0, -0.12))
        .limits(pincer_id.limits())
        .motor_position(pincer_id.default_position(), 60.0, 8.0)
        .motor_max_force(150.0)
        .motor_model(MotorModel::AccelerationBased);
    let mut typed: TypedJoint = built.into();
    let generic: &mut GenericJoint = typed.as_mut();
    generic.set_motor_position(pincer_id.joint_axis(), 0.06, 60.0, 8.0);

    let raw = generic.raw;
    assert!(!raw.locked_axes.contains(JointAxesMask::LIN_X));
    assert_eq!(
        raw.motors[0].target_pos, 0.06,
        "the pincer's motor target must land on LinX, the prismatic free axis"
    );
}

// ---------------------------------------------------------------------------
// Sim-level: command the legs, watch nothing happen
// ---------------------------------------------------------------------------

fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(bevy::window::WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .set(bevy::render::RenderPlugin {
                render_creation: bevy::render::settings::RenderCreation::Automatic(
                    bevy::render::settings::WgpuSettings {
                        backends: None,
                        ..default()
                    },
                ),
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<bevy::log::LogPlugin>(),
    );
    // One fixed tick (1/64 s) per app.update(), like headless training.
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f64(1.0 / 64.0),
    ));
    app.insert_resource(Visuals(false))
        .insert_resource(NumEnvs(1))
        .insert_resource(TimestepMode::Fixed {
            dt: 1.0 / 64.0,
            substeps: 1,
        })
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
        .add_plugins(PhysicsWorldPlugin)
        .add_plugins(BotPlugin);
    app
}

fn tick(app: &mut App, n: u32) {
    for _ in 0..n {
        app.update();
    }
}

fn joint_entity(app: &mut App, id: CrabJointId) -> Entity {
    let mut q = app.world_mut().query::<(Entity, &CrabJoint)>();
    q.iter(app.world())
        .find(|(_, j)| j.id == id)
        .map(|(e, _)| e)
        .expect("crab joint entity")
}

fn carapace_entity(app: &mut App) -> Entity {
    let mut q = app
        .world_mut()
        .query_filtered::<Entity, With<CrabCarapace>>();
    q.single(app.world()).expect("carapace")
}

fn rotation(app: &App, e: Entity) -> Quat {
    app.world().get::<Transform>(e).expect("transform").rotation
}

/// Signed hinge angle of `child` relative to `parent` about `axis` (the axis
/// the revolute builder was given, expressed in both bodies' local frames).
fn hinge_angle(app: &App, parent: Entity, child: Entity, axis: Vec3) -> f32 {
    let mut q = (rotation(app, parent).inverse() * rotation(app, child)).normalize();
    if q.w < 0.0 {
        q = -q; // canonical double-cover representative
    }
    2.0 * Vec3::new(q.x, q.y, q.z).dot(axis).atan2(q.w)
}

fn set_action(app: &mut App, idx: usize, v: f32) {
    app.world_mut().resource_mut::<CrabActions>().envs[0][idx] = v;
}

/// Render truthfulness at the transform level: every crab body part's bevy
/// `Transform` (what the mesh renders at) must equal rapier's rigid-body pose.
fn assert_transforms_match_rapier(app: &mut App) {
    let mut parts_q = app.world_mut().query_filtered::<(
        Entity,
        &Transform,
        &RapierRigidBodyHandle,
    ), With<CrabBodyPart>>();
    let parts: Vec<(Entity, Transform, bevy_rapier3d::rapier::dynamics::RigidBodyHandle)> =
        parts_q
            .iter(app.world())
            .map(|(e, t, h)| (e, *t, h.0))
            .collect();
    assert!(!parts.is_empty());

    let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
    let set = set_q.single(app.world()).expect("rapier context");

    for (e, t, h) in parts {
        let iso = set.bodies.get(h).expect("rapier body").position();
        let pt: Vec3 = iso.translation;
        let pq: Quat = iso.rotation;
        assert!(
            (t.translation - pt).length() < 1e-3,
            "{e:?}: bevy Transform {:?} != rapier body {:?}",
            t.translation,
            pt
        );
        assert!(
            t.rotation.dot(pq).abs() > 1.0 - 1e-4,
            "{e:?}: bevy rotation {:?} != rapier rotation {:?}",
            t.rotation,
            pq
        );
    }
}

#[test]
fn commanded_leg_motors_actually_move_the_legs() {
    let mut app = headless_app();

    // Settle into the motor-held rest stance (~3 s).
    tick(&mut app, 192);

    let carapace = carapace_entity(&mut app);

    // Hinges to probe: all eight femurs plus both eye stalks.
    // Axis is the one the builder got: s*Z for femurs, +X for eyes.
    struct Hinge {
        name: String,
        parent: Entity,
        child: Entity,
        axis: Vec3,
        action_idx: usize,
        rest: f32,
        target_at_plus_one: f32,
    }
    let mut hinges = Vec::new();
    for side in [Side::Left, Side::Right] {
        let s = if side == Side::Left { 1.0 } else { -1.0 };
        for leg in 0u8..4 {
            let femur_id = CrabJointId::LegFemur(side, leg);
            let coxa = joint_entity(&mut app, CrabJointId::LegCoxa(side, leg));
            let femur = joint_entity(&mut app, femur_id);
            hinges.push(Hinge {
                name: format!("{femur_id:?}"),
                parent: coxa,
                child: femur,
                axis: Vec3::new(0.0, 0.0, s),
                action_idx: femur_id.index(),
                rest: femur_id.default_position(),
                target_at_plus_one: femur_id.limits()[1],
            });
        }
        let eye_id = CrabJointId::EyeStalk(side);
        let eye = joint_entity(&mut app, eye_id);
        hinges.push(Hinge {
            name: format!("{eye_id:?}"),
            parent: carapace,
            child: eye,
            axis: Vec3::X,
            action_idx: eye_id.index(),
            rest: eye_id.default_position(),
            target_at_plus_one: eye_id.limits()[1],
        });
    }

    // The meshes do render exactly where rapier says the bodies are — the
    // transform-writeback path is honest.
    assert_transforms_match_rapier(&mut app);

    // The spawn builders' AngX motors hold the rest pose.
    let settled: Vec<f32> = hinges
        .iter()
        .map(|h| hinge_angle(&app, h.parent, h.child, h.axis))
        .collect();
    for (h, &a) in hinges.iter().zip(&settled) {
        println!("settled  {:24} angle {a:+.3} (rest {:+.3})", h.name, h.rest);
        assert!(
            (a - h.rest).abs() < 0.35,
            "{} did not settle near its rest pose: {a:+.3} vs {:+.3}",
            h.name,
            h.rest
        );
    }

    // Command action = +1 on every probed joint (and the tibias, so the knees
    // are asked to straighten too) and give the motors 2.5 s — an eternity for
    // stiffness-50 acceleration-based motors with max force 100.
    for h in &hinges {
        set_action(&mut app, h.action_idx, 1.0);
    }
    for side in [Side::Left, Side::Right] {
        for leg in 0u8..4 {
            set_action(&mut app, CrabJointId::LegTibia(side, leg).index(), 1.0);
        }
    }
    tick(&mut app, 160);

    // Still rendering the truth at the transform level...
    assert_transforms_match_rapier(&mut app);

    let after: Vec<f32> = hinges
        .iter()
        .map(|h| hinge_angle(&app, h.parent, h.child, h.axis))
        .collect();
    for ((h, &a0), &a1) in hinges.iter().zip(&settled).zip(&after) {
        println!(
            "commanded {:24} angle {a0:+.3} -> {a1:+.3} (target {:+.3})",
            h.name, h.target_at_plus_one
        );
    }

    // Eye stalks: builder axis X, so even a world-axis-named motor write hits
    // the live AngX slot. These moving proves the motor pipeline itself works
    // — if they fail, the problem is upstream of the axis mapping.
    for ((h, &a0), &a1) in hinges.iter().zip(&settled).zip(&after) {
        if h.name.contains("EyeStalk") {
            assert!(
                (a1 - a0).abs() > 0.2,
                "{} should have moved toward {:+.3} but stayed at {a1:+.3}",
                h.name,
                h.target_at_plus_one
            );
        }
    }

    // The legs must respond the same way. If the actuator's writes target a
    // locked axis, the femurs sit pinned at the spawn rest target no matter
    // what the policy commands.
    for ((h, &a0), &a1) in hinges.iter().zip(&settled).zip(&after) {
        if h.name.contains("LegFemur") {
            assert!(
                (a1 - a0).abs() > 0.3,
                "{}: commanded from rest {a0:+.3} to {:+.3} for 2.5 s but the \
                 joint only reached {a1:+.3} — the policy's motor targets never \
                 reach the physical joint",
                h.name,
                h.target_at_plus_one
            );
        }
    }
}
