use bevy::prelude::*;

use super::actuator::CrabActions;
use super::body::{CrabJoint, CrabJointId, Side, joint_angle};
use super::headless::{assert_transforms_match_rapier, flat_headless_app, headless_app, tick};

/// Rest-quiet ceilings. The angular one is sized to the CHAOS of the rest noise, not
/// one draw of it: the worst-claw-link mean is solver convergence noise on
/// near-massless links, and bit-level physics perturbations re-roll it — seven draws
/// during rl#340 stage 3 measured 0.24–0.38 rad/s, so 0.6 clears the 7-draw max ~60%
/// while still firing on the guarded diseases: removed rest support is 3-4x (rl#109)
/// and the pre-rl#340-stage-2 convergence regression measured 2.43 rad/s.
const QUIET_LIN_MPS: f32 = 0.2;
const QUIET_ANG_RADPS: f32 = 0.6;

fn joint_entity(app: &mut App, id: CrabJointId) -> Entity {
    let mut q = app.world_mut().query::<(Entity, &CrabJoint)>();
    q.iter(app.world())
        .find(|(_, j)| j.id == id)
        .map(|(e, _)| e)
        .expect("crab joint entity")
}

fn mean_merus_angle_under_torque(torque: f32, check_render: bool) -> f32 {
    let mut app = headless_app();
    tick(&mut app, 1);

    {
        let mut actions = app.world_mut().resource_mut::<CrabActions>();
        for side in [Side::Left, Side::Right] {
            for leg in 0u8..4 {
                assert!(actions.set_drive(0, CrabJointId::LegMerus(side, leg), torque));
                assert!(actions.set_drive(0, CrabJointId::LegCarpus(side, leg), torque));
            }
        }
    }
    tick(&mut app, 160);

    if check_render {
        assert_transforms_match_rapier(&mut app);
    }

    let mut pairs = Vec::new();
    for side in [Side::Left, Side::Right] {
        for leg in 0u8..4 {
            let merus = joint_entity(&mut app, CrabJointId::LegMerus(side, leg));
            let coxa = joint_entity(&mut app, CrabJointId::LegCoxa(side, leg));
            pairs.push((merus, coxa));
        }
    }
    let sum: f32 = pairs
        .iter()
        .map(|&(merus, coxa)| {
            let axis = app.world().get::<CrabJoint>(merus).unwrap().axis_local;
            let cr = app.world().get::<Transform>(merus).unwrap().rotation;
            let pr = app.world().get::<Transform>(coxa).unwrap().rotation;
            joint_angle(axis, pr, cr)
        })
        .sum();
    sum / pairs.len() as f32
}

#[test]
fn commanded_torque_moves_the_joints() {
    let plus = mean_merus_angle_under_torque(1.0, true);
    let minus = mean_merus_angle_under_torque(-1.0, false);
    println!("mean merus angle: +1 torque {plus:+.3}, -1 torque {minus:+.3}");
    assert!(
        (plus - minus).abs() > 0.5,
        "commanded torque did not reach the merus joints: +1 gave {plus:+.3}, -1 gave \
         {minus:+.3} — opposite commands should split the joint angle"
    );
}

#[test]
fn joint_friction_bounds_limb_speed() {
    use super::body::CrabBodyPart;
    use bevy_rapier3d::prelude::Velocity;

    let mut app = headless_app();
    tick(&mut app, 1);
    {
        assert!(app.world_mut().resource_mut::<CrabActions>().fill(0, 1.0));
    }
    tick(&mut app, 160);

    let mut max_ang = 0.0f32;
    let mut q = app
        .world_mut()
        .query_filtered::<&Velocity, With<CrabBodyPart>>();
    for vel in q.iter(app.world()) {
        max_ang = max_ang.max(vel.angular.length());
    }
    println!("max limb angular speed under full torque: {max_ang:.1} rad/s");
    assert!(
        max_ang < 100.0,
        "a limb is spinning at {max_ang:.1} rad/s under full torque — joint \
         friction/ceiling/mass regressed (pre-fix the carpus hit 300–600 rad/s and \
         the blow-up guard then killed every episode in ~8 steps)"
    );
}

#[test]
fn crab_spawns_in_rest_pose_inside_limits() {
    use bevy_rapier3d::prelude::MultibodyJoint;
    use std::collections::HashMap;

    let mut app = flat_headless_app();
    tick(&mut app, 3);

    let mut tf_q = app.world_mut().query::<(Entity, &Transform)>();
    let rot: HashMap<Entity, Quat> = tf_q
        .iter(app.world())
        .map(|(e, t)| (e, t.rotation))
        .collect();

    let mut joint_q = app
        .world_mut()
        .query::<(&CrabJoint, &MultibodyJoint, &Transform)>();
    let mut checked = 0;
    for (joint, mj, tf) in joint_q.iter(app.world()) {
        let id = joint.id;
        let angle = joint_angle(joint.axis_local, rot[&mj.parent], tf.rotation);
        let [lo, hi] = id.limits();
        assert!(
            angle.abs() < 0.15,
            "{id:?} spawned at {angle:+.3} rad, not its ~0 bind-pose rest — the rig \
             link is not spawning at joint coordinate 0"
        );
        assert!(
            angle >= lo - 1e-3 && angle <= hi + 1e-3,
            "{id:?} spawned at {angle:+.3} rad, outside its limits [{lo:+.3}, {hi:+.3}]"
        );
        checked += 1;
    }
    assert_eq!(checked, CrabJointId::COUNT);
}

#[test]
fn claw_joint_frames_and_point_trajectories_are_bilateral() {
    use bevy_rapier3d::plugin::context::{RapierContextJoints, RapierRigidBodySet};
    use bevy_rapier3d::prelude::RapierMultibodyJointHandle;
    use std::collections::HashMap;

    fn claw(side: Side) -> [CrabJointId; 3] {
        [
            CrabJointId::ClawShoulder(side),
            CrabJointId::ClawWrist(side),
            CrabJointId::ClawPincer(side),
        ]
    }

    fn mirror_point(v: Vec3) -> Vec3 {
        Vec3::new(-v.x, v.y, v.z)
    }

    fn mirror_rotation(q: Quat) -> Quat {
        Quat::from_xyzw(q.x, -q.y, -q.z, q.w)
    }

    fn rotation_distance(a: Quat, b: Quat) -> f32 {
        (a - b).length().min((a + b).length())
    }

    const TOL: f32 = 1e-3;
    const RIGHT_POINT: Vec3 = Vec3::new(-0.04, -0.08, 0.03);

    let mut app = flat_headless_app();
    tick(&mut app, 3);

    let handles: HashMap<CrabJointId, _> = {
        let mut q = app
            .world_mut()
            .query::<(&CrabJoint, &RapierMultibodyJointHandle)>();
        q.iter(app.world()).map(|(j, h)| (j.id, h.0)).collect()
    };
    let right = claw(Side::Right);
    let left = claw(Side::Left);
    let mut context_q = app
        .world_mut()
        .query::<(&RapierContextJoints, &RapierRigidBodySet)>();
    let (joints, bodies) = context_q.single(app.world()).expect("one rapier context");
    let (multibody, _) = joints
        .multibody_joints
        .get(handles[&right[0]])
        .expect("right claw is in a multibody");
    let mut links = HashMap::new();
    for id in right.into_iter().chain(left) {
        let (candidate, link_id) = joints
            .multibody_joints
            .get(handles[&id])
            .unwrap_or_else(|| panic!("{id:?} has no multibody link"));
        assert!(std::ptr::eq(multibody, candidate));
        links.insert(id, link_id);
    }

    let sweeps: Vec<[f32; 3]> = right
        .iter()
        .map(|id| {
            let [lo, hi] = id.limits();
            [lo, 0.0, hi]
        })
        .collect();
    for &shoulder in &sweeps[0] {
        for &wrist in &sweeps[1] {
            for &pincer in &sweeps[2] {
                let mut displacement = vec![0.0; multibody.ndofs()];
                for (r, l) in right.iter().zip(&left) {
                    let target = match r {
                        CrabJointId::ClawShoulder(_) => shoulder,
                        CrabJointId::ClawWrist(_) => wrist,
                        _ => pincer,
                    };
                    for id in [*r, *l] {
                        let link = multibody.link(links[&id]).expect("claw link");
                        displacement[link.assembly_id()] = target - link.joint().coords()[3];
                    }
                }

                let root = multibody.forward_kinematics_single_link(
                    &bodies.bodies,
                    0,
                    Some(&displacement),
                    None,
                );
                for (r, l) in right.iter().zip(&left) {
                    let rp = root.inverse()
                        * multibody.forward_kinematics_single_link(
                            &bodies.bodies,
                            links[r],
                            Some(&displacement),
                            None,
                        );
                    let lp = root.inverse()
                        * multibody.forward_kinematics_single_link(
                            &bodies.bodies,
                            links[l],
                            Some(&displacement),
                            None,
                        );
                    let position_error = (lp.translation - mirror_point(rp.translation)).length();
                    let rotation_error =
                        rotation_distance(lp.rotation, mirror_rotation(rp.rotation));
                    let point_error =
                        (lp * mirror_point(RIGHT_POINT) - mirror_point(rp * RIGHT_POINT)).length();
                    assert!(
                        position_error < TOL && rotation_error < TOL && point_error < TOL,
                        "{l:?} at (shoulder {shoulder:+.3}, wrist {wrist:+.3}, pincer \
                         {pincer:+.3}) does not reflect {r:?}: frame position \
                         {position_error:.4} m, rotation {rotation_error:.4}, point trajectory \
                         {point_error:.4} m"
                    );
                }
            }
        }
    }
}

#[test]
fn actuator_injects_no_net_wrench() {
    use super::actuator::CrabActions;
    use super::body::CrabBodyPart;
    use bevy_rapier3d::prelude::ExternalForce;

    use std::collections::HashMap;

    let mut app = headless_app();
    tick(&mut app, 1);
    {
        assert!(app.world_mut().resource_mut::<CrabActions>().fill(0, 1.0));
    }
    tick(&mut app, 40);

    let pos: HashMap<Entity, Vec3> = {
        let mut q = app
            .world_mut()
            .query_filtered::<(Entity, &Transform), With<CrabBodyPart>>();
        q.iter(app.world())
            .map(|(e, t)| (e, t.translation))
            .collect()
    };
    // `ExternalForce` after a tick also carries the rl#332 carapace drag — a
    // modeled EXTERNAL force. Predict it from the same pre-tick `Velocity` the
    // drag system reads, so the assert isolates the actuator's contribution.
    let expected_drag = {
        let (v, c) = carapace_vel_and_drag(&mut app);
        -(c * v.length()) * v
    };
    tick(&mut app, 1);

    let carapace = {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<super::body::CrabCarapace>>();
        q.single(app.world()).expect("one carapace")
    };
    let mut net_force = Vec3::ZERO;
    let mut net_torque = Vec3::ZERO;
    let mut q = app
        .world_mut()
        .query_filtered::<(Entity, &ExternalForce), With<CrabBodyPart>>();
    for (e, ef) in q.iter(app.world()) {
        let actuator_force = if e == carapace {
            ef.force - expected_drag
        } else {
            ef.force
        };
        net_force += actuator_force;
        net_torque += pos[&e].cross(actuator_force) + ef.torque;
    }
    println!(
        "actuator net force {:.5} N (drag {:.5} N removed), net torque {:.5} N·m",
        net_force.length(),
        expected_drag.length(),
        net_torque.length()
    );
    assert!(
        net_force.length() < 1e-2,
        "actuator injects net force {net_force:?} — not an internal wrench"
    );
    assert!(
        net_torque.length() < 1e-2,
        "actuator injects net torque {net_torque:?} ({:.3} N·m) — a momentum leak: \
         the crab can spin itself up mid-air with no external torque",
        net_torque.length()
    );
}

#[test]
fn unactuated_crab_crumples_under_load() {
    use super::body::CrabCarapace;

    fn carapace_y(app: &mut App) -> f32 {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        q.iter(app.world()).next().expect("carapace").translation.y
    }

    let mut app = headless_app();
    tick(&mut app, 3);
    let start_y = carapace_y(&mut app);

    tick(&mut app, 128);
    let end_y = carapace_y(&mut app);
    let leg_deflection = max_leg_joint_deflection(&mut app);
    println!(
        "carapace y: spawn {start_y:.3} -> unactuated+2s {end_y:.3} (Δ {:.3}); \
         max leg-joint deflection {leg_deflection:.3} rad",
        start_y - end_y
    );
    assert!(
        leg_deflection > 0.15,
        "no leg JOINT yielded ({leg_deflection:.3} rad max from rest) — the joints hold \
         the body up rigidly instead of folding to load (a passive standing statue, the \
         bug this guards against); friction too stiff to crumple"
    );
}

fn max_leg_joint_deflection(app: &mut App) -> f32 {
    use bevy_rapier3d::prelude::MultibodyJoint;
    use std::collections::HashMap;

    let rot: HashMap<Entity, Quat> = {
        let mut q = app.world_mut().query::<(Entity, &Transform)>();
        q.iter(app.world()).map(|(e, t)| (e, t.rotation)).collect()
    };
    let mut q = app
        .world_mut()
        .query::<(&CrabJoint, &MultibodyJoint, &Transform)>();
    let mut max_def = 0.0f32;
    for (joint, mj, tf) in q.iter(app.world()) {
        if !matches!(
            joint.id,
            CrabJointId::LegMerus(..) | CrabJointId::LegCarpus(..)
        ) {
            continue;
        }
        let angle = joint_angle(joint.axis_local, rot[&mj.parent], tf.rotation);
        max_def = max_def.max(angle.abs());
    }
    max_def
}

fn max_anchor_separation(app: &mut App) -> f32 {
    use bevy_rapier3d::plugin::context::RapierRigidBodySet;
    use bevy_rapier3d::prelude::{GenericJoint, MultibodyJoint, RapierRigidBodyHandle};
    use bevy_rapier3d::rapier::dynamics::RigidBodyHandle;
    use std::collections::HashMap;

    let handles: HashMap<Entity, RigidBodyHandle> = {
        let mut q = app.world_mut().query::<(Entity, &RapierRigidBodyHandle)>();
        q.iter(app.world()).map(|(e, h)| (e, h.0)).collect()
    };
    let joints: Vec<(Entity, Entity, Vec3, Vec3)> = {
        let mut q = app.world_mut().query::<(Entity, &MultibodyJoint)>();
        q.iter(app.world())
            .map(|(child, mj)| {
                let g: &GenericJoint = mj.data.as_ref();
                (child, mj.parent, g.local_anchor1(), g.local_anchor2())
            })
            .collect()
    };
    let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
    let set = set_q.single(app.world()).expect("rapier set");
    let mut max_gap = 0.0f32;
    for (child, parent, a1, a2) in joints {
        let (Some(&ph), Some(&ch)) = (handles.get(&parent), handles.get(&child)) else {
            continue;
        };
        let w1: Vec3 = set.bodies.get(ph).expect("parent body").position() * a1;
        let w2: Vec3 = set.bodies.get(ch).expect("child body").position() * a2;
        max_gap = max_gap.max((w1 - w2).length());
    }
    max_gap
}

#[test]
fn crab_settles_quietly_at_rest() {
    use super::body::CrabCarapace;
    use bevy_rapier3d::prelude::Velocity;

    fn carapace(app: &mut App) -> (f32, f32) {
        let mut q = app
            .world_mut()
            .query_filtered::<(&Velocity, &Transform), With<CrabCarapace>>();
        let (v, t) = q.iter(app.world()).next().expect("carapace");
        (v.angular.length(), t.translation.y)
    }

    let mut app = flat_headless_app();
    tick(&mut app, 1);

    tick(&mut app, 320);
    settle_to_sleep(&mut app, 1024);
    let crumple = max_leg_joint_deflection(&mut app);

    let mut ang_sum = 0.0f32;
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    let mut max_gap = 0.0f32;
    let window = 192u32;
    for _ in 0..window {
        tick(&mut app, 1);
        let (ang, y) = carapace(&mut app);
        ang_sum += ang;
        y_min = y_min.min(y);
        y_max = y_max.max(y);
        max_gap = max_gap.max(max_anchor_separation(&mut app));
    }
    let ang_mean = ang_sum / window as f32;
    let bounce = y_max - y_min;
    println!(
        "rest: carapace angular speed mean {ang_mean:.3} rad/s, bounce {bounce:.4} m, \
         leg crumple {crumple:.3} rad, max anchor gap {max_gap:.4} m"
    );

    assert!(
        ang_mean < QUIET_ANG_RADPS,
        "carapace still twitching at rest: angular speed mean {ang_mean:.3} rad/s \
         (want <{}; 12 Hz contact sits ~0.61, the 30 Hz / substeps=1 regressions ~1.5)",
        QUIET_ANG_RADPS
    );
    assert!(
        bounce < 0.024,
        "carapace bouncing at rest: {bounce:.4} m peak-to-peak (want <0.024 at the 0.04 \
         floppy cap; the 30 Hz contact regression ~0.036, substeps=1 ~0.030; the rapier-0.35 \
         plant measures 0.012-0.015 across 3 solver-noise draws — was 0.0014-0.0025 on 0.32 \
         — so headroom is ~1.6x, not the old 10x: before blaming a marginal red on \
         realization spread, re-measure the draw spread, stage-3 style)"
    );
    assert!(
        crumple > 0.4,
        "legs no longer crumple ({crumple:.3} rad) — the rest-quiet fix must not \
         stiffen the legs into a rigid brace; keep them floppy"
    );
    assert!(
        max_gap < 0.08,
        "a limb is separating from its parent: max anchor gap {max_gap:.4} m under \
         standing load (want <0.08; the joint positional lock has been softened too \
         far and the limbs are detaching)"
    );
}

/// Tick until every dynamic body sleeps, or `cap` ticks — the rl#392 settle for
/// rest measurements. Rest quiet is delivered by SLEEP, so a rest window starts
/// at "the crab fell asleep", not at a fixed tick count: under a loaded box the
/// async sally.glb load can spawn the crab hundreds of ticks late, and a fixed
/// settle then measures the settle transient instead of rest (the
/// contended-suite flake this replaces). A crab that never sleeps caps out and
/// the caller's bounds fail honestly.
fn settle_to_sleep(app: &mut App, cap: u32) {
    use bevy_rapier3d::plugin::context::RapierRigidBodySet;

    for _ in 0..cap {
        tick(app, 1);
        let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
        let set = set_q.single(app.world()).unwrap();
        let states: Vec<bool> = set
            .bodies
            .iter()
            .filter(|(_, rb)| rb.is_dynamic())
            .map(|(_, rb)| rb.is_sleeping())
            .collect();
        if !states.is_empty() && states.iter().all(|s| *s) {
            return;
        }
    }
}

#[test]
fn claws_quiet_at_rest() {
    use bevy_rapier3d::prelude::Velocity;

    let mut app = flat_headless_app();
    tick(&mut app, 1);
    tick(&mut app, 320);
    settle_to_sleep(&mut app, 1024);

    let (mut lin_sum, mut ang_sum) = (0.0f32, 0.0f32);
    let window = 192u32;
    for _ in 0..window {
        tick(&mut app, 1);
        let (mut lin, mut ang) = (0.0f32, 0.0f32);
        let mut q = app.world_mut().query::<(&CrabJoint, &Velocity)>();
        for (joint, v) in q.iter(app.world()) {
            if matches!(
                joint.id,
                CrabJointId::ClawShoulder(_)
                    | CrabJointId::ClawWrist(_)
                    | CrabJointId::ClawPincer(_)
            ) {
                lin = lin.max(v.linear.length());
                ang = ang.max(v.angular.length());
            }
        }
        lin_sum += lin;
        ang_sum += ang;
    }
    let lin_mean = lin_sum / window as f32;
    let ang_mean = ang_sum / window as f32;
    println!(
        "claws at rest: mean worst-link linear {lin_mean:.3} m/s, angular {ang_mean:.3} rad/s"
    );
    assert!(
        lin_mean < QUIET_LIN_MPS,
        "claw links shaking at rest: mean worst-link linear speed {lin_mean:.3} m/s \
         (want <{}) — the contact spring regressed stiffer",
        QUIET_LIN_MPS
    );
    assert!(
        ang_mean < QUIET_ANG_RADPS,
        "claw links shaking at rest: mean worst-link angular speed {ang_mean:.3} rad/s \
         (want <{}; the claws are HELD by load-bearing rest contacts — pincer on \
         shoulder, shell on leg bases; collision-group changes that remove that \
         support make this 3-4x worse, rl#109)",
        QUIET_ANG_RADPS
    );
}

#[cfg(test)]
fn crab_angular_momentum(app: &mut App) -> Vec3 {
    use super::body::CrabBodyPart;
    use bevy_rapier3d::plugin::context::RapierRigidBodySet;
    use bevy_rapier3d::prelude::RapierRigidBodyHandle;

    let handles: Vec<bevy_rapier3d::rapier::dynamics::RigidBodyHandle> = {
        let mut q = app
            .world_mut()
            .query_filtered::<&RapierRigidBodyHandle, With<CrabBodyPart>>();
        q.iter(app.world()).map(|h| h.0).collect()
    };
    let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
    let set = set_q.single(app.world()).expect("rapier set");

    struct Part {
        m: f32,
        r: Vec3,
        v: Vec3,
        i_world: Mat3,
        w: Vec3,
    }
    let (mut m_tot, mut mr, mut mv) = (0.0f32, Vec3::ZERO, Vec3::ZERO);
    let mut parts = Vec::with_capacity(handles.len());
    for h in &handles {
        let rb = set.bodies.get(*h).expect("rapier body");
        let m = rb.mass();
        let r = rb.center_of_mass();
        let v = rb.linvel();
        let rmat = Mat3::from_quat(rb.position().rotation);
        let i_world = rmat
            * rb.mass_properties()
                .local_mprops
                .reconstruct_inertia_matrix()
            * rmat.transpose();
        m_tot += m;
        mr += m * r;
        mv += m * v;
        parts.push(Part {
            m,
            r,
            v,
            i_world,
            w: rb.angvel(),
        });
    }
    let (r_com, v_com) = (mr / m_tot, mv / m_tot);
    parts.iter().fold(Vec3::ZERO, |l, p| {
        l + p.i_world * p.w + p.m * (p.r - r_com).cross(p.v - v_com)
    })
}

/// Teleport env 0's crab to an airborne respawn at `y` — the contact-free harness
/// both momentum-conservation tests build on.
#[cfg(test)]
fn respawn_airborne(app: &mut App, y: f32) {
    use super::body::{CrabAssets, CrabBodyPart, CrabEnvId};
    use super::respawn_crab;
    use bevy::ecs::system::RunSystemOnce;

    app.world_mut()
        .run_system_once(
            move |mut commands: Commands,
                  assets: Res<CrabAssets>,
                  terrain: Res<crate::terrain::Terrain>,
                  parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>| {
                respawn_crab(
                    &mut commands,
                    &assets,
                    &terrain,
                    parts.iter().filter(|(_, id)| id.0 == 0).map(|(e, _)| e),
                    Vec3::new(0.0, y, 0.0),
                    0,
                );
            },
        )
        .expect("airborne respawn");
}

/// Touching contact-point count across the whole narrow phase this tick.
#[cfg(test)]
fn contact_points(app: &mut App) -> usize {
    use bevy_rapier3d::plugin::context::RapierContextSimulation;

    let mut q = app.world_mut().query::<&RapierContextSimulation>();
    let sim = q.single(app.world()).expect("sim");
    sim.narrow_phase
        .contact_pairs()
        .flat_map(|p| p.manifolds.iter())
        .flat_map(|m| m.points.iter())
        .filter(|pt| -pt.dist > 0.0)
        .count()
}

#[test]
fn airborne_crab_conserves_angular_momentum() {
    use bevy_rapier3d::prelude::RapierConfiguration;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut app = headless_app();
    tick(&mut app, 1);

    {
        let mut q = app.world_mut().query::<&mut RapierConfiguration>();
        for mut cfg in q.iter_mut(app.world_mut()) {
            cfg.gravity = Vec3::ZERO;
        }
    }
    respawn_airborne(&mut app, 80.0);
    tick(&mut app, 4);

    tick(&mut app, 1);

    let l0 = crab_angular_momentum(&mut app).length();
    let mut rng = StdRng::seed_from_u64(3);
    let mut action = [0.0f32; CrabJointId::COUNT];
    let mut peak = l0;
    let mut total_contacts = 0usize;
    for _ in 0..800u32 {
        for a in action.iter_mut() {
            *a = (*a + rng.gen_range(-0.02..0.02)).clamp(-1.0, 1.0);
        }
        assert!(
            app.world_mut()
                .resource_mut::<CrabActions>()
                .set_row(0, action)
        );
        tick(&mut app, 1);
        peak = peak.max(crab_angular_momentum(&mut app).length());
        total_contacts += contact_points(&mut app);
    }

    println!(
        "airborne crab: |L| start={l0:.4}  peak={peak:.4}  ratio={:.1}x  contacts={total_contacts}",
        peak / l0.max(1e-9)
    );
    assert_eq!(
        total_contacts, 0,
        "airborne window had {total_contacts} contact-points — not contact-free, \
         so the momentum check isn't isolating internal forces"
    );
    assert!(
        peak < 0.3,
        "airborne crab spun ITSELF up: |L| grew from {l0:.4} to {peak:.4} with zero \
         contacts and no external torque — angular momentum is being injected by the \
         joint constraint solver (issue #17)"
    );
}

/// The spawned carapace's pre-tick velocity and drag coefficient, read together —
/// the same components `aero::apply_air_drag` reads this tick — so a test's drag
/// bookkeeping reproduces the applied force bit-for-bit.
#[cfg(test)]
fn carapace_vel_and_drag(app: &mut App) -> (Vec3, f32) {
    let mut q = app.world_mut().query_filtered::<(
        &bevy_rapier3d::prelude::Velocity,
        &super::aero::CarapaceDrag,
    ), bevy::prelude::With<super::body::CrabCarapace>>();
    let (v, d) = q.single(app.world()).expect("one carapace");
    (v.linear, d.coeff())
}

/// Σ m·v over every crab body plus the total mass, from the rapier set (ground
/// truth, not the bevy mirror).
#[cfg(test)]
fn crab_linear_momentum(app: &mut App) -> (Vec3, f32) {
    use super::body::CrabBodyPart;
    use bevy_rapier3d::plugin::context::RapierRigidBodySet;
    use bevy_rapier3d::prelude::RapierRigidBodyHandle;

    let handles: Vec<bevy_rapier3d::rapier::dynamics::RigidBodyHandle> = {
        let mut q = app
            .world_mut()
            .query_filtered::<&RapierRigidBodyHandle, With<CrabBodyPart>>();
        q.iter(app.world()).map(|h| h.0).collect()
    };
    let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
    let set = set_q.single(app.world()).expect("rapier set");

    let (mut p, mut m_tot) = (Vec3::ZERO, 0.0f32);
    for h in &handles {
        let rb = set.bodies.get(*h).expect("rapier body");
        let m = rb.mass();
        let v = rb.linvel();
        m_tot += m;
        p += m * v;
    }
    (p, m_tot)
}

#[derive(Clone, Copy, Debug)]
enum Thrash {
    /// Per-channel full-amplitude sinusoids at `freqs`/`phases`.
    Sinusoid,
    /// ±1 flipping every 5 ticks. Ignores `freqs` and repurposes each phase's SIGN
    /// as the channel's fixed polarity — sound only because the phase draw in
    /// `airborne_thrash_residual` is sign-symmetric over (−π, π).
    Squarewave,
}

/// The three drives every rl#321 momentum gate runs — one list so the strict
/// contract, the live ceiling, and the dated baselines in their docs can't drift
/// onto different drives.
#[cfg(test)]
const MOMENTUM_GATE_DRIVES: [(Thrash, u64); 3] = [
    (Thrash::Sinusoid, 11),
    (Thrash::Sinusoid, 12),
    (Thrash::Squarewave, 13),
];

/// One full-amplitude thrash row per tick.
#[cfg(test)]
fn thrash_row(
    kind: Thrash,
    t: u32,
    freqs: &[f32],
    phases: &[f32],
) -> [f32; super::actuator::ACTION_SIZE] {
    let secs = t as f32 * crate::physics::PHYSICS_DT;
    let mut row = [0.0f32; super::actuator::ACTION_SIZE];
    for (j, a) in row.iter_mut().enumerate() {
        *a = match kind {
            Thrash::Sinusoid => (std::f32::consts::TAU * freqs[j] * secs + phases[j]).sin(),
            Thrash::Squarewave => {
                let flip = if (t / 5).is_multiple_of(2) { 1.0 } else { -1.0 };
                flip * phases[j].signum()
            }
        };
    }
    row
}

/// The airborne spawn height every rl#321 gate uses. Over the 256-tick window the
/// crab falls ~78 m from here (v_end ≈ 39 m/s), staying far above the flat grid —
/// the strict bound's v_max derivation and the "never lands" argument both key off
/// this constant.
#[cfg(test)]
const MOMENTUM_SPAWN_Y: f32 = 400.0;

/// 4 s of full-amplitude airborne thrash (gravity ON), returning the COM momentum
/// residual as an equivalent acceleration — |Δp − m·g·Δt| / (m·Δt) — plus the
/// touching contact-point total across the window, asserted zero: airborne and with
/// no crab–crab pairs the window exercises motors + joints alone.
/// Deterministic: seeded drive, fixed-dt physics.
#[cfg(test)]
fn airborne_thrash_residual(kind: Thrash, seed: u64) -> (Vec3, usize) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const TICKS: u32 = 256;

    let mut app = flat_headless_app();
    tick(&mut app, 1);
    respawn_airborne(&mut app, MOMENTUM_SPAWN_Y);
    tick(&mut app, 4);
    tick(&mut app, 1);

    let mut rng = StdRng::seed_from_u64(seed);
    let n = super::actuator::ACTION_SIZE;
    let freqs: Vec<f32> = (0..n).map(|_| rng.gen_range(1.0..4.0)).collect();
    let phases: Vec<f32> = (0..n)
        .map(|_| rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI))
        .collect();

    let part_ids = |app: &mut App| -> std::collections::BTreeSet<Entity> {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<super::body::CrabBodyPart>>();
        q.iter(app.world()).collect()
    };
    let ids0 = part_ids(&mut app);

    let (p0, m) = crab_linear_momentum(&mut app);
    let mut total_contacts = 0usize;
    let mut j_drag = Vec3::ZERO;
    for t in 0..TICKS {
        assert!(
            app.world_mut()
                .resource_mut::<CrabActions>()
                .set_row(0, thrash_row(kind, t, &freqs, &phases))
        );
        // The rl#332 carapace drag is a modeled EXTERNAL force, so the momentum
        // books carry it explicitly. Read the same `Velocity` component the drag
        // system reads this tick (nothing writes it between here and the system),
        // so the subtraction reproduces the applied force bit-for-bit.
        {
            let (v, c) = carapace_vel_and_drag(&mut app);
            j_drag += -(c * v.length()) * v * crate::physics::PHYSICS_DT;
        }
        tick(&mut app, 1);
        total_contacts += contact_points(&mut app);
    }
    let (p1, _) = crab_linear_momentum(&mut app);
    assert_eq!(
        ids0,
        part_ids(&mut app),
        "the crab's part entities changed mid-window — a rescue respawned her, so \
         the momentum window mixes two bodies"
    );
    assert_eq!(
        total_contacts, 0,
        "airborne window had {total_contacts} contact-points — not contact-free, so \
         the momentum check isn't isolating internal forces"
    );

    let dt_total = TICKS as f32 * crate::physics::PHYSICS_DT;
    let g: Vec3 = crate::physics::PHYSICS_GRAVITY;
    let resid = (p1 - p0 - m * g * dt_total - j_drag) / (m * dt_total);
    println!(
        "airborne thrash ({kind:?} seed {seed}): m={m:.3} kg  \
         resid={:.5} m/s² ({:.3}% of g)  dir={:?}  contacts={total_contacts}",
        resid.length(),
        100.0 * resid.length() / g.length(),
        resid,
    );
    (resid, total_contacts)
}

/// bddap/rl#321 — THE physics contract: joint motors are INTERNAL forces, so an
/// airborne, contact-free crab can reorient itself but never translate its COM
/// beyond gravity: Δp = m·g·Δt over any window. A residual means an un-modeled
/// EXTERNAL force — the self-propulsion seen in GCR. Gravity stays ON
/// (unlike the angular twin): a velocity-proportional leak (world-frame damping) is
/// invisible at v ≈ 0 and grows with the free-fall speed.
///
/// The bound is principled, not tuned-to-pass: f32 rounding on the velocity
/// integration accumulates ≲ ε_f32·v_max·TICKS/2 ≈ 6e-8·39·128 ≈ 3e-4 m/s over the
/// window → ~7e-5 m/s²; 1e-2 m/s² (0.1% of g) sits two orders above that floor and
/// two below the ≥0.1 m/s² scale of visible self-propulsion.
///
/// Live since the 2026-07-28 solver fix in bddap-bot/rapier (momentum-exact
/// multibody substeps: per-substep base-momentum ledger, stabilization solves
/// re-derived against the current mass matrix, free-joint quaternion
/// renormalization). Post-fix the gate drives measure 0.00042 / 0.00001 /
/// 0.00034 m/s² — 24× under this bound (pre-fix: 0.159–0.574).
#[test]
fn airborne_crab_conserves_linear_momentum() {
    for (kind, seed) in MOMENTUM_GATE_DRIVES {
        let (resid, _) = airborne_thrash_residual(kind, seed);
        assert!(
            resid.length() < 1e-2,
            "phantom COM force ({kind:?} seed {seed}): |Δp − m·g·Δt| ≡ {:.4} m/s² \
             (direction {:?}) with zero contacts — an un-modeled EXTERNAL force is \
             acting on the crab (bddap/rl#321)",
            resid.length(),
            resid.normalize(),
        );
    }
}

/// bddap/rl#321's coarse ceiling, kept alongside the strict contract as the
/// far-backstop with slack for solver-tuning drift (the strict 1e-2 bound is
/// the primary gate; this one names the historical scale). Pre-fix the leak
/// measured 0.159 / 0.419 / 0.574 m/s² on [`MOMENTUM_GATE_DRIVES`]; post-fix
/// (2026-07-28 solver fix) ≤ 0.0005. The 0.05 ceiling sits 100× above today's
/// reality and 3× under the old best case, so it trips on any reappearance of
/// the leak class even if the strict bound is later loosened.
#[test]
fn airborne_contact_free_thrash_stays_below_known_leak() {
    for (kind, seed) in MOMENTUM_GATE_DRIVES {
        let (resid, _) = airborne_thrash_residual(kind, seed);
        assert!(
            resid.length() < 0.05,
            "phantom COM force is back at pre-fix scale ({kind:?} seed \
             {seed}): {:.4} m/s² vs the ≤0.0005 post-fix baseline — the \
             bddap/rl#321 momentum leak reappeared (limit softness? new \
             external-force path? solver regression?)",
            resid.length(),
        );
    }
}

/// NOT a regression test — a measurement INSTRUMENT (rl#20 stage 2/3), `#[ignore]`d
/// so suites never gate on it. Open-loop square-wave flail (the net ship-wiggle
/// drive: every channel ±1, period 10 ticks) — a policy-FREE gait whose pace
/// depends only on the body's mechanics, so running it on two baked tables
/// separates "this body is mechanically degraded" (the rl#277 fear) from "the old
/// policy is overfit to its contact geometry" (retrain territory). Prints
/// SCRIPTED_FLAIL with the pace in m/s and body-heights/s.
#[test]
#[ignore = "rl#20 measurement instrument — run explicitly with --ignored --nocapture"]
fn scripted_flail_gait_pace() {
    use super::body::CrabCarapace;

    let mut app = headless_app();
    tick(&mut app, 300);
    let carapace_xz = |app: &mut App| {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        let t = q.single(app.world()).expect("one carapace").translation;
        Vec2::new(t.x, t.z)
    };
    let start = carapace_xz(&mut app);
    let ticks = 640u32;
    for t in 0..ticks {
        let w = if (t / 5) % 2 == 0 { 1.0 } else { -1.0 };
        let _ = app.world_mut().resource_mut::<CrabActions>().fill(0, w);
        tick(&mut app, 1);
    }
    let dist = (carapace_xz(&mut app) - start).length();
    let secs = ticks as f32 / 64.0;
    let height = crate::bot::rig::natural_body_height().unwrap_or(f32::NAN);
    println!(
        "SCRIPTED_FLAIL dist={dist:.3} m over {secs:.1} s -> {:.4} m/s = {:.4} heights/s (h={height:.4})",
        dist / secs,
        dist / secs / height,
    );
}

/// bddap/rl#332 — the plant has air: an unactuated falling crab approaches the
/// carapace-drag terminal velocity instead of integrating gravity unboundedly.
/// Pre-drag this fall measured ~39 m/s at the window's end; with the coefficient
/// derived from the spawned body's own mass ([`super::aero::CarapaceDrag`],
/// v_t = [`super::aero::TERMINAL_SPEED`]) the same window must land in a band
/// around 15 m/s for mesh and fallback bodies alike (rl#340 stage 3: the old
/// fallback-sized CONSTANT let the 1.97 kg mesh body fall at 22 m/s). The band is
/// the drift alarm on BOTH sides: above 18 m/s the drag went missing or the
/// spawn-time mass sum diverged from what rapier integrates; below 11 m/s
/// something over-damps her and the trained gait is next.
#[test]
fn airborne_crab_reaches_terminal_velocity() {
    const TICKS: u32 = 256; // 4 s — ~2.6 drag time-constants past v_t

    let mut app = flat_headless_app();
    tick(&mut app, 1);
    respawn_airborne(&mut app, MOMENTUM_SPAWN_Y);
    tick(&mut app, 4);
    tick(&mut app, TICKS);

    let (p, m) = crab_linear_momentum(&mut app);
    let speed = (p / m).length();
    println!("terminal-velocity fall: |v_com|={speed:.2} m/s after {TICKS} ticks (m={m:.3} kg)");
    assert!(
        (11.0..18.0).contains(&speed),
        "free-fall speed {speed:.2} m/s is outside the 11–18 m/s terminal band — \
         carapace drag (bddap/rl#332) is mis-scaled for the current body mass, \
         missing, or doubled; unbounded speed is how Sally flies"
    );
}

/// bddap/rl#339 — the carapace drag brake must stay a brake at ANY speed. Explicit
/// integration of `F = −c·|v|·v` is a brake only below `|v| = m·Hz/c`; past it the
/// drag tick overshoots zero and amplifies geometrically (measured on the full crab:
/// 6000 → 788,000 m/s in ONE tick), which is what stretched a solver fling into the
/// rl#339 wedge flights. With the drag FORCE capped
/// ([`super::aero::CARAPACE_BRAKE_WEIGHT_MAX`]) the per-tick Δv is ~constant and
/// tiny, so hypersonic speed decays monotonically and stays finite.
///
/// A LONE carapace-tagged rigid body, deliberately: the integrator's stability is a
/// single-body property, and the full multibody masks it — a violent fling excites
/// the (open, rl#349) solver energy-injection class, which amplifies the COM with
/// or without drag. Collision-free and not a `CrabBodyPart`, so neither contacts
/// nor rescues touch the measurement.
#[test]
fn hypersonic_carapace_brake_decays() {
    use super::body::CrabCarapace;
    use bevy_rapier3d::prelude::{
        Collider, ColliderMassProperties, CollisionGroups, ExternalForce, Group,
        ReadMassProperties, RigidBody, Velocity,
    };

    let mut app = flat_headless_app();
    tick(&mut app, 1);
    // Arm the drag at spawn the way `spawn_crab` arms a crab: coefficient sized
    // off this body's own collider mass (same rapier mass_properties source), so
    // its terminal velocity is the design TERMINAL_SPEED — and the undragged-
    // carapace sentinel in `apply_air_drag` never sees it bare.
    let collider = Collider::cuboid(0.15, 0.06, 0.11);
    let drag = super::aero::CarapaceDrag::for_total_mass(collider.raw.mass_properties(50.0).mass());
    let body = app
        .world_mut()
        .spawn((
            CrabCarapace,
            RigidBody::Dynamic,
            collider,
            ColliderMassProperties::Density(50.0),
            CollisionGroups::new(Group::NONE, Group::NONE),
            ReadMassProperties::default(),
            Velocity {
                linear: Vec3::Y * 6000.0,
                ..default()
            },
            ExternalForce::default(),
            Transform::from_xyz(0.0, 400.0, 0.0),
            drag,
        ))
        .id();
    // One settle tick: rapier ingests the body and writes back its mass mirror (the
    // drag cap reads it; it is zero until the first writeback).
    tick(&mut app, 1);

    let speed = |app: &mut App| -> f32 {
        app.world()
            .get::<Velocity>(body)
            .expect("test body")
            .linear
            .length()
    };
    let v1 = speed(&mut app);
    assert!(v1 > 5000.0, "launch speed not held: |v|={v1:.0} m/s");

    // Per-tick, because the uncapped amplifier peaks (and dies to NaN) within a few
    // ticks of crossing the stability line. 5 s of ~20 g braking sheds ~1000 m/s.
    const TICKS: u32 = 320;
    let mut prev = v1;
    for t in 0..TICKS {
        // `apply_actions` zeroes ExternalForce for CRAB parts each tick; this lone
        // body isn't one, so zero it here or the += drag would accumulate forever.
        app.world_mut()
            .get_mut::<ExternalForce>(body)
            .expect("test body")
            .force = Vec3::ZERO;
        tick(&mut app, 1);
        let v = app.world().get::<Velocity>(body).expect("test body").linear;
        assert!(
            v.is_finite(),
            "velocity went non-finite {t} ticks in — the drag brake is amplifying \
             again (rl#339)"
        );
        let s = v.length();
        assert!(
            s < prev + 0.5,
            "|v| grew {prev:.0} -> {s:.0} m/s at tick {t} under drag+gravity alone — \
             the brake is adding energy (rl#339)"
        );
        prev = s;
    }
    println!("hypersonic brake: |v| {v1:.0} -> {prev:.0} m/s over {TICKS} ticks");
    assert!(
        prev < v1 - 800.0,
        "a {v1:.0} m/s carapace shed only {:.0} m/s in 4 s — the force-capped brake \
         is missing or mis-scaled (rl#339)",
        v1 - prev
    );
}

/// rl#392's structural pin. The solver runs two regimes split by drive state:
/// awake ticks pay only the cheap global [`crate::physics::SOLVER_ITERATIONS`]
/// (the frame budget), and a ZERO-DRIVE crab must actually leave the solver —
/// wake-loop-free force writers (actuator + carapace drag write-skips), the
/// noise-floor sleep gates, and the settle iteration elevation together put
/// every body of the multibody to sleep, which zeroes the velocities the
/// chatter tests above measure. If this fails, rest quiet is being bought by
/// awake solving again — the 2 fps regression's shape (rl#392). The driven
/// interlude makes the pin cover the wake → re-settle → re-sleep path, not
/// just the spawn settle.
#[test]
fn resting_crab_falls_asleep() {
    use bevy_rapier3d::plugin::context::RapierRigidBodySet;

    fn count(app: &mut App) -> (usize, usize) {
        let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
        let set = set_q.single(app.world()).unwrap();
        let (mut asleep, mut awake) = (0, 0);
        for (_, rb) in set.bodies.iter() {
            if !rb.is_dynamic() {
                continue;
            }
            if rb.is_sleeping() {
                asleep += 1;
            } else {
                awake += 1;
            }
        }
        (asleep, awake)
    }

    let mut app = flat_headless_app();
    tick(&mut app, 1);
    assert!(app.world_mut().resource_mut::<CrabActions>().fill(0, 0.6));
    tick(&mut app, 64);
    let (_, awake_driven) = count(&mut app);
    assert!(
        awake_driven > 0,
        "a fully driven crab has no awake bodies — the actuator's torque writes \
         stopped force-waking, so drives no longer reach the solver"
    );

    assert!(app.world_mut().resource_mut::<CrabActions>().fill(0, 0.0));
    settle_to_sleep(&mut app, 512);
    let (asleep, awake) = count(&mut app);
    assert!(
        awake == 0 && asleep > 10,
        "zero-drive crab still has {awake} awake bodies ({asleep} asleep) 8 s \
         after the drives went quiet — sleep is not engaging, so rest is being \
         paid for with awake solver ticks (and the chatter tests are measuring \
         live solver noise instead of sleep's exact zeros)"
    );

    let poses: Vec<_> = {
        let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
        let set = set_q.single(app.world()).unwrap();
        set.bodies
            .iter()
            .filter(|(_, rb)| rb.is_dynamic())
            .map(|(_, rb)| *rb.position())
            .collect()
    };
    tick(&mut app, 64);
    let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
    let set = set_q.single(app.world()).unwrap();
    for (i, (_, rb)) in set
        .bodies
        .iter()
        .filter(|(_, rb)| rb.is_dynamic())
        .enumerate()
    {
        assert_eq!(
            rb.position().translation,
            poses[i].translation,
            "a sleeping crab body moved — sleep is not bit-exact rest"
        );
    }
}
