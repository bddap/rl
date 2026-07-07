
use bevy::prelude::*;

use super::actuator::CrabActions;
use super::body::{CrabJoint, CrabJointId, Side, joint_angle};
use super::headless::{assert_transforms_match_rapier, headless_app, tick};

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
                actions.envs[0][CrabJointId::LegMerus(side, leg).index()] = torque;
                actions.envs[0][CrabJointId::LegCarpus(side, leg).index()] = torque;
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
        let mut actions = app.world_mut().resource_mut::<CrabActions>();
        for v in actions.envs[0].iter_mut() {
            *v = 1.0;
        }
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

    let mut app = headless_app();
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
fn actuator_injects_no_net_wrench() {
    use super::actuator::CrabActions;
    use super::body::CrabBodyPart;
    use bevy_rapier3d::prelude::ExternalForce;

    use std::collections::HashMap;

    let mut app = headless_app();
    tick(&mut app, 1);
    {
        let mut actions = app.world_mut().resource_mut::<CrabActions>();
        for v in actions.envs[0].iter_mut() {
            *v = 1.0;
        }
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
    tick(&mut app, 1);

    let mut net_force = Vec3::ZERO;
    let mut net_torque = Vec3::ZERO;
    let mut q = app
        .world_mut()
        .query_filtered::<(Entity, &ExternalForce), With<CrabBodyPart>>();
    for (e, ef) in q.iter(app.world()) {
        net_force += ef.force;
        net_torque += pos[&e].cross(ef.force) + ef.torque;
    }
    println!(
        "actuator net force {:.5} N, net torque {:.5} N·m",
        net_force.length(),
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

    if super::meshfit::model_path().is_none() {
        eprintln!("crab_settles_quietly_at_rest: no model — skipping (fallback body)");
        return;
    }

    let mut app = headless_app();
    tick(&mut app, 1);

    tick(&mut app, 320);
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
        ang_mean < super::collider_check::QUIET_ANG_RADPS,
        "carapace still twitching at rest: angular speed mean {ang_mean:.3} rad/s \
         (want <{}; 12 Hz contact sits ~0.61, the 30 Hz / substeps=1 regressions ~1.5)",
        super::collider_check::QUIET_ANG_RADPS
    );
    assert!(
        bounce < 0.024,
        "carapace bouncing at rest: {bounce:.4} m peak-to-peak (want <0.024 at the 0.04 \
         floppy cap; the 30 Hz contact regression ~0.036, substeps=1 ~0.030)"
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

#[test]
fn claws_quiet_at_rest() {
    use bevy_rapier3d::prelude::Velocity;

    if super::meshfit::model_path().is_none() {
        eprintln!("claws_quiet_at_rest: no model — skipping (fallback body)");
        return;
    }

    let mut app = headless_app();
    tick(&mut app, 1);
    tick(&mut app, 320);

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
        lin_mean < super::collider_check::QUIET_LIN_MPS,
        "claw links shaking at rest: mean worst-link linear speed {lin_mean:.3} m/s \
         (want <{}) — the contact spring regressed stiffer",
        super::collider_check::QUIET_LIN_MPS
    );
    assert!(
        ang_mean < super::collider_check::QUIET_ANG_RADPS,
        "claw links shaking at rest: mean worst-link angular speed {ang_mean:.3} rad/s \
         (want <{}; the claws are HELD by load-bearing rest contacts — pincer on \
         shoulder, shell on leg bases; collision-group changes that remove that \
         support make this 3-4x worse, rl#109)",
        super::collider_check::QUIET_ANG_RADPS
    );
}

#[test]
fn fallback_body_settles_without_blowing_up() {
    use super::body::{CrabBodyPart, CrabCarapace};
    use bevy_rapier3d::prelude::Velocity;

    if super::meshfit::model_path().is_some() {
        eprintln!(
            "fallback_body_settles_without_blowing_up: model present — skipping (not the fallback body)"
        );
        return;
    }

    let mut app = headless_app();
    tick(&mut app, 1);
    tick(&mut app, 320);

    let mut parts_q = app
        .world_mut()
        .query_filtered::<(&Transform, &Velocity), With<CrabBodyPart>>();
    let mut n = 0;
    for (t, v) in parts_q.iter(app.world()) {
        assert!(
            t.translation.is_finite() && t.rotation.is_finite(),
            "fallback part pose went non-finite at rest: {t:?}"
        );
        assert!(
            v.linear.is_finite() && v.angular.is_finite(),
            "fallback part velocity went non-finite at rest: {v:?}"
        );
        assert!(
            v.linear.length() < 5.0 && v.angular.length() < 50.0,
            "fallback part still moving fast at rest: lin {:.2} m/s ang {:.2} rad/s",
            v.linear.length(),
            v.angular.length()
        );
        n += 1;
    }
    assert!(n >= 30, "fallback crab failed to spawn its parts (got {n})");

    let mut car_q = app
        .world_mut()
        .query_filtered::<&Transform, With<CrabCarapace>>();
    let car_y = car_q
        .iter(app.world())
        .next()
        .expect("carapace")
        .translation
        .y;
    assert!(
        (0.0..2.0).contains(&car_y),
        "fallback carapace at y={car_y:.2} — sank through the floor or launched"
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

#[test]
fn airborne_crab_conserves_angular_momentum() {
    use super::body::{CrabAssets, CrabBodyPart, CrabEnvId};
    use super::respawn_crab;
    use bevy::ecs::system::RunSystemOnce;
    use bevy_rapier3d::plugin::context::RapierContextSimulation;
    use bevy_rapier3d::prelude::{CollisionGroups, Group, RapierConfiguration};
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
    app.world_mut()
        .run_system_once(
            |mut commands: Commands,
             assets: Res<CrabAssets>,
             parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>| {
                respawn_crab(
                    &mut commands,
                    &assets,
                    parts.iter().filter(|(_, id)| id.0 == 0).map(|(e, _)| e),
                    Vec3::new(0.0, 80.0, 0.0),
                    0,
                );
            },
        )
        .expect("airborne respawn");
    tick(&mut app, 4);

    {
        let mut q = app
            .world_mut()
            .query_filtered::<&mut CollisionGroups, With<CrabBodyPart>>();
        for mut g in q.iter_mut(app.world_mut()) {
            g.filters = Group::NONE;
        }
    }
    tick(&mut app, 1);

    let contacts = |app: &mut App| -> usize {
        let mut q = app.world_mut().query::<&RapierContextSimulation>();
        let sim = q.single(app.world()).expect("sim");
        sim.narrow_phase
            .contact_pairs()
            .flat_map(|p| p.manifolds.iter())
            .flat_map(|m| m.points.iter())
            .filter(|pt| -pt.dist > 0.0)
            .count()
    };

    let l0 = crab_angular_momentum(&mut app).length();
    let mut rng = StdRng::seed_from_u64(3);
    let mut action = [0.0f32; CrabJointId::COUNT];
    let mut peak = l0;
    let mut total_contacts = 0usize;
    for _ in 0..800u32 {
        for (i, a) in action.iter_mut().enumerate() {
            *a = (*a + rng.gen_range(-0.02..0.02)).clamp(-1.0, 1.0);
            app.world_mut().resource_mut::<CrabActions>().envs[0][i] = *a;
        }
        tick(&mut app, 1);
        peak = peak.max(crab_angular_momentum(&mut app).length());
        total_contacts += contacts(&mut app);
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
