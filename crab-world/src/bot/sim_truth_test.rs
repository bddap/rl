use bevy::prelude::*;

use super::actuator::CrabActions;
use super::body::{CrabJoint, CrabJointId, Side, joint_angle};
use super::headless::{assert_transforms_match_rapier, flat_headless_app, headless_app, tick};

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
        let mut q = app
            .world_mut()
            .query_filtered::<&bevy_rapier3d::prelude::Velocity, With<super::body::CrabCarapace>>();
        let v = q.single(app.world()).expect("one carapace").linear;
        -(super::aero::CARAPACE_DRAG * v.length()) * v
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

    // Keyed on the VERDICT, not path presence: a present-but-digest-mismatched glb
    // constructs the FALLBACK body (rl#20 Phase 2), and these are Sally assertions.
    if crate::mesh_fallback::usable_model().is_err() {
        eprintln!("crab_settles_quietly_at_rest: no usable model — skipping (fallback body)");
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

    if crate::mesh_fallback::usable_model().is_err() {
        eprintln!("claws_quiet_at_rest: no usable model — skipping (fallback body)");
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

    if crate::mesh_fallback::usable_model().is_ok() {
        eprintln!(
            "fallback_body_settles_without_blowing_up: usable model present — skipping (not the fallback body)"
        );
        return;
    }

    let mut app = flat_headless_app();
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

/// Kill every crab-part collision filter so the airborne window is contact-free by
/// construction — the momentum checks isolate motors + joints, not contact response.
#[cfg(test)]
fn disable_crab_collisions(app: &mut App) {
    use super::body::CrabBodyPart;
    use bevy_rapier3d::prelude::{CollisionGroups, Group};

    let mut q = app
        .world_mut()
        .query_filtered::<&mut CollisionGroups, With<CrabBodyPart>>();
    for mut g in q.iter_mut(app.world_mut()) {
        g.filters = Group::NONE;
    }
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

    disable_crab_collisions(&mut app);
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
/// touching contact-point total across the window. `isolate` kills the collision
/// filters first, so the window exercises motors + joints alone (asserted
/// contact-free — the harness-validity precondition, not the property under test);
/// without it the production groups stay and the limbs beat against each other.
/// Deterministic: seeded drive, fixed-dt physics.
#[cfg(test)]
fn airborne_thrash_residual(kind: Thrash, seed: u64, isolate: bool) -> (Vec3, usize) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const TICKS: u32 = 256;

    let mut app = flat_headless_app();
    tick(&mut app, 1);
    respawn_airborne(&mut app, MOMENTUM_SPAWN_Y);
    tick(&mut app, 4);
    if isolate {
        disable_crab_collisions(&mut app);
    }
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
            let mut q = app
                .world_mut()
                .query_filtered::<&bevy_rapier3d::prelude::Velocity, With<super::body::CrabCarapace>>();
            let v = q.single(app.world()).expect("one carapace").linear;
            j_drag += -(super::aero::CARAPACE_DRAG * v.length()) * v * crate::physics::PHYSICS_DT;
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
    if isolate {
        assert_eq!(
            total_contacts, 0,
            "isolated airborne window had {total_contacts} contact-points — not \
             contact-free, so the momentum check isn't isolating internal forces"
        );
    }

    let dt_total = TICKS as f32 * crate::physics::PHYSICS_DT;
    let g: Vec3 = crate::physics::PHYSICS_GRAVITY;
    let resid = (p1 - p0 - m * g * dt_total - j_drag) / (m * dt_total);
    println!(
        "airborne thrash ({kind:?} seed {seed} isolate={isolate}): m={m:.3} kg  \
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
/// EXTERNAL force — the self-propulsion the owner observed in GCR. Gravity stays ON
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
        let (resid, _) = airborne_thrash_residual(kind, seed, true);
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
        let (resid, _) = airborne_thrash_residual(kind, seed, true);
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

/// The self-collision half of bddap/rl#321: crab-part↔crab-part contacts are
/// also internal, so momentum must still follow gravity when the airborne thrash
/// runs WITH the production collision groups and the limbs beat against each
/// other. Pre-fix this measured 0.64 m/s²; after the 2026-07-28 solver fix it
/// measures 0.037 (contact impulses go through per-substep-fresh mass matrices,
/// but contact-pair rounding under hundreds of contact points keeps it above the
/// contact-free floor). The 0.5 ceiling is 13× above today's reality and under
/// the pre-fix baseline — it catches a contact-solve change that starts
/// injecting momentum wholesale.
#[test]
fn airborne_self_contact_thrash_stays_below_known_leak() {
    let (resid, contacts) = airborne_thrash_residual(Thrash::Sinusoid, 21, false);
    assert!(
        contacts > 0,
        "the thrash never produced a self-contact — this variant isn't exercising \
         contact resolution; make the drive more violent"
    );
    assert!(
        resid.length() < 0.5,
        "phantom COM force UNDER SELF-CONTACT is back at pre-fix scale \
         (bddap/rl#321): {:.4} m/s² (post-fix baseline 0.037) across {contacts} \
         contact-points — self-collision resolution injects momentum",
        resid.length(),
    );
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
    let height = crate::mesh_fallback::natural_body_height().unwrap_or(f32::NAN);
    println!(
        "SCRIPTED_FLAIL dist={dist:.3} m over {secs:.1} s -> {:.4} m/s = {:.4} heights/s (h={height:.4})",
        dist / secs,
        dist / secs / height,
    );
}

/// NOT a regression test — the bddap/rl#321 diagnosis instrument (the
/// `scripted_flail_gait_pace` pattern), `#[ignore]`d so suites never gate on it.
/// Residual accel across a mechanism matrix — thrash amplitude, substeps, solver
/// iterations, and a lone constant root torque — separating "joint-limit impulse
/// leak" from "convergence error" (2026-07-28 findings: substeps/iterations ×4
/// change nothing; amplitude 0.2 drops it 16×; a lone 5 N·m root torque leaks only
/// 0.008 m/s²; removing joint limits drops the full-thrash leak 6.5×).
#[test]
#[ignore = "rl#321 diagnosis instrument"]
fn phantom_force_instrument() {
    use bevy_rapier3d::plugin::context::RapierContextSimulation;
    use bevy_rapier3d::prelude::TimestepMode;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let run = |amp: f32, substeps: usize, iters: usize, root_torque: f32| -> f32 {
        use bevy_rapier3d::plugin::PhysicsSet;
        use bevy_rapier3d::prelude::ExternalForce;

        use super::BotSet;
        use super::body::CrabCarapace;

        let mut app = flat_headless_app();
        app.insert_resource(TimestepMode::Fixed {
            dt: crate::physics::PHYSICS_DT,
            substeps,
        });
        if root_torque != 0.0 {
            app.add_systems(
                FixedUpdate,
                (move |mut q: Query<&mut ExternalForce, With<CrabCarapace>>| {
                    for mut f in q.iter_mut() {
                        f.torque += Vec3::X * root_torque;
                    }
                })
                .after(BotSet::Act)
                .before(PhysicsSet::SyncBackend),
            );
        }
        tick(&mut app, 1);
        {
            let mut q = app.world_mut().query::<&mut RapierContextSimulation>();
            for mut sim in q.iter_mut(app.world_mut()) {
                sim.integration_parameters.num_solver_iterations = iters;
            }
        }
        respawn_airborne(&mut app, MOMENTUM_SPAWN_Y);
        tick(&mut app, 4);
        disable_crab_collisions(&mut app);
        tick(&mut app, 1);

        let mut rng = StdRng::seed_from_u64(11);
        let n = super::actuator::ACTION_SIZE;
        let freqs: Vec<f32> = (0..n).map(|_| rng.gen_range(1.0..4.0)).collect();
        let phases: Vec<f32> = (0..n)
            .map(|_| rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI))
            .collect();

        let (p0, m) = crab_linear_momentum(&mut app);
        let mut j_drag = Vec3::ZERO;
        for t in 0..256u32 {
            let mut row = thrash_row(Thrash::Sinusoid, t, &freqs, &phases);
            for a in row.iter_mut() {
                *a *= amp;
            }
            assert!(
                app.world_mut()
                    .resource_mut::<CrabActions>()
                    .set_row(0, row)
            );
            // Same drag-impulse bookkeeping as `airborne_thrash_residual`.
            {
                let mut q = app
                    .world_mut()
                    .query_filtered::<&bevy_rapier3d::prelude::Velocity, With<super::body::CrabCarapace>>();
                let v = q.single(app.world()).expect("one carapace").linear;
                j_drag +=
                    -(super::aero::CARAPACE_DRAG * v.length()) * v * crate::physics::PHYSICS_DT;
            }
            tick(&mut app, 1);
        }
        let (p1, _) = crab_linear_momentum(&mut app);
        let dt_total = 256.0 * crate::physics::PHYSICS_DT;
        let g: Vec3 = crate::physics::PHYSICS_GRAVITY;
        ((p1 - p0 - m * g * dt_total - j_drag) / (m * dt_total)).length()
    };

    for (label, amp, substeps, iters, root_tq) in [
        (
            "baseline            amp=1.0 sub=2 it=4",
            1.0,
            2usize,
            4usize,
            0.0,
        ),
        ("amp=0.5             ", 0.5, 2, 4, 0.0),
        ("amp=0.2             ", 0.2, 2, 4, 0.0),
        ("root-torque-only 1Nm", 0.0, 2, 4, 1.0),
        ("root-torque-only 5Nm", 0.0, 2, 4, 5.0),
        ("root-tq 5Nm sub=8   ", 0.0, 8, 4, 5.0),
    ] {
        let r = run(amp, substeps, iters, root_tq);
        println!("INSTRUMENT {label}: resid={r:.5} m/s²");
    }
}

/// Total mechanical energy of the crab from the rapier set —
/// Σ (½m|v|² + ½ω·I·ω + m·g·y). Passive bodies (zero drive) can only lose it;
/// growth is solver-injected.
#[cfg(test)]
fn crab_mechanical_energy(app: &mut App) -> f32 {
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

    let g = -crate::physics::PHYSICS_GRAVITY.y;
    handles
        .iter()
        .map(|h| {
            let rb = set.bodies.get(*h).expect("rapier body");
            let m = rb.mass();
            let v: Vec3 = rb.linvel();
            let w: Vec3 = rb.angvel();
            let rmat = Mat3::from_quat(rb.position().rotation);
            let i_world = rmat
                * rb.mass_properties()
                    .local_mprops
                    .reconstruct_inertia_matrix()
                * rmat.transpose();
            0.5 * m * v.length_squared() + 0.5 * w.dot(i_world * w) + m * g * rb.center_of_mass().y
        })
        .sum()
}

/// NOT a regression test — the bddap/rl#332 root-cause instrument, `#[ignore]`d.
/// Energize the crab with 300 ticks of grounded full-amplitude thrash (the
/// sally-soak storm regime), then ZERO all drives and watch total mechanical
/// energy over a passive window. A passive multibody under gravity + friction +
/// contacts can only dissipate; any sustained energy GROWTH is injected by the
/// solver. Ablation variants isolate the term: joint limits off, friction
/// motors off, both off.
#[test]
#[ignore = "rl#332 root-cause instrument — run explicitly with --ignored --nocapture"]
fn passive_storm_energy_instrument() {
    use bevy_rapier3d::prelude::MultibodyJoint;
    use bevy_rapier3d::rapier::dynamics::JointAxesMask;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const ENERGIZE: u32 = 300;
    const PASSIVE: u32 = 640;

    let run = |label: &str, kill_limits: bool, kill_motors: bool| {
        let mut app = flat_headless_app();
        tick(&mut app, 300);

        let mut rng = StdRng::seed_from_u64(11);
        let n = super::actuator::ACTION_SIZE;
        let freqs: Vec<f32> = (0..n).map(|_| rng.gen_range(1.0..4.0)).collect();
        let phases: Vec<f32> = (0..n)
            .map(|_| rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI))
            .collect();
        for t in 0..ENERGIZE {
            assert!(
                app.world_mut()
                    .resource_mut::<CrabActions>()
                    .set_row(0, thrash_row(Thrash::Sinusoid, t, &freqs, &phases))
            );
            tick(&mut app, 1);
        }
        // Zero drive; optionally strip the joint terms under test.
        assert!(app.world_mut().resource_mut::<CrabActions>().fill(0, 0.0));
        if kill_limits || kill_motors {
            let mut q = app.world_mut().query::<&mut MultibodyJoint>();
            for mut j in q.iter_mut(app.world_mut()) {
                let raw = &mut j.data.as_mut().raw;
                if kill_limits {
                    raw.limit_axes = JointAxesMask::empty();
                }
                if kill_motors {
                    raw.motor_axes = JointAxesMask::empty();
                }
            }
        }

        let e0 = crab_mechanical_energy(&mut app);
        let (mut e_max, mut e_end, mut grow_ticks) = (e0, e0, 0u32);
        let mut prev = e0;
        for _ in 0..PASSIVE {
            tick(&mut app, 1);
            let e = crab_mechanical_energy(&mut app);
            if e > prev + 1e-4 {
                grow_ticks += 1;
            }
            e_max = e_max.max(e);
            e_end = e;
            prev = e;
        }
        println!(
            "PASSIVE_STORM {label}: E0={e0:+.3} J  E_max={e_max:+.3} J  \
             E_end={e_end:+.3} J  gain_max={:+.3} J  grew_on {grow_ticks}/{PASSIVE} ticks",
            e_max - e0,
        );
    };

    run("baseline           ", false, false);
    run("limits OFF         ", true, false);
    run("friction-motors OFF", false, true);
    run("both OFF           ", true, true);
}

/// The [`passive_storm_energy_instrument`] on the CANONICAL GCR terrain instead of
/// the flat grid — the regime the sally-soak zero-drive ablation showed
/// accelerating passively (21 → 46 m/s with all drives zero). Flat ground
/// dissipates; if THIS grows, the injector lives in heightfield contact
/// resolution at speed, not in the joints.
#[test]
#[ignore = "rl#332 root-cause instrument — run explicitly with --ignored --nocapture"]
fn passive_storm_energy_on_terrain_instrument() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const ENERGIZE: u32 = 600;
    const PASSIVE: u32 = 1280;

    let run = |label: &str, amp: f32| {
        let mut app = headless_app();
        tick(&mut app, 300);

        let mut rng = StdRng::seed_from_u64(11);
        let n = super::actuator::ACTION_SIZE;
        let freqs: Vec<f32> = (0..n).map(|_| rng.gen_range(1.0..4.0)).collect();
        let phases: Vec<f32> = (0..n)
            .map(|_| rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI))
            .collect();
        for t in 0..ENERGIZE {
            let mut row = thrash_row(Thrash::Sinusoid, t, &freqs, &phases);
            for a in row.iter_mut() {
                *a *= amp;
            }
            assert!(
                app.world_mut()
                    .resource_mut::<CrabActions>()
                    .set_row(0, row)
            );
            tick(&mut app, 1);
        }
        assert!(app.world_mut().resource_mut::<CrabActions>().fill(0, 0.0));

        // On terrain, PE varies with where she wanders — energy alone still cannot
        // grow passively. Track E and speed both.
        let e0 = crab_mechanical_energy(&mut app);
        let (mut e_max, mut e_end) = (e0, e0);
        let mut grow_ticks = 0u32;
        let mut prev = e0;
        for t in 0..PASSIVE {
            tick(&mut app, 1);
            let e = crab_mechanical_energy(&mut app);
            if e > prev + 1e-3 {
                grow_ticks += 1;
            }
            e_max = e_max.max(e);
            e_end = e;
            prev = e;
            if t.is_multiple_of(160) {
                println!("  [{label}] passive t={t}: E={e:+.3} J");
            }
        }
        println!(
            "PASSIVE_TERRAIN {label}: E0={e0:+.3} J  E_max={e_max:+.3} J  \
             E_end={e_end:+.3} J  gain_max={:+.3} J  grew_on {grow_ticks}/{PASSIVE} ticks",
            e_max - e0,
        );
    };

    run("thrash amp=1.0", 1.0);
}

/// bddap/rl#332 — the plant has air: an unactuated falling crab approaches the
/// carapace-drag terminal velocity instead of integrating gravity unboundedly.
/// Pre-drag this fall measured ~39 m/s at the window's end; with
/// [`super::aero::CARAPACE_DRAG`] sized for v_t = √(m·g/DRAG) ≈ 15 m/s the same
/// window must land in a band around that. The band is the drift alarm on BOTH
/// sides: above 18 m/s the drag went missing (or mass grew — a re-bake changed
/// the MDP, rl#277); below 11 m/s something over-damps her and the trained gait
/// is next.
#[test]
fn airborne_crab_reaches_terminal_velocity() {
    const TICKS: u32 = 256; // 4 s — ~2.6 drag time-constants past v_t

    let mut app = flat_headless_app();
    tick(&mut app, 1);
    respawn_airborne(&mut app, MOMENTUM_SPAWN_Y);
    tick(&mut app, 4);
    disable_crab_collisions(&mut app);
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
