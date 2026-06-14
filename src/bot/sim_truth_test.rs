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

/// Joint friction must keep every limb's angular speed bounded under load.
///
/// The failure this guards: a directly-torqued light segment with no working
/// damping is a pure double-integrator, so the distal tibia ramped to 300–600
/// rad/s and tripped the blow-up speed guard every episode — training never got
/// an episode long enough to learn from. The friction has to live on the joint
/// (a velocity motor), because Rapier's per-body `Damping` is a no-op on
/// multibody links. Slamming every joint to full torque — harder than any policy
/// command — must still leave the limbs at a sane speed.
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
    // Under the capped-friction + inertia-graded-ceiling + tapered-mass regime the
    // small force-capped friction barely slows a limb, so what bounds the speed is
    // the hard joint limit a limb reaches in a couple of ticks (the heavier,
    // weaker-driven distal links no longer snap). The terminal scales inversely with
    // the friction cap — looser legs spin a touch faster into their stops, ~49 rad/s
    // at the current cap. Pre-fix was 300–600. 100 leaves headroom to loosen the
    // friction further (floppier legs) while still tripping far below the ~300 rad/s
    // blow-up guard.
    assert!(
        max_ang < 100.0,
        "a limb is spinning at {max_ang:.1} rad/s under full torque — joint \
         friction/ceiling/mass regressed (pre-fix the tibia hit 300–600 rad/s and \
         the blow-up guard then killed every episode in ~8 steps)"
    );
}

/// The crab must spawn already standing in its rest pose, every joint inside its
/// own limits.
///
/// A Rapier multibody initialises each joint coordinate to 0. Pre-bake that put
/// every limb at angle 0 — a flat splay that for the knees and coxae is *outside*
/// their limits, so the solver snapped them on the first tick instead of settling
/// from the intended stance. `revolute_joint` bakes the rest pose into each joint
/// frame so coordinate 0 is the planted stance; this pins that. The prismatic
/// pincer is excluded — its coordinate is a translation, so `joint_angle` reads
/// only the (near-zero) twist, not the DOF.
#[test]
fn crab_spawns_in_rest_pose_inside_limits() {
    use bevy_rapier3d::prelude::MultibodyJoint;
    use std::collections::HashMap;

    let mut app = headless_app();
    // Tick 1 builds the crab; Rapier writes the multibody spawn pose back into
    // the child Transforms on tick 2 (before that they hold a default identity
    // Transform, reading angle 0). A couple more ticks let the solver relax onto
    // the limits without the whole crab gravity-settling away from spawn.
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
        if matches!(id, CrabJointId::ClawPincer(_)) {
            continue;
        }
        let angle = joint_angle(id, rot[&mj.parent], tf.rotation);
        let rest = id.default_position();
        let [lo, hi] = id.limits();
        assert!(
            (angle - rest).abs() < 0.15,
            "{id:?} spawned at {angle:+.3} rad, not its rest pose {rest:+.3} — the \
             rest-pose frame bake is wrong (Rapier starts multibody joints at \
             coordinate 0, not default_position)"
        );
        assert!(
            angle >= lo - 1e-3 && angle <= hi + 1e-3,
            "{id:?} spawned at {angle:+.3} rad, outside its limits [{lo:+.3}, {hi:+.3}]"
        );
        checked += 1;
    }
    // 32 DOFs − 2 prismatic pincers = 30 revolute joints, every one verified.
    assert_eq!(checked, CrabJointId::COUNT - 2);
}

/// The actuator must be INTERNAL: every wrench it applies to the crab sums to
/// zero net force AND zero net torque, so it can never inject linear or angular
/// momentum. A crab in free-fall may reorient by swinging limbs (a falling-cat
/// turn conserves momentum), but it must NOT be able to spin itself UP — that
/// needs an external torque. Revolute joints push equal-and-opposite torque
/// couples (zero net by construction); the prismatic pincer pushes a *force*
/// couple, which only stays torque-free if the two forces share a line of action.
/// This pins that — a nonzero net torque here is the crab spinning itself out of
/// nothing (owner-reported "rotates mid-air in a way that shouldn't be possible").
#[test]
fn actuator_injects_no_net_wrench() {
    use super::actuator::CrabActions;
    use super::body::CrabBodyPart;
    use bevy_rapier3d::prelude::ExternalForce;

    use std::collections::HashMap;

    let mut app = headless_app();
    tick(&mut app, 1);
    // Drive every joint, pincers included, and let the claws rotate for a while
    // so the pincer slide axis is no longer aligned with the COM offset — the
    // configuration that turns its force couple into a net torque.
    {
        let mut actions = app.world_mut().resource_mut::<CrabActions>();
        for v in actions.envs[0].iter_mut() {
            *v = 1.0;
        }
    }
    tick(&mut app, 40);

    // ExternalForce and positions MUST be sampled at the same instant: the
    // actuator sets the wrench from the pre-step pose, so reading positions a
    // step later would show a spurious residual (one tick of motion × the force).
    // Snapshot the poses, then tick once more — that tick's actuator runs on
    // exactly these poses — and read the wrench it produced against the snapshot.
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
        // Net force is ~0 (couples), so torque about the origin equals torque
        // about the COM — no mass weighting needed.
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

/// Joints must YIELD to load above the friction cap — the legs crumple rather
/// than hold the body up as a rigid frame. With ZERO actuation the only thing
/// resisting a joint is the small force-capped friction (~0.3 N·m), below the
/// ~0.5 N·m a leg joint needs to bear its share of the body, so an unactuated
/// crab sags onto the ground. Standing must be ACTIVE: the policy holds the crab
/// up, physics does not do it for free — a joint stiff enough to hold the pose
/// passively would let training cheat by optimizing a frozen statue.
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
    // Build the crab and let Rapier write the spawn pose back, then read height
    // and joint angles before it can sag.
    tick(&mut app, 3);
    let start_y = carapace_y(&mut app);

    // ~2 s with no actions (CrabActions default to all-zero torque).
    tick(&mut app, 128);
    let end_y = carapace_y(&mut app);
    let leg_deflection = max_leg_joint_deflection(&mut app);
    println!(
        "carapace y: spawn {start_y:.3} -> unactuated+2s {end_y:.3} (sag {:.3}); \
         max leg-joint deflection {leg_deflection:.3} rad",
        start_y - end_y
    );
    assert!(
        end_y < start_y - 0.10,
        "unactuated crab did not sag (carapace {start_y:.3} -> {end_y:.3}): the joints \
         hold the body up rigidly instead of yielding to load — friction too stiff to \
         crumple (a passive standing statue, the bug this fix removes)"
    );
    assert!(
        leg_deflection > 0.15,
        "no leg JOINT yielded ({leg_deflection:.3} rad max from rest) — a carapace drop \
         with no joint bending is a rigid-body tip/fall, not compliant crumple; the \
         joints are too stiff to yield under load"
    );
}

/// Largest |angle − rest| over all leg femur/tibia joints — proof that a JOINT
/// yielded (legs folding), distinguishing compliant crumple from a rigid-body
/// tip-over (which also drops the carapace) or a blow-up respawn (every joint reset
/// to rest, so deflection ~0). Shared by the crumple and rest-quiet tests so their
/// "the legs are still floppy" check can't drift apart.
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
            CrabJointId::LegFemur(..) | CrabJointId::LegTibia(..)
        ) {
            continue;
        }
        let angle = joint_angle(joint.id, rot[&mj.parent], tf.rotation);
        max_def = max_def.max((angle - joint.id.default_position()).abs());
    }
    max_def
}

/// Largest gap, in metres, between the two world anchor points every multibody
/// joint pins together. A revolute joint forces `world(anchor1) == world(anchor2)`,
/// so this is ~0 for an attached limb; it grows only if the joint's positional
/// lock sags under load — the visible "limb detaching from its parent" failure.
/// The #17 limit softness deliberately softens the WHOLE joint (positional lock
/// included) to cap mid-air overshoot, so this is the metric that says how much
/// slack that bought: softening further to chase rest-quiet would pull the anchors
/// apart here first. Computed from the two rapier bodies' poses through the local
/// anchors the joint was built with.
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

/// Issues #18/#19: under the all-zero policy the crab must SETTLE and sit still,
/// not jiggle and bounce. The residual bounce (#19) is a CONTACT event — at rest
/// the floppy legs hang slack OFF their limit stops (crumple ~0.7 rad), so the
/// joint-limit spring is not engaged and does nothing to the bounce; it is the
/// foot-touchdown contact ringing up through the body. Softening the contact
/// spring (physics::CONTACT_SOFTNESS) absorbs the touchdown locally. This pins the
/// quiet: after settling, the carapace barely rotates and barely bounces.
///
/// Three counterweights keep a cheap "fix" from passing:
/// - crumple > 0.4: a fix that quiets the body by stiffening the legs into a rigid
///   brace would DROP the deflection and fail here (legs must stay floppy).
/// - anchor gap < 0.08 m: the rest-quiet lever must not be "soften the joint
///   positional lock until the limbs detach" — that pulls the anchors apart (#19).
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

    let mut app = headless_app();
    tick(&mut app, 1);

    // Fall and settle onto the ground under zero torque (~5 s), then read the legs.
    tick(&mut app, 320);
    let crumple = max_leg_joint_deflection(&mut app);

    // Watch a ~3 s window: a settled crab should be nearly motionless. Track the
    // worst limb-to-parent anchor gap across the window too — standing load is when
    // a soft positional lock sags most.
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

    // Shipped at substeps=2 + 12 Hz contact spring: ~0.6 rad/s, ~0.32 cm. The
    // regressions these bars catch, all with margin: the prior 16 Hz spring sits
    // ~0.82 / ~0.6 cm, the 30 Hz default ~1.44 / ~3.6 cm, substeps=1 ~1.54 / ~3.0 cm.
    // The sim is deterministic, so these are exact.
    assert!(
        ang_mean < 0.75,
        "carapace still twitching at rest: angular speed mean {ang_mean:.3} rad/s \
         (want <0.75; 16 Hz contact sits ~0.82, the 30 Hz / substeps=1 regressions ~1.5)"
    );
    assert!(
        bounce < 0.005,
        "carapace bouncing at rest: {bounce:.4} m peak-to-peak (want <0.005; 16 Hz \
         contact ~0.006, the 30 Hz regression ~0.036, substeps=1 ~0.030)"
    );
    assert!(
        crumple > 0.4,
        "legs no longer crumple ({crumple:.3} rad) — the rest-quiet fix must not \
         stiffen the legs into a rigid brace; keep them floppy"
    );
    // The #17 joint softness already drifts the anchors ~4.8 cm under standing load;
    // this bounds it well clear of visible detachment. Softening the joint
    // positional lock further to chase rest-quiet would trip here first.
    assert!(
        max_gap < 0.08,
        "a limb is separating from its parent: max anchor gap {max_gap:.4} m under \
         standing load (want <0.08; the joint positional lock has been softened too \
         far and the limbs are detaching)"
    );
}

/// Total angular momentum of env-0's crab about its own center of mass, in the
/// world frame: `L = Σ_parts ( I_world·ω + m·(r−r_com)×(v−v_com) )`, with
/// `I_world = R·I_local·Rᵀ`. This is the conserved quantity the
/// `airborne_crab_conserves_angular_momentum` invariant watches.
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
        let v = rb.linvel(); // velocity of the COM
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

/// THE momentum invariant: a driven, airborne crab cannot gain angular momentum.
///
/// With no external torque, the total angular momentum L of the whole multibody
/// about its center of mass is conserved — a falling-cat reorientation swings
/// limbs to *re-aim* a fixed L, it cannot grow |L|. |L| growing from nothing is
/// the owner-reported bug: the crab "spins itself up mid-air in a way that
/// shouldn't be possible" (#17).
///
/// This is the TRUE invariant that [`actuator_injects_no_net_wrench`] only
/// half-covers. That test sums the actuator's `ExternalForce` couples and finds
/// them balanced to ~1e-5 N·m — necessary but not sufficient, because it is
/// blind to forces the solver applies internally: the joint friction motors and
/// the hard joint-LIMIT constraints. Those are where the leak actually lived.
/// Driven hard into a near-rigid limit, a limb overshot by ~2.3 rad in a tick
/// and the violent position-correction snapping it back did not conserve L in
/// the reduced-coordinate solver; with all 30 joints slammed at once the
/// residual accumulated into real spin. Softening the joint limits
/// ([`LIMIT_SOFTNESS`]) caps the overshoot and removes the runaway.
///
/// The harness makes the airborne window unambiguous: gravity OFF (so the crab
/// never falls to the ground during the long measurement) and crab collision
/// OFF (so flailing limbs can't push off each other) — the test asserts ZERO
/// contacts throughout, so whatever L does is provably down to internal forces
/// alone. Gravity would in any case act through the COM and not change L about
/// it; removing it just buys an arbitrarily long contact-free window.
///
/// Drive is a temporally-correlated full-range random walk (like a real policy,
/// not white noise), the regime under which the bug shows. The bound is on
/// RUNAWAY, not machine-zero: the iterative solver has a small irreducible drift
/// floor, but it must not *pump*. Pre-fix this run grew |L| ~68×; post-fix it
/// stays a bounded few×.
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

    // Gravity off — no fall, so the crab stays airborne for the whole window.
    {
        let mut q = app.world_mut().query::<&mut RapierConfiguration>();
        for mut cfg in q.iter_mut(app.world_mut()) {
            cfg.gravity = Vec3::ZERO;
        }
    }
    // Respawn env-0 high up, far from the ground/walls.
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
    tick(&mut app, 4); // let rapier digest the respawn and write the spawn pose

    // Crab parts collide with nothing — guarantees the window is contact-free.
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
        // Correlated random walk: nudge each action a little and clamp.
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
    // The window must be genuinely contact-free, or "no external torque" is a
    // lie and a ground/self push could masquerade as conservation.
    assert_eq!(
        total_contacts, 0,
        "airborne window had {total_contacts} contact-points — not contact-free, \
         so the momentum check isn't isolating internal forces"
    );
    // No runaway: an airborne crab must not be able to pump its own spin up.
    // Pre-fix (rigid limits) this ratio was ~68×; the soft-limit fix holds it to
    // a bounded few×. 8× fails loudly on the old behaviour with margin to spare.
    //
    // The ratio reads ~2.3× here, not 1.0×, and that is NOT new spin: l0 is sampled
    // after a short respawn settle during which crab-vs-crab contacts are briefly
    // live (collisions are filtered off only afterwards), so the softer contact
    // spring (#19) leaves a smaller post-settle l0 — a smaller denominator inflates
    // the ratio. The ABSOLUTE peak |L| is ~unchanged (slightly lower) vs the old
    // 30 Hz contact, so conservation is intact; only the normalisation moved.
    assert!(
        peak < l0 * 8.0,
        "airborne crab spun ITSELF up: |L| grew from {l0:.4} to {peak:.4} \
         ({:.1}×) with zero contacts and no external torque — angular momentum is \
         being injected by the joint constraint solver (issue #17)",
        peak / l0.max(1e-9)
    );
}
