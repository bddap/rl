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

/// Issue #18: under the all-zero policy the crab must SETTLE and sit still, not
/// jiggle and bounce. The floppy legs collapse onto their joint limits at rest, and
/// with too few solver sub-steps the solver can't converge those unilateral
/// constraints so the whole body buzzes. This pins the quiet: after settling, the
/// carapace barely rotates and barely bounces. The crumple assert is the
/// counterweight — a "fix" that quieted the body by stiffening the legs back into a
/// rigid brace would drop the deflection and FAIL here rather than pass.
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

    // Watch a ~3 s window: a settled crab should be nearly motionless.
    let mut ang_sum = 0.0f32;
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    let window = 192u32;
    for _ in 0..window {
        tick(&mut app, 1);
        let (ang, y) = carapace(&mut app);
        ang_sum += ang;
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    let ang_mean = ang_sum / window as f32;
    let bounce = y_max - y_min;
    println!(
        "rest: carapace angular speed mean {ang_mean:.3} rad/s, bounce {bounce:.4} m, \
         leg crumple {crumple:.3} rad"
    );

    // Measured: substeps=4 (the fix) ~1.0 rad/s & ~0.03 m; substeps=1 (the bug) ~2.3
    // rad/s & ~0.076 m. Thresholds sit between, with margin for the deterministic sim.
    assert!(
        ang_mean < 1.6,
        "carapace still twitching at rest: angular speed mean {ang_mean:.3} rad/s \
         (want <1.6; the substeps=1 bug sits ~2.3)"
    );
    assert!(
        bounce < 0.055,
        "carapace bouncing at rest: {bounce:.4} m peak-to-peak (want <0.055; bug ~0.076)"
    );
    assert!(
        crumple > 0.4,
        "legs no longer crumple ({crumple:.3} rad) — the rest-quiet fix must not \
         stiffen the legs into a rigid brace; keep them floppy"
    );
}
