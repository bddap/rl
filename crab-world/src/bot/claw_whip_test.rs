//! rl#347: the claw-whip solver spike. A cold random policy spins the distal claw
//! pair (wrist+pincer) to O(1000) rad/s from nothing — energy the drives (capped
//! N·m) and the rl#315 damper (which out-brakes drive above the free rate) cannot
//! supply. These soaks reproduce it under a deterministic full-scale random drive
//! and pin the fix: no claw part may exceed the rl#343 integrity bound.

use bevy::prelude::*;

use super::body::{CrabBodyPart, CrabEnvId, CrabJoint};
use super::headless::{HeadlessStack, WorldRole, headless_stack, tick};
use crate::Visuals;
use crate::bot::actuator::{ACTION_SIZE, CrabActions};

/// Deterministic full-scale drive noise — splitmix64 per (env, tick, channel), so
/// a failure names an exact (seed, tick) to replay.
fn drive_row(seed: u64, env: usize, t: u32) -> [f32; ACTION_SIZE] {
    let mut row = [0.0f32; ACTION_SIZE];
    for (c, v) in row.iter_mut().enumerate() {
        let mut x = seed
            ^ (env as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ u64::from(t).wrapping_mul(0xBF58476D1CE4E5B9)
            ^ (c as u64).wrapping_mul(0x94D049BB133111EB);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        *v = (x >> 40) as f32 / ((1u64 << 24) as f32) * 2.0 - 1.0;
    }
    row
}

/// Step `envs` crabs under full-scale random drives for `ticks`, returning the
/// worst (bound-units speed, lin, ang, joint, tick) seen on any claw part —
/// `lin.max(ang / 3)`, the rl#343 integrity bound's units.
fn random_drive_soak(
    envs: usize,
    ticks: u32,
    seed: u64,
) -> (f32, f32, f32, Option<CrabJoint>, u32) {
    let mut app = headless_stack(HeadlessStack {
        num_envs: envs,
        role: WorldRole::RolloutWorker,
        grid: std::sync::Arc::new(crate::terrain::TerrainGrid::flat(512.0)),
        visuals: Visuals(false),
    });
    tick(&mut app, 64); // 1 s: spawn + touch down

    let mut worst = (0.0f32, 0.0f32, 0.0f32, None, 0u32);
    for t in 0..ticks {
        {
            let mut actions = app.world_mut().resource_mut::<CrabActions>();
            for e in 0..envs {
                assert!(actions.set_row(e, drive_row(seed, e, t)), "env {e} unsized");
            }
        }
        tick(&mut app, 1);
        let mut q = app.world_mut().query_filtered::<(
            &CrabEnvId,
            &bevy_rapier3d::prelude::Velocity,
            Option<&CrabJoint>,
        ), With<CrabBodyPart>>();
        for (_env, vel, joint) in q.iter(app.world()) {
            let lin = vel.linear.length();
            let ang = vel.angular.length();
            let s = lin.max(ang / 3.0);
            if s > worst.0 {
                worst = (s, lin, ang, joint.copied(), t);
            }
        }
    }
    worst
}

/// The rl#343 bound the trainer hard-fails at.
const INTEGRITY_BOUND: f32 = 100.0;

/// The rl#347 flail brake actually lands on the multibody: after spawn,
/// every crab articulation's joint dof carries [`CrabJointId::drive_damping`],
/// not rapier's 0.1 default. Guards the `set_flail_damping` wiring (system
/// ordering, `Added` detection, the `damping_mut` row fill) — a silent
/// miss would leave the plant on default damping with no error.
#[test]
fn flail_damping_lands_on_every_articulation() {
    use bevy_rapier3d::plugin::context::RapierContextJoints;
    use bevy_rapier3d::prelude::RapierMultibodyJointHandle;

    let mut app = super::headless::flat_headless_app();
    tick(&mut app, 3); // spawn + rapier sync + set_flail_damping

    let mut q = app
        .world_mut()
        .query::<(&RapierMultibodyJointHandle, &CrabJoint)>();
    let joints: Vec<_> = q.iter(app.world()).map(|(h, j)| (h.0, j.id)).collect();
    assert!(
        joints.len() > 30,
        "expected a whole crab, got {}",
        joints.len()
    );

    let mut ctx = app.world_mut().query::<&mut RapierContextJoints>();
    let mut ctx = ctx.single_mut(app.world_mut()).expect("one rapier context");
    for (handle, id) in joints {
        let (mb, link_id) = ctx
            .multibody_joints
            .get_mut(handle)
            .expect("every crab joint is a multibody joint");
        let link = mb.link(link_id).expect("handle names a live link");
        let (assembly_id, ndofs) = (link.assembly_id(), link.joint().ndofs());
        let damping: Vec<f32> = mb.damping().as_slice()[assembly_id..assembly_id + ndofs].to_vec();
        assert_eq!(
            damping,
            &[id.drive_damping()][..],
            "{id:?}: flail-brake damping did not land on the multibody dof"
        );
    }
}

/// The rl#347 acceptance: a long full-scale random-drive soak (the cold-policy
/// excitation that tripped rl#346 in minutes at 40 envs) stays under the
/// integrity bound on every part.
#[test]
#[ignore = "multi-minute soak — run explicitly (rl#347)"]
fn random_drive_soak_stays_bounded() {
    let (s, lin, ang, joint, t) = random_drive_soak(16, 20_000, 347);
    println!("worst bound-units speed {s:.1} (lin {lin:.1} ang {ang:.1}) at tick {t} on {joint:?}");
    assert!(
        s < INTEGRITY_BOUND,
        "claw whip reproduced: {s:.1} bound units (lin {lin:.1} m/s, ang {ang:.1} rad/s) \
         at tick {t} on {joint:?} (rl#347)"
    );
}
