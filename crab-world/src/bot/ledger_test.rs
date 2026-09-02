//! rl#332 energy ledger on the SHIPPED solver configuration under drive. A
//! settled crab in a steep valley is driven with saturated, sign-flipping drives
//! (the trained policy's stop-railing signature) and the whole-body mechanical
//! energy is audited over every [`LEDGER_WINDOW`]-tick window against the
//! actuators' gross power, and the narrow phase is audited for same-crab contact
//! rows, the row class PGS cannot converge (rl#332). Both must be clean at the
//! counts the game actually ships. Kicks (a part's speed ×4 in one tick) are
//! printed as data: unjammed legs under saturated flip-drives swing at the
//! drive's own rate, so here the count cannot tell a kick from a drive.

use bevy::prelude::*;
use bevy_rapier3d::plugin::context::{
    RapierContextColliders, RapierContextJoints, RapierContextSimulation, RapierRigidBodySet,
};
use bevy_rapier3d::prelude::{MultibodyJoint, RapierRigidBodyHandle, Velocity};

use super::actuator::{ACTION_SIZE, CrabActions, applied_torque};
use super::body::{CrabBodyPart, CrabEnvId, CrabJoint, CrabJointId, joint_angle};
use super::headless::{HeadlessStack, WorldRole, headless_stack, tick};
use super::sensor::CrabObservation;
use crate::Visuals;
use crate::physics::PHYSICS_DT;
use crate::physics::snapshot::{
    LEDGER_SLACK_J, LEDGER_WINDOW, SpringCoefficients, is_kick, mech_energy,
};

fn crab_mech_energy(app: &mut App) -> f32 {
    let handles: Vec<_> = {
        let mut q = app
            .world_mut()
            .query_filtered::<(&CrabEnvId, &RapierRigidBodyHandle), With<CrabBodyPart>>();
        q.iter(app.world())
            .filter(|(env, _)| env.0 == 0)
            .map(|(_, h)| h.0)
            .collect()
    };
    let mut set_q = app.world_mut().query::<&RapierRigidBodySet>();
    let set = set_q.single(app.world()).expect("rapier context");
    mech_energy(&set.bodies, &handles)
}

fn part_speeds(app: &mut App) -> Vec<f32> {
    let mut q = app
        .world_mut()
        .query_filtered::<(&CrabEnvId, &Velocity), With<CrabBodyPart>>();
    q.iter(app.world())
        .filter(|(env, _)| env.0 == 0)
        .map(|(_, v)| v.linear.length())
        .collect()
}

fn gross_power(app: &mut App, row: &[f32; ACTION_SIZE]) -> f32 {
    let obs = app.world().resource::<CrabObservation>();
    let view = obs.env(0).expect("env 0 observed");
    CrabJointId::all()
        .iter()
        .map(|id| (applied_torque(*id, row[id.index()]) * view.joint_rate(*id)).abs())
        .sum()
}

/// Active contact pairs whose BOTH colliders belong to env 0.s crab this tick.
fn same_crab_contacts(app: &mut App) -> usize {
    let crab: std::collections::HashSet<Entity> = {
        let mut q = app
            .world_mut()
            .query_filtered::<(Entity, &CrabEnvId), With<CrabBodyPart>>();
        q.iter(app.world())
            .filter(|(_, env)| env.0 == 0)
            .map(|(e, _)| e)
            .collect()
    };
    let mut q = app
        .world_mut()
        .query::<(&RapierContextColliders, &RapierContextSimulation)>();
    let (cols, sim) = q.single(app.world()).expect("rapier context");
    sim.narrow_phase
        .contact_pairs()
        .filter(|p| p.has_any_active_contact())
        .filter(|p| {
            [p.collider1, p.collider2]
                .iter()
                .all(|h| cols.collider_entity(*h).is_some_and(|e| crab.contains(&e)))
        })
        .count()
}

/// Largest excursion of any joint past its stop, rad.
fn max_stop_sag(app: &mut App) -> f32 {
    let mut q = app
        .world_mut()
        .query_filtered::<(&CrabEnvId, &CrabJoint, &MultibodyJoint, &Transform), With<CrabBodyPart>>();
    let rows: Vec<(CrabJointId, Vec3, Entity, Quat)> = q
        .iter(app.world())
        .filter(|(env, ..)| env.0 == 0)
        .map(|(_, j, mj, t)| (j.id, j.axis_local, mj.parent, t.rotation))
        .collect();
    let mut sag = 0.0f32;
    for (id, axis, parent, child_rot) in rows {
        let parent_rot = app
            .world()
            .get::<Transform>(parent)
            .expect("parent")
            .rotation;
        let angle = joint_angle(axis, parent_rot, child_rot);
        let [lo, hi] = id.limits();
        sag = sag.max(angle - hi).max(lo - angle);
    }
    sag
}

#[derive(Clone, Copy, Debug)]
struct Variant {
    iterations: Option<(usize, usize, usize)>,
    limit: Option<SpringCoefficients<f32>>,
}

const SHIPPED: Variant = Variant {
    iterations: None,
    limit: None,
};

#[derive(Debug)]
struct Audit {
    worst_over_budget: (f32, usize),
    kicks: Vec<(usize, usize, f32, f32)>,
    self_contacts: usize,
    max_speed: f32,
    max_sag: f32,
}

fn drive_and_audit(v: Variant) -> Audit {
    let grid = crate::terrain::TerrainGrid::test_grid(
        12,
        8,
        2.0,
        2.0,
        &[
            6, 5, 3, 1, 0, 0, 0, 0, 1, 3, 5, 6, //
            6, 4, 2, 1, 0, 0, 0, 0, 2, 4, 5, 6, //
            6, 5, 3, 2, 0, 0, 0, 0, 1, 2, 4, 6, //
            6, 4, 3, 1, 0, 0, 0, 0, 2, 3, 5, 6, //
            6, 5, 2, 1, 0, 0, 0, 0, 1, 4, 6, 6, //
            6, 4, 3, 2, 0, 0, 0, 0, 2, 3, 5, 6, //
            6, 5, 3, 1, 0, 0, 0, 0, 1, 2, 4, 6, //
            6, 4, 2, 1, 0, 0, 0, 0, 2, 4, 5, 6, //
        ],
    );
    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::RolloutWorker,
        grid: std::sync::Arc::new(grid),
        visuals: Visuals(false),
    });
    tick(&mut app, 2);
    if let Some((outer, pgs, stab)) = v.iterations {
        let mut q = app.world_mut().query::<&mut RapierContextSimulation>();
        let mut sim = q.single_mut(app.world_mut()).expect("rapier context");
        sim.integration_parameters.num_solver_iterations = outer;
        sim.integration_parameters.num_internal_pgs_iterations = pgs;
        sim.integration_parameters
            .num_internal_stabilization_iterations = stab;
    }
    if let Some(soft) = v.limit {
        let mut q = app.world_mut().query::<&mut RapierContextJoints>();
        let mut joints = q.single_mut(app.world_mut()).expect("rapier context");
        let handles: Vec<_> = joints.multibody_joints.iter().map(|(h, ..)| h).collect();
        for h in handles {
            let (mb, link_id) = joints.multibody_joints.get_mut(h).expect("joint");
            mb.link_mut(link_id).expect("link").joint.data.softness = soft;
        }
    }
    tick(&mut app, 62);

    const DRIVEN_TICKS: usize = 768;
    const FLIP_EVERY: usize = 16;
    let mut energies = vec![crab_mech_energy(&mut app)];
    let mut powers: Vec<f32> = Vec::new();
    let mut prev = part_speeds(&mut app);
    let mut audit = Audit {
        worst_over_budget: (f32::MIN, 0),
        kicks: Vec::new(),
        self_contacts: 0,
        max_speed: 0.0,
        max_sag: 0.0,
    };
    for t in 0..DRIVEN_TICKS {
        let sign = if (t / FLIP_EVERY).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let mut row = [0.0f32; ACTION_SIZE];
        for (i, v) in row.iter_mut().enumerate() {
            *v = sign * if i % 2 == 0 { -1.0 } else { 1.0 };
        }
        assert!(
            app.world_mut()
                .resource_mut::<CrabActions>()
                .set_row(0, row),
            "env 0 unsized"
        );
        tick(&mut app, 1);
        powers.push(gross_power(&mut app, &row));
        energies.push(crab_mech_energy(&mut app));
        audit.max_sag = audit.max_sag.max(max_stop_sag(&mut app));
        audit.self_contacts += same_crab_contacts(&mut app);
        let now = part_speeds(&mut app);
        for (i, (s0, s1)) in prev.iter().zip(&now).enumerate() {
            audit.max_speed = audit.max_speed.max(*s1);
            if is_kick(*s0, *s1) {
                audit.kicks.push((t, i, *s0, *s1));
            }
        }
        prev = now;
    }
    for end in LEDGER_WINDOW..energies.len() {
        let de = energies[end] - energies[end - LEDGER_WINDOW];
        let budget =
            powers[end - LEDGER_WINDOW..end].iter().sum::<f32>() * PHYSICS_DT + LEDGER_SLACK_J;
        if de - budget > audit.worst_over_budget.0 {
            audit.worst_over_budget = (de - budget, end);
        }
    }
    println!(
        "driven ledger {:?}: worst window {:+.1} J vs budget (ending tick {}), max part speed {:.2} m/s, max stop sag {:.3} rad, same-crab contacts {}, kicks {}{}",
        v,
        audit.worst_over_budget.0,
        audit.worst_over_budget.1,
        audit.max_speed,
        audit.max_sag,
        audit.self_contacts,
        audit.kicks.len(),
        audit
            .kicks
            .first()
            .map(|k| format!(" (first {k:?})"))
            .unwrap_or_default()
    );
    audit
}

#[test]
fn driven_crab_energy_ledger_holds_on_shipped_solver() {
    let audit = drive_and_audit(SHIPPED);
    assert!(
        audit.worst_over_budget.0 <= 0.0,
        "solver injected {:.1} J past the actuator budget over one {LEDGER_WINDOW}-tick window \
         (ending driven tick {}) on the shipped solver configuration (rl#332)",
        audit.worst_over_budget.0,
        audit.worst_over_budget.1
    );
    assert_eq!(
        audit.self_contacts, 0,
        "same-crab contact rows exist on the shipped collision filter (rl#332)"
    );
}

/// Data, not a gate: the same audit across the solver/limit-spring variants the
/// rl#332 one-tick replay matrix discriminated.
#[test]
#[ignore = "variant matrix — run explicitly (rl#332)"]
fn solver_variant_matrix() {
    let soft = |hz: f32, zeta: f32| {
        Some(SpringCoefficients {
            natural_frequency: hz,
            damping_ratio: zeta,
        })
    };
    for v in [
        SHIPPED,
        Variant {
            iterations: Some((2, 2, 3)),
            limit: None,
        },
        Variant {
            iterations: Some((2, 4, 3)),
            limit: None,
        },
        Variant {
            iterations: Some((3, 2, 3)),
            limit: None,
        },
        Variant {
            iterations: Some((4, 2, 3)),
            limit: None,
        },
        Variant {
            iterations: Some((2, 6, 3)),
            limit: None,
        },
        Variant {
            iterations: Some((2, 8, 3)),
            limit: None,
        },
        Variant {
            iterations: Some((2, 12, 3)),
            limit: None,
        },
        Variant {
            iterations: Some((2, 16, 3)),
            limit: None,
        },
        Variant {
            iterations: Some((2, 8, 2)),
            limit: None,
        },
        Variant {
            iterations: Some((1, 8, 3)),
            limit: None,
        },
        Variant {
            iterations: None,
            limit: soft(40.0, 2.0),
        },
    ] {
        drive_and_audit(v);
    }
}
