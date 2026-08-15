//! rl#349 solver-integrity probes. Two instruments, one lesson:
//!
//! - [`violent_fling_tick_stays_uniform`] — the rl#339 hypersonic class: a huge
//!   uniform kick must stay uniform (no internal-dof redistribution) and later
//!   near-force-free ticks must not amplify the COM. Red before the 9f01a0e
//!   drag-brake force cap; green since. The regression gate for that class.
//! - [`luge_impact_conserves_energy`] — the rl#349 storm class: a crab rammed
//!   into steep terrain at storm speed (30–120 m/s) throws distal links to
//!   1000+ rad/s, and the whole-body KE ledger proves the spins are CONVERTED
//!   from body KE (whip-crack, energy-legit), not injected by the solve. The
//!   assert is on the ledger, not the spins: spins at storm speed are honest
//!   physics — the rl#332 actuator rescale is what removes the speeds. A future
//!   solver injector (the thing six rl#349 candidate fixes went looking for)
//!   would turn this red.

use bevy::prelude::*;
use bevy_rapier3d::prelude::{ExternalForce, Velocity};

use super::body::{CrabBodyPart, CrabCarapace, CrabEnvId};
use super::headless::{HeadlessStack, WorldRole, flat_headless_app, headless_stack, tick};
use crate::Visuals;
use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::physics::PHYSICS_HZ;

/// Fling controller: while armed, add `force` to the carapace's `ExternalForce`
/// AFTER `apply_actions` zeroed it (the rl#298 shove seam). `one_shot` disarms
/// after a single tick; otherwise it holds — the sustained "rocket" that stands
/// in for rl#332 policy-catapult storm power.
#[derive(Resource, Default)]
struct Fling {
    force: Vec3,
    armed: bool,
    one_shot: bool,
}

fn apply_fling(
    mut fling: ResMut<Fling>,
    mut carapaces: Query<&mut ExternalForce, With<CrabCarapace>>,
) {
    if !fling.armed {
        return;
    }
    for mut ef in carapaces.iter_mut() {
        ef.force += fling.force;
    }
    if fling.one_shot {
        fling.armed = false;
    }
}

fn add_fling(app: &mut App) {
    app.insert_resource(Fling::default());
    app.add_systems(
        FixedUpdate,
        apply_fling
            .after(super::BotSet::Act)
            .before(bevy_rapier3d::plugin::PhysicsSet::SyncBackend),
    );
}

/// (COM velocity, max |part velocity − COM velocity|) over env 0's parts.
/// Unweighted: parts are similar-density and the probes read scales, not exact
/// momentum.
fn com_and_internal(app: &mut App) -> (Vec3, f32) {
    let mut q = app
        .world_mut()
        .query_filtered::<(&CrabEnvId, &Velocity), With<CrabBodyPart>>();
    let vels: Vec<Vec3> = q
        .iter(app.world())
        .filter(|(env, _)| env.0 == 0)
        .map(|(_, v)| v.linear)
        .collect();
    assert!(vels.len() > 30, "expected a whole crab, got {}", vels.len());
    let com = vels.iter().copied().sum::<Vec3>() / vels.len() as f32;
    let internal = vels
        .iter()
        .map(|v| (*v - com).length())
        .fold(0.0f32, f32::max);
    (com, internal)
}

/// Whole-crab kinetic energy (translational + rotational) and total mass, from
/// rapier's own masses/inertias — the ledger's ruler.
fn crab_energy(app: &mut App) -> (f32, f32) {
    use bevy_rapier3d::plugin::context::RapierRigidBodySet;
    use bevy_rapier3d::prelude::RapierRigidBodyHandle;
    let handles: Vec<bevy_rapier3d::rapier::dynamics::RigidBodyHandle> = {
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
    let mut ke = 0.0;
    let mut mass = 0.0;
    for h in handles {
        let rb = set.bodies.get(h).expect("rapier body");
        ke += rb.kinetic_energy();
        mass += rb.mass();
    }
    (ke, mass)
}

/// The storm-impact energy gate. Rocket the crab up to storm speed in a valley,
/// then cut power and let it ram a 30–60° wall with its drives saturated toward
/// their stops (the canary policy's signature). During the unpowered impact the
/// only energy sources are gravity and the damper-bounded drives — any per-tick
/// KE growth past that budget is the solver injecting. Distal links legitimately
/// reach 1000+ rad/s in these impacts (whip-crack conversion of body KE); the
/// gate deliberately does NOT bound them.
#[test]
#[ignore = "multi-second solver probe — run explicitly (rl#349)"]
fn luge_impact_conserves_energy() {
    // A valley between two 30–60° walls, 12×8 cells at 2 m pitch: whichever way
    // the rocket points, the crab rams UP a face with its feet catching — the
    // upslope-catch geometry of every rl#349 soak violation.
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
    add_fling(&mut app);
    tick(&mut app, 64); // settle standing

    // Saturated alternating drives — the pinned-past-stops signature from the
    // rl#349 canary violation rows.
    let mut row = [0.0f32; ACTION_SIZE];
    for (i, v) in row.iter_mut().enumerate() {
        *v = if i % 2 == 0 { -1.0 } else { 1.0 };
    }

    // Peak drive power: each joint maxes τ·ω − c·ω² at τ²/(4c).
    let p_drives: f32 = crate::bot::body::CrabJointId::all()
        .iter()
        .map(|id| {
            id.drive_torque_ceiling() * id.drive_torque_ceiling() / (4.0 * id.drive_damping())
        })
        .sum();

    const THRUST_N: f32 = 2000.0;
    const DT: f32 = 1.0 / PHYSICS_HZ as f32;
    let mut worst_ang = (0.0f32, 0.0f32, 0u32);
    let mut worst_injection = (f32::MIN, 0u32, 0u32); // (J over budget, burst, coast tick)
    for burst in 0..10u32 {
        let dir = if burst % 2 == 0 { 1.0 } else { -1.0 };
        {
            let mut fling = app.world_mut().resource_mut::<Fling>();
            fling.force = Vec3::X * (dir * THRUST_N);
            fling.armed = true;
            fling.one_shot = false;
        }
        for _ in 0..96 {
            {
                let mut actions = app.world_mut().resource_mut::<CrabActions>();
                assert!(actions.set_row(0, row), "env 0 unsized");
            }
            tick(&mut app, 1);
        }
        // Unpowered impact: the ledger is armed.
        app.world_mut().resource_mut::<Fling>().armed = false;
        let (mut ke_prev, mass) = crab_energy(&mut app);
        for t in 0..96u32 {
            {
                let mut actions = app.world_mut().resource_mut::<CrabActions>();
                assert!(actions.set_row(0, row), "env 0 unsized");
            }
            tick(&mut app, 1);
            let (ke, _) = crab_energy(&mut app);
            let (com, _) = com_and_internal(&mut app);
            // Falling at |v| trades PE→KE at m·g·|v|; generous 2x slack + 2 J.
            let budget = DT * (2.0 * mass * 9.81 * com.length() + p_drives) + 2.0;
            let over = (ke - ke_prev) - budget;
            if over > worst_injection.0 {
                worst_injection = (over, burst, t);
            }
            ke_prev = ke;
            let mut q = app
                .world_mut()
                .query_filtered::<(&CrabEnvId, &Velocity), With<CrabBodyPart>>();
            for (env, vel) in q.iter(app.world()) {
                if env.0 == 0 {
                    let ang = vel.angular.length();
                    if ang > worst_ang.0 {
                        worst_ang = (ang, vel.linear.length(), burst);
                    }
                }
            }
        }
    }
    println!(
        "storm impacts: worst coast part ang {:.0} rad/s (lin {:.0} m/s, burst {}); \
         worst coast KE delta {:.1} J relative to budget (burst {}, tick {})",
        worst_ang.0,
        worst_ang.1,
        worst_ang.2,
        worst_injection.0,
        worst_injection.1,
        worst_injection.2
    );
    assert!(
        worst_injection.0 <= 0.0,
        "solver injected {:.1} J past the gravity+drives budget in ONE unpowered \
         tick (burst {}, tick {}) — a NEW rl#349-class energy injection",
        worst_injection.0,
        worst_injection.1,
        worst_injection.2
    );
}

/// The rl#339 hypersonic gate: Δv 6000 m/s in one uniform-force tick must stay
/// uniform (internal spread at joint scale) and must not be amplified by later
/// near-force-free solves. Red before the 9f01a0e drag force cap (COM ×7 with
/// ≤113 N external); green since.
#[test]
#[ignore = "multi-second solver probe — run explicitly (rl#339/rl#349)"]
fn violent_fling_tick_stays_uniform() {
    let mut app = flat_headless_app();
    add_fling(&mut app);
    tick(&mut app, 64); // settle on the flat floor

    // Whole-crab mass ~0.78 kg (rl#332): F = Δv·m·Hz for one tick.
    let m = 0.781;
    let dv = 6000.0;
    {
        let mut fling = app.world_mut().resource_mut::<Fling>();
        fling.force = Vec3::Y * (dv * m * PHYSICS_HZ as f32);
        fling.armed = true;
        fling.one_shot = true;
    }
    tick(&mut app, 1);
    let (com1, internal1) = com_and_internal(&mut app);
    println!(
        "fling tick: com {:.0} m/s (y {:.0}), internal spread {:.0} m/s",
        com1.length(),
        com1.y,
        internal1
    );

    // A uniform kick must stay uniform: articulation under this jerk is O(10)
    // m/s, not kilometers/second.
    assert!(
        internal1 < 100.0,
        "fling tick redistributed a uniform force into {internal1:.0} m/s of \
         internal dof motion (rl#339/rl#349)"
    );

    let mut prev = com1.length();
    for t in 0..5 {
        tick(&mut app, 1);
        let (com, internal) = com_and_internal(&mut app);
        let s = com.length();
        println!(
            "tick +{}: com {s:.0} m/s, internal {internal:.0} m/s",
            t + 1
        );
        assert!(
            s.is_finite() && s < prev * 1.05 + 1.0,
            "COM amplified {prev:.0} -> {s:.0} m/s with no external force \
             (rl#339 drag-brake amplification)"
        );
        prev = s;
    }
}
