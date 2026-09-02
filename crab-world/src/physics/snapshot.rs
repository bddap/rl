//! A whole-plant rapier snapshot at the end of one tick, plus the drives the
//! ORIGINAL run applied on the tick that followed — enough to replay that one
//! tick outside bevy under a different solver configuration and read where its
//! energy came from (rl#332 T1). The snapshot is the physics state the solver
//! sees; the replay owns no policy, no sensors, no schedule: drives are a row of
//! numbers, so "zero the drives" and "change the solver" are independent levers.

use bevy::prelude::*;
use bevy_rapier3d::plugin::context::{
    RapierContextColliders, RapierContextJoints, RapierContextSimulation, RapierRigidBodySet,
};
use bevy_rapier3d::prelude::{MultibodyJoint, RapierRigidBodyHandle};
pub use bevy_rapier3d::rapier::dynamics::SpringCoefficients;
use bevy_rapier3d::rapier::dynamics::{
    CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
    RigidBodyHandle, RigidBodySet,
};
use bevy_rapier3d::rapier::geometry::{ColliderSet, DefaultBroadPhase, NarrowPhase};
use bevy_rapier3d::rapier::pipeline::PhysicsPipeline;
use serde::{Deserialize, Serialize};

use crate::bot::actuator::{CrabActions, applied_torque};
use crate::bot::aero::{CarapaceDrag, drag_force};
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId};
use crate::physics::{PHYSICS_DT, PHYSICS_GRAVITY};

#[derive(Serialize, Deserialize, Clone)]
pub struct SnapJoint {
    pub id: CrabJointId,
    pub axis_local: [f32; 3],
    pub child: RigidBodyHandle,
    pub parent: RigidBodyHandle,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PlantSnapshot {
    pub tick: u64,
    pub params: IntegrationParameters,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub islands: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub ccd_solver: CCDSolver,
    /// Env 0's crab, carapace first.
    pub parts: Vec<RigidBodyHandle>,
    pub joints: Vec<SnapJoint>,
    pub drag_coeff: f32,
    /// The drive row the original run applied on tick `tick + 1`.
    pub actions: Vec<f32>,
    /// (linvel, angvel) per part after the original run's tick `tick + 1` —
    /// the replay's self-check target.
    pub expected: Vec<([f32; 3], [f32; 3])>,
}

impl PlantSnapshot {
    /// Physics state at the end of `tick`; `actions`/`expected` are filled by
    /// [`Self::finish`] once the next tick has run.
    pub fn capture(world: &mut World, tick: u64) -> Self {
        let (parts, joints, drag_coeff) = {
            let mut q = world.query_filtered::<(
                Entity,
                &RapierRigidBodyHandle,
                &CrabEnvId,
                Option<&CrabCarapace>,
                Option<&CarapaceDrag>,
                Option<(&CrabJoint, &MultibodyJoint)>,
            ), With<CrabBodyPart>>();
            let rows: Vec<_> = q
                .iter(world)
                .filter(|(_, _, env, ..)| env.0 == 0)
                .map(|(e, h, _, cara, drag, joint)| {
                    (
                        e,
                        h.0,
                        cara.is_some(),
                        drag.map(CarapaceDrag::coeff),
                        joint.map(|(j, mj)| (j.id, j.axis_local, mj.parent)),
                    )
                })
                .collect();
            let handle_of = |e: Entity| {
                rows.iter()
                    .find(|r| r.0 == e)
                    .map(|r| r.1)
                    .expect("joint parent is a crab part")
            };
            let mut parts: Vec<RigidBodyHandle> =
                rows.iter().filter(|r| r.2).map(|r| r.1).collect();
            parts.extend(rows.iter().filter(|r| !r.2).map(|r| r.1));
            let joints = rows
                .iter()
                .filter_map(|r| {
                    r.4.map(|(id, axis, parent)| SnapJoint {
                        id,
                        axis_local: axis.to_array(),
                        child: r.1,
                        parent: handle_of(parent),
                    })
                })
                .collect();
            let drag_coeff = rows
                .iter()
                .find_map(|r| r.3)
                .expect("the carapace carries CarapaceDrag");
            (parts, joints, drag_coeff)
        };
        let bodies = world
            .query::<&RapierRigidBodySet>()
            .single(world)
            .expect("one rapier context")
            .bodies
            .clone();
        let colliders = world
            .query::<&RapierContextColliders>()
            .single(world)
            .expect("one rapier context")
            .colliders
            .clone();
        let joint_sets = world
            .query::<&RapierContextJoints>()
            .single(world)
            .expect("one rapier context");
        let (impulse_joints, multibody_joints) = (
            joint_sets.impulse_joints.clone(),
            joint_sets.multibody_joints.clone(),
        );
        let sim = world
            .query::<&RapierContextSimulation>()
            .single(world)
            .expect("one rapier context");
        Self {
            tick,
            params: sim.integration_parameters,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            islands: sim.islands.clone(),
            broad_phase: sim.broad_phase.clone(),
            narrow_phase: sim.narrow_phase.clone(),
            ccd_solver: sim.ccd_solver.clone(),
            parts,
            joints,
            drag_coeff,
            actions: Vec::new(),
            expected: Vec::new(),
        }
    }

    /// Records what the original run did on tick `tick + 1`: the drive row it
    /// applied and the velocities it ended with.
    pub fn finish(&mut self, world: &mut World) {
        self.actions = world.resource::<CrabActions>().rows()[0].to_vec();
        let set = world
            .query::<&RapierRigidBodySet>()
            .single(world)
            .expect("one rapier context");
        self.expected = self
            .parts
            .iter()
            .map(|h| {
                let rb = set.bodies.get(*h).expect("snapshot part still exists");
                (rb.linvel().to_array(), rb.angvel().to_array())
            })
            .collect();
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        bincode::deserialize(&std::fs::read(path)?).map_err(std::io::Error::other)
    }

    /// Max part speed the original run ended tick `tick + 1` with.
    pub fn original_max_speed(&self) -> f32 {
        self.expected
            .iter()
            .map(|(v, _)| Vec3::from(*v).length())
            .fold(0.0, f32::max)
    }

    /// Whole-body mechanical energy of the snapshot's crab.
    pub fn energy(&self) -> f32 {
        mech_energy(&self.bodies, &self.parts)
    }

    /// Replay tick `tick + 1` once under `cfg`, on a copy of the state.
    pub fn replay(&self, cfg: &ReplayConfig) -> ReplayOutcome {
        let mut s = self.clone();
        s.params.num_solver_iterations = cfg.iterations.0;
        s.params.num_internal_pgs_iterations = cfg.iterations.1;
        s.params.num_internal_stabilization_iterations = cfg.iterations.2;
        s.params.dt = PHYSICS_DT / cfg.substeps as f32;
        if let Some(soft) = cfg.limit_softness {
            let handles: Vec<_> = s.multibody_joints.iter().map(|(h, ..)| h).collect();
            for h in handles {
                if let Some((mb, link_id)) = s.multibody_joints.get_mut(h)
                    && let Some(link) = mb.link_mut(link_id)
                {
                    link.joint.data.softness = soft;
                }
            }
        }

        let mut torque: std::collections::HashMap<RigidBodyHandle, Vec3> =
            std::collections::HashMap::new();
        if !cfg.zero_drives {
            for j in &s.joints {
                let parent_rot = s
                    .bodies
                    .get(j.parent)
                    .expect("joint parent")
                    .position()
                    .rotation;
                let world_axis = parent_rot * Vec3::from(j.axis_local);
                let wrench = world_axis * applied_torque(j.id, s.actions[j.id.index()]);
                *torque.entry(j.child).or_default() += wrench;
                *torque.entry(j.parent).or_default() -= wrench;
            }
        }
        for (i, h) in s.parts.iter().enumerate() {
            let rb = s.bodies.get_mut(*h).expect("snapshot part");
            rb.reset_forces(true);
            rb.reset_torques(true);
            if i == 0 {
                let v = rb.linvel();
                rb.add_force(drag_force(s.drag_coeff, rb.mass(), v), true);
            }
            if let Some(t) = torque.get(h) {
                rb.add_torque(*t, true);
            }
        }

        let before: Vec<(Vec3, Vec3)> = s
            .parts
            .iter()
            .map(|h| {
                let rb = s.bodies.get(*h).expect("snapshot part");
                (rb.linvel(), rb.angvel())
            })
            .collect();
        let e0 = s.energy();
        let mut pipeline = PhysicsPipeline::new();
        for _ in 0..cfg.substeps {
            pipeline.step(
                PHYSICS_GRAVITY,
                &s.params,
                &mut s.islands,
                &mut s.broad_phase,
                &mut s.narrow_phase,
                &mut s.bodies,
                &mut s.colliders,
                &mut s.impulse_joints,
                &mut s.multibody_joints,
                &mut s.ccd_solver,
                &(),
                &(),
            );
        }
        let after: Vec<(Vec3, Vec3)> = s
            .parts
            .iter()
            .map(|h| {
                let rb = s.bodies.get(*h).expect("snapshot part");
                (rb.linvel(), rb.angvel())
            })
            .collect();
        let e1 = s.energy();

        let mut out = ReplayOutcome {
            energy_before: e0,
            energy_after: e1,
            max_speed_before: 0.0,
            max_speed_after: 0.0,
            max_angvel_before: 0.0,
            max_angvel_after: 0.0,
            kicks: 0,
            worst_kick: (0, 0.0, 0.0),
            max_dev_from_original: 0.0,
        };
        for (i, ((v0, w0), (v1, w1))) in before.iter().zip(&after).enumerate() {
            let (s0, s1) = (v0.length(), v1.length());
            out.max_speed_before = out.max_speed_before.max(s0);
            out.max_speed_after = out.max_speed_after.max(s1);
            out.max_angvel_before = out.max_angvel_before.max(w0.length());
            out.max_angvel_after = out.max_angvel_after.max(w1.length());
            if is_kick(s0, s1) {
                out.kicks += 1;
                if s1 / s0.max(KICK_FLOOR_M_S)
                    > out.worst_kick.2 / out.worst_kick.1.max(KICK_FLOOR_M_S)
                {
                    out.worst_kick = (i, s0, s1);
                }
            }
            if let Some((ev, ew)) = s.expected.get(i) {
                let dev = (*v1 - Vec3::from(*ev))
                    .length()
                    .max((*w1 - Vec3::from(*ew)).length());
                out.max_dev_from_original = out.max_dev_from_original.max(dev);
            }
        }
        out
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayConfig {
    pub zero_drives: bool,
    pub iterations: (usize, usize, usize),
    pub substeps: usize,
    /// Joint limit spring override; `None` keeps the snapshot's springs.
    pub limit_softness: Option<SpringCoefficients<f32>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayOutcome {
    pub energy_before: f32,
    pub energy_after: f32,
    pub max_speed_before: f32,
    pub max_speed_after: f32,
    pub max_angvel_before: f32,
    pub max_angvel_after: f32,
    pub kicks: usize,
    /// (part index, speed before, speed after) of the largest-ratio kick.
    pub worst_kick: (usize, f32, f32),
    /// Largest |Δv| or |Δω| between this replay and the original run's tick.
    pub max_dev_from_original: f32,
}

/// A part below this speed is not a kick source: a 0.1→0.5 m/s solver-noise
/// twitch would otherwise read as a 5× jump.
pub const KICK_FLOOR_M_S: f32 = 1.0;

/// The rl#332 F3 shape: a part's speed multiplies >4× in ONE tick. No drive can
/// do that to a link already moving — drives are rate-bounded at the joint —
/// and no strike can either: an impact redistributes body KE, so the struck
/// link's speed lands near the body's, not at several times its own.
pub fn is_kick(speed_before: f32, speed_after: f32) -> bool {
    speed_after > 4.0 * speed_before.max(KICK_FLOOR_M_S)
}

/// Σ ½m|v|² + ½ω·I·ω + m·g·y over `parts`.
pub fn mech_energy(bodies: &RigidBodySet, parts: &[RigidBodyHandle]) -> f32 {
    let g = -PHYSICS_GRAVITY.y;
    parts
        .iter()
        .filter_map(|h| bodies.get(*h))
        .map(|rb| {
            let m = rb.mass();
            let v = rb.linvel();
            let w = rb.angvel();
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

/// Energy-ledger window (ticks) and its slack: over any [`LEDGER_WINDOW`] ticks a
/// driven crab may gain at most Σ gross actuator power·dt + [`LEDGER_SLACK_J`].
/// The slack covers what the gross-power sum never books — energy the joint limit
/// springs store and return, and the one-tick sampling of torque × rate.
pub const LEDGER_WINDOW: usize = 32;
pub const LEDGER_SLACK_J: f32 = 100.0;
