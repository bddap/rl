use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::collision::{NESTED_COLLISION, crab_collision, no_adjacent_contacts};
use super::components::{
    CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint, CrabRestPose,
};
use super::joint_id::CrabJointId;
use crate::bot::rig;
use crate::terrain::TerrainGrid;

pub const SPAWN_HEIGHT: f32 = 0.05;

const FRICTION_RAMP: f32 = 4.0;

/// Foot (carpus tip) contact friction. Not free-standing: it Average-pairs with
/// [`crate::physics::world::GROUND_FRICTION`] to the μ≈2.0 the rl#318 slope-hold
/// acceptance is tuned against (`slope_hold_test`) — retune BOTH or the crab
/// toboggans again. Deliberately NOT raised (nor `Max`-combined) to do the ground's
/// job: feet also self-contact adjacent legs, and stiffer foot↔leg pairs jam them
/// (`collider_check` catches it).
const FOOT_FRICTION: Friction = Friction::coefficient(1.5);

/// Soft-CCD lookahead on every crab part (bddap/rl#315): the narrow phase widens its
/// speculative-contact margin to one tick's actual travel (capped at 0.5 m — never
/// binding at crab speeds), so a driven limb closing on a sibling at m/s speeds
/// meets an active contact BEFORE overlap instead of crossing a thin capsule
/// between two detection passes. Cheap (predictive constraints, no shape-cast) and
/// velocity-gated, so resting bodies pay nothing.
const SOFT_CCD: SoftCcd = SoftCcd { prediction: 0.5 };

/// Below this point-speed, a zero-drive crab body's reported motion is contact-solver
/// noise, not signal (rl#392, the rl#377 source-split): at rest the joint angles sit
/// static to sub-mrad while the under-converged contact/limit solve on the
/// near-massless distal links reports up to ~0.7 m/s of velocity chatter and
/// ~1 mm/tick of pose wander. Raising the crab's sleep threshold over that noise
/// floor lets rapier's pose-drift sleep check retire the settled multibody from the
/// solver — rest becomes bit-exact rest (velocities zeroed on sleep), which is what
/// the rl#340 stage-2 iteration cranking bought at ~3× solver cost, for free.
/// Anything commanding the crab (any actuator torque write) force-wakes it, so only
/// zero-drive bodies can cross this gate. `resting_crab_falls_asleep` pins engagement;
/// the claws/settle chatter tests pin the resulting quiet.
pub(crate) const CRAB_SLEEP_NOISE_FLOOR: f32 = 0.3;

/// Angular twin of [`CRAB_SLEEP_NOISE_FLOOR`]: the rest-noise angular velocity the
/// solver reports on the railed claw/leg links reaches ~5 rad/s with the pose static,
/// which would trip rapier's π/2 sleep sanity gate every step. Raising the body's
/// `angular_threshold` lifts that gate (fork patch, bddap-bot/rapier@8a5d985); the
/// pose-drift criterion — which folds real rotation in via the collider extent —
/// remains the arbiter, so a genuinely tumbling crab still cannot sleep.
pub(crate) const CRAB_SLEEP_ANGULAR_NOISE_FLOOR: f32 = 10.0;

/// Extra solver iterations a ZERO-DRIVE crab's island runs on top of
/// [`crate::physics::SOLVER_ITERATIONS`] (rl#392). Two jobs, both zero-drive-only:
/// on level ground the cheap driven-gait counts never converge the rest contact
/// stack on the near-massless distal links — the crab visibly creeps (~6 cm/s
/// measured) and can't pass the sleep gates, while at the elevated total the
/// settle converges and the multibody falls asleep within ~1 s, after which it
/// costs nothing; on steep terrain, where a passive crab may slide indefinitely
/// instead of resting, the elevated count is what keeps the load-bearing
/// self-stacks resolved under the tumble: at the cheap counts the carapace
/// crushes ~100 mm into the leg bases (historical FOUGHT findings), and 12
/// is the measured floor for sleep engagement across BOTH feature graphs — the render build realizes louder rest noise (worst-link spikes 11+ rad/s vs ~5 headless-only) and stays awake at 8 total outer. The actuator flips
/// this on drive-state edges (`apply_actions`) — island-wide data through
/// rapier's own per-body lever, not a config fork — so DRIVEN play never pays it.
pub(crate) const CRAB_SETTLE_EXTRA_ITERATIONS: usize = 12;

fn crab_sleep() -> Sleeping {
    Sleeping {
        normalized_linear_threshold: CRAB_SLEEP_NOISE_FLOOR,
        angular_threshold: CRAB_SLEEP_ANGULAR_NOISE_FLOOR,
        ..Sleeping::default()
    }
}

pub const LIMIT_SOFTNESS: bevy_rapier3d::rapier::dynamics::SpringCoefficients<f32> =
    bevy_rapier3d::rapier::dynamics::SpringCoefficients {
        natural_frequency: 400.0,
        damping_ratio: 2.0,
    };

/// A random spawn orientation for respawns (every training reset, and demo resets):
/// ~80% a mild tilt (≤ ~25°) off upright, ~20% a heavy tilt up to fully inverted —
/// each about a random horizontal axis, with a random yaw on top. Forces the policy
/// to stand and right itself from a varied start rather than memorising the one bind
/// pose.
///
/// Callers are training resets (live iff `wgpu` — see `training`'s module allow) and
/// demo resets (`play`, render-gated), hence the two-feature allow.
#[cfg_attr(not(any(feature = "wgpu", feature = "render")), allow(dead_code))]
pub(crate) fn random_spawn_rotation(rng: &mut impl rand::Rng) -> Quat {
    use std::f32::consts::{PI, TAU};
    let yaw = rng.gen_range(0.0..TAU);
    let tilt = if rng.r#gen::<f32>() < 0.8 {
        rng.gen_range(0.0f32..0.44)
    } else {
        rng.gen_range(0.44..PI)
    };
    let az = rng.gen_range(0.0..TAU);
    let tilt_axis = Vec3::new(az.cos(), 0.0, az.sin());
    Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(tilt_axis, tilt)
}

pub fn spawn_crab(
    commands: &mut Commands,
    assets: &CrabAssets,
    terrain: &TerrainGrid,
    position: Vec3,
    env: usize,
    init_rotation: Quat,
) -> Entity {
    let recipe = &assets.recipe;
    let origin = position + recipe.hub_bind_world + Vec3::new(0.0, SPAWN_HEIGHT, 0.0);

    let world_pos = rig::link_world_origins(&recipe.links, origin);

    // Every part's bounding sphere in the unrotated bind pose (the carapace's wraps its
    // offset compound around `origin`, matching the collider bound below).
    let carapace_r = recipe.carapace_offset.length() + recipe.carapace_half.length();
    let spheres: Vec<(Vec3, f32)> = std::iter::once((origin, carapace_r))
        .chain(
            recipe
                .links
                .iter()
                .zip(&world_pos)
                .map(|(link, &p)| (p, link.bounding_radius())),
        )
        .collect();
    let rotated = |p: Vec3| origin + init_rotation * (p - origin);
    let low_unrot = spheres
        .iter()
        .map(|(p, r)| p.y - r)
        .fold(f32::MAX, f32::min);
    let low_rot = spheres
        .iter()
        .map(|&(p, r)| rotated(p).y - r)
        .fold(f32::MAX, f32::min);
    // The bind pose clears a FLAT floor at `position.y` by construction. Two things can
    // still bury a part (rl#283): the init rotation swinging it below the unrotated low
    // point, and — on terrain — the ground itself rising above the origin sample under
    // an outlying part. Lift by both: restore the unrotated low point, then add the max
    // terrain rise under each part's bounding disc (exactly zero on flat grids, so
    // training placement is untouched). The 8-point rim + center sampling is approximate
    // — a gradient peaking between samples undershoots by ≲ (1−cos 22.5°)·r·slope, cm at
    // crab part radii — and the spheres are conservative bounds with SPAWN_HEIGHT on
    // top, so soft-contact depenetration absorbs the residual.
    const RIM: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let terrain_rise = spheres
        .iter()
        .flat_map(|&(p, r)| {
            let c = rotated(p);
            [
                (0.0, 0.0),
                (r, 0.0),
                (-r, 0.0),
                (0.0, r),
                (0.0, -r),
                (RIM * r, RIM * r),
                (RIM * r, -RIM * r),
                (-RIM * r, RIM * r),
                (-RIM * r, -RIM * r),
            ]
            .map(|(dx, dz)| terrain.height(c.x + dx, c.z + dz) - position.y)
        })
        .fold(0.0f32, f32::max);
    let lift = (low_unrot - low_rot).max(0.0) + terrain_rise;
    let place = |p: Vec3| {
        Transform::from_translation(origin + init_rotation * (p - origin) + Vec3::Y * lift)
            .with_rotation(init_rotation)
    };

    let carapace_collider = Collider::compound(vec![(
        recipe.carapace_offset,
        Quat::IDENTITY,
        Collider::cuboid(
            recipe.carapace_half.x,
            recipe.carapace_half.y,
            recipe.carapace_half.z,
        ),
    )]);
    // Whole-body mass for the drag coefficient (aero::CarapaceDrag): summed off the
    // very colliders spawned below, via rapier's own mass_properties — the same
    // density×shape products the solver integrates, so no second mass formula to
    // drift (rl#340 stage 3: a fallback-sized constant let the heavier mesh body
    // fall at 22 m/s).
    let mut total_mass = carapace_collider
        .raw
        .mass_properties(recipe.carapace_density)
        .mass();

    let carapace = commands
        .spawn((
            CrabCarapace,
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            SOFT_CCD,
            crab_sleep(),
            AdditionalSolverIterations(CRAB_SETTLE_EXTRA_ITERATIONS),
            carapace_collider,
            crab_collision(env),
            ColliderMassProperties::Density(recipe.carapace_density),
            // Live mass mirror for the drag brake's momentum-cancel cap
            // (`aero::apply_air_drag`) — the collider density is the one mass source.
            ReadMassProperties::default(),
            place(origin),
            CrabRestPose(place(origin)),
            Velocity::default(),
            ExternalForce::default(),
        ))
        .id();

    let mut ents: Vec<Entity> = Vec::with_capacity(recipe.links.len());
    let inside_carapace = |p: Vec3| {
        (p - origin - recipe.carapace_offset)
            .abs()
            .cmple(recipe.carapace_half)
            .all()
    };
    for (i, link) in recipe.links.iter().enumerate() {
        if link.actuated.is_none() {
            ents.push(carapace);
            continue;
        }
        let parent_ent = match link.parent {
            None => carapace,
            Some(idx) => ents[idx],
        };
        let here = world_pos[i];
        let collider = match rig::link_rest_shape(link, Vec3::ZERO) {
            rig::RestShape::Capsule { a, b, radius } => Collider::capsule(a, b, radius),
            // An oriented box has no direct bevy_rapier constructor; a one-shape
            // compound carries the rotation.
            rig::RestShape::Cuboid { center, rot, half } => Collider::compound(vec![(
                center,
                rot,
                Collider::cuboid(half.x, half.y, half.z),
            )]),
        };
        let groups = if inside_carapace(here + link.center) {
            NESTED_COLLISION
        } else {
            crab_collision(env)
        };
        total_mass += collider.raw.mass_properties(link.density).mass();
        let id = link
            .actuated
            .expect("locked links are skipped before spawn");
        let joint = rig_joint(id, link.axis_local, link.anchor1);
        let mut ec = commands.spawn((
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            SOFT_CCD,
            crab_sleep(),
            AdditionalSolverIterations(CRAB_SETTLE_EXTRA_ITERATIONS),
            collider,
            groups,
            ColliderMassProperties::Density(link.density),
            MultibodyJoint::new(parent_ent, joint),
            place(here),
            CrabRestPose(place(here)),
            Velocity::default(),
            ExternalForce::default(),
        ));
        ec.insert(CrabJoint {
            id,
            axis_local: link.axis_local,
        });
        if matches!(id, CrabJointId::ClawPincer(_)) {
            ec.insert(CrabClawTip);
        }
        if matches!(id, CrabJointId::LegCarpus(..)) {
            ec.insert(FOOT_FRICTION);
        }
        ents.push(ec.id());
    }

    commands
        .entity(carapace)
        .insert(crate::bot::aero::CarapaceDrag::for_total_mass(total_mass));

    carapace
}

/// The one joint of an articulation (bddap/rl#347 — was two in rl#315): the
/// MULTIBODY revolute carries kinematics, soft limits, and a pure-stiction
/// friction motor (it pins resting poses against slow gravity creep — rl#318's
/// zero-input slope hold). The rl#315 viscous flail brake (`-c·ω`, c sized so full
/// drive balances drag at [`CrabJointId::free_rate`]) is NOT a second joint: it is
/// per-dof damping set on the multibody itself ([`set_flail_damping`]), folded
/// into the augmented mass — implicit, unconditionally dissipative.
///
/// rl#315 rode the brake on a parallel ImpulseJoint velocity motor because the
/// multibody joint offers one motor line and stiction uses it. That construct was
/// the rl#347 energy source: an iterative velocity-motor constraint sharing dofs
/// with the reduced-coordinate multibody it constrains can rail against its
/// impulse bounds and inject its full cap — ~1500 rad/s in a tick on the
/// near-massless distal claw pair. The multibody damping replacement drops the
/// old headroom-capped drag (2× the drive ceiling) for uncapped viscous drag: identical
/// below 2× the free rate, strictly MORE braking past it, and it can only remove
/// energy.
fn rig_joint(id: CrabJointId, axis: Vec3, anchor1: Vec3) -> TypedJoint {
    let [lo, hi] = id.limits();
    let mut revolute = no_adjacent_contacts(
        RevoluteJointBuilder::new(axis)
            .local_anchor1(anchor1)
            .local_anchor2(Vec3::ZERO)
            .limits([lo, hi])
            .motor_velocity(0.0, FRICTION_RAMP)
            .motor_max_force(id.friction_cap())
            .motor_model(MotorModel::ForceBased),
    );
    let generic: &mut GenericJoint = revolute.as_mut();
    generic.raw.softness = LIMIT_SOFTNESS;
    revolute
}

/// Arms the rl#315 flail brake on every newly-created crab articulation: sets the
/// multibody's per-dof viscous damping to [`CrabJointId::drive_damping`] the tick
/// rapier materializes the joint (respawns re-run it via `Added`). Until it runs,
/// the joint sits one tick on rapier's default 0.1 N·m·s/rad — inert.
pub(in crate::bot) fn set_flail_damping(
    new: Query<(&RapierMultibodyJointHandle, &CrabJoint), Added<RapierMultibodyJointHandle>>,
    mut ctx: Query<&mut bevy_rapier3d::plugin::context::RapierContextJoints>,
) {
    if new.is_empty() {
        return;
    }
    let mut joints = ctx
        .single_mut()
        .expect("exactly one rapier context per crab world");
    for (handle, joint) in new.iter() {
        match joints.multibody_joints.get_mut(handle.0) {
            Some((multibody, link_id)) => {
                // rapier 0.35 exposes the raw per-dof damping DVector rather than the
                // fork's per-link setter; fill this link's dof rows ourselves.
                let Some(link) = multibody.link(link_id) else {
                    error!(
                        "flail brake NOT armed on {:?}: multibody link {link_id} is stale (rl#347)",
                        joint.id
                    );
                    continue;
                };
                let (assembly_id, ndofs) = (link.assembly_id(), link.joint().ndofs());
                multibody
                    .damping_mut()
                    .rows_mut(assembly_id, ndofs)
                    .fill(joint.id.drive_damping());
            }
            // A miss would leave the joint on rapier's 0.1 default — a silently
            // weaker flail brake, so it must announce itself.
            None => error!(
                "flail brake NOT armed on {:?}: multibody joint handle is stale (rl#347)",
                joint.id
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::ecs::world::CommandQueue;

    use super::super::components::CrabBodyPart;
    use super::*;

    /// A 2×2 grid whose one cell ramps 5 m over 1 m toward +x — steep enough that a
    /// footprint-blind spawn at the origin sample buries the +x parts in the hill.
    /// Note parse datum-shifts the declared heights to [-5, 0, -5, 0] — same ramp,
    /// shifted down.
    fn steep_ramp() -> TerrainGrid {
        TerrainGrid::test_grid(2, 2, 1.0, 1.0, &[0, 5, 0, 5])
    }

    /// rl#283: the spawn lift must clear the TERRAIN under the whole footprint, not just
    /// a flat plane through the origin sample — on a steep slope the uphill parts would
    /// otherwise spawn inside the hill and take a depenetration kick.
    #[test]
    fn spawn_lift_clears_the_hill_under_the_footprint() {
        let grid = steep_ramp();
        let mut world = World::new();
        let assets = CrabAssets {
            recipe: crate::bot::rig::baked_recipe(),
        };
        let position = grid.place(Vec2::ZERO, 0.0);
        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, &world);
        spawn_crab(&mut commands, &assets, &grid, position, 0, Quat::IDENTITY);
        queue.apply(&mut world);

        let mut parts = 0;
        let mut q = world.query_filtered::<&Transform, With<CrabBodyPart>>();
        for t in q.iter(&world) {
            let surface = grid.height(t.translation.x, t.translation.z);
            assert!(
                t.translation.y > surface,
                "part centre {:?} buried under the ramp (surface {surface})",
                t.translation
            );
            parts += 1;
        }
        assert!(parts > 10, "expected a whole crab, got {parts} parts");
    }
}
