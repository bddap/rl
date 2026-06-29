//! Instantiating the rig-derived recipe into a live Rapier multibody: [`spawn_crab`]
//! and its joint/collider helpers, plus the tuning constants the spawn consumes
//! (spawn clearance, the joint-friction ramp, and the limit spring).

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::collision::{NESTED_COLLISION, crab_collision, no_adjacent_contacts};
use super::components::{
    CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint, CrabRestPose,
};
use super::joint_id::CrabJointId;
use crate::bot::rig;

// ---------------------------------------------------------------------------
// Dimensions — tuned for a ~1m wide crab (game scale, not real life)
// ---------------------------------------------------------------------------

/// Clearance the body is lifted above its bind pose at spawn, so it drops the
/// last bit onto its feet rather than starting interpenetrating the ground. The
/// body otherwise spawns in the glTF bind-world frame (feet already near y=0), so
/// this is a small drop, not the full standing height — that height comes from the
/// bind pose itself.
pub const SPAWN_HEIGHT: f32 = 0.05;

/// Slope of the velocity-0 friction motor's ramp into its force cap
/// ([`CrabJointId::friction_cap`]): steep enough to saturate near-instantly into a
/// small CONSTANT opposing torque (dry/Coulomb friction, not a viscous damping gain).
/// The motor has stiffness 0 (no position servo — the policy commands all torque via
/// `ExternalForce`), and is ForceBased so the cap is an honest N·m, not rescaled by
/// the link's effective mass. Friction MUST live on the joint: Rapier's per-body
/// `Damping` is a no-op on multibody links — only joint constraints and external
/// forces reach them (#14).
const FRICTION_RAMP: f32 = 4.0;

/// Spring (natural frequency Hz, damping ratio) of every revolute joint's
/// constraint — the lock holding the limb attached AND the hard end stops. Softened
/// well below Rapier's near-rigid `1e6` Hz default because at that stiffness a limb
/// driven into its limit overshoots and the violent position-correction snapping it
/// back is NOT momentum-conserving in the reduced-coordinate multibody solver: with
/// every joint slammed at once the residual accumulates into net angular momentum,
/// letting an airborne crab spin up from nothing (issue #17 — actuator couples and
/// contact were ruled out, so the joint-LIMIT impulse was the leak). 400 Hz keeps the
/// stop firm under standing load yet caps the airborne overshoot enough to remove the
/// runaway pumping; damping 2.0 (> the default 1.0) softens the snap without letting
/// the limb sag through its stop. The iterative solver still leaves a small drift
/// floor, but no runaway. (Tuning figures: issue #17.)
pub const LIMIT_SOFTNESS: bevy_rapier3d::rapier::dynamics::SpringCoefficients<f32> =
    bevy_rapier3d::rapier::dynamics::SpringCoefficients {
        natural_frequency: 400.0,
        damping_ratio: 2.0,
    };

// ---------------------------------------------------------------------------
// Spawn the crab — instantiate the rig-derived recipe
// ---------------------------------------------------------------------------

/// A random spawn orientation for the `RL_RANDOM_INIT` curriculum: ~80% a mild tilt
/// (≤ ~25°) off upright, ~20% a heavy tilt up to fully inverted — each about a random
/// horizontal axis, with a random yaw on top. Forces the policy to stand and right
/// itself from a varied start rather than memorising the one bind pose.
pub(crate) fn random_spawn_rotation(rng: &mut impl rand::Rng) -> Quat {
    use std::f32::consts::{PI, TAU};
    let yaw = rng.gen_range(0.0..TAU);
    // `r#gen`: `gen` is a reserved keyword in edition 2024, so the rand 0.8 method
    // needs the raw identifier to parse.
    let tilt = if rng.r#gen::<f32>() < 0.8 {
        rng.gen_range(0.0f32..0.44) // ≤ ~25°: upright, lightly perturbed
    } else {
        rng.gen_range(0.44..PI) // up to fully upside-down
    };
    let az = rng.gen_range(0.0..TAU);
    let tilt_axis = Vec3::new(az.cos(), 0.0, az.sin());
    Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(tilt_axis, tilt)
}

/// Spawns a complete crab body at the given position, instantiating the
/// rig-derived [`crate::bot::rig::RigRecipe`] (the one model). Returns the carapace entity.
///
/// Links are emitted parent-before-child so each joint can reference its already-
/// spawned parent entity. At rest every link is axis-aligned, so a link's world
/// position is just its parent's plus the joint anchor — tracked here to seed each
/// body's initial `Transform` near the pose the multibody solver will hold.
pub fn spawn_crab(
    commands: &mut Commands,
    assets: &CrabAssets,
    position: Vec3,
    env: usize,
    init_rotation: Quat,
) -> Entity {
    let recipe = &assets.recipe;
    // Spawn in the glTF bind-world frame: the carapace root sits at the leg hub's
    // true bind-world position (offset by the spawn point + a clearance drop), so
    // every link's `anchor1` delta lands it at its real glTF bone origin — the exact
    // frame the cosmetic skin renders its bones in. The skin is the truth; the
    // physics aligns to it. Anchoring at a bare `(0, SPAWN_HEIGHT, 0)` instead pinned
    // the hub at an arbitrary height and dropped the hub's lateral/forward bind
    // offset, sliding the whole body ~0.1 (mostly −Z) off the skin.
    let origin = position + recipe.hub_bind_world + Vec3::new(0.0, SPAWN_HEIGHT, 0.0);

    // Every link's bind-pose world origin (unrotated), telescoped from the spawn hub
    // — used both to chain children below and to size the clearance lift.
    let world_pos = rig::link_world_origins(&recipe.links, origin);

    // `init_rotation` rigidly rotates the whole bind pose about `origin`. Every body
    // gets the SAME rotation, so parent-frame == child-frame still holds and the local
    // joint axes/anchors stay valid (the invariant `rig_revolute` relies on). Rotating
    // can swing limbs below the floor, so lift the assembly back to the upright pose's
    // ground clearance: `lift` = how much lower the rotated lowest body sits than the
    // unrotated one — exactly 0 for identity, so upright spawns are unchanged. Without
    // it an inverted spawn interpenetrates the floor on tick 0 and the solver NaNs the
    // env (a storm across every env on a randomized reset).
    let carapace_r = recipe.carapace_offset.length() + recipe.carapace_half.length();
    let mut low_unrot = origin.y - carapace_r;
    let mut low_rot = origin.y - carapace_r;
    for (link, &p) in recipe.links.iter().zip(&world_pos) {
        let r = link.center.length() + link.half_height + link.radius;
        low_unrot = low_unrot.min(p.y - r);
        low_rot = low_rot.min((origin + init_rotation * (p - origin)).y - r);
    }
    let lift = (low_unrot - low_rot).max(0.0);
    let place = |p: Vec3| {
        Transform::from_translation(origin + init_rotation * (p - origin) + Vec3::Y * lift)
            .with_rotation(init_rotation)
    };

    // -- Carapace (root): the rigid trunk; shell/thorax/rostrum/abdomen ride it.
    let carapace = commands
        .spawn((
            CrabCarapace,
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            // Offset cuboid: the trunk's bounding box isn't centred on the leg hub
            // the body root sits at, so the box rides at `carapace_offset` to cover
            // the shell without engulfing the legs.
            Collider::compound(vec![(
                recipe.carapace_offset,
                Quat::IDENTITY,
                Collider::cuboid(
                    recipe.carapace_half.x,
                    recipe.carapace_half.y,
                    recipe.carapace_half.z,
                ),
            )]),
            crab_collision(env),
            ColliderMassProperties::Density(recipe.carapace_density),
            // No stand-in Mesh3d: the visible body is the skin, and the true colliders
            // are shown by the shared `crab_view` collider wireframe (the render-mode cycle).
            // A primitive mesh here only risked drifting out of sync with the actual collider.
            place(origin),
            CrabRestPose(place(origin)),
            Velocity::default(),
            ExternalForce::default(),
            // No `Damping`: Rapier applies per-body damping only to non-multibody
            // bodies, and the carapace is the multibody root, so it would be a no-op.
        ))
        .id();

    let mut ents: Vec<Entity> = Vec::with_capacity(recipe.links.len());
    // A world point inside the carapace box (centred at `origin + carapace_offset`),
    // tested in the unrotated bind frame — this "is the stub tucked under the shell"
    // grouping is topological, hence rotation-invariant.
    let inside_carapace = |p: Vec3| {
        (p - origin - recipe.carapace_offset)
            .abs()
            .cmple(recipe.carapace_half)
            .all()
    };
    for (i, link) in recipe.links.iter().enumerate() {
        // Cosmetic locked links (the eye-stalks: `actuated == None`) are NOT spawned as
        // physics bodies. They carry no policy joint and aren't load-bearing (rig.rs),
        // never enter the observation, and ride atop the carapace where their colliders
        // never reach the ground — yet each one added a rigid body to the single-island
        // multibody solve (the per-step bottleneck) for zero return. The skin already
        // rides the eye bones off the carapace cosmetically (the eye link is fixed to it;
        // see `skin::link_map`), so dropping the bodies is invisible. They survive in the
        // rig (`recipe.links`) for the cosmetic/debug-collider view and the fallback body,
        // which read the rig directly, not these entities. A `carapace` placeholder keeps
        // `ents` index-aligned with `recipe.links`; it is never read as a parent because a
        // locked link is only ever the parent of another (also-skipped) locked link.
        if link.actuated.is_none() {
            ents.push(carapace);
            continue;
        }
        let parent_ent = match link.parent {
            None => carapace,
            Some(idx) => ents[idx],
        };
        let here = world_pos[i]; // unrotated bind-pose origin
        let collider = capsule_collider(link.center, link.col_rot, link.half_height, link.radius);
        // A link whose collider center sits inside the carapace box is a proximal
        // stub tucked under the shell; group it so it can't fight the carapace
        // collider (see [`NESTED_COLLISION`]). Distal limb segments reach outside the
        // box and keep full crab collision (ground + sibling limbs).
        let groups = if inside_carapace(here + link.center) {
            NESTED_COLLISION
        } else {
            crab_collision(env)
        };
        // Every link reaching here is actuated — locked (cosmetic) links were skipped above.
        let id = link.actuated.expect("locked links are skipped before spawn");
        let joint = rig_revolute(id, link.axis_local, link.anchor1);
        let mut ec = commands.spawn((
            CrabBodyPart,
            CrabEnvId(env),
            RigidBody::Dynamic,
            collider,
            groups,
            ColliderMassProperties::Density(link.density),
            // No stand-in Mesh3d (see carapace): skin + Rapier debug-render are the
            // truthful views; a fixed per-link capsule mesh misrepresented the colliders.
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
        // The movable-finger link is a claw tip (see [`CrabClawTip`]).
        if matches!(id, CrabJointId::ClawPincer(_)) {
            ec.insert(CrabClawTip);
        }
        // Grippy feet: the distal leg bone (`004`) is what plants on the ground.
        if link.bone.starts_with("Def_leg") && link.bone.contains(".004.") {
            ec.insert(Friction::coefficient(1.5));
        }
        ents.push(ec.id());
    }

    carapace
}

/// Revolute joint for a policy-actuated link: free about `axis` (parent frame),
/// hard-limited, with a friction motor and no contact against the adjacent
/// link. Coordinate 0 is the bind pose (links spawn axis-aligned), so the limits
/// straddle 0 directly — no rest bake needed.
///
/// `RevoluteJointBuilder::new(axis)` writes `axis` into BOTH bodies' local frames,
/// which is only correct while children spawn at identity (parent frame == child
/// frame == world). When phase 2 bakes each bone's bind orientation into the child,
/// the child-side axis/anchor must be re-expressed in the child frame, or the solver
/// will constrain a different axis than the sensor/actuator read.
fn rig_revolute(id: CrabJointId, axis: Vec3, anchor1: Vec3) -> TypedJoint {
    let [lo, hi] = id.limits();
    let mut joint = no_adjacent_contacts(
        RevoluteJointBuilder::new(axis)
            .local_anchor1(anchor1)
            .local_anchor2(Vec3::ZERO)
            .limits([lo, hi])
            .motor_velocity(0.0, FRICTION_RAMP)
            .motor_max_force(id.friction_cap())
            .motor_model(MotorModel::ForceBased),
    );
    let generic: &mut GenericJoint = joint.as_mut();
    generic.raw.softness = LIMIT_SOFTNESS;
    joint
}

/// A capsule collider offset within its link: the bone runs from the pivot (link
/// origin) along `rot·Y`, so the shape sits centred at `center` and oriented to
/// match — the same offset-baked-into-the-shape trick the fitted path uses.
fn capsule_collider(center: Vec3, rot: Quat, half_height: f32, radius: f32) -> Collider {
    let axis = rot * Vec3::Y * half_height;
    Collider::capsule(center - axis, center + axis, radius)
}
