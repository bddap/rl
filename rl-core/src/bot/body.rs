//! Crab body definition: the `CrabJointId` action/observation joint set, the
//! per-instance joint + marker components, and `spawn_crab`, which instantiates
//! the rig-derived body recipe ([`super::rig`] owns the geometry).
//!
//! The body is a Rapier multibody tree rooted at the carapace. Only the locomotion
//! joints are policy-actuated and carry a [`CrabJointId`]: each leg's
//! coxa/merus/carpus and each claw's shoulder/wrist/pincer (all revolute). Each
//! limb collapses to those joints — a leg's proximal bones ride its coxa link, so
//! there are no locked stubs. Only the eye-stalks spawn as locked (fixed-joint)
//! links (they carry the reward's eye-height marker but no `CrabJointId`); the rest
//! of the rig (shell, palpi, mouthparts) rides the carapace. Add articulation by
//! adding a `CrabJointId` variant and a [`super::rig`] `JointSpec`.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::meshfit::LoadedModel;
use super::rig::{self, RigRecipe};

// ---------------------------------------------------------------------------
// Collision groups
// ---------------------------------------------------------------------------

/// Bit 0 (`GROUP_1`): the arena (ground + walls). Every crab — whichever env —
/// must contact it, so its filter is "every group except my own" ([`Group::ALL`]
/// minus arena): the arena collides with all per-env crab bits and the nested-link
/// bit without having to enumerate them. Collision is an AND of both directions, so
/// the arena naming a group is what lets a part on that group touch the ground at all.
pub const ARENA_COLLISION: CollisionGroups =
    CollisionGroups::new(Group::GROUP_1, Group::ALL.difference(Group::GROUP_1));

/// Bit reserved for carapace-NESTED links: links whose collider center falls inside
/// the carapace box — on this model the actuated coxa/claw-shoulder (and the
/// eye-stalks) that ride just under the shell. Membership is purely geometric (see
/// `spawn_crab`), so it catches actuated joints, not only locked links. They keep
/// their mass but collide only with the arena — never with the carapace or each
/// other. A link jammed inside the carapace collider just fights the solver every
/// tick (the near-massless pincers ring it as rest jitter, bddap/rl#20), and
/// `no_adjacent_contacts` can't filter it because its joint parent is another nested
/// link, not the carapace. One shared bit across all envs is fine: nested links only
/// ever touch the arena, never another crab's parts.
pub const NESTED_COLLISION: CollisionGroups = CollisionGroups::new(Group::GROUP_2, Group::GROUP_1);

/// Highest env bit a crab may occupy (`GROUP_3`..`GROUP_18` ⇒ envs 0..15), which is
/// why `--envs` is capped at 16: env `e` takes bit `1 << (e + 2)` and we must stay
/// inside `Group`'s 32 bits with room for the arena (bit 0) and nested (bit 1) bits.
pub const MAX_ENVS: usize = 16;

// Envs occupy bits 2..=MAX_ENVS+1 of `Group`'s 32, so the highest env bit must fit.
// A compile-time guarantee, so raising MAX_ENVS past the budget fails the build
// instead of silently truncating env `e`'s membership bit at runtime.
const _: () = assert!(MAX_ENVS + 2 <= 32);

/// Collision membership for env `e`'s ordinary (non-nested, distal) crab parts.
///
/// **Each env gets its OWN bit**, so a crab's distal limbs collide with the arena
/// and with that SAME crab's other distal limbs — preserving self-collision (without
/// it the policy "tucks" legs through one another: free interpenetration is an
/// exploit, not a stance) — but NOT with any other env's crab, which keeps each env a
/// physically independent RL problem even as the M crabs walk across one shared arena
/// toward far targets and would otherwise plow into each other. Joint-adjacent
/// segments are separately contact-filtered (`no_adjacent_contacts`).
///
/// `e` must be `< MAX_ENVS`; the `--envs` clap range (1..=16) guarantees it.
pub fn crab_collision(env: usize) -> CollisionGroups {
    debug_assert!(
        env < MAX_ENVS,
        "env {env} exceeds the {MAX_ENVS}-env bit budget"
    );
    // Env 0 → GROUP_3 (bit 2); arena=bit 0, nested=bit 1 are reserved below it.
    let bit = Group::from_bits_truncate(1 << (env + 2));
    CollisionGroups::new(bit, Group::GROUP_1.union(bit))
}

/// Disable contacts between the two segments a joint connects. The joint
/// already constrains that pair, and their colliders overlap at the anchor by
/// construction — contacts there would only fight the articulation. All other
/// (non-adjacent) parts of the SAME crab DO collide; see [`crab_collision`].
fn no_adjacent_contacts(joint: impl Into<TypedJoint>) -> TypedJoint {
    let mut joint = joint.into();
    let generic: &mut GenericJoint = joint.as_mut();
    generic.set_contacts_enabled(false);
    joint
}

/// Signed joint coordinate read off the two links' world orientations: the twist
/// of child-in-parent about the joint's free `axis_local` (the rig-derived axis
/// the actuator drives). Shared by the observation and the telemetry overlay so
/// both report the same angle the policy acts on.
pub fn joint_angle(axis_local: Vec3, parent_rot: Quat, child_rot: Quat) -> f32 {
    let q = (parent_rot.inverse() * child_rot).normalize();
    let v = Vec3::new(q.x, q.y, q.z);
    2.0 * v.dot(axis_local).atan2(q.w)
}

// ---------------------------------------------------------------------------
// Dimensions — tuned for a ~1m wide crab (game scale, not real life)
// ---------------------------------------------------------------------------

/// Clearance the body is lifted above its bind pose at spawn, so it drops the
/// last bit onto its feet rather than starting interpenetrating the ground. The
/// body otherwise spawns in the glTF bind-world frame (feet already near y=0), so
/// this is a small drop, not the full standing height — that height comes from the
/// bind pose itself.
pub const SPAWN_HEIGHT: f32 = 0.05;

// ---------------------------------------------------------------------------
// Torque ceilings — the magnitude an action of ±1 commands on each joint type.
// DIRECT-DRIVE (the policy's output IS the torque) and GRADED BY INERTIA: a joint's
// snappiness is torque/inertia, so a flat ceiling would make a feather-light distal
// link a hair-trigger while staying modest on the hip. Each ceiling is still several×
// the ~0.5 N·m a joint needs to bear its share of the body, so standing keeps ample
// authority; only the surplus that fed mid-air spin-up is cut.
// ---------------------------------------------------------------------------

const COXA_TORQUE_CEILING: f32 = 6.0;
const FEMUR_TORQUE_CEILING: f32 = 4.0;
const TIBIA_TORQUE_CEILING: f32 = 2.5;
// Cheliped joints, weighted by relative muscle mass: the closer (pincer) is the
// dominant muscle of the claw, the wrist the smallest. Grapsus is a grazer with
// modest symmetric claws, so these stay within the legs' range — not crusher-scale.
const CLAW_PINCER_TORQUE_CEILING: f32 = 7.0;
const CLAW_SHOULDER_TORQUE_CEILING: f32 = 4.0;
const CLAW_WRIST_TORQUE_CEILING: f32 = 3.0;

// Cheliped shoulder travel is ASYMMETRIC about the bind rest, and a single source for
// both sides (`CrabJointId::limits` matches `ClawShoulder(_)`). +θ swings the arm DOWN
// toward the ground, keeping the full reach the last-metre grab needs; −θ LIFTS it. The
// old symmetric −1.0 up-stop let the palm swing to y≈0.75 — up through the shell top
// (≈0.61) and past the eye-stalks (≈0.52), the owner-reported "arms lift up through the
// carapace". −0.6 rad raises the claw to about eye/shell-top level (defensive reach
// kept) while the palm — the higher of the two reach effectors — stays below the shell.
// Read off the bind-pose geometry; pinned by `rig::tests::shoulder_upswing_stays_below_carapace`.
const CLAW_SHOULDER_UP_STOP: f32 = -0.6;
const CLAW_SHOULDER_DOWN_STOP: f32 = 1.0;

/// Slope of the velocity-0 friction motor's ramp into its force cap
/// ([`CrabJointId::friction_cap`]): steep enough to saturate near-instantly into a
/// small CONSTANT opposing torque (dry/Coulomb friction, not a viscous damping gain).
/// The motor has stiffness 0 (no position servo — the policy commands all torque via
/// `ExternalForce`), and is ForceBased so the cap is an honest N·m, not rescaled by
/// the link's effective mass. Friction MUST live on the joint: Rapier's per-body
/// `Damping` is a no-op on multibody links — only joint constraints and external
/// forces reach them (#14).
const FRICTION_RAMP: f32 = 4.0;

/// Breakaway torque of the joint-friction motor (see [`FRICTION_RAMP`]): the constant
/// an external load must exceed to back-drive the joint. Kept WELL below the ~0.5 N·m
/// needed to hold a leg (~1/5) so legs crumple under the lightest ground load instead
/// of propping the body up, yet nonzero because real joints have a little stiction.
pub const LEG_FRICTION_CAP: f32 = 0.04;
const CLAW_FRICTION_CAP: f32 = 0.04;

/// Spring (natural frequency Hz, damping ratio) of every revolute joint's
/// constraint — the lock holding the limb attached AND the hard end stops. Softened
/// well below Rapier's near-rigid `1e6` Hz default because at that stiffness a limb
/// driven into its limit overshoots and the violent position-correction snapping it
/// back is NOT momentum-conserving in the reduced-coordinate multibody solver: with
/// 30 joints slammed at once the residual accumulates into net angular momentum,
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
// Marker components for querying
// ---------------------------------------------------------------------------

/// The crab body recipe, derived once at startup. Held in a resource because
/// episode resets RESPAWN the whole crab (a teleport keeps the dying pose's joint
/// angles, which interpenetrate under self-collision and explode), so every spawn
/// re-instantiates this. The visible body is the skinned glTF and the colliders
/// are Rapier's debug-render; there are no per-body meshes to cache here.
#[derive(Resource)]
pub struct CrabAssets {
    /// The body to spawn. `RigRecipe`, not `Option`: preflight rejects a model that
    /// builds no recipe and an absent model falls back ([`rig::fallback_recipe`]), so
    /// construction always yields one.
    recipe: RigRecipe,
}

impl CrabAssets {
    /// Bind-pose world position of the leg hub the body spawns its root at. The skin
    /// reads it to place its own root in the same frame the body uses, so the two
    /// share one coordinate space (see [`super::skin::attach_skins`]).
    pub fn hub_bind_world(&self) -> Vec3 {
        self.recipe.hub_bind_world
    }
}

impl FromWorld for CrabAssets {
    fn from_world(_world: &mut World) -> Self {
        // A present-but-broken model is rejected by main's preflight (not silently
        // swapped for the stand-in), so this expect only fires for a future caller that
        // skips that preflight; no model at all falls back to the procedural stand-in.
        let recipe = match super::meshfit::model_path() {
            Some(p) => LoadedModel::load(&p)
                .ok()
                .and_then(|m| rig::build_recipe(&m))
                .expect("model preflight should have rejected a model that builds no recipe"),
            None => rig::fallback_recipe(),
        };
        Self { recipe }
    }
}

#[derive(Component)]
pub struct CrabCarapace;

/// Marker on the eye-tip link (the bone the eye rides). The reward reads its world
/// height (DeepMind-`stand`-style head height); the eye-stalks are locked, so they
/// carry no `CrabJoint` and this marker is how the reward locates them.
#[derive(Component)]
pub struct CrabEyeTip;

/// Marker on each claw's movable-finger link (the [`CrabJointId::ClawPincer`]
/// link — the dactyl that folds off the palm). The target-touch reward reads
/// these links' world positions as the crab's reach effectors, so it locates them
/// by marker rather than re-deriving "which link is the claw tip" from joint ids.
#[derive(Component)]
pub struct CrabClawTip;

/// Marker applied to ALL crab body parts (carapace + limb segments).
#[derive(Component)]
pub struct CrabBodyPart;

/// A part's REST (bind) world transform, captured at spawn before any physics
/// settle. The skin pairs its bones against this, not the live (already-settling)
/// transform, so the visual mesh reproduces the bind pose exactly and then tracks
/// the physics faithfully — without baking the limp body's sag into every bone (the
/// sag was the source of the skin riding above the colliders). A respawn re-creates
/// the identical rest, so the captured offsets stay valid across episode resets.
#[derive(Component, Clone, Copy)]
pub struct CrabRestPose(pub Transform);

/// Which training environment (crab instance) an entity belongs to. Every crab
/// entity carries one; systems group by it so N crabs sharing the world stay
/// independent samples. Demo/screenshot run a single env 0.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrabEnvId(pub usize);

/// A policy-driven joint on the crab: its observation/action slot key ([`id`](Self::id))
/// plus the per-instance data the sensor and actuator read. The free axis is
/// rig-derived — it varies per leg/side with the bind-pose bone geometry — so it
/// rides on the component, not a type-level constant. Locked rig links (the
/// eye-stalks) carry NO `CrabJoint`, so they are invisible to the policy: present
/// in the physics, out of the action space.
#[derive(Component, Clone, Copy, Debug)]
pub struct CrabJoint {
    pub id: CrabJointId,
    /// Free axis as a unit vector in the PARENT link's frame — the vector the
    /// actuator rotates into world to apply torque, and the sensor projects
    /// relative motion onto to read the DOF rate. Derived at spawn from the rig.
    pub axis_local: Vec3,
}

/// Every POLICY-ACTUATED joint — the locomotion-relevant subset of the crab's
/// articulation. The body also spawns the eye-stalks as locked links with no
/// [`CrabJoint`]; restoring finer articulation (bddap/rl#31) means adding a variant
/// here (which grows the observation/action vector and the net) plus a rig
/// [`super::rig::JointSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CrabJointId {
    // Legs (8 = side L/R × leg 0..3 front→back). Coxa swings the leg off the
    // body; merus and carpus are the two load-bearing bends down the limb.
    LegCoxa(Side, u8),
    LegMerus(Side, u8),
    LegCarpus(Side, u8),
    // Claws (1 per side): shoulder lifts the arm, wrist bends the hand, pincer
    // opens/closes the movable finger.
    ClawShoulder(Side),
    ClawWrist(Side),
    ClawPincer(Side),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Side {
    Left,
    Right,
}

impl CrabJointId {
    /// Every policy-actuated joint, in observation/action slot order — the ONE
    /// source of truth from which [`index`](Self::index) and [`COUNT`](Self::COUNT)
    /// derive, so adding a variant means editing only this list (and a wrong array
    /// length is a compile error, not the runtime slot-aliasing the bijection test
    /// once had to catch). The order is load-bearing: PPO obs/action vectors and
    /// saved checkpoints are keyed by a joint's position here, so reordering would
    /// invalidate trained checkpoints. Legs first (front→back per side, both
    /// sides), then claws (per side) — left side ahead of right within each.
    const fn all() -> [CrabJointId; 30] {
        [
            CrabJointId::LegCoxa(Side::Left, 0),
            CrabJointId::LegMerus(Side::Left, 0),
            CrabJointId::LegCarpus(Side::Left, 0),
            CrabJointId::LegCoxa(Side::Left, 1),
            CrabJointId::LegMerus(Side::Left, 1),
            CrabJointId::LegCarpus(Side::Left, 1),
            CrabJointId::LegCoxa(Side::Left, 2),
            CrabJointId::LegMerus(Side::Left, 2),
            CrabJointId::LegCarpus(Side::Left, 2),
            CrabJointId::LegCoxa(Side::Left, 3),
            CrabJointId::LegMerus(Side::Left, 3),
            CrabJointId::LegCarpus(Side::Left, 3),
            CrabJointId::LegCoxa(Side::Right, 0),
            CrabJointId::LegMerus(Side::Right, 0),
            CrabJointId::LegCarpus(Side::Right, 0),
            CrabJointId::LegCoxa(Side::Right, 1),
            CrabJointId::LegMerus(Side::Right, 1),
            CrabJointId::LegCarpus(Side::Right, 1),
            CrabJointId::LegCoxa(Side::Right, 2),
            CrabJointId::LegMerus(Side::Right, 2),
            CrabJointId::LegCarpus(Side::Right, 2),
            CrabJointId::LegCoxa(Side::Right, 3),
            CrabJointId::LegMerus(Side::Right, 3),
            CrabJointId::LegCarpus(Side::Right, 3),
            CrabJointId::ClawShoulder(Side::Left),
            CrabJointId::ClawWrist(Side::Left),
            CrabJointId::ClawPincer(Side::Left),
            CrabJointId::ClawShoulder(Side::Right),
            CrabJointId::ClawWrist(Side::Right),
            CrabJointId::ClawPincer(Side::Right),
        ]
    }

    /// Total policy-actuated DOFs — sets the observation/action vector width and
    /// the net's input/output size. Locked rig joints are excluded (no [`CrabJoint`]).
    pub const COUNT: usize = Self::all().len();

    /// Flat observation/action slot for this joint (0..COUNT): its position in
    /// [`all`](Self::all). A bijection because `all` lists each joint once; the
    /// `index_is_a_bijection` test pins that no variant is duplicated there. O(COUNT)
    /// and never on a hot path.
    pub fn index(&self) -> usize {
        Self::all()
            .iter()
            .position(|j| j == self)
            .expect("every CrabJointId is listed in all()")
    }
}

impl CrabJointId {
    /// Peak DIRECT-DRIVE torque an action of ±1 commands on this joint — the
    /// magnitude the actuator applies via `ExternalForce`, graded by inertia
    /// (weakest on the lightest, most distal links). The policy's drive authority,
    /// distinct from the joint's friction motor ([`Self::friction_cap`]).
    pub fn drive_torque_ceiling(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(..) => COXA_TORQUE_CEILING,
            CrabJointId::LegMerus(..) => FEMUR_TORQUE_CEILING,
            CrabJointId::LegCarpus(..) => TIBIA_TORQUE_CEILING,
            CrabJointId::ClawShoulder(_) => CLAW_SHOULDER_TORQUE_CEILING,
            CrabJointId::ClawWrist(_) => CLAW_WRIST_TORQUE_CEILING,
            CrabJointId::ClawPincer(_) => CLAW_PINCER_TORQUE_CEILING,
        }
    }

    /// Breakaway torque of this joint's friction motor (see [`FRICTION_RAMP`]):
    /// the constant an external load must beat to back-drive the joint, so legs
    /// crumple under ground contact and a modest command still actuates.
    pub fn friction_cap(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(..) | CrabJointId::LegMerus(..) | CrabJointId::LegCarpus(..) => {
                LEG_FRICTION_CAP
            }
            CrabJointId::ClawShoulder(_)
            | CrabJointId::ClawWrist(_)
            | CrabJointId::ClawPincer(_) => CLAW_FRICTION_CAP,
        }
    }

    /// Joint travel limits `[lo, hi]` in radians about the rig BIND POSE: the spawn
    /// bakes each link's bind orientation onto Rapier coordinate 0, so the range
    /// straddles 0 (0 = the bone's rest angle in the model). The policy commands
    /// torque, not position, so these are the hard stops the limb cannot pass, not
    /// a target. Per-family defaults, refined once the body stands.
    pub fn limits(&self) -> [f32; 2] {
        match self {
            CrabJointId::LegCoxa(..) => [-0.8, 0.8],
            CrabJointId::LegMerus(..) => [-1.0, 1.0],
            CrabJointId::LegCarpus(..) => [-1.1, 1.1],
            CrabJointId::ClawShoulder(_) => [CLAW_SHOULDER_UP_STOP, CLAW_SHOULDER_DOWN_STOP],
            // Tuned wrist sweep, clamped tight: ±13.7° (±0.239110 rad) about the bind
            // rest — much narrower than the other joints because the wrist rotated too far.
            CrabJointId::ClawWrist(_) => [-0.239110, 0.239110],
            CrabJointId::ClawPincer(_) => [-0.5, 0.2],
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn the crab — instantiate the rig-derived recipe
// ---------------------------------------------------------------------------

/// Spawns a complete crab body at the given position, instantiating the
/// rig-derived [`RigRecipe`] (the one model). Returns the carapace entity.
///
/// Links are emitted parent-before-child so each joint can reference its already-
/// spawned parent entity. At rest every link is axis-aligned, so a link's world
/// position is just its parent's plus the joint anchor — tracked here to seed each
/// body's initial `Transform` near the pose the multibody solver will hold.
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
            // are shown by Rapier's debug-render (RL_DEBUG_COLLIDERS). A primitive mesh
            // here only risked drifting out of sync with the actual collider.
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
        let joint = match link.actuated {
            Some(id) => rig_revolute(id, link.axis_local, link.anchor1),
            None => rig_fixed(link.anchor1),
        };
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
        if let Some(id) = link.actuated {
            ec.insert(CrabJoint {
                id,
                axis_local: link.axis_local,
            });
            // The movable-finger link is a claw tip (see [`CrabClawTip`]).
            if matches!(id, CrabJointId::ClawPincer(_)) {
                ec.insert(CrabClawTip);
            }
        }
        // Grippy feet: the distal leg bone (`004`) is what plants on the ground.
        if link.bone.starts_with("Def_leg") && link.bone.contains(".004.") {
            ec.insert(Friction::coefficient(1.5));
        }
        // The eye rides the stalk tip — mark it so the reward can read eye height.
        if link.bone.starts_with("Def_antennae_top") {
            ec.insert(CrabEyeTip);
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

/// Fixed joint for a locked rig link: welds the child to its parent at the bind
/// pose (no DOF), so the joint is present in the body — promotable to actuated by
/// adding a [`CrabJointId`] for it — but invisible to the policy.
fn rig_fixed(anchor1: Vec3) -> TypedJoint {
    no_adjacent_contacts(
        FixedJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(Vec3::ZERO),
    )
}

/// A capsule collider offset within its link: the bone runs from the pivot (link
/// origin) along `rot·Y`, so the shape sits centred at `center` and oriented to
/// match — the same offset-baked-into-the-shape trick the fitted path uses.
fn capsule_collider(center: Vec3, rot: Quat, half_height: f32, radius: f32) -> Collider {
    let axis = rot * Vec3::Y * half_height;
    Collider::capsule(center - axis, center + axis, radius)
}

// ---------------------------------------------------------------------------
// Debug: joint-pivot markers (RL_DEBUG_COLLIDERS)
// ---------------------------------------------------------------------------
// Render-only: gizmos draw through a camera, so this whole block is dead in the
// headless trainer and its types (Gizmos/GizmoConfig) don't exist without bevy's
// render feature. Gated out there; the demo/screenshot bins (render on) keep it.

/// A separate gizmo config group so the pivot markers can force `depth_bias = -1.0`
/// (always-in-front) WITHOUT changing how Rapier's collider wireframes render —
/// Rapier draws into the default group, which we leave depth-tested so the cage
/// still reads as 3D. The pivots sit buried inside the opaque skin, so without the
/// override every marker would be hidden by the body and the screenshot would prove
/// nothing.
#[cfg(feature = "render")]
#[derive(Default, Reflect, GizmoConfigGroup)]
#[reflect(Default)]
pub struct PivotGizmos;

/// Marker sphere radius (model units): big enough to spot against the skin yet
/// small enough not to swallow the joint it marks.
#[cfg(feature = "render")]
const PIVOT_MARKER_RADIUS: f32 = 0.02;

/// Draw a bright sphere at every physics link's world origin — which, by
/// construction, IS that link's joint pivot (`spawn_crab` anchors each child at its
/// parent with `local_anchor2 = ZERO`), plus the carapace root. Magenta to stay
/// distinct from both the Sally model's orange skin and Rapier's collider
/// wireframes. `GlobalTransform`, not `Transform`: a marker must sit at the pivot's
/// true world point even mid-tumble, and only the global has the full parent chain
/// resolved. Always-in-front comes from [`PivotGizmos`]'s `depth_bias`.
#[cfg(feature = "render")]
fn draw_pivot_markers(
    parts: Query<&GlobalTransform, With<CrabBodyPart>>,
    mut gizmos: Gizmos<PivotGizmos>,
) {
    let color = Color::srgb(1.0, 0.0, 1.0); // magenta
    for gt in &parts {
        gizmos.sphere(
            Isometry3d::from_translation(gt.translation()),
            PIVOT_MARKER_RADIUS,
            color,
        );
    }
}

/// Wire up the joint-pivot debug markers. Called from `main` alongside the Rapier
/// collider debug-render and behind the same `RL_DEBUG_COLLIDERS` gate, so the two
/// physics-truth overlays — collider cages and the pivots they hinge about — appear
/// together. Drawn through whatever camera renders the gizmos (the windowed demo's
/// or the offscreen screenshot's), so it shows up in `--screenshot` too.
#[cfg(feature = "render")]
pub fn register_pivot_markers(app: &mut App) {
    app.insert_gizmo_config(
        PivotGizmos,
        GizmoConfig {
            // -1.0 = always render in front; the pivots are inside the opaque body.
            depth_bias: -1.0,
            ..default()
        },
    );
    app.add_systems(Update, draw_pivot_markers);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `index` must be a bijection onto `0..COUNT`. Slots key every action,
    /// observation, and per-joint array, so a duplicated variant in [`all`] would
    /// give two joints the same `position` — one never actuated, the other twice —
    /// and the model would still "train". `all` being the lone source now makes
    /// that a duplicate-list bug rather than three drifting copies, but pin it
    /// regardless: every joint maps to a distinct slot covering the whole range.
    #[test]
    fn index_is_a_bijection() {
        let mut seen = [false; CrabJointId::COUNT];
        let mut count = 0;
        for id in CrabJointId::all() {
            let i = id.index();
            assert!(i < CrabJointId::COUNT, "{id:?} index {i} out of range");
            assert!(!seen[i], "{id:?} aliases slot {i}");
            seen[i] = true;
            count += 1;
        }
        assert_eq!(
            count,
            CrabJointId::COUNT,
            "all() yielded the wrong joint count"
        );
        assert!(seen.iter().all(|&s| s), "index leaves a slot unfilled");
    }
}
