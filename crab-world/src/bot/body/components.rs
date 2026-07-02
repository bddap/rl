//! The body's ECS surface: the marker components every crab part carries, the
//! per-instance [`CrabJoint`] the sensor/actuator read, and [`CrabAssets`] — the
//! once-derived [`RigRecipe`] resource each spawn re-instantiates. [`render_recipe`]
//! is the ONE place the model-vs-fallback body selection lives.

use bevy::prelude::*;

use super::joint_id::CrabJointId;
use crate::bot::rig::{self, RigRecipe};

/// The crab body recipe, derived once at startup. Held in a resource because
/// episode resets RESPAWN the whole crab (a teleport keeps the dying pose's joint
/// angles, which interpenetrate under self-collision and explode), so every spawn
/// re-instantiates this. The visible body is the skinned glTF and the colliders
/// are Rapier's debug-render; there are no per-body meshes to cache here.
#[derive(Resource)]
pub struct CrabAssets {
    /// The body to spawn. `RigRecipe`, not `Option`: preflight rejects a model that
    /// builds no recipe and an absent model falls back ([`rig::fallback_recipe`]), so
    /// construction always yields one. `pub(super)` so `spawn::spawn_crab` reads it.
    pub(super) recipe: RigRecipe,
}

impl CrabAssets {
    /// Bind-pose world position of the leg hub the body spawns its root at. The skin
    /// reads it to place its own root in the same frame the body uses, so the two
    /// share one coordinate space (see [`crate::bot::skin::pairing::attach_skins`]).
    pub fn hub_bind_world(&self) -> Vec3 {
        self.recipe.hub_bind_world
    }
}

/// The crab glTF THIS app renders ("which crab does this app show?"), decided ONCE at
/// construction: `Some(absolute path)` = the preflighted real Sally mesh; `None` = no usable
/// model, render the honest procedural/collider fallback. [`CrabAssets`] (physics body) and
/// [`crate::bot::skin::register`] (cosmetic skin) BOTH read this one resource, so a surface that has
/// preflighted "no Sally" flips body + skin together off a single explicit value.
///
/// WHY a resource and not a `CRAB_MODEL_PATH` env override: poisoning a global env var to
/// force the fallback is spooky action at a distance every other reader of the resolver
/// feels (bddap/rl#147); an explicit value can't mislead a third reader.
///
/// Defaults (FromWorld) to the shared preflight verdict [`crate::mesh_fallback::usable_model_path`],
/// so a present-but-unloadable glb resolves to `None` here (the honest fallback); the body then
/// spawns the fallback recipe, never a broken real one (bddap/rl#154). SCOPE: this resource is a `BotPlugin` resource —
/// only present where the bot stack is. `net::render`'s collider silhouette runs WITHOUT that stack
/// (the unarmed screenshot), so it reads the same [`crate::mesh_fallback::usable_model_path`] verdict
/// directly; since net never overrides the resolver, its reads and this resource's default agree.
/// rl-demo inserts its own `CrabModelPath` from the same verdict (it also needs the `Err` reason for
/// its banner). The *world-render scale* question ("how big is the crab THIS run draws?",
/// `net::render::world_render_scale`) reads the same verdict too, so on a broken glb it sizes the
/// world to the fallback height that's actually rendered — render and scale can't disagree.
#[derive(Resource, Clone)]
pub struct CrabModelPath(pub Option<std::path::PathBuf>);

impl FromWorld for CrabModelPath {
    fn from_world(_world: &mut World) -> Self {
        // The full preflight, NOT bare `model_path()`: a present-but-unloadable `sally.glb` resolves
        // to `None` here so the body degrades to the fallback recipe (bddap/rl#154). This is the
        // game's equivalent of rl-demo's explicit preflighted `CrabModelPath` insert — the same one
        // verdict (`mesh_fallback::usable_model`), which also OWNS the recipe `render_recipe` spawns.
        Self(crate::mesh_fallback::usable_model_path())
    }
}

/// The body recipe the game renders and spawns: the fitted real-Sally recipe when the mesh is
/// usable, else the procedural stand-in. The ONE place the model-vs-fallback BRANCH lives —
/// [`CrabAssets`] (the spawned/skinned body, fed its [`CrabModelPath`]) and the integer-crab
/// collider silhouette (`net::render::spawn_crab_silhouette`) both go through here, so the two can't
/// draw different geometry.
///
/// `has_model` is the surface's [`CrabModelPath`] flip: `true` → render the real crab, `false` → the
/// stand-in. When `true`, the recipe is the one the memoized [`crate::mesh_fallback::usable_model`]
/// verdict ALREADY built and validated — cloned out, NOT re-parsed from the 36 MB glb (bddap/rl#153:
/// the load+fit happens exactly ONCE, in the verdict; there is no second "is the mesh good?" check
/// here). A `bool`, not a
/// path: the caller can only say real-or-fallback, never hand in a path that DISAGREES with the one
/// asset the verdict validated — so the flip can't drift from the recipe. `has_model` co-derives with
/// the verdict (`usable_model_path().is_some()`), so the `Err` arm is unreachable but degrades to the
/// honest fallback rather than panicking.
pub fn render_recipe(has_model: bool) -> RigRecipe {
    if has_model {
        match crate::mesh_fallback::usable_model() {
            Ok(u) => u.recipe.clone(),
            Err(_) => rig::fallback_recipe(),
        }
    } else {
        rig::fallback_recipe()
    }
}

impl FromWorld for CrabAssets {
    fn from_world(world: &mut World) -> Self {
        // The body flips off the same `CrabModelPath` the skin reads (so the two can't disagree about
        // which crab is present); the fitted recipe itself comes from the memoized verdict.
        let has_model = world.resource::<CrabModelPath>().0.is_some();
        Self {
            recipe: render_recipe(has_model),
        }
    }
}

#[derive(Component)]
pub struct CrabCarapace;

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

/// A policy-driven joint on the crab: its observation/action slot key ([`CrabJointId::index`])
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
