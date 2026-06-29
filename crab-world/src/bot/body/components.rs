//! The body's ECS surface: the marker components every crab part carries, the
//! per-instance [`CrabJoint`] the sensor/actuator read, and [`CrabAssets`] — the
//! once-derived [`RigRecipe`] resource each spawn re-instantiates. [`render_recipe`]
//! is the ONE place the model-vs-fallback body selection lives.

use bevy::prelude::*;

use super::joint_id::CrabJointId;
use crate::bot::meshfit::LoadedModel;
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
/// WHY a resource and not the old way: rl-demo used to force the fallback by poisoning the
/// `CRAB_MODEL_PATH` env at a `/nonexistent` path so the global resolver would miss for every
/// reader — spooky action at a distance (a reader debugging "why is CRAB_MODEL_PATH nonsense?"
/// had to find rl-demo's main). The explicit resource replaces that (bddap/rl#147).
///
/// Defaults (FromWorld) to [`crate::bot::meshfit::model_path`], so a surface that does NOT override
/// gets the live resolution unchanged. SCOPE: this resource is a `BotPlugin` resource — only present
/// where the bot stack is. `net::render`'s collider silhouette runs WITHOUT that stack (the unarmed
/// screenshot), so it reads the global [`crate::bot::meshfit::model_path`] directly; since net never
/// overrides the resolver, its reads and this resource's default agree. Only a surface that
/// actually preflights and overrides (today: rl-demo) needs the resource. Also distinct from the
/// *world-render scale* question ("how big is the REAL crab?", `net::render::world_render_scale`),
/// which always reads the real asset — a fallback render must not resize the world.
#[derive(Resource, Clone)]
pub struct CrabModelPath(pub Option<std::path::PathBuf>);

impl FromWorld for CrabModelPath {
    fn from_world(_world: &mut World) -> Self {
        Self(crate::bot::meshfit::model_path())
    }
}

/// The body recipe the game renders and spawns from a resolved `model`: the fitted model when one
/// is present, else the procedural stand-in. The ONE place the model-vs-fallback BRANCH lives —
/// [`CrabAssets`] (the spawned/skinned body, fed its [`CrabModelPath`]) and the integer-crab
/// collider silhouette (`net::render::spawn_crab_silhouette`, fed `meshfit::model_path()`) both go
/// through here, so the two can't draw different geometry from the same input. A present-but-broken
/// model (`Some(p)` that loads to no recipe) makes the `expect` fire: callers that must not crash on
/// a bad asset run the full [`crate::mesh_fallback::canonical_mesh_status`] preflight and pass `None`
/// on failure, so a broken `Some` never reaches here (rl-demo does this + emits a LOUD OTEL error;
/// see its `main`). No model at all falls back to the procedural stand-in here directly.
pub fn render_recipe(model: Option<&std::path::Path>) -> RigRecipe {
    match model {
        Some(p) => LoadedModel::load(p)
            .ok()
            .and_then(|m| rig::build_recipe(&m))
            .expect("model preflight should have rejected a model that builds no recipe"),
        None => rig::fallback_recipe(),
    }
}

impl FromWorld for CrabAssets {
    fn from_world(world: &mut World) -> Self {
        let model = world.resource::<CrabModelPath>().0.clone();
        Self {
            recipe: render_recipe(model.as_deref()),
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
