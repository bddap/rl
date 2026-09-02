use bevy::prelude::*;

use super::joint_id::CrabJointId;
use crate::bot::rig::{self, RigRecipe};

#[derive(Resource)]
pub struct CrabAssets {
    pub(super) recipe: RigRecipe,
}

impl CrabAssets {
    pub fn hub_bind_world(&self) -> Vec3 {
        self.recipe.hub_bind_world
    }
}

/// The SKIN asset in its ASSET-TREE form (ready for `AssetServer::load`), `None`
/// unless a digest-matched sally.glb resolved
/// ([`crate::mesh_fallback::usable_model_path`]). Visual-only: physics builds
/// [`rig::baked_recipe`] regardless (rl#340 stage 10) — this only decides whether
/// the skin loads over it.
#[derive(Resource, Clone)]
pub struct CrabModelPath(pub Option<std::path::PathBuf>);

impl FromWorld for CrabModelPath {
    fn from_world(_world: &mut World) -> Self {
        Self(crate::mesh_fallback::usable_model_path())
    }
}

impl FromWorld for CrabAssets {
    fn from_world(_world: &mut World) -> Self {
        Self {
            recipe: rig::baked_recipe(),
        }
    }
}

#[derive(Component)]
pub struct CrabCarapace;

/// Whole-rig mass (kg) on the carapace, summed at spawn from the very colliders the
/// solver integrates, so every body-weight-relative force reads one number that
/// cannot drift from the rig actually spawned.
#[derive(Component, Clone, Copy)]
pub struct CrabBodyMass(pub f32);

#[derive(Component)]
pub struct CrabClawTip;

/// A rapier-driven rigid-body part of the crab. Its `Transform` belongs to the
/// physics solver: in any world that pumps `FixedUpdate`, a foreign write gets
/// synced back into the body at `SyncBackend` and blows up the multibody (the
/// GCR play-day crash, rl#116). Cosmetic/render placement rides the render-only
/// skin bones / `CrabRenderPose` instead; [`crate::bot::pose_sentinel`] enforces
/// this at runtime in visual worlds. Render consumers must also not READ this
/// `Transform` as the rendered pose directly — resolve it through
/// [`crate::bot::skin::CrabRenderPose::rendered`], or the 64:30 step staircase
/// leaks back into rendered motion (rl#274).
#[derive(Component)]
pub struct CrabBodyPart;

#[derive(Component, Clone, Copy)]
pub struct CrabRestPose(pub Transform);

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrabEnvId(pub usize);

#[derive(Component, Clone, Copy, Debug)]
pub struct CrabJoint {
    pub id: CrabJointId,
    pub axis_local: Vec3,
}
