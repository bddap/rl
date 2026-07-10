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

#[derive(Resource, Clone)]
pub struct CrabModelPath(pub Option<std::path::PathBuf>);

impl FromWorld for CrabModelPath {
    fn from_world(_world: &mut World) -> Self {
        Self(crate::mesh_fallback::usable_model_path())
    }
}

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
        let has_model = world.resource::<CrabModelPath>().0.is_some();
        Self {
            recipe: render_recipe(has_model),
        }
    }
}

#[derive(Component)]
pub struct CrabCarapace;

#[derive(Component)]
pub struct CrabClawTip;

/// A rapier-driven rigid-body part of the crab. Its `Transform` belongs to the
/// physics solver: in any world that pumps `FixedUpdate`, a foreign write gets
/// synced back into the body at `SyncBackend` and blows up the multibody (the
/// GCR play-day crash, rl#116). Cosmetic/render placement rides the render-only
/// skin bones / `CrabSkinRepose` instead; [`crate::bot::pose_sentinel`] enforces
/// this at runtime in visual worlds.
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
