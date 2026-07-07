use std::collections::HashMap;

use bevy::prelude::*;

use crate::bot::body::CrabJointId;
use crate::bot::meshfit::{LoadedModel, PartId};

mod colliders;
mod fallback;
mod recipe;

pub(crate) use colliders::link_capsule;
pub use colliders::{CrabSilhouette, RestCollider, RestShape, recipe_silhouette, rest_colliders};
pub use fallback::fallback_recipe;
pub(crate) use recipe::link_world_origins;
pub use recipe::{TRUNK_BONES, build_recipe, part_for_bone, parts_adjacent};

pub trait BindSource {
    fn bone_origin(&self, name: &str) -> Option<Vec3>;
    fn vertices_by_part(&self) -> HashMap<PartId, Vec<Vec3>>;
    fn vertices_for_bones(&self, names: &[&str]) -> Vec<Vec3>;
    fn radius_hint(&self, _part: PartId) -> Option<f32> {
        None
    }
}

impl BindSource for LoadedModel {
    fn bone_origin(&self, name: &str) -> Option<Vec3> {
        LoadedModel::bone_origin(self, name)
    }
    fn vertices_by_part(&self) -> HashMap<PartId, Vec<Vec3>> {
        LoadedModel::vertices_by_part(self)
    }
    fn vertices_for_bones(&self, names: &[&str]) -> Vec<Vec3> {
        LoadedModel::vertices_for_bones(self, names)
    }
}

#[derive(Clone)]
pub struct RigLink {
    pub bone: String,
    pub parent: Option<usize>,
    pub anchor1: Vec3,
    pub axis_local: Vec3,
    pub half_height: f32,
    pub radius: f32,
    pub center: Vec3,
    pub col_rot: Quat,
    pub density: f32,
    pub actuated: Option<CrabJointId>,
}

#[derive(Clone)]
pub struct RigRecipe {
    pub hub_bind_world: Vec3,
    pub carapace_half: Vec3,
    pub carapace_offset: Vec3,
    pub carapace_density: f32,
    pub links: Vec<RigLink>,
}

impl RigRecipe {
    pub(super) fn is_finite(&self) -> bool {
        self.hub_bind_world.is_finite()
            && self.carapace_half.is_finite()
            && self.carapace_offset.is_finite()
            && self.carapace_density.is_finite()
            && self.links.iter().all(RigLink::is_finite)
    }
}

impl RigLink {
    fn is_finite(&self) -> bool {
        self.anchor1.is_finite()
            && self.axis_local.is_finite()
            && self.half_height.is_finite()
            && self.radius.is_finite()
            && self.center.is_finite()
            && self.col_rot.is_finite()
            && self.density.is_finite()
    }
}
