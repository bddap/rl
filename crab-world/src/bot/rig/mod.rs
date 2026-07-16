use bevy::prelude::*;

use crate::bot::body::CrabJointId;

mod baked;
mod colliders;
mod fallback;
mod recipe;

pub use baked::{BAKED_ASSET_DIGEST, baked_recipe};
pub(crate) use colliders::link_capsule;
pub use colliders::{CrabSilhouette, RestCollider, RestShape, recipe_silhouette, rest_colliders};
pub use fallback::fallback_recipe;
pub(crate) use recipe::link_world_origins;
pub use recipe::{TRUNK_BONES, arc_to, build_recipe, part_for_bone, parts_adjacent};

/// Which physics part a skinned bone's flesh belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PartId {
    Carapace,
    Joint(CrabJointId),
}

impl PartId {
    pub fn is_rigid(self) -> bool {
        matches!(self, PartId::Carapace)
    }
}

/// A skeleton the rig recipe can be derived from: bone origins for topology and
/// anchors, a trunk vertex cloud for the carapace box. Implemented by the procedural
/// [`fallback::FallbackModel`] here and by the offline `meshfit` tool's glTF loader —
/// the runtime itself never reads mesh data for physics; it consumes
/// [`baked_recipe`] (bddap/rl#20).
pub trait BindSource {
    fn bone_origin(&self, name: &str) -> Option<Vec3>;
    /// The vertex cloud the carapace box is sized from ([`TRUNK_BONES`] flesh on a
    /// real model, synthetic box corners on the fallback).
    fn trunk_vertices(&self) -> Vec<Vec3>;
    fn radius_hint(&self, _part: PartId) -> Option<f32> {
        None
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
