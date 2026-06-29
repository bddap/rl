//! Derives the crab's physics body from the bind-pose skeleton of the glTF model:
//! one rigid link per actuated joint, each with a capsule collider fitted to that
//! joint's vertex cloud, plus a carapace box for the trunk. This is the single
//! source of the body's geometry, so the physics, the visual skin, and the
//! collider fit all share one coordinate space.
//!
//! [`joint_specs`](recipe::joint_specs) is the canonical decomposition of the rig
//! into physics parts: every consumer — the body spawn here, the skin
//! ([`super::skin`]), and the per-part vertex bucketing ([`super::meshfit`]) —
//! derives its bone→part view from it via [`part_for_bone`]. They can't drift,
//! because there is only one mapping; mismatched hand-kept copies of it are what
//! forked the rendered legs.
//!
//! A leg's six rig bones collapse to four physics links: the basal joint is 2-DOF,
//! split into a coxa (`000`/`000b`) that swings the leg fore/aft at the body and a
//! basis (`001`/`002`) that lifts it up/down just distal, then the merus and carpus
//! bend. Each link carries real flesh — no massless virtual link — and the bones
//! between the named pivots ride their link rather than spawning as locked stubs.
//!
//! The file is split by concern: [`recipe`] builds the [`RigRecipe`] from a bind
//! pose, [`fallback`] is the no-asset procedural body, and [`colliders`]
//! reconstructs the rest-pose collider shapes for scoring and rendering. The shared
//! vocabulary — the [`BindSource`] trait and the [`RigLink`]/[`RigRecipe`] data —
//! lives here.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::bot::body::CrabJointId;
use crate::bot::meshfit::{LoadedModel, PartId};

mod colliders;
mod fallback;
mod recipe;

pub use colliders::{
    CrabSilhouette, RestCollider, RestShape, recipe_silhouette, rest_colliders,
};
pub use fallback::fallback_recipe;
pub use recipe::{TRUNK_BONES, build_recipe, part_for_bone, parts_adjacent};
pub(crate) use recipe::link_world_origins;

/// Bind-pose geometry source, so the real [`LoadedModel`] and the procedural
/// [`FallbackModel`] share ONE recipe-builder ([`build_recipe`]) — no second spawn
/// path to drift.
pub trait BindSource {
    /// Bind-pose world origin of a bone by name (the joint pivot), if present.
    fn bone_origin(&self, name: &str) -> Option<Vec3>;
    /// Per-part flesh: world-space vertex cloud for each physics part. Drives the
    /// fitted capsule radius.
    fn vertices_by_part(&self) -> HashMap<PartId, Vec<Vec3>>;
    /// World-space vertices whose dominant bone is one of `names` — the trunk flesh
    /// the carapace box is sized from.
    fn vertices_for_bones(&self, names: &[&str]) -> Vec<Vec3>;
    /// Radius to use for a part whose cloud is too sparse to fit a capsule to. The
    /// real model has dense flesh so it returns `None`; the procedural body has no
    /// flesh and supplies its intended per-part thickness here so its limbs aren't
    /// pencil-thin.
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

/// One physics link to spawn, fully derived from the bind pose. Links are emitted
/// parent-before-child, so `parent` indexes an earlier entry.
pub struct RigLink {
    /// The deform bone at this link's joint pivot (skin mapping + debugging).
    pub bone: String,
    /// Index of the parent link in [`RigRecipe::links`], or `None` when the link
    /// hangs off the carapace root — so "root" can't masquerade as a real index.
    pub parent: Option<usize>,
    /// Joint pivot relative to the parent link's origin. Links spawn axis-aligned
    /// (identity orientation) at rest, so the parent frame is world and this is a
    /// plain world delta between bind-pose bone origins. (Reproducing each bone's
    /// bind *orientation* — needed to skin the visual mesh — is phase 2; at rest
    /// the capsule geometry below already points down the segment.)
    pub anchor1: Vec3,
    /// Free rotation axis in world (= the parent frame at rest; only meaningful
    /// when `actuated`).
    pub axis_local: Vec3,
    /// Capsule cylinder half-length (caps excluded) and radius.
    pub half_height: f32,
    pub radius: f32,
    /// Collider centre + orientation in the CHILD link frame (pivot at its origin).
    pub center: Vec3,
    pub col_rot: Quat,
    pub density: f32,
    /// `Some` → policy-actuated: carries a `CrabJoint`, a revolute joint, and an
    /// observation/action slot. `None` → locked (fixed joint, no `CrabJoint`,
    /// invisible to the policy — the eye-stalks).
    pub actuated: Option<CrabJointId>,
}

/// The full body recipe: a carapace root box plus the derived link chain.
pub struct RigRecipe {
    /// Bind-pose world position of the leg-hub centroid — the point the link chain
    /// anchors off (`anchor1` deltas telescope back to it). The body spawns its
    /// carapace root here (plus the spawn point) so every link lands at its true
    /// glTF bind-world origin, the same frame the cosmetic skin renders its bones
    /// in. Without it the spawn pinned the hub at an arbitrary height and lost the
    /// hub's lateral/forward bind offset, sliding the whole physics body ~0.1 off
    /// the skin (pivots fell outside the rendered mesh).
    pub hub_bind_world: Vec3,
    pub carapace_half: Vec3,
    /// Box centre relative to the body root (the leg-hub centroid the links anchor
    /// to). The trunk's bounding box isn't centred on the leg hub, so the carapace
    /// collider is offset to sit on the shell instead of engulfing the limbs.
    pub carapace_offset: Vec3,
    pub carapace_density: f32,
    pub links: Vec<RigLink>,
}

impl RigRecipe {
    /// Every geometric field across the recipe is finite. The spawn feeds these
    /// straight into Rapier; a single NaN/inf reaches the solver as an
    /// unrecoverable crash on the very first step, so this is the gate that keeps a
    /// degenerate asset from ever getting that far.
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
