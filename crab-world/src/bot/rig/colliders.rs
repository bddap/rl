//! Reconstruct a recipe's rest-pose collider shapes in bind-pose world: for
//! scoring the fit against the model's vertex clouds ([`rest_colliders`]) and for
//! drawing the cosmetic giant-crab silhouette ([`recipe_silhouette`]). Both walk the
//! same telescoped link origins and share one capsule definition, so the scored body
//! and the drawn one can't drift.

use bevy::prelude::*;

use crate::bot::meshfit::PartId;

use super::recipe::{leg_hub_centroid, link_world_origins};
use super::{BindSource, RigLink, RigRecipe};

/// A collider reconstructed in bind-pose world (the rest stance), paired with the
/// part whose vertex cloud it should hug. The verifier (the `rl --verify-colliders`
/// dev command) scores cloud-vs-collider, and the model's clouds live in bind-pose
/// world, so the collider must too. This mirrors the world accumulation in [`crate::bot::body`]'s
/// `spawn_crab` minus the constant spawn translation (it cancels — `anchor1` is a
/// parent-relative delta, and the clouds are already in this frame).
pub struct RestCollider {
    pub part: PartId,
    pub shape: RestShape,
    /// The link's bind-pose-world origin — where its revolute joint actually
    /// pivots (a leaf-most link's pivot is its own bone origin). This is the
    /// physical pivot the body spawns at, recovered from the same world walk that
    /// builds the shape, so it can't drift from `joint_specs`'s pivot bone names.
    /// For the carapace it's the leg-hub root the box is offset from.
    pub pivot: Vec3,
}

pub enum RestShape {
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    Cuboid { center: Vec3, half: Vec3 },
}

/// Reconstruct every scoreable collider of `recipe` in bind-pose world. Locked
/// eye-stalk links are skipped (no fitted cloud to score). The carapace box is
/// world-axis-aligned at the hub + offset.
pub fn rest_colliders(model: &impl BindSource, recipe: &RigRecipe) -> Vec<RestCollider> {
    let Some(o_root) = leg_hub_centroid(model) else {
        return Vec::new();
    };
    let world_origin = link_world_origins(&recipe.links, o_root);
    let mut out: Vec<RestCollider> = Vec::new();
    for (link, &origin) in recipe.links.iter().zip(&world_origin) {
        // Only actuated links carry a PartId and a fitted cloud; eye-stalks (locked,
        // fixed radius, cosmetic) have nothing to score against.
        if let Some(id) = link.actuated {
            out.push(RestCollider {
                part: PartId::Joint(id),
                shape: link_capsule(link, origin),
                pivot: origin,
            });
        }
    }
    out.push(RestCollider {
        part: PartId::Carapace,
        shape: carapace_cuboid(recipe, o_root),
        pivot: o_root,
    });
    out
}

/// The crab's cosmetic collider silhouette — the carapace box and a capsule for EVERY
/// link, the carapace kept as its OWN field (not the tail of a list) so a consumer
/// can't mistake it for a limb when it derives the crab's facing. See
/// [`recipe_silhouette`].
pub struct CrabSilhouette {
    /// One capsule per link, in `recipe.hub_bind_world`'s frame — legs, locked
    /// eye-stalks, and claws alike (the cosmetic view draws them all).
    pub limbs: Vec<RestShape>,
    /// The carapace box, same frame.
    pub carapace: RestShape,
}

impl CrabSilhouette {
    /// Every shape (limbs + carapace), for whole-body extent math.
    pub fn shapes(&self) -> impl Iterator<Item = &RestShape> {
        self.limbs.iter().chain(std::iter::once(&self.carapace))
    }

    /// The rig's natural standing height: the vertical (Y) extent of its rest-pose
    /// collider silhouette, in metres. This is the ONE source both giant-crab renders
    /// scale against — the integer silhouette (`net::render::spawn_crab_silhouette`)
    /// fits this extent to the giant height, and the armed NN rig
    /// (`net::external_crab`) scales the live body by the same target/height ratio,
    /// so the two crabs are the same size by construction (no drift). The recipe is oriented
    /// claws-forward by a pure YAW before rendering, which leaves the Y extent unchanged, so
    /// it's correct to take the extent in the rig's own frame here. `0.0` for a degenerate
    /// (empty/non-finite) recipe — callers guard that case.
    pub fn natural_height(&self) -> f32 {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for s in self.shapes() {
            match *s {
                RestShape::Capsule { a, b, radius } => {
                    lo = lo.min(a.y - radius).min(b.y - radius);
                    hi = hi.max(a.y + radius).max(b.y + radius);
                }
                RestShape::Cuboid { center, half } => {
                    lo = lo.min(center.y - half.y);
                    hi = hi.max(center.y + half.y);
                }
            }
        }
        if lo.is_finite() && hi.is_finite() {
            hi - lo
        } else {
            0.0
        }
    }
}

/// Reconstruct `recipe`'s collider silhouette for rendering. Unlike [`rest_colliders`]
/// (scoring: actuated links only, anchored at the model's leg hub) this is the COSMETIC
/// view — it draws every link including the locked eye-stalks and needs no model, the
/// recipe alone carries the geometry. Anchored at `recipe.hub_bind_world` so it shares
/// [`rest_colliders`]'s frame; a render that re-poses and re-centers the crab is free of
/// the absolute offset either way. This is the ONE shape source the giant-crab render
/// draws from (fed [`crate::bot::body::render_recipe`]), so the cosmetic crab can't drift
/// from the body it depicts.
pub fn recipe_silhouette(recipe: &RigRecipe) -> CrabSilhouette {
    let hub = recipe.hub_bind_world;
    let world_origin = link_world_origins(&recipe.links, hub);
    let limbs = recipe
        .links
        .iter()
        .zip(&world_origin)
        .map(|(link, &origin)| link_capsule(link, origin))
        .collect();
    CrabSilhouette {
        limbs,
        carapace: carapace_cuboid(recipe, hub),
    }
}

/// One link's capsule from its telescoped `origin`. Shared by [`rest_colliders`]
/// (scoring) and [`recipe_silhouette`] (rendering) so the capsule geometry has a
/// single definition that can't drift between the scored body and the drawn one.
fn link_capsule(link: &RigLink, origin: Vec3) -> RestShape {
    let axis = link.col_rot * Vec3::Y * link.half_height;
    let c = origin + link.center;
    RestShape::Capsule {
        a: c - axis,
        b: c + axis,
        radius: link.radius,
    }
}

/// The carapace box, world-axis-aligned at the leg `hub` + the recipe's offset.
fn carapace_cuboid(recipe: &RigRecipe, hub: Vec3) -> RestShape {
    RestShape::Cuboid {
        center: hub + recipe.carapace_offset,
        half: recipe.carapace_half,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::rig::fallback_recipe;

    /// The render silhouette (#108) draws EVERY link, eye-stalks included, as a capsule
    /// (one per link) plus the carapace as its own cuboid — and every shape must be
    /// finite or the giant crab would spawn NaN meshes. (Contrast [`rest_colliders`],
    /// which scores actuated links only.)
    #[test]
    fn recipe_silhouette_covers_all_links_plus_carapace() {
        let recipe = fallback_recipe();
        let sil = recipe_silhouette(&recipe);
        assert_eq!(
            sil.limbs.len(),
            recipe.links.len(),
            "one capsule per link, eye-stalks included"
        );
        for s in sil.limbs.iter().chain(std::iter::once(&sil.carapace)) {
            match *s {
                RestShape::Capsule { a, b, radius } => {
                    assert!(a.is_finite() && b.is_finite() && radius.is_finite() && radius > 0.0);
                }
                RestShape::Cuboid { center, half } => {
                    assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0);
                }
            }
        }
        assert!(
            matches!(sil.carapace, RestShape::Cuboid { .. }),
            "carapace must be a cuboid"
        );
    }
}
