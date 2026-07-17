use bevy::prelude::*;

use super::recipe::link_world_origins;
use super::{LinkShape, PartId, RigLink, RigRecipe};

pub struct RestCollider {
    pub part: PartId,
    pub shape: RestShape,
    pub pivot: Vec3,
}

pub enum RestShape {
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    Cuboid { center: Vec3, rot: Quat, half: Vec3 },
}

/// The eight world-space corners of an oriented cuboid.
pub fn cuboid_corners(center: Vec3, rot: Quat, half: Vec3) -> [Vec3; 8] {
    std::array::from_fn(|i| {
        let s = Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        center + rot * (s * half)
    })
}

pub fn rest_colliders(recipe: &RigRecipe) -> Vec<RestCollider> {
    let o_root = recipe.hub_bind_world;
    let world_origin = link_world_origins(&recipe.links, o_root);
    let mut out: Vec<RestCollider> = Vec::new();
    for (link, &origin) in recipe.links.iter().zip(&world_origin) {
        if let Some(id) = link.actuated {
            out.push(RestCollider {
                part: PartId::Joint(id),
                shape: link_rest_shape(link, origin),
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

pub struct CrabSilhouette {
    pub limbs: Vec<RestShape>,
    pub carapace: RestShape,
}

impl CrabSilhouette {
    pub fn shapes(&self) -> impl Iterator<Item = &RestShape> {
        self.limbs.iter().chain(std::iter::once(&self.carapace))
    }

    pub fn natural_height(&self) -> f32 {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for s in self.shapes() {
            match *s {
                RestShape::Capsule { a, b, radius } => {
                    lo = lo.min(a.y - radius).min(b.y - radius);
                    hi = hi.max(a.y + radius).max(b.y + radius);
                }
                RestShape::Cuboid { center, rot, half } => {
                    for c in cuboid_corners(center, rot, half) {
                        lo = lo.min(c.y);
                        hi = hi.max(c.y);
                    }
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

pub fn recipe_silhouette(recipe: &RigRecipe) -> CrabSilhouette {
    let hub = recipe.hub_bind_world;
    let world_origin = link_world_origins(&recipe.links, hub);
    let limbs = recipe
        .links
        .iter()
        .zip(&world_origin)
        .map(|(link, &origin)| link_rest_shape(link, origin))
        .collect();
    CrabSilhouette {
        limbs,
        carapace: carapace_cuboid(recipe, hub),
    }
}

pub fn link_rest_shape(link: &RigLink, origin: Vec3) -> RestShape {
    let c = origin + link.center;
    match link.shape {
        LinkShape::Capsule {
            half_height,
            radius,
        } => {
            let axis = link.col_rot * Vec3::Y * half_height;
            RestShape::Capsule {
                a: c - axis,
                b: c + axis,
                radius,
            }
        }
        LinkShape::Cuboid { half } => RestShape::Cuboid {
            center: c,
            rot: link.col_rot,
            half,
        },
    }
}

fn carapace_cuboid(recipe: &RigRecipe, hub: Vec3) -> RestShape {
    RestShape::Cuboid {
        center: hub + recipe.carapace_offset,
        rot: Quat::IDENTITY,
        half: recipe.carapace_half,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::rig::fallback_recipe;

    #[test]
    fn recipe_silhouette_covers_all_links_plus_carapace() {
        let recipe = fallback_recipe();
        let sil = recipe_silhouette(&recipe);
        assert_eq!(
            sil.limbs.len(),
            recipe.links.len(),
            "one shape per link, eye-stalks included"
        );
        for s in sil.limbs.iter().chain(std::iter::once(&sil.carapace)) {
            match *s {
                RestShape::Capsule { a, b, radius } => {
                    assert!(a.is_finite() && b.is_finite() && radius.is_finite() && radius > 0.0);
                }
                RestShape::Cuboid { center, rot, half } => {
                    assert!(
                        center.is_finite()
                            && rot.is_finite()
                            && half.is_finite()
                            && half.min_element() > 0.0
                    );
                }
            }
        }
        assert!(
            matches!(sil.carapace, RestShape::Cuboid { .. }),
            "carapace must be a cuboid"
        );
    }
}
