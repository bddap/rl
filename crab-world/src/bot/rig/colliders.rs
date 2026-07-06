
use bevy::prelude::*;

use crate::bot::meshfit::PartId;

use super::recipe::link_world_origins;
use super::{RigLink, RigRecipe};

pub struct RestCollider {
    pub part: PartId,
    pub shape: RestShape,
    pub pivot: Vec3,
}

pub enum RestShape {
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    Cuboid { center: Vec3, half: Vec3 },
}

pub fn rest_colliders(recipe: &RigRecipe) -> Vec<RestCollider> {
    let o_root = recipe.hub_bind_world;
    let world_origin = link_world_origins(&recipe.links, o_root);
    let mut out: Vec<RestCollider> = Vec::new();
    for (link, &origin) in recipe.links.iter().zip(&world_origin) {
        if let Some(id) = link.actuated {
            out.push(RestCollider {
                part: PartId::Joint(id),
                shape: link_capsule(link, origin).into(),
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

pub fn recipe_silhouette(recipe: &RigRecipe) -> CrabSilhouette {
    let hub = recipe.hub_bind_world;
    let world_origin = link_world_origins(&recipe.links, hub);
    let limbs = recipe
        .links
        .iter()
        .zip(&world_origin)
        .map(|(link, &origin)| link_capsule(link, origin).into())
        .collect();
    CrabSilhouette {
        limbs,
        carapace: carapace_cuboid(recipe, hub),
    }
}

pub(crate) struct LinkCapsule {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f32,
}

impl From<LinkCapsule> for RestShape {
    fn from(c: LinkCapsule) -> Self {
        RestShape::Capsule {
            a: c.a,
            b: c.b,
            radius: c.radius,
        }
    }
}

pub(crate) fn link_capsule(link: &RigLink, origin: Vec3) -> LinkCapsule {
    let axis = link.col_rot * Vec3::Y * link.half_height;
    let c = origin + link.center;
    LinkCapsule {
        a: c - axis,
        b: c + axis,
        radius: link.radius,
    }
}

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
