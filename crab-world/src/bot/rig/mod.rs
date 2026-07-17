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

    /// FNV-1a/64 over EVERY field of the recipe, bit-exact (f32s hashed as their raw
    /// bits, so any committed-table change — however small — is a different digest).
    /// This is the collider-geometry half of
    /// [`crate::mesh_fallback::constructed_body_digest`] (bddap/rl#20 stage 1): a
    /// `baked.rs` regen changes the body under an UNCHANGED asset digest, and this is
    /// what makes that change refuse loudly at every checkpoint load and MP handshake.
    /// Densities and joint anchors/axes are included deliberately — mass distribution
    /// and joint placement are as much "which body is this?" as the collider shapes
    /// (the rl#277 charge-gait collapse was a mass/geometry interaction).
    pub fn digest(&self) -> u64 {
        let mut h = crate::fnv::Fnv::new();
        write_vec3(&mut h, self.hub_bind_world);
        write_vec3(&mut h, self.carapace_half);
        write_vec3(&mut h, self.carapace_offset);
        write_f32(&mut h, self.carapace_density);
        for link in &self.links {
            h.write(link.bone.as_bytes());
            h.write(b"\n");
            h.write(&link.parent.map_or(u64::MAX, |p| p as u64).to_le_bytes());
            write_vec3(&mut h, link.anchor1);
            write_vec3(&mut h, link.axis_local);
            write_f32(&mut h, link.half_height);
            write_f32(&mut h, link.radius);
            write_vec3(&mut h, link.center);
            for c in [
                link.col_rot.x,
                link.col_rot.y,
                link.col_rot.z,
                link.col_rot.w,
            ] {
                write_f32(&mut h, c);
            }
            write_f32(&mut h, link.density);
            match link.actuated {
                Some(id) => h.write(format!("{id:?}\n").as_bytes()),
                None => h.write(b"-\n"),
            }
        }
        h.finish()
    }
}

fn write_f32(h: &mut crate::fnv::Fnv, v: f32) {
    h.write(&v.to_bits().to_le_bytes());
}

fn write_vec3(h: &mut crate::fnv::Fnv, v: Vec3) {
    for c in [v.x, v.y, v.z] {
        write_f32(h, c);
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

#[cfg(test)]
mod digest_tests {
    use super::baked_recipe;

    /// The recipe digest is deterministic across calls (it feeds a OnceLock'd global)
    /// and covers every geometry axis a `baked.rs` regen could move: shape dims,
    /// placement, orientation, mass, topology, and actuation. A single-ULP f32 nudge
    /// on any of them must change the digest — the rl#20 stage-1 guard is bit-exact.
    #[test]
    fn recipe_digest_is_deterministic_and_field_sensitive() {
        let base = baked_recipe().digest();
        assert_eq!(
            base,
            baked_recipe().digest(),
            "digest must be deterministic"
        );

        let nudged: Vec<(&str, super::RigRecipe)> = vec![
            ("carapace_half", {
                let mut r = baked_recipe();
                r.carapace_half.x = f32::from_bits(r.carapace_half.x.to_bits() ^ 1);
                r
            }),
            ("carapace_density", {
                let mut r = baked_recipe();
                r.carapace_density += 1.0;
                r
            }),
            ("link radius", {
                let mut r = baked_recipe();
                r.links[0].radius = f32::from_bits(r.links[0].radius.to_bits() ^ 1);
                r
            }),
            ("link half_height", {
                let mut r = baked_recipe();
                r.links[7].half_height *= 1.0 + 1e-6;
                r
            }),
            ("link center", {
                let mut r = baked_recipe();
                r.links[3].center.y = f32::from_bits(r.links[3].center.y.to_bits() ^ 1);
                r
            }),
            ("link col_rot", {
                let mut r = baked_recipe();
                r.links[5].col_rot.w = f32::from_bits(r.links[5].col_rot.w.to_bits() ^ 1);
                r
            }),
            ("link density", {
                let mut r = baked_recipe();
                r.links[9].density += 1.0;
                r
            }),
            ("link anchor1", {
                let mut r = baked_recipe();
                r.links[2].anchor1.z = f32::from_bits(r.links[2].anchor1.z.to_bits() ^ 1);
                r
            }),
            ("link axis_local", {
                let mut r = baked_recipe();
                r.links[4].axis_local.x = f32::from_bits(r.links[4].axis_local.x.to_bits() ^ 1);
                r
            }),
            ("link parent", {
                let mut r = baked_recipe();
                r.links[1].parent = None;
                r
            }),
            ("link bone", {
                let mut r = baked_recipe();
                r.links[0].bone.push('x');
                r
            }),
            ("link actuated", {
                let mut r = baked_recipe();
                r.links[0].actuated = None;
                r
            }),
            ("dropped link", {
                let mut r = baked_recipe();
                r.links.pop();
                r
            }),
        ];
        for (what, recipe) in nudged {
            assert_ne!(recipe.digest(), base, "digest must cover {what}");
        }
    }
}
