use bevy::prelude::*;

use crate::bot::body::CrabJointId;

mod baked;
mod colliders;
mod fallback;
mod recipe;

pub use baked::{BAKED_ASSET_DIGEST, baked_recipe};
pub use colliders::{
    CrabSilhouette, RestCollider, RestShape, cuboid_corners, link_rest_shape, recipe_silhouette,
    rest_colliders,
};
pub use fallback::fallback_recipe;
pub(crate) use recipe::link_world_origins;
pub use recipe::{TRUNK_BONES, arc_to, build_recipe, part_for_bone, parts_adjacent};

/// The full digest of THE baked Sally body this binary carries: the asset-byte digest
/// chained with the committed [`baked_recipe`] table's [`RigRecipe::digest`]. Both
/// inputs are compiled constants — no asset on disk needed — which is what lets the
/// rl#20 legacy-stamp pin and the golden test check it model-free. Whether a process
/// ADVERTISES it is [`crate::mesh_fallback::constructed_body_digest`]'s verdict-gated
/// call.
pub fn baked_body_digest() -> u64 {
    static D: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        let mut h = crate::fnv::Fnv::new();
        h.write(&BAKED_ASSET_DIGEST.to_le_bytes());
        h.write(&baked_recipe().digest().to_le_bytes());
        h.finish()
    })
}

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

/// A link's collider primitive, in the frame set by [`RigLink::center`] /
/// [`RigLink::col_rot`]. Capsules stand on the local Y axis (half_height to each
/// cap center); cuboids carry half-extents per local axis. Which primitive a part
/// wears is the offline fitter's scored choice (bddap/rl#20 Phase 1) — the runtime
/// only consumes the baked result.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LinkShape {
    Capsule { half_height: f32, radius: f32 },
    Cuboid { half: Vec3 },
}

impl LinkShape {
    fn is_finite(&self) -> bool {
        match *self {
            LinkShape::Capsule {
                half_height,
                radius,
            } => half_height.is_finite() && radius.is_finite(),
            LinkShape::Cuboid { half } => half.is_finite(),
        }
    }

    /// Radius of the shape's bounding sphere about its own center.
    pub fn bounding_radius(&self) -> f32 {
        match *self {
            LinkShape::Capsule {
                half_height,
                radius,
            } => half_height + radius,
            LinkShape::Cuboid { half } => half.length(),
        }
    }
}

#[derive(Clone)]
pub struct RigLink {
    pub bone: String,
    pub parent: Option<usize>,
    pub anchor1: Vec3,
    pub axis_local: Vec3,
    pub shape: LinkShape,
    pub center: Vec3,
    pub col_rot: Quat,
    pub density: f32,
    pub actuated: Option<CrabJointId>,
}

impl RigLink {
    /// Bounding-sphere radius about the link's pivot (covers the collider wherever
    /// `center` offsets it) — the spawn clearance bound.
    pub fn bounding_radius(&self) -> f32 {
        self.center.length() + self.shape.bounding_radius()
    }
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
    /// (the rl#277 charge-gait collapse was a mass/geometry interaction). `actuated`
    /// hashes as [`CrabJointId::index`] — the canonical channel ordinal — not the
    /// variant's name, so an enum rename stays where it belongs: the channel-layout
    /// guard's re-pin, not a fleet-wide body refusal.
    pub fn digest(&self) -> u64 {
        let mut h = crate::fnv::Fnv::new();
        write_vec3(&mut h, self.hub_bind_world);
        write_vec3(&mut h, self.carapace_half);
        write_vec3(&mut h, self.carapace_offset);
        write_f32(&mut h, self.carapace_density);
        for link in &self.links {
            // Length-prefixed: bone is the one variable-length field, and a byte-exact
            // frame is what lets nothing alias across field boundaries.
            h.write(&(link.bone.len() as u64).to_le_bytes());
            h.write(link.bone.as_bytes());
            write_opt_index(&mut h, link.parent.map(|p| p as u64));
            write_vec3(&mut h, link.anchor1);
            write_vec3(&mut h, link.axis_local);
            // Capsules hash exactly as the pre-LinkShape encoding (half_height then
            // radius, no tag) so an all-capsule table keeps its fleet-stamped digest
            // across the enum introduction. Cuboids lead with CUBOID_TAG — 8 bytes a
            // capsule can never emit (its first field would have to be a NaN bit
            // pattern, and `is_finite` bars that), so the variants cannot alias.
            match link.shape {
                LinkShape::Capsule {
                    half_height,
                    radius,
                } => {
                    write_f32(&mut h, half_height);
                    write_f32(&mut h, radius);
                }
                LinkShape::Cuboid { half } => {
                    h.write(&CUBOID_TAG.to_le_bytes());
                    write_vec3(&mut h, half);
                }
            }
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
            write_opt_index(&mut h, link.actuated.map(|id| id.index() as u64));
        }
        h.finish()
    }
}

/// Shape-variant discriminator in [`RigRecipe::digest`]'s cuboid arm. All-ones is
/// NaN as an f32 bit pattern, which no finite capsule field can produce.
const CUBOID_TAG: u64 = u64::MAX;

fn write_f32(h: &mut crate::fnv::Fnv, v: f32) {
    h.write(&v.to_bits().to_le_bytes());
}

/// `None` = `u64::MAX` — safe as a sentinel for both users (link indices and
/// [`CrabJointId::index`] ordinals are tiny), and ONE `Option` framing keeps the next
/// optional field from coining a third.
fn write_opt_index(h: &mut crate::fnv::Fnv, v: Option<u64>) {
    h.write(&v.unwrap_or(u64::MAX).to_le_bytes());
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
            && self.shape.is_finite()
            && self.center.is_finite()
            && self.col_rot.is_finite()
            && self.density.is_finite()
    }
}

#[cfg(test)]
mod digest_tests {
    use super::{Vec3, baked_recipe};

    /// Every field of `RigRecipe`/`RigLink` gets a nudge case here — if `digest()`
    /// silently stops covering one, the corresponding `assert_ne` fails. A single-ULP
    /// f32 nudge must change the digest: the rl#20 stage-1 guard is bit-exact.
    #[test]
    fn recipe_digest_is_deterministic_and_field_sensitive() {
        let base = baked_recipe().digest();
        assert_eq!(
            base,
            baked_recipe().digest(),
            "digest must be deterministic"
        );

        let nudged: Vec<(&str, super::RigRecipe)> = vec![
            ("hub_bind_world", {
                let mut r = baked_recipe();
                r.hub_bind_world.y = f32::from_bits(r.hub_bind_world.y.to_bits() ^ 1);
                r
            }),
            ("carapace_offset", {
                let mut r = baked_recipe();
                r.carapace_offset.z = f32::from_bits(r.carapace_offset.z.to_bits() ^ 1);
                r
            }),
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
            ("link capsule radius", {
                let mut r = baked_recipe();
                let super::LinkShape::Capsule {
                    half_height,
                    radius,
                } = r.links[0].shape
                else {
                    panic!("link 0 is a capsule in the committed table");
                };
                r.links[0].shape = super::LinkShape::Capsule {
                    half_height,
                    radius: f32::from_bits(radius.to_bits() ^ 1),
                };
                r
            }),
            ("link capsule half_height", {
                let mut r = baked_recipe();
                let super::LinkShape::Capsule {
                    half_height,
                    radius,
                } = r.links[7].shape
                else {
                    panic!("link 7 is a capsule in the committed table");
                };
                r.links[7].shape = super::LinkShape::Capsule {
                    half_height: half_height * (1.0 + 1e-6),
                    radius,
                };
                r
            }),
            ("link shape variant", {
                let mut r = baked_recipe();
                r.links[0].shape = super::LinkShape::Cuboid {
                    half: Vec3::splat(0.1),
                };
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

    /// GOLDEN pin of the full body digest, the durable sibling of the rl#20
    /// legacy-stamp shim's pin (which deletes itself with the shim): every stamped
    /// checkpoint in the fleet keys off this value, so an ACCIDENTAL change — a
    /// `digest()` encoding refactor as much as a table edit — must never slip through
    /// as a green build. On a DELIBERATE, reviewed `baked.rs` regen (a new MDP, plan
    /// the retrain per rl#277), re-pin this golden in the same commit — and delete
    /// the legacy shim per `legacy_stamp_pin_matches_current_body`, which forbids
    /// re-pinning ITS constant.
    #[test]
    fn body_digest_golden() {
        assert_eq!(super::baked_body_digest(), GOLDEN_BODY_DIGEST);
    }

    const GOLDEN_BODY_DIGEST: u64 = 0xcb56_1c71_d8fa_a748;
}
