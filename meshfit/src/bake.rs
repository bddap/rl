//! Codegen for `crab-world/src/bot/rig/baked.rs`: the committed fitted-collider table
//! the runtime consumes instead of fitting at spawn (bddap/rl#20 Phase 2). Floats are
//! emitted with Rust's shortest-roundtrip formatting, so the generated literals parse
//! back to bit-identical values — the bake is exact, not approximate.

use std::fmt::Write as _;

use bevy::prelude::*;
use crab_world::bot::body::{CrabJointId, Side};
use crab_world::bot::rig::{LinkShape, PartId, RigLink, RigRecipe, arc_to};

use crate::fit::{FittedShape, ShapePolicy, fit_link_shape};
use crate::gltf_load::LoadedModel;

/// Feet plant and roll on their spherical capsule tips, and the pincer collider is
/// read back via `as_capsule` for the sim's claw-touch decisions
/// (net::external_crab) — both stay capsules whatever the fit score says.
fn shape_policy(id: CrabJointId) -> ShapePolicy {
    match id {
        CrabJointId::LegCarpus(..) | CrabJointId::ClawPincer(_) => ShapePolicy::CapsuleOnly,
        _ => ShapePolicy::Any,
    }
}

/// The fitted recipe: `rig::build_recipe`'s bone-chain skeleton with each link's
/// collider geometry replaced by the best-scoring primitive fit of its skinned
/// vertex cloud (≥ 8 points; smaller/missing clouds keep the bone-chain stub), then
/// every density re-derived so the link keeps the mass it carries in the committed
/// [`baked_recipe`] table (rl#20 Phase 3 — the rl#277 charge-gait collapse was the
/// fuller shapes under the old volume-tuned densities). This fit runs only here, at
/// bake time; the runtime consumes the committed table.
pub fn fitted_recipe(model: &LoadedModel) -> Option<RigRecipe> {
    let mut recipe = crab_world::bot::rig::build_recipe(model)?;
    let clouds = model.vertices_by_part();
    for link in &mut recipe.links {
        let Some(id) = link.actuated else {
            continue;
        };
        let Some(pts) = clouds.get(&PartId::Joint(id)) else {
            continue;
        };
        if pts.len() < 8 {
            continue;
        }
        // The stub collider points down the bone chain by construction, so its
        // center direction IS the chain axis — the hint that rescues isotropic
        // clouds whose PCA axis is noise (the degenerate middle coxae).
        let chain_dir = link.center.normalize_or_zero();
        let Some(fitted) = fit_link_shape(
            pts,
            (chain_dir.length_squared() > 0.5).then_some(chain_dir),
            shape_policy(id),
        ) else {
            continue;
        };
        let origin = model
            .bone_origin(&link.bone)
            .expect("build_recipe only emits links whose pivot bone exists");
        match fitted {
            FittedShape::Capsule(cap) => {
                let seg = cap.b - cap.a;
                link.center = (cap.a + cap.b) * 0.5 - origin;
                link.col_rot = arc_to(Vec3::Y, seg);
                link.shape = LinkShape::Capsule {
                    half_height: seg.length() * 0.5,
                    radius: cap.radius,
                };
            }
            FittedShape::Cuboid { center, rot, half } => {
                link.center = center - origin;
                link.col_rot = rot;
                link.shape = LinkShape::Cuboid { half };
            }
        }
    }
    rederive_densities(&mut recipe, &crab_world::bot::rig::baked_recipe());
    Some(recipe)
}

fn shape_volume(shape: LinkShape) -> f32 {
    match shape {
        LinkShape::Capsule {
            half_height,
            radius,
        } => {
            let r = radius.max(1e-6);
            std::f32::consts::PI * r * r * (2.0 * half_height)
                + (4.0 / 3.0) * std::f32::consts::PI * r * r * r
        }
        LinkShape::Cuboid { half } => 8.0 * half.x * half.y * half.z,
    }
}

/// rl#20 Phase 3: every link keeps the MASS it carries in the reference (committed)
/// table — density_new = density_ref · V_ref / V_new — so a geometry re-fit changes
/// contact surfaces, never the tuned mass distribution or balance. Mass, not
/// density, is the gameplay tuning (the hand-coded densities were volume-tuned for
/// stick-thin colliders; carrying them onto fuller shapes tripled her mass and cut
/// the charge gait −90%, rl#277). Masses chain forward across re-bakes: the
/// committed table is definitionally the body every current checkpoint drives.
/// When the fit reproduces the reference geometry exactly (an unchanged-asset
/// re-bake), V_ref/V_new is exactly 1.0 and densities pass through bit-identical —
/// `baked_matches_refit` stays a fixed point.
fn rederive_densities(recipe: &mut RigRecipe, reference: &RigRecipe) {
    assert_eq!(
        recipe.links.len(),
        reference.links.len(),
        "link topology diverged from the committed table — a mass-reference decision \
         is needed per new link; re-derive by hand"
    );
    for (link, ref_link) in recipe.links.iter_mut().zip(&reference.links) {
        assert_eq!(
            link.bone, ref_link.bone,
            "link order diverged from the committed table"
        );
        link.density = ref_link.density * shape_volume(ref_link.shape) / shape_volume(link.shape);
    }
    let ref_carapace_vol =
        8.0 * reference.carapace_half.x * reference.carapace_half.y * reference.carapace_half.z;
    let carapace_vol =
        8.0 * recipe.carapace_half.x * recipe.carapace_half.y * recipe.carapace_half.z;
    recipe.carapace_density = reference.carapace_density * ref_carapace_vol / carapace_vol;
}

fn f(x: f32) -> String {
    assert!(x.is_finite(), "non-finite value in the fitted recipe: {x}");
    format!("{x:?}")
}

fn vec3(v: Vec3) -> String {
    format!("Vec3::new({}, {}, {})", f(v.x), f(v.y), f(v.z))
}

fn quat(q: Quat) -> String {
    format!(
        "Quat::from_xyzw({}, {}, {}, {})",
        f(q.x),
        f(q.y),
        f(q.z),
        f(q.w)
    )
}

fn side(s: Side) -> &'static str {
    match s {
        Side::Left => "Side::Left",
        Side::Right => "Side::Right",
    }
}

fn joint(id: CrabJointId) -> String {
    use CrabJointId::*;
    match id {
        LegCoxa(s, n) => format!("CrabJointId::LegCoxa({}, {n})", side(s)),
        LegBasis(s, n) => format!("CrabJointId::LegBasis({}, {n})", side(s)),
        LegMerus(s, n) => format!("CrabJointId::LegMerus({}, {n})", side(s)),
        LegCarpus(s, n) => format!("CrabJointId::LegCarpus({}, {n})", side(s)),
        ClawShoulder(s) => format!("CrabJointId::ClawShoulder({})", side(s)),
        ClawWrist(s) => format!("CrabJointId::ClawWrist({})", side(s)),
        ClawPincer(s) => format!("CrabJointId::ClawPincer({})", side(s)),
    }
}

fn shape(s: LinkShape) -> String {
    match s {
        LinkShape::Capsule {
            half_height,
            radius,
        } => format!(
            "LinkShape::Capsule {{ half_height: {}, radius: {} }}",
            f(half_height),
            f(radius)
        ),
        LinkShape::Cuboid { half } => format!("LinkShape::Cuboid {{ half: {} }}", vec3(half)),
    }
}

fn push_link(out: &mut String, l: &RigLink) {
    let parent = match l.parent {
        None => "None".to_string(),
        Some(i) => format!("Some({i})"),
    };
    let actuated = match l.actuated {
        None => "None".to_string(),
        Some(id) => format!("Some({})", joint(id)),
    };
    let _ = write!(
        out,
        "            RigLink {{\n                bone: {:?}.to_string(),\n                parent: {parent},\n                anchor1: {},\n                axis_local: {},\n                shape: {},\n                center: {},\n                col_rot: {},\n                density: {},\n                actuated: {actuated},\n            }},\n",
        l.bone,
        vec3(l.anchor1),
        vec3(l.axis_local),
        shape(l.shape),
        vec3(l.center),
        quat(l.col_rot),
        f(l.density),
    );
}

pub fn render_baked_rs(recipe: &RigRecipe, asset_digest: u64) -> String {
    let mut out = format!(
        "//! GENERATED by `cargo run -p meshfit -- bake` — DO NOT EDIT (bddap/rl#20).\n\
         //!\n\
         //! The committed fitted-collider table: THE one source of Sally's collider\n\
         //! geometry at runtime. Baked offline from sally.glb (FNV-1a/64\n\
         //! {asset_digest:#018x}); [`crate::mesh_fallback`] refuses any asset whose\n\
         //! digest disagrees, so a sally.glb change is loud and a re-bake is a\n\
         //! deliberate, reviewed diff of this file. Any geometry change here is a NEW\n\
         //! MDP: live checkpoints cannot drive it — plan a retrain (rl#277). A re-bake\n\
         //! must ship with the new sally.glb under a NEW rl-assets release tag (plus\n\
         //! the scripts/fetch-sally.sh bump) in the same commit — clobbering the old\n\
         //! tag strands every deployed binary on the refusal path.\n\n\
         use bevy::prelude::*;\n\n\
         use crate::bot::body::{{CrabJointId, Side}};\n\n\
         use super::{{LinkShape, RigLink, RigRecipe}};\n\n\
         /// FNV-1a/64 of the exact sally.glb bytes this table was fitted from.\n\
         pub const BAKED_ASSET_DIGEST: u64 = {asset_digest:#018x};\n\n\
         #[rustfmt::skip]\n\
         pub fn baked_recipe() -> RigRecipe {{\n    RigRecipe {{\n"
    );
    let _ = write!(
        out,
        "        hub_bind_world: {},\n        carapace_half: {},\n        carapace_offset: {},\n        carapace_density: {},\n",
        vec3(recipe.hub_bind_world),
        vec3(recipe.carapace_half),
        vec3(recipe.carapace_offset),
        f(recipe.carapace_density),
    );
    out.push_str("        links: vec![\n");
    for l in &recipe.links {
        push_link(&mut out, l);
    }
    out.push_str("        ],\n    }\n}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crab_world::bot::rig::{BAKED_ASSET_DIGEST, baked_recipe};
    use crab_world::mesh_fallback::model_path;

    /// THE fitted-vs-committed drift-guard (bddap/rl#20 Phase 0's promise, Phase 2's
    /// teeth): a re-fit of the canonical asset must reproduce the committed baked
    /// table BIT-EXACTLY. Red means either the fitter changed (deliberate → re-bake,
    /// review the geometry diff, plan a retrain) or the bake is stale. Skips without
    /// the model; hard-fails on an asset-digest mismatch (that's a sally.glb change —
    /// re-bake deliberately, never re-baseline blind).
    #[test]
    fn baked_matches_refit() {
        let Some(path) = model_path() else {
            eprintln!("baked_matches_refit: no model — skipping");
            return;
        };
        let bytes = std::fs::read(&path).expect("read model");
        let digest = crab_world::fnv::fnv1a(&bytes);
        assert_eq!(
            digest, BAKED_ASSET_DIGEST,
            "sally.glb changed (digest {digest:#018x} != baked {BAKED_ASSET_DIGEST:#018x}) — \
             run `cargo run -p meshfit -- bake`, review the geometry diff, expect a retrain"
        );
        let model = LoadedModel::from_slice(&bytes).expect("load model");
        let refit = fitted_recipe(&model).expect("fitted recipe");
        let baked = baked_recipe();

        assert_eq!(refit.hub_bind_world, baked.hub_bind_world, "hub_bind_world");
        assert_eq!(refit.carapace_half, baked.carapace_half, "carapace_half");
        assert_eq!(
            refit.carapace_offset, baked.carapace_offset,
            "carapace_offset"
        );
        assert_eq!(
            refit.carapace_density, baked.carapace_density,
            "carapace_density"
        );
        assert_eq!(refit.links.len(), baked.links.len(), "link count");
        for (i, (r, b)) in refit.links.iter().zip(&baked.links).enumerate() {
            assert_eq!(r.bone, b.bone, "link {i}: bone");
            assert_eq!(r.parent, b.parent, "link {i} ({}): parent", r.bone);
            assert_eq!(r.actuated, b.actuated, "link {i} ({}): actuated", r.bone);
            assert_eq!(r.anchor1, b.anchor1, "link {i} ({}): anchor1", r.bone);
            assert_eq!(
                r.axis_local, b.axis_local,
                "link {i} ({}): axis_local",
                r.bone
            );
            assert_eq!(r.shape, b.shape, "link {i} ({}): shape", r.bone);
            assert_eq!(r.center, b.center, "link {i} ({}): center", r.bone);
            assert_eq!(r.col_rot, b.col_rot, "link {i} ({}): col_rot", r.bone);
            assert_eq!(r.density, b.density, "link {i} ({}): density", r.bone);
        }
    }

    /// At the shoulder's up-stop, no part of the cheliped may reach above the
    /// carapace top — arm flesh would clip the shell/eye band. The check swings the
    /// skinned FLESH clouds, not the fitted colliders: a cuboid's empty corners
    /// would overstate the arm's reach (rl#20 Phase 1). Moved here from the runtime
    /// crate because flesh needs the model; skips without it.
    #[test]
    fn shoulder_upswing_stays_below_carapace() {
        use crab_world::bot::body::{CrabJointId, Side};

        let Some(path) = model_path() else {
            eprintln!("shoulder_upswing_stays_below_carapace: no model — skipping");
            return;
        };
        let model = LoadedModel::load(&path).expect("load model");
        let recipe = crab_world::bot::rig::build_recipe(&model).expect("recipe");
        let box_top = (recipe.hub_bind_world + recipe.carapace_offset).y + recipe.carapace_half.y;
        let pivots = crab_world::bot::rig::rest_colliders(&recipe);
        let clouds = model.vertices_by_part();
        for side in [Side::Left, Side::Right] {
            let shoulder = CrabJointId::ClawShoulder(side);
            let pivot = pivots
                .iter()
                .find(|rc| rc.part == PartId::Joint(shoulder))
                .expect("shoulder collider present")
                .pivot;
            let axis = recipe
                .links
                .iter()
                .find(|l| l.actuated == Some(shoulder))
                .expect("shoulder link present")
                .axis_local;
            let [lo, _hi] = shoulder.limits();
            let rot = Quat::from_axis_angle(axis, lo);
            for id in [
                shoulder,
                CrabJointId::ClawWrist(side),
                CrabJointId::ClawPincer(side),
            ] {
                let pts = clouds.get(&PartId::Joint(id)).expect("cheliped cloud");
                let top = pts
                    .iter()
                    .map(|&p| (pivot + rot * (p - pivot)).y)
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    top <= box_top + 1e-3,
                    "{side:?} cheliped {id:?} flesh reaches y={top:.3} at the up-stop \
                     θ={lo:.3}, above the carapace top {box_top:.3} — arm flesh clips \
                     the shell/eye band"
                );
            }
        }
    }

    /// Every Def_/Ctrl_ bone in the model must map to a physics part, or its flesh
    /// silently fits into nothing (moved from the old runtime meshfit tests).
    #[test]
    fn bone_map_covers_all_model_bones() {
        let Some(path) = model_path() else {
            eprintln!("bone_map_covers_all_model_bones: no model — skipping");
            return;
        };
        let model = LoadedModel::load(&path).expect("load model");
        let unmapped = model.unmapped_bones();
        assert!(
            unmapped.is_empty(),
            "bones map to no physics part: {unmapped:?}"
        );
    }
}
