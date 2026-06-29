//! The procedural stand-in body for when the purchased glTF isn't present: a
//! skeleton that names its bones exactly as the model's, so [`build_recipe`] yields
//! a full, simulatable [`RigRecipe`] — only the cosmetic skin is missing.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::bot::body::{CrabJointId, Side};
use crate::bot::meshfit::PartId;

use super::recipe::{antennae_bone, antennae_top_bone, build_recipe, leg_bone, pincer_bone};
use super::{BindSource, RigRecipe};

// Meters in the glTF bind-pose-world frame (Y up, feet ≈ y=0), so the recipe lands
// in the same space the real model's does.
const FB_CARAPACE_HALF_W: f32 = 0.40; // half-width  (±X), shell
const FB_CARAPACE_HALF_D: f32 = 0.30; // half-depth  (±Z), shell
const FB_CARAPACE_HALF_H: f32 = 0.12; // half-height (±Y), shell
/// Hub height above the ground: the legs reach down to y≈0, so the body rests roughly
/// a coxa+merus drop up.
const FB_HUB_HEIGHT: f32 = 0.30;
/// Per-segment limb radii, fed to `derive_link` via [`FallbackModel::radius_hint`].
const FB_LEG_RADIUS: f32 = 0.045;
const FB_CLAW_RADIUS: f32 = 0.06;

/// Procedural stand-in skeleton that names its bones exactly as the glTF's, so
/// [`build_recipe`] yields a full [`RigRecipe`] — simulatable and trainable; only the
/// cosmetic skin is missing, so the body shows as the Rapier debug wireframe.
pub(super) struct FallbackModel {
    origins: HashMap<String, Vec3>,
}

impl FallbackModel {
    /// Lay out the skeleton. Placement isn't tuned for fidelity; it only has to give
    /// every link a finite pivot and a bend pointing down the limb.
    pub fn new() -> Self {
        let mut o = HashMap::new();
        let hub = Vec3::new(0.0, FB_HUB_HEIGHT, 0.0);

        for side in [Side::Left, Side::Right] {
            let sx = match side {
                Side::Left => -1.0,
                Side::Right => 1.0,
            };
            // Legs
            for leg in 0u8..4 {
                let z = 0.18 - leg as f32 * 0.16;
                let root = hub + Vec3::new(sx * FB_CARAPACE_HALF_W, 0.0, z);
                let coxa = root; // 000: leg root at the shell (coxa swing pivot)
                let knee = root + Vec3::new(sx * 0.22, -0.02, 0.0); // 003: merus pivot
                // 001: coxo-basal joint, a short step out from the root — the basis lift
                // pivots here (the 2-DOF basal joint's second hinge).
                let basis = root + Vec3::new(sx * 0.09, -0.01, 0.0);
                let ankle = knee + Vec3::new(sx * 0.16, -0.14, 0.0); // 004: carpus pivot
                let foot = ankle + Vec3::new(sx * 0.05, -0.14, 0.0); // 005: foot tip on the ground
                o.insert(leg_bone(leg, "000", side), coxa);
                o.insert(leg_bone(leg, "001", side), basis);
                o.insert(leg_bone(leg, "003", side), knee);
                o.insert(leg_bone(leg, "004", side), ankle);
                o.insert(leg_bone(leg, "005", side), foot);
            }
            // Claw (cheliped)
            let shoulder = hub + Vec3::new(sx * 0.18, 0.04, FB_CARAPACE_HALF_D); // 000a
            let palm = shoulder + Vec3::new(sx * 0.06, 0.0, 0.22); // 005: palm base (wrist pivot)
            let finger = palm + Vec3::new(0.0, 0.02, 0.12); // 006b: movable-finger pivot
            let finger_tip = finger + Vec3::new(0.0, 0.0, 0.10); // 006: finger tip
            o.insert(pincer_bone("000a", side), shoulder);
            o.insert(pincer_bone("005", side), palm);
            o.insert(pincer_bone("006b", side), finger);
            o.insert(pincer_bone("006", side), finger_tip);
            // Eye-stalks
            let eye_base = hub + Vec3::new(sx * 0.12, FB_CARAPACE_HALF_H, FB_CARAPACE_HALF_D * 0.6);
            let eye_top = eye_base + Vec3::new(0.0, 0.10, 0.02);
            o.insert(antennae_bone(side), eye_base);
            o.insert(antennae_top_bone(side), eye_top);
        }
        FallbackModel { origins: o }
    }
}

impl Default for FallbackModel {
    fn default() -> Self {
        Self::new()
    }
}

impl BindSource for FallbackModel {
    fn bone_origin(&self, name: &str) -> Option<Vec3> {
        self.origins.get(name).copied()
    }

    /// Empty per part, routing every link through `derive_link`'s sparse-cloud branch
    /// sized at [`radius_hint`](Self::radius_hint).
    fn vertices_by_part(&self) -> HashMap<PartId, Vec<Vec3>> {
        HashMap::new()
    }

    /// The eight shell-box corners (bind-pose world at the hub), so `carapace_box`
    /// derives the same box it would from a cloud. `names` is ignored: the only
    /// `FallbackModel` caller is `carapace_box` asking for the trunk.
    fn vertices_for_bones(&self, _names: &[&str]) -> Vec<Vec3> {
        let c = Vec3::new(0.0, FB_HUB_HEIGHT, 0.0);
        let h = Vec3::new(FB_CARAPACE_HALF_W, FB_CARAPACE_HALF_H, FB_CARAPACE_HALF_D);
        let mut pts = Vec::with_capacity(8);
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    pts.push(c + Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                }
            }
        }
        pts
    }

    fn radius_hint(&self, part: PartId) -> Option<f32> {
        match part {
            PartId::Joint(
                CrabJointId::LegCoxa(..)
                | CrabJointId::LegBasis(..)
                | CrabJointId::LegMerus(..)
                | CrabJointId::LegCarpus(..),
            ) => Some(FB_LEG_RADIUS),
            PartId::Joint(
                CrabJointId::ClawShoulder(_)
                | CrabJointId::ClawWrist(_)
                | CrabJointId::ClawPincer(_),
            ) => Some(FB_CLAW_RADIUS),
            PartId::Carapace => None, // the box is sized from the corner cloud above
        }
    }
}

/// The no-asset fallback recipe. `expect` is sound: the layout names every bone the
/// builder requires with finite coords, so a `None` here is a bug in this file —
/// caught by `fallback_recipe_builds`, not something a contributor can trip.
pub fn fallback_recipe() -> RigRecipe {
    build_recipe(&FallbackModel::new())
        .expect("the procedural fallback skeleton must build a complete rig recipe")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stand-in must build a complete, finite recipe with NO asset present — the
    /// whole point of the fallback, so it runs in the default `cargo test` (no model,
    /// no App).
    #[test]
    fn fallback_recipe_builds() {
        let recipe = fallback_recipe();

        // Same actuated joint set as the real body — the RL obs/action layout is keyed
        // by these, so a missing/extra one would silently mistrain.
        let actuated: std::collections::HashSet<CrabJointId> =
            recipe.links.iter().filter_map(|l| l.actuated).collect();
        assert_eq!(
            actuated.len(),
            CrabJointId::COUNT,
            "fallback must spawn every actuated joint exactly once"
        );

        // The two cosmetic eye-tip links exist in the rig (for the cosmetic/debug view);
        // they are not given physics bodies and the reward does not read them.
        assert_eq!(
            recipe
                .links
                .iter()
                .filter(|l| l.bone.starts_with("Def_antennae_top"))
                .count(),
            2,
            "two eye-tip links"
        );
        assert_eq!(
            recipe
                .links
                .iter()
                .filter(|l| matches!(l.actuated, Some(CrabJointId::ClawPincer(_))))
                .count(),
            2,
            "two claw-tip links (the reach effectors)"
        );
        // Grippy feet attach on the distal leg bone `.004.`; one per leg (8).
        assert_eq!(
            recipe
                .links
                .iter()
                .filter(|l| l.bone.starts_with("Def_leg") && l.bone.contains(".004."))
                .count(),
            8,
            "eight feet (the .004 distal leg links that plant on the ground)"
        );

        // A real, non-degenerate carapace box.
        assert!(
            recipe.carapace_half.min_element() > 0.01,
            "carapace box must be non-degenerate, got {:?}",
            recipe.carapace_half
        );
    }

    /// `derive_link`'s sparse-cloud branch must honor [`BindSource::radius_hint`], or
    /// the stand-in's limbs would all be the pencil-thin generic fallback radius.
    #[test]
    fn fallback_uses_per_part_radius() {
        let recipe = fallback_recipe();
        let radius_of = |pred: fn(CrabJointId) -> bool| {
            recipe
                .links
                .iter()
                .find(|l| l.actuated.is_some_and(pred))
                .map(|l| l.radius)
                .expect("link present")
        };
        let leg_r = radius_of(|id| matches!(id, CrabJointId::LegMerus(..)));
        let claw_r = radius_of(|id| matches!(id, CrabJointId::ClawShoulder(_)));
        assert!(
            (leg_r - FB_LEG_RADIUS).abs() < 1e-6,
            "leg radius {leg_r} should be the leg hint {FB_LEG_RADIUS}"
        );
        assert!(
            (claw_r - FB_CLAW_RADIUS).abs() < 1e-6,
            "claw radius {claw_r} should be the claw hint {FB_CLAW_RADIUS}"
        );
    }

    /// Feet at y≈0 so the stand-in stands: the foot bones (`005`) sit at the ground in
    /// the bind-pose-world frame the body spawns in, else it topples at spawn.
    #[test]
    fn fallback_feet_reach_the_ground() {
        let m = FallbackModel::new();
        for side in [Side::Left, Side::Right] {
            for leg in 0u8..4 {
                let foot = m
                    .bone_origin(&leg_bone(leg, "005", side))
                    .expect("foot bone");
                assert!(
                    foot.y.abs() < 0.02,
                    "leg {leg} {side:?} foot at y={:.3}, should rest at the ground (y≈0)",
                    foot.y
                );
            }
        }
    }
}
