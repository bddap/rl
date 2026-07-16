use std::collections::HashMap;

use bevy::prelude::*;

use crate::bot::body::{CrabJointId, Side};

use super::recipe::{antennae_bone, antennae_top_bone, build_recipe, leg_bone, pincer_bone};
use super::{BindSource, PartId, RigRecipe};

const FB_CARAPACE_HALF_W: f32 = 0.40;
const FB_CARAPACE_HALF_D: f32 = 0.30;
const FB_CARAPACE_HALF_H: f32 = 0.12;
const FB_HUB_HEIGHT: f32 = 0.30;
const FB_LEG_RADIUS: f32 = 0.045;
const FB_CLAW_RADIUS: f32 = 0.06;

pub(super) struct FallbackModel {
    origins: HashMap<String, Vec3>,
}

impl FallbackModel {
    pub fn new() -> Self {
        let mut o = HashMap::new();
        let hub = Vec3::new(0.0, FB_HUB_HEIGHT, 0.0);

        for side in [Side::Left, Side::Right] {
            let sx = match side {
                Side::Left => -1.0,
                Side::Right => 1.0,
            };
            for leg in 0u8..4 {
                let z = 0.18 - leg as f32 * 0.16;
                let root = hub + Vec3::new(sx * FB_CARAPACE_HALF_W, 0.0, z);
                let coxa = root;
                let knee = root + Vec3::new(sx * 0.22, -0.02, 0.0);
                let basis = root + Vec3::new(sx * 0.09, -0.01, 0.0);
                let ankle = knee + Vec3::new(sx * 0.16, -0.14, 0.0);
                let foot = ankle + Vec3::new(sx * 0.05, -0.14, 0.0);
                o.insert(leg_bone(leg, "000", side), coxa);
                o.insert(leg_bone(leg, "001", side), basis);
                o.insert(leg_bone(leg, "003", side), knee);
                o.insert(leg_bone(leg, "004", side), ankle);
                o.insert(leg_bone(leg, "005", side), foot);
            }
            let shoulder = hub + Vec3::new(sx * 0.18, 0.04, FB_CARAPACE_HALF_D);
            let palm = shoulder + Vec3::new(sx * 0.06, 0.0, 0.22);
            let finger = palm + Vec3::new(0.0, 0.02, 0.12);
            let finger_tip = finger + Vec3::new(0.0, 0.0, 0.10);
            o.insert(pincer_bone("000a", side), shoulder);
            o.insert(pincer_bone("005", side), palm);
            o.insert(pincer_bone("006b", side), finger);
            o.insert(pincer_bone("006", side), finger_tip);
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

    fn trunk_vertices(&self) -> Vec<Vec3> {
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
            PartId::Carapace => None,
        }
    }
}

pub fn fallback_recipe() -> RigRecipe {
    build_recipe(&FallbackModel::new())
        .expect("the procedural fallback skeleton must build a complete rig recipe")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_recipe_builds() {
        let recipe = fallback_recipe();

        let actuated: std::collections::HashSet<CrabJointId> =
            recipe.links.iter().filter_map(|l| l.actuated).collect();
        assert_eq!(
            actuated.len(),
            CrabJointId::COUNT,
            "fallback must spawn every actuated joint exactly once"
        );

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
        assert_eq!(
            recipe
                .links
                .iter()
                .filter(|l| matches!(l.actuated, Some(CrabJointId::LegCarpus(..))))
                .count(),
            8,
            "eight feet (the leg-carpus links that plant on the ground)"
        );

        assert!(
            recipe.carapace_half.min_element() > 0.01,
            "carapace box must be non-degenerate, got {:?}",
            recipe.carapace_half
        );
    }

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
