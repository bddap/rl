use std::collections::HashMap;
use std::sync::OnceLock;

use bevy::prelude::*;

use crate::bot::body::{CrabJointId, Side};

use super::{BindSource, PartId, RigLink, RigRecipe};

const LEG_DENSITY: f32 = 8.0;
const CLAW_DENSITY: f32 = 1.0;
const EYE_DENSITY: f32 = 0.5;
const CARAPACE_DENSITY: f32 = 5.0;

const EYE_RADIUS: f32 = 0.03;
const FALLBACK_RADIUS: f32 = 0.03;

pub(crate) fn link_world_origins(links: &[RigLink], root: Vec3) -> Vec<Vec3> {
    let mut world = Vec::with_capacity(links.len());
    for link in links {
        let base = match link.parent {
            None => root,
            Some(idx) => world[idx],
        };
        world.push(base + link.anchor1);
    }
    world
}

struct JointSpec {
    id: CrabJointId,
    pivot: String,
    tip: Option<String>,
    members: Vec<String>,
    density: f32,
}

fn side_tag(side: Side) -> &'static str {
    match side {
        Side::Left => "L",
        Side::Right => "R",
    }
}

pub(super) fn leg_bone(leg: u8, seg: &str, side: Side) -> String {
    format!("Def_leg_0{}.{}.{}", leg + 1, seg, side_tag(side))
}
pub(super) fn pincer_bone(seg: &str, side: Side) -> String {
    format!("Def_pincer.{}.{}", seg, side_tag(side))
}
pub(super) fn antennae_bone(side: Side) -> String {
    format!("Def_antennae.{}", side_tag(side))
}
pub(super) fn antennae_top_bone(side: Side) -> String {
    format!("Def_antennae_top.{}", side_tag(side))
}

fn joint_specs() -> Vec<Vec<JointSpec>> {
    let mut chains = Vec::new();
    for side in [Side::Left, Side::Right] {
        for leg in 0u8..4 {
            let b = |seg: &str| leg_bone(leg, seg, side);
            chains.push(vec![
                JointSpec {
                    id: CrabJointId::LegCoxa(side, leg),
                    pivot: b("000"),
                    tip: Some(b("001")),
                    members: vec![b("000"), b("000b")],
                    density: LEG_DENSITY,
                },
                JointSpec {
                    id: CrabJointId::LegBasis(side, leg),
                    pivot: b("001"),
                    tip: Some(b("003")),
                    members: vec![b("001"), b("002")],
                    density: LEG_DENSITY,
                },
                JointSpec {
                    id: CrabJointId::LegMerus(side, leg),
                    pivot: b("003"),
                    tip: Some(b("004")),
                    members: vec![b("003")],
                    density: LEG_DENSITY,
                },
                JointSpec {
                    id: CrabJointId::LegCarpus(side, leg),
                    pivot: b("004"),
                    tip: Some(b("005")),
                    members: vec![b("004"), b("005")],
                    density: LEG_DENSITY,
                },
            ]);
        }
        let p = |seg: &str| pincer_bone(seg, side);
        chains.push(vec![
            JointSpec {
                id: CrabJointId::ClawShoulder(side),
                pivot: p("000a"),
                tip: Some(p("005")),
                members: vec![p("000a"), p("000"), p("001"), p("002"), p("003"), p("004")],
                density: CLAW_DENSITY,
            },
            JointSpec {
                id: CrabJointId::ClawWrist(side),
                pivot: p("005"),
                tip: Some(p("006b")),
                members: vec![p("005")],
                density: CLAW_DENSITY,
            },
            JointSpec {
                id: CrabJointId::ClawPincer(side),
                pivot: p("006b"),
                tip: Some(p("006")),
                members: vec![p("006b"), p("006")],
                density: CLAW_DENSITY,
            },
        ]);
    }
    chains
}

fn member_map() -> &'static HashMap<String, PartId> {
    static MAP: OnceLock<HashMap<String, PartId>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        for chain in joint_specs() {
            for spec in chain {
                for bone in spec.members {
                    m.insert(bone, PartId::Joint(spec.id));
                }
            }
        }
        m
    })
}

pub fn part_for_bone(name: &str) -> Option<PartId> {
    if !(name.starts_with("Def_") || name.starts_with("Ctrl_")) {
        return None;
    }
    Some(member_map().get(name).copied().unwrap_or(PartId::Carapace))
}

fn part_adjacency() -> &'static std::collections::HashSet<(PartId, PartId)> {
    static ADJ: OnceLock<std::collections::HashSet<(PartId, PartId)>> = OnceLock::new();
    ADJ.get_or_init(|| {
        let mut adj = std::collections::HashSet::new();
        let mut link = |a: PartId, b: PartId| {
            adj.insert((a, b));
            adj.insert((b, a));
        };
        for chain in joint_specs() {
            if let Some(first) = chain.first() {
                link(PartId::Carapace, PartId::Joint(first.id));
            }
            for pair in chain.windows(2) {
                link(PartId::Joint(pair[0].id), PartId::Joint(pair[1].id));
            }
        }
        adj
    })
}

pub fn parts_adjacent(a: PartId, b: PartId) -> bool {
    part_adjacency().contains(&(a, b))
}

/// Derive the bone-chain skeleton recipe from a [`BindSource`]: link topology,
/// anchors, bend axes, and stub capsules sized by the radius hints. This is the
/// SHAPE of the rig, not its fitted flesh — the mesh-fitted collider geometry lives
/// in the committed [`super::baked_recipe`] table, produced offline by the `meshfit`
/// tool overlaying capsule fits on exactly this skeleton (bddap/rl#20).
pub fn build_recipe(model: &impl BindSource) -> Option<RigRecipe> {
    let carapace_center = leg_hub_centroid(model)?;
    if !carapace_center.is_finite() {
        return None;
    }

    let mut links: Vec<RigLink> = Vec::new();
    for chain in joint_specs() {
        let mut parent_idx = None;
        let mut parent_pivot = carapace_center;
        for spec in &chain {
            let Some(pivot) = model.bone_origin(&spec.pivot) else {
                break;
            };
            let Some(link) = derive_link(
                model,
                &spec.pivot,
                spec.tip.as_deref(),
                LinkGeom {
                    parent_pivot,
                    parent_idx,
                    fixed_radius: model
                        .radius_hint(PartId::Joint(spec.id))
                        .unwrap_or(FALLBACK_RADIUS),
                    density: spec.density,
                },
                Some(spec.id),
            ) else {
                break;
            };
            parent_pivot = pivot;
            parent_idx = Some(links.len());
            links.push(link);
        }
    }

    for side in [Side::Left, Side::Right] {
        let base = antennae_bone(side);
        let tip = antennae_top_bone(side);
        let Some(base_link) = derive_link(
            model,
            &base,
            Some(&tip),
            LinkGeom {
                parent_pivot: carapace_center,
                parent_idx: None,
                fixed_radius: EYE_RADIUS,
                density: EYE_DENSITY,
            },
            None,
        ) else {
            continue;
        };
        let base_idx = links.len();
        let base_origin = model.bone_origin(&base).unwrap_or(carapace_center);
        links.push(base_link);
        if let Some(tip_link) = derive_link(
            model,
            &tip,
            None,
            LinkGeom {
                parent_pivot: base_origin,
                parent_idx: Some(base_idx),
                fixed_radius: EYE_RADIUS,
                density: EYE_DENSITY,
            },
            None,
        ) {
            links.push(tip_link);
        }
    }

    let (carapace_half, carapace_offset) = carapace_box(model, carapace_center);
    let recipe = RigRecipe {
        hub_bind_world: carapace_center,
        carapace_half,
        carapace_offset,
        carapace_density: CARAPACE_DENSITY,
        links,
    };
    assert_actuated_never_parents_locked(&recipe.links);
    recipe.is_finite().then_some(recipe)
}

fn assert_actuated_never_parents_locked(links: &[RigLink]) {
    for (i, link) in links.iter().enumerate() {
        if link.actuated.is_some()
            && let Some(p) = link.parent
        {
            assert!(
                links[p].actuated.is_some(),
                "actuated rig link {i} has a locked parent {p} — spawn_crab would \
                 misparent it onto the carapace placeholder"
            );
        }
    }
}

pub(super) fn leg_hub_centroid(model: &impl BindSource) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut n = 0u32;
    for side in [Side::Left, Side::Right] {
        for leg in 0u8..4 {
            if let Some(o) = model.bone_origin(&leg_bone(leg, "000", side)) {
                sum += o;
                n += 1;
            }
        }
    }
    (n > 0).then(|| sum / n as f32)
}

struct LinkGeom {
    parent_pivot: Vec3,
    parent_idx: Option<usize>,
    fixed_radius: f32,
    density: f32,
}

fn derive_link(
    model: &impl BindSource,
    pivot_name: &str,
    tip_name: Option<&str>,
    geom: LinkGeom,
    actuated: Option<CrabJointId>,
) -> Option<RigLink> {
    let LinkGeom {
        parent_pivot,
        parent_idx,
        fixed_radius,
        density,
    } = geom;
    let c_origin = model.bone_origin(pivot_name)?;
    let anchor1 = c_origin - parent_pivot;
    let in_dir = anchor1.normalize_or_zero();

    let chain_dir = match tip_name.and_then(|t| model.bone_origin(t)) {
        Some(g) => (g - c_origin).normalize_or_zero(),
        None => in_dir,
    };
    let axis_local = bend_axis(actuated, in_dir, chain_dir);

    let seg_world = match tip_name.and_then(|t| model.bone_origin(t)) {
        Some(g) => g - c_origin,
        None => in_dir * (fixed_radius * 3.0),
    };
    let seg_len = seg_world.length().max(1e-4);
    let (center, col_rot, half_height, radius) = (
        seg_world * 0.5,
        arc_to(Vec3::Y, seg_world / seg_len),
        (seg_len * 0.5 - fixed_radius).max(0.01),
        fixed_radius,
    );

    Some(RigLink {
        bone: pivot_name.to_string(),
        parent: parent_idx,
        anchor1,
        axis_local,
        half_height,
        radius,
        center,
        col_rot,
        density,
        actuated,
    })
}

fn bend_axis(actuated: Option<CrabJointId>, in_dir: Vec3, out_dir: Vec3) -> Vec3 {
    if matches!(actuated, Some(CrabJointId::LegCoxa(..))) {
        return Vec3::Y;
    }
    if matches!(actuated, Some(CrabJointId::LegBasis(..))) {
        let axis = out_dir.cross(Vec3::Y);
        return if axis.length() > 0.5 {
            axis.normalize()
        } else {
            Vec3::X
        };
    }
    if matches!(actuated, Some(CrabJointId::ClawWrist(_))) {
        return Vec3::new(0.86062, -0.27404, 0.42922);
    }
    let cross = in_dir.cross(out_dir);
    let axis = if cross.length() > 0.2 {
        cross.normalize()
    } else {
        out_dir.cross(Vec3::Y).normalize_or_zero()
    };
    if axis.length() > 0.5 { axis } else { Vec3::X }
}

/// Rotation taking `from` onto `to` — THE collider-orientation convention for every
/// capsule in the rig. Pub because the offline `meshfit` baker orients its fitted
/// capsules with the SAME convention; a second copy there could silently drift the
/// baked table from the stub path (one source, bddap/rl#20).
pub fn arc_to(from: Vec3, to: Vec3) -> Quat {
    if to.length_squared() < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_arc(from, to.normalize())
    }
}

pub const TRUNK_BONES: [&str; 10] = [
    "Def_shell.000.L",
    "Def_shell.002.L",
    "Def_shell.000.R",
    "Def_shell.002.R",
    "Def_thorax_back",
    "Def_thorax_front",
    "Def_Rostrum.L",
    "Def_Rostrum.R",
    "Ctrl_abdomen_end",
    "Ctrl_Def_neck",
];

fn carapace_box(model: &impl BindSource, center: Vec3) -> (Vec3, Vec3) {
    let pts = model.trunk_vertices();
    if pts.len() < 4 {
        return (Vec3::splat(0.1), Vec3::ZERO);
    }
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for p in &pts {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let half = (hi - lo) * 0.5;
    let box_center = (hi + lo) * 0.5;
    (half, box_center - center)
}

#[cfg(test)]
mod tests {
    use super::super::colliders::{LinkCapsule, link_capsule};
    use super::*;
    use crate::bot::rig::{baked_recipe, fallback_recipe};

    #[test]
    fn shoulder_upswing_stays_below_carapace() {
        let recipe = baked_recipe();
        let hub = recipe.hub_bind_world;
        let box_top = (hub + recipe.carapace_offset).y + recipe.carapace_half.y;
        let world = link_world_origins(&recipe.links, hub);
        for side in [Side::Left, Side::Right] {
            let sh_idx = recipe
                .links
                .iter()
                .position(|l| l.actuated == Some(CrabJointId::ClawShoulder(side)))
                .expect("shoulder link present");
            let pivot = world[sh_idx];
            let [lo, _hi] = CrabJointId::ClawShoulder(side).limits();
            let rot = Quat::from_axis_angle(recipe.links[sh_idx].axis_local, lo);
            for (i, link) in recipe.links.iter().enumerate().filter(|(_, l)| {
                matches!(
                    l.actuated,
                    Some(
                        CrabJointId::ClawShoulder(s)
                            | CrabJointId::ClawWrist(s)
                            | CrabJointId::ClawPincer(s)
                    ) if s == side
                )
            }) {
                let LinkCapsule { a, b, radius } = link_capsule(link, world[i]);
                for cap in [a, b] {
                    let top = (pivot + rot * (cap - pivot)).y + radius;
                    assert!(
                        top <= box_top + 1e-3,
                        "{side:?} cheliped {} capsule reaches y={top:.3} at the up-stop \
                         θ={lo:.3}, above the carapace top {box_top:.3} — arm flesh clips \
                         the shell/eye band",
                        link.bone
                    );
                }
            }
        }
    }

    #[test]
    fn joint_specs_cover_every_actuated_joint() {
        let mut ids: Vec<CrabJointId> = joint_specs().into_iter().flatten().map(|s| s.id).collect();
        let total = ids.len();
        ids.sort_by_key(|id| id.index());
        ids.dedup();
        assert_eq!(
            ids.len(),
            total,
            "a joint id appears in more than one JointSpec"
        );
        assert_eq!(
            total,
            CrabJointId::COUNT,
            "joint_specs count != CrabJointId::COUNT"
        );
    }

    #[test]
    fn members_route_to_their_joint() {
        for chain in joint_specs() {
            for spec in chain {
                for bone in &spec.members {
                    assert_eq!(
                        part_for_bone(bone),
                        Some(PartId::Joint(spec.id)),
                        "{bone} should route to {:?}",
                        spec.id
                    );
                }
            }
        }
    }

    /// The baked table and the procedural fallback must enumerate the SAME links
    /// (count, bone, actuated joint, parent): the RL obs/action layout and the skin
    /// mapping key off that sequence, so a fork would silently re-layout the policy
    /// or mis-route skin. Model-free — both recipes are in the binary.
    #[test]
    fn baked_and_fallback_share_link_topology() {
        let baked = baked_recipe();
        let fallback = fallback_recipe();
        assert_eq!(
            baked.links.len(),
            fallback.links.len(),
            "baked ({}) and fallback ({}) disagree on link count — the rig forked",
            baked.links.len(),
            fallback.links.len()
        );
        for (i, (b, h)) in baked.links.iter().zip(&fallback.links).enumerate() {
            assert_eq!(
                b.bone, h.bone,
                "link {i}: bone baked={} fallback={}",
                b.bone, h.bone
            );
            assert_eq!(
                b.actuated, h.actuated,
                "link {i} ({}): actuated joint baked={:?} fallback={:?}",
                b.bone, b.actuated, h.actuated
            );
            assert_eq!(
                b.parent, h.parent,
                "link {i} ({}): parent baked={:?} fallback={:?}",
                b.bone, b.parent, h.parent
            );
        }
    }

    /// The baked table must satisfy the same structural invariants `build_recipe`
    /// enforces on recipes it derives — it was generated from one, and a hand edit
    /// (the file says DO NOT EDIT) or codegen bug must not be able to sneak a
    /// non-finite value or a locked-parent actuated link into the live body.
    #[test]
    fn baked_recipe_is_structurally_sound() {
        let recipe = baked_recipe();
        assert!(recipe.is_finite(), "baked recipe has non-finite values");
        assert_actuated_never_parents_locked(&recipe.links);
        let actuated: std::collections::HashSet<CrabJointId> =
            recipe.links.iter().filter_map(|l| l.actuated).collect();
        assert_eq!(
            actuated.len(),
            CrabJointId::COUNT,
            "baked table must carry every actuated joint exactly once"
        );
    }
}
