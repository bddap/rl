
use std::collections::HashMap;
use std::sync::OnceLock;

use bevy::prelude::*;

use crate::bot::body::{CrabJointId, Side};
use crate::bot::meshfit::PartId;

use super::{BindSource, RigLink, RigRecipe};

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

pub fn build_recipe(model: &impl BindSource) -> Option<RigRecipe> {
    let carapace_center = leg_hub_centroid(model)?;
    if !carapace_center.is_finite() {
        return None;
    }
    let clouds = model.vertices_by_part();

    let mut links: Vec<RigLink> = Vec::new();
    for chain in joint_specs() {
        let mut parent_idx = None;
        let mut parent_pivot = carapace_center;
        for spec in &chain {
            let Some(pivot) = model.bone_origin(&spec.pivot) else {
                break;
            };
            let cloud = clouds
                .get(&PartId::Joint(spec.id))
                .map(|pts| pts.as_slice());
            let Some(link) = derive_link(
                model,
                &spec.pivot,
                spec.tip.as_deref(),
                LinkGeom {
                    parent_pivot,
                    parent_idx,
                    cloud,
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
                cloud: None,
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
                cloud: None,
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

struct LinkGeom<'a> {
    parent_pivot: Vec3,
    parent_idx: Option<usize>,
    cloud: Option<&'a [Vec3]>,
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
        cloud,
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

    let fitted = match cloud {
        Some(pts) if pts.len() >= 8 => crate::bot::meshfit::fit_capsule(pts),
        _ => None,
    };
    let (center, col_rot, half_height, radius) = match fitted {
        Some(cap) => {
            let seg = cap.b - cap.a;
            (
                (cap.a + cap.b) * 0.5 - c_origin,
                arc_to(Vec3::Y, seg),
                seg.length() * 0.5,
                cap.radius,
            )
        }
        None => {
            let seg_world = match tip_name.and_then(|t| model.bone_origin(t)) {
                Some(g) => g - c_origin,
                None => in_dir * (fixed_radius * 3.0),
            };
            let seg_len = seg_world.length().max(1e-4);
            (
                seg_world * 0.5,
                arc_to(Vec3::Y, seg_world / seg_len),
                (seg_len * 0.5 - fixed_radius).max(0.01),
                fixed_radius,
            )
        }
    };

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

fn arc_to(from: Vec3, to: Vec3) -> Quat {
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
    let pts = model.vertices_for_bones(&TRUNK_BONES);
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
    use crate::bot::meshfit::{LoadedModel, model_path};
    use crate::bot::rig::fallback_recipe;

    #[test]
    fn dump_left_cheliped_chain() {
        let Some(path) = model_path() else {
            eprintln!("dump_left_cheliped_chain: no model — skipping");
            return;
        };
        let model = LoadedModel::load(&path).expect("load model");
        let bytes = std::fs::read(&path).expect("read glb");
        let gltf = gltf::Gltf::from_slice(&bytes).expect("parse glb");

        let mut parent: HashMap<usize, usize> = HashMap::new();
        let mut name_of: HashMap<usize, String> = HashMap::new();
        for node in gltf.nodes() {
            if let Some(nm) = node.name() {
                name_of.insert(node.index(), nm.to_string());
            }
            for child in node.children() {
                parent.insert(child.index(), node.index());
            }
        }

        let mut rows: Vec<(String, String, Vec3)> = Vec::new();
        for node in gltf.nodes() {
            let Some(nm) = node.name() else { continue };
            if !(nm.starts_with("Def_pincer.") && nm.ends_with(".L")) {
                continue;
            }
            let origin = model.bone_origin(nm).unwrap_or(Vec3::ZERO);
            let parent_nm = parent
                .get(&node.index())
                .and_then(|p| name_of.get(p))
                .cloned()
                .unwrap_or_else(|| "<root>".into());
            rows.push((nm.to_string(), parent_nm, origin));
        }

        let names: std::collections::HashSet<&str> =
            rows.iter().map(|(n, _, _)| n.as_str()).collect();
        let mut child_of: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut root: Option<&str> = None;
        for (n, p, _) in &rows {
            if names.contains(p.as_str()) {
                child_of.entry(p.as_str()).or_default().push(n.as_str());
            } else {
                root = Some(n.as_str());
            }
        }
        let mut order: Vec<&str> = Vec::new();
        let mut stack: Vec<&str> = root.into_iter().collect();
        while let Some(n) = stack.pop() {
            order.push(n);
            if let Some(kids) = child_of.get(n) {
                let mut kids = kids.clone();
                kids.sort_unstable();
                stack.extend(kids.into_iter().rev());
            }
        }

        let origin_of = |name: &str| rows.iter().find(|(n, _, _)| n == name).map(|(_, _, o)| *o);
        let cloud_stats = |name: &str| -> (usize, Vec3) {
            let pts = model.vertices_for_bones(&[name]);
            if pts.is_empty() {
                return (0, Vec3::ZERO);
            }
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for p in &pts {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
            (pts.len(), hi - lo)
        };
        eprintln!("\n=== LEFT cheliped (Def_pincer.*.L) bind-pose chain, proximal->distal ===");
        let mut prev: Option<Vec3> = None;
        for n in &order {
            let o = origin_of(n).unwrap_or(Vec3::ZERO);
            let parent_nm = &rows.iter().find(|(name, _, _)| name == n).unwrap().1;
            let step = prev.map_or(0.0, |p| (o - p).length());
            let (nv, ext) = cloud_stats(n);
            eprintln!(
                "{n:<18} parent={parent_nm:<22} origin=({:+.4},{:+.4},{:+.4}) step={step:.4} verts={nv:<5} ext=({:.3},{:.3},{:.3})",
                o.x, o.y, o.z, ext.x, ext.y, ext.z
            );
            prev = Some(o);
        }
        eprintln!("--- all nodes matching \"pincer\" (case-insensitive) ---");
        let mut extras: Vec<(String, usize)> = gltf
            .nodes()
            .filter_map(|n| n.name().map(|nm| (nm.to_string(), n.index())))
            .filter(|(nm, _)| nm.to_lowercase().contains("pincer"))
            .collect();
        extras.sort_by(|a, b| a.0.cmp(&b.0));
        for (nm, idx) in &extras {
            let o = model.bone_origin(nm);
            let pnm = parent.get(idx).and_then(|p| name_of.get(p)).cloned();
            eprintln!(
                "{nm:<22} parent={:<22} origin={:?}",
                pnm.unwrap_or_else(|| "<root>".into()),
                o
            );
        }
        eprintln!(
            "=== {} Def_pincer.*.L bones, {} total pincer nodes ===\n",
            rows.len(),
            extras.len()
        );
    }

    #[test]
    fn shoulder_upswing_stays_below_carapace() {
        let Some(path) = model_path() else {
            eprintln!("shoulder_upswing_stays_below_carapace: no model — skipping");
            return;
        };
        let model = LoadedModel::load(&path).expect("load model");
        let recipe = build_recipe(&model).expect("recipe");
        let hub = leg_hub_centroid(&model).expect("hub");
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

    #[test]
    fn fitted_and_fallback_share_link_topology() {
        let Some(path) = model_path() else {
            eprintln!("fitted_and_fallback_share_link_topology: no model — skipping");
            return;
        };
        let model = LoadedModel::load(&path).expect("load model");
        let fitted = build_recipe(&model).expect("fitted recipe");
        let fallback = fallback_recipe();
        assert_eq!(
            fitted.links.len(),
            fallback.links.len(),
            "fitted ({}) and fallback ({}) disagree on link count — the rig forked",
            fitted.links.len(),
            fallback.links.len()
        );
        for (i, (f, h)) in fitted.links.iter().zip(&fallback.links).enumerate() {
            assert_eq!(
                f.bone, h.bone,
                "link {i}: bone fitted={} fallback={}",
                f.bone, h.bone
            );
            assert_eq!(
                f.actuated, h.actuated,
                "link {i} ({}): actuated joint fitted={:?} fallback={:?}",
                f.bone, f.actuated, h.actuated
            );
            assert_eq!(
                f.parent, h.parent,
                "link {i} ({}): parent fitted={:?} fallback={:?}",
                f.bone, f.parent, h.parent
            );
        }
    }

    const GOLDEN_ASSET_DIGEST: u64 = 0x5b29_217e_ad4c_7c57;
    #[rustfmt::skip]
    const FITTED_GOLDEN: &[(&str, [f32; 5])] = &[
        ("Def_leg_01.000.L", [0.094680, 0.057408,  0.024866, -0.016562,  0.001032]),
        ("Def_leg_01.001.L", [0.142467, 0.061588,  0.174220,  0.018777,  0.049600]),
        ("Def_leg_01.003.L", [0.060660, 0.061648,  0.087472, -0.032786, -0.010803]),
        ("Def_leg_01.004.L", [0.122645, 0.054299,  0.068538, -0.154283, -0.023502]),
        ("Def_leg_02.000.L", [0.000000, 0.176261,  0.026321, -0.013589, -0.001256]),
        ("Def_leg_02.001.L", [0.181178, 0.066546,  0.216730,  0.015234,  0.028623]),
        ("Def_leg_02.003.L", [0.054145, 0.071458,  0.097444, -0.034404, -0.027830]),
        ("Def_leg_02.004.L", [0.123662, 0.057603,  0.079551, -0.152684, -0.029877]),
        ("Def_leg_03.000.L", [0.073876, 0.066578,  0.087574,  0.007707, -0.028267]),
        ("Def_leg_03.001.L", [0.170628, 0.068805,  0.212199,  0.018997, -0.018147]),
        ("Def_leg_03.003.L", [0.066288, 0.066312,  0.095621, -0.037249, -0.052509]),
        ("Def_leg_03.004.L", [0.128524, 0.056455,  0.071998, -0.156979, -0.078652]),
        ("Def_leg_04.000.L", [0.123769, 0.060811,  0.022075, -0.015847, -0.036792]),
        ("Def_leg_04.001.L", [0.140865, 0.056084,  0.175245,  0.012156, -0.049158]),
        ("Def_leg_04.003.L", [0.067575, 0.056968,  0.080249, -0.043055, -0.069306]),
        ("Def_leg_04.004.L", [0.134749, 0.049772,  0.050016, -0.158901, -0.078713]),
        ("Def_pincer.000a.L", [0.228180, 0.099807,  0.146852, -0.112937,  0.199940]),
        ("Def_pincer.005.L", [0.106954, 0.073292, -0.054414, -0.106502,  0.033880]),
        ("Def_pincer.006b.L", [0.095839, 0.050577, -0.036690, -0.090157,  0.008504]),
        ("Def_leg_01.000.R", [0.106833, 0.057397, -0.013619, -0.016628, -0.004180]),
        ("Def_leg_01.001.R", [0.142467, 0.061588, -0.174220,  0.018777,  0.049600]),
        ("Def_leg_01.003.R", [0.060660, 0.061648, -0.087472, -0.032786, -0.010803]),
        ("Def_leg_01.004.R", [0.122645, 0.054299, -0.068537, -0.154283, -0.023502]),
        ("Def_leg_02.000.R", [0.006161, 0.164457, -0.013849, -0.016922, -0.001611]),
        ("Def_leg_02.001.R", [0.181178, 0.066546, -0.216730,  0.015234,  0.028623]),
        ("Def_leg_02.003.R", [0.054145, 0.071458, -0.097443, -0.034404, -0.027830]),
        ("Def_leg_02.004.R", [0.123662, 0.057603, -0.079551, -0.152684, -0.029877]),
        ("Def_leg_03.000.R", [0.073876, 0.066578, -0.087574,  0.007707, -0.028267]),
        ("Def_leg_03.001.R", [0.170628, 0.068806, -0.212199,  0.018997, -0.018147]),
        ("Def_leg_03.003.R", [0.066288, 0.066312, -0.095621, -0.037249, -0.052509]),
        ("Def_leg_03.004.R", [0.128524, 0.056455, -0.071998, -0.156979, -0.078652]),
        ("Def_leg_04.000.R", [0.113170, 0.059998, -0.030447, -0.015496, -0.044862]),
        ("Def_leg_04.001.R", [0.140865, 0.056084, -0.175245,  0.012156, -0.049158]),
        ("Def_leg_04.003.R", [0.067575, 0.056968, -0.080249, -0.043055, -0.069305]),
        ("Def_leg_04.004.R", [0.134749, 0.049772, -0.050016, -0.158901, -0.078713]),
        ("Def_pincer.000a.R", [0.239028, 0.100576, -0.142431, -0.113873,  0.189508]),
        ("Def_pincer.005.R", [0.103558, 0.076635,  0.058884, -0.105650,  0.032336]),
        ("Def_pincer.006b.R", [0.091664, 0.045815,  0.037039, -0.098493,  0.011171]),
        ("Def_antennae.L", [0.011954, 0.030000,  0.023860,  0.021847,  0.026712]),
        ("Def_antennae_top.L", [0.015000, 0.030000,  0.025592,  0.023433,  0.028652]),
        ("Def_antennae.R", [0.011954, 0.030000, -0.023860,  0.021847,  0.026712]),
        ("Def_antennae_top.R", [0.015000, 0.030000, -0.025592,  0.023433,  0.028652]),
    ];

    #[test]
    fn fitted_geometry_matches_golden() {
        use crate::mesh_fallback::constructed_body_digest;
        let Some(path) = model_path() else {
            eprintln!("fitted_geometry_matches_golden: no model — skipping");
            return;
        };
        let digest = constructed_body_digest();
        assert_eq!(
            digest, GOLDEN_ASSET_DIGEST,
            "crab asset changed (digest {digest:#018x} != golden {GOLDEN_ASSET_DIGEST:#018x}) — \
             the fitted colliders moved; re-capture FITTED_GOLDEN and expect a retrain"
        );

        let recipe = build_recipe(&LoadedModel::load(&path).expect("load model")).expect("recipe");
        let golden: std::collections::HashMap<&str, [f32; 5]> =
            FITTED_GOLDEN.iter().map(|(b, g)| (*b, *g)).collect();
        assert_eq!(
            golden.len(),
            FITTED_GOLDEN.len(),
            "duplicate bone in FITTED_GOLDEN"
        );
        assert_eq!(
            recipe.links.len(),
            FITTED_GOLDEN.len(),
            "fitted link count {} != golden {}",
            recipe.links.len(),
            FITTED_GOLDEN.len()
        );
        const TOL: f32 = 1e-4;
        for link in &recipe.links {
            let g = golden
                .get(link.bone.as_str())
                .unwrap_or_else(|| panic!("fitted link {} absent from golden", link.bone));
            let got = [
                link.half_height,
                link.radius,
                link.center.x,
                link.center.y,
                link.center.z,
            ];
            for (k, (a, b)) in got.iter().zip(g).enumerate() {
                let field = ["half_height", "radius", "center.x", "center.y", "center.z"][k];
                assert!(
                    (a - b).abs() <= TOL,
                    "{}: {field} drifted {a:.6} vs golden {b:.6} (Δ={:.6} > {TOL})",
                    link.bone,
                    (a - b).abs()
                );
            }
        }
    }
}
