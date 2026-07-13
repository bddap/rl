use std::collections::HashMap;
use std::sync::OnceLock;

use bevy::prelude::*;

use crate::bot::body::{CrabJointId, Side};
use crate::bot::meshfit::{FittedShape, PartId, ShapePolicy};

use super::{BindSource, LinkShape, RigLink, RigRecipe};

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
        Some(pts) if pts.len() >= 8 => crate::bot::meshfit::fit_link_shape(
            pts,
            (chain_dir.length_squared() > 0.5).then_some(chain_dir),
            shape_policy(actuated),
        ),
        _ => None,
    };
    let (center, col_rot, shape) = match fitted {
        Some(FittedShape::Capsule(cap)) => {
            let seg = cap.b - cap.a;
            (
                (cap.a + cap.b) * 0.5 - c_origin,
                arc_to(Vec3::Y, seg),
                LinkShape::Capsule {
                    half_height: seg.length() * 0.5,
                    radius: cap.radius,
                },
            )
        }
        Some(FittedShape::Cuboid { center, rot, half }) => {
            (center - c_origin, rot, LinkShape::Cuboid { half })
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
                LinkShape::Capsule {
                    half_height: (seg_len * 0.5 - fixed_radius).max(0.01),
                    radius: fixed_radius,
                },
            )
        }
    };

    Some(RigLink {
        bone: pivot_name.to_string(),
        parent: parent_idx,
        anchor1,
        axis_local,
        shape,
        center,
        col_rot,
        density,
        actuated,
    })
}

/// Feet plant and roll on their spherical capsule tips, and the pincer collider is
/// read back via `as_capsule` for the sim's claw-touch decisions
/// (net::external_crab) — both stay capsules whatever the fit score says.
fn shape_policy(actuated: Option<CrabJointId>) -> ShapePolicy {
    match actuated {
        Some(CrabJointId::LegCarpus(..) | CrabJointId::ClawPincer(_)) => ShapePolicy::CapsuleOnly,
        _ => ShapePolicy::Any,
    }
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
        // The check swings the arm FLESH (the skinned part clouds), not the fitted
        // colliders: a cuboid's empty corners would overstate the flesh reach.
        let clouds = model.vertices_by_part();
        for side in [Side::Left, Side::Right] {
            let sh_idx = recipe
                .links
                .iter()
                .position(|l| l.actuated == Some(CrabJointId::ClawShoulder(side)))
                .expect("shoulder link present");
            let pivot = world[sh_idx];
            let [lo, _hi] = CrabJointId::ClawShoulder(side).limits();
            let rot = Quat::from_axis_angle(recipe.links[sh_idx].axis_local, lo);
            for id in [
                CrabJointId::ClawShoulder(side),
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

    #[derive(Clone, Copy, Debug)]
    enum GoldenShape {
        Capsule { half_height: f32, radius: f32 },
        Cuboid { half: [f32; 3] },
    }

    struct GoldenLink {
        bone: &'static str,
        shape: GoldenShape,
        center: [f32; 3],
        rot: [f32; 4],
    }

    const fn cap(
        bone: &'static str,
        half_height: f32,
        radius: f32,
        center: [f32; 3],
        rot: [f32; 4],
    ) -> GoldenLink {
        GoldenLink {
            bone,
            shape: GoldenShape::Capsule {
                half_height,
                radius,
            },
            center,
            rot,
        }
    }

    const fn cub(
        bone: &'static str,
        half: [f32; 3],
        center: [f32; 3],
        rot: [f32; 4],
    ) -> GoldenLink {
        GoldenLink {
            bone,
            shape: GoldenShape::Cuboid { half },
            center,
            rot,
        }
    }

    /// Sign-canonicalised quat components — `from_rotation_arc`/`from_mat3` may land
    /// on either cover of the same rotation.
    fn canon_quat(q: Quat) -> [f32; 4] {
        let a = q.to_array();
        let flip = match a.iter().rev().find(|c| c.abs() > 1e-6) {
            Some(&c) => c < 0.0,
            None => false,
        };
        if flip {
            [-a[0], -a[1], -a[2], -a[3]]
        } else {
            a
        }
    }

    // Regenerates the FITTED_GOLDEN literal below — run with --ignored --nocapture
    // after a deliberate geometry change and paste the output over the table.
    #[test]
    #[ignore = "golden regeneration helper, not a check"]
    fn dump_fitted_golden() {
        let Some(path) = model_path() else {
            eprintln!("dump_fitted_golden: no model — skipping");
            return;
        };
        let recipe = build_recipe(&LoadedModel::load(&path).expect("load model")).expect("recipe");
        let f = |v: f32| format!("{v:.6}");
        let arr = |v: &[f32]| {
            format!(
                "[{}]",
                v.iter().map(|&x| f(x)).collect::<Vec<_>>().join(", ")
            )
        };
        for link in &recipe.links {
            let c = arr(&link.center.to_array());
            let q = arr(&canon_quat(link.col_rot));
            match link.shape {
                LinkShape::Capsule {
                    half_height,
                    radius,
                } => println!(
                    "        cap(\"{}\", {}, {}, {c}, {q}),",
                    link.bone,
                    f(half_height),
                    f(radius)
                ),
                LinkShape::Cuboid { half } => println!(
                    "        cub(\"{}\", {}, {c}, {q}),",
                    link.bone,
                    arr(&half.to_array())
                ),
            }
        }
    }

    #[rustfmt::skip]
    const FITTED_GOLDEN: &[GoldenLink] = &[
        cub("Def_leg_01.000.L", [0.152088, 0.076075, 0.041194], [0.021144, -0.017276, 0.012287], [0.611172, -0.099488, 0.133927, 0.773715]),
        cub("Def_leg_01.001.L", [0.204055, 0.087031, 0.053179], [0.168491, 0.034778, 0.062931], [0.306499, -0.143787, 0.084574, 0.937140]),
        cub("Def_leg_01.003.L", [0.122308, 0.072334, 0.037688], [0.089255, -0.029936, -0.011805], [0.318536, -0.000803, -0.318508, 0.892797]),
        cap("Def_leg_01.004.L", 0.122645, 0.054299, [0.068538, -0.154283, -0.023502], [0.109015, -0.000000, 0.167599, 0.979809]),
        cub("Def_leg_02.000.L", [0.155243, 0.133579, 0.074599], [0.026584, 0.075285, 0.006483], [0.000000, 0.000000, 0.000000, 1.000000]),
        cub("Def_leg_02.001.L", [0.247724, 0.087315, 0.055857], [0.213114, 0.029489, 0.041325], [0.294107, -0.074708, 0.068380, 0.950391]),
        cub("Def_leg_02.003.L", [0.125603, 0.081812, 0.036506], [0.095694, -0.036064, -0.030874], [0.377896, 0.001468, -0.280721, 0.882263]),
        cap("Def_leg_02.004.L", 0.123662, 0.057603, [0.079551, -0.152684, -0.029877], [0.127793, -0.000000, 0.188765, 0.973672]),
        cub("Def_leg_03.000.L", [0.140454, 0.113343, 0.062148], [0.096656, 0.042294, -0.032425], [-0.054584, 0.111633, -0.131877, 0.983447]),
        cub("Def_leg_03.001.L", [0.239433, 0.089718, 0.054107], [0.211514, 0.028718, -0.009547], [0.302863, 0.002104, 0.029747, 0.952567]),
        cub("Def_leg_03.003.L", [0.132601, 0.076186, 0.039267], [0.093246, -0.039953, -0.053762], [0.321689, 0.148990, -0.311887, 0.881501]),
        cap("Def_leg_03.004.L", 0.128524, 0.056455, [0.071998, -0.156979, -0.078652], [0.187891, -0.000000, 0.169116, 0.967521]),
        cub("Def_leg_04.000.L", [0.184580, 0.079608, 0.042361], [0.014989, -0.014968, -0.047346], [0.644533, 0.232046, -0.174906, 0.707206]),
        cub("Def_leg_04.001.L", [0.196948, 0.075875, 0.051187], [0.175866, 0.031600, -0.040112], [0.342983, 0.115145, -0.004672, 0.932246]),
        cub("Def_leg_04.003.L", [0.124543, 0.066511, 0.032990], [0.078340, -0.044795, -0.070163], [0.284312, 0.237543, -0.330143, 0.868185]),
        cap("Def_leg_04.004.L", 0.134749, 0.049772, [0.050016, -0.158901, -0.078713], [0.200942, -0.000000, 0.120450, 0.972170]),
        cub("Def_pincer.000a.L", [0.327987, 0.119014, 0.091226], [0.156361, -0.105584, 0.195727], [-0.588974, -0.440968, -0.369851, 0.567333]),
        cub("Def_pincer.005.L", [0.086848, 0.167349, 0.091270], [-0.033627, -0.107730, 0.018923], [0.000000, 0.000000, 0.000000, 1.000000]),
        cap("Def_pincer.006b.L", 0.082743, 0.063441, [-0.031627, -0.083131, 0.008357], [0.173875, -0.000000, 0.971086, 0.163580]),
        cub("Def_leg_01.000.R", [0.164229, 0.076084, 0.041290], [-0.009473, -0.016347, 0.007631], [0.611800, 0.107336, -0.133123, 0.772307]),
        cub("Def_leg_01.001.R", [0.204055, 0.087031, 0.053179], [-0.168491, 0.034778, 0.062931], [0.306499, 0.143787, -0.084574, 0.937140]),
        cub("Def_leg_01.003.R", [0.122308, 0.072334, 0.037688], [-0.089254, -0.029936, -0.011805], [0.318536, 0.000803, 0.318508, 0.892797]),
        cap("Def_leg_01.004.R", 0.122645, 0.054299, [-0.068537, -0.154283, -0.023502], [0.109015, 0.000000, -0.167599, 0.979809]),
        cub("Def_leg_02.000.R", [0.167668, 0.135392, 0.088010], [-0.014160, 0.077098, 0.019894], [0.000000, 0.000000, 0.000000, 1.000000]),
        cub("Def_leg_02.001.R", [0.247724, 0.087315, 0.055857], [-0.213114, 0.029489, 0.041325], [0.294107, 0.074708, -0.068380, 0.950391]),
        cub("Def_leg_02.003.R", [0.125603, 0.081812, 0.036506], [-0.095694, -0.036064, -0.030875], [0.377896, -0.001468, 0.280721, 0.882263]),
        cap("Def_leg_02.004.R", 0.123662, 0.057603, [-0.079551, -0.152684, -0.029877], [0.127793, 0.000000, -0.188765, 0.973672]),
        cub("Def_leg_03.000.R", [0.140454, 0.113343, 0.062148], [-0.096656, 0.042294, -0.032425], [-0.054584, -0.111633, 0.131877, 0.983447]),
        cub("Def_leg_03.001.R", [0.239433, 0.089718, 0.054107], [-0.211514, 0.028718, -0.009547], [0.302863, -0.002104, -0.029747, 0.952567]),
        cub("Def_leg_03.003.R", [0.132601, 0.076186, 0.039267], [-0.093245, -0.039953, -0.053762], [0.321689, -0.148990, 0.311887, 0.881502]),
        cap("Def_leg_03.004.R", 0.128524, 0.056455, [-0.071998, -0.156979, -0.078652], [0.187891, 0.000000, -0.169116, 0.967521]),
        cub("Def_leg_04.000.R", [0.173168, 0.080259, 0.042373], [-0.026232, -0.014095, -0.051513], [0.648582, -0.221440, 0.163258, 0.709685]),
        cub("Def_leg_04.001.R", [0.196948, 0.075875, 0.051187], [-0.175866, 0.031601, -0.040112], [0.342983, -0.115145, 0.004671, 0.932246]),
        cub("Def_leg_04.003.R", [0.124543, 0.066511, 0.032990], [-0.078340, -0.044795, -0.070163], [0.284312, -0.237542, 0.330143, 0.868185]),
        cap("Def_leg_04.004.R", 0.134749, 0.049772, [-0.050016, -0.158901, -0.078713], [0.200942, 0.000000, -0.120450, 0.972170]),
        cub("Def_pincer.000a.R", [0.339604, 0.118920, 0.090338], [-0.152483, -0.106471, 0.185150], [-0.441178, -0.584980, -0.571307, 0.369823]),
        cub("Def_pincer.005.R", [0.090010, 0.167350, 0.091270], [0.036789, -0.107730, 0.018923], [0.000000, 0.000000, 0.000000, 1.000000]),
        cap("Def_pincer.006b.R", 0.087952, 0.058100, [0.035186, -0.094717, 0.010937], [0.173878, 0.000000, -0.971086, 0.163581]),
        cap("Def_antennae.L", 0.011954, 0.030000, [0.023860, 0.021847, 0.026712], [0.365085, 0.000000, -0.326101, 0.871993]),
        cap("Def_antennae_top.L", 0.015000, 0.030000, [0.025592, 0.023433, 0.028652], [0.365085, 0.000000, -0.326101, 0.871993]),
        cap("Def_antennae.R", 0.011954, 0.030000, [-0.023860, 0.021847, 0.026712], [0.365085, -0.000000, 0.326101, 0.871993]),
        cap("Def_antennae_top.R", 0.015000, 0.030000, [-0.025592, 0.023433, 0.028652], [0.365085, -0.000000, 0.326101, 0.871993]),
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
        let golden: HashMap<&str, &GoldenLink> =
            FITTED_GOLDEN.iter().map(|g| (g.bone, g)).collect();
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
        let check = |bone: &str, field: &str, a: f32, b: f32| {
            assert!(
                (a - b).abs() <= TOL,
                "{bone}: {field} drifted {a:.6} vs golden {b:.6} (Δ={:.6} > {TOL})",
                (a - b).abs()
            );
        };
        for link in &recipe.links {
            let g = golden
                .get(link.bone.as_str())
                .unwrap_or_else(|| panic!("fitted link {} absent from golden", link.bone));
            for (k, (a, b)) in link.center.to_array().iter().zip(&g.center).enumerate() {
                check(&link.bone, ["center.x", "center.y", "center.z"][k], *a, *b);
            }
            for (k, (a, b)) in canon_quat(link.col_rot).iter().zip(&g.rot).enumerate() {
                check(&link.bone, ["rot.x", "rot.y", "rot.z", "rot.w"][k], *a, *b);
            }
            match (link.shape, g.shape) {
                (
                    LinkShape::Capsule {
                        half_height,
                        radius,
                    },
                    GoldenShape::Capsule {
                        half_height: gh,
                        radius: gr,
                    },
                ) => {
                    check(&link.bone, "half_height", half_height, gh);
                    check(&link.bone, "radius", radius, gr);
                }
                (LinkShape::Cuboid { half }, GoldenShape::Cuboid { half: gh }) => {
                    for (k, (a, b)) in half.to_array().iter().zip(&gh).enumerate() {
                        check(&link.bone, ["half.x", "half.y", "half.z"][k], *a, *b);
                    }
                }
                (got, want) => panic!(
                    "{}: primitive KIND changed — fitted {got:?} vs golden {want:?}; \
                     that is a new MDP and a determinism break, re-capture deliberately",
                    link.bone
                ),
            }
        }
    }
}
