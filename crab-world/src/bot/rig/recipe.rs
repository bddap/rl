//! The canonical rig decomposition and the bind-pose → [`RigRecipe`] builder: the
//! joint chains ([`joint_specs`]), the one bone→part mapping every consumer shares
//! ([`part_for_bone`]), and the per-link geometry derivation ([`derive_link`]).

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

/// Eye-stalks carry no policy joint and aren't load-bearing, so there's no cloud
/// worth fitting — a fixed thin radius is honest for a cosmetic stalk.
const EYE_RADIUS: f32 = 0.03;
/// Fallback radius when a joint's vertex cloud is too sparse to fit.
const FALLBACK_RADIUS: f32 = 0.03;

/// Every link's unrotated origin in `root`'s frame, by telescoping each `anchor1`
/// (a parent-relative delta) down the parent-before-child chain. Links are emitted
/// parent-first, so `world[parent]` is always filled before a child reads it. The
/// single source for this walk: the body spawn ([`crate::bot::body`]'s `spawn_crab`)
/// runs it from the spawn-translated hub, and `rest_colliders` from the bare
/// bind-world hub — same telescoping, the constant translation simply cancels.
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

/// One actuated joint, located against the bind-pose skeleton. The capsule runs
/// from `pivot` to `tip` (or stubs forward when `tip` is `None`, for a leaf like
/// the pincer finger); `members` are every deform bone whose flesh and skin ride
/// this link, used to bucket the vertex cloud the collider radius is fitted to.
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

// Deform-bone names in ONE place, so `joint_specs` and `FallbackModel` can't drift
// into an incomplete recipe.
/// `leg` is 0-based (the rig labels legs `01`..`04`), `seg` is the segment tag
/// (`000`,`003`,`004`,`005`, …).
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

/// The canonical rig decomposition: one chain of actuated joints per limb. Bones
/// are bracketed by chain position — a joint owns every deform bone from its pivot
/// up to the next joint's pivot — so flesh, skin, and collider all agree on which
/// link a bone belongs to. Bracket choices are verified against the collider
/// screenshots, not derived; adjust a `members`/`tip` if a capsule misses its mesh.
fn joint_specs() -> Vec<Vec<JointSpec>> {
    let mut chains = Vec::new();
    for side in [Side::Left, Side::Right] {
        // Legs: front→back, same 0..3 order as the policy. The basal joint is 2-DOF,
        // split into two short links sharing the proximal cluster: the coxa swings the
        // leg fore/aft at the body root (`000`), then the basis lifts it up/down at the
        // coxo-basal joint (`001`); the merus and carpus are the two load-bearing distal
        // bends. Splitting the old single rigid coxa (`000`..`002`) this way earns back
        // the levator/depressor DOF a real Sally has (bddap/rl#31) without a massless
        // virtual link — each link carries real flesh. The swing and lift hinges sit at
        // DIFFERENT pivots (`000` then `001`), not co-located: this is the real crustacean
        // thoraco-coxal → coxo-basal arrangement, not a ball joint at the hip, so lifting
        // also telescopes the merus origin slightly along the swung direction — intended.
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
        // Claw (cheliped), distal segments read off the bind pose (see the
        // `dump_left_cheliped_chain` diagnostic): the bone chain is linear
        // `000a→…→003→004→005→006b→006`, where `003` is the long forearm, `004` a
        // short narrow node (the carpus / true wrist), `005` the broad palm
        // (propodus — by far the fattest cloud, y-extent ~0.34, with the FIXED
        // finger/pollex baked into it as it has no own bone), and `006b→006` the
        // single movable finger (dactyl) that folds off the palm.
        //
        // So the joints bracket as the anatomy dictates: the shoulder owns the arm
        // up to and INCLUDING the carpus (`004`); the wrist pivots at the
        // carpus↔propodus joint (base of the palm `005`) and owns the palm, so
        // actuating it swings the whole hand — palm, pollex, and (as kinematic
        // descendants) the dactyl — as one rigid unit; the pincer pivots at the
        // propodus↔dactyl joint (`006b`) and owns only the movable finger. The old
        // rig folded the palm into the rigid shoulder link and pivoted the wrist out
        // at `006b`, so the wrist bent only the finger against a hand welded to the
        // arm — the owner-reported "wrist moves just the thumb" with stretched chitin.
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

/// The bone→part lookup, built once from [`joint_specs`]. A bone that names a
/// joint member maps to that joint; this is the live half of the one mapping.
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

/// Map a deform/control bone to the physics part it drives — the one mapping every
/// consumer shares. A bone listed in a [`JointSpec`] rides that joint's link;
/// every other rig bone (shell, thorax, eyes, palpi, mouth…) rides the carapace.
/// Returns `None` only for non-rig nodes (no `Def_`/`Ctrl_` prefix).
pub fn part_for_bone(name: &str) -> Option<PartId> {
    if !(name.starts_with("Def_") || name.starts_with("Ctrl_")) {
        return None;
    }
    Some(member_map().get(name).copied().unwrap_or(PartId::Carapace))
}

/// Unordered adjacency over the physics parts, derived from the joint graph — the
/// pairs of parts that meet at a hinge a skin vertex may legitimately blend across.
/// Two parts are adjacent iff their links are joined in the rig: consecutive joints
/// in a limb chain (the parent/child kinematic link), and each chain's *first* joint
/// with [`PartId::Carapace`], since that link hangs off the carapace root (see
/// [`build_recipe`], where a chain's first link's parent is `None` = the carapace).
/// Both `(a, b)` and `(b, a)` are present, so the lookup is a plain `contains`.
///
/// The skin weight strip ([`crate::bot::skin`]) keeps both parts' lanes at an adjacent
/// seam — a vertex spanning a hinge must bend with both links, not rigidly drag with
/// one — but still zeroes a lane on a NON-adjacent (spatially disjoint) part, so the
/// carapace-vs-arm bleed it fixes stays fixed. This is the one place that knows which
/// seams are real, and it reads the same [`joint_specs`] decomposition the body
/// spawns from, so it cannot disagree with the rig about which links are joined.
fn part_adjacency() -> &'static std::collections::HashSet<(PartId, PartId)> {
    static ADJ: OnceLock<std::collections::HashSet<(PartId, PartId)>> = OnceLock::new();
    ADJ.get_or_init(|| {
        let mut adj = std::collections::HashSet::new();
        let mut link = |a: PartId, b: PartId| {
            adj.insert((a, b));
            adj.insert((b, a));
        };
        for chain in joint_specs() {
            // The chain's first link hangs off the carapace root.
            if let Some(first) = chain.first() {
                link(PartId::Carapace, PartId::Joint(first.id));
            }
            // Consecutive joints share a kinematic parent/child link.
            for pair in chain.windows(2) {
                link(PartId::Joint(pair[0].id), PartId::Joint(pair[1].id));
            }
        }
        adj
    })
}

/// Whether two physics parts meet at a rig hinge (see [`part_adjacency`]). A part is
/// not adjacent to itself; the strip only consults this for two *distinct* parts.
pub fn parts_adjacent(a: PartId, b: PartId) -> bool {
    part_adjacency().contains(&(a, b))
}

/// Build the whole-body recipe from the model's bind pose, or `None` if the model
/// lacks the expected bones or carries a non-finite bind transform.
pub fn build_recipe(model: &impl BindSource) -> Option<RigRecipe> {
    let carapace_center = leg_hub_centroid(model)?;
    // The hub seeds every anchor (the link chain telescopes off it), so a NaN/inf
    // leg-root translation here would poison the whole body and crash the solver on
    // the spawn step — before `rescue_nonfinite_crabs` can ever see it. Reject the
    // model now; the recipe is re-checked whole below to catch a non-finite origin
    // on any other bone too.
    if !carapace_center.is_finite() {
        return None;
    }
    let clouds = model.vertices_by_part();

    let mut links: Vec<RigLink> = Vec::new();
    for chain in joint_specs() {
        let mut parent_idx = None; // first link in each chain hangs off the carapace
        let mut parent_pivot = carapace_center;
        for spec in &chain {
            let Some(pivot) = model.bone_origin(&spec.pivot) else {
                break; // a missing bone truncates this limb, not the whole body
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

    // Eye-stalks (locked, cosmetic): base (carapace-parented) + tip. The eye rides the
    // stalk, so the tip is re-parented onto the base here. These links stay in the rig
    // for the cosmetic/debug view + the fallback body, but `spawn_crab` does NOT give
    // them physics bodies (no policy joint, not observed, not load-bearing).
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
    // A non-finite origin on any non-hub bone slips past the hub check above but
    // still poisons that link's anchor. Make a non-finite recipe unrepresentable
    // downstream: reject the whole thing once here so spawn only ever sees clean
    // geometry (the degenerate-but-finite axis cases are already guarded in
    // `derive_link`/`bend_axis`).
    recipe.is_finite().then_some(recipe)
}

/// `spawn_crab` does not give the cosmetic locked links (eye-stalks, `actuated ==
/// None`) physics bodies; it leaves a carapace placeholder in their `ents` slot to
/// keep the index alignment with `recipe.links`. That placeholder is only sound while
/// no ACTUATED link parents a locked one — otherwise a kept link would silently
/// reparent onto the carapace. The rig builds locked links as leaves of the carapace
/// or of other locked links, so this holds; assert it at construction so a future rig
/// edit that violates it fails loudly here, not as a mysterious mis-jointed crab.
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

/// Body centre = the centroid of the eight leg roots (bone `000`), the hub the
/// limbs hang off: symmetric in x, mid-height in y. Carapace-relative anchors are
/// measured from here, and the carapace box is offset relative to it.
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

/// The bind-pose geometry inputs for one link, grouped so the run of similarly
/// typed args (the two `f32`s, the cloud, the parent index) can't be silently
/// transposed at a call site.
struct LinkGeom<'a> {
    /// The parent link's bind-pose-world origin; this link's `anchor1` is the delta
    /// from it.
    parent_pivot: Vec3,
    /// Index of the parent link, or `None` when the link hangs off the carapace root.
    parent_idx: Option<usize>,
    /// Flesh to fit the capsule radius to; `None`/sparse falls back to `fixed_radius`.
    cloud: Option<&'a [Vec3]>,
    fixed_radius: f32,
    density: f32,
}

/// Derive one link's joint + collider geometry from the bind pose. `tip = None`
/// makes a short stub along the incoming direction (a leaf bone). `cloud = Some`
/// fits the capsule radius to that vertex cloud; `None` uses `fixed_radius`.
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
    // Links spawn axis-aligned, so the parent frame is world: the anchor is a plain
    // world delta between bind-pose bone origins.
    let anchor1 = c_origin - parent_pivot;
    let in_dir = anchor1.normalize_or_zero();

    // The joint's free axis follows the KINEMATIC bone chain (toward the tip — the
    // direction the limb bends), independent of the collider shape below.
    let chain_dir = match tip_name.and_then(|t| model.bone_origin(t)) {
        Some(g) => (g - c_origin).normalize_or_zero(),
        None => in_dir,
    };
    let axis_local = bend_axis(actuated, in_dir, chain_dir);

    // Collider shape. With the flesh cloud, fit the capsule to it (best-fit
    // centre/axis/radius) so it hugs the actual limb — its curve and lateral offset
    // from the straight bone line included. Placing the capsule on the bone line
    // alone left 30–80% of the mesh poking out (see `--verify-colliders`). The joint
    // pivot still sits at the bone origin (the link spawns there); the capsule is
    // merely offset within the link, the same way a fitted collider hangs off its
    // joint. A sparse/absent cloud (eye-stalks) falls back to a bone-line stub.
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

/// The free rotation axis of a revolute joint. The coxa swings the leg fore/aft, so
/// it gets a vertical axis regardless of the (degenerate) body→coxa direction.
/// Otherwise the natural bend axis is `in × out`; when the bind chain is near-
/// straight that cross collapses, so fall back to a horizontal perpendicular — and
/// never a zero axis, which degenerates a revolute into NaNs that poison the
/// multibody and recur on every respawn.
fn bend_axis(actuated: Option<CrabJointId>, in_dir: Vec3, out_dir: Vec3) -> Vec3 {
    if matches!(actuated, Some(CrabJointId::LegCoxa(..))) {
        return Vec3::Y;
    }
    // The basis LIFTS the leg, so its axis is horizontal and perpendicular to the
    // leg's outward run — `out × Y` strips the bone's vertical component to leave a
    // pure horizontal hinge, so the DOF is up/down levation, orthogonal to the coxa's
    // fore/aft swing. Falls back to X only if the leg points straight up (it never does).
    if matches!(actuated, Some(CrabJointId::LegBasis(..))) {
        let axis = out_dir.cross(Vec3::Y);
        return if axis.length() > 0.5 {
            axis.normalize()
        } else {
            Vec3::X
        };
    }
    // Owner-tuned wrist sweep axis (az=63.5 el=-15.9). Same
    // parent/world-at-rest frame as the derived `in × out`, so it drops straight in
    // here; the owner dialed it off the kinematic bend to get the hand swinging the
    // way he wants, overriding the auto-derived cross.
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

/// Rotation taking `from` onto `to`, guarding the degenerate (near-zero `to`)
/// case `Quat::from_rotation_arc` would choke on.
fn arc_to(from: Vec3, to: Vec3) -> Quat {
    if to.length_squared() < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_arc(from, to.normalize())
    }
}

/// The trunk bones whose vertex flesh defines the carapace box. Leg- and claw-root
/// bones are excluded (their verts belong to the limb links), and so are the
/// lateral shoulder bones `Def_shell.003`/`.006`, which sit out over the leg sockets
/// and would stretch the box wide enough to swallow the legs.
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

/// Carapace box from the trunk's vertex cloud: half-extents and the box centre as
/// an offset from `center` (the leg hub the links anchor to). Using the actual
/// shell vertices — not bone origins — covers the trunk's flesh directly.
fn carapace_box(model: &impl BindSource, center: Vec3) -> (Vec3, Vec3) {
    let pts = model.vertices_for_bones(&TRUNK_BONES);
    if pts.len() < 4 {
        return (Vec3::splat(0.1), Vec3::ZERO); // sparse model: a small box at the hub
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
    use super::super::colliders::{RestShape, link_capsule};
    use crate::bot::meshfit::{LoadedModel, model_path};
    use crate::bot::rig::fallback_recipe;

    /// Diagnostic (run with `--nocapture`): dump the LEFT cheliped's bone chain from
    /// the model's bind pose — each `Def_pincer.*.L` bone's parent and bind-world
    /// origin, ordered proximal→distal — so the carpus/propodus/pollex/dactyl segment
    /// identities can be read off the geometry rather than guessed. Re-parses the GLB
    /// for the node parent links (the production [`LoadedModel`] keeps only world
    /// transforms, not the hierarchy) and reads origins through `bone_origin`. Skips
    /// cleanly when the model isn't present.
    #[test]
    fn dump_left_cheliped_chain() {
        let Some(path) = model_path() else {
            eprintln!("dump_left_cheliped_chain: no model — skipping");
            return;
        };
        let model = LoadedModel::load(&path).expect("load model");
        let bytes = std::fs::read(&path).expect("read glb");
        let gltf = gltf::Gltf::from_slice(&bytes).expect("parse glb");

        // node index -> parent node index, from the scene hierarchy.
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

        // Every left-cheliped bone, with its origin and parent name.
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

        // Order proximal→distal by walking the parent chain from the root-most pincer
        // bone, so the printed sequence is the actual kinematic order, not node order.
        let names: std::collections::HashSet<&str> =
            rows.iter().map(|(n, _, _)| n.as_str()).collect();
        let mut child_of: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut root: Option<&str> = None;
        for (n, p, _) in &rows {
            if names.contains(p.as_str()) {
                child_of.entry(p.as_str()).or_default().push(n.as_str());
            } else {
                root = Some(n.as_str()); // parent outside the pincer set = chain root
            }
        }
        let mut order: Vec<&str> = Vec::new();
        let mut stack: Vec<&str> = root.into_iter().collect();
        while let Some(n) = stack.pop() {
            order.push(n);
            if let Some(kids) = child_of.get(n) {
                // Deepest-last so siblings print in a stable order.
                let mut kids = kids.clone();
                kids.sort_unstable();
                stack.extend(kids.into_iter().rev());
            }
        }

        let origin_of = |name: &str| rows.iter().find(|(n, _, _)| n == name).map(|(_, _, o)| *o);
        // Per-bone vertex cloud size + extent distinguishes the broad propodus (hand,
        // bearing both fingers — many verts, fat box) from the thin arm segments.
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
        // EVERY node carrying "pincer" (any case, both sides) — to surface any
        // control/tail bone the `Def_pincer.NNN.L` chain dump doesn't cover, and to
        // confirm whether a separate dactyl bone exists.
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

    /// At the shoulder up-stop, the WHOLE cheliped CAPSULE ENVELOPE (the shoulder link
    /// and its rigid descendants — forearm, palm, finger) stays at or below the carapace
    /// top — the regression guard for the "arm intersects eye and carapace" bug
    /// (bddap/rl#41 + its refinement). It checks the fitted capsule FLESH, not the bare
    /// bone center: the highest point of each link's collider is `center + radius` along
    /// its capsule, and at −0.6 the palm BONE CENTER sat just under the shell while the
    /// fitted capsule still topped out ~0.09 above it — exactly the gap the old
    /// center-only check missed. Geometry is read straight from the spawned recipe (each
    /// link's own pivot, `axis_local`, capsule `center`/`col_rot`/`half_height`/`radius`),
    /// so it can't drift onto a phantom axis if the cheliped is re-bracketed. The whole
    /// subtree rides the shoulder rotation rigidly (wrist/pincer at their rest): a sweep
    /// over the wrist/pincer travel confirmed that driving the distal joints never lifts
    /// the flesh above this rigid-rest pose, so it is the empirical worst case for the
    /// up-reach. Pure geometry (reuses [`link_capsule`]); skips with no model.
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
        // Bind-world origin of every link — a link's origin IS its joint pivot.
        let world = link_world_origins(&recipe.links, hub);
        for side in [Side::Left, Side::Right] {
            let sh_idx = recipe
                .links
                .iter()
                .position(|l| l.actuated == Some(CrabJointId::ClawShoulder(side)))
                .expect("shoulder link present");
            let pivot = world[sh_idx];
            // `axis_local` is in the parent frame, which is world at the bind rest (the
            // parent carapace spawns unrotated), so it rotates the subtree directly.
            let [lo, _hi] = CrabJointId::ClawShoulder(side).limits();
            let rot = Quat::from_axis_angle(recipe.links[sh_idx].axis_local, lo);
            // Every cheliped link on this side rides the shoulder rotation as one rigid body.
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
                let RestShape::Capsule { a, b, radius } = link_capsule(link, world[i]) else {
                    unreachable!("cheliped links spawn as capsules");
                };
                // Highest point of the rotated capsule = the higher cylinder cap (endpoint
                // swung about the shoulder pivot) plus the hemispherical cap's radius.
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

    /// The chains must enumerate exactly the actuated joint set — same count, all
    /// distinct. Guards the "add a joint" footgun: a new `CrabJointId` with no
    /// matching `JointSpec` (or a duplicate) is caught here, not silently at runtime.
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

    /// Every member bone resolves to its own joint — the canonical map and the
    /// chains agree by construction. Pinned so a future edit can't route a limb
    /// bone to the wrong joint or the carapace (the divergence that forked the
    /// legs), including a member accidentally shared between two joints.
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

    /// Phase 0's promised but missing fitted-vs-hand-coded drift-guard (issue #20): the
    /// fitted body ([`build_recipe`] on the real glTF) and the hand-coded fallback
    /// ([`fallback_recipe`]) must enumerate the SAME links in the SAME order. The RL
    /// obs/action layout and the cosmetic skin mapping are both keyed by this link
    /// sequence (bone, actuated joint, parent), so if the model and the stand-in ever
    /// disagreed on it, swapping between them (asset present vs absent) would silently
    /// re-layout the policy or mis-route the skin. They share one builder today, but
    /// nothing pinned that — this is the "one source of truth" structural guard.
    /// Skips cleanly when the model isn't present (the fallback half is already covered
    /// by the model-free tests above).
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
            assert_eq!(f.bone, h.bone, "link {i}: bone fitted={} fallback={}", f.bone, h.bone);
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

    /// Golden snapshot of the fitted collider geometry, keyed by deform bone:
    /// `(half_height, radius, center.x, center.y, center.z)`, captured from
    /// `sally.glb` (digest [`GOLDEN_ASSET_DIGEST`]). Pins the output of the fit
    /// pipeline so a change to `fit_capsule`/`derive_link`/`carapace_box` — or to the
    /// asset — can't silently move the colliders. These colliders feed rapier and any
    /// shift is a new MDP (and a determinism break against an unchanged peer binary),
    /// so a silent move is exactly what must never happen.
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
        use crate::bot::meshfit::crab_asset_digest;
        let Some(path) = model_path() else {
            eprintln!("fitted_geometry_matches_golden: no model — skipping");
            return;
        };
        // A changed asset silently changes every collider (and the GCR asset digest),
        // so a digest mismatch is a HARD failure, not a skip: re-capture the golden
        // deliberately (it pins a new MDP + a retrain), don't let geometry drift slip
        // through unbaselined.
        let digest = crab_asset_digest();
        assert_eq!(
            digest, GOLDEN_ASSET_DIGEST,
            "crab asset changed (digest {digest:#018x} != golden {GOLDEN_ASSET_DIGEST:#018x}) — \
             the fitted colliders moved; re-capture FITTED_GOLDEN and expect a retrain"
        );

        let recipe = build_recipe(&LoadedModel::load(&path).expect("load model")).expect("recipe");
        let golden: std::collections::HashMap<&str, [f32; 5]> =
            FITTED_GOLDEN.iter().map(|(b, g)| (*b, *g)).collect();
        // A duplicate bone key would silently shrink the map, letting a recipe link
        // match a surviving row while the len check below still sees the full table.
        assert_eq!(golden.len(), FITTED_GOLDEN.len(), "duplicate bone in FITTED_GOLDEN");
        assert_eq!(
            recipe.links.len(),
            FITTED_GOLDEN.len(),
            "fitted link count {} != golden {}",
            recipe.links.len(),
            FITTED_GOLDEN.len()
        );
        // 0.1 mm: looser than any benign float reassociation, far tighter than a real
        // fit-algorithm change (which moves a capsule by millimetres or more).
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
