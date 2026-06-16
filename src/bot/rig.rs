//! Derives the crab's physics body from the bind-pose skeleton of the glTF model:
//! one rigid link per actuated joint, each with a capsule collider fitted to that
//! joint's vertex cloud, plus a carapace box for the trunk. This is the single
//! source of the body's geometry, so the physics, the visual skin, and the
//! collider fit all share one coordinate space.
//!
//! [`joint_specs`] is the canonical decomposition of the rig into physics parts:
//! every consumer — the body spawn here, the skin ([`super::skin`]), and the
//! per-part vertex bucketing ([`super::meshfit`]) — derives its bone→part view
//! from it via [`part_for_bone`]. They can't drift, because there is only one
//! mapping; mismatched hand-kept copies of it are what forked the rendered legs.
//!
//! A leg's six rig bones collapse to three physics links: the proximal cluster
//! (`000`/`000b`/`001`/`002`) is one rigid coxa that swings at the body, then the
//! merus and carpus bend. Only those three joints articulate in a real Sally, so
//! the intermediate bones ride their link rather than spawning as locked stubs —
//! that bookkeeping bought nothing and left the skin with undriven bones.

use std::collections::HashMap;
use std::sync::OnceLock;

use bevy::prelude::*;

use super::body::{CrabJointId, Side};
use super::meshfit::{LoadedModel, PartId};

const LEG_DENSITY: f32 = 8.0;
const CLAW_DENSITY: f32 = 1.0;
const EYE_DENSITY: f32 = 0.5;
const CARAPACE_DENSITY: f32 = 5.0;

/// Eye-stalks carry no policy joint and aren't load-bearing, so there's no cloud
/// worth fitting — a fixed thin radius is honest for a cosmetic stalk.
const EYE_RADIUS: f32 = 0.03;
/// Fallback radius when a joint's vertex cloud is too sparse to fit.
const FALLBACK_RADIUS: f32 = 0.03;

/// A [`RigLink::parent`] of `CARAPACE` means the link hangs off the carapace root
/// rather than another link.
pub const CARAPACE: usize = usize::MAX;

/// One physics link to spawn, fully derived from the bind pose. Links are emitted
/// parent-before-child, so `parent` indexes an earlier entry (or [`CARAPACE`]).
pub struct RigLink {
    /// The deform bone at this link's joint pivot (skin mapping + debugging).
    pub bone: String,
    pub parent: usize,
    /// Joint pivot relative to the parent link's origin. Links spawn axis-aligned
    /// (identity orientation) at rest, so the parent frame is world and this is a
    /// plain world delta between bind-pose bone origins. (Reproducing each bone's
    /// bind *orientation* — needed to skin the visual mesh — is phase 2; at rest
    /// the capsule geometry below already points down the segment.)
    pub anchor1: Vec3,
    /// Free rotation axis in world (= the parent frame at rest; only meaningful
    /// when `actuated`).
    pub axis_local: Vec3,
    /// Capsule cylinder half-length (caps excluded) and radius.
    pub half_height: f32,
    pub radius: f32,
    /// Collider centre + orientation in the CHILD link frame (pivot at its origin).
    pub center: Vec3,
    pub col_rot: Quat,
    pub density: f32,
    /// `Some` → policy-actuated: carries a `CrabJoint`, a revolute joint, and an
    /// observation/action slot. `None` → locked (fixed joint, no `CrabJoint`,
    /// invisible to the policy — the eye-stalks).
    pub actuated: Option<CrabJointId>,
}

/// The full body recipe: a carapace root box plus the derived link chain.
pub struct RigRecipe {
    pub carapace_half: Vec3,
    /// Box centre relative to the body root (the leg-hub centroid the links anchor
    /// to). The trunk's bounding box isn't centred on the leg hub, so the carapace
    /// collider is offset to sit on the shell instead of engulfing the limbs.
    pub carapace_offset: Vec3,
    pub carapace_density: f32,
    pub links: Vec<RigLink>,
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

/// The canonical rig decomposition: one chain of actuated joints per limb. Bones
/// are bracketed by chain position — a joint owns every deform bone from its pivot
/// up to the next joint's pivot — so flesh, skin, and collider all agree on which
/// link a bone belongs to. Bracket choices are verified against the collider
/// screenshots, not derived; adjust a `members`/`tip` if a capsule misses its mesh.
fn joint_specs() -> Vec<Vec<JointSpec>> {
    let mut chains = Vec::new();
    for side in [Side::Left, Side::Right] {
        let s = side_tag(side);
        // Legs: front→back, same 0..3 order as the policy. The long coxa swings at
        // the body; the merus and carpus are the two load-bearing distal bends.
        for leg in 0u8..4 {
            let b = |seg: &str| format!("Def_leg_0{}.{}.{}", leg + 1, seg, s);
            chains.push(vec![
                JointSpec {
                    id: CrabJointId::LegCoxa(side, leg),
                    pivot: b("000"),
                    tip: Some(b("003")),
                    members: vec![b("000"), b("000b"), b("001"), b("002")],
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
        // Claw: the long arm (shoulder) swings at the body, the wrist bends the
        // hand, and the pincer is the movable finger (`006`) against the fixed
        // pollex (`006b`).
        let p = |seg: &str| format!("Def_pincer.{}.{}", seg, s);
        chains.push(vec![
            JointSpec {
                id: CrabJointId::ClawShoulder(side),
                pivot: p("000a"),
                tip: Some(p("006b")),
                members: vec![
                    p("000a"),
                    p("000"),
                    p("001"),
                    p("002"),
                    p("003"),
                    p("004"),
                    p("005"),
                ],
                density: CLAW_DENSITY,
            },
            JointSpec {
                id: CrabJointId::ClawWrist(side),
                pivot: p("006b"),
                tip: Some(p("006")),
                members: vec![p("006b")],
                density: CLAW_DENSITY,
            },
            JointSpec {
                id: CrabJointId::ClawPincer(side),
                pivot: p("006"),
                tip: None,
                members: vec![p("006"), format!("Ctrl_pincer_tail.{s}")],
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

/// Build the whole-body recipe from the model's bind pose, or `None` if the model
/// lacks the expected bones.
pub fn build_recipe(model: &LoadedModel) -> Option<RigRecipe> {
    let carapace_center = leg_hub_centroid(model)?;
    let clouds = model.vertices_by_part();

    let mut links: Vec<RigLink> = Vec::new();
    for chain in joint_specs() {
        let mut parent_idx = CARAPACE;
        let mut parent_pivot = carapace_center;
        for spec in &chain {
            let Some(pivot) = model.bone_origin(&spec.pivot) else {
                break; // a missing bone truncates this limb, not the whole body
            };
            let cloud = clouds
                .get(&PartId::Joint(spec.id))
                .map(|(pts, _)| pts.as_slice());
            let Some(link) = derive_link(
                model,
                &spec.pivot,
                spec.tip.as_deref(),
                parent_pivot,
                parent_idx,
                cloud,
                FALLBACK_RADIUS,
                spec.density,
                Some(spec.id),
            ) else {
                break;
            };
            parent_pivot = pivot;
            parent_idx = links.len();
            links.push(link);
        }
    }

    // Eye-stalks (locked): base (carapace-parented) + tip. The eye rides the stalk,
    // so the tip is re-parented onto the base here. The tip carries the reward's
    // eye-height marker (`CrabEyeTip`, set by bone name in the spawn).
    for side in [Side::Left, Side::Right] {
        let s = side_tag(side);
        let base = format!("Def_antennae.{s}");
        let tip = format!("Def_antennae_top.{s}");
        let Some(base_link) = derive_link(
            model,
            &base,
            Some(&tip),
            carapace_center,
            CARAPACE,
            None,
            EYE_RADIUS,
            EYE_DENSITY,
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
            base_origin,
            base_idx,
            None,
            EYE_RADIUS,
            EYE_DENSITY,
            None,
        ) {
            links.push(tip_link);
        }
    }

    let (carapace_half, carapace_offset) = carapace_box(model, carapace_center);
    Some(RigRecipe {
        carapace_half,
        carapace_offset,
        carapace_density: CARAPACE_DENSITY,
        links,
    })
}

/// Body centre = the centroid of the eight leg roots (bone `000`), the hub the
/// limbs hang off: symmetric in x, mid-height in y. Carapace-relative anchors are
/// measured from here, and the carapace box is offset relative to it.
fn leg_hub_centroid(model: &LoadedModel) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut n = 0u32;
    for side in [Side::Left, Side::Right] {
        for leg in 0u8..4 {
            if let Some(o) =
                model.bone_origin(&format!("Def_leg_0{}.000.{}", leg + 1, side_tag(side)))
            {
                sum += o;
                n += 1;
            }
        }
    }
    (n > 0).then(|| sum / n as f32)
}

/// Derive one link's joint + collider geometry from the bind pose. `tip = None`
/// makes a short stub along the incoming direction (a leaf bone). `cloud = Some`
/// fits the capsule radius to that vertex cloud; `None` uses `fixed_radius`.
#[allow(clippy::too_many_arguments)]
fn derive_link(
    model: &LoadedModel,
    pivot_name: &str,
    tip_name: Option<&str>,
    parent_pivot: Vec3,
    parent_idx: usize,
    cloud: Option<&[Vec3]>,
    fixed_radius: f32,
    density: f32,
    actuated: Option<CrabJointId>,
) -> Option<RigLink> {
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
        Some(pts) if pts.len() >= 8 => super::meshfit::fit_capsule(pts),
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
pub(crate) const TRUNK_BONES: [&str; 10] = [
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
/// shell vertices — not bone origins — covers the trunk's flesh without the old
/// hand-tuned height clamp.
fn carapace_box(model: &LoadedModel, center: Vec3) -> (Vec3, Vec3) {
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

/// A collider reconstructed in bind-pose world (the rest stance), paired with the
/// part whose vertex cloud it should hug. The verifier ([`crate::verify_colliders`])
/// scores cloud-vs-collider, and the model's clouds live in bind-pose world, so the
/// collider must too. This mirrors the world accumulation in [`super::body`]'s
/// `spawn_crab` minus the constant spawn translation (it cancels — `anchor1` is a
/// parent-relative delta, and the clouds are already in this frame).
pub(crate) struct RestCollider {
    pub part: PartId,
    pub shape: RestShape,
    /// The link's bind-pose-world origin — where its revolute joint actually
    /// pivots (a leaf-most link's pivot is its own bone origin). This is the
    /// physical pivot the body spawns at, recovered from the same world walk that
    /// builds the shape, so it can't drift from `joint_specs`'s pivot bone names.
    /// For the carapace it's the leg-hub root the box is offset from.
    pub pivot: Vec3,
}

pub(crate) enum RestShape {
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    Cuboid { center: Vec3, half: Vec3 },
}

/// Reconstruct every scoreable collider of `recipe` in bind-pose world. Locked
/// eye-stalk links are skipped (no fitted cloud to score). The carapace box is
/// world-axis-aligned at the hub + offset.
pub(crate) fn rest_colliders(model: &LoadedModel, recipe: &RigRecipe) -> Vec<RestCollider> {
    let Some(o_root) = leg_hub_centroid(model) else {
        return Vec::new();
    };
    let mut world_origin: Vec<Vec3> = Vec::with_capacity(recipe.links.len());
    let mut out: Vec<RestCollider> = Vec::new();
    for link in &recipe.links {
        let base = if link.parent == CARAPACE {
            o_root
        } else {
            world_origin[link.parent]
        };
        let origin = base + link.anchor1;
        world_origin.push(origin);
        // Only actuated links carry a PartId and a fitted cloud; eye-stalks (locked,
        // fixed radius, cosmetic) have nothing to score against.
        if let Some(id) = link.actuated {
            let axis = link.col_rot * Vec3::Y * link.half_height;
            let c = origin + link.center;
            out.push(RestCollider {
                part: PartId::Joint(id),
                shape: RestShape::Capsule {
                    a: c - axis,
                    b: c + axis,
                    radius: link.radius,
                },
                pivot: origin,
            });
        }
    }
    out.push(RestCollider {
        part: PartId::Carapace,
        shape: RestShape::Cuboid {
            center: o_root + recipe.carapace_offset,
            half: recipe.carapace_half,
        },
        pivot: o_root,
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
