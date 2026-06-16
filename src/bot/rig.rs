//! Derives the crab's physics body straight from the bind-pose skeleton of the
//! glTF model: one physics link per deform bone, with joint anchors, free axes,
//! and capsule colliders read off the rig. This is the single source of the
//! body's geometry — replacing the old hand-coded segment lengths — so the visual
//! mesh and the physics share one coordinate space.
//!
//! Only the locomotion-relevant joints (a leg's coxa/merus/carpus, a claw's
//! shoulder/wrist/pincer) are policy-actuated; those carry a [`CrabJointId`]. The
//! rest of the rig (the proximal leg stubs, claw mid-segments, eye-stalks, palpi)
//! spawn as locked links, so the body articulates like a real Sally without
//! ballooning the action space. Promote a locked joint by giving its segment an
//! actuation mapping below and a [`CrabJointId`] variant.

use bevy::prelude::*;

use super::body::{CrabJointId, Side};
use super::meshfit::LoadedModel;

// Capsule radii by limb family (model units). Rough placeholders; the fitted
// collider pass (bddap/rl#16) refines them against the vertex cloud.
const LEG_RADIUS: f32 = 0.03;
const CLAW_RADIUS: f32 = 0.04;
const EYE_RADIUS: f32 = 0.03;
const PALP_RADIUS: f32 = 0.02;

const LEG_DENSITY: f32 = 8.0;
const CLAW_DENSITY: f32 = 1.0;
const EYE_DENSITY: f32 = 0.5;
const PALP_DENSITY: f32 = 0.5;
const CARAPACE_DENSITY: f32 = 5.0;

/// A [`RigLink::parent`] of `CARAPACE` means the link hangs off the carapace root
/// rather than another link.
pub const CARAPACE: usize = usize::MAX;

/// One physics link to spawn, fully derived from the bind pose. Links are emitted
/// parent-before-child, so `parent` indexes an earlier entry (or [`CARAPACE`]).
pub struct RigLink {
    /// The deform bone this link stands in for (skin mapping + debugging).
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
    /// invisible to the policy).
    pub actuated: Option<CrabJointId>,
}

/// The full body recipe: a carapace root box plus the derived link chain.
pub struct RigRecipe {
    pub carapace_half: Vec3,
    pub carapace_density: f32,
    pub links: Vec<RigLink>,
}

fn side_tag(side: Side) -> &'static str {
    match side {
        Side::Left => "L",
        Side::Right => "R",
    }
}

/// Leg chain proximal→distal. `005` is the root-parented foot-tip helper (not a
/// serial joint), used only as the distal target for the last segment's collider.
const LEG_BONES: [&str; 6] = ["000", "000b", "001", "002", "003", "004"];
/// Claw chain proximal→distal: arm (`000a`..`006b`) then the movable pincer finger.
const CLAW_BONES: [&str; 9] = [
    "000a", "000", "001", "002", "003", "004", "005", "006b", "006",
];

/// Which joint a leg-chain segment actuates (`None` = locked). Coxa swings the leg
/// off the body; the merus and carpus are the two load-bearing bends. The three
/// proximal stubs barely move in a real crab, so they ride locked.
fn leg_actuation(side: Side, leg: u8, seg: usize) -> Option<CrabJointId> {
    match seg {
        0 => Some(CrabJointId::LegCoxa(side, leg)),
        4 => Some(CrabJointId::LegMerus(side, leg)),
        5 => Some(CrabJointId::LegCarpus(side, leg)),
        _ => None,
    }
}

/// Which joint a claw-chain segment actuates: shoulder at the body, wrist at the
/// hand, and the movable pincer finger. The arm mid-segments ride locked.
fn claw_actuation(side: Side, seg: usize) -> Option<CrabJointId> {
    match seg {
        0 => Some(CrabJointId::ClawShoulder(side)),
        7 => Some(CrabJointId::ClawWrist(side)),
        8 => Some(CrabJointId::ClawPincer(side)),
        _ => None,
    }
}

/// Build the whole-body recipe from the model's bind pose, or `None` if the model
/// lacks the expected bones.
pub fn build_recipe(model: &LoadedModel) -> Option<RigRecipe> {
    // Body centre = the centroid of the eight leg roots (bone `000`), the hub the
    // limbs hang off: symmetric in x, mid-height in y. Carapace-relative anchors
    // are measured from here.
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
    if n == 0 {
        return None;
    }
    let carapace_center = sum / n as f32;

    let mut links: Vec<RigLink> = Vec::new();

    // -- Legs: 4 per side, the full 6-bone chain (3 actuated, 3 locked) ----------
    for side in [Side::Left, Side::Right] {
        let s = side_tag(side);
        for leg in 0u8..4 {
            let mut parent_idx = CARAPACE;
            let mut parent_name: Option<String> = None;
            for (seg, suffix) in LEG_BONES.iter().enumerate() {
                let bone = format!("Def_leg_0{}.{}.{}", leg + 1, suffix, s);
                // The last segment reaches to the foot tip (`005`) for its collider.
                let next = match LEG_BONES.get(seg + 1) {
                    Some(nb) => format!("Def_leg_0{}.{}.{}", leg + 1, nb, s),
                    None => format!("Def_leg_0{}.005.{}", leg + 1, s),
                };
                let Some(link) = derive_link(
                    model,
                    &bone,
                    parent_name.as_deref(),
                    carapace_center,
                    Some(&next),
                    leg_actuation(side, leg, seg),
                    LEG_RADIUS,
                    LEG_DENSITY,
                    parent_idx,
                ) else {
                    continue;
                };
                parent_idx = links.len();
                parent_name = Some(bone);
                links.push(link);
            }
        }
    }

    // -- Claws: the 9-bone arm+pincer chain (shoulder/wrist/pincer actuated) ------
    for side in [Side::Left, Side::Right] {
        let s = side_tag(side);
        let mut parent_idx = CARAPACE;
        let mut parent_name: Option<String> = None;
        for (seg, suffix) in CLAW_BONES.iter().enumerate() {
            let bone = format!("Def_pincer.{}.{}", suffix, s);
            let next = CLAW_BONES
                .get(seg + 1)
                .map(|nb| format!("Def_pincer.{}.{}", nb, s));
            let Some(link) = derive_link(
                model,
                &bone,
                parent_name.as_deref(),
                carapace_center,
                next.as_deref(),
                claw_actuation(side, seg),
                CLAW_RADIUS,
                CLAW_DENSITY,
                parent_idx,
            ) else {
                continue;
            };
            parent_idx = links.len();
            parent_name = Some(bone);
            links.push(link);
        }
    }

    // -- Eye-stalks (locked): base (carapace-parented) + tip. The rig parents both
    // to the armature, but physically the eye rides on the stalk, so the tip is
    // re-parented to the base here.
    for side in [Side::Left, Side::Right] {
        let s = side_tag(side);
        let base = format!("Def_antennae.{}", s);
        let tip = format!("Def_antennae_top.{}", s);
        let Some(base_link) = derive_link(
            model,
            &base,
            None,
            carapace_center,
            Some(&tip),
            None,
            EYE_RADIUS,
            EYE_DENSITY,
            CARAPACE,
        ) else {
            continue;
        };
        let base_idx = links.len();
        links.push(base_link);
        if let Some(tip_link) = derive_link(
            model,
            &tip,
            Some(&base),
            carapace_center,
            None,
            None,
            EYE_RADIUS,
            EYE_DENSITY,
            base_idx,
        ) {
            links.push(tip_link);
        }
    }

    // -- Palpi (locked): one mouthpart flap per side ----------------------------
    for side in [Side::Left, Side::Right] {
        let bone = format!("Def_palpi.001.{}", side_tag(side));
        if let Some(link) = derive_link(
            model,
            &bone,
            None,
            carapace_center,
            None,
            None,
            PALP_RADIUS,
            PALP_DENSITY,
            CARAPACE,
        ) {
            links.push(link);
        }
    }

    let carapace_half = carapace_extent(model, carapace_center);
    Some(RigRecipe {
        carapace_half,
        carapace_density: CARAPACE_DENSITY,
        links,
    })
}

/// Derive one link's joint + collider geometry from the bind pose. `parent = None`
/// hangs the link off the carapace root (identity basis at `carapace_center`).
#[allow(clippy::too_many_arguments)]
fn derive_link(
    model: &LoadedModel,
    child: &str,
    parent: Option<&str>,
    carapace_center: Vec3,
    next: Option<&str>,
    actuated: Option<CrabJointId>,
    radius: f32,
    density: f32,
    parent_idx: usize,
) -> Option<RigLink> {
    let c_origin = model.bone_origin(child)?;
    let p_origin = match parent {
        Some(p) => model.bone_origin(p)?,
        None => carapace_center,
    };
    // Links spawn axis-aligned, so the parent frame is world: anchors and the
    // collider are all plain world deltas between bind-pose bone origins.
    let anchor1 = c_origin - p_origin;

    // Segment direction: toward the next bone, else continue the incoming direction
    // as a short stub (a leaf bone has no outgoing segment).
    let in_dir = (c_origin - p_origin).normalize_or_zero();
    let seg_world = match next.and_then(|nb| model.bone_origin(nb)) {
        Some(g) => g - c_origin,
        None => in_dir * (radius * 3.0),
    };
    let seg_len = seg_world.length().max(1e-4);
    let out_dir = seg_world / seg_len;

    // Capsule from the pivot (link origin) down the segment, in the link's rest
    // frame (= world): centre halfway, oriented Y→segment.
    let half_height = (seg_len * 0.5 - radius).max(0.01);
    let center = seg_world * 0.5;
    let col_rot = arc_to(Vec3::Y, out_dir);

    // Free axis: the natural bend axis is `in × out`; when the bind chain is
    // near-straight that cross is tiny, so fall back to a horizontal axis
    // perpendicular to the limb. The coxa swings the leg fore/aft, so force a
    // vertical axis there regardless of the (degenerate) body→coxa direction.
    let axis_local = if matches!(actuated, Some(CrabJointId::LegCoxa(..))) {
        Vec3::Y
    } else {
        let cross = in_dir.cross(out_dir);
        let axis = if cross.length() > 0.2 {
            cross.normalize()
        } else {
            out_dir.cross(Vec3::Y).normalize_or_zero()
        };
        // Never hand Rapier a zero rotation axis: a vertical-and-straight bone
        // collapses both `in×out` and `out×Y` to ~0, and a zero-axis revolute
        // degenerates the joint frame into NaNs that poison the multibody and
        // recur on every respawn. Fall back to a fixed horizontal perpendicular.
        if axis.length() > 0.5 { axis } else { Vec3::X }
    };

    Some(RigLink {
        bone: child.to_string(),
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

/// Rotation taking `from` onto `to`, guarding the degenerate (near-zero `to`)
/// case `Quat::from_rotation_arc` would choke on.
fn arc_to(from: Vec3, to: Vec3) -> Quat {
    if to.length_squared() < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_arc(from, to.normalize())
    }
}

/// Half-extents of the carapace box: the bounding half-size (about `center`) of
/// the carapace-rigid bones (shell/thorax/rostrum/abdomen), so the root collider
/// covers the trunk without engulfing the limbs.
fn carapace_extent(model: &LoadedModel, center: Vec3) -> Vec3 {
    const RIGID: [&str; 14] = [
        "Def_shell.000.L",
        "Def_shell.002.L",
        "Def_shell.003.L",
        "Def_shell.006.L",
        "Def_shell.000.R",
        "Def_shell.002.R",
        "Def_shell.003.R",
        "Def_shell.006.R",
        "Def_thorax_back",
        "Def_thorax_front",
        "Def_Rostrum.L",
        "Def_Rostrum.R",
        "Ctrl_abdomen_end",
        "Ctrl_Def_neck",
    ];
    let mut half = Vec3::splat(0.08); // floor so a sparse model still gets a box
    for name in RIGID {
        if let Some(o) = model.bone_origin(name) {
            half = half.max((o - center).abs());
        }
    }
    // Trim height: the shell bones sit above the body midline, which would make a
    // tall box; keep it flattish like a real carapace.
    half.y = half.y.min(0.14);
    half
}
