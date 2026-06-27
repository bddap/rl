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

/// Bind-pose geometry source, so the real [`LoadedModel`] and the procedural
/// [`FallbackModel`] share ONE recipe-builder ([`build_recipe`]) — no second spawn
/// path to drift.
pub trait BindSource {
    /// Bind-pose world origin of a bone by name (the joint pivot), if present.
    fn bone_origin(&self, name: &str) -> Option<Vec3>;
    /// Per-part flesh: world-space vertex cloud for each physics part. Drives the
    /// fitted capsule radius.
    fn vertices_by_part(&self) -> HashMap<PartId, Vec<Vec3>>;
    /// World-space vertices whose dominant bone is one of `names` — the trunk flesh
    /// the carapace box is sized from.
    fn vertices_for_bones(&self, names: &[&str]) -> Vec<Vec3>;
    /// Radius to use for a part whose cloud is too sparse to fit a capsule to. The
    /// real model has dense flesh so it returns `None`; the procedural body has no
    /// flesh and supplies its intended per-part thickness here so its limbs aren't
    /// pencil-thin.
    fn radius_hint(&self, _part: PartId) -> Option<f32> {
        None
    }
}

impl BindSource for LoadedModel {
    fn bone_origin(&self, name: &str) -> Option<Vec3> {
        LoadedModel::bone_origin(self, name)
    }
    fn vertices_by_part(&self) -> HashMap<PartId, Vec<Vec3>> {
        LoadedModel::vertices_by_part(self)
    }
    fn vertices_for_bones(&self, names: &[&str]) -> Vec<Vec3> {
        LoadedModel::vertices_for_bones(self, names)
    }
}

const LEG_DENSITY: f32 = 8.0;
const CLAW_DENSITY: f32 = 1.0;
const EYE_DENSITY: f32 = 0.5;
const CARAPACE_DENSITY: f32 = 5.0;

/// Eye-stalks carry no policy joint and aren't load-bearing, so there's no cloud
/// worth fitting — a fixed thin radius is honest for a cosmetic stalk.
const EYE_RADIUS: f32 = 0.03;
/// Fallback radius when a joint's vertex cloud is too sparse to fit.
const FALLBACK_RADIUS: f32 = 0.03;

/// One physics link to spawn, fully derived from the bind pose. Links are emitted
/// parent-before-child, so `parent` indexes an earlier entry.
pub struct RigLink {
    /// The deform bone at this link's joint pivot (skin mapping + debugging).
    pub bone: String,
    /// Index of the parent link in [`RigRecipe::links`], or `None` when the link
    /// hangs off the carapace root — so "root" can't masquerade as a real index.
    pub parent: Option<usize>,
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
    /// Bind-pose world position of the leg-hub centroid — the point the link chain
    /// anchors off (`anchor1` deltas telescope back to it). The body spawns its
    /// carapace root here (plus the spawn point) so every link lands at its true
    /// glTF bind-world origin, the same frame the cosmetic skin renders its bones
    /// in. Without it the spawn pinned the hub at an arbitrary height and lost the
    /// hub's lateral/forward bind offset, sliding the whole physics body ~0.1 off
    /// the skin (pivots fell outside the rendered mesh).
    pub hub_bind_world: Vec3,
    pub carapace_half: Vec3,
    /// Box centre relative to the body root (the leg-hub centroid the links anchor
    /// to). The trunk's bounding box isn't centred on the leg hub, so the carapace
    /// collider is offset to sit on the shell instead of engulfing the limbs.
    pub carapace_offset: Vec3,
    pub carapace_density: f32,
    pub links: Vec<RigLink>,
}

impl RigRecipe {
    /// Every geometric field across the recipe is finite. The spawn feeds these
    /// straight into Rapier; a single NaN/inf reaches the solver as an
    /// unrecoverable crash on the very first step, so this is the gate that keeps a
    /// degenerate asset from ever getting that far.
    fn is_finite(&self) -> bool {
        self.hub_bind_world.is_finite()
            && self.carapace_half.is_finite()
            && self.carapace_offset.is_finite()
            && self.carapace_density.is_finite()
            && self.links.iter().all(RigLink::is_finite)
    }
}

impl RigLink {
    fn is_finite(&self) -> bool {
        self.anchor1.is_finite()
            && self.axis_local.is_finite()
            && self.half_height.is_finite()
            && self.radius.is_finite()
            && self.center.is_finite()
            && self.col_rot.is_finite()
            && self.density.is_finite()
    }
}

/// Every link's unrotated origin in `root`'s frame, by telescoping each `anchor1`
/// (a parent-relative delta) down the parent-before-child chain. Links are emitted
/// parent-first, so `world[parent]` is always filled before a child reads it. The
/// single source for this walk: the body spawn ([`super::body`]'s `spawn_crab`)
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
fn leg_bone(leg: u8, seg: &str, side: Side) -> String {
    format!("Def_leg_0{}.{}.{}", leg + 1, seg, side_tag(side))
}
fn pincer_bone(seg: &str, side: Side) -> String {
    format!("Def_pincer.{}.{}", seg, side_tag(side))
}
fn antennae_bone(side: Side) -> String {
    format!("Def_antennae.{}", side_tag(side))
}
fn antennae_top_bone(side: Side) -> String {
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
        // Legs: front→back, same 0..3 order as the policy. The long coxa swings at
        // the body; the merus and carpus are the two load-bearing distal bends.
        for leg in 0u8..4 {
            let b = |seg: &str| leg_bone(leg, seg, side);
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
/// The skin weight strip ([`super::skin`]) keeps both parts' lanes at an adjacent
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

    // Eye-stalks (locked): base (carapace-parented) + tip. The eye rides the stalk,
    // so the tip is re-parented onto the base here. The tip carries the reward's
    // eye-height marker (`CrabEyeTip`, set by bone name in the spawn).
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
    // A non-finite origin on any non-hub bone slips past the hub check above but
    // still poisons that link's anchor. Make a non-finite recipe unrepresentable
    // downstream: reject the whole thing once here so spawn only ever sees clean
    // geometry (the degenerate-but-finite axis cases are already guarded in
    // `derive_link`/`bend_axis`).
    recipe.is_finite().then_some(recipe)
}

// ===========================================================================
// Procedural fallback body (no purchased model)
// ===========================================================================

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
pub struct FallbackModel {
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
                let coxa = root; // 000: leg root at the shell
                let knee = root + Vec3::new(sx * 0.22, -0.02, 0.0); // 003: merus pivot
                let ankle = knee + Vec3::new(sx * 0.16, -0.14, 0.0); // 004: carpus pivot
                let foot = ankle + Vec3::new(sx * 0.05, -0.14, 0.0); // 005: foot tip on the ground
                o.insert(leg_bone(leg, "000", side), coxa);
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

    /// The eight shell-box corners (bind-pose world at the hub), so [`carapace_box`]
    /// derives the same box it would from a cloud. `names` is ignored: the only
    /// `FallbackModel` caller is [`carapace_box`] asking for the trunk.
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
                CrabJointId::LegCoxa(..) | CrabJointId::LegMerus(..) | CrabJointId::LegCarpus(..),
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

/// Body centre = the centroid of the eight leg roots (bone `000`), the hub the
/// limbs hang off: symmetric in x, mid-height in y. Carapace-relative anchors are
/// measured from here, and the carapace box is offset relative to it.
fn leg_hub_centroid(model: &impl BindSource) -> Option<Vec3> {
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

/// A collider reconstructed in bind-pose world (the rest stance), paired with the
/// part whose vertex cloud it should hug. The verifier (the `rl --verify-colliders`
/// dev command) scores cloud-vs-collider, and the model's clouds live in bind-pose
/// world, so the collider must too. This mirrors the world accumulation in [`super::body`]'s
/// `spawn_crab` minus the constant spawn translation (it cancels — `anchor1` is a
/// parent-relative delta, and the clouds are already in this frame).
pub struct RestCollider {
    pub part: PartId,
    pub shape: RestShape,
    /// The link's bind-pose-world origin — where its revolute joint actually
    /// pivots (a leaf-most link's pivot is its own bone origin). This is the
    /// physical pivot the body spawns at, recovered from the same world walk that
    /// builds the shape, so it can't drift from `joint_specs`'s pivot bone names.
    /// For the carapace it's the leg-hub root the box is offset from.
    pub pivot: Vec3,
}

pub enum RestShape {
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    Cuboid { center: Vec3, half: Vec3 },
}

/// Reconstruct every scoreable collider of `recipe` in bind-pose world. Locked
/// eye-stalk links are skipped (no fitted cloud to score). The carapace box is
/// world-axis-aligned at the hub + offset.
pub fn rest_colliders(model: &impl BindSource, recipe: &RigRecipe) -> Vec<RestCollider> {
    let Some(o_root) = leg_hub_centroid(model) else {
        return Vec::new();
    };
    let world_origin = link_world_origins(&recipe.links, o_root);
    let mut out: Vec<RestCollider> = Vec::new();
    for (link, &origin) in recipe.links.iter().zip(&world_origin) {
        // Only actuated links carry a PartId and a fitted cloud; eye-stalks (locked,
        // fixed radius, cosmetic) have nothing to score against.
        if let Some(id) = link.actuated {
            out.push(RestCollider {
                part: PartId::Joint(id),
                shape: link_capsule(link, origin),
                pivot: origin,
            });
        }
    }
    out.push(RestCollider {
        part: PartId::Carapace,
        shape: carapace_cuboid(recipe, o_root),
        pivot: o_root,
    });
    out
}

/// The crab's cosmetic collider silhouette — the carapace box and a capsule for EVERY
/// link, the carapace kept as its OWN field (not the tail of a list) so a consumer
/// can't mistake it for a limb when it derives the crab's facing. See
/// [`recipe_silhouette`].
pub struct CrabSilhouette {
    /// One capsule per link, in `recipe.hub_bind_world`'s frame — legs, locked
    /// eye-stalks, and claws alike (the cosmetic view draws them all).
    pub limbs: Vec<RestShape>,
    /// The carapace box, same frame.
    pub carapace: RestShape,
}

impl CrabSilhouette {
    /// Every shape (limbs + carapace), for whole-body extent math.
    pub fn shapes(&self) -> impl Iterator<Item = &RestShape> {
        self.limbs.iter().chain(std::iter::once(&self.carapace))
    }

    /// The rig's natural standing height: the vertical (Y) extent of its rest-pose
    /// collider silhouette, in metres. The ONE size source both giant-crab renders scale
    /// against — the integer silhouette (`spawn_crab_silhouette`) fits this extent to the giant
    /// height, and the armed NN rig ([`crate::bot::skin::CrabSkinRepose`]) scales the live skin by
    /// the same target/height ratio, so the two crabs are the same size by construction (no drift).
    /// Both renders orient the rig claws-forward by a pure YAW first, which leaves the Y
    /// extent unchanged, so it's correct to measure it in the rig's own frame here. `0.0` for
    /// a degenerate (empty/non-finite) recipe — callers guard that case.
    pub fn natural_height(&self) -> f32 {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for s in self.shapes() {
            match *s {
                RestShape::Capsule { a, b, radius } => {
                    lo = lo.min(a.y - radius).min(b.y - radius);
                    hi = hi.max(a.y + radius).max(b.y + radius);
                }
                RestShape::Cuboid { center, half } => {
                    lo = lo.min(center.y - half.y);
                    hi = hi.max(center.y + half.y);
                }
            }
        }
        if lo.is_finite() && hi.is_finite() {
            hi - lo
        } else {
            0.0
        }
    }
}

/// Reconstruct `recipe`'s collider silhouette for rendering. Unlike [`rest_colliders`]
/// (scoring: actuated links only, anchored at the model's leg hub) this is the COSMETIC
/// view — it draws every link including the locked eye-stalks and needs no model, the
/// recipe alone carries the geometry. Anchored at `recipe.hub_bind_world` so it shares
/// [`rest_colliders`]'s frame; a render that re-poses and re-centers the crab is free of
/// the absolute offset either way. This is the ONE shape source the giant-crab render
/// draws from (fed [`super::body::render_recipe`]), so the cosmetic crab can't drift
/// from the body it depicts.
pub fn recipe_silhouette(recipe: &RigRecipe) -> CrabSilhouette {
    let hub = recipe.hub_bind_world;
    let world_origin = link_world_origins(&recipe.links, hub);
    let limbs = recipe
        .links
        .iter()
        .zip(&world_origin)
        .map(|(link, &origin)| link_capsule(link, origin))
        .collect();
    CrabSilhouette {
        limbs,
        carapace: carapace_cuboid(recipe, hub),
    }
}

/// One link's capsule from its telescoped `origin`. Shared by [`rest_colliders`]
/// (scoring) and [`recipe_silhouette`] (rendering) so the capsule geometry has a
/// single definition that can't drift between the scored body and the drawn one.
fn link_capsule(link: &RigLink, origin: Vec3) -> RestShape {
    let axis = link.col_rot * Vec3::Y * link.half_height;
    let c = origin + link.center;
    RestShape::Capsule {
        a: c - axis,
        b: c + axis,
        radius: link.radius,
    }
}

/// The carapace box, world-axis-aligned at the leg `hub` + the recipe's offset.
fn carapace_cuboid(recipe: &RigRecipe, hub: Vec3) -> RestShape {
    RestShape::Cuboid {
        center: hub + recipe.carapace_offset,
        half: recipe.carapace_half,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::meshfit::{LoadedModel, model_path};

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

    /// The clamped shoulder up-swing keeps BOTH reach effectors (palm `005`, pincer tip
    /// `006`) at or below the carapace top — the regression guard for the "arm lifts up
    /// through the shell / past the eye-stalks" bug. It reads the SPAWNED shoulder link's
    /// own `axis_local` and bind-world pivot straight from the recipe — the exact axis the
    /// actuator drives and the joint limit clamps — rather than re-deriving them, so the
    /// guard can't silently test a phantom axis if the cheliped is ever re-bracketed. The
    /// palm is the higher of the two effectors, so it's the binding one. `lo` is the
    /// up-stop (−θ raises the arm; see the actuator). Pure geometry; skips with no model.
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
            let (link, &pivot) = recipe
                .links
                .iter()
                .zip(&world)
                .find(|(l, _)| l.actuated == Some(CrabJointId::ClawShoulder(side)))
                .expect("shoulder link present");
            // `axis_local` is in the parent frame, which is world at the bind rest (the
            // parent carapace spawns unrotated), so it rotates the effectors directly.
            let [lo, _hi] = CrabJointId::ClawShoulder(side).limits();
            let rot = Quat::from_axis_angle(link.axis_local, lo);
            for bone in ["005", "006"] {
                let p = model.bone_origin(&pincer_bone(bone, side)).unwrap();
                let y = (pivot + rot * (p - pivot)).y;
                assert!(
                    y <= box_top + 1e-3,
                    "{side:?} cheliped {bone} reaches y={y:.3} at the up-stop θ={lo:.3}, \
                     above the carapace top {box_top:.3} — arm clips the shell"
                );
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

        // The reward locates eye-tips and claw-tips by these bone-name patterns.
        assert_eq!(
            recipe
                .links
                .iter()
                .filter(|l| l.bone.starts_with("Def_antennae_top"))
                .count(),
            2,
            "two eye-tip links (the reward's eye-height markers)"
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

    /// The render silhouette (#108) draws EVERY link, eye-stalks included, as a capsule
    /// (one per link) plus the carapace as its own cuboid — and every shape must be
    /// finite or the giant crab would spawn NaN meshes. (Contrast [`rest_colliders`],
    /// which scores actuated links only.)
    #[test]
    fn recipe_silhouette_covers_all_links_plus_carapace() {
        let recipe = fallback_recipe();
        let sil = recipe_silhouette(&recipe);
        assert_eq!(
            sil.limbs.len(),
            recipe.links.len(),
            "one capsule per link, eye-stalks included"
        );
        for s in sil.limbs.iter().chain(std::iter::once(&sil.carapace)) {
            match *s {
                RestShape::Capsule { a, b, radius } => {
                    assert!(a.is_finite() && b.is_finite() && radius.is_finite() && radius > 0.0);
                }
                RestShape::Cuboid { center, half } => {
                    assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0);
                }
            }
        }
        assert!(
            matches!(sil.carapace, RestShape::Cuboid { .. }),
            "carapace must be a cuboid"
        );
    }

    /// `derive_link`'s sparse-cloud branch must honor [`BindSource::radius_hint`], or
    /// the stand-in's limbs would all be the pencil-thin generic [`FALLBACK_RADIUS`].
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
