//! `RL_SKIN_DIAG`: settled-pose point-in-mesh audit of the joint pivots.
//!
//! `--verify-pivots` answers a different question than the one the rendered crab
//! poses: it tests the *bind* pose (the static glTF skeleton) against the *bind*
//! skinned surface. What the eye actually sees is the *settled* pose — the body has
//! dropped and splayed under gravity for ~200 frames — wearing the *live* skin, whose
//! deform bones are driven by the physics links (see [`super::skin`]). A pivot can sit
//! inside the bind mesh yet outside the settled mesh, because the two surfaces are not
//! the same shape.
//!
//! This module reconstructs the surface Bevy actually rasterizes and tests the live
//! marker positions against it. The reconstruction is not an independent re-derivation
//! of the skin math (which could drift in frame or convention): it reads Bevy's own
//! [`SkinnedMesh`] (inverse-bind matrices + the live bone entities, in skin-joint
//! order) and composes each vertex exactly as the GPU does —
//! `world = Σ_i w_i · (boneGlobal[i].affine() · invBind[i]) · localpos` — so the
//! triangles tested are the triangles drawn. The marker positions are the same
//! `GlobalTransform`s [`super::body::draw_pivot_markers`] renders, so the numbers
//! describe the exact picture the owner sees.
//!
//! Emitted (once, at the settle frame, before the screenshot is taken):
//!  - per-pivot generalized winding number + signed distance (− inside, + outside);
//!  - per-link divergence between the physics link world position and its driven
//!    skin-bone world position (does the skin track the body, or drift off it?);
//!  - the bind-pose signed distance beside the settled one (the delta the bind-pose
//!    verifier cannot see);
//!  - an AABB size/offset check: live skinned mesh vs the physics link cloud.

use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::mesh::{MeshVertexAttribute, VertexAttributeValues};
use bevy::prelude::*;
use std::collections::HashMap;

use super::body::{CrabBodyPart, CrabCarapace, CrabJoint, CrabJointId};
use super::meshfit::PartId;

/// Fire the audit when the render-frame counter reaches this, matching the
/// screenshot's settle so the table describes the frame that gets photographed.
#[derive(Resource)]
pub struct SkinDiagAt(pub u32);

/// Register the settled-pose audit. Gated by the caller on `RL_SKIN_DIAG` so a
/// normal screenshot pays nothing; `settle` is the screenshot's settle frame so the
/// numbers and the picture agree.
pub fn register(app: &mut App, settle: u32) {
    app.insert_resource(SkinDiagAt(settle));
    app.add_systems(Update, run_at_settle);
}

/// Force every skin material translucent (`RL_SKIN_ALPHA`), so a settled screenshot
/// shows the solid pivot markers + collider cages THROUGH the skin and inside/outside
/// reads unambiguously. Re-applied each frame because the glTF materials finish
/// loading after spawn and a respawn brings fresh opaque ones.
pub fn register_translucent(app: &mut App) {
    app.add_systems(Update, make_skin_translucent);
}

/// Alpha the skin is rendered at under `RL_SKIN_ALPHA` — low enough to see the
/// markers/colliders behind it, high enough to still read the body's surface.
const SKIN_ALPHA: f32 = 0.4;

/// Set every skinned mesh's standard material to alpha-blended `SKIN_ALPHA`. Keyed off
/// the `SkinnedMesh` component so it touches only the crab skin, not the ground/props.
fn make_skin_translucent(
    skins: Query<&MeshMaterial3d<StandardMaterial>, With<SkinnedMesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for handle in skins.iter() {
        if let Some(mat) = materials.get_mut(&handle.0) {
            mat.base_color.set_alpha(SKIN_ALPHA);
            mat.alpha_mode = AlphaMode::Blend;
        }
    }
}

/// Drive the one-shot audit off a render-frame counter that mirrors the screenshot's
/// settle. Runs in `Update` (like `capture_when_settled`) and one frame *before* the
/// capture frame, so transforms are the settled pose the screenshot records.
fn run_at_settle(world: &mut World, mut frame: Local<u32>, mut done: Local<bool>) {
    if *done {
        return;
    }
    let at = world.resource::<SkinDiagAt>().0;
    *frame += 1;
    // One frame early: the capture system fires AT `settle`; we want the pose it sees,
    // so we read on the frame just before it spawns the screenshot and exits.
    if *frame < at.saturating_sub(1) {
        return;
    }
    *done = true;
    audit(world);
}

/// Assemble the live skinned triangle soup and run the point-in-mesh tests. Takes
/// `&mut World` so it can resolve the `SkinnedMesh`'s joint entities against their
/// live `GlobalTransform`s and pull the mesh asset — neither expressible as one
/// system param without fighting the borrow checker over the entity indirection.
fn audit(world: &mut World) {
    let Some((positions, triangles)) = live_skinned_soup(world) else {
        warn!("RL_SKIN_DIAG: no skinned mesh found (skin not spawned/paired?) — nothing to test");
        return;
    };
    if triangles.is_empty() {
        warn!("RL_SKIN_DIAG: skinned mesh has no triangles — nothing to test");
        return;
    }

    let (lo, hi) = aabb(&positions);
    let signed_vol = mesh_signed_volume(&positions, &triangles);
    let orient = if signed_vol < 0.0 { -1.0 } else { 1.0 };

    // A query point against the LIVE soup: winding (normalised so inside reads +1),
    // and the nearest-surface distance signed + for OUTSIDE. Same convention as
    // `--verify-pivots`, on the settled surface instead of the bind one.
    let probe = |p: Vec3| -> (f32, f32, bool) {
        let wn = winding_number(p, &positions, &triangles) * orient;
        let d = nearest_surface_distance(p, &positions, &triangles);
        let inside = wn > 0.5;
        (wn, if inside { -d } else { d }, inside)
    };

    // Bind-pose surface + bind-pose pivots, for the settled-vs-bind delta the
    // bind-only verifier can't show. Built from the model on disk; absent → skip the
    // delta column rather than fail the whole audit.
    let bind = load_bind_reference();

    // The 30 actuated link pivots = the marker positions for entities carrying a
    // CrabJoint (the carapace and the locked eye-stalks are reported separately).
    let mut joint_rows: Vec<JointRow> = Vec::new();
    {
        let mut q = world.query::<(&GlobalTransform, &CrabJoint)>();
        for (gt, joint) in q.iter(world) {
            let p = gt.translation();
            let (wn, dist, inside) = probe(p);
            let bind_dist = bind.as_ref().and_then(|b| b.pivot_signed_dist(joint.id));
            joint_rows.push(JointRow {
                id: joint.id,
                world: p,
                wn,
                dist,
                inside,
                bind_dist,
            });
        }
    }
    joint_rows.sort_by_key(|r| r.id.index());

    // Per-joint skin-vs-physics drift: each link vs its PIVOT deform bone (rest offset
    // ~0, so the gap is genuine drift). Large ⇒ the skin drifts off the body (cause
    // (a)); small ⇒ the skin tracks, so any outside-ness is the settled surface's own
    // shape at the joint, not a tracking failure.
    let divergence = link_skin_divergence(world);

    // ---- carapace + eye-stalks (the other CrabBodyPart markers), for completeness.
    let mut other_rows: Vec<(String, f32, f32, bool)> = Vec::new();
    {
        let mut q =
            world.query_filtered::<&GlobalTransform, (With<CrabBodyPart>, Without<CrabJoint>)>();
        let mut carapace = world.query_filtered::<&GlobalTransform, With<CrabCarapace>>();
        let carapace_set: Vec<Vec3> = carapace.iter(world).map(|g| g.translation()).collect();
        for gt in q.iter(world) {
            let p = gt.translation();
            let (wn, dist, inside) = probe(p);
            let label = if carapace_set.iter().any(|c| c.distance(p) < 1e-4) {
                "carapace".to_string()
            } else {
                "eye-stalk".to_string()
            };
            other_rows.push((label, wn, dist, inside));
        }
    }

    // ---- physics-body AABB (all CrabBodyPart marker points): a coarse size/offset
    // reference for the skin AABB. The marker points are link origins, so this is the
    // body's pivot envelope, not its collider hull — enough to catch a systematically
    // smaller or shifted skin.
    let body_pts = all_body_points(world);
    let (blo, bhi) = aabb(&body_pts);

    print_report(
        &positions,
        &triangles,
        lo,
        hi,
        signed_vol,
        orient,
        &joint_rows,
        &divergence,
        &other_rows,
        blo,
        bhi,
        bind.as_ref(),
    );
}

/// A measured pivot: its joint, world position, and settled-mesh verdict, plus the
/// bind-mesh signed distance for the same pivot when the model is on disk.
struct JointRow {
    id: CrabJointId,
    world: Vec3,
    wn: f32,
    dist: f32,
    inside: bool,
    /// Signed distance of this joint's BIND pivot against the BIND surface (− inside).
    /// `None` if the model couldn't be loaded for the bind reference.
    bind_dist: Option<f32>,
}

/// Build the live skinned surface as a world-space triangle soup, exactly as the GPU
/// skins it: for each mesh vertex, `Σ_i w_i · (boneGlobal[i].affine() · invBind[i]) ·
/// localpos`, using the driven bones' live `GlobalTransform`s. Concatenates every
/// skinned primitive with its indices offset. `None` if no skinned mesh exists yet.
fn live_skinned_soup(world: &mut World) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
    // The skin's mesh entities: a Mesh3d that is also a SkinnedMesh.
    let mut skinned = world.query::<(&Mesh3d, &SkinnedMesh)>();
    let skins: Vec<(Handle<Mesh>, SkinnedMesh)> = skinned
        .iter(world)
        .map(|(m, s): (&Mesh3d, &SkinnedMesh)| (m.0.clone(), s.clone()))
        .collect();
    if skins.is_empty() {
        return None;
    }

    let mut positions: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();

    for (mesh_handle, skin) in &skins {
        // Resolve each skin joint to its live world transform, in skin-joint order so
        // the index lines up with the inverse-bind array.
        let inv_binds: Vec<Mat4> = {
            let assets = world.resource::<Assets<SkinnedMeshInverseBindposes>>();
            match assets.get(&skin.inverse_bindposes) {
                Some(ib) => ib.iter().copied().collect::<Vec<Mat4>>(),
                None => continue,
            }
        };
        let mut joint_globals: Vec<Mat4> = Vec::with_capacity(skin.joints.len());
        for (i, &joint) in skin.joints.iter().enumerate() {
            // joint_matrix = boneGlobal.affine() · invBind — Bevy's exact formula.
            let g = world
                .get::<GlobalTransform>(joint)
                .map(|t| t.affine())
                .unwrap_or_default();
            let inv = inv_binds.get(i).copied().unwrap_or(Mat4::IDENTITY);
            joint_globals.push(Mat4::from(g) * inv);
        }

        let assets = world.resource::<Assets<Mesh>>();
        let Some(mesh) = assets.get(mesh_handle) else {
            continue;
        };
        let Some(raw) = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
        else {
            continue;
        };
        let (Some(joints), Some(weights)) = (
            read_u16x4(mesh, Mesh::ATTRIBUTE_JOINT_INDEX),
            read_f32x4(mesh, Mesh::ATTRIBUTE_JOINT_WEIGHT),
        ) else {
            continue;
        };

        let base = positions.len() as u32;
        for ((p, j), w) in raw.iter().zip(&joints).zip(&weights) {
            let local = Vec3::from_array(*p);
            let mut acc = Vec3::ZERO;
            let mut wsum = 0.0f32;
            for lane in 0..4 {
                let wt = w[lane];
                if wt <= 0.0 {
                    continue;
                }
                let jm = joint_globals
                    .get(j[lane] as usize)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY);
                acc += wt * jm.transform_point3(local);
                wsum += wt;
            }
            // glTF weights are normalised, but guard a degenerate (all-zero) vertex.
            positions.push(if wsum > 1e-6 { acc / wsum } else { local });
        }

        let indices: Vec<u32> = match mesh.indices() {
            Some(idx) => idx.iter().map(|i| i as u32).collect(),
            None => (0..raw.len() as u32).collect(),
        };
        for tri in indices.chunks_exact(3) {
            triangles.push([base + tri[0], base + tri[1], base + tri[2]]);
        }
    }

    Some((positions, triangles))
}

/// Per-joint drift between the physics link and the skin bone that shares its pivot.
/// Each link's PIVOT deform bone (`RigLink::bone`) starts at the link origin, so its
/// rest offset is ~0 and any settle-time gap is genuine skin-vs-physics drift — not
/// the legitimate rest offset a distal member bone carries. A *small* gap means the
/// skin surface rides with the body (so outside-ness is the surface's own shape); a
/// *large* one means the skin has drifted off the body.
struct Divergence {
    /// Per-joint pivot-bone↔link world-position gap.
    per_joint: HashMap<CrabJointId, f32>,
    median: f32,
    p95: f32,
    max: f32,
    max_label: String,
    /// Spread of the gap across joints. A near-zero std with a nonzero mean means the
    /// gap is a single rigid frame offset shared by every joint (the skin sits a fixed
    /// distance off the body) — NOT differential drift, where joints would diverge by
    /// different amounts. This is the number that tells the two apart.
    std: f32,
}

/// Measure each joint's pivot-bone↔link world gap. The recipe names the pivot bone of
/// every joint (`RigLink::bone`); its driven world position is where the skin places
/// that joint, the link's is where the physics does. The gap is the skin's bind-pose
/// pivot vs the physics spawn-pose pivot — which differ because links spawn
/// axis-aligned while the skin reproduces the model's natural bind pose. A *uniform*
/// gap (low std) is a rigid frame offset; a *scattered* one would be per-joint drift.
fn link_skin_divergence(world: &mut World) -> Divergence {
    use super::skin::BoneDrive;

    // joint → its pivot deform bone name, from the same recipe the body spawns from.
    let pivot_bone: HashMap<String, CrabJointId> = match load_bind_reference_recipe() {
        Some(links) => links
            .into_iter()
            .filter_map(|(bone, actuated)| actuated.map(|id| (bone, id)))
            .collect(),
        None => HashMap::new(),
    };

    // joint → live link world position.
    let mut link_world: HashMap<CrabJointId, Vec3> = HashMap::new();
    {
        let mut q = world.query::<(&GlobalTransform, &CrabJoint)>();
        for (gt, j) in q.iter(world) {
            link_world.insert(j.id, gt.translation());
        }
    }

    let mut per_joint: HashMap<CrabJointId, f32> = HashMap::new();
    let mut all: Vec<(String, f32)> = Vec::new();
    {
        let mut q = world.query::<(&GlobalTransform, &Name, &BoneDrive)>();
        for (gt, name, _) in q.iter(world) {
            let Some(&id) = pivot_bone.get(name.as_str()) else {
                continue;
            };
            let Some(&lp) = link_world.get(&id) else {
                continue;
            };
            let gap = gt.translation().distance(lp);
            all.push((name.as_str().to_string(), gap));
            per_joint.insert(id, gap);
        }
    }

    let mut gaps: Vec<f32> = all.iter().map(|(_, g)| *g).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (max_label, max) = all
        .iter()
        .cloned()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(("none".to_string(), 0.0));
    let mean = if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().sum::<f32>() / gaps.len() as f32
    };
    let std = if gaps.is_empty() {
        0.0
    } else {
        (gaps.iter().map(|g| (g - mean).powi(2)).sum::<f32>() / gaps.len() as f32).sqrt()
    };
    Divergence {
        per_joint,
        median: pctl(&gaps, 0.5),
        p95: pctl(&gaps, 0.95),
        max,
        max_label,
        std,
    }
}

/// The recipe's `(pivot_bone, actuated)` per link — the joint→pivot-bone mapping the
/// divergence metric needs. Reuses [`super::rig::build_recipe`] so it can't drift from
/// the body the physics spawns. `None` if the model isn't reachable.
fn load_bind_reference_recipe() -> Option<Vec<(String, Option<CrabJointId>)>> {
    let path = super::meshfit::model_path()?;
    let model = super::meshfit::LoadedModel::load(&path).ok()?;
    let recipe = super::rig::build_recipe(&model)?;
    Some(
        recipe
            .links
            .iter()
            .map(|l| (l.bone.clone(), l.actuated))
            .collect(),
    )
}

/// World positions of every `CrabBodyPart` link origin (the marker points), the
/// physics body's pivot envelope used for the AABB size/offset check.
fn all_body_points(world: &mut World) -> Vec<Vec3> {
    let mut q = world.query_filtered::<&GlobalTransform, With<CrabBodyPart>>();
    q.iter(world).map(|g| g.translation()).collect()
}

// ---------------------------------------------------------------------------
// Bind-pose reference (the settled-vs-bind delta)
// ---------------------------------------------------------------------------

/// The bind-pose surface + bind pivots, loaded from the model on disk, so the audit
/// can print the bind signed distance beside the settled one. Mirrors what
/// `--verify-pivots` measures, side-by-side with the settled result.
struct BindReference {
    positions: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    orient: f32,
    /// part → bind-pose pivot world position.
    pivots: HashMap<PartId, Vec3>,
}

impl BindReference {
    /// Signed distance of a joint's bind pivot against the bind surface (− inside).
    fn pivot_signed_dist(&self, id: CrabJointId) -> Option<f32> {
        let p = *self.pivots.get(&PartId::Joint(id))?;
        let wn = winding_number(p, &self.positions, &self.triangles) * self.orient;
        let d = nearest_surface_distance(p, &self.positions, &self.triangles);
        Some(if wn > 0.5 { -d } else { d })
    }
}

/// Load the bind-pose surface + pivots from the model, reusing the meshfit/rig bind
/// math (the exact basis `--verify-pivots` uses). `None` (and no delta column) if the
/// model isn't reachable.
fn load_bind_reference() -> Option<BindReference> {
    let path = super::meshfit::model_path()?;
    let model = super::meshfit::LoadedModel::load(&path).ok()?;
    let mesh = super::meshfit::load_bind_mesh(&path).ok()?;
    let recipe = super::rig::build_recipe(&model)?;
    let signed_vol = mesh_signed_volume(&mesh.positions, &mesh.triangles);
    let orient = if signed_vol < 0.0 { -1.0 } else { 1.0 };
    let mut pivots = HashMap::new();
    for rc in super::rig::rest_colliders(&model, &recipe) {
        pivots.insert(rc.part, rc.pivot);
    }
    Some(BindReference {
        positions: mesh.positions,
        triangles: mesh.triangles,
        orient,
        pivots,
    })
}

// ---------------------------------------------------------------------------
// Mesh attribute readers
// ---------------------------------------------------------------------------

/// Read a `[u16; 4]`-per-vertex attribute (glTF JOINTS_0). Bevy stores joint indices
/// as `Uint16x4`; nothing else is valid for `ATTRIBUTE_JOINT_INDEX`, so a different
/// format is a malformed mesh and yields `None`.
fn read_u16x4(mesh: &Mesh, attr: MeshVertexAttribute) -> Option<Vec<[u16; 4]>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Uint16x4(v) => Some(v.clone()),
        _ => None,
    }
}

/// Read a `[f32; 4]`-per-vertex attribute (glTF WEIGHTS_0).
fn read_f32x4(mesh: &Mesh, attr: MeshVertexAttribute) -> Option<Vec<[f32; 4]>> {
    match mesh.attribute(attr)? {
        VertexAttributeValues::Float32x4(v) => Some(v.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Geometry: winding number, signed volume, point-triangle distance.
// Re-implemented here rather than shared with main.rs's `--verify-pivots` helpers,
// which are private to the binary crate; the math is identical (same formulas).
// ---------------------------------------------------------------------------

fn winding_number(p: Vec3, positions: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    let mut acc = 0.0f64;
    for t in tris {
        let a = positions[t[0] as usize] - p;
        let b = positions[t[1] as usize] - p;
        let c = positions[t[2] as usize] - p;
        let (la, lb, lc) = (a.length() as f64, b.length() as f64, c.length() as f64);
        let num = a.dot(b.cross(c)) as f64;
        let den =
            la * lb * lc + (a.dot(b) as f64) * lc + (b.dot(c) as f64) * la + (c.dot(a) as f64) * lb;
        acc += 2.0 * num.atan2(den);
    }
    (acc / (4.0 * std::f64::consts::PI)) as f32
}

fn mesh_signed_volume(positions: &[Vec3], tris: &[[u32; 3]]) -> f64 {
    let mut acc = 0.0f64;
    for t in tris {
        let v0 = positions[t[0] as usize];
        let v1 = positions[t[1] as usize];
        let v2 = positions[t[2] as usize];
        acc += v0.dot(v1.cross(v2)) as f64 / 6.0;
    }
    acc
}

fn point_tri_distance(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> f32 {
    let ab = v1 - v0;
    let ac = v2 - v0;
    let ap = p - v0;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - v1;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let cp = p - v2;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (v0 + ab * v - p).length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (v0 + ac * w - p).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (v1 + (v2 - v1) * w - p).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (v0 + ab * v + ac * w - p).length()
}

fn nearest_surface_distance(p: Vec3, positions: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    let mut best = f32::INFINITY;
    for t in tris {
        let d = point_tri_distance(
            p,
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        if d < best {
            best = d;
        }
    }
    best
}

fn aabb(pts: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &p in pts {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (lo, hi)
}

fn pctl(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = q * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn print_report(
    positions: &[Vec3],
    triangles: &[[u32; 3]],
    lo: Vec3,
    hi: Vec3,
    signed_vol: f64,
    orient: f32,
    joint_rows: &[JointRow],
    div: &Divergence,
    other_rows: &[(String, f32, f32, bool)],
    blo: Vec3,
    bhi: Vec3,
    bind: Option<&BindReference>,
) {
    let yn = |b: bool| if b { "IN" } else { "OUT" };
    println!();
    println!("=== RL_SKIN_DIAG: settled-pose pivot containment (LIVE skinned mesh) ===");
    println!(
        "live skinned mesh: {} verts, {} triangles, bbox {:.3}..{:.3}, signed_vol={:.4} (orient {:+.0})",
        positions.len(),
        triangles.len(),
        lo,
        hi,
        signed_vol,
        orient,
    );

    println!();
    println!(
        "per-pivot (signed dist: + = OUTSIDE settled skin, - = inside; bindΔ = settled - bind):"
    );
    println!(
        "  {:<26} | {:>7} {:>9} {:>4} | {:>9} {:>9} {:>9}",
        "joint", "wn", "settled.d", "in?", "bind.d", "bindΔ", "link↔bone"
    );
    let mut outside = 0usize;
    let mut worst: Vec<(String, f32, Vec3)> = Vec::new();
    for r in joint_rows {
        if !r.inside {
            outside += 1;
            worst.push((format!("{:?}", r.id), r.dist, r.world));
        }
        let bind_s = r
            .bind_dist
            .map(|d| format!("{d:+.4}"))
            .unwrap_or_else(|| "    -".to_string());
        let delta = r
            .bind_dist
            .map(|bd| format!("{:+.4}", r.dist - bd))
            .unwrap_or_else(|| "    -".to_string());
        let gap = div
            .per_joint
            .get(&r.id)
            .map(|g| format!("{g:.4}"))
            .unwrap_or_else(|| "  -".to_string());
        println!(
            "  {:<26} | {:>+7.3} {:>+9.4} {:>4} | {:>9} {:>9} {:>9}",
            format!("{:?}", r.id),
            r.wn,
            r.dist,
            yn(r.inside),
            bind_s,
            delta,
            gap,
        );
    }

    // Carapace + eye-stalk markers (not in the 30, reported for context).
    if !other_rows.is_empty() {
        println!();
        println!("other markers (carapace + locked eye-stalks; not in the 30):");
        for (label, wn, dist, inside) in other_rows {
            println!(
                "  {:<26} | {:>+7.3} {:>+9.4} {:>4}",
                label,
                wn,
                dist,
                yn(*inside)
            );
        }
    }

    // Skin-vs-physics divergence summary (cause (a)). A near-zero std = the gap is one
    // rigid frame offset shared by every joint (skin sits a fixed distance off the
    // body), not per-joint drift.
    println!();
    println!(
        "skin↔physics pivot-bone gap (world units): median={:.4}, p95={:.4}, max={:.4}, std={:.4} ({})",
        div.median, div.p95, div.max, div.std, div.max_label
    );
    println!(
        "  std/median = {:.3} → {}",
        div.std / div.median.max(1e-6),
        if div.std < 0.1 * div.median.max(1e-6) {
            "UNIFORM: a rigid bind-vs-spawn frame offset, not per-joint drift (cause a = systematic offset)"
        } else {
            "SCATTERED: joints diverge by different amounts → genuine per-joint skin drift"
        }
    );

    // Size/offset check (cause (b)).
    let skin_size = hi - lo;
    let body_size = bhi - blo;
    let skin_c = (lo + hi) * 0.5;
    let body_c = (blo + bhi) * 0.5;
    println!();
    println!("AABB size/offset check (cause (b): is the skin systematically smaller/offset?):");
    println!(
        "  skin   bbox {:.3}..{:.3}  size {:.3}  center {:.3}",
        lo, hi, skin_size, skin_c
    );
    println!(
        "  body   bbox {:.3}..{:.3}  size {:.3}  center {:.3}  (link-pivot envelope)",
        blo, bhi, body_size, body_c
    );
    println!(
        "  skin/body size ratio = ({:.2}, {:.2}, {:.2}); center offset = {:.4} ({:.3})",
        skin_size.x / body_size.x.max(1e-6),
        skin_size.y / body_size.y.max(1e-6),
        skin_size.z / body_size.z.max(1e-6),
        (skin_c - body_c).length(),
        skin_c - body_c,
    );
    // The body envelope is link ORIGINS, so the skin (a surface) should be LARGER on
    // every axis; a ratio < 1 on an axis = the skin is narrower than the pivots it
    // wraps, i.e. systematically undersized there.
    println!("  (skin is a surface, body is pivot points → expect skin > body on every axis)");

    // Bind reference summary, if loaded.
    if let Some(b) = bind {
        let (blo2, bhi2) = aabb(&b.positions);
        let bind_outside = joint_rows
            .iter()
            .filter(|r| r.bind_dist.is_some_and(|d| d > 0.0))
            .count();
        println!();
        println!(
            "bind reference: {} verts, bbox {:.3}..{:.3}; {} of {} pivots OUTSIDE the BIND mesh (verify-pivots view)",
            b.positions.len(),
            blo2,
            bhi2,
            bind_outside,
            joint_rows.len(),
        );
    }

    // Headline.
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!();
    println!(
        "SUMMARY: {} of {} link pivots are OUTSIDE the SETTLED skinned mesh.",
        outside,
        joint_rows.len()
    );
    if !worst.is_empty() {
        println!("worst (model units outside the settled surface; world pos):");
        for (label, d, pos) in worst.iter().take(12) {
            println!("  {:<26} {:+.4}   at {:.3}", label, d, pos);
        }
    }
    println!("=== end RL_SKIN_DIAG ===");
    println!();
}
