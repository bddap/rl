//! The crab render-mode cycle + the ONE collider-wireframe implementation, shared by GCR
//! (`net::render`) and the rl-demo. Both render the crab at its TRUE physics size now
//! (render==physics — the giant feel comes from shrinking the human world, not inflating the
//! crab), so a collider wireframe overlays the mesh NATIVELY and one cage serves both binaries.
//!
//! [`RenderMode`] cycles `mesh → mesh+colliders → colliders`. A player-facing view toggle
//! (not a hidden debug flag): the binding + label come from each binary's controls-display
//! source ([`crate::controls`]), so the on-screen HUD can't drift from what the button does.
//! The MISSING-GLB fallback boots straight into [`RenderMode::Colliders`]: with no Sally mesh
//! the honest physics view IS the crab the player sees (never a placeholder box).
//!
//! The cage is drawn with gizmos (not Rapier's `RapierDebugRenderPlugin`) because GCR's crab
//! body lives at the ~1 m arena origin while the rendered crab is translated to its game spot:
//! [`crate::bot::skin::CrabSkinRepose`] carries that rigid shift, so reusing it (identity in
//! the rl-demo, where body == rendered crab) puts the cage exactly on the mesh in BOTH. No
//! scale anywhere — render==physics, so the cage is the colliders at true size.

use bevy::prelude::*;
use bevy_rapier3d::geometry::ColliderView;
use bevy_rapier3d::prelude::Collider;

use crate::bot::body::{CrabBodyPart, CrabEnvId};
use crate::bot::skin::CrabSkinRepose;

/// What the crab render shows. A 3-state cycle, driven by one controller button (+ key) wired
/// through each binary's controls source. `Mesh` is the default player-facing view; the two
/// collider views overlay (or replace) it with the honest physics cage.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum RenderMode {
    /// The skinned Sally mesh only (or, with no glb, the physics-bones silhouette) — the
    /// player-facing default.
    #[default]
    Mesh,
    /// The mesh WITH the collider wireframe drawn on top — "does my hitbox match what I see".
    /// Since render==physics, the cage sits exactly on the mesh.
    MeshColliders,
    /// The collider wireframe ALONE (mesh hidden) — the honest physics view, and the
    /// missing-glb fallback.
    Colliders,
}

impl RenderMode {
    /// Cycle `Mesh → MeshColliders → Colliders → Mesh`, so one button walks every view.
    pub fn next(self) -> Self {
        match self {
            RenderMode::Mesh => RenderMode::MeshColliders,
            RenderMode::MeshColliders => RenderMode::Colliders,
            RenderMode::Colliders => RenderMode::Mesh,
        }
    }

    /// Whether the crab MESH (skin / silhouette) is shown in this mode — `false` only for the
    /// colliders-only view, which hides the mesh so the cage reads clean.
    pub fn shows_mesh(self) -> bool {
        !matches!(self, RenderMode::Colliders)
    }

    /// Whether the collider wireframe is drawn in this mode.
    pub fn shows_colliders(self) -> bool {
        matches!(self, RenderMode::MeshColliders | RenderMode::Colliders)
    }

    /// The short HUD name of this mode — what the on-screen label reads. Stays in sync with the
    /// cycle because both the label and the input read this one enum.
    pub fn label(self) -> &'static str {
        match self {
            RenderMode::Mesh => "mesh",
            RenderMode::MeshColliders => "mesh + colliders",
            RenderMode::Colliders => "colliders",
        }
    }

    /// Parse a render-mode token (a CLI flag / env value). `None` for an unknown token (the
    /// caller reports it); the three modes map by name, with `mesh-colliders` as an alias for
    /// the `+` form that's awkward on a shell.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mesh" => Some(RenderMode::Mesh),
            "mesh+colliders" | "mesh-colliders" => Some(RenderMode::MeshColliders),
            "colliders" => Some(RenderMode::Colliders),
            _ => None,
        }
    }

    /// The initial mode from the environment, for callers with no explicit flag. Honors
    /// `RL_RENDER_MODE=<mode>`, else falls back to the legacy idiom where `RL_DEBUG_COLLIDERS`
    /// (any value) boots the colliders-only view. Default [`RenderMode::Mesh`].
    pub fn from_env() -> Self {
        if let Ok(v) = std::env::var("RL_RENDER_MODE") {
            return RenderMode::parse(&v).unwrap_or_else(|| {
                warn!("RL_RENDER_MODE={v:?} not one of mesh|mesh+colliders|colliders; defaulting mesh");
                RenderMode::Mesh
            });
        }
        if std::env::var_os("RL_DEBUG_COLLIDERS").is_some() {
            RenderMode::Colliders
        } else {
            RenderMode::Mesh
        }
    }
}

/// Wire the shared render-mode cage into a render `App`, booting in `initial` (the missing-glb
/// fallback passes [`RenderMode::Colliders`]). Inserts the [`RenderMode`] resource and the gizmo
/// cage. The mesh-visibility half lives where each mesh does — the skin in
/// [`crate::bot::skin`] (shared, reads this resource), GCR's silhouette in `net::render`. Call
/// once, after the bot/physics systems are installed.
pub fn register(app: &mut App, initial: RenderMode) {
    app.insert_resource(initial);
    // Draw AFTER transform propagation so each part's `GlobalTransform` holds this frame's
    // physics pose; otherwise the cage lags a frame.
    app.add_systems(
        PostUpdate,
        draw_crab_collider_wireframe.after(TransformSystems::Propagate),
    );
}

/// Draw the crab's live colliders as a gizmo wireframe at the crab's RENDERED pose. Active in
/// any mode that [`RenderMode::shows_colliders`]. The transform is `repose · part_global`: each
/// part's `GlobalTransform` is its raw physics pose, and [`crate::bot::skin::SkinRepose::matrix`]
/// is the rigid shift the skin applies to relocate the arena rig to its game spot (identity in
/// the rl-demo, where the body IS the rendered crab). Reusing it — not a re-derived factor — is
/// why the cage can't drift from the rendered crab. There is no scale: render==physics. Crab env
/// 0 only (both binaries render one env).
fn draw_crab_collider_wireframe(
    mode: Res<RenderMode>,
    repose: Option<Res<CrabSkinRepose>>,
    parts: Query<(&GlobalTransform, &Collider, &CrabEnvId), With<CrabBodyPart>>,
    mut gizmos: Gizmos,
) {
    if !mode.shows_colliders() {
        return;
    }
    // The crab's render placement: the skin's rigid shift in GCR, identity in the rl-demo (no
    // bridge ⇒ no repose ⇒ the body is already the rendered crab). render==physics, so there is
    // no scale to apply to the cage.
    let placement = repose
        .as_deref()
        .and_then(|r| r.0)
        .map(|s| s.matrix())
        .unwrap_or(Mat4::IDENTITY);
    let color = Color::srgb(0.2, 1.0, 0.4);
    for (gt, collider, env) in &parts {
        if env.0 != 0 {
            continue;
        }
        let world = placement * gt.to_matrix();
        draw_collider_view(&mut gizmos, collider.as_typed_shape(), world, color);
    }
}

/// Draw one collider view as gizmo lines under world transform `world`. Handles the shapes the
/// crab body actually uses — the carapace compound-of-cuboid and the per-link capsules; other
/// shapes are skipped (the crab has none). render==physics, so `world` carries no scale and the
/// shapes draw at true collider size.
fn draw_collider_view(gizmos: &mut Gizmos, view: ColliderView<'_>, world: Mat4, color: Color) {
    match view {
        ColliderView::Cuboid(c) => draw_cuboid(gizmos, world, c.half_extents(), color),
        ColliderView::Capsule(c) => {
            let seg = c.segment();
            draw_capsule(gizmos, world, seg.a(), seg.b(), c.radius(), color);
        }
        ColliderView::Compound(c) => {
            for (pos, rot, sub) in c.shapes() {
                let sub_world = world * Mat4::from_rotation_translation(rot, pos);
                draw_collider_view(gizmos, sub, sub_world, color);
            }
        }
        // The crab uses only the above; anything else (a future shape) is skipped rather than
        // mis-drawn.
        _ => {}
    }
}

/// Wireframe box: the 12 edges of the cuboid `±half`, each corner pushed through `world`.
/// bevy 0.18's gizmos have no cuboid primitive, so draw the edges directly.
fn draw_cuboid(gizmos: &mut Gizmos, world: Mat4, half: Vec3, color: Color) {
    let corner = |sx: f32, sy: f32, sz: f32| {
        world.transform_point3(Vec3::new(sx * half.x, sy * half.y, sz * half.z))
    };
    let c = [
        corner(-1.0, -1.0, -1.0),
        corner(1.0, -1.0, -1.0),
        corner(1.0, -1.0, 1.0),
        corner(-1.0, -1.0, 1.0),
        corner(-1.0, 1.0, -1.0),
        corner(1.0, 1.0, -1.0),
        corner(1.0, 1.0, 1.0),
        corner(-1.0, 1.0, 1.0),
    ];
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    for (a, b) in edges {
        gizmos.line(c[a], c[b], color);
    }
}

/// Wireframe capsule between the link-local segment endpoints `a`,`b` pushed through `world`.
/// The `Capsule3d` gizmo is Y-aligned, so rotate +Y onto the segment direction. render==physics
/// ⇒ `radius` is the true collider radius (no scale).
fn draw_capsule(gizmos: &mut Gizmos, world: Mat4, a: Vec3, b: Vec3, radius: f32, color: Color) {
    let pa = world.transform_point3(a);
    let pb = world.transform_point3(b);
    let seg = pb - pa;
    let len = seg.length();
    let rot = if len > 1e-6 {
        Quat::from_rotation_arc(Vec3::Y, seg / len)
    } else {
        Quat::IDENTITY
    };
    gizmos.primitive_3d(
        &Capsule3d::new(radius, len),
        Isometry3d::new((pa + pb) * 0.5, rot),
        color,
    );
}
