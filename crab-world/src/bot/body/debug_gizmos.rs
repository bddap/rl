//! Joint-pivot markers — the companion to the collider cage; drawn when the render mode
//! shows colliders (see [`draw_pivot_markers`]).
//!
//! Render-only: gizmos draw through a camera, so this whole module is dead in the
//! headless trainer and its types (Gizmos/GizmoConfig) don't exist without bevy's
//! render feature. The whole module is `render`-gated; the demo/screenshot bins keep it.

use bevy::prelude::*;

use super::components::CrabBodyPart;

/// A separate gizmo config group so the pivot markers can force `depth_bias = -1.0`
/// (always-in-front) WITHOUT changing how Rapier's collider wireframes render —
/// Rapier draws into the default group, which we leave depth-tested so the cage
/// still reads as 3D. The pivots sit buried inside the opaque skin, so without the
/// override every marker would be hidden by the body and the screenshot would prove
/// nothing.
#[derive(Default, Reflect, GizmoConfigGroup)]
#[reflect(Default)]
pub struct PivotGizmos;

/// Marker sphere radius (model units): big enough to spot against the skin yet
/// small enough not to swallow the joint it marks.
const PIVOT_MARKER_RADIUS: f32 = 0.02;

/// Draw a bright sphere at every physics link's world origin — which, by
/// construction, IS that link's joint pivot (`spawn_crab` anchors each child at its
/// parent with `local_anchor2 = ZERO`), plus the carapace root. Magenta to stay
/// distinct from both the Sally model's orange skin and Rapier's collider
/// wireframes. `GlobalTransform`, not `Transform`: a marker must sit at the pivot's
/// true world point even mid-tumble, and only the global has the full parent chain
/// resolved. Always-in-front comes from [`PivotGizmos`]'s `depth_bias`.
fn draw_pivot_markers(
    mode: Option<Res<crate::crab_view::RenderMode>>,
    parts: Query<&GlobalTransform, With<CrabBodyPart>>,
    mut gizmos: Gizmos<PivotGizmos>,
) {
    // The pivots are the companion to the collider cage, so they ride the same render mode —
    // shown only when colliders are. Absent resource (no cycle wired) ⇒ draw, the prior behavior.
    if !mode.map(|m| m.shows_colliders()).unwrap_or(true) {
        return;
    }
    let color = Color::srgb(1.0, 0.0, 1.0); // magenta
    for gt in &parts {
        gizmos.sphere(
            Isometry3d::from_translation(gt.translation()),
            PIVOT_MARKER_RADIUS,
            color,
        );
    }
}

/// Wire up the joint-pivot debug markers. Registered unconditionally by the rl-demo; the draw
/// self-gates on the render mode ([`crate::crab_view::RenderMode::shows_colliders`]), so the
/// pivots and the collider cage — the two physics-truth overlays — appear together. Drawn
/// through whatever camera renders the gizmos (the windowed demo's or the offscreen
/// screenshot's), so it shows up in `--screenshot` too.
pub fn register_pivot_markers(app: &mut App) {
    app.insert_gizmo_config(
        PivotGizmos,
        GizmoConfig {
            // -1.0 = always render in front; the pivots are inside the opaque body.
            depth_bias: -1.0,
            ..default()
        },
    );
    app.add_systems(Update, draw_pivot_markers);
}
