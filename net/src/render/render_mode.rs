//! GCR's render-mode glue: drive the SHARED [`crab_world::crab_view::RenderMode`] cycle from
//! GCR's controls source, hide the giant-crab SILHOUETTE in the colliders-only view, and name
//! the active mode on the HUD.
//!
//! The cage itself (the ONE collider-wireframe) and the SKIN's mesh-visibility live in the
//! shared [`crab_world::crab_view`] / [`crab_world::bot::skin`] — reused as-is by the rl-demo.
//! This module only adds what's GCR-specific: the keyboard/pad cycle (read through GCR's
//! [`crate::controls`] bindings, so the HUD can't drift from the button), the silhouette-hide
//! (GCR's own physics-bones entity, separate from the skin), and the persistent corner label
//! naming the mode.

use super::scene::CrabAvatar;
use super::*;
use crate::controls::{self, Action};
use bevy_rapier3d::prelude::Collider;
use crab_world::bot::skin::CrabSkinRepose;
pub use crab_world::crab_view::RenderMode;
use crab_world::crab_view::{COLLIDER_WIREFRAME_COLOR, draw_collider_wireframe};
use crab_world::vehicle::Vehicle;

/// Wire GCR's render-mode cycle into a render `App`, booting in `initial` (the missing-glb
/// fallback passes [`RenderMode::Colliders`]). Adds the shared cage + skin-visibility + the
/// mode-naming HUD label via [`crab_world::crab_view::register`], then GCR's silhouette-hide and
/// the live cycle. Call once, after the sim systems are installed.
pub fn register(app: &mut App, initial: RenderMode) {
    // Everything render-mode is gated on Playing (rl#211): gizmos render through ANY camera —
    // the menu's Camera2d included — and the crab body deliberately survives round teardown, so
    // ungated the cage draws over the post-disconnect menu; and pad East is ALSO menu Back, so
    // ungated cycle input means dismissing that screen cycles the mode. Both callers hold the
    // state: the windowed app inits it, the screenshot app boots pinned to Playing.
    crab_world::crab_view::register(app, initial, in_state(AppPhase::Playing));
    app.add_systems(
        Update,
        (
            cycle_render_mode.run_if(in_state(AppPhase::Playing)),
            manage_silhouette_visibility,
        ),
    );
    // The piloted craft's collider wireframe — same cycle, same drawer, same repose as the crab
    // cage. Drawn AFTER transform propagation so the body's `GlobalTransform` is this frame's
    // physics pose. Only GCR spawns a `Vehicle`, so the query is empty (a no-op) in the rl-demo,
    // which registers the shared crab cage directly and never calls this. The `RemoteVehicle`
    // mirror it reads is owned per-round by `install_round`, which runs before Playing (the
    // drawer's gate) on every path.
    app.add_systems(
        PostUpdate,
        draw_vehicle_collider_wireframe
            .after(TransformSystems::Propagate)
            .run_if(in_state(AppPhase::Playing)),
    );
}

/// Cycle the render mode on the `CycleRenderMode` control (keyboard V / pad B), read through
/// the SAME [`crate::controls`] bindings the on-screen legend shows — so the HUD names the
/// button that actually cycles. Pure client UI: it never touches the sim.
fn cycle_render_mode(
    keys: Res<ButtonInput<KeyCode>>,
    pads: Query<&Gamepad>,
    mut mode: ResMut<RenderMode>,
) {
    if crab_world::controls::just_pressed::<controls::GcrControls>(
        Action::CycleRenderMode,
        &keys,
        &pads,
    ) {
        *mode = mode.next();
        info!("render mode: {:?}", *mode);
    }
}

/// Show/hide GCR's giant-crab SILHOUETTE per render mode. The silhouette is the visible crab
/// ONLY when the skinned NN rig isn't (no armed crab, or no Sally model — then the silhouette
/// IS the honest physics-bones mesh). When it's the visible mesh, it follows
/// [`RenderMode::shows_mesh`]; the colliders-only view hides it so the shared cage reads clean.
/// When the skin is the crab, the silhouette stays hidden (the skin's own visibility, driven by
/// the same mode in [`crab_world::bot::skin`], handles the mesh). One authority per frame,
/// superseding `spawn_world`'s initial guess.
fn manage_silhouette_visibility(
    mode: Res<RenderMode>,
    armed: Option<Res<crate::external_crab::ExternalCrabArmed>>,
    mut q: Query<&mut Visibility, With<CrabAvatar>>,
) {
    // net's single asset source is the global preflight verdict (the silhouette runs without the bot
    // stack, so it can't read the `CrabModelPath` BotPlugin resource; net never overrides it anyway).
    // The memoized `usable_model` verdict, NOT existence-only `model_path()`, so this per-frame guard
    // agrees with `spawn_world`'s `have_model`: a present-but-broken glb is NOT the skin (the body
    // fell back to the silhouette), so the silhouette must stay shown (bddap/rl#154). Memoized ⇒ this
    // stays a cheap read every frame.
    let skin_is_the_crab =
        armed.is_some() && crab_world::mesh_fallback::usable_model_path().is_some();
    let want = if !skin_is_the_crab && mode.shows_mesh() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut q {
        if *vis != want {
            *vis = want;
        }
    }
}

/// Draw the PILOTED craft's collider as a gizmo wireframe at its RENDERED pose — the "physics debug
/// wireframe" for the plane/ship, the sibling of the crab cage in [`crab_world::crab_view`], sharing
/// the SAME [`RenderMode`] cycle, the SAME drawer, and the SAME [`CrabSkinRepose`] repose. Active in
/// any mode that [`RenderMode::shows_colliders`], so the ONE `CycleRenderMode` control toggles the
/// crab AND the craft together, every context.
///
/// The vehicle rapier body lives in the crab's ARENA frame (the open inference field with Sally — rl#209), so — like the
/// crab body — it must be reposed into render space to sit where the pilot sees it: the cockpit camera
/// applies exactly this `world(crab) − arena_carapace` shift (see [`super::scene`]'s `apply_transforms`),
/// so reusing the same [`CrabSkinRepose`] the crab cage uses puts the cage on the craft the camera
/// flies. No scale (render==physics). One body at a time (the player flies a single craft); despawned
/// on foot, so the query is empty then.
///
/// A remote MP client has no `Vehicle` body at all (host-authoritative — only the host steps the
/// craft's rapier); its view of the host's craft is the [`RemoteVehicle`] pose mirror off the wire,
/// drawn here in EVERY mode, not just the collider views: the craft has no mesh, so the cage IS its
/// visual (the missing-glb precedent — with no model the honest physics view is the thing itself),
/// and the second player must see the host fly regardless of their view cycle (rl#192). The
/// spectator is never inside it, so the always-on cage can't sit over the camera.
///
/// [`RemoteVehicle`]: super::articulation::RemoteVehicle
fn draw_vehicle_collider_wireframe(
    mode: Res<RenderMode>,
    repose: Option<Res<CrabSkinRepose>>,
    remote: Res<super::articulation::RemoteVehicle>,
    vehicles: Query<(&GlobalTransform, &Collider), With<Vehicle>>,
    mut gizmos: Gizmos,
) {
    // The piloted craft flies in the shared arena; env 0's repose is the render placement
    // reference (identity when absent, e.g. the rl-demo).
    let placement = repose
        .as_deref()
        .and_then(|r| r.0.get(&0).copied())
        .map(|s| s.matrix())
        .unwrap_or(Mat4::IDENTITY);
    // The locally-piloted body: collider views only — the pilot flies FP from inside it.
    if mode.shows_colliders() {
        for (gt, collider) in &vehicles {
            let world = placement * gt.to_matrix();
            draw_collider_wireframe(
                &mut gizmos,
                collider.as_typed_shape(),
                world,
                COLLIDER_WIREFRAME_COLOR,
            );
        }
    }
    // The host's craft mirrored off the wire, rebuilt from the ONE collider source
    // ([`crab_world::vehicle::vehicle_collider`]) so the drawn cage can't drift from the body
    // it depicts.
    if let Some(v) = remote.0 {
        let world = placement
            * Mat4::from_rotation_translation(Quat::from_array(v.rot), Vec3::from_array(v.pos));
        draw_collider_wireframe(
            &mut gizmos,
            crab_world::vehicle::vehicle_collider().as_typed_shape(),
            world,
            COLLIDER_WIREFRAME_COLOR,
        );
    }
}
