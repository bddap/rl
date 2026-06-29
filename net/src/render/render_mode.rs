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

use super::*;
use super::scene::CrabAvatar;
use crate::controls::{self, Action};
pub use crab_world::crab_view::RenderMode;

/// The corner text node naming the active render mode.
#[derive(Component)]
struct RenderModeLabel;

/// Wire GCR's render-mode cycle into a render `App`, booting in `initial` (the missing-glb
/// fallback passes [`RenderMode::Colliders`]). Adds the shared cage + skin-visibility via
/// [`crab_world::crab_view::register`], then GCR's silhouette-hide, the live cycle, and the
/// corner label. Call once, after the sim systems are installed.
pub fn register(app: &mut App, initial: RenderMode) {
    crab_world::crab_view::register(app, initial);
    app.add_systems(Startup, spawn_mode_label);
    app.add_systems(
        Update,
        (cycle_render_mode, manage_silhouette_visibility, update_mode_label),
    );
}

fn spawn_mode_label(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(0.4, 1.0, 0.55)),
        // Bottom-right: clear of the status HUD (top line) and the hold-to-reveal controls hint
        // (bottom-left), so the mode line never overlaps them.
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(14.0),
            right: Val::Px(14.0),
            ..default()
        },
        RenderModeLabel,
    ));
}

/// Cycle the render mode on the `CycleRenderMode` control (keyboard V / pad B), read through
/// the SAME [`crate::controls`] bindings the on-screen legend shows — so the HUD names the
/// button that actually cycles. Pure client UI: it never touches the sim.
fn cycle_render_mode(
    keys: Res<ButtonInput<KeyCode>>,
    pads: Query<&Gamepad>,
    mut mode: ResMut<RenderMode>,
) {
    let key = controls::key_code_for(Action::CycleRenderMode).is_some_and(|k| keys.just_pressed(k));
    let pad = pads.iter().any(|gp| {
        controls::gamepad_buttons_for(Action::CycleRenderMode).any(|b| gp.just_pressed(b))
    });
    if key || pad {
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
    let skin_is_the_crab = armed.is_some() && crab_world::bot::meshfit::model_path().is_some();
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

fn update_mode_label(mode: Res<RenderMode>, mut label: Query<&mut Text, With<RenderModeLabel>>) {
    if !mode.is_changed() {
        return;
    }
    if let Ok(mut text) = label.single_mut() {
        **text = format!("Render: {}", mode.label());
    }
}
