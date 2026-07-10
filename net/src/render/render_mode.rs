use super::scene::CrabAvatar;
use super::*;
use crate::controls::{self, Action};
use bevy_rapier3d::prelude::Collider;
pub use crab_world::crab_view::RenderMode;
use crab_world::crab_view::{COLLIDER_WIREFRAME_COLOR, draw_collider_wireframe};
use crab_world::vehicle::Vehicle;

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
    app.add_systems(
        PostUpdate,
        draw_vehicle_collider_wireframe
            .after(TransformSystems::Propagate)
            .run_if(in_state(AppPhase::Playing)),
    );
}

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

fn manage_silhouette_visibility(
    mode: Res<RenderMode>,
    armed: Option<Res<crate::external_crab::ExternalCrabArmed>>,
    mut q: Query<&mut Visibility, With<CrabAvatar>>,
) {
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

fn draw_vehicle_collider_wireframe(
    mode: Res<RenderMode>,
    anchor: Res<crate::external_crab::ArenaAnchor>,
    remote: Res<super::articulation::RemoteVehicle>,
    vehicles: Query<(&GlobalTransform, &Collider), With<Vehicle>>,
    mut gizmos: Gizmos,
) {
    // The STATIC arena→render frame (rl#224) — never the per-crab skin repose, which tracks
    // the live carapace and would drag Sally's every wiggle into each craft's rendered pose.
    let placement = Mat4::from_translation(anchor.0);
    if mode.shows_colliders() && !vehicles.is_empty() {
        for (gt, collider) in &vehicles {
            let world = placement * gt.to_matrix();
            draw_collider_wireframe(
                &mut gizmos,
                collider.as_typed_shape(),
                world,
                COLLIDER_WIREFRAME_COLOR,
            );
        }
        // On the HOST the entity query covers EVERY craft (its world simulates all pilots'),
        // so the RemoteVehicle pass below would ghost each remote craft a tick behind its
        // live entity. A client has no Vehicle entities and always takes the wire pass.
        return;
    }
    for v in &remote.0 {
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
