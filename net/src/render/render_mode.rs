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
    // Craft models ride the same registration seam so the windowed and screenshot apps
    // both get them (rl#260).
    super::vehicle_view::register(app);
    app.add_systems(
        Update,
        (
            cycle_view::<RenderMode>.run_if(in_state(AppPhase::Playing)),
            cycle_view::<crab_world::ground::GroundLook>.run_if(in_state(AppPhase::Playing)),
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

/// A view knob the player walks with one button: a `clap` value enum that is ALSO the
/// live resource, so the states its flag accepts and the states the button reaches are
/// one list. Both knobs are pure dressing — neither reaches simulated state, so cycling
/// mid-round changes nothing an eval or a peer would see.
trait ViewKnob: Resource + crab_world::CyclableView {
    const ACTION: Action;
    /// How the knob names itself in the log line.
    const LOG_LABEL: &'static str;
}

impl ViewKnob for RenderMode {
    const ACTION: Action = Action::CycleRenderMode;
    const LOG_LABEL: &'static str = "render mode";
}

impl ViewKnob for crab_world::ground::GroundLook {
    const ACTION: Action = Action::CycleGroundLook;
    const LOG_LABEL: &'static str = "ground look";
}

fn cycle_view<V: ViewKnob>(
    keys: Res<ButtonInput<KeyCode>>,
    pads: Query<&Gamepad>,
    chords: Res<crab_world::chord::Chords<controls::GcrControls>>,
    mut knob: ResMut<V>,
) {
    if crab_world::controls::just_pressed::<controls::GcrControls>(V::ACTION, &keys, &pads)
        || chords.executed(V::ACTION)
    {
        *knob = crab_world::next_view_variant(*knob);
        info!(
            "{}: {}",
            V::LOG_LABEL,
            crab_world::view_variant_name(&*knob)
        );
    }
}

fn manage_silhouette_visibility(
    mode: Res<RenderMode>,
    armed: Option<Res<crate::crab_slot::NnCrabsArmed>>,
    mut q: Query<&mut Visibility, With<CrabAvatar>>,
) {
    let skin_is_the_crab =
        armed.is_some() && crab_world::mesh_fallback::usable_model_path().is_some();
    let want = if skin_is_the_crab {
        Visibility::Hidden
    } else {
        mode.mesh_visibility()
    };
    for mut vis in &mut q {
        if *vis != want {
            *vis = want;
        }
    }
}

fn draw_vehicle_collider_wireframe(
    mode: Res<RenderMode>,
    remote: Res<super::articulation::RemoteVehicle>,
    clock: Res<super::driver::RenderClock>,
    vehicles: Query<(&GlobalTransform, &Collider), With<Vehicle>>,
    mut gizmos: Gizmos,
) {
    // Mesh mode: the craft models (`vehicle_view`, rl#260) are the visual.
    if !mode.shows_colliders() {
        return;
    }
    if !vehicles.is_empty() {
        for (gt, collider) in &vehicles {
            let world = gt.to_matrix();
            draw_collider_wireframe(
                &mut gizmos,
                collider.as_typed_shape(),
                world,
                COLLIDER_WIREFRAME_COLOR,
            );
        }
        // On the HOST the entity query covers EVERY craft (its world simulates all
        // pilots'), and this pass deliberately draws the LIVE rigidbody poses: the
        // colliders view is a physics debug surface, so it shows where physics IS —
        // in mesh+colliders mode it leads the sampled craft models by the window's
        // one-step render latency (rl#267), which is that latency made visible, not a
        // bug. A client has no Vehicle entities and always takes the sampled pass.
        return;
    }
    for c in &remote.sample(clock.tick, clock.frac) {
        let world = Mat4::from_rotation_translation(c.pose.orient, c.pose.pos);
        draw_collider_wireframe(
            &mut gizmos,
            crab_world::vehicle::vehicle_collider().as_typed_shape(),
            world,
            COLLIDER_WIREFRAME_COLOR,
        );
    }
}
