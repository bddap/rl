use super::scene::CrabAvatar;
use super::*;
use crate::controls::{self, Action};
use bevy_rapier3d::prelude::Collider;
pub use crab_world::crab_view::RenderMode;
use crab_world::crab_view::{COLLIDER_WIREFRAME_COLOR, draw_collider_wireframe};
use crab_world::moon::MoonSong;
use crab_world::vehicle::Vehicle;

pub fn register(app: &mut App, initial: RenderMode) {
    // Everything render-mode is gated on Playing (rl#211): gizmos render through ANY camera —
    // the menu's Camera2d included — and the crab body deliberately survives round teardown, so
    // ungated the cage draws over the post-disconnect menu; and the chord capture runs in every
    // phase, so ungated cycle input means a code typed in the menu cycles the mode. Both callers
    // hold the state: the windowed app inits it, the screenshot app boots pinned to Playing.
    crab_world::crab_view::register(app, initial, in_state(AppPhase::Playing));
    // Craft models ride the same registration seam so the windowed and screenshot apps
    // both get them (rl#260).
    super::vehicle_view::register(app);
    app.add_systems(
        Update,
        (
            select_view::<RenderMode>.run_if(in_state(AppPhase::Playing)),
            select_view::<crab_world::ground::GroundLook>.run_if(in_state(AppPhase::Playing)),
            play_moon_songs
                .run_if(in_state(AppPhase::Playing))
                .before(crab_world::moon::MoonMotionSet),
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

/// A view knob the player sets by chord code — one code per variant (rl#330 stage 5),
/// no cycle verb: a `clap` value enum that is ALSO the live resource, so the states its
/// flag accepts and the states a code reaches are one list. Both knobs are pure
/// dressing — neither reaches simulated state, so switching mid-round changes nothing
/// an eval or a peer would see.
pub(crate) trait ViewKnob:
    Resource<Mutability = bevy::ecs::component::Mutable> + crab_world::CyclableView
{
    /// How the knob names itself in the log line.
    const LOG_LABEL: &'static str;
    /// The variant's chord action. Exhaustive by construction: a new variant with no
    /// action — and hence no code in `GCR_CHORDS` — is a compile error here, and the
    /// registry-coverage test below catches an action left out of the table.
    fn action(self) -> Action;
}

impl ViewKnob for RenderMode {
    const LOG_LABEL: &'static str = "render mode";
    fn action(self) -> Action {
        match self {
            RenderMode::Mesh => Action::RenderMesh,
            RenderMode::MeshColliders => Action::RenderMeshColliders,
            RenderMode::Colliders => Action::RenderColliders,
        }
    }
}

impl ViewKnob for crab_world::ground::GroundLook {
    const LOG_LABEL: &'static str = "ground look";
    fn action(self) -> Action {
        use crab_world::ground::GroundLook::*;
        match self {
            Shipped => Action::GroundShipped,
            NightBloom => Action::GroundNightBloom,
            PatternedGround => Action::GroundPatternedGround,
            WindCombed => Action::GroundWindCombed,
            CrackedLoam => Action::GroundCrackedLoam,
            Watershed => Action::GroundWatershed,
            WatershedNaturalist => Action::GroundWatershedNaturalist,
            WatershedNocturne => Action::GroundWatershedNocturne,
            NightBloomAurora => Action::GroundNightBloomAurora,
            NightBloomEmber => Action::GroundNightBloomEmber,
            NightBloomFrost => Action::GroundNightBloomFrost,
            NightBloomRose => Action::GroundNightBloomRose,
            NightBloomFiligree => Action::GroundNightBloomFiligree,
        }
    }
}

/// The song's chord action. Exhaustive like [`ViewKnob::action`]: a new
/// [`MoonSong`] without an action — and hence no code in `GCR_CHORDS` — is a
/// compile error here, and the coverage test below catches an action left out
/// of the table.
fn song_action(song: MoonSong) -> Action {
    match song {
        MoonSong::PhaseFull => Action::MoonFull,
        MoonSong::PhaseWaxing => Action::MoonWaxing,
        MoonSong::PhaseWaning => Action::MoonWaning,
        MoonSong::PhaseNew => Action::MoonNew,
        MoonSong::HueSilver => Action::MoonSilver,
        MoonSong::HueBlood => Action::MoonBlood,
        MoonSong::HueHarvest => Action::MoonHarvest,
        MoonSong::HueVerdant => Action::MoonVerdant,
        MoonSong::TempoDrift => Action::MoonDrift,
        MoonSong::TempoFreeze => Action::MoonFreeze,
        MoonSong::TempoGallop => Action::MoonGallop,
        MoonSong::PoseZenith => Action::MoonZenith,
        MoonSong::PoseRise => Action::MoonRise,
    }
}

/// Chord-song dispatch onto the [`Moon`] knobs (rl#374). Ordered before the
/// motion step so a pose-freeze song lands before motion could overwrite the
/// posed frame. Deref only on a hit — the light/sky syncs filter on the moon's
/// change detection.
fn play_moon_songs(
    chords: Res<crab_world::chord::Chords<controls::GcrControls>>,
    mut moon: ResMut<crab_world::moon::Moon>,
) {
    for song in MoonSong::ALL {
        if chords.executed(song_action(song)) {
            song.apply(&mut moon);
            info!("moon song: {song:?} -> {:?}", *moon);
        }
    }
}

fn select_view<V: ViewKnob>(
    chords: Res<crab_world::chord::Chords<controls::GcrControls>>,
    mut knob: ResMut<V>,
) {
    for &v in crab_world::view_variants::<V>() {
        if chords.executed(v.action()) {
            *knob = v;
            info!("{}: {}", V::LOG_LABEL, crab_world::view_variant_name(&v));
        }
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
    origin: Res<super::RenderOrigin>,
    vehicles: Query<(&GlobalTransform, &Collider), With<Vehicle>>,
    mut gizmos: Gizmos,
) {
    // Mesh mode: the craft models (`vehicle_view`, rl#260) are the visual.
    if !mode.shows_colliders() {
        return;
    }
    // Physics transforms are absolute; gizmos draw in the render frame (rl#354).
    let to_frame = Mat4::from_translation(-origin.offset_m());
    if !vehicles.is_empty() {
        for (gt, collider) in &vehicles {
            let world = to_frame * gt.to_matrix();
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
        let world = to_frame * Mat4::from_rotation_translation(c.pose.orient, c.pose.pos);
        draw_collider_wireframe(
            &mut gizmos,
            crab_world::vehicle::vehicle_collider().as_typed_shape(),
            world,
            COLLIDER_WIREFRAME_COLOR,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every render/art variant is reachable by a chord code (rl#330 stage 5): the
    /// `ViewKnob::action` match makes a NEW variant a compile error, and this closes
    /// the other half — its action must actually sit in [`controls::GCR_CHORDS`].
    #[test]
    fn every_view_variant_has_a_chord_code() {
        fn assert_covered<V: ViewKnob>() {
            for &v in crab_world::view_variants::<V>() {
                assert!(
                    controls::GCR_CHORDS
                        .entries()
                        .iter()
                        .any(|e| e.action == v.action()),
                    "{} variant {} has no chord entry",
                    V::LOG_LABEL,
                    crab_world::view_variant_name(&v)
                );
            }
        }
        assert_covered::<RenderMode>();
        assert_covered::<crab_world::ground::GroundLook>();
    }

    /// The moon-song half of the same contract (rl#374): every song's action
    /// sits in the registry, i.e. every song is playable — and the registry's
    /// `Moon: ` family is exactly the songs, no orphan entry (the count check
    /// lives here, not in the controls tests, because `crab_world::moon` is
    /// render-gated and that test mod compiles without render).
    #[test]
    fn every_moon_song_has_a_chord_code() {
        for song in MoonSong::ALL {
            assert!(
                controls::GCR_CHORDS
                    .entries()
                    .iter()
                    .any(|e| e.action == song_action(song)),
                "moon song {song:?} has no chord entry"
            );
        }
        assert_eq!(
            controls::GCR_CHORDS
                .entries()
                .iter()
                .filter(|e| e.code.starts_with(controls::MOON_FAMILY))
                .count(),
            MoonSong::ALL.len(),
            "registry has a ^v-family entry no song claims"
        );
    }

    /// The variant→action maps must stay injective — two variants sharing an action
    /// would make one code set whichever variant iterates last, silently.
    #[test]
    fn view_variant_actions_are_distinct() {
        fn actions<V: ViewKnob>() -> Vec<Action> {
            crab_world::view_variants::<V>()
                .iter()
                .map(|&v| v.action())
                .collect()
        }
        let mut all = actions::<RenderMode>();
        all.extend(actions::<crab_world::ground::GroundLook>());
        all.extend(MoonSong::ALL.map(song_action));
        for (i, a) in all.iter().enumerate() {
            assert!(!all[i + 1..].contains(a), "action {a:?} mapped twice");
        }
    }
}
