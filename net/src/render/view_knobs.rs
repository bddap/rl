use super::*;
use crate::controls::{self, Action};
use crab_world::ground::GroundLook;
use crab_world::moon::MoonSong;

pub fn register(app: &mut App) {
    // Craft models ride the same registration seam so the windowed and screenshot apps
    // both get them (rl#260).
    super::vehicle_view::register(app);
    // Gated on Playing (rl#211): the chord capture runs in every phase, so ungated a code
    // typed in the menu would switch the look.
    app.add_systems(
        Update,
        (
            select_ground_look.run_if(in_state(AppPhase::Playing)),
            play_moon_songs
                .run_if(in_state(AppPhase::Playing))
                .before(crab_world::moon::MoonMotionSet),
        ),
    );
}

/// The look's chord action — one code per variant (rl#330 stage 5), no cycle verb.
/// Exhaustive by construction: a new [`GroundLook`] with no action — and hence no code in
/// `GCR_CHORDS` — is a compile error here, and the coverage test below catches an action
/// left out of the table.
fn look_action(look: GroundLook) -> Action {
    use GroundLook::*;
    match look {
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

/// The song's chord action. Exhaustive like [`look_action`].
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

fn select_ground_look(
    chords: Res<crab_world::chord::Chords<controls::GcrControls>>,
    mut knob: ResMut<GroundLook>,
) {
    for &v in crab_world::view_variants::<GroundLook>() {
        if chords.executed(look_action(v)) {
            *knob = v;
            info!("ground look: {}", crab_world::view_variant_name(&v));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every ground look is reachable by a chord code (rl#330 stage 5): the
    /// [`look_action`] match makes a NEW variant a compile error, and this closes the
    /// other half — its action must actually sit in [`controls::GCR_CHORDS`].
    #[test]
    fn every_ground_look_has_a_chord_code() {
        for &v in crab_world::view_variants::<GroundLook>() {
            assert!(
                controls::GCR_CHORDS
                    .entries()
                    .iter()
                    .any(|e| e.action == look_action(v)),
                "ground look {} has no chord entry",
                crab_world::view_variant_name(&v)
            );
        }
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
        let mut all: Vec<Action> = crab_world::view_variants::<GroundLook>()
            .iter()
            .map(|&v| look_action(v))
            .collect();
        all.extend(MoonSong::ALL.map(song_action));
        for (i, a) in all.iter().enumerate() {
            assert!(!all[i + 1..].contains(a), "action {a:?} mapped twice");
        }
    }
}
