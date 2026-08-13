//! The d-pad as instrument (rl#359): every tap the combo system consumes sounds a
//! note, so entering a code IS playing a melody — each item's code becomes a tune,
//! recruiting musical memory for code recall. Installed by
//! [`crate::chord::install_chords`], so every chord surface (GCR, the demo, the
//! offscreen evidence apps) sounds through the one path.
//!
//! The press→sound mapping is DATA, not code (owner amendment on rl#359): a scheme is
//! a pure `(scale, code-prefix, press) → NoteSpec` function plus a [`Scale`], held in
//! the [`InstrumentScheme`] resource — a candidate mapping from the design exploration
//! (heldbreath / patchwalk / harmonic-field, or the web playground's pick) swaps in by
//! replacing the resource, never by touching the capture or input code. The default
//! here is the PLACEHOLDER patch: a relative walk on a dark (minor-pentatonic,
//! matching the nightish/haunting atmosphere) scale, so arbitrary combos stay
//! consonant and successive presses read as a phrase; depth adds a little twin-voice
//! detune so a long code audibly deepens.
//!
//! Notes are procedurally synthesized plucks (additive partials, exponential decay) —
//! no asset dependency, every parameter tunable from the [`NoteSpec`]. The instrument
//! deliberately does NOT route through the external mix bus: while a code is being
//! entered the outside world ducks under it (`net`'s `ExternalBus` muffle), never the
//! reverse.

use bevy::audio::{AddAudioSource, Decodable, PlaybackSettings};
use bevy::prelude::*;

use crate::chord::{ChordDir, Chords};
use crate::controls::ControlScheme;
use crate::wav::SAMPLE_RATE;

/// The instrument's loudness ceiling. Sits ABOVE the muffled external mix by design:
/// while a code is entered the world ducks and the instrument carries the frame.
/// 0.35, not 0.5: the accepted-cadence's three overlapping notes summed to full
/// scale and clipped at the safety clamp (heard as a hard edge on the chime).
const MASTER: f32 = 0.35;

/// A musical scale as data: a root pitch plus the semitone degrees of one octave.
/// Scale-degree indices address it beyond the octave in either direction, so a
/// mapping walks one integer lattice and every reachable pitch is in-scale by
/// construction — no sour intervals whatever the combo.
#[derive(Clone, Copy)]
pub struct Scale {
    pub root_hz: f32,
    pub degrees: &'static [u8],
}

/// The default base scale: A minor pentatonic on A3 — dark, per the owner's mood
/// note (nightish/haunting/sad). A data parameter, not a decision: the final scale
/// is picked by ear on the web playground and swaps in here.
pub const DARK_PENTATONIC: Scale = Scale {
    root_hz: 220.0,
    degrees: &[0, 3, 5, 7, 10],
};

impl Scale {
    /// Pitch of scale-degree index `i` (0 = root; negative/overflowing indices fold
    /// into neighboring octaves).
    pub fn freq(&self, i: i32) -> f32 {
        let n = self.degrees.len() as i32;
        let semis = 12 * i.div_euclid(n) + self.degrees[i.rem_euclid(n) as usize] as i32;
        self.root_hz * (semis as f32 / 12.0).exp2()
    }
}

/// One synthesized note — everything the pluck voice can vary, so a mapping scheme
/// tunes sound entirely through data.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct NoteSpec {
    /// Seconds into the phrase this note starts.
    pub onset_s: f32,
    pub freq_hz: f32,
    /// Twin-voice spread: the two voices sit ± half this apart.
    pub detune_cents: f32,
    /// Amplitude time constant τ — the note rings ~6τ.
    pub decay_s: f32,
    /// 0..1: upper-partial level (1 = glassy chime, 0 = pure dark fundamental).
    pub brightness: f32,
    pub gain: f32,
}

/// A press→sound mapping over combo STATE (the code prefix the press lands after) —
/// swappable data, per the rl#359 amendment.
pub type PressFn = fn(&Scale, &[ChordDir], ChordDir) -> NoteSpec;
/// The resolution sound for a completed code: accepted (registered) or unknown.
pub type ResolveFn = fn(&Scale, &[ChordDir], bool) -> Vec<NoteSpec>;

/// The live mapping scheme. Replace the resource to swap schemes; the capture and
/// input code never change.
#[derive(Resource, Clone)]
pub struct InstrumentScheme {
    pub scale: Scale,
    pub press: PressFn,
    pub resolve: ResolveFn,
}

impl Default for InstrumentScheme {
    fn default() -> Self {
        Self {
            scale: DARK_PENTATONIC,
            press: walk_press,
            resolve: cadence_resolve,
        }
    }
}

/// The placeholder mapping: each direction is a STEP on the scale-degree lattice
/// (up/down big, right/left small), so a code is a melodic contour — two codes
/// sharing a prefix share an opening phrase, and every pitch is in-scale. The walk is
/// a pure function of the path, so a code always plays the same tune.
fn walk_step(d: ChordDir) -> i32 {
    match d {
        ChordDir::Up => 2,
        ChordDir::Right => 1,
        ChordDir::Left => -1,
        ChordDir::Down => -2,
    }
}

fn walk_index(path: &[ChordDir]) -> i32 {
    // Clamped to ±one-and-a-bit octaves: codes are uncapped (rl#380), so a long
    // monotone code pins at the clamp instead of walking off the piano.
    path.iter().map(|&d| walk_step(d)).sum::<i32>().clamp(-6, 8)
}

fn walk_press(scale: &Scale, prefix: &[ChordDir], press: ChordDir) -> NoteSpec {
    let depth = prefix.len();
    let idx = walk_index(&[prefix, &[press]].concat());
    NoteSpec {
        onset_s: 0.0,
        freq_hz: scale.freq(idx),
        // Deeper presses detune wider — a long code audibly needs resolution.
        detune_cents: 3.0 * (depth + 1) as f32,
        decay_s: 0.55,
        brightness: (0.7 - 0.06 * depth as f32).max(0.3),
        gain: 1.0,
    }
}

/// Accepted: a quick rising arpeggio landing on the root's octave, detune collapsed
/// to zero — the exhale. Unknown: a low, damped falling semitone (deliberately
/// OUT-of-scale — the one sour interval in the instrument, reserved for "that code
/// means nothing") that never resolves.
fn cadence_resolve(scale: &Scale, _code: &[ChordDir], accepted: bool) -> Vec<NoteSpec> {
    let note = |onset_s, freq_hz, brightness, decay_s, gain| NoteSpec {
        onset_s,
        freq_hz,
        detune_cents: 0.0,
        decay_s,
        brightness,
        gain,
    };
    if accepted {
        let n = scale.degrees.len() as i32;
        vec![
            note(0.0, scale.freq(0), 0.75, 0.5, 0.8),
            note(0.09, scale.freq(2), 0.75, 0.5, 0.8),
            note(0.18, scale.freq(n), 0.85, 0.9, 1.0),
        ]
    } else {
        let low = scale.root_hz * 0.5;
        vec![
            note(0.0, low, 0.25, 0.35, 0.9),
            note(0.14, low * (-1.0 / 12.0f32).exp2(), 0.2, 0.6, 0.9),
        ]
    }
}

/// One frame's worth of instrument sound as a bevy audio asset: a finite phrase of
/// [`NoteSpec`]s, synthesized by [`PhraseStream`]. Press notes are one-element
/// phrases; resolutions are short melodies.
#[derive(Asset, TypePath)]
pub struct Phrase(pub Vec<NoteSpec>);

impl Decodable for Phrase {
    type DecoderItem = f32;
    type Decoder = PhraseStream;
    fn decoder(&self) -> PhraseStream {
        PhraseStream::new(&self.0)
    }
}

/// One sounding note: twin slightly-detuned voices, each three partials with mild
/// inharmonicity (the chime shimmer), a 3 ms attack, and per-partial exponential
/// decay (upper partials die faster, so the tail mellows like a real pluck).
struct NoteVoice {
    start: u64,
    end: u64,
    /// (phase, phase-increment, amplitude, decay-per-sample) per partial per voice.
    partials: Vec<(f32, f32, f32, f32)>,
    attack_samples: f32,
    gain: f32,
}

/// Partial frequency multiples — a touch sharp of harmonic for chime character.
const PARTIALS: [f32; 3] = [1.0, 2.004, 3.009];

impl NoteVoice {
    fn new(spec: &NoteSpec) -> Self {
        let sr = SAMPLE_RATE as f32;
        let start = (spec.onset_s * sr) as u64;
        let spread = (spec.detune_cents / 2400.0).exp2();
        let mut partials = Vec::with_capacity(PARTIALS.len() * 2);
        for (k, mult) in PARTIALS.iter().enumerate() {
            // τ_k shrinks with partial order; amplitude follows brightness.
            let tau = spec.decay_s / (1.0 + 0.9 * k as f32);
            let decay = (-1.0 / (tau * sr)).exp();
            let amp = match k {
                0 => 1.0,
                1 => 0.6 * spec.brightness,
                _ => 0.35 * spec.brightness * spec.brightness,
            };
            for voice in [spread, 1.0 / spread] {
                let hz = spec.freq_hz * mult * voice;
                partials.push((0.0, std::f32::consts::TAU * hz / sr, amp * 0.5, decay));
            }
        }
        Self {
            start,
            // Ring ~7τ of the slowest partial, then the envelope is < 1e-3.
            end: start + (spec.decay_s * 7.0 * sr) as u64,
            partials,
            attack_samples: 0.003 * sr,
            gain: spec.gain,
        }
    }

    fn sample(&mut self, t: u64) -> f32 {
        if t < self.start || t >= self.end {
            return 0.0;
        }
        let attack = (((t - self.start) as f32) / self.attack_samples).min(1.0);
        let mut s = 0.0;
        for (phase, inc, amp, decay) in &mut self.partials {
            s += phase.sin() * *amp;
            *phase = (*phase + *inc) % std::f32::consts::TAU;
            *amp *= *decay;
        }
        s * attack * self.gain
    }
}

pub struct PhraseStream {
    voices: Vec<NoteVoice>,
    t: u64,
    end: u64,
}

impl PhraseStream {
    fn new(notes: &[NoteSpec]) -> Self {
        let voices: Vec<NoteVoice> = notes.iter().map(NoteVoice::new).collect();
        let end = voices.iter().map(|v| v.end).max().unwrap_or(0);
        Self { voices, t: 0, end }
    }
}

impl Iterator for PhraseStream {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.t >= self.end {
            return None;
        }
        let s: f32 = self.voices.iter_mut().map(|v| v.sample(self.t)).sum();
        self.t += 1;
        Some((s * MASTER).clamp(-1.0, 1.0))
    }
}

impl bevy::audio::Source for PhraseStream {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        1
    }
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        Some(std::time::Duration::from_secs_f64(
            self.end as f64 / SAMPLE_RATE as f64,
        ))
    }
}

/// Turn this frame's consumed taps + resolution into one phrase — pure, shared by the
/// live system and the evidence renderer so the proof clip can't drift from the game.
pub fn frame_phrase(
    scheme: &InstrumentScheme,
    presses: impl Iterator<Item = (impl AsRef<[ChordDir]>, ChordDir)>,
    resolution: Option<(&[ChordDir], bool)>,
) -> Vec<NoteSpec> {
    let mut notes: Vec<NoteSpec> = presses
        .map(|(prefix, d)| (scheme.press)(&scheme.scale, prefix.as_ref(), d))
        .collect();
    if let Some((code, accepted)) = resolution {
        notes.extend((scheme.resolve)(&scheme.scale, code, accepted));
    }
    notes
}

fn sound_chords<S: ControlScheme>(
    chords: Res<Chords<S>>,
    scheme: Res<InstrumentScheme>,
    live: Query<(), With<AudioPlayer<Phrase>>>,
    mut assets: ResMut<Assets<Phrase>>,
    mut commands: Commands,
) {
    let mut notes = frame_phrase(&scheme, chords.presses(), chords.resolution());
    if notes.is_empty() {
        return;
    }
    // Each frame's phrase is its own entity, and the per-stream clamp bounds only
    // that stream — rodio sums live streams unlimited. A note rings ~7τ, so a fast
    // code stacks several; attenuate new phrases by the count still sounding to
    // keep the device sum out of hard clipping.
    let atten = 1.0 / (1.0 + 0.3 * live.iter().count() as f32);
    for n in &mut notes {
        n.gain *= atten;
    }
    commands.spawn((
        AudioPlayer(assets.add(Phrase(notes))),
        PlaybackSettings::DESPAWN,
    ));
}

/// Wired by [`crate::chord::install_chords`] — not called directly. The per-scheme
/// system installs per call; the shared parts once, keyed on `Assets<Phrase>` (what
/// `add_audio_source` actually provides — a surface may legitimately insert its own
/// [`InstrumentScheme`] before installing chords).
pub(crate) fn install<S: ControlScheme>(app: &mut App) {
    if !app.world().contains_resource::<Assets<Phrase>>() {
        app.add_audio_source::<Phrase>();
    }
    app.init_resource::<InstrumentScheme>();
    app.add_systems(Update, sound_chords::<S>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ChordDir::*;

    fn scheme() -> InstrumentScheme {
        InstrumentScheme::default()
    }

    /// Every pitch a press can produce is a member of the scale — the "no sour
    /// intervals" guarantee, over every path up to depth 3.
    #[test]
    fn presses_stay_on_the_scale() {
        let s = scheme();
        let dirs = [Up, Down, Left, Right];
        let mut paths: Vec<Vec<ChordDir>> = vec![vec![]];
        for _ in 0..3 {
            paths = paths
                .iter()
                .flat_map(|p| {
                    dirs.map(|d| {
                        let mut q = p.clone();
                        q.push(d);
                        q
                    })
                })
                .collect();
            for p in &paths {
                let (prefix, &last) = (&p[..p.len() - 1], p.last().unwrap());
                let f = (s.press)(&s.scale, prefix, last).freq_hz;
                let semis = 12.0 * (f / s.scale.root_hz).log2();
                let folded = (semis.round() as i32).rem_euclid(12) as u8;
                assert!(
                    (semis - semis.round()).abs() < 1e-3 && s.scale.degrees.contains(&folded),
                    "off-scale pitch {f} Hz ({semis} semis) for {p:?}"
                );
            }
        }
    }

    /// The mapping is state-driven, not a key→note table: the same direction sounds
    /// different pitches after different prefixes, and the same full path always
    /// sounds the same pitch.
    #[test]
    fn pitch_depends_on_path_not_key() {
        let s = scheme();
        let a = (s.press)(&s.scale, &[], Up);
        let b = (s.press)(&s.scale, &[Up], Up);
        assert_ne!(a.freq_hz, b.freq_hz, "prefix must matter");
        let again = (s.press)(&s.scale, &[Up], Up);
        assert_eq!(b, again, "same path must always sound the same");
    }

    /// Depth is audible: detune widens as the code deepens.
    #[test]
    fn detune_widens_with_depth() {
        let s = scheme();
        let shallow = (s.press)(&s.scale, &[], Up).detune_cents;
        let deep = (s.press)(&s.scale, &[Up, Down, Left, Right], Up).detune_cents;
        assert!(deep > shallow);
    }

    /// Accepted and unknown resolutions are distinct in shape AND direction:
    /// accepted rises to the octave, unknown falls out of scale.
    #[test]
    fn resolutions_are_distinct() {
        let s = scheme();
        let ok = (s.resolve)(&s.scale, &[Up, Left], true);
        let bad = (s.resolve)(&s.scale, &[Up, Up], false);
        assert_ne!(ok.len(), bad.len());
        assert!(ok.last().unwrap().freq_hz > ok[0].freq_hz, "accepted rises");
        assert!(
            bad.last().unwrap().freq_hz < bad[0].freq_hz,
            "unknown falls"
        );
        assert!(
            (ok.last().unwrap().freq_hz / s.scale.root_hz - 2.0).abs() < 1e-3,
            "accepted lands the octave"
        );
    }

    /// The synth terminates, stays clamped, and actually sounds.
    #[test]
    fn phrase_stream_is_finite_bounded_and_audible() {
        let s = scheme();
        let notes = (s.resolve)(&s.scale, &[Up], true);
        let mut stream = PhraseStream::new(&notes);
        let expected = stream.end;
        let samples: Vec<f32> = stream.by_ref().collect();
        assert_eq!(samples.len() as u64, expected);
        assert!(samples.len() < 10 * SAMPLE_RATE as usize, "runaway tail");
        let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak <= 1.0);
        assert!(peak > 0.05, "inaudible phrase, peak {peak}");
        // The tail has decayed — no click at the hard stop.
        let tail = samples[samples.len() - 100..]
            .iter()
            .fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(tail < 0.01, "tail still hot at cutoff: {tail}");
    }

    /// Not a test — the rl#359 evidence generator: renders a real chord entry
    /// (every press through the live scheme, then the resolution) to WAV, at frame
    /// timings matching the fp-screenshot chord script, for muxing with the frame
    /// capture. `DPAD_EVIDENCE_DIR=docs/evidence/rl359 cargo test -p crab-world
    /// --features render dpad_evidence -- --ignored`
    #[test]
    #[ignore = "artifact generator, not a check"]
    fn dpad_evidence() {
        let Some(dir) = std::env::var_os("DPAD_EVIDENCE_DIR") else {
            return;
        };
        let s = scheme();
        // (name, taps at 60 fps frame numbers, release frame, registered?).
        // `v^^^` = GroundNightBloom (a real GCR code); `>>>` is unregistered.
        type Clip = (&'static str, &'static [(u64, ChordDir)], u64, bool);
        let clips: [Clip; 2] = [
            (
                "code-accepted-bloom",
                &[(60, Down), (105, Up), (150, Up), (195, Up)],
                255,
                true,
            ),
            (
                "code-unknown",
                &[(60, Right), (105, Right), (150, Right)],
                210,
                false,
            ),
        ];
        std::fs::create_dir_all(&dir).unwrap();
        for (name, taps, release_frame, registered) in clips {
            let total_s = release_frame as f32 / 60.0 + 2.5;
            let n = (total_s * SAMPLE_RATE as f32) as usize;
            let mut mix = vec![0.0f32; n];
            let mut render_at = |frame: u64, notes: Vec<NoteSpec>| {
                let base = (frame as f32 / 60.0 * SAMPLE_RATE as f32) as usize;
                for (i, v) in PhraseStream::new(&notes).enumerate() {
                    if let Some(slot) = mix.get_mut(base + i) {
                        *slot += v;
                    }
                }
            };
            let mut path: Vec<ChordDir> = Vec::new();
            for &(frame, d) in taps {
                render_at(
                    frame,
                    frame_phrase(&s, [(path.clone(), d)].into_iter(), None),
                );
                path.push(d);
            }
            render_at(
                release_frame,
                frame_phrase(
                    &s,
                    std::iter::empty::<(Vec<ChordDir>, ChordDir)>(),
                    Some((path.as_slice(), registered)),
                ),
            );
            let pcm: Vec<i16> = mix
                .iter()
                .map(|v| (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .collect();
            let path = std::path::Path::new(&dir).join(format!("{name}.wav"));
            std::fs::write(&path, crate::wav::wav_bytes(&pcm)).unwrap();
        }
    }
}
