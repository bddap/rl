//! The d-pad as instrument (rl#359): every tap the combo system consumes sounds a
//! note, so entering a code IS playing a melody — each item's code becomes a tune,
//! recruiting musical memory for code recall. Installed by
//! [`crate::chord::install_chords`], so every chord surface (GCR, the demo, the
//! offscreen evidence apps) sounds through the one path.
//!
//! The press→sound mapping is DATA, not code (owner amendment on rl#359): a scheme is
//! a pure `(scale, code-prefix, press) → NoteSpec` function plus a [`Scale`], held in
//! the [`InstrumentScheme`] resource — a replacement mapping swaps in by replacing the
//! resource, never by touching the capture or input code.
//!
//! THE scheme is **heldbreath** on **A hirajoshi** — the owner's pick from the design
//! exploration (rl#359, Telegram 2026-08-12: "held breath hirajoshi"). Entering a
//! code is inhaling; completing it is the exhale. Presses are intervals relative to
//! the melody so far (a code is a remembered *contour*), depth is a felt tension
//! gradient, and the first press shades the whole phrase's brightness (the playground
//! realizes heldbreath's timbre regions as brightness shades — that is the sound the
//! owner picked by ear, so that is what ships). The depth gradient follows the owner's
//! deep-code spec (rl#380 comment, 2026-08-13): note 0 is PURE; twin-voice detune
//! ramps as `min(1, sqrt(s/8))` of max; bitcrush is silent until note 4, pops in at
//! 25% intensity, and reaches full at note 12.
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

/// A hirajoshi on A2 (A B C E F) — the owner's by-ear pick on the web playground.
/// Rooted an octave below the playground's A3 because heldbreath voices its melody
/// one octave down (dark register); the walk starts an octave above this root, so
/// the first note of a code lands near A3.
pub const HIRAJOSHI: Scale = Scale {
    root_hz: 110.0,
    degrees: &[0, 2, 3, 7, 8],
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
    /// 0..1 bitcrush intensity: sample-rate decimation + amplitude quantization.
    /// 0 = clean; the deep-code tension layer (owner spec on rl#380).
    pub crush: f32,
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
            scale: HIRAJOSHI,
            press: heldbreath_press,
            resolve: heldbreath_resolve,
        }
    }
}

/// Press = motion, not a key: an interval RELATIVE to where the melody stands, so a
/// code is a contour (rise, fall, leap) and two paths to the same node are two
/// different tunes.
fn heldbreath_step(d: ChordDir) -> i32 {
    match d {
        ChordDir::Up => 2,
        ChordDir::Right => 1,
        ChordDir::Left => -1,
        ChordDir::Down => -2,
    }
}

/// Fold the path to the melody's current degree. Starts one octave above the root
/// (mid register); each step clamps to ~3 octaves, so an uncapped code (rl#380)
/// pins at the edge instead of walking off the piano.
fn heldbreath_degree(scale: &Scale, path: impl IntoIterator<Item = ChordDir>) -> i32 {
    let n = scale.degrees.len() as i32;
    let mut deg = n;
    for d in path {
        deg = (deg + heldbreath_step(d)).clamp(0, 3 * n - 1);
    }
    deg
}

/// Region = first press of a code: it shades the whole phrase's brightness
/// (L dark, D warm, R hollow-ish, U glassy) — the playground's realization of
/// heldbreath's timbre families, which is the sound the owner picked.
fn region_brightness(first: ChordDir) -> f32 {
    match first {
        ChordDir::Left => 0.3,
        ChordDir::Down => 0.55,
        ChordDir::Right => 0.4,
        ChordDir::Up => 0.75,
    }
}

/// The owner's deep-code tension curves (rl#380 spec, 2026-08-13), by note index
/// `s` (0-based): detune ramps immediately, bitcrush enters discontinuously at 25%
/// on note 4. Both hold at full from their end of ramp.
fn tension_detune(s: usize) -> f32 {
    (s as f32 / 8.0).sqrt().min(1.0)
}
fn tension_crush(s: usize) -> f32 {
    if s < 4 {
        0.0
    } else {
        (0.25 + 0.75 * (s - 4) as f32 / 8.0).min(1.0)
    }
}

/// Full twin-voice spread at max tension. ±24¢ beats at ~6 Hz in the walk's
/// register — unmistakable, still pitched.
const MAX_DETUNE_CENTS: f32 = 48.0;

fn heldbreath_press(scale: &Scale, prefix: &[ChordDir], press: ChordDir) -> NoteSpec {
    let depth = prefix.len();
    let first = prefix.first().copied().unwrap_or(press);
    let deg = heldbreath_degree(scale, prefix.iter().copied().chain([press]));
    NoteSpec {
        onset_s: 0.0,
        freq_hz: scale.freq(deg),
        detune_cents: MAX_DETUNE_CENTS * tension_detune(depth),
        // Breath tightens with depth. Playground decay is time-to-silence; τ rings
        // ~7× itself here, hence the /7.
        decay_s: (1.1 - 0.08 * depth as f32).max(0.4) / 7.0,
        brightness: (region_brightness(first) - 0.06 * depth as f32).max(0.15),
        crush: tension_crush(depth),
        gain: 0.8,
    }
}

/// Completion = cadence. Accepted is the EXHALE, and per the owner amendment
/// (2026-08-12: "confirmation chord … derived from the notes so far, not a single
/// chord for all combos") it is spelled from the code's own melody: the distinct
/// degrees the walk visited (most recent four, in visit order) strummed as a pure
/// chord — detune and crush collapse to zero — landing longest and loudest on the
/// home octave. Every code exhales into its own chord, and every exhale resolves
/// home. Unknown: a deceptive cadence a half-step off home (deliberately the one
/// out-of-scale interval the instrument can make), KEEPING the accumulated
/// detune+crush and damping early — the breath is not released.
fn heldbreath_resolve(scale: &Scale, code: &[ChordDir], accepted: bool) -> Vec<NoteSpec> {
    let n = scale.degrees.len() as i32;
    let home = n; // one octave above the root: where the walk begins
    if accepted {
        let mut deg = n;
        let mut visited: Vec<i32> = Vec::new();
        for &d in code {
            deg = (deg + heldbreath_step(d)).clamp(0, 3 * n - 1);
            if !visited.contains(&deg) {
                visited.push(deg);
            }
        }
        visited.retain(|&d| d != home);
        let tail_start = visited.len().saturating_sub(4);
        let strum = &visited[tail_start..];
        let mut notes: Vec<NoteSpec> = strum
            .iter()
            .enumerate()
            .map(|(i, &d)| NoteSpec {
                onset_s: 0.09 * i as f32,
                freq_hz: scale.freq(d),
                detune_cents: 0.0,
                decay_s: 0.28,
                brightness: 0.7,
                crush: 0.0,
                gain: 0.7,
            })
            .collect();
        notes.push(NoteSpec {
            onset_s: 0.09 * strum.len() as f32,
            freq_hz: scale.freq(home),
            detune_cents: 0.0,
            decay_s: 0.45,
            brightness: 0.85,
            crush: 0.0,
            gain: 1.0,
        });
        notes
    } else {
        let depth = code.len();
        let (detune, crush) = (
            MAX_DETUNE_CENTS * tension_detune(depth),
            tension_crush(depth),
        );
        let home_hz = scale.freq(home);
        vec![
            NoteSpec {
                onset_s: 0.0,
                freq_hz: home_hz * (1.0 / 12.0f32).exp2(),
                detune_cents: detune,
                decay_s: 0.16,
                brightness: 0.35,
                crush,
                gain: 0.9,
            },
            NoteSpec {
                onset_s: 0.07,
                freq_hz: home_hz * (-5.0 / 12.0f32).exp2(),
                detune_cents: detune,
                decay_s: 0.19,
                brightness: 0.2,
                crush,
                gain: 0.75,
            },
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
    /// Bitcrush: sample-hold length in samples (1 = clean) and quantization levels.
    crush_hold: u32,
    crush_levels: f32,
    held: f32,
    hold_left: u32,
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
        // Crush intensity → decimation + quantization. The 25% entry point (the
        // owner-spec'd pop-in) holds every 3rd sample (~15 kHz) at ~10 bits —
        // clearly audible grit; full crush is ~3.7 kHz at 4 bits.
        let (crush_hold, crush_levels) = if spec.crush > 0.0 {
            let bits = 12.0 - 8.0 * spec.crush;
            (1 + (spec.crush * 11.0) as u32, (bits - 1.0).exp2())
        } else {
            (1, 0.0)
        };
        Self {
            start,
            // Ring ~7τ of the slowest partial, then the envelope is < 1e-3.
            end: start + (spec.decay_s * 7.0 * sr) as u64,
            partials,
            attack_samples: 0.003 * sr,
            gain: spec.gain,
            crush_hold,
            crush_levels,
            held: 0.0,
            hold_left: 0,
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
        let s = s * attack * self.gain;
        if self.crush_hold <= 1 {
            return s;
        }
        if self.hold_left == 0 {
            self.held = (s * self.crush_levels).round() / self.crush_levels;
            self.hold_left = self.crush_hold;
        }
        self.hold_left -= 1;
        self.held
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

    /// The owner's deep-code tension spec (rl#380): note 0 PURE; detune at 50% of
    /// max on note 2, full by note 8; crush zero through note 3, 25% on note 4,
    /// full by note 12 — and both hold, never recede.
    #[test]
    fn tension_follows_the_owner_curve() {
        let s = scheme();
        let at = |depth: usize| {
            let prefix: Vec<ChordDir> = std::iter::repeat_n(Up, depth).collect();
            (s.press)(&s.scale, &prefix, Down)
        };
        let n0 = at(0);
        assert_eq!((n0.detune_cents, n0.crush), (0.0, 0.0), "note 0 is pure");
        assert!((at(2).detune_cents / MAX_DETUNE_CENTS - 0.5).abs() < 1e-3);
        assert_eq!(at(8).detune_cents, MAX_DETUNE_CENTS);
        assert_eq!(at(20).detune_cents, MAX_DETUNE_CENTS, "detune holds");
        assert_eq!(at(3).crush, 0.0, "crush silent before note 4");
        assert!((at(4).crush - 0.25).abs() < 1e-3, "crush pops in at 25%");
        assert_eq!(at(12).crush, 1.0);
        assert_eq!(at(20).crush, 1.0, "crush holds");
    }

    /// The accepted chord is DERIVED from the code's own melody (owner amendment
    /// 2026-08-12): different codes exhale into different chords, every chord is
    /// pure (no detune, no crush), and every one lands home on the octave.
    #[test]
    fn accepted_chord_derives_from_the_melody() {
        let s = scheme();
        let chord = |code: &[ChordDir]| (s.resolve)(&s.scale, code, true);
        let a = chord(&[Up, Up, Down]);
        let b = chord(&[Down, Down, Up]);
        let pitches = |c: &[NoteSpec]| {
            c.iter()
                .map(|n| n.freq_hz.round() as i32)
                .collect::<Vec<_>>()
        };
        assert_ne!(pitches(&a), pitches(&b), "chord must vary with the code");
        for c in [&a, &b] {
            assert!(c.iter().all(|n| n.detune_cents == 0.0 && n.crush == 0.0));
            assert!(
                (c.last().unwrap().freq_hz / s.scale.root_hz - 2.0).abs() < 1e-3,
                "exhale lands the home octave"
            );
        }
    }

    /// Unknown keeps the held breath: half-step-off pitch (the one out-of-scale
    /// interval), accumulated detune+crush retained, damped short.
    #[test]
    fn unknown_cadence_keeps_the_tension() {
        let s = scheme();
        let code = [Up, Up, Up, Up, Up];
        let bad = (s.resolve)(&s.scale, &code, false);
        assert!(bad[0].detune_cents > 0.0 && bad[0].crush > 0.0);
        assert!(bad.iter().all(|n| n.decay_s < 0.25), "damped, unreleased");
        let semis = 12.0 * (bad[0].freq_hz / s.scale.root_hz).log2();
        let folded = (semis.round() as i32).rem_euclid(12) as u8;
        assert!(!s.scale.degrees.contains(&folded), "lands OFF the scale");
    }

    /// The synth terminates, stays clamped, and actually sounds.
    #[test]
    fn phrase_stream_is_finite_bounded_and_audible() {
        let s = scheme();
        // A deep (fully crushed+detuned) press plus a cadence: the synth's whole
        // parameter range in one phrase.
        let deep: Vec<ChordDir> = std::iter::repeat_n(Down, 12).collect();
        let mut notes = vec![(s.press)(&s.scale, &deep, Up)];
        notes.extend((s.resolve)(&s.scale, &[Up], true));
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
