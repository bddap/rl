//! Speed-driven wind audio (rl#356) — pure client-side presentation off the shared
//! sim state (SP == MP: nothing here feeds back into the sim or the wire).
//!
//! The wind is PROCEDURAL — white noise shaped per context — rather than a looped
//! asset: the loudness/brightness must track a continuously varying airspeed, and
//! filtered noise modulates without the loop seams, pitch-shift artifacts, or a
//! binary-asset pipeline a recorded gust bed would need. One rodio source runs for
//! the whole round; the ECS side writes TARGET parameters (gain, lowpass cutoff,
//! whistle band, gust depth) into shared atomics each frame, and the audio thread
//! glides toward them per-sample, so context switches (board/exit a craft) crossfade
//! by construction instead of clicking.
//!
//! Timbres, per [`LocalVehicle`] context:
//! - on foot: gusty broadband rush — wide amplitude wobble, no whistle;
//! - plane: bright airstream with a resonant whistle whose pitch rises with speed;
//! - ship: low-pass rumble, a slow throb, no whistle.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bevy::audio::{AddAudioSource, Decodable, PlaybackSettings};
use bevy::prelude::*;

use super::driver::{GameState, LocalVehicle};
use crate::sim::{TICK_HZ, UNIT};
use crab_world::vehicle::VehicleKind;

/// The shared "external world" mix bus (rl#359): every diegetic outdoor sound routes
/// its parameters through this resource instead of straight to the master mix, so
/// one writer ([`drive_muffle`]: combo entry ducks + muffles the outside world while
/// the chord modifier is held) can shape them all. `gain` multiplies a member's
/// amplitude; `lowpass_hz` caps its lowpass cutoff. Defaults are transparent.
#[derive(Resource, Clone, Copy)]
pub struct ExternalBus {
    pub gain: f32,
    pub lowpass_hz: f32,
}

impl Default for ExternalBus {
    fn default() -> Self {
        Self {
            gain: 1.0,
            lowpass_hz: 20_000.0,
        }
    }
}

/// A bus member's muffle policy, declared per sound: `Duck` follows the bus,
/// `Exempt` bleeds through untouched. Nothing is exempted yet — the flag exists so a
/// specific sound can later be sent through a code-entry melody as artistic flair
/// without restructuring the bus (owner spec, rl#359).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Muffle {
    Duck,
    Exempt,
}

impl ExternalBus {
    /// Shape one member's (gain, lowpass cutoff) by its muffle policy.
    pub fn shape(&self, gain: f32, lowpass_hz: f32, muffle: Muffle) -> (f32, f32) {
        match muffle {
            Muffle::Duck => (gain * self.gain, lowpass_hz.min(self.lowpass_hz)),
            Muffle::Exempt => (gain, lowpass_hz),
        }
    }
}

/// The muffled world while a code is being typed: quiet and behind a closed window,
/// but present — the instrument sits alone on top, and the return to transparency on
/// release is itself the "exhale". Cutoffs glide at the members' 60 ms bus rate, so
/// the duck lands as one mix move.
const MUFFLE_GAIN: f32 = 0.22;
const MUFFLE_LOWPASS_HZ: f32 = 650.0;

/// Bus writer (the ONE writer): while chord entry is live, duck + low-pass every
/// non-exempt external sound so the d-pad instrument never competes with the world.
/// `typing`, not `capturing`: the release frame still counts as entry, so the world
/// swells back only after the resolution sound has started.
pub(super) fn drive_muffle(
    chords: Res<crab_world::chord::Chords<crate::controls::GcrControls>>,
    mut bus: ResMut<ExternalBus>,
) {
    *bus = if chords.typing() {
        ExternalBus {
            gain: MUFFLE_GAIN,
            lowpass_hz: MUFFLE_LOWPASS_HZ,
        }
    } else {
        ExternalBus::default()
    };
}

pub(super) use crab_world::wav::SAMPLE_RATE;

/// The wind's loudness ceiling — the loudest thing in the outdoor mix; the
/// ambience beds key their levels under it (ambience.rs).
pub(super) const WIND_MASTER: f32 = 0.45;

/// Wind's bus policy, declared like the ambience layers declare theirs (one style
/// of membership across the bus): ordinary outdoor audio, ducks under code entry.
const WIND_MUFFLE: Muffle = Muffle::Duck;

/// An f32 bit-stored in an `AtomicU32`: the lock-free handshake every audio
/// target struct uses between the frame-rate ECS writer and the audio thread.
pub(super) struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub(super) fn new(v: f32) -> Self {
        Self(AtomicU32::new(v.to_bits()))
    }
    pub(super) fn store(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed);
    }
    pub(super) fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Full-wind airspeed, m/s — the plane's full-throttle terminal (~4.5 m/s, the
/// fastest thing in the game; see `PLANE` in crab-world's vehicle.rs). At or above
/// it the wind is at full roar in every context. The plane engine layer
/// (ambience.rs) tops its band here too.
pub(super) const FULL_WIND_MPS: f32 = 4.5;

/// The plane airstream's level under [`WIND_MASTER`] — the ONE plane-wind knob.
/// In the cockpit the wind sits UNDER the engine and the speed roar, not on top
/// of the mix (owner acceptance call, rl#357); body and whistle both ride the
/// gain this scales, so it moves the whole airstream together.
pub(super) const PLANE_WIND_LEVEL: f32 = 0.5;

/// The ship's audible-band ceiling, m/s — it never nears plane speeds (sustained
/// top ~2.5, a dive slightly past), so its wind and thruster layers normalize
/// over this lower band to keep cruising present.
pub(super) const SHIP_TOP_MPS: f32 = 3.0;

/// One target parameter set, f32s bit-stored in atomics — the frame-rate ECS writer
/// and the audio-thread reader share this with no lock on the audio path.
#[derive(Default)]
pub(super) struct WindTargets([AtomicF32; 5]);

impl WindTargets {
    /// `[gain, cutoff_hz, whistle_hz, whistle_gain, gust_depth]` — the array
    /// [`profile`] produces, stored whole so callers can't transpose a pair.
    pub(super) fn store(&self, params: [f32; 5]) {
        for (slot, v) in self.0.iter().zip(params) {
            slot.store(v);
        }
    }

    fn load(&self) -> [f32; 5] {
        [&self.0[0], &self.0[1], &self.0[2], &self.0[3], &self.0[4]].map(|a| a.load())
    }
}

/// The ECS-side handle to the running wind synth's targets. Lives for the app;
/// the playing entity (and thus the audio thread's stream) only spans a round.
#[derive(Resource, Clone)]
pub(super) struct WindChannel(Arc<WindTargets>);

/// The wind synth as a bevy audio asset: `decoder()` hands the audio thread a fresh
/// infinite stream sharing the channel's target atomics.
#[derive(Asset, TypePath)]
pub(super) struct WindNoise(Arc<WindTargets>);

impl Decodable for WindNoise {
    type DecoderItem = f32;
    type Decoder = WindStream;
    fn decoder(&self) -> WindStream {
        WindStream::new(self.0.clone())
    }
}

/// Marks the one wind-playing entity of the live round.
#[derive(Component)]
pub(super) struct WindAudio;

pub(super) fn install(app: &mut App) {
    app.add_audio_source::<WindNoise>();
    app.init_resource::<ExternalBus>();
    app.insert_resource(WindChannel(Arc::new(WindTargets::default())));
}

pub(super) fn spawn_wind(
    mut commands: Commands,
    mut assets: ResMut<Assets<WindNoise>>,
    channel: Res<WindChannel>,
) {
    // Start the round silent regardless of how the last one ended.
    channel.0.store([0.0, 400.0, 0.0, 0.0, 0.0]);
    commands.spawn((
        WindAudio,
        AudioPlayer(assets.add(WindNoise(channel.0.clone()))),
        // The stream is infinite; Once just means "no restart logic".
        PlaybackSettings::ONCE,
    ));
}

pub(super) fn despawn_wind(
    mut commands: Commands,
    wind: Query<Entity, With<WindAudio>>,
    channel: Res<WindChannel>,
) {
    channel.0.store([0.0, 400.0, 0.0, 0.0, 0.0]);
    for e in &wind {
        commands.entity(e).despawn();
    }
}

/// The wind's airspeed, m/s — all three axes on foot (falling roars, rl#355).
/// The ambience bus projects its own GAIT speed instead (horizontal on foot):
/// same craft arm, different on-foot physics, deliberately not shared.
pub(super) fn local_speed_mps(state: &GameState, vehicle: &LocalVehicle) -> f32 {
    match vehicle {
        // The walker's sim velocity is per-tick grid units, all three axes (falling
        // counts — rl#355 airborne integration).
        LocalVehicle::OnFoot => state
            .client
            .sim()
            .player(state.client.me())
            .map(|p| {
                let v = p.vel();
                let (x, y, z) = (v.x as f32, v.y as f32, v.z as f32);
                (x * x + y * y + z * z).sqrt() / UNIT as f32 * TICK_HZ as f32
            })
            .unwrap_or(0.0),
        // The craft's pose window is the same tick-stamped stream the cockpit
        // renders from; empty (engage grace / teleport reset) reads as still air.
        LocalVehicle::Flying { poses, .. } => poses.speed_mps().unwrap_or(0.0),
    }
}

/// Per-frame: local airspeed → timbre targets, shaped by the external bus.
pub(super) fn drive_wind(
    state: NonSend<GameState>,
    vehicle: Res<LocalVehicle>,
    bus: Res<ExternalBus>,
    channel: Res<WindChannel>,
) {
    let speed = local_speed_mps(&state, &vehicle);
    let [gain, cutoff, whistle_hz, whistle_gain, gust] = profile(vehicle.kind(), speed);
    let (gain, cutoff) = bus.shape(gain, cutoff, WIND_MUFFLE);
    channel
        .0
        .store([gain, cutoff, whistle_hz, whistle_gain, gust]);
}

/// Timbre map: context + airspeed → `[gain, cutoff_hz, whistle_hz, whistle_gain,
/// gust_depth]`. Each context normalizes speed over its own audible band — quiet
/// floor to [`FULL_WIND_MPS`] — then shapes loudness and brightness from that.
pub(super) fn profile(kind: Option<VehicleKind>, speed_mps: f32) -> [f32; 5] {
    // Loudness ceiling: leave headroom under the rest of the mix.
    const MASTER: f32 = WIND_MASTER;
    // On foot the floor sits above sprint (~0.25 m/s): walking and sprinting stay
    // near-silent, a fall or a claw launch is what roars (owner spec: "hear wind
    // when I'm going fast — falling or flying").
    let sprint_mps = crate::sim::SPRINT_SPEED as f32 / UNIT as f32 * TICK_HZ as f32;
    let (floor, ceil) = match kind {
        None => (sprint_mps, FULL_WIND_MPS),
        // The plane is audible from just above slow flight; full roar at terminal.
        Some(VehicleKind::Plane) => (0.5, FULL_WIND_MPS),
        // The ship never nears plane speeds — its band tops out lower so cruising
        // still has presence.
        Some(VehicleKind::Ship) => (0.2, SHIP_TOP_MPS),
    };
    let n = ((speed_mps - floor) / (ceil - floor)).clamp(0.0, 1.0);
    match kind {
        None => [MASTER * n.powf(1.4), 300.0 + 2200.0 * n, 0.0, 0.0, 0.5],
        Some(VehicleKind::Plane) => [
            MASTER * PLANE_WIND_LEVEL * n.powf(1.3),
            600.0 + 3400.0 * n,
            900.0 + 2100.0 * n,
            0.4 * n * n,
            0.15,
        ],
        Some(VehicleKind::Ship) => [
            MASTER * 1.2 * n.powf(1.3),
            130.0 + 570.0 * n,
            0.0,
            0.0,
            0.35,
        ],
    }
}

/// The audio-thread synth: white noise → 2×one-pole lowpass (the body) + a
/// state-variable bandpass (the whistle), amplitude-wobbled by a slow gust LFO.
/// All five parameters glide toward the shared targets so ECS-rate writes never
/// zipper or click.
pub(super) struct WindStream {
    targets: Arc<WindTargets>,
    /// Smoothed [gain, cutoff_hz, whistle_hz, whistle_gain, gust_depth].
    params: [f32; 5],
    /// Lowpass coefficients for body/gust, refreshed each [`Self::BLOCK`].
    body_a: f32,
    whistle_f: f32,
    rng: u32,
    lp1: f32,
    lp2: f32,
    /// Chamberlin SVF state for the whistle band.
    svf_low: f32,
    svf_band: f32,
    /// Gust LFO: a target level re-rolled every ~0.4 s, glided toward.
    gust_level: f32,
    gust_target: f32,
    gust_countdown: u32,
    block_left: u32,
}

impl WindStream {
    /// Samples between target reloads / coefficient refreshes (~1.5 ms) — cheap
    /// enough to track frame-rate writes, coarse enough to keep `exp`/`sin` off the
    /// per-sample path.
    const BLOCK: u32 = 64;
    /// Per-sample one-pole toward targets: τ ≈ 60 ms at 44.1 kHz.
    const GLIDE: f32 = 1.0 / (0.060 * SAMPLE_RATE as f32);

    pub(super) fn new(targets: Arc<WindTargets>) -> Self {
        Self {
            targets,
            params: [0.0, 400.0, 0.0, 0.0, 0.0],
            body_a: 0.05,
            whistle_f: 0.1,
            rng: 0x9E37_79B9,
            lp1: 0.0,
            lp2: 0.0,
            svf_low: 0.0,
            svf_band: 0.0,
            gust_level: 0.0,
            gust_target: 0.0,
            gust_countdown: 0,
            block_left: 0,
        }
    }

    fn white(&mut self) -> f32 {
        // xorshift32 — audio-grade noise, no rand dependency on the audio thread.
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn refresh_block(&mut self) {
        self.block_left = Self::BLOCK;
        let [_, cutoff, whistle_hz, _, _] = self.params;
        let sr = SAMPLE_RATE as f32;
        self.body_a = 1.0 - (-std::f32::consts::TAU * cutoff / sr).exp();
        // SVF stays stable well past the whistle's 3 kHz ceiling at 44.1 kHz.
        self.whistle_f = 2.0 * (std::f32::consts::PI * whistle_hz / sr).sin();
    }
}

impl Iterator for WindStream {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.block_left == 0 {
            self.refresh_block();
        }
        self.block_left -= 1;

        let targets = self.targets.load();
        for (p, t) in self.params.iter_mut().zip(targets) {
            *p += (t - *p) * Self::GLIDE;
        }
        let [gain, _, _, whistle_gain, gust_depth] = self.params;

        let white = self.white();
        // Body: two cascaded one-poles ⇒ 12 dB/oct above the cutoff.
        self.lp1 += (white - self.lp1) * self.body_a;
        self.lp2 += (self.lp1 - self.lp2) * self.body_a;

        // Whistle: Chamberlin SVF bandpass, Q = 1/0.2 = 5.
        let high = white - self.svf_low - 0.2 * self.svf_band;
        self.svf_band += self.whistle_f * high;
        self.svf_low += self.whistle_f * self.svf_band;
        self.svf_band = self.svf_band.clamp(-4.0, 4.0);

        // Gust: glide toward a level re-rolled every 0.35–0.55 s.
        if self.gust_countdown == 0 {
            self.gust_target = self.white();
            self.gust_countdown =
                (0.35 * SAMPLE_RATE as f32) as u32 + (self.rng % (SAMPLE_RATE / 5));
        }
        self.gust_countdown -= 1;
        self.gust_level += (self.gust_target - self.gust_level) * (Self::GLIDE * 0.2);
        let gust = (1.0 + gust_depth * self.gust_level).max(0.15);

        // The lowpass body attenuates hard at low cutoffs — ×3 restores presence;
        // the clamp is the safety ceiling.
        let s = (self.lp2 * 3.0 + self.svf_band * whistle_gain) * gain * gust;
        Some(s.clamp(-1.0, 1.0))
    }
}

impl bevy::audio::Source for WindStream {
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
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(stream: &mut WindStream, samples: usize) -> f32 {
        let sum: f32 = stream.by_ref().take(samples).map(|s| s * s).sum();
        (sum / samples as f32).sqrt()
    }

    /// Render one second at a held context+speed, discarding a glide-in lead.
    fn settled_rms(kind: Option<VehicleKind>, speed: f32) -> f32 {
        let targets = Arc::new(WindTargets::default());
        targets.store(profile(kind, speed));
        let mut s = WindStream::new(targets);
        let _ = rms(&mut s, SAMPLE_RATE as usize / 2);
        rms(&mut s, SAMPLE_RATE as usize)
    }

    #[test]
    fn loudness_rises_with_speed_in_every_context() {
        for kind in [None, Some(VehicleKind::Plane), Some(VehicleKind::Ship)] {
            let quiet = settled_rms(kind, 0.1);
            let mid = settled_rms(kind, 2.0);
            let fast = settled_rms(kind, 4.5);
            assert!(
                quiet < mid && mid < fast,
                "{kind:?}: rms not monotone: {quiet} {mid} {fast}"
            );
        }
        // Walking pace is near-silent on foot.
        assert!(settled_rms(None, 0.15) < 0.01);
    }

    /// Timbres genuinely differ: at the same airspeed the ship's spectrum sits far
    /// darker than the plane's. Compared via zero-crossing rate — a cheap spectral
    /// centroid proxy needing no FFT.
    #[test]
    fn plane_brighter_than_ship() {
        let zcr = |kind| {
            let targets = Arc::new(WindTargets::default());
            targets.store(profile(kind, 3.0));
            let mut s = WindStream::new(targets);
            let lead = SAMPLE_RATE as usize / 2;
            let _ = s.by_ref().take(lead).count();
            let buf: Vec<f32> = s.by_ref().take(SAMPLE_RATE as usize).collect();
            buf.windows(2)
                .filter(|w| w[0].signum() != w[1].signum())
                .count()
        };
        let plane = zcr(Some(VehicleKind::Plane));
        let ship = zcr(Some(VehicleKind::Ship));
        assert!(
            plane > ship * 2,
            "plane zcr {plane} not clearly above ship {ship}"
        );
    }

    /// Not a test of the code — the evidence generator for rl#356: renders each
    /// context's 0→full speed ramp to WAV under `WIND_EVIDENCE_DIR` when that env
    /// var is set. `WIND_EVIDENCE_DIR=docs/evidence/rl356 cargo test -p net
    /// --features render wind_evidence -- --ignored`
    #[test]
    #[ignore = "artifact generator, not a check"]
    fn wind_evidence() {
        let Some(dir) = std::env::var_os("WIND_EVIDENCE_DIR") else {
            return;
        };
        std::fs::create_dir_all(&dir).unwrap();
        for (name, kind) in [
            ("wind-onfoot", None),
            ("wind-plane", Some(VehicleKind::Plane)),
            ("wind-ship", Some(VehicleKind::Ship)),
        ] {
            let targets = Arc::new(WindTargets::default());
            let mut s = WindStream::new(targets.clone());
            let secs = 8;
            let n = SAMPLE_RATE as usize * secs;
            let mut pcm = Vec::with_capacity(n);
            for i in 0..n {
                // Hold still for 1 s, ramp to full over 5 s, hold 2 s.
                let t = i as f32 / SAMPLE_RATE as f32;
                let speed = ((t - 1.0) / 5.0).clamp(0.0, 1.0) * FULL_WIND_MPS;
                targets.store(profile(kind, speed));
                pcm.push((s.next().unwrap() * i16::MAX as f32) as i16);
            }
            let path = std::path::Path::new(&dir).join(format!("{name}.wav"));
            std::fs::write(path, crab_world::wav::wav_bytes(&pcm)).unwrap();
        }
    }
}
