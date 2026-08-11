//! Layered ambient soundscape (rl#357 stage 1): a context bus driving looping
//! sampled beds that crossfade as the listener's context changes.
//!
//! [`AmbientContext`] is the bus: one system projects the shared sim/terrain state
//! (how grassy the ground under the listener is, height above ground, vehicle)
//! into it each frame; each layer's gain map turns the bus into a target level,
//! and the audio thread glides toward the targets per-sample — so a context
//! change (walk off the grass, take off, board a craft) crossfades by
//! construction, the wind synth's scheme (audio.rs) applied to sampled loops.
//! Stage 2 hangs more biome beds off the same bus; stage 3 adds vehicle/on-foot
//! layers.
//!
//! Beds are CC0 recordings fetched from bddap-bot/rl-assets
//! (scripts/fetch-ambience.sh; provenance in NOTICE) into the gitignored asset
//! dir. An absent bed is a silent layer plus one warning, never an error — the
//! sally.glb precedent — so plain checkouts and CI need no binaries.

use std::sync::Arc;

use bevy::audio::{AddAudioSource, Decodable, PlaybackSettings};
use bevy::prelude::*;

use super::audio::{AtomicF32, ExternalBus, SAMPLE_RATE, WIND_MASTER};
use super::driver::{GameState, LocalVehicle, RenderClock};
use crab_world::sky::smoothstep;
use crab_world::terrain::{Terrain, biome};
use crab_world::vehicle::VehicleKind;

/// The context bus: the listener's world context, written once per frame from
/// shared state, read by every ambient layer's gain map.
#[derive(Resource, Clone, Copy, Default)]
pub(super) struct AmbientContext {
    /// 0..1: how grassy the terrain under the listener is — crab-world's own tuft
    /// placement weight, so the beds live exactly where the tufts grow and thin
    /// out over scree, rock faces, and snow exactly where the ground art does.
    pub grass: f32,
    /// Listener height above the terrain, meters.
    pub alt_m: f32,
    /// Vehicle context (`None` = on foot).
    pub vehicle: Option<VehicleKind>,
}

struct LayerDef {
    /// Loop file under the asset root's `assets/` (see scripts/fetch-ambience.sh).
    file: &'static str,
    /// Full-context level, as a fraction of the wind's master ceiling so the
    /// beds sit under the loudest thing in the outdoor mix by construction.
    level: f32,
    /// Bus → 0..1 activation.
    gain: fn(&AmbientContext) -> f32,
}

/// Ground beds are heard on grassy terrain, fade out by ~60 m up (a low pass over
/// the meadow keeps a trace of it), and duck under a craft canopy.
fn grass_bed(ctx: &AmbientContext) -> f32 {
    let near_ground = 1.0 - smoothstep(12.0, 60.0, ctx.alt_m);
    let canopy = if ctx.vehicle.is_some() { 0.5 } else { 1.0 };
    ctx.grass * near_ground * canopy
}

/// Both beds are night sounds gated on biome only: the world is PERMANENTLY night
/// (a static `NightSkyPlugin` sky, one fixed moon-sun light — no time-of-day
/// exists anywhere in sim or render), so a "night" gate would be constant-true.
/// If a day/night cycle ever lands, it multiplies into [`grass_bed`].
const LAYERS: [LayerDef; 2] = [
    LayerDef {
        file: "ambience/grassland-birds.wav",
        level: 0.35 * WIND_MASTER,
        gain: grass_bed,
    },
    LayerDef {
        file: "ambience/crickets.wav",
        level: 0.5 * WIND_MASTER,
        gain: grass_bed,
    },
];

/// The one target formula: a layer's activation from the context — shared by the
/// live driver and the evidence generator, so the proof capture cannot drift from
/// the shipped path.
fn layer_target(def: &LayerDef, ctx: &AmbientContext) -> f32 {
    def.level * (def.gain)(ctx)
}

/// Per-layer activation targets plus the external-bus factors, kept SEPARATE so
/// they can glide at different rates: activations crossfade deliberately, while a
/// bus duck must land on beds and wind together (see the stream's glide consts).
struct AmbienceTargets {
    acts: [AtomicF32; LAYERS.len()],
    bus_gain: AtomicF32,
    bus_lowpass_hz: AtomicF32,
}

impl Default for AmbienceTargets {
    fn default() -> Self {
        Self {
            acts: Default::default(),
            bus_gain: AtomicF32::new(1.0),
            bus_lowpass_hz: AtomicF32::new(20_000.0),
        }
    }
}

impl AmbienceTargets {
    fn silence(&self) {
        for a in &self.acts {
            a.store(0.0);
        }
    }
}

/// The ECS-side handle: shared targets plus the decoded beds (loaded once at app
/// build; a bed that failed to load is an empty slice = a silent layer).
#[derive(Resource, Clone)]
pub(super) struct AmbienceChannel {
    targets: Arc<AmbienceTargets>,
    beds: [Arc<[f32]>; LAYERS.len()],
}

/// The bed mixer as a bevy audio asset: `decoder()` hands the audio thread a fresh
/// infinite stream sharing the channel's targets and beds.
#[derive(Asset, TypePath)]
pub(super) struct AmbienceMix(AmbienceChannel);

impl Decodable for AmbienceMix {
    type DecoderItem = f32;
    type Decoder = AmbienceStream;
    fn decoder(&self) -> AmbienceStream {
        AmbienceStream::new(self.0.targets.clone(), self.0.beds.clone())
    }
}

/// Marks the one ambience-playing entity of the live round.
#[derive(Component)]
pub(super) struct AmbienceAudio;

pub(super) fn install(app: &mut App) {
    app.add_audio_source::<AmbienceMix>();
    app.init_resource::<AmbientContext>();
    app.insert_resource(AmbienceChannel {
        targets: Arc::default(),
        beds: std::array::from_fn(|i| load_bed(&LAYERS[i])),
    });
}

fn load_bed(def: &LayerDef) -> Arc<[f32]> {
    let path = crab_world::assets::bevy_asset_path().join(def.file);
    match super::wav::read_mono_44k(&path) {
        Ok(mut pcm) => {
            seal_loop(&mut pcm);
            pcm.into()
        }
        Err(e) => {
            tracing::warn!(
                "ambient bed {} unavailable ({e}) — that layer stays silent (non-fatal). \
                 Fetch the CC0 beds with scripts/fetch-ambience.sh.",
                path.display()
            );
            Arc::from([])
        }
    }
}

/// Make any bed loop-clean by construction: equal-power-blend the last
/// half-second into the opening samples and drop the tail, so the wrap point is
/// seamless whatever the recording's ends look like.
fn seal_loop(pcm: &mut Vec<f32>) {
    let len = pcm.len();
    let n = ((SAMPLE_RATE / 2) as usize).min(len / 4);
    for i in 0..n {
        let t = i as f32 / n as f32;
        pcm[i] = pcm[i] * t.sqrt() + pcm[len - n + i] * (1.0 - t).sqrt();
    }
    pcm.truncate(len - n);
}

pub(super) fn spawn_ambience(
    mut commands: Commands,
    mut assets: ResMut<Assets<AmbienceMix>>,
    channel: Res<AmbienceChannel>,
) {
    // No beds fetched → nothing to play; the load-time warning already said why.
    if channel.beds.iter().all(|b| b.is_empty()) {
        return;
    }
    // Start the round silent regardless of how the last one ended.
    channel.targets.silence();
    commands.spawn((
        AmbienceAudio,
        AudioPlayer(assets.add(AmbienceMix(channel.clone()))),
        // The stream is infinite; Once just means "no restart logic".
        PlaybackSettings::ONCE,
    ));
}

pub(super) fn despawn_ambience(
    mut commands: Commands,
    ambience: Query<Entity, With<AmbienceAudio>>,
    channel: Res<AmbienceChannel>,
) {
    channel.targets.silence();
    for e in &ambience {
        commands.entity(e).despawn();
    }
}

/// Bus writer: project the shared sim/terrain state into [`AmbientContext`].
/// Runs after `apply_transforms`, so poses and the render clock are current.
pub(super) fn update_context(
    state: NonSend<GameState>,
    vehicle: Res<LocalVehicle>,
    clock: Res<RenderClock>,
    terrain: Res<Terrain>,
    mut ctx: ResMut<AmbientContext>,
) {
    // The walker's sim altitude is already height-above-terrain; the craft
    // reports absolute y, resolved against the ground height sampled below.
    enum Alt {
        AboveGround(f32),
        Absolute(f32),
    }
    // Absolute world xz for the terrain sampler (render-frame coordinates are
    // anchor-relative and must never touch it — rl#354).
    let (x, z, alt) = match &*vehicle {
        LocalVehicle::OnFoot => {
            let Some(p) = state.client.sim().player(state.client.me()) else {
                *ctx = AmbientContext::default();
                return;
            };
            let (x, z) = p.pos().to_meters_f64();
            (
                x,
                z,
                Alt::AboveGround(p.alt() as f32 / crate::sim::UNIT as f32),
            )
        }
        LocalVehicle::Flying { .. } => {
            let Some(pose) = vehicle.cockpit_sample(clock.tick, clock.frac) else {
                *ctx = AmbientContext::default();
                return;
            };
            (
                pose.pos.x as f64,
                pose.pos.z as f64,
                Alt::Absolute(pose.pos.y),
            )
        }
    };
    let h = terrain.height_f64(x, z) as f32;
    let alt_m = match alt {
        Alt::AboveGround(a) => a,
        Alt::Absolute(y) => y - h,
    };
    // Ground slope from central differences — plenty close to the mesh normal for
    // a gain weight.
    const D: f64 = 2.0;
    let gx = ((terrain.height_f64(x + D, z) - terrain.height_f64(x - D, z)) / (2.0 * D)) as f32;
    let gz = ((terrain.height_f64(x, z + D) - terrain.height_f64(x, z - D)) / (2.0 * D)) as f32;
    let normal_y = 1.0 / (1.0 + gx * gx + gz * gz).sqrt();
    *ctx = AmbientContext {
        grass: biome::tuft_weight(h, normal_y),
        alt_m,
        vehicle: vehicle.kind(),
    };
}

/// Bus reader: per-layer activation targets from the context, the external-bus
/// factors passed through untouched (they glide at their own rate downstream).
pub(super) fn drive_layers(
    ctx: Res<AmbientContext>,
    bus: Res<ExternalBus>,
    channel: Res<AmbienceChannel>,
) {
    for (def, slot) in LAYERS.iter().zip(&channel.targets.acts) {
        slot.store(layer_target(def, &ctx));
    }
    channel.targets.bus_gain.store(bus.gain);
    channel.targets.bus_lowpass_hz.store(bus.lowpass_hz);
}

/// The audio-thread mixer: each bed loops at its own cursor and rides its own
/// glided activation; the sum rides the glided bus gain, passes one one-pole
/// lowpass (the external bus's muffle hook), and a safety clamp.
pub(super) struct AmbienceStream {
    targets: Arc<AmbienceTargets>,
    beds: [Arc<[f32]>; LAYERS.len()],
    cursors: [usize; LAYERS.len()],
    /// Smoothed per-layer activations + bus gain + lowpass cutoff.
    acts: [f32; LAYERS.len()],
    bus_gain: f32,
    cutoff: f32,
    lp_a: f32,
    lp: f32,
    block_left: u32,
}

impl AmbienceStream {
    /// Samples between coefficient refreshes — same tradeoff as the wind's block.
    const BLOCK: u32 = 64;
    /// Activation glide: τ ≈ 0.4 s — a deliberate, audible crossfade (the wind's
    /// 60 ms would jump-cut an ambience bed).
    const GLIDE_ACT: f32 = 1.0 / (0.4 * SAMPLE_RATE as f32);
    /// Bus glide: the wind's 60 ms, so one combo-entry duck lands on the beds and
    /// the wind as a single mix move instead of two staggered ones.
    const GLIDE_BUS: f32 = 1.0 / (0.060 * SAMPLE_RATE as f32);

    fn new(targets: Arc<AmbienceTargets>, beds: [Arc<[f32]>; LAYERS.len()]) -> Self {
        Self {
            targets,
            beds,
            cursors: [0; LAYERS.len()],
            acts: [0.0; LAYERS.len()],
            bus_gain: 1.0,
            cutoff: 20_000.0,
            lp_a: 1.0,
            lp: 0.0,
            block_left: 0,
        }
    }

    fn refresh_block(&mut self) {
        self.block_left = Self::BLOCK;
        self.lp_a = 1.0 - (-std::f32::consts::TAU * self.cutoff / SAMPLE_RATE as f32).exp();
    }
}

impl Iterator for AmbienceStream {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.block_left == 0 {
            self.refresh_block();
        }
        self.block_left -= 1;

        self.bus_gain += (self.targets.bus_gain.load() - self.bus_gain) * Self::GLIDE_BUS;
        self.cutoff += (self.targets.bus_lowpass_hz.load() - self.cutoff) * Self::GLIDE_BUS;

        let mut s = 0.0;
        for i in 0..LAYERS.len() {
            self.acts[i] += (self.targets.acts[i].load() - self.acts[i]) * Self::GLIDE_ACT;
            let bed = &self.beds[i];
            if bed.is_empty() {
                continue;
            }
            s += bed[self.cursors[i]] * self.acts[i];
            self.cursors[i] = (self.cursors[i] + 1) % bed.len();
        }
        self.lp += (s * self.bus_gain - self.lp) * self.lp_a;
        Some(self.lp.clamp(-1.0, 1.0))
    }
}

impl bevy::audio::Source for AmbienceStream {
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

    #[test]
    fn grass_bed_follows_context() {
        let grassy = AmbientContext {
            grass: 1.0,
            alt_m: 0.0,
            vehicle: None,
        };
        assert_eq!(grass_bed(&grassy), 1.0);
        // Barren ground silences the bed; altitude fades it; a canopy halves it.
        assert_eq!(
            grass_bed(&AmbientContext {
                grass: 0.0,
                ..grassy
            }),
            0.0
        );
        assert_eq!(
            grass_bed(&AmbientContext {
                alt_m: 100.0,
                ..grassy
            }),
            0.0
        );
        let low_pass_over = grass_bed(&AmbientContext {
            alt_m: 30.0,
            vehicle: Some(VehicleKind::Plane),
            ..grassy
        });
        assert!(low_pass_over > 0.0 && low_pass_over < 0.5);
    }

    #[test]
    fn seal_loop_blends_tail_into_head() {
        // A 2 s ramp: after sealing, sample 0 is the pure tail (t=0 ⇒ head×0 +
        // tail×1) and the bed is shorter by the blend window.
        let len = 2 * SAMPLE_RATE as usize;
        let mut pcm: Vec<f32> = (0..len).map(|i| i as f32 / len as f32).collect();
        let n = (SAMPLE_RATE / 2) as usize;
        let tail_first = pcm[len - n];
        seal_loop(&mut pcm);
        assert_eq!(pcm.len(), len - n);
        assert!((pcm[0] - tail_first).abs() < 1e-4);
    }

    fn one_bed_stream(pcm: Vec<f32>) -> (Arc<AmbienceTargets>, AmbienceStream) {
        let targets = Arc::<AmbienceTargets>::default();
        let beds = [Arc::from(pcm), Arc::from([])];
        (targets.clone(), AmbienceStream::new(targets, beds))
    }

    #[test]
    fn mixer_settles_on_target_activation() {
        // A DC bed passes the (transparent-cutoff) lowpass whole, so the settled
        // output IS activation × sample.
        let (targets, mut s) = one_bed_stream(vec![0.5; 64]);
        targets.acts[0].store(0.8);
        let _ = s.by_ref().take(3 * SAMPLE_RATE as usize).count();
        let settled = s.next().unwrap();
        assert!((settled - 0.4).abs() < 0.01, "settled at {settled}");
    }

    #[test]
    fn activation_change_glides_not_jumps() {
        let (targets, mut s) = one_bed_stream(vec![0.5; 64]);
        targets.acts[0].store(0.8);
        let _ = s.by_ref().take(3 * SAMPLE_RATE as usize).count();
        targets.acts[0].store(0.0);
        // Mid-fade the signal is clearly alive, and by 3 s it is clearly gone.
        let mid = s.by_ref().nth(4096).unwrap();
        assert!(mid > 0.1, "jump-cut: {mid}");
        let _ = s.by_ref().take(3 * SAMPLE_RATE as usize).count();
        assert!(s.next().unwrap() < 1e-3);
    }

    #[test]
    fn bus_duck_lands_at_wind_rate() {
        // A bus duck settles in ~0.3 s (5τ at 60 ms) — the activation glide alone
        // would still be at ~47% after that long.
        let (targets, mut s) = one_bed_stream(vec![0.5; 64]);
        targets.acts[0].store(0.8);
        let _ = s.by_ref().take(3 * SAMPLE_RATE as usize).count();
        targets.bus_gain.store(0.0);
        let after = s.by_ref().nth((0.3 * SAMPLE_RATE as f32) as usize).unwrap();
        assert!(after < 0.02, "duck too slow: {after}");
    }

    /// Not a test — the evidence generator for rl#357 stage 1: renders a context
    /// sweep (grassland on foot → climb to rock → return → take off) through the
    /// real fetched beds to WAV. Skips (with a note) when the beds are absent.
    /// `AMBIENCE_EVIDENCE_DIR=docs/evidence/rl357-stage1 cargo test -p net
    /// --features render ambience_evidence -- --ignored`
    #[test]
    #[ignore = "artifact generator, not a check"]
    fn ambience_evidence() {
        let Some(dir) = std::env::var_os("AMBIENCE_EVIDENCE_DIR") else {
            return;
        };
        let beds: [Arc<[f32]>; LAYERS.len()] = std::array::from_fn(|i| load_bed(&LAYERS[i]));
        if beds.iter().any(|b| b.is_empty()) {
            eprintln!("beds absent — run scripts/fetch-ambience.sh first");
            return;
        }
        let targets = Arc::<AmbienceTargets>::default();
        let mut s = AmbienceStream::new(targets.clone(), beds);
        let secs = 20;
        let n = SAMPLE_RATE as usize * secs;
        let mut pcm = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE as f32;
            // 0–5 s deep grassland on foot; 5–10 s climb onto rock (grass → 0);
            // 10–13 s walk back down; 13–20 s board the plane and climb out.
            let (grass, alt_m, vehicle) = match t {
                t if t < 5.0 => (1.0, 0.0, None),
                t if t < 10.0 => (1.0 - (t - 5.0) / 5.0, 0.0, None),
                t if t < 13.0 => ((t - 10.0) / 3.0, 0.0, None),
                t => (1.0, (t - 13.0) * 12.0, Some(VehicleKind::Plane)),
            };
            let ctx = AmbientContext {
                grass,
                alt_m,
                vehicle,
            };
            for (def, slot) in LAYERS.iter().zip(&targets.acts) {
                slot.store(layer_target(def, &ctx));
            }
            pcm.push((s.next().unwrap() * i16::MAX as f32) as i16);
        }
        std::fs::create_dir_all(&dir).unwrap();
        let path = std::path::Path::new(&dir).join("ambience-context-sweep.wav");
        std::fs::write(path, super::super::wav::wav_bytes(&pcm)).unwrap();
    }
}
