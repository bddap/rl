//! Layered ambient soundscape (rl#357 stage 1): a context bus driving looping
//! sampled beds that crossfade as the listener's context changes.
//!
//! [`AmbientContext`] is the bus: one system projects the shared sim/terrain state
//! (how grassy the ground under the listener is, height above ground, vehicle)
//! into it each frame; each layer's gain map turns the bus into a target level,
//! and the audio thread glides toward the targets per-sample — so a context
//! change (walk off the grass, take off, board a craft) crossfades by
//! construction, the wind synth's scheme (audio.rs) applied to sampled loops.
//!
//! "Biome" is what the terrain actually has (rl#357 stage 2): elevation and
//! slope bands (crab-world's `biome` stops). The lush low greens double as the
//! moisture map — frogs sing where the valleys are green — and the mountain bed
//! lives on the complement of the grass, exactly where the tint paints rock and
//! scree. The listener's own sounds ride the same bus (rl#357 stage 3): movement
//! layers (footsteps everywhere walkable, a grass swish where the tufts grow)
//! keyed to gait, and per-craft engine loops keyed to boarding + speed.
//!
//! Beds are CC0 recordings fetched from bddap-bot/rl-assets
//! (scripts/fetch-ambience.sh; provenance in NOTICE) into the gitignored asset
//! dir. An absent bed is a silent layer plus one warning, never an error — the
//! sally.glb precedent — so plain checkouts and CI need no binaries.

use std::sync::Arc;

use bevy::audio::{AddAudioSource, Decodable, PlaybackSettings};
use bevy::prelude::*;

use super::audio::{AtomicF32, ExternalBus, Muffle, SAMPLE_RATE, WIND_MASTER};
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
    /// 0..1: how lush (green) the ground is — the complement of crab-world's
    /// grass-dryness ramp. High in the deep valley greens, zero by the dry-grass
    /// band; the terrain's moisture proxy, so it gates the wet-ground beds.
    pub lush: f32,
    /// 0..1: how rocky the ground is — crab-world's rock/scree weight (steep
    /// faces + the above-the-grass band, gone under snow).
    pub rocky: f32,
    /// Listener height above the terrain, meters.
    pub alt_m: f32,
    /// Listener speed, m/s, driving the movement and engine layers: horizontal
    /// gait on foot (vertical motion is the wind's business, not footsteps'),
    /// the craft pose window's speed when flying.
    pub speed_mps: f32,
    /// Vehicle context (`None` = on foot).
    pub vehicle: Option<VehicleKind>,
}

/// How a bed is made loop-clean at load: `Blend` equal-power-crossfades the tail
/// into the head — right for steady texture, but on transient material (footsteps)
/// it superimposes unrelated events into a double-hit at every wrap — so such an
/// asset is cut to start/end inside a between-events gap and `Splice`d untouched.
#[derive(Clone, Copy)]
enum Seal {
    Blend,
    Splice,
}

struct LayerDef {
    /// Loop file under the asset root's `assets/` (see scripts/fetch-ambience.sh).
    file: &'static str,
    /// Amplitude at full activation. The beds key off [`WIND_MASTER`] so the
    /// ambient background scales with the wind ceiling; the stage-3 layers are
    /// plain measured amplitudes (see the table comment) and may exceed 1 only
    /// while `level × activation × asset peak` stays under the mixer's clamp.
    level: f32,
    /// Bus → 0..1 activation.
    gain: fn(&AmbientContext) -> f32,
    /// Activation glide τ, seconds: beds and engines crossfade deliberately
    /// (0.4 s), gait-keyed layers must track the player's stops — a 0.4 s tail
    /// fires audible phantom steps after halting, so they release in 60 ms.
    glide_tau_s: f32,
    /// Loop-seal policy at load.
    seal: Seal,
    /// Per-sound external-bus policy (rl#359): an `Exempt` bed would bleed through
    /// the code-entry muffle. Nothing is exempted yet.
    muffle: Muffle,
}

/// Ground proximity shared by every ground bed: full on the ground, gone by
/// ~60 m up (a low pass over the meadow keeps a trace of it), halved under a
/// craft canopy.
fn near_ground(ctx: &AmbientContext) -> f32 {
    let near = 1.0 - smoothstep(12.0, 60.0, ctx.alt_m);
    let canopy = if ctx.vehicle.is_some() { 0.5 } else { 1.0 };
    near * canopy
}

/// Grassy-ground beds: anywhere the tufts grow.
fn grass_bed(ctx: &AmbientContext) -> f32 {
    ctx.grass * near_ground(ctx)
}

/// Wet-valley beds: grassy AND lush — the green low valleys, the closest thing
/// this terrain has to frog water.
fn valley_bed(ctx: &AmbientContext) -> f32 {
    ctx.grass * ctx.lush * near_ground(ctx)
}

/// Rock/scree beds: where the grass gives out — steep faces and the high band,
/// gone under snow cover (there the wind synth owns the mix alone).
fn mountain_bed(ctx: &AmbientContext) -> f32 {
    ctx.rocky * near_ground(ctx)
}

/// On-foot walking pace, m/s — the sim's tuned walk speed, the full-gain point of
/// the movement layers (sprint holds full, it does not get louder).
fn walk_mps() -> f32 {
    crate::sim::PLAYER_SPEED as f32 / crate::sim::UNIT as f32 * crate::sim::TICK_HZ as f32
}

/// Footsteps, any walkable ground: grounded (the jump apex is ~1.5 player
/// heights ≈ 0.08 m in this miniature world, so the fade sits inside it — steps
/// cut out mid-air) and moving, ramping in from a quarter of walking pace to
/// full at walk (sprint holds full, it does not get louder).
fn footsteps(ctx: &AmbientContext) -> f32 {
    if ctx.vehicle.is_some() {
        return 0.0;
    }
    let grounded = 1.0 - smoothstep(0.03, 0.08, ctx.alt_m);
    grounded * smoothstep(0.25 * walk_mps(), walk_mps(), ctx.speed_mps)
}

/// Grass swish: legs brushing through the tufts, so it lives exactly where the
/// tufts grow, on top of the neutral footsteps.
fn grass_swish(ctx: &AmbientContext) -> f32 {
    ctx.grass * footsteps(ctx)
}

/// An engine's presence the moment its craft is boarded — it idles audibly;
/// speed adds the rest of the band.
const ENGINE_IDLE: f32 = 0.6;

/// Engine loops: board/exit crossfades on the activation glide like every layer.
/// Each craft normalizes speed over its own audible band, the same ceilings its
/// wind profile uses (audio.rs).
fn engine(ctx: &AmbientContext, kind: VehicleKind, top_mps: f32) -> f32 {
    if ctx.vehicle != Some(kind) {
        return 0.0;
    }
    ENGINE_IDLE + (1.0 - ENGINE_IDLE) * (ctx.speed_mps / top_mps).clamp(0.0, 1.0)
}

/// Plane: the prop buzz, up to the full-throttle terminal.
fn plane_engine(ctx: &AmbientContext) -> f32 {
    engine(ctx, VehicleKind::Plane, super::audio::FULL_WIND_MPS)
}

/// Ship: the hover thrusters' hum, over the ship's own lower band.
fn ship_thruster(ctx: &AmbientContext) -> f32 {
    engine(ctx, VehicleKind::Ship, super::audio::SHIP_TOP_MPS)
}

/// All beds are gated on biome only, never time: the world is PERMANENTLY night
/// (a static `NightSkyPlugin` sky, one fixed moon-sun light — no time-of-day
/// exists anywhere in sim or render), so a "night" gate would be constant-true.
/// Deliberate crossfade for beds and engines; fast tracking for gait-keyed
/// layers (see [`LayerDef::glide_tau_s`]).
const BED_GLIDE_S: f32 = 0.4;
const GAIT_GLIDE_S: f32 = 0.06;

/// If a day/night cycle ever lands, it multiplies into the gain maps here.
///
/// Levels are the stage-3 deliberate mix pass, set from each asset's measured
/// integrated loudness (a scalar gain shifts LUFS by exactly 20·log10(gain)):
/// beds key off the wind ceiling as a band of background texture; movement
/// (steps −30.7 LUFS mastered, swish −26.8) lands a few dB above the bed stack
/// so the listener's own sounds read as foreground without drowning the world;
/// engines are mastered hot (−19.0 / −19.6 LUFS, loudnorm I=−16 — steady hums
/// take it) so full activation lands ~−18 LUFS, over the ship's ~−21 LUFS wind
/// rumble and the plane's cruise wind. The loudest SAMPLED layer is an engine;
/// the −8 LUFS full-throttle wind roar overtakes everything by design — speed
/// drowns the motor. Headroom is budgeted on the worst-case SUM: walking a lush
/// valley stacks four beds (Σ 0.72) + steps 0.7 + swish 0.35 ≈ 1.77 of level,
/// but the beds' instantaneous amplitude sits far under their ~0.84 true peaks,
/// so a step transient (0.7 × 0.84 ≈ 0.59) plus swish and beds stays clear of
/// the mixer's ±1 clamp except on vanishing coincident-peak alignments.
const LAYERS: [LayerDef; 9] = [
    LayerDef {
        file: "ambience/grassland-birds.wav",
        level: 0.35 * WIND_MASTER,
        gain: grass_bed,
        glide_tau_s: BED_GLIDE_S,
        seal: Seal::Blend,
        muffle: Muffle::Duck,
    },
    LayerDef {
        file: "ambience/crickets.wav",
        level: 0.5 * WIND_MASTER,
        gain: grass_bed,
        glide_tau_s: BED_GLIDE_S,
        seal: Seal::Blend,
        muffle: Muffle::Duck,
    },
    LayerDef {
        file: "ambience/valley-frogs.wav",
        level: 0.45 * WIND_MASTER,
        gain: valley_bed,
        glide_tau_s: BED_GLIDE_S,
        seal: Seal::Blend,
        muffle: Muffle::Duck,
    },
    LayerDef {
        file: "ambience/grass-rustle.wav",
        level: 0.3 * WIND_MASTER,
        gain: grass_bed,
        glide_tau_s: BED_GLIDE_S,
        seal: Seal::Blend,
        muffle: Muffle::Duck,
    },
    LayerDef {
        file: "ambience/mountain-birds.wav",
        level: 0.3 * WIND_MASTER,
        gain: mountain_bed,
        glide_tau_s: BED_GLIDE_S,
        seal: Seal::Blend,
        muffle: Muffle::Duck,
    },
    LayerDef {
        file: "ambience/footsteps.wav",
        level: 0.7,
        gain: footsteps,
        glide_tau_s: GAIT_GLIDE_S,
        seal: Seal::Splice,
        muffle: Muffle::Duck,
    },
    LayerDef {
        file: "ambience/grass-swish.wav",
        level: 0.35,
        gain: grass_swish,
        glide_tau_s: GAIT_GLIDE_S,
        seal: Seal::Blend,
        muffle: Muffle::Duck,
    },
    LayerDef {
        file: "ambience/plane-engine.wav",
        level: 1.15,
        gain: plane_engine,
        glide_tau_s: BED_GLIDE_S,
        seal: Seal::Blend,
        muffle: Muffle::Duck,
    },
    LayerDef {
        file: "ambience/ship-thruster.wav",
        level: 1.15,
        gain: ship_thruster,
        glide_tau_s: BED_GLIDE_S,
        seal: Seal::Blend,
        muffle: Muffle::Duck,
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
    match crab_world::wav::read_mono_44k(&path) {
        Ok(mut pcm) => {
            if let Seal::Blend = def.seal {
                seal_loop(&mut pcm);
            }
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
    // anchor-relative and must never touch it — rl#354). Speed here is the GAIT
    // speed, not the wind's `local_speed_mps`: on foot only the horizontal
    // components count — a standing jump moves at ~8.7× walking pace vertically,
    // which is wind, not footsteps.
    let (x, z, alt, speed_mps) = match &*vehicle {
        LocalVehicle::OnFoot => {
            let Some(p) = state.client.sim().player(state.client.me()) else {
                *ctx = AmbientContext::default();
                return;
            };
            let (x, z) = p.pos().to_meters_f64();
            let v = p.vel();
            let gait = ((v.x as f32).hypot(v.z as f32)) / crate::sim::UNIT as f32
                * crate::sim::TICK_HZ as f32;
            (
                x,
                z,
                Alt::AboveGround(p.alt() as f32 / crate::sim::UNIT as f32),
                gait,
            )
        }
        LocalVehicle::Flying { poses, .. } => {
            let Some(pose) = vehicle.cockpit_sample(clock.tick, clock.frac) else {
                *ctx = AmbientContext::default();
                return;
            };
            (
                pose.pos.x as f64,
                pose.pos.z as f64,
                Alt::Absolute(pose.pos.y),
                poses.speed_mps().unwrap_or(0.0),
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
        lush: 1.0 - biome::grass_dryness(h),
        rocky: biome::rocky_weight(h, normal_y),
        alt_m,
        speed_mps,
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
    /// Bus glide: the wind's 60 ms, so one combo-entry duck lands on the beds and
    /// the wind as a single mix move instead of two staggered ones.
    const GLIDE_BUS: f32 = 1.0 / (0.060 * SAMPLE_RATE as f32);

    /// Per-layer activation glide coefficient from its declared τ.
    fn glide_act(i: usize) -> f32 {
        1.0 / (LAYERS[i].glide_tau_s * SAMPLE_RATE as f32)
    }

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

        // Ducked layers ride the bus gain + lowpass; an exempt layer (none yet)
        // joins after the filter, untouched by the muffle.
        let (mut ducked, mut exempt) = (0.0, 0.0);
        for i in 0..LAYERS.len() {
            self.acts[i] += (self.targets.acts[i].load() - self.acts[i]) * Self::glide_act(i);
            let bed = &self.beds[i];
            if bed.is_empty() {
                continue;
            }
            let v = bed[self.cursors[i]] * self.acts[i];
            match LAYERS[i].muffle {
                Muffle::Duck => ducked += v,
                Muffle::Exempt => exempt += v,
            }
            self.cursors[i] = (self.cursors[i] + 1) % bed.len();
        }
        self.lp += (ducked * self.bus_gain - self.lp) * self.lp_a;
        Some((self.lp + exempt).clamp(-1.0, 1.0))
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
            lush: 0.0,
            rocky: 0.0,
            alt_m: 0.0,
            speed_mps: 0.0,
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
    fn valley_and_mountain_beds_split_the_terrain() {
        // Frogs need grassy AND lush; the dry plateau (lush 0) silences them
        // while the grass beds stay up.
        let dry_plateau = AmbientContext {
            grass: 1.0,
            lush: 0.0,
            rocky: 0.0,
            alt_m: 0.0,
            speed_mps: 0.0,
            vehicle: None,
        };
        assert_eq!(valley_bed(&dry_plateau), 0.0);
        assert_eq!(grass_bed(&dry_plateau), 1.0);
        let green_valley = AmbientContext {
            lush: 1.0,
            ..dry_plateau
        };
        assert_eq!(valley_bed(&green_valley), 1.0);
        // The rock face is the grass beds' complement: mountain up, grass out.
        let rock_face = AmbientContext {
            grass: 0.0,
            lush: 0.0,
            rocky: 1.0,
            ..dry_plateau
        };
        assert_eq!(mountain_bed(&rock_face), 1.0);
        assert_eq!(grass_bed(&rock_face), 0.0);
        assert_eq!(valley_bed(&rock_face), 0.0);
        // Same altitude fade as every ground bed.
        assert_eq!(
            mountain_bed(&AmbientContext {
                alt_m: 100.0,
                ..rock_face
            }),
            0.0
        );
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

    fn one_bed_stream_at(i: usize, pcm: Vec<f32>) -> (Arc<AmbienceTargets>, AmbienceStream) {
        let targets = Arc::<AmbienceTargets>::default();
        let mut beds: [Arc<[f32]>; LAYERS.len()] = std::array::from_fn(|_| Arc::from([]));
        beds[i] = pcm.into();
        (targets.clone(), AmbienceStream::new(targets, beds))
    }

    fn one_bed_stream(pcm: Vec<f32>) -> (Arc<AmbienceTargets>, AmbienceStream) {
        one_bed_stream_at(0, pcm)
    }

    /// The footsteps layer (a gait layer) releases in its declared 60 ms — a
    /// stop leaves no phantom step tail — where a bed at the same point in its
    /// fade is still clearly audible.
    #[test]
    fn gait_layer_releases_fast() {
        let steps_i = LAYERS
            .iter()
            .position(|l| l.file.contains("footsteps"))
            .unwrap();
        let release_at_300ms = |i: usize| {
            let (targets, mut s) = one_bed_stream_at(i, vec![0.5; 64]);
            targets.acts[i].store(0.8);
            let _ = s.by_ref().take(3 * SAMPLE_RATE as usize).count();
            targets.acts[i].store(0.0);
            s.by_ref().nth((0.3 * SAMPLE_RATE as f32) as usize).unwrap()
        };
        assert!(release_at_300ms(steps_i) < 0.01);
        assert!(release_at_300ms(0) > 0.1);
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

    #[test]
    fn movement_layers_follow_gait() {
        let walking = AmbientContext {
            grass: 1.0,
            lush: 0.0,
            rocky: 0.0,
            alt_m: 0.0,
            speed_mps: walk_mps(),
            vehicle: None,
        };
        assert_eq!(footsteps(&walking), 1.0);
        assert_eq!(grass_swish(&walking), 1.0);
        // Standing still is silent; sprint holds full rather than clipping past it.
        let still = AmbientContext {
            speed_mps: 0.0,
            ..walking
        };
        assert_eq!(footsteps(&still), 0.0);
        assert_eq!(
            footsteps(&AmbientContext {
                speed_mps: 1.8 * walk_mps(),
                ..walking
            }),
            1.0
        );
        // The swish needs grass under foot; the steps do not.
        let on_rock = AmbientContext {
            grass: 0.0,
            rocky: 1.0,
            ..walking
        };
        assert_eq!(grass_swish(&on_rock), 0.0);
        assert_eq!(footsteps(&on_rock), 1.0);
        // Mid-jump (past the ~0.08 m apex fade) and in a craft the steps cut out.
        assert_eq!(
            footsteps(&AmbientContext {
                alt_m: 0.1,
                ..walking
            }),
            0.0
        );
        assert_eq!(
            footsteps(&AmbientContext {
                vehicle: Some(VehicleKind::Plane),
                ..walking
            }),
            0.0
        );
    }

    #[test]
    fn engine_layers_gate_on_their_craft() {
        let in_plane = AmbientContext {
            grass: 0.0,
            lush: 0.0,
            rocky: 0.0,
            alt_m: 20.0,
            speed_mps: 0.0,
            vehicle: Some(VehicleKind::Plane),
        };
        // An idling engine is present the moment the craft is boarded, and speed
        // adds the top of the band.
        assert_eq!(plane_engine(&in_plane), 0.6);
        assert_eq!(
            plane_engine(&AmbientContext {
                speed_mps: 4.5,
                ..in_plane
            }),
            1.0
        );
        assert_eq!(ship_thruster(&in_plane), 0.0);
        let in_ship = AmbientContext {
            vehicle: Some(VehicleKind::Ship),
            ..in_plane
        };
        assert_eq!(plane_engine(&in_ship), 0.0);
        assert_eq!(ship_thruster(&in_ship), 0.6);
        let on_foot = AmbientContext {
            vehicle: None,
            ..in_plane
        };
        assert_eq!(plane_engine(&on_foot), 0.0);
        assert_eq!(ship_thruster(&on_foot), 0.0);
    }

    /// Not a test — the evidence generator for rl#357: renders a biome walk
    /// (green valley → dry plateau → rock face → past the snowline) through the
    /// real fetched beds to WAV. Skips (with a note) when the beds are absent.
    /// `AMBIENCE_EVIDENCE_DIR=docs/evidence/rl357-stage2 cargo test -p net
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
        let secs = 24;
        let n = SAMPLE_RATE as usize * secs;
        let mut pcm = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE as f32;
            // A walk up the whole biome ramp, 6 s per band: green valley (frogs
            // + birds + crickets + rustle) → dry plateau (frogs fade) → rock
            // face (grass beds out, mountain birds in) → past the snowline
            // (everything to the wind's silence).
            let (grass, lush, rocky) = match t {
                t if t < 6.0 => (1.0, 1.0, 0.0),
                t if t < 12.0 => (1.0, 1.0 - (t - 6.0) / 6.0, 0.0),
                t if t < 18.0 => {
                    let c = (t - 12.0) / 6.0;
                    (1.0 - c, 0.0, c)
                }
                t => (0.0, 0.0, 1.0 - (t - 18.0) / 6.0),
            };
            let ctx = AmbientContext {
                grass,
                lush,
                rocky,
                alt_m: 0.0,
                speed_mps: 0.0,
                vehicle: None,
            };
            for (def, slot) in LAYERS.iter().zip(&targets.acts) {
                slot.store(layer_target(def, &ctx));
            }
            pcm.push((s.next().unwrap() * i16::MAX as f32) as i16);
        }
        std::fs::create_dir_all(&dir).unwrap();
        let path = std::path::Path::new(&dir).join("ambience-context-sweep.wav");
        std::fs::write(path, crab_world::wav::wav_bytes(&pcm)).unwrap();
    }

    /// The stage-3 scenario, shared by the before and after renders so the A/B
    /// differs ONLY in the mix: stand in the green valley → walk → sprint →
    /// board the plane and climb to full speed → cruise the ship → back on foot
    /// in the grass. Returns the context plus the airspeed the wind synth sees.
    fn stage3_scenario(t: f32) -> (AmbientContext, f32) {
        let walk = walk_mps();
        let (speed, alt_m, vehicle) = match t {
            t if t < 4.0 => (0.0, 0.0, None),
            t if t < 10.0 => (walk, 0.0, None),
            t if t < 14.0 => (1.8 * walk, 0.0, None),
            t if t < 22.0 => {
                let n = ((t - 14.0) / 5.0).min(1.0);
                (
                    super::super::audio::FULL_WIND_MPS * n,
                    40.0 * n,
                    Some(VehicleKind::Plane),
                )
            }
            t if t < 28.0 => (2.5, 10.0, Some(VehicleKind::Ship)),
            _ => (walk, 0.0, None),
        };
        // The whole flight stays over the lush valley: the gain maps (canopy
        // halving, altitude fade) decide what survives boarding — the scenario
        // must not pre-silence the beds, or the render never exercises the
        // loudest engine+beds combination the level pass is accountable for.
        let ctx = AmbientContext {
            grass: 1.0,
            lush: 1.0,
            rocky: 0.0,
            alt_m,
            speed_mps: speed,
            vehicle,
        };
        (ctx, speed)
    }

    /// Not a test — the stage-3 before/after capture: the full outdoor mix (wind
    /// synth + every sampled layer) over one scenario, rendered twice. BEFORE is
    /// the mix as stage 2 left it — the five beds at their rough placement
    /// levels, no movement or engine layers; AFTER is the shipped LAYERS table
    /// (stage-3 layers + the deliberate level pass). Same scenario, same streams:
    /// the A/B is purely the mix. `AMBIENCE_EVIDENCE_DIR=docs/evidence/rl357-stage3
    /// cargo test -p net --features render stage3_mix_evidence -- --ignored`
    #[test]
    #[ignore = "artifact generator, not a check"]
    fn stage3_mix_evidence() {
        let Some(dir) = std::env::var_os("AMBIENCE_EVIDENCE_DIR") else {
            return;
        };
        let beds: [Arc<[f32]>; LAYERS.len()] = std::array::from_fn(|i| load_bed(&LAYERS[i]));
        if beds.iter().any(|b| b.is_empty()) {
            eprintln!("beds absent — run scripts/fetch-ambience.sh first");
            return;
        }
        // "Before" is the stage-2 mix: today's bed levels (the level pass left
        // the beds where stage 2 placed them) with the four stage-3 layers,
        // which sit last in the table, muted.
        let before: [f32; LAYERS.len()] = std::array::from_fn(|i| {
            if i < LAYERS.len() - 4 {
                LAYERS[i].level
            } else {
                0.0
            }
        });
        let after: [f32; LAYERS.len()] = std::array::from_fn(|i| LAYERS[i].level);
        for (name, levels) in [("mix-before.wav", before), ("mix-after.wav", after)] {
            let amb_targets = Arc::<AmbienceTargets>::default();
            let mut amb = AmbienceStream::new(amb_targets.clone(), beds.clone());
            let wind_targets = Arc::new(super::super::audio::WindTargets::default());
            let mut wind = super::super::audio::WindStream::new(wind_targets.clone());
            let n = SAMPLE_RATE as usize * 34;
            let mut pcm = Vec::with_capacity(n);
            for i in 0..n {
                let (ctx, speed) = stage3_scenario(i as f32 / SAMPLE_RATE as f32);
                for ((def, slot), level) in LAYERS.iter().zip(&amb_targets.acts).zip(levels) {
                    slot.store(level * (def.gain)(&ctx));
                }
                wind_targets.store(super::super::audio::profile(ctx.vehicle, speed));
                let s = (amb.next().unwrap() + wind.next().unwrap()).clamp(-1.0, 1.0);
                pcm.push((s * i16::MAX as f32) as i16);
            }
            std::fs::create_dir_all(&dir).unwrap();
            let path = std::path::Path::new(&dir).join(name);
            std::fs::write(path, crab_world::wav::wav_bytes(&pcm)).unwrap();
        }
    }
}
