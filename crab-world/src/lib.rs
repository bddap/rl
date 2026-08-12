#![cfg_attr(feature = "wgpu", recursion_limit = "512")]

use std::path::PathBuf;

use bevy::prelude::*;
use clap::Parser;

// rl#282: this suite has wedged for 1 h+ (every thread futex_wait at 0% CPU)
// under trainer saturation; abort loudly on a process-wide CPU flatline instead.
#[cfg(test)]
test_watchdog::arm!();

pub mod assets;
pub mod bot;
pub mod chord;
pub mod controls;
/// The headless training-SUCCESS eval — the true measure of the policy, distinct from the
/// training reward: reuses the demo/train crab+ball scenario headless, places the ball far,
/// drives the loaded policy deterministically, and reports real metres of progress toward the
/// ball plus the total applied joint torque. Pure physics + inference (no window), so it
/// stays out of the render gate.
pub mod eval;
pub mod fnv;
pub mod mesh_fallback;
pub mod physics;
pub mod policy;
pub mod terrain;
pub mod training;
pub mod vehicle;

#[cfg(feature = "render")]
pub mod app_boot;
#[cfg(feature = "render")]
pub mod crab_view;
#[cfg(feature = "render")]
pub mod debug_overlay;
#[cfg(feature = "render")]
pub mod frame_telemetry;
#[cfg(feature = "render")]
pub mod ground;
/// The d-pad instrument (rl#359): code entry sounds notes; wired by
/// [`chord::install_chords`].
#[cfg(feature = "render")]
pub mod instrument;
#[cfg(feature = "render")]
pub mod moisture;
/// The moon — THE light source (rl#374). Its plugin rides [`sky::NightSkyPlugin`].
#[cfg(feature = "render")]
pub mod moon;
#[cfg(feature = "render")]
pub mod play;
#[cfg(feature = "render")]
pub mod scatter;
#[cfg(feature = "render")]
pub mod screenshot;
/// Procedural night-sky skybox shared by both rendered surfaces (rl-demo + GCR). `pub`
/// because the `net` crate's GCR app builders add its [`sky::NightSkyPlugin`] too.
#[cfg(feature = "render")]
pub mod sky;
#[cfg(feature = "render")]
pub mod wav;

#[derive(Resource, Clone, Copy)]
pub struct Visuals(pub bool);

/// Truncate `s` to at most `max` BYTES, cutting back to a char boundary so the slice can't
/// panic mid-codepoint. THE one implementation for every human-facing string bound (the
/// brain-label display cap, the articulation wire's label clamp) — the loop is easy to
/// re-spell subtly wrong, so it lives once. A boundary exists within 3 bytes of any index,
/// so the cut lands at `max` or at most 3 bytes below it — never above.
pub fn truncate_at_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// The one knob shared by every mode that loads a checkpoint (`learn`, `eval`, rl-demo). Split
/// out of [`TrainConfig`] so binaries that only LOAD flatten just this, and a stray training
/// knob like `--envs` is a parse error there instead of a silent no-op (bddap/rl#217).
#[derive(clap::Args, Debug, Clone)]
pub struct CheckpointArgs {
    /// Directory of checkpoint files: loaded on startup if one is present; the
    /// trainer also saves here periodically and on exit.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,
}

/// The render-surface knob, flattened by every binary that opens a view on the crab world —
/// the ONE declaration of `--render-mode` and its env fallback, so a malformed value is a parse
/// error at t=0 on every surface (rl#275). `RL_DEBUG_COLLIDERS` is gone with it: it was a
/// second, less expressive spelling of `--render-mode colliders`.
#[cfg(feature = "render")]
#[derive(clap::Args, Debug, Clone, Copy, Default)]
pub struct RenderArgs {
    /// Which view to boot in. Unset: the mesh — or the honest collider wireframe when the
    /// canonical Sally mesh can't be resolved.
    #[arg(long, env = "RL_RENDER_MODE", value_enum)]
    pub render_mode: Option<crab_view::RenderMode>,
    /// Which ground look to boot in (rl#304). The in-game toggle walks the same set;
    /// this is the same one field, pre-set, so an evidence shot exercises the swap
    /// the keybind does.
    #[arg(long, env = "RL_GROUND_LOOK", value_enum)]
    pub ground_look: Option<ground::GroundLook>,
    /// Moon compass direction, degrees around +Y (rl#374). Setting this (or
    /// elevation) freezes sky motion so the pose holds; add --moon-timescale
    /// to move anyway.
    #[arg(long, env = "RL_MOON_AZIMUTH_DEG", allow_hyphen_values = true)]
    pub moon_azimuth_deg: Option<f32>,
    /// Moon height above the horizon, degrees (90 = zenith). Freezes sky
    /// motion like --moon-azimuth-deg.
    #[arg(long, env = "RL_MOON_ELEVATION_DEG", allow_hyphen_values = true)]
    pub moon_elevation_deg: Option<f32>,
    /// Moonlight + disc hue, degrees on the HSL wheel.
    #[arg(long, env = "RL_MOON_HUE_DEG", allow_hyphen_values = true)]
    pub moon_hue_deg: Option<f32>,
    /// Moon phase, wrapping [0, 1): 0 = new, 0.5 = full. Drives luminosity.
    /// Live even with motion on — the sweep advances it from here.
    #[arg(long, env = "RL_MOON_PHASE", allow_hyphen_values = true)]
    pub moon_phase: Option<f32>,
    /// Sky-motion timescale, sim seconds per real second (rl#374); default 24
    /// (a day-length moon traversal per real hour). 0 freezes the sky.
    #[arg(long, env = "RL_MOON_TIMESCALE", allow_hyphen_values = true)]
    pub moon_timescale: Option<f32>,
}

#[cfg(feature = "render")]
impl RenderArgs {
    /// The view `surface` boots in. With no usable canonical body, the flagless render
    /// default is the collider wireframe, and the fallback is LOGGED (latched for
    /// `surface`) — the render stays honest about what it is drawing, never a procedural
    /// stand-in posing as Sally. That logging is why this is `resolve` and not a getter:
    /// call it once, at the entrypoint.
    pub fn resolve(self, surface: mesh_fallback::Surface) -> BootView {
        let mesh_err = mesh_fallback::usable_model().as_ref().err();
        if let Some(reason) = mesh_err {
            mesh_fallback::log_fallback(surface, reason);
        }
        BootView {
            render_mode: self.render_mode.unwrap_or(if mesh_err.is_some() {
                crab_view::RenderMode::Colliders
            } else {
                crab_view::RenderMode::Mesh
            }),
            ground_look: self.ground_look.unwrap_or_default(),
            moon: self.boot_moon(),
        }
    }

    /// The boot [`moon::Moon`]: defaults overlaid with whatever knobs were
    /// passed. An explicit pose (azimuth or elevation) implies a frozen sky —
    /// motion would overwrite the posed angles on the first frame, so unless
    /// the timescale is ALSO explicit, posing zeroes it (self-correcting: you
    /// asked for that pose, you keep it).
    fn boot_moon(self) -> moon::Moon {
        let moon = moon::Moon::default();
        let posed = self.moon_azimuth_deg.is_some() || self.moon_elevation_deg.is_some();
        moon::Moon {
            azimuth_deg: self.moon_azimuth_deg.unwrap_or(moon.azimuth_deg),
            elevation_deg: self.moon_elevation_deg.unwrap_or(moon.elevation_deg),
            hue_deg: self.moon_hue_deg.unwrap_or(moon.hue_deg),
            phase: self.moon_phase.unwrap_or(moon.phase),
            timescale: self
                .moon_timescale
                .unwrap_or(if posed { 0.0 } else { moon.timescale }),
        }
    }
}

/// The resolved boot state of every runtime-cyclable view knob, so adding one does not
/// grow every app-builder signature.
#[cfg(feature = "render")]
#[derive(Debug, Clone, Copy)]
pub struct BootView {
    pub render_mode: crab_view::RenderMode,
    pub ground_look: ground::GroundLook,
    /// Boot values for the moon knobs (rl#374); the resource stays live-tweakable.
    pub moon: moon::Moon,
}

/// A view knob whose states ARE the values of its CLI flag. Blanket, so it names the
/// concept without anyone implementing it — and lets crates without a clap dependency
/// (`net`'s toggles) bound on it.
pub trait CyclableView: clap::ValueEnum + PartialEq + Copy {}
impl<T: clap::ValueEnum + PartialEq + Copy> CyclableView for T {}

/// Every state of a view knob, in clap declaration order — for callers without a clap
/// dependency (`net`'s per-variant chord dispatch, rl#330 stage 5).
pub fn view_variants<T: CyclableView>() -> &'static [T] {
    T::value_variants()
}

/// Advance a view knob one step, wrapping. The cycle order is clap's declaration
/// order — the very list the knob's flag accepts — so a knob has ONE set of states
/// and no hand-written `next` to forget a variant in.
pub fn next_view_variant<T: CyclableView>(current: T) -> T {
    let all = T::value_variants();
    let i = all
        .iter()
        .position(|v| *v == current)
        .expect("a value enum's variants include every value of the type");
    all[(i + 1) % all.len()]
}

/// A view knob's canonical name — what you'd type after its flag, so the readout and
/// the CLI never spell a state two ways.
pub fn view_variant_name<T: CyclableView>(current: &T) -> String {
    current
        .to_possible_value()
        .expect("a view knob has no skipped variants")
        .get_name()
        .to_string()
}

/// Training config, consumed by the learner and its rollout threads (which build a
/// `TrainingState`). Parsed only by the `learn` subcommand.
///
/// The run-shaping knobs keep an `env` fallback for the overnight loop's existing `RL_*`
/// exports, but they are real flags: visible in `--help`, echoed by the argv a run was launched
/// with, and a malformed value (flag OR env) is a parse error at t=0 — never a silent fallback
/// mid-run (rl#272).
#[derive(Parser, Debug, Clone)]
pub struct TrainConfig {
    #[command(flatten)]
    pub checkpoint: CheckpointArgs,

    #[arg(long, default_value_t = 0)]
    pub ticks: u64,

    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..=bot::body::MAX_ENVS as u64))]
    pub envs: u64,

    #[arg(long)]
    pub seed: Option<u64>,

    /// Exploration σ-floor (log-space) at the start of the anneal (rl#161).
    #[arg(long, env = "RL_LOG_STD_FLOOR_START", allow_negative_numbers = true,
          default_value_t = training::algorithm::LOG_STD_FLOOR_START_DEFAULT)]
    pub log_std_floor_start: f32,

    /// Exploration σ-floor (log-space) the anneal refines down to.
    #[arg(long, env = "RL_LOG_STD_FLOOR_END", allow_negative_numbers = true,
          default_value_t = bot::arch::LOG_STD_MIN)]
    pub log_std_floor_end: f32,

    /// Ticks over which the σ-floor anneals from start to end (0 = pinned at end).
    #[arg(long, env = "RL_LOG_STD_ANNEAL_TICKS",
          default_value_t = training::algorithm::LOG_STD_ANNEAL_TICKS_DEFAULT)]
    pub log_std_anneal_ticks: u64,

    /// Effort-tax coefficient on Σ|drive|² — the reward's only economy term (rl#268).
    #[arg(long, env = "RL_EFFORT_WEIGHT", value_parser = parse_effort_weight,
          default_value_t = training::reward::EFFORT_WEIGHT_DEFAULT)]
    pub effort_weight: f32,

    /// DIAGNOSTIC: log the rollout's mean Σ|drive|² and the tax it pays, per step.
    #[arg(long, env = "RL_LOG_EFFORT", value_parser = clap::builder::FalseyValueParser::new())]
    pub log_effort: bool,

    /// Hard cap on PPO minibatch steps per update — the rl#276 escalation lever: fewer
    /// steps per iteration slows the per-checkpoint policy walk at the σ-floor.
    #[arg(long, env = "RL_PPO_STEPS_CAP")]
    pub ppo_steps_cap: Option<std::num::NonZeroU32>,

    /// DIAGNOSTIC: the ROLLOUT worlds' ground. Default `gcr`, the canonical tile —
    /// the only ground a deployable policy trains on (rl#293). `flat` isolates the
    /// learning core from terrain (the 1807 flat-ground canary). The eval and the
    /// plant sidecar/digest stay canonical-GCR either way: the chase eval remains the
    /// one fixed instrument, and a non-gcr run's checkpoints are diagnostic artifacts,
    /// never deploy or warm-start candidates.
    #[arg(long, env = "RL_TERRAIN", value_enum, default_value_t = TrainTerrain::Gcr)]
    pub terrain: TrainTerrain,

    /// DIAGNOSTIC: far edge (m) of the target-band draw, in (BAND_START_MIN,
    /// [`training::targets::BAND_MAX_M`]]. Default the canonical band edge; smaller
    /// restricts rollouts to the near band (the 1807 canary trains 1.5–9 m) without
    /// touching the band geometry constants every other consumer (eval pace probe,
    /// GCR hunt poser, edge margins) is pinned to.
    #[arg(long, env = "RL_BAND_MAX_M", value_parser = parse_band_max,
          default_value_t = training::targets::BAND_MAX_M)]
    pub band_max_m: f32,
}

/// [`TrainConfig::terrain`]'s values — the ONE production seam for non-canonical
/// ground, so a diagnostic run forks a flag, never a code path.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrainTerrain {
    #[default]
    Gcr,
    Flat,
}

impl TrainTerrain {
    /// The rollout ground this choice builds. Flat spans ±512 m — the smallest
    /// round size clearing [`training::targets::sample_clamp_half`]'s edge-margin
    /// assert with the full canonical band, same as the band tests' fixture.
    pub fn grid(self) -> std::sync::Arc<terrain::TerrainGrid> {
        match self {
            Self::Gcr => terrain::TerrainGrid::gcr(),
            Self::Flat => std::sync::Arc::new(terrain::TerrainGrid::flat(512.0)),
        }
    }
}

impl TrainConfig {
    /// Envs per rollout world as a count — the ONE sizing formula, so a world's
    /// `NumEnvs`, its `TrainingState` buffers, and the learner's accounting can never
    /// disagree. (clap already rejects `--envs 0`; the clamp is a backstop for
    /// hand-built configs.)
    pub fn num_envs(&self) -> usize {
        self.envs.max(1) as usize
    }
}

#[cfg(test)]
impl TrainConfig {
    /// Test config CLAP-PARSED so every knob carries its real default — a struct
    /// literal here would grow a second default source that drifts.
    pub(crate) fn scratch(checkpoint_dir: &std::path::Path, envs: u64, seed: u64) -> Self {
        Self::try_parse_from([
            "rl",
            "--checkpoint-dir",
            checkpoint_dir.to_str().unwrap(),
            "--envs",
            &envs.to_string(),
            "--seed",
            &seed.to_string(),
        ])
        .expect("parse scratch TrainConfig")
    }
}

fn parse_band_max(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    // Below BAND_START_MIN the log-uniform draw's range inverts; above the canonical
    // edge the sampling clamp's edge margin (sized from the CONST) no longer bounds it.
    if v.is_finite() && v > training::targets::BAND_START_MIN && v <= training::targets::BAND_MAX_M
    {
        Ok(v)
    } else {
        Err(format!(
            "{v} is not in ({}, {}]",
            training::targets::BAND_START_MIN,
            training::targets::BAND_MAX_M
        ))
    }
}

fn parse_effort_weight(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    // Negative would PAY for flailing; NaN would poison every reward in the run.
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(format!("{v} is not a finite non-negative weight"))
    }
}

#[cfg(all(test, feature = "render"))]
mod render_args_tests {
    use super::*;

    /// An explicit moon pose freezes the sky unless the timescale is also
    /// explicit — motion would otherwise overwrite the posed flag on the
    /// first frame, a silent no-op of what the user typed.
    #[test]
    fn posing_the_moon_freezes_motion() {
        assert_eq!(
            RenderArgs::default().boot_moon().timescale,
            moon::DEFAULT_TIMESCALE
        );
        let posed = RenderArgs {
            moon_elevation_deg: Some(40.0),
            ..Default::default()
        }
        .boot_moon();
        assert_eq!((posed.timescale, posed.elevation_deg), (0.0, 40.0));
        let posed_moving = RenderArgs {
            moon_azimuth_deg: Some(100.0),
            moon_timescale: Some(moon::DEFAULT_TIMESCALE),
            ..Default::default()
        }
        .boot_moon();
        assert_eq!(posed_moving.timescale, moon::DEFAULT_TIMESCALE);
    }
}
