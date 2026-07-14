#![cfg_attr(feature = "wgpu", recursion_limit = "512")]

use std::path::PathBuf;

use bevy::prelude::*;
use clap::Parser;

pub mod assets;
pub mod bot;
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
pub mod training;
pub mod vehicle;

#[cfg(feature = "render")]
pub mod app_boot;
#[cfg(feature = "render")]
pub mod crab_view;
#[cfg(feature = "render")]
pub mod play;
#[cfg(feature = "render")]
pub mod screenshot;
/// Procedural night-sky skybox shared by both rendered surfaces (rl-demo + GCR). `pub`
/// because the `net` crate's GCR app builders add its [`sky::NightSkyPlugin`] too.
#[cfg(feature = "render")]
pub mod sky;

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

// The one knob shared by every mode that loads a checkpoint (`learn`, `eval`, rl-demo). Split
// out of `TrainConfig` so binaries that only LOAD flatten just this, and a stray training knob
// like `--envs` is a parse error there instead of a silent no-op (bddap/rl#217).
//
// Deliberately NOT a doc comment: clap adopts a flattened struct's docs as the enclosing
// command's `about`, which would overwrite the description of every subcommand flattening it.
#[derive(clap::Args, Debug, Clone)]
pub struct CheckpointArgs {
    /// Directory of checkpoint files: loaded on startup if one is present; the
    /// trainer also saves here periodically and on exit.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,
}

// The render-surface knob, flattened by every binary that opens a view on the crab world
// (rl-demo, `game play`/`net-join`/`fp-screenshot`/`net-screenshot`) — the ONE declaration of
// `--render-mode` and its env fallback. It used to be parsed deep in the view code, where a
// malformed value was warned about and IGNORED (leaving the default mode, as if the override
// had taken) and a hand-rolled `flag.or_else(env)` sat beside clap's own `env` (rl#275).
// `RL_DEBUG_COLLIDERS` is gone with it: it was a second, less expressive spelling of
// `--render-mode colliders`.
//
// Deliberately NOT a doc comment: clap adopts a flattened struct's docs as the enclosing
// command's `about`, which would overwrite the description of every subcommand flattening it.
#[cfg(feature = "render")]
#[derive(clap::Args, Debug, Clone, Copy, Default)]
pub struct RenderArgs {
    /// Which view to boot in. Unset: the mesh — or the honest collider wireframe when the
    /// canonical Sally mesh can't be resolved.
    #[arg(long, env = "RL_RENDER_MODE", value_enum)]
    pub render_mode: Option<crab_view::RenderMode>,
}

#[cfg(feature = "render")]
impl RenderArgs {
    /// The mode `surface` boots in. With no usable canonical body, the flagless default is
    /// the collider wireframe, and the fallback is latched for `surface` — the render stays
    /// honest about what it is drawing (never a procedural stand-in posing as Sally).
    pub fn initial(self, surface: mesh_fallback::Surface) -> crab_view::RenderMode {
        let mesh_err = mesh_fallback::usable_model().as_ref().err();
        if let Some(reason) = mesh_err {
            mesh_fallback::log_fallback(surface, reason);
        }
        self.render_mode.unwrap_or(if mesh_err.is_some() {
            crab_view::RenderMode::Colliders
        } else {
            crab_view::RenderMode::Mesh
        })
    }
}

// Training config, consumed by the learner and its rollout threads (which build a
// `TrainingState`). Parsed only by the `learn` subcommand.
//
// The run-shaping knobs below keep an `env` fallback for the overnight loop's existing `RL_*`
// exports, but they are real flags: visible in `--help`, echoed by the argv a run was launched
// with, and a malformed value (flag OR env) is a parse error at t=0 — never a silent fallback
// mid-run (rl#272).
//
// Deliberately NOT a doc comment: clap adopts a flattened struct's docs as the enclosing
// command's `about`, which would overwrite the description of every subcommand flattening it.
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

    /// Fraction of episodes whose target samples the close disc instead of the chase
    /// band — the rl#250 curriculum mix. 0.0 keeps the pure chase band.
    #[arg(long, env = "RL_TARGET_CLOSE_FRAC", value_parser = parse_unit_frac,
          default_value_t = 0.0)]
    pub target_close_frac: f32,

    /// Effort-tax coefficient on Σ|drive|² — the reward's only economy term (rl#268).
    #[arg(long, env = "RL_EFFORT_WEIGHT", value_parser = parse_effort_weight,
          default_value_t = training::reward::EFFORT_WEIGHT_DEFAULT)]
    pub effort_weight: f32,

    /// DIAGNOSTIC: log the rollout's mean Σ|drive|² and the tax it pays, per step.
    #[arg(long, env = "RL_LOG_EFFORT", value_parser = clap::builder::FalseyValueParser::new())]
    pub log_effort: bool,
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

fn parse_unit_frac(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{v} is outside 0..=1"))
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
