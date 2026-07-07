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

/// The one knob shared by every mode that loads a checkpoint (`learn`, `eval`,
/// rl-demo). Split out of [`TrainConfig`] so binaries that only LOAD flatten just
/// this, and a stray training knob like `--envs` is a parse error there instead of
/// a silent no-op (bddap/rl#217).
#[derive(clap::Args, Debug, Clone)]
pub struct CheckpointArgs {
    /// Directory of checkpoint files: loaded on startup if one is present; the
    /// trainer also saves here periodically and on exit.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,
}

/// Training config, consumed by the learner and its rollout threads (which build a
/// `TrainingState`). Parsed only by the `learn` subcommand.
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
}
