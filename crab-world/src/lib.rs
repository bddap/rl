//! `crab_world` library surface.
//!
//! The RL crab machinery — [`bot`], [`physics`], [`training`], [`play`] — that both the
//! headless trainer and the renderers build on. The networking layer lives in the separate
//! `net` crate, which depends on this one (it drives these crabs into multiplayer); a
//! dependency the other way would be a cycle. The training binary and the `game` binary both
//! link this one library, so there is ONE copy of every module — no duplicated crab /
//! physics / brain implementation across binaries.

// rl#49: the `--features wgpu` build resolves `Module: Send` on `load_record` for
// `CrabBrain<Autodiff<Wgpu>>`, chasing wgpu_core's nested generics past the default
// 128-deep recursion limit and aborting with E0275 (overflow, not a genuine !Send). A
// higher limit lets that finite-but-deep resolution complete. ONLY needed under `wgpu`
// — every other build (the headless CPU trainer, the renderers) compiles at the default
// 128, so the bump is gated on the feature instead of paid unconditionally.
#![cfg_attr(feature = "wgpu", recursion_limit = "512")]

use std::path::PathBuf;

use bevy::prelude::*;
use clap::Parser;

pub mod bot;
/// Build provenance (commit sha + build time) + the subtle corner stamp that shows it.
/// The consts compile everywhere; the overlay spawner is render-gated (UI-only).
pub mod build_info;
/// Reusable controls + hold-to-reveal-overlay framework, generic over an app's action set
/// (GCR and the demo each bring their own [`controls::ControlScheme`]).
pub mod controls;
/// The single FNV-1a/64 implementation every determinism guard folds bytes with — one offset,
/// one prime, one loop, so cross-peer digests can't drift apart. Shared with the `net` crate's
/// lockstep desync hash, so it is `pub`; the hash is a frozen wire format — changing it desyncs
/// peers, so treat it as append-only, not a free-to-edit internal.
pub mod fnv;
pub mod physics;
pub mod training;

// Rendering-only modules — gated out of the headless trainer build. They pull bevy's
// render/PBR/egui types (cameras, materials, screenshots), which don't even exist when
// bevy is built without `render`, so without this gate the trainer wouldn't compile.
// The trainer never renders, so it loses nothing.
#[cfg(feature = "render")]
pub mod play;
#[cfg(feature = "render")]
pub mod player;
/// Procedural night-sky skybox shared by both rendered surfaces (rl-demo + GCR). `pub`
/// because the `net` crate's GCR app builders add its [`sky::NightSkyPlugin`] too.
#[cfg(feature = "render")]
pub mod sky;
/// Shared offscreen render-to-PNG-on-settle primitive behind both headless shots
/// (the crab inspection shot in `play`, the FP game-view shot in the `net` crate's `render`).
/// `pub` because that `net::render` shot lives in the sibling crate and composes this.
#[cfg(feature = "render")]
pub mod screenshot;

/// Whether to spawn visual assets (meshes, lights). The `rl learn` rollout worlds
/// set this false (rendering off entirely); the rendering modes (demo/screenshot, and
/// the game's solo crab) set it true. A `Resource` so any plugin can read it.
#[derive(Resource, Clone, Copy)]
pub struct Visuals(pub bool);

/// Training config (consumed by the learner and its rollout threads, which build a
/// `TrainingState`) plus the render modes' shared knobs. The `learn` subcommand
/// flattens it so e.g. `--checkpoint-dir` / `--ticks` mean the same thing
/// everywhere.
#[derive(Parser, Debug, Clone)]
pub struct TrainConfig {
    /// Directory for checkpoint files. On startup, if the directory contains a
    /// previous checkpoint it will be loaded automatically. During training,
    /// checkpoints are saved here periodically and on exit.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,

    /// Stop training after this many physics ticks (0 = run until killed). The budget
    /// is counted in ticks, never wall-clock, so a run simulates an identical amount
    /// regardless of machine speed or load — the "fixed ticks, not real time"
    /// guarantee an assumed time↔tick relation can't give. The learner checks the
    /// budget once per PPO iteration, so it stops at the first iteration boundary at
    /// or after N (overshooting by up to one K·(--envs)·H iteration's worth of ticks).
    #[arg(long, default_value_t = 0)]
    pub ticks: u64,

    /// Benchmark only: skip NN inference in the train loop (hold zero actions),
    /// isolating physics + engine overhead from network cost. Training is
    /// meaningless under this flag — it exists to measure the per-step bottleneck.
    #[arg(long)]
    pub bench_skip_nn: bool,

    /// Environments M each rollout thread steps in its one world per tick (one
    /// batched NN pass over the M crabs, which sit on a 4 m grid). Total parallel
    /// envs = `--workers` × M. Capped at [`bot::body::MAX_ENVS`] — each env needs its
    /// own collision bit so independent crabs pass through each other, and `Group` has
    /// only 32 bits (see [`bot::body::crab_collision`]).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..=bot::body::MAX_ENVS as u64))]
    pub envs: u64,

    /// Master RNG seed for the run's stochastic choices — action-noise sampling, target
    /// placement, spawn rotation, the minibatch shuffle, and weight init. Omitted (the
    /// default) draws a fresh seed from OS entropy and LOGS it, so any run can be
    /// reproduced after the fact by passing the logged value back via `--seed`. The
    /// learner threads ONE base seed to every rollout worker, which mixes in its index so
    /// the K streams stay independent; a fixed seed therefore reproduces the whole run.
    #[arg(long)]
    pub seed: Option<u64>,
}
