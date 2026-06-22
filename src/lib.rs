//! `rl` library surface.
//!
//! Hosts BOTH the multiplayer netcode foundation ([`net`]) and the RL crab
//! machinery ([`bot`], [`physics`], [`training`], [`play`]) so the `game` binary can
//! drive the trained crab in its SOLO playtest. The `rl` training binary and the
//! `game` binary both link this one library, so there is ONE copy of every module —
//! no duplicated crab / physics / brain implementation across the two binaries.
//!
//! Until this consolidation the RL modules lived only in the `rl` training binary
//! (`src/main.rs`) and were invisible to `game`; that is why "reuse the trained crab
//! body in the game" was deferred. They now live behind this one crate root, which
//! the training binary re-imports (`use rl::{bot, physics, …}`) instead of declaring.

// rl#49: the `--features wgpu` build pulls bevy's wgpu 27 and burn-wgpu's wgpu 26 into
// one graph. Resolving `Module: Send` on `load_record` for `CrabBrain<Autodiff<Wgpu>>`
// chases wgpu_core's nested generics past the default 128-deep recursion limit and
// aborts with E0275 (overflow, not a genuine !Send). A higher limit lets that
// finite-but-deep resolution complete; no effect on the default CPU build, zero runtime
// cost. Set on the library root because the affected modules now live here.
#![recursion_limit = "512"]

use std::path::PathBuf;

use bevy::prelude::*;
use clap::Parser;

pub mod bot;
pub mod debug_sliders;
pub mod net;
pub mod physics;
pub mod play;
pub mod player;
pub mod training;

/// Whether to spawn visual assets (meshes, lights). The `rl learn` rollout worlds
/// set this false (rendering off entirely); the rendering modes (demo/screenshot, and
/// the game's solo crab) set it true. A `Resource` so any plugin can read it.
#[derive(Resource, Clone, Copy)]
pub struct Visuals(pub bool);

/// Device for the batched PPO update: CPU (the production ndarray path) or GPU (the
/// RTX via wgpu/Vulkan — rl#49). Rollout inference stays on CPU either way; only the
/// update moves.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateDevice {
    /// Run the update on the CPU (`Autodiff<NdArray>`) — byte-for-byte the production
    /// CPU path; the GPU backend is never touched.
    Cpu,
    /// Run the update on the GPU (`Autodiff<Wgpu>` over Vulkan): mirror the policy
    /// CPU→GPU, update on the discrete GPU, mirror back. Requires the binary be built
    /// with `--features wgpu`; otherwise the run fails with a clear error (never a
    /// silent CPU fallback). Forces + asserts a real discrete-GPU adapter (no software
    /// lavapipe masquerading as the GPU).
    Gpu,
}

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
}
