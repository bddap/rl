//! `crab_world` library surface.
//!
//! The RL crab machinery — [`bot`], [`physics`], [`training`], [`play`] — that both the
//! headless trainer and the renderers build on. The networking layer lives in the separate
//! `net` crate, which depends on this one (it drives these crabs into multiplayer); a
//! dependency the other way would be a cycle. The training binary and the `game` binary both
//! link this one library, so there is ONE copy of every module — no duplicated crab /
//! physics / brain implementation across binaries.

// rl#49: the `--features wgpu` build resolves `Module: Send` on `load_record` for
// `AnyBrain<Autodiff<Wgpu>>`, chasing wgpu_core's nested generics past the default
// 128-deep recursion limit and aborting with E0275 (overflow, not a genuine !Send). A
// higher limit lets that finite-but-deep resolution complete. ONLY needed under `wgpu`
// — every other build (the headless CPU trainer, the renderers) compiles at the default
// 128, so the bump is gated on the feature instead of paid unconditionally.
#![cfg_attr(feature = "wgpu", recursion_limit = "512")]

use std::path::PathBuf;

use bevy::prelude::*;
use clap::Parser;

/// Single source of truth for the bundled-asset root + the startup glyph-presence guard,
/// so a fresh clone's `cargo run` finds the committed control icons (and fails loud if not).
pub mod assets;
pub mod bot;
/// Reusable controls + hold-to-reveal-overlay framework, generic over an app's action set
/// (GCR and the demo each bring their own [`controls::ControlScheme`]).
pub mod controls;
/// The headless training-SUCCESS eval — the true measure of the policy, distinct from the
/// training reward: reuses the demo/train crab+ball scenario headless, places the ball far,
/// drives the loaded policy deterministically, and reports real metres of progress toward the
/// ball plus the total applied joint torque. Pure physics + inference (no window), so it
/// stays out of the render gate.
pub mod eval;
/// The single FNV-1a/64 implementation every determinism guard folds bytes with — one offset,
/// one prime, one loop, so cross-peer digests can't drift apart. Shared with the `net` crate's
/// lockstep desync hash, so it is `pub`; the hash is a frozen wire format — changing it desyncs
/// peers, so treat it as append-only, not a free-to-edit internal.
pub mod fnv;
/// LOUD missing-Sally-mesh signalling (OTEL error + on-screen banner) shared by the
/// player-facing surfaces (rl-demo + game/net), so they can't drift on how a missing Sally is
/// announced (bddap/rl#706). The banner half is render-gated inside; the OTEL half compiles
/// headless too (it's never called there).
pub mod mesh_fallback;
pub mod physics;
/// The trained-policy inference — load a checkpoint's brain + normalizer and run
/// deterministic mean-action inference. Pure (burn tensors + a checkpoint reader, no bevy
/// render types), so it lives OUTSIDE the render gate: the headless trainer-side eval
/// ([`eval`]) reuses the SAME policy the rendered demo runs (`play` re-exports it), instead
/// of a second copy that could drift.
pub mod policy;
pub mod training;
/// The player's single-player rapier flight vehicle (plane / Outer-Wilds ship), living in the crab's
/// rapier world so it collides with Sally. Headless (pure physics), so it stays out of the
/// render gate below.
pub mod vehicle;

// Rendering-only modules — gated out of the headless trainer build. They pull bevy's
// render/PBR/egui types (cameras, materials, screenshots), which don't even exist when
// bevy is built without `render`, so without this gate the trainer wouldn't compile.
// The trainer never renders, so it loses nothing.
/// The ONE `DefaultPlugins` recipe (asset root, LogPlugin-disabled, window/offscreen)
/// every rendered surface boots from — GCR windowed + screenshot, rl-demo both arms.
#[cfg(feature = "render")]
pub mod app_boot;
/// The crab render-mode cycle + the ONE shared collider-wireframe (GCR + rl-demo). `pub`
/// because both rendered surfaces (the `net` crate's GCR app and the rl-demo) drive it.
#[cfg(feature = "render")]
pub mod crab_view;
#[cfg(feature = "render")]
pub mod play;
/// Shared offscreen render-to-PNG-on-settle primitive behind both headless shots
/// (the crab inspection shot in `play`, the FP game-view shot in the `net` crate's `render`).
/// `pub` because that `net::render` shot lives in the sibling crate and composes this.
#[cfg(feature = "render")]
pub mod screenshot;
/// Procedural night-sky skybox shared by both rendered surfaces (rl-demo + GCR). `pub`
/// because the `net` crate's GCR app builders add its [`sky::NightSkyPlugin`] too.
#[cfg(feature = "render")]
pub mod sky;

/// Whether to spawn visual assets (meshes, lights). The `rl learn` rollout worlds
/// set this false (rendering off entirely); the rendering modes (demo/screenshot, and
/// the game's solo crab) set it true. A `Resource` so any plugin can read it.
#[derive(Resource, Clone, Copy)]
pub struct Visuals(pub bool);

/// Truncate `s` to at most `max` BYTES, cutting back to a char boundary so the slice can't
/// panic mid-codepoint. THE one implementation for every human-facing string bound (the
/// brain-label display cap, the articulation wire's label clamp) — the loop is easy to
/// re-spell subtly wrong, so it lives once. A boundary exists within 3 bytes of any index,
/// so this trims at most 3 bytes past `max`.
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

    /// Stop training after this many physics ticks (0 = run until killed). The budget
    /// is counted in ticks, never wall-clock, so a run simulates an identical amount
    /// regardless of machine speed or load — the "fixed ticks, not real time"
    /// guarantee an assumed time↔tick relation can't give. The learner checks the
    /// budget once per PPO iteration, so it stops at the first iteration boundary at
    /// or after N (overshooting by up to one K·(--envs)·H iteration's worth of ticks).
    #[arg(long, default_value_t = 0)]
    pub ticks: u64,

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
