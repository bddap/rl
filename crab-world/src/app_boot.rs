//! The ONE bevy plugin-group recipe every rendered rl surface boots from (GCR's windowed
//! client, its offscreen screenshot scaffold, rl-demo's demo + render-to-image arms).
//!
//! Sibling recipe: [`crate::bot::headless::headless_stack`] is the SIM-world counterpart
//! (GPU off, no asset root, compiles with render OFF) — a window/winit change here may
//! need mirroring there.

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

/// `DefaultPlugins` configured the way every rendered rl binary needs it.
///
/// - `AssetPlugin` rooted at the bundled `assets/` dir, so a fresh clone's `cargo run`
///   finds the committed control glyphs regardless of cwd or which workspace bin runs
///   (bevy's default root is the running bin's crate dir, which has no glyphs);
///   `BEVY_ASSET_ROOT` still overrides (deploy). See [`crate::assets`].
/// - `LogPlugin` DISABLED. Every rl binary installs the shared `otel` subscriber at the
///   top of `main` — it owns the process' tracing (stderr fmt + OTLP export) and the
///   `log`-crate bridge, so `LogPlugin` could only lose the subscriber race: bevy 0.18
///   `error!`s "already set" and no-ops.
/// - `Some(window)`: that window, winit event loop as normal.
/// - `None`: offscreen — no window but the GPU ON (render-to-image), winit disabled.
///   Callers pace frames with their own `ScheduleRunnerPlugin` cadence (the one thing
///   the offscreen surfaces genuinely differ on).
pub fn base_plugins(window: Option<Window>) -> PluginGroupBuilder {
    // With LogPlugin gone, a caller that skipped `otel::init` would boot with NO
    // subscriber at all — every log silently dropped, the exact silent-fallback class
    // this repo bans. Fail at boot naming the fix instead.
    assert!(
        tracing::dispatcher::has_been_set(),
        "install the shared tracing subscriber (otel::init) before base_plugins — LogPlugin is disabled here"
    );
    let plugins = DefaultPlugins
        .build()
        .set(AssetPlugin {
            file_path: crate::assets::bevy_asset_path()
                .to_string_lossy()
                .into_owned(),
            ..default()
        })
        .disable::<bevy::log::LogPlugin>();
    match window {
        Some(window) => plugins.set(WindowPlugin {
            primary_window: Some(window),
            ..default()
        }),
        None => plugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>(),
    }
}
