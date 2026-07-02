//! The ONE bevy plugin-group recipe every rendered rl surface boots from.
//!
//! Was open-coded ×4 (GCR's windowed client, its offscreen screenshot scaffold, and
//! rl-demo's demo + render-to-image arms) and the copies disagreed on `LogPlugin` —
//! rl-demo disabled it, the `net` copies kept it. [`base_plugins`] resolves that once.

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
///   `log`-crate bridge, so `LogPlugin` can only lose the subscriber race: bevy 0.18
///   `error!`s "already set" and no-ops. The pre-dedup copies disagreed here; disabled
///   is the one answer (rl-demo's arms already shipped this way).
/// - `Some(window)`: that window, winit event loop as normal.
/// - `None`: offscreen — no window but the GPU ON (render-to-image), winit disabled.
///   Callers pace frames with their own `ScheduleRunnerPlugin` cadence (the one thing
///   the offscreen surfaces genuinely differ on).
pub fn base_plugins(window: Option<Window>) -> PluginGroupBuilder {
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
