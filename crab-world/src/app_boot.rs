//! The ONE bevy plugin-group recipe every rendered rl surface boots from (GCR's windowed
//! client, its offscreen screenshot scaffold, rl-demo's demo + render-to-image arms).
//!
//! Sibling recipe: [`crate::bot::headless::headless_stack`] is the SIM-world counterpart
//! (GPU off, no asset root, compiles with render OFF) — a window/winit change here may
//! need mirroring there.

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

/// The physical height every UI size in this workspace is authored against (the Steam
/// Deck's 800 rows).
const UI_REFERENCE_HEIGHT: f32 = 800.0;

/// The ONE UI-size rule (rl#227): authored px render at `physical_height / 800` of
/// their authored size, regardless of the compositor's reported scale factor.
///
/// bevy renders UI at `authored × UiScale × scale_factor`, so returning
/// `physical_height / (800 × scale_factor)` cancels the factor — a Deck whose
/// compositor reports 2× draws the same physical UI as one reporting 1×, and a 4K TV
/// scales everything up ~2.7× instead of drawing desktop-sized specks. (This also
/// cancels deliberate OS-level UI scaling; on this fleet's fixed screens that's the
/// point, but it is a tradeoff.) The clamp bounds the PHYSICAL ratio, before the
/// cancellation, so the bounds don't drift with the reported factor either.
pub fn ui_scale_for(window: &Window) -> f32 {
    let sf = window.resolution.scale_factor();
    (window.resolution.physical_height() as f32 / UI_REFERENCE_HEIGHT).clamp(0.5, 3.0) / sf
}

/// Keeps [`UiScale`] synced to [`ui_scale_for`] on the primary window. Part of
/// [`base_plugins`] for every windowed surface; offscreen surfaces have no window, so
/// their render-to-image UI stays at authored size.
pub struct UiScalePlugin;

impl Plugin for UiScalePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, sync_ui_scale);
    }
}

fn sync_ui_scale(
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut ui_scale: ResMut<UiScale>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let scale = ui_scale_for(window);
    if (ui_scale.0 - scale).abs() > 1e-3 {
        ui_scale.0 = scale;
    }
}

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
        Some(window) => plugins
            .set(WindowPlugin {
                primary_window: Some(window),
                ..default()
            })
            .add(UiScalePlugin),
        None => plugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>(),
    }
}

#[cfg(test)]
mod tests {
    use super::ui_scale_for;
    use bevy::window::{Window, WindowResolution};

    fn window(physical_w: u32, physical_h: u32, sf: f32) -> Window {
        Window {
            resolution: WindowResolution::new(physical_w, physical_h)
                .with_scale_factor_override(sf),
            ..Default::default()
        }
    }

    #[test]
    fn ui_scale_is_physical_proportional_and_sf_invariant() {
        // What lands on screen is ui_scale × scale_factor; that product must depend
        // only on the physical resolution.
        let physical = |w: &Window| ui_scale_for(w) * w.resolution.scale_factor();
        assert_eq!(physical(&window(1280, 800, 1.0)), 1.0); // Deck at 1×
        assert_eq!(physical(&window(1280, 800, 2.0)), 1.0); // Deck reporting 2× — same physical UI
        assert_eq!(physical(&window(3840, 2160, 1.0)), 2.7); // 4K TV
        assert_eq!(physical(&window(400, 250, 1.0)), 0.5); // clamp floor
        assert_eq!(physical(&window(400, 250, 2.5)), 0.5); // clamp floor is sf-invariant too
        assert_eq!(physical(&window(6400, 4000, 1.0)), 3.0); // clamp ceiling
    }
}
