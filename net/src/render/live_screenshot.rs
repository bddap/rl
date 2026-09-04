//! In-game screenshot (rl#405): both sticks clicked (L3+R3) on the pad, F12 on the
//! keyboard. Steam's own screenshot chords don't reach the TV's nested-gamescope
//! setup, so the game captures its own frame: bevy's `Screenshot` readback — the one
//! capture machinery, same as the fp-screenshot evidence path — written as PNG at
//! native window res into the host's screenshots dir, with a HUD flash and an OTLP
//! line naming the file so tooling can fetch it.
//!
//! Input is the bottleneck (rl#381): the GPU readback is async by construction, and
//! the PNG encode + disk write run on a spawned thread — unlike the evidence apps'
//! stock `save_to_disk`, which encodes in the observer and would stall the frame the
//! shot lands on. The trigger frame does no I/O beyond `create_dir_all`.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};

use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};

/// Matches `otel::SCREENSHOT_TARGET` (tracing macros want a literal; tied by test) —
/// forced past quiet RUST_LOG defaults so the export line always ships.
const TARGET: &str = "screenshot";

/// How long the confirmation line stays up.
const FLASH_SECS: f32 = 2.5;

pub(super) fn install(app: &mut App) {
    app.init_resource::<ShotUx>();
    app.add_systems(Startup, spawn_hud);
    // Not gated on `AppPhase::Playing`: a menu shot is as legitimate as a gameplay
    // one, and the systems idle at a few button reads per frame.
    app.add_systems(Update, (trigger_screenshot, update_hud).chain());
}

#[derive(Resource)]
struct ShotUx {
    /// Writer threads report the written path (or the failure) here; `update_hud`
    /// polls it so the flash never claims a file that didn't land. The Mutex is only
    /// for Sync (`Receiver` isn't; a `Resource` must be) — one system ever reads it.
    tx: Sender<Result<PathBuf, String>>,
    rx: Mutex<Receiver<Result<PathBuf, String>>>,
    /// Transient confirmation line + time left to show it.
    flash: Option<(String, f32)>,
}

impl Default for ShotUx {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self {
            tx,
            rx: Mutex::new(rx),
            flash: None,
        }
    }
}

#[derive(Component)]
struct ShotHud;

/// `$RL_SCREENSHOT_DIR`, else `screenshots/` under the per-save data dir — per host
/// by construction (this host's local disk). `None` means no HOME at all.
fn shot_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("RL_SCREENSHOT_DIR") {
        return Some(PathBuf::from(dir));
    }
    Some(super::data_dir()?.join("screenshots"))
}

fn chord_hit(keys: &ButtonInput<KeyCode>, gamepads: &Query<&Gamepad>) -> bool {
    if keys.just_pressed(KeyCode::F12) {
        return true;
    }
    // Edge-triggered on either stick so a held chord fires exactly once. L3 alone is
    // the Sprint hold; the click-both-sticks chord costs at most a brief sprint blip.
    gamepads.iter().any(|gp| {
        use GamepadButton::{LeftThumb, RightThumb};
        (gp.just_pressed(LeftThumb) && gp.pressed(RightThumb))
            || (gp.pressed(LeftThumb) && gp.just_pressed(RightThumb))
    })
}

fn trigger_screenshot(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut ux: ResMut<ShotUx>,
) {
    if !chord_hit(&keys, &gamepads) {
        return;
    }
    let Some(dir) = shot_dir() else {
        ux.flash = Some(("screenshot: no HOME to write to".into(), FLASH_SECS));
        return;
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!("screenshot: create {} failed: {e}", dir.display());
        ux.flash = Some((format!("screenshot failed: {e}"), FLASH_SECS));
        return;
    }
    let path = dir.join(format!(
        "shot-{}.png",
        chrono::Local::now().format("%Y%m%d-%H%M%S%.3f")
    ));
    let tx = ux.tx.clone();
    commands.spawn(Screenshot::primary_window()).observe(
        move |captured: On<ScreenshotCaptured>| {
            let image = captured.image.clone();
            let path = path.clone();
            let tx = tx.clone();
            std::thread::Builder::new()
                .name("screenshot-write".into())
                .spawn(move || {
                    let _ = tx.send(write_png(image, path));
                })
                .expect("spawn screenshot-write thread");
        },
    );
}

fn write_png(image: bevy::image::Image, path: PathBuf) -> Result<PathBuf, String> {
    let dynamic = image
        .try_into_dynamic()
        .map_err(|e| format!("unsupported frame format: {e}"))?;
    // Drop alpha: with HDR it stores brightness, not opacity (same as bevy's
    // `save_to_disk`).
    dynamic
        .to_rgb8()
        .save(&path)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

fn update_hud(
    time: Res<Time>,
    mut ux: ResMut<ShotUx>,
    mut hud: Query<(&mut Text, &mut Visibility), With<ShotHud>>,
) {
    while let Ok(outcome) = ux.rx.get_mut().unwrap().try_recv() {
        match outcome {
            Ok(path) => {
                // The line tooling fetches by: ships over OTLP and stays on stderr.
                tracing::info!(target: TARGET, path = %path.display(), "screenshot saved");
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                // Plain ASCII: the bundled font has no emoji, a camera glyph is tofu.
                ux.flash = Some((format!("saved {name}"), FLASH_SECS));
            }
            Err(e) => {
                warn!("screenshot failed: {e}");
                ux.flash = Some((format!("screenshot failed: {e}"), FLASH_SECS));
            }
        }
    }
    if let Some((_, left)) = &mut ux.flash {
        *left -= time.delta_secs();
        if *left <= 0.0 {
            ux.flash = None;
        }
    }
    let Ok((mut text, mut vis)) = hud.single_mut() else {
        return;
    };
    match &ux.flash {
        Some((line, _)) => {
            if text.0 != *line {
                text.0 = line.clone();
            }
            *vis = Visibility::Visible;
        }
        None => *vis = Visibility::Hidden,
    }
}

fn spawn_hud(mut commands: Commands) {
    // A full-width transparent absolute strip so the text centers itself.
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(48.0),
            left: Val::Px(0.0),
            width: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            ..default()
        })
        .with_children(|p| {
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(18.0),
                    ..default()
                },
                TextColor(Color::srgb(0.55, 0.95, 0.55)),
                Visibility::Hidden,
                ShotHud,
            ));
        });
}

#[cfg(test)]
mod tests {
    #[test]
    fn target_matches_otel() {
        assert_eq!(otel::SCREENSHOT_TARGET, super::TARGET);
    }
}
