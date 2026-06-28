//! Shared offscreen render-to-PNG-on-settle primitive.
//!
//! Two headless screenshot subsystems sit on top of this: the trained-crab
//! inspection shot ([`crate::play::ScreenshotPlugin`]) and the first-person
//! game-view shot (`net::render::build_screenshot_app`). Both compose a
//! DIFFERENT scene and own their camera differently (a crab-tracking orbit camera
//! vs the FP camera driven by `apply_transforms`), so the camera-spawn and the
//! per-frame capture stay as their own systems. What they share — and what was
//! copied verbatim in both, the classic drift hazard — is the mechanics underneath:
//! the COPY_SRC render-target image, the settle→capture→exit-countdown bookkeeping,
//! and the save-to-disk spawn. That lives here once.
//!
//! Render-only, so nothing here touches sim/training determinism.

use std::path::PathBuf;

use bevy::app::AppExit;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};

/// The render-target image the offscreen camera draws into; the screenshot
/// machinery reads it back to encode the PNG. Inserted by each subsystem's
/// camera-spawn after it creates the target via [`new_render_target`].
#[derive(Resource)]
pub struct ShotTarget(pub Handle<Image>);

/// Progress through the settle→capture→exit cycle, advanced one render frame at a
/// time by [`advance_capture`].
#[derive(Resource, Default)]
pub struct ShotProgress {
    /// Render frames elapsed — NOT physics steps. The GPU pipeline renders black for
    /// the first few dozen frames (shaders/pipelines compiling, assets uploading), so
    /// settle is counted in rendered frames. `pub(crate)` because the crab shot's
    /// sibling systems (pose sweep, toss) read it to phase off the settle frame.
    pub(crate) frames: u32,
    /// `None` until the shot is captured; then `Some(n)` counts down so the async GPU
    /// readback + PNG encode finish before [`AppExit`]. Folding "captured?" and "how
    /// many frames left" into one `Option` makes the contradictory states (counting
    /// down but not captured; captured but no countdown) unrepresentable — and these
    /// internals stay private, since only [`advance_capture`]/[`finish_capture`] touch
    /// them.
    exit_countdown: Option<u32>,
}

/// Frames to keep running after the capture is spawned, so the async GPU readback
/// and PNG encode complete before [`AppExit`].
const EXIT_COUNTDOWN_FRAMES: u32 = 30;

/// Create an offscreen render-target image of the given size, marked `COPY_SRC` so
/// the screenshot machinery can read the rendered texture back. The caller adds it
/// to `Assets<Image>`, spawns its camera with `RenderTarget::Image(handle)` (and
/// whatever scene-specific markers/clear color it needs), and stashes the handle in
/// [`ShotTarget`].
pub fn new_render_target(width: u32, height: u32) -> Image {
    let mut image = Image::new_target_texture(width, height, TextureFormat::bevy_default(), None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image
}

/// Tick the shared settle→capture→exit bookkeeping one render frame (the part both
/// shots copied). Drives the post-capture exit countdown — emitting [`AppExit`] when
/// it runs out — and otherwise counts this frame. Returns `Some(frame)` (the
/// render-frame index, ≥ `settle`) on the frame the scene has settled enough to
/// shoot, and `None` while warming up or already captured. The caller does its own
/// scene-specific capture on `Some` then calls [`finish_capture`]; the crab shot
/// subtracts `settle` from `frame` to phase its video sweep. Ticks a counter and can
/// terminate the app, so call it exactly once per frame.
pub fn advance_capture(
    progress: &mut ShotProgress,
    settle: u32,
    exit: &mut MessageWriter<AppExit>,
) -> Option<u32> {
    if let Some(countdown) = &mut progress.exit_countdown {
        *countdown = countdown.saturating_sub(1);
        if *countdown == 0 {
            exit.write(AppExit::Success);
        }
        return None;
    }
    progress.frames += 1;
    (progress.frames >= settle).then_some(progress.frames)
}

/// Spawn the one-shot `Screenshot` of the render target, observed by `save_to_disk`,
/// to write `path` once the GPU readback lands. The shared half of every capture;
/// the caller chooses the path (a single shot, or a numbered video frame).
pub fn save_target_to(commands: &mut Commands, target: &ShotTarget, path: PathBuf) {
    commands
        .spawn(Screenshot::image(target.0.clone()))
        .observe(save_to_disk(path));
}

/// Mark the capture done and arm the exit countdown. After this, [`advance_capture`]
/// counts down and then exits, returning `None` every frame until it does.
pub fn finish_capture(progress: &mut ShotProgress) {
    progress.exit_countdown = Some(EXIT_COUNTDOWN_FRAMES);
}
