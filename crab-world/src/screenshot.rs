use std::path::PathBuf;

use bevy::app::AppExit;
use bevy::camera::RenderTarget;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};

#[derive(Resource)]
pub struct ShotTarget(pub Handle<Image>);

#[derive(Resource, Default)]
pub struct ShotProgress {
    pub(crate) frames: u32,
    exit_countdown: Option<u32>,
}

const EXIT_COUNTDOWN_FRAMES: u32 = 30;

pub const DEFAULT_WIDTH: u32 = 1280;
pub const DEFAULT_HEIGHT: u32 = 720;

pub fn new_render_target(width: u32, height: u32) -> Image {
    let mut image = Image::new_target_texture(width, height, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image
}

pub fn offscreen_camera_bundle(target: Handle<Image>) -> impl Bundle {
    (
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(crate::sky::NIGHT_CLEAR),
            ..default()
        },
        RenderTarget::Image(target.into()),
        Tonemapping::None,
        bevy::ui::IsDefaultUiCamera,
    )
}

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

pub fn save_target_to(commands: &mut Commands, target: &ShotTarget, path: PathBuf) {
    commands
        .spawn(Screenshot::image(target.0.clone()))
        .observe(save_to_disk(path));
}

pub fn finish_capture(progress: &mut ShotProgress) {
    progress.exit_countdown = Some(EXIT_COUNTDOWN_FRAMES);
}
