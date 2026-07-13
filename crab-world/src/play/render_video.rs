use std::path::{Path, PathBuf};

use bevy::app::{AppExit, RunFixedMainLoop, RunFixedMainLoopSystems};
use bevy::prelude::*;
use bevy::time::{Fixed, Virtual};

use crate::bot::BotSet;
use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::screenshot::{self, ShotTarget};
use crate::training::reward::action_effort;

use super::ShotConfig;
use super::cameras::{spawn_offscreen_camera, track_offscreen_camera};
use super::policy::{add_inference, policy_step};
use super::target_ball::{spawn_target_ball, target_ball};

pub const VIDEO_FPS: u32 = (1.0 / crate::physics::PHYSICS_DT) as u32;

const VIDEO_SETTLE_FRAMES: u32 = 60;

pub struct RenderVideoPlugin {
    pub checkpoint_dir: PathBuf,
    pub path: PathBuf,
    pub seconds: f32,
    pub width: u32,
    pub height: u32,
    pub overrides: super::PlayOverrides,
}

#[derive(Resource)]
struct VideoConfig {
    out_path: PathBuf,
    frame_dir: PathBuf,
    settle: u32,
    total: u32,
}

#[derive(Resource, Default)]
struct VideoProgress {
    frames: u32,
    captured: u32,
    done: bool,
    encode_countdown: i32,
}

#[derive(Resource, Default)]
struct DriveStats {
    ticks: u64,
    sum_mean_abs: f64,
    sum_effort: f64,
}

impl Plugin for RenderVideoPlugin {
    fn build(&self, app: &mut App) {
        let total = ((self.seconds.max(0.0)) * VIDEO_FPS as f32)
            .round()
            .max(1.0) as u32;
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let frame_dir = parent.join(format!(".rl-video-frames-{}", std::process::id()));
        if let Err(e) = std::fs::create_dir_all(&frame_dir) {
            panic!("render-video: cannot create frame dir {frame_dir:?}: {e}");
        }

        add_inference(
            app,
            &self.checkpoint_dir,
            None,
            self.overrides.random_policy,
        );
        self.overrides.apply_rng_and_ball(app);
        app.add_plugins(crate::sky::NightSkyPlugin);
        app.insert_resource(ShotConfig {
            path: self.path.clone(),
            settle: 0,
            width: self.width,
            height: self.height,
        });
        app.insert_resource(VideoConfig {
            out_path: self.path.clone(),
            frame_dir,
            settle: VIDEO_SETTLE_FRAMES,
            total,
        })
        .init_resource::<VideoProgress>()
        .init_resource::<DriveStats>()
        .add_systems(FixedUpdate, policy_step.in_set(BotSet::Think))
        .add_systems(FixedUpdate, accumulate_drive_stats.after(BotSet::Think))
        .add_systems(Startup, (spawn_offscreen_camera, spawn_target_ball))
        .add_systems(FixedUpdate, target_ball.after(BotSet::Sense))
        .add_systems(
            Update,
            (track_offscreen_camera, capture_video_frame).chain(),
        )
        .add_systems(
            RunFixedMainLoop,
            step_one_tick.in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
        )
        .add_systems(Startup, pause_virtual_clock);
    }
}

fn pause_virtual_clock(mut time: ResMut<Time<Virtual>>) {
    time.pause();
}

fn step_one_tick(mut fixed: ResMut<Time<Fixed>>) {
    let dt = fixed.timestep();
    fixed.accumulate_overstep(dt);
}

fn accumulate_drive_stats(
    actions: Res<CrabActions>,
    progress: Res<VideoProgress>,
    cfg: Res<VideoConfig>,
    mut stats: ResMut<DriveStats>,
) {
    if progress.frames < cfg.settle {
        return;
    }
    let Some(a) = actions.envs.first() else {
        return;
    };
    let mean_abs = a.iter().map(|d| d.abs() as f64).sum::<f64>() / ACTION_SIZE as f64;
    stats.sum_mean_abs += mean_abs;
    stats.sum_effort += action_effort(a) as f64;
    stats.ticks += 1;
}

fn video_seq_path(dir: &Path, n: u32) -> PathBuf {
    dir.join(format!("frame_{n:05}.png"))
}

fn capture_video_frame(
    mut commands: Commands,
    cfg: Res<VideoConfig>,
    target: Res<ShotTarget>,
    stats: Res<DriveStats>,
    mut progress: ResMut<VideoProgress>,
    mut exit: MessageWriter<AppExit>,
) {
    if progress.done {
        progress.encode_countdown -= 1;
        if progress.encode_countdown <= 0 {
            report_drive_stats(&stats);
            encode(&cfg);
            exit.write(AppExit::Success);
        }
        return;
    }

    progress.frames += 1;
    if progress.frames <= cfg.settle {
        return;
    }

    let n = progress.captured;
    screenshot::save_target_to(&mut commands, &target, video_seq_path(&cfg.frame_dir, n));
    progress.captured += 1;

    if progress.captured >= cfg.total {
        info!(
            "render-video: captured {} frames into {:?}, encoding…",
            progress.captured, cfg.frame_dir
        );
        progress.done = true;
        progress.encode_countdown = 30;
    }
}

fn report_drive_stats(stats: &DriveStats) {
    let n = stats.ticks.max(1) as f64;
    eprintln!(
        "DRIVE_STATS ticks={} mean_abs_drive={:.5} mean_effort={:.5}",
        stats.ticks,
        stats.sum_mean_abs / n,
        stats.sum_effort / n,
    );
}

fn encode(cfg: &VideoConfig) {
    let status = std::process::Command::new("ffmpeg")
        .arg("-y")
        .args(["-framerate", &VIDEO_FPS.to_string()])
        .args([
            "-i",
            &cfg.frame_dir.join("frame_%05d.png").to_string_lossy(),
        ])
        .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&cfg.out_path)
        .status();
    match status {
        Ok(s) if s.success() => {
            info!("render-video: wrote {:?}", cfg.out_path);
            if let Err(e) = std::fs::remove_dir_all(&cfg.frame_dir) {
                warn!(
                    "render-video: could not remove scratch {:?}: {e}",
                    cfg.frame_dir
                );
            }
        }
        Ok(s) => error!(
            "render-video: ffmpeg exited {s}; frames kept at {:?} for manual encode",
            cfg.frame_dir
        ),
        Err(e) => error!(
            "render-video: could not run ffmpeg ({e}); frames kept at {:?}",
            cfg.frame_dir
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_fps_tracks_physics_and_frame_count_is_seconds_times_fps() {
        assert_eq!(
            VIDEO_FPS,
            (1.0 / crate::physics::PHYSICS_DT) as u32,
            "playback rate must equal the sim tick rate so motion is true speed"
        );
        let ticks = |s: f32| ((s.max(0.0)) * VIDEO_FPS as f32).round().max(1.0) as u32;
        assert_eq!(ticks(1.0), VIDEO_FPS);
        assert_eq!(
            ticks(0.0),
            1,
            "a zero/negative length still renders ≥1 frame"
        );
        assert_eq!(ticks(2.5), (2.5 * VIDEO_FPS as f32).round() as u32);
    }

    #[test]
    fn video_frame_names_zero_pad_and_sort() {
        let dir = Path::new("/tmp/clip");
        assert_eq!(video_seq_path(dir, 7), dir.join("frame_00007.png"));
        assert_eq!(video_seq_path(dir, 12345), dir.join("frame_12345.png"));
        let a = video_seq_path(dir, 2).to_string_lossy().into_owned();
        let b = video_seq_path(dir, 10).to_string_lossy().into_owned();
        assert!(
            a < b,
            "padded frame 2 must sort before frame 10 ({a} vs {b})"
        );
    }
}
