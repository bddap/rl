//! Offline ball-chase video render: drive the trained crab after the moving target
//! headless, capture every sim tick to a PNG, and encode the sequence to an mp4 with
//! `ffmpeg`. The render half that produces an owner-facing motion clip — the still
//! [`super::ScreenshotPlugin`] can't show how frantic the gait is.
//!
//! It reuses the demo's exact pieces so there is ONE render path and ONE chase loop,
//! not a video-only fork: the offscreen render-to-image setup
//! ([`super::cameras::spawn_offscreen_camera`] + [`track_offscreen_camera`]), the policy
//! driver ([`policy_step`]), and the ball-chase loop ([`spawn_target_ball`] +
//! [`target_ball`]). What's unique here: the sim is stepped **exactly one tick per
//! rendered frame, decoupled from wall-clock** (see [`step_one_tick`]) — per-frame
//! policy+render is slower than realtime, so a wall-clock-paced sim would stutter — every
//! frame is captured, and at the end the frames are encoded.
//!
//! It also dumps an objective actuation-magnitude summary (mean |drive| and the
//! `Σ|dᵢ|²` effort the reward penalizes) over the captured window, so two policies can
//! be compared numerically — "is this gait calmer" answered by a number, not an eyeball.

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

/// Playback frame rate of the encoded mp4. The render captures one frame per SIM tick
/// (physics runs at [`crate::physics::PHYSICS_DT`]⁻¹), and we hand ffmpeg this as the
/// input frame rate, so the clip plays back at the rate the sim ticked — motion at true
/// speed, not sped up or slowed.
pub const VIDEO_FPS: u32 = (1.0 / crate::physics::PHYSICS_DT) as u32;

/// Render-video settle: frames rendered (and stepped) before capture + drive-stat
/// accounting start. A windowless render's GPU pipeline renders black for its first few
/// dozen frames while shaders/pipelines compile and assets upload, and the fresh crab
/// needs to drop and take load first — both the concerns the screenshot's settle handles.
/// Stepped the same as captured frames, so the crab is already chasing by clip frame 0.
const VIDEO_SETTLE_FRAMES: u32 = 60;

/// Render the demo's ball-chase headless to an mp4, then exit.
pub struct RenderVideoPlugin {
    pub checkpoint_dir: PathBuf,
    pub path: PathBuf,
    /// Clip length in SIMULATED seconds; ticks = `seconds` × [`VIDEO_FPS`].
    pub seconds: f32,
    pub width: u32,
    pub height: u32,
}

/// Render-video run config. `frame_dir` is a scratch dir for the numbered PNGs ffmpeg
/// globs; `settle` frames are rendered/stepped but neither captured nor counted in the
/// drive stats; `total` is the captured-frame target.
#[derive(Resource)]
struct VideoConfig {
    out_path: PathBuf,
    frame_dir: PathBuf,
    settle: u32,
    total: u32,
}

#[derive(Resource, Default)]
struct VideoProgress {
    /// App updates seen so far (each = one sim tick + one render).
    frames: u32,
    /// Frames actually written to disk (= app updates past `settle`).
    captured: u32,
    /// Set once all frames are captured; then we wait `encode_countdown` frames for the
    /// final GPU readbacks/PNG writes to land before encoding + exiting.
    done: bool,
    encode_countdown: i32,
}

/// Objective actuation summary over the captured window — the normalizer-robust
/// "franticness" proxy. `mean |drive|` is the average per-joint torque command magnitude;
/// `effort` is the `Σ|dᵢ|²` the reward taxes (via [`action_effort`]). Accumulated per tick
/// past the settle and reported at the end, so two checkpoints rendered the same way can be
/// compared by number.
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
        // Scratch dir for the PNG sequence, beside the mp4 (per-run by pid so two renders
        // never collide). ffmpeg globs `frame_%05d.png` out of it.
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let frame_dir = parent.join(format!(".rl-video-frames-{}", std::process::id()));
        // Create the scratch dir up front (and fail fast if we can't) rather than mid-
        // render — a render that reaches the first capture only to find it can't write
        // frames wastes the whole settle.
        if let Err(e) = std::fs::create_dir_all(&frame_dir) {
            panic!("render-video: cannot create frame dir {frame_dir:?}: {e}");
        }

        add_inference(app, &self.checkpoint_dir, None);
        // Night-sky skybox behind the captured frames (same sky as the windowed demo).
        app.add_plugins(crate::sky::NightSkyPlugin);
        // Reuse the screenshot's offscreen-camera spawn, which reads ShotConfig for its
        // size. path/settle are inert here (this path runs neither single-shot capture nor
        // the screenshot's settle counter — the video has its own VideoConfig).
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
        // Seeds target_ball's relocation RNG; RL_DEMO_SEED makes the rendered clip reproducible.
        .init_resource::<super::DemoRng>()
        .add_systems(FixedUpdate, policy_step.in_set(BotSet::Think))
        // Read the actions the policy just wrote, after Think, before they're consumed —
        // the objective effort accounting.
        .add_systems(FixedUpdate, accumulate_drive_stats.after(BotSet::Think))
        // The ball-chase loop — the SAME systems the demo runs (seed → drive → teleport on
        // reach). `target_ball` after Sense so it reads the post-physics state the
        // observation consumed; the policy then sees the relocated target next tick.
        .add_systems(Startup, (spawn_offscreen_camera, spawn_target_ball))
        .add_systems(FixedUpdate, target_ball.after(BotSet::Sense))
        .add_systems(
            Update,
            (track_offscreen_camera, capture_video_frame).chain(),
        )
        // Decouple the sim from wall-clock: inject exactly one fixed tick of overstep per
        // app update, BEFORE the fixed-main loop consumes it. With virtual time paused at
        // startup this is the ONLY thing that advances FixedUpdate, so the sim steps exactly
        // one tick per rendered frame regardless of how long that frame took.
        .add_systems(
            RunFixedMainLoop,
            step_one_tick.in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
        )
        .add_systems(Startup, pause_virtual_clock);
    }
}

/// Startup: pause virtual time so it stops tracking wall-clock. The render then advances
/// the sim itself, one fixed tick per frame, via [`step_one_tick`] — the decoupling that
/// keeps a slower-than-realtime render from stuttering or dropping frames.
fn pause_virtual_clock(mut time: ResMut<Time<Virtual>>) {
    time.pause();
}

/// System (`RunFixedMainLoop`, BeforeFixedMainLoop): hand the fixed-time accumulator
/// exactly one timestep of overstep so the fixed schedule runs the Sense→Think→Act +
/// physics loop exactly once this app update. Virtual time is paused (so it contributes no
/// overstep of its own), making this the sole driver of the sim clock — one sim tick per
/// rendered frame, independent of wall-clock.
fn step_one_tick(mut fixed: ResMut<Time<Fixed>>) {
    let dt = fixed.timestep();
    fixed.accumulate_overstep(dt);
}

/// System (FixedUpdate, after Think): accumulate the per-tick actuation magnitude of env 0
/// into [`DriveStats`], skipping the settle ticks. Reads the actions the policy wrote this
/// tick (the same `[f32; ACTION_SIZE]` the actuator applies and the reward taxes).
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

/// `dir/frame_00007.png` — zero-padded names ffmpeg globs into the clip in order.
fn video_seq_path(dir: &Path, n: u32) -> PathBuf {
    dir.join(format!("frame_{n:05}.png"))
}

/// System (Update, after `track_offscreen_camera`): the video capture pump. Each app
/// update (= one sim tick) past the settle, snapshot the offscreen target to the next
/// numbered PNG. Once `total` frames are written, wait a few frames for the final GPU
/// readbacks/encodes to land, then encode + exit. Paced by [`step_one_tick`], not
/// wall-clock, so it runs as fast as the machine renders.
fn capture_video_frame(
    mut commands: Commands,
    cfg: Res<VideoConfig>,
    target: Res<ShotTarget>,
    stats: Res<DriveStats>,
    mut progress: ResMut<VideoProgress>,
    mut exit: MessageWriter<AppExit>,
) {
    if progress.done {
        // All frames queued; let the last readbacks/PNG writes finish, then encode.
        progress.encode_countdown -= 1;
        if progress.encode_countdown <= 0 {
            report_drive_stats(&stats);
            encode(&cfg);
            exit.write(AppExit::Success);
        }
        return;
    }

    progress.frames += 1;
    // Settle frames are stepped (the sim advances) but not recorded.
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
        // Bevy's screenshot save is deferred (GPU readback → PNG on a later frame); give
        // the last few writes time to flush before ffmpeg reads the dir.
        progress.encode_countdown = 30;
    }
}

/// Print the objective actuation summary on a stable, greppable line. The mean per-joint
/// torque magnitude and the `Σ|dᵢ|²` effort the reward taxes, averaged over the captured
/// (post-settle) ticks — the franticness proxy that doesn't depend on the obs normalizer.
fn report_drive_stats(stats: &DriveStats) {
    let n = stats.ticks.max(1) as f64;
    eprintln!(
        "DRIVE_STATS ticks={} mean_abs_drive={:.5} mean_effort={:.5}",
        stats.ticks,
        stats.sum_mean_abs / n,
        stats.sum_effort / n,
    );
}

/// Shell out to ffmpeg to encode the captured PNG sequence into the mp4, then clean up the
/// scratch frames. ffmpeg is the project's sanctioned encoder (on PATH via nix); shelling
/// out keeps a heavyweight codec dependency out of the binary. `yuv420p` + even-dimension
/// padding for broad player/H.264 compatibility (odd dimensions are a classic silent encode
/// failure). On any failure we keep the frames and log loudly so the clip can be salvaged.
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

    /// The video plays back at the sim's own tick rate, so a clip's frame count is
    /// `seconds × tick_hz` and its duration is honest crab time. Pin both the rate (= the
    /// physics rate, not a hardcoded 30/60) and the rounding the plugin uses.
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

    /// Frame names must zero-pad to a fixed width and sort in capture order, or ffmpeg's
    /// `%05d` glob reorders/loses frames (e.g. `frame_2` before `frame_10`).
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
