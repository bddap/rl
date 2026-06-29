//! Interactive play + screenshot modes.
//!
//! `DemoPlugin` loads a trained checkpoint and drives the crab with the policy
//! (deterministic, no learning) behind an orbit camera with poke/reset controls
//! — the "launch it and play" experience.
//!
//! `ScreenshotPlugin` renders one frame to a PNG and exits. It runs windowless
//! but with the GPU on (render-to-image), so the trained crab can be inspected
//! without a display or a human in the loop.
//!
//! [`RenderVideoPlugin`] is the moving-clip cousin of the screenshot: it drives the
//! ball-chase headless, captures every sim tick, and encodes an mp4 — for showing how the
//! gait MOVES, which a still can't.
//!
//! These plugins live here as the wiring core; the systems they schedule are carved into
//! focused submodules ([`policy`], [`render_video`], [`hot_reload`], [`manual_control`],
//! [`cameras`], [`target_ball`], [`demo`], [`controls`]).

mod cameras;
mod controls;
mod demo;
mod hot_reload;
mod manual_control;
mod policy;
mod render_video;
mod rig_pose;
mod target_ball;

use std::path::PathBuf;

use bevy::prelude::*;
use bevy_rapier3d::plugin::PhysicsSet;

use crate::bot::BotSet;
use crate::bot::body::CrabCarapace;
use crate::screenshot::{self, ShotProgress, ShotTarget};

pub use policy::Policy;
pub use policy::{RigDims, RigFit, checkpoint_digest, checkpoint_fits_rig};
pub use render_video::RenderVideoPlugin;

/// The [`RigDims`] this binary's crab rig compiles to — the spec a checkpoint must satisfy
/// to drive the NN crab. The rig side of the fit check [`checkpoint_fits_rig`]
/// performs, exposed so the gate can name both sides; "does this checkpoint fit this
/// binary?" is answered by the binary itself, never a hand-maintained number that could drift.
pub fn rig_dims() -> RigDims {
    RigDims {
        obs: crate::bot::sensor::OBS_SIZE,
        action: crate::bot::actuator::ACTION_SIZE,
    }
}

use cameras::{orbit_camera, spawn_offscreen_camera, spawn_orbit_camera, track_offscreen_camera};
use controls::DemoControls;
use demo::{DemoSettle, PokeBurst, demo_controls, demo_poke, demo_settle};
use hot_reload::hot_reload_policy;
use manual_control::{ManualControl, manual_control_step, spawn_manual_hud};
use policy::{add_inference, policy_step};
use target_ball::{spawn_target_ball, target_ball};

// ---------------------------------------------------------------------------
// Demo: windowed, interactive
// ---------------------------------------------------------------------------

/// Windowed "play with the trained crab" mode.
pub struct DemoPlugin {
    pub checkpoint_dir: PathBuf,
    pub live_checkpoint_dir: Option<PathBuf>,
    /// Start in hands-on gamepad control instead of the policy (toggle live with
    /// Y). See [`manual_control`] — a physics feel-test, not a learned driver.
    pub manual_control: bool,
}

impl Plugin for DemoPlugin {
    fn build(&self, app: &mut App) {
        add_inference(app, &self.checkpoint_dir, self.live_checkpoint_dir.clone());
        crate::player::graph::register(app);
        // Night-sky skybox behind the orbit view.
        app.add_plugins(crate::sky::NightSkyPlugin);
        // The reusable controls overlay (corner hint + hold-to-reveal panel), driven by the
        // demo's own DEMO_CONTROL_MAP — replaces the old static bottom-left HUD text.
        app.add_plugins(crate::controls::ControlsOverlayPlugin::<DemoControls>::default());
        app.init_resource::<DemoSettle>()
            .init_resource::<PokeBurst>()
            .add_systems(
                Startup,
                (
                    spawn_orbit_camera,
                    spawn_target_ball,
                    crate::build_info::spawn_build_info_overlay,
                ),
            )
            .add_systems(Update, (orbit_camera, demo_controls))
            .add_systems(
                FixedUpdate,
                (
                    // Settle zeroes the actions a fresh crab holds, so it must land
                    // before Act applies them (not merely after Think) — otherwise the
                    // crab takes a tick of stale policy torque mid-drop. The
                    // Sense→Think→Act chain alone leaves settle-vs-Act to Bevy's
                    // implicit conflict ordering; pin it.
                    demo_settle.after(BotSet::Think).before(BotSet::Act),
                    demo_poke.after(BotSet::Act).before(PhysicsSet::SyncBackend),
                    // After Sense so it reads the same post-physics claw-tip and target
                    // state the observation just consumed — touch detection and the ball
                    // then agree with what the policy saw this tick.
                    target_ball.after(BotSet::Sense),
                ),
            );
        // Both drivers are always present; `ManualControl.active` (seeded by the
        // flag, toggled live with Y) decides which one writes the actions each tick.
        app.insert_resource(ManualControl {
            active: self.manual_control,
            selected: None,
        })
        .add_systems(Startup, spawn_manual_hud)
        .add_systems(
            FixedUpdate,
            (policy_step, manual_control_step).in_set(BotSet::Think),
        )
        .add_systems(Update, hot_reload_policy);
    }
}

// ---------------------------------------------------------------------------
// Screenshot: windowless render-to-PNG
// ---------------------------------------------------------------------------

/// Renders one frame to a PNG after the crab settles, then exits.
pub struct ScreenshotPlugin {
    pub checkpoint_dir: PathBuf,
    pub path: PathBuf,
    pub settle: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Resource)]
pub(in crate::play) struct ShotConfig {
    path: PathBuf,
    settle: u32,
    pub(in crate::play) width: u32,
    pub(in crate::play) height: u32,
}

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        add_inference(app, &self.checkpoint_dir, None);
        app.add_systems(FixedUpdate, policy_step.in_set(BotSet::Think));
        // Night-sky skybox behind the captured frame (same sky as the windowed demo).
        app.add_plugins(crate::sky::NightSkyPlugin);
        // RL_RIG_POSE: drive the chelipeds to their shoulder stop with the body pinned, so
        // a rig limit/axis change can be inspected headless in the offending pose. Inert by
        // default — a plain screenshot is unchanged. See [`rig_pose`].
        if let Some(pose) = rig_pose::rig_pose_from_env() {
            app.insert_resource(pose)
                .init_resource::<rig_pose::RigPosePin>()
                .add_systems(
                    FixedUpdate,
                    rig_pose::rig_pose_drive.in_set(BotSet::Think).after(policy_step),
                )
                .add_systems(
                    FixedUpdate,
                    rig_pose::rig_pose_pin
                        .after(BotSet::Act)
                        .before(PhysicsSet::SyncBackend),
                );
        }
        // RL_TARGET_BALL=1: render the demo's red target ball in the screenshot too,
        // off the same `CrabTargets` state, so the reach-target viz can be inspected
        // headless. `target_ball` seeds (honoring RL_TARGET_BALL_AT), drives, and
        // teleports the target exactly as in the demo. Inert by default — a plain
        // screenshot is unchanged.
        if std::env::var_os("RL_TARGET_BALL").is_some() {
            app.add_systems(Startup, spawn_target_ball)
                .add_systems(FixedUpdate, target_ball.after(BotSet::Sense));
        }
        app.insert_resource(ShotConfig {
            path: self.path.clone(),
            settle: self.settle,
            width: self.width,
            height: self.height,
        })
        .init_resource::<ShotProgress>()
        .add_systems(Startup, spawn_offscreen_camera)
        .add_systems(
            Update,
            (track_offscreen_camera, capture_when_settled).chain(),
        );
        // Controls overlay on the screenshot path too, so an evidence frame can prove the
        // demo's hold-to-reveal legend renders. The convenience plugin spawns the UI + seeds
        // `ForceRevealControls(false)`; the shared env override (see
        // [`crate::controls::reveal_overrides_from_env`]) opens it for the headless shot.
        let (force_reveal, active_device, active_context) =
            crate::controls::reveal_overrides_from_env::<DemoControls>();
        app.add_plugins(crate::controls::ControlsOverlayPlugin::<DemoControls>::default())
            .insert_resource(force_reveal)
            .insert_resource(active_device)
            .insert_resource(active_context);
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_when_settled(
    mut commands: Commands,
    cfg: Res<ShotConfig>,
    target: Res<ShotTarget>,
    mut progress: ResMut<ShotProgress>,
    mut exit: MessageWriter<AppExit>,
    carapace_q: Query<&Transform, With<CrabCarapace>>,
    meshes_q: Query<(), With<Mesh3d>>,
    lights_q: Query<(), With<DirectionalLight>>,
    cams_q: Query<&Camera>,
) {
    // settle counted in RENDER frames (the GPU pipeline renders black for the first
    // few dozen frames while shaders/pipelines compile and assets upload); the shared
    // helper owns that gate + the post-capture exit countdown.
    let Some(frame) = screenshot::advance_capture(&mut progress, cfg.settle, &mut exit) else {
        return;
    };

    let carapace = carapace_q
        .single()
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);
    debug!(
        "screenshot scene: carapace={carapace:?} meshes={} lights={} cameras={}",
        meshes_q.iter().count(),
        lights_q.iter().count(),
        cams_q.iter().count(),
    );
    screenshot::save_target_to(&mut commands, &target, cfg.path.clone());
    info!(
        "screenshot: captured at render frame {frame}, writing {}",
        cfg.path.display()
    );
    screenshot::finish_capture(&mut progress);
}
