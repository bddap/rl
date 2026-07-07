mod cameras;
mod controls;
mod demo;
mod graph;
mod hot_reload;
mod manual_control;
mod policy;
mod render_video;
mod rig_pose;
mod target_ball;

use std::path::PathBuf;

use bevy::prelude::*;
use bevy_rapier3d::plugin::PhysicsSet;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::bot::BotSet;
use crate::bot::body::CrabCarapace;
use crate::screenshot::{self, ShotProgress, ShotTarget};

pub use crate::policy::{Policy, RigDims, RigFit, checkpoint_fits_rig};
pub use render_video::RenderVideoPlugin;

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

#[derive(Resource)]
pub(super) struct DemoRng(pub(super) StdRng);

impl Default for DemoRng {
    fn default() -> Self {
        let seed = std::env::var("RL_DEMO_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(rand::random);
        Self(StdRng::seed_from_u64(seed))
    }
}

pub struct DemoPlugin {
    pub checkpoint_dir: PathBuf,
    pub live_checkpoint_dir: Option<PathBuf>,
    pub manual_control: bool,
}

impl Plugin for DemoPlugin {
    fn build(&self, app: &mut App) {
        add_inference(app, &self.checkpoint_dir, self.live_checkpoint_dir.clone());
        graph::register(app);
        app.add_plugins(crate::sky::NightSkyPlugin);
        app.add_plugins(crate::controls::ControlsOverlayPlugin::<DemoControls>::default());
        app.init_resource::<DemoSettle>()
            .init_resource::<PokeBurst>()
            .init_resource::<DemoRng>()
            .add_systems(Startup, (spawn_orbit_camera, spawn_target_ball))
            .add_systems(Update, (orbit_camera, demo_controls))
            .add_systems(
                FixedUpdate,
                (
                    demo_settle.after(BotSet::Think).before(BotSet::Act),
                    demo_poke.after(BotSet::Act).before(PhysicsSet::SyncBackend),
                    target_ball.after(BotSet::Sense),
                ),
            );
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
        app.add_plugins(crate::sky::NightSkyPlugin);
        if let Some(pose) = rig_pose::rig_pose_from_env() {
            app.insert_resource(pose)
                .init_resource::<rig_pose::RigPosePin>()
                .add_systems(
                    FixedUpdate,
                    rig_pose::rig_pose_drive
                        .in_set(BotSet::Think)
                        .after(policy_step),
                )
                .add_systems(
                    FixedUpdate,
                    rig_pose::rig_pose_pin
                        .after(BotSet::Act)
                        .before(PhysicsSet::SyncBackend),
                );
        }
        if std::env::var_os("RL_TARGET_BALL").is_some() {
            app.init_resource::<DemoRng>()
                .add_systems(Startup, spawn_target_ball)
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
