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

use crate::policy::RigDims;
/// The demo's control scheme — the ONE source of its bindings AND its legend. Public so the
/// entrypoint can resolve `--show-controls-context` against it at t=0 (rl#275).
pub use controls::DemoControls;
pub use render_video::RenderVideoPlugin;
pub use rig_pose::RigPosePart;

pub fn rig_dims() -> RigDims {
    RigDims {
        obs: crate::bot::sensor::OBS_SIZE,
        action: crate::bot::actuator::ACTION_SIZE,
    }
}

use cameras::{orbit_camera, spawn_offscreen_camera, spawn_orbit_camera, track_offscreen_camera};
use demo::{DemoSettle, PokeBurst, demo_controls, demo_poke, demo_settle};
use hot_reload::hot_reload_policy;
use manual_control::{ManualControl, manual_control_step, spawn_manual_hud};
use policy::{add_inference, policy_step};
use target_ball::{spawn_target_ball, target_ball};

#[derive(Resource)]
pub(super) struct DemoRng(pub(super) StdRng);

impl DemoRng {
    fn seeded(seed: Option<u64>) -> Self {
        Self(StdRng::seed_from_u64(seed.unwrap_or_else(rand::random)))
    }
}

impl Default for DemoRng {
    fn default() -> Self {
        Self::seeded(None)
    }
}

/// The rl-demo knobs shared by every play mode, parsed at the entrypoint (rl#272):
/// which seed drives the demo RNG, whether a rest-bound policy load drives a random
/// diagnostic brain instead, and whether the target ball holds a pinned position.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlayOverrides {
    pub seed: Option<u64>,
    pub random_policy: bool,
    pub target_ball_at: Option<Vec3>,
}

impl PlayOverrides {
    fn apply_rng_and_ball(&self, app: &mut App) {
        app.insert_resource(DemoRng::seeded(self.seed));
        app.insert_resource(target_ball::TargetBallAt(self.target_ball_at));
    }
}

pub struct DemoPlugin {
    pub checkpoint_dir: PathBuf,
    pub live_checkpoint_dir: Option<PathBuf>,
    pub manual_control: bool,
    pub overrides: PlayOverrides,
    /// Show the joint-trace graph overlay from launch.
    pub graph: bool,
    /// Capture the graph overlay to this path once its traces fill.
    pub graph_shot: Option<PathBuf>,
    pub controls: crate::controls::ControlsOverrides<DemoControls>,
}

impl Plugin for DemoPlugin {
    fn build(&self, app: &mut App) {
        add_inference(
            app,
            &self.checkpoint_dir,
            self.live_checkpoint_dir.clone(),
            self.overrides.random_policy,
        );
        graph::register(app, self.graph, self.graph_shot.clone());
        self.overrides.apply_rng_and_ball(app);
        app.add_plugins(crate::sky::NightSkyPlugin);
        crate::controls::install_overlay(app, &self.controls);
        // The demo is a single always-armed owner-facing crab: a rescue there is a
        // visible teleport, so it logs at the same fault/warn tier GCR arms.
        app.insert_resource(crate::bot::CrabRescueIsFault);
        app.init_resource::<DemoSettle>()
            .init_resource::<PokeBurst>()
            .add_systems(Startup, (spawn_orbit_camera, spawn_target_ball))
            .add_systems(Update, (orbit_camera, demo_controls))
            .add_systems(
                FixedUpdate,
                (
                    // Same ordering contract as GCR's registration (rl#303; since
                    // rl#311 the recenter is float-precision hygiene, not an obs
                    // guard): after rescue so a respawned env sees ~0 drift, before
                    // Sense to match the shared registration. Also what keeps the
                    // target ball re-seeding near HER locale instead of the boot
                    // spawn.
                    crate::bot::recenter_drifted_origins
                        .after(crate::bot::rescue_lost_crabs)
                        .before(BotSet::Sense),
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
    pub overrides: PlayOverrides,
    /// Spawn the chase target ball (the demo/video modes always have one).
    pub target_ball: bool,
    /// Drive these joints at this action value for a pose still.
    pub rig_pose: Option<(f32, RigPosePart)>,
    /// Fixed `(eye, look-at)` for the shot camera instead of the crab-tracking
    /// close-up — vista framing for the rl#281 terrain taste loop.
    pub shot_view: Option<(Vec3, Vec3)>,
    pub controls: crate::controls::ControlsOverrides<DemoControls>,
}

#[derive(Resource)]
pub(in crate::play) struct ShotConfig {
    path: PathBuf,
    settle: u32,
    pub(in crate::play) width: u32,
    pub(in crate::play) height: u32,
    pub(in crate::play) view: Option<(Vec3, Vec3)>,
}

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        add_inference(
            app,
            &self.checkpoint_dir,
            None,
            self.overrides.random_policy,
        );
        app.add_systems(FixedUpdate, policy_step.in_set(BotSet::Think));
        app.add_plugins(crate::sky::NightSkyPlugin);
        if let Some((action, part)) = self.rig_pose {
            app.insert_resource(rig_pose::RigPose::new(action, part))
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
        if self.target_ball {
            self.overrides.apply_rng_and_ball(app);
            app.add_systems(Startup, spawn_target_ball)
                .add_systems(FixedUpdate, target_ball.after(BotSet::Sense));
        }
        app.insert_resource(ShotConfig {
            path: self.path.clone(),
            settle: self.settle,
            width: self.width,
            height: self.height,
            view: self.shot_view,
        })
        .init_resource::<ShotProgress>()
        .add_systems(Startup, spawn_offscreen_camera)
        .add_systems(
            Update,
            (track_offscreen_camera, capture_when_settled).chain(),
        );
        crate::controls::install_overlay(app, &self.controls);
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
