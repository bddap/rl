use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use bevy::app::AppExit;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, MonitorSelection, PrimaryWindow, WindowMode};

use crate::cadence::PhysicsCadence;
use crate::controls::{self, Action, GcrContext, GcrControls};
use crate::lockstep::{Lockstep, TickMsg};
use crate::net_loop::{Coordinator, Exchanged, NetDriver};
use crate::sim::{
    CRAB_SCALE, Crab, Input, Outcome, Player, PlayerId, PlayerStatus, Pos, buttons, trig,
    trig_client,
};
use crate::telemetry::{TelemetryEvent, next_sample_tick};
use crab_world::controls::{
    ActiveContext, ActiveDevice, ForceRevealControls, PAD_STICK_DEADZONE, spawn_controls_ui,
    track_active_device, update_controls_ui,
};

pub use crate::sim::TICK_HZ;

pub use crate::sim::TICK_DT;

const MAX_TICKS_PER_FRAME: u32 = 8;

const EYE_HEIGHT: f32 = 1.6;

const PLAYER_RADIUS: f32 = 0.4;
const PLAYER_HEIGHT: f32 = 1.8;

const MOUSE_SENS: f32 = 0.0022;

const FLIGHT_MOUSE_SENS: f32 = 0.01;

const PAD_LOOK_SPEED: f32 = 2.5;

/// How long the pad Quit button (North/Y) must be HELD to quit (seconds). A hold, not a
/// tap, so a stray press can't end the round for everyone on the couch — the kid-safe
/// equivalent of Esc. Client-local (sends AppExit, never touches the sim), so it can't
/// desync a peer.
const PAD_QUIT_HOLD_SECS: f32 = 1.0;

const PITCH_LIMIT: f32 = 1.5;

fn world(pos: Pos, y: f32) -> Vec3 {
    let (x, z) = pos.to_meters();
    Vec3::new(x, y, z) * scene::world_render_scale()
}

/// The sim's per-tick yaw turn cap, in radians. We normalize our accrued look radians
/// by this same cap so full `look_yaw` deflection means exactly "the most the sim
/// turns in one tick" — commanding more would only make the camera lag the avatar,
/// since the sim would clamp it. Derived from the sim's own
/// [`crate::sim::MAX_YAW_TURNS_PER_TICK`], so the two can't drift.
const MAX_YAW_PER_TICK_RADIANS: f32 =
    crate::sim::MAX_YAW_TURNS_PER_TICK as f32 / trig::TURN as f32 * std::f32::consts::TAU;

mod app;
mod articulation;
mod driver;
mod hud;
mod input;
mod menu;
mod render_mode;
mod scene;
mod screenshot;
#[cfg(test)]
mod tests;

pub use app::{AppPhase, Boot, build_windowed_app};
pub use render_mode::RenderMode;
pub(crate) use scene::world_render_scale;
pub use screenshot::{
    PilotScript, ScreenshotConfig, build_net_screenshot_app, build_screenshot_app,
};
