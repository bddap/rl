use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use bevy::app::AppExit;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, MonitorSelection, PrimaryWindow, WindowMode};

use crate::client::{ClientSim, TickMsg};
use crate::controls::{self, Action, GcrContext, GcrControls};
use crate::net_loop::{Coordinator, Exchanged, NetDriver};
use crate::sim::{
    Crab, Input, Outcome, PLAYER_HEIGHT, Player, PlayerId, PlayerStatus, Pos, buttons, trig,
    trig_client,
};
use crate::telemetry::{TelemetryEvent, next_sample_tick};
use crab_world::controls::{ActiveContext, PAD_STICK_DEADZONE, update_controls_ui};

pub use crate::sim::TICK_HZ;

pub use crate::sim::TICK_DT;

const MAX_TICKS_PER_FRAME: u32 = 8;

/// Eye level: the pre-rl#256 1.6 m eyes on the 1.8 m player, as a stature fraction.
const EYE_HEIGHT: f32 = 1.6 / 1.8 * crate::sim::PLAYER_HEIGHT;

/// Eye level mid-slide (rl#368): a crouch-skid at half stature. The dip is the
/// first-person cue that the slide took — the sim's speed change alone was
/// unreadable in play. Render-only; the sim's collision/claw geometry is untouched.
const SLIDE_EYE_HEIGHT: f32 = 0.9 / 1.8 * crate::sim::PLAYER_HEIGHT;

/// The eye eases between stand and slide height with this per-second rate
/// (time-constant ≈ 1/12 s ≈ 80 ms — quick enough to read as a drop, not a cut).
const SLIDE_EYE_RATE: f32 = 12.0;

/// Avatar capsule radius: the pre-rl#256 0.4 m on the 1.8 m player, as a stature fraction.
const PLAYER_RADIUS: f32 = 0.4 / 1.8 * crate::sim::PLAYER_HEIGHT;

const MOUSE_SENS: f32 = 0.0022;

const FLIGHT_MOUSE_SENS: f32 = 0.01;

const PAD_LOOK_SPEED: f32 = 2.5;

const PITCH_LIMIT: f32 = 1.5;

/// The render frame's origin in sim space — the round's extraction point, synced
/// with [`crab_world::ground::GroundAnchor`] by `sync_ground_anchor` (rl#354).
/// EVERY Transform written render-side is world − this: subtracting on the i64
/// grid before the f32 conversion is what keeps the 0.051 m player's eye path
/// smooth at the rl#305 locales, where absolute f32 meters quantize at ~0.5–1.3 mm
/// — a large fraction of a per-tick walking step (the owner-visible nearby-ground
/// judder). Values already in absolute f32 world meters (rapier poses) subtract
/// [`RenderOrigin::offset_m`] instead — exact near the origin (Sterbenz), keeping
/// whatever precision physics gave them.
#[derive(Resource, Default, Clone, Copy)]
pub(crate) struct RenderOrigin(pub(crate) Pos);

impl RenderOrigin {
    /// The origin as absolute world meters, y = 0 (the anchor is xz-only) — the
    /// subtrahend for poses that only exist in f32 world meters.
    pub(crate) fn offset_m(self) -> Vec3 {
        let (x, z) = self.0.to_meters();
        Vec3::new(x, 0.0, z)
    }
}

/// A sim point in the render frame, at explicit height `y` (absolute — the frame
/// rebases xz only).
fn world(pos: Pos, y: f32, origin: RenderOrigin) -> Vec3 {
    let (x, z) = pos.rel_meters(origin.0);
    Vec3::new(x, y, z)
}

/// The sim's per-tick yaw turn cap, in radians. We normalize our accrued look radians
/// by this same cap so full `look_yaw` deflection means exactly "the most the sim
/// turns in one tick" — commanding more would only make the camera lag the avatar,
/// since the sim would clamp it. Derived from the sim's own
/// [`crate::sim::MAX_YAW_TURNS_PER_TICK`], so the two can't drift.
const MAX_YAW_PER_TICK_RADIANS: f32 =
    crate::sim::MAX_YAW_TURNS_PER_TICK as f32 / trig::TURN as f32 * std::f32::consts::TAU;

mod ambience;
mod app;
mod articulation;
mod audio;
mod brain_swap;
pub mod chord_map;
mod controls_sync;
mod driver;
mod input;
mod menu;
mod pos_trace;
mod pose;
mod render_mode;
mod scene;
mod screenshot;
#[cfg(test)]
mod tests;
mod vehicle_view;
mod voice;

/// The windowed client's per-save data dir: `$XDG_DATA_HOME|~/.local/share` +
/// `giant-crab-rescue`. `None` (no HOME at all) means callers run unpersisted
/// rather than refusing to boot.
fn data_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(base.join("giant-crab-rescue"))
}

pub use app::{AppPhase, Boot, build_windowed_app};
pub use audio::ExternalBus;
pub use render_mode::RenderMode;
pub use screenshot::{
    ChordScript, FrameDtModel, PilotScript, ScreenshotConfig, build_net_screenshot_app,
    build_screenshot_app,
};
