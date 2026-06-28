//! First-person Bevy client for the deterministic gray-box (rl#38 render sub).
//!
//! This is the windowed `play` mode of the `game` binary: it makes the
//! giant-crab-rescue sim VISIBLE and PLAYABLE on top of the existing lockstep +
//! transport netcode. It boots to a client-side Host / Join menu (rl#58,
//! [`AppPhase`]/[`menu`]) and builds the round only once the player chooses — the
//! menu is gated to its own pre-round phases and never touches the sim. The split it
//! honors is the one documented at the top of
//! [`crate::sim`]: **the sim is the authority, this client is a read-only
//! consumer that produces [`Input`]**. Rendering, the camera, mouse/gamepad input,
//! and tween interpolation are ALL client-side and add ZERO nondeterminism — the
//! only thing that ever crosses back into sim state is the per-tick [`Input`] each
//! peer broadcasts. Two peers running this client off the same input stream stay
//! bit-identical because none of the code here touches the sim except through
//! [`Lockstep::submit_local_input`].
//!
//! How the three layers wire together:
//! - **Lockstep** runs on a fixed-timestep accumulator ([`drive_lockstep`]) inside
//!   the Bevy app, NOT in Bevy's `FixedUpdate` — the sim's tick rate ([`TICK_HZ`])
//!   is its own clock, independent of the render/display rate. Each ready tick:
//!   drain the local [`PendingInput`] into `submit_local_input`, pump the transport
//!   (broadcast our [`TickMsg`], ingest peers'), then `try_advance`.
//! - **Render** ([`apply_transforms`]) reads `Lockstep::sim()` and tweens every
//!   entity between the previous tick's pose and the current one by the fractional
//!   accumulator, so motion is smooth at any frame rate even though the sim steps in
//!   discrete 30 Hz jumps.
//! - **Input** ([`gather_input`]) samples WASD + mouse + gamepad every render frame
//!   into [`PendingInput`]; the lockstep driver quantizes it to one [`Input`] per
//!   tick. Look pitch is integrated here and kept client-side (the sim models yaw
//!   only); the camera reads the authoritative yaw back from the sim.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use bevy::app::AppExit;
use bevy::camera::RenderTarget;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, MonitorSelection, PrimaryWindow, WindowMode};


use crab_world::controls::{
    ActiveContext, ActiveDevice, ForceRevealControls, PAD_STICK_DEADZONE, spawn_controls_ui,
    track_active_device, update_controls_ui,
};
use crate::cadence::PhysicsCadence;
use crate::controls::{self, Action, GcrContext, GcrControls};
use crate::lockstep::{Lockstep, TickMsg};
use crate::net_loop::{NetDriver, PeerMsg};
use crate::sim::{
    CRAB_SCALE, Crab, Input, Outcome, Plane, Player, PlayerId, PlayerStatus, Pos,
    Pos3, Sim, UNIT, buttons, trig, trig_client,
};
use crate::telemetry::{TELEMETRY_TICK_EVERY, TelemetryEvent};

/// Sim tick rate (Hz). Re-exported from [`crate::sim::TICK_HZ`] (the one source)
/// so this windowed client and the headless driver advance at the same rate and stay
/// in lockstep; the client renders faster and interpolates between ticks.
pub use crate::sim::TICK_HZ;

/// Seconds per sim tick — the fixed dt the lockstep accumulator drains in.
const TICK_DT: f64 = 1.0 / TICK_HZ as f64;

/// Most sim ticks to apply in a single render frame, so a long stall (window drag,
/// GPU hitch) can't trigger an unbounded catch-up spiral that freezes the client.
/// Extra accumulated time past this is dropped — the sim falls a little behind real
/// time rather than locking up.
const MAX_TICKS_PER_FRAME: u32 = 8;

/// Eye height of the first-person camera above the player's ground position, in
/// meters (a ~1.8 m capsule on the ground at Y=0; eyes near the top).
const EYE_HEIGHT: f32 = 1.6;

/// Player capsule dimensions (meters): a person-sized avatar for the other peers.
const PLAYER_RADIUS: f32 = 0.4;
const PLAYER_HEIGHT: f32 = 1.8;

/// Plane gray-box dimensions (meters): a fuselage box + a wider, thinner wing box.
/// Just enough shape to read as an aircraft and show its facing — a placeholder, like
/// the crab box.
const PLANE_FUSELAGE_LEN: f32 = 6.0;
const PLANE_FUSELAGE_W: f32 = 1.2;
const PLANE_WINGSPAN: f32 = 9.0;
const PLANE_WING_CHORD: f32 = 1.6;

/// Mouse look sensitivity (radians per pixel of motion). Yaw feeds the sim as a
/// per-tick delta; pitch stays client-side.
const MOUSE_SENS: f32 = 0.0022;

/// Gamepad look speed (radians/second at full right-stick deflection), scaled by the
/// frame dt so it's frame-rate independent.
const PAD_LOOK_SPEED: f32 = 2.5;

/// How long Select/Back must be HELD to quit (seconds). A hold, not a tap, so a stray
/// press can't end the round for everyone on the couch — the kid-safe equivalent of
/// Esc. Client-local (sends AppExit, never touches the sim), so it can't desync a peer.
const PAD_QUIT_HOLD_SECS: f32 = 1.0;

/// Pitch clamp (radians) so the FP camera can't flip over the poles.
const PITCH_LIMIT: f32 = 1.5;

/// Convert a sim fixed-point coordinate to meters.
fn meters(coord: i64) -> f32 {
    coord as f32 / UNIT as f32
}

/// A sim ground position (XZ at Y=0) as a Bevy world point at height `y`. The sim's
/// right-handed XZ frame (+X right, +Z forward, +Y up) IS Bevy's frame, so this is a
/// direct unit conversion with no axis remap.
fn world(pos: Pos, y: f32) -> Vec3 {
    Vec3::new(meters(pos.x), y, meters(pos.z))
}

/// A sim 3D position ([`Pos3`], includes altitude) as a Bevy world point — the same
/// direct unit conversion as [`world`], but with the entity's own Y (a flying plane),
/// not an externally supplied ground height.
fn world3(pos: Pos3) -> Vec3 {
    Vec3::new(meters(pos.x), meters(pos.y), meters(pos.z))
}

/// The sim's per-tick yaw turn cap, in radians. The sim clamps a tick's yaw delta to
/// `trig::TURN/24` turn-units (see [`crate::sim`]); we normalize our accrued
/// look radians by this same cap so full `look_yaw` deflection means exactly "the
/// most the sim turns in one tick" — commanding more would only make the camera lag
/// the avatar, since the sim would clamp it. Derived from the same integer `trig::TURN`
/// the sim uses, so the two can't drift.
const MAX_YAW_PER_TICK_RADIANS: f32 =
    (trig::TURN / 24) as f32 / trig::TURN as f32 * std::f32::consts::TAU;

mod app;
mod driver;
mod input;
mod scene;
mod hud;
mod screenshot;
mod menu;
#[cfg(test)]
mod tests;

pub use app::{AppPhase, Boot, build_windowed_app, pin_process_pools};
pub use screenshot::{ScreenshotConfig, build_screenshot_app};
pub(crate) use driver::{park_fixed_auto_pump, pump_fixed_steps};
pub(crate) use scene::crab_render_scale;
