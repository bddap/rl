//! rl#403 flight-recorder extension over the same OTLP pipe as
//! `crab_world::sally_track`: per-player walker kinematics and the local
//! controller-input summary, ~10 Hz samples batched into ~1 Hz log records.
//!
//! Split of duty: the HOST samples every sim player (the lockstep sim is identical on
//! all peers, so one exporter covers the roster — remote players stop being invisible
//! in telemetry), while EVERY peer samples its OWN issued input (arrival at the source
//! device is exactly what separates "input never arrived" from "sim ignored input").
//! Walker samples share the sally_track target and schema — keyed `"pl":<player>`
//! beside the crab's `"c"` and the craft's `"veh"` — so the plotter's per-body-key
//! grouping picks them up unchanged; a piloting player's walker mirrors its craft, and
//! the craft body itself is already the `"veh"` stream (one source per body, rl#401).

use std::fmt::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;

use super::driver::GameState;
use crate::sim::{Input, TICK_HZ, UNIT};

/// Sample cadence in sim ticks: 30 Hz / 3 = 10 Hz, the sally_track rate.
const SAMPLE_EVERY_TICKS: u64 = 3;

/// Batch cadence: one log record per second of accumulated samples, so the OTLP
/// envelope overhead is paid ~1 Hz instead of per sample.
const EMIT_EVERY_TICKS: u64 = TICK_HZ;

#[derive(Resource, Default)]
pub(super) struct NetTrack {
    /// Highest sim tick seen — a smaller current tick means a round reset, which
    /// rewinds the watermarks below instead of muting sampling until the old count.
    seen: u64,
    /// Next tick at/after which to sample ([`crate::telemetry::next_sample_tick`]
    /// shape) — a watermark, not a modulus, so remote-adopt jumps of several ticks
    /// still sample once.
    next_sample: u64,
    /// Next tick at/after which to hand the batches to `tracing`.
    next_emit: u64,
    /// Comma-joined compact-JSON walker samples (host only).
    players: String,
    /// Comma-joined compact-JSON local-input samples.
    inputs: String,
}

/// Per applied sim tick, from the drive loop. `is_host` gates the walker stream
/// (solo is a host with a roster of one); `input` is this tick's issued local input.
pub(super) fn sample(world: &mut World, is_host: bool, input: Input) {
    let tick = {
        let state = world.non_send::<GameState>();
        state.client.sim().tick()
    };
    {
        let mut track = world.get_resource_or_insert_with(NetTrack::default);
        if tick < track.seen {
            track.next_sample = 0;
            track.next_emit = 0;
            track.players.clear();
            track.inputs.clear();
        }
        track.seen = tick;
        if tick < track.next_sample {
            return;
        }
        track.next_sample = (tick / SAMPLE_EVERY_TICKS + 1) * SAMPLE_EVERY_TICKS;
    }

    let t_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let me = {
        let state = world.non_send::<GameState>();
        state.client.me().0
    };
    let players = is_host.then(|| player_samples(world, t_ms, tick));

    let mut track = world.resource_mut::<NetTrack>();
    if let Some(p) = players
        && !p.is_empty()
    {
        if !track.players.is_empty() {
            track.players.push(',');
        }
        track.players.push_str(&p);
    }
    {
        // Magnitudes only, by design (rl#403: gameplay telemetry, not a keylogger):
        // move-stick magnitude, |look-yaw|, and the button bitmask.
        let mv = f32::from(input.move_strafe).hypot(f32::from(input.move_forward))
            / f32::from(Input::AXIS_SCALE);
        let lk = f32::from(input.look_yaw.unsigned_abs()) / f32::from(Input::AXIS_SCALE);
        let b = input.buttons;
        let buf = &mut track.inputs;
        if !buf.is_empty() {
            buf.push(',');
        }
        let _ = write!(
            buf,
            r#"{{"t":{t_ms},"k":{tick},"pl":{me},"mv":{mv:.2},"lk":{lk:.2},"b":{b}}}"#
        );
    }

    if tick >= track.next_emit {
        track.next_emit = (tick / EMIT_EVERY_TICKS + 1) * EMIT_EVERY_TICKS;
        if !track.players.is_empty() {
            let samples = format!("[{}]", track.players);
            track.players.clear();
            // Targets must stay in lockstep with otel's constants; tracing macros
            // only accept literal targets (`targets_match_otel` guards drift).
            tracing::info!(target: "sally_track", samples = %samples, "player track batch");
        }
        if !track.inputs.is_empty() {
            let samples = format!("[{}]", track.inputs);
            track.inputs.clear();
            tracing::info!(target: "input_track", samples = %samples, "input track batch");
        }
    }
}

/// Every sim player's walker as comma-joined samples in the sally_track schema:
/// `p` absolute meters (y = local surface + altitude), `v` m/s, `a` above-ground m.
fn player_samples(world: &World, t_ms: u64, tick: u64) -> String {
    let state = world.non_send::<GameState>();
    let terrain = world.get_resource::<crab_world::terrain::Terrain>();
    let mut buf = String::new();
    for (pid, p) in state.client.sim().players() {
        let (x, z) = p.pos().to_meters();
        let a = p.alt() as f32 / UNIT as f32;
        let y = terrain.map_or(0.0, |t| t.height(x, z)) + a;
        // Vel is grid units per tick; one conversion to m/s for all three axes.
        let per_tick = TICK_HZ as f32 / UNIT as f32;
        let v = p.vel();
        let (vx, vy, vz) = (
            v.x as f32 * per_tick,
            v.y as f32 * per_tick,
            v.z as f32 * per_tick,
        );
        if !buf.is_empty() {
            buf.push(',');
        }
        let _ = write!(
            buf,
            r#"{{"t":{t_ms},"k":{tick},"pl":{},"p":[{x:.2},{y:.2},{z:.2}],"v":[{vx:.2},{vy:.2},{vz:.2}],"a":{a:.2}}}"#,
            pid.0
        );
    }
    buf
}

#[cfg(test)]
mod tests {
    #[test]
    fn targets_match_otel() {
        assert_eq!(otel::SALLY_TRACK_TARGET, "sally_track");
        assert_eq!(otel::INPUT_TRACK_TARGET, "input_track");
    }
}
