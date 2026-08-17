//! Sally flight recorder (bddap/rl#332): continuous ~10 Hz carapace kinematics from
//! every live surface, shipped over the process' existing OTLP pipe so the owner's
//! next "Sally flying" sighting can be reconstructed from a real track instead of
//! soak proxies. Bothouse-side extraction/plotting: `bothouse/telemetry/sally-track`.
//!
//! Frame-path budget (the standing input-latency constraint): per physics tick this
//! is one counter increment; ~10 Hz it formats ~90 bytes per crab into a buffer; 1 Hz
//! it hands the batch to `tracing`, whose OTLP bridge is a queue push — export runs on
//! the SDK's background batch thread. No lock, no syscall beyond the 10 Hz clock read.

use std::fmt::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use bevy_rapier3d::prelude::Velocity;

use crate::bot::body::{CrabCarapace, CrabEnvId};
use crate::physics::PHYSICS_HZ;
use crate::terrain::Terrain;

/// Sample cadence in physics ticks: 64 Hz / 6 ≈ 10.7 Hz. Flights last seconds, so an
/// arc is tens of samples.
const SAMPLE_EVERY_TICKS: u64 = 6;

/// Batch cadence: one log record per second of accumulated samples, so the OTLP
/// envelope overhead is paid ~1 Hz instead of per sample.
const EMIT_EVERY_TICKS: u64 = PHYSICS_HZ;

pub struct SallyTrackPlugin;

impl Plugin for SallyTrackPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SallyTrack>()
            .add_systems(FixedUpdate, sample);
    }
}

#[derive(Resource, Default)]
struct SallyTrack {
    /// Physics ticks since boot, counted here — the live surfaces expose no global
    /// sim-tick resource, and this recorder only needs a monotonic per-process ruler.
    tick: u64,
    /// Comma-joined compact-JSON samples for the batch under construction.
    buf: String,
}

fn sample(
    mut st: ResMut<SallyTrack>,
    terrain: Option<Res<Terrain>>,
    crabs: Query<(&CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>,
) {
    st.tick += 1;
    let tick = st.tick;
    if tick.is_multiple_of(SAMPLE_EVERY_TICKS) && !crabs.is_empty() {
        let t_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        for (id, tf, vel) in &crabs {
            let p = tf.translation;
            let v = vel.linear;
            if !st.buf.is_empty() {
                st.buf.push(',');
            }
            let _ = write!(
                st.buf,
                r#"{{"t":{t_ms},"k":{tick},"c":{},"p":[{:.2},{:.2},{:.2}],"v":[{:.2},{:.2},{:.2}]"#,
                id.0, p.x, p.y, p.z, v.x, v.y, v.z
            );
            // Above-ground height, the discriminator the rl#332 geometry work showed
            // matters (world-shallow arcs over receding slopes read as "flight").
            // Surfaces without a physics world (no Terrain) just omit it.
            if let Some(t) = &terrain {
                let _ = write!(st.buf, r#","a":{:.2}"#, p.y - t.height(p.x, p.z));
            }
            st.buf.push('}');
        }
    }
    if tick.is_multiple_of(EMIT_EVERY_TICKS) && !st.buf.is_empty() {
        let samples = format!("[{}]", st.buf);
        st.buf.clear();
        // Target must stay in lockstep with otel's stderr-suppression filter; the
        // tracing macros only accept a literal target, so the test below guards drift.
        tracing::info!(target: "sally_track", samples = %samples, "sally track batch");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn target_matches_otel_filter() {
        assert_eq!(otel::SALLY_TRACK_TARGET, "sally_track");
    }
}
