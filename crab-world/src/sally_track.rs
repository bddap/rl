//! Sally flight recorder (bddap/rl#332): continuous ~10 Hz kinematics from every live
//! surface, shipped over the process' existing OTLP pipe so the next "Sally
//! flying" sighting can be reconstructed from a real track instead of soak proxies.
//! Bothouse-side extraction/plotting: `bothouse/telemetry/sally-track`.
//!
//! Two body kinds per batch, discriminated by key (rl#401): crab samples carry
//! `"c":<env>`, craft samples carry `"veh":<pilot>,"kind":<wire byte>`. Crafts are in
//! the batch because a ride was otherwise invisible: the piloted walker only MIRRORS
//! the craft's pose in the fixed-point sim, so the craft rigidbody here is the one
//! physics source of ride motion — a whole TV flight session read as a parked world.
//!
//! Frame-path budget (the standing input-latency constraint): per physics tick this
//! is one counter increment; ~10 Hz it formats ~90 bytes per crab into a buffer; 1 Hz
//! it hands the batch to `tracing`, whose OTLP bridge is a queue push — export runs on
//! the SDK's background batch thread. No lock, no syscall beyond the 10 Hz clock read.

use std::fmt::Write;

use bevy::prelude::*;
use bevy_rapier3d::prelude::Velocity;

use crate::bot::body::{CrabCarapace, CrabEnvId};
use crate::physics::PHYSICS_HZ;
use crate::terrain::Terrain;
use crate::vehicle::{Vehicle, VehicleKind};

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
    /// Each pilot's craft kind last tick — the vehicle-transition edge detector
    /// (rl#403). Diffing the live craft SET catches every edge including exits (the
    /// rl#400 gap: the despawn side had no log at all), and only worlds with real
    /// craft bodies — the host's — carry this plugin, so remotes don't duplicate.
    riding: std::collections::BTreeMap<u8, VehicleKind>,
}

/// The mode edges between two craft snapshots: (player, from, to), `None` = on foot.
/// Pure so the exit/board/morph cases are testable without capturing tracing output.
fn mode_edges(
    prev: &std::collections::BTreeMap<u8, VehicleKind>,
    now: &std::collections::BTreeMap<u8, VehicleKind>,
) -> Vec<(u8, Option<VehicleKind>, Option<VehicleKind>)> {
    let mut edges = Vec::new();
    for (&player, &kind) in now {
        match prev.get(&player) {
            None => edges.push((player, None, Some(kind))),
            Some(&p) if p != kind => edges.push((player, Some(p), Some(kind))),
            _ => {}
        }
    }
    for (&player, &kind) in prev {
        if !now.contains_key(&player) {
            edges.push((player, Some(kind), None));
        }
    }
    edges
}

/// One first-class vehicle-mode transition (rl#403): `None` = on foot. Target
/// literal must stay in lockstep with `otel::VEHICLE_TRANSITION_TARGET` (tracing
/// macros only accept a literal target; the test below guards drift).
fn transition(player: u8, from: Option<VehicleKind>, to: Option<VehicleKind>) {
    let mode = |k: Option<VehicleKind>| match k {
        None => "foot",
        Some(VehicleKind::Plane) => "plane",
        Some(VehicleKind::Ship) => "ship",
    };
    tracing::info!(
        target: "vehicle_transition",
        player,
        from = mode(from),
        to = mode(to),
        "vehicle transition"
    );
}

/// One body's compact-JSON sample appended to the batch buffer. `who` is the
/// body-kind discriminator pair(s), e.g. `"c":0` or `"veh":0,"kind":2`.
fn push_sample(
    buf: &mut String,
    t_ms: u64,
    tick: u64,
    who: std::fmt::Arguments<'_>,
    tf: &Transform,
    vel: &Velocity,
    terrain: Option<&Terrain>,
) {
    let p = tf.translation;
    let v = vel.linear;
    if !buf.is_empty() {
        buf.push(',');
    }
    let _ = write!(
        buf,
        r#"{{"t":{t_ms},"k":{tick},{who},"p":[{:.2},{:.2},{:.2}],"v":[{:.2},{:.2},{:.2}]"#,
        p.x, p.y, p.z, v.x, v.y, v.z
    );
    // Above-ground height, the discriminator the rl#332 geometry work showed
    // matters (world-shallow arcs over receding slopes read as "flight").
    // Surfaces without a physics world (no Terrain) just omit it.
    if let Some(t) = terrain {
        let _ = write!(buf, r#","a":{:.2}"#, p.y - t.height(p.x, p.z));
    }
    buf.push('}');
}

fn sample(
    mut st: ResMut<SallyTrack>,
    terrain: Option<Res<Terrain>>,
    crabs: Query<(&CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>,
    crafts: Query<(&Vehicle, &Transform, &Velocity)>,
) {
    st.tick += 1;
    let tick = st.tick;
    let now: std::collections::BTreeMap<u8, VehicleKind> =
        crafts.iter().map(|(v, ..)| (v.pilot.0, v.kind)).collect();
    if now != st.riding {
        for (player, from, to) in mode_edges(&st.riding, &now) {
            transition(player, from, to);
        }
        st.riding = now;
    }
    if tick.is_multiple_of(SAMPLE_EVERY_TICKS) {
        let t_ms = crate::wall::unix_ms();
        let terrain = terrain.as_deref();
        let buf = &mut st.buf;
        for (id, tf, vel) in &crabs {
            push_sample(
                buf,
                t_ms,
                tick,
                format_args!(r#""c":{}"#, id.0),
                tf,
                vel,
                terrain,
            );
        }
        // Ridden crafts (rl#401): `kind` is the wire byte (1 plane, 2 ship).
        for (v, tf, vel) in &crafts {
            push_sample(
                buf,
                t_ms,
                tick,
                format_args!(r#""veh":{},"kind":{}"#, v.pilot.0, v.kind as u8),
                tf,
                vel,
                terrain,
            );
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
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::*;
    use bevy_rapier3d::prelude::Velocity;

    use super::{SAMPLE_EVERY_TICKS, SallyTrack, sample};
    use crate::vehicle::{PilotId, Vehicle, VehicleKind};

    #[test]
    fn target_matches_otel_filter() {
        assert_eq!(otel::SALLY_TRACK_TARGET, "sally_track");
        assert_eq!(otel::VEHICLE_TRANSITION_TARGET, "vehicle_transition");
    }

    /// rl#403: every mode edge is reported — board (foot→craft), morph
    /// (craft→craft), and the exit the rl#400 log lines lost (craft→foot).
    #[test]
    fn mode_edges_cover_board_morph_and_exit() {
        use std::collections::BTreeMap;

        use super::mode_edges;
        use crate::vehicle::VehicleKind::{Plane, Ship};

        let prev: BTreeMap<u8, _> = [(1, Plane), (2, Ship)].into();
        let now: BTreeMap<u8, _> = [(1, Ship), (3, Plane)].into();
        let edges = mode_edges(&prev, &now);
        assert_eq!(
            edges,
            vec![
                (1, Some(Plane), Some(Ship)), // morph
                (3, None, Some(Plane)),       // board
                (2, Some(Ship), None),        // exit
            ]
        );
        assert!(mode_edges(&now, &now).is_empty());
    }

    /// rl#401: a ridden craft is IN the batch — its rigidbody is the only physics
    /// source of ride motion (the sim's piloted walker merely mirrors it), so a
    /// tracker without craft samples records a parked world through a whole flight.
    #[test]
    fn crafts_are_sampled_with_pilot_and_kind() {
        let mut world = World::new();
        world.init_resource::<SallyTrack>();
        world.spawn((
            Vehicle::test_body(PilotId(3), VehicleKind::Ship),
            Transform::from_xyz(100.0, 20.0, -50.0),
            Velocity::linear(Vec3::new(8.0, 1.0, -2.0)),
        ));
        for _ in 0..SAMPLE_EVERY_TICKS {
            world.run_system_once(sample).expect("bare world");
        }
        let buf = &world.resource::<SallyTrack>().buf;
        assert!(
            buf.contains(r#""veh":3,"kind":2"#),
            "craft sample must carry pilot + wire-byte kind, got: {buf}"
        );
        assert!(
            buf.contains(r#""p":[100.00,20.00,-50.00]"#)
                && buf.contains(r#""v":[8.00,1.00,-2.00]"#),
            "craft sample must carry the craft's own kinematics, got: {buf}"
        );
    }
}
