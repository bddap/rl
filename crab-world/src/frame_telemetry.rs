//! Per-frame fps telemetry for every windowed surface (bddap/rl#309): one `Last`
//! system feeds the wall-clock frame delta to [`otel::frametime`], which histograms
//! it off the frame path and exports over the process' OTLP pipe. Part of
//! [`crate::app_boot::base_plugins`]' windowed arm; offscreen surfaces pace frames
//! artificially, so their deltas would be noise.

use bevy::prelude::*;

pub struct FrameTelemetryPlugin;

impl Plugin for FrameTelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FrameTelemetry(otel::frametime::spawn()))
            .add_systems(Last, record_frame);
    }
}

#[derive(Resource)]
struct FrameTelemetry(otel::frametime::FrameTelemetry);

fn record_frame(time: Res<Time<Real>>, mut telemetry: ResMut<FrameTelemetry>) {
    // Real (wall) delta, not virtual: a paused game still renders frames.
    telemetry.0.frame(time.delta_secs());
}
