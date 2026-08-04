//! End-to-end probe for the perf collapse black box (rl#331): real ≥100 ms frames →
//! ring → collapse detector → structured warn! with the ring attached → OTLP export.
//! Run against a sink:
//!
//! ```sh
//! DECK_ID=verify OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
//!   cargo run -q -p crab-world --features render --example perf_collapse_smoke
//! ```
//!
//! then check the sink's `otlp-verify.jsonl` for a `perf collapse:` log record whose
//! `samples_csv` attribute holds one CSV row per frame the run produced.

use bevy::prelude::*;
use crab_world::debug_overlay::DebugOverlayPlugin;

fn main() {
    let _otel = otel::init("perf-collapse-smoke", otel::OtelArgs { enabled: true });
    let mut app = App::new();
    // Headless: MinimalPlugins drives Time<Real> from the wall clock, so sleeping
    // between updates produces genuinely slow frames rather than faked deltas. The
    // overlay's UI entities spawn inert (no render/UI systems run them).
    app.add_plugins(MinimalPlugins)
        .init_resource::<ButtonInput<KeyCode>>()
        .add_plugins(DebugOverlayPlugin {
            collapse_dump: true,
        });
    // 35 × 110 ms ≥ the 30-frame collapse window; the 60 s cooldown keeps it to one emit.
    for _ in 0..35 {
        std::thread::sleep(std::time::Duration::from_millis(110));
        app.update();
    }
    println!("ran 35 slow frames; collapse record exports on shutdown");
}
