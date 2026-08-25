//! End-to-end probe for the fps-telemetry path (rl#309): synthetic frame deltas →
//! frame-path histogram → snapshot queue → flusher → OTLP export. Run against a sink:
//!
//! ```sh
//! DECK_ID=verify OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
//!   cargo run -q -p otel --example frametime_smoke
//! ```
//!
//! then check the sink's `otlp-verify.jsonl` for an `rl.frame_time_ms` histogram with
//! p50 ≈ 12.5 ms and a 100 ms hitch tail (count ≲ 360 — the trailing partial second
//! stays in the recorder, by design).

fn main() {
    let _otel = otel::init("frametime-smoke", otel::OtelArgs { enabled: true });
    let mut telemetry = frametime::start();
    // Three seconds of 80 fps with a couple of 100 ms hitches per second.
    for _ in 0..3 {
        for _ in 0..118 {
            telemetry.frame(0.0125);
        }
        telemetry.frame(0.1);
        telemetry.frame(0.1);
    }
    // Let the 500 ms flusher tick drain the queue into the SDK; the guard's drop
    // then force-flushes the exporter.
    std::thread::sleep(std::time::Duration::from_millis(1500));
    println!("fed 360 synthetic frames; exporting on shutdown");
}
