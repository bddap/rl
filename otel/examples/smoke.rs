//! Runtime smoke test for the `otel` init: with NO tokio runtime present, init the OTLP
//! pipelines and emit one of each signal (a span via `tracing`, an event → OTLP log, and
//! an OTLP metric), then drop the guard to flush. Run against a collector to confirm the
//! 0.28 thread-based exporters work without an async runtime and the bytes are valid OTLP.
//!
//!   OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:14318 cargo run -p otel --example smoke

fn main() {
    let _otel = otel::init("rl-otel-smoke");

    // A span → OTLP trace.
    let span = tracing::info_span!("smoke_span", phase = "demo");
    let _e = span.enter();

    // An event → OTLP log.
    tracing::info!(answer = 42, "hello-otel-from-rust-LOG");

    // A metric → OTLP metric.
    let counter = otel::meter().u64_counter("rl_otel_smoke_counter").build();
    counter.add(7, &[opentelemetry::KeyValue::new("kind", "demo")]);

    drop(_e);
    // _otel drops here, flushing all three pipelines before exit.
    println!("emitted span + log + metric; flushing on guard drop");
}
