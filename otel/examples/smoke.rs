fn main() {
    let _otel = otel::init("rl-otel-smoke", otel::OtelArgs { otel: true });

    let span = tracing::info_span!("smoke_span", phase = "demo");
    let _e = span.enter();

    tracing::info!(answer = 42, "hello-otel-from-rust-LOG");

    let counter = otel::meter().u64_counter("rl_otel_smoke_counter").build();
    counter.add(7, &[opentelemetry::KeyValue::new("kind", "demo")]);

    drop(_e);
    println!("emitted span + log + metric; flushing on guard drop");
}
