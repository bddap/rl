fn main() {
    let _otel = otel::init("rl-otel-smoke", otel::OtelArgs { enabled: true });

    tracing::info!(answer = 42, "hello-otel-from-rust-LOG");

    let counter = otel::meter().u64_counter("rl_otel_smoke_counter").build();
    counter.add(7, &[opentelemetry::KeyValue::new("kind", "demo")]);

    println!("emitted log + metric; flushing on guard drop");
}
