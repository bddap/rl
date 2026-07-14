use std::env;

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:4318";

#[must_use = "telemetry stops and unflushed data is lost when the guard is dropped"]
pub struct OtelGuard {
    tracer_provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
    logger_provider: Option<opentelemetry_sdk::logs::SdkLoggerProvider>,
    meter_provider: Option<opentelemetry_sdk::metrics::SdkMeterProvider>,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(p) = &self.tracer_provider {
            let _ = p.shutdown();
        }
        if let Some(p) = &self.logger_provider {
            let _ = p.shutdown();
        }
        if let Some(p) = &self.meter_provider {
            let _ = p.shutdown();
        }
    }
}

// The project's own telemetry switch, flattened by every binary that calls `init` — ONE
// declaration of `--otel` and its `RL_OTEL` env fallback, so a value clap doesn't recognize as
// falsey turns export ON instead of silently leaving it off (`RL_OTEL=true` used to mean OFF:
// the read was `v == "1"`, rl#275).
//
// The `OTEL_*` vars stay env-only on purpose: they are OTel ecosystem convention, and the SDK's
// own contract with whatever launches the process.
//
// Deliberately NOT a doc comment: clap adopts a flattened struct's docs as the enclosing
// command's `about`, which would overwrite the description of every binary that flattens this.
#[derive(clap::Args, Debug, Clone, Copy, Default)]
pub struct OtelArgs {
    /// Export traces/metrics/logs to the built-in OTLP endpoint. Setting
    /// `OTEL_EXPORTER_OTLP_ENDPOINT` enables export on its own, and wins.
    #[arg(long = "otel", env = "RL_OTEL", global = true,
          value_parser = clap::builder::FalseyValueParser::new())]
    pub otel: bool,
}

pub fn init(service_name: &str, args: OtelArgs) -> OtelGuard {
    // `log`-crate records (wgpu_hal, rapier, …) reach this subscriber via
    // tracing-subscriber's default `tracing-log` feature: every `.init()` below installs
    // the LogTracer bridge itself. Do NOT also call `tracing_log::LogTracer::init()` here —
    // a pre-set logger makes those `.init()` calls PANIC (SetLoggerError), which took down
    // every binary at startup and broke the rl-release checkpoint gate (2026-07-02).
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);

    let endpoint = resolve_endpoint(args.otel);
    let Some(endpoint) = endpoint else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
        return OtelGuard {
            tracer_provider: None,
            logger_provider: None,
            meter_provider: None,
        };
    };

    match build_providers(service_name, &endpoint) {
        Ok((tracer_provider, logger_provider, meter_provider)) => {
            let tracer = opentelemetry::trace::TracerProvider::tracer(&tracer_provider, "rl");
            let trace_layer = tracing_opentelemetry::layer()
                .with_tracer(tracer)
                .with_filter(tracing_subscriber::filter::filter_fn(not_otel_internal));
            let log_layer = opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
                &logger_provider,
            )
            .with_filter(tracing_subscriber::filter::filter_fn(not_otel_internal));
            opentelemetry::global::set_meter_provider(meter_provider.clone());

            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .with(trace_layer)
                .with(log_layer)
                .init();
            tracing::info!(service_name, endpoint, "OTLP telemetry enabled");
            OtelGuard {
                tracer_provider: Some(tracer_provider),
                logger_provider: Some(logger_provider),
                meter_provider: Some(meter_provider),
            }
        }
        Err(e) => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .init();
            tracing::warn!("OTLP telemetry setup failed, continuing with stderr only: {e:#}");
            OtelGuard {
                tracer_provider: None,
                logger_provider: None,
                meter_provider: None,
            }
        }
    }
}

fn not_otel_internal(meta: &tracing::Metadata<'_>) -> bool {
    let t = meta.target();
    !(t.starts_with("opentelemetry")
        || t.starts_with("hyper")
        || t.starts_with("reqwest")
        || t.starts_with("h2")
        || t.starts_with("tonic")
        || t.starts_with("tower"))
}

fn resolve_endpoint(otel: bool) -> Option<String> {
    if let Ok(ep) = env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        && !ep.is_empty()
    {
        return Some(ep);
    }
    otel.then(|| DEFAULT_ENDPOINT.to_string())
}

type Providers = (
    opentelemetry_sdk::trace::SdkTracerProvider,
    opentelemetry_sdk::logs::SdkLoggerProvider,
    opentelemetry_sdk::metrics::SdkMeterProvider,
);

fn build_providers(service_name: &str, endpoint: &str) -> anyhow::Result<Providers> {
    use opentelemetry_otlp::{LogExporter, MetricExporter, SpanExporter, WithExportConfig};

    let service = env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| service_name.to_string());
    let resource = Resource::builder()
        .with_service_name(service)
        .with_attribute(KeyValue::new("host.name", host_name()))
        .with_attributes(env_resource_attributes())
        .build();

    let base = endpoint.strip_suffix('/').unwrap_or(endpoint);

    let span_exporter = SpanExporter::builder()
        .with_http()
        .with_endpoint(format!("{base}/v1/traces"))
        .build()?;
    let tracer_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_batch_exporter(span_exporter)
        .build();

    let log_exporter = LogExporter::builder()
        .with_http()
        .with_endpoint(format!("{base}/v1/logs"))
        .build()?;
    let logger_provider = opentelemetry_sdk::logs::SdkLoggerProvider::builder()
        .with_resource(resource.clone())
        .with_batch_exporter(log_exporter)
        .build();

    let metric_exporter = MetricExporter::builder()
        .with_http()
        .with_endpoint(format!("{base}/v1/metrics"))
        .build()?;
    let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(metric_exporter).build();
    let meter_provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(reader)
        .build();

    Ok((tracer_provider, logger_provider, meter_provider))
}

fn env_resource_attributes() -> Vec<KeyValue> {
    let Ok(raw) = env::var("OTEL_RESOURCE_ATTRIBUTES") else {
        return Vec::new();
    };
    raw.split(',')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            let (k, v) = (k.trim(), v.trim());
            (!k.is_empty() && !v.is_empty()).then(|| KeyValue::new(k.to_string(), v.to_string()))
        })
        .collect()
}

fn host_name() -> String {
    env::var("DECK_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| env::var("HOSTNAME").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "unknown".to_string())
}

pub fn meter() -> opentelemetry::metrics::Meter {
    opentelemetry::global::meter("rl")
}
