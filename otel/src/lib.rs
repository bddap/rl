//! Vanilla OpenTelemetry, app side — one shared init for every rl binary.
//!
//! Emits standard OTLP (traces, metrics, logs) over OTLP/HTTP to a localhost endpoint
//! (the iroh tunnel's `forward` port). The app stays 100% stock OTEL and never knows iroh
//! exists; iroh is purely the transport hop to the bothouse sink. Full pipeline design:
//! bddap/bothouse `telemetry/README.md`.
//!
//! ## One source of truth for identity
//!
//! Each source tags its telemetry with `host.name` (the deck name — ablaised / kayleeza /
//! capippin — or `bothouse`) and `service.name` (which rl binary). The tag lives in the
//! OTEL **Resource** here, so it travels with the data through every hop and partitions
//! the sink by source — the OTEL-canonical place, not a collector processor.
//!
//! ## Off unless configured
//!
//! With `OTEL_EXPORTER_OTLP_ENDPOINT` unset, [`init`] installs ONLY the stderr `fmt`
//! subscriber and exports nothing — so linking this into the determinism-sacrosanct game
//! sim is inert until a deck is actually wired to a tunnel. Export adds no nondeterminism
//! regardless: it only READS the tracing stream the app already produces.

use std::env;

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Standard OTLP/HTTP base endpoint (the exporter appends `/v1/{traces,metrics,logs}`).
/// Points at the local iroh-tunnel `forward` listener by default — see the pipeline README.
const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:4318";

/// Holds the SDK providers so telemetry keeps flowing for the process lifetime, and
/// flushes + shuts them down on drop. Bind it for as long as you want telemetry:
/// `let _otel = otel::init("rl-train");` at the top of `main`. The `fmt`-only (disabled)
/// case carries no providers — dropping it is a no-op.
#[must_use = "telemetry stops and unflushed data is lost when the guard is dropped"]
pub struct OtelGuard {
    tracer_provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
    logger_provider: Option<opentelemetry_sdk::logs::SdkLoggerProvider>,
    meter_provider: Option<opentelemetry_sdk::metrics::SdkMeterProvider>,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        // Best-effort flush of each pipeline; a sink that's gone must not panic shutdown.
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

/// Install the process-wide tracing subscriber for `service_name` (e.g. `"rl-train"`,
/// `"rl-game"`). Always installs a stderr `fmt` layer (so logs surface headless, the
/// trainer's prior behavior). When `OTEL_EXPORTER_OTLP_ENDPOINT` is set — or
/// `RL_OTEL=1` to use the [`DEFAULT_ENDPOINT`] — it ALSO exports OTLP traces, logs, and
/// metrics tagged with this service + the host's `host.name`.
///
/// `RUST_LOG` overrides the default `info` filter. Call once at process start; the
/// returned guard must outlive the work whose telemetry you want delivered.
pub fn init(service_name: &str) -> OtelGuard {
    // `log`-crate records (wgpu_hal, rapier, …) reach this subscriber via
    // tracing-subscriber's default `tracing-log` feature: every `.init()` below installs
    // the LogTracer bridge itself. Do NOT also call `tracing_log::LogTracer::init()` here —
    // a pre-set logger makes those `.init()` calls PANIC (SetLoggerError), which took down
    // every binary at startup and broke the rl-release checkpoint gate (2026-07-02).
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);

    let endpoint = resolve_endpoint();
    let Some(endpoint) = endpoint else {
        // Disabled: stderr only, no export. Inert in the game sim until a deck is wired.
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
            // CRITICAL: exclude the exporters' own internal logs/spans (reqwest/hyper/the
            // otel crates) from the OTEL layers. Without this the log bridge captures the
            // OTLP HTTP client's logs, which it then exports, producing more logs — an
            // infinite feedback loop that overflows the stack on the first export.
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
            // Never let a telemetry-setup failure take down the app — fall back to stderr.
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

/// True unless the event/span comes from the telemetry export machinery itself. Used to
/// keep the OTEL layers from ingesting their own HTTP-client output (the recursion that
/// otherwise overflows the stack — see [`init`]). The stderr `fmt` layer is unaffected,
/// so these internal diagnostics still surface locally.
fn not_otel_internal(meta: &tracing::Metadata<'_>) -> bool {
    let t = meta.target();
    !(t.starts_with("opentelemetry")
        || t.starts_with("hyper")
        || t.starts_with("reqwest")
        || t.starts_with("h2")
        || t.starts_with("tonic")
        || t.starts_with("tower"))
}

/// The endpoint to export to, or `None` (telemetry disabled). Honors the standard
/// `OTEL_EXPORTER_OTLP_ENDPOINT`; `RL_OTEL=1` opts in at the default local tunnel port.
fn resolve_endpoint() -> Option<String> {
    if let Ok(ep) = env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        && !ep.is_empty()
    {
        return Some(ep);
    }
    if env::var("RL_OTEL").map(|v| v == "1").unwrap_or(false) {
        return Some(DEFAULT_ENDPOINT.to_string());
    }
    None
}

type Providers = (
    opentelemetry_sdk::trace::SdkTracerProvider,
    opentelemetry_sdk::logs::SdkLoggerProvider,
    opentelemetry_sdk::metrics::SdkMeterProvider,
);

/// Build the three OTLP/HTTP pipelines sharing one [`Resource`]. Batch span + log export
/// and the periodic metric reader all run on dedicated threads (the 0.27 model), so no
/// async runtime is required — the headless trainer can export without one.
fn build_providers(service_name: &str, endpoint: &str) -> anyhow::Result<Providers> {
    use opentelemetry_otlp::{LogExporter, MetricExporter, SpanExporter, WithExportConfig};

    // Honor the standard OTEL env vars 0.28's Resource::builder() does NOT auto-ingest, so
    // a deployer can override identity the vanilla way. Order matters: later attributes win
    // for a repeated key, so the explicit OTEL_RESOURCE_ATTRIBUTES override our host.name
    // default if it sets one. OTEL_SERVICE_NAME overrides the passed name.
    let service = env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| service_name.to_string());
    let resource = Resource::builder()
        .with_service_name(service)
        .with_attribute(KeyValue::new("host.name", host_name()))
        .with_attributes(env_resource_attributes())
        .build();

    // Per-signal endpoint = base + `/v1/{traces,logs,metrics}`. The OTLP/HTTP exporter only
    // appends that path itself when the endpoint comes from `OTEL_EXPORTER_OTLP_ENDPOINT`;
    // a `.with_endpoint(...)` value (our `RL_OTEL` default path) is used VERBATIM, so a bare
    // base posts to `/` and the collector 404s — telemetry silently dropped. Append the path
    // ourselves so the convenience opt-in actually delivers. (A signal-specific
    // `OTEL_EXPORTER_OTLP_{LOGS,TRACES,METRICS}_ENDPOINT` env still wins inside the exporter
    // and is taken verbatim, so this never double-appends.)
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

/// Parse the standard `OTEL_RESOURCE_ATTRIBUTES` env (`key=value,key2=value2`) into
/// resource attributes. Malformed entries (no `=`) are skipped. Empty/unset → none. This
/// is the vanilla way to attach extra identity (e.g. `deployment.environment`) without a
/// code change; 0.28's `Resource::builder()` doesn't read it on its own.
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

/// This source's `host.name` tag: the deck name (`DECK_ID`) if set, else `HOSTNAME`, else
/// `"unknown"`. The deck units set `DECK_ID=ablaised|kayleeza|capippin`; bothouse has
/// `HOSTNAME=bothouse`.
fn host_name() -> String {
    env::var("DECK_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| env::var("HOSTNAME").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// A meter for app-defined OTLP metrics (counters, histograms, gauges), bound to the
/// global provider [`init`] installed. No-op metrics until/unless telemetry is enabled.
/// Example: `otel::meter().u64_counter("rl.iterations").build().add(1, &[]);`
pub fn meter() -> opentelemetry::metrics::Meter {
    opentelemetry::global::meter("rl")
}
