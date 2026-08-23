use std::env;

pub mod frametime;

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:4318";

/// Target of the Sally flight-recorder batches (`crab_world::sally_track`, rl#332):
/// OTLP-export-only data records — ~1 Hz of ~1 KB JSON that would be pure noise on
/// stderr/journal, so the fmt layer drops them while the OTLP log layer ships them.
pub const SALLY_TRACK_TARGET: &str = "sally_track";

/// Target of the controller-input summary batches (`net::render::net_track`, rl#403):
/// same export-only contract as [`SALLY_TRACK_TARGET`].
pub const INPUT_TRACK_TARGET: &str = "input_track";

/// Target of the vehicle-mode transition events (`crab_world::sally_track`, rl#403):
/// rare first-class events (player, from-mode, to-mode). Forced past quiet RUST_LOG
/// defaults like the batch targets, but NOT dropped from stderr — a boarding edge is
/// worth a journal line.
pub const VEHICLE_TRANSITION_TARGET: &str = "vehicle_transition";

/// Target of in-game screenshot events (`net::render::live_screenshot`, rl#405):
/// rare first-class events naming the written PNG so tooling can fetch it. Same
/// contract as [`VEHICLE_TRANSITION_TARGET`] — forced past quiet RUST_LOG, kept on
/// stderr.
pub const SCREENSHOT_TARGET: &str = "screenshot";

/// The ~1 Hz batch targets: export-only (dropped from the stderr fmt layer).
const BATCH_TARGETS: &[&str] = &[SALLY_TRACK_TARGET, INPUT_TRACK_TARGET];

#[must_use = "telemetry stops and unflushed data is lost when the guard is dropped"]
pub struct OtelGuard {
    logger_provider: Option<opentelemetry_sdk::logs::SdkLoggerProvider>,
    meter_provider: Option<opentelemetry_sdk::metrics::SdkMeterProvider>,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(p) = &self.logger_provider {
            let _ = p.shutdown();
        }
        if let Some(p) = &self.meter_provider {
            let _ = p.shutdown();
        }
    }
}

/// The project's own telemetry switch, flattened by every binary that calls [`init`] — one
/// declaration of `--otel` and its `RL_OTEL` env fallback, so a value clap does not recognize
/// as falsey turns export ON rather than silently leaving it off (rl#275).
///
/// The `OTEL_*` vars stay env-only by design: they are OTel ecosystem convention, the SDK's
/// contract with whatever launches the process.
#[derive(clap::Args, Debug, Clone, Copy, Default)]
pub struct OtelArgs {
    /// Export metrics/logs to the built-in OTLP endpoint. Setting
    /// `OTEL_EXPORTER_OTLP_ENDPOINT` enables export on its own, and wins.
    #[arg(long = "otel", env = "RL_OTEL", global = true,
          value_parser = clap::builder::FalseyValueParser::new())]
    pub enabled: bool,
}

pub fn init(service_name: &str, args: OtelArgs) -> OtelGuard {
    // `log`-crate records (wgpu_hal, rapier, …) reach this subscriber via
    // tracing-subscriber's default `tracing-log` feature: every `.init()` below installs
    // the LogTracer bridge itself. Do NOT also call `tracing_log::LogTracer::init()` here —
    // a pre-set logger makes those `.init()` calls PANIC (SetLoggerError), which took down
    // every binary at startup and broke the rl-release checkpoint gate (2026-07-02).
    // The flight-recorder target rides on top of whatever RUST_LOG asks for: surfaces
    // default to `warn`-ish filters (rl-demo pre-sets RUST_LOG), and a recorder that a
    // quieter default silently disables is the gap rl#332 exists to close.
    let mut filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    for t in BATCH_TARGETS
        .iter()
        .chain([&VEHICLE_TRANSITION_TARGET, &SCREENSHOT_TARGET])
    {
        filter = filter.add_directive(format!("{t}=info").parse().expect("static directive"));
    }
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(tracing_subscriber::filter::filter_fn(|meta| {
            !BATCH_TARGETS.contains(&meta.target())
        }));

    let endpoint = resolve_endpoint(args.enabled);
    let Some(endpoint) = endpoint else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
        return OtelGuard {
            logger_provider: None,
            meter_provider: None,
        };
    };

    // No span export on purpose: bevy_render opens `info_span!`s per frame with no
    // feature gate, so a span pipeline ships ~6k spans/s from a running game (3 GiB/h
    // at the sink, tv 2026-07-21) — and nothing fleet-side reads traces, only logs.
    match build_providers(service_name, &endpoint) {
        Ok((logger_provider, meter_provider)) => {
            let log_layer = opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
                &logger_provider,
            )
            .with_filter(tracing_subscriber::filter::filter_fn(|meta| {
                exportable_target(meta.target())
            }));
            opentelemetry::global::set_meter_provider(meter_provider.clone());

            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .with(log_layer)
                .init();
            tracing::info!(service_name, endpoint, "OTLP telemetry enabled");
            OtelGuard {
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
                logger_provider: None,
                meter_provider: None,
            }
        }
    }
}

/// Whether a record with this `tracing` target belongs in the OTLP export (it always
/// still reaches stderr). Excluded:
///
/// - The export pipeline's own crates — exporting their records feeds back into export.
/// - gilrs's force-feedback module. No rl surface plays rumble, but bevy_gilrs 0.18
///   hardcodes ff ON, so gilrs holds a second fd per gamepad and pokes it every 50ms
///   tick; when a Bluetooth pad sleeps, the ticks that race the Disconnected event hit
///   the deleted node and log expected ENODEV at ERROR (upstream's own Drop impl
///   treats ENODEV as expected), firing a fleet-error per controller sleep (tv,
///   2026-08-11). Reconnect self-heals: gilrs re-opens the device with the fresh node.
fn exportable_target(t: &str) -> bool {
    !(t.starts_with("opentelemetry")
        || t.starts_with("hyper")
        || t.starts_with("reqwest")
        || t.starts_with("h2")
        || t.starts_with("tonic")
        || t.starts_with("tower")
        || (t.starts_with("gilrs") && t.ends_with("::ff")))
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
    opentelemetry_sdk::logs::SdkLoggerProvider,
    opentelemetry_sdk::metrics::SdkMeterProvider,
);

fn build_providers(service_name: &str, endpoint: &str) -> anyhow::Result<Providers> {
    use opentelemetry_otlp::{LogExporter, MetricExporter, WithExportConfig};

    let service = env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| service_name.to_string());
    let resource = Resource::builder()
        .with_service_name(service)
        .with_attribute(KeyValue::new("host.name", host_name()))
        .with_attribute(KeyValue::new("service.version", build_digest()))
        .with_attributes(env_resource_attributes())
        .build();

    let base = endpoint.strip_suffix('/').unwrap_or(endpoint);

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

    Ok((logger_provider, meter_provider))
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

/// The build the telemetry came from (rl#309). Launchers export `RL_BUILD_DIGEST`
/// (the release's `rl_commit` — deck kits read manifest.json, the TV deploy stages a
/// digest file); an ad-hoc run without one reports `dev`.
fn build_digest() -> String {
    env::var("RL_BUILD_DIGEST")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "dev".to_string())
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

#[cfg(test)]
mod tests {
    use super::exportable_target;

    #[test]
    fn export_filter_drops_pipeline_and_ff_noise_only() {
        for t in [
            "opentelemetry_sdk::logs",
            "hyper::client",
            "gilrs_core::platform::platform::ff",
            "gilrs::ff",
        ] {
            assert!(!exportable_target(t), "{t} should be dropped");
        }
        for t in [
            "gilrs_core::platform::platform::gamepad", // connect/disconnect stays visible
            "gilrs::mapping",
            "net::crab_slot",
            "wgpu_hal",
        ] {
            assert!(exportable_target(t), "{t} should export");
        }
    }
}
