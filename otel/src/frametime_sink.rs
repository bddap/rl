//! The native sink for [`frametime`]'s snapshot queue (bddap/rl#309): a background
//! thread replays each one-second bucket snapshot into the OTLP metrics SDK, which
//! exports through the existing OTLP pipe with the process Resource (host.name =
//! device, service.version = build digest). With export disabled the global meter is
//! a no-op, so the thread idles at ~2 wakeups/s and the recorder still costs the
//! frame path nothing.

use std::time::Duration;

/// Register the OTLP flusher as the process' frametime sink — called by
/// [`crate::init`] on every path (export on or off), before any recorder starts.
pub(crate) fn install() {
    frametime::install_sink(|rx| {
        std::thread::Builder::new()
            .name("otel-frametime".into())
            .spawn(move || flusher(rx))
            .expect("spawning otel-frametime flusher");
    });
}

fn flusher(rx: frametime::SnapshotRx) {
    let histogram = crate::meter()
        .f64_histogram("rl.frame_time_ms")
        .with_unit("ms")
        .with_boundaries(frametime::boundaries_ms())
        .build();
    loop {
        std::thread::sleep(Duration::from_millis(500));
        while let Some(counts) = rx.pop() {
            for (i, &c) in counts.iter().enumerate() {
                let mid = frametime::midpoint_ms(i);
                for _ in 0..c {
                    histogram.record(mid, &[]);
                }
            }
        }
    }
}
