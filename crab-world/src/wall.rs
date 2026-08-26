//! Wall-clock unix millis, platform-safe: the track schemas (sally_track,
//! net_track) stamp samples with epoch time, and `std::time::SystemTime` panics on
//! wasm — the browser's clock is `Date.now()`. One helper so no sample path reads a
//! platform clock directly (rl#411 stage 5).

#[cfg(not(target_family = "wasm"))]
pub fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(target_family = "wasm")]
pub fn unix_ms() -> u64 {
    js_sys::Date::now() as u64
}
