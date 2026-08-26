// rl#282: sibling sim suites have wedged (all threads futex_wait, 0% CPU) under
// trainer saturation; abort loudly on a process-wide CPU flatline instead.
#[cfg(test)]
test_watchdog::arm!();

pub mod controls;
pub mod formation;
pub mod net_loop;
pub mod telemetry;
// Voice notes ride bothouse-side HTTP + local files — native tooling, no web story.
#[cfg(not(target_family = "wasm"))]
pub mod voice_delivery;
#[cfg(not(target_family = "wasm"))]
pub mod voice_reply;

// The link layer lives in `net-link` (rl#411 stage 4: one poll-driven Session
// surface, native + web platform impls behind it, wasm32-checked). Re-exported under
// the same path so `net::transport::Session` etc. stay stable for every consumer.
pub use net_link as transport;

// The protocol core lives in `net-proto` (rl#411 stage 2: platform-free by
// construction — no tokio/iroh, wasm32-checked). Re-exported under the same paths
// so `net::sim` etc. stay stable for every consumer.
pub use net_proto::{
    SyncStamp, SyncVerdict, articulation, cadence, client, cordic, may_arm_crabs, membership,
    roster, server, sim, snapshot, wire,
};

// Render-free since rl#298 stage 4: only the label publisher inside stays
// render-gated — it feeds UI.
pub mod crab_slot;
#[cfg(feature = "render")]
pub mod menu;
pub mod probe;
#[cfg(feature = "render")]
pub mod render;

/// Serializes the `#[ignore]`d real-endpoint tests: every live iroh endpoint on the box
/// mDNS-discovers and dials every other, so two lobby tests running at once merge into
/// one oversized roster. A lock they all take beats a `--test-threads=1` flag someone
/// must remember to pass.
#[cfg(test)]
pub(crate) fn real_net_serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
