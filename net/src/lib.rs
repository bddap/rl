pub mod controls;
pub mod formation;
pub mod net_loop;

// The link layer lives in `net-link` (rl#411 stage 4: one poll-driven Session
// surface, native + web platform impls behind it, wasm32-checked). Re-exported under
// the same path so `net::transport::Session` etc. stay stable for every consumer.
pub use net_link as transport;

// The protocol core lives in `net-proto` (rl#411 stage 2: platform-free by
// construction — no tokio/iroh, wasm32-checked). Re-exported under the same paths
// so `net::sim` etc. stay stable for every consumer.
pub use net_proto::{
    SyncStamp, cadence, client, cordic, membership, roster, server, sim, snapshot, vehicles, wire,
};

pub mod menu;
#[cfg(feature = "render")]
pub mod render;

/// Long enough that only a real deadlock reaches it; never how a test tells success apart.
#[cfg(test)]
pub(crate) const HANG_GUARD: std::time::Duration = std::time::Duration::from_secs(120);

/// Serializes the real-endpoint tests: every live iroh endpoint in the process
/// mDNS-discovers and dials every other, so two lobby tests running at once merge into
/// one oversized roster; the lock serializes them, and the per-process service name keeps
/// other processes' endpoints out.
#[cfg(test)]
pub(crate) fn real_net_serial() -> std::sync::MutexGuard<'static, ()> {
    transport::confine_discovery_to_process();
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
