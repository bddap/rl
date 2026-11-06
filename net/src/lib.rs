pub mod controls;
pub mod formation;
pub mod net_loop;

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
    net_link::confine_discovery_to_process();
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
