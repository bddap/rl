//! The demo's checkpoint hot-reload: a timer-driven system that swaps in a newer
//! checkpoint from the live training dir so a left-open demo tracks training without a
//! relaunch. The swap mechanism (and its mid-save-race safety) lives on the policy itself
//! ([`Policy::try_hot_reload`]); this module owns only the cadence.

use bevy::prelude::*;

use crate::policy::Policy;

/// Interval between demo hot-reload checks. Loading the brain is cheap but not
/// free, and training saves no faster than its PPO-update cadence, so a couple of
/// seconds keeps the demo live without re-reading the file every frame.
const HOT_RELOAD_INTERVAL_S: f32 = 2.0;

/// System (demo only): every [`HOT_RELOAD_INTERVAL_S`], swap in a newer checkpoint
/// from the live training dir so a left-open demo tracks training without a
/// relaunch. No-op unless `--live-checkpoint-dir` was given. See
/// [`Policy::try_hot_reload`].
pub(super) fn hot_reload_policy(
    time: Res<Time>,
    mut policy: NonSendMut<Policy>,
    mut since: Local<f32>,
) {
    *since += time.delta_secs();
    if *since < HOT_RELOAD_INTERVAL_S {
        return;
    }
    *since = 0.0;
    if policy.try_hot_reload() {
        info!("play: hot-reloaded a newer checkpoint from live training");
    }
}
