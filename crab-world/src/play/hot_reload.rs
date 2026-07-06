
use bevy::prelude::*;

use crate::policy::Policy;

const HOT_RELOAD_INTERVAL_S: f32 = 2.0;

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
