//! The demo/screenshot bevy wiring for the trained [`Policy`]: the `Think`-set system that
//! turns the loaded policy into per-tick actions, plus the `add_inference` helper that
//! installs it. The policy itself — checkpoint loading + deterministic mean-action inference —
//! is the headless [`crate::policy`], reused unchanged by these renderers and the
//! trainer-side eval ([`crate::eval`]), so there is ONE inference path, not a rendered copy
//! and a headless copy that could drift.

use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::sensor::CrabObservation;
use crate::crab_view::CrabBrainLabels;
use crate::policy::Policy;

use super::manual_control::ManualControl;

/// System (BotSet::Think): run the policy and write the actions the actuator will
/// apply — unless manual control has taken over (then `manual_control_step` drives).
pub(super) fn policy_step(
    policy: NonSend<Policy>,
    manual: Option<Res<ManualControl>>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    mut warned_no_env: Local<bool>,
) {
    if manual.is_some_and(|m| m.active) {
        return;
    }
    let (Some(o), Some(a)) = (obs.envs.first(), actions.envs.first_mut()) else {
        // Env 0 is sized at startup and lives the whole run, so a missing slot here is a
        // wiring bug (the policy could never drive the crab), not a respawn transient —
        // surface it. Latched so a persistent miswire logs once, not every tick.
        if !*warned_no_env {
            error!("play: env-0 observation/action slot missing — policy cannot drive the crab");
            *warned_no_env = true;
        }
        return;
    };
    *a = policy.act(o);
}

/// Load the trained policy as a resource. The driver system that turns it into
/// actions (`policy_step`) is added by the caller, so `--manual-control` can swap
/// in [`super::manual_control::manual_control_step`] instead.
pub(super) fn add_inference(app: &mut App, checkpoint_dir: &Path, live_dir: Option<PathBuf>) {
    let mut policy = Policy::load(checkpoint_dir);
    policy.set_live_dir(live_dir);
    app.insert_non_send_resource(policy);
    // The demo's single crab wears its brain's identity on screen (rl#200 increment 7).
    // Republished every frame (write-on-change) rather than set once so a hot-reload swap
    // relabels the crab the same tick the new brain takes over.
    app.add_systems(Update, publish_brain_label);
}

/// Keep env 0's world-space brain label current with the (possibly hot-reloaded) policy.
fn publish_brain_label(policy: NonSend<Policy>, mut labels: ResMut<CrabBrainLabels>) {
    let want = policy.brain_label();
    if labels.0.first() != Some(&want) {
        labels.0 = vec![want];
    }
}
