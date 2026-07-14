use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::sensor::CrabObservation;
use crate::crab_view::CrabBrainLabels;
use crate::policy::{Policy, RestFallback};

use super::manual_control::ManualControl;

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
    let landed = match obs.rows().first() {
        Some(o) => actions.set_row(0, policy.act(o)),
        None => false,
    };
    if !landed && !*warned_no_env {
        error!("play: env-0 observation/action slot missing — policy cannot drive the crab");
        *warned_no_env = true;
    }
}

pub(super) fn add_inference(
    app: &mut App,
    checkpoint_dir: &Path,
    live_dir: Option<PathBuf>,
    random_policy: bool,
) {
    let fallback = if random_policy {
        RestFallback::RandomBrain
    } else {
        RestFallback::Rest
    };
    let mut policy = Policy::load(checkpoint_dir, fallback);
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
