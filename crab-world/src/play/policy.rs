use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::sensor::CrabObservation;
use crate::crab_view::CrabBrainLabels;
use crate::policy::Policy;

use super::manual_control::ManualControl;

pub(super) fn policy_step(
    policy: NonSend<Policy>,
    manual: Option<Res<ManualControl>>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    mut warned_no_env: Local<bool>,
    mut rollout: Option<ResMut<super::RenderRollout>>,
) {
    if manual.is_some_and(|m| m.active) {
        return;
    }
    let landed = match obs.rows().first() {
        Some(o) => actions.set_row(
            0,
            policy.act_with_noise(o, rollout.as_mut().and_then(|r| r.exploration())),
        ),
        None => false,
    };
    if !landed && !*warned_no_env {
        error!("play: env-0 observation/action slot missing — policy cannot drive the crab");
        *warned_no_env = true;
    }
}

pub(super) fn add_inference(app: &mut App, checkpoint_dir: &Path, live_dir: Option<PathBuf>) {
    let mut policy = Policy::load(checkpoint_dir);
    policy.set_live_dir(live_dir);
    app.insert_non_send(policy);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::headless::{flat_headless_app, tick};
    use bevy::ecs::system::RunSystemOnce;

    #[test]
    fn policy_step_uses_optional_seeded_exploration() {
        let mut app = flat_headless_app();
        tick(&mut app, 1);
        let policy = Policy::untrained();
        let mean = policy.act(&app.world().resource::<CrabObservation>().rows()[0]);
        app.insert_non_send(policy);
        app.world_mut().run_system_once(policy_step).unwrap();
        assert_eq!(app.world().resource::<CrabActions>().rows()[0], mean);
        let mut previous = None;
        for seed in [351, 352, 351] {
            app.insert_resource(super::super::RenderRollout::new(9.0, Some(-1.0), seed));
            app.world_mut().run_system_once(policy_step).unwrap();
            let action = app.world().resource::<CrabActions>().rows()[0];
            assert_ne!(action, mean);
            if let Some(prior) = previous {
                assert_ne!(action, prior);
            }
            previous = Some(action);
            app.world_mut().run_system_once(policy_step).unwrap();
            assert_ne!(app.world().resource::<CrabActions>().rows()[0], action);
        }
        app.world_mut()
            .remove_resource::<super::super::RenderRollout>();
        app.world_mut().run_system_once(policy_step).unwrap();
        assert_eq!(app.world().resource::<CrabActions>().rows()[0], mean);
    }
}
