//! The GCR brain-swap button (rl#232): cycle which trained brain drives each NN crab,
//! live, through the ONE swap path ([`crab_world::policy::Policy::cycle_brain`]) the demo
//! also uses. Sally's brain selection stays where Sally is simulated — the solo/host
//! peer; a remote-adopt client renders whatever the host drives (its label arrives on
//! the articulation wire), so its button is a no-op by design, not a second code path.

use bevy::prelude::*;

use crab_world::controls::ActiveContext;

use crate::controls::{Action, GcrContext, GcrControls};
use crate::external_crab::CrabPolicies;

use super::driver::GameState;

pub(super) fn swap_brain(
    keys: Res<ButtonInput<KeyCode>>,
    pads: Query<&Gamepad>,
    ctx: Res<ActiveContext<GcrControls>>,
    state: Option<NonSend<GameState>>,
    policies: Option<NonSendMut<CrabPolicies>>,
) {
    // On-foot only, matching the one FOOT_ROWS legend row — and R3 is the click of the
    // stick that aims the ship / attitudes the plane, so a piloting hand must not be
    // able to swap Sally by accident.
    if ctx.get() != GcrContext::OnFoot
        || !crab_world::controls::just_pressed::<GcrControls>(Action::SwapBrain, &keys, &pads)
    {
        return;
    }
    let Some(state) = state else {
        return;
    };
    if state.coord.is_remote_client() {
        info!("brain swap: this peer adopts the host's Sally — only the host swaps brains");
        return;
    }
    let Some(mut policies) = policies else {
        return;
    };
    // Every bridged crab cycles within its own roster (its boot dir + that dir's brain
    // subdirs) — with one crab this is the latest↔keep-best toggle; the swapped label
    // reaches every peer through `publish_brain_labels` and the articulation wire.
    for policy in policies.0.iter_mut() {
        policy.cycle_brain();
    }
}
