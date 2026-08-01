//! The GCR brain-swap button (rl#232): cycle which trained brain drives each NN crab,
//! live, through the ONE swap path ([`crab_world::policy::Policy::cycle_brain`]) the demo
//! also uses. Sally's brain selection stays where Sally is simulated — the solo/host
//! peer; a remote-adopt client renders whatever the host drives (its label arrives on
//! the articulation wire), so its button is a no-op by design, not a second code path.

use bevy::prelude::*;

use crab_world::controls::ActiveContext;

use crate::controls::{Action, GcrContext, GcrControls};
use crate::crab_slot::CrabPolicies;

use super::driver::GameState;

pub(super) fn swap_brain(
    chords: Res<crab_world::chord::Chords<GcrControls>>,
    ctx: Res<ActiveContext<GcrControls>>,
    state: Option<NonSend<GameState>>,
    policies: Option<NonSendMut<CrabPolicies>>,
) {
    // On-foot only: while piloting, a mistyped code must not be able to swap Sally.
    // Context gating stays HERE, dispatcher-side (rl#330 stage-2 decision): the
    // registry is pure code→command data, and which contexts allow a command is the
    // dispatcher's business, same as the Playing gate on the view cycles.
    if ctx.get() != GcrContext::OnFoot || !chords.executed(Action::SwapBrain) {
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
