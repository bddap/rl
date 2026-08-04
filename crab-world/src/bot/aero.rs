//! Aerodynamic drag on the carapace — the speed bound the plant was missing
//! (bddap/rl#332).
//!
//! Without any air resistance the GCR mountainsides turn Sally into a lossless
//! luge: the policy (or a passive tumble — the zero-drive soak ablation) trades
//! 100+ m of descent for 30–55 m/s and sails off terrain lips tens of meters into
//! the air — the owner-reported "flight". The rl#321/#332 instruments cleared the
//! solver (passive energy strictly dissipates on flat AND on terrain); the
//! remaining cause is un-modeled aerodynamics, so the fix is to model it.
//!
//! Quadratic drag on the carapace only: the carapace is the bluff body and
//! carries most of the mass, while per-limb drag would damp the fast limb whips
//! the trained gait is built from (an in-distribution plant change the frozen
//! policy would feel). `F = −DRAG·|v|·v` gives terminal velocity
//! `v_t = √(m·g/DRAG)` ≈ 15 m/s at her measured 0.78 kg — real-animal scale for a
//! 0.6 m body, ~5× her full-charge gait speed (so in-distribution locomotion sees
//! only a few percent of body weight), and low enough that a lip at speed is a
//! hop, not a flight. `airborne_crab_reaches_terminal_velocity` pins the realized
//! v_t against mass/collider drift.

use bevy::prelude::*;
use bevy_rapier3d::prelude::{ExternalForce, Velocity};

use super::body::CrabCarapace;

/// Drag coefficient, N·s²/m²: `m·g / v_t²` = 0.781 kg × 9.81 / (15 m/s)². As a
/// bluff-body check, ½·ρ·C_d·A with air ρ = 1.2 kg/m³, frontal area ≈ 0.14 m²
/// gives C_d ≈ 0.4 — an unremarkable shell.
pub const CARAPACE_DRAG: f32 = 0.034;

/// Ordered after [`super::BotSet::Act`] (whose `apply_actions` zeroes
/// `ExternalForce.force` every tick) and before the rapier sync — the same seam
/// the training shoves use.
pub(crate) fn apply_air_drag(
    mut carapaces: Query<(&Velocity, &mut ExternalForce), With<CrabCarapace>>,
) {
    for (vel, mut force) in carapaces.iter_mut() {
        force.force += -(CARAPACE_DRAG * vel.linear.length()) * vel.linear;
    }
}
