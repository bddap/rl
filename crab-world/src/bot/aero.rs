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
use bevy_rapier3d::prelude::{ExternalForce, ReadMassProperties, Velocity};

use super::body::CrabCarapace;

/// Drag coefficient, N·s²/m²: `m·g / v_t²` = 0.781 kg × 9.81 / (15 m/s)². As a
/// bluff-body check, ½·ρ·C_d·A with air ρ = 1.2 kg/m³, frontal area ≈ 0.14 m²
/// gives C_d ≈ 0.4 — an unremarkable shell.
pub const CARAPACE_DRAG: f32 = 0.034;

/// Cap on the drag FORCE, in multiples of body weight (`m·g`). Two jobs (rl#339):
///
/// - **Integrator correctness**: explicit `F = −c·|v|·v` is a brake only while its
///   one-tick impulse stays under the momentum it cancels; past `|v| = m·Hz/c`
///   (≈ 1470 m/s here) it amplifies geometrically — measured 6000 → 788,000 m/s in
///   ONE tick. With the force capped, the per-tick Δv is a constant
///   `20·g/Hz ≈ 3 m/s` — it can never overshoot a hypersonic velocity, at any speed.
/// - **Solver sanity**: the craft's fix for the same instability
///   ([`crate::physics::brake_coeff_max`], the one-step momentum-cancel coefficient
///   cap) is exact for a lone rigid body but produces ~2×10⁵ N on the carapace, and
///   forces of that scale through the MULTIBODY solve corrupt the generalized
///   position outright — measured: a capped-coefficient brake tick slammed her
///   1×10⁷ m off the map with |v| still ~5×10³ m/s (the rl#347/rl#349 huge-force
///   class, and the actual shape of the rl#339 wedge jumps). So the carapace bound
///   must live on the force, not the coefficient.
///
/// 20× weight engages at `√(20·m·g/CARAPACE_DRAG)` ≈ 67 m/s — 4.5× the rl#332
/// terminal velocity and beyond anything a legit gait, lip, or fall reaches, so
/// tuned locomotion never feels it. Past it, deceleration is a constant ~20 g; the
/// escaped-crab rescue is the flight-time bound, this is the no-new-energy bound.
const CARAPACE_BRAKE_WEIGHT_MAX: f32 = 20.0;

/// Ordered after [`super::BotSet::Act`] (whose `apply_actions` zeroes
/// `ExternalForce.force` every tick) and before the rapier sync — the same seam
/// the training shoves use.
///
/// The live rapier mass mirror reads zero until the first writeback, so the first
/// tick's drag cap is zero and the drag is dropped — at gait speeds drag is a few
/// percent of body weight, so one tick is nothing.
pub(crate) fn apply_air_drag(
    mut carapaces: Query<(&Velocity, &ReadMassProperties, &mut ExternalForce), With<CrabCarapace>>,
) {
    for (vel, mass, mut force) in carapaces.iter_mut() {
        let speed = vel.linear.length();
        let f_max = CARAPACE_BRAKE_WEIGHT_MAX * mass.mass * -crate::physics::PHYSICS_GRAVITY.y;
        // Coefficient form so the in-band expression is BIT-identical to the uncapped
        // original — the 300-tick flail pin in net is sensitive to bit-level physics
        // drift. speed = 0 ⇒ f_max/speed = +inf ⇒ the cap arm never wins; mass = 0
        // (pre-writeback tick) ⇒ 0/0 = NaN and `f32::min` returns the OTHER arm —
        // uncapped for that one tick, exactly the pre-cap behavior.
        let coeff = (CARAPACE_DRAG * speed).min(f_max / speed);
        force.force += -coeff * vel.linear;
    }
}
