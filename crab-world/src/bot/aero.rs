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
//! policy would feel). `F = −c·|v|·v` with `c = m·g/v_t²` sized from the WHOLE
//! body's mass at spawn ([`CarapaceDrag`]), so terminal velocity is
//! [`TERMINAL_SPEED`] whatever the body masses — the rl#340 stage-3 lesson: a
//! constant sized for the then-extant 0.78 kg fallback body let the 1.97 kg
//! mesh body fall at 22 m/s. 15 m/s is real-animal scale for a 0.6 m body, ~5× her full-charge gait
//! speed (so in-distribution locomotion sees only a few percent of body weight),
//! and low enough that a lip at speed is a hop, not a flight.
//! `airborne_crab_reaches_terminal_velocity` pins the realized v_t against
//! mass/collider drift.

use bevy::prelude::*;
use bevy_rapier3d::prelude::{ExternalForce, ReadMassProperties, Velocity};

use super::body::CrabCarapace;

/// Design free-fall terminal speed, m/s — the one aero tuning knob.
///
/// As a bluff-body check, `c = m·g/v_t²` vs `½·ρ·C_d·A` with air ρ = 1.2 kg/m³,
/// frontal area ≈ 0.14 m² gives C_d ≈ 1.0 (m = 1.97 kg, c ≈ 0.086) — an
/// unremarkable shell.
pub const TERMINAL_SPEED: f32 = 15.0;

/// The carapace's drag coefficient, computed by `spawn_crab` from the summed
/// collider masses of the whole body it just spawned (the same density×shape
/// products rapier integrates). Component state, not a constant, so the
/// coefficient can never drift from the mass of the body actually spawned
/// (rl#340 stage 3). The field is private and the only constructor takes a mass, so "drag not
/// derived from the body's mass" is unrepresentable.
#[derive(Component)]
pub struct CarapaceDrag {
    coeff: f32,
}

impl CarapaceDrag {
    /// The coefficient (N·s²/m²) that gives [`TERMINAL_SPEED`] for a body of
    /// `total_mass` kg: at terminal, `c·v_t² = m·g`.
    pub fn for_total_mass(total_mass: f32) -> Self {
        Self {
            coeff: total_mass * -crate::physics::PHYSICS_GRAVITY.y
                / (TERMINAL_SPEED * TERMINAL_SPEED),
        }
    }

    /// Tests that book drag impulses bit-for-bit read this same value.
    pub fn coeff(&self) -> f32 {
        self.coeff
    }
}

/// Cap on the drag FORCE, in multiples of body weight (`m·g`). Two jobs (rl#339):
///
/// - **Integrator correctness**: explicit `F = −c·|v|·v` is a brake only while its
///   one-tick impulse stays under the momentum of the body it acts on — the
///   CARAPACE link, so past `|v| = m_carapace·Hz/c =
///   (m_carapace/m_total)·Hz·v_t²/g` (~720 m/s) it
///   amplifies geometrically: measured 6000 → 788,000 m/s in ONE tick. With the
///   force capped, the per-tick Δv is a constant `20·g/Hz ≈ 3 m/s` — it can
///   never overshoot a hypersonic velocity, at any speed.
/// - **Solver sanity**: the craft's fix for the same instability
///   ([`crate::physics::brake_coeff_max`], the one-step momentum-cancel coefficient
///   cap) is exact for a lone rigid body but produces ~2×10⁵ N on the carapace, and
///   forces of that scale through the MULTIBODY solve corrupt the generalized
///   position outright — measured: a capped-coefficient brake tick slammed her
///   1×10⁷ m off the map with |v| still ~5×10³ m/s (the rl#347/rl#349 huge-force
///   class, and the actual shape of the rl#339 wedge jumps). So the carapace bound
///   must live on the force, not the coefficient.
///
/// 20× carapace weight engages at `√(20·m_carapace·g/c)` =
/// `v_t·√(20·m_carapace/m_total)` ≈ 47 m/s — over 3× the terminal velocity and
/// beyond anything a legit gait, lip, or fall reaches, so tuned locomotion
/// never feels it. Past it, deceleration is a constant ~20 g; the
/// escaped-crab rescue is the flight-time bound, this is the no-new-energy bound.
const CARAPACE_BRAKE_WEIGHT_MAX: f32 = 20.0;

/// Ordered after [`super::BotSet::Act`] (whose `apply_actions` zeroes
/// `ExternalForce.force` every tick) and before the rapier sync — the same seam
/// the training shoves use.
///
/// The live rapier mass mirror reads zero until the first writeback, so the first
/// tick's drag cap is zero and the drag is dropped — at gait speeds drag is a few
/// percent of body weight, so one tick is nothing.
#[allow(clippy::type_complexity)]
pub(crate) fn apply_air_drag(
    mut carapaces: Query<
        (
            &Velocity,
            &ReadMassProperties,
            &CarapaceDrag,
            &mut ExternalForce,
        ),
        With<CrabCarapace>,
    >,
    undragged: Query<
        Entity,
        (
            With<CrabCarapace>,
            With<ExternalForce>,
            Without<CarapaceDrag>,
        ),
    >,
) {
    // A dynamic carapace with a force accumulator but no drag coefficient is the
    // illegal state this system's query would otherwise skip SILENTLY — no air
    // drag, no error, Sally flies again. Only `spawn_crab` (and the lone-body
    // rl#339 test) may spawn one, and both attach the coefficient at spawn.
    debug_assert!(
        undragged.is_empty(),
        "a physics carapace has no CarapaceDrag — spawned outside spawn_crab?"
    );
    for (vel, mass, drag, mut force) in carapaces.iter_mut() {
        let speed = vel.linear.length();
        let f_max = CARAPACE_BRAKE_WEIGHT_MAX * mass.mass * -crate::physics::PHYSICS_GRAVITY.y;
        // Coefficient form so the in-band expression is BIT-identical to the uncapped
        // original — the 300-tick flail pin in net is sensitive to bit-level physics
        // drift. speed = 0 ⇒ f_max/speed = +inf ⇒ the cap arm never wins; mass = 0
        // (pre-writeback tick) with speed > 0 ⇒ f_max/speed = 0 wins ⇒ the drag is
        // dropped for that one tick (at speed = 0 the zero drag arm wins — `min`
        // returns the non-NaN arm even against the cap's 0/0).
        let coeff = (drag.coeff() * speed).min(f_max / speed);
        // A resting crab must be allowed to SLEEP (rl#392, the rl#377 pattern):
        // bevy_rapier force-wakes on any Changed ExternalForce, and at rest this
        // drag feeds on the contact solver's own noise velocity, re-deriving a
        // not-quite-equal force each tick — a wake loop that pins the sleep timer
        // at zero forever. Below the sleep band the velocity is noise by
        // definition and the drag on it is sub-milli-Newton; snap it to exactly
        // zero and skip the write so sleep can engage. Dissipative drag can never
        // sustain motion, so nothing physical is lost; in-band the expression and
        // the write are bit-identical to before (the 300-tick flail pin).
        let add = if speed < super::body::CRAB_SLEEP_NOISE_FLOOR {
            Vec3::ZERO
        } else {
            -coeff * vel.linear
        };
        if add != Vec3::ZERO {
            force.force += add;
        }
    }
}
