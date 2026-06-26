pub mod world;

pub use world::PhysicsWorldPlugin;

use bevy_rapier3d::plugin::RapierContextInitialization;
use bevy_rapier3d::prelude::{RapierConfiguration, TimestepMode};
use bevy_rapier3d::rapier::dynamics::{IntegrationParameters, SpringCoefficients};

/// Physics step rate (Hz) — the CANONICAL source. The crab body + the brain's
/// Sense→Think→Act loop advance at this rate; the sim/lockstep ([`crate::net::sim::TICK_HZ`])
/// runs slower, so the networked arm reconciles the two with a deterministic integer cadence
/// ([`crate::net::cadence::PhysicsCadence`]). Kept as the source (and [`PHYSICS_DT`] derived
/// from it) so the cadence math and the timestep can never disagree about the rate.
pub const PHYSICS_HZ: u64 = 64;

/// One Bevy tick advances physics by exactly this, split into [`PHYSICS_SUBSTEPS`]
/// solver sub-steps. Shared by production AND every headless test via
/// [`fixed_timestep`], so the three can never drift — a test that ran different
/// physics than the demo/training loop would prove nothing. Derived from
/// [`PHYSICS_HZ`] so the rate has one source.
pub const PHYSICS_DT: f32 = 1.0 / PHYSICS_HZ as f32;

/// Solver sub-steps per tick. At one sub-step the solver can't converge the floppy
/// legs + soft joint limits resting under the zero policy, so the crab buzzes
/// (issue #18); more sub-steps shrink the per-solve dt and the twitch falls off.
/// But sub-steps cost ~linearly and 4 tanked the realtime demo framerate, so 2 is
/// the balance: airborne angular momentum stays bounded (no runaway, same as 4) and
/// rest clears the quiet bar, at half the solver cost. The residual rest bounce at
/// 2 sub-steps was the *contact* spring, not sub-step count — see [`CONTACT_SOFTNESS`].
pub const PHYSICS_SUBSTEPS: usize = 2;

/// The fixed physics timestep. Use everywhere (main + headless tests) so the
/// simulation is identical and reproducible across all three.
pub fn fixed_timestep() -> TimestepMode {
    TimestepMode::Fixed {
        dt: PHYSICS_DT,
        substeps: PHYSICS_SUBSTEPS,
    }
}

/// Contact constraint spring (natural frequency Hz, damping ratio), overriding
/// Rapier's 30 Hz / 5.0 default. The resting-crab jitter (issue #19) is a CONTACT
/// event, not a joint one: at rest the floppy legs hang slack *off* their limit
/// stops (they crumple ~0.7 rad from rest), so the joint-limit spring is not even
/// engaged and softening it does nothing measurable — a parameter sweep found rest
/// jitter flat across limit frequencies 30–400 Hz. The jitter lives in this contact
/// spring: at 2 sub-steps a stiff (high-frequency) spring is under-resolved against a
/// foot touchdown, so the corrective impulse rings up through the body. The fix is to
/// drop the frequency until the spring's period is long relative to a sub-step
/// (~7.8 ms): 5 Hz has a 200 ms period and the solver resolves it cleanly. Measured
/// rest: carapace angular speed ~0.11 rad/s, bounce ~0.15 cm — quieter than any
/// stiffer setting tested; legs still crumple (the quiet must not come from stiffening
/// them into a brace).
///
/// The cost is visual: a softer contact lets the feet sink further into the floor
/// before it pushes back. The owner judged that sink preferable to the residual
/// wobble a stiffer contact leaves, so this sits at the soft end. It does shift the
/// standing/contact dynamics the policy trains against (unlike the airborne phase
/// below), so changing this frequency wants a training resume.
///
/// This contact spring is the ONLY lever that moves the jitter — a measured sweep
/// ruled out two other guesses: `length_unit` is flat from 1.0 down to 0.1 (not
/// tolerance-limited), and scaling the whole model up makes the jitter relative to
/// size WORSE and would force a retrain. So `length_unit` stays 1.0 and the model is
/// not rescaled.
///
/// Crucially the contact spring is dead code mid-air (an airborne crab makes no
/// contact), so this is FREE for the #17 mid-air invariant — `airborne_crab_…` stays
/// well under its runaway guard, and #17's joint-limit softness is untouched.
pub const CONTACT_SOFTNESS: SpringCoefficients<f32> = SpringCoefficients {
    natural_frequency: 5.0,
    damping_ratio: 5.0,
};

/// How Rapier seeds its default physics context — same as the built-in default
/// but with [`CONTACT_SOFTNESS`] baked into the integration parameters from
/// creation. Insert this BEFORE `RapierPhysicsPlugin` (main + training + tests)
/// so all three share one contact spring and can't drift; the plugin spawns the
/// context from it at `PreStartup`.
pub fn rapier_context_init() -> RapierContextInitialization {
    RapierContextInitialization::InitializeDefaultRapierContext {
        integration_parameters: IntegrationParameters {
            contact_softness: CONTACT_SOFTNESS,
            ..IntegrationParameters::default()
        },
        rapier_configuration: RapierConfiguration::new(1.0),
    }
}
