pub mod world;

pub use world::PhysicsWorldPlugin;

use bevy_rapier3d::plugin::RapierContextInitialization;
use bevy_rapier3d::prelude::{RapierConfiguration, TimestepMode};
use bevy_rapier3d::rapier::dynamics::{IntegrationParameters, SpringCoefficients};

/// One Bevy tick advances physics by exactly this, split into [`PHYSICS_SUBSTEPS`]
/// solver sub-steps. Shared by production AND every headless test via
/// [`fixed_timestep`], so the three can never drift — a test that ran different
/// physics than the demo/training loop would prove nothing.
pub const PHYSICS_DT: f32 = 1.0 / 64.0;

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
/// Rapier's 30 Hz / 5.0 default. The resting-crab bounce (issue #19) is a CONTACT
/// event, not a joint one: at rest the floppy legs hang slack *off* their limit
/// stops (they crumple ~0.7 rad from rest), so the joint-limit spring is not even
/// engaged and softening it does nothing measurable to the bounce — a parameter
/// sweep found rest jitter completely flat across limit frequencies 30–400 Hz.
/// What the bounce *is* sensitive to is this contact spring: at 2 sub-steps the
/// 30 Hz default is under-resolved against a foot touchdown, so the impulse rings
/// up through the body. Lowering it to 12 Hz absorbs the touchdown locally —
/// carapace bounce 3.6 cm → 0.32 cm (it roughly halves at each step 30 → 16 → 12)
/// and rest angular speed to ~0.6 rad/s, feet still penetrating < 4 mm. 12 Hz is
/// the floor: 10 Hz climbs back as the over-soft contact lets the body wallow.
/// Legs still crumple (the rest-quiet fix must not stiffen them into a brace).
///
/// This contact spring is the ONLY lever that moves the bounce — a measured sweep
/// ruled out two other guesses: `length_unit` is flat from 1.0 down to 0.1 (the
/// touchdown is not tolerance-limited, it's contact-spring stiffness), and scaling
/// the whole model up makes the bounce relative to size WORSE and would force a
/// retrain. So `length_unit` stays 1.0 and the model is not rescaled.
///
/// Crucially this is FREE for the #17 mid-air invariant: an airborne crab is
/// contact-free, so the contact spring is dead code there — `airborne_crab_…`
/// stays well under its runaway guard. That sidesteps the joint-limit tension (a
/// stop soft enough to cap mid-air overshoot vs stiff enough not to jitter at
/// rest); the rest jitter never lived in the joint limit, so #17's limit softness
/// is untouched.
pub const CONTACT_SOFTNESS: SpringCoefficients<f32> = SpringCoefficients {
    natural_frequency: 12.0,
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
