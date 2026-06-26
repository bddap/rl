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

/// The fixed physics timestep. Private: [`CrabPhysicsPlugin`] is the one entry
/// point that installs it, so no caller can pick a different timestep and run
/// physics the policy never trained under.
fn fixed_timestep() -> TimestepMode {
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
/// creation. Not called directly by app builders: [`CrabPhysicsPlugin`] inserts
/// it (in the one order that works) so all worlds share one contact spring and
/// can't drift; `RapierPhysicsPlugin` spawns the context from it at `PreStartup`.
fn rapier_context_init() -> RapierContextInitialization {
    RapierContextInitialization::InitializeDefaultRapierContext {
        integration_parameters: IntegrationParameters {
            contact_softness: CONTACT_SOFTNESS,
            ..IntegrationParameters::default()
        },
        rapier_configuration: RapierConfiguration::new(1.0),
    }
}

/// The crab's complete Rapier setup, bundled so the init ordering is impossible to
/// get wrong. The contact-spring-seeded context init ([`rapier_context_init`]) MUST
/// already be present when `RapierPhysicsPlugin::build` runs — the plugin only keeps
/// a pre-existing [`RapierContextInitialization`], otherwise it inserts its own
/// default and the softened contact spring silently never applies (wrong physics, no
/// compile error, no panic). Registering both here, in order, replaces that
/// comment-enforced invariant with a structural one: every world (demo, headless
/// training/tests, solo netcode render) does a single `add_plugins(CrabPhysicsPlugin)`
/// and cannot insert the two out of order. The shared [`fixed_timestep`] rides along
/// for the same one-source reason.
///
/// One residual trap Bevy can't make unrepresentable: resources are last-write-wins,
/// so a caller who ALSO inserts their own [`RapierContextInitialization`] (or adds a
/// bare `RapierPhysicsPlugin`) after this plugin silently clobbers the spring. So this
/// plugin also installs [`assert_contact_spring_applied`] at `PostStartup` — a
/// fail-loud runtime guard, in EVERY world (demo/render/train), that the spawned
/// context really carries [`CONTACT_SOFTNESS`]. The `contact_spring_is_applied` test
/// proves the guard catches a dropped spring on the headless path.
pub struct CrabPhysicsPlugin;

impl bevy::app::Plugin for CrabPhysicsPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        use bevy::app::PostStartup;
        use bevy_rapier3d::plugin::{NoUserData, RapierPhysicsPlugin};
        app.insert_resource(fixed_timestep())
            .insert_resource(rapier_context_init())
            .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
            .add_systems(PostStartup, assert_contact_spring_applied);
    }
}

/// Fail loud, in every world, if the spawned Rapier context doesn't carry the
/// crab's softened contact spring — the runtime backstop for the last-write-wins
/// resource trap [`CrabPhysicsPlugin`] can't close at compile time. Runs once at
/// `PostStartup` (after `RapierPhysicsPlugin` spawns the context at `PreStartup`):
/// read-only, so it can't perturb the physics it guards. A panic here means the
/// init was overridden after the plugin — wrong physics that would otherwise pass
/// silently into the GCR lockstep.
fn assert_contact_spring_applied(
    ctx: bevy::ecs::system::Query<
        &bevy_rapier3d::plugin::context::RapierContextSimulation,
        bevy::ecs::query::With<bevy_rapier3d::plugin::context::DefaultRapierContext>,
    >,
) {
    let spring = ctx
        .single()
        .expect("CrabPhysicsPlugin: exactly one default Rapier context")
        .integration_parameters
        .contact_softness;
    assert_eq!(
        (spring.natural_frequency, spring.damping_ratio),
        (
            CONTACT_SOFTNESS.natural_frequency,
            CONTACT_SOFTNESS.damping_ratio
        ),
        "CrabPhysicsPlugin: the spawned Rapier context lost CONTACT_SOFTNESS — its \
         RapierContextInitialization was overridden after the plugin (last-write-wins). \
         The contact spring is silently wrong; fix the init ordering at the call site."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::test_util::headless_app;
    use bevy::prelude::With;
    use bevy_rapier3d::prelude::{DefaultRapierContext, RapierContextSimulation};

    /// The runtime guard the init-ordering comment used to be: after the real
    /// production stack builds and runs `PreStartup`, the spawned Rapier context must
    /// carry our softened [`CONTACT_SOFTNESS`], not Rapier's stiff default. If a
    /// future plugin reorder broke the ordering [`CrabPhysicsPlugin`] enforces, the
    /// plugin would seed its own default context and this assert fails loudly —
    /// instead of the contact spring silently dropping and the physics drifting under
    /// the GCR lockstep that proved bit-identical on hardware.
    #[test]
    fn contact_spring_is_applied() {
        let mut app = headless_app();
        // First update runs PreStartup, where RapierPhysicsPlugin spawns the context
        // from the init resource CrabPhysicsPlugin inserted ahead of it.
        app.update();
        let mut q = app
            .world_mut()
            .query_filtered::<&RapierContextSimulation, With<DefaultRapierContext>>();
        let ctx = q.single(app.world()).expect("one default rapier context");
        let spring = ctx.integration_parameters.contact_softness;
        assert_eq!(
            spring.natural_frequency, CONTACT_SOFTNESS.natural_frequency,
            "contact spring natural_frequency lost — init ordering broke"
        );
        assert_eq!(
            spring.damping_ratio, CONTACT_SOFTNESS.damping_ratio,
            "contact spring damping_ratio lost — init ordering broke"
        );
    }
}
