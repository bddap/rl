pub mod world;

pub use world::PhysicsWorldPlugin;

use bevy_rapier3d::math::Vect;
use bevy_rapier3d::plugin::RapierContextInitialization;
use bevy_rapier3d::prelude::{RapierConfiguration, TimestepMode};
use bevy_rapier3d::rapier::dynamics::{IntegrationParameters, SpringCoefficients};

/// Physics step rate (Hz) — the CANONICAL source. The crab body + the brain's
/// Sense→Think→Act loop advance at this rate; the sim/lockstep (`net::sim::TICK_HZ`)
/// runs slower, so the networked arm reconciles the two with a deterministic integer cadence
/// (`net::cadence::PhysicsCadence`). Kept as the source (and [`PHYSICS_DT`] derived
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

/// World length unit (1 sim unit = 1 metre): the scale Rapier's solver tolerances and
/// its default gravity are expressed in. The one source [`PHYSICS_GRAVITY`] is pinned
/// against (`gravity_matches_rapier_default`); we override gravity itself explicitly.
const LENGTH_UNIT: f32 = 1.0;

/// Gravity, set EXPLICITLY on the Rapier config rather than inherited from
/// `RapierConfiguration::new`'s default. The determinism contract is bit-identity, and
/// the GCR lockstep (proven cross-machine on hardware, rl#82) integrates this exact
/// vector every tick — so it must not be an unguarded dependency on an upstream
/// constant a Rapier bump or a stray config edit could silently flip and desync the
/// trained checkpoint.
///
/// Value-equal to Rapier's default at [`LENGTH_UNIT`] (`Vect::Y * -9.81`): y is the
/// same `-9.81` f32 and the horizontal components are zero, so the trajectory is
/// bit-identical and the determinism/lockstep tests hold. `gravity_matches_rapier_default`
/// pins this equality — an upstream change to the default fails there, in review,
/// instead of silently shifting physics on the next rebuild.
pub const PHYSICS_GRAVITY: Vect = Vect::new(0.0, -9.81, 0.0);

/// How Rapier seeds its default physics context — Rapier's default but with
/// [`CONTACT_SOFTNESS`] baked into the integration parameters and [`PHYSICS_GRAVITY`]
/// set explicitly. Not called directly by app builders: [`CrabPhysicsPlugin`] inserts
/// it (in the one order that works) so all worlds share one contact spring and
/// can't drift; `RapierPhysicsPlugin` spawns the context from it at `PreStartup`.
fn rapier_context_init() -> RapierContextInitialization {
    RapierContextInitialization::InitializeDefaultRapierContext {
        integration_parameters: IntegrationParameters {
            contact_softness: CONTACT_SOFTNESS,
            ..IntegrationParameters::default()
        },
        rapier_configuration: RapierConfiguration {
            gravity: PHYSICS_GRAVITY,
            ..RapierConfiguration::new(LENGTH_UNIT)
        },
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
/// plugin also installs [`assert_contact_spring_applied`] and
/// [`assert_gravity_applied`] at `PostStartup` — fail-loud runtime guards, in EVERY
/// world (demo/render/train), that the spawned context really carries
/// [`CONTACT_SOFTNESS`] and [`PHYSICS_GRAVITY`]. The `contact_spring_is_applied` and
/// `gravity_is_applied` tests prove the guards catch a dropped value on the headless
/// path.
pub struct CrabPhysicsPlugin;

impl bevy::app::Plugin for CrabPhysicsPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        use bevy::app::PostStartup;
        use bevy_rapier3d::plugin::{NoUserData, RapierPhysicsPlugin};
        app.insert_resource(fixed_timestep())
            .insert_resource(rapier_context_init())
            .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
            .add_systems(
                PostStartup,
                (assert_contact_spring_applied, assert_gravity_applied),
            );
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

/// Fail loud, in every world, if the spawned Rapier context's gravity isn't our
/// explicit [`PHYSICS_GRAVITY`] — the runtime backstop, alongside
/// [`assert_contact_spring_applied`], for the same last-write-wins resource trap.
/// Gravity lives on the `RapierConfiguration` component (not the integration
/// parameters), so it needs its own read-only query. A panic here means the config
/// was overridden after the plugin — wrong physics that would otherwise pass silently
/// into the GCR lockstep.
fn assert_gravity_applied(
    config: bevy::ecs::system::Query<
        &RapierConfiguration,
        bevy::ecs::query::With<bevy_rapier3d::plugin::context::DefaultRapierContext>,
    >,
) {
    let gravity = config
        .single()
        .expect("CrabPhysicsPlugin: exactly one default Rapier context")
        .gravity;
    assert_eq!(
        gravity, PHYSICS_GRAVITY,
        "CrabPhysicsPlugin: the spawned Rapier context's gravity is not PHYSICS_GRAVITY — \
         its RapierConfiguration was overridden after the plugin (last-write-wins). \
         Gravity is silently wrong; fix the init ordering at the call site."
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

    /// Pins [`PHYSICS_GRAVITY`] to the value Rapier's own `RapierConfiguration::new`
    /// seeds at [`LENGTH_UNIT`] — see [`PHYSICS_GRAVITY`] for why the equality matters.
    /// A Rapier bump that changed the default fails here, in review.
    #[test]
    fn gravity_matches_rapier_default() {
        assert_eq!(
            PHYSICS_GRAVITY,
            RapierConfiguration::new(LENGTH_UNIT).gravity,
            "PHYSICS_GRAVITY diverged from Rapier's default — verify the new value is \
             intentional and resume training; a silent change desyncs the checkpoint."
        );
    }

    /// The gravity twin of [`contact_spring_is_applied`]: after the production stack
    /// builds, the spawned context's active gravity must be our explicit
    /// [`PHYSICS_GRAVITY`], proving [`assert_gravity_applied`]'s invariant holds on the
    /// real headless path rather than leaning on Rapier's undocumented default.
    #[test]
    fn gravity_is_applied() {
        let mut app = headless_app();
        app.update();
        let mut q = app
            .world_mut()
            .query_filtered::<&RapierConfiguration, With<DefaultRapierContext>>();
        let config = q.single(app.world()).expect("one default rapier context");
        assert_eq!(
            config.gravity, PHYSICS_GRAVITY,
            "active context gravity != PHYSICS_GRAVITY — init ordering broke"
        );
    }

    /// Free-fall determinism regression: a bare dynamic body dropped into the real
    /// physics stack must (a) trace a bit-identical per-tick state hash across two
    /// independently-built worlds — the determinism contract gravity feeds into — and
    /// (b) actually accelerate downward, so the test can't pass vacuously on a frozen
    /// or zero-gravity body. Hashes the body's full `(pos, quat, linvel, angvel)` bits
    /// via the same [`crate::bot::physics_digest`] machinery the GCR lockstep folds.
    #[test]
    fn falling_body_is_deterministic() {
        use crate::bot::physics_digest::{DIGEST_SEED, body_bits, fold_bodies};
        use bevy::prelude::{Transform, Vec3};
        use bevy_rapier3d::prelude::{Collider, RigidBody, Velocity};

        const TICKS: usize = 32;
        // Far from the crab + ground so the fall is pure gravity, never a contact.
        const START: Vec3 = Vec3::new(1000.0, 500.0, 1000.0);

        // One world: spawn a free body, step `TICKS`, return (per-tick hashes, final).
        fn run() -> (Vec<u64>, Transform, Velocity) {
            let mut app = headless_app();
            let body = app
                .world_mut()
                .spawn((
                    RigidBody::Dynamic,
                    Collider::ball(0.1),
                    Velocity::zero(),
                    Transform::from_translation(START),
                ))
                .id();
            let mut hashes = Vec::with_capacity(TICKS);
            for _ in 0..TICKS {
                app.update();
                let t = *app.world().entity(body).get::<Transform>().unwrap();
                let v = *app.world().entity(body).get::<Velocity>().unwrap();
                hashes.push(fold_bodies(DIGEST_SEED, &[(0, body_bits(&t, &v))]));
            }
            let t = *app.world().entity(body).get::<Transform>().unwrap();
            let v = *app.world().entity(body).get::<Velocity>().unwrap();
            (hashes, t, v)
        }

        let (a_hashes, a_final, a_vel) = run();
        let (b_hashes, _, _) = run();

        // (a) Bit-identical across independent worlds — the determinism contract.
        assert_eq!(a_hashes, b_hashes, "free-fall trajectory not reproducible");
        // Non-vacuous: the body genuinely fell under PHYSICS_GRAVITY (negative Y), and
        // the state changed every tick (not frozen, so the hash equality means something).
        assert!(
            a_final.translation.y < START.y - 1.0,
            "body did not fall: y {} -> {}",
            START.y,
            a_final.translation.y
        );
        assert!(
            a_vel.linear.y < -1.0,
            "downward velocity never built: {a_vel:?}"
        );
        let distinct = a_hashes.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(
            distinct.len(),
            TICKS,
            "state hash repeated across ticks — body wasn't actually moving"
        );
    }
}
