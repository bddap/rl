pub mod world;

#[cfg(feature = "render")]
pub use world::ArenaVisualsPlugin;
pub use world::{Arena, PhysicsWorldPlugin};

use bevy_rapier3d::math::Vect;
use bevy_rapier3d::plugin::RapierContextInitialization;
use bevy_rapier3d::prelude::{RapierConfiguration, TimestepMode};
use bevy_rapier3d::rapier::dynamics::{IntegrationParameters, SpringCoefficients};

pub const PHYSICS_HZ: u64 = 64;

pub const PHYSICS_DT: f32 = 1.0 / PHYSICS_HZ as f32;

pub const PHYSICS_SUBSTEPS: usize = 2;

fn fixed_timestep() -> TimestepMode {
    TimestepMode::Fixed {
        dt: PHYSICS_DT,
        substeps: PHYSICS_SUBSTEPS,
    }
}

pub const CONTACT_SOFTNESS: SpringCoefficients<f32> = SpringCoefficients {
    natural_frequency: 5.0,
    damping_ratio: 5.0,
};

const LENGTH_UNIT: f32 = 1.0;

pub const PHYSICS_GRAVITY: Vect = Vect::new(0.0, -9.81, 0.0);

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

/// Runtime backstop that the spawned context carries [`CONTACT_SOFTNESS`] — see
/// [`CrabPhysicsPlugin`] for the last-write-wins trap this guards.
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

/// Runtime backstop that the spawned context's gravity is [`PHYSICS_GRAVITY`] — see
/// [`CrabPhysicsPlugin`] for the last-write-wins trap this guards. Gravity lives on
/// `RapierConfiguration` (not the integration parameters), hence its own query.
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
    use crate::bot::headless::headless_app;
    use bevy::prelude::With;
    use bevy_rapier3d::prelude::{DefaultRapierContext, RapierContextSimulation};

    #[test]
    fn contact_spring_is_applied() {
        let mut app = headless_app();
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

    #[test]
    fn gravity_matches_rapier_default() {
        assert_eq!(
            PHYSICS_GRAVITY,
            RapierConfiguration::new(LENGTH_UNIT).gravity,
            "PHYSICS_GRAVITY diverged from Rapier's default — verify the new value is \
             intentional and resume training; a silent change desyncs the checkpoint."
        );
    }

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

    #[test]
    fn falling_body_is_deterministic() {
        use crate::bot::physics_digest::{DIGEST_SEED, body_bits, fold_bodies};
        use bevy::prelude::{Transform, Vec3};
        use bevy_rapier3d::prelude::{Collider, RigidBody, Velocity};

        const TICKS: usize = 32;
        const START: Vec3 = Vec3::new(1000.0, 500.0, 1000.0);

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
                hashes.push(fold_bodies(DIGEST_SEED, vec![(0, body_bits(&t, &v))]));
            }
            let t = *app.world().entity(body).get::<Transform>().unwrap();
            let v = *app.world().entity(body).get::<Velocity>().unwrap();
            (hashes, t, v)
        }

        let (a_hashes, a_final, a_vel) = run();
        let (b_hashes, _, _) = run();

        assert_eq!(a_hashes, b_hashes, "free-fall trajectory not reproducible");
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
