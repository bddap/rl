pub mod world;

pub use world::PhysicsWorldPlugin;
#[cfg(feature = "render")]
pub use world::{ArenaVisualsPlugin, ArenaWorldPlugin};

use bevy_rapier3d::math::Vect;
use bevy_rapier3d::plugin::{NoUserData, RapierPhysicsPlugin};
use bevy_rapier3d::prelude::TimestepMode;

pub const PHYSICS_HZ: u64 = 64;

pub const PHYSICS_DT: f32 = 1.0 / PHYSICS_HZ as f32;

pub const PHYSICS_GRAVITY: Vect = Vect::new(0.0, -9.81, 0.0);

/// At `c = m·Hz` one explicit tick of brake force `F = −c·v` cancels the body's
/// momentum; past it the brake overshoots zero and grows geometrically each step.
pub const fn brake_coeff_max(mass: f32) -> f32 {
    mass * PHYSICS_HZ as f32
}

pub struct WorldPhysicsPlugin;

impl bevy::app::Plugin for WorldPhysicsPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.insert_resource(TimestepMode::Fixed {
            dt: PHYSICS_DT,
            substeps: 1,
        })
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rapier_default_gravity_is_physics_gravity() {
        assert_eq!(
            bevy_rapier3d::prelude::RapierConfiguration::new(1.0).gravity,
            super::PHYSICS_GRAVITY
        );
    }
}
