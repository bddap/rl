mod bot;
mod combat;
mod physics;
mod player;
mod training;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(physics::PhysicsWorldPlugin)
        .add_plugins(bot::BotPlugin)
        .run();
}
