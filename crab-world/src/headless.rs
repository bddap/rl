use std::sync::Arc;
use std::time::Duration;

use bevy::prelude::*;

use crate::physics::{PHYSICS_DT, PhysicsWorldPlugin, WorldPhysicsPlugin};
use crate::terrain::TerrainGrid;

/// A windowless world on `grid`, stepped one [`PHYSICS_DT`] per `update()`. With
/// `render` on, the renderer is still built with no backends, so no GPU is touched.
fn headless_world(grid: Arc<TerrainGrid>) -> App {
    let mut app = App::new();
    let plugins = DefaultPlugins.build().disable::<bevy::log::LogPlugin>();
    #[cfg(feature = "render")]
    let plugins = plugins
        .set(bevy::window::WindowPlugin {
            primary_window: None,
            exit_condition: bevy::window::ExitCondition::DontExit,
            ..default()
        })
        .set(bevy::render::RenderPlugin {
            render_creation: bevy::render::settings::RenderCreation::Automatic(Box::new(
                bevy::render::settings::WgpuSettings {
                    backends: None,
                    ..default()
                },
            )),
            ..default()
        })
        .disable::<bevy::winit::WinitPlugin>()
        .disable::<bevy::gilrs::GilrsPlugin>();
    app.add_plugins(plugins);
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        Duration::from_secs_f32(PHYSICS_DT),
    ));
    app.insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f32(
        PHYSICS_DT,
    )));
    app.add_plugins(WorldPhysicsPlugin)
        .add_plugins(PhysicsWorldPlugin { grid });
    app
}

/// The host's world: the canonical tile plus the crafts.
pub fn headless_server_world(grid: Arc<TerrainGrid>) -> App {
    let mut app = headless_world(grid);
    app.add_plugins(crate::vehicle::VehiclePlugin);
    app
}

pub fn headless_app() -> App {
    headless_world(TerrainGrid::gcr())
}

#[cfg(test)]
pub fn flat_headless_app() -> App {
    headless_world(Arc::new(TerrainGrid::flat(512.0)))
}

pub fn tick(app: &mut App, n: u32) {
    for _ in 0..n {
        app.update();
    }
}
