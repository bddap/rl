
use bevy::prelude::*;

mod pairing;
mod weights;

pub use pairing::{CrabSkinRepose, SkinRepose};

#[derive(Resource)]
struct CrabModel {
    scene: Handle<Scene>,
}

pub fn register(app: &mut App) {
    let Some(model) = app
        .world()
        .resource::<super::body::CrabModelPath>()
        .0
        .clone()
    else {
        return;
    };
    let rel = model
        .strip_prefix(crate::assets::bevy_asset_path())
        .map_or_else(|_| model.clone(), std::path::Path::to_path_buf);
    let scene = app
        .world()
        .resource::<AssetServer>()
        .load(GltfAssetLabel::Scene(0).from_asset(rel));
    app.insert_resource(CrabModel { scene });
    pairing::register(app);
    weights::register(app);
}
