use bevy::prelude::*;

mod pairing;
mod weights;

pub use pairing::CrabRenderPose;

#[derive(Resource)]
struct CrabModel {
    scene: Handle<WorldAsset>,
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
    // `CrabModelPath` is already the asset-tree form (rl#411) — the same path the
    // digest gate fetched through the one asset byte path.
    let scene = app
        .world()
        .resource::<AssetServer>()
        .load(GltfAssetLabel::Scene(0).from_asset(model));
    app.insert_resource(CrabModel { scene });
    pairing::register(app);
    weights::register(app);
}
