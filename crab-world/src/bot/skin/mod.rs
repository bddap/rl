//! Optional skinned crab model riding the physics body.
//!
//! When `CRAB_MODEL_PATH` names a glTF inside the app's `assets/` directory
//! (e.g. `sally.glb`, fetched from the private bddap-bot/rl-assets repo), each
//! crab gets a skinned-mesh skin whose deform bones follow the physics links.
//! The physics body stays the single source of truth; the model is cosmetic,
//! and the colliders themselves are only ever shown by the shared `crab_view`
//! collider wireframe (the render-mode cycle), never as stand-in meshes.
//!
//! This module is the facade. [`register`] gates on the model resource, then
//! wires the two submodules:
//! - [`pairing`] — the skin lifecycle: pair each deform bone to its physics link,
//!   drive it every frame, re-pair after an episode reset. Owns the render-only
//!   giant blow-up ([`SkinRepose`]/[`CrabSkinRepose`]), re-exported here so
//!   `net::external_crab` and `crate::crab_view` keep their `bot::skin::…` paths.
//! - [`weights`] — a one-time skin-weight strip confining each vertex's weights to
//!   its dominant physics part, so a rigidly-driven bone can't drag the wrong flesh.

use bevy::prelude::*;

mod pairing;
mod weights;

pub use pairing::{CrabSkinRepose, SkinRepose};

/// Present only when a model path was configured; all systems key off this.
#[derive(Resource)]
struct CrabModel {
    scene: Handle<Scene>,
}

pub fn register(app: &mut App) {
    // Skin iff a model resolves — read the SAME `CrabModelPath` the body reads, so skin and
    // physics can't disagree about which crab is present (see `body::CrabModelPath`).
    let Some(model) = app
        .world()
        .resource::<super::body::CrabModelPath>()
        .0
        .clone()
    else {
        return;
    };
    // The AssetServer resolves names RELATIVE to its root (`assets::bevy_asset_path`), so hand it
    // the resource's resolved path made relative to that root — NOT a fresh `CRAB_MODEL_PATH` env
    // read, which could name a different file than the resource resolved. An out-of-root absolute
    // override (the rare dev escape hatch) has no relative form, so pass it as-is.
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
