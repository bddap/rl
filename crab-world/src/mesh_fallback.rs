//! sally.glb is VISUAL-ONLY (bddap/rl#340 stage 10): physics always builds the
//! committed [`bot::rig::baked_recipe`] table — no surface gates a body on the
//! asset. What this module gates is the SKIN: a resolvable glb whose bytes match
//! [`bot::rig::BAKED_ASSET_DIGEST`] may be drawn over the physics body; anything
//! else (absent, corrupt, or changed without a re-bake) falls back to the honest
//! collider view, loudly — a stale skin over current colliders would silently
//! desync render from physics, the one thing the digest check protects.

use std::path::PathBuf;
use std::sync::OnceLock;

pub const MESH_ABSENT_REASON: &str = "no crab model resolved (CRAB_MODEL_PATH / default `sally.glb` not found under \
     BEVY_ASSET_ROOT/assets)";

#[derive(Clone, Copy)]
pub enum Surface {
    RlDemo,
    Game,
}

impl Surface {
    fn as_str(self) -> &'static str {
        match self {
            Surface::RlDemo => "rl-demo",
            Surface::Game => "game",
        }
    }
}

/// The model's ASSET path — the form every render-side consumer speaks (rl#411: the
/// digest gate fetches it via [`crate::assets::read_asset`], the skin hands it to the
/// `AssetServer`, so both ride the one asset tree): `CRAB_MODEL_PATH`
/// (asset-tree-relative, or an absolute native dev path) with `sally.glb` as the
/// default.
fn model_rel() -> PathBuf {
    rel_from(std::env::var_os("CRAB_MODEL_PATH").as_deref())
}

fn rel_from(crab_model_path: Option<&std::ffi::OsStr>) -> PathBuf {
    crab_model_path.map_or_else(|| PathBuf::from("sally.glb"), PathBuf::from)
}

/// The model's real FILE path, for the offline native tooling that reads and bakes it
/// (meshfit). Render surfaces never touch this — they gate on [`usable_model`].
pub fn model_path() -> Option<PathBuf> {
    let p = crate::assets::bevy_asset_path().join(model_rel());
    p.exists().then_some(p)
}

fn checked_model() -> Result<PathBuf, String> {
    let rel = model_rel();
    let bytes = match crate::assets::read_asset(&rel) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(MESH_ABSENT_REASON.to_string());
        }
        Err(e) => return Err(format!("crab model {rel:?}: read: {e}")),
    };
    checked_skin(&rel, &bytes)?;
    Ok(rel)
}

/// The skin digest gate: the asset's bytes must be EXACTLY the ones the committed
/// [`bot::rig::baked_recipe`](crate::bot::rig::baked_recipe) table was baked from.
/// Any other byte state (a changed model, a corrupt download) refuses the SKIN
/// loudly: physics runs the baked table regardless (rl#340 stage 10), so drawing a
/// mismatched mesh over it would silently fork render from physics. A re-fit is a
/// deliberate offline event (`cargo run -p meshfit -- bake`), never a side effect
/// of swapping a file.
fn checked_skin(p: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    let digest = crate::fnv::fnv1a(bytes);
    if digest != crate::bot::rig::BAKED_ASSET_DIGEST {
        return Err(format!(
            "crab model {p:?}: digest {digest:#018x} does not match the baked collider \
             table ({:#018x}) — the asset changed (or is corrupt) without a re-bake. \
             Fetch the canonical sally.glb (scripts/fetch-sally.sh), or re-bake \
             deliberately (`cargo run -p meshfit -- bake`): a geometry change is a new \
             MDP — review the baked.rs diff and plan a retrain (rl#277)",
            crate::bot::rig::BAKED_ASSET_DIGEST
        ));
    }
    Ok(())
}

/// The memoized skin verdict: `Ok(asset path)` iff the model's bytes fetch through
/// the one asset path AND match the baked digest — the only state in which the skin
/// may be drawn; the path is the ASSET-tree form, ready for `AssetServer::load`.
/// `Err` carries the human-facing reason for the banner/collider-view surfaces.
/// Physics never consults this (rl#340 stage 10).
pub fn usable_model() -> &'static Result<PathBuf, String> {
    static VERDICT: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    VERDICT.get_or_init(checked_model)
}

pub fn usable_model_path() -> Option<PathBuf> {
    usable_model().as_ref().ok().cloned()
}

pub fn log_fallback(surface: Surface, reason: &str) {
    tracing::error!(
        target: "crab_world::canonical_mesh",
        surface = %surface.as_str(),
        host = %hostname(),
        reason = %reason,
        "canonical Sally mesh could not be resolved — falling back to the honest collider \
         wireframe (the real physics colliders, NOT the real Sally rig). Fetch it with \
         scripts/fetch-sally.sh or point CRAB_MODEL_PATH at the model."
    );
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(feature = "render")]
pub use banner::spawn_banner;

#[cfg(feature = "render")]
mod banner {
    use bevy::prelude::*;

    const BANNER_HEADLINE: &str =
        "SALLY MESH NOT LOADED — showing physics colliders (NOT the real Sally rig)";

    pub fn spawn_banner(commands: &mut Commands, reason: &str) -> Entity {
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(0.0),
                    left: Val::Px(0.0),
                    right: Val::Px(0.0),
                    padding: UiRect::all(Val::Px(8.0)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgba(0.15, 0.0, 0.0, 0.85)),
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new(BANNER_HEADLINE),
                    TextFont {
                        font_size: FontSize::Px(20.0),
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.5, 0.5)),
                ));
                b.spawn((
                    Text::new(format!("{reason}  —  fetch with scripts/fetch-sally.sh")),
                    TextFont {
                        font_size: FontSize::Px(13.0),
                        ..default()
                    },
                    TextColor(Color::srgba(0.95, 0.85, 0.85, 0.9)),
                ));
            })
            .id()
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_skin, rel_from};
    use std::path::{Path, PathBuf};

    #[test]
    fn defaults_to_sally_and_honors_the_override() {
        assert_eq!(rel_from(None), PathBuf::from("sally.glb"));
        assert_eq!(
            rel_from(Some("models/x.glb".as_ref())),
            PathBuf::from("models/x.glb")
        );
    }

    #[test]
    fn present_but_mismatched_glb_refuses_the_skin() {
        let status = checked_skin(
            Path::new("sally.glb"),
            b"this is not a glb, it is garbage bytes",
        );
        assert!(
            status.is_err(),
            "present-but-mismatched glb bytes must refuse the skin (rl#154/rl#340 stage 10)"
        );
    }
}
