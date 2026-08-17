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

/// Resolve the crab model file: `CRAB_MODEL_PATH` (absolute, or relative to the
/// asset root) with `sally.glb` under `BEVY_ASSET_ROOT/assets` as the default.
pub fn model_path() -> Option<PathBuf> {
    resolve(
        std::env::var_os("CRAB_MODEL_PATH").as_deref(),
        &crate::assets::asset_root(),
        |p| p.exists(),
    )
}

fn resolve(
    crab_model_path: Option<&std::ffi::OsStr>,
    asset_root: &std::path::Path,
    exists: impl Fn(&std::path::Path) -> bool,
) -> Option<PathBuf> {
    let rel = crab_model_path.map_or_else(|| PathBuf::from("sally.glb"), PathBuf::from);
    if rel.is_absolute() {
        return exists(&rel).then_some(rel);
    }
    let asset = asset_root.join("assets").join(rel);
    exists(&asset).then_some(asset)
}

fn checked_model() -> Result<PathBuf, String> {
    let Some(path) = model_path() else {
        return Err(MESH_ABSENT_REASON.to_string());
    };
    checked_skin(&path)?;
    Ok(path)
}

/// The skin digest gate: the asset's bytes must be EXACTLY the ones the committed
/// [`bot::rig::baked_recipe`](crate::bot::rig::baked_recipe) table was baked from.
/// Any other byte state (a changed model, a corrupt download) refuses the SKIN
/// loudly: physics runs the baked table regardless (rl#340 stage 10), so drawing a
/// mismatched mesh over it would silently fork render from physics. A re-fit is a
/// deliberate offline event (`cargo run -p meshfit -- bake`), never a side effect
/// of swapping a file.
fn checked_skin(p: &std::path::Path) -> Result<(), String> {
    let bytes = std::fs::read(p).map_err(|e| format!("crab model {p:?}: read: {e}"))?;
    let digest = crate::fnv::fnv1a(&bytes);
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

/// The memoized skin verdict: `Ok(path)` iff a model resolves AND its bytes match
/// the baked digest — the only state in which the skin may be drawn. `Err` carries
/// the human-facing reason for the banner/collider-view surfaces. Physics never
/// consults this (rl#340 stage 10).
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
mod model_path_tests {
    use super::resolve;
    use std::path::{Path, PathBuf};

    #[test]
    fn relative_resolves_under_asset_root() {
        let got = resolve(Some("sally.glb".as_ref()), Path::new("/srv/app"), |p| {
            p == Path::new("/srv/app/assets/sally.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/srv/app/assets/sally.glb")));
    }

    #[test]
    fn defaults_to_sally_under_asset_root() {
        let got = resolve(None, Path::new("/crate"), |p| {
            p == Path::new("/crate/assets/sally.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/crate/assets/sally.glb")));
    }

    #[test]
    fn absolute_path_used_as_is() {
        let got = resolve(Some("/models/x.glb".as_ref()), Path::new("/srv"), |p| {
            p == Path::new("/models/x.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/models/x.glb")));
    }

    #[test]
    fn none_when_missing() {
        assert_eq!(
            resolve(Some("sally.glb".as_ref()), Path::new("/srv"), |_| false),
            None
        );
    }
}

#[cfg(test)]
mod tests {
    use super::checked_skin;

    #[test]
    fn present_but_mismatched_glb_refuses_the_skin() {
        let dir = std::env::temp_dir().join(format!("rl154-badglb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("sally.glb");
        std::fs::write(&bad, b"this is not a glb, it is garbage bytes").unwrap();

        let status = checked_skin(&bad);
        assert!(
            status.is_err(),
            "a present-but-mismatched glb must refuse the skin (rl#154/rl#340 stage 10)"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
