//! Shared LOUD signalling for a missing canonical-Sally mesh on the player-facing surfaces.
//!
//! rl-demo and game/net both render the crab through the SAME `crab_view` collider cage, but
//! each binary decides on its own to fall back to that cage when no `sally.glb` resolves. The
//! WARNING they raise about it — the OTEL error on the telemetry stream and the on-screen
//! banner — is ONE impl here, so the two surfaces can't grow two drifting copies of the same
//! message (the silent-fallback / duplicate-warning bug the owner most hates; bddap/rl#706).
//! The headless trainer keeps the procedural body by design and never calls this — the loud
//! fallback is purely a player-facing concern, so the banner half is render-gated.

use crate::bot;

/// The reason a surface fell back: no `sally.glb` resolved at all (the common case). Shared so
/// the game's cheap `model_path().is_none()` check and rl-demo's full [`canonical_mesh_status`]
/// preflight name the absent mesh identically.
pub const MESH_ABSENT_REASON: &str =
    "no crab model resolved (CRAB_MODEL_PATH / default `sally.glb` not found under \
     BEVY_ASSET_ROOT/assets)";

/// Which player-facing binary raised the fallback. A closed enum, not a free string, so the
/// `surface` field the telemetry sink partitions on can't be typo'd or drift between call
/// sites (`"game"` vs `"rl-game"` vs `"net"`).
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

/// Is the canonical Sally mesh present AND usable (loads + has the crab bones the rig needs)?
/// `Ok(())` ⇒ render the real skinned crab; `Err(reason)` carries a human-readable cause for
/// the OTEL error and the collider fallback. Mirrors the model-vs-fallback selection
/// [`crate::bot::body::render_recipe`] performs, so a surface's "is the mesh good?" verdict
/// can't disagree with what the body would actually spawn.
pub fn canonical_mesh_status() -> Result<(), String> {
    let Some(p) = bot::meshfit::model_path() else {
        return Err(MESH_ABSENT_REASON.to_string());
    };
    let model = bot::meshfit::LoadedModel::load(&p).map_err(|e| format!("crab model {p:?}: {e}"))?;
    if bot::rig::build_recipe(&model).is_none() {
        return Err(format!(
            "crab model {p:?}: loaded but has none of the expected crab bones (e.g. Def_leg_01.000.L)"
        ));
    }
    Ok(())
}

/// Emit the LOUD canonical-mesh error: a `tracing::error!` that names the absent/broken Sally
/// mesh, the `surface` that fell back, and the `host` it fired on — surfaced on stderr always
/// and exported to the OTLP sink when a binary wires `otel::init`. The matching on-screen
/// banner is [`spawn_banner`] (render only). One target (`crab_world::canonical_mesh`) so the
/// sink filters every surface's fallback through one query.
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

/// This host's name for telemetry tagging, so a missing-asset error names the device it fired
/// on. Best-effort: an unreadable hostname reports `unknown` rather than failing the run.
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

    /// The can't-miss headline, named honestly so the collider view can never be mistaken for
    /// the real Sally rig (rl#706). Private: [`spawn_banner`] is the only way to put it on
    /// screen, so a second caller can't re-render its own band and re-grow the drift.
    const BANNER_HEADLINE: &str =
        "SALLY MESH NOT LOADED — showing physics colliders (NOT the real Sally rig)";

    /// Spawn the top-center red warning band naming the missing Sally mesh on screen. Call it
    /// from a startup system (rl-demo) or directly during scene spawn (game/net) — the caller
    /// owns WHEN and gates it to the windowed surface; this owns the one banner both share.
    /// `reason` is the human-readable cause (from [`super::canonical_mesh_status`] /
    /// [`super::MESH_ABSENT_REASON`]).
    pub fn spawn_banner(commands: &mut Commands, reason: &str) {
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
                // Opaque dark band so the warning reads against any scene behind it.
                BackgroundColor(Color::srgba(0.15, 0.0, 0.0, 0.85)),
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new(BANNER_HEADLINE),
                    TextFont {
                        font_size: 20.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.5, 0.5)),
                ));
                b.spawn((
                    Text::new(format!("{reason}  —  fetch with scripts/fetch-sally.sh")),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgba(0.95, 0.85, 0.85, 0.9)),
                ));
            });
    }
}
