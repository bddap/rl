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
use std::path::PathBuf;
use std::sync::OnceLock;

/// The reason a surface fell back: no `sally.glb` resolved at all (the common case, vs a
/// present-but-unloadable one). The absent branch of the shared [`usable_model`] verdict, so every
/// surface names the missing mesh identically.
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

/// Resolve the crab model path AND run the full load+recipe preflight: `Ok(path)` ⇒ the canonical
/// Sally mesh is present AND usable (loads + has the crab bones the rig needs), render the real
/// skinned crab from `path`; `Err(reason)` carries a human-readable cause (absent, or
/// present-but-unloadable) for the OTEL error and the collider fallback. THE single verdict
/// function — [`usable_model`] memoizes it — so the "is the mesh good?" answer has one source and
/// mirrors the model-vs-fallback selection [`crate::bot::body::render_recipe`] performs, meaning a
/// surface's verdict can't disagree with what the body would actually spawn.
fn checked_model_path() -> Result<PathBuf, String> {
    let Some(p) = bot::meshfit::model_path() else {
        return Err(MESH_ABSENT_REASON.to_string());
    };
    mesh_status_of(&p).map(|()| p)
}

/// The load+recipe verdict for an already-resolved model path — split out so the
/// present-but-unloadable case (bddap/rl#154) is unit-testable without touching process env or the
/// memoized [`usable_model`]. `Err` names the cause: a load failure (corrupt/truncated/wrong-format
/// glb) carries the parser's message; a load that succeeds but yields no crab bones is called out
/// distinctly.
fn mesh_status_of(p: &std::path::Path) -> Result<(), String> {
    let model = bot::meshfit::LoadedModel::load(p).map_err(|e| format!("crab model {p:?}: {e}"))?;
    if bot::rig::build_recipe(&model).is_none() {
        return Err(format!(
            "crab model {p:?}: loaded but has none of the expected crab bones (e.g. Def_leg_01.000.L)"
        ));
    }
    Ok(())
}

/// Memoized player-facing crab-mesh verdict shared by the `game` and `net` render surfaces:
/// `Ok(path)` — render the real skinned Sally from `path`; `Err(reason)` — the mesh is absent OR
/// present-but-unloadable (corrupt/truncated, or parses with none of the crab bones), so fall back
/// to the honest collider silhouette and go LOUD with `reason`. THE one "is the mesh good?" answer
/// every render decision on these surfaces reads, so a present-but-broken `sally.glb` degrades once,
/// everywhere, to `None` instead of a hard `.expect()` in [`crate::bot::body::render_recipe`]
/// (bddap/rl#154) — collapsing the old existence-only `model_path().is_some()` checks and the full
/// [`checked_model_path`] preflight into ONE verdict, so a surface's "is the mesh present?" and "is
/// the mesh usable?" answers can no longer disagree.
///
/// Memoized because the preflight re-parses the 36 MB glb, the crab mesh never changes at runtime (a
/// fixed binary+asset constant, like [`crate::bot::rig`]'s memoized natural height), and the
/// per-frame silhouette-visibility system reads this — so it must be cheap after the first call. All
/// three player-facing surfaces read this one verdict for their `CrabModelPath`/silhouette; rl-demo
/// additionally reads the `Err` reason (`usable_model().err()`) to thread the cause into its own
/// on-screen banner and forced render mode.
pub fn usable_model() -> &'static Result<PathBuf, String> {
    static VERDICT: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    VERDICT.get_or_init(checked_model_path)
}

/// The crab model to RENDER from on the player-facing surfaces: `Some(path)` iff the mesh is present
/// AND usable, else `None` → the honest fallback. The [`usable_model`] verdict as a plain path option
/// for the sites that pick geometry (they can't crash on a broken `Some`); sites that go LOUD read
/// the `Err(reason)` from [`usable_model`] directly.
pub fn usable_model_path() -> Option<PathBuf> {
    usable_model().as_ref().ok().cloned()
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
    /// `reason` is the human-readable cause (from [`super::usable_model`] /
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

#[cfg(test)]
mod tests {
    use super::mesh_status_of;

    /// The bug this fixes (bddap/rl#154): a present-but-unloadable `sally.glb` must yield `Err`, so
    /// the player-facing surfaces fall back to `None` instead of `render_recipe` `.expect()`-panicking.
    /// A garbage file (fails glTF parse) is the corrupt/truncated case; the verdict is an honest
    /// `Err`, never a panic.
    #[test]
    fn present_but_unloadable_glb_is_err_not_panic() {
        let dir = std::env::temp_dir().join(format!("rl154-badglb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("sally.glb");
        std::fs::write(&bad, b"this is not a glb, it is garbage bytes").unwrap();

        let status = mesh_status_of(&bad);
        assert!(
            status.is_err(),
            "a garbage present-but-unloadable glb must report Err (rl#154), got {status:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
