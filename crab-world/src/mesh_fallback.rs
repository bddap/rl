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
pub const MESH_ABSENT_REASON: &str = "no crab model resolved (CRAB_MODEL_PATH / default `sally.glb` not found under \
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

/// The canonical crab mesh, resolved AND validated in one shot: its file `path` plus the body
/// [`RigRecipe`](bot::rig::RigRecipe) built from it. A value of this type EXISTS only when the mesh
/// loaded and yielded the crab bones the rig needs — so holding one is the type-level proof of
/// "usable by construction", and the recipe was built ONCE (the 36 MB parse + collider fit) rather
/// than re-derived per render surface (bddap/rl#153).
pub struct UsableModel {
    pub path: PathBuf,
    pub recipe: bot::rig::RigRecipe,
}

/// Resolve the crab model path AND run the full load+recipe preflight, KEEPING the built recipe:
/// `Ok(UsableModel)` ⇒ the canonical Sally mesh is present AND usable (loads + has the crab bones the
/// rig needs), carrying the path to render from and the recipe to spawn; `Err(reason)` carries a
/// human-readable cause (absent, or present-but-unloadable) for the OTEL error and the collider
/// fallback. THE single verdict function — [`usable_model`] memoizes it — so the "is the mesh good?"
/// answer has one source AND the recipe [`crate::bot::body::render_recipe`] spawns is the very one
/// this validated, meaning a surface's verdict can't disagree with what the body actually spawns.
fn checked_model() -> Result<UsableModel, String> {
    let Some(path) = bot::meshfit::model_path() else {
        return Err(MESH_ABSENT_REASON.to_string());
    };
    let recipe = checked_recipe(&path)?;
    Ok(UsableModel { path, recipe })
}

/// Load an already-resolved model path and build its rig recipe — the actual usability test, split
/// out so the present-but-unloadable case (bddap/rl#154) is unit-testable without touching process
/// env or the memoized [`usable_model`]. `Err` names the cause: a load failure
/// (corrupt/truncated/wrong-format glb) carries the parser's message; a load that succeeds but yields
/// no crab bones is called out distinctly. `Ok` returns the built recipe so the caller need not
/// rebuild it — the load+fit happens exactly once (bddap/rl#153).
fn checked_recipe(p: &std::path::Path) -> Result<bot::rig::RigRecipe, String> {
    let model = bot::meshfit::LoadedModel::load(p).map_err(|e| format!("crab model {p:?}: {e}"))?;
    bot::rig::build_recipe(&model).ok_or_else(|| {
        format!(
            "crab model {p:?}: loaded but has none of the expected crab bones (e.g. Def_leg_01.000.L)"
        )
    })
}

/// Memoized player-facing crab-mesh verdict shared by every render surface (`game`, `net`, rl-demo):
/// `Ok(UsableModel)` — render the real skinned Sally from its `path` and spawn its `recipe`;
/// `Err(reason)` — the mesh is absent OR present-but-unloadable (corrupt/truncated, or parses with
/// none of the crab bones), so fall back to the honest collider silhouette and go LOUD with `reason`.
/// THE one "is the mesh good?" answer every render decision reads, so a present-but-broken `sally.glb`
/// degrades once, everywhere, to the fallback instead of a hard `.expect()` in
/// [`crate::bot::body::render_recipe`] (bddap/rl#154) — collapsing the old existence-only
/// `model_path().is_some()` checks and the load+recipe preflight into ONE verdict that also OWNS the
/// built recipe, so a surface's "is the mesh present?", "is the mesh usable?", and "which recipe do I
/// spawn?" answers can no longer disagree (bddap/rl#153).
///
/// Memoized because the preflight re-parses the 36 MB glb + fits the collider cloud (~1 s), the crab
/// mesh never changes at runtime (a fixed binary+asset constant, like [`crate::bot::rig`]'s memoized
/// natural height), and the per-frame silhouette-visibility system reads this — so it must be cheap
/// after the first call. Every surface reads this one verdict for its `CrabModelPath`/recipe/
/// silhouette; rl-demo + game additionally read the `Err` reason (`usable_model().err()`) to thread
/// the cause into the banner and forced render mode.
pub fn usable_model() -> &'static Result<UsableModel, String> {
    static VERDICT: OnceLock<Result<UsableModel, String>> = OnceLock::new();
    VERDICT.get_or_init(checked_model)
}

/// The crab model to RENDER from on the player-facing surfaces: `Some(path)` iff the mesh is present
/// AND usable, else `None` → the honest fallback. The [`usable_model`] verdict as a plain path option
/// for the sites that pick geometry (the skin, the world scale); the body recipe comes from
/// [`usable_model`] directly, and sites that go LOUD read its `Err(reason)`.
pub fn usable_model_path() -> Option<PathBuf> {
    usable_model().as_ref().ok().map(|u| u.path.clone())
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
    use super::checked_recipe;

    /// The bug this fixes (bddap/rl#154): a present-but-unloadable `sally.glb` must yield `Err`, so
    /// the player-facing surfaces fall back instead of `render_recipe` `.expect()`-panicking. A
    /// garbage file (fails glTF parse) is the corrupt/truncated case; the verdict is an honest `Err`,
    /// never a panic — and never a recipe.
    #[test]
    fn present_but_unloadable_glb_is_err_not_panic() {
        let dir = std::env::temp_dir().join(format!("rl154-badglb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("sally.glb");
        std::fs::write(&bad, b"this is not a glb, it is garbage bytes").unwrap();

        let status = checked_recipe(&bad);
        assert!(
            status.is_err(),
            "a garbage present-but-unloadable glb must report Err (rl#154), got a recipe: {:?}",
            status.is_ok()
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
