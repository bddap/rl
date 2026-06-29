//! Single source of truth for the bundled-asset root, plus a startup check that loudly
//! flags any bundled control glyph that didn't resolve (without crashing the match).
//!
//! The committed CC0 control glyphs (`assets/controls/*.png`) must show on the controls
//! overlay after a bare `git clone` + `cargo run`, with no env or cwd setup. They didn't:
//! bevy's `AssetPlugin` defaults its root to the *running binary's* crate dir
//! (`game/assets` under a `cargo run`, or the exe dir for a built binary) — neither of
//! which holds the glyphs, which live under this crate. So the overlay silently drew blank
//! boxes. [`asset_root`] is the one root the rendering binaries point bevy at; it matches
//! the root [`crate::bot::meshfit::model_path`] already resolves the crab model against.

use std::path::{Path, PathBuf};

/// Directory whose `assets/` subdir holds the bundled art the renderers load: the CC0
/// control glyphs committed at `assets/controls/`, plus a fetched `assets/sally.glb` when
/// present.
///
/// `BEVY_ASSET_ROOT` (set by deploy, which stages `assets/` beside the shipped binary)
/// wins. Otherwise this crate's source dir, baked in at compile time — so a fresh clone
/// resolves the committed glyphs with no setup. Matches the root
/// [`crate::bot::meshfit::model_path`] resolves the crab model against, so the bevy glyph
/// loads and the model load agree. EVERY asset the renderers resolve now roots here:
/// [`crate::bot::meshfit::model_path`] (the crab mesh), `scripts/fetch-sally.sh`'s fetch dest
/// (it asks `cargo metadata` for this same crate dir, so a fetched `sally.glb` lands exactly
/// where the renderer looks — it used to drop it at the repo root and silently fall back), and
/// the `game` NN-crab `assets/weights` checkpoint dir. One root, no drift (bddap/rl#146/#148).
pub fn asset_root() -> PathBuf {
    std::env::var_os("BEVY_ASSET_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf())
}

/// Absolute `assets/` path to hand bevy's `AssetPlugin.file_path`. An absolute file_path
/// makes every `asset_server.load("controls/…")` resolve here regardless of cwd or which
/// workspace binary is running — the default left bevy looking under the running bin's
/// crate dir, where the glyphs aren't.
pub fn bevy_asset_path() -> PathBuf {
    asset_root().join("assets")
}

/// Loudly WARN — but never crash — for any bundled glyph the overlay will request that is
/// absent under the resolved asset root. Called from [`crate::controls::spawn_controls_ui`]
/// at spawn. A control-prompt icon is decorative HUD: bevy renders a missing image handle
/// as nothing, so the overlay degrades to a blank slot for that one binding while the match
/// runs normally. Killing the whole match over a missing decorative glyph is the wrong
/// failure mode for the deployed game (it cost the owner a match-start panic when a deploy
/// staged the binary but not `assets/controls/`). We still log every missing path at WARN so
/// the blank slot has a findable cause — loud, not silent (the silent-fallback anti-pattern
/// is about silence, not about being fatal); a missing NON-ESSENTIAL asset degrades, it
/// doesn't abort. `paths` are the `controls/…` asset paths the active scheme can surface
/// (see [`crate::controls::icon_asset_paths`]).
pub fn warn_missing_glyphs<I: IntoIterator<Item = &'static str>>(paths: I) {
    let base = bevy_asset_path();
    let missing: Vec<&str> = paths
        .into_iter()
        .filter(|p| !base.join(p).exists())
        .collect();
    if !missing.is_empty() {
        tracing::warn!(
            "control overlay glyphs missing under {} — those bindings will show a blank \
             slot (non-fatal): {missing:?}. These are CC0 Kenney Input Prompts committed at \
             crab-world/assets/controls/ and should be present in any checkout; if your \
             assets live elsewhere, set BEVY_ASSET_ROOT.",
            base.display(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of this module's change: a missing decorative glyph must NOT abort.
    /// A bogus `controls/…` path can't exist under any asset root, so this exercises the
    /// missing branch — it must return (warn), not panic, or a deployed match-start dies on
    /// an un-staged HUD icon (the bug this replaced).
    #[test]
    fn missing_glyph_warns_does_not_panic() {
        warn_missing_glyphs(["controls/__surely_absent_glyph__.png"]);
    }

    /// An empty path set is trivially all-present — no warn, no panic.
    #[test]
    fn no_glyphs_is_fine() {
        warn_missing_glyphs(std::iter::empty());
    }
}
