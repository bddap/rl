
use std::path::{Path, PathBuf};

pub fn asset_root() -> PathBuf {
    std::env::var_os("BEVY_ASSET_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf())
}

pub fn bevy_asset_path() -> PathBuf {
    asset_root().join("assets")
}

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

    #[test]
    fn missing_glyph_warns_does_not_panic() {
        warn_missing_glyphs(["controls/__surely_absent_glyph__.png"]);
    }

    #[test]
    fn no_glyphs_is_fine() {
        warn_missing_glyphs(std::iter::empty());
    }
}
