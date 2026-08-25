use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Pin the asset root for this process — an entry adapter passes it via its config
/// (rl#411: `run_game(GameConfig)`), BEFORE anything resolves an asset. Pinning twice
/// is a wiring bug and dies loudly: a second root would mean two asset paths in one
/// process, the exact drift the one-path rule forbids.
pub fn set_asset_root(root: PathBuf) {
    if let Err(root) = ROOT.set(root) {
        panic!(
            "asset root configured twice (second value {}) — one process, one asset root",
            root.display()
        );
    }
}

/// The asset root of record: the pinned config value, else the native default —
/// binaries without a config entrypoint (probes, trainers) resolve the same way the
/// native adapter does.
pub fn asset_root() -> PathBuf {
    ROOT.get().cloned().unwrap_or_else(native_asset_root)
}

/// How NATIVE launches find assets: the deploy env override, else the dev checkout
/// (this crate's manifest dir, so a fresh clone's `cargo run` finds the committed
/// glyphs regardless of cwd). Entry adapters resolve this into their config; only
/// native code may call it — a web adapter supplies its own root.
pub fn native_asset_root() -> PathBuf {
    std::env::var_os("BEVY_ASSET_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf())
}

pub fn bevy_asset_path() -> PathBuf {
    asset_root().join("assets")
}

/// What a load path does when an asset it wants is not there (rl#375, owner
/// directive): every release repeatedly shipped code without its assets and the
/// resulting degradation was silent-by-design, found by ear. So missing-asset is now
/// a PANIC at load — a broken bundle refuses to run instead of quietly shipping less
/// game — and an environment that legitimately has no assets (plain dev checkout, CI)
/// must say so with `RL_ALLOW_MISSING_ASSETS=1`, never by just lacking the file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MissingAssetPolicy {
    Panic,
    /// Explicitly declared asset-less environment: keep the pre-rl#375 behavior
    /// (silent layer / blank glyph slot / collider view), logged at WARN.
    Degrade,
}

/// The one env flag, parsed strictly: unset or `0` = panic, `1` = degrade, anything
/// else is a config typo and dies at first use — a silently-ignored `true` here would
/// re-create the exact silent-fallback class this policy exists to kill.
pub fn missing_asset_policy() -> MissingAssetPolicy {
    static POLICY: std::sync::OnceLock<MissingAssetPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| policy_from(std::env::var("RL_ALLOW_MISSING_ASSETS").ok().as_deref()))
}

fn policy_from(env: Option<&str>) -> MissingAssetPolicy {
    match env {
        None | Some("0") => MissingAssetPolicy::Panic,
        Some("1") => MissingAssetPolicy::Degrade,
        Some(other) => panic!(
            "RL_ALLOW_MISSING_ASSETS={other:?} is malformed — set 1 (explicitly asset-less \
             environment) or 0/unset (missing asset panics)"
        ),
    }
}

/// The single missing-asset chokepoint: every loader that finds an asset absent (or
/// unreadable) reports it here. Panics under [`MissingAssetPolicy::Panic`] naming the
/// path and the packaging step that should have shipped it; under `Degrade` it logs
/// and returns so the caller can fall back exactly as before.
///
/// `ship_step` names how a correct deployment gets this asset (the packaging step
/// and/or the dev fetch script) — the panic message must leave the reader with the
/// fix, not just the fact.
pub fn missing_asset(path: &Path, detail: &str, ship_step: &str) {
    match missing_asset_policy() {
        MissingAssetPolicy::Panic => panic!(
            "missing asset: {} ({detail}). {ship_step} A shipped build hitting this is a \
             packaging bug (rl#375); a legitimately asset-less environment must declare \
             itself with RL_ALLOW_MISSING_ASSETS=1.",
            path.display()
        ),
        MissingAssetPolicy::Degrade => tracing::warn!(
            "missing asset: {} ({detail}) — degrading (RL_ALLOW_MISSING_ASSETS=1). {ship_step}",
            path.display()
        ),
    }
}

/// Control-overlay glyphs: committed at crab-world/assets/controls/ and staged into
/// every release, so an absent one is a broken checkout or bundle.
pub fn require_glyphs<I: IntoIterator<Item = &'static str>>(paths: I) {
    let base = bevy_asset_path();
    for p in paths {
        let full = base.join(p);
        if !full.exists() {
            missing_asset(
                &full,
                "control glyph not found",
                "These CC0 Kenney glyphs are committed at crab-world/assets/controls/ and \
                 packaged by rl-release-build; if your assets live elsewhere, set \
                 BEVY_ASSET_ROOT.",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_parses_strictly() {
        assert_eq!(policy_from(None), MissingAssetPolicy::Panic);
        assert_eq!(policy_from(Some("0")), MissingAssetPolicy::Panic);
        assert_eq!(policy_from(Some("1")), MissingAssetPolicy::Degrade);
    }

    #[test]
    #[should_panic(expected = "malformed")]
    fn malformed_policy_value_dies_loudly() {
        policy_from(Some("true"));
    }

    #[test]
    #[should_panic(expected = "missing asset")]
    fn missing_glyph_panics_by_default() {
        // The suite runs without RL_ALLOW_MISSING_ASSETS, so the strict policy applies.
        require_glyphs(["controls/__surely_absent_glyph__.png"]);
    }

    #[test]
    fn no_glyphs_is_fine() {
        require_glyphs(std::iter::empty());
    }
}
