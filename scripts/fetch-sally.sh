#!/usr/bin/env bash
# Fetch the licensed Sally Lightfoot model from the private bddap-bot/rl-assets
# release into the ONE directory the renderers resolve assets against:
# `crab_world::assets::asset_root()/assets` (bevy's AssetServer and meshfit::model_path
# both root on `asset_root()` — see crab-world/src/assets.rs). Dropping it elsewhere is
# the trap the owner hit (bddap/rl#148): present in ./assets, yet the renderer looks
# under the resolved root and falls back.
#
# The dest is DERIVED from that same resolver, not hard-coded, so it can't drift:
# `cargo metadata` reports crab-world's real manifest path (move the crate dir and the
# dest follows; rename the *package* and the selector below fails LOUD instead of silently
# dropping the mesh), and BEVY_ASSET_ROOT is honored with asset_root()'s exact precedence
# (below). The rl#148 failure class made impossible by construction.
#
# Requires `gh` authenticated as an account with read access to that repo (the
# asset is paid and not redistributable — see NOTICE). Without it, the build has
# no Sally mesh and the demo/game show the loud physics-bones fallback.
set -euo pipefail

repo="bddap-bot/rl-assets"
tag="v1"

manifest="$(cargo metadata --format-version 1 --no-deps \
    | jq -r '.packages[] | select(.name == "crab-world") | .manifest_path')"
if [[ -z "$manifest" ]]; then
    echo "error: cargo metadata reported no 'crab-world' package — was it renamed?" >&2
    echo "       update this script's selector to match crab_world::assets::asset_root()." >&2
    exit 1
fi
# Mirror asset_root()'s precedence EXACTLY: BEVY_ASSET_ROOT (set by deploy, which stages assets/
# beside the shipped binary) wins, else the crab-world crate dir. Without honoring it, fetching on
# a host where deploy set BEVY_ASSET_ROOT would drop sally.glb in the crate dir while the renderer
# looks under BEVY_ASSET_ROOT — the same silent-fallback trap on a different axis (bddap/rl#148).
dest="${BEVY_ASSET_ROOT:-$(dirname "$manifest")}/assets"

mkdir -p "$dest"
gh release download "$tag" -R "$repo" -p sally.glb -D "$dest" --clobber
echo "fetched sally.glb -> $dest"
echo "run with: cargo run --release -p rl-demo -- --demo"
