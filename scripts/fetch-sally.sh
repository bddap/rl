#!/usr/bin/env bash
# Fetch the licensed Sally Lightfoot model from the private bddap-bot/rl-assets
# release into the ONE directory the renderers resolve assets against: the
# crab-world crate's `assets/` dir (bevy's AssetServer and meshfit::model_path
# both root on `crab_world::assets::asset_root()` == that crate's manifest dir —
# see crab-world/src/assets.rs). Dropping it at the repo root instead is the trap
# the owner hit (bddap/rl#706): present in ./assets, yet the renderer looks under
# crab-world/assets and falls back.
#
# The dest is ASKED of cargo, not hard-coded: `cargo metadata` reports crab-world's
# real manifest path, so sally.glb lands wherever the crate actually lives. Move the
# crate dir and the dest follows; rename the *package* and the selector below fails
# LOUD (no match) instead of silently dropping the mesh in the old place — the rl#706
# failure class made impossible by construction.
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
dest="$(dirname "$manifest")/assets"

mkdir -p "$dest"
gh release download "$tag" -R "$repo" -p sally.glb -D "$dest" --clobber
echo "fetched sally.glb -> $dest"
echo "run with: cargo run --release -p rl-demo -- --demo"
