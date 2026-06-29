#!/usr/bin/env bash
# Fetch the licensed Sally Lightfoot model from the private bddap-bot/rl-assets
# release into the ONE directory the renderers resolve assets against:
# crab-world/assets/ (the crate dir bevy's AssetServer and meshfit::model_path
# both root on — see crab-world/src/assets.rs::asset_root). Dropping it at the
# repo root instead is the trap the owner hit (bddap/rl#706): present in
# ./assets, yet the renderer looks under crab-world/assets and falls back.
# Requires `gh` authenticated as an account with read access to that repo (the
# asset is paid and not redistributable — see NOTICE). Without it, the build has
# no Sally mesh and rl-demo shows the loud physics-bones fallback.
set -euo pipefail

repo="bddap-bot/rl-assets"
tag="v1"
dest="$(git rev-parse --show-toplevel)/crab-world/assets"

mkdir -p "$dest"
gh release download "$tag" -R "$repo" -p sally.glb -D "$dest" --clobber
echo "fetched sally.glb -> $dest"
echo "run with: cargo run --release -p rl-demo -- --demo"
