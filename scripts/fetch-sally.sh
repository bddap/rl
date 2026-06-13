#!/usr/bin/env bash
# Fetch the licensed Sally Lightfoot model into assets/ from the private
# bddap-bot/rl-assets release. Requires `gh` authenticated as an account with
# read access to that repo (the asset is paid and not redistributable — see
# NOTICE). Without it, the build uses the primitive-mesh crab.
set -euo pipefail

repo="bddap-bot/rl-assets"
tag="v1"
dest="$(git rev-parse --show-toplevel)/assets"

mkdir -p "$dest"
gh release download "$tag" -R "$repo" -p sally.glb -D "$dest" --clobber
echo "fetched sally.glb -> $dest"
echo "run with: CRAB_MODEL_PATH=sally.glb cargo run --release -- --demo"
