#!/usr/bin/env bash
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
dest="${BEVY_ASSET_ROOT:-$(dirname "$manifest")}/assets"

mkdir -p "$dest"
gh release download "$tag" -R "$repo" -p sally.glb -D "$dest" --clobber
echo "fetched sally.glb -> $dest"
echo "run with: cargo run --release -p rl-demo -- --demo"
