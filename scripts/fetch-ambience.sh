#!/usr/bin/env bash
# CC0 ambient beds for the soundscape (rl#357) — see NOTICE for provenance.
# Absent beds are non-fatal: the ambience layers just stay silent.
set -euo pipefail

repo="bddap-bot/rl-assets"
tag="ambience-v3"

manifest="$(cargo metadata --format-version 1 --no-deps \
    | jq -r '.packages[] | select(.name == "crab-world") | .manifest_path')"
if [[ -z "$manifest" ]]; then
    echo "error: cargo metadata reported no 'crab-world' package — was it renamed?" >&2
    echo "       update this script's selector to match crab_world::assets::asset_root()." >&2
    exit 1
fi
dest="${BEVY_ASSET_ROOT:-$(dirname "$manifest")}/assets/ambience"

mkdir -p "$dest"
gh release download "$tag" -R "$repo" -p '*.wav' -D "$dest" --clobber
echo "fetched ambience beds -> $dest"
