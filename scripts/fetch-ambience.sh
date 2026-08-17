#!/usr/bin/env bash
# CC0 ambient beds for the soundscape (rl#357) — see NOTICE for provenance.
# The release packager (bothouse rl-release-build) runs this at package time so
# assets/ambience/ ships with every release (rl#375); dev checkouts run it by hand.
set -euo pipefail

repo="bddap-bot/rl-assets"
tag="ambience-v3"

# BEVY_ASSET_ROOT is the same override crab_world::assets::asset_root() honors; with
# it set (the packager's case) no cargo is needed to locate the destination.
if [[ -n "${BEVY_ASSET_ROOT:-}" ]]; then
    dest="$BEVY_ASSET_ROOT/assets/ambience"
else
    manifest="$(cargo metadata --format-version 1 --no-deps \
        | jq -r '.packages[] | select(.name == "crab-world") | .manifest_path')"
    if [[ -z "$manifest" ]]; then
        echo "error: cargo metadata reported no 'crab-world' package — was it renamed?" >&2
        echo "       update this script's selector to match crab_world::assets::asset_root()." >&2
        exit 1
    fi
    dest="$(dirname "$manifest")/assets/ambience"
fi

mkdir -p "$dest"

# Skip the download when this tag's beds are already here: release assets are immutable
# per tag, and hitting GitHub on every package tick makes API flakes fail the release
# red (2026-08-17: transient REST 404s on an existing release). The marker records which
# tag the on-disk beds came from, so a tag bump here still re-fetches.
marker="$dest/.fetched-tag"
if [[ "$(cat "$marker" 2>/dev/null)" == "$tag" ]] && compgen -G "$dest/*.wav" >/dev/null; then
    echo "ambience beds already at $tag -> $dest"
    exit 0
fi

gh release download "$tag" -R "$repo" -p '*.wav' -D "$dest" --clobber
echo "$tag" > "$marker"
echo "fetched ambience beds -> $dest"
