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

# Verify the download against the committed collider table's asset pin
# (crab-world/src/bot/rig/baked.rs — the ONE digest source; rl#340 stage 6). The
# runtime already refuses a mismatched sally.glb, but the sally-gated tests key on
# that verdict and SKIP under it, so an unverified fetch that landed wrong bytes
# would silently un-run the mesh suite — the rot channel this check closes. A
# mismatch here means the release tag and the baked table are out of step
# (a re-bake must ship both in one commit); the bad file is removed so nothing
# downstream half-trusts it.
baked="$(dirname "$manifest")/src/bot/rig/baked.rs"
pin="$(grep -oE 'BAKED_ASSET_DIGEST: u64 = 0x[0-9a-fA-F]+' "$baked" \
    | grep -oE '0x[0-9a-fA-F]+')"
if [[ -z "$pin" ]]; then
    echo "error: no BAKED_ASSET_DIGEST pin found in $baked — cannot verify sally.glb" >&2
    exit 1
fi
got="$(node -e '
    const fs = require("fs");
    const bytes = fs.readFileSync(process.argv[1]);
    let h = 0xcbf29ce484222325n;
    const prime = 0x100000001b3n, mask = 0xffffffffffffffffn;
    for (const b of bytes) { h ^= BigInt(b); h = (h * prime) & mask; }
    console.log("0x" + h.toString(16).padStart(16, "0"));
' "$dest/sally.glb")"
if (( got != pin )); then
    rm -f "$dest/sally.glb"
    echo "error: sally.glb digest $got does not match the baked collider pin $pin" >&2
    echo "       ($baked). The release asset and the committed table are out of" >&2
    echo "       step — a re-bake ships both in one commit. Removed the bad file." >&2
    exit 1
fi

echo "fetched sally.glb -> $dest (digest $got matches baked pin)"
echo "run with: cargo run --release -p rl-demo -- demo"
