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

# Verify the download against the committed collider table's asset pin
# (crab-world/src/bot/rig/baked.rs — the ONE digest source; rl#340 stage 6). The
# runtime already refuses a mismatched sally.glb, but the sally-gated tests key on
# that verdict and SKIP under it, so an unverified fetch that landed wrong bytes
# would silently un-run the mesh suite — the rot channel this check closes. A
# mismatch means the release tag and the baked table are out of step (a re-bake
# ships both in one commit). Download to a temp name and rename only after the
# digest passes: no failure mode leaves unverified bytes at the real name, and a
# previously-verified copy survives a failed re-fetch.
baked="$(dirname "$manifest")/src/bot/rig/baked.rs"
mapfile -t pins < <(grep -oE 'BAKED_ASSET_DIGEST: u64 = 0x[0-9a-fA-F]+' "$baked" \
    | grep -oE '0x[0-9a-fA-F]+' || true)
if [[ ${#pins[@]} -ne 1 ]]; then
    echo "error: expected exactly one BAKED_ASSET_DIGEST pin in $baked, found ${#pins[@]}" >&2
    echo "       — cannot verify sally.glb (did a re-bake or reformat change the line?)" >&2
    exit 1
fi
pin="${pins[0]}"

mkdir -p "$dest"
tmp="$dest/sally.glb.fetching"
gh release download "$tag" -R "$repo" -p sally.glb -O "$tmp" --clobber
# Node computes FNV-1a/64 and does the comparison itself (BigInt — no bash
# arithmetic on 64-bit hex); its exit status IS the verdict, so a node crash
# lands in the failure arm too.
if got="$(node -e '
    const fs = require("fs");
    const bytes = fs.readFileSync(process.argv[1]);
    let h = 0xcbf29ce484222325n;
    const prime = 0x100000001b3n, mask = 0xffffffffffffffffn;
    for (const b of bytes) { h ^= BigInt(b); h = (h * prime) & mask; }
    console.log("0x" + h.toString(16).padStart(16, "0"));
    process.exit(h === BigInt(process.argv[2]) ? 0 : 1);
' "$tmp" "$pin")"; then
    mv -f "$tmp" "$dest/sally.glb"
else
    rm -f "$tmp"
    echo "error: sally.glb digest ${got:-"(digest computation failed)"} does not match" >&2
    echo "       the baked collider pin $pin ($baked). The release asset and the" >&2
    echo "       committed table are out of step — a re-bake ships both in one" >&2
    echo "       commit. Removed the unverified download." >&2
    exit 1
fi

echo "fetched sally.glb -> $dest (digest $got matches baked pin)"
echo "run with: cargo run --release -p rl-demo -- demo"
