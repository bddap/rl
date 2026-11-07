#!/usr/bin/env bash
# Usage: sally.sh export <out.glb> [<render-dir>]
#        sally.sh rig <out.glb> [<render-dir>]
# export: the purchased mesh as bought, checked against the .blend.
# rig: the mesh skinned to the skeleton in skeleton.md, its report beside it, then
#      dropped on flat ground in rapier to check the bodies and hinges derived from
#      the bones hold together.
# The purchased .blend may not be redistributed (NOTICE): it is read from
# $SALLY_BLEND, else fetched from the private release; the .glb it writes is gitignored.
set -euo pipefail

usage="usage: sally.sh export|rig <out.glb> [<render-dir>]"
mode="${1:?$usage}"
out="$(realpath -m "${2:?$usage}")"
render_dir="${3:+$(realpath -m "$3")}"

blend_sha256=18fc06ca7967cb7d69b05bee034c3b293acdf8afb005e6798523e0e4cb006060
crab=Grapsus_grapsus

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

blend="${SALLY_BLEND:-}"
if [[ -z "$blend" ]]; then
    cache="${XDG_CACHE_HOME:-$HOME/.cache}/rl-assets"
    blend="$cache/grapsus_grapsus_crab_280.blend"
    if [[ ! -f "$blend" ]]; then
        mkdir -p "$cache"
        gh release download v1 -R bddap-bot/rl-assets -p grapsus_grapsus_crab_280.blend -O "$blend.part" --clobber
        mv "$blend.part" "$blend"
    fi
fi
echo "$blend_sha256  $blend" | sha256sum --check --quiet

blender="$(nix-build --no-out-link "$repo/nixpkgs.nix" -A blender)/bin/blender"
run() { "$blender" -b --factory-startup --python-exit-code 1 "$@"; }

mkdir -p "$(dirname "$out")"
[[ -z "$render_dir" ]] || mkdir -p "$render_dir"
case "$mode" in
export)
    run "$blend" --python "$here/export.py" -- "$crab" "$out"
    run "$blend" --python "$here/verify.py" -- "$crab" "$out"
    if [[ -n "$render_dir" ]]; then
        for look in clay own; do
            run "$blend" --python "$here/render.py" -- "$crab" "$blend" "$render_dir/blend-$look" "$look"
            run "$blend" --python "$here/render.py" -- "$crab" "$out" "$render_dir/glb-$look" "$look"
        done
        run --python "$here/compare.py" -- "$render_dir" "$render_dir/sheet.png"
    fi
    ;;
rig)
    run "$blend" --python "$here/rig.py" -- "$here/skeleton.md" "$crab" "$out" "${out%.glb}.json"
    (cd "$repo" && nix-shell --run "cargo run -q --release -p crab-world --bin sally-drop -- '$out'")
    if [[ -n "$render_dir" ]]; then
        for look in mesh hulls; do
            run --python "$here/pose.py" -- "$out" "$render_dir/$look" "$look"
        done
    fi
    ;;
*)
    echo "$usage" >&2
    exit 2
    ;;
esac
