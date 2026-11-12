#!/usr/bin/env bash
# Usage: sally.sh export <out.glb> [<render-dir>]
#        sally.sh rig <out.glb> [<render-dir>]
#        sally.sh run <out-dir> [<ticks> [<seed>]]
#        sally.sh clips <out-dir> <policy.json> [<task>]
# export: the purchased mesh as bought, checked against the .blend.
# rig: the mesh skinned to the skeleton in skeleton.md, its report beside it, then
#      dropped on flat ground in rapier to check the bodies and hinges derived from
#      the bones hold together.
# run: the whole crab pipeline in one go: rig, search for the policy with seed <seed>
#      for <ticks> simulated ticks, score it and both baselines on the evals, and
#      write the three side by side to <out-dir>/scores.md. Run again on the same
#      <out-dir>, an interrupted search goes on from its log.
# clips: one episode of <task> (walk_left by default) from the same start under the
#      zero, random and trained policies, each rendered from its recorded physics
#      bodies to <out-dir>/<policy>.mp4, labelled with its score.
# The purchased .blend may not be redistributed (NOTICE): it is read from
# $SALLY_BLEND, else fetched from the private release; the .glb it writes is gitignored.
set -euo pipefail

usage="usage: sally.sh export|rig <out.glb> [<render-dir>] | run <out-dir> [<ticks> [<seed>]] | clips <out-dir> <policy.json> [<task>]"
mode="${1:?$usage}"
out="$(realpath -m "${2:?$usage}")"
if [[ "$mode" == run ]]; then
    train_args=(${3:+--ticks "$3"} ${4:+--seed "$4"})
    render_dir=
elif [[ "$mode" == clips ]]; then
    weights="$(realpath "${3:?$usage}")"
    task="${4:-walk_left}"
    render_dir=
else
    render_dir="${3:+$(realpath -m "$3")}"
fi

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
crab_world() { (cd "$repo" && nix-shell --run "cargo run -q --release -p crab-world --bin $1 -- $(printf '%q ' "${@:2}")"); }
rig() {
    run "$blend" --python "$here/rig.py" -- "$here/skeleton.md" "$crab" "$1" "${1%.glb}.json"
    crab_world sally-drop "$1"
}

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
    rig "$out"
    if [[ -n "$render_dir" ]]; then
        for look in mesh hulls; do
            run --python "$here/pose.py" -- "$out" "$render_dir/$look" "$look"
        done
    fi
    ;;
run)
    mkdir -p "$out"
    glb="$out/sally.glb"
    rig "$glb"
    crab_world sally-train "$glb" --out "$out/train" "${train_args[@]}"
    crab_world sally-eval "$glb" --policy zero >"$out/zero.eval"
    crab_world sally-eval "$glb" --policy random >"$out/random.eval"
    crab_world sally-eval "$glb" --policy trained --weights "$out/train/policy.json" >"$out/trained.eval"
    nix-shell -p python3 --run "python3 $(printf '%q ' "$here/scores.py" -- "$out/zero.eval" "$out/random.eval" "$out/trained.eval" "$out/scores.md")"
    cat "$out/scores.md"
    ;;
clips)
    mkdir -p "$out"
    glb="$out/sally.glb"
    rig "$glb"
    for policy in zero random trained; do
        args=(--policy "$policy" --task "$task" --seeds 1 --record "$out/$policy")
        [[ "$policy" != trained ]] || args+=(--weights "$weights")
        crab_world sally-eval "$glb" "${args[@]}" >"$out/$policy.eval"
        score="$(sed -n 's/^SALLY_EVAL //p' "$out/$policy.eval" | nix-shell -p jq --run "jq -r '.tasks[0].mean'")"
        run --python "$here/clip.py" -- \
            "$glb" "$out/$policy/$task-0.json" "$out/$policy.mp4" "$policy  $task  $score"
    done
    ls -l "$out"/*.mp4
    ;;
*)
    echo "$usage" >&2
    exit 2
    ;;
esac
