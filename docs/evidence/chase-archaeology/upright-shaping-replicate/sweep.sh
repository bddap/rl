#!/usr/bin/env bash
set -euo pipefail
: "${RUNS:?set RUNS to the directory holding the run instances}"
I=$1; L=$2; O=$(realpath -m "$3"); shift 3
D=$RUNS/$I
SEEDS=${*:-$(seq 351 370)}
eval "$(grep -E '^export (LD_LIBRARY_PATH|VK_DRIVER_FILES|VK_ICD_FILENAMES|WGPU_BACKEND)=' "$D/run-train.sh")"
export BEVY_ASSET_ROOT=$D/plant CRAB_MODEL_PATH=sally.glb
mkdir -p "$O/traces" "$O/logs" "$O/videos"
cd "$RUNS"
for s in $SEEDS; do
  "$D/rl-demo" render-video --checkpoint-dir "$I/ckpt/marks/40000000" --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed "$s" "$O/videos/$L-$s.mp4" > "$O/logs/$L-$s.log" 2>&1
  echo "$L $s exit=$?"
  sed 's/\x1b\[[0-9;]*m//g' "$O/logs/$L-$s.log" | grep -oE '(RENDER_CHECKPOINT|ROLLOUT_TRACE) .*' > "$O/traces/$L-$s.txt"
done
