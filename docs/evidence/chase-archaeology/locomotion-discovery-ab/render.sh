#!/usr/bin/env bash
# render.sh <arm off|on> <training-seed> <eval-seed> <outdir>
# RUNS holds the run dirs (<arm>-s<seed>/ckpt/marks/40000000), DEMO the pinned rl-demo,
# PLANT an asset root whose assets/sally.glb is the pinned plant.
set -uo pipefail
: "${RUNS:?}" "${DEMO:?}" "${PLANT:?}"
A=$1; TS=$2; S=$3; O=$(realpath -m "$4")
export BEVY_ASSET_ROOT=$PLANT CRAB_MODEL_PATH=sally.glb
mkdir -p "$O/videos" "$O/logs" "$O/traces"
L=$A-s$TS-$S
timeout 400 "$DEMO" render-video --checkpoint-dir "$RUNS/$A-s$TS/ckpt/marks/40000000" \
  --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed "$S" \
  "$O/videos/$L.mp4" 2>&1 | grep -v 'returned POLLERR' > "$O/logs/$L.log"
rc=${PIPESTATUS[0]}
sed 's/\x1b\[[0-9;]*m//g' "$O/logs/$L.log" | grep -oE '(RENDER_ROLLOUT|RENDER_CHECKPOINT|ROLLOUT_TRACE|DRIVE_STATS) .*' > "$O/traces/$L.txt"
echo "$L exit=$rc samples=$(grep -c ROLLOUT_TRACE "$O/traces/$L.txt")"
