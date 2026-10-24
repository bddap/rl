#!/usr/bin/env bash
# render.sh <arm ctrl|floor07> <training-seed> <mean|sampled> <eval-seed> <outdir>
# RUNS holds the registered entropy-floor run dirs (<arm>-s<seed>/ckpt/marks/40000000),
# DEMO the pinned rl-demo, PLANT an asset root whose assets/sally.glb is the pinned plant.
set -uo pipefail
: "${RUNS:?}" "${DEMO:?}" "${PLANT:?}"
A=$1; TS=$2; MODE=$3; S=$4; O=$(realpath -m "$5")
case $A in ctrl) FLOOR=-1 ;; floor07) FLOOR=-0.7 ;; *) exit 2 ;; esac
EXTRA=()
[ "$MODE" = sampled ] && EXTRA=(--exploration-log-std-floor "$FLOOR")
export BEVY_ASSET_ROOT=$PLANT CRAB_MODEL_PATH=sally.glb
mkdir -p "$O/videos" "$O/logs" "$O/traces"
L=$A-s$TS-$MODE-$S
cd "$RUNS"
timeout 400 "$DEMO" render-video --checkpoint-dir "$A-s$TS/ckpt/marks/40000000" \
  --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed "$S" "${EXTRA[@]}" \
  "$O/videos/$L.mp4" > "$O/logs/$L.log" 2>&1
rc=$?
sed 's/\x1b\[[0-9;]*m//g' "$O/logs/$L.log" | grep -oE '(RENDER_ROLLOUT|RENDER_CHECKPOINT|ROLLOUT_TRACE) .*' > "$O/traces/$L.txt"
echo "$L exit=$rc"
