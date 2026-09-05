#!/usr/bin/env bash
# capture.sh <game-binary> <out-dir>: the rl#367 jump-feel capture — seed 7, on foot,
# a 2-frame JUMP tap at frame 95, JUMP held over 130–175; per-tick altitude to trace.csv.
set -euo pipefail
GAME=$1 OUT=$2
mkdir -p "$OUT"
LOADER=$(dirname "$(find /nix/store -maxdepth 3 -name libvulkan.so.1 -path '*vulkan-loader-1.4.313*' | head -1)")
cd "$OUT"
LD_LIBRARY_PATH="$LOADER:/run/opengl-driver/lib" \
VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/lvp_icd.x86_64.json \
RL_POS_TRACE=trace.csv \
  "$GAME" fp-screenshot --seed 7 --players 1 --settle 90 --anim-frames 150 --anim-every 1 \
    --cam-pitch=-25 --width 640 --height 360 --jump-holds 95:97,130:175 --out f.png
