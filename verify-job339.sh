#!/usr/bin/env bash
# Verify job-339 headless via the rl-demo screenshot path:
#   present)  sally.glb resolvable -> real Sally renders to a PNG (exit 0)
#   absent)   no model -> rl-demo fails LOUD and refuses (nonzero exit, clear stderr)
# lavapipe software render (no GPU), mirroring render-shot.sh.
set -uo pipefail
WT=/home/bot/.cache/botq-wt/339
BIN="$WT/target/release/rl-demo"
SRC_SALLY=/home/bot/.cache/rl-train-main/assets/sally.glb
CKPT_SRC=/home/bot/.local/state/rl-target/ckpt
SNAP=/tmp/job339-ckpt
OUT=/tmp/job339-shots
rm -rf "$SNAP" "$OUT"; mkdir -p "$SNAP" "$OUT"
for f in brain.bin normalizer.bin return_normalizer.bin; do cp "$CKPT_SRC/$f" "$SNAP/$f"; done

export LD_LIBRARY_PATH=/nix/store/0jgicjfcml2v3plj470ggf8q88xkxq4d-vulkan-loader-1.4.313.0/lib:/run/opengl-driver/lib
export VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/lvp_icd.x86_64.json
export WGPU_BACKEND=vulkan
export CRAB_MODEL_PATH=sally.glb

echo "########## STATE 1: sally.glb PRESENT -> expect real Sally PNG ##########"
PRESENT_ROOT=/tmp/job339-present
rm -rf "$PRESENT_ROOT"; mkdir -p "$PRESENT_ROOT/assets"
cp "$SRC_SALLY" "$PRESENT_ROOT/assets/sally.glb"
BEVY_ASSET_ROOT="$PRESENT_ROOT" timeout 200 "$BIN" \
  --checkpoint-dir "$SNAP" --screenshot "$OUT/present.png" --screenshot-settle 240 2>&1 | tail -6
echo "present exit: ${PIPESTATUS[0]}"
ls -l "$OUT/present.png" 2>&1 || echo "NO PNG (unexpected)"

echo
echo "########## STATE 2: sally.glb ABSENT -> expect LOUD refuse, nonzero exit ##########"
ABSENT_ROOT=/tmp/job339-absent
rm -rf "$ABSENT_ROOT"; mkdir -p "$ABSENT_ROOT/assets"   # assets dir exists but NO sally.glb
BEVY_ASSET_ROOT="$ABSENT_ROOT" timeout 120 "$BIN" \
  --checkpoint-dir "$SNAP" --screenshot "$OUT/absent.png" --screenshot-settle 240 2>&1 | tail -6
echo "absent exit: ${PIPESTATUS[0]}  (nonzero = correct loud-fail)"
ls -l "$OUT/absent.png" 2>/dev/null && echo "!! UNEXPECTED PNG on absent (silent fallback?)" || echo "no PNG on absent (correct)"
