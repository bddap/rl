#!/usr/bin/env bash
# Launch the terrain-diffusion model API in a run-untrusted sandbox (third-party
# ML code never runs unsandboxed — CLAUDE.md SECURITY). The server binds
# 127.0.0.1:$PORT in the shared net namespace; bake.py talks to it from outside.
#
# usage: serve-model.sh [WORKDIR] [PORT]
#   WORKDIR (default ~/.cache/terrain-bake) holds the upstream clone, venv,
#   pip + huggingface caches. Everything untrusted stays inside it.
set -euo pipefail

WORK="${1:-$HOME/.cache/terrain-bake}"
PORT="${2:-8017}"
UPSTREAM=https://github.com/xandergos/terrain-diffusion
# Pin: the artifact must be re-bakeable; a moving master is not a provenance.
# gcr-seed281 was baked at this rev (recorded in its metadata as model_rev).
UPSTREAM_REV="${UPSTREAM_REV:-82a0431281f21a6ec3d691a12ee61525de5b0790}"
PY="$(nix-shell -p python312 --run 'command -v python3')"

mkdir -p "$WORK"
[ -d "$WORK/terrain-diffusion" ] || git clone "$UPSTREAM" "$WORK/terrain-diffusion"
git -C "$WORK/terrain-diffusion" fetch -q origin
git -C "$WORK/terrain-diffusion" checkout -q "$UPSTREAM_REV"

cd "$WORK"
cat > sandbox-entry.sh <<EOF
set -euo pipefail
export LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib:/run/opengl-driver/lib
export PIP_CACHE_DIR=\$PWD/pipcache
export HF_HOME=\$PWD/hf
# -x probe, not -d: a GC'd store python leaves a venv of dangling shebangs,
# and an interrupted first run leaves a half-made dir; both must rebuild.
[ -x venv/bin/pip ] && venv/bin/pip --version >/dev/null 2>&1 || { rm -rf venv; $PY -m venv venv; }
# Inference-only subset of upstream requirements.txt (skips wandb/optuna/
# earthengine/cartopy + the rest of the training stack).
./venv/bin/pip install -q --no-input torch torchvision numpy diffusers \
  safetensors ema-pytorch Flask infinite-tensor matplotlib numba Pillow \
  "pyfastnoiselite==0.0.6" rasterio scipy tqdm click h5py scikit-image \
  easydict pyyaml
cd terrain-diffusion
# --no-compile: torch.compile needs triton warmup and can be flaky on older
# GPUs (sm_75); a one-off bake doesn't need the steady-state speedup.
exec ../venv/bin/python -m terrain_diffusion.inference.api \
  --no-compile --host 127.0.0.1 --port $PORT
EOF
exec run-untrusted -g bash sandbox-entry.sh
