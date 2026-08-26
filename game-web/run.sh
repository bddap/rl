#!/usr/bin/env bash
# The browser build, end to end (rl#411): build the game-web wasm bundle,
# serve it (with the asset tree over HTTP), and — for CI/evidence — drive solo play
# in headless chromium and assert ZERO non-local network contact.
#
#   run.sh build    — wasm build + wasm-bindgen into game-web/pkg
#   run.sh serve    — build, then serve on http://127.0.0.1:8643 (Ctrl-C to stop)
#   run.sh verify   — build, serve, headless-chromium solo-play probe + net assert;
#                     artifacts (screenshots, console log, netlog) in $OUT
#
# The served asset root is ${BEVY_ASSET_ROOT:-crab-world/} — the SAME resolution
# native launches use. Solo play needs the full tree there: committed glyphs/terrain
# plus sally.glb + ambience (scripts/fetch-sally.sh, scripts/fetch-ambience.sh) and
# trained weights under assets/weights/ — a partial tree refuses at boot (rl#375).
#
# Toolchain gotchas inherited from net-link/examples/web-echo/run.sh: wasm linking
# needs lld; ring's C sources need an UNWRAPPED clang (the nix-wrapped one emits x86
# objects that only fail at link time); the wasm-bindgen CLI must match Cargo.lock.
set -euo pipefail
cd "$(dirname "$0")"   # game-web/

MODE=${1:-serve}
PORT=${PORT:-8643}
OUT=${OUT:-/tmp/gcr-web-verify}
ASSET_ROOT=${BEVY_ASSET_ROOT:-$(cd ../crab-world && pwd)}
ASSETS_DIR="$ASSET_ROOT/assets"
TARGET_DIR=${CARGO_TARGET_DIR:-../target}

build() {
  local clang_unwrapped llvm_ar wb_lock
  clang_unwrapped=$(nix-build '<nixpkgs>' -A llvmPackages.clang-unwrapped --no-out-link)/bin/clang
  llvm_ar=$(nix-build '<nixpkgs>' -A llvm --no-out-link)/bin/llvm-ar
  wb_lock=$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/version = |"/,""); print; exit}' ../Cargo.lock)
  nix-shell -p lld --run "
    export CC_wasm32_unknown_unknown=$clang_unwrapped AR_wasm32_unknown_unknown=$llvm_ar
    nix-shell ../shell.nix --run 'cargo build --target wasm32-unknown-unknown --release -p game-web'
  "
  nix-shell -p wasm-bindgen-cli --run "
    [ \"\$(wasm-bindgen --version)\" = 'wasm-bindgen $wb_lock' ] || {
      echo \"wasm-bindgen CLI (\$(wasm-bindgen --version)) != workspace lock ($wb_lock) — use a nixpkgs (or cargo install --version $wb_lock) that matches\"; exit 1; }
    wasm-bindgen --target web --out-dir pkg \
      $TARGET_DIR/wasm32-unknown-unknown/release/game_web.wasm
  "
}

# The read_asset prefetch manifest: exactly the byte-path consumer set — the
# checkpoint set under weights/, the control-glyph probe's PNGs, and the model
# digest gate's .glb — derived from the LIVE tree at serve time. Everything else
# (wavs, textures) streams through the AssetServer's own HTTP reader on demand,
# and terrain is compiled in (include_bytes).
gen_manifest() {
  [ -d "$ASSETS_DIR" ] || { echo "no asset tree at $ASSETS_DIR (set BEVY_ASSET_ROOT)"; exit 1; }
  (cd "$ASSETS_DIR" && {
    find weights controls -type f 2>/dev/null
    find . -maxdepth 1 -name '*.glb' | sed 's|^\./||'
  } | sort) > web-assets.txt
  echo "manifest: $(wc -l < web-assets.txt) files from $ASSETS_DIR"
}

serve() {
  ASSETS_DIR="$ASSETS_DIR" PORT="$PORT" exec node server.mjs
}

case "$MODE" in
  build) build ;;
  serve) build; gen_manifest; serve ;;
  verify)
    build; gen_manifest
    mkdir -p "$OUT"
    ASSETS_DIR="$ASSETS_DIR" PORT="$PORT" node server.mjs & SERVER_PID=$!
    trap '[ -z "${SERVER_PID:-}" ] || kill "$SERVER_PID" 2>/dev/null || true' EXIT
    sleep 1
    # This bevy build carries the webgl2 backend (bevy's webgpu feature is a
    # build-time either/or), so the probe exercises WebGL2 on SwiftShader — software
    # rasterization of the real pipeline. The resolver rule is the belt (any
    # non-local lookup fails at the socket layer); verify.mjs asserts the page layer.
    # setsid: $! would be the nix-shell wrapper, and killing only it on a failure
    # path orphans chromium holding the debug port + profile lock — kill the group.
    setsid nix-shell -p chromium --run "
      chromium --headless=new --no-sandbox --window-size=800,450 \
        --remote-debugging-port=9333 \
        --autoplay-policy=no-user-gesture-required \
        --user-data-dir=$OUT/chrome-profile \
        --no-first-run --no-default-browser-check --metrics-recording-only \
        --disable-background-networking --disable-component-update \
        --disable-sync --disable-default-apps --disable-domain-reliability \
        --disable-client-side-phishing-detection \
        --disable-features=DnsOverHttps,AsyncDns,OptimizationHints,Translate,MediaRouter \
        --host-resolver-rules='MAP * ~NOTFOUND, EXCLUDE 127.0.0.1' \
        --log-net-log=$OUT/netlog.json \
        about:blank
    " & CHROMIUM_PID=$!
    trap '[ -z "${SERVER_PID:-}" ] || kill "$SERVER_PID" 2>/dev/null || true; [ -z "${CHROMIUM_PID:-}" ] || kill -- "-$CHROMIUM_PID" 2>/dev/null || true' EXIT
    for _ in $(seq 1 60); do
      curl -sf --max-time 2 http://127.0.0.1:9333/json/version >/dev/null && break
      sleep 0.5
    done
    node verify.mjs "http://127.0.0.1:$PORT/index.html" "$OUT"
    wait "$CHROMIUM_PID" 2>/dev/null || true   # Browser.close flushes the netlog artifact
    CHROMIUM_PID=
    ;;
  *) echo "usage: run.sh build|serve|verify"; exit 2 ;;
esac
