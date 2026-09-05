#!/usr/bin/env bash
# The browser build, end to end (rl#411): build the game-web wasm bundle, assemble
# the self-contained static site, serve it, and — for CI/evidence — drive solo play
# in headless chromium asserting ZERO non-local network contact.
#
#   run.sh build             — wasm build + wasm-bindgen into game-web/pkg
#   run.sh dist              — build, then assemble game-web/dist: index.html, pkg/,
#                              assets.pack (the WHOLE asset tree baked into one blob —
#                              a hosted build serves no loose asset files, rl#411
#                              stage 6), version.txt
#   run.sh serve             — dist, then serve it on http://127.0.0.1:8643
#   run.sh verify            — dist, serve, headless-chromium solo-play probe
#   run.sh verify-live URL   — the same probe against an already-HOSTED bundle
#
# The baked asset root is ${BEVY_ASSET_ROOT:-crab-world/} — the SAME resolution
# native launches use. dist needs the full tree there: committed glyphs plus
# sally.glb + ambience (scripts/fetch-sally.sh, scripts/fetch-ambience.sh) and
# trained weights under assets/weights/ — a partial tree refuses HERE, mirroring the
# game's own boot refusal (rl#375), so a broken bundle never ships or serves.
#
# The whole wasm toolchain — lld, the ring-needs-unwrapped-clang env, and the
# wasm-bindgen CLI (which MUST match Cargo.lock's wasm-bindgen; game-web/Cargo.toml
# pins the crate family to shell.nix's CLI version and the assert below keeps them
# honest) — rides in ../shell.nix, so this script needs no channel/<nixpkgs> and runs
# identically for a dev, CI, and the release builder. chromium (verify only) is the
# one remaining channel fetch.
set -euo pipefail
cd "$(dirname "$0")"   # game-web/

MODE=${1:-serve}
PORT=${PORT:-8643}
OUT=${OUT:-/tmp/gcr-web-verify}
ASSET_ROOT=${BEVY_ASSET_ROOT:-$(cd ../crab-world && pwd)}
ASSETS_DIR="$ASSET_ROOT/assets"
TARGET_DIR=${CARGO_TARGET_DIR:-../target}
DIST="$PWD/dist"

build() {
  local wb_lock
  wb_lock=$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/version = |"/,""); print; exit}' ../Cargo.lock)
  nix-shell ../shell.nix --run "
    set -e
    [ \"\$(wasm-bindgen --version)\" = 'wasm-bindgen $wb_lock' ] || {
      echo \"wasm-bindgen CLI (\$(wasm-bindgen --version)) != workspace lock ($wb_lock) — bump the game-web/Cargo.toml pins and the lock to shell.nix's CLI version (they move together)\"; exit 1; }
    cargo build --target wasm32-unknown-unknown --profile web-release -p game-web
    # --remove-name-section: rustc's strip leaves the wasm name section (tens of MB
    # of symbol names) and wasm-bindgen re-emits it; browser backtraces degrade to
    # frame indexes, which a shipped bundle accepts for the size.
    wasm-bindgen --target web --remove-name-section --out-dir pkg \
      $TARGET_DIR/wasm32-unknown-unknown/web-release/game_web.wasm
  "
}

# The pieces of the tree a playable bundle cannot lack (rl#375: a partial tree must
# refuse loudly at packaging, never ship and degrade). The game's own boot chokepoint
# stays the authority; this is the packaging-time mirror that fails minutes earlier.
require_tree() {
  local missing=0 f
  for f in sally.glb weights/brain.bin weights/normalizer.bin; do
    [ -f "$ASSETS_DIR/$f" ] || { echo "missing $ASSETS_DIR/$f" >&2; missing=1; }
  done
  ls "$ASSETS_DIR"/ambience/*.wav >/dev/null 2>&1 || { echo "no ambience beds under $ASSETS_DIR/ambience/" >&2; missing=1; }
  ls "$ASSETS_DIR"/controls/*.png >/dev/null 2>&1 || { echo "no control glyphs under $ASSETS_DIR/controls/" >&2; missing=1; }
  [ "$missing" = 0 ] || {
    echo "asset tree at $ASSETS_DIR is incomplete (set BEVY_ASSET_ROOT; fetch via scripts/fetch-sally.sh + scripts/fetch-ambience.sh; stage weights under assets/weights/)" >&2
    exit 1
  }
}

dist() {
  build
  require_tree
  rm -rf "$DIST"
  mkdir -p "$DIST/pkg"
  cp index.html "$DIST/"
  cp pkg/game_web.js pkg/game_web_bg.wasm "$DIST/pkg/"
  [ ! -d pkg/snippets ] || cp -r pkg/snippets "$DIST/pkg/"
  nix-shell ../shell.nix --run "cargo run --release -p crab-world --bin pack-assets -- \
    --root '$ASSETS_DIR' --out '$DIST/assets.pack'"
  # Pages runs Jekyll on branch-served sites unless told not to; serve bytes as-is.
  touch "$DIST/.nojekyll"
  { git -C .. rev-parse --short=12 HEAD; date -u '+%F %T'; } > "$DIST/version.txt"
  du -sh "$DIST"/pkg/game_web_bg.wasm "$DIST"/assets.pack >&2
}

# Chromium hermeticity flags shared by verify and verify-live: the resolver rule is
# the belt (any lookup outside $2 fails at the socket layer); verify.mjs asserts the
# page layer. setsid: $! would be the nix-shell wrapper, and killing only it on a
# failure path orphans chromium holding the debug port + profile lock — kill the group.
start_chromium() {
  local resolver=$1
  setsid nix-shell -p chromium --run "
    chromium --headless=new --no-sandbox --window-size=800,450 \
      --remote-debugging-port=9333 \
      --user-data-dir=$OUT/chrome-profile \
      --no-first-run --no-default-browser-check --metrics-recording-only \
      --disable-background-networking --disable-component-update \
      --disable-sync --disable-default-apps --disable-domain-reliability \
      --disable-client-side-phishing-detection \
      --disable-features=DnsOverHttps,AsyncDns,OptimizationHints,Translate,MediaRouter \
      --host-resolver-rules='$resolver' \
      --log-net-log=$OUT/netlog.json \
      about:blank
  " & CHROMIUM_PID=$!
  for _ in $(seq 1 60); do
    curl -sf --max-time 2 http://127.0.0.1:9333/json/version >/dev/null && break
    sleep 0.5
  done
}

probe() {
  local url=$1
  mkdir -p "$OUT"
  node verify.mjs "$url" "$OUT"
  wait "$CHROMIUM_PID" 2>/dev/null || true   # Browser.close flushes the netlog artifact
  CHROMIUM_PID=
  echo "verify artifacts in $OUT"
}

case "$MODE" in
  build) build ;;
  dist) dist ;;
  serve)
    dist
    DIST_DIR="$DIST" PORT="$PORT" exec node server.mjs
    ;;
  verify)
    dist
    DIST_DIR="$DIST" PORT="$PORT" node server.mjs & SERVER_PID=$!
    trap '[ -z "${SERVER_PID:-}" ] || kill "$SERVER_PID" 2>/dev/null || true; [ -z "${CHROMIUM_PID:-}" ] || kill -- "-$CHROMIUM_PID" 2>/dev/null || true' EXIT
    sleep 1
    # This bevy build carries the webgl2 backend (bevy's webgpu feature is a
    # build-time either/or), so the probe exercises WebGL2 on SwiftShader — software
    # rasterization of the real pipeline.
    start_chromium 'MAP * ~NOTFOUND, EXCLUDE 127.0.0.1'
    probe "http://127.0.0.1:$PORT/index.html"
    ;;
  verify-live)
    LIVE_URL=${2:?usage: run.sh verify-live https://host/path/}
    LIVE_HOST=$(node -e 'console.log(new URL(process.argv[1]).host)' "$LIVE_URL")
    trap '[ -z "${CHROMIUM_PID:-}" ] || kill -- "-$CHROMIUM_PID" 2>/dev/null || true' EXIT
    start_chromium "MAP * ~NOTFOUND, EXCLUDE $LIVE_HOST"
    probe "$LIVE_URL"
    ;;
  *)
    echo "usage: run.sh {build|dist|serve|verify|verify-live URL}" >&2
    exit 1
    ;;
esac
