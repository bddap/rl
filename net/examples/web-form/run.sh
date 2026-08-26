#!/usr/bin/env bash
# Browser↔native CROSS-PLAY verify (rl#412): a wasm peer in headless chromium takes
# the REAL browser join path — pollable bind, pkarr-resolved dial of the host's bare
# join code, relay transport, formation to agreement, then the real round drivers —
# against a native `game net` host on this box. PASS requires formation to complete
# AND per-tick state hashes to MATCH between the host's --hash-log and the browser's
# adopted-snapshot log: byte-identical state that could only exist if the browser's
# inputs reached the host and the host's authority reached the browser.
#
# The formation-level sibling of net-link/examples/web-echo (the transport canary) —
# rerun on lobby/bind/relay changes. Toolchain gotchas (unwrapped clang for ring,
# wasm-bindgen CLI == lock) are the same; see that script.
set -euo pipefail
cd "$(dirname "$0")/../.."   # net/

RUN_SECS=${RUN_SECS:-15}
LOG_DIR=${LOG_DIR:-/tmp/xplay-verify}
mkdir -p "$LOG_DIR"

CLANG_UNWRAPPED=$(nix-build '<nixpkgs>' -A llvmPackages.clang-unwrapped --no-out-link)/bin/clang
LLVM_AR=$(nix-build '<nixpkgs>' -A llvm --no-out-link)/bin/llvm-ar
TARGET_DIR=${CARGO_TARGET_DIR:-../target}
WB_LOCK=$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/version = |"/,""); print; exit}' ../Cargo.lock)

nix-shell -p lld --run "
  export CC_wasm32_unknown_unknown=$CLANG_UNWRAPPED AR_wasm32_unknown_unknown=$LLVM_AR
  nix-shell ../shell.nix --run 'cargo build --target wasm32-unknown-unknown --release --example form_web && cargo build --release -p game'
"
nix-shell -p wasm-bindgen-cli --run "
  [ \"\$(wasm-bindgen --version)\" = 'wasm-bindgen $WB_LOCK' ] || {
    echo \"wasm-bindgen CLI (\$(wasm-bindgen --version)) != workspace lock ($WB_LOCK)\"; exit 1; }
  wasm-bindgen --target web --out-dir examples/web-form/pkg \
    $TARGET_DIR/wasm32-unknown-unknown/release/examples/form_web.wasm
"

HOST_LOG="$LOG_DIR/host.log"
HOST_HASHES="$LOG_DIR/host-hashes.log"
"$TARGET_DIR/release/game" net --expect 2 --discover-secs 120 \
  --run-secs $((RUN_SECS + 10)) --hash-log "$HOST_HASHES" >"$HOST_LOG" 2>&1 &
HOST_PID=$!
trap '[ -z "${HOST_PID:-}" ] || kill "$HOST_PID" 2>/dev/null || true; [ -z "${SERVER_PID:-}" ] || kill "$SERVER_PID" 2>/dev/null || true' EXIT
until grep -q 'game endpoint id:' "$HOST_LOG"; do sleep 0.5; done
HOST_ID=$(grep -oP 'game endpoint id: \K\S*' "$HOST_LOG")
echo "native host: $HOST_ID"

(cd examples/web-form && exec node -e '
const http=require("http"),fs=require("fs"),path=require("path");
const mime={".html":"text/html",".js":"text/javascript",".wasm":"application/wasm"};
http.createServer((req,res)=>{
  let p=req.url.split("?")[0]; if(p==="/")p="/index.html";
  fs.readFile(path.join(process.cwd(),p),(e,d)=>{ if(e){res.writeHead(404);res.end();return;}
    res.writeHead(200,{"Content-Type":mime[path.extname(p)]||"application/octet-stream"});res.end(d);});
}).listen(8644);') &
SERVER_PID=$!
sleep 1

BROWSER_LOG="$LOG_DIR/browser.log"
nix-shell -p chromium --run "
  timeout 240 chromium --headless=new --no-sandbox --disable-gpu --enable-logging=stderr --v=0 \
    'http://127.0.0.1:8644/index.html?host=$HOST_ID&secs=$RUN_SECS'
" 2>&1 | grep -o '\(FORM\|HASH\|XPLAY\)[^"]*' > "$BROWSER_LOG" || true
grep -v '^HASH' "$BROWSER_LOG"

wait "$HOST_PID" || true
HOST_PID=

# The verdict: formation completed, and the two sides logged IDENTICAL state hashes
# for a healthy overlap of ticks.
grep -q 'XPLAY_OK' "$BROWSER_LOG" || { echo "FAIL: browser probe did not finish"; exit 1; }
grep -oP 'HASH \K.*' "$BROWSER_LOG" > "$LOG_DIR/browser-hashes.log"
BROWSER_TICKS=$(wc -l < "$LOG_DIR/browser-hashes.log")
MATCHES=$(awk 'NR==FNR{h[$1]=$2;next} ($1 in h) && h[$1]==$2' "$HOST_HASHES" "$LOG_DIR/browser-hashes.log" | wc -l)
DIVERGED=$(awk 'NR==FNR{h[$1]=$2;next} ($1 in h) && h[$1]!=$2' "$HOST_HASHES" "$LOG_DIR/browser-hashes.log" | wc -l)
echo "hash join: $MATCHES identical tick-hashes of $BROWSER_TICKS browser-adopted ticks"
[ "$MATCHES" -ge 50 ] || { echo "FAIL: fewer than 50 matching tick-hashes"; exit 1; }
[ "$DIVERGED" -eq 0 ] || { echo "FAIL: $DIVERGED ticks present on both sides with DIFFERENT hashes"; exit 1; }
echo "XPLAY_VERIFY_PASS matches=$MATCHES diverged=0"
