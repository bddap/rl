#!/usr/bin/env bash
# Browser↔native relay echo probe (rl#411 stage 4): a wasm iroh endpoint in headless
# chromium round-trips datagrams with a native peer through the n0 dev relays, on the
# game's ALPN. The transport canary for net-link's web half — rerun it when bumping
# iroh, changing relay posture, or debugging the browser path.
#
# Verified 2026-08-25 (iroh 1.0, chromium headless): PROBE_OK rounds=10
# median_rtt_ms=4.0 (connect 147 ms through https://usw1-1.relay.n0.iroh.link./).
#
# Gotchas encoded below:
# - wasm LINKING needs lld, and ring's C sources need a clang that can emit wasm
#   objects — the nix-WRAPPED clang silently produces x86 .o's that only fail at link
#   time ("neither Wasm object file nor LLVM bitcode"), hence clang-unwrapped +
#   CC_wasm32_unknown_unknown.
# - wasm-bindgen CLI must match the workspace lock's wasm-bindgen version; a mismatch
#   fails loudly at bindgen time.
set -euo pipefail
cd "$(dirname "$0")/../.."   # net-link/

CLANG_UNWRAPPED=$(nix-build '<nixpkgs>' -A llvmPackages.clang-unwrapped --no-out-link)/bin/clang
LLVM_AR=$(nix-build '<nixpkgs>' -A llvm --no-out-link)/bin/llvm-ar
TARGET_DIR=${CARGO_TARGET_DIR:-../target}
WB_LOCK=$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/version = |"/,""); print; exit}' ../Cargo.lock)

nix-shell -p lld --run "
  export CC_wasm32_unknown_unknown=$CLANG_UNWRAPPED AR_wasm32_unknown_unknown=$LLVM_AR
  nix-shell ../shell.nix --run 'cargo build --target wasm32-unknown-unknown --release --example echo_web && cargo build --release --example echo_native'
"
nix-shell -p wasm-bindgen-cli --run "
  [ \"\$(wasm-bindgen --version)\" = 'wasm-bindgen $WB_LOCK' ] || {
    echo \"wasm-bindgen CLI (\$(wasm-bindgen --version)) != workspace lock ($WB_LOCK) — use a nixpkgs (or cargo install --version $WB_LOCK) that matches\"; exit 1; }
  wasm-bindgen --target web --out-dir examples/web-echo/pkg \
    $TARGET_DIR/wasm32-unknown-unknown/release/examples/echo_web.wasm
"

"$TARGET_DIR/release/examples/echo_native" > /tmp/echo-native.log &
NATIVE_PID=$!
trap '[ -n "${NATIVE_PID:-}" ] && kill "$NATIVE_PID" 2>/dev/null; [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null; true' EXIT
until grep -q PROBE_RELAY /tmp/echo-native.log; do sleep 0.5; done
ID=$(grep -oP 'PROBE_ID=\K.*' /tmp/echo-native.log)
RELAY=$(grep -oP 'PROBE_RELAY=\K.*' /tmp/echo-native.log)
echo "native peer: $ID via $RELAY"

(cd examples/web-echo && exec node -e '
const http=require("http"),fs=require("fs"),path=require("path");
const mime={".html":"text/html",".js":"text/javascript",".wasm":"application/wasm"};
http.createServer((req,res)=>{
  let p=req.url.split("?")[0]; if(p==="/")p="/index.html";
  fs.readFile(path.join(process.cwd(),p),(e,d)=>{ if(e){res.writeHead(404);res.end();return;}
    res.writeHead(200,{"Content-Type":mime[path.extname(p)]||"application/octet-stream"});res.end(d);});
}).listen(8642);') &
SERVER_PID=$!
sleep 1

nix-shell -p chromium --run "
  timeout 120 chromium --headless=new --no-sandbox --disable-gpu --enable-logging=stderr --v=0 \
    'http://127.0.0.1:8642/index.html?id=$ID&relay=$RELAY&rounds=10'
" 2>&1 | grep -o 'PROBE_[^"]*'
