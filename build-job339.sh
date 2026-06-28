#!/usr/bin/env bash
# Build job-339's changed crates in the isolated worktree. Sandboxed (bevy/rapier
# build scripts are third-party), niced + pinned to cores 0-13 so the live trainer
# and GCR game keep their cores. Reuses the warm copied target/.
set -euo pipefail
WT=/home/bot/.cache/botq-wt/339
export CARGO_HOME=/home/bot/.cache/rl-target-train/.cargo-home
export CARGO_TARGET_DIR="$WT/target"
cd "$WT"
ACTION="${1:-build}"
case "$ACTION" in
  check)  CMD="cargo check -p net -p game -p rl-demo -p crab-world --tests" ;;
  build)  CMD="cargo build --release --bin rl-demo" ;;
  clippy) CMD="cargo clippy -p net -p game -p rl-demo -p crab-world --release -- -D warnings" ;;
  *) echo "usage: build-job339.sh [check|build|clippy]" >&2; exit 2 ;;
esac
exec nix-shell "$WT/shell.nix" --run "nice -n 19 taskset -c 0-13 $CMD"
