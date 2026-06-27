#!/usr/bin/env bash
# GCR #82 proof: the real trained NN crab is the DETERMINISTIC multiplayer crab.
#
# Runs the `game nn-crab-xpeer` gate — two independent in-process peers, each stepping its OWN
# float rapier NN crab under the trained policy and exchanging lockstep inputs — across several
# seeds, and confirms every peer pair's per-tick state-hash logs `diff` byte-identically with
# zero lockstep desyncs. A single diverging tick fails the script (and is the netcode-rethink
# trigger). This is the headless, same-arch stand-in for the on-Deck 2-machine gate; the binary
# itself exits nonzero on divergence, so it is CI-runnable directly.
#
# Usage: scripts/gcr82-xpeer-determinism.sh [CHECKPOINT_DIR]
#   CHECKPOINT_DIR defaults to $RL_CRAB_CHECKPOINT_DIR, else assets/weights under the repo root.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
ckpt="${1:-${RL_CRAB_CHECKPOINT_DIR:-$root/assets/weights}}"
game="${GAME_BIN:-$root/target/release/game}"
ticks="${TICKS:-1200}"
seeds=(1 42 999999 1666463330)
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [[ ! -x "$game" ]]; then
  echo "building game (release)…" >&2
  (cd "$root" && cargo build --release -p game)
fi
if [[ ! -f "$ckpt/brain.bin" ]]; then
  echo "no brain.bin under $ckpt — pass a trained checkpoint dir or set RL_CRAB_CHECKPOINT_DIR" >&2
  exit 2
fi

echo "GCR#82 cross-peer NN-crab determinism — checkpoint=$ckpt ticks=$ticks"
fail=0
for s in "${seeds[@]}"; do
  a="$work/s${s}_a.log"; b="$work/s${s}_b.log"
  "$game" nn-crab-xpeer --checkpoint "$ckpt" --seed "$s" --ticks "$ticks" \
    --hash-log-a "$a" --hash-log-b "$b" >/dev/null
  if diff -q "$a" "$b" >/dev/null; then
    distinct="$(sort -u "$a" | wc -l)"
    echo "  seed $s: PASS — $(wc -l < "$a") ticks byte-identical across peers ($distinct distinct hashes)"
  else
    echo "  seed $s: FAIL — peers DIVERGED (first diff: $(diff "$a" "$b" | head -3 | tr '\n' ' '))"
    fail=1
  fi
done

if [[ $fail -eq 0 ]]; then
  echo "GCR#82: PASS — the trained NN crab is the deterministic multiplayer crab."
else
  echo "GCR#82: FAIL — float NN crab diverged across peers → netcode-rethink trigger."
fi
exit $fail
