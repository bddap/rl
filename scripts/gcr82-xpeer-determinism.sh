#!/usr/bin/env bash
# GCR #82 proof: the real trained NN crab is the DETERMINISTIC multiplayer crab.
#
# Runs the `game nn-crab-xpeer` gate — two independent peers, each stepping its OWN float rapier
# NN crab under the trained policy at the production 64:30 physics cadence and exchanging lockstep
# inputs — across several seeds, and confirms two things:
#   (1) WITHIN a process: the two peers' per-tick state-hash logs `diff` byte-identically, 0 desyncs.
#   (2) ACROSS processes: the same seed run in two SEPARATE OS processes yields identical logs —
#       so the determinism doesn't lean on process-global state two real machines wouldn't share.
# A single diverging tick fails the script (and is the netcode-rethink trigger). The binary itself
# exits nonzero on divergence, so it is CI-runnable directly.
#
# SCOPE: this is the headless SAME-ARCH (x86_64) stand-in for the on-Deck 2-machine gate — it does
# NOT prove cross-architecture float determinism (the all-x86_64 Deck fleet doesn't need it; the
# integer-pursuit fallback remains the guard if a non-x86_64 peer ever appears).
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
  # Process 1: the in-process two-peer gate (check 1 — A vs B within one process).
  a="$work/s${s}_a.log"; b="$work/s${s}_b.log"
  "$game" nn-crab-xpeer --checkpoint "$ckpt" --seed "$s" --ticks "$ticks" \
    --hash-log-a "$a" --hash-log-b "$b" >/dev/null
  # Process 2: a SEPARATE OS process, same seed (check 2 — peer A across processes).
  a2="$work/s${s}_a2.log"; b2="$work/s${s}_b2.log"
  "$game" nn-crab-xpeer --checkpoint "$ckpt" --seed "$s" --ticks "$ticks" \
    --hash-log-a "$a2" --hash-log-b "$b2" >/dev/null

  if diff -q "$a" "$b" >/dev/null && diff -q "$a" "$a2" >/dev/null; then
    distinct="$(sort -u "$a" | wc -l)"
    echo "  seed $s: PASS — $(wc -l < "$a") ticks byte-identical in-process AND cross-process ($distinct distinct hashes)"
  else
    diff -q "$a" "$b"  >/dev/null || { echo "  seed $s: FAIL (in-process A!=B): $(diff "$a" "$b" | head -1)"; fail=1; }
    diff -q "$a" "$a2" >/dev/null || { echo "  seed $s: FAIL (cross-process): $(diff "$a" "$a2" | head -1)"; fail=1; }
  fi
done

if [[ $fail -eq 0 ]]; then
  echo "GCR#82: PASS — the trained NN crab is the deterministic multiplayer crab."
else
  echo "GCR#82: FAIL — float NN crab diverged across peers → netcode-rethink trigger."
fi
exit $fail
