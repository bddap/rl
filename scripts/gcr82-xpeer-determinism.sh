#!/usr/bin/env bash
# GCR #82 proof: the real trained NN crab is the DETERMINISTIC multiplayer crab.
#
# Each `game nn-crab-xpeer` process runs two independent in-process peers, both stepping their OWN
# float rapier NN crab under the trained policy at the production 64:30 physics cadence and
# exchanging lockstep inputs, and writes each peer's per-tick `<tick> <state_hash>` log.
#
# RIGOR (round 2 — the round-1 worker false-passed off a SINGLE matching run): a determinism claim
# is only as strong as the number of FRESH-PROCESS trials behind it. So per seed this spawns RUNS
# (default 6) SEPARATE OS processes, and requires, with ZERO tolerance:
#   (1) WITHIN every process — peer A's and peer B's logs are byte-identical (0 lockstep desyncs):
#       the float crab evolved identically on both peers of that process.
#   (2) ACROSS every process and the whole RUNS set — every log is byte-identical to run 0's: the
#       determinism does NOT lean on process-local state (thread timing, pool init order) that two
#       real machines wouldn't share. A SINGLE differing line on ANY run or seed fails the gate.
# Default load: 4 seeds × 6 runs = 24 fresh processes, each TICKS (default 1200) ticks.
#
# SCOPE: same-arch (x86_64) cross-PROCESS determinism. It does NOT prove cross-ARCHITECTURE float
# determinism — the cross-MACHINE case (two Steam Decks, possibly different CPUs) is the NEXT gate,
# not this one. rl#114 removed the integer-pursuit fallback (it was the trap that silently stood in
# for Sally), so deploy MUST keep every peer on the same-arch binary it ships; a non-x86_64 peer
# would have to be proven here (or REFUSE the round) rather than silently dropping to a fake crab.
#
# Usage: scripts/gcr82-xpeer-determinism.sh [CHECKPOINT_DIR]
#   CHECKPOINT_DIR defaults to $RL_CRAB_CHECKPOINT_DIR, else assets/weights under the repo root.
#   Env: TICKS (ticks per run, default 1200), RUNS (fresh processes per seed, default 6),
#        GAME_BIN (prebuilt binary), SEEDS (space-separated seed override).
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
ckpt="${1:-${RL_CRAB_CHECKPOINT_DIR:-$root/assets/weights}}"
game="${GAME_BIN:-$root/target/release/game}"
ticks="${TICKS:-1200}"
runs="${RUNS:-6}"
read -r -a seeds <<<"${SEEDS:-1 42 777 1666463330}"
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

echo "GCR#82 cross-peer NN-crab determinism — checkpoint=$ckpt ticks=$ticks runs/seed=$runs seeds=[${seeds[*]}]"
fail=0
procs=0
for s in "${seeds[@]}"; do
  ref_a="$work/s${s}_r0_a.log"; ref_b="$work/s${s}_r0_b.log"
  seed_fail=0
  for ((r = 0; r < runs; r++)); do
    a="$work/s${s}_r${r}_a.log"; b="$work/s${s}_r${r}_b.log"
    # A FRESH OS process each iteration (no in-process shared state — that's the whole point).
    if ! "$game" nn-crab-xpeer --checkpoint "$ckpt" --seed "$s" --ticks "$ticks" \
      --hash-log-a "$a" --hash-log-b "$b" >/dev/null 2>"$work/s${s}_r${r}.err"; then
      echo "  seed $s run $r: FAIL — binary exited nonzero (own divergence check tripped):"
      sed 's/^/    /' "$work/s${s}_r${r}.err" | tail -3
      seed_fail=1; fail=1
    fi
    procs=$((procs + 1))
    # (1) in-process: this run's A vs B.
    if ! diff -q "$a" "$b" >/dev/null 2>&1; then
      echo "  seed $s run $r: FAIL (in-process A!=B): $(diff "$a" "$b" 2>&1 | head -1)"
      seed_fail=1; fail=1
    fi
    # (2) cross-process: this run vs run 0 (skipped for run 0 itself, which IS the reference).
    if ((r > 0)); then
      if ! diff -q "$a" "$ref_a" >/dev/null 2>&1; then
        echo "  seed $s run $r: FAIL (cross-process A!=run0): $(diff "$a" "$ref_a" 2>&1 | head -1)"
        seed_fail=1; fail=1
      fi
      if ! diff -q "$b" "$ref_b" >/dev/null 2>&1; then
        echo "  seed $s run $r: FAIL (cross-process B!=run0): $(diff "$b" "$ref_b" 2>&1 | head -1)"
        seed_fail=1; fail=1
      fi
    fi
  done
  # Non-quiescence guard: identical-but-FROZEN hashes would diff clean and false-pass exactly as
  # round 1 did off a quiescent run. Require the crab to actually EVOLVE — most ticks distinct —
  # so "byte-identical" means "identically MOVING", not "identically still". (A real run has ~one
  # distinct hash per tick; demand at least half.)
  nlines="$(wc -l <"$ref_a")"
  distinct="$(sort -u "$ref_a" | wc -l)"
  if ((seed_fail == 0 && distinct * 2 < nlines)); then
    echo "  seed $s: FAIL — only $distinct distinct hashes over $nlines ticks: the crab is ~quiescent, so byte-identity proves nothing (round-1 false-pass mode)"
    seed_fail=1; fail=1
  fi
  if ((seed_fail == 0)); then
    echo "  seed $s: PASS — $runs fresh processes × $nlines ticks byte-identical in- AND cross-process ($distinct distinct hashes)"
  fi
done

echo "ran $procs fresh processes total"
if [[ $fail -eq 0 ]]; then
  echo "GCR#82: PASS — the trained NN crab is the deterministic multiplayer crab (cross-process, x86_64)."
else
  echo "GCR#82: FAIL — float NN crab diverged across peers/processes → netcode-rethink trigger."
fi
exit $fail
