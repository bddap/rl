#!/usr/bin/env bash
# Runs ../upright-shaping-ab/analyze.py unchanged per training seed. Its fixed labels are
# ctrl and k12; k12 stages the gait-clock arm.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d)
for t in 351 352 353; do
  d=$T/s$t; mkdir -p "$d/traces"
  for s in $(seq 351 370); do
    cp "traces/off-s$t-$s.txt" "$d/traces/ctrl-$s.txt"
    cp "traces/on-s$t-$s.txt" "$d/traces/k12-$s.txt"
  done
  (cd "$d" && python3 "$OLDPWD/../upright-shaping-ab/analyze.py") > "scorer-s$t.txt"
done
rm -r "$T"
