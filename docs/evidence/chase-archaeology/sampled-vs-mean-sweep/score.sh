#!/usr/bin/env bash
# Runs ../upright-shaping-ab/analyze.py unchanged per training seed and mode. Its fixed
# labels are ctrl and k12; k12 stages the floor −0.7 arm (no shaping in either arm).
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d)
for t in 351 352 353; do for m in mean sampled; do
  d=$T/$m-s$t; mkdir -p "$d/traces"
  for s in $(seq 351 370); do
    if [ "$m" = mean ]; then
      cp "../entropy-floor-ab/s$t/traces/ctrl-$s.txt" "$d/traces/ctrl-$s.txt"
      cp "../entropy-floor-ab/s$t/traces/floor07-$s.txt" "$d/traces/k12-$s.txt"
    else
      cp "traces/ctrl-s$t-sampled-$s.txt" "$d/traces/ctrl-$s.txt"
      cp "traces/floor07-s$t-sampled-$s.txt" "$d/traces/k12-$s.txt"
    fi
  done
  (cd "$d" && python3 "$OLDPWD/../upright-shaping-ab/analyze.py") > "scorer-$m-s$t.txt"
done; done
rm -r "$T"
python3 paired.py > paired.txt
python3 by-spawn.py > by-spawn.txt
