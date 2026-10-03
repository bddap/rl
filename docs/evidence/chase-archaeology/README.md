# Checkpoint archaeology — the last Sally that actually chased

Pre-terrain (< rl#281, 2026-07-16), pre-chase-eval (< 79d11ae, 2026-06-30) candidates,
re-verified 2026-08-06 by rendering each under the archived era binary
`rl-demo-3ddcf6f` (commit 3ddcf6f, 2026-07-01 — the newest binary that loads the
Jun-28 checkpoint format; pre-2026-06-29 source history did not survive the rewrite,
so archived binaries are the only era pins).

Checkpoints archived durably at `~/.local/state/rl-target/archive/sally377-jun28-chase/`
on the build host. `RL_TARGET_BALL=1`, flat-floor era, lavapipe headless render.

- `sally377-recheck.*` — Jun-28 brain.bin (it~500 after the 2x energy tax), paired normalizer.
- `prepenalty-paired.*` — Jun-28 brain-pre.bin (pre-energy-penalty), paired normalizer.
- `reach1.0-iter1046.*` — Jun-28 03:19 brain that hit reach 1.0 at iter 1046
  (from ckpt-backup-pre-job650); normalizer is the Jul-01 lineage one (unpaired — closest surviving).
- `era-jun28-*` — job-377 originals rendered ON 2026-06-28 (before/after the energy tax).

Bisect probes (rl#351, post-shortlist):

- `probe-a-8012a4f-endofbound.*` — Probe A end-of-bound render (2026-08-10): main
  `8012a4f8` cold canary, `--terrain flat --band-max-m 9`, 24 h / 19.2M ticks.
  Rendered with the pin's own `rl-demo` (built at `8012a4f8`), end-of-bound
  checkpoint, seed 351, canonical GCR tile (the demo's only ground, rl#293 — same
  locale family as chase-eval). Policy armed (mean|drive| 0.885) but no chase:
  aimless downhill drift, reach 0.00. Verdict on the issue: confounded (1 rl#343
  hard-fail), red-leaning.

- `probe-f-endbound.*`, `probe-f-vs-e-reach.png` — Probe F (2026-08-18): the
  mlp512x3 single-flip on the same `3ddcf6f` tree that went green as Probe E
  (arch the ONLY diff; patch archived at
  `~/.local/state/rl-target/probe-f-mlp512x3-backport.patch`). Cold, flat band
  ≤9 m, 24 h / 39.1M ticks / 12.7k iters. Train-side reach plateaued 0.26–0.35
  from ~4k iters (episode-weighted 500-iter buckets), never touching E's 0.6
  crossing (~2.5k iters) or 0.83–0.87 tail — see the overlay chart. End-of-bound
  render (patched `rl-demo-3ddcf6f-mlp512x3`, era flat floor): policy armed
  (mean|drive| 0.869) but no chase. Verdict on the issue: DEGRADED — convicts
  `56754c32` (mlp256 → mlp512x3) for cold-start learning at this rev.

- `probe-i-2760e243-endofbound.*`, `probe-i-2760e243-reach-curve.png` — Probe I
  (2026-08-25): bisect at the 07-17 collider-regen boundary, pin `2760e243`
  (post-collider-regen; pre-terrain-ground/band128/obs117/env-swap — flat arena +
  band ≤9 m are the rev's defaults, mlp512x3, REACH 0.2 native). Cold, 24 h +
  the ONE extension (upward trend, no cross) = 73.3M ticks / 23.5k iters.
  Train-side reach (0.2 m criterion, like-for-like vs A's trendless 0.01–0.05):
  climbed 0.02 → flat ~0.045 → 0.07 → late surge to 0.14–0.15, still rising at
  the bound — see the curve. End-of-bound render (the pin's own `rl-demo`, flat
  native = in-distribution, seed 351): **she chases** — closes from a distant
  spawn and stays at the ball for the rest of the clip (mean|drive| 0.890).
  Verdict on the issue: NOT-FLOOR (D/F-like slow learning) — acquits everything
  ≤07-17 incl. the collider regen; with Probe C's floor on `4cdf934c` (07-27)
  the breaker narrows to `2760e243..4cdf934c` (07-18..07-27).

- `integrity-repro/` — flat short-band integrity reproduction (rl#351, 2026-10-03).
  The last coherent checkpoint of the `a9372f7` hull baseline (34,240,512 ticks,
  `brain.bin` sha256 `bc44ff79…7a18`) rolled frozen by `rl-train repro` (`e49bd48`):
  `--terrain flat --band-max-m 9 --envs 2 --horizon 512 --log-std-floor-end -1
  --seed 3510 --workers 12 --ticks 1048576`, rl#343 thresholds unchanged.
  Stopped at ≥6,259,163 worker-ticks (49.7 % of the registered budget) with **4
  rl#343 trips** (`trips.txt`, `per-worker-ticks.txt`), one of them the right claw
  wrist at 110.7 m/s. Each `w*-e*-t*.bin` is the failing env's plant one tick
  before a kick onset (`w4-e1-t19561.bin` is the 427.9 m/s right-wrist trip from
  the first hunt); `game sally-replay --state <bin>…` reproduces every kick exactly
  (self-check dev 0.000) and prints `sally-replay.txt`. In the 192 pre-trip
  flight-recorder ticks (48 per trip), 21–36 of 38 joints sit at or past a limit
  (`joints-at-limit-histogram.txt`: count, joints). The kick survives zeroed
  drives in 3/7, grows with more PGS iterations (to 7e13 m/s), ignores capsule
  radius and single claw-limit removal, and vanishes only under solver
  (8,4,4)×4 / (32,8,8)×8 or a whole-body shape swap.
