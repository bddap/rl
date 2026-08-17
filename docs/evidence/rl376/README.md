# rl#376 — irregular plane-wind wooshes: the 64:30 staircase in `speed_mps`

Verdict from the wind-airspeed trace: the craft's true velocity is uniform —
NOT a sim/physics bug. The surge was introduced at the measurement:
`PoseWindow::speed_mps` (the one airspeed signal behind the wind synth and the
craft engine layer, rl#356/357) divided per-tick displacement by tick TIME,
while a craft covers distance per physics STEP — and the 64:30 cadence bunches
2 vs 3 steps per tick. So on every 3-step tick the wind heard 3/2 the true
per-step pace for one tick (~2 frames at 60 Hz), at the staircase's irregular
7/8-tick spacing ≈ 4×/s: the reported "speed bumping up for a couple frames
every few frames". This is exactly the surge the window's `sample()` was built
to correct for rendered motion (rl#264); the speed read had skipped the
correction.

The perf-graph correlation falls out of the same clock: a 3-step tick's frame
does 50% more physics work. In the capture below the sim portion measures
71.1 ms (2-step) vs 88.7 ms (3-step) per frame — on the TV kit those are the
frames that tip over the >33.4 ms red line, the same ticks the wind surged.

Fix: `speed_mps` divides by the staircase's step count (`cumulative_steps`,
the same clock `sample()` walks) — uniform by construction, tick gaps exact.
Regression pin: `speed_mps_is_uniform_through_the_staircase`.

Capture (offscreen lavapipe, plane boarded at frame 60, full throttle):

```
RL_POS_TRACE=<out.csv> game fp-screenshot --seed 7 --settle 200 \
  --pilot-toggle-at 60 --width 160 --height 90 --nn-crab-checkpoint <ckpt> \
  --out <shot.png>
```

(The before capture ran `--settle 250` and was killed at ~tick 200 by a
session timeout — its CSV tail is torn; `plot.py` skips torn lines. The
cruise window both captures share is what the figure plots.)

Trace records added for this issue (versionless additions to the rl#371 CSV):
`V` (craft pose entering the window), `W` (airspeed the wind synth is driven
with — a shared observer system, since the offscreen app has no audio
systems), `S` (measured per-frame sim cost + pumped ticks).

- `wind-speed-before-after.png` — before: 8.010 m/s on 2-step ticks jumping
  to 12.015 (+50%, exactly 3/2) on every 3-step tick; after: flat
  8.544 m/s (= the same displacement measured over steps) through the same
  staircase. Lower panel: measured per-frame sim cost, 3-step ticks banded.
- `plot.py` — renders it from the two trace CSVs.

Cruise-window numbers (ticks 120–200):

| trace  | 2-step ticks | 3-step ticks |
|--------|--------------|--------------|
| before | 8.010 m/s    | 12.015 m/s   |
| after  | 8.544 m/s    | 8.544 m/s    |
