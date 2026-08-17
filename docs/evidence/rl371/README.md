# rl#371 — residual walking jitter: render-side, in `lerp_pos`

Verdict from the position trace: the sim path is perfectly uniform (per-tick
step p2p 0.00 mm/s over 1700 ticks) — NOT true jitter. The residual was
introduced between sim and pixels, at the frame interpolation stage:
`lerp_pos` interpolated the ABSOLUTE i64 grid coordinate through f32. At a
14.4 km locale the coordinate is ~1.0e9 grid units, where f32 quantizes at
64 units = 0.64 mm — so every interpolated camera/avatar position snapped to
a 0.64 mm lattice, turning a uniform 2.25 mm 60 Hz step into alternating
2.56 / 1.92 mm steps (±14 %). rl#354 removed exactly this quantization from
the transform path (`rel_meters` before f32); the lerp had re-introduced it
upstream. Fix: lerp the per-tick DELTA in f32 (a few hundred units, exact)
and add it back on the i64 grid.

Every capture: `game fp-screenshot --seed 7` (spawn (−10310.8, −10003.7) m,
14.4 km — the rl#354 class), walking dead straight from frame 1 for 40 s
(1200 ticks / 2400 frames), offscreen lavapipe, position trace armed:

```
RL_POS_TRACE=<out.csv> game fp-screenshot --seed 7 --settle 2400 --walk-at 1 \
  --walk-straight --frame-hz 60 [--frame-jitter-ms 1.5] --out <shot.png>
```

The trace (`RL_POS_TRACE`, rl#371) records per-tick sim position on the exact
i64 grid and the per-frame render-resolved camera translation — live sessions
can arm the same env var. `--frame-hz/--frame-jitter-ms` model a real display
cadence beating against the 30 Hz sim; `--walk-straight` gives the
constant-velocity ground truth.

- `fixed60.png` — exact 60 Hz clock: sim uniform, camera-before alternating
  ±19 mm/s (13 % of walk speed) at 30 Hz — the visible shimmer — camera-after
  uniform to 0.1 mm/s (one grid unit).
- `jitter60.png` — ±1.5 ms frame-time jitter: before compounds to ±47 mm/s;
  after stays within 0.6 mm/s of ideal.
- `plot.py` — renders both from the trace CSVs.

Full-run stats (frames 600–2300, horizontal speed per step, mm/s):

| trace                    | median | p2p   |
|--------------------------|--------|-------|
| sim per-tick             | 140.77 | 0.00  |
| camera before            | 158.31 | 36.92 |
| camera after             | 140.78 | 0.20  |
| camera before, jittered  | 146.02 | 98.18 |
| camera after, jittered   | 140.78 | 1.38  |

Regression pin: `interpolated_eye_path_is_uniform_at_every_locale` — samples
the real `lerp_pos` at a 60 Hz cadence at the same locales as the rl#354 pin
(which walks raw tick positions and so missed this) and holds per-frame steps
to the same 2e-5 m bound. It fails on the absolute-form lerp, passes on the
delta form.
