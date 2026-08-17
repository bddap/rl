# rl#377 — parked aircraft wiggles: the craft could never sleep

Verdict: the wiggle is ROTATIONAL chatter of a grounded craft that is never
allowed to fall asleep. `apply_vehicle_forces` writes `ExternalForce` every
tick, bevy_rapier force-wakes a body on every `Changed<ExternalForce>`
(resetting rapier's sleep timer), and the passive terms — drag, grip, lift,
angular drag — re-derive a not-quite-equal force each tick from the contact
solver's own rest-noise velocities. The loop holds a parked craft awake
forever, and the awake contact solve rings its orientation: at far-coordinate
locales (f32 position ULP ~1 mm at the ~13 km seed-7 spot) a headless sweep
measured up to ~1 rad/s of sustained chatter, while position pins at the ULP
and reads perfectly still — which is why the rl#371/#376 position traces
never saw it.

Fix: with the COMMAND neutral (no thrust, no control torque) and both
velocities inside rapier's own sleep thresholds (0.4 m/s, 0.5 rad/s), the
passive terms are rest noise by definition — the force snaps to exactly zero
and the no-op write is skipped, so sleep engages and rest is bit-exact rest.
Dissipative terms can never sustain motion, so dropping them only inside the
sleep band changes no trajectory; any commanded input writes (and wakes) as
before. Regression pin: `parked_craft_falls_asleep` (asserts the rapier body
is ASLEEP and the transform bit-frozen; fails pre-fix).

Not the rl#376 mechanism: that was a measurement-clock artifact in the wind's
airspeed read (tick-average vs physics-step), fixed separately — the causes
do not join.

Capture (offscreen lavapipe, plane boarded at frame 60, parked — zero flight
input — through the shot):

```
RL_POS_TRACE=<out.csv> game fp-screenshot --seed 7 --settle 600 \
  --pilot-toggle-at 60 --pilot-park --frame-hz 60 --width 640 --height 360 \
  --nn-crab-checkpoint <ckpt> --out <shot.png>
```

Trace record added for this issue (versionless addition to the rl#371 CSV):
`Q` (craft orientation entering the pose window) — the wiggle is invisible to
the position-only `V` record.

- `parked-rotation-before-after.png` — per-tick orientation delta, ticks
  150–403 (parked, settled): before chatters up to 0.044°/tick on 38/254
  ticks and is still moving at the last recorded tick; after is 0.0000° on
  every tick (asleep).
- `plot.py` — renders it from the two trace CSVs.
- `crab-world/tests/park_probe.rs` (ignored, informational) — the headless
  sweep: parked-craft motion, sleep state, and peak angular velocity across
  terrain spots and residual throttles; the far-locale spots chattered at
  0.1–1.1 rad/s pre-fix and sleep bit-exact post-fix. Steep slopes (μ 0.5 <
  tan 49°) slide and residual throttle taxis — both physically consistent,
  neither is the wiggle.
