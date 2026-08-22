# rl#401 — sally_track never records vehicle ride motion; boarding "double-teleport"

Two findings, one fix.

## The "double-teleport" was two round restarts, not a tracking defect

The TV session's 11.1 km + 16.2 km jumps at 01:59:00.674/01:59:01.312Z (0.64 s
apart, 11 s after `OnFoot -> Ship`) line up exactly with two
`rl#305 restart spawn` log lines in the same sink — same match seed, successive
rng frames, each with a matching `crab skin re-paired after reset`. The sim only
resets on the edge-triggered RESTART button (`Sim::step`), so this was two
restart presses; the crab teleporting to each fresh round origin is the designed
restart behavior (`restart_crabs_to_spawns`). Velocity stayed ~0 through both
jumps — transform re-placement, not physics. Not related to boarding mechanics,
and nothing here to fix. (rl#398's command-map fix is unrelated: that was
discovery-set replication.)

## The real defect: ride motion had NO recorder

`sally_track` samples crab carapaces only. The piloted walker merely MIRRORS the
craft's pose inside the fixed-point sim (`Sim::step`'s pilot feed), so the craft
rigidbody in the one bevy world is the only physics source of ride motion — and
nothing sampled it. Any consumer reading the stream saw a parked world through
every flight; the TV session's "idle creep" tracked body was simply Sally
walking at crab pace while the unrecorded ship flew away (the `crab_slot`
prey-distance lines carried the only trace of the flight).

Fix: craft samples in the same ~10 Hz batch, same emit path — crab samples keep
`"c":<env>`, craft samples carry `"veh":<pilot>,"kind":<wire byte>` (1 plane,
2 ship). Bothouse-side `telemetry/sally-track-plot.py` groups per body key.

## Verification (offscreen lavapipe, this build)

```
game fp-screenshot --seed 7 --settle 1020 --pilot-toggle-at 120 \
  --nn-crab-checkpoint <ckpt> --width 320 --height 200   # + OTLP env
```

Scripted board at frame 120, full forward drive; sink `otlp-verify2659.jsonl`.

- `plane-track.png` — the craft stream: spawns AT the walker's spot
  (`a`=0.05 m, v≈0), takeoff, steady 8.3 m/s climb to 46 m above ground,
  ~200 m straight x-z path. First craft sample p=[9368.85,-168.85,-7772.74];
  last p=[9471.97,-26.97,-7603.47].
- `crab-track.png` — Sally's stream, continuous through the boarding edge.
- No teleport in either stream: max inter-sample step 0.78 m (craft),
  0.31 m (crab) across the whole session.
