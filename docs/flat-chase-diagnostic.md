# Matched flat chase diagnostic

A short-band training reach rate does not establish mean-policy chase. Training
samples correlated exploration; the normal demo uses action means and the full
target band. Changing only the ground in a local demo patch leaves the target
distribution unmatched.

Use a checkpoint trained from scratch on the current hull body. Keep the trainer
and renderer pinned to the same source and record their binary hashes, checkpoint
hashes, body and simulation identities, sensor layout, architecture, tick count,
and integrity failures. Diagnostic flat checkpoints remain unsuitable for release;
the diagnostic options do not override body, simulation, or friction checks.

Train a finite flat baseline with the existing sensors and architecture:

```sh
rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 \
  --workers 6 --envs 2 --horizon 512 --seed 351 \
  --log-std-floor-end -1 --ticks 40000000 --checkpoint-dir "$CKPT"
```

Render both policies from the same complete checkpoint after training stops:

```sh
rl-demo render-video mean.mp4 --checkpoint-dir "$CKPT" \
  --rollout-terrain flat --band-max-m 9 --seed 351 \
  --seconds 20 --width 640 --height 360
rl-demo render-video stochastic.mp4 --checkpoint-dir "$CKPT" \
  --rollout-terrain flat --band-max-m 9 --seed 351 \
  --exploration-log-std-floor -1 --seconds 20 --width 640 --height 360
```

Repeat the paired renders for seeds 352 and 353. At 40M ticks the example run has
finished its 5M-tick anneal, so its floor is -1. For earlier checkpoints, pass the
floor actually logged by the learner rather than assuming the endpoint.

The stochastic render uses the trainer's Gaussian head and stationary OU noise.
Its independent RNG leaves the initial target draw unchanged between policy modes.
Both modes draw from the same close-disc/short-band mixture and re-seed the target
when a claw touches it. Neither mode disables sensors. The render is continuous,
not a replay of training's settle/reset/terminal episodes or its evolving
normalizer, so this is an action-selection comparison, not an episode-success-rate
equivalence claim.

`RENDER_ROLLOUT` prints the requested distribution. `ROLLOUT_TRACE video_tick=...`
reports position, target, horizontal body distance, 3D claw-tip distance, and upright
orientation once per simulated second, within the captured interval only. Rendering
includes 60 unrecorded startup ticks. Target changes mark new pursuit segments;
do not subtract distances across them. Review the clips as well as the trace and
`DRIVE_STATS`; count rescue/integrity reports separately from CCD clamp candidates.
A clamp candidate alone is not an integrity failure.

Pre-register episode-weighted training reach in 5M-tick buckets and the paired
20-second renders. A mean-policy closing-and-holding sequence is the gate for
widening the task. If stochastic motion succeeds while mean motion stalls, isolate
action selection next; if both fail, retain the flat short band and define the
next finite learning diagnostic. Training reach or a successful render command is
not a successful chase verdict. Do not change physics based on unmatched renders.
