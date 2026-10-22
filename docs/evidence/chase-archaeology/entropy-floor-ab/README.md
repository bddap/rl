# Entropy-floor A/B: pre-registration

Status: complete — **reject the constant −0.7 floor; retain −1**.
[Pooled verdict](verdict.md): training reach 0.1260 versus control 0.0724 passes
the no-harm gate, but mean-policy close-and-hold 0/60 versus 4/60 fails the
one-sided gain test (Fisher p = 1). All six 40M runs and 120 mean-policy renders
completed. Per-training-seed evidence: [351](s351/README.md),
[352](s352/README.md), [353](s353/README.md).

The protocol below was registered before launch. This is the architecture/optimizer
increment of [#351](https://github.com/bddap/rl/issues/351).

## One change

Control: `--log-std-floor-end -1`. Treatment: `--log-std-floor-end -0.7`.
The existing schedule starts at -0.7 and reaches its endpoint after 5M ticks;
treatment therefore holds the initial floor throughout. No source change is needed.
All other optimizer settings remain at the pin's defaults, including learning rate
0.0003, target KL 0.03 and four epochs per update. Shaping is absent.

The reading in `../inversion-sweep/optim-4m.txt` shows entropy saturating at 0.419,
with KL 0.077–0.091 and updates commonly cut short. Holding the initial floor
raises minimum Gaussian standard deviation from exp(-1) = 0.368 to exp(-0.7) =
0.497. This tests whether early loss of exploration contributes to the failure to
close. A larger variance also reduces the mean-shift contribution to Gaussian KL
for an equal mean displacement; this is a hypothesis, not an attribution of the
observed KL. The tradeoff is noisier training actions, so training reach and the
deterministic mean policy must both be measured. No floor sweep or interim selection.

## Matched runs

Build `rl-train` and `rl-demo` from main `24b3c4928da635ef13ba7b2589fefca6f1ba98f0`.
Use that single trainer artifact for all six cold runs, and that demo for all reads.
Record full binary and plant SHA-256 digests before launch. Plant: the pinned
`sally.glb`, SHA-256 beginning `119d5a4f`; shipped solver (2,12,3)×2.
This pin includes the warm-up change; none of the earlier shaping runs substitutes
for a new control.

For each training seed 351, 352, 353, run both endpoints:

```
rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 \
  --workers 6 --envs 2 --horizon 512 --seed SEED \
  --log-std-floor-end ENDPOINT --ticks 40000000 --read-mark 40000000 \
  --checkpoint-dir RUN/ckpt
```

Run each matched pair concurrently, seeds in ascending order, using the same
runner and resources for both arms. Each starts with no checkpoint and cold
optimizer moments. Record interruptions and resumes. Do not change the rollout
shape for throughput. Stop an arm on an integrity hard-fail; no silent restart.
A hard-fail censors the experiment and prevents a keep decision.

## Reads and fixed decision

- Per run, report episode-weighted 5M reach buckets using
  `../upright-shaping-ab/reach5m.awk`, plus rl#343 hard-fail counts.
- At 40M, render the mean policy from `ckpt/marks/40000000/`, seeds 351–370,
  `--rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360`,
  using NVIDIA Vulkan and the pinned plant. Require a successful render and the
  expected `RENDER_CHECKPOINT` for every read. Preserve all 120 traces.
- Score with `../upright-shaping-ab/analyze.py` unchanged: inverted is upright < 0
  at 20 s; close-and-hold is body-to-ball distance ≤0.5 m at every sample from
  some t ≤18 s through 20 s. Report every training seed separately and pool 60
  evaluation cases per arm. These are the epic's pooled descriptive tests;
  shared training policies mean evaluation cases are not independent training
  replications.
- Pool episode-weighted training reach over 30–40M and training seeds 351–353.
  Use the same one-sided two-proportion z-test as the shaping replicate:
  H1 control > treatment. Significant harm is p < 0.05; absence of significant
  harm is the stipulated 'not below' gate, not proof of equivalence. Log-rounded
  reach rates give approximate reached-episode totals, as in the prior ruler.
- For pooled close-and-hold, use a one-sided Fisher exact test, H1 treatment >
  control, p < 0.05. This handles the small success counts and zero cells.
  Both gates must pass: no significant reach harm AND significant closing gain.
  Report counts, rates and p-values, including the per-training-seed breakdown.
- Keep -0.7 only if both gates pass and there are no integrity failures. Otherwise
  retain the existing configuration and record why. No production default changes
  before that decision. If kept, apply the floor change and run its covering suites.

Also report KL, epochs and entropy in fixed 5M buckets as mechanism diagnostics;
these do not override either decision gate.

## First pair launched

Seed 351 control and treatment both launched cold. [Launch telemetry](launch.txt)
records their identical first update, cold optimizer moments, configured schedules,
and simulation identity `ac3ad4a12cc60fdc`. Both saved initial checkpoints and
advanced beyond 190,000 ticks with no panic recorded at this launch check. This is
launch validation only; the 40M reads and seeds 352–353 remain outstanding.

Source pin is `24b3c49`; the build worktree also contains the documentation-only
pre-registration commit `c14e253`, with no source differences. Release builds both
exited 0: trainer in 3m 14s, evaluator as recorded in `builds.txt`. The binaries are
preserved for every subsequent run and read. Full SHA-256 digests:

```
7281ec601676b1ec288d84ac54e43522bae1453eb773c16447ea8aaa9c90d881  rl-train
70746b5ec9bf34de11b1dc83d774709b3583b834d9122e153e7575f4df59bfe9  rl-demo
119d5a4f9986f3739e01677910f9880ddc48a47b0a12a24236c1781426a7bf0b  sally.glb
```
