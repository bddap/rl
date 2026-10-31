# Locomotion-discovery A/B: pre-registration

Status: pre-registered; runs not yet launched. This is the locomotion-discovery increment of [#351](https://github.com/bddap/rl/issues/351).

**Reading.** On the friction-fixed flat plant the registered flail travels 1.51 m in 20 s and random periodic open-loop gaits a median 1.48 m, but 4 of 96 cross 9 m ([gait search](../long-distance-closing-ab/gait-search.txt)). No trained policy has closed to within 0.5 m from 2.5 m or farther: 0/71 in the [last A/B](../long-distance-closing-ab/verdict.md). Spawn distance, entropy floor and upright shaping did not change that. The policy is a feed-forward map from the current observation to joint drives, and nothing in the observation oscillates, so a rhythm has to emerge from body state alone or from per-tick exploration noise.

**The one change.** Treatment observes a gait clock: two channels, sin and cos of 2π·t/32 with t the world's sense tick (32 ticks = 0.5 s at 64 Hz). Control trains with the same two channels held at zero (`--gait-clock off`). A channel that never varied in training normalizes to zero, so a control checkpoint ignores the clock wherever it runs and renders need no matching flag. The clock free-runs across episodes, so each episode starts at an arbitrary phase. Everything else is identical in both arms: the reward, spawns, band, network and the 40M budget.

**Why this change, why 32 ticks.** Five of the 96 open-loop gaits travel more than 6 m in 20 s; their periods are 32, 24, 16, 16 and 6 ticks. A 32-tick clock carries period 32 and, through the MLP's nonlinearity, its harmonics at 16, 10.7 and 8 ticks, so three of those five rhythms are expressible as a function of phase alone. The two alternatives the box names stay in reserve: seeding exploration with the gait search's travelling gaits changes the exploration process and the action path, and rl#395's σ-learnability fix changes the optimiser; the clock changes only what the policy can see.

**Layout.** The clock adds two observation channels in both arms, so the channel-layout digest moves and checkpoints trained before this change no longer arm on main. A reject removes the channels and restores the previous layout.

**Runs.** Six cold runs, one trainer binary built from main at the option's landing, with `rl-demo` from the same pin and the shipped plant `sally.glb`:

`rl-train learn --terrain flat --band-max-m 9 --gait-clock {off|on} --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --seed {351,352,353} --log-std-floor-end -1 --ticks 40000000 --read-mark 40000000`

All six run under the same runner. Warm resumes are recorded, and settings never change on a resume. An rl#343 hard-fail stops that arm with no silent restart and censors the experiment, which blocks a keep.

**Reads, per run.**
- 5M reach buckets using `../upright-shaping-ab/reach5m.awk`, unchanged.
- The rl#343 hard-fail count.
- The 40M mean sweep on `ckpt/marks/40000000`, eval seeds 351–370: `render-video --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed S`, scored by `../upright-shaping-ab/analyze.py` unchanged.
- Closing by spawn distance with `../sampled-vs-mean-sweep/by-spawn.py`.

**Decision, fixed now.** Keep the gait clock only if both gates pass and no arm hard-failed:
1. Pooled mean-policy close-and-hold over 60 cases per arm is above control's, by a one-sided Fisher exact test at p < 0.05.
2. Pooled episode-weighted 30–40M training reach is not below control's: the one-sided two-proportion z test of H1 control > treatment at p < 0.05 counts as harm.

A keep makes the clock the flat baseline recipe and removes `--gait-clock`. A reject removes the clock channels and `--gait-clock` from main and records why.
