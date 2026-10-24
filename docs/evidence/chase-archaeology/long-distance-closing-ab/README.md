# Long-distance closing A/B: pre-registration

Status: registered; all six runs launched cold on 2026-10-24 ([launch telemetry](launch.txt)); 40M reads outstanding. This is the long-distance closing increment of [#351](https://github.com/bddap/rl/issues/351).

**Plant bound first.** At main `1ab752d`, on flat ground, 20 s from a 300-tick settle:

| Gait | Planar travel in 20 s |
|---|---:|
| `scripted_flail_gait_pace` (every drive ±1, 10-tick period; restored build-locally, it was deleted from main) | 1.51 m |
| best of 96 random open-loop gaits (per-joint sine or square, periods 6–96 ticks, random phase, sign and bias) | 16.55 m |
| median of those 96 | 1.48 m |

Four of the 96 open-loop gaits cover more than 9 m. The plant can cross the band, so no plant box is appended. The registered flail is one fixed gait, and it falls short. The search only gives a lower bound on what the plant can do. It does not check orientation, so a gait that slides while inverted would count. Logs: [pace.txt](pace.txt) and [gait-search.txt](gait-search.txt); the build-local instrument is [pace-instrument.patch](pace-instrument.patch), with the deleted `scripted_flail_gait_pace` extended to 20 s on both grounds and the gait search beside it. On the canonical GCR ground the same flail slides 66 m downhill, so only the flat reading bounds the gait.

**The one change.** Treatment draws the band distance uniformly in metres over [1.5, 9] m (`--band-draw uniform-distance`). Control keeps the log-uniform draw (`--band-draw log-uniform`, the default). The 10% close disc, the reward, the reach criterion and everything else are the same in both arms.

| Training spawn share (close disc included) | < 2.5 m | 2.5–4.5 m | ≥ 4.5 m |
|---|---:|---:|---:|
| control, log-uniform | 35.7% | 29.5% | 34.8% |
| treatment, uniform | 22.0% | 24.0% | 54.0% |

**Why this change.** In the sampled-vs-mean sweep, every approach to within 0.5 m but one started within 2.5 m. From 4.5 m or farther, the median closed in 20 s was 0.66 m with the mean action and 1.06 m sampled, which is no better than a random open-loop gait. Training reach therefore looks like credit from short spawns: a lunge and grab earns +50 there without any locomotion. A uniform draw cuts the share of short spawns and puts most episodes where only travel pays.

The other two options were rejected:
- The travel incentive (24 per metre closed) is already dense and the same at every distance, and it beats the grab bonus on a far approach. Raising it would mostly shrink the effort tax and the grab bonus relative to it, and the effort tax is already a minor term.
- Changing the reach criterion would change the reach ruler that the keep gate reads.

Evaluation stays log-uniform (`rl-demo` render rollouts), so both arms are scored on the same distribution as earlier sweeps.

**Runs.** Six cold runs, one trainer binary built at `1ab752d`, with `rl-demo` from the same pin and plant `sally.glb` `119d5a4f…`:

`rl-train learn --terrain flat --band-max-m 9 --band-draw {log-uniform|uniform-distance} --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --seed {351,352,353} --log-std-floor-end -1 --ticks 40000000 --read-mark 40000000`

All six run concurrently under the same runner and share the CPU. The 6×2×512 rollout is the recipe, so throughput is not traded against it. Warm resumes are recorded, and settings never change on a resume. An rl#343 hard-fail stops that arm with no silent restart and censors the experiment, which blocks a keep.

**Reads, per run.**
- 5M reach buckets using `../upright-shaping-ab/reach5m.awk`, unchanged.
- The rl#343 hard-fail count.
- The 40M mean sweep on `ckpt/marks/40000000`, eval seeds 351–370: `render-video --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed S` on NVIDIA Vulkan, scored by `../upright-shaping-ab/analyze.py` unchanged.
- Closing by spawn distance with `../sampled-vs-mean-sweep/by-spawn.py`.

**Decision, fixed now.** Keep uniform-distance only if both gates pass and no arm hard-failed:
1. Pooled mean-policy close-and-hold over 60 cases per arm is above control's, by a one-sided Fisher exact test at p < 0.05.
2. Pooled episode-weighted 30–40M training reach is not below control's. The test is the replicate's one-sided two-proportion z test, H1 control > treatment, where p < 0.05 counts as harm.

Training reach is stochastic and is not evidence of chase. If the change is kept, it becomes the flat baseline recipe. If it is rejected, `--band-draw` is removed from main and the reason is recorded.

Full SHA-256 digests (release builds of `-p rl-train -p rl-demo` at `1ab752d`, exit 0):

```
ba75f0450a046ee7ed76dd5815baeb22489900e02d11764d9f9f4366e99b9ec8  rl-train
bb342275aff9148f73ccfe13df9c16896085296a29f04465b1cc32b0ce5d3bfc  rl-demo
119d5a4f9986f3739e01677910f9880ddc48a47b0a12a24236c1781426a7bf0b  sally.glb
```

No plant, physics or body source changed between the entropy-floor pin `24b3c49` and `1ab752d`.
