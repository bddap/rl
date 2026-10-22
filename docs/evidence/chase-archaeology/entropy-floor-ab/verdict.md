# Verdict: reject the constant −0.7 log-standard-deviation floor

Retain the existing −1 endpoint. The higher floor improves sampled training
reach but fails the required deterministic mean-policy closing gate: zero holds
across all three treatment policies. Both registered gates were required. The
experiment changed only a command-line option, so rejection requires no source
revert or production default change.

| Training seed | Control reach, 30–40M | Treatment reach, 30–40M | Control holds | Treatment holds | Control inverted | Treatment inverted |
|---|---:|---:|---:|---:|---:|---:|
| [351](s351/README.md) | 0.0966 | 0.1727 | 1/20 | 0/20 | 8/20 | 19/20 |
| [352](s352/README.md) | 0.0781 | 0.1008 | 2/20 | 0/20 | 2/20 | 11/20 |
| [353](s353/README.md) | 0.0419 | 0.1021 | 1/20 | 0/20 | 10/20 | 10/20 |
| Pooled | 0.0724 | 0.1260 | 4/60 | 0/60 | 20/60 | 40/60 |

## Registered decision

Training reach: 40,581 control episodes and 41,795 treatment episodes in the
registered 30–40M window. The unchanged replicate `pooled.py` gives z = −25.647,
p = 1 for H1 control > treatment. No significant reach harm: **pass**. This is
not an equivalence claim. Counts are estimated from log-rounded reach rates;
see [the per-seed totals and test](pooled-reach.txt). `k12` in that output is
only the analyzer's treatment alias, with no shaping applied.

Closing: one-sided Fisher exact test, H1 treatment > control, is p = 1 for
0/60 versus 4/60. Significant closing gain: **fail**. Conditioning on four total
holds, the tail starting at zero contains every feasible allocation; its exact
numerator and denominator are both 8,214,570. See [the calculation](pooled-hold.txt).
The keep conjunction therefore fails. Inversion is a diagnostic, not an extra
post-hoc selection rule.

All six runs started cold on source pin
`24b3c4928da635ef13ba7b2589fefca6f1ba98f0`, using the same trainer and evaluator
artifacts, plant, flat ≤9 m terrain, mlp512x3 and 6×2×512 rollout. Each ended at
40,000,512 ticks with exit 0 and zero integrity panic lines. Warm resumes were
control/treatment 1/1 for seed 351, 3/4 for seed 352, and 0/0 for seed 353; no
settings changed on resume. All 120 accepted renders exited 0 and read the
registered mark, with 2,400 ordered samples scored by the unchanged analyzer.

Two earlier render-driver attempts (control 351/eval 353 and treatment 352/eval
354) ended with status 143 without a successful exit receipt; their causes are
unresolved. Two seed-353/eval-351 startup attempts exited 127 for absent runtime
libraries. These four attempts were excluded and preserved separately. Identical
seeds were rerun successfully. Exact runtime dependencies were restored for the
last pair; the executable hashes stayed equal to the pre-registration hashes.

## Interpretation and scope

The floor intervention did not solve the observed mean-policy failure. More
training reach did not transfer to more body-distance holds. The logged entropy
and optimizer diagnostics are included in each seed directory; they do not
establish a causal explanation for failed closing or for the inversion increase.
There are three trained policies per arm, not 60 independent training replicates;
the pooled tests follow the registered descriptive ruler. A body-distance hold
also differs from a target grab. No claim is made about full-distance terrain,
slopes, or on-device chasing.

This increment contains documentation and measured evidence only. The existing
release builds exited 0; their real tails are in [builds.txt](builds.txt). The
repository's evidence mapping requires no additional build/test commands for
these paths. The review directive exempts artifacts and runs with no source
change; no code review loop or speculative source fix was introduced.
