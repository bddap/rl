# Training seed 351: complete, experiment still pending

Both arms ended at 40,000,512 ticks with exit 0, zero panic lines and one warm
resume each. Their immutable 40M checkpoints were used for every evaluation.
The pin, binary hashes and common recipe are in the parent protocol.

| Read | Control, floor −1 | Treatment, floor −0.7 |
|---|---:|---:|
| 30–40M episode-weighted reach | 0.0966 | 0.1727 |
| Episodes in that window | 13,700 | 14,372 |
| Mean-policy close-and-hold | 1/20 | 0/20 |
| Inverted at 20 s | 8/20 | 19/20 |
| Ever within 0.5 m | 1/20 | 3/20 |
| Median distance closed | 0.56 m | 0.05 m |

No keep/reject decision is made from this seed. Training seeds 352 and 353 remain
required, followed by the pre-registered pooled tests.

## Evidence and validation

- `*-buckets.txt`: unchanged `../../upright-shaping-ab/reach5m.awk` applied to each
  complete training log. The final update falls just above 40M and appears in a
  separate bucket; it is excluded from the registered 30–40M window.
- `*-provenance.txt`: training exit, launch count, final budget lines, marked tick
  count and full checkpoint-file hashes.
- `*-optimizer.txt`: arithmetic means over logged updates in fixed 5M buckets
  below 40M; also the fraction with at most two epochs. These are diagnostics.
- `traces/`: 40 mean-policy traces. All contain exactly the ordered video ticks
  64, 128, …, 1280 and `RENDER_CHECKPOINT` at 40,000,512. Checkpoint directory
  prefixes are normalized to arm names; no observation values are changed.
- `render-status-*.txt`: all 40 accepted renders exited 0. One earlier control
  evaluation-seed-353 driver exited 143 without an exit receipt. That attempt is
  retained separately; the same seed was rerun successfully, with no change to
  settings or scoring. It is not counted as a successful render.
- `scorer-output.txt`: the unchanged `../../upright-shaping-ab/analyze.py` output.
  Its hardcoded `k12` label aliases the floor −0.7 treatment in the scoring
  directory; **there is no upright shaping in either arm**. Published trace
  filenames use `floor07` instead. All 800 trace samples passed validation.

The 10-second frames of both evaluation-seed-351 videos were inspected: both
show Sally and the intended scene. Close-and-hold retains the registered
body-distance definition; it is not a count of target grabs.
