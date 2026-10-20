# Training seed 352: complete, experiment still pending

Both arms ended at 40,000,512 ticks with exit 0 and zero panic lines. Control had
three warm resumes and treatment four. Their immutable 40M checkpoints were
used for every evaluation. The pin, binary hashes and common recipe are in the
parent protocol.

| Read | Control, floor −1 | Treatment, floor −0.7 |
|---|---:|---:|
| 30–40M episode-weighted reach | 0.0781 | 0.1008 |
| Episodes in that window | 13,506 | 13,722 |
| Mean-policy close-and-hold | 2/20 | 0/20 |
| Inverted at 20 s | 2/20 | 11/20 |
| Ever within 0.5 m | 2/20 | 4/20 |
| Median distance closed | 0.31 m | 0.35 m |

No keep/reject decision is made from this seed. Training seed 353 remains
required, followed by the pre-registered pooled tests.

## Evidence and validation

- `*-buckets.txt`: unchanged `../../upright-shaping-ab/reach5m.awk` applied to each
  complete training log. The final update falls just above 40M and appears in a
  separate bucket; it is excluded from the registered 30–40M window.
- `*-provenance.txt`: training exit, launch count, budget lines, marked tick
  count and full checkpoint-file hashes. A DONE line describes its uninterrupted
  segment; it does not measure the whole run across warm resumes.
- `*-optimizer.txt`: arithmetic means over logged updates in fixed 5M buckets
  below 40M; also the fraction with at most two epochs. These are diagnostics.
- `traces/`: 40 mean-policy traces. All contain exactly the ordered video ticks
  64, 128, …, 1280 and `RENDER_CHECKPOINT` at 40,000,512. Checkpoint directory
  prefixes are normalized to arm names; no observation values are changed.
- `render-status-*.txt`: all 40 accepted renders exited 0. One earlier treatment
  evaluation-seed-354 driver exited 143 without an exit receipt. That attempt is
  retained separately; the same seed was rerun successfully, with no change to
  settings or scoring. It is not counted as a successful render.
- `scorer-output.txt`: the unchanged `../../upright-shaping-ab/analyze.py` output.
  Its hardcoded `k12` label aliases the floor −0.7 treatment in the scoring
  directory; **there is no upright shaping in either arm**. Published trace
  filenames use `floor07` instead. All 800 trace samples passed validation.

The 10-second frames of both evaluation-seed-351 videos were inspected: both
show Sally and the intended scene. Close-and-hold retains the registered
body-distance definition; it is not a count of target grabs.
