# Training seed 353: complete

Both arms ended at 40,000,512 ticks with exit 0, no warm resumes and zero panic
lines. Every evaluation used the registered immutable 40M checkpoint. The pin,
binary hashes and common recipe are in the parent protocol.

| Read | Control, floor −1 | Treatment, floor −0.7 |
|---|---:|---:|
| 30–40M episode-weighted reach | 0.0419 | 0.1021 |
| Episodes in that window | 13,375 | 13,701 |
| Mean-policy close-and-hold | 1/20 | 0/20 |
| Inverted at 20 s | 10/20 | 10/20 |
| Ever within 0.5 m | 3/20 | 3/20 |
| Median distance closed | −0.09 m | 0.47 m |

## Evidence and validation

- `*-buckets.txt`: unchanged `../../upright-shaping-ab/reach5m.awk` on each full
  training log. The final update just above 40M is excluded from the registered
  30–40M window.
- `*-provenance.txt`: exit status, launch count, budget lines, marked tick count
  and hashes for the entire checkpoint envelope.
- `*-optimizer.txt`: arithmetic means over logged updates in fixed 5M buckets
  below 40M, plus the fraction with at most two epochs; diagnostic only.
- `traces/`: all 40 traces contain exactly video ticks 64, 128, …, 1280 and
  `RENDER_CHECKPOINT` at 40,000,512. Only checkpoint directory prefixes were
  normalized to arm names; observations are unchanged.
- `render-status-*.txt`: all 40 accepted renders exited 0. Both initial evaluation
  seed-351 attempts exited 127 before rendering because the pinned executable's
  runtime libraries were absent. Restoring those exact dependency versions
  allowed the same seeds to run; neither executable was rebuilt or modified.
  Failed attempts are archived separately and excluded from accepted renders.
- `scorer-output.txt`: unchanged `../../upright-shaping-ab/analyze.py` output.
  Its hardcoded `k12` label aliases floor −0.7; neither arm uses upright shaping.
  Published filenames use `floor07`. All 800 ordered samples passed validation.

The 10-second frames of both evaluation-seed-351 videos were inspected and show
Sally and the intended scene. Close-and-hold is the registered body-distance
measurement, not a count of target grabs. See the [pooled verdict](../verdict.md).
