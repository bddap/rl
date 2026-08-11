# Checkpoint archaeology — the last Sally that actually chased (owner ask, 2026-08-04)

Pre-terrain (< rl#281, 2026-07-16), pre-chase-eval (< 79d11ae, 2026-06-30) candidates,
re-verified 2026-08-06 by rendering each under the archived era binary
`rl-demo-3ddcf6f` (commit 3ddcf6f, 2026-07-01 — the newest binary that loads the
Jun-28 checkpoint format; pre-2026-06-29 source history did not survive the rewrite,
so archived binaries are the only era pins).

Checkpoints archived durably at `~/.local/state/rl-target/archive/sally377-jun28-chase/`
on bothouse. `RL_TARGET_BALL=1`, flat-floor era, lavapipe headless render.

- `sally377-recheck.*` — Jun-28 brain.bin (it~500 after the 2x energy tax), paired normalizer.
- `prepenalty-paired.*` — Jun-28 brain-pre.bin (pre-energy-penalty), paired normalizer.
- `reach1.0-iter1046.*` — Jun-28 03:19 brain that hit reach 1.0 at iter 1046
  (from ckpt-backup-pre-job650); normalizer is the Jul-01 lineage one (unpaired — closest surviving).
- `era-jun28-*` — job-377 originals rendered ON 2026-06-28 (before/after the energy tax).

Bisect probes (rl#351, post-shortlist):

- `probe-a-8012a4f-endofbound.*` — Probe A end-of-bound render (2026-08-10): main
  `8012a4f8` cold canary, `--terrain flat --band-max-m 9`, 24 h / 19.2M ticks.
  Rendered with the pin's own `rl-demo` (built at `8012a4f8`), end-of-bound
  checkpoint, seed 351, canonical GCR tile (the demo's only ground, rl#293 — same
  locale family as chase-eval). Policy armed (mean|drive| 0.885) but no chase:
  aimless downhill drift, reach 0.00. Verdict on the issue: confounded (1 rl#343
  hard-fail), red-leaning.
