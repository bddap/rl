# rl#373 — relief normals showed the value-noise lattice; LOD fades rang

Two defects, one module (`ground_detail.wgsl` + `noise.wgsl`):

1. **Lattice.** Value noise puts one height per grid cell; its derivative
   concentrates energy at cell scale, so the relief octaves read as "pixels
   with interpolation ramps". Fix: gradient noise (`gnoise` — Perlin-class,
 same hash family/lattice/quintic fade, RMS-matched ×2.1) in both adaptive
   descents. `vnoise` stays for threshold/vein/field uses: gnoise is zero at
   every lattice point, so thresholding it or tracing its zero-set would
   re-inherit the lattice.
2. **Rings.** Each octave died over a `grain_fade` window of edge-ratio 2.5
   while the descent steps wavelengths by 3 — a plateau between one octave's
   death and the next's fade start, i.e. a visible amplitude ring per octave.
   Fix: window widened to [0.07 wl, 0.28 wl] (ratio 4 ≥ 3) — continuous
   rolloff, octaves now die at ~3.6 px/wavelength instead of 5.3 (nearer
   Nyquist: some distant shimmer, preferred per the issue's tiebreaker over
   visible rings).

Retune: relief `nweight` 0.06 → 0.0405 (× 0.675, the measured vnoise/gnoise
finite-difference gradient-RMS ratio at the 0.27·wl step — `noise_sim.py`),
holding the pre-fix composed relief strength. Not pixel-identical anywhere
relief/grain contributes — that is the point; the fixed-cam pairs below bound
the collateral: composition, albedo fields, veins, sparkle all unchanged,
only the micro-relief/grain character moves.

## Captures

All at moon az 270°, el 12° (grazing — the reported case), default seed/pose.
`-x5`/`-x5crop` files are the same pixels ×5 exposure (night scenes are dim;
the boost is stated, not hidden), crops are the frame center.

- `before/after-down75(-x5crop).png` — straight down at on-foot range: the
  lattice (blocky cells + ramps) vs isotropic grain at matched depth.
- `before/after-vista-rings(-x5).png` — elevated walking-away view: before,
  checkerboard lattice near + banded fade rings out; after, continuous
  rolloff, no rings.
- `before/after-foot-grazing.png` — the as-reported on-foot grazing view.
- `before/after-{watershed,nightbloom}-foot-x5.png` — collateral bound on
  two other looks: relief character changes, look identity does not.
