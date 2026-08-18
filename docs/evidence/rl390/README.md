# rl#390 — post-rl#373 ground still blurry: the descent floor was the blur, not the fades

Step 1 (was the blur the email pipeline?): native 1280×720 captures at the
reported pose (moon az 270°, el 12°, cam pitch −75°) are soft at 1:1 —
in-render blur, real.

Root cause — instrumented, three candidates eliminated in order:

1. **grain_fade window (the issue's lead suspect): innocent.** Widening
   [0.07 wl, 0.28 wl] → [0.125 wl, 0.5 wl] changed the frame by ≤2 LSB
   (mean |diff| 0.016) — nothing the fades gate was being cut.
2. **fw instrumentation** (footprint rendered as color): 53% of the frame sat
   AT the 1e-4 clamp; the rest 0.1–0.6 mm/px. A camera-distance render
   explains it: at pitch −75° the whole frame is ground **4–17 cm** from the
   lens — the crab eye is centimeters up, so a down-look is a macro shot, and
   near ground runs 0.02–0.2 mm/px, not the ~2 mm/px the descent floor's
   comment assumed (a standing human eye).
3. So the **3.5 mm descent floor** was the resolution ceiling: ~35 px per
   wavelength at frame right — exactly the "upscaled low-res texture" failure
   rl#324's adaptive descent exists to prevent, reintroduced by its own
   backstop. Mip/texture filtering: no texture in the path (procedural).

Fix: floors 3.5 mm → 0.1 mm (finest octaves now: color 0.13 mm, relief
0.21 mm), fw clamp 1e-4 → 1e-5 so it can't lie to the descent again. The fade
exit stays the resolution law — distant pixels still leave the loop at the
same octave as before.

Depth re-match (the rl#373 gradient-RMS method, `rms_sim.py`): the composed
RMS table IS the re-match —

- color: ≤ +2% at any footprint;
- relief: **ratio 1.00 for every fw ≥ 0.5 mm** (all walking-height and vista
  views — pixel-identical captures below confirm), rising to 1.56× at 0.1 mm
  and 2.46× at 0.02 mm pre-soft-limit, ~1.5× post-limit.

That rise is the fix operating (energy the old floor deleted), not drift: a
global nweight rescale to hold near-field RMS at 1.0 would cut tuned
mid-range relief ~2× — that would be the drift. Tuned regimes hold by
construction; new energy exists only where the old output was the blur.

Collateral bound (fixed-cam before/after, mean |diff| in LSB by frame third):

| pose                | far   | mid   | near  | max |
|---------------------|-------|-------|-------|-----|
| vista (+10 m, −15°) | 0.000 | 0.000 | 0.000 | 0   |
| foot grazing (0°)   | 0.000 | 0.000 | 0.042 | 2   |
| mid (−30°)          | 0.024 | 0.175 | 0.446 | 4   |
| watershed (−75°)    | 0.741 | 0.802 | 0.825 | 6   |

Cost (lavapipe, worst case — full-frame macro close-up, 120 frames): +17%
CPU, +3% wall.

## Captures

Same conventions as rl#373: moon az 270°, el 12°, default seed/pose; `-x5crop`
= center-right 400×400 crop ×5 exposure (night scenes are dim; the boost is
stated, not hidden).

- `before/after-down75(.png, -x5crop.png)` — the reported pose, native res.
- `before/after-watershed-x5crop.png` — collateral on a second look: same
  fix, look identity unchanged.
