# rl#372 — shadow-side hillside renders a screen-corner "sunlit" wash

> **SUPERSEDED — wrong diagnosis.** The owner ruled the captured "artifact"
> below is an expected specular highlight, not the reported glitch (which was
> some sort of clipping). The reflectance-ramp fix was reverted. Kept for
> reference only; the issue remains open pending a position-anchored live repro.

Repro (deterministic): `game fp-screenshot --walk-at 1 --anim-frames 17
--anim-every 12 --cam-pitch=-12 --cam-yaw=45` (default seed), frame 16 — the
walking player is on a hillside with the camera a little inward and down, the
reported recipe. Pin the moon with `--moon-azimuth-deg/--moon-elevation-deg`
to hold the pose.

## Diagnosis chain (all at that exact frame, pixel-comparable)

1. **Shadows are not the term.** `shadow_maps_enabled: false` reproduces the
   baseline at PSNR 97 dB (identical) — cast shadows contribute nothing here.
2. **Specular is the term.** `reflectance: 0.0` on the ground material removes
   exactly the white wash (PSNR 26 dB vs baseline — a large, targeted change).
3. **View-anchored, not world-anchored.** The wash brightens toward the screen
   corner where view rays graze the slope, and sweeps with camera yaw at a
   fixed player position; with the moon pinned at a grazing azimuth (270°) it
   is strongest (`before-moon-grazing.png`). Un-tinted by albedo (white against
   green ground), it reads as sunlight on a shadowed hillside.

Mechanism: single-scatter GGX at perceptual_roughness 0.95 + Schlick Fresnel
(f90 → 1 at grazing) still returns a broad bright sheet from the 9500-lux moon.

## Fix

`ground.wgsl` (the one scaffold): fade `material.reflectance` to zero as the
look's roughness approaches matte — `*= 1.0 - smoothstep(0.75, 0.92,
a.roughness)`. Matte ground (every look's dry base, 0.95) carries no lobe;
watershed's deliberate sheens (fully-wet basins ≈0.719, wet micro 0.18,
puddle cores 0.05) sit below the ramp and keep full reflectance.

## Captures

- `before-artifact.png` / `after-fixed.png` — repro pose, free-running moon.
- `before-moon-grazing.png` / `after-moon-grazing.png` — repro pose, moon
  pinned az 270° el 30° (worst case). The after-frame is pixel-identical
  (PSNR inf) to the reflectance-0 control at this pose.
- `watershed-before.png` / `watershed-after.png` — fixed-cam watershed look:
  wet-basin comb sheen and dew glints preserved; only the dry grazing wash is
  gone.
