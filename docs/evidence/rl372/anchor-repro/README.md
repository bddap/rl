# rl#372 round 2 — position-anchored repro: view-dependent wrong-side lighting

A fresh sighting anchored the artifact to world position ≈ (14081, −174, −4644)
on build 36cc5757 (ground look watershed, render mode mesh), with the note that
it "feels like a frustum thing" and that it became invisible after the in-game
lighting moved a few minutes later.

**Reproduced, deterministically, at that anchor.** A near receiver renders
fully sunlit while standing inside a terrain cast shadow, and flips
shadowed⇄sunlit with CAMERA YAW ALONE — world and light untouched. It also
flips with small moon-elevation changes inside the legal traversal band, which
matches the "went away when the lighting moved" observation.

## The repro pose

The sighting locale is restart draw 3 of seed `0x631a539793517b5e` — reachable
headlessly with the `--restart-taps` flag (added for this repro; each tap draws
the next rl#305 layout off the seed's stream):

```
game fp-screenshot --seed $((0x631a539793517b5e)) --restart-taps 20,30,40 \
  --ground-look watershed --settle 45 --cam-pitch=-12 --cam-yaw=90 \
  --moon-azimuth-deg=62.7 --moon-elevation-deg=34 --out el34-yaw90.png
```

That lands the layout `origin=(14081.7, -4637.2) heading=106°
extraction=(14089.9, -4639.5)` — the player standing at the anchor on a steep
shadow-side hillside. The second player's capsule stands a couple of meters
away and is the clearest receiver to watch (any near receiver works; the
capsule just shows it unambiguously).

A terrain ray-march from the anchor toward the moon confirms the anchor is
inside a terrain cast shadow for every traversal elevation up to ~40°
(occluding ridge 4–170 m up-slope along the light azimuth; e.g. at el 34°/az
62.7° the occluder is 168 m away, ~95 m higher). At all of the poses below the
receiver SHOULD be shadowed (el ≤ 40) — yet:

| capture | camera yaw | moon el | capsule renders |
|---|---|---|---|
| `el34-yaw70-shadowed.png` | 70 | 34 | shadowed ✓ |
| `el34-yaw90-sunlit.png` | 90 | 34 | **fully sunlit ✗** |
| `el34-yaw110-sunlit-edge.png` | 110 | 34 | **sunlit, at the screen edge ✗** |
| `el29-yaw90-sunlit.png` (pitch 0) | 90 | 29 | **fully sunlit ✗** |
| `el31-yaw90-shadowed.png` (pitch 0) | 90 | 31 | shadowed ✓ |
| `el37-yaw90-healed.png` | 90 | 37 | shadowed ✓ (hillside behind lit ✓) |

Yaw 70→90→110 changes NOTHING but the camera: the flip is view-driven — the
reported "frustum thing". The el34→el37 pair flips the capsule dark while the
hillside behind it brightens (the moon rose): the two receivers move in
opposite directions, so one of the frames is unphysical by construction.

## Reading

Same class as the deliberately-fenced limitation recorded in
`docs/evidence/rl387` (near-field receivers missing shadows from far casters —
there believed confined to below the moon's elevation floor). At this locale
the terrain is mountain-scale: occluders sit 25–170 m up-slope and cast long
shadows at legal elevations, so the seam manifests inside the traversal band
the floor is supposed to fence off. Root-cause (why the near cascade's shadow
map misses the far terrain caster at some view orientations, given casters are
depth-pancaked) is the next increment.

## Bounds of the sweep (what did NOT show an artifact)

- Ground-pixel version at this anchor: not found in a static grid of
  el {16,18,20,21.5,23,25,27,29,31,34,37,40} × yaw {0,45,…,315} × pitch
  {0,−12,−25} (watershed, NVIDIA) nor in walk-arc clips (20 frames × 4 yaws,
  shipped AND watershed, lavapipe). The steep local slope faces away from the
  moon, so a missing-shadow patch on GROUND is masked by N·L darkness here;
  the capsule (curved) is what makes the miss visible at this locale.
- Fresh-load/warp confound: excluded — frames captured immediately after the
  29 km restart-warp are byte-identical (PSNR ∞) to frames 550 render-frames
  later with the moon frozen, on both lavapipe and NVIDIA.
- Driver: the flip reproduces on the NVIDIA 2080 (the hardware behind the
  sighting lineage) — lavapipe renders were used for the bounded negative
  sweeps only.
