# rl#387 — do the moon's cast shadows work at all? Yes.

The rl#372 diagnosis found `shadow_maps_enabled: false` pixel-identical to
baseline (PSNR 97 dB) at that one pose and asked for a deliberate check: a tall
occluder at a low moon angle, shadows on vs off, is there a visible delta.

Method: two builds of `game` off the same tree, differing only in
`shadow_maps_enabled` on the moon light (moon.rs); identical `fp-screenshot`
invocations (deterministic, default seed); PSNR via ffmpeg. Rendered headless
on lavapipe.

## Verdict — shadows contribute, massively, once the moon is low

| pose | shadows on vs off |
|---|---|
| occluder pose, moon el 5° az 270° (`--settle 30 --cam-pitch=-20 --cam-yaw 45`, 640×360) | **14.4 dB** — night and day (`occluder-el5-*.png`) |
| rl#372 repro pose pinned el 10° az 315° (`--walk-at 1 --anim-frames 17 --anim-every 12 --cam-pitch=-12 --cam-yaw 45`, frame 16) | **12.4 dB** (`repro-el10-*.png`) |
| same repro pose, moon free (default el 21.5°) | 37.6 dB — crab/capsule self-shadowing, little on the ground |
| occluder pose at el 25° | 43.4 dB — delta confined to the capsule |

In the low-moon on-frames: hillsides cast broad shadow bands across the
valley, terrain self-shadows, the crab and grass scatter go shadow-side dark.
Off-frames render fully lit. Cast shadows work — terrain, ground, and skinned
crab all cast and receive.

## Why the rl#372 measurement saw nothing

Two compounding reasons, no bug:

- At the shipped elevation floor (21.5°, `ELEVATION_FLOOR_DEG`) sun angles are
  steep enough that on mild slopes almost nothing occludes anything — the
  ground delta only appears well below the floor, which in-game never happens.
- The 97 dB figure no longer reproduces on main anyway: the same repro pair
  today is 37.6 dB (self-shadowing on the crab that walks through the frame).

## Known, deliberately-fenced limitation (visible in `repro-el10-shadows-on.png`)

A hard horizontal seam crosses the on-frame: near-field ground (inside the
first cascades) misses shadows from far casters, so the foreground stays lit
while the midfield sits in a hill's shadow. This is the exact artifact the
moon's elevation floor exists to keep off-screen (see the `Moon::elevation_deg`
doc comment) — it only shows when you pin the moon below the floor, which the
traversal never does. Nothing to fix while the floor stands.
