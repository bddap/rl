# rl#420 — the night blooms breathe

Every night-bloom variant's glow swells and fades on an 8 s cycle
(`BLOOM_CYCLE_S`, ground.rs — lane `[5].w` of every row), 1 ± 0.7 of the tuned
level, phase-drifted across the ground so the web pulses in slow travelling
swells rather than in unison. One mechanism in `night_bloom.wgsl`, driven by the
scaffold's `ctx.time`.

Clips: 9.6 s each (1.2 cycles), 10 fps real time, h264 yuv420p, captured through
the game's own frame-sequence path on the RTX 2080:

```
game fp-screenshot --ground-look night-bloom-<variant> --players 1 --settle 90 \
  --anim-frames 96 --anim-every 3 --cam-pitch=-55 --cam-height 25 \
  --moon-azimuth-deg 200 --moon-elevation-deg 45 --out f.png
ffmpeg -framerate 10 -i f.%04d.png -c:v h264_nvenc -pix_fmt yuv420p -preset p5 -cq 23 <variant>.mp4
```

`montage.png`: rows classic, aurora, ember, frost, rose, filigree; columns one
frame per second across one cycle (t = 0…7 s).

Guard: `game/tests/bloom_anim.rs` — half a cycle apart the rendered frame
changes, a whole cycle apart it repeats bit-exactly.
