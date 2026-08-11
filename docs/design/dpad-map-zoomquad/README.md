# Chord Atlas — zoom-fractal quadrant proposal (rl#358)

The combo space presented as a self-similar place instead of a list. Every node has
four child quadrants laid out like the d-pad (up top, down bottom, left/right beside);
a press dives into that quadrant, which zooms to fill the screen while the parent's
ring ghosts past the edges; the opposite press surfaces. Commands are rooms at their
code's address, so the ^-family really is "the sky wing" and muscle memory becomes
route memory. The space is exponentially bigger than any screen — the non-euclidean
structure IS the presentation.

Owner amendments covered:

- **Unlock growth** — a locked family renders as teased fog: you see how many codes
  sleep there (one faint seed each, placed where they truly live) but not what they
  are. On unlock the fog dissolves and the subtree unfurls with a flash — the atlas
  visibly grows.
- **Melody (rl#359)** — one note per direction (pitch echoes the vertical axis:
  ^ high, v low). The HUD ribbon draws the entered code as a pitch contour; every
  room carries its tune as a sparkline, so melodies are landmarks.

`extract.py` parses `GCR_CHORDS` out of `net/src/controls.rs` (never hand-copied),
`render.py` draws the animation from that data:

```
nix-shell -p python3 --run 'python3 extract.py ../../../net/src/controls.rs'
nix-shell -p python3 python3Packages.pillow dejavu_fonts --run 'python3 render.py'
ffmpeg -framerate 12 -i frames/f%04d.png -vf palettegen=stats_mode=diff palette.png
ffmpeg -framerate 12 -i frames/f%04d.png -i palette.png \
  -lavfi 'paletteuse=dither=bayer:bayer_scale=4' ../dpad-map-zoomquad.gif
```

In-game this maps onto the existing held-X capture unchanged: the atlas replaces the
`chord_menu_text` list while the modifier is held, `Chords::entered()` drives the
camera, and the registry (plus an unlock set) is the whole data dependency.
