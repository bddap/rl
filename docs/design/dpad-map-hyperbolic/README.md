# rl#358 proposal: hyperbolic recentering chord map

The chord code space is a 4-ary tree — exponential in depth, so no flat layout
scales. This proposal lays the REAL GCR registry (net/src/controls.rs) out in
the Poincare disk: exponential room is what hyperbolic space is FOR. Each d-pad
press applies a Mobius translation gliding the pressed child to the center, so
the map always fits, nearby options are always legible, and the glide itself is
the code you are typing — motion encodes the path.

- Unlock growth (rl#358 amendment): a locked family is a sealed bud — position
  teased, content hidden. Unlocking blooms the subtree in place; the map
  visibly GROWS where you already knew something was buried.
- Instrument hook (rl#359): each direction is a fixed note (L=A, D=C, R=E,
  U=G). Every edge carries a 4-rung pitch glyph; the strip at bottom-left plays
  back the typed code as a melody contour. Unlocks could extend the scale.

`render.py` produces the animation (frames + ffmpeg palette pass; see its
docstring). Mock only — the in-game version would live in the held-modifier
menu path (crab-world/src/chord.rs glue).
