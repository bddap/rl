# rl#358 proposal: The Hyperbolic Atlas

The chord registry is a 4-ary prefix tree — exponential in depth, so it doesn't
fit flat euclidean space. This proposal takes that literally: lay the real
`GCR_CHORDS` tree out in a Poincare disk, where exponential branching has room
by construction. Each d-pad press is a hyperbolic translation that glides the
map one node deeper; release folds you home.

- **Place memory**: each prefix subtree is a district with stable color +
  landmark glyphs (Night-Bloom Garden `v^`, Loam Quarter `v<`, The Watershed
  `v>`, Sky Harbor `^`, Render Observatory `^^`, the Monoliths `<`/`>`). A code
  is remembered as a walk through recognizable places, not a list row.
- **Growth (unlock progression)**: locked districts sit at the rim as fogged
  silhouettes labeled `? ? ?` — teased, not hidden. Unlocking blooms the region
  open, nodes scaling in staggered by depth.
- **Melody (rl#359)**: each direction is a scale note (C/D/E/G by "upness");
  pitch-ladder glyphs sit on every edge and the traversed code accumulates on a
  staff — a code IS a tune. Unlocks extend the scale (the A rung).

In-game this replaces the held-modifier text menu (`chord_menu_text`): hold X,
the atlas appears centered on your entered prefix, each tap glides.

`generate.py` renders `dpad-map-hyperatlas.gif` from the real 24-entry registry
(command in its docstring).
