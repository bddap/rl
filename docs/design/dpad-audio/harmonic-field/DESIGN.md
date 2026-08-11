# Harmonic Field — d-pad instrument scheme (rl#359)

**One sentence:** the combo space is a chord landscape — each node is a harmonic
context, each press sounds a note *in* that context — so typing a code plays a
phrase, and just wandering the d-pad plays a coherent piece.

## The mapping

The combo tree is projected onto a pitch **lattice** (Tonnetz-style). Your path
so far puts you at a lattice position `(x, y)`; the two axes carry different
musical meaning:

- **U/D move the melody axis** — one scale step up/down. Vertical runs are
  melodic lines.
- **L/R move the chord axis** — one scale-third sideways. Horizontal moves are
  harmonic motion: the chord under your feet rotates through the scale's
  third-stacks, the way folk harmony walks between related chords.

A press sounds two things at once:

1. **Lead** — the scale note at your new melody position. Same direction,
   different place → different note, but *always* consonant, because pitch
   comes from the field, not from the button.
2. **Pad** — the chord rooted at your new position, quiet, an octave below.
   This is the "context": it tells your ear where you are.

**Depth darkens.** Each press deeper adds detune haze (+1.3 cents/press),
rolls off brightness, and thickens the pad. Shallow codes are clear chimes;
deep codes sink into fog. You can *hear* how deep you are.

**Resolution is a place, not a jingle.** An accepted code arpeggiates the full
chord of the node you landed on — so every code's accept sound is the sound of
its *address*, and codes in the same region resolve with a family resemblance.
An unknown code gets a damped low thud: no bloom, no place.

## Why it hooks musical and spatial memory

Because position — not the press — determines pitch, the two memories are the
*same* memory: remembering a code's tune IS remembering its walk through the
lattice. Reversed steps cancel audibly (U then D returns to the note you left),
sibling codes rhyme, and codes that end in the same region share a cadence.
The lattice sum is order-independent, so two different paths to one node play
different melodies but arrive at the *identical* chord — the space is genuinely
a place you navigate, not a list of key sequences. This also pairs exactly with
the #358 spatial-map proposals: the map shows the field, the ear confirms it.

## Growth with unlocks

The field's scale is the progression resource. Tiers add degrees:

| tier | scale (over A) | color |
|---|---|---|
| 0 | A C E G | bare minor 7 — sparse, night |
| 1 | + D | minor pentatonic |
| 2 | + B | 9th shimmer |
| 3 | + F | the b6 — hirajoshi/insen darkness |

An unlock plays the **new notes themselves** shimmering in over a tonic drone —
the field audibly gains vocabulary it keeps forever. Every old code then sounds
subtly richer (denser accept chords, new passing tones in the lattice), so
progression retunes the whole world rather than adding a menu entry. Dark
throughout: rooted on A minor at A3, and the *brightest* colors arrive last as
earned depth, not cheer.

## Sketches (`sketches/`)

| file | shows |
|---|---|
| `a-shallow-code.mp3` | 2-press code, twice: a motif you could hum back |
| `b-deep-code.mp3` | 7-press code, twice: haze/density growing with depth |
| `c-two-paths-one-node.mp3` | R,R,U,U vs U,R,U,R — different tunes, same arrival chord |
| `d-unlock-bloom.mp3` | code at tier 0 → unlock shimmer → same code at tier 2, richer |
| `e-wander.mp3` | aimless wandering at tier 3 — still music |

All pure synthesis (`scheme.js` + `render.mjs`, node + ffmpeg), no samples.
`scheme.js` is the playground-shape module: `(comboState, press) → soundEvent`,
stateless, dependency-free — the sketches are rendered through the exact
function the playground would run.
