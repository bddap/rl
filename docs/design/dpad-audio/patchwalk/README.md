# patchwalk — timbre as place

D-pad audio scheme proposal for rl#359. One idea: **the combo tree is a territory,
and timbre is your position in it.** Every press morphs a continuous synth patch;
a code is a walk you can recognize with your eyes closed.

## The mapping

State is a patch vector — `brightness, inharmonicity, breath, warmth, shimmer` —
starting from a dark, soft pluck at the root. Each direction pulls the patch toward
a fixed character, and the pull **shrinks geometrically with depth** (×0.6 per press):

| direction | region character | pull |
|---|---|---|
| **L** | glass | inharmonic++, colder, bell-like |
| **R** | wood | warmer, harmonic, duller |
| **U** | air | breathy, bright |
| **D** | depth | dark, warm, low |

So the four depth-1 subtrees are four far-apart **timbral regions**, and deeper
presses are progressively finer morphs within a neighborhood — matching the tree's
geometry: siblings near the root are distant places, deep siblings are next-door
rooms. Pitch walks the **hirajoshi** scale (dark, nightish — fits the game's
haunting atmosphere): U/D step by one degree, R/L leap by two, so any code is an
in-scale phrase with contour, never four unrelated beeps. Depth adds the owner's
requested decay-of-certainty: register sinks (~⅓ semitone per press, continuous)
and detune spread widens (+4 cents per press) — deep space sounds lower, wider,
less resolved.

Resolution: an accepted code blooms the **destination patch** as a soft root+fifth
dyad — the place you arrived at gets the last word. An unknown code is a damped,
falling knock in whatever patch you dragged there.

## Why it hooks musical *and* spatial memory

The melody (pitch contour) and the journey (timbre morph) are two independent
encodings of the same code, learned together. Contour carries short codes;
timbre carries *where* — you know you're deep in wood-country before you count
presses. Because patch deltas are additive, **two orderings of the same presses
converge on the same destination sound while being audibly different
performances** (sketch c): the *place* is stable, the *route* is expressive. That
is the spatial hook — codes stop being sequences and become paths through somewhere.

## Growth with unlocks

Locked territory is **veiled, not silent**: presses into it sound distant and
muffled (steep spectral rolloff, half gain) — the map teases what it isn't giving
you, matching #358's "teased buds". An unlock is a **bloom**: a slow rising
arpeggio played through the new region's own patches — the territory introduces
itself in its own voice (sketch d). Unlocked regions also gain **shimmer** (a
slow chorus glow that grows with depth), so hard-won deep territory audibly
glitters: progression literally opens new timbral country.

## Sketches (all synthesized through `scheme.js` itself — no hand-faking)

| file | demonstrates |
|---|---|
| `sketches/a-shallow-code.mp3` | 3-press code + accept: a short phrase that resolves |
| `sketches/b-deep-code.mp3` | 7-press code: register sinks, detune widens, timbre narrows in |
| `sketches/c-two-paths-same-node.mp3` | LUR then RUL: same destination sound, two different journeys |
| `sketches/d-unlock-event.mp3` | veiled knocks on locked territory → unlock bloom → same code lands open |
| `sketches/e-region-tour.mp3` | two steps into each region: glass, air, wood, depth |

## Files

- `scheme.js` — the whole mapping as one **pure function
  `scheme(comboState, press) → soundEvent`**, playground-ready
  (`comboState = {path, unlocked}`; the soundEvent is declarative voice params a
  WebAudio or in-game interpreter can render directly).
- `render.js` — offline Node renderer (inharmonic-partial pluck + Schroeder
  reverb) used to produce the sketches. Pure synthesis, no samples, no assets.
