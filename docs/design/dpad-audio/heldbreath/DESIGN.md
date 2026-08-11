# heldbreath — tension-gradient d-pad instrument

Entering a code is inhaling; completing it is the exhale. Every press adds
melodic motion AND harmonic tension; the cadence at the end releases it — or
pointedly doesn't.

## The mapping

**Press = interval, not note.** Each direction moves the melody relative to
where it already stands, in scale degrees of A hirajoshi (A-B-C-E-F, root A2 —
dark, nightish, no sour intervals): `U +2, R +1, L −1, D −2`. So a code is a
*contour* — rise, fall, leap — not a key sequence. Two different paths to the
same node are two different melodies that end in the same place (audible in
sketch c). The map is a function of combo state, never a fixed key→note table.

**Depth = tension.** Every event carries the gradient of how deep you are:

| depth grows → | detune (twin voices) widens 6¢/press → audible beating |
|---|---|
| | brightness falls (lowpass closes — the room darkens) |
| | note decay shortens (breath tightening) |
| | a root drone underneath gains a minor-2nd shimmer |

Shallow codes are calm plucks; a 7-press code audibly *needs* resolution —
held breath.

**Region = timbre.** The first press of a code picks the voice family for the
whole phrase: `L` dark pluck, `D` soft bell, `R` hollow, `U` glass. Regions of
the combo tree literally sound like different instruments, so spatial position
in the map has a sonic signature.

**Completion = cadence.**
- *Accepted:* authentic cadence home — staggered root/fifth/octave chord,
  detune collapses to zero, long release, drone resolves. Exhale.
- *Unknown:* deceptive cadence — lands a half-step off home, keeps the
  accumulated detune, damps early. The breath is not released; you feel the
  wrongness without a buzzer.

**Unlock = the gradient in reverse.** A bloom arpeggio that starts wide-detuned
and converges to pure as it climbs into a note the scale didn't have before —
each unlock tier admits a color degree (b7, then #4, then 4th) into the region's
scale. Progression literally gives the instrument more notes, so late-game
codes can be melodies early-game ones couldn't be.

## Why it hooks memory

- **Musical:** contour + tension-arc is what melodic memory actually stores
  (people remember shapes and resolutions, not absolute pitches). A code is
  recalled as "rise-rise-dip, then the exhale."
- **Spatial:** depth is *felt* (beating, darkening, tightening) — you know how
  deep you are with your eyes closed, and the region timbre tells you which
  quarter of the tree you're in. The audio is a position sense for the
  non-euclidean map (#358).
- **Progression (#358 growth):** unlocks change the palette, not just the map —
  the satisfying growth moment has a sound (the bloom) and a lasting audible
  consequence (new degrees).

## Artifacts

- `scheme.js` — the pure `(comboState, press) → soundEvent` mapping, playground-
  ready (the sketches are rendered from this exact file).
- `sketches/` — `a-shallow`, `b-deep`, `c-twopaths`, `d-unknown`, `e-unlock`
  (.ogg). All pure synthesis, no samples.
