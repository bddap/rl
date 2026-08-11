// HARMONIC FIELD — d-pad instrument mapping scheme (rl#359)
//
// Pure playground-shape module: (comboState, press) -> soundEvent.
// No state, no I/O, no deps. Drop into bddap-bot/dpad-audio as-is.
//
// comboState = {
//   path:       array of 'U'|'D'|'L'|'R' — presses consumed so far (NOT incl. this one)
//   unlockTier: 0..3 — progression tier (0 = fresh save)
// }
// press = 'U'|'D'|'L'|'R'         — a combo press
//       | 'accept' | 'reject'    — code resolution
//       | 'unlock'               — an unlock event just fired (tier went up)
//
// soundEvent = {
//   kind: 'note'|'accept'|'reject'|'unlock',
//   notes: [{ freq, gain, decay, detuneCents, brightness, at }]
//     notes[0] is the lead voice; the rest are the harmonic pad under it.
//     at = seconds offset (arpeggiation); decay = exp tau seconds.
// }

// The field: each node of the combo tree is a POSITION in a pitch lattice.
//   L/R step the chord axis (motion by scale-thirds — harmonic color)
//   U/D step the melody axis (motion by scale steps)
// The node's chord = a third-stack rooted where you stand, so every press
// sounds a note that is consonant IN CONTEXT, and the context itself moves
// under your feet as you type. Two orderings of the same presses land on the
// same lattice point: different melody, identical arrival harmony.

const ROOT_HZ = 220; // A3 — dark register

// Scale grows with unlocks: Am add-nothing -> minor pentatonic -> +9 -> +b6
// (the b6 is the hirajoshi/insen darkness, saved for late game).
const TIER_SCALES = [
  [0, 3, 7, 10],           // A C E G
  [0, 3, 5, 7, 10],        // + D  (minor pentatonic)
  [0, 2, 3, 5, 7, 10],     // + B  (9th color)
  [0, 2, 3, 5, 7, 8, 10],  // + F  (b6 — the haunted note)
];

const MOVES = { L: [-1, 0], R: [1, 0], D: [0, -1], U: [0, 1] };

const mod = (a, n) => ((a % n) + n) % n;

// Scale degree k (any integer) -> semitones from root, octave-extended.
function degree(scale, k) {
  const n = scale.length;
  return scale[mod(k, n)] + 12 * Math.floor(k / n);
}

const hz = (semis) => ROOT_HZ * Math.pow(2, semis / 12);

function position(path) {
  let x = 0, y = 0;
  for (const p of path) { const m = MOVES[p]; x += m[0]; y += m[1]; }
  return { x, y };
}

// The chord standing at lattice (x, y): third-stack rooted at degree 2x,
// voiced low (pad register), denser when deeper in a code.
function nodeChord(scale, x, y, density) {
  const notes = [];
  for (let i = 0; i < density; i++) {
    notes.push(degree(scale, 2 * x + 2 * i) - 12); // an octave under the lead
  }
  return notes;
}

export function scheme(comboState, press) {
  const tier = Math.max(0, Math.min(3, comboState.unlockTier | 0));
  const scale = TIER_SCALES[tier];
  const path = comboState.path || [];

  if (press === 'unlock') return unlockEvent(tier);
  if (press === 'reject') return rejectEvent(path, scale);
  if (press === 'accept') return acceptEvent(path, scale, tier);

  const m = MOVES[press];
  const { x: px, y: py } = position(path);
  const x = px + m[0], y = py + m[1];
  const depth = path.length + 1;

  // Deeper = hazier and darker: more detune, less brightness, denser pad.
  const detuneCents = Math.min(depth, 8) * 1.3;
  const brightness = Math.max(0.25, 1 - depth * 0.09);
  const density = Math.min(2 + (depth >> 1), 2 + tier + 1);

  const lead = {
    freq: hz(degree(scale, y)),
    gain: 0.9, decay: 0.9, detuneCents, brightness, at: 0,
  };
  const pad = nodeChord(scale, x, y, density).map((s, i) => ({
    freq: hz(s),
    gain: 0.16, decay: 1.6, detuneCents: detuneCents * 0.5,
    brightness: brightness * 0.5, at: 0.012 * (i + 1),
  }));
  return { kind: 'note', notes: [lead, ...pad] };
}

// Accepted code: the node you landed on blooms — its full chord arpeggiates
// upward. The resolution IS the place: each code's accept chord is the sound
// of its address, so sibling codes share a family resemblance.
function acceptEvent(path, scale, tier) {
  const { x, y } = position(path);
  const size = 3 + tier; // richer accepts as the field grows
  const notes = [];
  for (let i = 0; i < size; i++) {
    notes.push({
      freq: hz(degree(scale, 2 * x + 2 * i) - 12),
      gain: 0.55 - i * 0.05, decay: 2.4, detuneCents: 3,
      brightness: 0.7, at: i * 0.07,
    });
  }
  // crown: the melody note you ended on, an octave up, late and soft
  notes.push({
    freq: hz(degree(scale, y) + 12),
    gain: 0.3, decay: 2.8, detuneCents: 5, brightness: 0.5,
    at: size * 0.07 + 0.05,
  });
  return { kind: 'accept', notes };
}

// Unknown code: no bloom — a damped low cluster, felt more than heard.
function rejectEvent(path, scale) {
  const { x } = position(path);
  const root = degree(scale, 2 * x) - 24;
  return {
    kind: 'reject',
    notes: [
      { freq: hz(root), gain: 0.5, decay: 0.22, detuneCents: 0, brightness: 0.2, at: 0 },
      { freq: hz(root + 1), gain: 0.3, decay: 0.18, detuneCents: 0, brightness: 0.15, at: 0.02 },
    ],
  };
}

// Unlock: the NEW degrees of the wider scale shimmer in, ascending — the
// field itself audibly gains notes it will keep from now on.
function unlockEvent(tierNow) {
  const prev = TIER_SCALES[Math.max(0, tierNow - 1)];
  const now = TIER_SCALES[tierNow];
  const fresh = now.filter((s) => !prev.includes(s));
  const ladder = [...fresh.map((s) => s + 12), ...fresh.map((s) => s + 24)];
  const notes = ladder.map((s, i) => ({
    freq: hz(s), gain: 0.4, decay: 2.2, detuneCents: 6,
    brightness: 0.8, at: i * 0.16,
  }));
  // ground it: tonic drone under the shimmer
  notes.unshift({ freq: hz(-12), gain: 0.3, decay: 3.5, detuneCents: 2, brightness: 0.3, at: 0 });
  return { kind: 'unlock', notes };
}

export default scheme;
