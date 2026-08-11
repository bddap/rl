// patchwalk — d-pad audio scheme: TIMBRE AS PLACE.
//
// Pure function (comboState, press) -> soundEvent, no side effects, no state.
// comboState: { path: ["L"|"R"|"U"|"D", ...],   // presses BEFORE this one
//               unlocked: ["", "L", "DR", ...] } // unlocked region prefixes ("" = root region)
// press: "L"|"R"|"U"|"D" | "accept" | "reject" | "unlock"
//   ("unlock" celebrates the region the current path points into)
//
// soundEvent: { kind: "note"|"resolve"|"reject"|"unlock",
//               veiled: bool,            // locked territory: render muffled/distant
//               voices: [ { freq,        // Hz
//                           cents,       // additional detune spread (render as ± split)
//                           amp,         // 0..1
//                           dur,         // seconds (decay target)
//                           delay,       // seconds after event start
//                           patch: { brightness, inharm, breath, warmth, shimmer } } ] }
//
// The idea: the combo tree is a territory. Each of the four depth-1 subtrees is a
// timbral REGION (L glass, R wood, U air, D depth). Every press adds a
// direction-specific delta to a continuous patch vector, with step size shrinking
// geometrically with depth — so regions are far apart, neighborhoods are close, and
// a code is a WALK whose sound tells you where you are with your eyes closed.
// Deltas are additive+clamped, so permuted codes converge on (nearly) the same
// destination patch while sounding like different performances on the way.
// Depth also drags register down and detune up (the owner's "detune as combos
// progress deeper") — deep space is darker, wider, less certain.
// Pitch walks the hirajoshi scale (dark, nightish) — U/D step, R/L leap.

"use strict";

// hirajoshi on A: A B C E F — semitone offsets from root
const SCALE = [0, 2, 3, 7, 8];
const ROOT_HZ = 220; // A3

const ROOT_PATCH = { brightness: 0.35, inharm: 0.15, breath: 0.20, warmth: 0.70, shimmer: 0.0 };

// per-direction character: where each region pulls the patch
const DELTA = {
  L: { brightness: +0.18, inharm: +0.55, breath: -0.10, warmth: -0.25 }, // glass
  R: { brightness: -0.15, inharm: -0.20, breath: +0.05, warmth: +0.35 }, // wood
  U: { brightness: +0.30, inharm: +0.10, breath: +0.45, warmth: -0.15 }, // air
  D: { brightness: -0.30, inharm: +0.15, breath: +0.10, warmth: +0.30 }, // depth
};
const STEP_AT_DEPTH = (d) => 0.85 * Math.pow(0.60, d); // d = 0 for the first press

// scale-degree motion: U/D step, R/L leap (melodic, not sour — everything stays in scale)
const DEGREE_STEP = { U: +1, D: -1, R: +2, L: -2 };

const clamp01 = (x) => Math.min(1, Math.max(0, x));

function walk(path) {
  // fold a path into (patch, degree, depth)
  const p = { ...ROOT_PATCH };
  let degree = 0;
  path.forEach((dir, i) => {
    const s = STEP_AT_DEPTH(i);
    for (const k in DELTA[dir]) p[k] = clamp01(p[k] + DELTA[dir][k] * s);
    degree += DEGREE_STEP[dir];
  });
  return { patch: p, degree, depth: path.length };
}

function degreeToFreq(degree, depth) {
  const oct = Math.floor(degree / SCALE.length);
  const idx = ((degree % SCALE.length) + SCALE.length) % SCALE.length;
  const semis = SCALE[idx] + 12 * oct;
  // deep space sinks: -1 semitone of drift per 3 presses, continuous
  const sink = -depth / 3;
  return ROOT_HZ * Math.pow(2, (semis + sink) / 12);
}

function isUnlocked(path, unlocked) {
  // a node is open when SOME unlocked prefix covers it
  const code = path.join("");
  return unlocked.some((u) => code.startsWith(u) || u.startsWith(code));
}

function scheme(comboState, press) {
  const before = comboState.path;
  const unlocked = comboState.unlocked || [""];

  if (press === "accept" || press === "resolve") {
    // resolution: the destination patch blooms as a dark dyad (root + fifth below)
    const { patch, degree, depth } = walk(before);
    const f = degreeToFreq(degree, depth);
    return {
      kind: "resolve",
      veiled: false,
      voices: [
        { freq: f, cents: 4 + depth * 2, amp: 0.55, dur: 2.8, delay: 0, patch },
        { freq: f / 1.4983, cents: 3, amp: 0.40, dur: 3.4, delay: 0.07, patch: { ...patch, warmth: clamp01(patch.warmth + 0.2) } },
        { freq: f * 2, cents: 6, amp: 0.12, dur: 2.0, delay: 0.14, patch: { ...patch, brightness: clamp01(patch.brightness + 0.2) } },
      ],
    };
  }

  if (press === "reject") {
    // unknown code: a damped, falling thud in the patch you arrived with
    const { patch, degree, depth } = walk(before);
    const f = degreeToFreq(degree, depth) / 2;
    const dull = { ...patch, brightness: 0.1, inharm: clamp01(patch.inharm + 0.3), breath: 0.05 };
    return {
      kind: "reject",
      veiled: false,
      voices: [
        { freq: f, cents: 18, amp: 0.5, dur: 0.5, delay: 0, patch: dull },
        { freq: f * 0.944, cents: 18, amp: 0.35, dur: 0.7, delay: 0.12, patch: dull },
      ],
    };
  }

  if (press === "unlock") {
    // the veil lifts on the region the current path points into: a slow rising
    // arpeggio THROUGH that region's own patches — new territory introduces itself
    const ev = { kind: "unlock", veiled: false, voices: [] };
    for (let i = 1; i <= 4; i++) {
      const sub = before.slice(0, i);
      const { patch, degree, depth } = walk(sub.length ? sub : before);
      ev.voices.push({
        freq: degreeToFreq(degree + i, depth),
        cents: 5,
        amp: 0.30 + i * 0.05,
        dur: 1.6 + i * 0.3,
        delay: i * 0.22,
        patch: { ...patch, shimmer: clamp01(0.25 * i) },
      });
    }
    return ev;
  }

  // a directional press: sound the node we land on
  const after = [...before, press];
  const { patch, degree, depth } = walk(after);
  const open = isUnlocked(after, unlocked);
  // shimmer is unlock-bought territory: open regions glow slightly with depth
  const shimmer = open ? clamp01((depth - 1) * 0.12) : 0;
  return {
    kind: "note",
    veiled: !open,
    voices: [
      {
        freq: degreeToFreq(degree, depth),
        cents: 3 + depth * 4, // deeper = wider (the owner's progressive detune)
        amp: open ? 0.6 : 0.25,
        dur: 1.1 + 0.08 * depth,
        delay: 0,
        patch: { ...patch, shimmer },
      },
    ],
  };
}

if (typeof module !== "undefined") module.exports = { scheme, walk, degreeToFreq, SCALE, ROOT_HZ };
