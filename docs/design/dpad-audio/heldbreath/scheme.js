// heldbreath — tension-gradient d-pad instrument scheme.
// Pure mapping: (comboState, press) -> soundEvent. No audio code here; a
// renderer (the web playground, the game, sketches/render.js) interprets the
// event. CommonJS+browser-safe single function export.
//
// comboState = {
//   path:     ['U'|'D'|'L'|'R', ...]  // presses already consumed, oldest first
//   unlocked: number                  // 0 = base 5-note scale; each unlock
//                                     // tier adds a color degree to the scale
// }
// press = 'U'|'D'|'L'|'R'            // a d-pad press, or one of the meta
//       | 'accept'                    // code completed, recognized
//       | 'reject'                    // code completed, unknown
//       | 'unlock'                    // a new region just unlocked
//
// soundEvent = {
//   kind: 'note'|'cadence-accept'|'cadence-reject'|'unlock-bloom',
//   notes: [{ freqHz, amp, detuneCents, decaySec, delaySec }],
//   timbre: 'darkpluck'|'bell'|'hollow'|'glass',
//   brightness: 0..1,   // lowpass-ish macro: falls as tension rises
//   tension: 0..1,      // for the drone layer / visuals; 0 = resolved home
// }

(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.heldbreath = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var ROOT_HZ = 110; // A2 — low, dark register
  // A hirajoshi: A B C E F — the dark pentatonic the night palette asks for.
  var BASE_SCALE = [0, 2, 3, 7, 8];
  // Unlock tiers each admit one color degree into the scale (revelation =
  // the space literally gains notes): b7 (G), then #4 shade (D#), then D.
  var UNLOCK_DEGREES = [10, 6, 5];

  // Press = motion, not a key: interval RELATIVE to the melody so far.
  var STEP = { U: 2, R: 1, L: -1, D: -2 };

  // Region = first press of the code; it owns the timbre family.
  var REGION_TIMBRE = { L: 'darkpluck', D: 'bell', R: 'hollow', U: 'glass' };

  function scaleFor(unlocked) {
    var s = BASE_SCALE.slice();
    for (var i = 0; i < Math.min(unlocked | 0, UNLOCK_DEGREES.length); i++)
      s.push(UNLOCK_DEGREES[i]);
    return s.sort(function (a, b) { return a - b; });
  }

  function degreeToHz(deg, scale) {
    var n = scale.length;
    var oct = Math.floor(deg / n);
    var st = scale[((deg % n) + n) % n] + 12 * oct;
    return ROOT_HZ * Math.pow(2, st / 12);
  }

  // Fold the path to the current melodic degree. Start one octave above the
  // root (mid register), clamp to ~3 octaves so deep codes can't run away.
  function degreeAfter(path, scale) {
    var deg = scale.length; // 1 octave up
    for (var i = 0; i < path.length; i++) {
      deg += STEP[path[i]] || 0;
      deg = Math.max(0, Math.min(3 * scale.length - 1, deg));
    }
    return deg;
  }

  function tensionAt(depth) { return Math.min(1, depth / 8); }

  return function heldbreath(comboState, press) {
    var path = (comboState && comboState.path) || [];
    var scale = scaleFor((comboState && comboState.unlocked) || 0);
    var depth = path.length;
    var t = tensionAt(depth);
    var timbre = REGION_TIMBRE[path[0] || press] || 'darkpluck';
    var homeHz = ROOT_HZ * 2;

    if (press === 'accept') {
      // Authentic cadence: exhale. Home chord (root, fifth, octave) staggered,
      // detune collapses to zero, long release.
      return {
        kind: 'cadence-accept', timbre: timbre, brightness: 0.9, tension: 0,
        notes: [
          { freqHz: ROOT_HZ,     amp: 0.55, detuneCents: 0, decaySec: 3.2, delaySec: 0 },
          { freqHz: homeHz * Math.pow(2, 7 / 12) / 2, amp: 0.4, detuneCents: 0, decaySec: 3.0, delaySec: 0.10 },
          { freqHz: homeHz,      amp: 0.5,  detuneCents: 0, decaySec: 3.4, delaySec: 0.22 },
        ],
      };
    }

    if (press === 'reject') {
      // Deceptive cadence: lands a half-step off home, keeps the accumulated
      // detune, damps early — the breath is not released.
      return {
        kind: 'cadence-reject', timbre: timbre, brightness: 0.35, tension: t,
        notes: [
          { freqHz: homeHz * Math.pow(2, 1 / 12), amp: 0.5, detuneCents: 6 * depth, decaySec: 1.1, delaySec: 0 },
          { freqHz: homeHz * Math.pow(2, -5 / 12), amp: 0.35, detuneCents: 6 * depth, decaySec: 1.3, delaySec: 0.07 },
        ],
      };
    }

    if (press === 'unlock') {
      // Bloom: tension gradient run in reverse — the arpeggio STARTS wide and
      // converges to pure as it climbs into the newly admitted degree.
      var notes = [];
      var top = degreeAfter(path, scale) + scale.length;
      for (var i = 0; i < 5; i++) {
        var frac = i / 4;
        notes.push({
          freqHz: degreeToHz(top - 4 + i, scale),
          amp: 0.3 + 0.08 * i,
          detuneCents: 24 * (1 - frac),
          decaySec: 1.2 + 1.6 * frac,
          delaySec: 0.16 * i,
        });
      }
      return { kind: 'unlock-bloom', timbre: 'glass', brightness: 1, tension: 0, notes: notes };
    }

    // An ordinary press: one interval of motion from where the melody stands.
    var deg = degreeAfter(path.concat([press]), scale);
    return {
      kind: 'note', timbre: timbre,
      brightness: Math.max(0.25, 1 - 0.09 * depth),
      tension: tensionAt(depth + 1),
      notes: [{
        freqHz: degreeToHz(deg, scale),
        amp: 0.5,
        detuneCents: 6 * (depth + 1),          // the gradient: beating grows
        decaySec: Math.max(0.4, 1.1 - 0.08 * depth), // breath tightens
        delaySec: 0,
      }],
    };
  };
});
