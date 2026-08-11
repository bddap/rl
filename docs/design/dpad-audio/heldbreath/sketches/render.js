#!/usr/bin/env node
// Renders the heldbreath sketches from scheme.js (the SAME pure mapping the
// playground gets). Pure synthesis → WAV; ffmpeg converts to ogg.
'use strict';
const fs = require('fs');
const path = require('path');
const heldbreath = require('../scheme.js');

const SR = 44100;

// Partial sets per timbre: [ratio, amp] pairs. Bell/glass are inharmonic.
const TIMBRES = {
  darkpluck: [[1, 1], [2, 0.4], [3, 0.18], [5, 0.06]],
  bell: [[1, 1], [2.756, 0.32], [5.404, 0.1]],
  hollow: [[1, 1], [3, 0.28], [5, 0.09]],
  glass: [[1, 1], [3.01, 0.22], [4.16, 0.14], [6.7, 0.05]],
};

function mkBuf(sec) {
  return [new Float64Array(Math.ceil(sec * SR)), new Float64Array(Math.ceil(sec * SR))];
}

// One decaying additive voice into buf at time t0, panned.
function voice(buf, t0, freq, amp, decaySec, partials, brightness, pan) {
  const n0 = Math.floor(t0 * SR);
  const len = Math.min(Math.ceil(decaySec * 4 * SR), buf[0].length - n0);
  const gl = Math.sqrt(0.5 * (1 - pan)), gr = Math.sqrt(0.5 * (1 + pan));
  for (let i = 0; i < len; i++) {
    const t = i / SR;
    const env = Math.min(t / 0.006, 1) * Math.exp(-t / (decaySec * 0.55));
    let s = 0;
    for (let p = 0; p < partials.length; p++) {
      const [ratio, pa] = partials[p];
      if (freq * ratio > SR * 0.45) continue;
      // brightness closes the top of the spectrum as tension rises
      const ba = pa * Math.pow(brightness, p) * Math.exp(-t * ratio * 0.9);
      s += ba * Math.sin(2 * Math.PI * freq * ratio * t);
    }
    const v = amp * env * s;
    buf[0][n0 + i] += v * gl;
    buf[1][n0 + i] += v * gr;
  }
}

// A soundEvent from the scheme, realized. Detune = twin voices split L/R.
function playEvent(buf, t0, ev) {
  const partials = TIMBRES[ev.timbre] || TIMBRES.darkpluck;
  for (const n of ev.notes) {
    const d = (n.detuneCents || 0) / 2;
    const f1 = n.freqHz * Math.pow(2, -d / 1200);
    const f2 = n.freqHz * Math.pow(2, d / 1200);
    voice(buf, t0 + n.delaySec, f1, n.amp * 0.5, n.decaySec, partials, ev.brightness, -0.35);
    voice(buf, t0 + n.delaySec, f2, n.amp * 0.5, n.decaySec, partials, ev.brightness, 0.35);
  }
}

// Root drone: soft low A + octave, plus a minor-2nd shimmer whose level follows
// the tension curve — the "held breath" under the melody.
function drone(buf, t0, t1, tensionAt, release) {
  const n0 = Math.floor(t0 * SR), n1 = Math.min(Math.floor((t1 + release) * SR), buf[0].length);
  for (let i = n0; i < n1; i++) {
    const t = i / SR;
    const fadeIn = Math.min((t - t0) / 0.8, 1);
    const fadeOut = t > t1 ? Math.max(0, 1 - (t - t1) / release) : 1;
    const ten = tensionAt(Math.min(t, t1));
    const trem = 1 + 0.12 * Math.sin(2 * Math.PI * (0.9 + 2.2 * ten) * t);
    let s = 0.5 * Math.sin(2 * Math.PI * 55 * t) + 0.3 * Math.sin(2 * Math.PI * 110 * t);
    s += 0.55 * ten * trem * Math.sin(2 * Math.PI * 110 * Math.pow(2, 1 / 12) * t);
    const v = 0.085 * fadeIn * fadeOut * s;
    buf[0][i] += v; buf[1][i] += v;
  }
}

// Cheap space: three cross-panned echo taps.
function reverb(buf) {
  const taps = [[0.089, 0.24], [0.131, 0.17], [0.197, 0.11]];
  for (const [dt, g] of taps) {
    const d = Math.floor(dt * SR);
    for (let i = buf[0].length - 1; i >= d; i--) {
      buf[0][i] += g * buf[1][i - d];
      buf[1][i] += g * buf[0][i - d];
    }
  }
}

function writeWav(file, buf) {
  reverb(buf);
  let peak = 1e-9;
  for (let c = 0; c < 2; c++) for (const v of buf[c]) peak = Math.max(peak, Math.abs(v));
  const g = 0.89 / peak;
  const n = buf[0].length;
  const data = Buffer.alloc(44 + n * 4);
  data.write('RIFF', 0); data.writeUInt32LE(36 + n * 4, 4); data.write('WAVEfmt ', 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(2, 22);
  data.writeUInt32LE(SR, 24); data.writeUInt32LE(SR * 4, 28); data.writeUInt16LE(4, 32);
  data.writeUInt16LE(16, 34); data.write('data', 36); data.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) {
    data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, buf[0][i] * g)) * 32767), 44 + i * 4);
    data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, buf[1][i] * g)) * 32767), 46 + i * 4);
  }
  fs.writeFileSync(file, data);
}

// Play a code: fold presses through the scheme, tightening the press interval
// with depth, then the ending event. Returns tension keyframes for the drone.
function playCode(buf, t0, presses, ending, unlocked) {
  let t = t0;
  const keys = [[t0, 0]];
  const state = { path: [], unlocked: unlocked || 0 };
  for (const p of presses) {
    const ev = heldbreath(state, p);
    playEvent(buf, t, ev);
    keys.push([t, ev.tension]);
    state.path.push(p);
    t += Math.max(0.34, 0.52 - 0.022 * state.path.length);
  }
  t += 0.15;
  if (ending) {
    const ev = heldbreath(state, ending);
    playEvent(buf, t, ev);
    keys.push([t + 0.05, ev.tension]);
  }
  const tensionAt = (x) => {
    let v = 0;
    for (const [kt, kv] of keys) if (x >= kt) v = kv; else break;
    return v;
  };
  drone(buf, t0, t + 0.3, tensionAt, ending === 'accept' ? 2.2 : 0.7);
  return t;
}

const out = process.argv[2] || '.';
const sketches = {
  // (a) shallow code: 3 presses, calm, quick exhale.
  'a-shallow': (buf) => playCode(buf, 0.3, ['R', 'U', 'D'], 'accept'),
  // (b) deep code: 8 presses — beating widens, room darkens, breath tightens —
  // then the exhale.
  'b-deep': (buf) => playCode(buf, 0.3, ['L', 'U', 'R', 'U', 'D', 'R', 'U', 'D'], 'accept'),
  // (c) two paths to the same node: different contours, same destination,
  // same exhale. U+R+D and R+D+U both sum to +1.
  'c-twopaths': (buf) => {
    const t = playCode(buf, 0.3, ['U', 'R', 'D'], 'accept');
    playCode(buf, t + 3.2, ['R', 'D', 'U'], 'accept');
  },
  // (d) unknown code: 5 presses of build-up, deceptive cadence — no release.
  'd-unknown': (buf) => playCode(buf, 0.3, ['U', 'U', 'L', 'D', 'L'], 'reject'),
  // (e) unlock: a code lands, then the bloom — wide detune converging to pure
  // as it climbs into the newly admitted degree; then a short phrase USING the
  // unlocked scale.
  'e-unlock': (buf) => {
    let t = playCode(buf, 0.3, ['D', 'R', 'U'], 'accept');
    const ev = heldbreath({ path: ['D', 'R', 'U'], unlocked: 1 }, 'unlock');
    playEvent(buf, t + 2.4, ev);
    playCode(buf, t + 5.6, ['D', 'U', 'R', 'R'], 'accept', 1);
  },
};

const durations = { 'a-shallow': 8, 'b-deep': 12, 'c-twopaths': 15, 'd-unknown': 8, 'e-unlock': 14 };
for (const [name, fn] of Object.entries(sketches)) {
  const buf = mkBuf(durations[name]);
  fn(buf);
  writeWav(path.join(out, name + '.wav'), buf);
  console.log('rendered', name);
}
