// Offline renderer for the patchwalk sketches. Every sound in every sketch is
// produced by calling scheme() — the renderer only interprets soundEvents, so the
// audio demonstrates the actual mapping, not a hand-tuned imitation of it.
// Usage: node render.js  (writes sketches/*.wav; encode with ffmpeg afterwards)

"use strict";
const fs = require("fs");
const path = require("path");
const { scheme } = require("./scheme.js");

const SR = 44100;

// ---------- voice synth: inharmonic partial pluck ----------
function renderVoice(v, veiled) {
  const { freq, cents, amp, dur, patch } = v;
  const { brightness, inharm, breath, warmth, shimmer } = patch;
  const n = Math.floor(SR * (dur + 0.5));
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  const B = inharm * 0.004; // partial stretch
  const nPartials = 14;
  const detunes = [-cents, cents]; // split the spread across the stereo pair
  for (let side = 0; side < 2; side++) {
    const ch = side === 0 ? L : R;
    const f0 = freq * Math.pow(2, detunes[side] / 1200);
    for (let k = 1; k <= nPartials; k++) {
      const ratio = k * Math.sqrt(1 + B * k * k);
      const fk = f0 * ratio;
      if (fk > SR / 2 - 1000) break;
      // brightness sets rolloff; warmth boosts the fundamental & 2nd
      let a = Math.pow(1 / k, 1.4 + (1 - brightness) * 1.8);
      if (k <= 2) a *= 1 + warmth * 0.8;
      if (veiled) a *= Math.pow(1 / k, 2.2); // veil: steep extra rolloff
      const decay = dur * (k === 1 ? 1 : 1 / (1 + k * (0.25 + (1 - brightness) * 0.35)));
      const w = (2 * Math.PI * fk) / SR;
      const phase = (k * 0.7 + side * 1.3) % (2 * Math.PI);
      const shimRate = (2 * Math.PI * (0.9 + side * 0.35)) / SR;
      for (let i = 0; i < n; i++) {
        const t = i / SR;
        const env = Math.exp(-t / (decay * 0.35)) * Math.min(1, t / 0.004);
        let s = Math.sin(w * i + phase) * a * env;
        if (shimmer > 0 && k <= 4) s *= 1 + shimmer * 0.5 * Math.sin(shimRate * i + k);
        ch[i] += s;
      }
    }
    // breath: short filtered noise chiff
    if (breath > 0.02) {
      let lp = 0;
      const cut = 0.05 + brightness * 0.25;
      for (let i = 0; i < Math.min(n, SR * 0.09); i++) {
        const t = i / SR;
        lp += cut * ((Math.sin(i * 12.9898 + side * 78.233) * 43758.5453) % 1 - lp);
        ch[i] += lp * breath * 0.8 * Math.exp(-t / 0.03);
      }
    }
  }
  // normalize-ish and apply amp
  let peak = 1e-9;
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
  const g = (amp * (veiled ? 0.5 : 1)) / peak;
  for (let i = 0; i < n; i++) { L[i] *= g; R[i] *= g; }
  return { L, R };
}

// ---------- event -> mix at time t ----------
function mixEvent(bufL, bufR, t0, ev) {
  for (const v of ev.voices) {
    const { L, R } = renderVoice(v, ev.veiled);
    const off = Math.floor((t0 + v.delay) * SR);
    for (let i = 0; i < L.length && off + i < bufL.length; i++) {
      bufL[off + i] += L[i];
      bufR[off + i] += R[i];
    }
  }
}

// ---------- Schroeder reverb (dark, damped — the nightish room) ----------
function reverb(dry, wet, sr, offsetSamp) {
  const combs = [1687, 1601, 2053, 2251].map((d) => ({
    buf: new Float32Array(d + offsetSamp), i: 0, fb: 0.80, damp: 0.4, store: 0,
  }));
  const aps = [225, 556].map((d) => ({ buf: new Float32Array(d), i: 0, g: 0.5 }));
  const out = new Float32Array(dry.length);
  for (let i = 0; i < dry.length; i++) {
    const x = dry[i];
    let s = 0;
    for (const c of combs) {
      const y = c.buf[c.i];
      c.store = y * (1 - c.damp) + c.store * c.damp;
      c.buf[c.i] = x + c.store * c.fb;
      c.i = (c.i + 1) % c.buf.length;
      s += y;
    }
    s /= combs.length;
    for (const a of aps) {
      const y = a.buf[a.i];
      const inn = s + y * a.g;
      a.buf[a.i] = inn;
      a.i = (a.i + 1) % a.buf.length;
      s = y - inn * a.g;
    }
    out[i] = dry[i] + s * wet;
  }
  return out;
}

// ---------- WAV out ----------
function writeWav(file, L, R) {
  let peak = 1e-9;
  for (let i = 0; i < L.length; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
  const g = 0.89 / peak; // -1 dBFS headroom
  const data = Buffer.alloc(L.length * 4);
  for (let i = 0; i < L.length; i++) {
    data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, L[i] * g)) * 32767), i * 4);
    data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, R[i] * g)) * 32767), i * 4 + 2);
  }
  const h = Buffer.alloc(44);
  h.write("RIFF", 0); h.writeUInt32LE(36 + data.length, 4); h.write("WAVE", 8);
  h.write("fmt ", 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(2, 22);
  h.writeUInt32LE(SR, 24); h.writeUInt32LE(SR * 4, 28); h.writeUInt16LE(4, 32); h.writeUInt16LE(16, 34);
  h.write("data", 36); h.writeUInt32LE(data.length, 40);
  fs.writeFileSync(file, Buffer.concat([h, data]));
}

// ---------- sequencing: a sketch is [(t, path-so-far, press, unlocked)] ----------
function renderSketch(name, events, seconds) {
  const n = Math.floor(SR * seconds);
  let L = new Float32Array(n);
  let R = new Float32Array(n);
  for (const [t, pathSoFar, press, unlocked] of events) {
    mixEvent(L, R, t, scheme({ path: pathSoFar, unlocked: unlocked || [""] }, press));
  }
  L = reverb(L, 0.4, SR, 0);
  R = reverb(R, 0.4, SR, 23);
  const file = path.join(__dirname, "sketches", name + ".wav");
  writeWav(file, L, R);
  console.log("wrote", file);
}

// helper: play a code from the root at a tempo, then resolve
function playCode(code, t0, dt, unlocked, resolve = "accept") {
  const ev = [];
  const dirs = code.split("");
  for (let i = 0; i < dirs.length; i++) ev.push([t0 + i * dt, dirs.slice(0, i), dirs[i], unlocked]);
  if (resolve) ev.push([t0 + dirs.length * dt + 0.15, dirs, resolve, unlocked]);
  return ev;
}

const ALL = [""]; // "" prefix-covers everything: fully unlocked world

// (a) shallow code: three presses, an easy phrase, accepted
renderSketch("a-shallow-code", playCode("RUL", 0.6, 0.48, ALL), 8);

// (b) deep code: seven presses — register sinks, detune widens, timbre narrows in
renderSketch("b-deep-code", playCode("DRDLUDR", 0.6, 0.44, ALL), 12);

// (c) two paths, same destination: LUR vs RUL — same multiset, so the walk
// converges on the same patch+pitch, but the journeys are different performances
renderSketch(
  "c-two-paths-same-node",
  [...playCode("LUR", 0.6, 0.48, ALL), ...playCode("RUL", 5.6, 0.48, ALL)],
  11
);

// (d) unlock: knock on veiled territory (muffled, distant), the veil lifts,
// then the same code played open — the region now answers in full voice
renderSketch(
  "d-unlock-event",
  [
    ...playCode("DR", 0.6, 0.5, ["", "L", "R", "U"], "reject"), // D-subtree locked: veiled taps, refused
    [3.4, ["D", "R"], "unlock", ALL],                             // the bloom
    ...playCode("DRD", 6.4, 0.48, ALL),                           // territory open, code lands
  ],
  13
);

// (e) region tour: one step into each of the four territories — glass, air, wood, depth
renderSketch(
  "e-region-tour",
  [
    ...playCode("LL", 0.6, 0.5, ALL, null),
    ...playCode("UU", 2.8, 0.5, ALL, null),
    ...playCode("RR", 5.0, 0.5, ALL, null),
    ...playCode("DD", 7.2, 0.5, ALL, null),
  ],
  11
);
