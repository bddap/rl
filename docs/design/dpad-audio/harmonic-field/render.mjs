// Render harness for the HARMONIC FIELD sketches (rl#359 design exploration).
// Feeds press sequences through scheme.js — the exact module the playground
// gets — and synthesizes the resulting soundEvents to WAV.
//   node render.mjs   -> sketches/*.wav   (encode with ffmpeg after)
import { writeFileSync } from 'node:fs';
import scheme from './scheme.js';

const SR = 44100;

// ---- voice synth: soft pluck — sine + gentle harmonics, exp decay ----------
function renderVoice(buf, t0, v) {
  const start = Math.floor((t0 + v.at) * SR);
  const dur = Math.min(v.decay * 5, 6);
  const n = Math.min(Math.floor(dur * SR), buf.length - start);
  const w = 2 * Math.PI * v.freq / SR;
  const w2 = w * Math.pow(2, v.detuneCents / 1200);
  const atk = Math.floor(0.008 * SR);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const env = (i < atk ? i / atk : 1) * Math.exp(-t / v.decay);
    const envH = env * Math.exp(-t * 3); // harmonics die faster
    let s = Math.sin(w * i) * 0.6 + Math.sin(w2 * i) * 0.4;
    s += v.brightness * (0.30 * Math.sin(2 * w * i) + 0.10 * Math.sin(3 * w * i)) * (envH / (env || 1));
    buf[start + i] += s * env * v.gain;
  }
}

// ---- tiny reverb: two feedback combs + wet mix -----------------------------
function reverb(buf) {
  const wet = new Float64Array(buf.length);
  for (const [d, g] of [[Math.floor(0.041 * SR), 0.38], [Math.floor(0.053 * SR), 0.33]]) {
    const line = new Float64Array(d);
    for (let i = 0; i < buf.length; i++) {
      const j = i % d;
      const out = line[j];
      line[j] = buf[i] + out * g;
      wet[i] += out;
    }
  }
  for (let i = 0; i < buf.length; i++) buf[i] = buf[i] + wet[i] * 0.22;
}

function writeWav(path, buf) {
  let peak = 0;
  for (const s of buf) peak = Math.max(peak, Math.abs(s));
  const norm = 0.85 / (peak || 1);
  const pcm = Buffer.alloc(buf.length * 2);
  for (let i = 0; i < buf.length; i++) {
    pcm.writeInt16LE(Math.round(Math.tanh(buf[i] * norm * 1.1) * 32000), i * 2);
  }
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + pcm.length, 4); h.write('WAVE', 8);
  h.write('fmt ', 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20);
  h.writeUInt16LE(1, 22); h.writeUInt32LE(SR, 24); h.writeUInt32LE(SR * 2, 28);
  h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34);
  h.write('data', 36); h.writeUInt32LE(pcm.length, 40);
  writeFileSync(path, Buffer.concat([h, pcm]));
}

// ---- sequencer: walk a script of {t, press, state} through the scheme ------
function renderScript(name, seconds, events) {
  const buf = new Float64Array(Math.ceil(seconds * SR));
  for (const e of events) {
    const ev = scheme(e.state, e.press);
    for (const v of ev.notes) renderVoice(buf, e.t, v);
  }
  reverb(buf);
  writeWav(`sketches/${name}.wav`, buf);
  console.log(`${name}: ${events.length} events, ${seconds}s`);
}

// Helper: type a code starting at t; returns events + final path.
// gap ~human entry cadence, slight lilt so it phrases.
function typeCode(code, t, tier, gap = 0.20) {
  const events = [];
  const path = [];
  for (const p of code) {
    events.push({ t, press: p, state: { path: [...path], unlockTier: tier } });
    path.push(p);
    t += gap * (0.92 + 0.16 * (path.length % 2)); // lilt
  }
  return { events, path, tEnd: t };
}

function codeWithAccept(code, t, tier, gap) {
  const { events, path, tEnd } = typeCode(code, t, tier, gap);
  events.push({ t: tEnd + 0.15, press: 'accept', state: { path, unlockTier: tier } });
  return { events, tEnd: tEnd + 0.15 };
}

// ---- sketch a: shallow code (R,U), heard twice — a 2-note motif + bloom ----
{
  const a = codeWithAccept(['R', 'U'], 0.5, 1);
  const b = codeWithAccept(['R', 'U'], 4.6, 1);
  renderScript('a-shallow-code', 9, [...a.events, ...b.events]);
}

// ---- sketch b: deep code (7 presses) — haze and pad density grow with depth
{
  const code = ['D', 'D', 'L', 'U', 'L', 'D', 'R'];
  const a = codeWithAccept(code, 0.5, 1, 0.21);
  const b = codeWithAccept(code, 7.6, 1, 0.19);
  renderScript('b-deep-code', 14, [...a.events, ...b.events]);
}

// ---- sketch c: two paths, one node — RRUU vs URUR both land on (2,2):
// different melodies, identical arrival harmony (accept chord matches) ------
{
  const p1 = codeWithAccept(['R', 'R', 'U', 'U'], 0.5, 1);
  const p2 = codeWithAccept(['U', 'R', 'U', 'R'], 6.5, 1);
  renderScript('c-two-paths-one-node', 12, [...p1.events, ...p2.events]);
}

// ---- sketch d: unlock — same code at tier 0, the field gains its notes
// (unlock shimmer), then the same code again at tier 2: audibly richer ------
{
  const code = ['L', 'D', 'U', 'D'];
  const before = codeWithAccept(code, 0.5, 0);
  const unlock = { t: 6.2, press: 'unlock', state: { path: [], unlockTier: 2 } };
  const after = codeWithAccept(code, 9.8, 2);
  renderScript('d-unlock-bloom', 16, [...before.events, unlock, ...after.events]);
}

// ---- sketch e: free wander — no code in mind, just walking the field;
// the thesis: aimless d-pad input still comes out as a coherent piece -------
{
  const walk = ['U', 'U', 'R', 'D', 'U', 'L', 'L', 'D', 'D', 'R', 'U', 'R', 'R', 'U', 'D', 'L', 'U', 'U'];
  const gaps = [.4, .2, .5, .2, .2, .7, .2, .3, .2, .5, .2, .2, .4, .2, .2, .6, .2, .2];
  const events = [];
  const path = [];
  let t = 0.6;
  walk.forEach((p, i) => {
    events.push({ t, press: p, state: { path: [...path], unlockTier: 3 } });
    path.push(p);
    t += gaps[i] * 1.5;
  });
  renderScript('e-wander', 15, events);
}
