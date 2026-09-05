// Headless-chromium solo-play probe (rl#411). Drives the REAL menu with
// REAL key events over CDP: boot → menu renders → keyboard-select "Play solo" →
// round arms → frames flow → WASD held. Artifacts: screenshots + full console log.
// Exit 0 only if every gate below passed.
//
// Zero-network assert lives HERE, at the page layer (CDP Network domain): every
// request the PAGE makes — fetch, XHR — must target the bundle's own origin (the
// local dev server or the live host), and a WebSocket (an iroh relay dial would be
// one) is a violation outright — solo play has no business opening any. Chromium's
// own service phone-homes (accounts/GCM/network-time) are browser plumbing outside
// the page and are further neutered by run.sh's flags + the ~NOTFOUND resolver rule,
// which blocks any real off-host contact at the socket layer.
import fs from 'node:fs';

const [url, outdir] = process.argv.slice(2);
const DEBUG = 'http://127.0.0.1:9333';
const deadline = Date.now() + 240_000;

const consoleLines = [];
let ws, nextId = 1;
const pending = new Map();

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForLine(substr, label) {
  process.stderr.write(`waiting for: ${label}\n`);
  for (;;) {
    if (consoleLines.some((l) => l.includes(substr))) return;
    const panic = consoleLines.find((l) => l.includes('panicked at'));
    if (panic) throw new Error(`page panicked while waiting for ${label}: ${panic}`);
    if (Date.now() > deadline) throw new Error(`timeout waiting for ${label}`);
    await sleep(250);
  }
}

async function screenshot(name) {
  const { data } = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(`${outdir}/${name}`, Buffer.from(data, 'base64'));
  process.stderr.write(`screenshot: ${name}\n`);
}

async function key(code, keyName, type) {
  // `code`/`key` are what winit's DOM listeners read; no legacy vkey needed.
  await send('Input.dispatchKeyEvent', { type, code, key: keyName });
}
async function tap(code, keyName) {
  await key(code, keyName, 'keyDown');
  await sleep(80);
  await key(code, keyName, 'keyUp');
  await sleep(250);
}

const listRes = await fetch(`${DEBUG}/json`);
const targets = await listRes.json();
const page = targets.find((t) => t.type === 'page');
if (!page) throw new Error('no page target');
ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
const pageRequests = [];
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
  } else if (msg.method === 'Runtime.consoleAPICalled') {
    const line = msg.params.args.map((a) => a.value ?? a.description ?? '').join(' ');
    consoleLines.push(line);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    consoleLines.push('EXCEPTION ' + JSON.stringify(msg.params.exceptionDetails?.exception?.description ?? msg.params));
  } else if (msg.method === 'Network.requestWillBeSent') {
    pageRequests.push(msg.params.request.url);
  } else if (msg.method === 'Network.webSocketCreated') {
    pageRequests.push('ws: ' + msg.params.url);
  }
};
await send('Runtime.enable');
await send('Network.enable');
await send('Page.enable');
// rl#419: Web Audio instrumentation, installed before any page script runs. The
// page creates its AudioContext through the global constructor and drives output
// by starting AudioBufferSourceNodes, so wrapping those two counts what the API
// itself sees — no console-log matching.
await send('Page.addScriptToEvaluateOnNewDocument', { source: `
  window.__gcrAudio = { contexts: [], starts: 0 };
  window.AudioContext = class extends AudioContext {
    constructor(...args) { super(...args); window.__gcrAudio.contexts.push(this); }
  };
  const start = AudioBufferSourceNode.prototype.start;
  AudioBufferSourceNode.prototype.start = function (...args) {
    window.__gcrAudio.starts += 1;
    return start.apply(this, args);
  };
` });
await send('Page.navigate', { url });

async function evaluate(expression) {
  const { result } = await send('Runtime.evaluate', { expression, returnByValue: true });
  return result.value;
}
const loadingOverlayPresent = () => evaluate("!!document.getElementById('gcr-loading')");
const focusedOnCanvas = () => evaluate("document.activeElement === document.getElementById('gcr-canvas')");
const audioStates = () => evaluate('window.__gcrAudio.contexts.map((c) => c.state)');
const audioStarts = () => evaluate('window.__gcrAudio.starts');
async function click() {
  for (const type of ['mousePressed', 'mouseReleased'])
    await send('Input.dispatchMouseEvent', { type, x: 400, y: 225, button: 'left', clickCount: 1 });
}
const synthetic = (ctor, type, init) => evaluate(
  `!document.getElementById('gcr-canvas').dispatchEvent(new ${ctor}(${JSON.stringify(type)}, ${JSON.stringify(init)}))`);

// rl#413: the loading overlay is the page's only boot feedback — it must exist
// while the bundle downloads and be gone once real frames flow. Poll: navigation
// may not have committed yet, but removal can't beat the poll (it waits on wasm
// boot + first frames, seconds away).
async function awaitLoadingOverlay() {
  const until = Date.now() + 10_000;
  while (!(await loadingOverlayPresent())) {
    if (Date.now() > until) throw new Error('no #gcr-loading overlay during load (rl#413 regression)');
    await sleep(100);
  }
}

try {
  await awaitLoadingOverlay();
  await waitForLine('WEB_ASSETS_PRELOADED', 'asset prefetch');
  await waitForLine('WEB_FRAMETIME', 'first frame-rate snapshot (menu rendering)');
  await sleep(1500);
  if (await loadingOverlayPresent()) throw new Error('#gcr-loading overlay still up after frames flow (rl#413 regression)');
  await screenshot('menu.png');

  // rl#419: chromium's autoplay policy holds an AudioContext created before the
  // first gesture in `suspended` — asserted BEFORE any synthetic input (a key tap is
  // a gesture too) so the running-state check after the click cannot pass
  // vacuously; a bypass flag on the browser would show up here.
  {
    const states = await audioStates();
    if (states.length === 0) throw new Error('no AudioContext created by the page (rl#419)');
    if (!states.every((s) => s === 'suspended'))
      throw new Error(`AudioContext ${states} before any gesture — autoplay policy not enforced, the audio check is vacuous (rl#419)`);
  }
  const startsBeforeGesture = await audioStarts();

  // rl#418: winit owns canvas focus + preventDefault only while bevy's
  // prevent_default_event_handling stays on; this is what catches it flipping off.
  if (!(await focusedOnCanvas())) throw new Error('canvas not focused from the first frames (rl#418)');
  await tap('Tab', 'Tab');
  if (!(await focusedOnCanvas())) throw new Error('Tab moved focus off the canvas (rl#418)');
  if (!(await synthetic('KeyboardEvent', 'keydown', { key: 'Tab', code: 'Tab', bubbles: true, cancelable: true })))
    throw new Error('Tab keydown on the canvas not defaultPrevented (rl#418)');
  if (!(await synthetic('MouseEvent', 'contextmenu', { button: 2, bubbles: true, cancelable: true })))
    throw new Error('contextmenu on the canvas not defaultPrevented (rl#418)');
  await evaluate("document.getElementById('gcr-canvas').blur()");
  if (await focusedOnCanvas()) throw new Error('blur() left the canvas focused — the focus probe is vacuous');
  await click();
  await sleep(250);
  if (!(await focusedOnCanvas())) throw new Error('click did not refocus the canvas (rl#418)');
  process.stderr.write('FOCUS_OK canvas keeps focus across Tab and click; Tab + contextmenu defaultPrevented\n');

  await sleep(1000);
  {
    const states = await audioStates();
    const starts = await audioStarts();
    if (!states.every((s) => s === 'running'))
      throw new Error(`AudioContext ${states} after the click — the gesture did not resume it (rl#419)`);
    if (starts <= startsBeforeGesture)
      throw new Error(`no source node started after the gesture (${starts} before and after) — the output chain is stalled (rl#419)`);
    process.stderr.write(`AUDIO_OK AudioContext running after the click; ${starts - startsBeforeGesture} source nodes started since\n`);
  }

  // Chooser: focus starts on Host; one Down lands on "Play solo (offline)".
  await tap('ArrowDown', 'ArrowDown');
  await tap('Enter', 'Enter');
  await waitForLine('ROUND_ARMED net=solo', 'solo round armed by keyboard');

  // Let the round render, then prove movement input reaches the sim: walk,
  // strafe, and hop, and require the rendered view to actually change (a spawn
  // facing a steep slope can make a pure W-hold a no-op).
  await sleep(4000);
  await screenshot('round.png');
  await key('KeyW', 'w', 'keyDown');
  await sleep(4000);
  await key('KeyW', 'w', 'keyUp');
  await key('KeyD', 'd', 'keyDown');
  await sleep(2500);
  await key('KeyD', 'd', 'keyUp');
  await tap('Space', ' ');
  await sleep(1200);
  await screenshot('round-moved.png');
  await sleep(1500);
  const before = fs.readFileSync(`${outdir}/round.png`);
  const after = fs.readFileSync(`${outdir}/round-moved.png`);
  if (before.equals(after)) throw new Error('movement keys produced a pixel-identical view — input not reaching the sim?');
  process.stderr.write('MOVEMENT_OK rendered view changed under WASD+jump\n');

  const frametime = consoleLines.filter((l) => l.includes('WEB_FRAMETIME'));
  if (frametime.length < 3) throw new Error(`only ${frametime.length} WEB_FRAMETIME snapshots`);
  process.stderr.write('frame-rate report (1 Hz snapshots):\n');
  for (const l of frametime.slice(-8)) process.stderr.write(`  ${l}\n`);

  const audio = consoleLines.filter((l) => /audio|AudioContext/i.test(l));
  for (const l of audio.slice(0, 5)) process.stderr.write(`audio: ${l}\n`);

  const panic = consoleLines.find((l) => l.includes('panicked at'));
  if (panic) throw new Error(`page panicked: ${panic}`);

  const origin = new URL(url).origin;
  const nonLocal = pageRequests.filter(
    (u) => u.startsWith('ws: ') || (!u.startsWith(origin + '/') && !u.startsWith('data:')),
  );
  const relayish = consoleLines.filter((l) => /\b(relay|relays|dial|dialing|iroh)\b|endpoint bind/i.test(l));
  if (nonLocal.length) throw new Error('page made non-local requests:\n  ' + nonLocal.join('\n  '));
  if (relayish.length) throw new Error('console shows link activity in solo:\n  ' + relayish.join('\n  '));
  process.stderr.write(`NETCHECK_OK page made ${pageRequests.length} requests, all same-origin (${origin}); no link activity logged\n`);

  // rl#419, the other ordering: the gesture lands while the bundle is still
  // downloading, before the wasm creates its AudioContext. The page must come up
  // audible with no further input. A fresh navigation resets user activation.
  consoleLines.length = 0;
  await send('Page.navigate', { url });
  await awaitLoadingOverlay();
  await click();
  if ((await audioStates()).length !== 0)
    throw new Error('boot created its AudioContext before the click landed — the gesture-first ordering was not exercised (rl#419)');
  await waitForLine('WEB_FRAMETIME', 'first frame-rate snapshot after a gesture-first reload');
  await sleep(1000);
  {
    const states = await audioStates();
    if (states.length === 0) throw new Error('no AudioContext created after the gesture-first reload (rl#419)');
    if (!states.every((s) => s === 'running'))
      throw new Error(`AudioContext ${states} when the gesture came before boot — the page stays silent (rl#419)`);
    process.stderr.write('AUDIO_OK AudioContext running with the gesture before boot\n');
  }
  process.stderr.write('VERIFY_PLAY_OK\n');
} finally {
  fs.writeFileSync(`${outdir}/console.log`, consoleLines.join('\n') + '\n');
  await send('Browser.close').catch(() => {});
  ws.close();
}
