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
await send('Page.navigate', { url });

async function loadingOverlayPresent() {
  const { result } = await send('Runtime.evaluate', {
    expression: "!!document.getElementById('gcr-loading')",
    returnByValue: true,
  });
  return result.value;
}

try {
  // rl#413: the loading overlay is the page's only boot feedback — it must exist
  // while the bundle downloads and be gone once real frames flow. Poll: navigation
  // may not have committed yet, but removal can't beat the poll (it waits on wasm
  // boot + first frames, seconds away).
  {
    const until = Date.now() + 10_000;
    while (!(await loadingOverlayPresent())) {
      if (Date.now() > until) throw new Error('no #gcr-loading overlay during load (rl#413 regression)');
      await sleep(100);
    }
  }
  await waitForLine('WEB_ASSETS_PRELOADED', 'asset prefetch');
  await waitForLine('WEB_FRAMETIME', 'first frame-rate snapshot (menu rendering)');
  await sleep(1500);
  if (await loadingOverlayPresent()) throw new Error('#gcr-loading overlay still up after frames flow (rl#413 regression)');
  await screenshot('menu.png');

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
  process.stderr.write('VERIFY_PLAY_OK\n');
} finally {
  fs.writeFileSync(`${outdir}/console.log`, consoleLines.join('\n') + '\n');
  await send('Browser.close').catch(() => {});
  ws.close();
}
