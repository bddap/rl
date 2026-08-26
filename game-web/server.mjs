// Static server for the web bundle: the game-web dir (index.html, pkg/,
// web-assets.txt) plus /assets/* mapped onto $ASSETS_DIR. Dev/verify tool — the
// deploy stage owns real hosting.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const PORT = Number(process.env.PORT || 8643);
const ASSETS_DIR = process.env.ASSETS_DIR;
if (!ASSETS_DIR) { console.error('ASSETS_DIR unset'); process.exit(1); }

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.txt': 'text/plain', '.png': 'image/png', '.wav': 'audio/wav',
  '.glb': 'model/gltf-binary', '.json': 'application/json',
};

http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/index.html';
  // Resolve within the intended root; reject traversal.
  const root = p.startsWith('/assets/') ? ASSETS_DIR : process.cwd();
  const rel = p.startsWith('/assets/') ? p.slice('/assets/'.length) : p.slice(1);
  const file = path.resolve(root, rel);
  if (!file.startsWith(path.resolve(root) + path.sep) && file !== path.resolve(root, 'index.html')) {
    res.writeHead(403); res.end(); return;
  }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(PORT, '127.0.0.1', () => console.log(`serving on http://127.0.0.1:${PORT}`));
