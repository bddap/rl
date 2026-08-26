// Static server for the assembled web bundle ($DIST_DIR): exactly what a real host
// serves — index.html, pkg/, assets.pack. Dev/verify tool; deploy.sh owns hosting.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const PORT = Number(process.env.PORT || 8643);
const DIST_DIR = process.env.DIST_DIR;
if (!DIST_DIR) { console.error('DIST_DIR unset'); process.exit(1); }

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.txt': 'text/plain', '.pack': 'application/octet-stream',
};

http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/index.html';
  // Resolve within the bundle root; reject traversal.
  const file = path.resolve(DIST_DIR, p.slice(1));
  if (!file.startsWith(path.resolve(DIST_DIR) + path.sep)) {
    res.writeHead(403); res.end(); return;
  }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(PORT, '127.0.0.1', () => console.log(`serving on http://127.0.0.1:${PORT}`));
