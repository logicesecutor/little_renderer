// serve.mjs - a static file server for the web build (docs/WEB.md): the files of a folder,
// nothing else, with the two headers that make the page cross-origin isolated (COOP/COEP),
// which a page needs for SharedArrayBuffer, so for the threaded build (make web THREADS=1).
// Node 18+, nothing to install.
//
//   node tools/serve.mjs [folder] [-port 8000] [-host 127.0.0.1] [-noisolate]
//
// Also used by tools/webbench.mjs (-serve). Without -noisolate every response carries
//   Cross-Origin-Opener-Policy: same-origin, Cross-Origin-Embedder-Policy: require-corp
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { join, normalize, extname, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm',
                '.data': 'application/octet-stream', '.json': 'application/json', '.png': 'image/png', '.css': 'text/css' };

export function serve(folder, port = 8000, host = '127.0.0.1', isolate = true) {
  const root = resolve(folder);
  const server = createServer((req, res) => {
    let path;
    try { path = decodeURIComponent((req.url || '/').split('?')[0]); }
    catch { res.writeHead(400); res.end('bad request'); return; }   // a malformed escape ("/%")
    let file = normalize(join(root, path));
    // inside the folder: root itself or below it (not a sibling whose name begins the same)
    if (file !== root && !file.startsWith(root + sep)) { res.writeHead(403); res.end(); return; }
    try { if (statSync(file).isDirectory()) file = join(file, 'index.html'); } catch {}
    let size = 0;
    try { size = statSync(file).size; } catch { res.writeHead(404); res.end('not found'); return; }
    const h = { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream', 'Content-Length': size, 'Cache-Control': 'no-cache' };
    if (isolate) { h['Cross-Origin-Opener-Policy'] = 'same-origin'; h['Cross-Origin-Embedder-Policy'] = 'require-corp'; }
    res.writeHead(200, h);
    createReadStream(file).pipe(res);
  });
  return new Promise(ok => server.listen(port, host, () => ok(server)));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const argv = process.argv.slice(2);
  const opt = (name, def) => { const i = argv.indexOf(name); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def; };
  const folder = argv[0] && !argv[0].startsWith('-') ? argv[0] : 'web/build';
  const port = +opt('-port', 8000), host = opt('-host', '127.0.0.1'), isolate = !argv.includes('-noisolate');
  await serve(folder, port, host, isolate);
  console.log(`serving ${folder} on http://${host}:${port}/${isolate ? ' (cross-origin isolated: COOP/COEP)' : ''}`);
}
