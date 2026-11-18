// webbench.mjs - the performance suite in a browser (docs/STATUS.md 1.3, group J): opens
// web/build/bench.html in a headless Chrome of its own (a temporary profile, this machine's
// GPU), prints the results as the page makes them, and appends its CSV lines to a file.
// Node 22+, nothing to install. The page is served by -serve (tools/serve.mjs, in this
// process, cross-origin isolated) or by any server at -url (make serve: http://localhost:8000).
//
//   node tools/webbench.mjs [quick|full|check] [-only A,F3] [-tag label] [-o out/perf/x_web.csv]
//                           [-serve] [-page bench.html] [-url http://127.0.0.1:8000/bench.html]
//                           [-chrome path] [-q "a=b&c=d"] [-flags "--chrome-switch ..."] [-log file]
//                           [-shots dir]
//
// -shots dir: instead of the cases, the page's images (bench.c -shots: path traced, denoised,
// rasterized) as PNG files <dir>/<name>_web.png, to compare with bin/bench -shots dir's
// <name>_wgpu.png (bin/bench -imgdiff a.png b.png).
// check (make web-check): one round of the cases that make every pipeline the page uses (the
// path tracer with and without the sky, the guide, the light tracer, the rasterizer, the
// denoiser, the trees built on the GPU, the bodies, the air), failing on any GPU error or a
// skipped case.
// -q: more of the page's options (bench.c: env=NAME=value sets an environment variable);
// -flags: more switches for Chrome, for measuring only (Dawn's toggles:
// --enable-dawn-features=allow_unsafe_apis,disable_robustness); the page never needs them;
// -log: every console line of the page into a file.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync, appendFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './serve.mjs';

const argv = process.argv.slice(2);
const opt = (name, def) => { const i = argv.indexOf(name); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def; };
const check = argv.includes('check');
const set = argv.includes('full') ? 'full' : 'quick';
const only = opt('-only', check ? 'A1,A4,A5,A7,A8,C1,D1,E5,F3,G1' : ''), tag = opt('-tag', ''), out = opt('-o', '');
const log = opt('-log', ''), more = opt('-q', ''), shots = opt('-shots', '');
// -until text: any page (the viewer: -page index.html -q "sim=on&autoquit=8"): every line it
// prints is shown, and the run ends at the first line holding text
const until = opt('-until', '');
if (shots) mkdirSync(shots, { recursive: true });
let base = opt('-url', 'http://127.0.0.1:8000/bench.html');
const chromes = [opt('-chrome', ''), 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
                 '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'];
const exe = chromes.find(p => p && existsSync(p));
if (!exe) { console.error('no Chrome found (-chrome path)'); process.exit(1); }

let server = null;
if (argv.includes('-serve')) {
  const folder = join(dirname(fileURLToPath(import.meta.url)), '..', 'web', 'build');
  const port = 8400 + Math.floor(Math.random() * 400);
  server = await serve(folder, port, '127.0.0.1', !argv.includes('-noisolate'));   // (-noisolate: as a static host without the headers)
  base = `http://127.0.0.1:${port}/${opt('-page', 'bench.html')}`;
}

const q = until ? [] : [set === 'full' ? 'full' : 'quick'];   // (another page: only -q's options)
if (only) q.push('only=' + only);
if (tag) q.push('tag=' + tag);
if (check) q.push('reps=1');
if (shots) q.push('shots=page');
if (more) q.push(more);
const url = `${base}?${q.join('&')}`;
const dir = mkdtempSync(join(tmpdir(), 'lrbench-'));
const port = 9334;
const extra = opt('-flags', '').split(' ').filter(s => s);
const chrome = spawn(exe, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`, '--no-first-run',
                           '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu', ...extra, 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws;
for (let k = 0; k < 100 && !ws; k++) {
  try { const t = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); const p = t.find(x => x.type === 'page'); if (p) ws = new WebSocket(p.webSocketDebuggerUrl); } catch {}
  if (!ws) await sleep(200);
}
if (!ws) { console.error('Chrome did not start'); chrome.kill(); if (server) server.close(); process.exit(1); }
await new Promise(r => ws.onopen = r);
let id = 0, finished = false, rows = 0, errors = 0;
const send = (method, params = {}) => ws.send(JSON.stringify({ id: ++id, method, params }));
const bad = line => /GPU error|GPU lost|^GPU: |^RESULT .* skipped:|Aborted|RuntimeError/.test(line);
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.method === 'Runtime.consoleAPICalled') {
    const s = m.params.args.map(a => a.value ?? a.description ?? '').join(' ');
    if (log) appendFileSync(log, s + '\n');
    for (const line of s.split('\n')) {
      if (line.startsWith('PNG ') && shots) {
        const [, file, b64] = line.split(' ');
        writeFileSync(join(shots, file), Buffer.from(b64, 'base64'));
        continue;
      }
      if (line.startsWith('SHOT ')) console.log(line);
      if (line.startsWith('CSV ')) {
        if (out) { if (!existsSync(out)) appendFileSync(out, 'tag,backend,gpu,set,case,what,unit,median,min,max,reps,gpu_ms,note\n'); appendFileSync(out, line.slice(4) + '\n'); }
        rows++;
      } else if (line.startsWith('RESULT ') || line.startsWith('bench:') || (check && bad(line))) console.log(line);
      if (bad(line)) errors++;
      if (until){ console.log(line); if (line.includes(until)) finished = true; }
      if (line.startsWith('bench: done')) finished = true;
    }
  }
  if (m.method === 'Log.entryAdded' && log) appendFileSync(log, '[log] ' + m.params.entry.text + '\n');
  if (m.method === 'Runtime.exceptionThrown') { console.log('page error:', JSON.stringify(m.params.exceptionDetails).slice(0, 300)); errors++; finished = true; }
};
send('Runtime.enable');
if (log) send('Log.enable');
send('Page.navigate', { url });
const limit = Date.now() + (set === 'full' ? 1800 : 600) * 1000;
while (!finished && Date.now() < limit) await sleep(500);
if (!finished) console.log('webbench: stopped waiting (the page did not finish)');
ws.close(); chrome.kill();
if (server) server.close();
await sleep(500);
try { rmSync(dir, { recursive: true, force: true }); } catch {}
console.log(`webbench: ${rows} results${out ? ' appended to ' + out : ''}${check ? `, ${errors} errors` : ''}`);
process.exit(finished && (!check || errors === 0) ? 0 : 1);
