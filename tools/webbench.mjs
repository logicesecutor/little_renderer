// webbench.mjs - the performance suite in a browser (docs/STATUS.md 1.3, group J): opens
// web/build/bench.html in a headless Chrome of its own (a temporary profile, this machine's
// GPU), prints the results as the page makes them, and appends its CSV lines to a file.
// Node 22+, nothing to install; the page must be served (make serve: http://localhost:8000).
//
//   node tools/webbench.mjs [quick|full] [-only A,F3] [-tag label] [-o out/perf/x_web.csv]
//                           [-url http://127.0.0.1:8000/bench.html] [-chrome path]
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync, appendFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const argv = process.argv.slice(2);
const opt = (name, def) => { const i = argv.indexOf(name); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def; };
const set = argv.includes('full') ? 'full' : 'quick';
const only = opt('-only', ''), tag = opt('-tag', ''), out = opt('-o', '');
const base = opt('-url', 'http://127.0.0.1:8000/bench.html');
const chromes = [opt('-chrome', ''), 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
                 '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'];
const exe = chromes.find(p => p && existsSync(p));
if (!exe) { console.error('no Chrome found (-chrome path)'); process.exit(1); }

const q = [set === 'full' ? 'full' : 'quick'];
if (only) q.push('only=' + only);
if (tag) q.push('tag=' + tag);
const url = `${base}?${q.join('&')}`;
const dir = mkdtempSync(join(tmpdir(), 'lrbench-'));
const port = 9334;
const chrome = spawn(exe, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`, '--no-first-run',
                           '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws;
for (let k = 0; k < 100 && !ws; k++) {
  try { const t = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); const p = t.find(x => x.type === 'page'); if (p) ws = new WebSocket(p.webSocketDebuggerUrl); } catch {}
  if (!ws) await sleep(200);
}
if (!ws) { console.error('Chrome did not start'); chrome.kill(); process.exit(1); }
await new Promise(r => ws.onopen = r);
let id = 0, finished = false, rows = 0;
const send = (method, params = {}) => ws.send(JSON.stringify({ id: ++id, method, params }));
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.method === 'Runtime.consoleAPICalled') {
    const s = m.params.args.map(a => a.value ?? a.description ?? '').join(' ');
    for (const line of s.split('\n')) {
      if (line.startsWith('CSV ')) {
        if (out) { if (!existsSync(out)) appendFileSync(out, 'tag,backend,gpu,set,case,what,unit,median,min,max,reps,gpu_ms,note\n'); appendFileSync(out, line.slice(4) + '\n'); }
        rows++;
      } else if (line.startsWith('RESULT ') || line.startsWith('bench:')) console.log(line);
      if (line.startsWith('bench: done')) finished = true;
    }
  }
  if (m.method === 'Runtime.exceptionThrown') { console.log('page error:', JSON.stringify(m.params.exceptionDetails).slice(0, 300)); finished = true; }
};
send('Runtime.enable');
send('Page.navigate', { url });
const limit = Date.now() + (set === 'full' ? 1800 : 600) * 1000;
while (!finished && Date.now() < limit) await sleep(500);
if (!finished) console.log('webbench: stopped waiting (the page did not finish)');
ws.close(); chrome.kill();
await sleep(500);
try { rmSync(dir, { recursive: true, force: true }); } catch {}
console.log(`webbench: ${rows} results${out ? ' appended to ' + out : ''}`);
process.exit(finished ? 0 : 1);
