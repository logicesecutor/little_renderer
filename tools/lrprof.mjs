// lrprof.mjs - the samples of tools/lrprof.c as functions and lines (addr2line; the program
// built with -g):  node tools/lrprof.mjs program.exe samples.prof [-top 30] [-lines 25]
//   by function: where the instruction is (the function as compiled, with what was inlined
//   into it; time in a system library or the C runtime goes to the program's function that
//   called it), by the innermost inlined function, the time outside the program by its
//   caller, then the hottest source lines.
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2);
const exe = args[0], prof = args[1];
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? Number(args[i + 1]) : def; };
const top = opt('-top', 30), nlines = opt('-lines', 25), only = args.includes('-thread') ? String(opt('-thread', 0)) : null;
if (!exe || !prof){ console.error('usage: node tools/lrprof.mjs program.exe samples.prof [-top n] [-lines n]'); process.exit(1); }

const text = readFileSync(prof, 'utf8').split(/\r?\n/);
const head = text[0];
const mods = [];
const counts = new Map();            // "module offset caller" -> samples
const perThread = new Map();
let total = 0;
for (const line of text.slice(1)){
    if (line.startsWith('module ')){ const p = line.split(' '); mods[Number(p[1])] = p.slice(3).join(' '); }
    else if (line.startsWith('s ')){
        const p = line.split(' ');
        perThread.set(p[1], (perThread.get(p[1]) || 0) + 1);
        if (only !== null && p[1] !== only) continue;
        const key = p[2] + ' ' + p[3] + ' ' + (p[4] || '0');
        counts.set(key, (counts.get(key) || 0) + 1);
        total++;
    }
}
const base = s => (s || '?').replace(/\\/g, '/').split('/').pop();
const exeName = base(exe).toLowerCase();
const exeMod = mods.findIndex(m => m && base(m).toLowerCase() === exeName);

// the program's preferred base (the samples are offsets from where it was loaded)
const od = spawnSync('objdump', ['-p', exe], { encoding: 'utf8', maxBuffer: 1 << 28 });
const m = /ImageBase\s+([0-9a-fA-F]+)/.exec(od.stdout || '');
const imageBase = m ? BigInt('0x' + m[1]) : 0x140000000n;

const want = new Set();
for (const key of counts.keys()){
    const [mod, off, from] = key.split(' ');
    if (Number(mod) === exeMod) want.add(off);
    else if (from !== '0') want.add(from);
}
const addrs = [...want];
const sym = new Map();               // offset -> [{fn, loc}] innermost first
if (addrs.length){
    const input = addrs.map(a => '0x' + (imageBase + BigInt('0x' + a)).toString(16)).join('\n') + '\n';
    const r = spawnSync('addr2line', ['-f', '-i', '-a', '-e', exe], { input, encoding: 'utf8', maxBuffer: 1 << 30 });
    let cur = null, k = -1;
    const out = (r.stdout || '').split(/\r?\n/);
    for (let i = 0; i < out.length; i++){
        if (/^0x[0-9a-f]+$/.test(out[i])){ k++; cur = []; sym.set(addrs[k], cur); continue; }
        if (cur && out[i] !== '' && i + 1 < out.length){ cur.push({ fn: out[i], loc: out[i + 1].replace(/^.*[\\/](engine|apps|webgpu|tools)[\\/]/, '$1/').replace(/ \(discriminator \d+\)/, '') }); i++; }
    }
}

// functions without debug information (the C library's, the GPU driver's): their names from the symbol table
const nm = spawnSync('nm', ['-n', exe], { encoding: 'utf8', maxBuffer: 1 << 30 }).stdout || '';
const syms = [];
for (const l of nm.split(/\r?\n/)){ const s = /^([0-9a-f]+) [tT] (\S+)/.exec(l); if (s) syms.push([BigInt('0x' + s[1]) - imageBase, s[2]]); }
syms.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
const symAt = off => { const a = BigInt('0x' + off); let lo = 0, hi = syms.length - 1; if (hi < 0 || syms[0][0] > a) return '??'; while (lo < hi){ const mid = (lo + hi + 1) >> 1; if (syms[mid][0] <= a) lo = mid; else hi = mid - 1; } return syms[lo][1].slice(0, 60); };
for (const [off, fr] of sym) for (const f of fr) if (f.fn === '??') f.fn = symAt(off);

const byOuter = new Map(), byInner = new Map(), byLine = new Map(), byMod = new Map(), byCaller = new Map();
const add = (map, k, n) => map.set(k, (map.get(k) || 0) + n);
for (const [key, n] of counts){
    const [mod, off, from] = key.split(' ');
    const name = base(mods[Number(mod)]);
    add(byMod, name, n);
    if (Number(mod) !== exeMod){
        const fr = from !== '0' ? sym.get(from) : null;
        add(byInner, '[' + name + ']', n);
        if (fr && fr.length){
            add(byOuter, fr[fr.length - 1].fn, n);
            add(byCaller, '[' + name + '] < ' + fr[0].fn + '  ' + fr[0].loc + (fr.length > 1 ? '  < ' + fr[fr.length - 1].fn : ''), n);
        } else { add(byOuter, '[' + name + ']', n); add(byCaller, '[' + name + '] (not called from the program: a thread waiting or starting)', n); }
        continue;
    }
    const fr = sym.get(off);
    if (!fr || !fr.length){ add(byOuter, '??', n); add(byInner, '??', n); continue; }
    add(byOuter, fr[fr.length - 1].fn, n);
    add(byInner, fr[0].fn, n);
    add(byLine, fr[0].loc + '  ' + fr[0].fn + (fr.length > 1 ? '  < ' + fr[fr.length - 1].fn : ''), n);
}
const show = (title, map, n) => {
    console.log('\n' + title);
    [...map].sort((a, b) => b[1] - a[1]).slice(0, n).forEach(([k, v]) => console.log(`${(100 * v / total).toFixed(1).padStart(6)}%  ${String(v).padStart(8)}  ${k}`));
};
console.log(`${prof}: ${head}`);
console.log(`${total} samples of CPU time` + (only !== null ? ` of thread ${only}` : "") + `; per thread: ` + [...perThread].sort((a, b) => b[1] - a[1]).slice(0, 16).map(([k, v]) => k + ":" + v).join(" "));
show('by module', byMod, 12);
show('by function (as compiled: inlined callees and the libraries it calls inside)', byOuter, top);
show('by innermost (inlined) function', byInner, top);
show('outside the program, by caller', byCaller, top);
show('hottest lines', byLine, nlines);
