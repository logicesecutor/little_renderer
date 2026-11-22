// editor.js - Kiln, the web editor's interface: HTML over the engine's canvas. It talks to
// the engine (apps/editor.c, built into engine.js) only through its exported functions
// (Module._lr_...), and reads its state as JSON (lr_status, lr_scene).
//
// Milestone 1: the render view, the renderer switch, the resolution, the camera (orbit, pan,
// zoom, fly), the scene list, the status card and the internals. Adding, selecting and
// moving objects, materials, skies and saving come in the next milestones.
'use strict';

(() => {
  const $ = (id) => document.getElementById(id);

  // ---- the catalogues: what the tray shows (the engine has the same, apps/editor.c) ----
  const MATERIALS = {
    wood: { label: 'Painted wood', base: '#C9A27E' },
    terracotta: { label: 'Terracotta', base: '#C9694A' },
    stone: { label: 'Stone', base: '#8C857A' },
    steel: { label: 'Brushed steel', base: '#A9B0BA', metal: true },
    gold: { label: 'Gold', base: '#E3B04B', metal: true },
    rubber: { label: 'Blue rubber', base: '#3E6FA8' },
    glass: { label: 'Glass', base: '#BFD8EE', glass: true },
    glow: { label: 'Emissive', base: '#FFE2B8', emit: true },
  };
  const MAT_ORDER = ['wood', 'terracotta', 'stone', 'steel', 'gold', 'rubber', 'glass', 'glow'];
  const SHAPES = [
    ['box', 'Box', '#D4AB84', 'M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3zM12 12l8-4.5M12 12v9M12 12L4 7.5'],
    ['sphere', 'Sphere', '#F08A5E', 'M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18zM3 12c3 3 15 3 18 0'],
    ['glassball', 'Glass ball', '#9FC4E8', 'M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18zM15 7.5a4 4 0 0 1 2 3'],
    ['rock', 'Rock', '#A48A70', 'M3 19l2-7 6-5 7 1 3 5 0 6z'],
    ['lamp', 'Lamp', '#FFD39A', 'M12 3a5 5 0 1 0 0 10a5 5 0 1 0 0-10zM12 13v7M8 21h8'],
  ];
  const SKIES = [
    ['dusk', 'Dusk', 'linear-gradient(180deg, #1A2340 0%, #B8674C 58%, #3A2D28 60%, #110E0D 100%)', '#B8674C'],
    ['noon', 'Noon', 'linear-gradient(180deg, #5E95CF 0%, #D5E4EE 58%, #8A7E6C 60%, #4A4136 100%)', '#D5E4EE'],
    ['overcast', 'Overcast', 'linear-gradient(180deg, #7D8792 0%, #C3C8CC 58%, #6B6660 60%, #3A3734 100%)', '#C3C8CC'],
  ];
  const RES = [[1280, 720, 'HD'], [1920, 1080, 'Full HD · default'], [2560, 1440, 'QHD'], [3840, 2160, '4K UHD']];
  const BACKEND = { RT: 0, RASTER: 1, HYBRID: 2 };
  const BACKEND_NAME = ['BACKEND_RT', 'BACKEND_RASTER', 'BACKEND_HYBRID'];
  const FLY_KEYS = { w: 1, a: 2, s: 4, d: 8, e: 16, q: 32, shift: 64 };

  const ui = {
    ready: false, backend: BACKEND.HYBRID, res: [1920, 1080], internals: false, calls: [],
    scene: null, sel: null, tray: 'shapes', status: null, flying: false, flyKeys: 0,
  };

  // ---- the engine: its functions, and the calls the interface made (Internals) ----
  function call(name, args, text) {
    ui.calls.push(text || name + '(' + args.join(', ') + ')');
    if (ui.calls.length > 7) ui.calls.shift();
    if (ui.internals) renderInternals();
    return ui.ready ? Module['_' + name](...args) : 0;
  }
  function quiet(name, ...args) { return ui.ready ? Module['_' + name](...args) : 0; }   // (camera moves: not logged)
  function engineJSON(name) {
    try { return JSON.parse(Module.UTF8ToString(Module['_' + name]())); } catch (e) { return null; }
  }

  // the address's options go to the engine as bin/view's: ?n=960 -> -n 960
  const args = [];
  for (const [k, v] of new URLSearchParams(location.search)) { args.push('-' + k); if (v !== '') args.push(v); }

  window.Module = {
    canvas: $('canvas'),
    arguments: args,
    print: (t) => console.log(t),
    printErr: (t) => console.warn(t),
    // (before the engine's main: its scene and frame are made there, so the page reads them
    // at the first status that has a frame, sync below)
    onRuntimeInitialized: () => { ui.ready = true; },
  };

  // the engine's main has made its scene and opened its frame: the page takes them over
  function sync(st) {
    ui.synced = true;
    ui.scene = engineJSON('lr_scene');
    ui.res = [st.width, st.height];
    ui.backend = st.backend;
    renderAll();
    fitCanvas();
  }

  // ---- the frame, shown whole: scaled to fit the window, bars around it (5.15) ----
  const fitted = [0, 0];
  function fitCanvas() {
    const c = $('canvas');
    const rw = ui.synced ? c.width : ui.res[0], rh = ui.synced ? c.height : ui.res[1];   // (before: the default 300 x 150)
    fitted[0] = c.width;
    fitted[1] = c.height;
    const W = window.innerWidth, H = window.innerHeight;
    const k = Math.min(W / rw, H / rh);
    const w = Math.round(rw * k), h = Math.round(rh * k);
    c.style.width = w + 'px';
    c.style.height = h + 'px';
    c.style.left = Math.round((W - w) / 2) + 'px';
    c.style.top = Math.round((H - h) / 2) + 'px';
  }
  window.addEventListener('resize', fitCanvas);

  // ---- the camera: drag orbit, Shift drag pan, wheel zoom, right button + W A S D fly (6.7) ----
  const view = $('view');
  let drag = null;
  view.addEventListener('contextmenu', (e) => e.preventDefault());
  view.addEventListener('pointerdown', (e) => {
    if (drag) return;
    if (e.button === 0) drag = { mode: e.shiftKey ? 'pan' : 'orbit', x: e.clientX, y: e.clientY, id: e.pointerId };
    else if (e.button === 2) {
      drag = { mode: 'fly', x: e.clientX, y: e.clientY, id: e.pointerId };
      ui.flying = true;
      ui.flyKeys = 0;
    } else return;
    view.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  view.addEventListener('pointermove', (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (!dx && !dy) return;
    const c = $('canvas');
    const toPx = c.height / Math.max(1, c.clientHeight);       // window pixels -> the render's
    if (drag.mode === 'orbit') quiet('lr_camera_orbit', dx, dy);
    else if (drag.mode === 'pan') quiet('lr_camera_pan', dx * toPx, dy * toPx);
    else quiet('lr_camera_look', dx, dy);
  });
  function endDrag(e) {
    if (!drag || (e && e.pointerId !== drag.id)) return;
    if (drag.mode === 'fly') { ui.flying = false; ui.flyKeys = 0; quiet('lr_camera_fly', 0); }
    drag = null;
  }
  view.addEventListener('pointerup', endDrag);
  view.addEventListener('pointercancel', endDrag);
  view.addEventListener('wheel', (e) => {
    e.preventDefault();
    const steps = -e.deltaY / (e.deltaMode === 1 ? 3 : 100);
    quiet('lr_camera_dolly', Math.max(-5, Math.min(5, steps)));
  }, { passive: false });
  function flyKey(e, down) {
    const k = e.key.toLowerCase();
    const bit = FLY_KEYS[k];
    if (!bit) return false;
    ui.flyKeys = down ? ui.flyKeys | bit : ui.flyKeys & ~bit;
    quiet('lr_camera_fly', ui.flyKeys);
    return true;
  }
  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.isContentEditable)) return;
    if (ui.flying && !e.ctrlKey && !e.metaKey && flyKey(e, true)) { e.preventDefault(); return; }
    if (e.key === 'Escape') closePopovers();
  });
  window.addEventListener('keyup', (e) => { if (ui.flying) flyKey(e, false); });
  window.addEventListener('blur', () => { if (ui.flying) { ui.flyKeys = 0; quiet('lr_camera_fly', 0); } });

  // ---- the renderer switch (5.5) ----
  $('renderer').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    ui.backend = Number(b.dataset.backend);
    call('lr_set_backend', [ui.backend], 'lr_set_backend(' + BACKEND_NAME[ui.backend] + ')');
    renderRenderer();
  });
  function renderRenderer() {
    for (const b of $('renderer').querySelectorAll('button')) b.setAttribute('aria-pressed', Number(b.dataset.backend) === ui.backend ? 'true' : 'false');
  }

  // ---- the resolution popover (5.6) ----
  const resPop = $('res-popover'), resBtn = $('res-button');
  function openRes(open) {
    resPop.hidden = !open;
    resBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
      closeSave();
      $('res-w').value = ui.res[0];
      $('res-h').value = ui.res[1];
      $('res-w').removeAttribute('aria-invalid');
      $('res-h').removeAttribute('aria-invalid');
      renderPresets();
      const checked = resPop.querySelector('[aria-checked="true"]') || resPop.querySelector('button');
      checked.focus();
    }
  }
  resBtn.addEventListener('click', () => openRes(resPop.hidden));
  function renderPresets() {
    const box = $('res-presets');
    box.textContent = '';
    for (const [w, h, name] of RES) {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', w === ui.res[0] && h === ui.res[1] ? 'true' : 'false');
      b.innerHTML = '<span class="dims">' + w + ' × ' + h + '</span><span class="name"></span>';
      b.querySelector('.name').textContent = name;
      b.addEventListener('click', () => { setResolution(w, h); openRes(false); resBtn.focus(); });
      box.appendChild(b);
    }
  }
  $('res-custom').addEventListener('submit', (e) => {
    e.preventDefault();
    const w = Math.round(Number($('res-w').value)), h = Math.round(Number($('res-h').value));
    const okW = w >= 64 && w <= 7680, okH = h >= 64 && h <= 4320;
    $('res-w').setAttribute('aria-invalid', okW ? 'false' : 'true');
    $('res-h').setAttribute('aria-invalid', okH ? 'false' : 'true');
    if (!okW || !okH) return;
    setResolution(w, h);
    openRes(false);
    resBtn.focus();
  });
  function setResolution(w, h) {
    if (call('lr_set_resolution', [w, h]) !== 0 && ui.ready) { toast('Could not use ' + w + ' × ' + h + ' (out of memory)'); return; }
    ui.res = [w, h];
    $('res-label').textContent = w + ' × ' + h;
    fitCanvas();
    renderInternals();
  }

  // ---- the save menu (5.7: saving comes with milestone 4) and the internals switch ----
  const saveMenu = $('save-menu'), saveBtn = $('save-button');
  function closeSave() { saveMenu.hidden = true; saveBtn.setAttribute('aria-expanded', 'false'); }
  saveBtn.addEventListener('click', () => {
    const open = saveMenu.hidden;
    if (open) openRes(false);
    saveMenu.hidden = !open;
    saveBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  function closePopovers() { if (!resPop.hidden) { openRes(false); resBtn.focus(); } if (!saveMenu.hidden) { closeSave(); saveBtn.focus(); } }
  document.addEventListener('pointerdown', (e) => {
    if (!resPop.hidden && !resPop.contains(e.target) && !resBtn.contains(e.target)) openRes(false);
    if (!saveMenu.hidden && !saveMenu.contains(e.target) && !saveBtn.contains(e.target)) closeSave();
  });
  $('internals').addEventListener('click', () => {
    ui.internals = !ui.internals;
    $('internals').setAttribute('aria-pressed', ui.internals ? 'true' : 'false');
    $('internals-block').hidden = !ui.internals;
    renderInternals();
  });

  let toastTimer = 0;
  function toast(text) {
    $('toast-text').textContent = text;
    $('toast').hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 2200);
  }

  // ---- the scene list (5.8): the objects as a tree, what rests on what indented ----
  function renderScene() {
    const sc = ui.scene;
    const ul = $('objects');
    ul.textContent = '';
    if (!sc) { $('count').textContent = ''; return; }
    $('count').textContent = sc.objects.length + ' objects';
    const sky = SKIES.find((s) => s[1] === sc.sky);
    $('sky-label').textContent = sc.sky || '';
    $('sky-swatch').style.background = sky ? sky[3] : '#C3C8CC';
    const add = (o, depth) => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('aria-current', ui.sel === o.id ? 'true' : 'false');
      b.style.paddingLeft = (10 + depth * 18) + 'px';
      const m = MATERIALS[o.material];
      const sw = document.createElement('span');
      sw.className = 'swatch' + (o.kind === 'sphere' || o.kind === 'lamp' ? ' round' : '');
      sw.style.background = m ? m.base : '#B8AFA3';
      const nm = document.createElement('span');
      nm.className = 'name';
      nm.textContent = o.name;
      const body = document.createElement('span');
      body.className = 'body';
      body.textContent = o.dynamic ? 'dyn' : '';
      b.append(sw, nm, body);
      b.addEventListener('click', () => { ui.sel = ui.sel === o.id ? null : o.id; renderScene(); });
      li.appendChild(b);
      ul.appendChild(li);
      for (const c of sc.objects) if (c.support === o.id) add(c, depth + 1);
    };
    for (const o of sc.objects) if (!o.support) add(o, 0);
  }

  // ---- the tray (5.10): shown; adding and applying come with milestones 2 and 3 ----
  $('tray-tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    ui.tray = b.dataset.tab;
    for (const t of $('tray-tabs').querySelectorAll('button')) t.setAttribute('aria-selected', t === b ? 'true' : 'false');
    renderTiles();
  });
  function tile(cls, aria, title, inner) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tile ' + cls;
    b.draggable = true;
    b.setAttribute('aria-label', aria);
    b.title = title;
    b.innerHTML = inner;
    b.addEventListener('click', () => toast('Adding and changing objects come in the next milestone'));
    b.addEventListener('dragstart', (e) => e.preventDefault());
    return b;
  }
  function renderTiles() {
    const box = $('tiles');
    box.textContent = '';
    if (ui.tray === 'shapes') {
      for (const [, label, ink, icon] of SHAPES)
        box.appendChild(tile('', 'Add ' + label.toLowerCase() + ': drag into the scene, or press to place it in the middle', 'Drag into the scene, or click to add',
          '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="' + ink + '" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="' + icon + '"/></svg><span class="label">' + label + '</span>'));
    } else if (ui.tray === 'materials') {
      for (const k of MAT_ORDER) {
        const m = MATERIALS[k];
        const short = m.label.split(' ').pop().replace(/^./, (c) => c.toUpperCase());
        const sw = m.glass ? 'radial-gradient(circle at 65% 35%, #FFFFFF, #9FC4E8 40%, #3A4A5A)'
          : m.metal ? 'radial-gradient(circle at 65% 35%, #FFFFFF, ' + m.base + ' 45%, #2A2A2A)' : m.base;
        const glow = m.emit ? 'box-shadow: 0 0 14px #F2A65A;' : '';
        box.appendChild(tile('', m.label + ': drag onto an object, or press to apply it to the selection', 'Drag onto an object, or click to apply to the selection',
          '<span class="mat-swatch" style="background: ' + sw + '; ' + glow + '"></span><span class="label">' + short + '</span>'));
      }
    } else {
      for (const [, label, preview] of SKIES) {
        const t = tile('sky', label + ' sky: drag onto the background, or press to use it', 'Drag onto the background, or click to use', '<span class="label">' + label + '</span>');
        t.style.background = preview;
        t.setAttribute('aria-pressed', ui.scene && ui.scene.sky === label ? 'true' : 'false');
        box.appendChild(t);
      }
    }
  }

  // ---- the status card (5.11, 6.1) ----
  function statusOf(st) {
    if (!ui.ready || !st) return { title: 'Starting the engine', right: '', pct: 0, green: false, note: 'Loading the renderer and its GPU.' };
    if (st.gpuStarting) return { title: 'Starting the GPU', right: '', pct: 0, green: true, note: 'The CPU draws until the GPU is ready.' };
    const cpu = st.gpu ? '' : ' (on the CPU: this browser gave no WebGPU)';
    const auto = st.backend === BACKEND.HYBRID;
    if (!st.tracing) return {
      title: auto ? 'Auto · raster while moving' : 'Raster', right: 'real time', pct: 100, green: true,
      note: 'Light probe up to date. Hard shadows, no bounced light.' + cpu,
    };
    const done = st.spp >= st.target;
    return {
      title: (auto ? 'Auto · ' : '') + (done ? 'Converged' : 'Refining'),
      right: Math.min(st.spp, st.target) + ' / ' + st.target + ' spp',
      pct: Math.round(100 * Math.min(1, st.spp / Math.max(1, st.target))), green: false,
      note: (done ? 'Soft shadows, bounced light, glass and emissive light.' : 'Samples add up; the grain clears.') + cpu,
    };
  }
  let lastTitle = '';
  function renderStatus() {
    const s = statusOf(ui.status);
    if (s.title !== lastTitle) { $('status-title').textContent = s.title; lastTitle = s.title; }   // (announced politely)
    $('status-right').textContent = s.right;
    $('status-fill').style.width = s.pct + '%';
    $('status-fill').classList.toggle('green', s.green);
    $('status-bar').setAttribute('aria-valuenow', String(s.pct));
    $('status-bar').setAttribute('aria-label', s.green ? 'Light probe' : 'Path tracer samples');
    $('status-note').textContent = s.note;
    if (ui.internals) renderInternals();
  }
  function renderInternals() {
    if (!ui.internals) return;
    const st = ui.status;
    const lines = [
      ['resolution', ui.res[0] + ' × ' + ui.res[1]],
      ['renderer', !st ? '' : st.backend === BACKEND.HYBRID ? 'hybrid (' + (st.tracing ? 'tracing' : 'raster') + ')' : st.backend === BACKEND.RT ? 'rt' : 'raster'],
      ['instances', st ? String(st.instances) : ''],
      ['emitters', st ? String(st.emitters) : ''],
      ['frames/s', st ? st.fps.toFixed(0) : ''],
      ['device', st ? (st.gpu ? 'GPU (WebGPU)' : st.gpuStarting ? 'GPU starting' : 'CPU') : ''],
    ];
    const box = $('internal-lines');
    box.textContent = '';
    for (const [k, v] of lines) {
      const d = document.createElement('div');
      d.className = 'line';
      const a = document.createElement('span'); a.className = 'k'; a.textContent = k;
      const b = document.createElement('span'); b.textContent = v;
      d.append(a, b);
      box.appendChild(d);
    }
    const calls = $('calls');
    calls.textContent = '';
    for (const c of ui.calls.slice().reverse()) { const s = document.createElement('span'); s.textContent = c; calls.appendChild(s); }
  }

  function renderAll() {
    renderRenderer();
    $('res-label').textContent = ui.res[0] + ' × ' + ui.res[1];
    renderScene();
    renderTiles();
    renderStatus();
  }

  // the engine's state, ten times a second
  setInterval(() => {
    if (ui.ready) {
      const st = engineJSON('lr_status');
      ui.status = st && st.width > 0 ? st : null;
      if (ui.status && !ui.synced) sync(ui.status);
      const c = $('canvas');                       // (the engine sizes the canvas: fit it again)
      if (c.width !== fitted[0] || c.height !== fitted[1]) fitCanvas();
    }
    renderStatus();
  }, 100);

  renderAll();
  fitCanvas();
})();
