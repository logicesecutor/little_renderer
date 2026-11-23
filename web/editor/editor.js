// editor.js - Kiln, the web editor's interface: HTML over the engine's canvas. It talks to
// the engine (apps/editor.c, built into engine.js) only through its exported functions
// (Module._lr_...), and reads its state as JSON (lr_status, lr_scene).
//
// Milestone 1: the render view, the renderer switch, the resolution, the camera (orbit, pan,
// zoom, fly), the scene list, the status card and the internals.
// Milestone 2: shapes added from the tray (dragged into the view or clicked), objects picked
// by the engine's rays, selected (brackets, the card), dragged over the surfaces, sized,
// made static or dynamic, deleted; undo and redo.
// Milestone 3: materials (dragged onto objects or pressed for the selection), skies (dragged
// onto the view or pressed), files dropped from the computer (.glb / .gltf at their real size,
// .hdr as the sky), and the raster's light made again after each change (Relighting).
// The frame: by default the whole window in the screen's real pixels, following it (a preset
// or a custom size fixes it).
// The middle button held pans, as Shift with the left one.
// .obj / .ply files from the computer: in metres (fitted to 1 m when that makes no sense), of clay.
// Milestone 4: the gizmo (W Move, E Rotate, R Scale).
// Saving: the scene as glTF 2.0 (.glb), downloaded.
// Drags without an edge: the pointer locked while a drag moves.
// Milestone 5: the Simulate tab, and the samples: Sky rain, the engine's physics demo
// (bin/view -sim on -demo sky) in the view, simulated, with Play / Pause / Reset.
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
    clay: { label: 'Clay (the file has no material)', base: '#B8AFA3' },   // (.obj / .ply)
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
  const RES = [[1280, 720, 'HD'], [1920, 1080, 'Full HD'], [2560, 1440, 'QHD'], [3840, 2160, '4K UHD']];
  const BACKEND = { RT: 0, RASTER: 1, HYBRID: 2 };
  const BACKEND_NAME = ['BACKEND_RT', 'BACKEND_RASTER', 'BACKEND_HYBRID'];
  const FLY_KEYS = { w: 1, a: 2, s: 4, d: 8, e: 16, q: 32, shift: 64 };
  // the samples (Simulate): the engine's demos, opened with ?sample=<key> (in a new tab).
  // args: bin/view's options for it; files: what it reads, fetched from samples/ (make
  // web-editor copies them from models/) into the engine's file system before it starts
  const SKY_RAIN_HDR = 'models/skies/kloofendal_48d_partly_cloudy_2k.hdr';
  const SAMPLES = {
    skyrain: {
      name: 'Sky rain', file: 'sky_rain',
      args: ['-sim', 'on', '-demo', 'sky', '-sky', SKY_RAIN_HDR, '-backend', 'rt', '-s', '256'],
      files: ['models/dragon_recon/dragon_vrip_res4.ply', 'models/torus.obj', SKY_RAIN_HDR],
    },
  };

  const ui = {
    ready: false, backend: BACKEND.HYBRID, res: [1920, 1080], internals: false, calls: [],
    scene: null, sel: null, tray: 'shapes', status: null, flying: false, flyKeys: 0,
    boxes: {}, dragging: null, adds: 0,
    follow: true,                  // the resolution: the window's, in the screen's real pixels (else fixed)
    mode: 'build',                 // the mode tab: build, simulate
    sample: null,                  // the sample shown (SAMPLES), not the editor's own scene
  };
  // the window in the screen's real pixels (its size times the display's scale), within the
  // engine's limits (64..7680 x 64..4320, the shape kept)
  function windowRes() {
    const dpr = window.devicePixelRatio || 1;
    let w = Math.max(64, Math.round(window.innerWidth * dpr)), h = Math.max(64, Math.round(window.innerHeight * dpr));
    const k = Math.min(1, 7680 / w, 4320 / h);
    return [Math.max(64, Math.round(w * k)), Math.max(64, Math.round(h * k))];
  }
  const SHAPE_INDEX = { box: 0, sphere: 1, glassball: 2, rock: 3, lamp: 4 };   // (lr_add_shape)
  const SHAPE_WIDTH = { box: 1, sphere: 1, glassball: 1, rock: 1.4, lamp: 0.4 };   // metres: the ground preview's size

  // ---- the engine: its functions, and the calls the interface made (Internals) ----
  function log(text) {                              // (the Internals' list of engine calls)
    ui.calls.push(text);
    if (ui.calls.length > 7) ui.calls.shift();
    if (ui.internals) renderInternals();
  }
  function call(name, args, text) {
    log(text || name + '(' + args.join(', ') + ')');
    return ui.ready ? Module['_' + name](...args) : 0;
  }
  function quiet(name, ...args) { return ui.ready ? Module['_' + name](...args) : 0; }   // (camera moves: not logged)
  function engineJSON(name) {
    try { return JSON.parse(Module.UTF8ToString(Module['_' + name]())); } catch (e) { return null; }
  }

  // the address's options go to the engine as bin/view's: ?n=960 -> -n 960
  const args = ['-width', String(windowRes()[0]), '-height', String(windowRes()[1])];   // (the default: the whole window)
  const query = new URLSearchParams(location.search);
  ui.sample = SAMPLES[query.get('sample')] || null;
  if (ui.sample) { args.push(...ui.sample.args); ui.mode = 'simulate'; }   // (the address's own options after: they can change it)
  for (const [k, v] of query) { if (k === 'sample') continue; args.push('-' + k); if (v !== '') args.push(v); }
  ui.res = windowRes();
  { const q = new URLSearchParams(location.search); if (q.has('width') || q.has('height')) ui.follow = false; }   // (a size in the address: fixed)

  window.Module = {
    canvas: $('canvas'),
    arguments: args,
    print: (t) => console.log(t),
    printErr: (t) => console.warn(t),
    // (before the engine's main: its scene and frame are made there, so the page reads them
    // at the first status that has a frame, sync below)
    onRuntimeInitialized: () => { ui.ready = true; },
    // a sample's files into the engine's file system before its main reads them; without
    // them (a build without the models) the editor's own scene starts instead
    preRun: [() => {
      const sm = ui.sample;
      if (!sm) return;
      Module.addRunDependency('sample');
      Promise.all(sm.files.map(async (f) => {
        const r = await fetch('samples/' + f);
        if (!r.ok) throw new Error(f);
        const data = new Uint8Array(await r.arrayBuffer());
        Module.FS.mkdirTree('/' + f.slice(0, f.lastIndexOf('/')));
        Module.FS.writeFile('/' + f, data);
      })).catch((e) => {
        const at = args.indexOf(sm.args[0]);
        if (at >= 0) args.splice(at, sm.args.length);
        ui.sampleMissing = sm.name + ' needs ' + e.message + ' (make models dragon-download skies-download, then make web-editor)';
        ui.sample = null;
        ui.mode = 'build';
      }).finally(() => Module.removeRunDependency('sample'));
    }],
  };

  // the engine's main has made its scene and opened its frame: the page takes them over
  function sync(st) {
    ui.synced = true;
    if (ui.sampleMissing) toast(ui.sampleMissing);
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
    const o = $('overlay').style;                   // (the page's drawings over the frame)
    o.left = c.style.left; o.top = c.style.top; o.width = c.style.width; o.height = c.style.height;
  }
  // a point of the window on the frame (0..1 from its top left), and back
  function toFrame(x, y) {
    const r = $('canvas').getBoundingClientRect();
    return [(x - r.left) / Math.max(1, r.width), (y - r.top) / Math.max(1, r.height)];
  }
  function frameRect() { const r = $('canvas').getBoundingClientRect(); return r; }
  // the window changed (its size, or the display's scale: a move to another screen, the
  // browser's zoom): the frame fitted again, and, following the window, made its new size
  // once it stops changing (a new size restarts the image). Not only on 'resize': browsers
  // do not always send one when only the scale changes, so the page also looks (below).
  let resizeTimer = 0, seenWin = '';
  function windowChanged() {
    seenWin = window.innerWidth + 'x' + window.innerHeight + '@' + (window.devicePixelRatio || 1);
    fitCanvas();
    if (!ui.follow) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const [w, h] = windowRes();
      if (ui.synced && ui.follow && (w !== ui.res[0] || h !== ui.res[1])) setResolution(w, h, true);
    }, 250);
  }
  window.addEventListener('resize', windowChanged);
  setInterval(() => { if (seenWin !== window.innerWidth + 'x' + window.innerHeight + '@' + (window.devicePixelRatio || 1)) windowChanged(); }, 100);

  // ---- the mouse on the view (6.5, 6.7): an object pressed and dragged slides over the
  // surfaces; elsewhere drag orbits, Shift drag pans, the wheel zooms, the right button held
  // with W A S D flies. A click selects what is under it, or nothing. ----
  const view = $('view');
  let drag = null;

  // ---- drags without an edge: the pointer locked (the browser's Pointer Lock) ----
  // Once a drag moves, the cursor is hidden and the mouse's own movement is used, so the
  // window's edge never stops it: orbit, pan and fly turn on, a gizmo handle goes on, an
  // object carried over the surfaces takes the view with it past the frame's edge
  // (lr_camera_slide). Each drag keeps a cursor of its own (vx, vy: where the cursor would
  // be), drawn while the real one is hidden where it matters (carrying, the gizmo). Without
  // the lock (not granted, Esc pressed, a browser without it) the same code runs on the
  // real cursor's moves.
  const lock = { want: false, reset: false };
  const locked = () => document.pointerLockElement === view;
  function lockPointer() {
    if (lock.want || !view.requestPointerLock) return;
    lock.want = true;
    try { const p = view.requestPointerLock(); if (p && p.catch) p.catch(() => {}); } catch (e) { /* (no lock: the real cursor) */ }
  }
  function unlockPointer() {
    lock.want = false;
    if (locked()) document.exitPointerLock();
    $('vcursor').hidden = true;
  }
  // (the lock taken or lost: the next move is measured from where the cursor is then)
  document.addEventListener('pointerlockchange', () => { lock.reset = true; if (!locked()) $('vcursor').hidden = true; });
  function track(d, x, y) { d.lx = x; d.ly = y; d.vx = x; d.vy = y; }
  // this event's move in window pixels; it moves the drag's cursor too
  function moved(e, d) {
    let mx, my;
    if (locked()) {
      const cap = (m) => Math.max(-250, Math.min(250, m || 0));   // (a first move after the lock can be a jump)
      mx = cap(e.movementX); my = cap(e.movementY);
      if (lock.reset) { lock.reset = false; if (Math.abs(mx) > 60 || Math.abs(my) > 60) mx = my = 0; }
    } else if (lock.reset) {
      lock.reset = false; mx = my = 0;
    } else { mx = e.clientX - d.lx; my = e.clientY - d.ly; }
    d.lx = e.clientX; d.ly = e.clientY;
    d.vx += mx; d.vy += my;
    return [mx, my];
  }
  function showVCursor(x, y) {
    const el = $('vcursor'), R = frameRect();
    el.style.left = (x - R.left) + 'px';
    el.style.top = (y - R.top) + 'px';
    el.hidden = !locked();
  }
  view.addEventListener('contextmenu', (e) => e.preventDefault());
  view.addEventListener('mousedown', (e) => { if (e.button === 1) e.preventDefault(); });   // (no auto-scroll on the middle button)
  view.addEventListener('auxclick', (e) => e.preventDefault());
  view.addEventListener('pointerdown', (e) => {
    if (drag || !ui.synced) return;
    if ((e.button === 0 && e.shiftKey) || e.button === 1) drag = { mode: 'pan' };
    else if (e.button === 0) {
      const [sx, sy] = toFrame(e.clientX, e.clientY);
      const pk = ui.mode === 'build' ? pick(sx, sy, 0) : null;     // (objects are picked in Build only)
      drag = { mode: pk && pk.id ? 'press-object' : 'press', obj: pk && pk.id };
    } else if (e.button === 2) {
      drag = { mode: 'fly' };
      ui.flying = true;
      ui.flyKeys = 0;
    } else return;
    drag.id = e.pointerId;
    drag.x0 = e.clientX;
    drag.y0 = e.clientY;
    track(drag, e.clientX, e.clientY);
    lock.reset = false;
    view.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  view.addEventListener('pointermove', (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const [dx, dy] = moved(e, drag);
    if (!dx && !dy) return;
    const far = Math.hypot(drag.vx - drag.x0, drag.vy - drag.y0) > 4;
    if (drag.mode === 'press' && far) drag.mode = 'orbit';
    if (drag.mode === 'press-object' && far) startObjectDrag(drag);
    if (drag.mode !== 'press' && drag.mode !== 'press-object') lockPointer();   // (a click: the cursor stays)
    const c = $('canvas');
    const toPx = c.height / Math.max(1, c.clientHeight);       // window pixels -> the render's
    if (drag.mode === 'orbit') quiet('lr_camera_orbit', dx, dy);
    else if (drag.mode === 'pan') quiet('lr_camera_pan', dx * toPx, dy * toPx);
    else if (drag.mode === 'fly') quiet('lr_camera_look', dx, dy);
    else if (drag.mode === 'object') dragObject(drag);
  });
  function endDrag(e) {
    if (!drag || (e && e.pointerId !== drag.id)) return;
    if (drag.mode === 'fly') { ui.flying = false; ui.flyKeys = 0; quiet('lr_camera_fly', 0); }
    if (drag.mode === 'press-object') select(drag.obj);
    if (drag.mode === 'press') select(null);
    if (drag.mode === 'object') endObjectDrag(drag);
    drag = null;
    unlockPointer();
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
  // the keys (6.5): Delete removes the selection, Esc deselects (and closes a popover), F
  // frames it, Ctrl Z / Ctrl Shift Z undo and redo; not while typing in a field
  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.isContentEditable)) return;
    if (ui.flying && !e.ctrlKey && !e.metaKey && flyKey(e, true)) { e.preventDefault(); return; }
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
    if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); redo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === ' ' && (ui.sample || ui.mode === 'simulate') && e.target.tagName !== 'BUTTON') { e.preventDefault(); simPlay(); return; }
    if (e.key === 'Escape') { if (!closePopovers()) { if (ui.tool !== 'drag') setTool('drag'); else select(null); } return; }
    if (ui.sel && (k === 'w' || k === 'e' || k === 'r')) { e.preventDefault(); setTool({ w: 'move', e: 'rotate', r: 'scale' }[k]); return; }
    if ((e.key === 'Delete' || e.key === 'Backspace') && ui.sel) { e.preventDefault(); removeSelected(); return; }
    if (k === 'f' && ui.sel) { e.preventDefault(); call('lr_frame_object', [ui.sel]); }
  });
  window.addEventListener('keyup', (e) => { if (ui.flying) flyKey(e, false); });
  window.addEventListener('blur', () => { if (ui.flying) { ui.flyKeys = 0; quiet('lr_camera_fly', 0); } });

  // ---- the objects: picking, adding, moving, removing (the engine does it; the page asks) ----
  function engineText(name, ...a) {
    try { return JSON.parse(Module.UTF8ToString(Module['_' + name](...a))); } catch (e) { return null; }
  }
  function pick(sx, sy, exclude) { return ui.ready ? engineText('lr_pick', sx, sy, exclude || 0) : null; }
  function project(p) { return ui.ready ? engineText('lr_project', p[0], p[1], p[2]) : null; }
  function obj(id) { return ui.scene && id ? ui.scene.objects.find((o) => o.id === id) : null; }
  function refreshScene() {
    ui.scene = engineJSON('lr_scene');
    if (ui.sel && !obj(ui.sel)) ui.sel = null;
    renderScene();
    renderHistory();
    renderCard();
    renderTiles();
  }
  function select(id) {
    if (id && ui.mode !== 'build') return;          // (objects are chosen and edited in Build)
    ui.sel = id && obj(id) ? id : null;
    renderScene();
    renderCard();
    placeOverlay();
  }
  const fmt = (v) => v.toFixed(2);
  const where = (p) => '(' + p.map(fmt).join(', ') + ')';

  // a shape added where a ray through (sx, sy) lands: on an object's top, or the ground
  function addShapeAt(key, sx, sy) {
    const pk = pick(sx, sy, 0);
    if (!pk || !pk.hit) return 0;
    quiet('lr_checkpoint');
    log('lr_pick(' + sx.toFixed(3) + ', ' + sy.toFixed(3) + ') -> ' + (pk.id ? 'object ' + pk.id : 'ground'));
    const id = quiet('lr_add_shape', SHAPE_INDEX[key], pk.land[0], pk.land[1], pk.land[2], pk.id || 0);
    log('lr_add_shape(' + key + ', ' + where(pk.land) + (pk.id ? ', on ' + pk.id : '') + ') -> ' + id);
    refreshScene();
    select(id);
    return id;
  }

  function startObjectDrag(d) {
    d.mode = 'object';
    quiet('lr_checkpoint');                       // (the whole drag: one undo step)
    d.moved = false;
    select(d.obj);
  }
  // the object under the drag's cursor; the cursor stays on the frame, and what goes past
  // its edge slides the view (by as much, at the object's depth), so the object can be
  // carried as far as the mouse goes
  function dragObject(d) {
    const R = frameRect(), m = 2;
    const x = Math.max(R.left + m, Math.min(R.right - m, d.vx)), y = Math.max(R.top + m, Math.min(R.bottom - m, d.vy));
    const ox = d.vx - x, oy = d.vy - y;
    d.vx = x;
    d.vy = y;
    const at = obj(d.obj);
    if ((ox || oy) && at) {
      const c = $('canvas'), toPx = c.height / Math.max(1, c.clientHeight);
      quiet('lr_camera_slide', ox * toPx, oy * toPx, at.centre[0], at.centre[1], at.centre[2]);
      d.slid = true;
    }
    showVCursor(x, y);
    const [sx, sy] = toFrame(x, y);
    const pk = pick(sx, sy, d.obj);               // (not onto itself or what rests on it)
    if (!pk || !pk.hit) return;
    if (quiet('lr_move_object', d.obj, pk.land[0], pk.land[1], pk.land[2], pk.id || 0) === 0) {
      d.moved = true;
      d.land = pk.land;
      d.onto = pk.id || 0;
    }
    const o = obj(d.obj), onto = obj(d.onto);
    ui.scene = engineJSON('lr_scene');
    showPill(o ? 'Move ' + o.name + (onto ? ' onto ' + onto.name : ' here') : '', x, y - 24, false);
    renderCard();
  }
  function endObjectDrag(d) {
    hideGhosts();
    if (d.moved) log('lr_move_object(' + d.obj + ', ' + where(d.land) + (d.onto ? ', on ' + d.onto : '') + ')');
    refreshScene();
  }
  function removeSelected() {
    const id = ui.sel;
    if (!id) return;
    quiet('lr_checkpoint');
    call('lr_remove_object', [id]);
    ui.sel = null;
    refreshScene();
    placeOverlay();
  }
  function undo() { if (quiet('lr_undo')) { log('lr_undo()'); refreshScene(); placeOverlay(); } }
  function redo() { if (quiet('lr_redo')) { log('lr_redo()'); refreshScene(); placeOverlay(); } }
  $('undo').addEventListener('click', undo);
  $('redo').addEventListener('click', redo);
  function renderHistory() {
    $('undo').disabled = !(ui.scene && ui.scene.undo > 0);
    $('redo').disabled = !(ui.scene && ui.scene.redo > 0);
  }

  // ---- materials and skies (6.4, 7.2, 7.3) ----
  function applyMaterial(id, key) {
    const o = obj(id);
    if (!o) return;
    quiet('lr_checkpoint');
    call('lr_set_material', [id, MAT_ORDER.indexOf(key)], 'lr_set_material(' + id + ', "' + key + '")');
    refreshScene();
  }
  function useSky(key) {
    if (ui.scene && ui.scene.skyKey === key) return;
    quiet('lr_checkpoint');
    call('lr_set_sky', [SKIES.findIndex((k) => k[0] === key)], 'lr_set_sky("' + key + '")');
    refreshScene();
  }

  // ---- files from the computer (6.4): .glb / .gltf as an object at its real size, .hdr as
  // the sky; written into the engine's file system (/drop) first ----
  let dropDir = false;
  async function dropFiles(files, sx, sy) {
    const model = files.find((f) => /\.(glb|gltf|obj|ply)$/i.test(f.name)), hdr = files.find((f) => /\.hdr$/i.test(f.name));
    if (!model && !hdr) { toast('Drop .glb, .gltf, .obj or .hdr files'); return; }
    if (!dropDir) { try { Module.FS.mkdir('/drop'); } catch (e) { /* (there already) */ } dropDir = true; }
    // all of them (a .gltf names its buffers and images, beside it)
    for (const f of files) Module.FS.writeFile('/drop/' + f.name, new Uint8Array(await f.arrayBuffer()));
    if (model) {
      const pk = pick(sx, sy, 0);
      if (!pk || !pk.hit) { toast('Drop it on the ground or on an object'); return; }
      toast('Reading ' + model.name);
      await new Promise((r) => setTimeout(r, 30));      // (the toast shows before the reading)
      quiet('lr_checkpoint');
      const fn = /\.(obj|ply)$/i.test(model.name) ? 'lr_import_mesh' : 'lr_import_gltf';   // (a mesh file: metres, of clay)
      const id = Module.ccall(fn, 'number', ['string', 'number', 'number', 'number', 'number'],
                              ['/drop/' + model.name, pk.land[0], pk.land[1], pk.land[2], pk.id || 0]);
      log(fn + '("/drop/' + model.name + '", ' + where(pk.land) + (pk.id ? ', on ' + pk.id : '') + ') -> ' + id);
      if (!id) { quiet('lr_forget_checkpoint'); toast('Could not read ' + model.name); refreshScene(); return; }
      if (fn === 'lr_import_mesh' && quiet('lr_last_import_fitted')) toast(model.name + ': its size made no sense in metres, so it was fitted to 1 m');
      refreshScene();
      select(id);
    } else {
      quiet('lr_checkpoint');
      const r = Module.ccall('lr_load_sky', 'number', ['string', 'string'], ['/drop/' + hdr.name, hdr.name]);
      log('lr_load_sky("/drop/' + hdr.name + '")');
      if (r) { quiet('lr_forget_checkpoint'); toast('Could not read ' + hdr.name); }
      refreshScene();
    }
  }

  // ---- what is dragged into the view (6.4): a preview where it would land or what it
  // would change, and its label ----
  function dragKind(e) {
    if (ui.sample) return null;                     // (a sample is shown, not edited)
    if (ui.dragging) return ui.dragging;
    return e.dataTransfer && [...(e.dataTransfer.types || [])].includes('Files') ? { type: 'files' } : null;
  }
  view.addEventListener('dragover', (e) => {
    const d = dragKind(e);
    if (!d || !ui.synced) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    hideGhosts();
    if (d.type === 'sky') {                            // anywhere: the whole view
      $('ghost-frame').hidden = false;
      const R = frameRect();
      showPill('Drop to use the ' + SKIES.find((k) => k[0] === d.key)[1].toLowerCase() + ' sky', R.left + R.width / 2, R.top + 160, false);
      return;
    }
    const [sx, sy] = toFrame(e.clientX, e.clientY);
    const pk = pick(sx, sy, 0);
    const o = pk && pk.id ? obj(pk.id) : null, b = o && ui.boxes[o.id];
    if (d.type === 'material') {
      const m = MATERIALS[d.key];
      if (o && b) { showRing(b); showPill('Apply ' + m.label.toLowerCase() + ' to ' + o.name, null, null, false, b); }
      else { e.dataTransfer.dropEffect = 'none'; showPill('Drop on an object', e.clientX, e.clientY - 24, true); }
      return;
    }
    if (!pk || !pk.hit) { e.dataTransfer.dropEffect = d.type === 'files' ? 'copy' : 'none'; if (d.type === 'files') showPill('.glb / .gltf / .obj as an object, .hdr as the sky', e.clientX, e.clientY - 24, true); return; }
    const label = d.type === 'files' ? null : SHAPES.find((t) => t[0] === d.key)[1].toLowerCase();
    if (o && b) {
      showRing(b);
      showPill(label ? 'Add ' + label + ' onto ' + o.name : '.glb / .obj on ' + o.name + ', .hdr as the sky', null, null, false, b);
    } else {
      const e1 = showEllipse(pk.land, 0.6 * (label ? SHAPE_WIDTH[d.key] : 1));
      showPill(label ? 'Add ' + label + ' here' : '.glb / .gltf / .obj as an object, .hdr as the sky', e1 ? e1.cx : e.clientX, e1 ? e1.top - 8 : e.clientY - 24, false);
    }
  });
  view.addEventListener('dragleave', (e) => { if (!view.contains(e.relatedTarget)) hideGhosts(); });
  view.addEventListener('drop', (e) => {
    const d = dragKind(e);
    ui.dragging = null;
    hideGhosts();
    if (!d || !ui.synced) return;
    e.preventDefault();
    const [sx, sy] = toFrame(e.clientX, e.clientY);
    if (d.type === 'shape') addShapeAt(d.key, sx, sy);
    else if (d.type === 'sky') useSky(d.key);
    else if (d.type === 'material') { const pk = pick(sx, sy, 0); if (pk && pk.id) { applyMaterial(pk.id, d.key); select(pk.id); } }
    else if (d.type === 'files') dropFiles([...e.dataTransfer.files], sx, sy);
  });
  document.addEventListener('dragend', () => { ui.dragging = null; hideGhosts(); });
  // (a file dropped anywhere else on the page: not opened by the browser)
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());

  // ---- what the page draws over the render: brackets, previews, lamp notes, the objects' buttons ----
  function px(b) {                                // a frame box (0..1) in the overlay's pixels
    const r = frameRect();
    return { left: b[0] * r.width, top: b[1] * r.height, w: (b[2] - b[0]) * r.width, h: (b[3] - b[1]) * r.height };
  }
  function place(el, l, t, w, h) { el.style.left = l + 'px'; el.style.top = t + 'px'; el.style.width = w + 'px'; el.style.height = h + 'px'; }
  function showRing(b) {
    const q = px(b);
    const el = $('ghost-ring');
    place(el, q.left - 10, q.top - 10, q.w + 20, q.h + 20);
    el.hidden = false;
    $('ghost-ground').hidden = true;
  }
  // the dashed ellipse on the ground: a circle of radius r (metres) around p, as the camera sees it
  function showEllipse(p, r) {
    const pts = [[p[0] + r, p[1], p[2]], [p[0] - r, p[1], p[2]], [p[0], p[1], p[2] + r], [p[0], p[1], p[2] - r]].map(project);
    if (pts.some((q) => !q)) { hideGhosts(); return null; }
    const R = frameRect();
    const xs = pts.map((q) => q[0] * R.width), ys = pts.map((q) => q[1] * R.height);
    const l = Math.min(...xs), t = Math.min(...ys), w = Math.max(...xs) - l, h = Math.max(6, Math.max(...ys) - t);
    const el = $('ghost-ground');
    place(el, l, t, w, h);
    el.hidden = false;
    $('ghost-ring').hidden = true;
    return { cx: R.left + l + w / 2, top: R.top + t };
  }
  // the label over a preview: at (x, y) in the window, or over the box b
  function showPill(text, x, y, dark, b) {
    const el = $('ghost-pill'), R = frameRect();
    if (b) { const q = px(b); x = R.left + q.left + q.w / 2; y = R.top + q.top - 16; }
    el.textContent = text;
    el.classList.toggle('dark', !!dark);
    el.style.left = (x - R.left) + 'px';
    el.style.top = (y - R.top) + 'px';
    el.hidden = !text;
  }
  function hideGhosts() { for (const id of ['ghost-ground', 'ghost-ring', 'ghost-frame', 'ghost-pill']) $(id).hidden = true; }

  // every frame: the objects' boxes from the engine, and what follows them
  function placeOverlay() {
    if (!ui.synced) return;
    ui.boxes = engineJSON('lr_boxes') || {};
    const R = frameRect();
    // the selection's brackets, 8 px out
    const br = $('brackets'), sb = ui.sel && ui.boxes[ui.sel];
    if (sb) { const q = px(sb); place(br, q.left - 8, q.top - 8, q.w + 16, q.h + 16); }
    br.hidden = !sb;
    // the raster's lamp notes: over what glows, while the raster shows
    const notes = $('lamp-notes');
    const raster = ui.status && !ui.status.tracing;
    const glow = raster && ui.scene ? ui.scene.objects.filter((o) => o.material === 'glow' && ui.boxes[o.id]) : [];
    while (notes.children.length < glow.length) { const n = document.createElement('span'); n.className = 'lamp-note'; n.textContent = 'Lights the scene in path tracing only'; notes.appendChild(n); }
    while (notes.children.length > glow.length) notes.lastChild.remove();
    glow.forEach((o, i) => { const q = px(ui.boxes[o.id]); const n = notes.children[i]; n.style.left = (q.left + q.w / 2) + 'px'; n.style.top = (q.top - 8) + 'px'; });
    // the objects' buttons, nearest last (on top)
    const hits = $('hits');
    const list = ui.scene ? ui.scene.objects.filter((o) => ui.boxes[o.id]).sort((a, b) => ui.boxes[b.id][4] - ui.boxes[a.id][4]) : [];
    const have = new Map([...hits.children].map((b) => [Number(b.dataset.id), b]));
    for (const [id, b] of have) if (!list.find((o) => o.id === id)) b.remove();
    for (const o of list) {
      let b = have.get(o.id);
      if (!b) {
        b = document.createElement('button');
        b.type = 'button';
        b.dataset.id = String(o.id);
        b.addEventListener('click', () => select(o.id));
        b.addEventListener('focus', () => select(o.id));
        hits.appendChild(b);
      }
      const m = MATERIALS[o.material];
      b.setAttribute('aria-label', o.name + ', ' + (m ? m.label : 'its own materials') + ', ' + (o.dynamic ? 'dynamic' : 'static') + '. Drag to move; Delete to remove; F to frame it.');
      b.setAttribute('aria-pressed', ui.sel === o.id ? 'true' : 'false');
      const q = px(ui.boxes[o.id]);
      place(b, q.left, q.top, Math.max(34, q.w), Math.max(34, q.h));
    }
    placeCard(R);
    drawGizmo();
  }
  function frameLoop() { placeOverlay(); requestAnimationFrame(frameLoop); }
  requestAnimationFrame(frameLoop);

  // ---- the object card (5.9): next to the selection ----
  const card = $('card');
  function renderCard() {
    const o = obj(ui.sel);
    card.hidden = !o;
    if (!o) return;
    const m = MATERIALS[o.material] || { label: 'From the file', base: '#B8AFA3' };
    $('card-name').textContent = o.name;
    card.setAttribute('aria-label', o.name + ' properties');
    $('card-delete').setAttribute('aria-label', 'Delete ' + o.name);
    $('card-swatch').style.background = m.base;
    $('card-mat-swatch').style.background = m.base;
    $('card-mat-swatch').style.boxShadow = m.emit ? '0 0 12px #F2A65A' : 'none';
    $('card-mat').textContent = m.label;
    for (const b of $('card-body').querySelectorAll('button')) b.setAttribute('aria-pressed', Number(b.dataset.dynamic) === o.dynamic ? 'true' : 'false');
    const sl = $('card-scale');
    if (document.activeElement !== sl) sl.value = String(o.scale);
    $('card-scale-text').textContent = o.scale.toFixed(2) + '×' + (o.kind === 'model' ? ' file size' : '');
    $('card-model-note').hidden = o.kind !== 'model';
    $('card-model-note').textContent = /\.(obj|ply)$/i.test(o.name) ? 'Its units taken as metres (the file has none).' : 'Imported at the real size in the file (glTF metres).';
    $('card-px').textContent = fmt(o.pos[0]);
    $('card-py').textContent = fmt(o.pos[1]);
    $('card-pz').textContent = fmt(o.pos[2]);
    const notes = [];
    const sup = obj(o.support);
    if (sup) notes.push('Resting on ' + sup.name + '; moves with it.');
    if (m.emit && ui.status && !ui.status.tracing) notes.push('Emissive: lights other objects in path tracing only.');
    $('card-note').textContent = notes.join(' ');
    $('card-note').hidden = !notes.length;
  }
  // 28 px right of the selection's box (or left, at the right margin), its top 20 px above the box's
  function placeCard(R) {
    const b = ui.sel && ui.boxes[ui.sel];
    if (card.hidden || !b) { if (!card.hidden && !b) card.style.visibility = 'hidden'; return; }
    card.style.visibility = '';
    const q = px(b), W = window.innerWidth, H = window.innerHeight, ch = card.offsetHeight;
    let left = R.left + q.left + q.w + 28;
    if (left + 316 > W - 24) left = R.left + q.left - 28 - 316;
    left = Math.max(24, Math.min(W - 24 - 316, left));
    const top = Math.max(88, Math.min(H - 24 - ch, R.top + q.top - 20));
    card.style.left = Math.round(left) + 'px';
    card.style.top = Math.round(top) + 'px';
  }
  $('card-close').addEventListener('click', () => select(null));
  $('card-delete').addEventListener('click', removeSelected);
  $('card-body').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    const o = obj(ui.sel);
    if (!b || !o || Number(b.dataset.dynamic) === o.dynamic) return;
    quiet('lr_checkpoint');
    call('lr_set_body', [o.id, Number(b.dataset.dynamic)], 'lr_set_body(' + o.id + ', ' + (b.dataset.dynamic === '1' ? 'dynamic' : 'static') + ')');
    refreshScene();
  });
  // the size: live while the slider moves, one undo step per release
  const scaleInput = $('card-scale');
  let scaling = false;
  scaleInput.addEventListener('input', () => {
    const o = obj(ui.sel);
    if (!o) return;
    if (!scaling) { quiet('lr_checkpoint'); scaling = true; }
    quiet('lr_scale_object', o.id, Number(scaleInput.value));
    ui.scene = engineJSON('lr_scene');
    renderCard();
  });
  scaleInput.addEventListener('change', () => {
    if (!scaling) return;
    scaling = false;
    log('lr_scale_object(' + ui.sel + ', ' + Number(scaleInput.value).toFixed(2) + ')');
    refreshScene();
  });
  $('card-tools').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (b) setTool(b.dataset.tool);
  });

  // ---- the gizmo (5.13, 6.5): W Move, E Rotate, R Scale; Drag (surface) without one ----
  // It is drawn here from the engine's camera (lr_camera), 200 x 200 px round the object's
  // middle whatever its distance. A handle dragged changes the object along that axis only
  // (Move: the world's X, Y, Z; Scale: the object's own), its centre in the screen's plane
  // (Move) or all at once (Scale); Rotate turns it about its middle, about an axis or the
  // view. One undo step per drag; what rests on it comes along.
  const AXES = [[1, 0, 0, '#F0565A'], [0, 1, 0, '#5BD16B'], [0, 0, 1, '#5C9BFF']];
  const V3 = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    mul: (a, k) => [a[0] * k, a[1] * k, a[2] * k], dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  };
  const Q = {                                        // quaternions [w, x, y, z]
    mul: (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]],
    axis: (v, ang) => { const s2 = Math.sin(ang / 2); return [Math.cos(ang / 2), v[0] * s2, v[1] * s2, v[2] * s2]; },
    rotate: (q, v) => { const p = Q.mul(Q.mul(q, [0, v[0], v[1], v[2]]), [q[0], -q[1], -q[2], -q[3]]); return [p[1], p[2], p[3]]; },
  };
  let cam = null;
  // a point (metres) in the overlay's pixels, and how many pixels a metre is there; null behind the camera
  function proj(p) {
    if (!cam) return null;
    const r = V3.sub(p, cam.pos), z = V3.dot(r, cam.fwd);
    if (z <= 1e-3) return null;
    const R = frameRect(), h = Math.tan(cam.fov * Math.PI / 360), a = cam.w / cam.h;
    return [(0.5 + V3.dot(r, cam.right) / (z * 2 * h * a)) * R.width, (0.5 - V3.dot(r, cam.up) / (z * 2 * h)) * R.height, R.height / (2 * z * h)];
  }
  function setTool(t) {
    if (!['drag', 'move', 'rotate', 'scale'].includes(t)) return;
    ui.tool = t;
    for (const b of $('card-tools').querySelectorAll('button')) b.setAttribute('aria-pressed', b.dataset.tool === t ? 'true' : 'false');
    placeOverlay();
  }
  ui.tool = 'drag';
  const SVGNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const giz = $('gizmo');
  let gdrag = null;                                  // a handle being dragged
  // what the handles are: {kind: 'axis' | 'plane' | 'ring' | 'view' | 'uniform', axis: 0..2}
  function handle(kind, axis, shape, attrs, hit) {
    const g = el('g', {}, giz);
    el(shape, attrs, g);
    const h = el(hit.tag, Object.assign({ class: hit.fill ? 'hit-fill' : 'hit' }, hit.attrs), g);
    h.addEventListener('pointerdown', (e) => startGizmo(e, kind, axis, g));
    return g;
  }
  function drawGizmo() {
    const o = obj(ui.sel);
    giz.textContent = '';
    // (an svg: shown and hidden by the attribute itself; it has no .hidden)
    if (!o || ui.tool === 'drag' || !ui.synced) { giz.toggleAttribute('hidden', true); return; }
    cam = engineText('lr_camera');
    const c = proj(o.centre);
    if (!cam || !c) { giz.toggleAttribute('hidden', true); return; }
    const R = frameRect();
    giz.toggleAttribute('hidden', false);
    // (the svg's own box the size of the frame: the browser takes the mouse on an svg's
    // children only inside it, whatever they overflow)
    giz.setAttribute('width', R.width); giz.setAttribute('height', R.height);
    giz.style.left = '0px'; giz.style.top = '0px';
    giz.style.width = R.width + 'px'; giz.style.height = R.height + 'px';
    const k = c[2];                                  // pixels per metre at the object
    const rot = [o.rot[0], o.rot[1], o.rot[2], o.rot[3]];
    const dirs = AXES.map((A, i) => {                // each axis on the screen: a unit 2D direction
      const v = ui.tool === 'scale' ? Q.rotate(rot, [A[0], A[1], A[2]]) : [A[0], A[1], A[2]];
      const q = proj(V3.add(o.centre, V3.mul(v, 0.5 / k * 100)));   // (a point 50 px along it, about)
      if (!q) return null;
      const dx = q[0] - c[0], dy = q[1] - c[1], l = Math.hypot(dx, dy);
      return l < 6 ? null : { v, d: [dx / l, dy / l], i };          // (pointing at the camera: no handle)
    });
    const at = (d, len) => [c[0] + d[0] * len, c[1] + d[1] * len];
    if (ui.tool === 'move' || ui.tool === 'scale') {
      for (const a of dirs) {
        if (!a) continue;
        const col = AXES[a.i][3], e1 = at(a.d, ui.tool === 'move' ? 70 : 68);
        const g = handle('axis', a.i, 'line', { x1: c[0], y1: c[1], x2: e1[0], y2: e1[1], stroke: col, 'stroke-width': 3, 'stroke-linecap': 'round' },
                         { tag: 'line', attrs: { x1: c[0], y1: c[1], x2: e1[0] + a.d[0] * 12, y2: e1[1] + a.d[1] * 12 } });
        if (ui.tool === 'move') {
          const tip = at(a.d, 82), n = [-a.d[1], a.d[0]];
          el('polygon', { points: [[e1[0] + n[0] * 6, e1[1] + n[1] * 6], tip, [e1[0] - n[0] * 6, e1[1] - n[1] * 6]].map((p) => p.join(',')).join(' '), fill: col }, g);
        } else el('rect', { x: e1[0] - 6, y: e1[1] - 6, width: 12, height: 12, fill: col }, g);
      }
      const sq = ui.tool === 'move' ? 5 : 6;
      handle(ui.tool === 'move' ? 'plane' : 'uniform', -1, 'rect', { x: c[0] - sq, y: c[1] - sq, width: 2 * sq, height: 2 * sq, fill: '#ECEDEF' },
             { tag: 'rect', fill: true, attrs: { x: c[0] - 10, y: c[1] - 10, width: 20, height: 20 } });
    } else {                                         // rotate: a ring per axis (round it in the world), the view's ring
      const rw = 55 / k;
      for (let i = 2; i >= 0; i--) {
        const A = AXES[i], ax = [A[0], A[1], A[2]];
        const u = V3.norm(V3.cross(ax, Math.abs(ax[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0])), v = V3.cross(ax, u);
        const pts = [];
        for (let j = 0; j <= 64; j++) {
          const t = j / 64 * 2 * Math.PI, q = proj(V3.add(o.centre, V3.add(V3.mul(u, rw * Math.cos(t)), V3.mul(v, rw * Math.sin(t)))));
          if (q) pts.push(q[0].toFixed(1) + ',' + q[1].toFixed(1));
        }
        const d = 'M' + pts.join('L');
        handle('ring', i, 'path', { d, fill: 'none', stroke: A[3], 'stroke-width': 2.5 }, { tag: 'path', attrs: { d } });
      }
      handle('view', -1, 'circle', { cx: c[0], cy: c[1], r: 64, fill: 'none', stroke: '#ECEDEF', 'stroke-opacity': 0.35, 'stroke-width': 2.5 },
             { tag: 'circle', attrs: { cx: c[0], cy: c[1], r: 64 } });
    }
    giz.__c = c;
    giz.__dirs = dirs;
  }
  function startGizmo(e, kind, axis, g) {
    const o = obj(ui.sel);
    if (!o || e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    quiet('lr_checkpoint');                          // (the whole drag: one undo step)
    const c = giz.__c, dirs = giz.__dirs;
    gdrag = { kind, axis, id: o.id, x0: e.clientX, y0: e.clientY, pos: o.pos.slice(), centre: o.centre.slice(), rot: o.rot.slice(),
              stretch: o.stretch.slice(), c: [c[0], c[1]], k: c[2], dir: axis >= 0 && dirs[axis] ? dirs[axis] : null, pointer: e.pointerId };
    track(gdrag, e.clientX, e.clientY);
    lock.reset = false;
    g.classList.add('hot');
    window.addEventListener('pointermove', moveGizmo);
    window.addEventListener('pointerup', endGizmo);
    window.addEventListener('pointercancel', endGizmo);
  }
  function moveGizmo(e) {
    const d = gdrag;
    if (!d || e.pointerId !== d.pointer) return;
    const [mx, my] = moved(e, d);
    if (!mx && !my) return;
    lockPointer();
    const dx = d.vx - d.x0, dy = d.vy - d.y0;
    let pos = d.pos, rot = d.rot, st = d.stretch;
    const o = obj(d.id);
    const uniform = o && (o.kind === 'sphere' || o.kind === 'lamp');   // (they keep their shape)
    if (d.kind === 'axis' && d.dir && ui.tool === 'move') pos = V3.add(d.pos, V3.mul(d.dir.v, (dx * d.dir.d[0] + dy * d.dir.d[1]) / d.k));
    else if (d.kind === 'plane') pos = V3.add(d.pos, V3.add(V3.mul(cam.right, dx / d.k), V3.mul(cam.up, -dy / d.k)));
    else if (d.kind === 'axis' && d.dir) {           // scale along its own axis (100 px: twice as long)
      const f = Math.max(0.05, 1 + (dx * d.dir.d[0] + dy * d.dir.d[1]) / 100);
      st = uniform ? d.stretch.map((x) => x * f) : d.stretch.map((x, i) => (i === d.axis ? x * f : x));
    } else if (d.kind === 'uniform') {
      const f = Math.max(0.05, 1 + (dx - dy) / 150);
      st = d.stretch.map((x) => x * f);
    } else if (d.kind === 'ring' || d.kind === 'view') {
      const a0 = Math.atan2(d.y0 - frameRect().top - d.c[1], d.x0 - frameRect().left - d.c[0]);
      const a1 = Math.atan2(d.vy - frameRect().top - d.c[1], d.vx - frameRect().left - d.c[0]);
      showVCursor(d.vx, d.vy);                    // (round the ring: where the hidden cursor is)
      let da = a1 - a0;
      while (da > Math.PI) da -= 2 * Math.PI;
      while (da < -Math.PI) da += 2 * Math.PI;
      const ax = d.kind === 'view' ? cam.fwd : AXES[d.axis].slice(0, 3);
      const turn = Q.axis(ax, da * Math.sign(V3.dot(ax, cam.fwd) || 1));   // (as the mouse goes round, seen from the camera)
      rot = Q.mul(turn, d.rot);
      pos = V3.add(d.centre, Q.rotate(turn, V3.sub(d.pos, d.centre)));    // (about its middle)
    }
    quiet('lr_set_transform', d.id, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3], st[0], st[1], st[2]);
    d.moved = true;
    ui.scene = engineJSON('lr_scene');
    renderCard();
  }
  function endGizmo(e) {
    const d = gdrag;
    if (!d || (e && e.pointerId !== d.pointer)) return;
    gdrag = null;
    window.removeEventListener('pointermove', moveGizmo);
    window.removeEventListener('pointerup', endGizmo);
    window.removeEventListener('pointercancel', endGizmo);
    unlockPointer();
    const o = obj(d.id);
    if (d.moved && o) log('lr_set_transform(' + d.id + ', ' + where(o.pos) + ', [' + o.rot.map((x) => x.toFixed(3)).join(', ') + '], [' + o.stretch.map(fmt).join(', ') + '])');
    else if (!d.moved) quiet('lr_forget_checkpoint');   // (a press without a move: no step)
    refreshScene();
  }

  // ---- saving (5.7, 6.8): the scene as glTF 2.0 (.glb), downloaded ----
  $('save-glb').addEventListener('click', () => {
    closeSave();
    if (!ui.ready) return;
    const name = 'untitled_scene.glb';
    try { Module.FS.mkdir('/save'); } catch (e) { /* (there already) */ }
    const r = Module.ccall('lr_save_gltf', 'number', ['string'], ['/save/' + name]);
    log('lr_save_gltf("' + name + '")');
    if (r) { toast('Could not save the scene'); return; }
    const data = Module.FS.readFile('/save/' + name);
    const url = URL.createObjectURL(new Blob([data], { type: 'model/gltf-binary' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    window.__lastSave = data;                        // (for a test to look at)
    toast('Saved ' + name);
  });

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
    const win = windowRes();
    const wb = document.createElement('button');
    wb.type = 'button';
    wb.setAttribute('role', 'radio');
    wb.setAttribute('aria-checked', ui.follow ? 'true' : 'false');
    wb.innerHTML = '<span class="dims">' + win[0] + ' × ' + win[1] + '</span><span class="name">Window · default</span>';
    wb.title = 'The whole window, in the screen\'s real pixels; it follows the window';
    wb.addEventListener('click', () => { setResolution(win[0], win[1], true); openRes(false); resBtn.focus(); });
    box.appendChild(wb);
    for (const [w, h, name] of RES) {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', !ui.follow && w === ui.res[0] && h === ui.res[1] ? 'true' : 'false');
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
  // follow: the window's size, and again whenever the window changes (else fixed at w x h)
  function setResolution(w, h, follow) {
    if (call('lr_set_resolution', [w, h]) !== 0 && ui.ready) { toast('Could not use ' + w + ' × ' + h + ' (out of memory)'); return; }
    ui.follow = !!follow;
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
  function closePopovers() {                       // (1: one was open)
    if (!resPop.hidden) { openRes(false); resBtn.focus(); return 1; }
    if (!saveMenu.hidden) { closeSave(); saveBtn.focus(); return 1; }
    return 0;
  }
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
    $('count').textContent = sc.sample ? sc.sample.instances + ' instances' : sc.objects.length + ' objects';
    if (sc.sample) {                               // (a sample: what it is made of)
      $('sky-label').textContent = sc.sky || '';
      $('sky-label').title = sc.sky || '';
      $('sky-swatch').style.background = '#A9C4DE';
      const rows = [['#B5664A', false, 'Brick paving', ''], ['conic-gradient(#E3B04B, #3E6FA8, #5FCB8A, #C9694A, #E3B04B)', true, 'Bodies', sc.sample.bodies + ' · dyn'],
        ['#A9B0BA', false, 'Props', String(sc.sample.props)]];
      for (const [bg, round, name, body] of rows) {
        const li = document.createElement('li');
        li.className = 'row';
        li.innerHTML = '<span class="swatch' + (round ? ' round' : '') + '"></span><span class="name"></span><span class="body"></span>';
        li.firstChild.style.background = bg;
        li.children[1].textContent = name;
        li.children[2].textContent = body;
        ul.appendChild(li);
      }
      return;
    }
    const sky = SKIES.find((k) => k[0] === sc.skyKey);
    $('sky-label').textContent = sc.sky || '';
    $('sky-label').title = sc.sky || '';
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
      b.addEventListener('click', () => select(o.id));
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
  // a tray tile: pressed (Enter, Space, a click) or dragged; act: what pressing does,
  // drag: what dragging carries (null: nothing yet)
  function tile(cls, aria, title, inner, act, drag) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tile ' + cls;
    b.draggable = true;
    b.setAttribute('aria-label', aria);
    b.title = title;
    b.innerHTML = inner;
    b.addEventListener('click', act);
    b.addEventListener('dragstart', (e) => {
      if (!drag || !ui.synced) { e.preventDefault(); return; }
      ui.dragging = drag;
      e.dataTransfer.setData('text/plain', 'kiln:' + drag.type + ':' + drag.key);
      e.dataTransfer.effectAllowed = 'copy';
    });
    return b;
  }
  // a shape pressed in the tray: in the middle of the view, a little aside from the last one
  function addInFront(key) {
    const n = ui.adds++;
    const sx = 0.5 + ((n * 0.137) % 0.3) - 0.15, sy = 0.62 + ((n * 0.061) % 0.12) - 0.06;
    if (!addShapeAt(key, sx, sy) && !addShapeAt(key, 0.5, 0.75)) toast('Nowhere to put it: turn the camera towards the ground');
  }
  function renderTiles() {
    const box = $('tiles');
    box.textContent = '';
    if (ui.tray === 'shapes') {
      for (const [key, label, ink, icon] of SHAPES)
        box.appendChild(tile('', 'Add ' + label.toLowerCase() + ': drag into the scene, or press to place it in the middle', 'Drag into the scene, or click to add',
          '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="' + ink + '" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="' + icon + '"/></svg><span class="label">' + label + '</span>',
          () => addInFront(key), { type: 'shape', key }));
    } else if (ui.tray === 'materials') {
      for (const k of MAT_ORDER) {
        const m = MATERIALS[k];
        const short = m.label.split(' ').pop().replace(/^./, (c) => c.toUpperCase());
        const sw = m.glass ? 'radial-gradient(circle at 65% 35%, #FFFFFF, #9FC4E8 40%, #3A4A5A)'
          : m.metal ? 'radial-gradient(circle at 65% 35%, #FFFFFF, ' + m.base + ' 45%, #2A2A2A)' : m.base;
        const glow = m.emit ? 'box-shadow: 0 0 14px #F2A65A;' : '';
        box.appendChild(tile('', m.label + ': drag onto an object, or press to apply it to the selection', 'Drag onto an object, or click to apply to the selection',
          '<span class="mat-swatch" style="background: ' + sw + '; ' + glow + '"></span><span class="label">' + short + '</span>',
          () => { if (ui.sel) applyMaterial(ui.sel, k); else toast('Select an object first, or drag the material onto one'); }, { type: 'material', key: k }));
      }
    } else {
      for (const [key, label, preview] of SKIES) {
        const t = tile('sky', label + ' sky: drag onto the background, or press to use it', 'Drag onto the background, or click to use', '<span class="label">' + label + '</span>',
                       () => useSky(key), { type: 'sky', key });
        t.style.background = preview;
        t.setAttribute('aria-pressed', ui.scene && ui.scene.skyKey === key ? 'true' : 'false');
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
      title: auto ? 'Auto · raster while moving' : 'Raster', right: st.relighting ? 'relighting' : 'real time',
      pct: st.relighting ? Math.round(100 * st.relight) : 100, green: true,
      note: (st.relighting ? 'Relighting: the light probe is being rebuilt after the change.' : 'Light probe up to date. Hard shadows, no bounced light.') + cpu,
    };
    const done = st.spp >= st.target;
    if (st.sim && st.sim.on && !st.sim.paused && (!st.sim.own || st.sim.moving > 0)) return {
      title: 'Simulating · path traced', right: Math.min(st.spp, st.target) + ' / ' + st.target + ' spp',
      pct: Math.round(100 * Math.min(1, st.spp / Math.max(1, st.target))), green: false,
      note: 'Each step of the bodies starts the samples again; pause to let the image refine.' + cpu,
    };
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
    if (st && st.sim) lines.push(['simulation', st.sim.line]);
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

  // ---- the modes (5.4): Build, and Simulate (its controls where the tray is) ----
  document.querySelector('.modes').addEventListener('click', (e) => {
    const a = e.target.closest('a');
    if (!a) return;
    e.preventDefault();
    if (a.getAttribute('aria-disabled') === 'true' || !a.dataset.mode) return;
    setMode(a.dataset.mode);
  });
  const simHolds = () => !!(ui.status && ui.status.sim && ui.status.sim.own && ui.status.sim.on);   // (a simulation holds the objects)
  function setMode(m) {
    if ((ui.sample || simHolds()) && m === 'build') return;
    ui.mode = m;
    if (m !== 'build') { if (ui.tool !== 'drag') setTool('drag'); select(null); closePopovers(); }
    renderModes();
  }
  function renderModes() {
    for (const a of document.querySelectorAll('.modes a[data-mode]')) {
      if (a.dataset.mode === ui.mode) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
      if (a.dataset.mode !== 'build') continue;
      if (ui.sample) { a.setAttribute('aria-disabled', 'true'); a.title = ui.sample.name + ' is the engine\'s sample: shown and simulated, not edited'; }
      else if (simHolds()) { a.setAttribute('aria-disabled', 'true'); a.title = 'Reset or Keep the simulation to edit again'; }
      else { a.removeAttribute('aria-disabled'); a.title = ''; }
    }
    document.querySelector('.tray:not(.sim-bar)').hidden = ui.mode !== 'build';
    $('sim-bar').hidden = ui.mode !== 'simulate';
    $('sim-controls').hidden = false;
    $('sim-samples').hidden = !!ui.sample;
    if (ui.sample) {
      document.querySelector('.scene-name').textContent = ui.sample.file + ' · sample';
      saveBtn.disabled = true;
      saveBtn.title = 'A sample has nothing of yours to save';
    }
    renderSim();
  }
  function renderSim() {
    const sim = ui.status && ui.status.sim, paused = !sim || sim.paused;
    const own = !ui.sample, on = !!(sim && sim.on), was = $('sim-keep').hidden;
    $('sim-keep').hidden = !(own && on);
    if (own && was === on) renderModes();                         // (a simulation made or ended: Build off or on again)
    $('sim-play').setAttribute('aria-label', paused ? 'Play' : 'Pause');
    $('sim-icon-play').toggleAttribute('hidden', !paused);
    $('sim-icon-pause').toggleAttribute('hidden', paused);
    $('sim-time').textContent = (sim ? sim.t : 0).toFixed(2) + ' s';
    $('sim-reset').disabled = own && !on;
    if (own) {
      const n = sim ? sim.bodies : 0;
      $('sim-bodies').textContent = !n ? 'Nothing is dynamic yet' : on ? sim.moving + ' of ' + n + ' dynamic moving' : n + ' dynamic object' + (n === 1 ? '' : 's');
      $('sim-note').innerHTML = !n ? 'Make an object <b>Dynamic</b> in its card (Build), then Play: it falls and collides.'
        : '<span class="mono key">Space</span> plays and pauses. Reset puts everything back; Keep keeps the pose.';
    } else {
      $('sim-bodies').textContent = sim && ui.scene && ui.scene.sample ? sim.bodies + ' of ' + ui.scene.sample.bodies + ' bodies' : '';
    }
    // (the list as it is now: the instances grow as the bodies come; the props are the rest)
    const sm = ui.scene && ui.scene.sample;
    if (sim && sm && (sm.instances !== ui.status.instances || sm.props !== ui.status.instances - sim.bodies)) {
      sm.instances = ui.status.instances;
      sm.props = ui.status.instances - sim.bodies;
      renderScene();
    }
  }
  function simPlay() {
    const sim = ui.status && ui.status.sim;
    if (!sim) return;
    call('lr_sim_play', [sim.paused ? 1 : 0], 'lr_sim_play(' + (sim.paused ? 1 : 0) + ')');
    sim.paused = !sim.paused;                      // (shown at once; the next status says the same)
    renderSim();
  }
  $('sim-play').addEventListener('click', simPlay);
  $('sim-reset').addEventListener('click', () => {
    if (!ui.status || !ui.status.sim) return;
    call('lr_sim_reset', [], 'lr_sim_reset()');
    if (!ui.sample) { ui.status.sim.on = false; ui.status.sim.paused = true; ui.status.sim.t = 0; refreshScene(); renderSim(); }
  });
  $('sim-keep').addEventListener('click', () => {
    call('lr_sim_keep', [], 'lr_sim_keep()');
    if (ui.status && ui.status.sim) { ui.status.sim.on = false; ui.status.sim.paused = true; ui.status.sim.t = 0; }
    refreshScene();
    renderSim();
    toast('Kept the pose. Undo goes back to before Play.');
  });
  $('open-sample').addEventListener('click', () => window.open(location.pathname + '?sample=skyrain', '_blank'));

  function renderAll() {
    renderModes();
    renderRenderer();
    $('res-label').textContent = ui.res[0] + ' × ' + ui.res[1];
    renderScene();
    renderTiles();
    renderStatus();
    renderHistory();
    renderCard();
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
    renderSim();
  }, 100);

  renderAll();
  fitCanvas();
})();
