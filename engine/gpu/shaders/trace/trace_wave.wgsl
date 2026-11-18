// trace_wave.wgsl - used by engine/gpu/trace_gpu.c (WAVE)
/* ---- the wavefront path tracer (-wavefront on; docs/STATUS.md R1) ----
   The same light transport as trace() above, cut into small kernels that each do one step
   for every path of the band at once, the paths' state kept in WF between them:
     wf_gen      a camera ray per pixel of the band (the paths, in queue 0)
     wf_ext      the closest hit of each path in the queue (the BVH walk alone)
     wf_shade    what the hit does: media, lamps, materials, the next ray (into the other
                 queue) and the shadow ray toward the lamp (into the shadow queue)
     wf_args     the queues' sizes into the next dispatches' workgroup counts (indirect)
     wf_shadow   whether each shadow ray reaches the lamp (and through the media)
     wf_resolve  every path's radiance into the image, as main() does
   wf_ext, wf_shade, wf_args and wf_shadow run once per bounce. Each kernel holds only what
   its step needs, so the GPU runs more threads of it at once, and a path that ends early
   no longer keeps a thread waiting for the longest one of its group.
   WF (vec4u): [0] the control: x = which queue is current (0 / 1), y = its size, z = the
   shadow queue's size; then per path 4 (state), 1 (its hit), 3 (its shadow ray); then the
   queues (path indices, 4 per vec4): 0, 1 and the shadow queue. WC: the counters the shade
   step adds to. WA: the indirect dispatches' workgroup counts (x, y, z) for the paths and the
   shadow rays. */
@group(0) @binding(12) var<storage, read_write> WF: array<vec4u>;
@group(0) @binding(13) var<storage, read_write> WC: array<atomic<u32>, 4>;
@group(0) @binding(14) var<storage, read_write> WA: array<u32, 8>;

fn wf_n() -> u32 { return T.dims.x * T.dims.w; }
fn wf_st(p: u32) -> u32 { return 1u + 4u * p; }
fn wf_hit(p: u32) -> u32 { return 1u + 4u * wf_n() + p; }
fn wf_sh(p: u32) -> u32 { return 1u + 5u * wf_n() + 3u * p; }
fn wf_q(which: u32) -> u32 { return 1u + 8u * wf_n() + which * ((wf_n() + 3u) / 4u); }
fn q_get(which: u32, i: u32) -> u32 { return WF[wf_q(which) + i / 4u][i % 4u]; }
fn q_set(which: u32, i: u32, v: u32) { WF[wf_q(which) + i / 4u][i % 4u] = v; }

/* a path: its ray, the pdf of the bounce that made it, its pixel, throughput, random state,
   radiance so far, and flags: depth (bits 0-15), emit (16-17), chain (18), spec (19) */
struct PS { o: vec3f, pdf: f32, d: vec3f, pix: u32, thr: vec3f, rng: u32, rad: vec3f, fl: u32 };
fn ps_load(p: u32) -> PS {
    let s = wf_st(p);
    let a = WF[s]; let b = WF[s + 1u]; let c = WF[s + 2u]; let e = WF[s + 3u];
    return PS(bitcast<vec3f>(a.xyz), bitcast<f32>(a.w), bitcast<vec3f>(b.xyz), b.w,
              bitcast<vec3f>(c.xyz), c.w, bitcast<vec3f>(e.xyz), e.w);
}
fn ps_store(p: u32, q: PS) {
    let s = wf_st(p);
    WF[s] = vec4u(bitcast<vec3u>(q.o), bitcast<u32>(q.pdf));
    WF[s + 1u] = vec4u(bitcast<vec3u>(q.d), q.pix);
    WF[s + 2u] = vec4u(bitcast<vec3u>(q.thr), q.rng);
    WF[s + 3u] = vec4u(bitcast<vec3u>(q.rad), q.fl);
}
fn wf_id(gid: vec3u, nw: vec3u) -> u32 { return gid.x + gid.y * nw.x * 64u; }

@compute @workgroup_size(64)
fn wf_gen(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = wf_id(gid, nw);
    let n = wf_n();
    if (id >= n) { return; }
    if (id == 0u) { WF[0] = vec4u(0u, n, 0u, 0u); }
    let W = T.dims.x;
    let x = id % W;
    let y = T.dims.z + id / W;
    let k = y * W + x;
    let acc = accum[k];
    var rng = hash32(k ^ hash32(T.cnt.z * 0x9e3779b9u + u32(acc.w)));
    let sm = Smp(x, y, T.cnt.z, T.ext.x);
    let j = s2(sm, 0u, &rng);
    let u = (f32(x) + j.x) / f32(W);
    let v = (f32(y) + j.y) / f32(T.dims.y);
    let through = T.upper_left.xyz + T.horizontal.xyz * u - T.vertical.xyz * v;
    ps_store(id, PS(T.origin.xyz, 0.0, normalize(through - T.origin.xyz), k, vec3f(1.0), rng, vec3f(0.0), 0u));
    q_set(0u, id, id);
}

@compute @workgroup_size(64)
fn wf_ext(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    tlid = li;
    let id = wf_id(gid, nw);
    let c = WF[0];
    if (id >= c.y) { return; }
    let p = q_get(c.x & 1u, id);
    let s = wf_st(p);
    let o = bitcast<vec3f>(WF[s].xyz);
    let d = bitcast<vec3f>(WF[s + 1u].xyz);
    let f = traverse(Ray(o, d), T.f.y, 1e30, false);
    WF[wf_hit(p)] = vec4u(bitcast<u32>(f.t), f.id, f.inst, pack2x16float(f.b));
}

/* the shade step's shadow ray: from p along wi to the lamp at dist; c is what it brings if
   nothing is in the way (and times the media's transmittance) */
fn wf_shadow_ray(p: u32, o: vec3f, wi: vec3f, dist: f32, c: vec3f, seed: u32) {
    let s = wf_sh(p);
    WF[s] = vec4u(bitcast<vec3u>(o), bitcast<u32>(dist));
    WF[s + 1u] = vec4u(bitcast<vec3u>(wi), seed);
    WF[s + 2u] = vec4u(bitcast<vec3u>(c), 0u);
    let i = atomicAdd(&WC[1], 1u);
    q_set(2u, i, p);
}

/* direct_surface / direct_medium without the shadow ray: the light sample, and what it brings
   if the lamp is in sight (w = 0: nothing to send). One shadow ray per bounce: with both a
   lamp and a sky, one of the two, each half the time (its light twice, its density in MIS
   half: wf_sel) */
struct LS { wi: vec3f, dist: f32, c: vec3f };
fn wf_sel() -> f32 { return select(1.0, 0.5, T.opt.w != NONE && sky_on()); }
fn wf_sky_surface(h: Hit, b: ptr<function, Bsdf>, v: vec3f, u: vec2f) -> LS {
    var r = LS(vec3f(0.0), 0.0, vec3f(0.0));
    let sp = sky_sample(u);
    if (!(sp.w > 0.0)) { return r; }
    let fp = bsdf_eval(b, h.n, v, sp.xyz);
    if (!(max(fp.x, max(fp.y, fp.z)) > 0.0)) { return r; }
    let sel = wf_sel();
    var g = abs(dot(h.n, sp.xyz)) / (sp.w * sel);
    if (T.opt.y != 0u) { g *= power_heuristic(sp.w * sel, fp.w); }
    return LS(sp.xyz, 1e6, fp.xyz * sky_eval(sp.xyz) * g);
}
fn wf_sky_medium(p: vec3f, d: vec3f, gg: f32, u: vec2f) -> LS {
    var r = LS(vec3f(0.0), 0.0, vec3f(0.0));
    let sp = sky_sample(u);
    if (!(sp.w > 0.0)) { return r; }
    let sel = wf_sel();
    let ph = hg(dot(d, sp.xyz), gg);
    var w = ph / (sp.w * sel);
    if (T.opt.y != 0u) { w *= power_heuristic(sp.w * sel, ph); }
    return LS(sp.xyz, 1e6, sky_eval(sp.xyz) * w);
}
fn wf_light_surface(h: Hit, b: ptr<function, Bsdf>, v: vec3f, u: vec2f) -> LS {
    var r = LS(vec3f(0.0), 0.0, vec3f(0.0));
    if (T.opt.w == NONE) { return r; }
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let to_light = lt.q.xyz + lt.u.xyz * u.x + lt.v.xyz * u.y - h.p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_surf = dot(h.n, wi);
    let cos_light = -dot(lt.n.xyz, wi);
    if (cos_light <= 0.0) { return r; }
    let fp = bsdf_eval(b, h.n, v, wi);
    if (!(max(fp.x, max(fp.y, fp.z)) > 0.0)) { return r; }
    let sel = wf_sel();
    var g = abs(cos_surf) * cos_light * lt.v.w / (dist2 * sel);
    if (T.opt.y != 0u) { g *= power_heuristic(sel * light_pdf(lt.v.w, dist2, cos_light), fp.w); }
    return LS(wi, dist, fp.xyz * le * g);
}
fn wf_light_medium(p: vec3f, d: vec3f, gg: f32, u: vec2f) -> LS {
    var r = LS(vec3f(0.0), 0.0, vec3f(0.0));
    if (T.opt.w == NONE) { return r; }
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let to_light = lt.q.xyz + lt.u.xyz * u.x + lt.v.xyz * u.y - p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_light = -dot(lt.n.xyz, wi);
    if (cos_light <= 0.0) { return r; }
    let ph = hg(dot(d, wi), gg);
    let sel = wf_sel();
    var w = ph * cos_light * lt.v.w / (dist2 * sel);
    if (T.opt.y != 0u) { w *= power_heuristic(sel * light_pdf(lt.v.w, dist2, cos_light), ph); }
    return LS(wi, dist, le * w);
}

/* one bounce of trace() for each path of the queue */
@compute @workgroup_size(64)
fn wf_shade(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = wf_id(gid, nw);
    let cq = WF[0];
    if (id >= cq.y) { return; }
    let par = cq.x & 1u;
    let p = q_get(par, id);
    var s = ps_load(p);
    let hv = WF[wf_hit(p)];
    let f = Found(bitcast<f32>(hv.x), hv.y, unpack2x16float(hv.w), hv.z);
    var rng = s.rng;
    let depth = i32(s.fl & 0xffffu);
    var emit = i32((s.fl >> 16u) & 3u);
    var chain = (s.fl & (1u << 18u)) != 0u;
    var spec = (s.fl & (1u << 19u)) != 0u;
    var thr = s.thr;
    var ray = Ray(s.o, s.d);
    let max_depth = i32(T.opt.x);
    let after_bounce = select(1, 2, T.opt.y != 0u);
    let W = T.dims.x;
    let sm = Smp(s.pix % W, s.pix / W, T.cnt.z, T.ext.x);
    let found = f.id != NONE;
    var go_on = false;
    var nxt_o = vec3f(0.0);
    var nxt_d = vec3f(0.0);
    var done = false;
    if (T.cnt.w > 0u) {
        /* the ray may scatter in a medium before the surface: the earliest collision wins */
        var ts = select(1e30, f.t, found);
        var which = NONE;
        for (var k = 0u; k < T.cnt.w; k++) {
            let ti = track(k, ray, 0.0, ts, &rng, false);
            if (ti >= 0.0) { ts = ti; which = k; }
        }
        if (which != NONE) {
            done = true;
            if (depth < max_depth) {
                let vol = F.vols[which];
                let pp = ray.o + ray.d * ts;
                thr *= vol.albedo.xyz;
                let dd = 3u * u32(depth);
                var ls: LS;
                if (sky_on() && (T.opt.w == NONE || rnd(&rng) < 0.5)) { ls = wf_sky_medium(pp, ray.d, vol.p.x, sky_u(sm, depth, &rng)); }
                else { ls = wf_light_medium(pp, ray.d, vol.p.x, s2(sm, 1u + dd, &rng)); }
                if (ls.dist > 0.0) { wf_shadow_ray(p, pp, ls.wi, ls.dist, thr * ls.c, hash32(rng ^ 0x68bc21ebu)); }
                let nd = sample_phase(ray.d, vol.p.x, s2(sm, 2u + dd, &rng));
                s.pdf = hg(dot(ray.d, nd), vol.p.x);
                emit = after_bounce;
                chain = false;
                if (roulette(&thr, depth, sm, &rng)) { go_on = true; nxt_o = pp; nxt_d = nd; }
            }
        }
    }
    if (!done && !found && sky_on() && emit != 1) {   /* escaped: the sky */
        var le = sky_eval(ray.d);
        if (emit == 2) { le *= power_heuristic(s.pdf, wf_sel() * sky_pdf(ray.d)); }
        s.rad += thr * le;
    }
    if (!done && found) {
        var h = make_hit(ray, f);
        let m = shade_tex(T.mats[hit_mat(f)], ray, f, &h);
        let kind = u32(m.albedo.w);
        if (kind != 3u) { s.rad += thr * m.emission.rgb; }
        if (kind == 3u) {
            if (h.front && !(chain && spec)) {
                if (emit == 0) {
                    s.rad += thr * m.emission.rgb;
                } else if (emit == 2 && f.id == T.opt.w) {
                    let lt = prims[T.opt.w];
                    let cos_light = -dot(lt.n.xyz, ray.d);
                    if (cos_light > 0.0) {
                        s.rad += thr * m.emission.rgb * power_heuristic(s.pdf, wf_sel() * light_pdf(lt.v.w, h.t * h.t, cos_light));
                    }
                }
            }
        } else if (depth < max_depth) {
            let v = -ray.d;
            var b = mat_bsdf(m, h.front);
            let bo = surface_bounce(&b, h, v, depth, sm, &rng);
            if (bsdf_rough_part(&b)) {
                var ls: LS;
                if (sky_on() && (T.opt.w == NONE || rnd(&rng) < 0.5)) { ls = wf_sky_surface(h, &b, v, sky_u(sm, depth, &rng)); }
                else { ls = wf_light_surface(h, &b, v, s2(sm, 1u + 3u * u32(depth), &rng)); }
                if (ls.dist > 0.0) { wf_shadow_ray(p, h.p, ls.wi, ls.dist, thr * ls.c, hash32(rng ^ 0x68bc21ebu)); }
            }
            if (bo.s.ok) {
                thr *= bo.s.w;
                if (bo.s.delta) {
                    emit = 0;
                    spec = true;
                } else {
                    s.pdf = bo.s.pdf;
                    emit = after_bounce;
                    if (depth == 0) { chain = T.ext.z != 0u; }
                    spec = false;
                }
                if (roulette_u(&thr, depth, bo.ur)) { go_on = true; nxt_o = h.p; nxt_d = normalize(bo.s.l); }
            }
        }
    }
    s.thr = thr;
    s.rng = rng;
    s.fl = u32(depth + 1) | (u32(emit) << 16u) | select(0u, 1u << 18u, chain) | select(0u, 1u << 19u, spec);
    if (go_on) {
        s.o = nxt_o;
        s.d = nxt_d;
        let i = atomicAdd(&WC[0], 1u);
        q_set(1u - par, i, p);
    }
    ps_store(p, s);
}

fn wf_groups(n: u32, at: u32) {
    let g = (n + 63u) / 64u;
    let x = min(g, 65535u);
    WA[at] = x;
    WA[at + 1u] = select(1u, (g + x - 1u) / x, x > 0u);
    WA[at + 2u] = 1u;
}

@compute @workgroup_size(1)
fn wf_args() {
    let nn = atomicLoad(&WC[0]);
    let ns = atomicLoad(&WC[1]);
    var c = WF[0];
    c.x = c.x ^ 1u;
    c.y = nn;
    c.z = ns;
    WF[0] = c;
    wf_groups(nn, 0u);
    wf_groups(ns, 4u);
    atomicStore(&WC[0], 0u);
    atomicStore(&WC[1], 0u);
}

@compute @workgroup_size(64)
fn wf_shadow(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u, @builtin(local_invocation_index) li: u32) {
    tlid = li;
    let id = wf_id(gid, nw);
    if (id >= WF[0].z) { return; }
    let p = q_get(2u, id);
    let s = wf_sh(p);
    let a = WF[s];
    let b = WF[s + 1u];
    let dist = bitcast<f32>(a.w);
    let r = Ray(bitcast<vec3f>(a.xyz), bitcast<vec3f>(b.xyz));
    if (traverse(r, T.f.y, dist - T.f.z, true).id != NONE) { return; }
    var c = bitcast<vec3f>(WF[s + 2u].xyz);
    if (T.cnt.w > 0u) {
        var rng = b.w;
        c *= transmittance(r, 0.0, dist, &rng);
    }
    let st = wf_st(p) + 3u;
    let e = WF[st];
    WF[st] = vec4u(bitcast<vec3u>(bitcast<vec3f>(e.xyz) + c), e.w);
}

@compute @workgroup_size(64)
fn wf_resolve(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = wf_id(gid, nw);
    if (id >= wf_n()) { return; }
    let st = wf_st(id);
    let k = WF[st + 1u].w;
    var c = bitcast<vec3f>(WF[st + 3u].xyz);
    let mx = max(c.x, max(c.y, c.z));
    if (!(mx >= 0.0 && mx < 1e30)) {
        c = vec3f(0.0);
    } else if (T.f.x > 0.0 && mx > T.f.x) {
        c *= T.f.x / mx;
    }
    accum[k] = accum[k] + vec4f(c, 1.0);
    if (T.f.w > 0.5) {
        let N = T.dims.x * T.dims.y;
        let ga = accum[3u * N + k];
        let l = luma(select(c, c / max(ga.xyz, vec3f(1e-3)), ga.w == 0.0));
        accum[N + k] += vec4f(l, l * l, 0.0, 0.0);
    }
}
