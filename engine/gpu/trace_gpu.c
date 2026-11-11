// trace_gpu.c - the path tracer of rt/tracer.c (trace_path_ex), on the GPU.
//
// One compute kernel, one thread per pixel, one sample per thread: the whole path is traced
// in a loop inside the shader (a "megakernel"), and the sample is added to an accumulation
// buffer (sum of the samples, and their count, per pixel) that the view shows. The same light
// transport as the CPU: diffuse / mirror / glass / lamp, next event estimation with multiple
// importance sampling, Russian roulette, and the media (delta tracking for the collisions,
// ratio tracking for the shadow rays, both walking the majorant bricks with a 3D DDA).
//
// The scene is the CPU's, flattened into buffers:
//   nodes   the BVH (rt/bvh.h), 32 bytes per node: box, and children or a range of `aux`
//   aux     the BVH's shape ids (leaves are ranges of it), then the vertex normals
//   prims   spheres and quads, with the precomputed data of core/scene.h
//   tris    mesh triangles, 48 bytes: corner, two edges, 3 normal indices, material
// What differs from the CPU: 32-bit floats whatever the build, and another random number
// generator (PCG with a 32-bit state: WGSL has no 64-bit integers). So the two images are
// not equal sample by sample; they converge to the same picture.
#include "gpu/gpu_internal.h"

#ifdef LR_GPU

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/volume.h"
#include "rt/accel.h"
#include "rt/bvh.h"

static const char *TRACE = WGSL(
struct Node { lo: vec3f, a: u32, hi: vec3f, b: u32 };       /* leaf: a = first, b = 0x80000000 | count */
struct Prim {
    c: vec4f,      /* sphere: center, radius */
    q: vec4f,      /* quad: corner; w = 0 sphere, 1 quad */
    u: vec4f,      /* edge; w = material */
    v: vec4f,      /* edge; w = area */
    n: vec4f,      /* unit normal; w = plane_d */
    w: vec4f,      /* (u x v) / |u x v|^2 */
};
struct Tri { q: vec3f, n0: u32, u: vec3f, n1: u32, v: vec3f, nm: u32 };   /* nm: normal 2, material in the top 4 bits */
struct TMat { albedo: vec4f, emission: vec4f };             /* albedo.w = type, emission.w = ior */
struct TP {
    origin: vec4f, upper_left: vec4f, horizontal: vec4f, vertical: vec4f,   /* the camera */
    dims: vec4u,   /* image width, height; first row and rows of this band */
    opt: vec4u,    /* max depth, mis, number of prims, the lamp (a prim) */
    cnt: vec4u,    /* BVH nodes, where the normals start in aux, seed, media */
    f: vec4f,      /* clamp, ray epsilon, shadow margin */
    mats: array<TMat, 16>,
};
@group(0) @binding(3) var<storage, read> VS: array<VStat, 2>;
@group(0) @binding(4) var<uniform> T: TP;
@group(0) @binding(5) var<storage, read> nodes: array<Node>;
@group(0) @binding(6) var<storage, read> aux: array<u32>;
@group(0) @binding(7) var<storage, read> prims: array<Prim>;
@group(0) @binding(8) var<storage, read> tris: array<Tri>;
@group(0) @binding(9) var<storage, read_write> accum: array<vec4f>;

const NONE: u32 = 0xffffffffu;
const PI_: f32 = 3.14159265;

struct Ray { o: vec3f, d: vec3f };
struct Found { t: f32, id: u32, b: vec2f };
struct Hit { t: f32, p: vec3f, n: vec3f, ng: vec3f, front: bool };

/* random numbers: a 32-bit PCG (an LCG whose output is scrambled) */
fn rnd(s: ptr<function, u32>) -> f32 {
    let old = *s;
    *s = old * 747796405u + 2891336453u;
    let w = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return f32(((w >> 22u) ^ w) >> 8u) * (1.0 / 16777216.0);
}
fn hash32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}

/* ---- intersections (rt/intersect.c); a miss is t = -1 ---- */

fn hit_sphere(pr: Prim, r: Ray, tmin: f32, tmax: f32) -> f32 {
    let oc = pr.c.xyz - r.o;
    let a = dot(r.d, r.d);
    let hh = dot(r.d, oc);
    let c = dot(oc, oc) - pr.c.w * pr.c.w;
    let disc = hh * hh - a * c;
    if (disc < 0.0) { return -1.0; }
    let sq = sqrt(disc);
    var t = (hh - sq) / a;
    if (t <= tmin || t >= tmax) {
        t = (hh + sq) / a;
        if (t <= tmin || t >= tmax) { return -1.0; }
    }
    return t;
}

fn hit_quad(pr: Prim, r: Ray, tmin: f32, tmax: f32) -> f32 {
    let denom = dot(pr.n.xyz, r.d);
    if (abs(denom) < 1e-12) { return -1.0; }
    let t = (pr.n.w - dot(pr.n.xyz, r.o)) / denom;
    if (t <= tmin || t >= tmax) { return -1.0; }
    let rel = r.o + r.d * t - pr.q.xyz;
    let a = dot(pr.w.xyz, cross(rel, pr.v.xyz));
    let b = dot(pr.w.xyz, cross(pr.u.xyz, rel));
    if (a < 0.0 || a > 1.0 || b < 0.0 || b > 1.0) { return -1.0; }
    return t;
}

/* Moller-Trumbore: (t, b1, b2) */
fn hit_tri(tr: Tri, r: Ray, tmin: f32, tmax: f32) -> vec3f {
    let pvec = cross(r.d, tr.v);
    let det = dot(tr.u, pvec);
    if (abs(det) < 1e-12) { return vec3f(-1.0); }
    let inv = 1.0 / det;
    let tvec = r.o - tr.q;
    let b1 = dot(tvec, pvec) * inv;
    if (b1 < 0.0 || b1 > 1.0) { return vec3f(-1.0); }
    let qvec = cross(tvec, tr.u);
    let b2 = dot(r.d, qvec) * inv;
    if (b2 < 0.0 || b1 + b2 > 1.0) { return vec3f(-1.0); }
    let t = dot(tr.v, qvec) * inv;
    if (t <= tmin || t >= tmax) { return vec3f(-1.0); }
    return vec3f(t, b1, b2);
}

/* slab test: the distance at which the ray enters the box, -1 if it misses (tmin > 0) */
fn box_t(lo: vec3f, hi: vec3f, o: vec3f, inv: vec3f, tmin: f32, tmax: f32) -> f32 {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let a = min(t0, t1);
    let b = max(t0, t1);
    let enter = max(max(a.x, a.y), max(a.z, tmin));
    let leave = min(min(b.x, b.y), min(b.z, tmax));
    return select(enter, -1.0, leave < enter);
}

/* the BVH (rt/bvh.c): the closest hit in (tmin, tmax), or with stop_at_first any hit */
fn traverse(r: Ray, tmin: f32, tmax: f32, stop_at_first: bool) -> Found {
    var best = Found(tmax, NONE, vec2f(0.0));
    if (T.cnt.x == 0u) { return best; }
    let inv = safe_inv(r.d);
    var stack: array<u32, 64>;
    var stack_t: array<f32, 64>;
    var sp = 0;
    let t_root = box_t(nodes[0].lo, nodes[0].hi, r.o, inv, tmin, tmax);
    if (t_root < 0.0) { return best; }
    stack[0] = 0u;
    stack_t[0] = t_root;
    sp = 1;
    while (sp > 0) {
        sp--;
        if (stack_t[sp] > best.t) { continue; }
        let nd = nodes[stack[sp]];
        if ((nd.b & 0x80000000u) != 0u) {
            let count = nd.b & 0x7fffffffu;
            for (var k = 0u; k < count; k++) {
                let sid = aux[nd.a + k];
                if (sid < T.opt.z) {
                    let pr = prims[sid];
                    var t = -1.0;
                    if (pr.q.w < 0.5) { t = hit_sphere(pr, r, tmin, best.t); } else { t = hit_quad(pr, r, tmin, best.t); }
                    if (t > 0.0) {
                        best = Found(t, sid, vec2f(0.0));
                        if (stop_at_first) { return best; }
                    }
                } else {
                    let h = hit_tri(tris[sid - T.opt.z], r, tmin, best.t);
                    if (h.x > 0.0) {
                        best = Found(h.x, sid, h.yz);
                        if (stop_at_first) { return best; }
                    }
                }
            }
            continue;
        }
        /* inner node: the farther child is pushed first, so the nearer one is visited first */
        let tl = box_t(nodes[nd.a].lo, nodes[nd.a].hi, r.o, inv, tmin, best.t);
        let tr = box_t(nodes[nd.b].lo, nodes[nd.b].hi, r.o, inv, tmin, best.t);
        if (tl >= 0.0 && tr >= 0.0 && sp < 62) {
            let left_first = tl <= tr;
            stack[sp] = select(nd.a, nd.b, left_first);
            stack_t[sp] = max(tl, tr);
            sp++;
            stack[sp] = select(nd.b, nd.a, left_first);
            stack_t[sp] = min(tl, tr);
            sp++;
        } else if (tl >= 0.0 && sp < 63) {
            stack[sp] = nd.a;
            stack_t[sp] = tl;
            sp++;
        } else if (tr >= 0.0 && sp < 63) {
            stack[sp] = nd.b;
            stack_t[sp] = tr;
            sp++;
        }
    }
    return best;
}

fn normal_at(i: u32) -> vec3f {
    let o = T.cnt.y + 3u * i;
    return vec3f(bitcast<f32>(aux[o]), bitcast<f32>(aux[o + 1u]), bitcast<f32>(aux[o + 2u]));
}

fn shape_mat(id: u32) -> u32 {
    if (id < T.opt.z) { return u32(prims[id].u.w); }
    return tris[id - T.opt.z].nm >> 28u;
}

/* the hit record: point, normals facing the ray (shading and geometric), which side */
fn make_hit(r: Ray, f: Found) -> Hit {
    var h: Hit;
    h.t = f.t;
    h.p = r.o + r.d * f.t;
    var outward = vec3f(0.0, 1.0, 0.0);
    var ns = vec3f(0.0);
    var smooth_n = false;
    if (f.id < T.opt.z) {
        let pr = prims[f.id];
        if (pr.q.w < 0.5) { outward = (h.p - pr.c.xyz) / pr.c.w; } else { outward = pr.n.xyz; }
    } else {
        let tr = tris[f.id - T.opt.z];
        outward = normalize(cross(tr.u, tr.v));
        if (tr.n0 != NONE) {
            ns = normalize(normal_at(tr.n0) * (1.0 - f.b.x - f.b.y) + normal_at(tr.n1) * f.b.x +
                           normal_at(tr.nm & 0x0fffffffu) * f.b.y);
            smooth_n = true;
        }
    }
    h.front = dot(r.d, outward) < 0.0;
    h.n = select(-outward, outward, h.front);
    h.ng = h.n;
    if (smooth_n) {
        if (dot(ns, h.n) < 0.0) { ns = -ns; }
        if (dot(ns, ns) > 0.5) { h.n = ns; }
    }
    return h;
}

/* ---- media (rt/medium.c) ---- */

/* the part of the ray inside a box: (t0, t1), t1 <= t0 when there is none */
fn clip_box(lo: vec3f, hi: vec3f, r: Ray, tmin: f32, tmax: f32) -> vec2f {
    var a = tmin;
    var b = tmax;
    for (var i = 0; i < 3; i++) {
        let d = r.d[i];
        let o = r.o[i];
        if (d == 0.0) {
            if (o < lo[i] || o > hi[i]) { return vec2f(1.0, 0.0); }
            continue;
        }
        let inv = 1.0 / d;
        let ta = min((lo[i] - o) * inv, (hi[i] - o) * inv);
        let tb = max((lo[i] - o) * inv, (hi[i] - o) * inv);
        a = max(a, ta);
        b = min(b, tb);
        if (a >= b) { return vec2f(1.0, 0.0); }
    }
    return vec2f(a, b);
}

/* Tracking through medium k along the ray, brick by brick (Amanatides and Woo DDA over the
   majorant grid), each brick with its own majorant.
   ratio = false: delta tracking, returns the distance of the first real collision, or -1;
   ratio = true: ratio tracking, returns the transmittance. */
fn track(k: u32, r: Ray, tmin: f32, tmax: f32, rng: ptr<function, u32>, ratio: bool) -> f32 {
    let nothing = select(-1.0, 1.0, ratio);
    let v = F.vols[k];
    let sig = v.albedo.w;
    let mx = VS[k].mx.x;
    if (!(mx > 0.0) || sig <= 0.0) { return nothing; }
    let seg = clip_box(VS[k].alo.xyz, VS[k].ahi.xyz, r, tmin, tmax);
    if (seg.y <= seg.x) { return nothing; }
    var t = seg.x;
    let t1 = seg.y;
    var tr = 1.0;
    /* thin: one segment with the global majorant is cheaper than walking the bricks */
    let whole = sig * mx * (t1 - t) < 4.0;
    var bi = vec3i(0);
    var sgn = vec3i(0);
    var nxt = vec3f(1e30);
    var dlt = vec3f(1e30);
    let nb = vec3i(v.bdims.xyz);
    if (!whole) {
        let bs = (v.hi.xyz - v.lo.xyz) / vec3f(v.dims.xyz) * 8.0;
        for (var a = 0; a < 3; a++) {
            let p = r.o[a] + r.d[a] * t;
            let i = clamp(i32(floor((p - v.lo[a]) / bs[a])), 0, nb[a] - 1);
            bi[a] = i;
            if (r.d[a] > 0.0) {
                sgn[a] = 1;
                nxt[a] = (v.lo[a] + f32(i + 1) * bs[a] - r.o[a]) / r.d[a];
                dlt[a] = bs[a] / r.d[a];
            } else if (r.d[a] < 0.0) {
                sgn[a] = -1;
                nxt[a] = (v.lo[a] + f32(i) * bs[a] - r.o[a]) / r.d[a];
                dlt[a] = -bs[a] / r.d[a];
            }
        }
    }
    for (var guard = 0; guard < 4096; guard++) {
        if (t >= t1) { break; }
        let ta = t;
        var tb = t1;
        var smax = sig * mx;
        if (!whole) {
            if (any(bi < vec3i(0)) || any(bi >= nb)) { break; }
            var ax = 2;
            if (nxt.x < nxt.y) { if (nxt.x < nxt.z) { ax = 0; } } else if (nxt.y < nxt.z) { ax = 1; }
            tb = max(min(nxt[ax], t1), t);
            smax = sig * bricks[v.bdims.w + u32((bi.z * nb.y + bi.y) * nb.x + bi.x)];
            bi[ax] += sgn[ax];
            nxt[ax] += dlt[ax];
        }
        t = tb;
        if (smax <= 0.0) { continue; }
        var tt = ta;
        for (var n = 0; n < 65536; n++) {
            tt -= log(1.0 - rnd(rng)) / smax;
            if (tt >= tb) { break; }
            let s = sig * density(k, r.o + r.d * tt);
            if (ratio) {
                tr *= 1.0 - min(s, smax) / smax;
                if (tr < 0.1) {
                    if (rnd(rng) < 0.5) { return 0.0; }
                    tr *= 2.0;
                }
            } else if (rnd(rng) * smax < s) {
                return tt;
            }
        }
    }
    return select(-1.0, tr, ratio);
}

fn transmittance(r: Ray, tmin: f32, tmax: f32, rng: ptr<function, u32>) -> f32 {
    var tr = 1.0;
    for (var k = 0u; k < T.cnt.w; k++) {
        if (tr <= 0.0) { break; }
        tr *= track(k, r, tmin, tmax, rng, true);
    }
    return tr;
}

fn hg(c: f32, g: f32) -> f32 {
    let d = 1.0 + g * g - 2.0 * g * c;
    return (1.0 - g * g) / (4.0 * PI_ * d * sqrt(d));
}

/* two tangents of n (Duff et al. 2017) */
fn onb(n: vec3f) -> mat2x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let c = n.x * n.y * a;
    return mat2x3f(vec3f(1.0 + s * n.x * n.x * a, s * c, -s * n.x), vec3f(c, s + n.y * n.y * a, -n.y));
}

fn cosine_hemisphere(n: vec3f, rng: ptr<function, u32>) -> vec3f {
    let u1 = rnd(rng);
    let u2 = rnd(rng);
    let phi = 2.0 * PI_ * u1;
    let rad = sqrt(u2);
    let tb = onb(n);
    return tb[0] * (rad * cos(phi)) + tb[1] * (rad * sin(phi)) + n * sqrt(max(0.0, 1.0 - u2));
}

/* a direction drawn from the Henyey-Greenstein phase function around d */
fn sample_phase(d: vec3f, g: f32, rng: ptr<function, u32>) -> vec3f {
    let xi = rnd(rng);
    var cos_t = 1.0 - 2.0 * xi;
    if (abs(g) >= 1e-3) {
        let sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * xi);
        cos_t = (1.0 + g * g - sq * sq) / (2.0 * g);
    }
    cos_t = clamp(cos_t, -1.0, 1.0);
    let sin_t = sqrt(1.0 - cos_t * cos_t);
    let phi = 2.0 * PI_ * rnd(rng);
    let tb = onb(d);
    return tb[0] * (sin_t * cos(phi)) + tb[1] * (sin_t * sin(phi)) + d * cos_t;
}

/* ---- light transport (rt/tracer.c) ---- */

fn power_heuristic(pa: f32, pb: f32) -> f32 {
    let a = pa * pa;
    let b = pb * pb;
    return select(0.0, a / (a + b), a + b > 0.0);
}
fn light_pdf(area: f32, dist2: f32, cos_light: f32) -> f32 { return dist2 / (cos_light * area); }

/* next event estimation for a diffuse surface: one point of the lamp */
fn direct_surface(h: Hit, albedo: vec3f, rng: ptr<function, u32>) -> vec3f {
    if (T.opt.w == NONE) { return vec3f(0.0); }
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let r1 = rnd(rng);
    let r2 = rnd(rng);
    let to_light = lt.q.xyz + lt.u.xyz * r1 + lt.v.xyz * r2 - h.p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_surf = dot(h.n, wi);
    let cos_light = -dot(lt.n.xyz, wi);
    if (cos_surf <= 0.0 || cos_light <= 0.0) { return vec3f(0.0); }
    let shadow = Ray(h.p, wi);
    if (traverse(shadow, T.f.y, dist - T.f.z, true).id != NONE) { return vec3f(0.0); }
    var g = cos_surf * cos_light * lt.v.w / dist2;
    if (T.cnt.w > 0u) { g *= transmittance(shadow, 0.0, dist, rng); }
    if (T.opt.y != 0u) { g *= power_heuristic(light_pdf(lt.v.w, dist2, cos_light), cos_surf / PI_); }
    return albedo * le * (g / PI_);
}

/* the same for a point inside a medium, reached along d: the phase function replaces the BRDF */
fn direct_medium(p: vec3f, d: vec3f, g: f32, rng: ptr<function, u32>) -> vec3f {
    if (T.opt.w == NONE) { return vec3f(0.0); }
    let lt = prims[T.opt.w];
    let le = T.mats[u32(lt.u.w)].emission.rgb;
    let r1 = rnd(rng);
    let r2 = rnd(rng);
    let to_light = lt.q.xyz + lt.u.xyz * r1 + lt.v.xyz * r2 - p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_light = -dot(lt.n.xyz, wi);
    if (cos_light <= 0.0) { return vec3f(0.0); }
    let shadow = Ray(p, wi);
    if (traverse(shadow, T.f.y, dist - T.f.z, true).id != NONE) { return vec3f(0.0); }
    let tr = transmittance(shadow, 0.0, dist, rng);
    if (tr <= 0.0) { return vec3f(0.0); }
    let ph = hg(dot(d, wi), g);
    var w = ph * tr * cos_light * lt.v.w / dist2;
    if (T.opt.y != 0u) { w *= power_heuristic(light_pdf(lt.v.w, dist2, cos_light), ph); }
    return le * w;
}

fn schlick(cos_i: f32, eta: f32) -> f32 {
    let r0 = (1.0 - eta) / (1.0 + eta);
    return r0 * r0 + (1.0 - r0 * r0) * pow(1.0 - cos_i, 5.0);
}

fn refract_dir(d: vec3f, n: vec3f, eta: f32) -> vec3f {
    let cos_i = min(dot(-d, n), 1.0);
    let perp = (d + n * cos_i) * eta;
    return perp - n * sqrt(abs(1.0 - dot(perp, perp)));
}

/* Russian roulette after 3 bounces: false = the path stops */
fn roulette(thr: ptr<function, vec3f>, depth: i32, rng: ptr<function, u32>) -> bool {
    if (depth < 3) { return true; }
    let c = *thr;
    let p = min(max(c.x, max(c.y, c.z)), 0.95);
    if (rnd(rng) >= p) { return false; }
    *thr = c / p;
    return true;
}

/* the radiance arriving along one camera ray */
fn trace(ray0: Ray, rng: ptr<function, u32>) -> vec3f {
    var ray = ray0;
    var radiance = vec3f(0.0);
    var thr = vec3f(1.0);
    var emit = 0;             /* a lamp hit counts: 0 fully, 1 not (NEE had it), 2 weighted against NEE */
    var bounce_pdf = 0.0;
    let max_depth = i32(T.opt.x);
    let mis = T.opt.y != 0u;
    let after_bounce = select(1, 2, mis);
    for (var depth = 0; depth < 4096; depth++) {
        let f = traverse(ray, T.f.y, 1e30, false);
        let found = f.id != NONE;
        if (T.cnt.w > 0u) {
            /* the ray may scatter in a medium before the surface: the earliest collision wins */
            var ts = select(1e30, f.t, found);
            var which = NONE;
            for (var k = 0u; k < T.cnt.w; k++) {
                let ti = track(k, ray, 0.0, ts, rng, false);
                if (ti >= 0.0) { ts = ti; which = k; }
            }
            if (which != NONE) {
                if (depth >= max_depth) { break; }
                let vol = F.vols[which];
                let p = ray.o + ray.d * ts;
                thr *= vol.albedo.xyz;
                radiance += thr * direct_medium(p, ray.d, vol.p.x, rng);
                let nd = sample_phase(ray.d, vol.p.x, rng);
                bounce_pdf = hg(dot(ray.d, nd), vol.p.x);
                ray = Ray(p, nd);
                emit = after_bounce;
                if (!roulette(&thr, depth, rng)) { break; }
                continue;
            }
        }
        if (!found) { break; }
        let h = make_hit(ray, f);
        let m = T.mats[shape_mat(f.id)];
        let kind = u32(m.albedo.w);
        if (kind == 3u) {
            if (h.front) {
                if (emit == 0) {
                    radiance += thr * m.emission.rgb;
                } else if (emit == 2 && f.id == T.opt.w) {
                    let lt = prims[T.opt.w];
                    let cos_light = -dot(lt.n.xyz, ray.d);
                    if (cos_light > 0.0) {
                        radiance += thr * m.emission.rgb * power_heuristic(bounce_pdf, light_pdf(lt.v.w, h.t * h.t, cos_light));
                    }
                }
            }
            break;
        }
        if (depth >= max_depth) { break; }
        let d = ray.d;
        var nxt = d;
        if (kind == 0u) {
            radiance += thr * direct_surface(h, m.albedo.rgb, rng);
            nxt = cosine_hemisphere(h.n, rng);
            bounce_pdf = dot(h.n, nxt) / PI_;
            thr *= m.albedo.rgb;
            emit = after_bounce;
        } else if (kind == 1u) {
            nxt = reflect(d, h.n);
            thr *= m.albedo.rgb;
            emit = 0;
        } else {
            let ior = m.emission.w;
            let eta = select(ior, 1.0 / ior, h.front);
            let cos_i = min(dot(-d, h.n), 1.0);
            let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
            if (sin_t2 > 1.0 || rnd(rng) < schlick(cos_i, eta)) { nxt = reflect(d, h.n); } else { nxt = refract_dir(d, h.n, eta); }
            thr *= m.albedo.rgb;
            emit = 0;
        }
        if (!roulette(&thr, depth, rng)) { break; }
        ray = Ray(h.p, normalize(nxt));
    }
    return radiance;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let id = gid.x + gid.y * nw.x * 64u;
    let W = T.dims.x;
    if (id >= W * T.dims.w) { return; }
    let x = id % W;
    let y = T.dims.z + id / W;
    if (y >= T.dims.y) { return; }
    let k = y * W + x;
    let acc = accum[k];
    /* seeded from (pixel, sample number): every sample of every pixel is different */
    var rng = hash32(k ^ hash32(u32(acc.w) * 0x9e3779b9u + T.cnt.z));
    let u = (f32(x) + rnd(&rng)) / f32(W);
    let v = (f32(y) + rnd(&rng)) / f32(T.dims.y);
    let through = T.upper_left.xyz + T.horizontal.xyz * u - T.vertical.xyz * v;
    var c = trace(Ray(T.origin.xyz, normalize(through - T.origin.xyz)), &rng);
    let mx = max(c.x, max(c.y, c.z));
    if (!(mx >= 0.0 && mx < 1e30)) {
        c = vec3f(0.0);
    } else if (T.f.x > 0.0 && mx > T.f.x) {
        c *= T.f.x / mx;
    }
    accum[k] = acc + vec4f(c, 1.0);
}
);

#define RT_EPS_F 2e-3f           // rt/tracer.c, float build: minimum ray distance
#define RT_SHADOW_MARGIN_F 2e-2f // shadow rays stop this far before the lamp
#define CHUNK_TRIS 32768

typedef struct {
    float origin[4], upper_left[4], horizontal[4], vertical[4];
    uint32_t dims[4], opt[4], cnt[4];
    float f[4];
    float mats[MAX_MATS][8];
} GTP;

struct GpuTrace {
    Gpu *g;
    int w, h;
    WGPUBindGroupLayout bgl;
    WGPUComputePipeline pipe;
    WGPUBuffer params, accum;
    WGPUBuffer nodes, aux, prims, tris;
    size_t nodes_cap, aux_cap, prims_cap, tris_cap;
    uint32_t nnodes, normals_at;
    // what is on the GPU now
    Primitive *prim_copy;
    int nprim_copy;
    const Triangle *mesh_tris;
    const Vec3 *mesh_normals;
    int mesh_ntris, mesh_nnormals;
    int valid;
    Bvh own;                 // built here when the scene has no BVH of its own
    int own_built;
};

GpuTrace *gpu_trace_create(Gpu *g, int width, int height, const char *types_wgsl, const char *media_wgsl){
    if (!g || g->state != GPU_READY) return NULL;
    GpuTrace *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->g = g;
    t->w = width;
    t->h = height;
    char *src = malloc(strlen(types_wgsl) + strlen(media_wgsl) + strlen(TRACE) + 8);
    if (!src){ free(t); return NULL; }
    strcpy(src, types_wgsl); strcat(src, "\n"); strcat(src, media_wgsl); strcat(src, "\n"); strcat(src, TRACE);
    // frame, densities, bricks, media boxes, parameters, nodes, aux, prims, tris, accumulation
    t->bgl = gpu_layout(g, WGPUShaderStage_Compute, "URRRURRRRW");
    t->pipe = gpu_compute(g, "path tracer", src, "main", t->bgl);
    free(src);
    t->params = gpu_buffer(g, "path tracer parameters", sizeof(GTP), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    t->accum = gpu_buffer(g, "path tracer samples", (size_t)width * height * 16,
                          WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
    return t;
}

WGPUBuffer gpu_trace_accum(GpuTrace *t){ return t->accum; }

static void put(float *d, Vec3 a, float w){ d[0] = (float)a.x; d[1] = (float)a.y; d[2] = (float)a.z; d[3] = w; }

static int geometry_same(const GpuTrace *t, const Scene *s){
    return t->valid && t->nprim_copy == s->nprims && t->mesh_tris == s->tris && t->mesh_ntris == s->ntris &&
           t->mesh_normals == s->normals && t->mesh_nnormals == s->nnormals &&
           (s->nprims == 0 || memcmp(t->prim_copy, s->prims, (size_t)s->nprims * sizeof(Primitive)) == 0);
}

int gpu_trace_sync(GpuTrace *t, const Scene *s){
    Gpu *g = t->g;
    if (geometry_same(t, s)) return 1;
    t->valid = 0;
    const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

    // the BVH: the scene's own if it has one (it must be up to date: scene_set_accel after
    // the geometry changed), otherwise one is built here
    const Bvh *b = scene_bvh(s);
    if (!b){
        if (t->own_built) bvh_free(&t->own);
        AccelParams ap = accel_default_params();
        bvh_build(&t->own, s, ap.bvh_leaf, ap.bvh_bins, ap.bvh_split);
        t->own_built = 1;
        b = &t->own;
    }
    if (b->depth > 60) fprintf(stderr, "GPU path tracer: the BVH is %d levels deep, its stack holds 64: some hits may be missed\n", b->depth);
    size_t bytes_nodes = (size_t)b->nnodes * 32, bytes_aux = ((size_t)b->nrefs + 3 * (size_t)s->nnormals) * 4;
    size_t bytes_tris = (size_t)s->ntris * 48;
    size_t most = bytes_nodes > bytes_aux ? bytes_nodes : bytes_aux;
    if (bytes_tris > most) most = bytes_tris;
    if (most > g->max_storage){
        fprintf(stderr, "GPU: the scene needs a %.0f MB buffer, this GPU allows %.0f MB: the CPU path traces\n",
                (double)most / 1048576, (double)g->max_storage / 1048576);
        return 0;
    }

    // nodes
    {
        uint32_t *nd = malloc((size_t)(b->nnodes ? b->nnodes : 1) * 32);
        if (!nd) return 0;
        for (int i = 0; i < b->nnodes; i++){
            const BvhNode *n = &b->nodes[i];
            float lo[4], hi[4];
            put(lo, n->box.lo, 0);
            put(hi, n->box.hi, 0);
            uint32_t *o = nd + 8 * (size_t)i;
            memcpy(o, lo, 12);
            memcpy(o + 4, hi, 12);
            o[3] = n->count > 0 ? (uint32_t)n->start : (uint32_t)n->left;
            o[7] = n->count > 0 ? (0x80000000u | (uint32_t)n->count) : (uint32_t)n->right;
        }
        gpu_buffer_reserve(g, &t->nodes, &t->nodes_cap, bytes_nodes, st, "BVH nodes");
        if (b->nnodes) wgpuQueueWriteBuffer(g->queue, t->nodes, 0, nd, bytes_nodes);
        free(nd);
        t->nnodes = (uint32_t)b->nnodes;
    }
    // aux: the shape ids of the leaves, then the normals
    {
        uint32_t *ax = malloc(bytes_aux ? bytes_aux : 4);
        if (!ax) return 0;
        for (int i = 0; i < b->nrefs; i++) ax[i] = (uint32_t)b->refs[i];
        float *nf = (float *)(ax + b->nrefs);
        for (int i = 0; i < s->nnormals; i++){
            nf[3 * i] = (float)s->normals[i].x; nf[3 * i + 1] = (float)s->normals[i].y; nf[3 * i + 2] = (float)s->normals[i].z;
        }
        gpu_buffer_reserve(g, &t->aux, &t->aux_cap, bytes_aux, st, "BVH shapes and normals");
        if (bytes_aux) wgpuQueueWriteBuffer(g->queue, t->aux, 0, ax, bytes_aux);
        free(ax);
        t->normals_at = (uint32_t)b->nrefs;
    }
    // spheres and quads
    {
        float *pr = malloc((size_t)(s->nprims ? s->nprims : 1) * 96);
        if (!pr) return 0;
        for (int i = 0; i < s->nprims; i++){
            const Primitive *p = &s->prims[i];
            float *o = pr + 24 * (size_t)i;
            put(o, p->center, (float)p->radius);
            put(o + 4, p->q, p->type == PRIM_QUAD ? 1.0f : 0.0f);
            put(o + 8, p->u, (float)p->material);
            put(o + 12, p->v, (float)p->area);
            put(o + 16, p->normal, (float)p->plane_d);
            put(o + 20, p->w, 0);
        }
        gpu_buffer_reserve(g, &t->prims, &t->prims_cap, (size_t)s->nprims * 96, st, "primitives");
        if (s->nprims) wgpuQueueWriteBuffer(g->queue, t->prims, 0, pr, (size_t)s->nprims * 96);
        free(pr);
    }
    // mesh triangles (only when they changed: they are many)
    if (!(t->tris && t->mesh_tris == s->tris && t->mesh_ntris == s->ntris)){
        gpu_buffer_reserve(g, &t->tris, &t->tris_cap, bytes_tris, st, "triangles");
        uint32_t *tr = malloc((size_t)CHUNK_TRIS * 48);
        if (!tr) return 0;
        for (int first = 0; first < s->ntris; first += CHUNK_TRIS){
            int n = s->ntris - first < CHUNK_TRIS ? s->ntris - first : CHUNK_TRIS;
            for (int i = 0; i < n; i++){
                const Triangle *q = &s->tris[first + i];
                uint32_t *o = tr + 12 * (size_t)i;
                float a[4];
                put(a, q->q, 0); memcpy(o, a, 12);
                put(a, q->u, 0); memcpy(o + 4, a, 12);
                put(a, q->v, 0); memcpy(o + 8, a, 12);
                int smooth_n = q->n[0] >= 0;
                o[3] = smooth_n ? (uint32_t)q->n[0] : 0xffffffffu;
                o[7] = smooth_n ? (uint32_t)q->n[1] : 0;
                o[11] = ((uint32_t)(q->material & 15) << 28) | (smooth_n ? ((uint32_t)q->n[2] & 0x0fffffffu) : 0);
            }
            wgpuQueueWriteBuffer(g->queue, t->tris, (uint64_t)first * 48, tr, (size_t)n * 48);
        }
        free(tr);
    }
    Primitive *c = realloc(t->prim_copy, (size_t)(s->nprims ? s->nprims : 1) * sizeof(Primitive));
    if (!c) return 0;
    t->prim_copy = c;
    if (s->nprims) memcpy(c, s->prims, (size_t)s->nprims * sizeof(Primitive));
    t->nprim_copy = s->nprims;
    t->mesh_tris = s->tris;
    t->mesh_normals = s->normals;
    t->mesh_ntris = s->ntris;
    t->mesh_nnormals = s->nnormals;
    t->valid = 1;
    return 1;
}

void gpu_trace_clear(GpuTrace *t, WGPUCommandEncoder enc){
    wgpuCommandEncoderClearBuffer(enc, t->accum, 0, (uint64_t)t->w * t->h * 16);
}

void gpu_trace_encode(GpuTrace *t, WGPUCommandEncoder enc, const Scene *s, const GpuTraceParams *tp,
                      int y0, int rows, int repeats, int nvolumes,
                      WGPUBuffer frame_u, WGPUBuffer dens, WGPUBuffer bricks, WGPUBuffer vstat){
    Gpu *g = t->g;
    GTP P;
    memset(&P, 0, sizeof(P));
    put(P.origin, tp->cam.origin, 0);
    put(P.upper_left, tp->cam.upper_left, 0);
    put(P.horizontal, tp->cam.horizontal, 0);
    put(P.vertical, tp->cam.vertical, 0);
    P.dims[0] = (uint32_t)t->w; P.dims[1] = (uint32_t)t->h; P.dims[2] = (uint32_t)y0; P.dims[3] = (uint32_t)rows;
    P.opt[0] = (uint32_t)(tp->max_depth > 0 ? tp->max_depth : 0);
    P.opt[1] = tp->mis ? 1u : 0u;
    P.opt[2] = (uint32_t)s->nprims;
    P.opt[3] = s->light >= 0 && s->light < s->nprims ? (uint32_t)s->light : 0xffffffffu;
    P.cnt[0] = t->nnodes;
    P.cnt[1] = t->normals_at;
    P.cnt[2] = 0;
    P.cnt[3] = (uint32_t)nvolumes;
    P.f[0] = (float)tp->clamp;
    P.f[1] = RT_EPS_F;
    P.f[2] = RT_SHADOW_MARGIN_F;
    for (int m = 0; m < s->nmats && m < MAX_MATS; m++){
        put(P.mats[m], s->mats[m].albedo, (float)s->mats[m].type);
        put(P.mats[m] + 4, s->mats[m].emission, (float)s->mats[m].ior);
    }
    wgpuQueueWriteBuffer(g->queue, t->params, 0, &P, sizeof(P));
    WGPUBuffer b[10] = {frame_u, dens, bricks, vstat, t->params, t->nodes, t->aux, t->prims, t->tris, t->accum};
    WGPUBindGroup bg = gpu_bind(g, t->bgl, b, 10);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, t->pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    for (int k = 0; k < repeats; k++) gpu_dispatch(pass, (size_t)t->w * rows);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuBindGroupRelease(bg);
}

void gpu_trace_destroy(GpuTrace *t){
    if (!t) return;
    WGPUBuffer bufs[] = {t->params, t->accum, t->nodes, t->aux, t->prims, t->tris};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]){ wgpuBufferDestroy(bufs[i]); wgpuBufferRelease(bufs[i]); }
    if (t->pipe) wgpuComputePipelineRelease(t->pipe);
    if (t->bgl) wgpuBindGroupLayoutRelease(t->bgl);
    if (t->own_built) bvh_free(&t->own);
    free(t->prim_copy);
    free(t);
}

#else
typedef int trace_gpu_unused;
#endif
